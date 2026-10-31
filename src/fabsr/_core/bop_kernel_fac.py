"""Factored F3+F1 Bop: no full-matrix v or score; optional bf16 SR.

Ordinary steps: fused momentum + tiled g^2 sums, then finalize EMAs and
clear scratch (two custom launches). Check steps add a sampled quantile
and flip/reset. State costs 8*(n_out+n_in) bytes beyond weight and m.
Calls for one State must run in order on the same CUDA stream.
Atomic reduction order can change the last few fp32 bits versus PyTorch.
"""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

BLOCK = 1024
ROWS, COLS = 32, 128
SAMPLE = 65536


@dataclass
class State:
    row: torch.Tensor
    col: torch.Tensor
    row_sum: torch.Tensor
    col_sum: torch.Tensor


def make_state(p):
    assert p.is_cuda and p.ndim == 2 and p.is_contiguous() and p.numel() > 0
    nr, nc = p.shape
    return State(*(torch.zeros(n, dtype=torch.float32, device=p.device)
                   for n in (nr, nc, nr, nc)))


@triton.jit
def _mv_stats(g_ptr, m_ptr, rs_ptr, cs_ptr, nr, nc, gamma, seed,
              SR: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr):
    rr = tl.program_id(0) * BR + tl.arange(0, BR)
    cc = tl.program_id(1) * BC + tl.arange(0, BC)
    offsets = rr[:, None] * nc + cc[None, :]
    mask = (rr[:, None] < nr) & (cc[None, :] < nc)
    g = tl.load(g_ptr + offsets, mask, other=0.0).to(tl.float32)
    m = tl.load(m_ptr + offsets, mask, other=0.0).to(tl.float32)
    gm = g
    if not SR:
        # hook() casts g to m.dtype before a non-SR lerp.
        gm = g.to(m_ptr.dtype.element_ty).to(tl.float32)
    m = m + gamma * (gm - m)
    if SR:
        bits = m.to(tl.int32, bitcast=True)
        rnd = tl.randint(seed, offsets.to(tl.uint32)).to(tl.int32) & 65535
        m = ((bits + rnd) & -65536).to(tl.float32, bitcast=True)
    tl.store(m_ptr + offsets, m.to(m_ptr.dtype.element_ty), mask)
    g2 = g * g
    tl.atomic_add(rs_ptr + rr, tl.sum(g2, axis=1), mask=rr < nr, sem="relaxed")
    tl.atomic_add(cs_ptr + cc, tl.sum(g2, axis=0), mask=cc < nc, sem="relaxed")


@triton.jit
def _finish_stats(r_ptr, c_ptr, rs_ptr, cs_ptr, nr, nc, beta2,
                  BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rm = idx < nr
    ci = idx - nr
    cm = (ci >= 0) & (ci < nc)
    r = tl.load(r_ptr + idx, rm, other=0.0)
    c = tl.load(c_ptr + ci, cm, other=0.0)
    rs = tl.load(rs_ptr + idx, rm, other=0.0) / nc
    cs = tl.load(cs_ptr + ci, cm, other=0.0) / nr
    tl.store(r_ptr + idx, r + beta2 * (rs - r), rm)
    tl.store(c_ptr + ci, c + beta2 * (cs - c), cm)
    # Each scratch element has one writer after all accumulation CTAs finish.
    tl.store(rs_ptr + idx, 0.0, rm)
    tl.store(cs_ptr + ci, 0.0, cm)


@triton.jit
def _corrected_mean(r_ptr, nr, bias_corr, BLOCK: tl.constexpr):
    ii = tl.arange(0, BLOCK)
    accum = tl.full((BLOCK,), 0.0, tl.float32)
    for start in range(0, nr, BLOCK):
        r = tl.load(r_ptr + start + ii, start + ii < nr, other=0.0)
        accum += r / bias_corr
    return tl.maximum(tl.sum(accum, axis=0) / nr, 1e-30)


@triton.jit
def _row_mean(r_ptr, mean_ptr, nr, bias_corr, BLOCK: tl.constexpr):
    tl.store(mean_ptr, _corrected_mean(r_ptr, nr, bias_corr, BLOCK))


@triton.jit
def _sample_keys(p_ptr, m_ptr, r_ptr, c_ptr, rows_ptr, cols_ptr,
                 keys_ptr, counts_ptr, mean_ptr, nr, nc, bias_corr,
                 BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    ii = pid * BLOCK + tl.arange(0, BLOCK)
    rr = tl.load(rows_ptr + ii)
    cc = tl.load(cols_ptr + ii)
    # R is small. Repeat its reduction across sample CTAs to save a mean
    # launch; only CTA 0 writes the scalar subsequently consumed by flip().
    mean = _corrected_mean(r_ptr, nr, bias_corr, BLOCK)
    if pid == 0:
        tl.store(mean_ptr, mean)
    p = tl.load(p_ptr + rr * nc + cc).to(tl.float32)
    m = tl.load(m_ptr + rr * nc + cc).to(tl.float32)
    r = tl.load(r_ptr + rr) / bias_corr
    c = tl.load(c_ptr + cc) / bias_corr
    src = m / (tl.sqrt(r * c / mean) + 1e-12)
    movable = ~(((p > 0) & (src < 0)) | ((p < 0) & (src > 0)))
    tl.store(keys_ptr + ii, tl.where(movable, tl.abs(src), -1.0))
    tl.store(counts_ptr + pid, tl.sum(movable.to(tl.int32), axis=0))


@triton.jit
def _quantile(sorted_ptr, counts_ptr, tau_ptr, rate, every,
               SAMPLE: tl.constexpr, COUNT_BLOCK: tl.constexpr):
    ii = tl.arange(0, COUNT_BLOCK)
    n_mov = tl.sum(tl.load(counts_ptr + ii), axis=0)
    f_mov = tl.maximum(n_mov.to(tl.float32) / SAMPLE, 0.05)
    q = 1.0 - tl.minimum(tl.div_rn(rate * every, f_mov), 0.5)
    last = tl.maximum(n_mov - 1, 0)
    # torch.quantile computes the rank in double precision even for fp32
    # samples. fp32 rank arithmetic loses fractional precision at ~65k.
    pos = q.to(tl.float64) * last.to(tl.float64)
    lo = tl.floor(pos).to(tl.int32)
    hi = tl.minimum(lo + 1, last)
    base = SAMPLE - n_mov
    a = tl.load(sorted_ptr + tl.minimum(base + lo, SAMPLE - 1))
    b = tl.load(sorted_ptr + tl.minimum(base + hi, SAMPLE - 1))
    frac = (pos - lo.to(tl.float64)).to(tl.float32)
    tau = tl.where(frac < 0.5, a + frac * (b - a), b - (1.0 - frac) * (b - a))
    tl.store(tau_ptr, tl.where(n_mov > 0, tau, float("inf")))


@triton.jit
def _flip(p_ptr, m_ptr, r_ptr, c_ptr, mean_ptr, tau_ptr, flips_ptr,
          n, nc, bias_corr, BLOCK: tl.constexpr):
    ii = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = ii < n
    p = tl.load(p_ptr + ii, mask, other=0.0).to(tl.float32)
    m = tl.load(m_ptr + ii, mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + ii // nc, mask, other=0.0) / bias_corr
    c = tl.load(c_ptr + ii % nc, mask, other=0.0) / bias_corr
    mean = tl.load(mean_ptr)
    tau = tl.load(tau_ptr)
    src = m / (tl.sqrt(r * c / mean) + 1e-12)
    new = tl.minimum(tl.maximum(
        p + (src < -tau).to(tl.float32) - (src > tau).to(tl.float32), -1.0), 1.0)
    changed = mask & (new != p)
    tl.store(p_ptr + ii, new.to(p_ptr.dtype.element_ty), mask)
    tl.store(m_ptr + ii, tl.where(changed, 0.0, m).to(m_ptr.dtype.element_ty), mask)
    tl.atomic_add(flips_ptr, tl.sum(changed.to(tl.int32), axis=0).to(tl.int64),
                  sem="relaxed")


@torch.no_grad()
def mv_update(g, m, state, gamma, beta2, seed=0, sr=False):
    assert g.is_cuda and m.device == g.device and g.shape == m.shape
    assert g.is_contiguous() and m.is_contiguous()
    assert m.ndim == 2 and m.shape == (state.row.numel(), state.col.numel())
    assert m.dtype in (torch.bfloat16, torch.float32)
    nr, nc = m.shape
    _mv_stats[(triton.cdiv(nr, ROWS), triton.cdiv(nc, COLS))](
        g, m, state.row_sum, state.col_sum, nr, nc, gamma, seed,
        SR=bool(sr and m.dtype == torch.bfloat16), BR=ROWS, BC=COLS)
    _finish_stats[(triton.cdiv(nr + nc, BLOCK),)](
        state.row, state.col, state.row_sum, state.col_sum,
        nr, nc, beta2, BLOCK=BLOCK)


@torch.no_grad()
def row_mean(state, bias_corr):
    """Corrected, clamped mean(R), for testing or an externally supplied tau."""
    mean = torch.empty((), dtype=torch.float32, device=state.row.device)
    _row_mean[(1,)](state.row, mean, state.row.numel(), bias_corr, BLOCK=BLOCK)
    return mean


@torch.no_grad()
def threshold(p, m, state, bias_corr, rate, every, gen=None):
    """Return (device tau, corrected row mean) from the existing F1 quantile.

    Same two randint calls, movable mask, cap and interpolation as F3+F1.
    Reuse the returned mean in flip(), before updating state again.
    """
    assert p.is_contiguous() and m.is_contiguous() and p.shape == m.shape
    nr, nc = p.shape
    rows = torch.randint(0, nr, (SAMPLE,), device=p.device, generator=gen)
    cols = torch.randint(0, nc, (SAMPLE,), device=p.device, generator=gen)
    keys = torch.empty(SAMPLE, device=p.device, dtype=torch.float32)
    counts = torch.empty(SAMPLE // BLOCK, device=p.device, dtype=torch.int32)
    mean = torch.empty((), device=p.device, dtype=torch.float32)
    tau = torch.empty_like(mean)
    _sample_keys[(SAMPLE // BLOCK,)](
        p, m, state.row, state.col, rows, cols, keys, counts, mean,
        nr, nc, bias_corr, BLOCK=BLOCK)
    keys = torch.sort(keys).values
    # Match torch's separate quantile interpolation arithmetic.
    _quantile[(1,)](keys, counts, tau, rate, every, SAMPLE=SAMPLE,
                    COUNT_BLOCK=SAMPLE // BLOCK, enable_fp_fusion=False)
    return tau, mean


@torch.no_grad()
def flip(p, m, state, tau, flips, bias_corr, mean):
    assert p.is_contiguous() and m.is_contiguous() and p.shape == m.shape
    assert flips.dtype == torch.int64, "use an int64 cumulative flip counter"
    _flip[(triton.cdiv(p.numel(), BLOCK),)](
        p, m, state.row, state.col, mean, tau, flips,
        p.numel(), p.shape[1], bias_corr, BLOCK=BLOCK)
