"""Fused Triton update for the F3+F1 Bop rule (per-weight second moment, threshold over movable weights only).

Per layer and step the PyTorch hook used about 25 small kernels. Here it is:
  every step:        one pass over (grad, m, v)         -> m, v updated in place          (_mv_update)
  every `every`-th:  a fixed-shape threshold from a 65,536-element sample (about 10 small device ops, no host sync),
                     then one pass over (weight, m, v)  -> weight moved, m reset, flips counted (_flip)
The threshold is a device scalar, so nothing here forces a host sync. `m` can be stored in bf16 with stochastic rounding
(sr=True), which keeps a slow EMA alive without the extra 2 bytes per weight of an fp32 accumulator.
Usage from tune.py: set cfg fused=True (rule quantile, secmom, movable_q required).
"""
import torch
import triton
import triton.language as tl

BLOCK = 1024
SAMPLE = 65536


@triton.jit
def _mv_update(g_ptr, m_ptr, v_ptr, n, gamma, beta2, seed, SR: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + offs, mask=mask, other=0.0)
    m = m + gamma * (g - m)
    v = v + beta2 * (g * g - v)
    if SR:  # stochastic rounding to bf16: add 16 random bits below the kept mantissa, then truncate
        r = tl.randint(seed, offs).to(tl.int32)
        mi = m.to(tl.int32, bitcast=True)
        mi = (mi + (r & 65535)) & -65536
        m = mi.to(tl.float32, bitcast=True)
    tl.store(m_ptr + offs, m.to(m_ptr.dtype.element_ty), mask=mask)
    tl.store(v_ptr + offs, v, mask=mask)


@triton.jit
def _flip(p_ptr, m_ptr, v_ptr, tau_ptr, flips_ptr, n, bias_corr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    p = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + offs, mask=mask, other=1.0)
    tau = tl.load(tau_ptr)
    src = m / (tl.sqrt(v / bias_corr) + 1e-12)
    new = p + (src < -tau).to(tl.float32) - (src > tau).to(tl.float32)
    new = tl.minimum(tl.maximum(new, -1.0), 1.0)
    changed = (new != p) & mask
    tl.store(p_ptr + offs, new.to(p_ptr.dtype.element_ty), mask=mask)
    m = tl.where(changed, 0.0, m)
    tl.store(m_ptr + offs, m.to(m_ptr.dtype.element_ty), mask=mask)
    tl.atomic_add(flips_ptr, tl.sum(changed.to(tl.float32), axis=0))


def mv_update(g, m, v, gamma, beta2, seed=0, sr=False):
    n = m.numel()
    _mv_update[(triton.cdiv(n, BLOCK),)](g, m, v, n, gamma, beta2, seed, SR=sr, BLOCK=BLOCK)


def flip(p, m, v, tau, flips, bias_corr):
    n = p.numel()
    _flip[(triton.cdiv(n, BLOCK),)](p, m, v, tau, flips, n, bias_corr, BLOCK=BLOCK)


def threshold(p, m, v, bias_corr, rate, every, gen):
    """Fixed-shape version of F1's conditional quantile: the |score| quantile over movable weights in a random sample.
    Blocked entries get key -1 so they sort first; the quantile position is offset past them. Returns a 0-dim device tensor."""
    n_out, n_in = p.shape
    rows = torch.randint(0, n_out, (SAMPLE,), device=p.device, generator=gen)
    cols = torch.randint(0, n_in, (SAMPLE,), device=p.device, generator=gen)
    ps, ms, vs = p[rows, cols].float(), m[rows, cols].float(), v[rows, cols]
    src = ms / (torch.sqrt(vs / bias_corr) + 1e-12)
    mov = ~(((ps > 0) & (src < 0)) | ((ps < 0) & (src > 0)))
    n_mov = mov.sum()
    f_mov = (n_mov.float() / SAMPLE).clamp_min(0.05)
    q = 1 - torch.clamp(rate * every / f_mov, max=0.5)
    key = torch.sort(torch.where(mov, src.abs(), torch.full_like(src, -1.0))).values
    pos = q * (n_mov - 1).clamp_min(0).float()
    lo = pos.floor().long()
    hi = torch.minimum(lo + 1, (n_mov - 1).clamp_min(0))
    base = SAMPLE - n_mov
    a, b = key[(base + lo).clamp(max=SAMPLE - 1)], key[(base + hi).clamp(max=SAMPLE - 1)]
    tau = a + (pos - lo.float()) * (b - a)
    return torch.where(n_mov > 0, tau, torch.full_like(tau, float("inf")))
