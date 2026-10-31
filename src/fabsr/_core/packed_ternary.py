"""2-bit packed ternary weights for Fabsr.

A ternary matrix T in {-1, 0, +1} (out x in, in % 4 == 0) is stored as uint8 (out x in/4): byte j of a row holds columns 4j..4j+3,
2 bits each, code = value + 1 (so -1 -> 0, 0 -> 1, +1 -> 2; code 3 is never written). 969M weights: 1.94 GB bf16 -> 0.24 GB packed.

There is no dense Parameter. PackedLinearFn unpacks one matrix into a transient bf16 buffer for the matmul, and in backward it computes the
dense weight gradient, hands it straight to the optimizer update (momentum + factored second moment + threshold + flip, all on the packed
codes) and drops it. The flip kernels are copies of bop_kernel_fac._sample_keys/_flip that read and write the packed bytes; everything else
(mv_update, quantile, State) is reused from bop_kernel_fac unchanged, so with the same seeds the packed run flips exactly the weights the
dense run flips (test_packed.py checks this).

Momentum is dense bf16 (1.94 GB at 1B) unless OPT["m_int8"] is on.

OPT: every optional feature, all OFF by default. With OPT at its defaults this file runs the same statements as the version that was
verified in training (baselines/PRE_TOGGLE/packed_ternary.py); each feature below only adds work behind its own flag.

  wgrad_every  k    weight gradient only every k-th step (flip steps always); the averages decay as if no step were skipped. UNVALIDATED for quality.
  m_int8            momentum stored as int8 + one fp32 scale per row, stochastic rounding. Halves the momentum memory, slower. UNVALIDATED for quality.
  sparse_store      the flip kernel writes only the bytes / momentum entries that changed. Same results, less write traffic.
  fast_unpack       unpack() as one Triton pass instead of four elementwise kernels (shift, mask, convert, subtract). Identical output (test_speedups checks torch.equal).
  fast_actquant     lowmem_models.fake_quant_act (the BitNet int8 activation quantisation) as one Triton pass instead of ~10 elementwise kernels. Identical output and gradient.

Per-module switches (set by speedups.py): PackedTLinear.frozen (no weight gradient, never flips), PackedTLinear.int8_act (real int8 forward).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from . import bop_kernel_fac as K
from .bop_kernel_fac import _corrected_mean

BYTES = 256  # packed bytes per program in the flip kernel = 1024 weights, same work per program as the dense kernel

OPT = {"wgrad_every": 1, "m_int8": False, "sparse_store": False, "fast_unpack": False, "fast_actquant": False}


# ---------------------------------------------------------------- format
def pack(T):
    """(out, in) tensor of {-1, 0, +1} -> uint8 (out, in // 4)."""
    out, n_in = T.shape
    assert n_in % 4 == 0, "input dim must be a multiple of 4"
    c = (T.to(torch.int8) + 1).to(torch.uint8).reshape(out, n_in // 4, 4)
    return c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)


def _shifts(codes):
    return torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=codes.device)


@triton.jit
def _unpack_kernel(pk_ptr, out_ptr, n_bytes, BLOCK: tl.constexpr):
    bi = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lane = tl.arange(0, 4)
    bmask = bi < n_bytes
    byte = tl.load(pk_ptr + bi, bmask, other=1).to(tl.int32)
    v = ((byte[:, None] >> (lane[None, :] * 2)) & 3) - 1
    tl.store(out_ptr + bi[:, None] * 4 + lane[None, :], v.to(out_ptr.dtype.element_ty), bmask[:, None] & (lane[None, :] < 4))


@triton.jit
def _bf16(t):  # round an fp32 value to bf16 and back, as PyTorch does after every bf16 op
    return t.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _actquant_kernel(x_ptr, y_ptr, K, min_amax, BLOCK: tl.constexpr):
    """lowmem_models.fake_quant_act on one row, with every intermediate rounded to bf16 exactly where the PyTorch op chain rounds it."""
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < K
    x = tl.load(x_ptr + row * K + offs, mask, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), min_amax)  # clamp(min=1e-5) with 1e-5 as a bf16 value
    scale = _bf16(_bf16(tl.math.div_rn(1.0, amax)) * 127.0)  # 127.0 / t  is  reciprocal(t) * 127.0 in PyTorch: two roundings
    v = _bf16(x * scale)
    f = tl.floor(v)
    d = v - f
    odd = (f - 2.0 * tl.floor(f * 0.5)) > 0.5
    q = tl.where(d > 0.5, f + 1.0, tl.where(d < 0.5, f, tl.where(odd, f + 1.0, f)))  # round half to even
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    out = _bf16(x + _bf16(_bf16(tl.math.div_rn(q, scale)) - x))
    tl.store(y_ptr + row * K + offs, out.to(y_ptr.dtype.element_ty), mask)


_MIN_AMAX = float(torch.tensor(1e-5, dtype=torch.bfloat16))  # clamp(min=1e-5) on a bf16 tensor clamps at this bf16 value


class _FastActQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        y = torch.empty_like(x2)
        K = x2.shape[1]
        _actquant_kernel[(x2.shape[0],)](x2, y, K, _MIN_AMAX, BLOCK=triton.next_power_of_2(K), num_warps=8)
        return y.reshape(x.shape)

    @staticmethod
    def backward(ctx, g):  # straight-through, as x + (q - x).detach()
        return g


def fast_fake_quant_act(x):
    """OPT["fast_actquant"]: lowmem_models.fake_quant_act in one pass (bf16 CUDA input), identical output and gradient."""
    return _FastActQuant.apply(x)


def unpack(codes, dtype=torch.bfloat16):
    """uint8 (out, in // 4) -> (out, in) tensor of {-1, 0, +1} in `dtype`."""
    out, nb = codes.shape
    if OPT["fast_unpack"] and codes.is_cuda and codes.is_contiguous():  # one pass instead of shift, mask, convert, subtract; same values
        res = torch.empty((out, nb * 4), dtype=dtype, device=codes.device)
        _unpack_kernel[(triton.cdiv(codes.numel(), 1024),)](codes, res, codes.numel(), BLOCK=1024)
        return res
    return ((codes[..., None] >> _shifts(codes)) & 3).reshape(out, nb * 4).to(dtype).sub_(1)


def count_zeros(codes):
    return int((((codes[..., None] >> _shifts(codes)) & 3) == 1).sum())


# ---------------------------------------------------------------- kernels (copies of the dense ones, reading/writing packed bytes)
@triton.jit
def _sample_keys_packed(pk_ptr, m_ptr, r_ptr, c_ptr, rows_ptr, cols_ptr,
                        keys_ptr, counts_ptr, mean_ptr, nr, nc, bias_corr,
                        BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    ii = pid * BLOCK + tl.arange(0, BLOCK)
    rr = tl.load(rows_ptr + ii)
    cc = tl.load(cols_ptr + ii)
    mean = _corrected_mean(r_ptr, nr, bias_corr, BLOCK)
    if pid == 0:
        tl.store(mean_ptr, mean)
    flat = rr * nc + cc
    byte = tl.load(pk_ptr + flat // 4).to(tl.int32)
    p = (((byte >> ((flat % 4) * 2)) & 3) - 1).to(tl.float32)
    m = tl.load(m_ptr + flat).to(tl.float32)
    r = tl.load(r_ptr + rr) / bias_corr
    c = tl.load(c_ptr + cc) / bias_corr
    src = m / (tl.sqrt(r * c / mean) + 1e-12)
    movable = ~(((p > 0) & (src < 0)) | ((p < 0) & (src > 0)))
    tl.store(keys_ptr + ii, tl.where(movable, tl.abs(src), -1.0))
    tl.store(counts_ptr + pid, tl.sum(movable.to(tl.int32), axis=0))


@triton.jit
def _flip_packed(pk_ptr, m_ptr, r_ptr, c_ptr, mean_ptr, tau_ptr, flips_ptr,
                 n, nc, bias_corr, BYTES: tl.constexpr, SPARSE: tl.constexpr):
    bi = tl.program_id(0) * BYTES + tl.arange(0, BYTES)
    lane = tl.arange(0, 4)
    ii = bi[:, None] * 4 + lane[None, :]
    bmask = bi < n // 4
    mask = (bi[:, None] < n // 4) & (lane[None, :] < 4)
    byte = tl.load(pk_ptr + bi, bmask, other=1).to(tl.int32)
    p = (((byte[:, None] >> (lane[None, :] * 2)) & 3) - 1).to(tl.float32)
    m = tl.load(m_ptr + ii, mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + ii // nc, mask, other=0.0) / bias_corr
    c = tl.load(c_ptr + ii % nc, mask, other=0.0) / bias_corr
    mean = tl.load(mean_ptr)
    tau = tl.load(tau_ptr)
    src = m / (tl.sqrt(r * c / mean) + 1e-12)
    new = tl.minimum(tl.maximum(
        p + (src < -tau).to(tl.float32) - (src > tau).to(tl.float32), -1.0), 1.0)
    changed = mask & (new != p)
    code = (new + 1.0).to(tl.int32)
    packed = tl.sum(code << (lane[None, :] * 2), axis=1)
    if SPARSE:  # OPT["sparse_store"]: write only what changed (a flip touches ~1 in 1000 weights)
        byte_changed = tl.sum(changed.to(tl.int32), axis=1) > 0
        tl.store(pk_ptr + bi, packed.to(tl.uint8), bmask & byte_changed)
        tl.store(m_ptr + ii, tl.zeros_like(m).to(m_ptr.dtype.element_ty), changed)
    else:
        tl.store(pk_ptr + bi, packed.to(tl.uint8), bmask)
        tl.store(m_ptr + ii, tl.where(changed, 0.0, m).to(m_ptr.dtype.element_ty), mask)
    tl.atomic_add(flips_ptr, tl.sum(tl.sum(changed.to(tl.int32), axis=1), axis=0).to(tl.int64),
                  sem="relaxed")


@torch.no_grad()
def threshold_packed(codes, m, state, bias_corr, rate, every, gen=None):
    """K.threshold on packed weights: same two randint calls, movable mask, cap and interpolation."""
    nr, nc = codes.shape[0], codes.shape[1] * 4
    assert codes.is_contiguous() and m.is_contiguous() and m.shape == (nr, nc)
    rows = torch.randint(0, nr, (K.SAMPLE,), device=codes.device, generator=gen)
    cols = torch.randint(0, nc, (K.SAMPLE,), device=codes.device, generator=gen)
    keys = torch.empty(K.SAMPLE, device=codes.device, dtype=torch.float32)
    counts = torch.empty(K.SAMPLE // K.BLOCK, device=codes.device, dtype=torch.int32)
    mean = torch.empty((), device=codes.device, dtype=torch.float32)
    tau = torch.empty_like(mean)
    _sample_keys_packed[(K.SAMPLE // K.BLOCK,)](
        codes, m, state.row, state.col, rows, cols, keys, counts, mean,
        nr, nc, bias_corr, BLOCK=K.BLOCK)
    keys = torch.sort(keys).values
    K._quantile[(1,)](keys, counts, tau, rate, every, SAMPLE=K.SAMPLE,
                      COUNT_BLOCK=K.SAMPLE // K.BLOCK, enable_fp_fusion=False)
    return tau, mean


@torch.no_grad()
def flip_packed(codes, m, state, tau, flips, bias_corr, mean):
    assert codes.is_contiguous() and m.is_contiguous()
    assert flips.dtype == torch.int64, "use an int64 cumulative flip counter"
    n = codes.numel() * 4
    _flip_packed[(triton.cdiv(n // 4, BYTES),)](
        codes, m, state.row, state.col, mean, tau, flips,
        n, codes.shape[1] * 4, bias_corr, BYTES=BYTES, SPARSE=bool(OPT["sparse_store"]))


# ---------------------------------------------------------------- optimizer step on one matrix (mirrors tune.fused_fac_hook)
@torch.no_grad()
def update(mod, g):
    """Momentum / second-moment update with the dense weight gradient g, then (on check steps) the packed flip."""
    from . import tune  # lazy: tune creates a CUDA generator at import
    ctl = tune.CTL
    nr, nc = mod.shape
    int8m = OPT["m_int8"]
    if mod.m is None:
        if int8m:
            assert ctl["m_dtype"] == torch.bfloat16, "OPT['m_int8'] re-quantises a bf16 momentum"
            mod.m = torch.zeros(mod.shape, dtype=torch.int8, device=g.device)
            mod.ms = torch.full((nr,), 1e-30, dtype=torch.float32, device=g.device)
        else:
            mod.m = torch.zeros(mod.shape, dtype=ctl["m_dtype"], device=g.device)
        mod.vstate = K.State(*(torch.zeros(n, dtype=torch.float32, device=g.device) for n in (nr, nc, nr, nc)))
        mod.layer = tune.LAYER_IDS.setdefault(id(mod), len(tune.LAYER_IDS))
    seed = (ctl["sr_seed"] + ctl["step"] * 7919 + mod.layer * 104729) & 0x7fffffff
    gamma, beta2v = ctl["gamma"], ctl["beta2v"]
    if OPT["wgrad_every"] > 1:  # this gradient stands in for `gap` steps, so decay the averages as many times
        gap = 1 if mod.last_g is None else max(1, ctl["step"] - mod.last_g)
        mod.last_g = ctl["step"]
        if gap > 1:
            gamma, beta2v = 1 - (1 - gamma) ** gap, 1 - (1 - beta2v) ** gap
    assert (mod.m.dtype == torch.int8) == int8m, "OPT['m_int8'] changed after the momentum was created"
    m = _dequant_m(mod) if int8m else mod.m  # int8 mode: a transient bf16 copy for the kernels, re-quantised at the end
    K.mv_update(g, m, mod.vstate, gamma, beta2v, seed=seed, sr=ctl["m_sr"])
    if ctl["step"] % ctl["every"] == 0 and ctl["step"] >= ctl["flip_start"]:
        bc = 1 - (1 - ctl["beta2v"]) ** ctl["step"]
        tau, mean = threshold_packed(mod.codes, m, mod.vstate, bc, ctl["rate"], ctl["every"],
                                     tune.GEN if ctl["own_rng"] else None)
        if ctl["score_floor"]:
            tau = torch.clamp_min(tau, ctl["score_floor"] * (ctl["gamma"] / (2 - ctl["gamma"])) ** 0.5)
        flip_packed(mod.codes, m, mod.vstate, tau, ctl["flips"], bc, mean)
    if int8m:
        _quant_m(mod, m, tune.GEN if ctl["own_rng"] else None)


def _dequant_m(mod):
    return mod.m.to(torch.bfloat16) * mod.ms.to(torch.bfloat16)[:, None]


def _quant_m(mod, m16, gen=None):
    """int8 momentum with one fp32 scale per row and stochastic rounding (unbiased: floor(x + u))."""
    mf = m16.float()
    ms = (mf.abs().amax(dim=1) / 127.0).clamp_(min=1e-30)
    q = mf.div_(ms[:, None])
    q.add_(torch.rand(q.shape, device=q.device, generator=gen)).floor_().clamp_(-127, 127)
    mod.m.copy_(q.to(torch.int8))
    mod.ms.copy_(ms)


# ---------------------------------------------------------------- layer
class PackedLinearFn(torch.autograd.Function):
    """y = x @ (T * s)^T with T unpacked on the fly. Backward returns grad_x and feeds the dense weight gradient to update()."""

    @staticmethod
    def forward(ctx, x, mod):
        W = unpack(mod.codes, x.dtype)
        s = mod.s.to(x.dtype)
        y = F.linear(x * s, W) if mod.axis == "in" else F.linear(x, W) * s
        ctx.mod = mod
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, gy):
        (x,) = ctx.saved_tensors
        return _bwd(ctx.mod, x, gy), None


def _bwd(mod, x, gy):
    """grad_x for y = x @ (T * s)^T, and the weight gradient handed to the optimizer when this matrix needs one."""
    W = unpack(mod.codes, gy.dtype)
    s = mod.s.to(gy.dtype)
    x2 = x.reshape(-1, x.shape[-1])
    gy2 = gy.reshape(-1, gy.shape[-1])
    want = mod.needs_wgrad()  # always True unless the matrix is frozen or OPT["wgrad_every"] > 1
    if mod.axis == "in":
        gW = (gy2.t() @ (x2 * s)).contiguous() if want else None
        gx = (gy @ W) * s
    else:
        gys = gy * s
        gW = (gys.reshape(-1, gys.shape[-1]).t() @ x2).contiguous() if want else None
        gx = gys @ W
    del W
    if want:
        mod_update(mod, gW)
    return gx


def mod_update(mod, gW):  # indirection so tests can capture the gradient instead of running the GPU kernels
    update(mod, gW)


def _fake_quant_act(x):  # copy of lowmem_models.fake_quant_act (that module imports this one)
    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    q = (x * scale).round().clamp(-128, 127) / scale
    return x + (q - x).detach()


class PackedLinearInt8Fn(torch.autograd.Function):
    """Same math as BitNet-style fake-quantised activations followed by PackedLinearFn (axis 'out'), but the matmul is a real int8 x int8 -> int32 product
    (ternary weights are exact in int8).  The activation scale and rounding are computed exactly as lowmem_models.fake_quant_act does, in the input dtype,
    and backward rebuilds the same dequantised activations from the saved int8, so the weight gradient is unchanged (straight-through for grad_x)."""

    @staticmethod
    def forward(ctx, x, mod):
        x2 = x.reshape(-1, x.shape[-1])
        scale = 127.0 / x2.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        xq = (x2 * scale).round().clamp(-128, 127).to(torch.int8)
        acc = torch._int_mm(xq, unpack(mod.codes, torch.int8).t())  # (tokens, out) int32, exact
        y = (acc.float() / scale.float() * mod.s.float()).to(x.dtype)
        ctx.mod = mod
        ctx.save_for_backward(xq, scale)
        ctx.shape = x.shape
        return y.reshape(*x.shape[:-1], y.shape[-1])

    @staticmethod
    def backward(ctx, gy):
        xq, scale = ctx.saved_tensors
        x = (xq.to(gy.dtype) / scale.to(gy.dtype)).reshape(ctx.shape)
        return _bwd(ctx.mod, x, gy), None


class PackedTLinear(nn.Module):
    """Drop-in for the dense TLinear in absorb_probe: same forward math, weights stored 2-bit, optimizer step inside backward."""

    def __init__(self, T, s, axis):
        super().__init__()
        self.shape = tuple(T.shape)
        self.register_buffer("codes", pack(T))
        self.register_buffer("s", s)
        self.axis = axis
        self.m = self.vstate = self.layer = self.ms = None  # ms: per-row scale of the int8 momentum (OPT["m_int8"])
        self.last_g = None  # step of the last gradient this matrix was updated with (OPT["wgrad_every"])
        self.frozen = False  # True: dgrad only, no weight gradient and no flips (see speedups.restrict)
        self.int8_act = False  # True: int8 activation quantisation + int8 matmul (see speedups.int8_forward); replaces the fake-quant pre-hook

    def forward(self, x):
        if self.int8_act:
            if x.numel() // x.shape[-1] > 16:  # torch._int_mm needs more than 16 rows
                return PackedLinearInt8Fn.apply(x, self)
            return PackedLinearFn.apply(_fake_quant_act(x), self)  # tiny batches (decoding): same quantisation, bf16 matmul
        return PackedLinearFn.apply(x, self)

    def needs_wgrad(self):
        if self.frozen:
            return False
        k = OPT["wgrad_every"]
        if k <= 1:
            return True
        from . import tune
        ctl = tune.CTL
        return self.m is None or ctl["step"] % k == 0 or ctl["step"] % ctl["every"] == 0  # flip steps always see a fresh gradient

    # interface shared with TLinear
    @property
    def n_ternary(self):
        return self.shape[0] * self.shape[1]

    def zeros(self):
        return float(count_zeros(self.codes))

    def set_ternary(self, T):
        self.codes.copy_(pack(T))
        self.m = self.vstate = self.layer = self.last_g = self.ms = None

    def attach(self, hook):  # the update runs inside backward; nothing to register
        return []
