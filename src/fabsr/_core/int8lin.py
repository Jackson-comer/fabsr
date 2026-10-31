"""int8 matmul for the ternary layers, switched by the TERN_INT8 environment variable (off | fwd | fwd_dx; default off).

Ternary weights are exact in int8. Activations are quantized per token (absmax / 127) by one fused Triton pass, the product runs on the int8 tensor
cores (torch._int_mm), and one more fused pass rescales the int32 result to bf16.
  fwd      int8 forward; backward is the usual bf16 straight-through (grad_x = grad_out @ W, grad_w = grad_out^T @ x)
  fwd_dx   additionally runs grad_x = grad_out @ W in int8 (grad_out quantized per token); grad_w stays bf16
The optimizers never see any of this: the layer still owns the same bf16 ternary tensor, and the gradient comes back in its dtype.
"""
import os

import torch
import triton
import triton.language as tl

MODE = os.environ.get("TERN_INT8", "off")


@triton.jit
def _rowquant(x_ptr, q_ptr, s_ptr, K, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < K
    x = tl.load(x_ptr + row * K + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-5) / 127.0
    v = x / s
    q = (v + tl.where(v >= 0, 0.5, -0.5)).to(tl.int32)          # round half away from zero
    q = tl.minimum(tl.maximum(q, -127), 127)
    tl.store(q_ptr + row * K + offs, q.to(tl.int8), mask=mask)
    tl.store(s_ptr + row, s)


@triton.jit
def _dequant(y_ptr, s_ptr, o_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < N
    y = tl.load(y_ptr + row * N + col, mask=mask, other=0).to(tl.float32)
    s = tl.load(s_ptr + row)
    tl.store(o_ptr + row * N + col, (y * s).to(o_ptr.dtype.element_ty), mask=mask)


def quant_rows(x):
    """x: (m, k) bf16/fp16/fp32 contiguous -> (int8 (m, k), fp32 scale (m,))"""
    m, k = x.shape
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    s = torch.empty((m,), dtype=torch.float32, device=x.device)
    _rowquant[(m,)](x, q, s, k, BLOCK=triton.next_power_of_2(k))
    return q, s


def dequant(y, s, dtype):
    m, n = y.shape
    out = torch.empty((m, n), dtype=dtype, device=y.device)
    _dequant[(m, triton.cdiv(n, 1024))](y, s, out, n, BLOCK=1024)
    return out


class Int8Linear(torch.autograd.Function):
    """y = x @ w^T with x int8-quantized per token and w ternary (bf16 holding -1/0/+1)."""

    @staticmethod
    def forward(ctx, x, w, mode):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1]).contiguous()
        xq, xs = quant_rows(x2)
        wq = w.to(torch.int8)
        y = dequant(torch._int_mm(xq, wq.t()), xs, x.dtype)
        ctx.save_for_backward(x2, w, wq)
        ctx.mode, ctx.shp = mode, shp
        return y.reshape(*shp[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, go):
        x2, w, wq = ctx.saved_tensors
        go2 = go.reshape(-1, go.shape[-1]).contiguous()
        if ctx.mode == "fwd_dx":
            gq, gs = quant_rows(go2)
            gx = dequant(torch._int_mm(gq, wq), gs, go.dtype)
        else:
            gx = go2 @ w.to(go.dtype)
        gw = (go2.t() @ x2.to(go.dtype)).to(w.dtype)
        return gx.reshape(ctx.shp), gw, None


def linear(x, w):
    return Int8Linear.apply(x, w, MODE)
