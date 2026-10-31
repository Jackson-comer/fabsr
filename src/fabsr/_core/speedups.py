"""Opt-in speed toggles for Fabsr training on packed ternary models.  Every default leaves the training step exactly as it was.

  restrict(model, tl, layers, types)   train only some layers / matrix types.  Frozen matrices skip their weight gradient and never flip; if the lowest trainable
                                       layer is above 0, the autograd graph starts there, so the layers below run forward only (no backward, no stored activations).
                                       layers: ''=all | 'top:12' | '12-23' | '0,3,5'      types: ''=all | 'attn' | 'mlp' | comma list of q,k,v,o,gate,up,down
  chunked_ce(h, W, y, chunk)           cross-entropy from hidden states with the logits built chunk by chunk and recomputed in backward (no full fp32 logits)
  int8_forward(tl)                     real int8 matmuls for the forward (and the checkpoint recompute) instead of bf16 on fake-quantised activations

The flags that live inside the optimizer step (wgrad_every, m_int8, sparse_store) are in packed_ternary.OPT; see that module's docstring.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from . import absorb_probe as A

SHORT = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj",
         "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}
GROUPS = {"attn": ["q", "k", "v", "o"], "mlp": ["gate", "up", "down"]}


def parse_layers(spec, n):
    spec = (spec or "").strip()
    if not spec:
        return set(range(n))
    if spec.startswith("top:"):
        return set(range(n - int(spec[4:]), n))
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return set(range(int(a), int(b) + 1))
    return {int(v) for v in spec.split(",")}


def parse_types(spec):
    spec = (spec or "").strip()
    if not spec:
        return set(A.LINEARS)
    names = []
    for part in spec.split(","):
        names += GROUPS.get(part, [part])
    return {SHORT[v] for v in names}


def restrict(model, tl, layers="", types=""):
    """Returns undo(). Replaces absorb_probe.prepare_packed (it also keeps autograd alive for the packed layers)."""
    n = len(model.model.layers)
    per = len(A.LINEARS)
    assert len(tl) == n * per, (len(tl), n, per)
    S, T = parse_layers(layers, n), parse_types(types)
    assert S and all(0 <= i < n for i in S) and T, (layers, types)
    for i, m in enumerate(tl):
        m.frozen = not (i // per in S and A.LINEARS[i % per] in T)
    lo = min(S)
    if lo > 0:
        def pre(mod, args, kwargs):  # graph starts here: everything below is forward only
            if not torch.is_grad_enabled():
                return None
            if args:
                return (args[0].detach().requires_grad_(True),) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = kwargs["hidden_states"].detach().requires_grad_(True)
            return args, kwargs
        handle = model.model.layers[lo].register_forward_pre_hook(pre, with_kwargs=True)
    else:
        handle = model.get_input_embeddings().register_forward_hook(lambda m, i, o: o.requires_grad_(True))

    def undo():
        handle.remove()
        for m in tl:
            m.frozen = False
    return undo


def int8_forward(tl):
    """Replace the BitNet activation fake-quant pre-hook with a real int8 x int8 -> int32 matmul (packed_ternary.PackedLinearInt8Fn). Returns undo()."""
    assert hasattr(torch, "_int_mm"), "needs torch._int_mm (CUDA, sm80+)"
    saved = []
    for m in tl:
        assert m._forward_pre_hooks, "this model was built without act_quant, so there is no activation quantisation to replace"
        assert m.axis == "out", "int8 forward is written for per-output-row scales"
        saved.append(dict(m._forward_pre_hooks))
        m._forward_pre_hooks.clear()
        m.int8_act = True

    def undo():
        for m, h in zip(tl, saved):
            m._forward_pre_hooks.update(h)
            m.int8_act = False
    return undo


def chunked_ce(h, W, y, chunk=2048, ignore_index=-100):
    """Mean cross-entropy of softmax(h @ W.T) over targets y != ignore_index; h (..., d), W (vocab, d) frozen, y (...)."""
    h, y = h.reshape(-1, h.shape[-1]), y.reshape(-1)
    n = (y != ignore_index).sum().clamp(min=1)

    def f(hc, yc):
        return F.cross_entropy((hc @ W.t()).float(), yc, ignore_index=ignore_index, reduction="sum")
    tot = 0.0
    for i in range(0, h.shape[0], chunk):
        tot = tot + checkpoint(f, h[i:i + chunk], y[i:i + chunk], use_reentrant=False)
    return tot / n
