"""Public API of fabsr: load a native-ternary model, wrap it in Fabsr, train with the usual loss.backward().

    import fabsr
    model, tok = fabsr.load_falcon_e()                    # Falcon-E-1B, ternary weights held as 2-bit codes
    opt = fabsr.Fabsr(model, flip_rate=1.5e-5, flip_rate_end=2e-6, steps=1500)
    for batch in data:
        loss = ce(model(batch).logits, labels)
        loss.backward()                                    # the weight update happens inside this call
        opt.step()                                         # advances the step counter and the flip-rate schedule

Everything heavy (torch, triton, transformers) is imported lazily, so `import fabsr` is cheap and works on a CPU-only machine.
"""
import importlib
import math
import os
import warnings

__all__ = ["Fabsr", "load_falcon_e", "ternary_layers", "save_codes", "load_codes", "rate_schedule"]

_CORE = {}
_ACTIVE = []  # the optimizer state is process-global (inherited from the research code), so only one Fabsr can be live
FLAGS = ("fast_unpack", "fast_actquant", "sparse_store")           # exact: same outputs as the plain path, only faster
EXPERIMENTAL = ("wgrad_every", "m_int8")                           # run, but not validated for training quality


def rate_schedule(step, start, end=None, steps=None):
    """Flip rate at a 1-based step: constant, or log-linear from `start` to `end` over `steps` (then held at `end`)."""
    if end is None or not steps:
        return start
    return start * (end / start) ** min(step / steps, 1.0)


def _core(name):
    """Import one vendored research module (fabsr/_core/<name>.py) on first use."""
    if name not in _CORE:
        _CORE[name] = importlib.import_module(f"fabsr._core.{name}")
    return _CORE[name]


def _hf_home():
    if "HF_HOME" not in os.environ:
        from huggingface_hub import constants
        os.environ["HF_HOME"] = constants.HF_HOME   # the library default, made explicit because the vendored loader reads it at import time


def load_falcon_e(repo="tiiuae/Falcon-E-1B-Base", *, act_quant=True, download=True):
    """Load a Falcon-E checkpoint with every projection as a 2-bit packed layer (no dense copy is ever built on the GPU).

    Returns (model, tokenizer). `download=True` fetches the packed `main` revision (about 0.7 GB for the 1B) into the Hugging Face cache if missing.
    act_quant keeps the model's own 8-bit per-token input quantisation, as in the original BitLinear (leave it on for Falcon-E).
    Tested with tiiuae/Falcon-E-1B-Base only. Embeddings, norms and the output head stay frozen bf16.
    """
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("fabsr needs a CUDA GPU (the update kernels are Triton)")
    _hf_home()
    L = _core("lowmem_models")
    if download:
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo, allow_patterns=["model.safetensors", "config.json", "generation_config.json", "tokenizer*", "special_tokens_map.json"])
    else:
        path = None
    L.FALCON_REPO = repo
    if path is not None:
        L.FALCON_GLOB = path
    model, _layers, tok, _ids = L.build_falcon(act_quant=act_quant)
    return model, tok


def ternary_layers(model):
    """The packed ternary layers of a model, in module order."""
    PT = _core("packed_ternary")
    return [m for m in model.modules() if isinstance(m, PT.PackedTLinear)]


class Fabsr:
    """Weight-flip optimizer for packed ternary layers. There is no optimizer.step() that updates weights: the update runs inside backward.

    flip_rate        target fraction of weights flipped per step (the tuned Falcon-E value is 1.5e-5); with flip_rate_end and steps it decays log-linearly
    momentum         gradient EMA rate (tuned 0.025); second_moment: row/column second-moment EMA rate (tuned 0.01); flip_every: flip check every N steps (tuned 4)
    checkpoint       turn on activation checkpointing (needed to fit 1B on 8 GB); model.train() is called either way
    fast             turn on the three exact speed flags (fast_unpack, fast_actquant, sparse_store); each can also be set on its own by name
    experimental     dict for flags that run but are NOT validated for quality: {"wgrad_every": k, "m_int8": True}
    train_layers / train_types   train only some layers or matrix types (see fabsr._core.speedups.restrict): "top:12", "12-23", "attn", "mlp", ...
    seed             reseed the flip-threshold sampler (None keeps the tuned scripts' default stream)
    Only one Fabsr can exist per process; creating a second one resets the first.
    """

    def __init__(self, model, flip_rate=1.5e-5, *, flip_rate_end=None, steps=None, momentum=0.025, second_moment=0.01, flip_every=4,
                 checkpoint=True, fast=False, fast_unpack=None, fast_actquant=None, sparse_store=None, experimental=None,
                 train_layers="", train_types="", seed=None):
        import torch
        PT, tune, SP = _core("packed_ternary"), _core("tune"), _core("speedups")
        self.layers = ternary_layers(model)
        if not self.layers:
            raise ValueError("no packed ternary layers found: build the model with fabsr.load_falcon_e()")
        if _ACTIVE:
            warnings.warn("a Fabsr optimizer already exists in this process; its state is replaced", stacklevel=2)
        self.model, self._start, self._end, self._steps = model, flip_rate, flip_rate_end, steps
        self.step_count = 1                       # the step the next backward() belongs to (1-based, like the tuned scripts)
        self.n_ternary = sum(m.n_ternary for m in self.layers)
        opts = {k: bool(fast) for k in FLAGS}
        for k, v in zip(FLAGS, (fast_unpack, fast_actquant, sparse_store)):
            if v is not None:
                opts[k] = bool(v)
        exp = dict(experimental or {})
        bad = set(exp) - set(EXPERIMENTAL)
        if bad:
            raise ValueError(f"unknown experimental flags {sorted(bad)}; known: {EXPERIMENTAL}")
        if exp:
            warnings.warn(f"experimental flags {sorted(exp)} are not validated for training quality", stacklevel=2)
        PT.OPT.update(wgrad_every=int(exp.get("wgrad_every", 1)), m_int8=bool(exp.get("m_int8", False)), **opts)
        tune.STATE.clear(); tune.VFAC.clear(); tune.V2.clear(); tune.LAYER_IDS.clear()
        dev = "cuda"
        tune.CTL.update(rule="quantile", fused=True, vfac=True, secmom=False, movable_q=True, m_sr=True, every=flip_every, gamma=momentum,
                        beta2v=second_moment, rate=rate_schedule(1, flip_rate, flip_rate_end, steps), m_dtype=torch.bfloat16, flip_start=0,
                        score_floor=0.0, own_rng=True, sr_seed=0, step=1, rownorm=False, diag=False,
                        flips=torch.zeros((), device=dev, dtype=torch.int64))
        if seed is not None:
            tune.GEN.manual_seed(seed)
            tune.CTL["sr_seed"] = int(seed)
        if checkpoint:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self._undo = SP.restrict(model, self.layers, train_layers, train_types)   # also keeps autograd alive through the frozen embedding
        model.train()
        _ACTIVE[:] = [self]

    @property
    def flip_rate(self):
        return _core("tune").CTL["rate"]

    @flip_rate.setter
    def flip_rate(self, value):
        """Override the rate by hand; also replaces the schedule from here on."""
        self._start, self._end, self._steps = float(value), None, None
        _core("tune").CTL["rate"] = float(value)

    def step(self):
        """Call after loss.backward(): moves to the next step and sets that step's flip rate."""
        self.step_count += 1
        tune = _core("tune")
        tune.CTL["step"] = self.step_count
        tune.CTL["rate"] = rate_schedule(self.step_count, self._start, self._end, self._steps)

    def zero_grad(self, set_to_none=True):
        """No-op, for loops written against torch.optim: packed layers hold no gradients."""

    def flipped_fraction(self):
        """Cumulative flips divided by the number of ternary weights (reads a GPU counter, so it synchronises)."""
        return float(_core("tune").CTL["flips"]) / self.n_ternary

    def state_dict(self):
        return {"step": self.step_count, "codes": [m.codes.cpu() for m in self.layers]}

    def load_state_dict(self, blob):
        """Restore the ternary codes and the step counter. The momentum is not saved: it restarts from zero, as in the tuned scripts' resume."""
        for m, c in zip(self.layers, blob["codes"]):
            m.codes.copy_(c)
        if "step" in blob:                      # written by state_dict(): the next step to run
            self.step_count = int(blob["step"])
        elif "steps" in blob:                   # written by save_codes() or the research scripts: steps completed
            self.step_count = int(blob["steps"]) + 1
        tune = _core("tune")
        tune.CTL["step"] = self.step_count
        tune.CTL["rate"] = rate_schedule(self.step_count, self._start, self._end, self._steps)

    def close(self):
        """Detach the training-only hooks and clear the optimizer state."""
        self._undo()
        tune = _core("tune")
        tune.STATE.clear(); tune.VFAC.clear(); tune.V2.clear(); tune.LAYER_IDS.clear()
        if _ACTIVE and _ACTIVE[0] is self:
            _ACTIVE.clear()


def save_codes(model, path, step=0):
    """Write the trained ternary codes (about 0.4 GB for the 1B) in the format the research scripts use: {'codes': [...], 'steps': n}."""
    import torch
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"steps": step, "codes": [m.codes.cpu() for m in ternary_layers(model)]}, tmp)
    os.replace(tmp, path)


def load_codes(model, path):
    """Load codes written by save_codes() (or by the research scripts) into a model from load_falcon_e()."""
    import torch
    blob = torch.load(path, map_location="cpu")
    layers = ternary_layers(model)
    if len(blob["codes"]) != len(layers):
        raise ValueError(f"{path} has {len(blob['codes'])} matrices, the model has {len(layers)}")
    for m, c in zip(layers, blob["codes"]):
        m.codes.copy_(c)
    return blob.get("steps", 0)
