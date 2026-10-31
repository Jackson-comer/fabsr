"""Tune the Bop recipe at 3M without changing memory per weight (bf16 weights + one bf16 gradient average, update in backward).

Options explored (all keep per-weight memory the same):
  rule=fixed       absolute threshold tau on the gradient average
  rule=quantile    each layer's threshold is chosen so about `rate` of weights qualify per step (estimated from a random sample)
  chan=True        a learnable per-output-channel scale (n_out floats per matrix)
  stream=bf16      residual stream in bf16 instead of fp32
  gamma_end        geometric schedule for the gradient-average rate gamma
  flip_start       steps before any flip is allowed
Usage: python tune.py configs.json   (a JSON list of config dicts; results are appended to tune_results.jsonl)
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import int8lin
from . import twin

CTL = {"rule": "fixed", "tau": 1e-6, "rate": 1e-3, "gamma": 1e-3, "every": 8, "flip_start": 0, "step": 0, "flips": None,
       "rownorm": False, "beta2": 1e-3, "secmom": False, "beta2v": 1e-3, "diag": False, "diag_every": 250, "movable_q": False, "m_dtype": torch.bfloat16, "own_rng": True, "score_floor": 0.0, "fused": False, "m_sr": False, "vfac": False, "sr_seed": 0, "legacy_sr_seed": os.environ.get("TERN_LEGACY_SR") == "1"}
STATE = {}
GEN = torch.Generator(device="cuda")  # threshold-sampling RNG, reseeded in run()
V2 = {}  # id(param) -> fp32 per-weight second moment of the gradient (Adam-style normalization; +4 bytes per weight)

DSTATE = {}  # id(param) -> persistent diagnostic masks (diagnostic runs only; about 8 bytes per weight)
DSUM = {}    # layer name -> GPU-side sums since the last dump
DLOG = []    # one record per dump
NAMES = {}   # id(param) -> layer name such as blocks.3.qkv
LAYER_IDS = {}  # stable module-order indices for independent factored SR streams


@torch.no_grad()
def diag_update(p, src, thr, new, changed, qstats):
    """Flip diagnostics, called at every check before the weights change (diag runs only)."""
    st = DSTATE.get(id(p))
    if st is None:
        st = DSTATE[id(p)] = {"prev": torch.zeros_like(p, dtype=torch.bool),
                              "last_step": torch.full(p.shape, -10**9, dtype=torch.int32, device=p.device),
                              "last_dir": torch.zeros_like(p, dtype=torch.int8),
                              "count": torch.zeros_like(p, dtype=torch.int16)}
    step = CTL["step"]
    qual = (src < -thr) | (src > thr)  # weights whose accumulator passes the threshold
    delta = (new - p).to(torch.int8)
    recent = (step - st["last_step"]) <= 20 * CTL["every"]
    rev = changed & recent & (delta == -st["last_dir"])  # flips that undo a flip made in the last 20 checks
    tau, q50, q99 = qstats if qstats is not None else (torch.zeros((), device=p.device),) * 3
    vals = {"n_qual": qual.sum(), "n_blocked": (qual & ~changed).sum(), "n_changed": changed.sum(),
            "n_persist": (qual & st["prev"]).sum(), "n_rev": rev.sum(), "tau": tau, "q50": q50, "q99": q99}
    s = DSUM.setdefault(NAMES[id(p)], {"checks": 0})
    s["checks"] += 1
    for k, v in vals.items():
        s[k] = s.get(k, 0) + v
    st["prev"] = qual
    st["last_step"] = torch.where(changed, torch.full_like(st["last_step"], step), st["last_step"])
    st["last_dir"] = torch.where(changed, delta, st["last_dir"])
    st["count"] += changed.to(torch.int16)


@torch.no_grad()
def dump_diag(step, weights_by_name):
    rec = {"step": step, "layers": {}}
    for name, s in DSUM.items():
        g = {k: (v.item() if torch.is_tensor(v) else v) for k, v in s.items()}
        w = weights_by_name[name]
        n, checks = w.numel(), g["checks"]
        rec["layers"][name] = {
            "realized_per_step": g["n_changed"] / (n * checks * CTL["every"]),
            "target_per_step": CTL["rate"],
            "blocked_of_qual": g["n_blocked"] / max(1, g["n_qual"]),
            "persist_of_qual": g["n_persist"] / max(1, g["n_qual"]),
            "reversal_of_flips": g["n_rev"] / max(1, g["n_changed"]),
            "tau": g["tau"] / checks, "q50": g["q50"] / checks, "q99": g["q99"] / checks,
            "frac_neg": (w == -1).float().mean().item(), "frac_zero": (w == 0).float().mean().item()}
    DSUM.clear()
    DLOG.append(rec)
VFAC = {}  # id(param) -> (R, C) in PyTorch, or bop_kernel_fac.State in fused mode
VSTATE = {}  # id(param) -> fp32 per-output-row running mean of squared gradient (n_out floats per matrix)


class RMS(nn.Module):
    """RMSNorm that computes in fp32 and returns the input's dtype."""

    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d, device="cuda"))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w).to(x.dtype)


class Lin(nn.Module):
    def __init__(self, n_in, n_out, extra, chan, p0):
        super().__init__()
        u = torch.rand(n_out, n_in, device="cuda")
        t = torch.where(u < p0, 0.0, torch.where(u < p0 + (1 - p0) / 2, -1.0, 1.0))
        self.weight = nn.Parameter(t.to(torch.bfloat16))
        self.scale = n_in**-0.5 * extra
        self.gain = nn.Parameter(torch.ones(n_out, device="cuda")) if chan else None

    def forward(self, x):
        y = (F.linear(x, self.weight) if int8lin.MODE == "off" else int8lin.linear(x, self.weight)) * self.scale   # TERN_INT8=fwd switches the matmul to int8
        return y if self.gain is None else y * self.gain.to(y.dtype)


class Block(nn.Module):
    def __init__(self, d, heads, blocks, chan, p0):
        super().__init__()
        res = (2 * blocks) ** -0.5
        self.heads = heads
        self.n1, self.n2 = RMS(d), RMS(d)
        self.qkv, self.o = Lin(d, 3 * d, 1.0, chan, p0), Lin(d, d, res, chan, p0)
        self.up, self.down = Lin(d, 4 * d, 1.0, chan, p0), Lin(4 * d, d, res, chan, p0)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, d)
        x = x + self.o(a)
        return x + self.down(F.relu(self.up(self.n2(x))))


class Model(nn.Module):
    def __init__(self, d, blocks, heads, chan, stream, p0):
        super().__init__()
        self.stream = stream
        self.emb, self.pos = nn.Embedding(twin.VOCAB, d, device="cuda"), nn.Embedding(twin.T, d, device="cuda")
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.blocks = nn.ModuleList(Block(d, heads, blocks, chan, p0) for _ in range(blocks))
        self.norm = RMS(d)

    def forward(self, idx):
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        if self.stream == "bf16":
            x = x.to(torch.bfloat16)
        for blk in self.blocks:
            x = blk(x)
        return F.linear(self.norm(x), self.emb.weight)


@torch.no_grad()
def fused_hook(p):
    """Dispatch factored F3+F1 or the original per-weight F3+F1 Triton path."""
    if CTL["vfac"]:
        return fused_fac_hook(p)
    from . import bop_kernel as K
    if p.grad is None:
        return
    m = STATE.get(id(p))
    if m is None:
        m = STATE[id(p)] = torch.zeros(p.shape, dtype=CTL["m_dtype"], device=p.device)
        V2[id(p)] = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
    v2 = V2[id(p)]
    if CTL["legacy_sr_seed"]:  # the studies up to the Fabsr work used one seed for every layer; kept to reproduce them
        seed = CTL["step"] * 7919 + len(STATE)
    else:
        layer = LAYER_IDS.setdefault(id(p), len(LAYER_IDS))
        seed = (CTL["sr_seed"] + CTL["step"] * 7919 + layer * 104729) & 0x7fffffff
    K.mv_update(p.grad, m, v2, CTL["gamma"], CTL["beta2v"], seed=seed, sr=CTL["m_sr"])
    if CTL["step"] % CTL["every"] == 0 and CTL["step"] >= CTL["flip_start"]:
        bc = 1 - (1 - CTL["beta2v"]) ** CTL["step"]
        tau = K.threshold(p, m, v2, bc, CTL["rate"], CTL["every"], GEN if CTL["own_rng"] else None)
        if CTL["score_floor"]:
            tau = torch.clamp_min(tau, CTL["score_floor"] * (CTL["gamma"] / (2 - CTL["gamma"])) ** 0.5)
        K.flip(p, m, v2, tau.reshape(1), CTL["flips"], bc)
    p.grad = None


@torch.no_grad()
def fused_fac_hook(p):
    """Factored F3+F1; original per-weight F3 kernel and RNG remain unchanged."""
    from . import bop_kernel_fac as K
    if p.grad is None:
        return
    m = STATE.get(id(p))
    if m is None:
        m = STATE[id(p)] = torch.zeros(p.shape, dtype=CTL["m_dtype"], device=p.device)
        VFAC[id(p)] = K.make_state(p)
    state = VFAC[id(p)]
    layer = LAYER_IDS.setdefault(id(p), len(LAYER_IDS))
    seed = (CTL["sr_seed"] + CTL["step"] * 7919 + layer * 104729) & 0x7fffffff
    K.mv_update(p.grad, m, state, CTL["gamma"], CTL["beta2v"],
                seed=seed, sr=CTL["m_sr"])
    if CTL["step"] % CTL["every"] == 0 and CTL["step"] >= CTL["flip_start"]:
        bc = 1 - (1 - CTL["beta2v"]) ** CTL["step"]
        tau, mean = K.threshold(p, m, state, bc, CTL["rate"], CTL["every"],
                                GEN if CTL["own_rng"] else None)
        if CTL["score_floor"]:
            tau = torch.clamp_min(tau, CTL["score_floor"] * (CTL["gamma"] / (2 - CTL["gamma"])) ** 0.5)
        K.flip(p, m, state, tau, CTL["flips"], bc, mean)
    p.grad = None


def sr_bf16(x):
    """Stochastic rounding of an fp32 tensor to bf16 (add 16 random low bits, then truncate)."""
    xi = x.contiguous().view(torch.int32)
    r = torch.randint(0, 1 << 16, xi.shape, device=x.device, dtype=torch.int32)
    return ((xi + r) & -65536).view(torch.float32).to(torch.bfloat16)


@torch.no_grad()
def hook(p):
    if CTL["fused"]:
        return fused_hook(p)
    if p.grad is None:
        return
    m = STATE.get(id(p))
    if m is None:  # allocate once (setdefault would build a throwaway zero tensor on every call)
        m = STATE[id(p)] = torch.zeros(p.shape, dtype=CTL["m_dtype"], device=p.device)
    if CTL["m_sr"] and m.dtype == torch.bfloat16:  # stochastic rounding keeps the bf16 average decaying (same idea as the Triton path)
        m.copy_(sr_bf16(torch.lerp(m.float(), p.grad.float(), CTL["gamma"])))
    else:
        m.lerp_(p.grad.to(m.dtype), CTL["gamma"])
    if CTL["rownorm"]:
        v = VSTATE.get(id(p))
        if v is None:
            v = VSTATE[id(p)] = torch.zeros(p.shape[0], device=p.device)
        v.lerp_(p.grad.pow(2).mean(1, dtype=torch.float32), CTL["beta2"])
    if CTL["secmom"]:
        v2 = V2.get(id(p))
        if v2 is None:
            v2 = V2[id(p)] = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
        v2.lerp_(p.grad.float().pow(2), CTL["beta2v"])
    if CTL["vfac"]:
        fr = VFAC.get(id(p))
        if fr is None:
            fr = VFAC[id(p)] = (torch.zeros(p.shape[0], device=p.device), torch.zeros(p.shape[1], device=p.device))
        g2 = p.grad.float().pow(2)
        fr[0].lerp_(g2.mean(1), CTL["beta2v"])
        fr[1].lerp_(g2.mean(0), CTL["beta2v"])
    if CTL["step"] % CTL["every"] == 0 and CTL["step"] >= CTL["flip_start"]:
        if CTL["rownorm"]:  # per-row scale, bias corrected; thresholds are in units of each row's gradient rms
            scale = torch.sqrt(VSTATE[id(p)] / (1 - (1 - CTL["beta2"]) ** CTL["step"]) + 1e-20)
        src = m
        if CTL["secmom"]:  # per-weight Adam-style normalization: compare m / sqrt(v), bias corrected
            src = m.float() / (torch.sqrt(V2[id(p)] / (1 - (1 - CTL["beta2v"]) ** CTL["step"])) + 1e-12)
        elif CTL["vfac"]:  # same normalization from the factored estimate v_ij ~ R_i * C_j / mean(R)
            bcv = 1 - (1 - CTL["beta2v"]) ** CTL["step"]
            R, C = VFAC[id(p)][0] / bcv, VFAC[id(p)][1] / bcv
            src = m.float() / (torch.sqrt(R[:, None] * C[None, :] / R.mean().clamp_min(1e-30)) + 1e-12)
        qstats = None
        if CTL["rule"] == "fixed":
            tau = CTL["tau"]
        else:  # quantile: about `rate` of weights per step qualify; one check covers `every` steps of evidence
            n_out, n_in = m.shape
            gen = GEN if CTL["own_rng"] else None  # keep threshold sampling off the training-data RNG stream
            rows = torch.randint(0, n_out, (65536,), device=m.device, generator=gen)
            cols = torch.randint(0, n_in, (65536,), device=m.device, generator=gen)
            sample = src[rows, cols].abs().float()
            if CTL["rownorm"]:
                sample = sample / scale[rows]
            if CTL["movable_q"]:  # F1: rank only weights that can actually move, and aim the budget at those
                ps, ss = p[rows, cols], src[rows, cols]
                mov = ~(((ps > 0) & (ss < 0)) | ((ps < 0) & (ss > 0)))  # blocked: wants to pass +1 or -1
                f_mov = mov.float().mean().clamp_min(0.05)
                sample = sample[mov]
                q = 1 - torch.clamp(CTL["rate"] * CTL["every"] / f_mov, max=0.5)
                tau = torch.quantile(sample, q) if sample.numel() else torch.full((), float("inf"), device=m.device)
            else:
                tau = torch.quantile(sample, 1 - min(0.5, CTL["rate"] * CTL["every"]))
            if CTL["diag"]:
                qs = torch.quantile(sample, torch.tensor([0.5, 0.99], device=sample.device))
                qstats = (tau, qs[0], qs[1])
        if CTL["score_floor"] and CTL["rule"] != "fixed":  # m/sqrt(v) of pure-noise gradients has sd ~ sqrt(gamma / (2 - gamma))
            tau = torch.clamp_min(tau, CTL["score_floor"] * (CTL["gamma"] / (2 - CTL["gamma"])) ** 0.5)
        thr = tau * scale[:, None] if CTL["rownorm"] else tau
        new = (p + (src < -thr).to(p.dtype) - (src > thr).to(p.dtype)).clamp_(-1, 1)
        changed = new != p
        if CTL["diag"]:
            diag_update(p, src, thr, new, changed, qstats)
        CTL["flips"] += changed.sum()
        p.copy_(new)
        m.masked_fill_(changed, 0)
    p.grad = None


def gpu_stats():
    """GPU clocks, power, temperature, utilization from nvidia-smi (empty dict if unavailable)."""
    try:
        import subprocess
        o = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip().split(", ")
        keys = ["sm_mhz", "mem_mhz", "power_w", "temp_c", "util_pct", "mem_used_mib"]
        return {k: (float(v) if v.replace(".", "", 1).isdigit() else None) for k, v in zip(keys, o)}
    except Exception:
        return {}


RICH = {"step": 0, "sec": 0.0}


def rich_point(step, batch, seconds, weights, flips):
    """Extra per-evaluation measurements for the fair-comparison runs."""
    dt, ds = seconds - RICH["sec"], step - RICH["step"]
    pt = {"tokens": step * batch * twin.T, "tok_s_interval": (ds * batch * twin.T / dt) if dt > 0 else None,
          "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1e9, "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
          "gpu": gpu_stats()}
    if weights is not None:
        pt["zero_frac"] = float(np.mean([(w == 0).float().mean().item() for w in weights]))
    if flips is not None:
        pt["flips_total"] = float(flips)
    RICH["step"], RICH["sec"] = step, seconds
    return pt


def run(cfg, train_tokens, val_tokens):
    d, blocks, heads = twin.SIZES[cfg.get("size", "3M")]
    steps, batch, seed = cfg["steps"], cfg.get("batch", 4), cfg.get("seed", 0)
    lr0, warmup = cfg.get("lr", 1e-3), cfg.get("warmup", 100)
    gamma0 = cfg["gamma"]
    gamma1 = cfg.get("gamma_end", gamma0)
    rate0 = cfg.get("rate", 1e-3)
    rate1 = cfg.get("rate_end", rate0)  # geometric decay of the target flip rate over training
    cool = cfg.get("rate_cool_start")  # run fraction where the taper to rate_end begins; None = decay over the whole run
    CTL.update(rule=cfg.get("rule", "fixed"), tau=cfg.get("tau", 1e-6), rate=cfg.get("rate", 1e-3), gamma=gamma0,
               every=cfg.get("every", 8), flip_start=cfg.get("flip_start", 0), step=0,
               rownorm=cfg.get("rownorm", False), beta2=cfg.get("beta2", 1e-3),
               secmom=cfg.get("secmom", False), beta2v=cfg.get("beta2v", 1e-3),
               diag=cfg.get("diag", False), diag_every=cfg.get("diag_every", 250),
               movable_q=cfg.get("movable_q", False),
               m_dtype=torch.float32 if cfg.get("m_fp32", False) else torch.bfloat16, own_rng=cfg.get("own_rng", True), score_floor=cfg.get("score_floor", 0.0),
               fused=cfg.get("fused", False), m_sr=cfg.get("m_sr", False), vfac=cfg.get("vfac", False), sr_seed=seed)
    if CTL["fused"]:
        assert CTL["rule"] == "quantile" and (CTL["secmom"] != CTL["vfac"]) and CTL["movable_q"] and not CTL["rownorm"] and not CTL["diag"], "fused requires F1 and exactly one of secmom/vfac"
    STATE.clear()
    VSTATE.clear()
    VFAC.clear()
    V2.clear()
    CTL["flips"] = torch.zeros((), device="cuda", dtype=torch.int64 if CTL["fused"] and CTL["vfac"] else torch.float32)
    torch.manual_seed(seed)
    GEN.manual_seed(seed + 12345)
    dgen = None if cfg.get("data_seed") is None else torch.Generator(device="cuda").manual_seed(cfg["data_seed"])  # batch order independent of the method
    RICH.update(step=0, sec=0.0)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    model = Model(d, blocks, heads, cfg.get("chan", False), cfg.get("stream", "fp32"), cfg.get("p0", 1 / 3))
    weights = [m.weight for m in model.modules() if isinstance(m, Lin)]
    weights_by_name = {n: m.weight for n, m in model.named_modules() if isinstance(m, Lin)}
    NAMES.clear()
    NAMES.update({id(w): n for n, w in weights_by_name.items()})
    LAYER_IDS.clear()
    LAYER_IDS.update({id(w): i for i, w in enumerate(weights)})
    DSTATE.clear()
    DSUM.clear()
    DLOG.clear()
    for w in weights:
        w.register_post_accumulate_grad_hook(hook)
    ids = {id(w) for w in weights}
    fp_params = [p for p in model.parameters() if id(p) not in ids]
    opt = torch.optim.Adam(fp_params, lr=lr0)
    n_weights = sum(w.numel() for w in weights)

    curve, recent, t0, flips_at_3 = [], [], None, 0.0
    eval_every = max(1, steps // 4)
    eval_every = cfg.get("eval_every", eval_every)
    t_start, eval_time = time.time(), 0.0
    for step in range(1, steps + 1):
        CTL["step"] = step
        CTL["gamma"] = gamma0 * (gamma1 / gamma0) ** (step / steps)
        taper = step / steps if cool is None else max(0.0, (step / steps - cool) / (1 - cool))
        CTL["rate"] = rate0 * (rate1 / rate0) ** taper
        lr = lr0 * min(1.0, step / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        starts = torch.randint(0, len(train_tokens) - twin.T - 1, (batch,), device="cuda", generator=dgen)
        if cfg.get("debug_starts") is not None and step <= 5:
            cfg["debug_starts"].append(starts.tolist())
        data = train_tokens[starts[:, None] + torch.arange(twin.T + 1, device="cuda")].long()  # tokens may be stored as int16
        x, y = data[:, :-1], data[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, twin.VOCAB), y.reshape(-1))
        model.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        recent.append(loss.detach())  # stays on the device: no per-step host sync
        if step % 100 == 0 and not bool(torch.isfinite(torch.stack(recent[-100:])).all()):
            return {"cfg": cfg, "error": f"non-finite loss near step {step}"}
        if CTL["diag"] and step % CTL["diag_every"] == 0:
            dump_diag(step, weights_by_name)
        if step == 3:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0, flips_at_3 = time.time(), CTL["flips"].item()
        if not cfg.get("no_eval") and (step % eval_every == 0 or step == steps):
            torch.cuda.synchronize()
            t_eval = time.time()
            val_loss = twin.evaluate(model, val_tokens, n_windows=cfg.get("eval_windows", 128))
            pt = {"step": step, "val_loss": val_loss, "train_loss": float(torch.stack(recent[-eval_every:]).float().mean()),
                  "seconds": (t_eval - t_start) - eval_time}  # wall clock with evaluation time excluded
            if cfg.get("rich_log"):
                pt.update(rich_point(step, batch, pt["seconds"], weights, CTL["flips"].item()))
            curve.append(pt)
            eval_time += time.time() - t_eval
            if cfg.get("log_path"):  # one line per evaluation, so a crash keeps everything so far
                with open(cfg["log_path"], "a") as f:
                    f.write(json.dumps(curve[-1]) + "\n")
            if cfg.get("save_path") and cfg.get("ckpt_every_eval"):  # rolling checkpoint
                torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                            "cfg": dict(cfg, vocab=twin.VOCAB, method="bop"), "curve": curve}, cfg["save_path"])
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() - base
    out = {"cfg": cfg, "final_val_loss": curve[-1]["val_loss"] if curve else None, "curve": curve,
           "tokens_per_s": (steps - 3) * batch * twin.T / elapsed,
           "peak_gb": peak / 1e9, "weights_gb": twin.nbytes(weights) / 1e9,
           "grad_average_gb": twin.nbytes(STATE.values()) / 1e9, "n_weights": n_weights,
           "flip_rate": (CTL["flips"].item() - flips_at_3) / n_weights / max(1, steps - 3),
           "zero_frac": float(np.mean([(w == 0).float().mean().item() for w in weights]))}
    if CTL["diag"]:  # per-layer histogram of how many times each weight flipped: 0, 1, 2, 3-4, 5-9, 10+
        hist = {}
        for name, w in weights_by_name.items():
            c = DSTATE[id(w)]["count"]
            hist[name] = [(c == 0).float().mean().item(), (c == 1).float().mean().item(), (c == 2).float().mean().item(),
                          ((c >= 3) & (c <= 4)).float().mean().item(), ((c >= 5) & (c <= 9)).float().mean().item(),
                          (c >= 10).float().mean().item()]
        out["diag"], out["flip_hist"] = list(DLOG), hist
    if cfg.get("save_path"):  # weights, model settings, and the loss curve, for reloading and sampling
        torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "cfg": dict(cfg, vocab=twin.VOCAB, method="bop"), "curve": curve}, cfg["save_path"])
    del model, opt
    STATE.clear()
    VSTATE.clear()
    VFAC.clear()
    V2.clear()
    torch.cuda.empty_cache()
    return out


def main():
    configs = json.load(open(sys.argv[1]))
    train_tokens = torch.from_numpy(np.load("train_tokens.npy").astype(np.int64)).cuda()
    val_tokens = torch.from_numpy(np.load("val_tokens.npy").astype(np.int64)).cuda()
    for cfg in configs:
        r = run(cfg, train_tokens, val_tokens)
        with open("tune_results.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")
        label = {k: v for k, v in cfg.items() if k not in ("steps",)}
        if "error" in r:
            print(label, r["error"], flush=True)
        else:
            print(f"val {r['final_val_loss'] if r['final_val_loss'] is not None else float('nan'):.3f}  flip {r['flip_rate']:.1e}  zero {r['zero_frac']:.2f}  "
                  f"{r['tokens_per_s']:.0f} tok/s  peak {r['peak_gb']:.2f} GB  {label}", flush=True)


if __name__ == "__main__":
    main()
