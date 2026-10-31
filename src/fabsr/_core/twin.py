"""The 1B recipe at any size, with measured memory and speed, plus 3M twin runs against dense and latent baselines.

Recipe (same at every size): pre-norm blocks, attention QKV/O and MLP up/down as the ternary layers, output scale
1/sqrt(fan-in), residual-branch scale (2*blocks)^-0.5 on the O and down projections, context 256, TinyStories BPE vocab 2,048,
1,024 tokens per step (4 sequences), bf16 autocast. Bop: bf16 ternary weights, bf16 gradient average, ternary ladder rule,
flip check every 8th step, applied layer by layer during backward. fp32 embedding, position, and norms trained with Adam.
Baselines (3M only): dense fp32 Adam, and latent-weight ternary (STE) with Adam.

  python twin.py --mode ladder         memory and speed for 3M ... 1B (short runs)
  python twin.py --mode twin           3M: dense, latent, and a Bop threshold grid
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB, T, BATCH = 2048, 256, 4
LR, WARMUP = 1e-3, 100
SIZES = {"3M": (192, 6, 3), "12M": (384, 7, 6), "50M": (640, 10, 10),
         "135M": (768, 19, 12), "350M": (1280, 18, 20), "1B": (2048, 20, 16)}
CTL = {"every": 8, "step": 0, "tau": 1e-6, "gamma": 1e-3, "flips": None}
STATE = {}  # id(param) -> bf16 gradient average


class Lin(nn.Module):
    def __init__(self, n_in, n_out, method, extra=1.0):
        super().__init__()
        self.method, self.extra = method, extra
        self.scale = n_in**-0.5 * extra
        if method == "bop":
            self.weight = nn.Parameter(torch.randint(-1, 2, (n_out, n_in), device="cuda").to(torch.bfloat16))
        else:
            self.weight = nn.Parameter(torch.randn(n_out, n_in, device="cuda") * n_in**-0.5)

    def quantized(self):
        w = self.weight
        return torch.clamp(torch.round(w / w.abs().mean().clamp_min(1e-8)), -1, 1)

    def forward(self, x):
        if self.method == "latent":
            w = self.weight
            return F.linear(x, w + (self.quantized() - w).detach()) * self.scale
        if self.method == "dense":
            return F.linear(x, self.weight) * self.extra
        return F.linear(x, self.weight) * self.scale


class Block(nn.Module):
    def __init__(self, d, heads, blocks, method):
        super().__init__()
        res = (2 * blocks) ** -0.5
        self.heads = heads
        self.n1, self.n2 = nn.RMSNorm(d, device="cuda"), nn.RMSNorm(d, device="cuda")
        self.qkv, self.o = Lin(d, 3 * d, method), Lin(d, d, method, res)
        self.up, self.down = Lin(d, 4 * d, method), Lin(4 * d, d, method, res)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, d)
        x = x + self.o(a)
        return x + self.down(F.relu(self.up(self.n2(x))))


class Model(nn.Module):
    def __init__(self, d, blocks, heads, method):
        super().__init__()
        self.emb, self.pos = nn.Embedding(VOCAB, d, device="cuda"), nn.Embedding(T, d, device="cuda")
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.blocks = nn.ModuleList(Block(d, heads, blocks, method) for _ in range(blocks))
        self.norm = nn.RMSNorm(d, device="cuda")

    def forward(self, idx):
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return F.linear(self.norm(x), self.emb.weight)


@torch.no_grad()
def bop_hook(p):
    if p.grad is None:
        return
    m = STATE.setdefault(id(p), torch.zeros_like(p))
    m.lerp_(p.grad, CTL["gamma"])
    if CTL["step"] % CTL["every"] == 0:
        tau = CTL["tau"]
        new = (p + (m < -tau).to(p.dtype) - (m > tau).to(p.dtype)).clamp_(-1, 1)
        changed = new != p
        CTL["flips"] += changed.sum()
        p.copy_(new)
        m.masked_fill_(changed, 0)
    p.grad = None


@torch.no_grad()
def evaluate(model, val, n_windows=128, chunk=32):
    starts = torch.linspace(0, len(val) - T - 2, n_windows).long()
    losses = []
    model.eval()
    for i in range(0, n_windows, chunk):
        s = starts[i:i + chunk]
        x = torch.stack([val[j:j + T] for j in s])
        y = torch.stack([val[j + 1:j + T + 1] for j in s])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        losses.append(F.cross_entropy(logits.float().view(-1, VOCAB), y.reshape(-1)).item())
    model.train()
    return float(np.mean(losses))


def nbytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


def train(size, method, tau, gamma, every, steps, seed, train_tokens, val_tokens, eval_every):
    d, blocks, heads = SIZES[size]
    CTL.update(every=every, tau=tau, gamma=gamma, step=0)
    STATE.clear()
    torch.manual_seed(seed)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    model = Model(d, blocks, heads, method)
    lins = [m for m in model.modules() if isinstance(m, Lin)]
    weights = [m.weight for m in lins]
    n_weights = sum(w.numel() for w in weights)
    if method == "bop":
        for w in weights:
            w.register_post_accumulate_grad_hook(bop_hook)
        ids = {id(w) for w in weights}
        fp_params = [p for p in model.parameters() if id(p) not in ids]
        opt = torch.optim.Adam(fp_params, lr=LR)
        CTL["flips"] = torch.zeros((), device="cuda")
    else:
        fp_params = [p for p in model.parameters() if all(p is not w for w in weights)]
        opt = torch.optim.Adam(model.parameters(), lr=LR)
    mem_build = torch.cuda.memory_allocated() - base

    curve, recent = [], []
    t0, last_flips, last_step = None, 0.0, 3
    for step in range(1, steps + 1):
        CTL["step"] = step
        lr = LR * min(1.0, step / WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        starts = torch.randint(0, len(train_tokens) - T - 1, (BATCH,), device="cuda")
        batch = train_tokens[starts[:, None] + torch.arange(T + 1, device="cuda")]
        x, y = batch[:, :-1], batch[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, VOCAB), y.reshape(-1))
        model.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        recent.append(loss.item())
        if not math.isfinite(recent[-1]):
            return {"size": size, "method": method, "error": f"non-finite loss at step {step}"}
        if step == 3:  # warm: optimizer and Bop state now exist
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            flips_at_3 = CTL["flips"].item() if method == "bop" else 0.0
        if step % eval_every == 0 or step == steps:
            point = {"step": step, "train_loss": float(np.mean(recent[-eval_every:])),
                     "val_loss": evaluate(model, val_tokens)}
            curve.append(point)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() - base

    w_bytes, fp_bytes = nbytes(weights), nbytes(fp_params)
    m_bytes = nbytes(STATE.values()) if method == "bop" else 0
    adam_bytes = 2 * (fp_bytes if method == "bop" else w_bytes + fp_bytes)
    grad_bytes = 0 if method == "bop" else w_bytes + fp_bytes
    out = {"size": size, "method": method, "tau": tau, "gamma": gamma, "every": every, "seed": seed, "steps": steps,
           "ternary_or_linear_params": n_weights, "tokens_per_s": (steps - 3) * BATCH * T / elapsed,
           "gb": {"weights": w_bytes / 1e9, "fp_params": fp_bytes / 1e9, "grad_average": m_bytes / 1e9,
                  "adam_states": adam_bytes / 1e9, "full_gradients": grad_bytes / 1e9,
                  "activations_and_transients": (peak - w_bytes - fp_bytes - m_bytes - adam_bytes - grad_bytes) / 1e9,
                  "built_model": mem_build / 1e9, "peak": peak / 1e9},
           "curve": curve, "final_val_loss": curve[-1]["val_loss"] if curve else None}
    if method == "bop":
        out["flip_rate"] = (CTL["flips"].item() - flips_at_3) / n_weights / max(1, steps - 3)
        out["zero_frac"] = float(np.mean([(w == 0).float().mean().item() for w in weights]))
    elif method == "latent":
        out["zero_frac"] = float(np.mean([(m.quantized() == 0).float().mean().item() for m in lins]))
    del model, opt
    STATE.clear()
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ladder", "twin"], required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--ladder-steps", type=int, default=25)
    args = ap.parse_args()
    train_tokens = torch.from_numpy(np.load("train_tokens.npy").astype(np.int64)).cuda()
    val_tokens = torch.from_numpy(np.load("val_tokens.npy").astype(np.int64)).cuda()

    rows = []
    if args.mode == "ladder":
        for size in SIZES:
            r = train(size, "bop", 1e-6, 1e-3, 8, args.ladder_steps, 0, train_tokens, val_tokens, args.ladder_steps)
            rows.append(r)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                              if k not in ("curve", "gb")}), {k: round(v, 2) for k, v in r.get("gb", {}).items()},
                  flush=True)
            with open("twin_ladder.json", "w") as f:
                json.dump(rows, f, indent=1)
    else:
        configs = [("dense", 0, 0), ("latent", 0, 0)] + [("bop", t, g) for g in [1e-3, 1e-2] for t in [1e-7, 1e-6, 1e-5]]
        for seed in range(args.seeds):
            for method, tau, gamma in configs:
                r = train("3M", method, tau, gamma, 8, args.steps, seed, train_tokens, val_tokens, 500)
                rows.append(r)
                print(f"{method:7s} tau={tau:.0e} gamma={gamma:.0e} seed={seed}  "
                      f"val {r.get('final_val_loss', float('nan')):.3f}  {r.get('tokens_per_s', 0):.0f} tok/s  "
                      f"peak {r.get('gb', {}).get('peak', 0):.2f} GB  flip {r.get('flip_rate', float('nan')):.1e}",
                      flush=True)
                with open("twin_3m.json", "w") as f:
                    json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
