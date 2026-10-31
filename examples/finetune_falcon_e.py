"""Smallest end-to-end use of fabsr: tune Falcon-E-1B on a few chat lines and watch the weights flip.

  python examples/finetune_falcon_e.py --steps 40

Needs an NVIDIA GPU (about 5 GB at this size) and downloads the 0.7 GB packed checkpoint on first use. Swap `lines` for your own data.
"""
import argparse

import torch
import torch.nn.functional as F

import fabsr

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--fast", action="store_true", help="turn on the exact speed flags (same results, faster)")
ap.add_argument("--save", default="")
args = ap.parse_args()

lines = ["User: What is 17 plus 25?\nAssistant: 17 plus 25 is 42.",
         "User: Name a primary colour.\nAssistant: Red is a primary colour.",
         "User: What is the capital of France?\nAssistant: The capital of France is Paris."]

model, tok = fabsr.load_falcon_e()
opt = fabsr.Fabsr(model, flip_rate=1.5e-5, flip_rate_end=2e-6, steps=args.steps, fast=args.fast)
print(f"{opt.n_ternary / 1e9:.2f}B ternary weights, {torch.cuda.memory_allocated() / 1e9:.2f} GB on the GPU")

ids = [tok.encode(t) for t in lines]
L = max(map(len, ids))
x = torch.full((len(ids), L), tok.pad_token_id or 0, dtype=torch.long)
y = torch.full((len(ids), L), -100, dtype=torch.long)
for i, s in enumerate(ids):
    x[i, :len(s)] = torch.tensor(s)
    y[i, :len(s) - 1] = torch.tensor(s[1:])
x, y = x.cuda(), y.cuda()

for step in range(1, args.steps + 1):
    logits = model(x).logits.float()
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)
    loss.backward()          # weights flip in here
    opt.step()
    if step % 10 == 0 or step == 1:
        print(f"step {step:3d}  loss {loss.item():.3f}  flipped {opt.flipped_fraction() * 100:.4f}% of weights  peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

if args.save:
    fabsr.save_codes(model, args.save, step=args.steps)
    print("saved", args.save)
