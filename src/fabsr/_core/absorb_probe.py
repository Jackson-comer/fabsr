"""Can Fabsr teach a pretrained ternary LLM (TriLM 1.1B, Spectra) new facts by flipping ternary weights only?

The checkpoint stores each projection as {-s, 0, +s}. We keep the exact ternary pattern as a bf16 tensor in {-1, 0, +1} plus a fixed
scale vector per output row (or per input column, whichever the shard layout allows). Embeddings, lm_head and norms stay frozen.
The Fabsr hook flips the ternary weights; no latent weights, bf16 momentum plus factored second-moment state only.

The new information is invented, so the base model cannot know it: 20 made-up countries, each with a made-up capital and river.
Each fact is trained in 4 sentence templates and tested three ways by likelihood ranking over the 20 candidate names (chance = 5%):
  seen     first training template, forward direction      ("The capital of X is" -> name)
  novel    phrasing never used in training                  ("Travelers arriving in X will find that its capital city is" -> name)
  reverse  second training template, other direction       ("<name> is the capital city of" -> X)
Also reports wikitext-2 loss before and after, to see what the flips cost on unrelated text.

  python absorb_probe.py --rates 1e-4 --steps 300
"""
import argparse
import json
import os
import random
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("BENCH_DIR", os.path.join(HERE, "..", "bench")))

# Near 8 GB, the allocator's cached blocks push Windows into swapping GPU memory (7 s/step instead of 1). Make it reclaim cache early.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.7,max_split_size_mb:256")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from . import packed_ternary as PT
from . import tune

MODEL = os.environ.get("TRILM_DIR", "")
LINEARS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
dev = "cuda"


# ---------------------------------------------------------------- checkpoint (read without mmap: Windows commit limit)
class Ckpt:
    def __init__(self, path):
        self.fh = open(path, "rb")
        n = struct.unpack("<Q", self.fh.read(8))[0]
        self.hdr = json.loads(self.fh.read(n))
        self.base = 8 + n
        self.keys = [k for k in self.hdr if k != "__metadata__"]

    def get(self, k):
        a, b = self.hdr[k]["data_offsets"]
        self.fh.seek(self.base + a)
        return torch.frombuffer(bytearray(self.fh.read(b - a)), dtype=torch.float16).reshape(self.hdr[k]["shape"])


class TLinear(nn.Module):
    """y = x @ (T * s)^T with T in {-1, 0, +1} (trainable by flips) and s a fixed per-row (out) or per-column (in) scale."""

    def __init__(self, T, s, axis):
        super().__init__()
        self.weight = nn.Parameter(T, requires_grad=True)
        self.register_buffer("s", s)
        self.axis = axis

    def forward(self, x):
        if self.axis == "in":
            return F.linear(x * self.s.to(x.dtype), self.weight)
        return F.linear(x, self.weight) * self.s.to(x.dtype)

    # interface shared with packed_ternary.PackedTLinear
    @property
    def n_ternary(self):
        return self.weight.numel()

    def zeros(self):
        return float((self.weight == 0).sum())

    def set_ternary(self, T):
        self.weight.data.copy_(T)
        self.weight.grad = None

    def attach(self, hook):
        return [self.weight.register_post_accumulate_grad_hook(hook)]


def ternarize(W16):
    """fp16 {-s, 0, +s} matrix (the scale can differ per shard) -> (bf16 {-1,0,1}, fp32 scale vector, axis, worst reconstruction error)."""
    W = W16.to(dev).float()
    a = W.abs()
    big = torch.finfo(torch.float32).max
    nz = torch.where(a > 0, a, torch.full_like(a, big))
    out_ok = bool(((a.amax(1) == nz.amin(1)) | (a.amax(1) == 0)).all())
    in_ok = bool(((a.amax(0) == nz.amin(0)) | (a.amax(0) == 0)).all())
    if out_ok:
        s, axis = a.amax(1), "out"
    elif in_ok:
        s, axis = a.amax(0), "in"
    else:
        raise RuntimeError("scale pattern is neither per-row nor per-column")
    s = torch.where(s == 0, s.max(), s)
    T = torch.sign(W)
    recon = T * (s[:, None] if axis == "out" else s[None, :])
    return T.to(torch.bfloat16), s, axis, float((recon - W).abs().max())


def build_model(ck):
    cfg = LlamaConfig.from_pretrained(MODEL)
    torch.set_default_dtype(torch.bfloat16)
    with torch.device(dev):
        model = LlamaForCausalLM(cfg)
    torch.set_default_dtype(torch.float32)
    sd = model.state_dict()
    for k in ck.keys:
        sd[k].copy_(ck.get(k).to(dev))
    model.config.use_cache = False
    return model


def prepare_packed(model):
    """Packed layers have no Parameter that requires grad, so make the (frozen) embedding output require grad to keep autograd alive."""
    model.get_input_embeddings().register_forward_hook(lambda m, i, o: o.requires_grad_(True))


def convert(model, ck, packed=False):
    """Swap every projection for a TLinear (or 2-bit PackedTLinear) holding the exact ternary pattern; returns the layers, worst reconstruction error, axis counts."""
    tl, worst, axes = [], 0.0, {"out": 0, "in": 0}
    for i, layer in enumerate(model.model.layers):
        for name in LINEARS:
            parent_name, attr = name.split(".")
            parent = getattr(layer, parent_name)
            T, s, axis, err = ternarize(ck.get(f"model.layers.{i}.{name}.weight"))
            worst, axes[axis] = max(worst, err), axes[axis] + 1
            m = PT.PackedTLinear(T, s, axis) if packed else TLinear(T, s, axis)
            del T
            setattr(parent, attr, m)
            tl.append(m)
    torch.cuda.empty_cache()
    return tl, worst, axes


# ---------------------------------------------------------------- data
def make_facts(seed=0, n=20):
    rng = random.Random(seed)
    syl = ["ve", "lt", "ra", "na", "os", "su", "me", "re", "ka", "dor", "mi", "zu", "bel", "tho", "pan", "gri", "qua", "lin", "sor", "ven", "dax", "yu", "fen", "ori"]
    used, out = set(), []

    def name():
        while True:
            w = "".join(rng.choice(syl) for _ in range(rng.choice([2, 3]))).capitalize()
            if w not in used and len(w) > 4:
                used.add(w)
                return w
    for _ in range(n):
        out.append({"country": name(), "capital": name(), "river": name()})
    return out


TRAIN = {"capital": ["The capital of {C} is {N}.", "{N} is the capital city of {C}.", "Q: What is the capital of {C}? A: {N}", "In {C}, the capital is {N}, and many people live there."],
         "river": ["The main river of {C} is the {N}.", "The {N} is the longest river in {C}.", "Q: Which river flows through {C}? A: the {N}", "In {C}, the {N} river provides water to many farms."]}
EVAL = {"capital": {"seen": "The capital of {C} is", "novel": "Travelers arriving in {C} will find that its capital city is", "reverse": "{N} is the capital city of"},
        "river": {"seen": "The main river of {C} is the", "novel": "Boats on the great river of {C} sail along the", "reverse": "The {N} is the longest river in"}}


class Scorer:
    def __init__(self, model, tok, facts):
        self.model, self.tok, self.facts = model, tok, facts
        self.enc = lambda s: tok.encode(s, add_special_tokens=False)

    @torch.no_grad()
    def logprobs(self, seqs):
        """seqs: list of (prompt_ids, answer_ids) -> summed log prob of the answer tokens."""
        out = []
        for i in range(0, len(seqs), 40):
            chunk = seqs[i:i + 40]
            L = max(len(p) + len(a) for p, a in chunk)
            ids = torch.zeros(len(chunk), L, dtype=torch.long, device=dev)
            for j, (p, a) in enumerate(chunk):
                ids[j, :len(p) + len(a)] = torch.tensor(p + a, device=dev)
            lp = F.log_softmax(self.model(ids).logits.float(), -1)
            for j, (p, a) in enumerate(chunk):
                pos = torch.arange(len(p) - 1, len(p) + len(a) - 1, device=dev)
                out.append(lp[j, pos, torch.tensor(a, device=dev)].sum().item())
        return out

    def run(self):
        res = {}
        for attr in ("capital", "river"):
            names = [f[attr] for f in self.facts]
            for kind in ("seen", "novel", "reverse"):
                rev = kind == "reverse"
                cands = [f["country"] for f in self.facts] if rev else names
                seqs, truth = [], []
                for k, f in enumerate(self.facts):
                    prompt = [0] + self.enc(EVAL[attr][kind].format(C=f["country"], N=f[attr]))
                    for c in cands:
                        seqs.append((prompt, self.enc(" " + c)))
                    truth.append(k)
                lps = torch.tensor(self.logprobs(seqs)).reshape(len(self.facts), len(cands))
                tgt = torch.tensor(truth)
                res[f"{attr}/{kind}"] = {"acc": float((lps.argmax(1) == tgt).float().mean()),
                                         "top3": float((lps.topk(3, 1).indices == tgt[:, None]).any(1).float().mean()),
                                         "lp": float(lps[torch.arange(len(truth)), tgt].mean())}
        for kind in ("seen", "novel", "reverse"):
            res[f"all/{kind}"] = {m: (res[f"capital/{kind}"][m] + res[f"river/{kind}"][m]) / 2 for m in ("acc", "top3", "lp")}
        return res


def wiki_chunks(tok, n=24, T=256):
    path = os.path.join(HERE, "wiki_test.json")
    try:
        if os.path.exists(path):
            text = json.load(open(path))["text"]
        else:
            from datasets import load_dataset
            last = None
            for repo in ("Salesforce/wikitext", "wikitext"):
                try:
                    text = "\n".join(load_dataset(repo, "wikitext-2-raw-v1", split="test")["text"][:1500])
                    break
                except Exception as e:
                    last = e
            else:
                raise last
            json.dump({"text": text}, open(path, "w"))
        src = "wikitext-2 test"
    except Exception as e:
        print("wikitext unavailable:", repr(e)[:100], "-> using the fallback paragraph (repetitive, weaker forgetting check)")
        text = ("The history of the printing press changed how ideas moved across Europe. Before movable type, books were copied by hand, which made them rare and "
                "expensive. Within fifty years of its invention, millions of volumes had been printed, and literacy rose steadily in cities and towns. ") * 60
        src = "fallback"
    ids = tok.encode(text, add_special_tokens=False)
    chunks = [[0] + ids[i:i + T - 1] for i in range(0, len(ids) - T, T)][:n]
    return torch.tensor(chunks, device=dev), src


@torch.no_grad()
def lm_loss(model, chunks):
    tot = 0.0
    for i in range(0, len(chunks), 8):
        x = chunks[i:i + 8]
        logits = model(x).logits.float()
        tot += F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="sum").item()
    return tot / (chunks.shape[0] * (chunks.shape[1] - 1))


@torch.no_grad()
def greedy(model, tok, prompt, n=10):
    prompt_ids = [0] + tok.encode(prompt, add_special_tokens=False)
    ids = list(prompt_ids)
    for _ in range(n):
        lg = model(torch.tensor([ids], device=dev)).logits[0, -1]
        ids.append(int(lg.argmax()))
    return tok.decode(ids[len(prompt_ids):])


# ---------------------------------------------------------------- training
def replay_pool(tok, T=64, n=4000):
    """Chunks of wikitext-2 TRAIN (the forgetting eval uses the test split) to mix into every batch."""
    path = os.path.join(HERE, "wiki_train.json")
    if os.path.exists(path):
        text = json.load(open(path))["text"]
    else:
        from datasets import load_dataset
        text = "\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"][:20000])
        json.dump({"text": text}, open(path, "w"))
    ids = tok.encode(text, add_special_tokens=False)
    return [[0] + ids[i:i + T - 1] for i in range(0, len(ids) - T, T)][:n]


def train_batch(tok, facts, rng, bs, pool=None, n_replay=0):
    texts = []
    for _ in range(bs):
        f, attr = rng.choice(facts), rng.choice(["capital", "river"])
        texts.append(rng.choice(TRAIN[attr]).format(C=f["country"], N=f[attr]))
    seqs = [[0] + tok.encode(t, add_special_tokens=False) for t in texts]
    if pool and n_replay:
        seqs += [rng.choice(pool) for _ in range(n_replay)]
    bs = len(seqs)
    L = max(map(len, seqs))
    x = torch.zeros(bs, L, dtype=torch.long)
    y = torch.full((bs, L), -100, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
        y[i, :len(s) - 1] = torch.tensor(s[1:])
    return x.to(dev), y.to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="1e-4")
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--beta2v", type=float, default=0.01)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--replay", type=int, default=0, help="wikitext-2 train sequences (64 tokens) added to every batch")
    ap.add_argument("--packed", action="store_true", help="store the ternary weights 2-bit packed (0.24 GB instead of 1.94 GB)")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(MODEL, "tokenizer.json"))
    ck = Ckpt(os.path.join(MODEL, "model.safetensors"))
    facts = make_facts()
    pool = replay_pool(tok) if args.replay else None
    print("example facts:", facts[:2], flush=True)
    model = build_model(ck)
    wchunks, wsrc = wiki_chunks(tok)
    model.eval()
    base_ref = lm_loss(model, wchunks)
    print(f"original checkpoint ({wsrc}): loss {base_ref:.4f} (ppl {torch.exp(torch.tensor(base_ref)):.1f})", flush=True)
    for p in model.parameters():
        p.requires_grad_(False)
    tl, worst, axes = convert(model, ck, args.packed)
    n_t = sum(m.n_ternary for m in tl)
    split_loss = lm_loss(model, wchunks)
    print(f"ternary split: {len(tl)} matrices {axes}, worst |reconstruction - original| = {worst:.2e}; {n_t / 1e6:.0f}M ternary weights; "
          f"loss after split {split_loss:.4f} (should match {base_ref:.4f})", flush=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.packed:
        prepare_packed(model)
    model.train()
    scorer = Scorer(model, tok, facts)
    zero0 = sum(m.zeros() for m in tl) / n_t
    model.eval()
    base_scores = scorer.run()
    model.train()
    print("before tuning:", {k: round(v["acc"], 2) for k, v in base_scores.items() if k.startswith("all")}, "(chance 0.05)", flush=True)
    torch.cuda.empty_cache()
    model.eval()
    print("  base greedy:", repr(greedy(model, tok, EVAL["capital"]["seen"].format(C=facts[0]["country"]))), "| truth:", facts[0]["capital"], flush=True)
    model.train()

    results = []
    for rate in [float(r) for r in args.rates.split(",")]:
        # restore the checkpoint's ternary pattern before each setting
        for j, m in enumerate(tl):
            T, _, _, _ = ternarize(ck.get(f"model.layers.{j // len(LINEARS)}.{LINEARS[j % len(LINEARS)]}.weight"))
            m.set_ternary(T)
        tune.STATE.clear(); tune.VFAC.clear(); tune.V2.clear(); tune.LAYER_IDS.clear()
        tune.CTL.update(rule="quantile", fused=True, vfac=True, secmom=False, movable_q=True, m_sr=True, every=args.every, gamma=args.gamma,
                        beta2v=args.beta2v, rate=rate, m_dtype=torch.bfloat16, flip_start=0, score_floor=0.0, own_rng=True, sr_seed=0, step=0,
                        rownorm=False, diag=False, flips=torch.zeros((), device=dev, dtype=torch.int64))
        hooks = [h for m in tl for h in m.attach(tune.hook)]
        rng = random.Random(1)
        curve, t0 = [], time.time()
        torch.cuda.reset_peak_memory_stats()
        for step in range(1, args.steps + 1):
            tune.CTL["step"] = step
            x, y = train_batch(tok, facts, rng, args.bs, pool, args.replay)
            logits = model(x).logits.float()
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), y[:, :-1].reshape(-1), ignore_index=-100)
            loss.backward()
            if step % args.eval_every == 0 or step == args.steps or step == 1:
                model.eval()
                sc = scorer.run()
                wl = lm_loss(model, wchunks)
                model.train()
                torch.cuda.empty_cache()  # eval batches bloat the allocator cache; near 8 GB Windows starts swapping GPU memory
                train_loss = float(loss.detach())
                pt = {"step": step, "train_loss": train_loss, "wiki_loss": wl, "flipped_frac_cum": float(tune.CTL["flips"]) / n_t, "scores": sc, "sec": time.time() - t0}
                curve.append(pt)
                print(f"rate {rate:g} step {step:4d} loss {train_loss:6.3f}| seen {sc['all/seen']['acc']:.2f} novel {sc['all/novel']['acc']:.2f} "
                      f"reverse {sc['all/reverse']['acc']:.2f} | wiki {wl:.3f} ({wl - base_ref:+.3f}) | flips so far {pt['flipped_frac_cum']:.5f}/w | "
                      f"{time.time() - t0:.0f}s | peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
        model.eval()
        print("  greedy after:", repr(greedy(model, tok, EVAL["capital"]["seen"].format(C=facts[0]["country"]))), "| truth:", facts[0]["capital"], flush=True)
        print("  greedy novel:", repr(greedy(model, tok, EVAL["capital"]["novel"].format(C=facts[1]["country"]))), "| truth:", facts[1]["capital"], flush=True)
        zero1 = sum(m.zeros() for m in tl) / n_t
        results.append({"rate": rate, "args": vars(args), "curve": curve, "zero_frac_before": zero0, "zero_frac_after": zero1,
                        "before": base_scores, "wiki_ref": base_ref, "wiki_src": wsrc, "peak_gb": torch.cuda.max_memory_allocated() / 1e9})
        for h in hooks:
            h.remove()
        json.dump(results, open(os.path.join(args.out, "absorb_probe.json"), "w"), indent=1)
        model.train()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
