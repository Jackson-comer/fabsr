# fabsr

Train ternary (-1/0/+1) language models by **flipping weights stored as 2-bit codes**. There is no full-precision copy of the weights and no `optimizer.step()` that touches them: each matrix computes its weight gradient in `backward`, hands it to the optimizer, and drops it. Optimizer state is a bf16 momentum plus one vector per matrix row and column.

On an 8 GB RTX 4060, this fine-tunes the 1.58B ternary weights of TII's Falcon-E-1B directly, using about 4 to 5 GB.

## Install

```bash
pip install .            # from this folder
```

Needs an NVIDIA GPU, PyTorch with CUDA, and Triton (installed automatically: `triton-windows` on Windows). On Windows, Triton compiles a small helper at first use and needs a C compiler on `CC` (we used `zig cc`); on Linux it just works.

## Use

```python
import torch, torch.nn.functional as F
import fabsr

model, tok = fabsr.load_falcon_e()                       # downloads the 0.7 GB packed checkpoint on first use
opt = fabsr.Fabsr(model, flip_rate=1.5e-5, flip_rate_end=2e-6, steps=1500)

for x, y in batches:                                      # x, y: token ids on the GPU, y = -100 where there is no loss
    logits = model(x).logits.float()
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)
    loss.backward()                                       # the weights flip in here
    opt.step()                                            # next step number and flip rate

fabsr.save_codes(model, "tuned.pt", step=1500)           # about 0.4 GB
fabsr.load_codes(model, "tuned.pt")                      # into a fresh load_falcon_e() model
```

A runnable version is `examples/finetune_falcon_e.py`.

`flip_rate` is the target fraction of weights flipped per step (not a learning rate); the tuned Falcon-E values are 1.5e-5 decaying to 2e-6. The other tuned settings (`momentum=0.025`, `second_moment=0.01`, `flip_every=4`) are the defaults and should not be changed without a reason.

## Flags

Everything below is off by default; the default path is the one the Falcon-E results were tuned with.

| Argument | What it does | Status |
|---|---|---|
| `fast=True` | turns on the three flags below | exact: identical outputs, only faster |
| `fast_unpack` | unpack the 2-bit codes in one Triton pass | exact; measured +23% tokens/s |
| `fast_actquant` | BitNet 8-bit activation quantisation in one pass | exact; +38% together with `fast_unpack` |
| `sparse_store` | the flip kernel writes only changed bytes | exact; no measurable speed change |
| `train_layers`, `train_types` | train only some layers or matrix types, e.g. `"top:12"`, `"mlp"` | works; skips the weight gradient for frozen matrices |
| `experimental={"wgrad_every": k}` | weight gradient only every k-th step | runs, **not validated for quality** |
| `experimental={"m_int8": True}` | int8 momentum, about half the momentum memory | runs, slower, **not validated for quality** |

Speeds are Falcon-E-1B, batch 4x448, activation checkpointing on, RTX 4060, measured interleaved in one process. The "exact" claims are tested with `torch.equal` on the pure functions and with flip-count statistics on the whole trainer (the trainer itself is not bit-reproducible run to run because of float atomics, about 0.7% spread in flip counts).

## Limits

- Tested on `tiiuae/Falcon-E-1B-Base` only. Other ternary models need their own loader.
- One `Fabsr` per process (its state is process-global). Each `backward()` is one optimizer step: there is no gradient accumulation.
- Embeddings, norms and the output head stay frozen.
- Needs CUDA. `transformers` is pinned below 5.18 because a call in the loader is removed there.
- Falcon-E weights are under TII's Falcon-LLM license; if you redistribute a tuned model you have to follow it (attribution, acceptable-use policy).

## How the package is built

`src/fabsr/_core/` is generated from the research tree (`tools/build_package.py`): the files are copied and only their sibling imports are rewritten, then every file is parsed and compared with its source with those imports normalised. `_core/MANIFEST.json` records the source hash of each file. Do not edit `_core` by hand.

## Tests

```bash
pip install pytest
pytest tests                       # CPU-safe checks
FABSR_RUN_GPU=1 pytest tests       # plus a GPU smoke test (needs the Falcon-E checkpoint in the HF cache)
```

## License and citation

Apache-2.0 (see `LICENSE` and `NOTICE`): free to use, change and sell, as long as you keep the copyright notice. The Falcon-E models are not part of this package and keep TII's own license.

If this helps your work, please cite it (`CITATION.cff`; GitHub shows a "Cite this repository" button).

Status: research code, version 0.1. It does what the README says on the setup it was tested on, and nothing more is promised.
