"""Load ternary Llama checkpoints straight into 2-bit packed layers without ever holding a dense copy on the GPU (the model is built on the meta device).

  build_falcon()   Falcon-E-1B-Base (TII): native ternary Llama, 1.7B params (1.58B ternary), vocab 32768. The packed 'main' revision stores 4 weights per byte
                   along dim 0 with values code-1 in {-1,0,1}, one bf16 scale per matrix: W = T / weight_scale. Verified against the bf16 'prequantized' latents
                   (92-99.8% of entries agree; the latents are only an approximation of what the packed model was trained with, so we start from the packed one).
                   Like the original BitLinear, every projection sees its input fake-quantised to int8 per token (disable with act_quant=False).
  build_trilm()    TriLM 1.1B, same result as absorb_probe.build_model + convert(packed=True) with a much smaller transient footprint.

Both return (model, packed_layers, tokenizer, ids) with ids = dict(bos=..., eos=..., pad=...).
"""
import glob
import json
import os
import struct

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from . import absorb_probe as A
from . import packed_ternary as PT

DT = {"BF16": torch.bfloat16, "U8": torch.uint8, "F16": torch.float16, "F32": torch.float32}
FALCON_REPO = os.environ.get("FALCON_REPO", "tiiuae/Falcon-E-1B-Base")  # e.g. tiiuae/Falcon-E-3B-Base
FALCON_GLOB = os.path.join(os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")), "hub", "models--" + FALCON_REPO.replace("/", "--"), "snapshots", "*")


class Reader:
    """safetensors reader without mmap (Windows commit limit), any dtype."""

    def __init__(self, path):
        self.fh = open(path, "rb")
        n = struct.unpack("<Q", self.fh.read(8))[0]
        self.hdr = json.loads(self.fh.read(n))
        self.hdr.pop("__metadata__", None)
        self.base = 8 + n
        self.keys = list(self.hdr)

    def get(self, k):
        h = self.hdr[k]
        a, b = h["data_offsets"]
        self.fh.seek(self.base + a)
        return torch.frombuffer(bytearray(self.fh.read(b - a)), dtype=DT[h["dtype"]]).reshape(h["shape"])


def falcon_snapshot():
    for d in glob.glob(FALCON_GLOB):
        f = os.path.join(d, "model.safetensors")
        if os.path.exists(f) and os.path.getsize(f) < float(os.environ.get("FALCON_MAX_BYTES", 1e9)):  # the packed 'main' revision (665 MB for the 1B), not the 3.4 GB latents
            return d
    raise FileNotFoundError(f"{FALCON_REPO} main revision is not in the HF cache")


def unpack_rows(p):
    """Falcon-E / HF-bitnet packing: uint8 (R, C) -> int8 (4R, C) with value = code - 1; byte bits 2i..2i+1 hold row block i."""
    R = p.shape[0]
    out = torch.zeros((4 * R, *p.shape[1:]), dtype=torch.uint8, device=p.device)
    for i in range(4):
        out[i * R:(i + 1) * R] = (p & (3 << (2 * i))) >> (2 * i)
    return out.to(torch.int8) - 1


def fake_quant_act(x):
    """BitNet 8-bit per-token absmax activation quantisation (straight-through)."""
    if PT.OPT["fast_actquant"] and x.is_cuda and x.dtype == torch.bfloat16:  # one fused pass, identical output (see packed_ternary._actquant_kernel)
        return PT.fast_fake_quant_act(x)
    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    q = (x * scale).round().clamp(-128, 127) / scale
    return x + (q - x).detach()


def _meta_model(cfg):
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            return LlamaForCausalLM(cfg)
    finally:
        torch.set_default_dtype(torch.float32)


def _load_rest(model, cfg, rd, skip):
    """Real GPU parameters for everything that is not a replaced projection; recreate the rotary buffer (it was built on meta)."""
    for k in rd.keys:
        if k in skip:
            continue
        path, attr = k.rsplit(".", 1)
        setattr(model.get_submodule(path), attr, nn.Parameter(rd.get(k).to(torch.bfloat16).to(A.dev), requires_grad=False))
    if model.lm_head.weight.is_meta:  # tied embeddings
        model.lm_head.weight = model.model.embed_tokens.weight
    model.model.rotary_emb = LlamaRotaryEmbedding(config=cfg, device=A.dev)
    assert not any(p.is_meta for p in model.parameters()) and not any(b.is_meta for b in model.buffers())
    model.config.use_cache = True
    model.eval()


def build_falcon(act_quant=True):
    snap = falcon_snapshot()
    rd = Reader(os.path.join(snap, "model.safetensors"))
    cfg = LlamaConfig.from_pretrained(snap)
    cfg.quantization_config = None
    model = _meta_model(cfg)
    tl, skip = [], set()
    for i, layer in enumerate(model.model.layers):
        for name in A.LINEARS:
            parent_name, attr = name.split(".")
            key = f"model.layers.{i}.{name}"
            T = unpack_rows(rd.get(key + ".weight").to(A.dev))
            ws = float(rd.get(key + ".weight_scale").float())
            m = PT.PackedTLinear(T, torch.full((T.shape[0],), 1.0 / ws, dtype=torch.float32, device=A.dev), "out")
            del T
            if act_quant:
                m.register_forward_pre_hook(lambda mod, args: (fake_quant_act(args[0]),))
            setattr(getattr(layer, parent_name), attr, m)
            tl.append(m)
            skip |= {key + ".weight", key + ".weight_scale"}
    _load_rest(model, cfg, rd, skip)
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(snap, "tokenizer.json"))
    ids = {"bos": tok.convert_tokens_to_ids("<|begin_of_text|>"), "eos": tok.convert_tokens_to_ids("<|end_of_text|>"), "pad": tok.convert_tokens_to_ids("<|pad|>"), "prefix": []}
    torch.cuda.empty_cache()
    return model, tl, tok, ids


def build_trilm():
    ck = A.Ckpt(os.path.join(A.MODEL, "model.safetensors"))
    cfg = LlamaConfig.from_pretrained(A.MODEL)
    model = _meta_model(cfg)
    tl, skip = [], set()
    for i, layer in enumerate(model.model.layers):
        for name in A.LINEARS:
            parent_name, attr = name.split(".")
            key = f"model.layers.{i}.{name}.weight"
            T, s, axis, _ = A.ternarize(ck.get(key))
            m = PT.PackedTLinear(T, s, axis)
            del T
            setattr(getattr(layer, parent_name), attr, m)
            tl.append(m)
            skip.add(key)

    class _Fp16Reader:  # ck.get returns fp16; _load_rest wants .keys/.get/.hdr
        keys = ck.keys
        hdr = ck.hdr
        get = staticmethod(ck.get)

    _load_rest(model, cfg, _Fp16Reader, skip)
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(A.MODEL, "tokenizer.json"))
    torch.cuda.empty_cache()
    return model, tl, tok, {"bos": 0, "eos": 0, "pad": 1, "prefix": [0]}

