import importlib
import os
import subprocess
import sys

import pytest

import fabsr


def test_import_is_light_and_cpu_safe():
    # a fresh interpreter: `import fabsr` must not pull in torch, transformers or triton (so it cannot need CUDA)
    code = "import sys, fabsr; bad = [m for m in ('torch', 'triton', 'transformers') if m in sys.modules]; print(fabsr.__version__, bad); sys.exit(1 if bad else 0)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_rate_schedule():
    assert fabsr.rate_schedule(1, 1e-5) == 1e-5
    assert fabsr.rate_schedule(1, 1e-5, 1e-6, 0) == 1e-5            # no step count: constant
    r = [fabsr.rate_schedule(s, 1.5e-5, 2e-6, 1500) for s in (1, 750, 1500, 3000)]
    assert r[0] > r[1] > r[2]
    assert r[2] == pytest.approx(2e-6)
    assert r[3] == pytest.approx(2e-6)                              # held at the end value
    assert r[1] == pytest.approx((1.5e-5 * 2e-6) ** 0.5, rel=1e-3)  # log-linear: halfway is the geometric mean


def test_public_names():
    for n in ("Fabsr", "load_falcon_e", "ternary_layers", "save_codes", "load_codes", "rate_schedule"):
        assert hasattr(fabsr, n)


def test_flag_lists_match_the_core():
    cuda = pytest.importorskip("torch").cuda.is_available()
    if not cuda:
        pytest.skip("the vendored core creates a CUDA generator at import")
    PT = importlib.import_module("fabsr._core.packed_ternary")
    assert set(fabsr.api.FLAGS) | set(fabsr.api.EXPERIMENTAL) == set(PT.OPT)
    assert not any(PT.OPT.values()) or PT.OPT["wgrad_every"] == 1   # core defaults: everything off


@pytest.mark.skipif(os.environ.get("FABSR_RUN_GPU") != "1", reason="set FABSR_RUN_GPU=1 (needs a CUDA GPU and the Falcon-E checkpoint in the HF cache)")
def test_gpu_smoke_tune_a_few_steps():
    import torch
    import torch.nn.functional as F
    model, tok = fabsr.load_falcon_e()
    opt = fabsr.Fabsr(model, flip_rate=1e-4, flip_every=2)
    x = torch.tensor([tok.encode("User: What is 2 plus 3?\nAssistant: 5.")]).cuda()
    before = [m.codes.clone() for m in opt.layers[:4]]
    for _ in range(6):
        lg = model(x).logits.float()
        loss = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1))
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
    assert opt.step_count == 7
    assert opt.flipped_fraction() > 0                                 # flips happened
    assert any(not torch.equal(b, m.codes) for b, m in zip(before, opt.layers[:4]))
    opt.close()
