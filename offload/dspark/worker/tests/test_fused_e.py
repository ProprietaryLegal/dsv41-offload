"""fused_e.py against the torch / fused.py ops it replaces (one GPU)."""

from __future__ import annotations

import os

import pytest
import torch

from worker import fused as FU
from worker import fused_e as FE
from worker.ops import route

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
ROOT = os.environ.get("DSPARK_ROOT", "/opt/dsv41/x86host-data/dspark")
DEV = torch.device("cuda", 0)


def test_collapse_norm2_and_rms_norm_fp16():
    g = torch.Generator(device=DEV).manual_seed(1)
    for T in (5, 20):
        stream = (torch.randn(T, 4, 5120, generator=g, device=DEV) * 3).to(torch.bfloat16)
        pre = torch.rand(T, 4, generator=g, device=DEV)
        w = torch.randn(5120, generator=g, device=DEV).to(torch.bfloat16)
        for p in (None, pre):
            a = FU.collapse_norm(stream, p, w, 1e-20)
            b, b16 = FE.collapse_norm2(stream, p, w, 1e-20)
            assert torch.equal(a, b) and torch.equal(b16, a.to(torch.float16))
        x = torch.randn(T, 1280, generator=g, device=DEV) * 7
        wq = torch.randn(1280, generator=g, device=DEV).to(torch.bfloat16)
        assert torch.equal(FE.rms_norm_fp16(x, wq, 1e-20), FU.rms_norm(x, wq, 1e-20).to(torch.float16))


def test_swiglu_fp16():
    g = torch.Generator(device=DEV).manual_seed(2)
    gt = torch.randn(20, 1152, generator=g, device=DEV) * 8
    ut = torch.randn(20, 1152, generator=g, device=DEV) * 8
    lim = 10.0
    want = (torch.nn.functional.silu(gt.clamp(max=lim)) * ut.clamp(min=-lim, max=lim)).to(torch.float16)
    got = FE.swiglu_fp16(gt, ut, lim)
    diff = (got.float() - want.float()).abs()
    # same formula and libdevice exp; allow one fp16 ulp where the fp32 product sits on a rounding boundary
    ulp = torch.finfo(torch.float16).eps * want.float().abs().clamp_min(2 ** -14)
    assert bool((diff <= ulp).all()), float(diff.max())
    print("swiglu bit-identical fraction", float((got == want).float().mean()))


@pytest.mark.skipif(not os.path.exists(ROOT), reason="checkpoint")
def test_route_fused_vs_torch():
    from worker.checkpoint import Checkpoint
    ck = Checkpoint(ROOT, verify=False)
    ck.verified = True                       # gate tensors only; the worker itself always verifies
    g = torch.Generator(device=DEV).manual_seed(3)
    n_rows = n_same = 0
    max_w = 0.0
    for i in range(3):
        gw = ck.get(f"mtp.{i}.ffn.gate.weight").to(DEV, torch.float32).contiguous()
        gb = ck.get(f"mtp.{i}.ffn.gate.bias").to(DEV, torch.float32).contiguous()
        perm = torch.randperm(128, device=DEV)
        for T in (5, 20, 20, 20):
            x = (torch.randn(T, 5120, generator=g, device=DEV)).to(torch.bfloat16)
            ids0, w0 = route(x, gw, gb, 3, 1.5)
            ids1, sel1, w1, w16 = FE.route_fused(x, gw, gb, perm, 3, 1.5)
            same = (ids0.sort(-1).values == ids1.sort(-1).values).all(-1)
            n_rows += T
            n_same += int(same.sum())
            assert torch.equal(sel1, perm[ids1]) and torch.equal(w16, w1.to(torch.float16))
            # same expert set: weights agree to fp32 rounding (logits summed in another order)
            o0 = ids0.sort(-1)
            o1 = ids1.sort(-1)
            wa = w0.gather(1, o0.indices)[same]
            wb = w1.gather(1, o1.indices)[same]
            if wa.numel():
                max_w = max(max_w, float(((wa - wb).abs() / wa.abs()).max()))
            # a different set only at a 3rd / 4th near-tie of the torch scores
            if not bool(same.all()):
                sc = torch.nn.functional.softplus(x.float() @ gw.t()).sqrt() + gb
                v = sc[~same].topk(4, dim=-1).values
                gap = ((v[:, 2] - v[:, 3]) / v[:, 2].abs())
                assert bool((gap < 1e-5).all()), gap
    print(f"route_fused: {n_same}/{n_rows} rows with the same expert set, weight rel diff max {max_w:.2e}")
    assert max_w < 1e-5 and n_same >= n_rows - 2
