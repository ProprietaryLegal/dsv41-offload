"""Exactness of the format decoders and the act_qdq / reference ops (CPU; real tensors when native-1017 exists).

  python -m pytest -q offload/dspark/native/tests/test_formats.py
"""
from __future__ import annotations

import os

import pytest
import torch

from offload.dspark.native import formats as F
from offload.dspark.native import reference as R
from offload.dspark.native.official import OfficialShards

ROOT = os.environ.get("NATIVE_1017_ROOT", "/opt/dsv41/x86host-data/native-1017")
HAVE = os.path.exists(os.path.join(ROOT, "model-00044-of-00048.safetensors"))


def test_fp4_table_and_nibble_order():
    # byte 0x7A: low nibble 0xA (-1.0) is input 0, high nibble 0x7 (6.0) is input 1
    packed = torch.tensor([[0x7A] * 16], dtype=torch.uint8)
    scale = torch.tensor([[127]], dtype=torch.uint8)
    v = F.decode_fp4(packed, scale)
    assert v.shape == (1, 32)
    assert v[0, 0].item() == -1.0 and v[0, 1].item() == 6.0
    scale2 = torch.tensor([[127 - 3]], dtype=torch.uint8)
    assert F.decode_fp4(packed, scale2)[0, 1].item() == 6.0 / 8


def test_every_fp4_code_every_scale_exact_in_bf16_and_fp16_range():
    codes = torch.arange(16, dtype=torch.uint8)
    packed = (codes | (codes << 4)).repeat(2).view(1, 32)[:, :16]
    for b in range(127 - 16, 127 + 1):                       # the official checkpoint's exponent range -16..0
        v = F.decode_fp4(packed, torch.tensor([[b]], dtype=torch.uint8))
        assert F.exact_in(v, torch.bfloat16) and F.exact_in(v, torch.float16)


def test_nan_scale_refused():
    with pytest.raises(ValueError):
        F.decode_fp4(torch.zeros(1, 16, dtype=torch.uint8), torch.tensor([[255]], dtype=torch.uint8))


def test_act_qdq_matches_definition():
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(7, 256, generator=g) * torch.logspace(-6, 3, 256)).to(torch.bfloat16)
    x[0, :32] = 0                                              # amax floor 1e-4
    x[1, 32:64] = 448.0                                        # exact power-of-two boundary: s = 1
    q, s = F.act_quant(x)
    assert torch.all(s[0, 0] == 2.0 ** -22)                    # ceil(log2(1e-4/448)) = -22
    assert s[1, 1].item() == 1.0
    assert q.abs().max() <= 448
    y = F.act_qdq(x)
    ref = (q.view(7, 8, 32) * s.unsqueeze(-1)).view(7, 256)
    assert torch.equal(y.float(), ref)                          # bf16 holds q * s exactly
    # idempotent: QDQ of a QDQ'd tensor is itself
    assert torch.equal(F.act_qdq(y), y)


def test_linear_modes_agree_synthetic():
    g = torch.Generator().manual_seed(1)
    w = torch.randint(0, 256, (64, 128), generator=g, dtype=torch.uint8)
    s = torch.randint(127 - 8, 127 - 2, (64, 8), generator=g, dtype=torch.uint8)
    x = torch.randn(5, 256, generator=g).to(torch.bfloat16)
    a = R.linear_fp4(x, w, s, "f64").float()
    b = R.linear_fp4(x, w, s, "blocked").float()
    assert (a - b).abs().max() <= 2 * torch.finfo(torch.bfloat16).eps * a.abs().max()


@pytest.mark.skipif(not HAVE, reason="official shards not present")
def test_real_expert_and_dense_modes_agree():
    sh = OfficialShards(ROOT)
    p = "mtp.0.ffn.experts.5."
    t = [sh.get(p + n) for n in ("w1.weight", "w1.scale", "w3.weight", "w3.scale", "w2.weight", "w2.scale")]
    g = torch.Generator().manual_seed(2)
    x = torch.randn(4, 5120, generator=g).to(torch.bfloat16)
    rw = torch.rand(4, 1, generator=g)
    a = R.expert(x, *t, rw, "fp4", "f64").float()
    b = R.expert(x, *t, rw, "fp4", "blocked").float()
    rel = ((a - b).norm() / a.norm()).item()
    assert rel < 1e-3, rel
    w, s = sh.get("mtp.0.attn.wq_a.weight"), sh.get("mtp.0.attn.wq_a.scale")
    a = R.linear_fp8(x, w, s, "f64").float()
    b = R.linear_fp8(x, w, s, "blocked").float()
    assert ((a - b).norm() / a.norm()).item() < 1e-3
