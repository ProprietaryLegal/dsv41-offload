"""The native GPU ops against formats.py / reference.py on the real official tensors.

GPU:  CUDA_VISIBLE_DEVICES=GPU-7744b44b-... python -m pytest -q offload/dspark/native/tests/test_ops.py
CPU:  TRITON_INTERPRET=1 python -m pytest -q offload/dspark/native/tests/test_ops.py   (Triton interpreter; slow)
"""
from __future__ import annotations

import os

import pytest
import torch

from offload.dspark.native import formats as F
from offload.dspark.native import ops as NO
from offload.dspark.native import reference as R
from offload.dspark.native.loader import LoadStats, load_dense, load_expert_banks
from offload.dspark.native.official import OfficialShards

ROOT = os.environ.get("NATIVE_1017_ROOT", "/opt/dsv41/x86host-data/native-1017")
HAVE = os.path.exists(os.path.join(ROOT, "model-00044-of-00048.safetensors"))
INTERP = os.environ.get("TRITON_INTERPRET") == "1"
DEV = torch.device("cpu") if INTERP or not torch.cuda.is_available() else torch.device("cuda")
pytestmark = pytest.mark.skipif(not INTERP and not torch.cuda.is_available(), reason="needs a GPU or the interpreter")


def test_act_qdq_bit_identical():
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(9, 640, generator=g) * torch.logspace(-7, 4, 640)).to(torch.bfloat16)
    x[0, :32] = 0
    x[1, 32:64] = 448.0
    x[2, 64:96] = -448.0 * 2 ** -5
    x[3] = x[3] * 1e-30
    want = F.act_qdq(x)
    got = NO.act_qdq(x.to(DEV)).cpu()
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


@pytest.mark.skipif(not HAVE, reason="official shards not present")
@pytest.mark.parametrize("kind", ["fp8", "bf16"])
def test_dense_matches_reference(kind):
    sh = OfficialShards(ROOT)
    st = LoadStats()
    d = load_dense(sh, "mtp.1.attn.wkv", DEV, kind, st)
    w, s = sh.get("mtp.1.attn.wkv.weight"), sh.get("mtp.1.attn.wkv.scale")
    g = torch.Generator().manual_seed(3)
    x = torch.randn(5, 5120, generator=g).to(torch.bfloat16)
    want = R.linear_fp8(x, w, s, "f64").float()
    xq = NO.act_qdq(x.to(DEV))
    got = d(xq).cpu().float()
    tol = torch.finfo(torch.bfloat16).eps * want.abs().amax()
    assert (got - want).abs().max() <= tol
    assert (got == want).float().mean() > 0.97
    got32 = d(xq, out_f32=True).cpu()
    ref32 = (F.act_qdq(x).double() @ F.decode_fp8_block(w, s).double().t()).float()
    assert ((got32 - ref32).abs() <= 1e-5 * ref32.abs().amax()).all()
    if kind == "bf16" and DEV.type == "cuda":          # the Triton kernel on bf16 weights agrees too
        dt = NO.NativeDense("wkv-triton", "bf16", d.w, None, backend="triton")
        assert (dt(xq).cpu().float() - want).abs().max() <= tol
        assert ((dt(xq, out_f32=True).cpu() - ref32).abs() <= 1e-5 * ref32.abs().amax()).all()
    # a row shard keeps the full tensor's scale blocks
    d2 = load_dense(sh, "mtp.1.attn.wkv", DEV, kind, st, rows=(256, 512))
    assert torch.equal(d2(xq).cpu(), d(xq).cpu()[:, 256:512]) or kind == "bf16"


@pytest.mark.skipif(not HAVE, reason="official shards not present")
@pytest.mark.parametrize("kind", ["packed", "expanded"])
def test_expert_slots_match_reference(kind):
    sh = OfficialShards(ROOT)
    E = 6
    banks = load_expert_banks(sh, 2, [(0, DEV), (1, DEV)], 2, kind, LoadStats(), verify_sample=2, n_experts=E)
    g = torch.Generator().manual_seed(4)
    T = 2
    x = (torch.randn(T, 5120, generator=g) * 0.7).to(torch.bfloat16)
    ids = torch.tensor([[0, 5, 3], [2, 0, 4]], dtype=torch.int64)
    rw = torch.rand(T, 3, generator=g)
    xq = F.act_qdq(x)
    full = torch.zeros(T, 3, 5120, dtype=torch.float32)
    hs = {}
    for r in (0, 1):
        h = torch.empty(T * 3, 1152, dtype=torch.bfloat16, device=DEV)
        full += banks[r].forward_slots(xq.to(DEV), ids.to(DEV), rw.to(DEV), h=h).cpu()
        hs[r] = h.cpu()
    for t in range(T):
        for j in range(3):
            e = int(ids[t, j])
            p = f"mtp.2.ffn.experts.{e}."
            w1, s1, w3, s3, w2, s2 = (sh.get(p + n) for n in ("w1.weight", "w1.scale", "w3.weight", "w3.scale",
                                                               "w2.weight", "w2.scale"))
            gg = R.linear_fp4(x[t:t + 1], w1, s1)
            uu = R.linear_fp4(x[t:t + 1], w3, s3)
            hq = F.act_qdq(R.swiglu(gg, uu, rw[t:t + 1, j:j + 1]))
            got_h = torch.cat([hs[0][t * 3 + j], hs[1][t * 3 + j]])
            frac = (got_h == hq[0]).float().mean().item()
            assert frac > 0.995, frac                      # bf16 g/u ties may flip by fp32 order: a few elements
            y = (hq.double() @ F.decode_fp4(w2, s2).double().t()).float()[0]
            rel = ((full[t, j] - y).norm() / y.norm()).item()
            assert rel < 5e-3, rel


def test_official_combine_order():
    parts = torch.randn(2, 3, 8) * 100
    ids = torch.tensor([[7, 1, 4], [0, 9, 2]])
    sh = torch.randn(2, 8)
    got = NO.moe_combine_official(parts, sh, ids)
    want = torch.zeros(2, 8)
    for t in range(2):
        for j in ids[t].argsort().tolist():
            want[t] += parts[t, j].to(torch.bfloat16).float()
    want = (want + sh.to(torch.bfloat16).float()).to(torch.bfloat16)
    assert torch.equal(got, want)


def test_markov_logits_fp32_exact_bias():
    g = torch.Generator().manual_seed(6)
    V, Vr, Rk = 1000, 300, 256
    w1 = (torch.randn(V, Rk, generator=g) * 0.05).to(torch.bfloat16)
    w2 = (torch.randn(Vr, Rk, generator=g) * 0.05).to(torch.bfloat16)
    base = torch.randn(3, Vr, generator=g) * 4
    prev = torch.tensor([5, 999, 0])
    got = NO.markov_logits_fp32(base.to(DEV), prev.to(DEV), w1.to(DEV), w2.to(DEV)).cpu()
    want = (base.double() + w1[prev].double() @ w2.double().t()).float()
    assert ((got - want).abs() <= 1e-5 * want.abs().amax()).all()


def test_markov_hot_rows_bit_identical_to_computed():
    g = torch.Generator().manual_seed(7)
    V, Vr = 700, 260
    w1 = (torch.randn(V, 256, generator=g) * 0.05).to(torch.bfloat16).to(DEV)
    w2 = (torch.randn(Vr, 256, generator=g) * 0.05).to(torch.bfloat16).to(DEV)
    base = (torch.randn(3, Vr, generator=g) * 4).to(DEV)
    prev = torch.tensor([5, 699, 17]).to(DEV)
    rows = NO.MarkovRowsFP32(w1, w2, [17, 5, 3], rows=2)          # 5 and 17 cached, 699 computed
    assert rows.rows == 2
    a = NO.markov_logits_fp32(base, prev, w1, w2, rows).cpu()
    b = NO.markov_logits_fp32(base, prev, w1, w2, None).cpu()
    assert torch.equal(a, b)


def test_swiglu_qdq_matches_reference():
    g_ = torch.Generator().manual_seed(8)
    T, I = 4, 1152
    g = torch.randn(T, 3, I, generator=g_) * 6
    u = torch.randn(T, 3, I, generator=g_) * 6
    rw = torch.rand(T, 3, generator=g_)
    got = NO.swiglu_qdq(g.to(DEV), u.to(DEV), rw.to(DEV)).cpu()
    want = F.act_qdq(R.swiglu(g.to(torch.bfloat16), u.to(torch.bfloat16), rw.reshape(T, 3, 1))).view(T * 3, I)
    assert (got == want).float().mean() > 0.999
