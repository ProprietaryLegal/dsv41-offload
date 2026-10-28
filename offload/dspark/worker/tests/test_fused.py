"""Fused Triton kernels (fused.py) against the plain torch ops (ops.py) they replace. GPU required.

Run on x86host: CUDA_VISIBLE_DEVICES=GPU-c4e7a371-... python -m pytest -q worker/tests/test_fused.py
"""

import pytest
import torch

from worker import fused, ops

dev = torch.device("cuda", 0)
H, HC = 5120, 4


def rnd(*shape, scale=1.0, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    return (torch.randn(*shape, generator=g, device=dev) * scale).to(dtype)


def ulps_bf16(a, b):
    """max distance in bf16 representation steps between two bf16 tensors."""
    ai = a.view(torch.int16).to(torch.int32)
    bi = b.view(torch.int16).to(torch.int32)
    ai = torch.where(ai < 0, -32768 - ai, ai)
    bi = torch.where(bi < 0, -32768 - bi, bi)
    return int((ai - bi).abs().max())


def close_bf16(got, ref, atol):
    """|got - ref| <= max(one bf16 step of ref, atol): sums that cancel lose relative precision in any order."""
    step = ref.float().abs().clamp_min(1e-30) * 2.0 ** -7
    return bool(((got.float() - ref.float()).abs() <= torch.maximum(step, torch.full_like(step, atol))).all())


@pytest.mark.parametrize("T", [1, 5, 20])
def test_hc_mixes(T):
    stream = rnd(T, HC, H, scale=3.0)
    fn = rnd(24, HC * H, scale=0.01, dtype=torch.float32, seed=1)
    scale = torch.tensor([0.9, 1.1, 0.7], device=dev)
    base = rnd(24, scale=0.5, dtype=torch.float32, seed=2)
    ref = ops.hc_mixes(stream, fn, scale, base, 1e-20, 1e-6, 20)
    got = fused.hc_mixes(stream, fn, scale, base, 1e-20, 1e-6, 20)
    for r, g in zip(ref, got):
        assert g.shape == r.shape and g.dtype == torch.float32
        assert torch.allclose(g, r, rtol=2e-5, atol=2e-6), (g - r).abs().max()


@pytest.mark.parametrize("T", [1, 5, 20])
@pytest.mark.parametrize("with_pre", [False, True])
def test_collapse_norm(T, with_pre):
    stream = rnd(T, HC, H, scale=4.0)
    pre = (torch.rand(T, HC, device=dev) + 0.1) if with_pre else None
    w = rnd(H, scale=0.3, seed=3)
    ref = ops.rms_norm(ops.hc_collapse(stream, pre), w, 1e-20)
    got = fused.collapse_norm(stream, pre, w, 1e-20)
    assert got.dtype == torch.bfloat16 and ulps_bf16(got, ref) <= 1


@pytest.mark.parametrize("T", [1, 20])
def test_hc_post(T):
    stream = rnd(T, HC, H, scale=4.0)
    y = rnd(T, H, scale=2.0, seed=4)
    post = torch.rand(T, HC, device=dev) * 2
    comb = torch.softmax(torch.randn(T, HC, HC, device=dev), -1)
    ref = ops.hc_post(y, stream, post, comb)
    got = fused.hc_post(y, stream, post, comb)
    # terms are up to ~4 * 16 + 2 * 8 in magnitude; fp32 reordering error ~1e-5 of that
    assert close_bf16(got, ref, atol=1e-4 * 80)


@pytest.mark.parametrize("D", [512, 1280, 5120])
def test_rms_norm(D):
    x = rnd(7, D, scale=5.0)
    w = rnd(D, scale=0.5, seed=5)
    assert ulps_bf16(fused.rms_norm(x, w, 1e-20), ops.rms_norm(x, w, 1e-20)) <= 1


@pytest.mark.parametrize("heads", [None, 64])
@pytest.mark.parametrize("inverse", [False, True])
def test_rope(heads, inverse):
    cs = ops.rope_cos_sin(4096).to(dev)
    n = 9
    x = rnd(n, 512, scale=3.0) if heads is None else rnd(n, heads, 512, scale=3.0)
    pos = torch.tensor([0, 1, 5, 127, 128, 1000, 2047, 3000, 4095], device=dev)
    ref = ops.apply_rope(x, pos, cs, inverse=inverse)
    got = fused.rope(x, pos, cs, inverse=inverse)
    assert torch.equal(got[..., :448], x[..., :448])
    assert ulps_bf16(got, ref) <= 1


def test_kv_rope_qdq():
    cs = ops.rope_cos_sin(4096).to(dev)
    kv = rnd(33, 512, scale=2.0)
    kv[3, :64] = 0                       # an all-zero group
    kv[4, 64:128] *= 1e-3
    pos = torch.arange(33, device=dev) * 97
    ref = ops.kv_record_qdq(ops.apply_rope(kv, pos, cs))
    got = fused.kv_rope_qdq(kv, pos, cs)
    assert torch.equal(got[:, :448], ref[:, :448])          # the quantized NoPE values must match exactly
    assert ulps_bf16(got[:, 448:], ref[:, 448:]) <= 1


@pytest.mark.parametrize("B", [1, 4])
def test_markov_argmax(B):
    V, R = 129280, 256
    w1 = rnd(V, R, scale=0.05, seed=6)
    w2 = rnd(V, R, scale=0.05, seed=7)
    mb = ops.MarkovBias(w1, w2, list(range(0, 50000, 7)), rows=512)
    base = rnd(B, V, scale=4.0, seed=8)
    prev = torch.tensor([0, 7, 1234, 99999][:B], device=dev)      # cached, cached, uncached, uncached
    e = mb.embed(prev)
    ref_logits = base + mb(prev, e)
    ref = ref_logits.argmax(-1)
    got = fused.markov_argmax(base, prev, e, mb)
    assert torch.equal(got, ref)
    # a forced tie: two equal maxima in different blocks -> lowest id
    base2 = base.clone()
    bias = mb(prev, e)
    for b in range(B):
        base2[b] = (-100.0 - bias[b].float()).to(torch.bfloat16)
        base2[b, 70000] = (50.0 - bias[b, 70000].float()).to(torch.bfloat16)
        base2[b, 300] = (50.0 - bias[b, 300].float()).to(torch.bfloat16)
    l2 = base2 + bias
    assert torch.equal(fused.markov_argmax(base2, prev, e, mb), l2.argmax(-1))


@pytest.mark.parametrize("B,k", [(1, 5), (3, 5), (4, 4)])
def test_block_attention_ring(B, k):
    S, W, D, Hh = 6, 128, 512, 64
    ring = rnd(S, 3, W, D, scale=1.0, seed=11)
    pos = torch.full((S, W), -1, dtype=torch.int64, device=dev)
    slots = torch.tensor([4, 1, 5, 0][:B], device=dev)
    last = torch.tensor([300, 60, 1000, 129][:B], device=dev)
    lo = torch.tensor([300 - 127, 0, 990, 2][:B], device=dev)      # a full window, a short one, a gap, a partial
    for b in range(B):
        p = torch.arange(max(0, int(last[b]) - 140), int(last[b]) + 1, device=dev)
        pos[int(slots[b]), p % W] = p
    q = rnd(B, k, Hh, D, scale=0.5, seed=12)
    blk = rnd(B, k, D, scale=1.0, seed=13)
    sink = torch.randn(Hh, device=dev)
    for stage in range(3):
        rs = ring[:, stage]
        got = fused.block_attention_ring(q, rs, pos, slots, lo, last, blk, sink, D ** -0.5)
        cpos = pos[slots]
        valid = (cpos >= lo[:, None]) & (cpos <= last[:, None])
        ref = ops.block_attention(q, rs[slots], valid, blk, sink, D ** -0.5)
        assert close_bf16(got, ref, atol=2e-3 * float(ref.float().abs().max())), (got.float() - ref.float()).abs().max()


@pytest.mark.parametrize("amax", [3.0, 52.25, 16384.0, 16385.0, 1.6e5, 7e5])
def test_prescale_matches_plugin(amax):
    x = rnd(6, 15360, scale=1.0, seed=21)
    x[2, 77] = amax
    x16, sc = fused.prescale_fp16(x)
    ref_sc = torch.exp2(torch.ceil(torch.log2(x.abs().amax().float().clamp_min(1.0) / 16384.0)).clamp_min(0.0))
    assert torch.equal(sc.view(()), ref_sc)
    assert torch.equal(x16, (x.float() / ref_sc).to(torch.float16))


def test_scaled_rms_norm():
    y = torch.randn(7, 5120, device=dev) * 40
    w = rnd(5120, scale=1.0, seed=22)
    for sc in (1.0, 4.0, 64.0):
        s_ = torch.tensor([sc], device=dev)
        assert torch.equal(fused.scaled_rms_norm(y, s_, w, 1e-20), fused.rms_norm(y * sc, w, 1e-20))


def test_kv_store():
    S, W, D = 5, 128, 512
    y = torch.randn(9, D, device=dev) * 3
    w = rnd(D, scale=1.0, seed=23)
    pos = torch.tensor([0, 5, 127, 128, 300, 301, 302, 999, 1000], device=dev)
    slots = torch.tensor([0, 0, 0, 1, 2, 2, 2, 4, 3], device=dev)
    cs = ops.rope_cos_sin(4096, 64, 10000.0).to(dev)
    ring = torch.zeros(S, 3, W, D, dtype=torch.bfloat16, device=dev)
    rpos = torch.full((S, W), -1, dtype=torch.int64, device=dev)
    fused.kv_store(y, w, 1e-20, pos, cs, ring[:, 1], slots, rpos)
    want = fused.kv_rope_qdq(fused.rms_norm(y, w, 1e-20), pos, cs)
    assert torch.equal(ring[slots, 1, pos % W], want)
    assert torch.equal(rpos[slots, pos % W], pos)
    assert int((ring[:, 0] != 0).sum()) == 0 and int((ring[:, 2] != 0).sum()) == 0


def _wo_a_case(G=8, R=1024, C=4096, seed=31):
    g = torch.Generator(device=dev).manual_seed(seed)
    w8 = torch.randint(0, 256, (G * R, C), generator=g, device=dev, dtype=torch.int32).to(torch.uint8)
    w8[(w8 & 0x7F) == 0x7F] = 0x3C                      # no NaN (as in the checkpoint)
    sc = torch.exp2(torch.randint(-12, -7, (G * R // 32, C // 32), generator=g, device=dev).float())
    wf = w8.view(torch.float8_e4m3fn).float() * sc.repeat_interleave(32, 0).repeat_interleave(32, 1)
    return w8, sc, wf


def test_wo_a_fp8_dequant_exact():
    G, R, C = 2, 64, 256
    w8, sc, wf = _wo_a_case(G, R, C)
    # one-hot rows read single weights back: out[t, g*R + r] = wf[g*R + r, c_t] exactly
    cols = torch.tensor([0, 3, 31, 32, 100, 255], device=dev)
    o = torch.zeros(len(cols), G * C, dtype=torch.bfloat16, device=dev)
    o[torch.arange(len(cols)), cols] = 1.0
    o[torch.arange(len(cols)), C + cols] = 1.0
    out = fused.wo_a_fp8(o, w8, sc, G).float()
    for i, c in enumerate(cols.tolist()):
        for g_ in range(G):
            assert torch.equal(out[i, g_ * R:(g_ + 1) * R], wf[g_ * R:(g_ + 1) * R, c].to(torch.bfloat16).float())


@pytest.mark.parametrize("T", [5, 16, 20])
def test_wo_a_fp8_matches_bmm(T):
    G, R, C = 8, 1024, 4096
    w8, sc, wf = _wo_a_case(G, R, C)
    w_t = wf.to(torch.bfloat16).view(G, R, C).transpose(1, 2).contiguous()          # model.dequant_wo_a layout
    o = rnd(T, G * C, scale=0.5, seed=32)
    ref = torch.bmm(o.view(T, G, C).transpose(0, 1), w_t).transpose(0, 1).reshape(T, -1)
    got = fused.wo_a_fp8(o, w8, sc, G)
    assert close_bf16(got, ref, atol=1e-3 * float(ref.float().abs().max())), (got.float() - ref.float()).abs().max()


@pytest.mark.parametrize("B", [1, 2, 4])
def test_markov_argmax_into(B):
    V, R = 129280, 256
    w1 = rnd(V, R, scale=0.05, seed=6)
    w2 = rnd(V, R, scale=0.05, seed=7)
    mb = ops.MarkovBias(w1, w2, list(range(0, 50000, 7)), rows=512)
    base = rnd(B, 3, V, scale=4.0, seed=9)
    prev0 = torch.tensor([0, 7, 1234, 99999][:B], device=dev)      # cached, cached, uncached, uncached
    out = torch.full((B, 5), -1, dtype=torch.int64, device=dev)
    prev = prev0
    for i in range(3):
        want = fused.markov_argmax(base[:, i], prev, mb.embed(prev), mb)
        fused.markov_argmax_into(base[:, i], prev0 if i == 0 else out[:, i - 1], mb, out, i)
        assert torch.equal(out[:, i], want), i
        prev = want
    assert int((out[:, 3:] != -1).sum()) == 0
    # forced tie across blocks -> lowest id
    bias = mb(prev0, mb.embed(prev0))
    b2 = (-100.0 - bias.float()).to(torch.bfloat16)
    b2[:, 70000] = (50.0 - bias[:, 70000].float()).to(torch.bfloat16)
    b2[:, 300] = (50.0 - bias[:, 300].float()).to(torch.bfloat16)
    fused.markov_argmax_into(b2, prev0, mb, out, 4)
    assert torch.equal(out[:, 4], (b2 + bias).argmax(-1))
