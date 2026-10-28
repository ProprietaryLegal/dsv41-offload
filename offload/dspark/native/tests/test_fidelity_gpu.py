"""Run under the N4 UUID/flock wrapper; fixtures are synthetic, never captured."""
from __future__ import annotations

import math

import pytest
import torch

from offload.dspark.native.attention import attention_into
from offload.dspark.native.fidelity_reference import markov_logits, online_attention
from offload.dspark.native.ops import markov_logits_fp32

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires authorized N4 GPU hold")


@pytest.mark.parametrize("last", [1, 62, 63, 64, 126, 127, 128, 255, 900000])
def test_online64_ring_and_graph(last: int) -> None:
    g = torch.Generator().manual_seed(38)
    b, k, h, d = 2, 5, 32, 512
    # BF16 x BF16 integer dots are exact, isolating online softmax scheduling.
    q = torch.randint(-2, 3, (b, k, h, d), generator=g).bfloat16()
    ring = torch.randint(-2, 3, (4, 3, 128, d), generator=g).bfloat16()[:, 1]
    block = torch.randint(-2, 3, (b, k, d), generator=g).bfloat16()
    positions = torch.full((4, 128), -1, dtype=torch.int64)
    lo = max(0, last - 127)
    pp = torch.arange(lo, last + 1)
    for slot in (1, 3):
        positions[slot, pp % 128] = pp
    slots = torch.tensor([3, 1])
    lows = torch.full((b,), lo)
    highs = torch.full((b,), last)
    sink = torch.linspace(-2., 4., h)
    nc = min(128, last + 1)
    kv = torch.cat((ring[slots], block), 1)
    idx = torch.cat((torch.arange(nc), 128 + torch.arange(k))).view(1, 1, -1).expand(b, k, -1)
    want = online_attention(q, kv, sink, idx, d ** -.5, fp64=True)
    args = (q.cuda(), ring.cuda(), positions.cuda(), slots.cuda(), lows.cuda(), highs.cuda(), block.cuda(), sink.cuda())
    out = torch.empty_like(args[0])
    ptr = out.data_ptr()
    attention_into(*args, d ** -.5, out)
    got = out.cpu()
    # FP32 exp/reduction differences only; no weight or QDQ error is involved.
    assert (got.float() - want.float()).norm() / want.float().norm() < .003
    torch.testing.assert_close(got, want, rtol=.008, atol=.004)
    stream = torch.cuda.Stream()  # type: ignore[no-untyped-call]
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        attention_into(*args, d ** -.5, out)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        attention_into(*args, d ** -.5, out)
    graph.replay()
    torch.cuda.synchronize()
    assert out.data_ptr() == ptr
    assert torch.equal(out.cpu(), got)


def test_fp32_markov_exact_dyadic_and_near_tie() -> None:
    g = torch.Generator().manual_seed(42)
    embed = torch.randint(-2, 3, (32, 256), generator=g).bfloat16()
    weight = torch.randint(-2, 3, (513, 256), generator=g).bfloat16()
    base = torch.randn(3, 513, generator=g)
    prev = torch.tensor([0, 15, 31])
    want = markov_logits(base, prev, embed, weight, fp64=True)
    got = markov_logits_fp32(base.cuda(), prev.cuda(), embed.cuda(), weight.cuda())
    assert torch.equal(got.cpu(), want)


def test_online64_exact_constant_values() -> None:
    q = torch.zeros(1, 5, 32, 512, dtype=torch.bfloat16, device="cuda")
    ring = torch.ones(1, 128, 512, dtype=torch.bfloat16, device="cuda")
    positions = torch.arange(128, device="cuda").view(1, -1)
    slot = torch.zeros(1, dtype=torch.int64, device="cuda")
    last = torch.full_like(slot, 127)
    block = torch.ones(1, 5, 512, dtype=torch.bfloat16, device="cuda")
    sink = torch.zeros(32, device="cuda")
    out = torch.empty_like(q)
    attention_into(q, ring, positions, slot, slot, last, block, sink, 512 ** -.5, out)
    assert torch.equal(out, torch.full_like(q, 133 / 134))


def test_mixed_short_and_wrapped_windows() -> None:
    g = torch.Generator().manual_seed(91)
    q = torch.randint(-2, 3, (2, 5, 32, 512), generator=g).bfloat16()
    ring = torch.randint(-2, 3, (4, 128, 512), generator=g).bfloat16()
    block = torch.randint(-2, 3, (2, 5, 512), generator=g).bfloat16()
    sink = torch.linspace(-2., 4., 32)
    positions = torch.full((4, 128), -1, dtype=torch.int64)
    slots, last = torch.tensor([3, 0]), torch.tensor([62, 900000])
    lo = (last - 127).clamp_min(0)
    expected = []
    for b in range(2):
        p = torch.arange(int(lo[b]), int(last[b]) + 1)
        positions[slots[b], p % 128] = p
        kv = torch.cat((ring[slots[b]], block[b]), 0).unsqueeze(0)
        idx = torch.cat((torch.arange(min(128, int(last[b]) + 1)), 128 + torch.arange(5)))
        expected.append(online_attention(q[b:b+1], kv, sink, idx.view(1, 1, -1).expand(1, 5, -1),
                                         512 ** -.5, fp64=True))
    args = (q.cuda(), ring.cuda(), positions.cuda(), slots.cuda(), lo.cuda(), last.cuda(), block.cuda(), sink.cuda())
    out = torch.empty_like(args[0])
    attention_into(*args, 512 ** -.5, out)
    want = torch.cat(expected)
    got = out.cpu()
    assert (got.float() - want.float()).norm() / want.float().norm() < .003
    torch.testing.assert_close(got, want, rtol=.008, atol=.004)


def test_online_blocks_and_sink_rounding_are_required() -> None:
    # First block:64 keys at score0/value1. Next block:5 keys at log(3)/value0.
    # Sink weight9 is added last. Official output=64/(64+15+9)=8/11.
    # Rounding P around the global/sink maximum instead produces a different
    # BF16 value (~.73047 versus .72656), so a single-pass implementation fails.
    q = torch.zeros(1, 5, 32, 512, dtype=torch.bfloat16, device="cuda")
    q[..., 0] = 1
    ring = torch.zeros(1, 128, 512, dtype=torch.bfloat16, device="cuda")
    ring[:, :64, 1] = 1
    positions = torch.full((1, 128), -1, dtype=torch.int64, device="cuda")
    positions[:, :64] = torch.arange(64, device="cuda")
    slot = torch.zeros(1, dtype=torch.int64, device="cuda")
    last = torch.full_like(slot, 63)
    block = torch.zeros(1, 5, 512, dtype=torch.bfloat16, device="cuda")
    block[..., 0] = 1
    sink = torch.full((32,), 2 * math.log(3), device="cuda")
    out = torch.empty_like(q)
    attention_into(q, ring, positions, slot, slot, last, block, sink, math.log(3), out)
    assert torch.equal(out[..., 1], torch.full_like(out[..., 1], 8 / 11))
