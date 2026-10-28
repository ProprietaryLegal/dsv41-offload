from __future__ import annotations

import torch

from offload.dspark.native.fidelity_reference import head_logits, markov_logits, online_attention


def test_head_and_markov_preserve_fp32_near_tie() -> None:
    hidden = torch.tensor([[1., 1.]], dtype=torch.bfloat16)
    weight = torch.tensor([[1., 2 ** -9], [1., 2 ** -8]], dtype=torch.bfloat16)
    base = head_logits(hidden, weight)
    assert torch.equal(base, head_logits(hidden, weight, fp64=True))
    emb = torch.tensor([[1., 0.]], dtype=torch.bfloat16)
    mw = torch.tensor([[0., 0.], [2 ** -10, 0.]], dtype=torch.bfloat16)
    got = markov_logits(base, torch.tensor([0]), emb, mw)
    assert torch.equal(got, markov_logits(base, torch.tensor([0]), emb, mw, fp64=True))
    assert got.argmax(-1).item() == 1
    assert (base.bfloat16() + head_logits(emb, mw).bfloat16()).argmax(-1).item() == 0


def test_online_exact_dots_and_block_rescaling() -> None:
    g = torch.Generator().manual_seed(37)
    q = torch.randint(-2, 3, (1, 2, 3, 16), generator=g).bfloat16()
    kv = torch.randint(-2, 3, (1, 133, 16), generator=g).bfloat16()
    idx = torch.arange(133).view(1, 1, -1).expand(1, 2, -1)
    sink = torch.tensor([0., 2., -3.])
    # Integer dot products and dyadic BF16 probabilities have exact sums here.
    a = online_attention(q, kv, sink, idx, .25)
    b = online_attention(q, kv, sink, idx, .25, fp64=True)
    assert torch.equal(a, b)


def test_sink_last_and_empty_blocks() -> None:
    q = torch.zeros(1, 1, 1, 16, dtype=torch.bfloat16)
    kv = torch.ones(1, 1, 16, dtype=torch.bfloat16)
    idx = torch.full((1, 1, 129), -1)
    assert torch.equal(online_attention(q, kv, torch.tensor([0.]), idx, 1.), q)
    idx[..., -1] = 0
    assert torch.equal(online_attention(q, kv, torch.tensor([0.]), idx, 1.), q + .5)


def test_online_differs_from_single_pass_bf16_p() -> None:
    g = torch.Generator().manual_seed(45)
    q = torch.randn(1, 2, 4, 32, generator=g).bfloat16()
    kv = torch.randn(1, 133, 32, generator=g).bfloat16()
    idx = torch.arange(133).view(1, 1, -1).expand(1, 2, -1)
    sink = torch.full((4,), 4.)
    got = online_attention(q, kv, sink, idx, .2)
    s = torch.einsum("bmhd,btd->bmht", q.float(), kv.float()) * .2
    mx = torch.maximum(s.amax(-1), sink)
    p = torch.exp(s - mx[..., None])
    den = p.sum(-1) + torch.exp(sink - mx)
    old = (torch.einsum("bmht,btd->bmhd", p.bfloat16().float(), kv.float()) / den[..., None]).bfloat16()
    assert not torch.equal(got, old)
