from __future__ import annotations

import torch
from typing import Callable, cast

from offload.dspark.d2.train.loss import bucket_statistics, distillation_loss
from offload.dspark.d2.train.loss import confidence_loss
from offload.dspark.d2.train.model import Drafter
from offload.dspark.d2.train.synthetic import SyntheticSource, inputs, tiny_config


class Logits:
    def __init__(self, z: torch.Tensor):
        self.z = z
        self.c = type("C", (), {"vocab": z.shape[-1]})()

    def logit_chunk(self, hidden: torch.Tensor, previous: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
        return self.z[..., lo:hi]


def test_bucket_kl_and_gradients_against_full_distribution() -> None:
    torch.manual_seed(2)
    z = torch.randn(2, 5, 96, requires_grad=True)
    pfull = torch.randn_like(z).softmax(-1)
    p, ids = pfull.topk(20, -1)
    tail = 1-p.sum(-1)
    hidden, prev = torch.zeros(2, 5, 1), torch.zeros(2, 5, dtype=torch.long)
    mask = torch.ones(2, 5, dtype=torch.bool)
    got, per = distillation_loss(Logits(z), hidden, prev, ids, p.log(), tail, prev, mask, mask, chunk=7)
    lq = z.log_softmax(-1)
    qt = lq.exp().scatter(-1, ids, 0).sum(-1)
    expected = (p*(p.log()-lq.gather(-1, ids))).sum(-1) + tail*(tail.log()-qt.log())
    weights = torch.tensor([1., .8, .65, .5, .4])[None].expand(2, -1)
    expected_loss = (weights*expected).sum()/weights.sum()
    torch.testing.assert_close(per, expected.detach(), rtol=2e-5, atol=3e-7)
    grad = torch.autograd.grad(got, z, retain_graph=True)[0]
    want = torch.autograd.grad(expected_loss, z)[0]
    torch.testing.assert_close(grad, want, rtol=2e-5, atol=3e-8)
    assert torch.isfinite(grad).all()


def test_missing_top20_ce_only_and_zero_tail() -> None:
    z = torch.randn(1, 5, 32, requires_grad=True)
    ids = torch.tensor([[[0, 1], [-1, -1], [0, 1], [-1, -1], [0, 1]]])
    present = ids[..., 0] >= 0
    p = torch.tensor([.7, .3]).expand(1, 5, 2)
    labels = torch.tensor([[7, 8, 9, 10, 11]])
    loss, per = distillation_loss(Logits(z), torch.zeros(1, 5, 1), labels, ids, p.log(),
                                   torch.zeros(1, 5), labels, present, torch.ones_like(present), chunk=1)
    expected_ce = -.1 * z.log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)
    torch.testing.assert_close(per[~present], expected_ce[~present])
    cast(Callable[[], None], loss.backward)()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_extreme_top_mass_retains_tail_without_cancellation() -> None:
    z = torch.tensor([[[80., -80., -81.]]], requires_grad=True)
    lse, tlse, top = bucket_statistics(lambda lo, hi: z[..., lo:hi], torch.tensor([[[0]]]), 3, 1)
    assert torch.isfinite(tlse).all()
    cast(Callable[[], None], (-tlse+lse+top.sum()*0).sum().backward)()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_tvd_switch_is_bucket_total_variation() -> None:
    z = torch.randn(1, 5, 32, requires_grad=True)
    ids = torch.tensor([0, 1]).expand(1, 5, -1)
    p = torch.tensor([.4, .3]).expand(1, 5, -1)
    previous = torch.zeros(1, 5, dtype=torch.long)
    present = torch.ones(1, 5, dtype=torch.bool)
    loss, per = distillation_loss(Logits(z), torch.zeros(1, 5, 1), previous, ids, p.log(),
                                 torch.full((1, 5), .3), previous, present, present, chunk=7, divergence="tvd")
    q = z.softmax(-1)
    want = .5*((p-q[..., :2]).abs().sum(-1)+(.3-q[..., 2:].sum(-1)).abs())
    torch.testing.assert_close(per, want.detach(), atol=2e-7, rtol=1e-6)
    assert torch.isfinite(torch.autograd.grad(loss, z)[0]).all()


def test_confidence_uses_known_argmax_probability_and_only_head_gradients() -> None:
    c = tiny_config()
    m = Drafter(SyntheticSource(c), c)
    for n, p in m.named_parameters():
        p.requires_grad_("confidence_head" in n)
    inp = inputs(c, batch=1)
    previous = torch.zeros(1, 5, dtype=torch.long)
    with torch.no_grad():
        h, x = m.hidden_from(inp)
        ids = m.logit_chunk(h, previous, 0, c.vocab).topk(20, -1).indices
    lp = torch.full((1, 5, 20), .04).log()
    mask = torch.ones(1, 5, dtype=torch.bool)
    loss, known_count = confidence_loss(m, h, x, previous, ids, lp, mask, chunk=17)
    assert known_count == 5 and loss is not None
    cast(Callable[[], None], loss.backward)()
    assert all(p.grad is None for name, p in m.named_parameters() if "confidence_head" not in name)
    assert m.last("confidence_head.proj.weight").grad is not None
    skipped, known = confidence_loss(m, h, x, previous, torch.full_like(ids, -1), lp, mask)
    assert skipped is None and known == 0
