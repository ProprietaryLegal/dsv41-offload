"""Independent dense-oracle checks for zero-mass and full-support buckets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from offload.dspark.d2.train.fast_loss import bucket_loss
from offload.dspark.d2.train.evaluate import bucket_kl
from offload.dspark.d2.train.loss import POSITION_WEIGHTS, distillation_loss


@dataclass
class Vocabulary:
    vocab: int


@dataclass
class Head:
    head: Tensor
    embed: Tensor
    markov: Tensor

    @property
    def c(self) -> Vocabulary:
        return Vocabulary(self.head.shape[0])

    def logit_chunk(self, hidden: Tensor, previous: Tensor, start: int, end: int) -> Tensor:
        return F.linear(hidden, self.head[start:end]) + F.linear(F.embedding(previous, self.embed), self.markov[start:end])


@pytest.mark.parametrize("backend", ("reference", "analytic"))
@pytest.mark.parametrize("case", ("full_support", "zero_top_mass", "rounded_mass", "missing"))
@pytest.mark.parametrize("divergence", ("kl", "tvd"))
@pytest.mark.parametrize("chunk", (1, 2, 7))
def test_objective_edges_match_dense_value_and_gradients(
    backend: str, case: str, divergence: Literal["kl", "tvd"], chunk: int,
) -> None:
    g = torch.Generator().manual_seed(812)
    h = torch.randn(2, 5, 4, generator=g, requires_grad=True)
    m = Head(torch.randn(7, 4, generator=g), torch.randn(7, 3, generator=g, requires_grad=True),
             torch.randn(7, 3, generator=g, requires_grad=True))
    prev = torch.tensor([[0, 0, 1, 1, 2], [2, 1, 0, 0, 1]])
    labels = torch.full_like(prev, 6)  # outside top support; CE must still find it
    ids = torch.arange(7 if case == "full_support" else 2).expand(2, 5, -1).clone()
    p = torch.full(ids.shape, 1 / ids.shape[-1]) if case == "full_support" else torch.tensor([.5, .2]).expand(2, 5, -1).clone()
    tail = 1 - p.sum(-1)
    if case == "zero_top_mass":
        p[..., 1] = 0
        tail[:] = .5
    if case == "rounded_mass":
        tail += 8e-5  # accepted input-rounding tolerance; derivatives must use actual mass
    present = torch.ones_like(prev, dtype=torch.bool)
    mask = present.clone()
    mask[0, -2:] = False
    if case == "missing":
        present[1, 2] = False
        ids[1, 2] = -1
        p[1, 2] = 0
        tail[1, 2] = 1
    lp = p.log()
    z = m.logit_chunk(h, prev, 0, 7)
    lq = z.log_softmax(-1)
    q = lq.exp()
    valid = ids >= 0
    safe_ids = ids.clamp_min(0)
    selected_lq = lq.gather(-1, safe_ids)
    qtop = q.gather(-1, safe_ids).masked_fill(~valid, 0)
    retained = torch.zeros_like(q, dtype=torch.int64).scatter_add(-1, safe_ids, valid.long()).bool()
    qt = q.masked_fill(retained, 0).sum(-1)
    if divergence == "kl":
        # A zero teacher bucket contributes exactly zero, even when the student bucket is empty.
        logqt = qt.masked_fill(tail == 0, 1).log()
        per = (torch.xlogy(p, p) - p * selected_lq).sum(-1) + torch.xlogy(tail, tail) - tail * logqt
    else:
        per = .5 * ((p - qtop).abs().sum(-1) + (tail - qt).abs())
    per = torch.where(present, per, -.1 * lq.gather(-1, labels[..., None]).squeeze(-1))
    weights = torch.tensor(POSITION_WEIGHTS).expand_as(per) * mask
    expected = (per * weights).sum() / weights.sum()
    if backend == "reference":
        got, actual_per = distillation_loss(m, h, prev, ids, lp, tail, labels, present, mask, chunk=chunk, divergence=divergence)
    else:
        got, actual_per = bucket_loss(h, m.head, m.embed, m.markov, prev, ids, lp, tail, labels, present, mask, chunk, divergence)
    torch.testing.assert_close(got, expected, rtol=2e-6, atol=5e-7)
    torch.testing.assert_close(actual_per, per.detach(), rtol=2e-6, atol=5e-7)
    parameters = h, m.embed, m.markov
    # Non-unit incoming gradient, repeated embedding IDs and partly masked blocks.
    actual_grad = torch.autograd.grad(3.7 * got, parameters, retain_graph=True)
    expected_grad = torch.autograd.grad(3.7 * expected, parameters)
    for actual, want in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, want, rtol=3e-5, atol=2e-7)
        assert torch.isfinite(actual).all()


@pytest.mark.parametrize("full_support", (False, True))
def test_evaluation_handles_zero_mass_buckets(full_support: bool) -> None:
    logits = torch.tensor([[.2, -.7, 1.1]], dtype=torch.float64)
    ids = torch.tensor([[0, 1, 2]] if full_support else [[0, 1]])
    p = torch.tensor([[.6, 0, .4]] if full_support else [[.6, 0]], dtype=torch.float64)
    tail = torch.tensor([0.] if full_support else [.4], dtype=torch.float64)
    result = bucket_kl(logits, ids, p.log(), tail)
    logq = logits.log_softmax(-1)
    # Both cases describe the same full three-token teacher distribution.
    expected = .6 * (.6 / logq[0, 0].exp()).log() + .4 * (.4 / logq[0, 2].exp()).log()
    torch.testing.assert_close(result, expected[None], rtol=1e-12, atol=1e-12)
