from __future__ import annotations

from typing import Literal

import pytest
import torch

from offload.dspark.d2.train.fast_loss import bucket_loss
from offload.dspark.d2.train.loss import distillation_loss
from offload.dspark.d2.train.model import Drafter
from offload.dspark.d2.train.synthetic import SyntheticSource, tiny_config


@pytest.mark.parametrize("divergence", ("kl", "tvd"))
@pytest.mark.parametrize("missing", (False, True))
def test_explicit_bucket_gradients_match_streamed_autograd(divergence: Literal["kl", "tvd"], missing: bool) -> None:
    torch.manual_seed(91)
    m = Drafter(SyntheticSource(tiny_config()), tiny_config(), checkpoint_stages=False)
    h = torch.randn(2, 5, m.c.dim, requires_grad=True)
    previous = torch.tensor([[3, 3, 7, 8, 9], [9, 1, 5, 5, 6]])
    distribution = torch.randn(2, 5, m.c.vocab).softmax(-1)
    top, ids = distribution.topk(20, -1)
    present = torch.ones(2, 5, dtype=torch.bool)
    mask = present.clone()
    mask[0, -1] = False
    labels = previous + 1
    if missing:
        present[1, 2] = False
        ids[1, 2] = -1
        top[1, 2] = 0
    tail = 1 - top.sum(-1)
    expected, _ = distillation_loss(m, h, previous, ids, top.log(), tail, labels, present, mask,
                                     chunk=17, divergence=divergence)
    params = (h, m.last("markov_head.embed.weight"), m.last("markov_head.head.weight"))
    gradients = torch.autograd.grad(expected, params)
    got, _ = bucket_loss(h, m.t["head.weight"], params[1], params[2], previous,
                          ids, top.log(), tail, labels, present, mask, 17, divergence)
    torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-6)
    actual = torch.autograd.grad(got, params)
    for a, b in zip(actual, gradients):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-7)
