from __future__ import annotations

from typing import Literal

import pytest
import torch
from torch.nn import functional as F

from offload.dspark.d2.train.audits import one_step
from offload.dspark.d2.train.data import SyntheticBlocks
from offload.dspark.d2.train.fast_loss import bucket_loss
from offload.dspark.d2.train.loss import POSITION_WEIGHTS
from offload.dspark.d2.train.ops import Dense, qdq
from offload.dspark.d2.train.tests.test_loop import model


def test_one_step_reports_every_nonzero_gradient_and_zero_A_is_expected() -> None:
    torch.set_num_threads(2)
    m = model()
    result = one_step(m, SyntheticBlocks(m.c, 1)[0], chunk=32)
    assert result["nonzero_gradient_parameters"] == [n for n, p in m.named_parameters() if p.grad is not None and bool(p.grad.count_nonzero())]
    assert any(n.endswith(".b") for n in result["nonzero_gradient_parameters"])
    assert all(n.endswith(".a") for n in result["zero_gradient_parameters"])
    assert result["missing_gradient_parameters"] == []
    assert all("confidence_head" in n for n in result["excluded_parameters"])
    first = next(layer for layer in m.modules() if isinstance(layer, Dense) and layer.rank)
    with torch.no_grad():
        first.a.zero_()
        first.b.zero_()
    with pytest.raises(ValueError, match="nonzero A and zero B"):
        one_step(m, SyntheticBlocks(m.c, 1)[0])


def test_projection_cache_refused_even_if_detached_because_input_or_weight_can_change() -> None:
    m = model()
    block = SyntheticBlocks(m.c, 1)[0]
    inp = block["inputs"]
    with torch.no_grad():
        cached = m.main_proj(inp["taps"].float())
    assert not cached.requires_grad
    for change in ("input", "weight"):
        with torch.no_grad():
            if change == "input":
                inp["taps"].add_(1)
            else:
                m.main_proj.weight.add_(1)
        with pytest.raises(ValueError, match="frozen weights AND frozen inputs"):
            m.hidden_from({**inp, "main_projection": cached})


@pytest.mark.parametrize("divergence", ["kl", "tvd"])
@pytest.mark.parametrize("chunk", [7, 31])
def test_analytic_backward_against_unfused_full_vocab_with_intentional_ste(
    divergence: Literal["kl", "tvd"], chunk: int,
) -> None:
    torch.manual_seed(932)
    # Nontrivial QDQ rounding; compare STE, never finite-difference rounding.
    raw = torch.randn(2, 5, 32, requires_grad=True)
    head = torch.randn(31, 32).bfloat16()
    embed = torch.randn(31, 7, requires_grad=True)
    markov = torch.randn(31, 7, requires_grad=True)
    prev = torch.tensor([[2, 2, 4, 4, 6], [1, 3, 3, 5, 1]])
    teacher = torch.randn(2, 5, 31).softmax(-1)
    p, ids = teacher.topk(9, -1)
    tail = 1-p.sum(-1)
    labels = (prev+8) % 31  # includes labels outside the retained support
    present = torch.ones(2, 5, dtype=torch.bool)
    present[0, 2] = False  # ordinary sampled CE fallback
    mask = torch.ones_like(present)
    mask[1, 4] = False
    h = qdq(raw)
    assert not torch.equal(h, raw)
    # Unfused independent reference: full global softmax then exact bucket sum.
    z = F.linear(h, head.float())+F.linear(F.embedding(prev, embed), markov)
    q = z.softmax(-1)
    qt = q.gather(-1, ids)
    complement = torch.ones_like(q).scatter_(-1, ids, 0)
    qtail = (q*complement).sum(-1)
    if divergence == "kl":
        per = (p*(p.log()-qt.log())).sum(-1)+tail*(tail.log()-qtail.log())
    else:
        per = .5*((p-qt).abs().sum(-1)+(tail-qtail).abs())
    ce = -z.log_softmax(-1).gather(-1, labels[..., None])[..., 0]
    per = torch.where(present, per, .1*ce)
    weights = torch.tensor(POSITION_WEIGHTS)*mask
    expected = (per*weights).sum()/weights.sum()
    params = (raw, embed, markov)
    reference_grads = torch.autograd.grad(expected, params, retain_graph=True)
    actual, _ = bucket_loss(h, head, embed, markov, prev, ids, p.log(), tail, labels, present, mask, chunk, divergence)
    actual_grads = torch.autograd.grad(actual, params)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    for got, reference in zip(actual_grads, reference_grads):
        torch.testing.assert_close(got, reference, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("dependency", ["markov_embed", "markov_head", "main_norm", "wkv_lora_B"])
def test_markov_and_feature_state_invalidates_on_adapted_dependency_change(dependency: str) -> None:
    from offload.dspark.d2.train.cache_guard import DependencyGuard
    weights = {name: torch.ones(2, 2) for name in ("markov_embed", "markov_head", "main_norm", "wkv_lora_B")}
    state = {"Markov": torch.ones(2), "feature": torch.ones(2), "graph": torch.ones(2)}
    guard = DependencyGuard(weights, state.clear)
    guard.check()
    weights[dependency].add_(1)
    with pytest.raises(RuntimeError, match="caches and graphs invalidated"):
        guard.check()
    assert not state
    with pytest.raises(RuntimeError, match="recreate worker"):
        guard.check()
