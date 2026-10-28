from __future__ import annotations

from dataclasses import fields

import pytest
import torch
from torch import Tensor

from offload.dspark.d2.train.vocab_audit import VocabAudit, audit_logits


@pytest.mark.parametrize("chunk", (1, 3, 11))
def test_streamed_full_vocab_matches_dense_with_global_normalization(chunk: int) -> None:
    g = torch.Generator().manual_seed(189)
    teacher = torch.randn(3, 11, generator=g, dtype=torch.float64) + 900
    student = torch.randn(3, 11, generator=g, dtype=torch.float64) - 900
    top = teacher.topk(3, -1).indices
    labels = torch.tensor([0, 1, 10])
    proposals = torch.tensor([7, 10, 0])
    calls: list[int] = []

    def supply(value: Tensor, lo: int, hi: int) -> Tensor:
        calls.append(hi-lo)
        return value[:, lo:hi]

    got = audit_logits(lambda lo, hi: supply(teacher, lo, hi), lambda lo, hi: supply(student, lo, hi),
                       top, labels, proposals, 11, chunk)
    lp, lq = teacher.log_softmax(-1), student.log_softmax(-1)
    p, q = lp.exp(), lq.exp()
    pt, qt = p.scatter(-1, top, 0).sum(-1), q.scatter(-1, top, 0).sum(-1)
    ptop, qtop = p.gather(-1, top), q.gather(-1, top)
    expected = VocabAudit(teacher.logsumexp(-1), student.logsumexp(-1), pt, qt,
                          (p*(lp-lq)).sum(-1), .5*(p-q).abs().sum(-1),
                          (ptop*(ptop.log()-qtop.log())).sum(-1) + pt*(pt.log()-qt.log()),
                          .5*((ptop-qtop).abs().sum(-1)+(pt-qt).abs()),
                          -lp.gather(-1, labels[:, None]).squeeze(-1),
                          -lq.gather(-1, labels[:, None]).squeeze(-1),
                          p.gather(-1, proposals[:, None]).squeeze(-1), q.gather(-1, proposals[:, None]).squeeze(-1))
    for field in fields(got):
        torch.testing.assert_close(getattr(got, field.name), getattr(expected, field.name), rtol=1e-11, atol=1e-11)
    assert max(calls) <= chunk
    assert bool((got.full_kl >= got.bucket_kl-1e-12).all())
    assert bool((got.full_tv >= got.bucket_tv-1e-12).all())


def test_audit_zero_mass_full_support_and_padded_support() -> None:
    z = torch.tensor([[0., -torch.inf, 1.]])
    ids = torch.tensor([[0, 1, 2, -1]])
    pick = torch.tensor([2])
    got = audit_logits(lambda lo, hi: z[:, lo:hi], lambda lo, hi: z[:, lo:hi], ids, pick, pick, 3, 1)
    assert got.teacher_tail.item() == got.student_tail.item() == 0
    for name in ("full_kl", "full_tv", "bucket_kl", "bucket_tv"):
        assert getattr(got, name).item() == 0


def test_support_mismatch_is_infinite_kl_not_nan_or_a_probability_floor() -> None:
    p = torch.tensor([[0., -torch.inf, 1.]])
    q = torch.tensor([[-torch.inf, 0., 1.]])
    pick = torch.tensor([0])
    got = audit_logits(lambda lo, hi: p[:, lo:hi], lambda lo, hi: q[:, lo:hi], pick[:, None], pick, pick, 3, 2)
    assert torch.isposinf(got.full_kl).all() and torch.isposinf(got.bucket_kl).all()
    assert torch.isfinite(got.full_tv).all()


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), -float("inf")))
def test_invalid_teacher_distribution_is_refused(bad: float) -> None:
    ids = torch.tensor([[0]])
    label = torch.tensor([0])
    z = torch.full((1, 3), bad)
    with pytest.raises(ValueError, match=r"NaN/\+inf|entirely masked"):
        audit_logits(lambda lo, hi: z[:, lo:hi], lambda lo, hi: z[:, lo:hi], ids, label, label, 3)


def test_mutated_second_pass_is_refused() -> None:
    calls = 0

    def changed(lo: int, hi: int) -> Tensor:
        nonlocal calls
        calls += 1
        return torch.full((1, hi-lo), float(calls > 2))

    ids = torch.tensor([[0]])
    label = torch.tensor([0])
    with pytest.raises(ValueError, match="changed between audit passes"):
        audit_logits(changed, lambda lo, hi: torch.zeros(1, hi-lo), ids, label, label, 3, 2)
