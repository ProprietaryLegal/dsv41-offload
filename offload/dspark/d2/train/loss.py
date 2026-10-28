"""Top-20 plus complementary-tail forward KL, with chunk-recomputed logits."""
from __future__ import annotations

from typing import Callable, Literal, Protocol

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .model import Drafter

POSITION_WEIGHTS = (1., .8, .65, .5, .4)


class Vocab(Protocol):
    @property
    def vocab(self) -> int: ...


class LogitModel(Protocol):
    @property
    def c(self) -> Vocab: ...
    def logit_chunk(self, hidden: Tensor, previous: Tensor, start: int, end: int) -> Tensor: ...


def complementary_logaddexp(a: Tensor, b: Tensor) -> Tensor:
    """Merge possibly empty chunks without the undefined (-inf, -inf) gradient."""
    empty = torch.isneginf(a) & torch.isneginf(b)
    return torch.logaddexp(a.masked_fill(empty, 0), b.masked_fill(empty, 0)).masked_fill(empty, -torch.inf)


def bucket_divergence(logq: Tensor, logqt: Tensor, ids: Tensor, logp: Tensor, tail: Tensor,
                      divergence: Literal["kl", "tvd"]) -> Tensor:
    """Coarse top-k plus tail divergence; zero teacher buckets contribute zero.

    This is not full-vocabulary KL/TV: within-tail allocation is unobserved.
    logqt comes from the complementary logsumexp, never 1-sum(q_top).
    """
    p = logp.exp().masked_fill(ids < 0, 0)
    if divergence == "kl":
        top_loss = (torch.xlogy(p, p) - p * logq.masked_fill(p == 0, 0)).sum(-1)
        tail_loss = torch.xlogy(tail, tail) - tail * logqt.masked_fill(tail == 0, 0)
        return top_loss + tail_loss
    q = logq.exp().masked_fill(ids < 0, 0)
    return .5 * ((p-q).abs().sum(-1) + (tail-logqt.exp()).abs())


def bucket_statistics(logits: Callable[[int, int], Tensor], ids: Tensor, vocab: int, chunk: int,
                      recompute: bool = True) -> tuple[Tensor, Tensor, Tensor]:
    """logZ, logZ(complement), selected logits. No subtraction of nearly equal masses.

    Each chunk checkpoint recomputes its logits in backward rather than retaining
    the 129K-vocabulary activation/Markov head graph. -1 is missing top-k padding.
    """
    valid = ids >= 0
    if chunk < 1 or bool((ids >= vocab).any()):
        raise ValueError("invalid chunk size or top-k id")
    sorted_ids = ids.masked_fill(~valid, vocab).sort(-1).values
    if bool(((sorted_ids[..., 1:] == sorted_ids[..., :-1]) & (sorted_ids[..., 1:] != vocab)).any()):
        raise ValueError("duplicate top-k ids")
    lse = torch.full(ids.shape[:-1], -torch.inf, device=ids.device, dtype=torch.float32)
    tail = lse.clone()
    selected = torch.zeros_like(ids, dtype=torch.float32)
    for lo in range(0, vocab, chunk):
        hi = min(vocab, lo + chunk)

        def part(dummy: Tensor, start: int = lo, end: int = hi) -> tuple[Tensor, Tensor, Tensor]:
            z = logits(start, end).float()
            inside = valid & (ids >= start) & (ids < end)
            loc = (ids - start).clamp(0, end-start-1)
            chosen = z.gather(-1, loc).masked_fill(~inside, 0)
            # scatter_add counts instead of scatter avoids padding overwriting
            # a real top-id at the same clamped index.
            counts = torch.zeros_like(z, dtype=torch.int32).scatter_add(-1, loc, inside.int())
            remainder = z.masked_fill(counts != 0, -torch.inf)
            any_tail = (counts == 0).any(-1)
            safe = torch.where(any_tail.unsqueeze(-1), remainder, torch.zeros_like(remainder))
            tail_z = safe.logsumexp(-1).masked_fill(~any_tail, -torch.inf)
            return z.logsumexp(-1), tail_z, chosen

        if recompute and torch.is_grad_enabled():
            # Non-reentrant checkpoint tracks closed-over model parameters too.
            a, b, t = checkpoint(part, lse, use_reentrant=False)
        else:
            a, b, t = part(lse)
        lse, tail, selected = torch.logaddexp(lse, a), complementary_logaddexp(tail, b), selected + t
    return lse, tail, selected


def teacher_argmax(ids: Tensor, logp: Tensor, vocab: int) -> Tensor:
    """Teacher top-k mode; equal log probabilities choose the lowest token id.

    Empty rows get a harmless lookup id and are excluded by the ACE mask.
    """
    valid = ids >= 0
    best = logp.masked_fill(~valid, -torch.inf).amax(-1, keepdim=True)
    picks = ids.masked_fill(~valid | (logp != best), vocab).amin(-1)
    return picks.clamp_max(vocab-1)


def distillation_loss(model: LogitModel, hidden: Tensor, previous: Tensor, top_ids: Tensor, top_logp: Tensor,
                      tail_mass: Tensor, labels: Tensor, present: Tensor, position_mask: Tensor,
                      chunk: int = 4096, divergence: Literal["kl", "tvd", "ace"] = "kl"
                      ) -> tuple[Tensor, Tensor]:
    if divergence not in ("kl", "tvd", "ace"):
        raise ValueError(divergence)
    valid = top_ids >= 0
    p = top_logp.exp().masked_fill(~valid, 0)
    if bool((~torch.isfinite(p)).any()) or bool((~torch.isfinite(tail_mass)).any()):
        raise ValueError("nonfinite target distribution")
    if bool(((tail_mass < 0) | (tail_mass > 1)).any()):
        raise ValueError("tail mass outside [0,1]")
    observed = present & valid.any(-1) if divergence == "ace" else present
    if not torch.allclose((p.sum(-1)+tail_mass)[observed], torch.ones_like(tail_mass[observed]), atol=1e-4, rtol=0):
        raise ValueError("top-k probabilities plus tail do not sum to one")
    if divergence != "ace" and bool((present & ~valid.any(-1)).any()):
        raise ValueError("present top-k distribution is empty")
    if divergence == "ace":
        position_mask = position_mask & present & valid.any(-1)
    if isinstance(model, Drafter) and model.fast_training:
        from .fast_loss import bucket_loss
        if chunk < 1 or bool((top_ids >= model.c.vocab).any()):
            raise ValueError("invalid chunk size or top-k id")
        ordered = top_ids.masked_fill(~valid, model.c.vocab).sort(-1).values
        if bool(((ordered[..., 1:] == ordered[..., :-1]) & (ordered[..., 1:] != model.c.vocab)).any()):
            raise ValueError("duplicate top-k ids")
        missing = ~present & position_mask
        if bool(missing.any()) and (bool(((labels < 0) & missing).any()) or bool((labels >= model.c.vocab).any())):
            raise ValueError("missing-top20 position lacks a sampled label")
        if not bool(position_mask.any()):
            raise ValueError("batch has no scored positions")
        loss, per = bucket_loss(hidden, model.t["head.weight"], model.last("markov_head.embed.weight"),
                                model.last("markov_head.head.weight"), previous, top_ids, top_logp,
                                tail_mass, labels, present, position_mask, chunk, divergence)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite distillation loss")
        return loss, per
    # Include sampled label in statistics only as a separate lookup; it must
    # not be removed from the complement bucket unless actually in the top-k.
    fn = lambda lo, hi: model.logit_chunk(hidden, previous, lo, hi)
    lse, tail_lse, selected = bucket_statistics(fn, top_ids, model.c.vocab, chunk)
    logq = selected - lse.unsqueeze(-1)
    logqt = tail_lse - lse
    if divergence == "ace":
        picks = teacher_argmax(top_ids, top_logp, model.c.vocab)
        per = lse - selected.masked_fill(top_ids != picks.unsqueeze(-1), 0).sum(-1)
        per = per.masked_fill(~position_mask, 0)
    else:
        per = bucket_divergence(logq, logqt, top_ids, top_logp, tail_mass, divergence)
    missing = ~present & position_mask
    if bool(missing.any()):
        safe_label = labels.clamp_min(0)
        if bool(((labels < 0) & missing).any()) or bool((labels >= model.c.vocab).any()):
            raise ValueError("missing-top20 position lacks a sampled label")
        # This second streamed lookup computes ordinary sampled CE only for
        # positions with missing top20. It never substitutes CE on known targets.
        _, _, label_logits = bucket_statistics(fn, safe_label.unsqueeze(-1), model.c.vocab, chunk)
        per = torch.where(present, per, .1 * (lse-label_logits.squeeze(-1)))
    else:
        per = torch.where(present, per, 0)
    weights = hidden.new_tensor(POSITION_WEIGHTS).expand_as(per) * position_mask
    if not bool(weights.sum() > 0):
        raise ValueError("batch has no scored positions")
    loss = (per * weights).sum() / weights.sum()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("nonfinite distillation loss")
    return loss, per.detach()


def confidence_loss(model: Drafter, hidden: Tensor, head_x: Tensor, previous: Tensor, top_ids: Tensor,
                    top_logp: Tensor, position_mask: Tensor, chunk: int = 4096) -> tuple[Tensor | None, int]:
    # Calibrate each greedy conditional at a committed previous token. If q's
    # argmax is outside top20 its exact p is unknown; do not invent a zero label.
    with torch.no_grad():
        best = torch.full(previous.shape, -torch.inf, device=hidden.device)
        picks = torch.zeros_like(previous)
        for lo in range(0, model.c.vocab, chunk):
            val, idx = model.logit_chunk(hidden, previous, lo, min(lo+chunk, model.c.vocab)).max(-1)
            picks = torch.where(val > best, idx+lo, picks)
            best = torch.maximum(best, val)
        match = top_ids == picks.unsqueeze(-1)
        known = match.any(-1) & position_mask
        target = top_logp.exp().masked_fill(~match, 0).sum(-1)
    if not bool(known.any()):
        return None, 0  # an unlabelled block is skipped, never assigned a fabricated zero probability
    prediction = model.confidence(head_x.detach(), previous).sigmoid()
    return (prediction[known]-target[known]).square().mean(), int(known.sum())
