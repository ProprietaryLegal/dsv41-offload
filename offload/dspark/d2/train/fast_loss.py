"""Chunked GPU bucket loss with explicit head derivatives and bounded logit cache.

Only FP32 logits are retained, rather than the per-chunk autograd graphs and
full-table slice gradients. This saves a repeated frozen-head forward in
backward. No CPU reductions or trainable weight copies leave the GPU.
"""
from __future__ import annotations

from typing import Callable, Literal, Protocol, cast

import torch
from torch import Tensor
from torch.nn import functional as F

from .ops import check_cancelled, head_product


class BucketContext(Protocol):
    chunk: int
    divergence: str
    shape: torch.Size
    saved_tensors: tuple[Tensor, ...]

    def save_for_backward(self, *tensors: Tensor) -> None: ...
    def mark_non_differentiable(self, *tensors: Tensor) -> None: ...


class _BucketLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx: BucketContext, hidden: Tensor, head: Tensor, embed: Tensor, markov: Tensor, previous: Tensor,
                ids: Tensor, logp: Tensor, tail: Tensor, labels: Tensor, present: Tensor, mask: Tensor,
                chunk: int, divergence: Literal["kl", "tvd", "ace"]) -> tuple[Tensor, Tensor]:
        from .loss import POSITION_WEIGHTS, bucket_divergence, complementary_logaddexp, teacher_argmax
        if divergence == "ace":
            mask = mask & present & (ids >= 0).any(-1)
            labels = teacher_argmax(ids, logp, head.shape[0])
            if not bool(mask.any()):
                raise ValueError("batch has no scored positions")
        flat = hidden.reshape(-1, hidden.shape[-1])
        emb = F.embedding(previous, embed).reshape(-1, embed.shape[-1])
        shape = previous.shape
        valid = ids >= 0
        lse = torch.full(shape, -torch.inf, device=hidden.device)
        tail_lse = lse.clone()
        selected = torch.zeros_like(logp)
        label_z = torch.zeros_like(tail)
        saved_logits = []
        for lo in range(0, head.shape[0], chunk):
            check_cancelled()
            hi = min(lo + chunk, head.shape[0])
            z = (head_product(flat, head[lo:hi].T) + F.linear(emb, markov[lo:hi])).reshape(*shape, hi-lo)
            saved_logits.append(z)
            inside = valid & (ids >= lo) & (ids < hi)
            loc = (ids-lo).clamp(0, hi-lo-1)
            selected += z.gather(-1, loc).masked_fill(~inside, 0)
            counts = torch.zeros_like(z, dtype=torch.int32).scatter_add(-1, loc, inside.int())
            remainder = z.masked_fill(counts != 0, -torch.inf)
            any_tail = (counts == 0).any(-1)
            safe = torch.where(any_tail.unsqueeze(-1), remainder, torch.zeros_like(remainder))
            part_tail = safe.logsumexp(-1).masked_fill(~any_tail, -torch.inf)
            lse = torch.logaddexp(lse, z.logsumexp(-1))
            tail_lse = complementary_logaddexp(tail_lse, part_tail)
            chosen_label = z.gather(-1, (labels-lo).clamp(0, hi-lo-1).unsqueeze(-1)).squeeze(-1)
            label_z += chosen_label.masked_fill((labels < lo) | (labels >= hi), 0)
        p = logp.exp().masked_fill(~valid, 0)
        logq = selected - lse.unsqueeze(-1)
        logqt = tail_lse - lse
        if divergence == "ace":
            per = (lse-label_z).masked_fill(~mask, 0)
        else:
            per = bucket_divergence(logq, logqt, ids, logp, tail, divergence)
            per = torch.where(present, per, .1 * (lse-label_z) if bool((~present & mask).any()) else 0.)
        weights = hidden.new_tensor(POSITION_WEIGHTS).expand_as(per) * mask
        norm_weights = weights / weights.sum()
        loss = (per * weights).sum() / weights.sum()
        ctx.chunk, ctx.divergence, ctx.shape = chunk, divergence, hidden.shape
        ctx.save_for_backward(head, embed, markov, previous, ids, p, tail, labels, present, norm_weights,
                              lse, tail_lse, selected, *saved_logits)
        detached = per.detach().clone()
        ctx.mark_non_differentiable(detached)
        return loss, detached

    @staticmethod
    def backward(ctx: BucketContext, grad: Tensor, grad_per: Tensor | None) -> tuple[Tensor | None, ...]:
        (head, embed, markov, previous, ids, p, tail, labels, present, weights,
         lse, tail_lse, selected, *logits) = ctx.saved_tensors
        emb = F.embedding(previous, embed).reshape(-1, embed.shape[-1])
        dh = torch.zeros(previous.numel(), head.shape[-1], device=head.device)
        de = torch.zeros_like(emb)
        dm = torch.empty_like(markov)
        valid = ids >= 0
        if ctx.divergence == "tvd":
            qtop = (selected-lse.unsqueeze(-1)).exp().masked_fill(~valid, 0)
            qt = (tail_lse-lse).exp()
            atop = .5 * (qtop-p).sign()
            atail = .5 * (qt-tail).sign()
            weighted = (atop*qtop).sum(-1) + atail*qt
        for part, z in enumerate(logits):
            check_cancelled()
            lo = part * ctx.chunk
            hi = lo + z.shape[-1]
            inside = valid & (ids >= lo) & (ids < hi)
            loc = (ids-lo).clamp(0, hi-lo-1)
            counts = torch.zeros_like(z, dtype=torch.int32).scatter_add(-1, loc, inside.int())
            q = (z-lse.unsqueeze(-1)).exp()
            if ctx.divergence == "ace":
                dz = q.clone()
                label_inside = (labels >= lo) & (labels < hi)
                dz.scatter_add_(-1, (labels-lo).clamp(0, hi-lo-1).unsqueeze(-1), -label_inside.float().unsqueeze(-1))
            elif ctx.divergence == "kl":
                # Use the supplied mass, including its permitted rounding error;
                # the reference differentiates this exact, unrenormalized target.
                dz = q * (p.sum(-1)+tail).unsqueeze(-1)
                dz -= tail.unsqueeze(-1) * (z-tail_lse.unsqueeze(-1)).exp().masked_fill(counts != 0, 0)
                dz.scatter_add_(-1, loc, -p.masked_fill(~inside, 0))
            else:
                coeff = atail.unsqueeze(-1).expand_as(z).clone()
                coeff.scatter_add_(-1, loc, (atop-atail.unsqueeze(-1)).masked_fill(~inside, 0))
                dz = q * (coeff-weighted.unsqueeze(-1))
            ce = q.clone()
            label_inside = (labels >= lo) & (labels < hi)
            ce.scatter_add_(-1, (labels-lo).clamp(0, hi-lo-1).unsqueeze(-1), -label_inside.float().unsqueeze(-1))
            if ctx.divergence != "ace":
                dz = torch.where(present.unsqueeze(-1), dz, .1 * ce)
            dz *= (weights * grad).unsqueeze(-1)
            dz2 = dz.reshape(-1, hi-lo)
            dh += head_product(dz2, head[lo:hi])
            de.addmm_(dz2, markov[lo:hi])
            dm[lo:hi] = dz2.T @ emb
        dembed = torch.zeros_like(embed)
        dembed.index_add_(0, previous.flatten(), de)
        return (dh.reshape(ctx.shape), None, dembed, dm, None, None, None, None, None, None, None, None, None)


def bucket_loss(hidden: Tensor, head: Tensor, embed: Tensor, markov: Tensor, previous: Tensor,
                ids: Tensor, logp: Tensor, tail: Tensor, labels: Tensor, present: Tensor, mask: Tensor,
                chunk: int, divergence: Literal["kl", "tvd", "ace"]) -> tuple[Tensor, Tensor]:
    return cast(Callable[..., tuple[Tensor, Tensor]], _BucketLoss.apply)(
        hidden, head, embed, markov, previous, ids, logp, tail, labels, present, mask, chunk, divergence)
