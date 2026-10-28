"""Two-pass CPU audit of frozen teacher/student vocabulary chunk providers.

No network, model loading, training or serving side effects. Providers must expose
identical logits on both passes; chunk hashes enforce that contract. Normalizers
are global FP64 logsumexp, never a softmax within retained top-k support.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable

import torch
from torch import Tensor

from .loss import bucket_divergence, complementary_logaddexp

ChunkLogits = Callable[[int, int], Tensor]


@dataclass(frozen=True)
class VocabAudit:
    teacher_logz: Tensor
    student_logz: Tensor
    teacher_tail: Tensor
    student_tail: Tensor
    full_kl: Tensor
    full_tv: Tensor
    bucket_kl: Tensor
    bucket_tv: Tensor
    teacher_sample_ce: Tensor
    student_sample_ce: Tensor
    proposal_p: Tensor
    proposal_q: Tensor


@torch.no_grad()
def audit_logits(teacher: ChunkLogits, student: ChunkLogits, top_ids: Tensor,
                 sampled_ids: Tensor, proposal_ids: Tensor, vocab: int, chunk: int = 4096) -> VocabAudit:
    """Audit [rows, vocab] logits while retaining only O(rows*chunk) workspace.

    The selected support is the teacher's captured top-k (or a declared audit
    support), with -1 padding. Sample/proposal IDs can lie outside it. -inf logits
    express hard masks; NaN/+inf and entirely masked distributions are refused.
    Full KL can be +inf for a true teacher/student support mismatch.
    Outputs stay per row so a caller can aggregate by private family/segment/position.
    This checks supplied logits only, not their equivalence to a serving engine.
    """
    if top_ids.ndim != 2 or top_ids.shape[0] < 1 or vocab < 1 or chunk < 1:
        raise ValueError("expected nonempty [rows, top-k] IDs, positive vocabulary and chunk")
    rows = top_ids.shape[0]
    for ids in (top_ids, sampled_ids, proposal_ids):
        if ids.device.type != "cpu" or ids.dtype != torch.int64:
            raise ValueError("audit IDs must be CPU int64")
    if sampled_ids.shape != (rows,) or proposal_ids.shape != (rows,):
        raise ValueError("sample/proposal IDs must have one entry per row")
    if bool(((top_ids < -1) | (top_ids >= vocab)).any()) or any(
        bool(((ids < 0) | (ids >= vocab)).any()) for ids in (sampled_ids, proposal_ids)
    ):
        raise ValueError("audit token ID outside vocabulary")
    ordered = top_ids.masked_fill(top_ids < 0, vocab).sort(-1).values
    if bool(((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] != vocab)).any()):
        raise ValueError("duplicate retained token IDs")
    hashes: dict[tuple[str, int], str] = {}

    def checked(provider: ChunkLogits, name: str, lo: int, hi: int, verify: bool) -> Tensor:
        value = provider(lo, hi)
        if value.device.type != "cpu" or not value.is_floating_point() or value.shape != (rows, hi-lo):
            raise ValueError(f"{name} chunk must be floating CPU [rows, chunk]")
        z = value.detach().double()
        if bool((torch.isnan(z) | torch.isposinf(z)).any()):
            raise ValueError(f"{name} chunk contains NaN/+inf")
        digest = hashlib.sha256(z.contiguous().numpy().tobytes()).hexdigest()
        if verify and hashes[(name, lo)] != digest:
            raise ValueError(f"{name} logits changed between audit passes")
        hashes[(name, lo)] = digest
        return z

    def normalize(provider: ChunkLogits, name: str) -> tuple[Tensor, Tensor, Tensor]:
        lse = torch.full((rows,), -torch.inf, dtype=torch.float64)
        tail_lse = lse.clone()
        selected = torch.zeros(top_ids.shape, dtype=torch.float64)
        for lo in range(0, vocab, chunk):
            hi = min(vocab, lo+chunk)
            z = checked(provider, name, lo, hi, False)
            inside = (top_ids >= lo) & (top_ids < hi)
            local = (top_ids-lo).clamp(0, hi-lo-1)
            selected += z.gather(-1, local).masked_fill(~inside, 0)
            retained = torch.zeros_like(z, dtype=torch.int64).scatter_add(-1, local, inside.long()).bool()
            remainder = z.masked_fill(retained, -torch.inf)
            has_tail = torch.isfinite(remainder).any(-1)
            # Stable even for an empty chunk or a chunk masked out by processors.
            safe = torch.where(has_tail[:, None], remainder, torch.zeros_like(remainder))
            part_tail = safe.logsumexp(-1).masked_fill(~has_tail, -torch.inf)
            lse = complementary_logaddexp(lse, z.logsumexp(-1))
            tail_lse = complementary_logaddexp(tail_lse, part_tail)
        if not bool(torch.isfinite(lse).all()):
            raise ValueError(f"{name} has an entirely masked distribution")
        return lse, tail_lse-lse, (selected-lse[:, None]).masked_fill(top_ids < 0, -torch.inf)

    pz, pt, lp_top = normalize(teacher, "teacher")
    qz, qt, lq_top = normalize(student, "student")
    kl = torch.zeros(rows, dtype=torch.float64)
    tv = kl.clone()
    label_lp, label_lq, proposal_lp, proposal_lq = [kl.clone() for _ in range(4)]
    for lo in range(0, vocab, chunk):
        hi = min(vocab, lo+chunk)
        lp = checked(teacher, "teacher", lo, hi, True)-pz[:, None]
        lq = checked(student, "student", lo, hi, True)-qz[:, None]
        p, q = lp.exp(), lq.exp()
        # Zero teacher probabilities contribute zero, including a jointly masked token.
        zero = p == 0
        kl += (p * (lp.masked_fill(zero, 0)-lq.masked_fill(zero, 0))).sum(-1)
        tv += .5 * (p-q).abs().sum(-1)
        for ids, dest_p, dest_q in ((sampled_ids, label_lp, label_lq), (proposal_ids, proposal_lp, proposal_lq)):
            inside = (ids >= lo) & (ids < hi)
            local = (ids-lo).clamp(0, hi-lo-1)[:, None]
            dest_p += lp.gather(-1, local).squeeze(-1).masked_fill(~inside, 0)
            dest_q += lq.gather(-1, local).squeeze(-1).masked_fill(~inside, 0)
    return VocabAudit(pz, qz, pt.exp(), qt.exp(), kl, tv,
                      bucket_divergence(lq_top, qt, top_ids, lp_top, pt.exp(), "kl"),
                      bucket_divergence(lq_top, qt, top_ids, lp_top, pt.exp(), "tvd"),
                      -label_lp, -label_lq, proposal_lp.exp(), proposal_lq.exp())
