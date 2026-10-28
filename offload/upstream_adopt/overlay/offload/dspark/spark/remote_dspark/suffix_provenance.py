# SPDX-License-Identifier: Apache-2.0
"""Optional replay qualification metadata, never text/tensors or a wire change.

observe() runs on BOTH engine ranks only under VLLM_DSV41_REPLAY_AUDIT=1.
The owner epoch is engine-observation-local; STEP's worker generation remains
separately checked by RoundCore. Only metadata digests/counts may be logged.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence


@dataclass(frozen=True)
class SuffixRow:
    owner_digest: str
    slot: int
    owner_epoch: int
    q: int
    kept: int
    first_pos: int
    valid: int
    anchor_pos: int
    anchor_token: int
    need_draft: bool
    lo: int


Snapshot = tuple[str | None, tuple[SuffixRow, ...]]


class SuffixAudit:
    def __init__(self, window: int, max_slots: int) -> None:
        self.window = window
        self.max_slots = max_slots
        self.owners: dict[int, str] = {}
        self.epochs: dict[int, int] = {}
        self.ranges: dict[int, tuple[int, int]] = {}

    def observe(self, owners: Sequence[str], slots: Sequence[int], qsl: Sequence[int],
                kept: Sequence[int] | None, positions: Sequence[int],
                rejected: Sequence[int], sampled: Sequence[int],
                last: Sequence[int], next_prefill: Sequence[int],
                need: Sequence[bool]) -> tuple[SuffixRow, ...]:
        n = len(owners)
        if (n < 1 or n > self.max_slots or len(qsl) != n + 1
            or qsl[0] != 0 or qsl[-1] != len(positions)
            or any(len(v) != n for v in (slots, rejected, sampled, need))
            or kept is not None and len(kept) != n or len(set(slots)) != n):
            raise ValueError('invalid replay audit batch')
        result: list[SuffixRow] = []
        for i, owner in enumerate(owners):
            slot = slots[i]
            if not 0 <= slot < self.max_slots:
                raise ValueError('invalid replay audit slot')
            q = qsl[i + 1] - qsl[i]
            k = q if kept is None else kept[i]
            if q <= 0 or not 0 < k <= q or not 0 <= rejected[i] <= q:
                raise ValueError('invalid replay audit provenance')
            if need[i] and k < min(q, self.window):
                raise ValueError('replay audit draft lacks computed taps')
            count = min(q, k, self.window)
            start = qsl[i + 1] - count
            first = positions[start]
            if first < 0 or list(positions[start:qsl[i + 1]]) != list(range(first, first + count)):
                raise ValueError('replay audit positions are not contiguous')
            valid = max(0, count - rejected[i])
            anchor_pos = first + valid
            if self.owners.get(slot) != owner:
                self.epochs[slot] = (self.epochs.get(slot, 0) + 1) & 0xFFFFFFFF
                self.owners[slot] = owner
                self.ranges.pop(slot, None)
            old = self.ranges.get(slot)
            if valid == 0:
                self.ranges.pop(slot, None)
                lo = anchor_pos
            else:
                lo = first if old is None or first > old[1] + 1 else min(old[0], first)
                lo = max(lo, anchor_pos - self.window)
                self.ranges[slot] = (lo, anchor_pos - 1)
            if need[i] and anchor_pos - lo < min(anchor_pos, self.window):
                raise ValueError('replay audit final suffix incomplete')
            anchor = last[slot] if sampled[i] > 0 else next_prefill[slot]
            result.append(SuffixRow(hashlib.sha256(owner.encode()).hexdigest(), slot,
                                    self.epochs[slot], q, k, first, valid, anchor_pos,
                                    anchor, bool(need[i]), lo))
        return tuple(result)


def agree(snapshots: Sequence[Snapshot]) -> None:
    """All participants exchange even local failures before raising together."""
    if not snapshots or any(error is not None for error, _ in snapshots):
        raise RuntimeError('0075 rank provenance invalid')
    if any(rows != snapshots[0][1] for _, rows in snapshots[1:]):
        raise RuntimeError('0075 rank provenance mismatch')
