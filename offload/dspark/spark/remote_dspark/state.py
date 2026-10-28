# SPDX-License-Identifier: Apache-2.0
"""Window bookkeeping shared by the Spark proxy (rank-0 ring) and the mock worker.

CONTRACT.md 7.2: a request's rows form one contiguous position range [lo, hi], clamped
to the last W positions. Rows are stored at ``pos mod W``. Applying rows [p0, p0+n):
  empty or gap (p0 > hi+1)  -> lo = p0
  otherwise (p0 <= hi+1)    -> roll back rows >= p0, lo = min(lo, p0)
  then hi = p0+n-1, lo = max(lo, hi-W+1).
n == 0 is a pure rollback to anchor_pos-1 (requires hi+1 >= anchor_pos).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import wire


@dataclass
class WindowRange:
    """Contiguous [lo, hi] of positions held, or empty (lo is None)."""
    window: int
    lo: int | None = None
    hi: int | None = None

    @property
    def empty(self) -> bool:
        return self.lo is None

    def clear(self) -> None:
        self.lo = self.hi = None

    def apply(self, first_pos: int, n_rows: int, anchor_pos: int) -> None:
        """Apply rows [first_pos, first_pos+n_rows) ending at anchor_pos-1. Raises
        ValueError when the entry is inconsistent (CONTRACT 7.2 steps 2-4)."""
        if not 0 <= n_rows <= self.window:
            raise ValueError(f"n_rows {n_rows} outside 0..{self.window}")
        if anchor_pos < 0:
            raise ValueError(f"anchor_pos {anchor_pos} < 0")
        if n_rows > 0:
            if first_pos < 0:
                raise ValueError(f"first_pos {first_pos} < 0")
            if anchor_pos != first_pos + n_rows:
                raise ValueError(
                    f"anchor_pos {anchor_pos} != first_pos {first_pos} + n_rows {n_rows}")
            p0 = first_pos
            if self.empty or p0 > self.hi + 1:
                self.lo = p0
            else:
                self.lo = min(self.lo, p0)
            self.hi = p0 + n_rows - 1
        else:
            if self.empty or self.hi + 1 < anchor_pos:
                raise ValueError(
                    f"no rows sent and the window ends at {self.hi}, anchor_pos {anchor_pos}")
            self.hi = anchor_pos - 1
            if self.hi < self.lo:
                self.clear()
                return
        self.lo = max(self.lo, self.hi - self.window + 1)
        if self.hi != anchor_pos - 1:
            raise ValueError(f"window ends at {self.hi}, anchor_pos {anchor_pos}")

    def rows_for_anchor(self, anchor_pos: int) -> int:
        """Rows held in [max(0, anchor_pos-W), anchor_pos-1]."""
        if self.empty or self.hi != anchor_pos - 1:
            return 0
        start = max(self.lo, anchor_pos - self.window, 0)
        return max(0, self.hi - start + 1)

    def expected_rows(self, anchor_pos: int) -> int:
        return min(self.window, max(anchor_pos, 0))

    def positions(self, anchor_pos: int | None = None) -> range:
        """Positions held (optionally only those an anchor at anchor_pos may read)."""
        if self.empty:
            return range(0)
        start = self.lo if anchor_pos is None else max(self.lo, anchor_pos - self.window, 0)
        return range(start, self.hi + 1)


@dataclass
class SlotState:
    """Rank-0 view of one request slot (CONTRACT 7.1)."""
    window: int
    owner: str | None = None
    generation: int = 0
    rng: WindowRange = field(default=None)  # type: ignore[assignment]
    dirty: bool = True
    # True once the worker has been told about this generation (NEW_GEN sent and the
    # STEP carrying it succeeded).
    announced: bool = False

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = WindowRange(self.window)


class SlotTable:
    """Per-slot owner / generation / ring range on rank 0."""

    def __init__(self, max_slots: int, window: int):
        self.window = window
        self.slots = [SlotState(window) for _ in range(max_slots)]

    def __getitem__(self, slot: int) -> SlotState:
        return self.slots[slot]

    def begin_request(self, slot: int, req_id: str) -> bool:
        """Note that ``req_id`` owns ``slot`` this round. Returns True when the slot
        changed hands (new generation)."""
        s = self.slots[slot]
        if s.owner == req_id:
            return False
        s.owner = req_id
        s.generation = (s.generation + 1) & wire.U32_MASK
        s.rng.clear()
        s.dirty = True
        s.announced = False
        return True

    def mark_all_dirty(self) -> None:
        for s in self.slots:
            s.dirty = True
            s.announced = False

    def live_slots(self) -> list[int]:
        return [i for i, s in enumerate(self.slots) if s.owner is not None]
