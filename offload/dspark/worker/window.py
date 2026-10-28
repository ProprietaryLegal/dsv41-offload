"""Per-slot window bookkeeping of CONTRACT.md 7.2 (v1.3), the worker's copy.

The rule (same as lane O3's remote_dspark/state.py WindowRange, which the Spark proxy uses for its rank-0 ring): a
request's rows form one contiguous position range [lo, hi], at most W long, stored at pos mod W. Applying rows
[p0, p0+n): empty or gap (p0 > hi+1) -> lo = p0; else roll back rows >= p0 and lo = min(lo, p0); then hi = p0+n-1,
lo = max(lo, hi-W+1). n == 0 is a pure rollback to anchor_pos-1 (requires hi+1 >= anchor_pos).
Because [lo, hi] never exceeds W consecutive positions, its pos-mod-W slots are distinct, and every position in it was
written by the current generation (lane O3's argument, CONTRACT 7.2); stale ring entries lie outside [lo, hi].
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WindowRange:
    window: int
    lo: int | None = None
    hi: int | None = None

    @property
    def empty(self) -> bool:
        return self.lo is None

    def clear(self) -> None:
        self.lo = self.hi = None

    def apply(self, first_pos: int, n_rows: int, anchor_pos: int) -> None:
        """Raises ValueError (-> BAD_ENTRY) when the entry is inconsistent (CONTRACT 7.2 steps 2-4)."""
        if not 0 <= n_rows <= self.window:
            raise ValueError(f"n_rows {n_rows} outside 0..{self.window}")
        if anchor_pos < 0:
            raise ValueError(f"anchor_pos {anchor_pos} < 0")
        if n_rows > 0:
            if first_pos < 0:
                raise ValueError(f"first_pos {first_pos} < 0")
            if anchor_pos != first_pos + n_rows:
                raise ValueError(f"anchor_pos {anchor_pos} != first_pos {first_pos} + n_rows {n_rows}")
            p0 = first_pos
            if self.empty or p0 > self.hi + 1:
                self.lo = p0
            else:
                self.lo = min(self.lo, p0)
            self.hi = p0 + n_rows - 1
        else:
            if self.empty or self.hi + 1 < anchor_pos:
                raise ValueError(f"no rows sent and the window ends at {self.hi}, anchor_pos {anchor_pos}")
            self.hi = anchor_pos - 1
            if self.hi < self.lo:
                self.clear()
                return
        self.lo = max(self.lo, self.hi - self.window + 1)
        if self.hi != anchor_pos - 1:
            raise ValueError(f"window ends at {self.hi}, anchor_pos {anchor_pos}")

    def read_lo(self, anchor_pos: int) -> int:
        """Lowest position an anchor at anchor_pos reads: max(lo, anchor_pos - W, 0)."""
        return max(self.lo if self.lo is not None else anchor_pos, anchor_pos - self.window, 0)

    def rows_for_anchor(self, anchor_pos: int) -> int:
        if self.empty or self.hi != anchor_pos - 1:
            return 0
        return max(0, self.hi - self.read_lo(anchor_pos) + 1)

    def expected_rows(self, anchor_pos: int) -> int:
        return min(self.window, max(anchor_pos, 0))

    def positions(self, anchor_pos: int | None = None) -> range:
        if self.empty:
            return range(0)
        start = self.lo if anchor_pos is None else self.read_lo(anchor_pos)
        return range(start, self.hi + 1)
