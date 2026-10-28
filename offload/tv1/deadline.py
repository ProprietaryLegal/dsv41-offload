"""Immutable HOLD-based TV1 admission/recovery budget, shared by every call."""
from __future__ import annotations

import math
from dataclasses import dataclass

SUBMIT_S, ABORT_S, RESTORE_S = 85*60, 86*60, 98*60


@dataclass(frozen=True)
class HoldBudget:
    opened: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.opened):
            raise ValueError('TV1 invalid HOLD timestamp')

    @property
    def submission(self) -> float:
        return self.opened + SUBMIT_S

    @property
    def abort(self) -> float:
        return self.opened + ABORT_S

    @property
    def restore(self) -> float:
        return self.opened + RESTORE_S

    def cap(self, label: str, timeout: float, now: float) -> float:
        left = self.restore - now
        if not math.isfinite(now) or not math.isfinite(timeout) or left <= 0 or timeout <= 0:
            raise ValueError(f'TV1 HOLD+98 recovery deadline exhausted before {label}; HOLD stays open, manual restore required')
        return min(left, timeout)
