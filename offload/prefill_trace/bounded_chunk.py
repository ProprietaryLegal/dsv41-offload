"""Detached CPU design for a time-budget proposer. NOT connected to the scheduler.

Default disabled returns the caller's upstream choice unchanged. An enabled call
is only valid AFTER admission/restore/KV checks reserve the stated token budget.
Time is a soft bound: a ready admitted request always gets its positive floor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    fixed_ms: float
    token_ms: float
    token_prefix_ms_per_100K: float

    def __post_init__(self) -> None:
        if any(not math.isfinite(x) or x < 0 for x in
               (self.fixed_ms, self.token_ms, self.token_prefix_ms_per_100K)):
            raise ValueError('cost coefficients must be finite nonnegative measurements')
        if self.token_ms == self.token_prefix_ms_per_100K == 0:
            raise ValueError('cost model must charge new tokens')

    def predict(self, new_tokens: int, prefix: int) -> float:
        if new_tokens <= 0 or prefix < 0:
            raise ValueError('predict requires positive new rows and nonnegative prefix')
        # Integrate increasing context over the new rows, rather than charging
        # every token at the beginning or final prefix. This is a fitted lead,
        # not a claim that real sparse/EXL3 work is exactly quadratic.
        prefix_work = new_tokens * (prefix + (new_tokens - 1) / 2) / 100000
        value = self.fixed_ms + self.token_ms * new_tokens + self.token_prefix_ms_per_100K * prefix_work
        if not math.isfinite(value):
            raise ValueError('cost prediction overflow')
        return value


@dataclass(frozen=True)
class Decision:
    tokens: int
    predicted_ms: float | None
    exceeds_soft_budget: bool
    floor: int


def propose(upstream_tokens: int, remaining: int, prefix: int, reserved_tokens: int,
            budget_ms: float, model: CostModel, *, enabled: bool = False,
            minimum_tokens: int = 1, must_trim_successor: bool = False,
            replay_window: int = 128) -> Decision:
    """Largest fitted chunk within budget, or a positive reserved progress floor.

    No queue classification, priority reordering, KV admission, prefix restore or
    speculative rollback is performed here. The caller owns each of those gates.
    reserved_tokens must already exclude decode work, async restore constraints
    and reservations for other admitted requests; an unadmittable waiter is not
    an input. Impossible floors refuse rather than returning a silent zero.
    """
    if not enabled:
        return Decision(upstream_tokens, None, False, 0)
    if not math.isfinite(budget_ms) or budget_ms < 0:
        raise ValueError('time budget must be finite and nonnegative')
    if min(upstream_tokens, remaining, reserved_tokens, minimum_tokens, replay_window) < 1 or prefix < 0:
        raise ValueError('enabled proposer requires a ready admitted request with reserved positive resources')
    if must_trim_successor and remaining <= replay_window:
        raise ValueError('invalid replay obligation: a skipped predecessor cannot leave only a short tail')
    floor = min(remaining, max(minimum_tokens, replay_window + 1 if must_trim_successor else 1))
    ceiling = min(upstream_tokens, remaining, reserved_tokens)
    if ceiling < floor:
        raise ValueError('reserved token floor unavailable: recover/rebuild real suffix or reserve next turn; never spin at zero')
    lo, hi, choice = floor, ceiling, floor
    while lo <= hi:
        mid = (lo + hi) // 2
        if model.predict(mid, prefix) <= budget_ms:
            choice, lo = mid, mid + 1
        else:
            hi = mid - 1
    predicted = model.predict(choice, prefix)
    return Decision(choice, predicted, predicted > budget_ms, floor)
