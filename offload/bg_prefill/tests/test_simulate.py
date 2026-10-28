"""Conserved work and honest cost-model accounting, not a speedup acceptance test."""
from __future__ import annotations

import pytest

from offload.bg_prefill.simulate import Cost, simulate


@pytest.mark.parametrize('backgrounds', [1, 2])
@pytest.mark.parametrize('priority', [0, 10])
def test_equal_full_work_and_reported_fairness_cost(backgrounds: int, priority: int) -> None:
    control = simulate(backgrounds, priority, False)
    candidate = simulate(backgrounds, priority, True)
    for row in (control, candidate):
        assert row['background_prompt_tokens_total'] == backgrounds * 60000
        assert row['estimated_foreground_tokens'] == 600
        # Additive model: all work has the same exact cost, regardless of scheduling.
        assert row['all_work_s'] == pytest.approx(300 * .039 + backgrounds * 60000 / 2200)
    assert candidate['background_tokens_at_foreground_end'] == backgrounds * 12 * 32
    assert candidate['max_background_service_gap_calls_during_overlap'] == 24


def test_simulation_refuses_invalid_costs_and_workloads() -> None:
    with pytest.raises(ValueError):
        Cost(decode_ms=0)
    with pytest.raises(ValueError):
        simulate(3, 0, True)
