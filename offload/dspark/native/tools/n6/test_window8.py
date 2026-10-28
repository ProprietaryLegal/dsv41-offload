from __future__ import annotations

import pytest

from offload.dspark.native.tools.n6.window8 import Run, summarize


def run(rounds: float, positions: list[float], proposed: float) -> Run:
    return dict(type='run', concurrency=3, mixed=False, rep=0, error=None, aggregate_tps=10.0,
                spec=dict(rounds=rounds, proposed_tokens=proposed, accepted_tokens=sum(positions),
                          mean_accepted_per_round=sum(positions)/rounds,
                          accepted_per_position={str(i): p for i, p in enumerate(positions)}))


def test_survivals_are_not_multiplied_again() -> None:
    summary = summarize([run(10, [8, 5, 2, 1, 0], 40), run(20, [16, 10, 4, 2, 0], 80)])
    assert summary['survival_per_position'] == [0.8, 0.5, 0.2, 0.1, 0.0]
    assert summary['acceptance_per_round'] == pytest.approx(1.6)
    assert summary['proposals_per_round'] == 4


def test_run_means_and_pooled_denominators_are_separate() -> None:
    summary = summarize([run(10, [5, 0, 0, 0, 0], 20), run(20, [20, 20, 0, 0, 0], 60)])
    assert summary['mean_run_acceptance'] == 1.25
    assert summary['acceptance_per_round'] == 1.5
    assert summary['accepted_fraction'] == 45/80


def test_censored_fifth_position_is_allowed() -> None:
    summary = summarize([run(10, [8, 5, 2, 1], 40), run(10, [8, 5, 2, 1], 40)])
    assert summary['survival_per_position'][-1] == 0


@pytest.mark.parametrize('bad', ['sum', 'survival', 'empty', 'failed'])
def test_bad_evidence_refused(bad: str) -> None:
    row = run(10, [8, 5, 2, 1, 0], 40)
    if bad == 'sum':
        row['spec']['accepted_tokens'] += 1
    elif bad == 'survival':
        row = run(10, [1, 2, 3, 4, 5], 40)
    elif bad == 'empty':
        row['spec']['proposed_tokens'] = 0
    else:
        row['error'] = 'failed'
    with pytest.raises(ValueError):
        summarize([row, row])
