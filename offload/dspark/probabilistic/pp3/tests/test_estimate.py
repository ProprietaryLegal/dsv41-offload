from __future__ import annotations

import json
from pathlib import Path

import pytest

from offload.dspark.probabilistic.pp3.estimate import estimate, rate, wire_bytes


def test_cost_break_even_with_bonus_token() -> None:
    # Three committed tokens/round +10% tokens, +10% round time: same60tok/s.
    assert rate(3.,50.,.3,2.,3.) == pytest.approx(60.)
    with pytest.raises(ValueError):
        rate(3.,50.,.3,2.,-1.)
    assert wire_bytes(1,5) == 1416
    assert wire_bytes(4,5) == 5412
    with pytest.raises(ValueError):
        wire_bytes(5,5)


def test_no_missing_cost_becomes_zero_or_live_prediction() -> None:
    root=Path(__file__).parents[1]
    producer=json.loads((root/'producer-final.json').read_text())
    gpu=json.loads((root/'gpu-qualification.json').read_text())
    frozen=json.loads((root/'pp2-frozen.json').read_text())
    result=estimate(producer,gpu,frozen)
    assert result['additional_unmeasured_ms'] is None
    assert not result['selector_absolute_005ms_goal_met']
    for row in result['historical_adaptive_sensitivities']:
        assert row['expected_live_tps'] is None
        assert row['sensitivity'][0]['conditional_tps'] > row['sensitivity'][-1]['conditional_tps']
    assert result['historical_adaptive_sensitivities'][2]['measured_acceptance_at150k'] is None
    producer['complete']=False
    with pytest.raises(ValueError,match='incomplete'):
        estimate(producer,gpu,frozen)
