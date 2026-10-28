"""Regression checks for the screen's main inference hazards, using numeric fixtures only."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from offload.dspark.depth.screen import DRAFT, VERIFY, expected, rate, read_suite


def test_prefix_survivals_are_summed_not_multiplied() -> None:
    assert expected([.8,.5,.3,.2,.1],5) == pytest.approx(2.9)
    assert expected([.8,.5,.3,.2,.1],2) == pytest.approx(2.3)
    with pytest.raises(ValueError,match='maximum'):
        expected([.8,.5,.3,.2,.1],6)
    with pytest.raises(ValueError,match='increase'):
        expected([.8,.9,.3,.2,.1],5)


def test_overlap_not_double_counted_and_batch_not_squared() -> None:
    # At the nominal cap, central exposure exactly cancels: T is already end-to-end.
    s = [.8,.5,.3,.2,.1]
    for b in range(1,5):
        k = 5 if b <= 2 else 4
        assert rate(s,k,b,80,.5,.75,.3) == pytest.approx(1000*b*expected(s,k)/80)
        assert rate(s,k,b,80,.5,1,.3) > rate(s,k,b,80,.5,0,.3)
    assert len(VERIFY) == 6 and len(DRAFT) == 4


def test_reader_counts_request_rounds_and_refuses_missing_tail(tmp_path: Path) -> None:
    run: dict[str,object] = dict(type='run',run_id='synthetic',error=None,target_ctx=32000,concurrency=2,mixed=False,
        settings=dict(temperature=1,top_p=1),
        spec=dict(rounds=10,accepted_tokens=19,accepted_per_position=dict(zip(map(str,range(5)),[8,5,3,2,1]))))
    req: dict[str,object] = dict(type='request',run_id='synthetic',error=None,step_gap_ms=dict(p50=80))
    p = tmp_path/'fixture.jsonl'
    p.write_text('\n'.join(json.dumps(r) for r in [req,req,run]))
    result = read_suite([p],'synthetic')
    assert result[0]['prefix_survival_MEASURED'] == [.8,.5,.3,.2,.1]
    assert result[0]['request_rounds'] == 10  # not five engine steps, not twenty request-rounds
    assert 'run_id' not in json.dumps(result)  # no request identifiers exported
    spec = run['spec']
    assert isinstance(spec,dict)
    counts = spec['accepted_per_position']
    assert isinstance(counts,dict)
    del counts['4']
    p.write_text('\n'.join(json.dumps(r) for r in [req,req,run]))
    with pytest.raises(KeyError):
        read_suite([p],'synthetic')


def test_flat_target_cost_never_prefers_losing_accepted_tokens() -> None:
    s = [.8,.5,.3,.2,.1]
    assert all(rate(s,k+1,1,40,0,1,0) > rate(s,k,1,40,0,1,0) for k in range(1,5))
