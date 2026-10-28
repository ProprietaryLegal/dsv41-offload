from __future__ import annotations

import copy
import math
from typing import cast

import pytest

from offload.tv1 import metrics as m
from offload.prefill_maxlen.evidence import Json, record


def trial(batch: int = 1, rounds: float = 100, seconds: float = 4, accepted: float = 100, offset: float = 100) -> dict[str, Json]:
    before = {name+'{model="public"}': offset for name in m.NAMES.values()}
    after = dict(before)
    values = {'rounds': rounds, 'decode_seconds': seconds, 'decode_requests': batch,
              'completed_requests': batch, 'accepted_tokens': accepted, 'proposed_tokens': 5*rounds}
    for key, value in values.items():
        after[m.NAMES[key]+'{model="public"}'] += value
    return {'batch': batch, 'streams': [{} for _ in range(batch)], 'server_counters': m.counters(before, after, batch)}


def test_pool_sums_numerators_and_denominators_instead_of_averaging_ratios() -> None:
    trials = [trial(rounds=10,seconds=1,accepted=2), trial(rounds=90,seconds=3,accepted=180,offset=1000),
              trial(rounds=10,seconds=1,accepted=2,offset=2000), trial(rounds=90,seconds=3,accepted=180,offset=3000)]
    result = m.pool(trials)
    assert result['requests'] == 4
    assert result['server_decode_ms_per_spec_request_round'] == 40
    assert result['mean_accepted_per_round'] == 1.82
    assert result['acceptance_fraction'] == 364/1000
    assert m.pool([trial(batch=4)])['requests'] == 4
    with pytest.raises(ValueError, match='>= 4'):
        m.pool([trial()])


@pytest.mark.parametrize('defect', ['missing','reset','nan','infinity','foreign','unfinished','accepted','forged'])
def test_raw_counter_failures_never_become_a_decision(defect: str) -> None:
    trials = [trial(batch=4)]
    counters = record(trials[0]['server_counters'])
    after = record(counters['metrics_after'])
    key = m.NAMES['decode_seconds']+'{model="public"}'
    if defect == 'missing':
        after.pop(key)
    elif defect == 'reset':
        after[key] = 0
    elif defect == 'nan':
        after[key] = math.nan
    elif defect == 'infinity':
        after[key] = math.inf
    elif defect == 'foreign':
        after[m.NAMES['completed_requests']+'{model="public"}'] = 105
    elif defect == 'unfinished':
        after[m.NAMES['decode_requests']+'{model="public"}'] = 103
    elif defect == 'accepted':
        after[m.NAMES['accepted_tokens']+'{model="public"}'] = 10000
    else:
        counters['rounds'] = 200
    with pytest.raises(ValueError):
        m.pool(trials)


def test_new_lazy_label_is_counted_but_disappearing_label_refuses() -> None:
    original = record(trial(batch=4)['server_counters'])
    before = {k: float(cast(float, v)) for k,v in record(original['metrics_before']).items()}
    after = {k: float(cast(float, v)) for k,v in record(original['metrics_after']).items()}
    after[m.NAMES['rounds']+'{model="lazy"}'] = 7
    assert m.delta(before, after, m.NAMES['rounds']) == 107
    broken = copy.copy(after)
    broken.pop(next(k for k in before if k.startswith(m.NAMES['rounds']+'{')))
    with pytest.raises(ValueError, match='counter'):
        m.delta(before, broken, m.NAMES['rounds'])


def test_replayed_counter_interval_cannot_count_same_request_four_times() -> None:
    one = trial()
    with pytest.raises(ValueError, match='counted twice'):
        m.pool([one]*4)
