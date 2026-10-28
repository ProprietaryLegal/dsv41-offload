"""Pooled arithmetic, paired uncertainty, and actual repeated CPU cell driver."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import cast

import pytest

from offload.launch.windows import up3_window as w, up2_checks as ck
from offload.prefill_maxlen.evidence import Json, record
from offload.up3 import cells, metrics as m
from offload.up3.tests.test_up3_cells import Api
from offload.up3.tests.test_up3_window import measured, starts


def test_pool_counter_totals_instead_of_mean_of_ratios() -> None:
    rows = [m.Group(10, 10, 100), m.Group(100, 200, 2000)]
    accepted, ms, rate = m.pooled(rows)
    assert accepted == pytest.approx(210/110)
    assert ms == pytest.approx(2100/110)
    assert rate == pytest.approx(320/2100)
    assert rate != pytest.approx((.2+.15)/2)


def test_client_rate_cannot_hide_a_pooled_loss() -> None:
    data = starts('k5')
    for value in record(record(data['s2-k5'])['cells']).values():
        for repeat in ck.rows(record(value)['repetitions']):
            # Faster rounds, but far fewer accepted tokens: committed rate loses.
            repeat.update({'accepted_tokens': 100000, 'server_decode_ms': 300000000})
            for stream in ck.rows(repeat['streams']):
                stream['decode_tps'] = 999
    assert w.decision('k5', data)['status'] == 'NOT A CANDIDATE'


def test_noisy_point_win_fails_paired_lower_bound() -> None:
    records = [measured('short-b1', gain, salt=str(i)) for i, gain in enumerate((0., .4, 0.))]
    for i, group in enumerate(ck.rows(records[1]['repetitions'])):
        group['accepted_tokens'] = 1800000 if i < 2 else 5000000
    result = m.compare(records, 1, 4)
    assert ck.rows(result['start_estimates'])[1]['committed_per_ms'] == pytest.approx(.011)
    assert result['win'] is False
    assert result == m.compare(records, 1, 4)


@pytest.mark.parametrize('fault', ['short', 'seed', 'salt', 'nan', 'fractional', 'completed', 'budget', 'cached', 'prompt'])
def test_invalid_evidence_refuses(fault: str) -> None:
    row = measured('short-b1')
    groups = ck.rows(row['repetitions'])
    stream = ck.rows(groups[0]['streams'])[0]
    if fault == 'short': row['repetitions'] = cast(Json, groups[:3])
    elif fault == 'seed': record(stream['sampling'])['seed'] = 41
    elif fault == 'salt': ck.rows(groups[1]['streams'])[0]['cache_salt_sha256'] = stream['cache_salt_sha256']
    elif fault == 'nan': groups[0]['server_decode_ms'] = float('nan')
    elif fault == 'fractional': groups[0]['spec_request_rounds'] = .5
    elif fault == 'completed': groups[0]['completed_requests'] = 2
    elif fault == 'budget': stream['budget'] = 4096
    elif fault == 'cached': record(record(stream['usage'])['prompt_tokens_details'])['cached_tokens'] = 1
    else: record(stream['usage'])['prompt_tokens'] = 1
    with pytest.raises(ValueError):
        m.groups(row, 1, 4)


@pytest.mark.parametrize('name', cells.CELLS)
def test_repeated_driver_measures_at_least_four_complete_requests(tmp_path: Path, name: str) -> None:
    api = Api()
    output = tmp_path/'cell.json'
    prompts = {key: ('long' if key == '150k' else 'secret public fixture', 150000 if key == '150k' else 32000)
               for key in cells.KEYS}
    result = cells.repeated(api, name, prompts, deadline=time.time()+10000, salt='tag', ledger=tmp_path/'ledger',
                            output=output, sleep=lambda _: None)
    assert len(api.requests) == cells.batch_for(name)*cells.REPEATS[name] >= 4
    assert len({str(r['cache_salt']) for r in api.requests}) == len(api.requests)
    assert all(r['max_tokens'] == 512 and r['temperature'] == r['top_p'] == 1 for r in api.requests)
    assert {r['seed'] for r in api.requests} == set(range(42, 42+cells.REPEATS[name]))
    groups = m.groups(result, cells.batch_for(name), cells.REPEATS[name])
    assert m.pooled(groups) == (3., 40., .1)
    assert 'secret public fixture' not in output.read_text()
    assert json.loads(output.read_text()) == result


def test_partial_result_and_mismatched_prompt_never_qualify() -> None:
    records = [measured('short-b1', gain, salt=str(i)) for i, gain in enumerate((0., .4, 0.))]
    row = copy.deepcopy(records[1])
    ck.rows(ck.rows(row['repetitions'])[0]['streams'])[0]['prompt_sha256'] = 'changed'
    with pytest.raises(ValueError, match='prompt mismatch'):
        m.compare([records[0], row, records[2]], 1, 4)
    assert m.compare(records, 1, 4)['win'] is True
    assert m.compare([measured('short-b1', gain) for gain in (0., .2, 0.)], 1, 4)['win'] is False
