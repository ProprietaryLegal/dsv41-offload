"""CPU checks of pooled counters, uncertainty and actual four-wave driver sequencing."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from offload.gb10_budget import window_cells as wc
from offload.launch.windows import pp4_cells as c
from offload.launch.windows import pp4_metrics as m
from offload.launch.windows import pp4_window as w
from offload.launch.windows.tests.test_pp4_window import complete
from offload.prefill_maxlen.evidence import Json, number, record


def test_pool_uses_round_weighted_totals_and_milliseconds() -> None:
    rows: list[dict[str, Json]] = [
        {'spec_request_rounds': 10., 'accepted_tokens': 0., 'server_decode_seconds': .1},
        {'spec_request_rounds': 90., 'accepted_tokens': 180., 'server_decode_seconds': .9}]
    result = m.pool(rows)
    assert result['acceptance_per_round'] == 1.8
    assert result['server_decode_ms_per_spec_request_round'] == 10.
    assert result['expected_tps'] == 280.


@pytest.mark.parametrize('key,value', [('spec_request_rounds', 0.), ('spec_request_rounds', 1.2),
    ('accepted_tokens', -1.), ('accepted_tokens', 501.), ('server_decode_seconds', float('nan')),
    ('server_decode_seconds', float('inf')), ('server_decode_seconds', 0.)])
def test_invalid_counter_evidence_refuses(key: str, value: float) -> None:
    row: dict[str, Json] = {'spec_request_rounds': 100., 'accepted_tokens': 100., 'server_decode_seconds': 2.}
    row[key] = value
    with pytest.raises(ValueError):
        m.pool([row])


def test_client_rate_cannot_determine_round3_decision() -> None:
    rows = complete()
    for arm in w.ARMS:
        for cell in record(record(rows[arm.name])['cells']).values():
            record(cell)['median_decode_tps'] = 1. if arm.role == 'B' else 10000.
    decision = w.decision(rows)
    assert decision['status'] == 'CANDIDATE FOR REVIEW'
    cell = record(record(decision['cells'])['short-b1'])
    assert record(cast(list[Json], cell['starts'])[0])['expected_tps'] == 100.
    assert record(cast(list[Json], cell['starts'])[1])['expected_tps'] == 110.
    assert record(record(cast(list[Json], cell['starts'])[1])['pooled_95pct'])['acceptance_per_round'] == [1.2, 1.2]


@pytest.mark.parametrize('change', ['three-waves', 'duplicate-prompt', 'source-order', 'old-schema'])
def test_independent_request_and_registration_gates(change: str) -> None:
    rows = complete()
    cell = record(record(record(rows[w.ARMS[1].name])['cells'])['short-b1'])
    waves = cast(list[Json], cell['waves'])
    if change == 'three-waves': waves.pop()
    if change == 'duplicate-prompt': record(waves[2])['streams'] = copy.deepcopy(record(waves[0])['streams'])
    if change == 'source-order': record(waves[0])['family'] = 'filings'
    if change == 'old-schema': cell['metric_registration'] = 'old'
    with pytest.raises(ValueError):
        w.decision(rows)


def test_interval_preserves_request_wave_and_source_variation() -> None:
    rows = complete()
    for arm in (w.ARMS[1],):
        for cell in record(record(rows[arm.name])['cells']).values():
            for wave in cast(list[Json], record(cell)['waves']):
                record(wave)['accepted_tokens'] = 500. if record(wave)['family'] == 'opinions' else 2000.
    result = w.decision(rows)
    assert result['status'] == 'INCONCLUSIVE'
    assert number(record(record(result['cells'])['short-b1'])['point_gain']) > .05
    assert number(cast(list[Json], record(record(result['cells'])['short-b1'])['paired_delta_95pct_tps'])[0]) < 0
    for cell in record(result['cells']).values():
        bounds = cast(list[Json], record(record(cast(list[Json], record(cell)['starts'])[1])['pooled_95pct'])['acceptance_per_round'])
        assert number(bounds[0]) < 1.25 < number(bounds[1])


@pytest.mark.parametrize('name', list(c.CELLS))
def test_four_actual_driver_waves_and_incremental_receipts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    state = {'rounds': 100., 'accepted': 100., 'seconds': 2., 'completed': 0.}
    calls: list[tuple[str, tuple[str, ...]]] = []
    class Api:
        def metrics(self) -> str:
            return json.dumps({'vllm:spec_decode_num_drafts_total': state['rounds'],
                'vllm:spec_decode_num_accepted_tokens_total': state['accepted'],
                'vllm:request_decode_time_seconds_sum': state['seconds'], 'vllm:request_decode_time_seconds_count': state['completed']})
    def run(api: wc.Api, cell: wc.Cell, texts: dict[str, tuple[str, int]], plan: wc.Plan) -> dict[str, Json]:
        calls.append((cell.name, cell.prompts))
        assert isinstance(api, c.SeededApi) and api.seed == 42+len(calls)-1
        assert cell.max_tokens == 512
        state['completed'] += cell.batch
        state['rounds'] += 100.; state['accepted'] += 120.; state['seconds'] += 2.
        return {'cell': cell.name, 'batch': cell.batch, 'streams': [
            {'usage': {'completion_tokens': 512}, 'finish': 'length', 'first_by_kind_s': {'reasoning': 1.},
             'last_by_kind_s': {'reasoning': 4.}, 'ttft_s': 1., 'prompt_sha256': texts[key][0], 'cache_salt_sha256': plan.salt_prefix+key,
             'sampling': {'temperature': 1, 'top_p': 1, 'seed': 42, 'priority': 10, 'max_tokens': 512}} for key in cell.prompts]}
    monkeypatch.setattr(c, 'load_bench', lambda: SimpleNamespace(parse_metrics=json.loads, spec_delta=lambda *a: {}))
    monkeypatch.setattr(c, 'family_corpus', lambda root, family: (family, {}))
    monkeypatch.setattr(c, 'texts_from', lambda cuts, family, hashes: {key: (family+'-'+key, 32000) for key in c.PROMPTS})
    monkeypatch.setattr(wc, 'run_cell', run)
    monkeypatch.setattr(time, 'sleep', lambda seconds: None)
    output = tmp_path/'partial.json'
    result = c.repeated_cell(cast(wc.Api, Api()), name, {f: {} for f in c.FAMILIES}, tmp_path,
                             wc.Plan(9999999999., 0, [], 'fixed', None), output)
    assert len(calls) == m.group_count(name) and len({key for key, _ in calls}) == m.group_count(name)
    assert [keys for _, keys in calls] == [c.wave_prompts(name, i) for i in range(m.group_count(name))]
    assert result['independent_requests'] == m.group_count(name)*c.CELLS[name].batch
    assert result['acceptance_per_round'] == pytest.approx(1.2)
    assert record(result['pooled_95pct'])['acceptance_per_round'] == [1.2, 1.2]
    assert record(cast(Json, json.loads(output.read_text())))['status'] == 'PARTIAL'
    assert len(m.waves(result, c.CELLS[name].batch)) == m.group_count(name)


@pytest.mark.parametrize('kind', ['ttft', 'server-cost'])
def test_family_regression_cannot_hide_in_pooled_median(kind: str) -> None:
    rows = complete()
    for arm in (w.ARMS[1],):
        for cell in record(record(rows[arm.name])['cells']).values():
            data = record(cell)
            if kind == 'ttft':
                for wave in cast(list[Json], data['waves']):
                    if record(wave)['family'] == 'opinions':
                        record(wave)['median_ttft_s'] = 1.051
            else:
                for wave in cast(list[Json], data['waves']):
                    if record(wave)['family'] == 'opinions':
                        record(wave)['server_decode_seconds'] = 21.01
    assert w.decision(rows)['status'] == 'INCONCLUSIVE'


def test_retired_calibration_gate_cannot_admit_measurement() -> None:
    from offload.launch.windows import pp4_prerequisites as p
    assert 'measured_complete_block_budget' not in p.GATES
    assert 'measured_complete_block_budget' in p.RETIRED_GATES
