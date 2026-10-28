"""DEP1/NBW greedy port: actual checker, UP3 CLI, start ordering and STOP."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import cast

import pytest

from offload.launch.windows import nbw_cells as driver, up3_window as w
from offload.launch.windows.tests.test_nbw_window import GreedyApi
from offload.prefill_maxlen.evidence import Json, record
from offload.up3 import cells
from offload.up3.tests.test_up3_window import Remote, rehearsal as rehearsal, starts, greedy_result


@pytest.mark.parametrize('margin', [None, .5, .501, .502, .75])
def test_up3_cli_uses_actual_nbw_near_tie_checker(margin: float | None, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    api = GreedyApi(margin)
    output = tmp_path/'greedy.json'
    monkeypatch.setenv('WINDOW_COORDINATOR', 'Deepseek/coordinator')
    monkeypatch.setattr(cells, 'Http', lambda url: api)
    monkeypatch.setattr(driver, 'greedy_inputs', lambda: (['public']*6,
        [{'i': i, 'text': 'public reference'} for i in range(6)]))
    def no_corpus(root: Path) -> tuple[str, dict[str, Json]]:
        raise AssertionError('greedy mode must use pinned W5 prompts, no replacement corpus')
    monkeypatch.setattr(cells, 'corpus', no_corpus)
    monkeypatch.setattr(sys, 'argv', ['cells', '--execute', '--mode', 'greedy', '--output', str(output),
        '--deadline', '1000000000000', '--data-dir', str(tmp_path)])
    if margin is not None and margin > driver.NEAR_TIE_NATS:
        with pytest.raises(ValueError, match='not a near-tie'):
            cells.main()
        assert api.chats == 4
    else:
        assert cells.main() == 0
        assert api.chats == 6
        w.greedy_checks(record(cast(Json, json.loads(output.read_text()))))
    assert 'public reference' not in output.read_text()


@pytest.mark.parametrize('fault', ['missing-logprobs', 'outside-top20', 'prefix'])
def test_actual_checker_refuses_missing_diagnosis_and_prefix(fault: str, tmp_path: Path) -> None:
    class Broken(GreedyApi):
        def post(self, endpoint: str, body: dict[str, Json]) -> dict[str, Json]:
            if endpoint == '/v1/completions':
                if fault == 'missing-logprobs':
                    return {'choices': [{'logprobs': {}}]}
                if fault == 'outside-top20':
                    return {'choices': [{'logprobs': {'top_logprobs': [{'token_id:12': -.6, 'token_id:7': -3.0}]}}]}
            if endpoint == '/tokenize' and fault == 'prefix' and body.get('prompt') == 'public changed output':
                return {'tokens': [10, 11, 12]}
            return super().post(endpoint, body)
    api = Broken(.5)
    with pytest.raises((ValueError, KeyError)):
        driver.greedy(api, 1e12, ['public']*6, [{'i': i, 'text': 'public reference'} for i in range(6)], tmp_path/'stop.json')
    assert api.chats == 4


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'unordered', 'diverge', 'counts'])
def test_decision_requires_six_ordered_experimental_checks(fault: str) -> None:
    data = starts('k5')
    row = record(data['s2-k5'])
    evidence = record(row['greedy'])
    checks = cast(list[Json], evidence['checks'])
    if fault == 'missing': row.pop('greedy')
    elif fault == 'duplicate': checks[5] = copy.deepcopy(checks[4])
    elif fault == 'unordered': checks[0], checks[1] = checks[1], checks[0]
    elif fault == 'diverge': record(checks[3])['identical'] = False
    else: evidence.update(identical=5, near_tie=1)
    with pytest.raises(ValueError, match='greedy'):
        w.decision('k5', data)


def test_every_start_greedy_precedes_warmup_and_timing(rehearsal: tuple[w.Live, Remote]) -> None:
    live, fake = rehearsal
    result = w.window(live, 'k5', lambda row: None)
    assert result['status'] == 'COMPLETE', result
    for arm in w.arms('k5'):
        row = record(record(result['starts'])[arm.name])
        assert row['greedy'] == greedy_result()
        assert fake.commands.index(arm.name+'-start') < fake.commands.index(arm.name+'-greedy')
        assert fake.commands.index(arm.name+'-greedy') < fake.commands.index(arm.name+'-warmup')
        assert fake.commands.index(arm.name+'-warmup') < fake.commands.index(arm.name+'-short-b1')


def test_candidate_divergence_stops_before_timing_and_restores(rehearsal: tuple[w.Live, Remote], monkeypatch: pytest.MonkeyPatch) -> None:
    live, fake = rehearsal
    original = live.greedy
    def diverge(arm: w.Arm) -> dict[str, Json]:
        if arm.role == 'k5':
            fake.fail = 'experiment-greedy'
        return original(arm)
    monkeypatch.setattr(live, 'greedy', diverge)
    result = w.window(live, 'k5', lambda row: None)
    assert 'greedy divergence' in str(result['stop'])
    assert 's1-control-short-b1' in fake.commands
    assert 's2-k5-short-b1' not in fake.commands
    assert str(record(result['restore'])['status']).startswith('RESTORED')
    assert record(result['decision'])['status'] == 'NOT A CANDIDATE'


def test_greedy_refuses_exhausted_premeasurement_budget(rehearsal: tuple[w.Live, Remote]) -> None:
    live, fake = rehearsal
    live.started_at = fake.clock.now - w.PREPARE_S
    with pytest.raises(ValueError, match='greedy budget exhausted'):
        live.greedy(w.arms('k5')[0])
    assert not fake.commands
