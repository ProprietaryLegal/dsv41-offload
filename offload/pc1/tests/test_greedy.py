"""DEP1/NBW greedy port: actual checker, PC1 CLI, start ordering and STOP."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import cast

import pytest

from offload.launch.windows import nbw_cells as driver
from offload.pc1 import window as w
from offload.launch.windows.tests.test_nbw_window import GreedyApi
from offload.prefill_maxlen.evidence import Json, record
from offload.pc1 import cells
from offload.pc1.tests.test_window import Remote, rehearsal as rehearsal, starts, greedy_result


@pytest.mark.parametrize('margin', [None, .5, .501, .502, .75, .9, .901, .902, 1.])
def test_pc1_cli_uses_actual_nbw_near_tie_checker(margin: float | None, tmp_path: Path,
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
    if margin is not None and margin > cells.GREEDY_NEAR_TIE_NATS:
        with pytest.raises(ValueError, match='not a near-tie'):
            cells.main()
        assert api.chats == 4
    else:
        assert cells.main() == 0
        assert api.chats == 6
        result = record(cast(Json, json.loads(output.read_text())))
        assert result['policy'] == 'pc1-r3-near-tie-0p9'
        assert result['near_tie_nats'] == .9 + 1e-3
        w.greedy_checks(result)
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
        driver.greedy(api, 1e12, ['public']*6, [{'i': i, 'text': 'public reference'} for i in range(6)], tmp_path/'stop.json',
                      near_tie_nats=cells.GREEDY_NEAR_TIE_NATS)
    assert api.chats == 4


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'unordered', 'diverge', 'counts',
                                 'policy-missing', 'policy-other', 'threshold-missing', 'threshold-other'])
def test_decision_requires_six_ordered_experimental_checks(fault: str) -> None:
    data = starts('chunk2048')
    row = record(data['s2-chunk2048'])
    evidence = record(row['greedy'])
    checks = cast(list[Json], evidence['checks'])
    if fault == 'missing': row.pop('greedy')
    elif fault == 'duplicate': checks[5] = copy.deepcopy(checks[4])
    elif fault == 'unordered': checks[0], checks[1] = checks[1], checks[0]
    elif fault == 'diverge': record(checks[3])['identical'] = False
    elif fault == 'policy-missing': evidence.pop('policy')
    elif fault == 'policy-other': evidence['policy'] = 'unregistered'
    elif fault == 'threshold-missing': evidence.pop('near_tie_nats')
    elif fault == 'threshold-other': evidence['near_tie_nats'] = .501
    else: evidence.update(identical=5, near_tie=1)
    with pytest.raises(ValueError, match='greedy'):
        w.decision('chunk2048', data)


def test_every_start_greedy_precedes_warmup_and_timing(rehearsal: tuple[w.Live, Remote]) -> None:
    live, fake = rehearsal
    result = w.window(live, 'chunk2048', lambda row: None)
    assert result['status'] == 'COMPLETE', result
    for arm in w.arms('chunk2048'):
        row = record(record(result['starts'])[arm.name])
        assert row['greedy'] == greedy_result()
        assert fake.commands.index(arm.name+'-start') < fake.commands.index(arm.name+'-greedy')
        assert fake.commands.index(arm.name+'-greedy') < fake.commands.index(arm.name+'-warmup')
        assert fake.commands.index(arm.name+'-warmup') < fake.commands.index(arm.name+'-short-b1-r0')


def test_candidate_divergence_stops_before_timing_and_restores(rehearsal: tuple[w.Live, Remote], monkeypatch: pytest.MonkeyPatch) -> None:
    live, fake = rehearsal
    original = live.greedy
    def diverge(arm: w.Arm) -> dict[str, Json]:
        if arm.role == 'chunk2048':
            fake.fail = 'experiment-greedy'
        return original(arm)
    monkeypatch.setattr(live, 'greedy', diverge)
    result = w.window(live, 'chunk2048', lambda row: None)
    assert 'greedy divergence' in str(result['stop'])
    assert 's1-control-short-b1-r0' in fake.commands
    assert not any(c.startswith('s2-chunk2048-short-b1') for c in fake.commands)
    assert str(record(result['restore'])['status']).startswith('RESTORED')
    assert record(result['decision'])['status'] == 'NOT A CANDIDATE'


def test_greedy_refuses_exhausted_premeasurement_budget(rehearsal: tuple[w.Live, Remote]) -> None:
    live, fake = rehearsal
    live.started_at = fake.clock.now - w.PREPARE_S
    with pytest.raises(ValueError, match='greedy budget exhausted'):
        live.greedy(w.arms('chunk2048')[0])
    assert not fake.commands


def test_shared_nbw_default_stays_stricter_than_pc1(tmp_path: Path) -> None:
    assert driver.NEAR_TIE_NATS == .5 + 1e-3
    with pytest.raises(ValueError, match='not a near-tie'):
        driver.greedy(GreedyApi(.75), 1e12, ['public'] * 6,
            [{'i': i, 'text': 'public reference'} for i in range(6)], tmp_path / 'nbw.json')


def test_pc1_threshold_keeps_shared_top_choice_gap_rule(tmp_path: Path) -> None:
    class Gap(GreedyApi):
        def post(self, endpoint: str, body: dict[str, Json]) -> dict[str, Json]:
            if endpoint == '/v1/completions':
                return {'choices': [{'logprobs': {'top_logprobs': [
                    {'token_id:12': -1.0, 'token_id:99': -1.5, 'token_id:14': 0.0}]}}]}
            return super().post(endpoint, body)
    with pytest.raises(ValueError, match='not a near-tie'):
        driver.greedy(Gap(.5), 1e12, ['public'] * 6,
            [{'i': i, 'text': 'public reference'} for i in range(6)], tmp_path / 'gap.json',
            near_tie_nats=cells.GREEDY_NEAR_TIE_NATS)
