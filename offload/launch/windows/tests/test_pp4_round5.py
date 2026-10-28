"""Round5 whole-HOLD admission, registered historical budget and forced output checks."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from offload.launch.windows import pf1_window as pf
from offload.launch.windows import pp4_budget as b
from offload.launch.windows import pp4_cells as c
from offload.launch.windows import pp4_metrics as m
from offload.launch.windows import pp4_prerequisites as prerequisites
from offload.launch.windows import pp4_window as w
from offload.launch.windows.tests import test_pf1_window as base
from offload.launch.windows.tests.test_pp4_window import WORKER
from offload.prefill_maxlen.evidence import Json, number, record


def test_budget_recomputes_gap_scaling_worker_cuts_and_explicit_profile_retirement() -> None:
    result = b.estimate(pair={'seconds': 600})
    assert result['total_s'] == 4743 and result['fits'] is True
    assert result['headline_per_start_s'] == 220
    assert result['cells'] == ['short-b1', '32k-b1'] and result['all_cells_total_s'] == 5373
    assert result['profile_per_start_s'] == 0
    assert result['block_budget_s'] == {'s1-A': 1167., 's2-B': 1089., 's3-A': 1047.}
    assert 'measured_complete_block_budget' in prerequisites.RETIRED_GATES
    assert 'measured_complete_block_budget' not in prerequisites.GATES


def test_all_nine_measured_cold_cell_source_hashes_and_numeric_values() -> None:
    plan = record(json.loads(b.PATH.read_text()))
    paths: set[str] = set()
    for value in record(plan['cells']).values():
        row = record(value)
        for observation in record_list(row['observations']):
            path = pf.CAMPAIGN/str(observation['source'])
            assert pf.digest(path) == observation['sha256']
            paths.add(str(path))
            streams = record_list(record(json.loads(path.read_text()))['streams'])
            assert all(record(s['usage'])['completion_tokens'] == 1500 for s in streams)
            assert observation['max_ttft_s'] == max(number(s['ttft_s']) for s in streams)
            spans = [max(number(x) for x in record(s['last_by_kind_s']).values())
                -min(number(x) for x in record(s['first_by_kind_s']).values()) for s in streams]
            assert observation['max_decode_span_s'] == max(spans)
    assert len(paths) == 9


def record_list(value: Json) -> list[dict[str, Json]]:
    assert isinstance(value, list)
    return [record(v) for v in value]


def test_missing_admission_cannot_be_replaced_by_old_pair_pointer(tmp_path: Path) -> None:
    path = tmp_path/'missing-admission.json'
    with pytest.raises(FileNotFoundError): b.pair_reserve(path, WORKER.commit)
    path.write_text('{"schema":"pp4-physical-pair-v1","complete":true}')
    with pytest.raises(ValueError): b.pair_reserve(path, WORKER.commit)


def test_late_start_needs_all_remaining_blocks_and_cannot_skip_closing_off(tmp_path: Path) -> None:
    clock = base.Clock()
    live = w.Live(tmp_path, pf.Supervisor(tmp_path, clock=clock), native=WORKER, remote='experiment',
        expected=WORKER.commit, data_dir=tmp_path, tag='owned', admission={
            'verifier_sha256': {}, 'block_budget_s': {'s1-A': 100., 's2-B': 110., 's3-A': 100.}}, clock=clock)
    try:
        live.submit = clock()+310
        assert live.fits(w.ARMS[0])
        clock.now += .001
        assert not live.fits(w.ARMS[0])
        live.submit = clock()+210
        assert live.fits(w.ARMS[1])
        clock.now += .001
        assert not live.fits(w.ARMS[1])
        live.submit = clock()+100
        assert live.fits(w.ARMS[2])
        clock.now += .001
        assert not live.fits(w.ARMS[2])
        assert w.SUBMIT_S == w.ABORT_S == 73*60 and w.RESTORE_S == 85*60
    finally:
        live.sup.close()


def test_seeded_http_preserves_settings_and_forces_eos() -> None:
    seen: list[dict[str, Json]] = []
    class Api(c.Http):
        def stream(self, body: dict[str, Json], first: threading.Event) -> dict[str, Json]:
            seen.append(body)
            return {}
    c.SeededApi(Api('http://unused'), 45).stream({'seed': 42, 'max_tokens': 512, 'temperature': 1, 'top_p': 1,
        'cache_salt': 'owned-start-wave:cell:0'}, threading.Event())
    assert seen == [{'seed': 45, 'max_tokens': 512, 'temperature': 1, 'top_p': 1, 'ignore_eos': True,
                     'cache_salt': 'owned-start-wave:cell:0'}]


@pytest.mark.parametrize('tokens,finish', [(511, 'length'), (1500, 'length'), (512, 'stop')])
def test_short_eos_or_legacy_output_never_enters512_summary(tokens: int, finish: str) -> None:
    with pytest.raises(ValueError, match='budget incomplete'):
        c.summarize({'batch': 1, 'streams': [{'usage': {'completion_tokens': tokens}, 'finish': finish}]}, {}, {})


def test_split_window_input_is_retired_not_reinterpreted() -> None:
    assert [a.role for a in w.ARMS] == ['A', 'B', 'A']
    with pytest.raises(ValueError, match='one complete'):
        w.validate_prior({}, 'a'*40)
    assert m.REGISTRATION == 'pp4-r6-512-aba-no-measurement-profiles-v1'
    assert [m.group_count(name) for name in c.CELLS] == [4, 4, 2]


def constant_groups(rate: int, count: int = 4) -> list[dict[str, Json]]:
    return [{'spec_request_rounds': 100., 'accepted_tokens': float(rate*2-100), 'server_decode_seconds': 2.}
            for _ in range(count)]


def test_strict_win_boundary_and_inclusive_loss_boundary() -> None:
    equal = m.compare([constant_groups(100), constant_groups(102), constant_groups(100)])
    assert equal['required_delta_tps'] == equal['point_delta_tps'] == 2.
    assert equal['win'] is False
    assert m.compare([constant_groups(100), constant_groups(103), constant_groups(100)])['win'] is True
    loss = m.compare([constant_groups(100), constant_groups(95), constant_groups(110)])
    assert loss['point_delta_tps'] == -10. and loss['off_spread_tps'] == 10.
    assert loss['no_loss_beyond_spread'] is True
    assert m.compare([constant_groups(100), constant_groups(94), constant_groups(110)])['no_loss_beyond_spread'] is False


def test_off_spread_overrides_two_percent_and_every_group_stays_intact() -> None:
    point = m.compare([constant_groups(100, 2), constant_groups(115, 2), constant_groups(110, 2)])
    assert point['required_delta_tps'] == 10. and point['win'] is False
    assert point['bootstrap_seed'] == 4103 and point['resamples'] == 2000
    assert point['independent_groups'] == 2
    assert point['paired_delta_95pct_tps'] == [10., 10.]


def test_b4_cannot_win_and_any_cell_loss_stops() -> None:
    from offload.launch.windows.tests.test_pp4_window import complete
    starts = complete()
    on = record(record(starts[w.ARMS[1].name])['cells'])
    for name in ('short-b1', '32k-b1'):
        for row in record_list(record(on[name])['waves']): row['accepted_tokens'] = 1000.
    assert w.decision(starts)['status'] == 'INCONCLUSIVE'
    starts = complete()
    for row in record_list(record(record(record(starts[w.ARMS[1].name])['cells'])['32k-b4'])['waves']):
        row['accepted_tokens'] = 999.
    assert w.decision(starts)['status'] == 'INCONCLUSIVE'


@pytest.mark.parametrize('change', ['512', 'eos', 'seed', 'salt', 'complete', 'greedy'])
def test_each_evidence_contract_refuses_changed_receipt(change: str) -> None:
    from offload.launch.windows.tests.test_pp4_window import complete
    starts = complete()
    block = record(starts[w.ARMS[1].name])
    group = record_list(record(record(block['cells'])['short-b1'])['waves'])[0]
    stream = record_list(group['streams'])[0]
    if change == '512': record(stream['usage'])['completion_tokens'] = 511
    if change == 'eos': record(stream['sampling'])['ignore_eos'] = False
    if change == 'seed': record(stream['sampling'])['seed'] = 43
    if change == 'salt':
        off = record_list(record(record(record(starts[w.ARMS[0].name])['cells'])['short-b1'])['waves'])[0]
        stream['cache_salt_sha256'] = record_list(off['streams'])[0]['cache_salt_sha256']
    if change == 'complete': group['completed_requests'] = 0
    if change == 'greedy': record_list(record(block['greedy'])['checks'])[5]['identical'] = False
    with pytest.raises(ValueError): w.decision(starts)


def test_nearest_rank95_interval_is_preregistered() -> None:
    assert m.interval(list(range(2000))) == [49, 1949]


def test_actual_execute_refuses_budget_before_output_or_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    import subprocess
    monkeypatch.setenv('WINDOW_COORDINATOR', 'Deepseek/coordinator')
    monkeypatch.setattr(b, 'pair_reserve', lambda path, expected: {'seconds': 960})
    proof = tmp_path/'primary.json'; proof.write_text('{}')
    receipt: dict[str, Json] = {'schema': 'pp4-live-admission-v1', 'source_commit': WORKER.commit,
        'verdict': 'QUALIFIED', 'depth': 5, 'worker_identity': WORKER.identity, 'markov_identity': WORKER.markov,
        'runtime_python': prerequisites.PYTHON, 'runtime_torch': '2.12.1+cu130',
        'runtime_artifact_sha256': {prerequisites.STOCK: 'd'*64, prerequisites.FACC: 'e'*64,
            'p2p/'+pf.digest(pf.ROOT/'offload/dspark/worker/p2p/p2p.cu')[:16]+'/dspark_p2p.so': 'f'*64},
        'gates': {name: {'pass': True, 'evidence_path': str(proof.resolve()), 'sha256': pf.digest(proof)}
                  for name in prerequisites.GATES}}
    path = tmp_path/'qualification.json'; path.write_text(json.dumps(receipt))
    def forbidden(*args: object, **kwargs: object) -> None: raise AssertionError('process/remote action before budget refusal')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    out = tmp_path/'PP4-never-created'
    assert w.main(['--execute', '--qualification', str(path), '--expected-experiment-commit', WORKER.commit,
                   '--out', str(out)]) == 1
    assert not out.exists()
    assert 'ESTIMATED 85.05min exceeds85min after B4 omission' in capsys.readouterr().err
