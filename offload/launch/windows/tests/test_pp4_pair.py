"""CPU pair-only/one-HOLD qualification rehearsals; no CUDA/host calls."""
from __future__ import annotations

from pathlib import Path

import pytest

from offload.dspark.probabilistic.pp4 import pair_contract as contract
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import pp4_window as w
from offload.launch.windows import up2_checks as ck
from offload.launch.windows.tests.test_pp4_window import WORKER, Fake
from offload.prefill_maxlen.evidence import Json, record


def receipt() -> dict[str, Json]:
    return {'schema': contract.SCHEMA, 'source_commit': WORKER.commit, 'run': 'fixture', 'complete': True,
        'classification': 'MEASURED physical pair', 'gpus': pf.WORKER_GPUS,
        'cases': [{'batch': b, 'depth': d, 'mode': m, 'rank0_exact': True, 'rank1_exact': True,
                   'changed_input_graph_replays': 10, 'solo_sha256': 'a'*64} for b, d, m in contract.CASES],
        'timeout0': {'missing_rank': 1, 'timeout_refused': True, 'sticky_retry_refused': True, 'process_exited': True},
        'timeout1': {'missing_rank': 0, 'timeout_refused': True, 'sticky_retry_refused': True, 'process_exited': True},
        'whole_worker': {'physical_pair': True, 'complete': True, 'identity_checks': 120, 'identity_exact': 120,
                         'archived_ids_exact': 48, 'archived_confidence_exact': 48}, 'source_sha256': {'fixture': 'b'*64}}


@pytest.mark.parametrize('mutation', ['solo', 'incomplete', 'B3', 'rank1', 'graph', 'sticky', 'teardown', 'archive', 'source'])
def test_solo_and_partial_physical_evidence_never_qualify(mutation: str) -> None:
    data = receipt(); contract.validate(data, WORKER.commit, 'fixture')
    if mutation == 'solo': data['classification'] = 'MEASURED synthetic solo'
    elif mutation == 'incomplete': data['complete'] = False
    elif mutation == 'B3': data['cases'] = [r for r in ck.rows(data['cases']) if r['batch'] != 3]
    elif mutation == 'rank1': ck.rows(data['cases'])[0]['rank1_exact'] = False
    elif mutation == 'graph': ck.rows(data['cases'])[0]['changed_input_graph_replays'] = 0
    elif mutation == 'sticky': record(data['timeout0'])['sticky_retry_refused'] = False
    elif mutation == 'teardown': record(data['timeout1'])['process_exited'] = False
    elif mutation == 'archive': record(data['whole_worker'])['archived_ids_exact'] = 47
    else: data['source_sha256'] = {}
    with pytest.raises(ValueError): contract.validate(data, WORKER.commit, 'fixture')


def test_pair_failure_restores_before_any_engine_or_request() -> None:
    fake = Fake('pair-qualification')
    result = w.window(fake, lambda row: None)
    assert fake.events == ['preflight', 'hold', 'snapshot', 'pair-qualification', 'restore']
    assert record(result['decision'])['status'] == 'REJECT'


def test_split_segments_are_refused_before_any_host_contact() -> None:
    fake = Fake()
    first = w.window(fake, lambda row: None, selected=w.ARMS[:2])
    assert 'stop' in first and fake.events == []
    second = w.window(fake, lambda row: None, selected=w.ARMS[2:], prior=first)
    assert 'stop' in second and fake.events == []
    assert record(second['decision'])['status'] == 'REJECT'


@pytest.mark.parametrize('bad', ['restore', 'commit', 'solo', 'missing-block', 'config'])
def test_all_old_segment_evidence_is_explicitly_retired(bad: str) -> None:
    prior: dict[str, Json] = {'segment': 'first', 'status': bad, 'physical_pair': receipt()}
    with pytest.raises(ValueError, match='Round5 retires'):
        w.validate_prior(prior, WORKER.commit)


def test_pair_driver_default_never_touches_a_process(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import subprocess
    import sys

    from offload.dspark.probabilistic.pp4 import pair_qualify
    def forbidden(*args: object, **kwargs: object) -> None: raise AssertionError('process forbidden')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(sys, 'argv', ['pair_qualify'])
    pair_qualify.main()
    assert capsys.readouterr().out.startswith('DRY RUN')


def test_qualification_shell_replaces_server_but_keeps_pair_open_owned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from offload.dspark.probabilistic.pp4.pair_qualify import SOURCES
    from offload.launch.windows.tests.test_pf1_window import Clock
    clock = Clock(); sup = pf.Supervisor(tmp_path, clock=clock)
    live = w.Live(tmp_path, sup, native=WORKER, remote='experiment', expected=WORKER.commit,
                  data_dir=tmp_path, tag='fixture', admission={'verifier_sha256': {}}, clock=clock)
    live.submit = clock()+w.SUBMIT_S
    live.pair_only = True
    calls: list[str] = []
    data = receipt(); data['source_sha256'] = {name: pf.digest(pf.ROOT/name) for name in SOURCES}
    def owned(label: str, host: str, tag: str, script: str, *, timeout: float) -> str:
        assert tag in live.native_tags and live.pair_open
        assert 'prod_window.sh pair COORD' in script
        assert '-m worker.server' not in script and '-m offload.dspark.probabilistic.pp4.pair_qualify --execute' in script
        assert 'gpu_locked.sh' in script and 'PP4_PAIR_LOCK_FD0=3 PP4_PAIR_LOCK_FD1=4' in script
        calls.append('owned-pair'); return ''
    def forbidden_budget(arm: w.Arm) -> float:
        raise AssertionError('pair-only must not require measurement budget')
    monkeypatch.setattr(live, 'block_budget', forbidden_budget)
    monkeypatch.setattr(live, 'owned', owned)
    monkeypatch.setattr(live, 'json', lambda *args: data)
    monkeypatch.setattr(live, 'pair_close', lambda **kwargs: calls.append('fenced-pair-close'))
    try:
        assert live.qualify_pair() == data
        assert calls == ['owned-pair', 'fenced-pair-close']
    finally:
        sup.close()


def test_pair_only_does_not_start_an_engine_or_claim_measurements() -> None:
    fake = Fake(); result = w.window(fake, lambda row: None, selected=())
    assert result['status'] == 'PAIR_QUALIFIED'
    assert record(result['decision'])['status'] == 'INCONCLUSIVE'
    assert fake.events == ['preflight', 'hold', 'snapshot', 'pair-qualification', 'restore']


@pytest.mark.parametrize('failure', ['', 'timeout0', 'whole_worker'])
def test_real_pair_driver_process_sequence_and_failed_child_never_qualifies(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    import json
    import subprocess
    import sys
    import time

    from offload.dspark.probabilistic.pp4 import pair_qualify as driver
    monkeypatch.setattr(driver, 'guard', lambda: None)  # CPU-only process boundary
    monkeypatch.setattr(driver, 'checked_output', lambda path: None)  # tmp_path is authorized PP4 CPU scratch
    monkeypatch.setattr(sys, 'executable', '/home/operator/miniconda3/bin/python3')
    monkeypatch.setenv('PP4_PAIR_LOCK_FD0', '3'); monkeypatch.setenv('PP4_PAIR_LOCK_FD1', '4')
    monkeypatch.setattr(subprocess, 'check_output', lambda *args, **kwargs: WORKER.commit+'\n')
    calls: list[str] = []
    def child(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        mode = argv[argv.index('--mode')+1] if '--mode' in argv else 'whole_worker'
        calls.append(mode)
        assert kwargs['pass_fds'] == (3, 4) and kwargs['check'] is True
        path = Path(argv[argv.index('--out')+1])
        if mode == failure:
            raise subprocess.TimeoutExpired(argv, 1)
        data = receipt()
        payload = {'cases': data['cases']} if mode == 'correctness' else data[mode]
        path.write_text(json.dumps(payload))
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(subprocess, 'run', child)
    out = tmp_path/'probe'
    if failure:
        with pytest.raises(subprocess.TimeoutExpired): driver.execute(out, WORKER.commit, 'fixture', time.time()+600)
        assert not ck.read(out/'receipt.json')['complete']
        assert calls[-1] == failure
    else:
        data = driver.execute(out, WORKER.commit, 'fixture', time.time()+600)
        contract.validate(data, WORKER.commit, 'fixture')
        assert calls == ['correctness', 'timeout0', 'timeout1', 'whole_worker']
        assert ck.read(out/'receipt.json')['complete'] is True


def test_hold_prefix_collision_is_not_adopted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import datetime
    import time
    sup = pf.Supervisor(tmp_path)
    live = w.Live(tmp_path, sup, native=WORKER, remote='experiment', expected=WORKER.commit,
                  data_dir=tmp_path, tag='owner', admission={'verifier_sha256': {}})
    monkeypatch.setattr(live, 'owned', lambda *args, **kwargs: '')
    hold = datetime.datetime.fromtimestamp(time.time(), datetime.timezone.utc).isoformat()+' coordinator PP4-WINDOW owner-other: unrelated'
    monkeypatch.setattr(live, 'ssh', lambda *args, **kwargs: hold)
    try:
        with pytest.raises(ValueError, match='exact run'): live.open_hold()
        assert not live.hold_written
    finally:
        sup.close()
