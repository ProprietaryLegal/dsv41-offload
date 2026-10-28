"""PP4 shared-owner integration; CPU fixtures only, real scopes for receiving hops."""
from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import cast

import pytest

from offload.launch.windows import hold_admission
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import pp4_window as w
from offload.launch.windows import scoped_remote as owner
from offload.launch.windows.tests.test_pp4_window import WORKER
from offload.launch.windows.tests.test_scoped_remote import Sandbox, wait_for

pytest_plugins = ["offload.launch.windows.tests.test_scoped_remote"]


@pytest.mark.parametrize('kind', ['stop', 'run'])
@pytest.mark.parametrize('pause', ['before-registration', 'active'])
def test_real_pp4_owned_second_hop_and_retry(sandbox: Sandbox, kind: str, pause: str, monkeypatch: pytest.MonkeyPatch) -> None:
    s = sandbox
    tag = s.tag('pp4-' + kind)
    home = s.homes['sparkworker-wired']
    state = home/'production'
    state.write_text('active')
    release = home/'release'
    if pause == 'before-registration':
        (s.root/'sparkworker-wired-transport-gate').touch()
    body = f'touch {home}/active; while [ ! -e {release} ]; do sleep .02; done; echo {kind} > {state}'
    live = w.Live(s.root, pf.Supervisor(s.root), native=WORKER, remote='experiment', expected=WORKER.commit,
                  data_dir=s.root, tag=tag, admission={'verifier_sha256': {}})
    live.sup.abort_at = s.deadlines[tag]
    submitted: list[str] = []
    def ssh(label: str, host: str, command: str, stdin: str | None = None, *, timeout: float = 120) -> str:
        submitted.append(command)
        return ''
    monkeypatch.setattr(live, 'ssh', ssh)
    live.owned('engine-start', 'sparkhead', tag, 'ssh sparkworker ' + shlex.quote(body), timeout=30)
    assert live.owner_tags == [tag] and live.owner_deadlines[tag] == s.deadlines[tag]
    process = subprocess.Popen(['bash', '-c', submitted[0]], env=s.env(), stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, text=True)
    s.processes.append(process)
    wait_for(s.root/'sparkworker-wired-accepted' if pause == 'before-registration' else home/'active')
    s.close('new', tag)
    (s.root/'sparkworker-wired-transport-gate').unlink(missing_ok=True)
    release.touch()
    process.wait(timeout=10)
    assert state.read_text() == 'active'
    retry = s.run(s.command('new', 'launch', tag, ['bash', '-c', body]), 'sparkworker-wired')
    assert retry.returncode == 75 and state.read_text() == 'active'
    live.sup.close()


def test_s55_lease_origin_admission_and_no_renewal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [1000.]
    sup = pf.Supervisor(tmp_path, clock=lambda: clock[0])
    live = w.Live(tmp_path, sup, native=WORKER, remote='experiment', expected=WORKER.commit,
                  data_dir=tmp_path, tag='lease', admission={'verifier_sha256': {}}, clock=lambda: clock[0])
    clock[0] += 1080  # long preflight must not consume placeholder lifetime
    before = clock[0]
    calls: list[str] = []
    def ssh(label: str, host: str, command: str, stdin: str | None = None, *, timeout: float = 120) -> str:
        calls.append(label)
        if label == 'hold-open':
            assert 'SCOPED_REMOTE_DEADLINE=' in command and hold_admission.NOT_OWNED in command
            clock[0] += pf.DRAIN_S - 10
        from offload.launch.windows.tests.test_pf1_window import iso
        return iso(clock[0]) + ' DSV41-OFFLOAD (session coordinator): PP4-WINDOW lease: fixture'
    monkeypatch.setattr(live, 'ssh', ssh)
    monkeypatch.setattr(sup, 'tunnel', lambda argv: None)
    live.open_hold()
    assert live.open_deadline == before + pf.DRAIN_S + 60
    assert live.owner_deadlines['lease-hold'] == live.hold_scope_deadline >= live.restore_by
    saved = live.restore_by, live.hold_scope_deadline
    with pytest.raises(ValueError, match='cannot be reopened or renewed'):
        live.open_hold()
    assert saved == (live.restore_by, live.hold_scope_deadline)
    sup.close()


@pytest.mark.parametrize('mode', ['fence', 'reap', 'absent'])
@pytest.mark.parametrize('host', w.OWNER_HOSTS)
def test_all_host_control_failures_prevent_pair_and_w8(tmp_path: Path, mode: str, host: str) -> None:
    calls: list[str] = []
    def plain(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        calls.append(label)
        if label == f'owned-{mode}-engine-{host}':
            return 1, 'injected missing proof'
        return 0, owner.FENCED if label.startswith('owned-fence-') else owner.ABSENT
    sup = pf.Supervisor(tmp_path)
    live = w.Live(tmp_path, sup, native=WORKER, remote='experiment', expected=WORKER.commit,
                  data_dir=tmp_path, tag='fixture', admission={'verifier_sha256': {}}, plain=plain)
    live.hold_written = True
    live.remember('engine', live.restore_by - 90)
    with pytest.raises(ValueError, match='HOLD stays open'):
        live.restore(production_stopped=True)
    assert 'pair-close' not in calls and 'start' not in calls and 'commit-close' not in calls
    if mode == 'fence':
        assert not any(label.startswith(('owned-reap-', 'owned-absent-')) for label in calls)


def test_pins_only_shared_ownership_and_recovery_is_scoped() -> None:
    assert 'offload/launch/windows/scoped_remote.py' in w.PIN_FILES
    assert not (pf.ROOT/'offload/launch/windows/pp4_remote.py').exists()
    assert {'offload/launch/windows/'+name+'.py' for name in
            ('post_close_watch', 'hold_admission', 'spark_quiet', 'production_health')} <= set(w.PIN_FILES)


def test_repeated_recovery_proofs_keep_distinct_logs(tmp_path: Path) -> None:
    sup = pf.Supervisor(tmp_path)
    (tmp_path/'cmd').mkdir(exist_ok=True)
    live = w.Live(tmp_path, sup, native=WORKER, remote='experiment', expected=WORKER.commit,
                  data_dir=tmp_path, tag='repeat', admission={'verifier_sha256': {}})
    for text in ('first proof', 'second proof'):
        assert live.local('same-phase-absent', ['bash', '-c', 'printf %s '+shlex.quote(text)], 5) == (0, text)
    logs = sorted((tmp_path/'cmd').glob('restore-*.log'))
    assert [p.name for p in logs] == ['restore-0001-same-phase-absent.log', 'restore-0002-same-phase-absent.log']
    assert not (tmp_path/'cmd'/'restore-same-phase-absent.log').exists()
    assert [p.read_text() for p in logs] == ['first proof', 'second proof']
    sup.close()


@pytest.mark.parametrize('restore,watch_status,exit_code', [
    ('RESTORED: verified', 'PASS', 0), ('RESTORED: verified', 'FAILED', 2),
    ('FAILED', 'PASS', 2), ('NOT NEEDED: preflight only', 'PASS', 0),
])
def test_main_watchdog_is_mandatory_after_restore(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        restore: str, watch_status: str, exit_code: int) -> None:
    from types import SimpleNamespace

    from offload.launch.windows import post_close_watch as watch
    from offload.launch.windows import pp4_prerequisites as prereq
    from offload.prefill_maxlen.evidence import Json, record

    monkeypatch.setenv('WINDOW_COORDINATOR', 'Deepseek/coordinator')
    monkeypatch.setattr(pf, 'CAMPAIGN', tmp_path)
    monkeypatch.setattr(subprocess, 'check_output', lambda argv, **kwargs: WORKER.commit if argv[1] == 'rev-parse' else '')
    monkeypatch.setattr(prereq, 'staging', lambda *args: {'worker_identity': WORKER.identity, 'markov_identity': WORKER.markov})
    calls: list[str] = []
    monkeypatch.setattr(pf, 'Supervisor', lambda out: SimpleNamespace(close=lambda: calls.append('close')))
    monkeypatch.setattr(w, 'Live', lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(w, 'window', lambda *args, **kwargs: {
        'status': 'PAIR_QUALIFIED', 'restore': {'status': restore}, 'decision': {'status': 'CANDIDATE'}})
    monkeypatch.setattr(watch, 'ProductionHealth', object)
    def recovery(tag: str, out: Path) -> object:
        assert calls == ['close']
        assert tag == 'PP4-test-watch'
        calls.append('recovery')
        return object()
    def watchdog(health: object, recovery: object, *, log: object) -> watch.Result:
        calls.append('watch')
        return watch.Result(watch_status, 30, 0, 0)
    monkeypatch.setattr(watch, 'ScopedRecovery', recovery)
    monkeypatch.setattr(watch, 'watch', watchdog)
    import signal
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        rc = w.main(['--execute', '--pair-only', '--qualification', str(tmp_path/'staging.json'),
            '--expected-experiment-commit', WORKER.commit, '--worker-commit', WORKER.commit,
            '--worker-tree', WORKER.tree, '--worker-identity', WORKER.identity,
            '--worker-markov-identity', WORKER.markov, '--out', str(tmp_path/'PP4-test')])
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)
    assert rc == exit_code
    assert ('watch' in calls) == restore.startswith('RESTORED')
    result = record(cast(Json, json.loads((tmp_path/'PP4-test/summary.json').read_text())))
    if restore.startswith('RESTORED'):
        assert record(result['post_close_watch'])['status'] == watch_status
        assert record(result['decision'])['status'] == ('CANDIDATE' if watch_status == 'PASS' else 'REJECT')


def test_local_control_precheck_never_requires_loopback_ssh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sup = pf.Supervisor(tmp_path)
    live = w.Live(tmp_path, sup, native=WORKER, remote='experiment', expected=WORKER.commit,
                  data_dir=tmp_path, tag='fresh', admission={'verifier_sha256': {}})
    calls: list[list[str]] = []
    def call(label: str, argv: list[str], **kwargs: object) -> str:
        calls.append(argv)
        return ''
    monkeypatch.setattr(live, 'call', call)
    live.fresh_controls(['fresh-hold', 'fresh-worker'])
    assert [argv[-2] for argv in calls if argv[0] == 'ssh'] == list(owner.NODES)
    local = [argv for argv in calls if argv[0] != 'ssh']
    assert len(local) == 1 and local[0][:2] == ['bash', '-c']
    assert 'scoped-control/fresh-hold' in local[0][-1] and 'scoped-control/fresh-worker' in local[0][-1]
    sup.close()
