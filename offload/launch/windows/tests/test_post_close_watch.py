"""CPU fakes only: no HTTP server, SSH, production endpoint or GPU is used."""
from __future__ import annotations

from collections.abc import Iterator
import hashlib
from pathlib import Path
import subprocess
import urllib.request
import shlex
import signal

import pytest

from offload.launch.windows import post_close_watch as w
from offload.launch.windows import scoped_remote as owner

CHECK = ''.join(f'greedy K1-post vs restored: {i} IDENTICAL 100\n' for i in range(6)) + 'PRODUCTION HOLD OK\nCHECK: DONE\n'
W8 = {'start': 'START: K1 START VERIFIED\n', 'check': CHECK, 'close': 'CLOSE: window closed\n'}


class Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, amount: float) -> None:
        self.value += amount


class Health:
    def __init__(self, failures: set[int]) -> None:
        self.failures = failures
        self.calls = 0

    def probe(self) -> None:
        self.calls += 1
        if self.calls in self.failures:
            raise OSError('fake production unavailable')


class Recovery:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[float] = []
        self.fail = fail

    def recover(self, deadline: float) -> None:
        self.calls.append(deadline)
        if self.fail:
            raise w.WatchError('fake missing greedy index')


def test_schedule_is_fifteen_minutes_every_thirty_seconds() -> None:
    clock, health, recovery = Clock(), Health(set()), Recovery()
    logs: list[str] = []
    result = w.watch(health, recovery, log=logs.append, clock=clock, sleep=clock.sleep)
    assert result == w.Result('PASS', 30, 0, 0)
    assert clock() == 900
    assert not recovery.calls
    assert logs[0].startswith('WATCH START:') and logs[-1].startswith('WATCH COMPLETE:')


def test_success_resets_failure_streak() -> None:
    clock, health, recovery = Clock(), Health({1, 3, 5}), Recovery()
    result = w.watch(health, recovery, log=lambda _: None, clock=clock, sleep=clock.sleep)
    assert result.recoveries == 0 and result.failures == 3


def test_two_consecutive_failures_recover_once_and_recurrence_is_loud() -> None:
    clock, health, recovery = Clock(), Health({2, 3, 6, 7}), Recovery()
    logs: list[str] = []
    result = w.watch(health, recovery, log=logs.append, clock=clock, sleep=clock.sleep)
    assert result.status.startswith('FAILED: recurrence')
    assert result.recoveries == 1 and len(recovery.calls) == 1
    assert recovery.calls == [60 + w.RECOVERY_S]
    assert any('allowance exhausted' in line for line in logs)


def test_failed_recovery_is_never_retried() -> None:
    clock, health, recovery = Clock(), Health({1, 2, 3}), Recovery(True)
    logs: list[str] = []
    result = w.watch(health, recovery, log=logs.append, clock=clock, sleep=clock.sleep)
    assert result == w.Result('FAILED: recovery unverified; retain HOLD', 2, 2, 1)
    assert len(recovery.calls) == 1
    assert any('HOLD stays open' in line for line in logs)


@pytest.mark.parametrize('text', [
    CHECK.replace('5 IDENTICAL', '4 IDENTICAL'),
    CHECK + 'greedy K1-post vs restored: 0 IDENTICAL 100\n',
    CHECK.replace('5 IDENTICAL', '6 IDENTICAL'),
    CHECK.replace('5 IDENTICAL 100', '5 DIVERGES at char 1'),
    CHECK.replace('5 IDENTICAL 100', '5 IDENTICAL 100 junk'),
    CHECK.replace('PRODUCTION HOLD OK', ''),
    CHECK.replace('CHECK: DONE', ''),
    CHECK + 'STOP: wrong KV size\n',
])
def test_strict_greedy_six_distinct_indices(text: str) -> None:
    w.restore_verified('check', 0, CHECK)
    with pytest.raises(w.WatchError):
        w.restore_verified('check', 0, text)


@pytest.mark.parametrize('step', ['start', 'check', 'close'])
def test_w8_status_and_success_markers_required(step: str) -> None:
    w.restore_verified(step, 0, W8[step])
    with pytest.raises(w.WatchError):
        w.restore_verified(step, 1, W8[step])
    with pytest.raises(w.WatchError):
        w.restore_verified(step, 0, '')


def test_probe_checks_models_and_sends_one_token_greedy_priority_twenty() -> None:
    calls: list[tuple[str, dict[str, object] | None]] = []
    def request(path: str, body: dict[str, object] | None) -> object:
        calls.append((path, body))
        if body is None:
            return {'data': [{'root': w.MODEL_ROOT, 'id': 'legal-v2a', 'max_model_len': 327680}]}
        return {'choices': [{'text': 'OK', 'finish_reason': 'length'}], 'usage': {'completion_tokens': 1}}
    w.ProductionHealth(request=request).probe()
    assert [path for path, _ in calls] == ['/v1/models', '/v1/completions']
    assert calls[1][1] == {'model': 'legal-v2a', 'prompt': 'Reply OK.', 'max_tokens': 1,
                            'temperature': 0, 'top_p': 1, 'seed': 42, 'priority': 20, 'stream': False}


@pytest.mark.parametrize('reply', [
    {'data': []}, {'data': [{'root': '/wrong', 'id': 'x'}]}, {'data': 'wrong'},
    {'data': [{'root': w.MODEL_ROOT, 'id': 'model', 'max_model_len': 524288}]},
    {'data': [{'root': w.MODEL_ROOT, 'id': 'model'}]},
])
def test_wrong_model_never_gets_completion_probe(reply: object) -> None:
    calls: list[str] = []
    def request(path: str, body: dict[str, object] | None) -> object:
        calls.append(path)
        return reply
    with pytest.raises(w.WatchError):
        w.ProductionHealth(request=request).raw()
    assert calls == ['/v1/models']


@pytest.mark.parametrize('reply', [
    {'choices': [], 'usage': {'completion_tokens': 1}},
    {'choices': [{'text': 'OK', 'finish_reason': None}], 'usage': {'completion_tokens': 1}},
    {'choices': [{'text': 'OK', 'finish_reason': 'length'}], 'usage': {'completion_tokens': 0}},
    {'choices': [{'text': 'OK', 'finish_reason': 'length'}], 'usage': {'completion_tokens': True}},
])
def test_incomplete_completion_is_health_failure(reply: object) -> None:
    def request(path: str, body: dict[str, object] | None) -> object:
        return {'data': [{'root': w.MODEL_ROOT, 'id': 'model', 'max_model_len': 327680}]} if body is None else reply
    with pytest.raises(w.WatchError):
        w.ProductionHealth(request=request).raw()


class Transport:
    """Commands are parsed as argv via shlex; never executed."""
    def __init__(self, failure: str = '') -> None:
        self.calls: list[tuple[str, list[str], float]] = []
        self.failure = failure

    def __call__(self, label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        self.calls.append((label, argv, timeout))
        if label == self.failure:
            return 1, 'fake failed transport'
        if label == 'run-hold-open':
            return 0, 'window open: POST-CLOSE-WATCH fake:\n'
        if label == 'run-stop':
            return 0, ''
        if label.startswith('quiet-'):
            return 0, 'WATCH SPARK QUIET\n'
        if label.startswith('run-restore-'):
            return 0, W8[label.removeprefix('run-restore-')]
        for prefix, marker in (('fence-', owner.FENCED), ('reap-', owner.ABSENT), ('absent-', owner.ABSENT)):
            if label.startswith(prefix):
                return 0, marker + '\n'
        return 0, {'commit-close': owner.CLOSED, 'cancel-close': 'SCOPED CLOSE COMMIT FENCED',
                   'retain-close': owner.RETAINED}[label] + '\n'


@pytest.fixture
def recovery(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[w.ScopedRecovery, Transport]]:
    transport = Transport()
    # Hash pin is exercised separately; no filesystem writes or /tmp fixture.
    monkeypatch.setattr(Path, 'read_bytes', lambda _: b'reviewed W8')
    monkeypatch.setattr(w, 'RESTORE_SHA256', hashlib.sha256(b'reviewed W8').hexdigest())
    backend = w.ScopedRecovery('fake', Path('/unused/not-created'), call=transport, clock=lambda: 0,
                               wall=lambda: 1000)
    yield backend, transport


def test_recovery_owns_all_phases_and_proves_all_hosts_before_close(
        recovery: tuple[w.ScopedRecovery, Transport]) -> None:
    backend, transport = recovery
    backend.recover(1800)
    labels = [label for label, _, _ in transport.calls]
    assert [label for label in labels if label.startswith('run-')] == [
        'run-hold-open', 'run-stop', 'run-restore-start', 'run-restore-check', 'run-restore-close']
    assert labels.index('quiet-sparkhead') < labels.index('run-restore-start')
    quiet = next(argv[-1] for label, argv, _ in transport.calls if label == 'quiet-sparkhead')
    assert 'docker' in quiet and 'inspect' in quiet and '{{.ID}}' in quiet
    assert 'tf_experts' in quiet and '_load_engram_remote_ext' in quiet
    assert '--query-compute-apps=pid' in quiet
    opened = next(argv[-1] for label, argv, _ in transport.calls if label == 'run-hold-open')
    stop = next(argv[-1] for label, argv, _ in transport.calls if label == 'run-stop')
    assert 'os.link(candidate, hold)' in opened and 'offload/launch/window.sh open' not in opened
    assert 'first stop refuses unowned/changed HOLD' in stop and 'guarded_stop' in stop
    assert labels[-3:] == ['absent-fake-restore-close-x86host', 'commit-close', 'cancel-close']
    for phase in backend.phases:
        fences = [labels.index('fence-' + phase + '-' + host) for host in w.HOSTS]
        reaps = [labels.index('reap-' + phase + '-' + host) for host in w.HOSTS]
        absent = [labels.index('absent-' + phase + '-' + host) for host in w.HOSTS]
        assert max(fences) < min(reaps) and max(reaps) < min(absent)
    for label, argv, timeout in transport.calls:
        assert timeout <= 1800
        if label.startswith('run-'):
            assert 'launch' in argv[-1] and 'SCOPED_REMOTE_DEADLINE=2795.000000' in argv[-1]
    with pytest.raises(w.WatchError, match='already admitted'):
        backend.recover(1900)


@pytest.mark.parametrize('failure', ['run-hold-open', 'run-stop', 'quiet-sparkhead', 'quiet-sparkworker-wired',
                                    'run-restore-start', 'run-restore-check',
                                    'fence-fake-restore-close-x86host', 'absent-fake-restore-close-x86host'])
def test_failures_retain_hold_and_never_commit(recovery: tuple[w.ScopedRecovery, Transport], failure: str) -> None:
    backend, transport = recovery
    transport.failure = failure
    with pytest.raises(w.WatchError):
        backend.recover(1800)
    assert 'commit-close' not in [label for label, _, _ in transport.calls]


def test_fence_failure_prevents_every_reap_but_fences_all_hosts(recovery: tuple[w.ScopedRecovery, Transport]) -> None:
    backend, transport = recovery
    transport.failure = 'fence-fake-hold-open-sparkhead'
    with pytest.raises(w.WatchError):
        backend.recover(1800)
    labels = [label for label, _, _ in transport.calls]
    assert 'fence-fake-hold-open-x86host' in labels
    assert not any(label.startswith('reap-') for label in labels)


def test_lost_commit_reply_cancels_and_retains_original_hold(recovery: tuple[w.ScopedRecovery, Transport]) -> None:
    backend, transport = recovery
    transport.failure = 'commit-close'
    with pytest.raises(w.WatchError, match='unconfirmed'):
        backend.recover(1800)
    assert [label for label, _, _ in transport.calls][-3:] == ['commit-close', 'cancel-close', 'retain-close']


def test_expired_lease_never_calls_transport(recovery: tuple[w.ScopedRecovery, Transport]) -> None:
    backend, transport = recovery
    backend.deadline = -1
    with pytest.raises(w.WatchError, match='deadline'):
        backend.invoke('late', 'sparkhead', 'false')
    assert not transport.calls


def test_dry_default_has_no_network_or_files(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError('dry plan performed I/O')
    monkeypatch.setattr(urllib.request, 'urlopen', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(Path, 'mkdir', forbidden)
    assert w.main([]) == 0
    assert 'PLAN ONLY' in capsys.readouterr().out


def test_recovery_restores_signal_handlers_on_failure() -> None:
    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    before = {sig: signal.getsignal(sig) for sig in signals}
    with pytest.raises(RuntimeError):
        with w.masked_signals():
            assert all(signal.getsignal(sig) == signal.SIG_IGN for sig in signals)
            raise RuntimeError('fake failure')
    assert {sig: signal.getsignal(sig) for sig in signals} == before


def test_w8_pin_drift_refuses_before_any_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = Transport()
    monkeypatch.setattr(Path, 'read_bytes', lambda _: b'changed W8')
    backend = w.ScopedRecovery('fake', Path('/unused'), call=transport, clock=lambda: 0, wall=lambda: 1000)
    with pytest.raises(w.WatchError, match='changed'):
        backend.recover(1800)
    assert not transport.calls


def test_deadline_crossing_cannot_advance_to_next_phase(recovery: tuple[w.ScopedRecovery, Transport]) -> None:
    backend, transport = recovery
    clock = Clock()
    backend.clock = clock
    original = transport.__call__
    def call(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        result = original(label, argv, timeout)
        if label == 'run-stop':
            clock.value = 1801
        return result
    backend.call = call
    with pytest.raises(w.WatchError, match='deadline|absence unproved'):
        backend.recover(1800)
    labels = [label for label, _, _ in transport.calls]
    assert 'run-stop' in labels
    assert 'run-restore-start' not in labels and 'commit-close' not in labels
    assert backend.deadline == 1800  # never renew the admission deadline


def test_ambiguous_launch_is_fenced_on_every_destination(recovery: tuple[w.ScopedRecovery, Transport]) -> None:
    backend, transport = recovery
    original = transport.__call__
    def call(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        if label == 'run-hold-open':
            transport.calls.append((label, argv, timeout))
            raise OSError('fake reply lost after remote HOLD write')
        return original(label, argv, timeout)
    backend.call = call
    with pytest.raises(OSError, match='reply lost'):
        backend.recover(1800)
    labels = [label for label, _, _ in transport.calls]
    for host in w.HOSTS:
        assert 'fence-fake-hold-open-' + host in labels
        assert 'absent-fake-hold-open-' + host in labels
    assert not any(label.startswith('run-restore-') for label in labels)


def test_execute_without_coordinator_refuses_before_any_write_or_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('WINDOW_COORDINATOR', raising=False)
    out = tmp_path/'watch'
    with pytest.raises(w.WatchError, match='WINDOW_COORDINATOR'):
        w.main(['--execute', '--tag', 'fixture', '--out', str(out)])
    assert not out.exists()


def test_late_cancel_reply_still_retains_hold_after_deadline(recovery: tuple[w.ScopedRecovery, Transport]) -> None:
    backend, transport = recovery
    clock = Clock()
    backend.clock = clock
    original = transport.__call__
    def late(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        result = original(label, argv, timeout)
        if label == 'cancel-close':
            clock.value = 1801
        return result
    backend.call = late
    with pytest.raises(w.WatchError, match='after immutable deadline'):
        backend.recover(1800)
    assert [label for label, _, _ in transport.calls][-3:] == ['commit-close', 'cancel-close', 'retain-close']
    assert backend.deadline == 1800


def test_default_logger_survives_repeated_phase_labels(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # S66: the real exclusive logger must survive a complete recovery whose phases are fenced again.
    monkeypatch.setattr(Path, 'read_bytes', lambda _: b'reviewed W8')
    monkeypatch.setattr(w, 'RESTORE_SHA256', hashlib.sha256(b'reviewed W8').hexdigest())
    transport = Transport()
    backend = w.ScopedRecovery('fake', tmp_path / 'commands', clock=lambda: 0, wall=lambda: 1000)
    def logged(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        rc, text = transport(label, argv, timeout)
        return backend.local(label, ['bash', '-c', f'printf %s {shlex.quote(text)}; exit {rc}'], timeout)
    backend.call = logged
    backend.recover(1800)
    labels = [label for label, _, _ in transport.calls]
    assert len(set(labels)) < len(labels)
    logs = sorted((tmp_path / 'commands').iterdir())
    assert [p.name for p in logs] == [f'{n:04d}-{label}.log' for n, label in enumerate(labels, 1)]

