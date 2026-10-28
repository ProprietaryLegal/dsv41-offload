"""NBW recovery invariants using CPU fakes, with no SSH or inference traffic."""
from __future__ import annotations

import json
import shlex
import signal
import subprocess
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest

from offload.launch.windows import nbw_window as n
from offload.launch.windows import scoped_remote as owner
from offload.prefill_maxlen.evidence import Json, record
from offload.launch.windows import pf1_window as pf, post_close_watch as watchdog
from offload.launch.windows.tests import test_nbw_window as pure, test_pf1_window as base
from offload.launch.windows.tests.test_nbw_rehearsal import Remote, rehearsal, scoped_argv


@contextmanager
def masked() -> Iterator[None]:
    previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def prepare(live: n.Live, fake: Remote) -> None:
    live.sup.abort_at = fake.clock() + n.ABORT_S
    live.hold_written = True
    fake.hold = fake.original_hold = 'original NBW HOLD'
    live.remember('tag-engine', live.sup.abort_at)


@pytest.mark.parametrize('host', [*owner.NODES, 'localhost'])
@pytest.mark.parametrize('mode', ['fence', 'reap', 'absent'])
@pytest.mark.parametrize('failure', ['rc', 'receipt', 'transport'])
def test_every_host_and_control_phase_refuses_restore_without_proof(
    rehearsal: tuple[n.Live, Remote], monkeypatch: pytest.MonkeyPatch, host: str, mode: str, failure: str,
) -> None:
    live, fake = rehearsal
    prepare(live, fake)
    original = live.plain
    label = f'owned-{mode}-tag-engine-{host}'

    def fail(step: str, argv: list[str], timeout: float) -> tuple[int, str]:
        if step == label:
            fake.events.append(step)
            if failure == 'transport':
                raise ValueError('injected control transport loss')
            return (1, owner.FENCED if mode == 'fence' else owner.ABSENT) if failure == 'rc' else (0, 'missing receipt')
        return original(step, argv, timeout)

    monkeypatch.setattr(live, 'plain', fail)
    with masked(), pytest.raises(ValueError):
        live.restore(production_stopped=True)
    assert label in fake.events
    assert not fake.restores
    assert 'pair-close' not in fake.events and 'commit-close' not in fake.events
    assert fake.hold == 'original NBW HOLD'
    # No kill/proof can follow an incomplete fence phase.
    if mode == 'fence':
        assert not any(step.startswith(('owned-reap-', 'owned-absent-')) for step in fake.events)


@pytest.mark.parametrize('step', ['pair-close', 'experiment-stop', 'jit-wait-sparkhead', 'jit-wait-sparkworker-wired',
                                  'start', 'check', 'close'])
@pytest.mark.parametrize('failure', ['rc', 'transport'])
def test_failed_recovery_phase_keeps_hold_and_fences_submitted_mutations(
    rehearsal: tuple[n.Live, Remote], monkeypatch: pytest.MonkeyPatch, step: str, failure: str,
) -> None:
    live, fake = rehearsal
    prepare(live, fake)
    original = live.plain
    submitted: list[str] = []

    def fail(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        if label == step:
            fake.events.append(label)
            if label not in ('jit-wait-sparkhead', 'jit-wait-sparkworker-wired'):
                submitted.append(scoped_argv(argv[-1])[0])
            if failure == 'transport':
                raise ValueError('injected recovery transport loss')
            return 7, 'injected recovery failure'
        return original(label, argv, timeout)

    monkeypatch.setattr(live, 'plain', fail)
    with masked(), pytest.raises(ValueError):
        live.restore(production_stopped=True)
    assert 'commit-close' not in fake.events
    assert fake.hold == 'original NBW HOLD'
    for tag in submitted:
        for host in (*owner.NODES, 'localhost'):
            assert f'owned-absent-{tag}-{host}' in fake.events


def test_recovery_retry_uses_fresh_scopes_with_original_deadline(
    rehearsal: tuple[n.Live, Remote], monkeypatch: pytest.MonkeyPatch,
) -> None:
    live, fake = rehearsal
    prepare(live, fake)
    original = live.plain
    attempts: list[tuple[str, float]] = []

    def once(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        if label == 'start':
            tag, _ = scoped_argv(argv[-1])
            deadline = float(shlex.split(argv[-1])[1].split('=', 1)[1])
            attempts.append((tag, deadline))
            if len(attempts) == 1:
                raise ValueError('lost first recovery reply')
        return original(label, argv, timeout)

    monkeypatch.setattr(live, 'plain', once)
    deadline = live.restore_by
    with masked(), pytest.raises(ValueError, match='lost first recovery reply'):
        live.restore(production_stopped=True)
    fake.clock.now += 20
    with masked():
        result = live.restore(production_stopped=True)
    assert str(result['status']).startswith('RESTORED')
    assert len(attempts) == 2 and attempts[0][0] != attempts[1][0]
    assert attempts[0][1] == attempts[1][1] == pytest.approx(deadline - 90, abs=.000001)
    assert live.restore_by == deadline
    for tag, _ in attempts:
        assert f'owned-absent-{tag}-localhost' in fake.events


@pytest.mark.parametrize('loss', ['reply', 'receipt', 'rc', 'cancel'])
def test_ambiguous_close_retains_hold_and_cancels_late_commit(
    rehearsal: tuple[n.Live, Remote], monkeypatch: pytest.MonkeyPatch, loss: str,
) -> None:
    live, fake = rehearsal
    prepare(live, fake)
    original = live.plain

    def ambiguous(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        result = original(label, argv, timeout)
        if label == 'commit-close' and loss in ('receipt', 'rc'):
            return (0, 'reply missing receipt') if loss == 'receipt' else (7, owner.CLOSED)
        if label == ('cancel-close' if loss == 'cancel' else 'commit-close'):
            # The mutation happened; only the reply is missing.
            raise ValueError('injected final close ambiguity')
        return result

    monkeypatch.setattr(live, 'plain', ambiguous)
    message = 'close ambiguity' if loss in ('reply', 'cancel') else 'final owned HOLD close unconfirmed'
    with masked(), pytest.raises(ValueError, match=message):
        live.restore(production_stopped=True)
    assert fake.events.index('close') < fake.events.index('commit-close') < fake.events.index('cancel-close') < fake.events.index('retain-close')
    assert fake.hold == fake.original_hold == 'original NBW HOLD'
    assert fake.restores == ['start', 'check', 'close']


def test_restore_never_extends_deadline_or_submits_after_expiry(
    rehearsal: tuple[n.Live, Remote], monkeypatch: pytest.MonkeyPatch,
) -> None:
    live, fake = rehearsal
    prepare(live, fake)
    deadline = live.restore_by
    original = live.plain
    timeouts: list[float] = []

    def exhaust(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        timeouts.append(timeout)
        result = original(label, argv, timeout)
        if label == 'experiment-stop':
            fake.clock.now = deadline + 1
        return result

    monkeypatch.setattr(live, 'plain', exhaust)
    with masked(), pytest.raises(ValueError, match=r'HOLD\+90 exhausted'):
        live.restore(production_stopped=True)
    assert live.restore_by == deadline
    assert not fake.restores and 'commit-close' not in fake.events
    assert all(0 < timeout <= 900 for timeout in timeouts)
    fake.events.clear()
    with masked(), pytest.raises(ValueError, match=r'HOLD\+90 exhausted'):
        live.restore(production_stopped=True)
    assert not fake.events and live.restore_by == deadline


def test_phase_deadline_cannot_be_replaced(rehearsal: tuple[n.Live, Remote]) -> None:
    live, _ = rehearsal
    live.remember('phase', live.restore_by)
    with pytest.raises(ValueError, match='immutable phase deadline changed'):
        live.remember('phase', live.restore_by + 1)
    assert live.owner_deadlines['phase'] == live.restore_by
    for bad in (float('nan'), float('inf')):
        with pytest.raises(ValueError, match='finite scoped deadline'):
            live.remember('bad', bad)
    assert 'bad' not in live.owner_tags


def test_deferred_close_lost_reply_and_late_commit_cannot_remove_retained_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # Direct control calls exercise the real on-disk commit lock/fence protocol.
    # No scopes or external commands are started by this test.
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('SCOPED_REMOTE_DEADLINE', '9999999999')
    monkeypatch.setattr(owner, 'units', lambda tag: [])
    tag = 'fixture-restore-close'
    hold = tmp_path/'dsv41-deploy/HOLD'
    hold.parent.mkdir()
    original = 'original NBW-WINDOW fixture: HOLD\n'
    hold.write_text(original)
    assert owner.main(['rm', tag, '-f', str(hold)]) == 0
    assert hold.read_text() == original, 'W8 close must defer unlink until all scopes are absent'
    assert owner.main(['fence', tag]) == 0
    assert owner.main(['commit-close', tag]) == 0
    assert not hold.exists()  # committed reply is then lost
    assert owner.main(['cancel-close', tag]) == 0
    assert owner.main(['retain-close', tag]) == 0
    assert hold.read_text() == original
    with pytest.raises(ValueError, match='committed/cancelled'):
        owner.main(['commit-close', tag])
    assert hold.read_text() == original
    assert owner.CLOSED in capsys.readouterr().out


@pytest.mark.parametrize('host', [*owner.NODES, 'localhost'])
def test_absence_is_reproved_for_old_mutations_before_final_hold_commit(
    rehearsal: tuple[n.Live, Remote], monkeypatch: pytest.MonkeyPatch, host: str,
) -> None:
    live, fake = rehearsal
    prepare(live, fake)
    original = live.plain
    refused = f'owned-absent-tag-engine-{host}'

    def fail_final_proof(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        if label == refused and 'close' in fake.restores:
            fake.events.append(label)
            return 0, 'lost final absence proof'
        return original(label, argv, timeout)

    monkeypatch.setattr(live, 'plain', fail_final_proof)
    with masked(), pytest.raises(ValueError, match='HOLD stays open'):
        live.restore(production_stopped=True)
    assert fake.restores == ['start', 'check', 'close']
    assert fake.events.count(refused) >= 2
    assert fake.hold == 'original NBW HOLD' and 'commit-close' not in fake.events



def test_successful_cancel_reply_arriving_after_deadline_retains_hold(
    rehearsal: tuple[n.Live, Remote], monkeypatch: pytest.MonkeyPatch,
) -> None:
    live, fake = rehearsal
    prepare(live, fake)
    original = live.plain
    deadline = live.restore_by

    def late(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        reply = original(label, argv, timeout)
        if label == 'cancel-close':
            fake.clock.now = deadline + 1
        return reply

    monkeypatch.setattr(live, 'plain', late)
    with masked(), pytest.raises(ValueError, match='restore completed late'):
        live.restore(production_stopped=True)
    assert fake.events[-1] == 'retain-close'
    assert fake.hold == fake.original_hold == 'original NBW HOLD'
    assert live.restore_by == deadline


def test_hold_open_can_never_be_repeated_or_renew_deadlines(rehearsal: tuple[n.Live, Remote]) -> None:
    live, fake = rehearsal
    live.open_hold()
    deadlines = live.submit, live.sup.abort_at, live.restore_by, live.open_deadline, live.hold_scope_deadline
    fake.clock.now += 1
    calls = list(fake.commands)
    with pytest.raises(ValueError, match='cannot be reopened or renewed'):
        live.open_hold()
    assert fake.commands == calls
    assert (live.submit, live.sup.abort_at, live.restore_by, live.open_deadline, live.hold_scope_deadline) == deadlines


def test_hold_scope_lease_outlives_the_whole_window(rehearsal: tuple[n.Live, Remote]) -> None:
    # The HOLD-opener scope hosts the placeholder server between arms; its guardian must not kill it after the drain.
    # S55: a long preflight before open_hold() must not shorten the lease relative to the HOLD it protects.
    live, fake = rehearsal
    fake.clock.now += pf.DRAIN_S + 120  # preflight longer than the old construction-based opening ceiling
    live.open_hold()
    tag = live.tag + '-hold'
    assert live.owner_deadlines[tag] == live.hold_scope_deadline
    assert live.owner_deadlines[tag] >= live.restore_by
    assert live.owner_deadlines[tag] - live.open_deadline == n.RESTORE_S


def test_hold_published_after_its_opening_ceiling_is_refused(rehearsal: tuple[n.Live, Remote],
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    # S55: a HOLD that lands after the frozen opening ceiling would outlive the placeholder lease; refuse it.
    live, fake = rehearsal
    original = fake.call
    def slow(label: str, argv: list[str], **kwargs: object) -> str:
        if label == 'hold-open':
            fake.clock.now += pf.DRAIN_S + 61
        return original(label, argv, **kwargs)  # type: ignore[arg-type]
    monkeypatch.setattr(live, 'call', slow)
    with pytest.raises(ValueError, match='after its opening ceiling'):
        live.open_hold()


def test_wrong_hold_marker_with_incidental_tag_never_adopts_or_starts_experiment(
    rehearsal: tuple[n.Live, Remote], monkeypatch: pytest.MonkeyPatch,
) -> None:
    live, fake = rehearsal
    original = live.call
    other_hold = f'{base.iso(fake.clock())} DSV41-OFFLOAD (session coordinator): OTHER-WINDOW somebody: mentions tag'

    def wrong(label: str, argv: list[str], *, stdin: str | None = None,
              timeout: float = 120, inflight: int = 0) -> str:
        if label == 'hold-read':
            fake.hold = other_hold
            return other_hold
        return original(label, argv, stdin=stdin, timeout=timeout, inflight=inflight)

    monkeypatch.setattr(live, 'call', wrong)
    result = n.window(live, lambda row: None)
    assert 'stop' in result and not live.hold_written
    assert str(record(result['restore'])['status']).startswith('NOT NEEDED')
    assert not fake.started and not fake.restores and 'pair-close' not in fake.events
    assert fake.hold == other_hold


@pytest.mark.parametrize('restored,watch_status,expected', [
    ('RESTORED: verified', 'PASS', 0), ('RESTORED: verified', 'FAILED', 2),
    ('FAILED', 'PASS', 2), ('NOT NEEDED: no HOLD', 'PASS', 0),
])
def test_main_runs_watchdog_only_after_verified_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restored: str, watch_status: str, expected: int,
) -> None:
    monkeypatch.setenv('WINDOW_COORDINATOR', 'Deepseek/coordinator')
    monkeypatch.setattr(pf, 'CAMPAIGN', tmp_path)
    expected_commit = 'a' * 40
    calls: list[str] = []

    def git(argv: list[str], **kwargs: object) -> str:
        assert argv[0] == 'git'
        return expected_commit if argv[1] == 'rev-parse' else ''

    class Supervisor:
        def __init__(self, out: Path) -> None:
            calls.append('supervisor')

        def close(self) -> None:
            calls.append('supervisor-close')

    def live(*args: object, **kwargs: object) -> object:
        calls.append('live')
        return object()

    def window(backend: object, checkpoint: Callable[[dict[str, Json]], None]) -> dict[str, Json]:
        checkpoint({'phase': 'fake-complete'})
        return {'status': 'COMPLETE', 'restore': {'status': restored}, 'decision': {'status': 'CANDIDATE'}}

    def recovery(tag: str, out: Path) -> object:
        assert tag == 'NBW-main-watch' and out == tmp_path/'NBW-main/post-close-commands'
        calls.append('watch-recovery')
        return object()

    def watch(health: object, recovery: object, *, log: Callable[[str], None]) -> watchdog.Result:
        assert calls.index('supervisor-close') < calls.index('watch-recovery')
        calls.append('watch')
        log('fake CPU watchdog result')
        return watchdog.Result(watch_status, 30, 0, 0)

    monkeypatch.setattr(subprocess, 'check_output', git)
    monkeypatch.setattr(pf, 'Supervisor', Supervisor)
    monkeypatch.setattr(n, 'Live', live)
    monkeypatch.setattr(n, 'window', window)
    monkeypatch.setattr(watchdog, 'ProductionHealth', object)
    monkeypatch.setattr(watchdog, 'ScopedRecovery', recovery)
    monkeypatch.setattr(watchdog, 'watch', watch)
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        rc = n.main(['--execute', '--out', str(tmp_path/'NBW-main'),
            '--expected-experiment-commit', expected_commit, '--native-tree', pure.NATIVE.tree,
            '--native-commit', pure.NATIVE.commit, '--native-identity', pure.NATIVE.identity,
            '--native-markov-identity', pure.NATIVE.markov])
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)
    assert rc == expected
    result = record(cast(Json, json.loads((tmp_path/'NBW-main/summary.json').read_text())))
    assert ('watch' in calls) == restored.startswith('RESTORED')
    if restored.startswith('RESTORED'):
        assert record(result['post_close_watch'])['status'] == watch_status
        assert (tmp_path/'NBW-main/post-close-watch.log').read_text() == 'fake CPU watchdog result\n'
        assert record(result['decision'])['status'] == ('CANDIDATE' if watch_status == 'PASS' else 'NOT A CANDIDATE')
    else:
        assert 'post_close_watch' not in result
