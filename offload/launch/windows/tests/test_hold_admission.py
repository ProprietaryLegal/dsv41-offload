"""S54 FIX3: actual admission commands in real Controller scopes, private HOME.

curl, services and production state are file-only fixtures. No HTTP/SSH/GPU or
Docker request goes to a real host. Scope cleanup uses only own1test identities.
"""
from __future__ import annotations

import os
from pathlib import Path
import shlex
import sys
import time
import uuid

import pytest

from offload.launch.windows import hold_admission as a
from offload.launch.windows.tests.test_scoped_remote import (
    OPS1, Sandbox, frozen, sandbox as sandbox, wait_for,
)
from offload.launch.windows.tests.test_post_close_watch import Clock


def prepare(s: Sandbox) -> Path:
    home = s.homes['sparkhead']
    prod = home/'dsv41-deploy'
    (prod/'offload/launch').mkdir(parents=True)
    (prod/'offload/launch/window.sh').write_text(frozen(OPS1, 'offload/launch/window.sh'))
    (prod/'offload/launch/window.sh').chmod(0o700)
    (prod/'dsv41.env').write_text('HEAD=spark\nWORKER=sparkworker\n')
    curl = s.binary/'curl'
    curl.write_text('#!' + sys.executable + '\nimport os,sys\nfrom pathlib import Path\n'
        'home=Path(os.environ["HOME"])\n'
        'if sys.argv[-1].endswith("/metrics"):\n'
        ' (home/("draining-"+os.environ.get("OWN1_ADMISSION","old"))).touch()\n'
        ' value=0 if (home/"release-drain").exists() else 1\n'
        ' print("vllm:num_requests_running{model=\\\"x\\\"} "+str(value))\n'
        ' print("vllm:num_requests_waiting{model=\\\"x\\\"} 0")\n'
        'else: print("healthy")\n')
    curl.chmod(0o700)
    sleep = s.binary/'sleep'
    sleep.write_text('#!/bin/bash\nexec /usr/bin/sleep .02\n')
    sleep.chmod(0o700)
    return home


def admit(s: Sandbox, tag: str, token: str, purpose: str, gate: str) -> str:
    return ('export OWN1_ADMISSION=' + shlex.quote(gate) + '; '
            + a.command('admit', tag, token, [purpose, '20', 'require-serving']))


@pytest.mark.parametrize('kind', ['NBW-WINDOW', 'POST-CLOSE-WATCH'])
@pytest.mark.parametrize('old', [True, False])
def test_foreign_hold_during_drain_old_overwrites_new_refuses(
    sandbox: Sandbox, kind: str, old: bool,
) -> None:
    s = sandbox
    home = prepare(s)
    tag, token = s.tag('admitrace'), uuid.uuid4().hex
    purpose = kind + ' ' + tag + ': fixture'
    script = ('cd ~/dsv41-deploy && test ! -e HOLD && offload/launch/window.sh open '
              + shlex.quote(purpose) + ' && cat HOLD') if old else admit(s, tag, token, purpose, 'new')
    process = s.launch('new', tag, script)
    wait_for(home/('draining-old' if old else 'draining-new'))
    hold = home/'dsv41-deploy/HOLD'
    foreign = '2026-10-27T00:00:00Z OTHER-WINDOW somebody: active HOLD\n'
    hold.write_text(foreign)
    stat = hold.stat()
    (home/'release-drain').touch()
    rc = process.wait(timeout=10)
    if old:
        assert rc == 0 and purpose in hold.read_text() and hold.read_text() != foreign
    else:
        assert rc != 0 and hold.read_text() == foreign
        assert (hold.stat().st_ino, hold.stat().st_mtime_ns) == (stat.st_ino, stat.st_mtime_ns)
        proof = s.run(a.command('check', tag, token))
        assert proof.returncode == 0 and a.NOT_OWNED in proof.stdout
        stop = s.run('export SCOPED_REMOTE_DEADLINE=' + str(time.time() + 10) + '; '
            + a.command('stop', tag, token, ['bash', '-c', 'touch ' + str(home/'stopped')]))
        assert stop.returncode != 0 and not (home/'stopped').exists()
    s.close('new', tag)


@pytest.mark.parametrize('kind', ['NBW-WINDOW', 'POST-CLOSE-WATCH'])
def test_two_simultaneous_admissions_exactly_one_stops_and_loser_cannot_adopt(
    sandbox: Sandbox, kind: str,
) -> None:
    s = sandbox
    home = prepare(s)
    # Same run tag/purpose is deliberately reused: the per-caller nonce still
    # prevents the losing invocation from adopting the winning receipt.
    tag = s.tag('concurrent')
    tokens = [uuid.uuid4().hex, uuid.uuid4().hex]
    processes = [s.launch('new', tag, admit(s, tag, token, kind + ' ' + tag + ': fixture', str(i)))
                 for i, token in enumerate(tokens)]
    for i in range(2):
        wait_for(home/('draining-' + str(i)))
    (home/'release-drain').touch()
    codes = [p.wait(timeout=10) for p in processes]
    assert codes.count(0) == 1
    winner = codes.index(0)
    hold = home/'dsv41-deploy/HOLD'
    original = hold.read_bytes(), hold.stat().st_mtime_ns
    for i, token in enumerate(tokens):
        proof = s.run(a.command('check', tag, token))
        assert proof.returncode == 0
        assert (a.NOT_OWNED not in proof.stdout) == (i == winner)
        command = a.command('stop', tag, token,
            ['bash', '-c', 'echo ' + str(i) + ' >> ' + str(home/'stops')])
        stop_tag = s.tag('firststop')
        p = s.launch('new', stop_tag, command)
        assert (p.wait(timeout=10) == 0) == (i == winner)
        s.close('new', stop_tag)
    assert (home/'stops').read_text().splitlines() == [str(winner)]
    assert (hold.read_bytes(), hold.stat().st_mtime_ns) == original
    s.close('new', tag)


def test_shared_lock_covers_final_check_through_actual_stop(sandbox: Sandbox) -> None:
    s = sandbox
    home = prepare(s)
    (home/'release-drain').touch()
    tag, token = s.tag('lockedstop'), uuid.uuid4().hex
    p = s.launch('new', tag, admit(s, tag, token, 'NBW-WINDOW ' + tag + ': fixture', 'first'))
    assert p.wait(timeout=10) == 0
    body = ('touch ' + str(home/'stopping') + '; while [ ! -e ' + str(home/'release-stop')
            + ' ]; do /usr/bin/sleep .02; done; touch ' + str(home/'stopped'))
    stopping_tag = s.tag('stopping')
    stopping = s.launch('new', stopping_tag, a.command('stop', tag, token, ['bash', '-c', body]))
    wait_for(home/'stopping')
    # Even if another actor removes HOLD, a new compliant admission cannot
    # publish a new owner while the first stop is still running under its lock.
    (home/'dsv41-deploy/HOLD').rename(home/'original-HOLD')
    second_tag, second_token = s.tag('second'), uuid.uuid4().hex
    second = s.launch('new', second_tag, admit(s, second_tag, second_token,
        'POST-CLOSE-WATCH ' + second_tag + ': fixture', 'second'))
    wait_for(home/'draining-second')
    time.sleep(.2)
    assert second.poll() is None and not (home/'dsv41-deploy/HOLD').exists()
    (home/'release-stop').touch()
    assert stopping.wait(timeout=10) == second.wait(timeout=10) == 0
    assert (home/'stopped').exists() and second_tag in (home/'dsv41-deploy/HOLD').read_text()
    for phase in (tag, stopping_tag, second_tag):
        s.close('new', phase)


@pytest.mark.parametrize('change', ['bytes', 'replacement', 'symlink'])
def test_changed_hold_before_first_stop_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('SCOPED_REMOTE_DEADLINE', str(time.time() + 20))
    (tmp_path/'dsv41-deploy').mkdir()
    monkeypatch.setattr(a, 'fetch', lambda _: 'healthy')
    monkeypatch.setattr(a, 'idle', lambda: True)
    tag, token = 'own1test-change', uuid.uuid4().hex
    a.admit(tag, token, 'NBW-WINDOW own1test-change: fixture', 10)
    hold, _, _ = a.paths(tag, token)
    assert a.owned(tag, token)
    content = hold.read_bytes()
    if change == 'bytes':
        hold.write_bytes(content + b'foreign')
    else:
        hold.unlink()
        if change == 'replacement':
            hold.write_bytes(content)
        else:
            other = tmp_path/'foreign'
            other.write_bytes(content)
            hold.symlink_to(other)
    with pytest.raises(ValueError, match='unowned/changed'):
        a.stop(tag, token, ['bash', '-c', 'touch ' + str(tmp_path/'stopped')])
    assert not (tmp_path/'stopped').exists()


@pytest.mark.parametrize('metrics', ['busy-then-idle', 'unreachable', 'malformed'])
def test_failed_models_never_bypasses_watchdog_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metrics: str,
) -> None:
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('SCOPED_REMOTE_DEADLINE', '30')
    (tmp_path/'dsv41-deploy').mkdir()
    clock = Clock()
    monkeypatch.setattr(time, 'time', clock)
    tag, token = 'own1test-watch-drain', uuid.uuid4().hex
    hold, _, _ = a.paths(tag, token)
    calls: list[str] = []
    def fetch(path: str) -> str:
        calls.append(path)
        if path == '/v1/models' or metrics == 'unreachable':
            raise OSError('connection failed')
        if metrics == 'malformed':
            return 'vllm:num_requests_running 0\n'
        return 'vllm:num_requests_running ' + ('4' if clock() < 10 else '0') + '\nvllm:num_requests_waiting 0\n'
    def sleep(amount: float) -> None:
        assert not hold.exists(), 'HOLD must be published after the drain'
        clock.sleep(amount)
    monkeypatch.setattr(time, 'sleep', sleep)
    monkeypatch.setattr(a, 'fetch', fetch)
    if metrics == 'malformed':
        with pytest.raises(ValueError, match='lacks both'):
            a.admit(tag, token, 'POST-CLOSE-WATCH fixture: recovery', 20, allow_unreachable=True)
        assert not hold.exists()
    else:
        a.admit(tag, token, 'POST-CLOSE-WATCH fixture: recovery', 20, allow_unreachable=True)
        assert a.owned(tag, token)
        assert clock() == (10 if metrics == 'busy-then-idle' else 5)
        assert calls.count('/metrics') >= 2


def test_foreign_publication_between_check_and_link_cannot_be_overwritten_or_adopted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('SCOPED_REMOTE_DEADLINE', str(time.time() + 20))
    (tmp_path/'dsv41-deploy').mkdir()
    monkeypatch.setattr(a, 'fetch', lambda _: 'healthy')
    monkeypatch.setattr(a, 'idle', lambda: True)
    tag, token = 'own1test-linkrace', uuid.uuid4().hex
    hold, _, _ = a.paths(tag, token)
    original = os.link
    def race(src: Path, dst: Path) -> None:
        dst.write_text('OTHER-WINDOW somebody: active HOLD\n')
        original(src, dst)
    monkeypatch.setattr(os, 'link', race)
    with pytest.raises(FileExistsError):
        a.admit(tag, token, 'NBW-WINDOW own1test-linkrace: fixture', 10)
    assert hold.read_text() == 'OTHER-WINDOW somebody: active HOLD\n'
    assert not a.owned(tag, token)
