"""WIN2-R3 placeholder root fix, rehearsed under the REAL scoped_remote shims (R-NBW's defect, scratch/RNBW/placeholder-proof.txt).

Controller only, no host contact. Each fake host (sparkhead, sparkworker-wired, x86host) is a private HOME plus a private TMUX_TMPDIR (short,
under /run/user/<uid>, so tmux sockets stay below 108 bytes). The fake ssh transport runs the remote command in a fresh real
user systemd scope with that host's environment, modelling sshd: a plain `ssh host cmd` from the test process (the runner),
and every nested hop that a scope's ssh shim makes (scoped_remote SSH_BINARY points at it). Window scopes are the real
scoped_remote launch/guard/fence/reap/absent/rm/commit-close code, with real tmux 3.6.

What is proved:
- a window's `-hold` scope placeholder lives on that scope's private socket and dies with the fence (unchanged);
- chain placeholders started by `chain.placeholders_up` (unscoped, default socket, owner in the title) outlive every window
  scope, are seen by `chain.chained_close` from inside a `-restore-close` scope, by `chain.placeholders_present` and by
  chain_prev_ok.sh, and come down only in `chain.end_chain`;
- a foreign placeholder (window.sh open's, no owner) or a foreign HOLD is refused and never touched;
- `up` and `down` refuse inside a window scope.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from offload.launch.windows import chain, hold_admission, scoped_remote as owner

REPO = Path(__file__).resolve().parents[4]
WINDOWS = REPO / 'offload/launch/windows'
HOSTS = ('sparkhead', 'sparkworker-wired', 'x86host')
TMUX = '/usr/bin/tmux'
# hold_admission.open_script's placeholder command, as the window's -hold scope runs it on each Spark (checked below).
HOLDPROC = ("tmux has-session -t dsv41-hold 2>/dev/null || tmux new-session -d -s dsv41-hold "
            + shlex.quote('exec -a "vllm serve [DSV41-OFFLOAD hold]" sleep 86400'))
# WIN2-R2's chain.CHAINED_CLOSE (2b48eec), kept to show the rehearsal reproduces R-NBW's defect: inside the -restore-close
# scope its `tmux has-session` reaches each scope's private socket, never the placeholders.
R2_CHAINED_CLOSE = ('set -euo pipefail; cd ~/dsv41-deploy; set -a; . ./dsv41.env; set +a; '
                    'for node in "$HEAD" "$WORKER"; do ssh -o BatchMode=yes "$node" "tmux has-session -t dsv41-hold" '
                    '|| { echo "hold placeholder missing on $node"; exit 1; }; done; '
                    'rm -f "$HOME/dsv41-deploy/HOLD"; echo CHAINED CLOSE RECORDED')


class Rehearsal:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        root.mkdir(parents=True)
        runtime = Path(os.environ.get('XDG_RUNTIME_DIR') or f'/run/user/{os.getuid()}')
        self.sockets = Path(tempfile.mkdtemp(prefix='w2r3-', dir=runtime))
        self.homes = {h: root / h for h in HOSTS}
        self.tmux = {h: self.sockets / h[0] for h in HOSTS}
        self.bin = root / 'bin'
        self.bin.mkdir()
        self.tags: set[str] = set()
        self.path = str(self.bin) + os.pathsep + os.environ['PATH']
        for host in HOSTS:
            self.homes[host].mkdir()
            self.tmux[host].mkdir()
            (root / (host + '.env.json')).write_text(json.dumps(
                {'HOME': str(self.homes[host]), 'TMUX_TMPDIR': str(self.tmux[host]), 'PATH': self.path}))
        self.transport = self.bin / 'ssh'
        self.transport.write_text(
            '#!' + sys.executable + '\n'
            'import json, os, subprocess, sys, uuid\nfrom pathlib import Path\n'
            f'root = Path({str(root)!r})\n'
            'host, command = sys.argv[-2:]\n'
            "host = {'spark': 'sparkhead', 'sparkworker': 'sparkworker-wired'}.get(host, host)\n"
            "assert host in ('sparkhead', 'sparkworker-wired', 'x86host'), host\n"
            "unit = 'w2r3sshd-' + uuid.uuid4().hex\n"
            "with (root / 'transport-units').open('a') as f:\n    f.write(unit + '\\n')\n"
            "env = {k: v for k, v in os.environ.items() if k not in ('TMUX', 'TMUX_PANE', 'SCOPED_REMOTE_DEADLINE')}\n"
            "env.update(json.loads((root / (host + '.env.json')).read_text()))\n"
            "raise SystemExit(subprocess.call(['systemd-run', '--user', '--scope', '--collect', '--quiet', '--unit=' + unit,\n"
            "                                  'bash', '-c', command], env=env))\n")
        self.transport.chmod(0o700)
        original = owner.source()
        patched = original.replace("SSH_BINARY = '/usr/bin/ssh'", 'SSH_BINARY = ' + repr(str(self.transport)))
        assert patched != original
        monkeypatch.setattr(owner, 'source', lambda: patched)
        # The runner's own plain ssh (chain.plain_ssh) resolves `ssh` on PATH: the transport.
        monkeypatch.setenv('PATH', self.path)
        deploy = self.homes['sparkhead'] / 'dsv41-deploy'
        deploy.mkdir()
        (deploy / 'dsv41.env').write_text('HEAD=spark\nWORKER=sparkworker\n')

    def env(self, host: str) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in ('TMUX', 'TMUX_PANE')}
        env.update(json.loads((self.root / (host + '.env.json')).read_text()))
        return env

    def ssh(self, host: str, command: str, timeout: float = 60) -> subprocess.CompletedProcess[str]:
        """The runner's ordinary ssh (unscoped) to `host`."""
        return subprocess.run([str(self.transport), '-o', 'BatchMode=yes', host, command], text=True,
                              capture_output=True, timeout=timeout, env=self.env('sparkhead') | {'PATH': self.path})

    def tag(self, label: str) -> str:
        return 'w2r3test-' + label + '-' + uuid.uuid4().hex[:8]

    def launch(self, tag: str, body: str, *, seconds: float = 90) -> subprocess.CompletedProcess[str]:
        """The runner launches a window phase on sparkhead: ssh sparkhead <scoped launch>."""
        self.tags.add(tag)
        return self.ssh('sparkhead', owner.command('launch', tag, ['bash', '-c', body], deadline=time.time() + seconds),
                        timeout=seconds)

    def control(self, mode: str, tag: str, host: str = 'sparkhead', *,
                deadline: float | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(['bash', '-c', owner.command(mode, tag, deadline=deadline)], env=self.env(host), text=True,
                              capture_output=True, timeout=60)

    def fence(self, tag: str) -> None:
        """owner.fence_all over every fake host: fence everywhere, then reap, then prove absent."""
        for mode, marker in (('fence', owner.FENCED), ('reap', owner.ABSENT), ('absent', owner.ABSENT)):
            for host in HOSTS:
                p = self.control(mode, tag, host)
                assert p.returncode == 0 and marker in p.stdout.splitlines(), (host, mode, p.stdout + p.stderr)

    def private(self, host: str, tag: str) -> int:
        """has-session of dsv41-hold on the scope's private socket (rc 0 = present)."""
        socket = 'dsv41w-' + hashlib.sha256(tag.encode()).hexdigest()[:32]
        return subprocess.run([TMUX, '-L', socket, 'has-session', '-t', '=dsv41-hold'], env=self.env(host),
                              capture_output=True, timeout=10).returncode

    def default(self, host: str, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([TMUX, '-L', 'default', *args], env=self.env(host), text=True, capture_output=True,
                              timeout=10)

    def states(self, action: str, who: str) -> dict[str, str]:
        return chain.placeholder_states(action, who)

    def cleanup(self) -> None:
        errors: list[str] = []
        for tag in self.tags:
            for mode in ('fence', 'reap'):
                for host in HOSTS:
                    p = self.control(mode, tag, host)
                    if p.returncode:
                        errors.append(f'{tag}/{host}/{mode}: {p.stdout}{p.stderr}')
        for host in HOSTS:
            sockets = self.tmux[host] / f'tmux-{os.getuid()}'
            for sock in sockets.glob('*') if sockets.is_dir() else []:
                subprocess.run([TMUX, '-S', str(sock), 'kill-server'], capture_output=True, timeout=10)
        record = self.root / 'transport-units'
        for unit in record.read_text().split() if record.exists() else []:
            assert unit.startswith('w2r3sshd-')
            subprocess.run(['systemctl', '--user', 'kill', '--signal=KILL', '--kill-whom=all', unit + '.scope'],
                           capture_output=True, timeout=10)
        shutil.rmtree(self.sockets, ignore_errors=True)
        if errors:
            raise AssertionError('rehearsal cleanup failed: ' + '\n'.join(errors))


@pytest.fixture
def rehearsal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Rehearsal]:
    assert not tmp_path.resolve().is_relative_to('/tmp'), 'supply an authorized private --basetemp'
    assert Path('/sys/fs/cgroup/cgroup.controllers').is_file() and Path(TMUX).is_file()
    check = subprocess.run(['systemctl', '--user', 'show-environment'], text=True, capture_output=True, timeout=10)
    assert check.returncode == 0, 'real user systemd is required: ' + check.stderr
    marker = tmp_path / 'CHAINED'
    monkeypatch.setattr(chain, 'MARKER', marker)
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    monkeypatch.delenv('WINDOW_CHAIN_LAST', raising=False)
    r = Rehearsal(tmp_path / 'r', monkeypatch)
    try:
        yield r
    finally:
        r.cleanup()


def _hold(r: Rehearsal, tag: str) -> Path:
    hold = r.homes['sparkhead'] / 'dsv41-deploy/HOLD'
    hold.write_text(f'2026-10-27T22:00:00Z DSV41-OFFLOAD (session coordinator): {tag}: rehearsal window\n')
    return hold


def _up(states: dict[str, str], who: str) -> bool:
    return all(s.startswith(f'UP owner={who} ') for s in states.values())


def _gate(r: Rehearsal, tmp_path: Path, prev_summary: dict[str, object]) -> subprocess.CompletedProcess[str]:
    camp = tmp_path / 'campaign'
    (camp / 'launch').mkdir(parents=True, exist_ok=True)
    (camp / 'launch/no-stale-containers.sh').write_text('#!/bin/bash\nexit 0\n')
    (camp / 'launch/no-stale-containers.sh').chmod(0o755)
    abort = r.bin / 'abort.sh'
    abort.write_text(f'#!/bin/sh\necho ABORT-CALLED >> {tmp_path}/abort.log\n')
    abort.chmod(0o755)
    prev = tmp_path / 'W-prev'
    prev.mkdir(exist_ok=True)
    (prev / 'summary.json').write_text(json.dumps(prev_summary))
    (tmp_path / 'prev.txt').write_text(str(prev) + '\n')
    env = {k: v for k, v in os.environ.items() if k not in ('TMUX', 'TMUX_PANE', 'WINDOW_CHAIN_LAST')}
    env.update({'PATH': r.path, 'WINDOW_CHAIN': '1', 'DSV41_CHAIN_MARKER': str(chain.MARKER),
                'DSV41_CHAIN_PRODUCTION_URL': 'http://127.0.0.1:9/v1/models', 'DSV41_CHAIN_PY': sys.executable,
                'DSV41_CHAIN_CAMPAIGN': str(camp), 'DSV41_CHAIN_ABORT': str(abort)})
    return subprocess.run(['bash', str(WINDOWS / 'chain_prev_ok.sh'), 'w2r3-no-such-tmux', str(tmp_path / 'prev.txt')],
                          capture_output=True, text=True, env=env, timeout=180, cwd=REPO)


def test_holdproc_matches_hold_admission() -> None:
    assert shlex.quote(HOLDPROC) in hold_admission.open_script('t', '0' * 32, 'p', 60.0)


def test_chain_placeholders_outlive_window_scopes_and_the_scoped_close_sees_them(rehearsal: Rehearsal,
                                                                                tmp_path: Path) -> None:
    r = rehearsal
    tag = r.tag('w1')
    hold = _hold(r, tag)
    who = chain.chain_owner()                                # no marker yet: a new chain's owner
    # 1. The window's -hold scope makes its placeholders as open_script does: on its PRIVATE socket on both Sparks.
    p = r.launch(tag + '-hold', 'for node in spark sparkworker; do ssh -o BatchMode=yes "$node" ' + shlex.quote(HOLDPROC)
                 + '; done; echo HOLD-SCOPE-PLACEHOLDERS')
    assert p.returncode == 0 and 'HOLD-SCOPE-PLACEHOLDERS' in p.stdout, p.stdout + p.stderr
    assert r.private('sparkhead', tag + '-hold') == 0 and r.private('sparkworker-wired', tag + '-hold') == 0
    assert r.states('probe', who) == {'sparkhead': 'ABSENT', 'sparkworker-wired': 'ABSENT'}   # invisible on the default socket
    # 2. The runner starts this chain's placeholders, unscoped, BEFORE the first fence.
    first = chain.placeholders_up(who)
    assert _up(first, who)
    # 3. Fencing the -hold scope kills its private placeholders; the chain placeholders stay up.
    r.fence(tag + '-hold')
    assert r.private('sparkhead', tag + '-hold') != 0 and r.private('sparkworker-wired', tag + '-hold') != 0
    assert r.states('probe', who) == first
    # R2's close, in a throwaway -restore-close scope, cannot see them (the defect); its rm is never reached.
    old = r.tag('r2close')
    p = r.launch(old + '-restore-close', R2_CHAINED_CLOSE)
    assert p.returncode != 0 and 'hold placeholder missing on spark' in p.stdout and hold.exists(), p.stdout + p.stderr
    r.fence(old + '-restore-close')
    # 4. The shared chained close runs INSIDE the -restore-close scope (scope ssh + tmux shims) and sees them.
    p = r.launch(tag + '-restore-close', chain.chained_close(who))
    assert p.returncode == 0 and chain.CHAINED_CLOSE_OK in p.stdout.splitlines(), p.stdout + p.stderr
    # each Spark's probe ran through the scope's ssh shim, in a nested scope of the same tag on that host
    assert all(list((r.homes[h] / owner.CONTROL / (tag + '-restore-close')).glob('*.cgroup'))
               for h in ('sparkhead', 'sparkworker-wired'))
    control = r.homes['sparkhead'] / owner.CONTROL / (tag + '-restore-close')
    assert hold.exists() and (control / 'CLOSE_HOLD').read_text() == hold.read_text()   # the rm shim only records it
    r.fence(tag + '-restore-close')
    c = r.control('commit-close', tag + '-restore-close', deadline=time.time() + 25)
    assert c.returncode == 0 and owner.CLOSED in c.stdout.splitlines() and not hold.exists(), c.stdout + c.stderr
    # 5. After every window scope is fenced and reaped, the chain placeholders are the same processes.
    assert r.states('probe', who) == first
    chain.finish_chained(tmp_path / 'W1', {'sparkhead': 'QUIET', 'sparkworker-wired': 'QUIET'}, placeholder_owner=who)
    assert chain.marker_owner() == who and chain.placeholders_present() is True
    # 6. The coordinator gate (chain_prev_ok.sh) proves them through its own plain ssh.
    g = _gate(r, tmp_path, {'chain': 'CHAINED', 'restore': {'status': 'CHAINED: production left down'}})
    assert g.returncode == 0 and 'chain gate: chained' in g.stdout and not (tmp_path / 'abort.log').exists(), \
        g.stdout + g.stderr
    # 7. The next chained window keeps the same owner and the same placeholder processes.
    assert chain.chain_owner() == who and chain.placeholders_up(who) == first
    # 8. Chain end (after the strict W8 restore + watch PASS): placeholders down, processes gone, marker cleared.
    pids = [int(s.rsplit('pid=', 1)[1]) for s in first.values()]
    ended = chain.end_chain()
    assert ended['status'] == 'ENDED: chain placeholders down on both Sparks' and chain.read_marker() is None, ended
    assert r.states('probe', who) == {'sparkhead': 'ABSENT', 'sparkworker-wired': 'ABSENT'}
    assert not any(Path('/proc', str(pid)).exists() for pid in pids)


def test_foreign_placeholder_is_refused_and_never_touched(rehearsal: Rehearsal, tmp_path: Path) -> None:
    r = rehearsal
    who = chain.new_owner()
    # window.sh open's placeholder (no owner) on sparkworker's default socket, made by an ordinary ssh
    p = r.ssh('sparkworker-wired', HOLDPROC)
    assert p.returncode == 0, p.stdout + p.stderr
    with pytest.raises(ValueError, match='not up on both Sparks'):
        chain.placeholders_up(who)
    states = r.states('probe', who)
    assert states['sparkhead'].startswith(f'UP owner={who} ') and states['sparkworker-wired'].startswith('FOREIGN owner=none ')
    foreign_pid = int(states['sparkworker-wired'].rsplit('pid=', 1)[1])
    chain.finish_chained(tmp_path / 'W1', {'s': 'QUIET'}, placeholder_owner=who)
    g = _gate(r, tmp_path, {'chain': 'CHAINED', 'restore': {'status': 'CHAINED: production left down'}})
    assert g.returncode == 3 and 'CHAIN STOPPED: PRODUCTION DOWN' in g.stdout, g.stdout + g.stderr
    assert (tmp_path / 'abort.log').read_text().split() == ['ABORT-CALLED']
    ended = chain.end_chain()
    assert str(ended['status']).startswith('ENDED WITH ALERT') and chain.read_marker() is None
    assert ended['placeholders'] == {'sparkhead': 'DOWN', 'sparkworker-wired': f'FOREIGN owner=none pid={foreign_pid}'}
    assert Path('/proc', str(foreign_pid)).exists()          # the foreign placeholder was left alone
    assert r.default('sparkworker-wired', 'has-session', '-t', '=dsv41-hold').returncode == 0


def test_start_and_stop_refuse_inside_a_window_scope(rehearsal: Rehearsal) -> None:
    r = rehearsal
    who = chain.new_owner()
    tag = r.tag('scoped')
    p = r.launch(tag + '-hold', chain.placeholder_script('up', who))
    assert 'PLACEHOLDER REFUSED: up inside a scoped_remote window scope' in p.stdout and p.returncode != 0, \
        p.stdout + p.stderr
    assert r.states('probe', who)['sparkhead'] == 'ABSENT'
    chain.placeholders_up(who)
    p = r.launch(tag + '-restore-close', chain.placeholder_script('down', who))
    assert 'PLACEHOLDER REFUSED: down inside a scoped_remote window scope' in p.stdout and p.returncode != 0
    assert _up(r.states('probe', who), who)
    r.fence(tag + '-hold')
    r.fence(tag + '-restore-close')
    assert _up(r.states('probe', who), who)
    assert all(s == 'DOWN' for s in chain.placeholders_down(who).values())


def test_scoped_close_refuses_a_missing_placeholder_and_a_foreign_hold(rehearsal: Rehearsal) -> None:
    r = rehearsal
    who = chain.new_owner()
    tag = r.tag('missing')
    hold = _hold(r, tag)
    assert r.ssh('sparkhead', chain.placeholder_script('up', who)).stdout.startswith(f'PLACEHOLDER UP owner={who} ')
    p = r.launch(tag + '-restore-close', chain.chained_close(who))                 # sparkworker has none
    assert p.returncode != 0 and f'hold placeholder of {who} missing on sparkworker: PLACEHOLDER ABSENT' in p.stdout
    control = r.homes['sparkhead'] / owner.CONTROL / (tag + '-restore-close')
    assert not (control / 'CLOSE_HOLD').exists() and hold.exists()
    r.fence(tag + '-restore-close')
    chain.placeholders_up(who)
    hold.write_text(f'2026-10-27T22:00:00Z DSV41-OFFLOAD (session coordinator): {r.tag("other")}: another window\n')
    tag3 = r.tag('foreignhold')
    p = r.launch(tag3 + '-restore-close', chain.chained_close(who))
    assert p.returncode != 0 and 'HOLD belongs to another window' in p.stdout + p.stderr and hold.exists(), \
        p.stdout + p.stderr
    assert chain.CHAINED_CLOSE_OK not in p.stdout
    assert all(s == 'DOWN' for s in chain.placeholders_down(who).values())
