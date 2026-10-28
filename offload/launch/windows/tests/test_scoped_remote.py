"""Real Controller cgroup probes for the remote ownership failures S47--S53.

Only private CPU files change. Fake SSH uses a separate real user scope to
model sshd's independent receiving-host process tree; setsid alone would not
faithfully model that boundary. There is deliberately no systemd fake/skip.
Supply private TMPDIR/TMUX_TMPDIR and pytest --basetemp outside /tmp.
"""
from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from offload.launch.windows import kl1_remote, scoped_remote as owner

REPO = Path(__file__).resolve().parents[4]
OPS1 = '596ba1cf4e3b11be29b41a57e7f69a5d8efa4799'
TV1 = '63cc9a1f0ad931c6064f41d141feb0d84e58398a'


def wait_for(path: Path, seconds: float = 10) -> None:
    until = time.monotonic() + seconds
    while not path.exists():
        if time.monotonic() >= until:
            raise AssertionError('timed out waiting for fixture marker: ' + str(path))
        time.sleep(.02)


def frozen(commit: str, path: str) -> str:
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=REPO, text=True)


class Sandbox:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self.root.mkdir(parents=True)
        self.homes = {host: root/host for host in ('sparkhead', 'sparkworker-wired', 'x86host')}
        self.tags: set[str] = set()
        self.processes: list[subprocess.Popen[str]] = []
        self.groups: set[int] = set()
        self.deadlines: dict[str, float] = {}
        self.simple_transport = False
        self.tmux_tmp = Path(os.environ['TMUX_TMPDIR'])
        self.binary = root/'bin'
        self.binary.mkdir()
        for home in self.homes.values():
            home.mkdir()
        transport = self.binary/'ssh'
        # The outer transport scope is independent of the sender, just like
        # sshd. All nested destination scopes still use the real user manager.
        transport.write_text('#!' + sys.executable + '\n' +
            'import os,subprocess,sys,uuid\nfrom pathlib import Path\n' +
            f'root=Path({str(root)!r})\n' +
            'host,command=sys.argv[-2:]\n' +
            'assert host in ("sparkhead","sparkworker-wired","x86host","spark","sparkworker")\n' +
            'host={"spark":"sparkhead","sparkworker":"sparkworker-wired"}.get(host,host)\n' +
            'unit="dsv41w-own1test-transport-"+uuid.uuid4().hex\n' +
            'with (root/"transport-units").open("a") as f:f.write(unit+"\\n")\n' +
            'env={**os.environ,"HOME":str(root/host)}\n' +
            'if os.environ.get("OWN1_TEST_READONLY_HOPS")=="1":\n' +
            ' raise SystemExit(subprocess.Popen(["bash","-c",command],env=env,start_new_session=True).wait())\n' +
            'gate=root/(host+"-transport-gate")\n' +
            'body="cat /proc/self/cgroup > "+str(root/(unit+".cgroup"))+"; touch "+str(root/(host+"-accepted"))+"; "\n' +
            'body+="while [ -e "+str(gate)+" ]; do sleep .02; done; exec bash -c "+__import__("shlex").quote(command)\n' +
            'raise SystemExit(subprocess.call(["systemd-run","--user","--scope","--collect","--quiet",' +
            '"--unit="+unit,"--property=RuntimeMaxSec=30s","setsid","--wait","bash","-c",body],env=env))\n')
        transport.chmod(0o700)
        docker = self.binary/'docker'
        docker.write_text('#!/bin/bash\nexit 0\n')
        docker.chmod(0o700)
        original = owner.source()
        self.source = original.replace("SSH_BINARY = '/usr/bin/ssh'", 'SSH_BINARY = ' + repr(str(transport)))
        assert self.source != original
        monkeypatch.setattr(owner, 'source', lambda: self.source)
        self.tv1_source = frozen(TV1, 'offload/launch/windows/tv1_remote.py').replace(
            "SSH_BINARY = '/usr/bin/ssh'", 'SSH_BINARY = ' + repr(str(transport)))

    def env(self, host: str = 'sparkhead') -> dict[str, str]:
        return {**os.environ, 'HOME': str(self.homes[host]),
                'OWN1_TEST_SANDBOX': str(self.root),
                'OWN1_TEST_READONLY_HOPS': '1' if self.simple_transport else '0',
                'TMUX_TMPDIR': str(self.tmux_tmp),
                'PATH': str(self.binary) + os.pathsep + os.environ['PATH']}

    def tag(self, label: str, seconds: float = 25) -> str:
        tag = 'own1test-' + label + '-' + uuid.uuid4().hex[:8]
        self.tags.add(tag)
        self.deadlines[tag] = time.time() + seconds
        return tag

    def command(self, mechanism: str, mode: str, tag: str, argv: list[str] | None = None) -> str:
        if mechanism == 'new':
            return owner.command(mode, tag, argv, deadline=self.deadlines[tag] if mode == 'launch' else None)
        if mechanism == 'kl1':
            return kl1_remote.command(mode, tag, argv)
        if mechanism == 'tv1':
            bootstrap = 'SOURCE = ' + repr(self.tv1_source) + '\nexec(compile(SOURCE,"<old-tv1>","exec"))'
            prefix = ['env', 'TV1_OWNER_DEADLINE=' + str(self.deadlines[tag])]
            if mode == 'launch':
                prefix += ['setsid', '--wait']
            return shlex.join([*prefix, 'python3', '-c', bootstrap, mode, tag,
                               *([uuid.uuid4().hex] if mode == 'launch' else []), *(argv or [])])
        if mode != 'launch' or argv is None:
            return 'true'
        return shlex.join(argv)

    def run(self, command: str, host: str = 'sparkhead') -> subprocess.CompletedProcess[str]:
        return subprocess.run(['bash', '-c', command], env=self.env(host), text=True,
                              capture_output=True, timeout=25)

    def launch(self, mechanism: str, tag: str, body: str, host: str = 'sparkhead') -> subprocess.Popen[str]:
        log = self.root/(uuid.uuid4().hex + '.log')
        with log.open('w') as handle:
            p = subprocess.Popen(['bash', '-c', self.command(mechanism, 'launch', tag, ['bash', '-c', body])],
                                 env=self.env(host), stdout=handle, stderr=handle, text=True,
                                 start_new_session=True)
        self.processes.append(p)
        self.groups.add(p.pid)
        return p

    def close(self, mechanism: str, tag: str) -> None:
        # ALL destinations fence first. This remains true even when a fixture
        # launch has not reached its destination registration yet.
        for mode in ('fence', 'reap', 'absent'):
            if mechanism in ('kl1', 'tv1') and mode == 'absent':
                continue
            for host in self.homes:
                p = self.run(self.command(mechanism, mode, tag), host)
                assert p.returncode == 0, p.stdout + p.stderr
                if mechanism == 'new':
                    assert (owner.FENCED if mode == 'fence' else owner.ABSENT) in p.stdout.splitlines()

    def cleanup(self) -> None:
        errors: list[str] = []
        for tag in self.tags:
            for mode in ('fence', 'reap'):
                for host in self.homes:
                    p = self.run(owner.command(mode, tag), host)
                    if p.returncode:
                        errors.append(p.stdout + p.stderr)
        # Explicitly recorded fake-transport units are ours, even if the old
        # ownership mechanism deliberately missed their descendants.
        record = self.root/'transport-units'
        units = record.read_text().splitlines() if record.exists() else []
        for unit in units:
            assert unit.startswith('dsv41w-own1test-transport-')
            subprocess.run(['systemctl', '--user', 'kill', '--signal=KILL', '--kill-whom=all', unit],
                           text=True, capture_output=True, timeout=10)
        until = time.monotonic() + 5
        while units:
            listing = subprocess.run(['systemctl', '--user', 'list-units', '--all', '--plain',
                                      '--no-legend', '--no-pager', 'dsv41w-own1test-transport-*.scope'],
                                     check=True, text=True, capture_output=True, timeout=10)
            listed = {line.split()[0] for line in listing.stdout.splitlines()}
            retained = False
            for unit in units:
                record = self.root/(unit + '.cgroup')
                if record.exists():
                    value = record.read_text().strip().removeprefix('0::')
                    assert Path(value).name == unit + '.scope'
                    assert value.startswith(f'/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/')
                    retained |= owner.populated(Path('/sys/fs/cgroup')/value.lstrip('/'))
                retained |= unit + '.scope' in listed
            if not retained:
                break
            if time.monotonic() >= until:
                errors.append('owned fake SSH transport scope/cgroup remained after cleanup')
                break
            time.sleep(.02)
        for pidfile in self.root.glob('**/child.pid'):
            pid = int(pidfile.read_text())
            try:
                if Path('/proc', str(pid)).exists():
                    # The private PID was emitted by our exact JIT shell.
                    self.groups.add(os.getpgid(pid))
            except ProcessLookupError:
                pass
        for record in self.root.glob('*/dsv41-offload-run/*/*/*.pid'):
            assert record.parent.name in self.tags
            # Old launchers can fork away from Popen's group before the test
            # fails. Their exact private records make teardown cover that case.
            self.groups.add(int(record.read_text().splitlines()[0]))
        for group in self.groups:
            if group <= 1 or group == os.getpgrp():
                raise AssertionError('unsafe fixture process group')
            owned = False
            for proc in Path('/proc').iterdir():
                if not proc.name.isdigit():
                    continue
                try:
                    stat = (proc/'stat').read_text().rsplit(')', 1)[1].split()
                    if int(stat[2]) != group or stat[0] in ('Z', 'X'):
                        continue
                    token = ('OWN1_TEST_SANDBOX=' + str(self.root)).encode()
                    assert token in (proc/'environ').read_bytes().split(b'\0'), 'foreign fixture group member'
                    owned = True
                except (FileNotFoundError, ProcessLookupError):
                    continue
            if owned:
                try:
                    os.killpg(group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for process in self.processes:
            process.wait(timeout=10)
        if errors:
            raise AssertionError('scope cleanup failed: ' + '\n'.join(errors))


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Sandbox]:
    assert not tmp_path.resolve().is_relative_to('/tmp'), 'supply an authorized private --basetemp'
    assert 'TMUX' not in os.environ and 'TMUX_PANE' not in os.environ
    for name in ('TMPDIR', 'TMUX_TMPDIR'):
        private = Path(os.environ[name])  # no implicit system-temporary fallback
        assert private.is_absolute() and private.is_dir() and not private.resolve().is_relative_to('/tmp')
    root = tmp_path/'node'
    s = Sandbox(root, monkeypatch)
    try:
        # Failure is explicit on a host without real user cgroup-v2 support.
        assert Path('/sys/fs/cgroup/cgroup.controllers').is_file()
        check = subprocess.run(['systemctl', '--user', 'show-environment'], text=True,
                               capture_output=True, timeout=10)
        assert check.returncode == 0, 'real user systemd is required: ' + check.stderr
        yield s
    finally:
        s.cleanup()


@pytest.mark.parametrize('mechanism', ['bare', 'kl1', 'new'])
def test_s47_late_registration_and_lost_reply_retry(sandbox: Sandbox, mechanism: str) -> None:
    s = sandbox
    tag = s.tag('s47')
    state, release, pending = (s.root/name for name in ('production', 'release', 'pending'))
    state.write_text('restored')
    # Transport accepted the request, but it has not entered its owner yet.
    inner = s.command(mechanism, 'launch', tag, ['bash', '-c', 'echo stopped > ' + str(state)])
    p = s.launch('bare', tag, 'touch ' + str(pending) + '; while [ ! -e ' + str(release) +
                 ' ]; do sleep .02; done; ' + inner)
    wait_for(pending)
    s.close(mechanism, tag)
    release.touch()
    p.wait(timeout=10)
    assert state.read_text().strip() == ('stopped' if mechanism == 'bare' else 'restored')
    retry = s.run(inner)
    assert (retry.returncode == 0) == (mechanism == 'bare')


@pytest.mark.parametrize('mechanism', ['bare', 'new'])
@pytest.mark.parametrize('timing', ['before-registration', 'active'])
def test_s49_pair_open_delayed_or_active(sandbox: Sandbox, mechanism: str, timing: str) -> None:
    s = sandbox
    tag = s.tag('s49')
    state, pair, release, pending = (s.root/name for name in ('production', 'PAIR_OK', 'release', 'pending'))
    state.write_text('active')
    body = ('touch ' + str(pending) + '; while [ ! -e ' + str(release) +
            ' ]; do sleep .02; done; echo inactive > ' + str(state) + '; touch ' + str(pair))
    if timing == 'before-registration':
        inner = s.command(mechanism, 'launch', tag, ['bash', '-c', 'echo inactive > ' + str(state) + '; touch ' + str(pair)])
        p = s.launch('bare', tag, 'touch ' + str(pending) + '; while [ ! -e ' + str(release) +
                     ' ]; do sleep .02; done; ' + inner, 'x86host')
    else:
        p = s.launch(mechanism, tag, body, 'x86host')
    wait_for(pending)
    s.close(mechanism, tag)
    state.write_text('active')  # file-only pair restoration checkpoint
    release.touch()
    p.wait(timeout=10)
    assert state.read_text().strip() == ('inactive' if mechanism == 'bare' else 'active')
    assert pair.exists() == (mechanism == 'bare')
    retry = s.run(s.command(mechanism, 'launch', tag, ['touch', str(pair)]), 'x86host')
    assert (retry.returncode == 0) == (mechanism == 'bare')


@pytest.mark.parametrize('mechanism', ['kl1', 'new'])
@pytest.mark.parametrize('timing', ['before-registration', 'active'])
@pytest.mark.parametrize('mutation', ['stop', 'run'])
def test_s51_s52_second_hop_receiving_host_ownership(
        sandbox: Sandbox, mechanism: str, timing: str, mutation: str) -> None:
    s = sandbox
    tag = s.tag('s51-' + mutation)
    state, release, pending = (s.root/name for name in ('production', 'release', 'pending'))
    state.write_text('restored')
    payload = 'echo ' + mutation + ' > ' + str(state)
    if timing == 'before-registration':
        gate = s.root/'sparkworker-wired-transport-gate'
        gate.touch()
        remote = payload
        pending = s.root/'sparkworker-wired-accepted'
    else:
        gate = release
        remote = ('touch ' + str(pending) + '; while [ ! -e ' + str(release) +
                  ' ]; do sleep .02; done; ' + payload)
    p = s.launch(mechanism, tag, 'ssh -o BatchMode=yes sparkworker-wired ' + shlex.quote(remote))
    wait_for(pending)
    s.close(mechanism, tag)
    state.write_text('restored')
    if timing == 'before-registration':
        gate.unlink()
    else:
        release.touch()
    # The first-hop reply may be lost/killed; destination marker is decisive.
    time.sleep(.6)
    assert state.read_text().strip() == (mutation if mechanism == 'kl1' else 'restored')
    retry = s.run(s.command(mechanism, 'launch', tag, ['bash', '-c', payload]), 'sparkworker-wired')
    assert retry.returncode != 0  # old fences new requests but misses the accepted unowned hop
    if p.poll() is None and mechanism == 'new':
        p.wait(timeout=10)


def install_ops1(s: Sandbox, topology: str) -> tuple[Path, Path, Path, Path]:
    # OPS1's only SSH calls in this fixture are completed read-only docker ps
    # checks. Match S53's transport exactly; the direct JIT stays local.
    s.simple_transport = True
    prod = s.homes['sparkhead']/'dsv41-deploy'
    lane = prod/'offload/promote/k1'
    lane.mkdir(parents=True)
    (lane/'start_checked.sh').write_text(frozen(OPS1, 'offload/promote/k1/start_checked.sh'))
    (prod/'dsv41.env').write_text('HEAD=spark\nWORKER=sparkworker\nPROD_K1_PROFILE=327680\n')
    (lane/'manage.py').write_text('import sys\n' +
        ('raise SystemExit(1 if "verify" in sys.argv else 0)\n' if topology != 'start' else
         'print("fixture verify")\n'))
    (lane/'stage.sh').write_text('#!/bin/bash\nexit 0\n')
    (prod/'stop.sh').write_text('#!/bin/bash\nexit 0\n')
    pending, release, state, pidfile = (s.root/name for name in ('pending', 'release', 'production', 'child.pid'))
    state.write_text('restored')
    jit = ('echo $$ > ' + str(pidfile) + '; cat /proc/self/cgroup > ' + str(s.root/'jit.cgroup') +
           '; touch ' + str(pending) + '; while [ ! -e ' + str(release) +
           ' ]; do sleep .02; done; echo late-local-JIT > ' + str(state))
    jit_script = '#!/bin/bash\nJIT=' + shlex.quote(jit) + '\nbash -c "$JIT" & JIT_HEAD=$!\nwait $JIT_HEAD\n'
    (prod/'start.sh').write_text(jit_script if topology != 'fallback-stop' else '#!/bin/bash\nexit 0\n')
    if topology == 'fallback-stop':
        (prod/'stop.sh').write_text(jit_script)
    return lane/'start_checked.sh', pending, release, state


@pytest.mark.parametrize('mechanism', ['tv1', 'new'])
@pytest.mark.parametrize('finish', ['reap', 'deadline'])
@pytest.mark.parametrize('topology', ['start', 'fallback-stop', 'fallback-start'])
def test_s53_frozen_ops1_nested_timeout_direct_head_jit(
        sandbox: Sandbox, mechanism: str, finish: str, topology: str) -> None:
    s = sandbox
    script, pending, release, state = install_ops1(s, topology)
    tag = s.tag('s53', seconds=2 if finish == 'deadline' else 25)
    if mechanism == 'tv1' and finish == 'reap':
        # Isolate the unchanged old PGID reaper from its separate transient
        # guardian/environ race on short completed read-only SSH sessions.
        # Both old/new deadline controls below retain finite server ceilings.
        s.deadlines[tag] = float('inf')
    p = s.launch(mechanism, tag, 'bash ' + str(script))
    wait_for(pending)
    jit_pid = int((s.root/'child.pid').read_text())
    # Frozen OPS1 timeout creates a new PGID, defeating old PGID-only members().
    assert os.getpgid(jit_pid) != p.pid
    if mechanism == 'tv1':
        registered = s.homes['sparkhead']/'dsv41-offload-run/tv1-control'/tag
        groups = {int(record.read_text().splitlines()[0]) for record in registered.glob('*.pid')}
        assert groups and os.getpgid(jit_pid) not in groups
    if mechanism == 'new':
        assert 'dsv41w-' + tag + '-' in (s.root/'jit.cgroup').read_text()
    if finish == 'reap':
        s.close(mechanism, tag)
    else:
        # Old TV1's guardian waits five more seconds after TERM; the escaped
        # timeout group still survives that full escalation interval.
        time.sleep(max(0, s.deadlines[tag] - time.time()) + 5.3)
        if mechanism == 'new':
            s.close(mechanism, tag)
    release.touch()
    time.sleep(.3)
    assert state.read_text().strip() == ('late-local-JIT' if mechanism == 'tv1' else 'restored')


def test_reap_requires_fence_and_finite_immutable_deadline(sandbox: Sandbox) -> None:
    s = sandbox
    tag = s.tag('contracts')
    for invalid in (float('inf'), float('nan')):
        with pytest.raises(ValueError, match='finite'):
            owner.command('launch', tag, ['true'], deadline=invalid)
    refused = s.run(owner.command('reap', tag))
    assert refused.returncode != 0 and 'persistent fence' in refused.stderr
    p = s.launch('new', tag, 'true')
    assert p.wait(timeout=10) == 0
    changed = s.run(owner.command('launch', tag, ['true'], deadline=s.deadlines[tag] + 1))
    assert changed.returncode != 0 and 'immutable' in changed.stderr
    s.close('new', tag)


def test_fence_all_orders_destinations_and_rejects_missing_receipt(sandbox: Sandbox) -> None:
    s = sandbox
    tag = s.tag('all')
    calls: list[tuple[str, str]] = []

    def invoke(host: str, command: str) -> tuple[int, str]:
        p = s.run(command, host)
        calls.append((host, p.stdout.strip()))
        return p.returncode, p.stdout + p.stderr

    owner.fence_all(tag, invoke)
    assert [host for host, _ in calls[:3]] == list(owner.NODES)
    assert all(text == owner.FENCED for _, text in calls[:3])
    assert all(text == owner.ABSENT for _, text in calls[3:])
    with pytest.raises(ValueError, match='HOLD stays open'):
        owner.fence_all(tag, lambda host, command: (0, 'lost response'))


def test_real_tmux_placeholder_daemon_stays_in_owned_scope(sandbox: Sandbox) -> None:
    s = sandbox
    # Caller-supplied private TMUX_TMPDIR must keep the socket below 108 bytes.
    assert len(str(s.tmux_tmp/f'tmux-{os.getuid()}'/('dsv41w-' + '0'*32)).encode()) < 108
    tag = s.tag('tm')
    pending, release, state = (s.root/name for name in ('pending', 'release', 'production'))
    state.write_text('restored')
    child = ('echo $$ > ' + str(s.root/'child.pid') + '; cat /proc/self/cgroup > ' +
             str(s.root/'placeholder.cgroup') + '; touch ' + str(s.root/'placeholder-ready') +
             '; while [ ! -e ' + str(release) +
             ' ]; do sleep .02; done; echo late-placeholder > ' + str(state))
    script = s.root/'placeholder.sh'
    script.write_text('#!/bin/bash\n' + child + '\n')
    body = ('tmux has-session -t dsv41-hold 2>/dev/null || tmux new-session -d -s dsv41-hold ' +
            shlex.quote('bash ' + str(script)) +
            '; server=$(tmux display-message -p -t dsv41-hold "#{pid}"); ' +
            'cat /proc/$server/cgroup > ' + str(s.root/'server.cgroup') +
            '; touch ' + str(pending) + '; sleep 20')
    s.launch('new', tag, body)
    wait_for(pending)
    wait_for(s.root/'placeholder-ready')
    assert (s.root/'server.cgroup').read_text() == (s.root/'placeholder.cgroup').read_text()
    assert 'dsv41w-' + tag + '-' in (s.root/'server.cgroup').read_text()
    s.close('new', tag)
    release.touch()
    time.sleep(.1)
    assert state.read_text() == 'restored'
    assert all(not log.read_text() for log in s.root.glob('*.log'))


@pytest.mark.parametrize('submission_delay', [.6, 2.0])
def test_absolute_deadline_survives_slow_scope_submission(sandbox: Sandbox, submission_delay: float) -> None:
    s = sandbox
    tag = s.tag('slow', seconds=1.3)
    delayed = s.binary/'systemd-run'
    delayed.write_text('#!/bin/bash\nsleep ' + str(submission_delay) + '\nexec /usr/bin/systemd-run "$@"\n')
    delayed.chmod(0o700)
    state, release, pending = (s.root/name for name in ('production', 'release', 'pending'))
    state.write_text('restored')
    body = ('touch ' + str(pending) + '; while [ ! -e ' + str(release) +
            ' ]; do sleep .02; done; echo late > ' + str(state))
    process = s.launch('new', tag, body)
    if submission_delay < 1.3:
        wait_for(pending)
        # This is earlier than the old RuntimeMaxSec=1.3 submitted before the
        # .6-second delay could expire. Only the absolute guardian stops it.
        time.sleep(max(0, s.deadlines[tag] + .2 - time.time()))
        release.touch()
        process.wait(timeout=10)
    else:
        assert process.wait(timeout=10) != 0
        assert not pending.exists()  # in-scope expiry refuses before mutation
    assert state.read_text() == 'restored'
    s.close('new', tag)
