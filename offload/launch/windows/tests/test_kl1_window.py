"""KL1 CPU rehearsal: real adapter/supervisor, fake command layer, owned local monitor processes only."""
from __future__ import annotations

import hashlib
import json
import os
import re
import ast
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from offload.launch.windows import kl1_window as k
from offload.launch.windows import kl1_remote as kr
from offload.launch.windows import pf1_window as pf
from offload.launch.windows.tests import test_pf1_window as base
from offload.prefill_maxlen.evidence import Json, record


class Remote(base.FakeRemote):
    def __init__(self, out: Path, sup: pf.Supervisor, clock: base.Clock, fail: str, trip: Path) -> None:
        super().__init__(out, sup, clock, fail, trip=trip)
        self.files = {h: {'prod-first.json': b'{"variant":"prod"}', 'prod-last.json': b'{"variant":"prod"}',
                          'failures.json': b'[]'} for h in pf.HOSTS}
        self.events: list[str] = []
        self.stopped = False

    def call(self, label: str, argv: list[str], *, stdin: str | None = None, timeout: float = 120, inflight: int = 0) -> str:
        if label == 'kl1-bench':
            self.sup.poll()
            self.commands.append(label)
            self.events.append('bench')
            if self.fail in ('start', 'crash'):
                return self.sup.run(label, [sys.executable, '-c', 'raise SystemExit(7)'], timeout=1)
            if self.fail == 'hang':
                return self.sup.run(label, [sys.executable, '-c', 'import time; time.sleep(30)'], timeout=.15)
            if self.fail == 'signal':
                raise KeyboardInterrupt('signal 15')
            if self.fail == 'deadline':
                self.clock.now += k.ABORT_S
            if self.fail == 'STOP':
                (self.out / 'STOP').touch()
            if self.fail == 'memory':
                assert self.trip is not None
                self.trip.touch()
            end = time.monotonic() + .15
            while time.monotonic() < end:
                self.sup.poll()
            if self.fail == 'variant':
                self.files['sparkhead']['failures.json'] = b'[{"step":"pipe4","rc":1}]'
            return ''
        if label.startswith('kl1-result-copy-'):
            host, name = argv[-2].split(':')[0], argv[-2].rsplit('/', 1)[1]
            (Path(argv[-1]) / name).write_bytes(self.files[host][name] + (b'x' if self.fail == 'copy' else b''))
            return ''
        return super().call(label, argv, stdin=stdin, timeout=timeout, inflight=inflight)

    def remote(self, label: str, host: str, command: str, stdin: str | None) -> str:
        if 'window.sh open' in command:
            assert 'test ! -e HOLD && timeout' in command
            self.events.append('hold')
            self.hold = f'{base.iso(self.clock())} DSV41-OFFLOAD (session coordinator): {k.KL1Live.hold_purpose(live_stub())}'
            if self.fail == 'hold-open':
                raise ValueError('lost hold-open reply')
            return 'window open'
        if label == 'ledger' and self.fail == 'existing-HOLD':
            raise ValueError('existing HOLD')
        if label.startswith('kl1-dir-'):
            return '/dev/nvme0n1 100000000 1000000 90000000 1% /home\n'
        if label.startswith('kl1-plan-'):
            return json.dumps({k.PLAN: ('b' if self.fail == 'plan' and host == pf.HOSTS[1] else 'a') * 64})
        if label.startswith(('kl1-host-', 'kl1-memory-')):
            mem = (39 if self.fail == 'low-before-bench' else 60) * 1024**2
            return f'{k.KERNEL}\n{k.DRIVER}\nMemAvailable: {mem} kB\n'
        if label == 'prod-identity-sparkhead' and self.fail == 'snapshot':
            raise ValueError('snapshot failed')
        if label == 'kl1-stop':
            self.stopped = True
            self.events.append('stop')
            if self.fail == 'stop':
                raise ValueError('stop failed')
            return ''
        if label.startswith('kl1-idle-'):
            return '1234\n' if self.fail == 'gpu-busy' else ''
        if label.startswith('kl1-mkdir-'):
            if self.fail == 'stage':
                raise ValueError('stage failed')
            self.events.append('stage-' + host)
            return ''
        if label.startswith('kl1-source-hash-'):
            return '\n'.join(f'{pf.digest(pf.ROOT / p)}  {p}' for p in k.SOURCES)
        if label.startswith('kl1-results-') and not label.startswith('kl1-results-remove-'):
            return ''.join(f'f {len(b)} {n}\n' for n, b in self.files[host].items())
        if label.startswith('kl1-result-hashes-'):
            return json.dumps({n: hashlib.sha256(b).hexdigest() for n, b in self.files[host].items()})
        if label.startswith('kl1-results-remove-'):
            self.files[host] = {}
            return ''
        return super().remote(label, host, command, stdin)

    def plain(self, step: str, command: list[str], timeout: float) -> tuple[int, str]:
        assert all(signal.getsignal(s) == signal.SIG_IGN for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP))
        self.events.append(step)
        if step.startswith('kl1-fence-'):
            assert 'ABORT' in command[-1] and self.events[-1] == step
            return (1, 'fence failed') if self.fail == 'fence' else (0, kr.FENCED + '\n')
        if step.startswith('kl1-rm-'):
            assert 'reap' in shlex.split(command[-1])
            assert f'docker rm -f {k.CONTAINER}' in command[-1]
            assert k.IDLE in command[-1] or not self.stopped
            marker = 'KL1 CLEAN GPU IDLE' if self.stopped else 'KL1 CLEAN BENCH ABSENT'
            if self.fail == 'launcher-live':
                return 0, marker + '\n'  # Container removal alone is insufficient.
            return (1, 'GPU busy') if self.fail == 'cleanup' else (0, kr.ABSENT + '\n' + marker + '\n')
        if step == 'reap':
            assert 'start_exp' not in command[-1] and 'KL1-WINDOW tag:' in command[-1]
            return 0, 'reap done\n'
        if step.startswith('kl1-dir-clean-'):
            return 0, ''
        if step in base.RESTORE_OK and self.fail == 'restore-strings':
            self.restores.append(step)
            return 0, 'invented success'
        if step == 'close' and self.fail == 'restore-deadline':
            self.clock.now += k.RESTORE_S + 1
        if step in base.RESTORE_OK:
            self.restores.append(step)
            return 0, base.RESTORE_OK[step]
        return super().plain(step, command, timeout)


def live_stub() -> k.KL1Live:
    live = object.__new__(k.KL1Live)
    live.tag = 'tag'
    return live


@pytest.fixture
def rehearsal(tmp_path: Path, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[k.KL1Live, Remote]]:
    fail = str(getattr(request, 'param', ''))
    monitor = tmp_path / 'monitor.py'
    monitor.write_text(base.MONITOR)
    out = tmp_path / 'KL1-W-fake'
    out.mkdir()
    clock = base.Clock()
    trip = tmp_path / 'trip'
    sup = base.NoTunnel(out, clock=clock, monitor=lambda host: [sys.executable, str(monitor), 'low', str(trip)], demand=lambda: 0.0)
    fake = Remote(out, sup, clock, fail, trip)
    live = k.KL1Live(out, sup, remote='dsv41-deploy-offload', expected='0' * 40, data_dir=tmp_path, tag='tag',
                     stage_bundle=False, call=fake.call, plain=fake.plain, clock=clock)
    import offload.prefill_trace.window_requests as wr
    monkeypatch.setattr(wr, 'corpus', lambda root: ('x', {}))
    try:
        yield live, fake
    finally:
        sup.close()


def assert_restored(result: dict[str, Json], fake: Remote) -> None:
    assert fake.restores == ['start', 'check', 'close']
    assert str(record(result['restore'])['status']).startswith('RESTORED')
    assert fake.events.index('kl1-fence-sparkhead') < fake.events.index('kl1-fence-sparkworker-wired') < fake.events.index('kl1-rm-sparkhead')
    assert fake.events.index('kl1-rm-sparkhead') < fake.events.index('kl1-rm-sparkworker-wired') < fake.events.index('start')
    assert record(result['restore'])['restore_deadline_met'] is True


def test_full_sequence(rehearsal: tuple[k.KL1Live, Remote]) -> None:
    live, fake = rehearsal
    result = k.window(live, lambda row: None)
    assert 'stop' not in result, result
    assert result['status'] == 'COMPLETE'
    assert_restored(result, fake)
    assert fake.events[:5] == ['hold', 'stop', 'stage-sparkhead', 'stage-sparkworker-wired', 'bench']
    assert all(not f for f in fake.files.values())
    assert len(json.loads((live.out / 'results-manifest.json').read_text())) == 6
    deadlines = json.loads((live.out / 'deadlines.json').read_text())
    assert deadlines['abort'] - deadlines['opened'] == 40 * 60
    assert deadlines['restore_by'] - deadlines['opened'] == 55 * 60


@pytest.mark.parametrize('rehearsal,message', [('start', 'rc7'), ('crash', 'rc7'), ('hang', 'timeout'),
    ('memory', 'below 4 GiB'), ('deadline', '40-minute KL1'), ('signal', 'signal 15'), ('STOP', 'STOP file'),
    ('stop', 'stop failed'), ('stage', 'stage failed'), ('gpu-busy', 'compute apps'),
    ('low-before-bench', '40 GiB'), ('copy', 'copy differs'), ('variant', 'reported failures')], indirect=['rehearsal'])
def test_failures_restore(rehearsal: tuple[k.KL1Live, Remote], message: str) -> None:
    live, fake = rehearsal
    result = k.window(live, lambda row: None)
    assert message in str(result.get('stop')), result
    assert_restored(result, fake)
    if fake.fail == 'copy':
        assert all(fake.files.values()), 'unverified results must remain on the node'


@pytest.mark.parametrize('rehearsal', ['existing-HOLD', 'plan'], indirect=True)
def test_preflight_refuses_without_hold(rehearsal: tuple[k.KL1Live, Remote]) -> None:
    live, fake = rehearsal
    result = k.window(live, lambda row: None)
    assert 'stop' in result and not fake.events and not fake.restores


@pytest.mark.parametrize('rehearsal', ['hold-open', 'snapshot'], indirect=True)
def test_failed_hold_reply_or_snapshot_does_not_stop_production(rehearsal: tuple[k.KL1Live, Remote]) -> None:
    live, fake = rehearsal
    result = k.window(live, lambda row: None)
    assert 'stop' in result and not fake.stopped
    assert fake.restores == ['check', 'close']
    assert str(record(result['restore'])['status']).startswith('RESTORED')
    cleanup = record(record(result['restore'])['container_cleanup'])
    assert all('BENCH ABSENT' in str(record(v)['text']) for v in cleanup.values())


@pytest.mark.parametrize('rehearsal', ['fence', 'launcher-live', 'cleanup', 'restore-strings', 'restore-deadline'], indirect=True)
def test_restore_failure_is_visible(rehearsal: tuple[k.KL1Live, Remote]) -> None:
    live, fake = rehearsal
    result = k.window(live, lambda row: None)
    assert record(result['restore'])['status'] == 'FAILED'
    if fake.fail in ('fence', 'launcher-live', 'cleanup'):
        assert not fake.restores
        assert fake.events[-2:] == ['kl1-rm-sparkhead', 'kl1-rm-sparkworker-wired']


def test_second_signal_after_supervisor_close_still_restores(rehearsal: tuple[k.KL1Live, Remote], monkeypatch: pytest.MonkeyPatch) -> None:
    live, fake = rehearsal
    close = live.sup.close
    def closing() -> None:
        close()
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
        signal.raise_signal(signal.SIGTERM)
    monkeypatch.setattr(live.sup, 'close', closing)
    try:
        result = k.window(live, lambda row: None)
        assert_restored(result, fake)
    finally:
        monkeypatch.setattr(live.sup, 'close', close)


def test_mutations_cannot_claim_success(rehearsal: tuple[k.KL1Live, Remote], monkeypatch: pytest.MonkeyPatch) -> None:
    live, fake = rehearsal
    # Mutation: omit the inherited W8 restore. The independent order/restore assertion must reject it.
    monkeypatch.setattr(pf.Live, 'restore', lambda self, **kw: {'status': 'RESTORED'})
    result = k.window(live, lambda row: None)
    with pytest.raises(AssertionError):
        assert_restored(result, fake)
    # Mutation: erase container cleanup evidence from an otherwise claimed success.
    fake.restores = ['start', 'check', 'close']
    fake.events = ['start', 'check', 'close']
    with pytest.raises(ValueError):
        assert_restored(result, fake)


def test_skipping_container_removal_mutation_fails(rehearsal: tuple[k.KL1Live, Remote], monkeypatch: pytest.MonkeyPatch) -> None:
    live, fake = rehearsal
    plain = live.plain
    def mutated(step: str, command: list[str], timeout: float) -> tuple[int, str]:
        if step.startswith('kl1-rm-'):
            command = [*command[:-1], command[-1].replace(f'docker rm -f {k.CONTAINER}', 'true')]
        return plain(step, command, timeout)
    monkeypatch.setattr(live, 'plain', mutated)
    result = k.window(live, lambda row: None)
    assert record(result['restore'])['status'] == 'FAILED'
    assert not fake.restores


def test_generated_commands_are_bounded_and_node_local(tmp_path: Path) -> None:
    for rank in (0, 1):
        command = k.remote_bench(rank, 'KL1-W-test')
        assert '--name dsv41-kl1-bench' in command and 'timeout --signal=TERM --kill-after=15s --foreground 1500s' in command
        assert shlex.split(command)[:3] == ['setsid', '--wait', 'python3']
        assert 'test ! -e' in command and f'{kr.CONTROL}/KL1-W-test/ABORT' in command
        assert '--memory=32g --memory-swap=32g' in command and '--network none' in command
        assert '--user "$(id -u):$(id -g)"' in command
        assert 'src=$HOME/dsv41-offload-run/kl1/src' in command and '/home/operator' not in command
        assert '--mount type=bind,src=' + k.PLAN + ',dst=/plan.json,readonly' in command
        script = k.bench_script(rank, 'KL1-W-test')
        assert script.index('--out /kl1-out/prod-first.json') < script.index('--variant pipe4')
        assert script.index('--out /kl1-out/prod-last.json') > script.index('--out /kl1-out/dense.json')
        assert all(f'{m}; {m}.ext()' in script for m in ('mrow', 'router', 'l2pf'))
        assert '24*1024**3' in script
        words = shlex.split(script)
        for i in range(len(words) - 2):
            if words[i:i + 2] == ['python3', '-c']:
                ast.parse(words[i + 2])
        path = tmp_path / f'bench{rank}.sh'
        path.write_text(script)
        subprocess.run(['bash', '-n', str(path)], check=True)
        subprocess.run(['bash', '-n', '-c', command], check=True)
    purpose = live_stub().hold_purpose()
    assert 'start_exp' not in live_stub().reap_script()
    assert re.search('window[.]sh open KL1-WINDOW tag:', 'window.sh open ' + purpose)


def test_dry_default_and_authorization_refusals(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    assert k.main([]) == 0
    assert capsys.readouterr().out.startswith('DRY RUN')
    monkeypatch.delenv('WINDOW_COORDINATOR', raising=False)
    assert k.main(['--execute']) == 1
    assert k.main(['--precheck']) == 1
    assert 'before any remote action' in capsys.readouterr().err


FAKE_DOCKER = '''import os,sys,time
from pathlib import Path
root=Path(os.environ['HOME'])
if sys.argv[1]=='run':
    (root/'pending').touch()
    while not (root/'release').exists(): time.sleep(.01)
    (root/'started').touch()
    (root/'container').touch()
    while True: time.sleep(.1)
elif sys.argv[1]=='rm':
    (root/'container').unlink(missing_ok=True)
elif sys.argv[1]=='ps':
    if (root/'container').exists(): print('fake-container')
else: raise SystemExit(99)
'''


def wait_file(path: Path) -> None:
    until = time.monotonic() + 5
    while not path.exists():
        if time.monotonic() >= until:
            raise AssertionError(f'owned fake did not reach {path.name}')
        time.sleep(.01)


def fake_node(root: Path) -> dict[str, str]:
    root.mkdir()
    for name in ('out', 'src', 'ext'):
        (root / k.RUN_DIR / name).mkdir(parents=True)
    bin_dir = root / 'bin'
    bin_dir.mkdir()
    # Explicit test interpreter, even for the generated remote command's `python3`.
    (bin_dir / 'python3').symlink_to(sys.executable)
    (bin_dir / 'docker').write_text(f'#!{sys.executable}\n' + FAKE_DOCKER)
    (bin_dir / 'nvidia-smi').write_text(f'#!{sys.executable}\nimport os\nfrom pathlib import Path\n'
        'if (Path(os.environ["HOME"])/"container").exists(): print(12345)\n')
    for name in ('docker', 'nvidia-smi'):
        (bin_dir / name).chmod(0o700)
    return {**os.environ, 'HOME': str(root), 'PATH': str(bin_dir) + ':' + os.environ['PATH'], 'CUDA_VISIBLE_DEVICES': ''}


@pytest.mark.parametrize('pause,skip_reap', [('before-register', False), ('before-container', False),
                                         ('container-active', False), ('before-container', True)])
def test_remote_owned_late_launch_is_fenced_before_w8(
    rehearsal: tuple[k.KL1Live, Remote], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pause: str, skip_reap: bool,
) -> None:
    """Real generated shells and remote owner on BOTH fake nodes; local Supervisor cannot own/kill them.

    Release pending work only at fake W8 start, after restore's cleanup checks. No late bench may appear.
    """
    live, fake = rehearsal
    envs = {host: fake_node(tmp_path / host) for host in pf.HOSTS}
    owners: list[subprocess.Popen[str]] = []
    plain = live.plain
    def launch() -> None:
        for rank, host in enumerate(pf.HOSTS):
            command = k.remote_bench(rank, live.tag)
            if pause == 'before-register':
                # SSH-server-owned session accepted but not yet at the registration lock.
                prefix = 'touch "$HOME/pending"; while [ ! -e "$HOME/release" ]; do sleep .01; done\n'
                command = prefix + command
            proc = subprocess.Popen(['bash', '-c', command], env=envs[host], start_new_session=True,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
            owners.append(proc)
            root = Path(envs[host]['HOME'])
            wait_file(root / 'pending')
            if pause == 'container-active':
                (root / 'release').touch()
                wait_file(root / 'container')
        raise ValueError('CPU abort: remote session remains independently owned')

    def local_cleanup(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        if label.startswith(('kl1-fence-', 'kl1-rm-')):
            fake.events.append(label)
            command = argv[-1]
            if skip_reap and label.startswith('kl1-rm-'):
                # Mutation: forge absence without reaping. The independent W8-start assertion must catch it.
                command = command.replace(kr.command('reap', live.tag), 'echo ' + shlex.quote(kr.ABSENT))
            proc = subprocess.run(['bash', '-c', command], env=envs[argv[-2]], capture_output=True, text=True,
                                  timeout=min(timeout, 30))
            return proc.returncode, proc.stdout + proc.stderr
        if label == 'start':
            for host in pf.HOSTS:
                root = Path(envs[host]['HOME'])
                pidfile = root / kr.CONTROL / live.tag / 'launcher.pid'
                if pidfile.exists():
                    assert not kr.group_members(int(pidfile.read_text().splitlines()[0])), 'live remote launcher at W8 start'
                assert not (root / 'container').exists(), 'bench container at W8 start'
                (root / 'release').touch()
            for owner in owners:
                owner.wait(timeout=5)
            assert all(not (Path(e['HOME']) / 'started').exists() for e in envs.values()) or pause == 'container-active'
        if label.startswith('kl1-dir-clean-'):
            fake.events.append(label)
            proc = subprocess.run(['bash', '-c', argv[-1]], env=envs[argv[-2]], capture_output=True, text=True, timeout=5)
            return proc.returncode, proc.stdout + proc.stderr
        return plain(label, argv, timeout)

    monkeypatch.setattr(live, 'bench', launch)
    monkeypatch.setattr(live, 'plain', local_cleanup)
    try:
        result = k.window(live, lambda row: None)
        assert 'CPU abort' in str(result.get('stop')), result
        if skip_reap:
            assert record(result['restore'])['status'] == 'FAILED'
            assert 'live remote launcher at W8 start' in str(record(result['restore'])['error'])
            assert not fake.restores
            return
        assert_restored(result, fake)
        for host, env in envs.items():
            root = Path(env['HOME'])
            assert not (root / 'container').exists()
            assert (root / kr.CONTROL / live.tag / 'ABORT').exists(), 'persistent fence removed by directory cleanup'
            # A brand-new late arrival after disposable directory cleanup must also refuse this run.
            retry = subprocess.run(['bash', '-c', k.remote_bench(0, live.tag)], env=env,
                                   capture_output=True, text=True, timeout=5)
            assert retry.returncode != 0 and 'fenced by ABORT' in retry.stderr
            proof = record(record(record(result['restore'])['container_cleanup'])[host])
            assert kr.ABSENT in str(proof['text'])
    finally:
        for proc in owners:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        # setsid may fork from the SSH session; clean only identities recorded by these owned fakes if a test fails.
        for env in envs.values():
            root = Path(env['HOME'])
            for mode in ('fence', 'reap'):
                subprocess.run(['bash', '-c', kr.command(mode, live.tag)], env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30, check=True)


def test_remote_reaper_refuses_reused_identity_and_live_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = fake_node(tmp_path / 'node')
    # This owned process deliberately represents an unrelated group. A wrong start time must NEVER signal it.
    other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
    root = Path(env['HOME']) / kr.CONTROL / 'tag'
    root.mkdir(parents=True)
    ticks = kr.proc_stat(other.pid)[2]
    boot = Path('/proc/sys/kernel/random/boot_id').read_text()
    (root / 'ABORT').touch()
    (root / 'launcher.pid').write_text(f'{other.pid}\n{ticks + 1}\n{boot}')
    try:
        result = subprocess.run(['bash', '-c', kr.command('reap', 'tag')], env=env, capture_output=True, text=True, timeout=5)
        assert result.returncode != 0 and 'refusing to signal' in result.stderr
        assert kr.ABSENT not in result.stdout and other.poll() is None
        # A surviving, identity-verified group cannot produce absence proof even if TERM/KILL do not take effect.
        (root / 'launcher.pid').write_text(f'{other.pid}\n{ticks}\n{boot}')
        monkeypatch.setattr(os, 'killpg', lambda pgid, sig: None)
        with pytest.raises(ValueError, match='still alive'):
            kr.reap(root, 'tag', term_s=0, kill_s=0)
    finally:
        other.kill()
        other.wait(timeout=5)
