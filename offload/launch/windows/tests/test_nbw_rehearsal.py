"""Real NBW Live/Supervisor with fake remote commands and owned CPU telemetry only."""
from __future__ import annotations

import json
import shlex
import signal
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from offload.launch.windows import scoped_remote as owner
from offload.launch.windows import nbw_cells as c
from offload.launch.windows import nbw_window as n
from offload.launch.windows import pf1_window as pf
from offload.launch.windows.tests import test_nbw_window as pure
from offload.launch.windows.tests import test_pf1_window as base
from offload.prefill_maxlen.evidence import Json, record


def scoped_argv(command: str) -> tuple[str, list[str]]:
    """Decode the submitted scoped payload without executing the inline owner."""
    tokens = shlex.split(command.removeprefix('(nohup ').removesuffix(' > /dev/null 2>&1 < /dev/null &)'))
    at = tokens.index('launch')
    assert tokens[0] == 'env' and tokens[1].startswith('SCOPED_REMOTE_DEADLINE=')
    assert 'systemd-run' in tokens[tokens.index('-c') + 1]
    return tokens[at + 1], tokens[at + 3:]


def control_receipt(label: str) -> str:
    assert label.startswith(('owned-fence-', 'owned-reap-', 'owned-absent-'))
    return (owner.FENCED if label.startswith('owned-fence-') else owner.ABSENT) + '\n'


class Remote(base.FakeRemote):
    def __init__(self, out: Path, sup: pf.Supervisor, clock: base.Clock, fail: str, trip: Path) -> None:
        super().__init__(out, sup, clock, fail, trip=trip)
        self.events: list[str] = []
        self.tag = "tag"
        self.original_hold = ""
        self.native_arm = n.ARMS[1]
        self.started: list[str] = []
        self.greedy_starts: list[str] = []
        self.measured: list[tuple[str, str]] = []

    def identity(self, rank: int) -> dict[str, Json]:
        if self.arm is None:
            return pure.production(rank)
        row = pure.identity(cast(n.Arm, self.arm), rank)
        if self.fail == 'runtime-identity' and cast(n.Arm, self.arm).role == 'I':
            record(row['env'])['VLLM_DSPARK_REMOTE_IDENTITY'] = 'f' * 64
        return row

    def native_probe(self) -> str:
        self.steps += 1
        status: dict[str, Json] = {'tp': 2, 'tp_comm_errors': 0, 'steps': self.steps,
            'time': base.iso(self.clock()), 'arm': 'native-hybrid-kx2',
            'drafter_identity': pure.NATIVE.identity, 'markov_identity': pure.NATIVE.markov}
        if self.fail == 'identity-refusal':
            status['drafter_identity'] = 'f' * 64
        return json.dumps({'status': status, 'cwd': '/home/fake/' + pure.NATIVE.tree,
            'expected_cwd': '/home/fake/' + pure.NATIVE.tree, 'revision': pure.NATIVE.commit,
            'gpus': pf.WORKER_GPUS, 'level': '2', 'cmd': ['python', '--tp', '2', '--max-model-len', '1048576']})

    def remote(self, label: str, host: str, command: str, stdin: str | None) -> str:
        if label.startswith(('owned-fence-', 'owned-reap-', 'owned-absent-')):
            self.events.append(label)
            return control_receipt(label)
        if label == 'pair-close':
            self.events.append(label)
            return 'PRODUCTION HOLD OK\n'
        if label == 'ledger':
            assert n.PROD_COMMIT in command and n.PROD_TAG in command
            assert 'offload/launch/warmup.py offload/launch/warmup_serve.sh' in command
            if self.fail == 'existing-HOLD':
                raise ValueError('existing HOLD')
            return ''
        if label == 'hold-open':
            tag, payload = scoped_argv(command)
            assert tag == self.tag + '-hold'
            assert 'os.link(candidate, hold)' in payload[-1] and f'NBW-WINDOW {self.tag}:' in payload[-1]
            assert 'offload/launch/window.sh open' not in payload[-1]
            self.hold = f'{base.iso(self.clock())} DSV41-OFFLOAD (session coordinator): NBW-WINDOW {self.tag}: native'
            self.original_hold = self.hold
            self.events.append('hold')
            if self.fail == 'hold-reply':
                raise ValueError('lost HOLD reply')
            return 'opened'
        if label == 'hold-read':
            tokens = shlex.split(command)
            assert tokens[-3:-1] == ['check', self.tag + '-hold']
            assert len(tokens[-1]) == 32
            return self.hold
        if command == pf.HOST_PARITY:
            mem = pf.START_FLOOR_KB - 1 if self.fail == 'before-start-memory' and label.startswith('s2-I') else 7_000_000
            return f'{n.KERNEL}\n{n.DRIVER}\nMemAvailable: {mem} kB\n'
        if label == 'native-tree':
            assert command.count('cd ~/') == 1 and command.count('test -d ~/dsv41-offload/n4/build') == 1
            assert f'true && test ! -e ~/dsv41-offload-run/nbw/{self.tag}-s2-I' in command
            return ''
        if label == 'native-reviewed-identity':
            return json.dumps({'drafter_identity': pure.NATIVE.identity, 'markov_identity': pure.NATIVE.markov})
        assert label != 'pair-open', 'pair-open must be inside the registered native launcher'
        if label.endswith('-native-launch'):
            self.events.append(label)
            self.native_arm = next(a for a in n.ARMS if label.startswith(a.name + '-'))
            tag, payload = scoped_argv(command)
            assert tag == f'{self.tag}-{self.native_arm.name}' and 'nohup' in command
            assert 'prod_window.sh pair COORD' in payload[-1]
            if self.fail == 'pair-reply':
                raise ValueError('lost pair reply after production worker stopped')
            if self.fail == 'native-start':
                raise ValueError('native worker failed to start')
            return ''
        if label.endswith('-native-status'):
            return self.native_probe()
        if host == 'x86host' and 'expected_cwd' in command:
            return self.native_probe()
        if label.endswith('-start'):
            self.arm = next(a for a in n.ARMS if label == a.name + '-start')
            self.started.append(self.arm.name)
            self.events.append('start-' + self.arm.name)
            tag, payload = scoped_argv(command)
            assert tag == f'{self.tag}-{self.arm.name}-engine'
            assert n.CONFIG in payload[-1] and f'NBW_ROLE={self.arm.role}' in payload[-1]
            if self.fail == 'spark-start' and self.arm.role == 'I':
                raise ValueError('Spark start failed rc7')
            return 'ready'
        if '--pid' in command and stdin is not None:
            text = super().remote(label, host, command, stdin)
            if self.fail == 'failed-round' and self.arm is not None and cast(n.Arm, self.arm).role == 'I' \
                    and label.endswith('-after'):
                obj = json.loads(text)
                obj['counters']['rounds_failed'] = 1
                return json.dumps(obj)
            if self.fail == 'startup-failed-round' and self.arm is not None:
                obj = json.loads(text)
                obj['counters']['rounds_failed'] = 1   # failed during warm-up, never grows while measuring
                return json.dumps(obj)
            return text
        return super().remote(label, host, command, stdin)

    def call(self, label: str, argv: list[str], *, stdin: str | None = None, timeout: float = 120, inflight: int = 0) -> str:
        if label.startswith(('owned-fence-', 'owned-reap-', 'owned-absent-')):
            self.events.append(label)
            self.sup.poll()
            return control_receipt(label)
        if argv[1:3] != ['-m', 'offload.launch.windows.nbw_cells']:
            return super().call(label, argv, stdin=stdin, timeout=timeout, inflight=inflight)
        self.sup.poll()
        mode, path = argv[argv.index('--mode') + 1], Path(argv[argv.index('--output') + 1])
        arm = cast(n.Arm, self.arm)
        if mode == 'build':
            path.write_text('{}\n')
            return ''
        if mode == 'greedy':
            self.greedy_starts.append(arm.name)
            self.events.append('greedy-' + arm.name)
            diverged = self.fail in ('greedy-divergence', 'greedy-near-tie')
            checks = [{'i': i, 'identical': i != 3 or not diverged} for i in range(6)]
            if self.fail == 'greedy-near-tie':
                checks[3]['near_tie'] = True
            path.write_text(json.dumps({'identical': 5 if diverged else 6,
                                        'near_tie': 1 if self.fail == 'greedy-near-tie' else 0, 'checks': checks}))
            return ''
        name = argv[argv.index('--cell') + 1]
        self.measured.append((arm.name, name))
        assert inflight == c.CELLS[name].batch
        if arm.role == 'I':
            if self.fail == 'memory':
                cast(Path, self.trip).touch()
                until = time.monotonic() + .2
                while time.monotonic() < until:
                    self.sup.poll()
            if self.fail == 'deadline':
                self.clock.now = self.sup.abort_at + 1
                self.sup.poll()
            if self.fail == 'signal':
                raise KeyboardInterrupt('signal 15')
            if self.fail == 'STOP':
                (self.out / 'STOP').touch()
                self.sup.poll()
        path.write_text(json.dumps(pure.sample_cell(name, 55 if arm.role == 'I' else 50 + .1 * n.ARMS.index(arm))))
        if self.fail == 'optional-no-fit' and arm == n.ARMS[3] and name == '32k-b3':
            self.clock.now = self.sup.abort_at - 6 * 60
        self.clock.now += 10
        return ''

    def plain(self, step: str, argv: list[str], timeout: float) -> tuple[int, str]:
        assert all(signal.getsignal(sig) == signal.SIG_IGN for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP))
        self.events.append(step)
        assert timeout <= 900
        if step.startswith(('owned-fence-', 'owned-reap-', 'owned-absent-')):
            mode = argv[-1].rsplit(' ', 2)[-2]
            assert mode in ('fence', 'reap', 'absent')
            if self.fail == 'native-reap-failed' and mode == 'reap':
                return 0, 'no verified absence'
            return 0, control_receipt(step)
        if step == 'pair-close':
            _, payload = scoped_argv(argv[-1])
            assert payload[-1].endswith('prod_window.sh close') and 'PROD_WINDOW_WAIT_S=60' in payload[-1]
            return (1, 'unit failed') if self.fail == 'pair-close-failed' else (0, 'PRODUCTION HOLD OK\n')
        if step == 'experiment-stop':
            _, payload = scoped_argv(argv[-1])
            assert payload[0] == 'ssh' and './stop.sh </dev/null' in payload[-1]
            return 0, 'experiment stopped\n'
        if step.startswith('jit-wait-'):
            assert 'tf_experts' in argv[-1] and '_load_engram_remote_ext' in argv[-1]
            assert 'NBW EXPERIMENT ABSENT' in argv[-1]
            return 0, 'NBW EXPERIMENT ABSENT\n'
        if step == 'commit-close':
            self.hold = ''
            return 0, owner.CLOSED + '\n'
        if step == 'cancel-close':
            return 0, 'SCOPED CLOSE COMMIT FENCED\n'
        if step == 'retain-close':
            self.hold = self.original_hold
            return 0, owner.RETAINED + '\n'
        if step in ('hold-reread', 'nbw-hold-reread'):
            return 0, self.hold
        if step in base.RESTORE_OK:
            tag, payload = scoped_argv(argv[-1])
            assert tag.startswith(self.tag + '-restore-') and payload == ['bash', str(pf.RESTORE), step]
            self.restores.append(step)
            if self.fail == 'restore-greedy' and step == 'check':
                return 0, base.RESTORE_OK[step].replace('3 IDENTICAL', '3 DIVERGES at char 1')
            if self.fail == 'restore-missing' and step == 'check':
                return 0, base.RESTORE_OK[step].replace('greedy K1-post vs restored: 5 IDENTICAL 105\n', '')
            if self.fail == 'restore-duplicate' and step == 'check':
                return 0, base.RESTORE_OK[step].replace('5 IDENTICAL', '4 IDENTICAL')
            return 0, base.RESTORE_OK[step]
        raise AssertionError(f'unexpected unsupervised command {step}')


@pytest.fixture
def rehearsal(tmp_path: Path, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[n.Live, Remote]]:
    fail = str(getattr(request, 'param', ''))
    monitor = tmp_path / 'monitor.py'
    monitor.write_text(base.MONITOR)
    out = tmp_path / 'NBW-fake'
    out.mkdir()
    clock = base.Clock()
    trip = tmp_path / 'trip'
    sup = base.NoTunnel(out, clock=clock, monitor=lambda host: [sys.executable, str(monitor), 'low', str(trip)], demand=lambda: 0.0)
    fake = Remote(out, sup, clock, fail, trip)
    live = n.Live(out, sup, native=pure.NATIVE, remote='dsv41-deploy-offload', expected='0' * 40,
                  data_dir=tmp_path, tag='tag', call=fake.call, plain=fake.plain, clock=clock)
    import offload.prefill_trace.window_requests as wr
    monkeypatch.setattr(wr, 'corpus', lambda root: ('public fake data', {}))
    monkeypatch.setattr(c, 'greedy_inputs', lambda: (['public prompt'] * 6, [{'i': i, 'text': 'public reference'} for i in range(6)]))
    try:
        yield live, fake
    finally:
        sup.close()


def assert_restored(result: dict[str, Json], fake: Remote, *, stopped: bool = True) -> None:
    assert fake.restores == (['start', 'check', 'close'] if stopped else ['check', 'close'])
    assert str(record(result['restore'])['status']).startswith('RESTORED'), result
    assert fake.events.index('pair-close') < fake.events.index('check')
    assert record(result['restore'])['pair_close'] == 'PRODUCTION HOLD OK'
    assert record(result['restore'])['restore_deadline_met'] is True
    assert not fake.hold, 'verified restore must commit the owned HOLD close'
    assert fake.events.index('close') < fake.events.index('commit-close') < fake.events.index('cancel-close')
    for host in (*owner.NODES, 'localhost'):
        assert fake.events.index(f'owned-absent-{fake.tag}-restore-close-{host}') < fake.events.index('commit-close')


def test_full_live_sequence(rehearsal: tuple[n.Live, Remote]) -> None:
    live, fake = rehearsal
    result = n.window(live, lambda row: None)
    assert 'stop' not in result, result
    assert result['status'] == 'COMPLETE' and record(result['decision'])['status'] == 'CANDIDATE'
    assert fake.started == [a.name for a in n.ARMS]
    assert fake.greedy_starts == ['s2-I', 's4-I']
    assert fake.measured == [(a.name, cell) for a in n.ARMS for cell in c.CELLS]
    assert_restored(result, fake)
    for arm in ('s2-I', 's4-I'):
        assert fake.events.index('greedy-' + arm) > fake.events.index('start-' + arm)
        assert fake.events.index('owned-fence-tag-' + arm + '-sparkhead') < fake.events.index('owned-reap-tag-' + arm + '-sparkhead')
    deadlines = json.loads((live.out / 'deadlines.json').read_text())
    assert deadlines['restore_by'] - deadlines['opened'] == 90 * 60
    rollback = json.loads((live.out / 'rollback.json').read_text())
    assert rollback['production_commit'] == n.PROD_COMMIT


@pytest.mark.parametrize('rehearsal', ['greedy-near-tie'], indirect=True)
def test_native_near_tie_greedy_divergence_is_measured(rehearsal: tuple[n.Live, Remote]) -> None:
    # Live NBW-W-20261026T224001Z stopped on a native greedy flip; W8 showed such flips are near-ties (batch-shape numerics).
    live, fake = rehearsal
    result = n.window(live, lambda row: None)
    assert 'stop' not in result, result
    assert result['status'] == 'COMPLETE'
    assert fake.measured == [(a.name, cell) for a in n.ARMS for cell in c.CELLS]
    assert_restored(result, fake)


@pytest.mark.parametrize('rehearsal', ['startup-failed-round'], indirect=True)
def test_warmup_failed_round_is_recorded_not_a_stop(rehearsal: tuple[n.Live, Remote]) -> None:
    # Live NBW-W-20261026T221846Z: rounds_failed was 1 from engine warm-up and never grew while measuring.
    live, fake = rehearsal
    result = n.window(live, lambda row: None)
    assert 'stop' not in result, result
    assert result['status'] == 'COMPLETE'
    for arm in n.ARMS:
        checks = record(record(record(result['starts'])[arm.name])['checks'])
        assert record(checks['startup_failed_rounds'])['rounds_failed'] == 1
    assert_restored(result, fake)


@pytest.mark.parametrize('rehearsal,message', [('native-start', 'failed to start'), ('identity-refusal', 'identity refusal'),
    ('runtime-identity', 'env drift'), ('spark-start', 'rc7'), ('memory', 'below 4 GiB'),
    ('before-start-memory', '5 GiB'), ('deadline', '75-minute NBW'), ('signal', 'signal 15'),
    ('STOP', 'STOP file'), ('greedy-divergence', 'greedy divergence'), ('pair-reply', 'lost pair reply'),
    ('failed-round', 'rounds_failed')], indirect=['rehearsal'])
def test_every_stop_restores(rehearsal: tuple[n.Live, Remote], message: str) -> None:
    live, fake = rehearsal
    result = n.window(live, lambda row: None)
    assert message in str(result.get('stop')), result
    assert record(result['decision'])['status'] == 'NOT A CANDIDATE'
    assert_restored(result, fake)
    if fake.fail == 'greedy-divergence':
        assert not any(arm == 's2-I' for arm, cell in fake.measured), 'divergent arm measured after STOP'


@pytest.mark.parametrize('rehearsal', ['restore-greedy', 'restore-missing', 'restore-duplicate',
                                        'native-reap-failed', 'pair-close-failed'], indirect=True)
def test_restore_refusal_keeps_hold(rehearsal: tuple[n.Live, Remote]) -> None:
    live, fake = rehearsal
    result = n.window(live, lambda row: None)
    assert record(result['restore'])['status'] == 'FAILED'
    assert record(result['decision'])['status'] == 'NOT A CANDIDATE'
    assert 'close' not in fake.restores
    if fake.fail.startswith('restore-'):
        assert fake.restores == ['start', 'check']
    else:
        assert not fake.restores


@pytest.mark.parametrize('rehearsal', ['existing-HOLD'], indirect=True)
def test_preflight_stop_touches_no_hold(rehearsal: tuple[n.Live, Remote]) -> None:
    live, fake = rehearsal
    result = n.window(live, lambda row: None)
    assert str(record(result['restore'])['status']).startswith('NOT NEEDED')
    assert not fake.events and not fake.restores


@pytest.mark.parametrize('rehearsal', ['hold-reply'], indirect=True)
def test_lost_hold_reply_checks_and_closes_without_restarting(rehearsal: tuple[n.Live, Remote]) -> None:
    live, fake = rehearsal
    result = n.window(live, lambda row: None)
    assert 'lost HOLD reply' in str(result.get('stop'))
    assert_restored(result, fake, stopped=False)
    assert not fake.started


@pytest.mark.parametrize('rehearsal', ['optional-no-fit'], indirect=True)
def test_closing_k_is_skipped_when_recovery_budget_is_needed(rehearsal: tuple[n.Live, Remote]) -> None:
    live, fake = rehearsal
    result = n.window(live, lambda row: None)
    assert 'stop' not in result, result
    assert fake.started == ['s1-K', 's2-I', 's3-K', 's4-I']
    assert record(record(result['starts'])['s5-K'])['status'] == 'SKIPPED_TIME'
    assert_restored(result, fake)


def test_second_signal_during_restore_is_ignored(rehearsal: tuple[n.Live, Remote], monkeypatch: pytest.MonkeyPatch) -> None:
    live, fake = rehearsal
    close = live.sup.close
    def closing() -> None:
        close()
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
        signal.raise_signal(signal.SIGTERM)
    monkeypatch.setattr(live.sup, 'close', closing)
    try:
        result = n.window(live, lambda row: None)
        assert_restored(result, fake)
    finally:
        monkeypatch.setattr(live.sup, 'close', close)


@pytest.mark.parametrize('pause', ['before-register', 'active'])
def test_real_owned_native_launcher_is_absent_before_pair_close(
    rehearsal: tuple[n.Live, Remote], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pause: str,
) -> None:
    """Use real owner commands on a fake X86host. A late SSH session must never start after recovery."""
    import os
    import subprocess
    live, fake = rehearsal
    node = tmp_path / 'fake-x86host'
    node.mkdir()
    bin_dir = node / 'bin'
    bin_dir.mkdir()
    (bin_dir / 'python3').symlink_to(sys.executable)
    env = {**os.environ, 'HOME': str(node), 'PATH': str(bin_dir) + ':' + os.environ['PATH'], 'CUDA_VISIBLE_DEVICES': ''}
    processes: list[subprocess.Popen[str]] = []
    call, plain = live.call, live.plain
    import uuid
    live.tag = fake.tag = 'own1test-nbw-' + uuid.uuid4().hex[:10]
    tag = live.tag + '-s2-I'
    def wait_file(path: Path) -> None:
        until = time.monotonic() + 5
        while not path.exists():
            assert time.monotonic() < until, f'owned fake never reached {path.name}'
            time.sleep(.01)
    def command_layer(label: str, argv: list[str], *, stdin: str | None = None,
                      timeout: float = 120, inflight: int = 0) -> str:
        if label == 's2-I-native-launch':
            worker = ('import os,time; from pathlib import Path; '
                      '(Path.home()/"started").touch(); time.sleep(60)')
            command = owner.command('launch', tag, [sys.executable, '-c', worker], deadline=live.sup.abort_at)
            if pause == 'before-register':
                command = 'touch "$HOME/pending"; while [ ! -e "$HOME/release" ]; do sleep .01; done\n' + command
            proc = subprocess.Popen(['bash', '-c', command], env=env, start_new_session=True,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
            processes.append(proc)
            wait_file(node / ('pending' if pause == 'before-register' else 'started'))
            raise ValueError('CPU STOP with independently owned native launcher')
        return call(label, argv, stdin=stdin, timeout=timeout, inflight=inflight)
    def cleanup(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        if label.startswith(tuple(f'owned-{mode}-{tag}-' for mode in ('fence', 'reap', 'absent'))):
            fake.events.append(label)
            proc = subprocess.run(['bash', '-c', argv[-1]], env=env, capture_output=True, text=True, timeout=min(timeout, 30))
            return proc.returncode, proc.stdout + proc.stderr
        if label == 'pair-close':
            assert not owner.units(tag)
            assert not any(owner.populated(path) for path in owner.paths(node / owner.CONTROL / tag, tag))
            (node / 'release').touch()
            for child in processes:
                child.wait(timeout=5)
            assert (node / 'started').exists() == (pause == 'active')
        return plain(label, argv, timeout)
    monkeypatch.setattr(live, 'call', command_layer)
    monkeypatch.setattr(live, 'plain', cleanup)
    try:
        result = n.window(live, lambda row: None)
        assert 'independently owned native launcher' in str(result.get('stop'))
        assert_restored(result, fake)
        retry = subprocess.run(['bash', '-c', owner.command('launch', tag, [sys.executable, '-c', 'raise SystemExit(99)'], deadline=live.sup.abort_at)],
                               env=env, capture_output=True, text=True, timeout=5)
        assert retry.returncode == 75
    finally:
        (node / 'release').touch()
        for proc in processes:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        for mode in ('fence', 'reap'):
            subprocess.run(['bash', '-c', owner.command(mode, tag)], env=env, check=True, timeout=30,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@pytest.mark.parametrize('pause', ['before-register', 'active'])
def test_pair_open_is_fenced_even_when_ssh_reply_is_lost(
    rehearsal: tuple[n.Live, Remote], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pause: str,
) -> None:
    """S49: execute the submitted shell and real pair dispatch; release only AFTER W8 closes HOLD.

    Fake unit/GPU primitives use files. The SSH-server session is independent of Supervisor;
    an active pair-open and its descendants must be reaped before production restarts.
    """
    import os
    import subprocess
    live, fake = rehearsal
    node = tmp_path / 'pair-x86host'
    directory = node / 'dsv41-offload'
    (directory / 'd2').mkdir(parents=True)
    cache = node / n.TUNE_CACHE
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b'public fake reviewed tune cache')
    monkeypatch.setattr(n, 'TUNE_SHA256', pf.digest(cache))
    (node / 'production-state').write_text('active\n')
    head, dispatch = (pf.ROOT / 'offload/launch/prod_window.sh').read_text().split('case "${1:-}" in', 1)
    primitives = '''
need_promoted() { :; }
waiters() { :; }
stop_prod() {
  echo inactive > "$HOME/production-state"
  touch "$HOME/pair-active"
  while [ ! -e "$HOME/release" ]; do sleep .01; done
}
state() { if [ "$1" = "$ONE" ]; then echo inactive; else cat "$HOME/production-state"; fi; }
cards() { :; }
if [ "${1:-}" = close ]; then
  rm -f "$PAIR_OK" "$TRAIN_OK"
  echo active > "$HOME/production-state"
  echo 'PRODUCTION HOLD OK'
  exit 0
fi
'''
    (directory / 'prod_window.sh').write_text(head + primitives + 'case "${1:-}" in' + dispatch)
    bin_dir = node / 'bin'
    bin_dir.mkdir()
    (bin_dir / 'python3').symlink_to(sys.executable)
    env = {**os.environ, 'HOME': str(node), 'PATH': str(bin_dir) + ':' + os.environ['PATH'], 'CUDA_VISIBLE_DEVICES': ''}
    processes: list[subprocess.Popen[str]] = []
    call, plain = live.call, live.plain
    import uuid
    live.tag = fake.tag = 'own1test-nbw-' + uuid.uuid4().hex[:10]
    tag = live.tag + '-s2-I'
    labels: list[str] = []

    def wait_file(path: Path) -> None:
        until = time.monotonic() + 5
        while not path.exists():
            assert time.monotonic() < until, f'pair-open never reached {path.name}'
            time.sleep(.01)

    def command_layer(label: str, argv: list[str], *, stdin: str | None = None,
                      timeout: float = 120, inflight: int = 0) -> str:
        if label in ('pair-open', 's2-I-native-launch'):
            labels.append(label)
            command = argv[-1]  # EXACT submitted remote shell, including native_script and owner registration
            if pause == 'before-register':
                command = 'touch "$HOME/pending"; while [ ! -e "$HOME/release" ]; do sleep .01; done\n' + command
            proc = subprocess.Popen(['bash', '-c', command], env=env, start_new_session=True,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
            processes.append(proc)
            wait_file(node / ('pending' if pause == 'before-register' else 'pair-active'))
            raise ValueError('lost pair-open reply; remote SSH session is independently owned')
        return call(label, argv, stdin=stdin, timeout=timeout, inflight=inflight)

    def cleanup(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        if label.startswith(tuple(f'owned-{mode}-{tag}-' for mode in ('fence', 'reap', 'absent'))) or label == 'pair-close':
            fake.events.append(label)
            if label == 'pair-close':
                assert not owner.units(tag)
                assert not any(owner.populated(path) for path in owner.paths(node / owner.CONTROL / tag, tag))
            proc = subprocess.run(['bash', '-c', argv[-1]], env=env, capture_output=True, text=True,
                                  timeout=min(timeout, 30))
            return proc.returncode, proc.stdout + proc.stderr
        if label == 'commit-close':
            assert (node / 'production-state').read_text().strip() == 'active'
            assert not (directory / 'PAIR_OK').exists()
            result = plain(label, argv, timeout)  # Final commit releases HOLD after deferred W8 close
            (node / 'release').touch()
            for child in processes:
                child.wait(timeout=5)
            if labels == ['s2-I-native-launch'] and pause == 'before-register':
                assert not (node / f'dsv41-offload-run/nbw/{tag}').exists(), 'late launcher passed its fence'
            return result
        return plain(label, argv, timeout)

    monkeypatch.setattr(live, 'call', command_layer)
    monkeypatch.setattr(live, 'plain', cleanup)
    try:
        result = n.window(live, lambda row: None)
        assert 'lost pair-open reply' in str(result.get('stop')), result
        assert_restored(result, fake)
        assert (node / 'production-state').read_text().strip() == 'active'
        assert not (directory / 'PAIR_OK').exists(), 'late pair-open mutated production after HOLD closed'
        assert labels == ['s2-I-native-launch'], 'pair-open was submitted outside the owned launcher'
        assert (node / 'pair-active').exists() == (pause == 'active')
        retry = subprocess.run(['bash', '-c', owner.command('launch', tag, ['bash', '-c', n.native_script(pure.NATIVE, tag)], deadline=live.sup.abort_at)],
                               env=env, capture_output=True, text=True, timeout=5)
        assert retry.returncode == 75
        assert (node / 'production-state').read_text().strip() == 'active'
        assert not (directory / 'PAIR_OK').exists()
    finally:
        (node / 'release').touch()
        for proc in processes:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        for mode in ('fence', 'reap'):
            subprocess.run(['bash', '-c', owner.command(mode, tag)], env=env, check=True, timeout=30,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
