"""CPU-only runtime, decision and complete real-Live rehearsal. Every transport and GPU operation is fake."""
from __future__ import annotations

import copy
import functools
import json
import os
import shlex
import signal
import time
from collections.abc import Iterator
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from offload.launch.windows import up3_window as w, pf1_window as pf
from offload.launch.windows.tests import test_pf1_window as base
from offload.prefill_maxlen.evidence import Json, record, number
from offload.up3 import build as b, gates, cells, host_gates
from offload.launch.windows import chain, scoped_remote as owner, nbw_cells as greedy_driver


@pytest.fixture(autouse=True, scope='module')
def cached_immutable_manifests() -> Iterator[None]:
    # These transport fixtures never mutate arm files. Exact rebuild/drift tests
    # exercise disk checks separately; reuse immutable verified bytes here.
    with patch.object(b, 'verify', functools.lru_cache(maxsize=None)(b.verify)):
        yield


def identity(arm: w.Arm | None, rank: int = 0, serial: int = 0) -> dict[str, Json]:
    prod = base.prod_identity(rank)
    prod['argv'] = str(prod['argv']) + ' --speculative-config ' + shlex.quote('{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,2,5],[3,4,4]]}')
    if arm is None:
        return prod
    live = copy.deepcopy(prod)
    live['id'] = arm.name + str(rank) + str(serial)
    record(live['env']).update(b.ENV[arm.role])
    record(live['mounted']).update({b.DIST + name: sha for name, sha in b.verify(arm.role)['files'].items()})
    argv = shlex.split(str(live['argv']).replace('\\\n', ' '))
    argv[argv.index('--port') + 1] = '8001'
    if arm.role == 'k5':
        argv[argv.index('--speculative-config') + 1] = b.SPEC
    live['argv'] = shlex.join(argv)
    return live


def receipt(arm: str, rank: int, run: str, container: str, now: float, program_sha: str) -> dict[str, Json]:
    cases: dict[str, Json] = {key: {'pass': True, 'samples': 1} for key in gates.CASES[arm]}
    if arm == 'split':
        for m in (33, 42, 43, 63):
            record(cases[f'm{m}']).update({'rows': m, 'capacity': 42, 'comparison': 'every-real-output-byte-before-TP',
                'oracle': 'single-smoe' if m <= 42 else 'per-row-decode', 'matched_inputs_weights_ids_perm_pad_limit': True})
        for m in (32, 64):
            record(cases[f'dispatch{m}'])['unchanged'] = True
        record(cases['teacher-forced-rounding']).update({'same_prefix_and_batch_shape': True, 'captured_mass': 1.,
                                                       'kl': .001, 'cc_kl': .001, 'task_review': 'pass'})
    elif arm == 'vor':
        for key in ('valid', 'lens', 'indices', 'padded', 'invalid', 'replay', 'ctx327680'):
            record(cases[key])['comparison'] = 'every-byte-and-index'
        record(cases['actual-groups']).update({'count': 15, 'capacity_mode': True, 'derived_from_loaded_engine': True})
        record(cases['launch']).update({'once_per_eligible_step': True, 'request_time_compile': False,
            'exception_disabled': False, 'replay_copy_only_on_change': True})
    elif arm == 'k5':
        record(cases['five-proposals'])['proposals'] = 5
        record(cases['b4-24-rows'])['rows'] = 24
        record(cases['noncausal-stock'])['byte_identical'] = True
    elif arm == 'dual-ring':
        for i, dev in enumerate(('rocep1s0f1', 'roceP2p1s0f1')):
            record(cases[f'gid-device{i}']).update({'device': dev, 'gid_index': 3, 'ipv4_roce_v2': True})
            record(cases[f'traffic-device{i}'])['bytes_delta'] = 100
        record(cases['collective-bytes'])['byte_identical'] = True
        record(cases['rank-log'])['line'] = f'EXL3 RDMA all-reduce: rank {rank} of 2 on rocep1s0f1,roceP2p1s0f1, gid 3, proxy on cpu 14'
    else:
        record(cases['all-thread-affinities']).update({'every_tid': True, 'thread_count': 100, 'allowed_cpus': '5-9,15-19'})
        record(cases['poller14'])['cpu'] = 14
    return {'schema': 'up3-gb10-gates-v1', 'production': b.PRODUCTION, 'arm': arm, 'rank': rank, 'run': run,
            'container': container, 'program_sha256': program_sha, 'manifest_sha256': b.digest((b.HERE / 'manifests' / (arm + '.json')).read_bytes()),
            'device_capability': [12, 1], 'scope': 'actual-engine-inputs', 'time': now, 'cases': cases,
            'nccl_only_fallback': False, 'FIXTURE_ONLY': True}


def measured(name: str, gain: float = 0, salt: str = 'fixture') -> dict[str, Json]:
    batch = cells.batch_for(name)
    repeats: list[Json] = []
    for repeat in range(cells.REPEATS[name]):
        streams: list[Json] = []
        for i in range(batch):
            n = 150000 if (name == cells.CELLS[-1] and i == 2) or name == cells.LONG else 32000
            streams.append({'prompt_sha256': str(i) * 64, 'prompt_tokens': n, 'budget': 512,
                'decode_tps': 10.0 + gain, 'ttft_s': 1., 'p99_inter_token_delivery_gap_ms': 100.,
                'finish': 'length', 'usage': {'completion_tokens': 512, 'prompt_tokens': n,
                    'prompt_tokens_details': {'cached_tokens': 0}},
                'sampling': {'temperature': 1, 'top_p': 1, 'seed': 42+repeat, 'priority': 10, 'max_tokens': 512},
                'cache_salt_sha256': f'{salt}-{name}-{repeat}-{i}'})
        repeats.append({'repeat': repeat, 'batch': batch, 'streams': streams,
            'spec_request_rounds': 1000000, 'accepted_tokens': round(3000000 + gain*400000),
            'server_decode_ms': 400000000, 'completed_requests': batch,
            'mixed_three_active_overlap': name == cells.CELLS[-1]})
    return {'cell': name, 'batch': batch, 'repetitions': repeats}


def greedy_result() -> dict[str, Json]:
    return {'checks': [{'i': i, 'identical': True} for i in range(6)], 'identical': 6, 'near_tie': 0}


def starts(group: str, gain: float = .4) -> dict[str, Json]:
    return {arm.name: {'status': 'COMPLETE', 'role': arm.role, 'greedy': greedy_result(),
        'cells': {name: measured(name, gain if arm.role != 'control' else 0, salt=arm.name) for name in cells.CELLS}}
        for arm in w.arms(group)}


@pytest.mark.parametrize('group', list(gates.CASES))
def test_current_runtime_and_decision(group: str) -> None:
    for arm in w.arms(group):
        w.runtime(identity(None), identity(arm), arm)
        live = identity(arm)
        record(live['env'])['VLLM_ENGRAM_REMOTE_REQUIRED'] = '1'
        with pytest.raises(ValueError, match='env drift'):
            w.runtime(identity(None), live, arm)
    assert w.decision(group, starts(group))['status'] == 'CANDIDATE FOR REVIEW'
    assert w.decision(group, starts(group, .2))['status'] == 'NOT A CANDIDATE'
    assert w.decision(group, starts(group, .20001))['status'] == 'CANDIDATE FOR REVIEW'
    rows = starts(group)
    rows.pop(next(reversed(rows)))
    assert w.decision(group, rows)['status'] == 'NOT A CANDIDATE'


@pytest.mark.parametrize('field', ['image', 'engine', 'files', 'engram_identity', 'mounted', 'argv'])
def test_drift_is_rejected(field: str) -> None:
    arm = w.arms('split')[1]
    live = identity(arm)
    live[field] = 'changed' if field in ('image', 'engine', 'argv') else {}
    with pytest.raises(ValueError):
        w.runtime(identity(None), live, arm)


def test_per_stream_tail_and_pooled_regression_not_hidden() -> None:
    for name, metric, value in (('32k-b4', 'ttft_s', 1.01),
                                (cells.CELLS[-1], 'p99_inter_token_delivery_gap_ms', 101.)):
        rows = starts('k5')
        cell = record(record(record(rows['s2-k5'])['cells'])[name])
        first = record(cast(list[Json], cell['repetitions'])[0])
        record(cast(list[Json], first['streams'])[0])[metric] = value
        assert w.decision('k5', rows)['status'] == 'NOT A CANDIDATE'
    rows = starts('k5')
    cell = record(record(record(rows['s2-k5'])['cells'])['32k-b4'])
    for repeat in cast(list[Json], cell['repetitions']):
        record(repeat)['server_decode_ms'] = 500000000
    assert w.decision('k5', rows)['status'] == 'NOT A CANDIDATE'


class Remote(base.FakeRemote):
    live: w.Live
    provenance = 0
    down = False  # WIN2 chain: production left down by the previous chained window
    placeholders = True  # WIN2-R2: hold placeholders up on both Sparks between chained windows
    clock_caps = {'spark': 2300, 'sparkworker': 2300}  # what start_exp.sh prints into each start log

    def identity(self, rank: int) -> dict[str, Json]:
        return identity(cast(w.Arm | None, self.arm), rank, self.serial)

    def remote(self, label: str, host: str, command: str, stdin: str | None) -> str:
        if label.startswith('owned-'):
            return owner.FENCED if label.startswith('owned-fence-') else owner.ABSENT
        if label == 'hold-read':
            return self.hold
        if label == 'chain-probe':
            return ('HOLD' if self.hold else 'NOHOLD') + '\n' + ('DOWN' if self.down else 'SERVING') + '\n'
        if label.startswith('chain-placeholders-'):
            assert chain.PLACEHOLDER_PROBE in command and host == label.removeprefix('chain-placeholders-')
            return ('PRESENT' if self.placeholders else 'ABSENT') + '\n'
        if label.startswith('chain-quiet-'):
            return 'UP3 CHAIN QUIET\n'
        if label == 'hold-open':
            # A chained window admits its HOLD with production down; otherwise production must serve.
            assert (f'{pf.DRAIN_S} allow-unreachable' in command) == self.down, 'HOLD admission mode'
            if self.fail == 'hold-open-refused':
                raise ValueError('admission refused before any HOLD was written')
            self.hold =f'{base.iso(self.clock())} DSV41-OFFLOAD (session coordinator): UP3-WINDOW tag: isolated k5'
            if self.fail == 'hold-open':
                raise ValueError('lost HOLD reply')
            return ''
        if label.startswith('host-') or label.endswith(tuple('-host-' + h for h in pf.HOSTS)):
            return f'{w.KERNEL}\n{w.DRIVER}\nMemAvailable: 7000000 kB\n'
        if label.startswith('pins-'):
            return '\n'.join(f'{pf.digest(b.ROOT / p)}  {p}' for p in self.live.pin_files)
        if label.endswith('-start'):
            self.serial += 1
            self.arm = next(a for a in w.arms(self.live.group) if label == a.name + '-start')
            if self.fail == 'start':
                raise ValueError('failed start rc7')
            if self.fail == 'signal':
                raise KeyboardInterrupt('signal 15')
            if self.fail == 'deadline':
                self.clock.now += w.ABORT_S + 1
                self.sup.poll()
            # The Supervisor logs every command as cmd/NNNN-<label>.log; the start log is the archive's clock proof.
            text = ''.join(f'{node} clock cap {mhz} MHz\n' for node, mhz in self.clock_caps.items())
            (self.out / 'cmd').mkdir(exist_ok=True)
            (self.out / 'cmd' / f'{9000 + self.serial:04d}-{label}.log').write_text(text)
            return text
        if '-sources-' in label:
            role = label.split('-sources-')[0]
            return json.dumps(w.source_pins(role))
        if '-host-gate-' in label:
            rank = pf.HOSTS.index(host)
            result: dict[str, Json] = {'arm': self.live.group, 'rank': rank, 'container': self.live.ids[rank]}
            if self.live.group == 'dual-ring':
                result.update(rdma=[f'EXL3 RDMA all-reduce: rank {rank} of 2 on rocep1s0f1,roceP2p1s0f1, gid 3, proxy on cpu 14, '
                    'up to 256 KB (slots 3072 KB, all-gathers up to 3072 KB)'],
                    gids=[{'device': dev, 'index': 3, 'type': 'RoCE v2', 'ipv4': f'10.202.{26+i}.{rank+1}', 'netdev': 'eth'+str(i)}
                          for i, dev in enumerate(host_gates.DEVICES)])
                if self.fail == 'gate':
                    result['rdma'] = []
            else:
                from offload.up3.tests.pin_fixtures import evidence
                result = evidence(rank, self.live.ids[rank])
            return json.dumps(result)
        if '-logs-' in label:
            return json.dumps(base.log_summary(327680))
        if stdin is not None and '--pid' in command and self.provenance:
            stats = json.loads(super().remote(label, host, command, stdin))
            stats['counters']['provenance_errors'] = self.provenance
            return json.dumps(stats)
        if self.fail == 'unfiltered-env' and '{{json .Config}}' in command and self.arm is not None and cast(w.Arm, self.arm).role == 'k5':
            env_result = json.loads(super().remote(label, host, command, stdin))
            env_result['Env'].append('OMP_NUM_THREADS=999')
            return json.dumps(env_result)
        return super().remote(label, host, command, stdin)

    def call(self, label: str, argv: list[str], *, stdin: str | None = None, timeout: float = 120, inflight: int = 0) -> str:
        if label.startswith('owned-'):
            self.sup.poll()
            self.commands.append(label)
            return owner.FENCED if label.startswith('owned-fence-') else owner.ABSENT
        if argv[1:3] == ['-m', 'offload.up3.cells']:
            if self.fail == 'STOP':
                (self.out / 'STOP').touch()
            if self.fail == 'memory':
                assert self.trip is not None
                self.trip.touch()
                until = time.monotonic() + .3
                while time.monotonic() < until:
                    self.sup.poll()
                    time.sleep(.01)
            self.sup.poll()
            self.commands.append(label)
            output = Path(argv[argv.index('--output') + 1])
            if argv[argv.index('--mode')+1] == 'greedy':
                result = greedy_result()
                if self.fail == 'experiment-greedy':
                    record(cast(list[Json], result['checks'])[3])['identical'] = False
                if '--near-tie-nats' in argv:
                    # cells.main records the replay registration only for the registered 0.9+1e-3 limit.
                    assert float(argv[argv.index('--near-tie-nats') + 1]) == cells.REPLAY_NEAR_TIE_NATS
                    recorded = '--record-distribution' in argv
                    assert recorded == (cast(w.Arm, self.arm).role == w.REPLAY)
                    result.update(greedy_registration=cells.REPLAY_REGISTRATION, near_tie_nats=cells.REPLAY_NEAR_TIE_NATS,
                                  distribution_recorded=recorded)
                output.write_text(json.dumps(result))
            elif argv[argv.index('--mode')+1] == 'greedy-kl':
                assert cast(w.Arm, self.arm).name == 's3-control' and Path(argv[argv.index('--candidate') + 1]).is_file()
                output.write_text(json.dumps({'mode': 'greedy-kl', 'registration': cells.REPLAY_REGISTRATION,
                    'bound': float(argv[argv.index('--kl-bound') + 1]), 'checks': [],
                    'status': 'FAIL' if self.fail == 'greedy-kl' else 'PASS'}))
            elif '--cell' not in argv:
                output.write_text('{}')
            else:
                name = argv[argv.index('--cell') + 1]
                output.write_text(json.dumps(measured(name, .4 if cast(w.Arm, self.arm).role != 'control' else 0, salt=cast(w.Arm, self.arm).name)))
                self.rounds += 10
            return ''
        if argv[1:3] == ['-m', 'offload.launch.windows.up2_request']:
            self.sup.poll()
            self.commands.append(label)
            assert inflight == 1 and '--ledger' in argv
            output = Path(argv[argv.index('--output') + 1])
            assert not output.exists()
            if '--needle' in argv:
                output.write_text(json.dumps({'context': 150000, 'recall': 5 if self.fail == 'needle' else 6, 'of': 6,
                                              'foreign_codes': 0}))
            else:
                target = argv[argv.index('--target') + 1]
                on = cast(w.Arm, self.arm).role == w.REPLAY
                probabilities = {'x': .9, 'y': .099} if on and self.fail == 'kl' else {'x': .995, 'y': .004}
                output.write_text(json.dumps({'prompt_sha256': target * 4, 'text_sha256': 't' * 64, 'finish': 'length',
                                              'probabilities': probabilities}))
            if self.fail == 'provenance' and label.endswith('needle150000'):
                self.provenance = 1
            return ''
        return super().call(label, argv, stdin=stdin, timeout=timeout, inflight=inflight)

    def plain(self, label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        assert all(signal.getsignal(s) == signal.SIG_IGN for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP))
        self.commands.append(label)
        if label.startswith('owned-'):
            if self.fail == label or self.fail == 'cleanup' and label.startswith('owned-reap-') and label.endswith('-sparkworker-wired'):
                return 1, 'cleanup failure'
            return 0, owner.FENCED + '\n' if '-fence-' in label else owner.ABSENT + '\n'
        if label.startswith('jit-wait-'):
            return (1, 'JIT stuck') if self.fail == label else (0, 'UP3 EXPERIMENT ABSENT\n')
        if label == 'hold-reread':
            if self.fail == 'slow-restore':
                self.clock.now = self.live.restore_by - 20
            return 0, self.hold
        if label == 'experiment-stop':
            if self.fail == 'slow-restore':
                self.clock.now = self.live.restore_by + 1
            return 0, ''
        if label == 'chained-close':
            # WIN2-R2: the shared CHAINED_CLOSE inside the window's own -restore-close scope; it never kills the
            # placeholders, and its `rm -f HOLD` is only RECORDED by the scope's rm shim (commit-close removes HOLD).
            scoped = shlex.split(argv[-1])
            assert scoped[scoped.index('launch') + 1] == 'tag-close-txn-1-restore-close'  # numbered close txn
            assert 'SCOPED_CLOSE_OWNER=tag' in scoped and scoped[-2:] == ['-c', chain.CHAINED_CLOSE]
            assert 'kill-session' not in chain.CHAINED_CLOSE
            if self.fail == 'chained-close':
                return 124, 'lost chained close reply'
            return 0, chain.CHAINED_CLOSE_OK + '\n'
        if label == 'commit-close':
            self.hold = ''
            return 0, owner.CLOSED
        if label == 'cancel-close':
            return 0, 'SCOPED CLOSE COMMIT FENCED'
        if label == 'retain-close':
            self.hold = 'UP3-WINDOW tag: original retained'
            return 0, owner.RETAINED
        if label in base.RESTORE_OK:
            self.restores.append(label)
            if label == 'start':
                self.arm = None
                if self.fail == 'second-signal':
                    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                        os.kill(os.getpid(), sig)
            if self.fail == 'slow-restore':
                self.clock.now += timeout + 9
            if label == 'check' and self.fail == 'greedy':
                return 0, base.RESTORE_OK[label].replace('5 IDENTICAL', '4 IDENTICAL')
            if label == 'close':
                # Shared rm shim defers HOLD removal until commit-close.
                if self.fail == 'lost-close':
                    return 124, 'lost close reply'
            return 0, base.RESTORE_OK[label]
        if label == 'retain-hold':
            self.hold = 'UP3-WINDOW tag: failed restore'
            return 0, ''
        if label.startswith('verify-'):
            # Source/identity probes are unsupervised bounded recovery reads, never the closed Supervisor.
            probe = label.removeprefix('verify-')
            command = argv[-1]
            if '-sources-' in probe:
                return 0, json.dumps(w.source_pins('control'))
            if probe.startswith('restored-identity-'):
                ident = self.identity(pf.HOSTS.index(argv[-2]))
                return 0, json.dumps({'id': ident['id'], 'state': ident['state'], 'image': ident['image']}) + '\n' + json.dumps({
                    'engine': ident['engine'], 'files': ident['files'], 'mounted_code_sha256': ident['mounted'],
                    'generation_config': {'exists': False, 'sha256': None}})
            if probe.startswith(('restored-config-', 'restored-fullenv-')):
                return 0, json.dumps({'Env': [f'{k}={v}' for k, v in record(self.identity(0)['env']).items()] + ['HOME=/root']})
            if probe.startswith('restored-engram-'):
                return 0, json.dumps(self.identity(0)['engram_identity'])
            if probe.startswith('restored-argv-'):
                return 0, str(self.identity(pf.HOSTS.index(argv[-2]))['argv'])
            raise AssertionError((probe, command))
        raise AssertionError((label, argv))


@pytest.fixture
def rehearsal(tmp_path: Path, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[w.Live, Remote]]:
    fail = str(getattr(request, 'param', ''))
    monitor = tmp_path / 'monitor.py'
    monitor.write_text(base.MONITOR)
    clock = base.Clock()
    out = tmp_path / 'out'
    out.mkdir()
    sup = base.NoTunnel(out, clock=clock, monitor=lambda host: [os.environ.get('UP3_TEST_PYTHON', '/home/operator/miniconda3/bin/python3'), str(monitor), 'low', str(tmp_path / 'trip')], demand=lambda: 0.)
    remote = Remote(out, sup, clock, fail, trip=tmp_path / 'trip')
    live = w.Live(out, sup, group='k5',
        remote='dsv41-deploy-offload', expected='a' * 40, data_dir=tmp_path, tag='tag', call=remote.call, plain=remote.plain, clock=clock)
    remote.live = live
    import offload.prefill_trace.window_requests as wr
    monkeypatch.setattr(greedy_driver, 'greedy_inputs', lambda: (['public']*6, [{'i': i, 'text': 'public'} for i in range(6)]))
    monkeypatch.setattr(wr, 'corpus', lambda root: ('public fixture', {}))
    monkeypatch.setattr(pf, 'CAMPAIGN', tmp_path)
    (tmp_path / 'D2-GEN.PAUSE').touch()
    try:
        yield live, remote
    finally:
        sup.close()


def test_full_live_sequence(rehearsal: tuple[w.Live, Remote]) -> None:
    live, fake = rehearsal
    result = w.window(live, 'k5', lambda row: None)
    assert 'stop' not in result, result.get('stop')
    assert str(record(result['restore'])['status']).startswith('RESTORED'), result
    assert result['status'] == 'COMPLETE'
    assert record(result['decision'])['status'] == 'CANDIDATE FOR REVIEW'
    assert fake.restores == ['start', 'check', 'close']
    assert not fake.hold
    assert list(record(result['starts'])) == [a.name for a in w.arms('k5')]
    assert fake.commands.index('owned-absent-tag-s1-control-engine-sparkworker-wired') < fake.commands.index('start')
    assert fake.commands.index('owned-absent-tag-restore-start-2-sparkworker-wired') < fake.commands.index('check')
    assert fake.commands.index('owned-absent-tag-restore-check-3-sparkworker-wired') < fake.commands.index('close')
    deadlines = json.loads((live.out / 'deadlines.json').read_text())
    assert deadlines['restore_by'] - deadlines['opened'] == 85 * 60


@pytest.mark.parametrize('rehearsal', ['start', 'gate', 'experiment-greedy', 'signal', 'deadline', 'hold-open', 'STOP', 'memory', 'unfiltered-env'], indirect=True)
def test_stop_then_restore(rehearsal: tuple[w.Live, Remote]) -> None:
    live, fake = rehearsal
    if fake.fail == 'gate':
        live.group = 'dual-ring'
    result = w.window(live, live.group, lambda row: None)
    assert 'stop' in result
    assert str(record(result['restore'])['status']).startswith('RESTORED'), result
    assert record(result['decision'])['status'] == 'NOT A CANDIDATE'
    if fake.fail == 'gate':
        assert 's2-dual-ring-short-b1' not in fake.commands
    assert fake.restores[-2:] == ['check', 'close']


@pytest.mark.parametrize('rehearsal', ['cleanup', 'jit-wait-sparkhead', 'jit-wait-sparkworker-wired', 'greedy', 'slow-restore'], indirect=True)
def test_failed_cleanup_or_proof_keeps_hold(rehearsal: tuple[w.Live, Remote]) -> None:
    live, fake = rehearsal
    result = w.window(live, 'k5', lambda row: None)
    assert record(result['restore'])['status'] == 'FAILED', result
    assert 'close' not in fake.restores
    assert fake.hold
    assert record(result['decision'])['status'] == 'NOT A CANDIDATE'


def test_parked_default_refuses_before_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('WINDOW_COORDINATOR', 'Deepseek/coordinator')
    monkeypatch.setattr(pf, 'CAMPAIGN', tmp_path)
    assert w.main(['--execute', '--out', str(tmp_path / 'fresh'), '--expected-experiment-commit', 'a' * 40]) == 1
    assert not (tmp_path / 'fresh').exists()


@pytest.mark.parametrize('rehearsal', ['second-signal'], indirect=True)
def test_second_signal_cannot_abandon_recovery(rehearsal: tuple[w.Live, Remote]) -> None:
    live, fake = rehearsal
    result = w.window(live, 'k5', lambda row: None)
    assert str(record(result['restore'])['status']).startswith('RESTORED'), result
    assert fake.restores == ['start', 'check', 'close']


@pytest.mark.parametrize('rehearsal', ['lost-close', 'owned-reap-tag-restore-close-sparkworker-wired'], indirect=True)
def test_ambiguous_close_recreates_hold(rehearsal: tuple[w.Live, Remote]) -> None:
    live, fake = rehearsal
    result = w.window(live, 'k5', lambda row: None)
    assert record(result['restore'])['status'] == 'FAILED', result
    assert fake.hold
    assert 'commit-close' not in fake.commands
    assert record(result['decision'])['status'] == 'NOT A CANDIDATE'


@pytest.mark.parametrize('text', [
    base.RESTORE_OK['check'].replace('5 IDENTICAL', '4 IDENTICAL'),
    base.RESTORE_OK['check'].replace('5 IDENTICAL', '5 DIVERGES'),
    base.RESTORE_OK['check'].replace('greedy K1-post vs restored: 5 IDENTICAL 105\n', ''),
    base.RESTORE_OK['check'] + 'greedy K1-post vs restored: 5 IDENTICAL 105\n'])
def test_six_greedy_lines_are_distinct_and_complete(text: str) -> None:
    with pytest.raises(ValueError, match='6/6'):
        w.strict_restored('check', 0, text)


def test_restore_rejects_embedded_or_suffixed_identity_claims() -> None:
    w.strict_restored('check', 0, base.RESTORE_OK['check'])
    for row in ('prefix greedy K1-post vs restored: 5 IDENTICAL 105', 'greedy K1-post vs restored: 5 IDENTICAL 105 garbage'):
        text = base.RESTORE_OK['check'].replace('greedy K1-post vs restored: 5 IDENTICAL 105', row)
        with pytest.raises(ValueError, match='six distinct'):
            w.strict_restored('check', 0, text)


def test_restore_logs_survive_repeated_phase_fence(tmp_path: Path) -> None:
    # TV1-COMPACT-W-20261027T031304Z: the post-close fence pass repeats each owned phase's
    # label; exclusive unnumbered logs raised FileExistsError and kept HOLD.
    run = w.plain_owned(tmp_path)
    label = 'owned-fence-UP3-k5-W-x-hold-sparkhead'
    assert run(label, ['bash', '-c', 'echo first'], 10) == (0, 'first\n')
    assert run(label, ['bash', '-c', 'echo second'], 10) == (0, 'second\n')
    assert sorted(p.name for p in (tmp_path / 'cmd').iterdir()) == [
        f'restore-0001-{label}.log', f'restore-0002-{label}.log']


def test_control_pins_match_the_mounted_remote_dspark_package() -> None:
    # Production mounts only remote_dspark/*.py (offload/dspark/spark/stage_overlay.sh); the lock also records hostflag.c.
    lock = json.loads((b.HERE / 'inputs.json').read_text())['inputs']
    mounted = {b.DIST + 'vllm/v1/worker/gpu/spec_decode/remote_dspark/' + key.rsplit('/', 1)[1]: sha
               for key, sha in lock.items()
               if key.startswith('git:offload/dspark/spark/remote_dspark/') and key.endswith('.py')}
    assert any(key.endswith('/remote_dspark/hostflag.c') for key in lock)
    for arm in ('control', 'k5', 'dual-ring', 'cpu-pin'):
        pins = w.source_pins(arm)
        package = {k: v for k, v in pins.items() if '/spec_decode/remote_dspark/' in k}
        assert package == mounted


def replay_starts(gain: float = .4, *, registration: str = cells.REPLAY_REGISTRATION,
                  ttft: float = 1.) -> dict[str, Json]:
    """Fixture starts; `ttft` scales the replay arm's TTFT in every decision cell (below 1 = faster)."""
    rows: dict[str, Json] = {}
    for arm in w.arms(w.REPLAY):
        greedy = {**greedy_result(), 'greedy_registration': registration, 'near_tie_nats': cells.REPLAY_NEAR_TIE_NATS}
        measured_cells = {name: measured(name, gain if arm.role != 'control' else 0, salt=arm.name) for name in w.REPLAY_CELLS}
        for name in w.REPLAY_DECISION_CELLS:
            for k, repeat in enumerate(cast(list[Json], record(measured_cells[name])['repetitions'])):
                stream = record(cast(list[Json], record(repeat)['streams'])[0])
                # Real cold TTFT varies run to run: give the controls a 1 % repeat-to-repeat spread.
                stream['ttft_s'] = (1 + .01 * (k % 2)) * (ttft if arm.role == w.REPLAY else 1.)
        gates_row: Json = ({'status': 'PASS', 'registration': w.REPLAY_REGISTRATION} if arm.role == w.REPLAY else
            {'status': 'x', 'greedy_kl': {'status': 'PASS', 'registration': w.REPLAY_REGISTRATION}} if arm.name == 's3-control'
            else {'status': 'x'})
        row: dict[str, Json] = {'status': 'COMPLETE', 'role': arm.role, 'greedy': greedy, 'gates': gates_row,
            'cells': cast(Json, measured_cells)}
        if arm.role == 'control':
            row['checks'] = {'archive': {'clock_cap_mhz': 2300, 'clock_source': 'start-log'}}
        rows[arm.name] = row
    return rows


def test_replay_runtime_isolates_one_switch_and_the_reviewed_files() -> None:
    for arm in w.arms(w.REPLAY):
        w.runtime(identity(None), identity(arm), arm)
    arm = w.arms(w.REPLAY)[1]
    assert b.ENV[arm.role] == {'VLLM_DSV41_REPLAY_SKIP': '1'}
    pins = w.source_pins(w.REPLAY)
    assert pins[b.DIST + b.REMOTE + 'suffix_provenance.py'] == b.verify(w.REPLAY)['files'][b.REMOTE + 'suffix_provenance.py']
    assert {k for k in pins if '/remote_dspark/' in k} == {b.DIST + b.REMOTE + n for n in (*b.REMOTE_PACKAGE, 'suffix_provenance.py')}
    # Production's remote package stays the control pin set; the replay arm replaces it whole.
    assert {k for k in w.source_pins('control') if '/remote_dspark/' in k} == {b.DIST + b.REMOTE + n for n in b.REMOTE_PACKAGE}
    live = identity(arm)
    record(live['mounted'])[b.DIST + b.REMOTE + 'suffix_provenance.py'] = '0' * 64
    with pytest.raises(ValueError, match='mounted source drift'):
        w.runtime(identity(None), live, arm)
    live = identity(arm)
    record(live['env'])['VLLM_DSV41_REPLAY_SKIP'] = '0'
    with pytest.raises(ValueError, match='env drift'):
        w.runtime(identity(None), live, arm)


def test_replay_decision_is_win2_no_loss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pf, 'CAMPAIGN', tmp_path)
    same = w.decision(w.REPLAY, replay_starts())
    # WIN2 checklist 3: the summary decision IS decide_window's dict, with the adoption verdict beside it.
    assert same['adopt'] == 'ADOPT CANDIDATE' and same['status'] == 'INCONCLUSIVE'
    assert same['rule'] == 'win2-v1' and same['decision_cells'] == ['150k-b1', '32k-b1']
    assert record(record(same['cells'])['short-b1'])['decides'] is False
    assert same['registration'] == 'up2-0075-adopt-0p9'
    assert same['replay_files'] == b.verify(w.REPLAY)['files']
    assert same['archive_spread_max_tokens_512'] == {'150k-b1/ttft_s': None, '150k-b1/ms_per_round': None,
                                                     '32k-b1/ttft_s': None, '32k-b1/ms_per_round': None}
    assert same['archive_spread_where'] == {'max_tokens': cells.BUDGET, 'commit': b.PRODUCTION, 'clock_cap_mhz': 2300}
    # Controls that proved different clock caps have no single archive definition: the spread is reported unreadable.
    mixed = replay_starts()
    record(record(record(mixed['s3-control'])['checks'])['archive'])['clock_cap_mhz'] = 2150
    spreads = record(w.decision(w.REPLAY, mixed)['archive_spread_max_tokens_512'])
    assert all(str(v).startswith('UNREADABLE: STOP: control starts prove no single clock cap') for v in spreads.values())
    faster = w.decision(w.REPLAY, replay_starts(ttft=.95))
    assert faster['adopt'] == 'ADOPT CANDIDATE' and faster['status'] == 'CANDIDATE'
    slower = w.decision(w.REPLAY, replay_starts(ttft=1.03))
    assert slower['adopt'] == 'NOT A CANDIDATE' and slower['status'] == 'NOT A CANDIDATE'
    # A reported-only short-cell loss never decides.
    rows = replay_starts()
    for repeat in cast(list[Json], record(record(record(rows['s2-replay'])['cells'])['short-b1'])['repetitions']):
        record(repeat)['server_decode_ms'] = 600000000
    assert w.decision(w.REPLAY, rows)['adopt'] == 'ADOPT CANDIDATE'
    # The ms/round guard can turn a decision cell into a LOSS.
    rows = replay_starts()
    for repeat in cast(list[Json], record(record(record(rows['s2-replay'])['cells'])['150k-b1'])['repetitions']):
        record(repeat)['server_decode_ms'] = 500000000
    assert w.decision(w.REPLAY, rows)['adopt'] == 'NOT A CANDIDATE'
    rows = replay_starts()
    record(rows['s2-replay'])['gates'] = {'status': 'unchanged production source/env verified'}
    assert w.decision(w.REPLAY, rows)['adopt'] == 'NOT A CANDIDATE'
    rows = replay_starts()
    record(rows['s3-control'])['gates'] = {'status': 'unchanged production source/env verified'}
    assert w.decision(w.REPLAY, rows)['adopt'] == 'NOT A CANDIDATE'
    rows = replay_starts()
    rows.pop('s3-control')
    assert w.decision(w.REPLAY, rows)['adopt'] == 'NOT A CANDIDATE'
    rows = replay_starts()
    record(record(record(rows['s2-replay'])['cells'])['150k-b1'])['repetitions'] = cast(list[Json], record(record(
        record(rows['s2-replay'])['cells'])['150k-b1'])['repetitions'])[:2]
    with pytest.raises(ValueError, match='repeat count differs'):
        w.decision(w.REPLAY, rows)
    with pytest.raises(ValueError, match='registration/threshold'):
        w.decision(w.REPLAY, replay_starts(registration='nbw-0p5'))


def test_replay_hold_fits_and_other_groups_unchanged() -> None:
    assert w.budget_minutes(w.REPLAY) <= 85
    assert all(w.budget_minutes(g) == pytest.approx(83.95) for g in gates.CASES)
    control, replay, last = w.arms(w.REPLAY)
    assert w.block_s(w.REPLAY, control, True) == w.PREPARE_S + w.AA_S + w.REPLAY_CELL_S
    assert w.block_s(w.REPLAY, replay, False) == w.PREPARE_S + w.REPLAY_GATES_S + w.REPLAY_CELL_S
    assert w.block_s(w.REPLAY, last, False) == w.PREPARE_S + w.GREEDY_KL_S + w.REPLAY_CELL_S
    assert w.NUMERIC_TARGETS == (32000, 150000)
    text = w.plan(w.REPLAY)
    assert 'up2-0075-adopt-0p9' in text and '0.901nat' in text and 'NOT RUN' in text
    with pytest.raises(ValueError):
        w.arms('core')


@pytest.fixture
def replay_rehearsal(tmp_path: Path, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[w.Live, Remote]]:
    fail = str(getattr(request, 'param', ''))
    monitor = tmp_path / 'monitor.py'
    monitor.write_text(base.MONITOR)
    clock = base.Clock()
    out = tmp_path / 'out'
    out.mkdir()
    sup = base.NoTunnel(out, clock=clock, monitor=lambda host: [os.environ.get('UP3_TEST_PYTHON', '/home/operator/miniconda3/bin/python3'), str(monitor), 'low', str(tmp_path / 'trip')], demand=lambda: 0.)
    remote = Remote(out, sup, clock, fail, trip=tmp_path / 'trip')
    live = w.Live(out, sup, group=w.REPLAY,
        remote='dsv41-deploy-offload', expected='a' * 40, data_dir=tmp_path, tag='tag', call=remote.call, plain=remote.plain, clock=clock)
    remote.live = live
    import offload.prefill_trace.window_requests as wr
    monkeypatch.setattr(greedy_driver, 'greedy_inputs', lambda: (['public']*6, [{'i': i, 'text': 'public'} for i in range(6)]))
    monkeypatch.setattr(wr, 'corpus', lambda root: ('public fixture', {}))
    monkeypatch.setattr(pf, 'CAMPAIGN', tmp_path)
    (tmp_path / 'D2-GEN.PAUSE').touch()
    try:
        yield live, remote
    finally:
        sup.close()


def test_replay_full_live_sequence(replay_rehearsal: tuple[w.Live, Remote]) -> None:
    live, fake = replay_rehearsal
    result = w.window(live, w.REPLAY, lambda row: None)
    assert 'stop' not in result, result.get('stop')
    assert str(record(result['restore'])['status']).startswith('RESTORED'), result
    assert result['status'] == 'COMPLETE'
    assert record(result['decision'])['adopt'] == 'ADOPT CANDIDATE'
    assert fake.restores == ['start', 'check', 'close']
    assert not fake.hold
    assert 's3-control-greedy-kl' in fake.commands
    probes = [c for c in fake.commands if 'numeric' in c or 'needle' in c]
    assert probes == ['s1-control-numeric32000-0', 's1-control-numeric32000-1', 's1-control-numeric150000-0',
                      's1-control-numeric150000-1', 's2-replay-numeric32000-0', 's2-replay-numeric150000-0',
                      's2-replay-needle150000']
    starts = record(result['starts'])
    for name in ('s1-control', 's2-replay', 's3-control'):
        assert list(record(record(starts[name])['cells'])) == list(w.REPLAY_CELLS)
        assert record(record(starts[name])['greedy'])['greedy_registration'] == 'up2-0075-adopt-0p9'
    gate = record(record(starts['s2-replay'])['gates'])
    assert gate['status'] == 'PASS' and set(record(gate['numerical'])) == {'32000', '150000'}
    assert record(record(record(starts['s3-control'])['gates'])['greedy_kl'])['bound'] == .005
    assert not any('-host-gate-' in c for c in fake.commands)
    # Every control cell repeat is archived (sha-chained) for the WIN2 day-to-day spread.
    from offload.launch.windows import decision_v2 as dv2
    archived = [json.loads(line) for line in (pf.CAMPAIGN / 'control-archive.jsonl').read_text().splitlines()]
    assert {(r['start'], r['cell']) for r in archived} == {(a, c) for a in ('s1-control', 's3-control') for c in w.REPLAY_CELLS}
    assert {r['max_tokens'] for r in archived} == {cells.BUDGET}
    assert len(archived) == 2 * sum(cells.REPEATS[c] * len(w.REPLAY_METRICS[c]) for c in w.REPLAY_CELLS)
    # Archive v3 (WIN2-R2): 40-hex PRODUCTION commit, the cap both Sparks printed in the row's own start log, YYYYMMDD.
    assert {(r['commit'], r['clock_cap_mhz'], r['clock_source']) for r in archived} == {(b.PRODUCTION, 2300, 'start-log')}
    assert {r['date'] for r in archived} == {time.strftime('%Y%m%d', time.gmtime(live.clock()))}
    assert all(len(r['date']) == 8 and isinstance(r['repeat'], int) and 'arm' not in r for r in archived)
    heads = [record(record(record(starts[a])['checks'])['archive']) for a in ('s1-control', 's3-control')]
    assert record(heads[0]['after']) == record(heads[1]['before']) and record(heads[0]['before'])['rows'] == 0
    assert record(heads[1]['after'])['rows'] == len(archived)
    assert dv2.archive_spread(pf.CAMPAIGN / 'control-archive.jsonl', '32k-b1', 'ttft_s') is None   # one window only
    assert record(result['decision'])['archive_spread_where'] == {'max_tokens': cells.BUDGET, 'commit': b.PRODUCTION,
                                                                  'clock_cap_mhz': 2300}


@pytest.mark.parametrize('replay_rehearsal', ['kl', 'needle', 'provenance', 'greedy-kl'], indirect=True)
def test_replay_gate_failure_stops_and_restores(replay_rehearsal: tuple[w.Live, Remote]) -> None:
    live, fake = replay_rehearsal
    result = w.window(live, w.REPLAY, lambda row: None)
    assert 'stop' in result
    assert {'kl': 'numerical floor exceeded', 'needle': 'needle 6/6', 'provenance': 'new provenance_errors',
            'greedy-kl': 'registered KL bound'}[fake.fail] in str(result['stop'])
    archived = pf.CAMPAIGN / 'control-archive.jsonl'
    # Only FINISHED control starts are archived: s1 always finishes before the replay arm's gates.
    assert {json.loads(line)['start'] for line in archived.read_text().splitlines()} == {'s1-control'}
    assert str(record(result['restore'])['status']).startswith('RESTORED'), result
    assert record(result['decision'])['status'] == 'NOT A CANDIDATE'
    assert fake.restores[-2:] == ['check', 'close']
    if fake.fail != 'greedy-kl':   # the KL re-probe runs on the last control, after the replay arm's cells
        assert not any(c.startswith('s2-replay-') and c.endswith(w.REPLAY_CELLS) for c in fake.commands)


def test_replay_window_chains_like_every_up3_group(replay_rehearsal: tuple[w.Live, Remote],
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    # WIN2 chain (r-up3-fork 1530c9c) is group-agnostic: a clean non-last replay window leaves production DOWN.
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    live, fake = replay_rehearsal
    result = w.window(live, w.REPLAY, lambda row: None)
    assert 'stop' not in result, result.get('stop')
    assert str(record(result['restore'])['status']).startswith('CHAINED') and result['chain'] == 'CHAINED'
    assert record(result['decision'])['adopt'] == 'ADOPT CANDIDATE' and fake.restores == []
    assert chain.read_marker() is not None
    archived = [json.loads(line) for line in (pf.CAMPAIGN / 'control-archive.jsonl').read_text().splitlines()]
    assert {r['start'] for r in archived} == {'s1-control', 's3-control'}


@pytest.mark.parametrize('replay_rehearsal', ['needle'], indirect=True)
def test_replay_stop_inside_a_chain_runs_the_strict_restore(replay_rehearsal: tuple[w.Live, Remote],
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    live, fake = replay_rehearsal
    result = w.window(live, w.REPLAY, lambda row: None)
    assert 'stop' in result and str(record(result['restore'])['status']).startswith('RESTORED'), result
    assert 'chain' not in result and chain.read_marker() is None
    assert record(result['decision'])['status'] == 'NOT A CANDIDATE'


def slower_starts(cells_slow: tuple[str, ...]) -> dict[str, Json]:
    """S97 CX2 values: 1 % slower cold TTFT in every repeat of the named decision cells, inside the control spread."""
    slow = {'150k-b1': ([68.6, 69.0, 69.4], [69.3, 69.7, 70.1]),
            '32k-b1': ([14.4, 14.6, 14.8, 14.9], [14.55, 14.75, 14.95, 15.05])}
    rows = replay_starts()
    for arm in w.arms(w.REPLAY):
        for name in cells_slow:
            control, replay = slow[name]
            values = replay if arm.role == w.REPLAY else control
            for repeat, value in zip(cast(list[Json], record(record(record(rows[arm.name])['cells'])[name])['repetitions']), values):
                for stream in cast(list[Json], record(repeat)['streams']):
                    record(stream)['ttft_s'] = value
    return rows


def test_replay_adopt_refuses_a_significantly_slower_cold_ttft(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # S97 CX2 (inverted): both cells INCONCLUSIVE (no repeat beyond the control spread, so no LOSS), but each whole 95 %
    # interval is below zero: never an ADOPT CANDIDATE.
    monkeypatch.setattr(pf, 'CAMPAIGN', tmp_path)   # empty archive: the day-to-day spread is reporting only
    decided = w.decision(w.REPLAY, slower_starts(('150k-b1', '32k-b1')))
    for name in ('150k-b1', '32k-b1'):
        cell = record(record(record(record(decided['cells'])[name])['metrics'])['ttft_s'])
        assert cast(list[float], cell['interval95'])[1] < 0 and cell['verdict'] == 'INCONCLUSIVE', cell
    assert decided['status'] == 'INCONCLUSIVE' and decided['significantly_slower'] == ['150k-b1', '32k-b1']
    assert decided['adopt'] == 'NOT A CANDIDATE'
    # One significantly slower decision cell is enough; equal TTFTs adopt with nothing listed.
    one = w.decision(w.REPLAY, slower_starts(('32k-b1',)))
    assert one['significantly_slower'] == ['32k-b1'] and one['adopt'] == 'NOT A CANDIDATE'
    same = w.decision(w.REPLAY, replay_starts())
    assert same['significantly_slower'] == [] and same['adopt'] == 'ADOPT CANDIDATE'


def test_replay_provenance_counters_are_checked_from_zero_on_every_start(replay_rehearsal: tuple[w.Live, Remote]) -> None:
    # S97 CX1 (inverted): a provenance error during the replay start's engine start, greedy or OPS1 warm-up (before the
    # post-warm-up baseline) used to be absorbed into that baseline and the window completed as ADOPT CANDIDATE.
    live, fake = replay_rehearsal
    original = live.warmup

    def warmup(arm: w.Arm) -> None:
        if arm.role == w.REPLAY:
            fake.provenance = 1
        original(arm)
    live.warmup = warmup  # type: ignore[method-assign]
    result = w.window(live, w.REPLAY, lambda row: None)
    assert 'new provenance_errors=1' in str(result['stop'])
    assert record(record(result['starts'])['s2-replay'])['status'] == 'STARTED'   # stopped at its baseline, no cells
    assert record(result['decision'])['status'] == 'NOT A CANDIDATE' and 'adopt' not in record(result['decision'])
    assert str(record(result['restore'])['status']).startswith('RESTORED') and fake.restores == ['start', 'check', 'close']


def test_replay_archive_refuses_a_start_without_one_proven_clock_cap(replay_rehearsal: tuple[w.Live, Remote]) -> None:
    # Archive v3 (WIN2-R2): a control start whose log does not show ONE cap on both Sparks stops the window (its
    # comparison would mix clocks); nothing is archived and production is restored strictly.
    live, fake = replay_rehearsal
    fake.clock_caps = {'spark': 2300, 'sparkworker': 2150}
    result = w.window(live, w.REPLAY, lambda row: None)
    assert 'does not prove one clock cap on both Sparks' in str(result['stop'])
    assert not (pf.CAMPAIGN / 'control-archive.jsonl').exists()
    assert str(record(result['restore'])['status']).startswith('RESTORED') and fake.restores == ['start', 'check', 'close']
