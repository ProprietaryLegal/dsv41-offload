"""CPU-only runtime, decision and complete real-Live rehearsal. Every transport and GPU operation is fake."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shlex
import signal
import time
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from offload.pc1 import window as w
from offload.launch.windows import pf1_window as pf
from offload.launch.windows.tests import test_pf1_window as base
from offload.prefill_maxlen.evidence import Json, record, number
from offload.pc1 import build as b, cells, host_gates, capacity
from offload.launch.windows import chain, scoped_remote as owner, nbw_cells as greedy_driver


def identity(arm: w.Arm | None, rank: int = 0, serial: int = 0) -> dict[str, Json]:
    prod = base.prod_identity(rank)
    prod['argv'] = (b.HERE / 'tests/fixtures' / f'production-serve-rank{rank}.sh').read_text()
    if arm is None:
        return prod
    live = copy.deepcopy(prod)
    live['id'] = arm.name + str(rank) + str(serial)
    script = str(live['argv']).replace('--port 8000', '--port 8001')
    for flag, old, value in zip(('--max-num-batched-tokens', '--long-prefill-token-threshold'),
                               b.SETTINGS['control'], b.SETTINGS[arm.role]):
        script = script.replace(f'{flag} {old}', f'{flag} {value}')
    live['argv'] = script
    return live


def measured(name: str, gain: float = 0, salt: str = 'fixture', repeats: int = cells.DECISION_REPEATS,
             first: int = 0) -> dict[str, Json]:
    batch = cells.batch_for(name)
    rows: list[Json] = []
    for repeat in range(first, first + repeats):
        # Repeat-to-repeat drift shared by every arm: a nonzero control spread, unchanged paired gains.
        drift = 1 + repeat / 1000
        streams: list[Json] = []
        for i in range(batch):
            n = 1024 if name == 'short-b1' else 150000 if name == '150k-b1' or name == cells.CELLS[-1] and i == 2 else 32000
            streams.append({'prompt_sha256': str(i) * 64, 'prompt_tokens': n, 'budget': 512,
                'decode_tps': (10.0 + gain) / drift, 'ttft_s': (1. - gain / 10) * drift,
                'total_s': (1. - gain / 10 + 511 / (10.0 + gain)) * drift,
                'p95_inter_token_delivery_gap_ms': 100., 'p99_inter_token_delivery_gap_ms': 100.,
                'finish': 'length', 'delivery_gap_samples': 511, 'usage': {'completion_tokens': 512, 'prompt_tokens': n,
                    'prompt_tokens_details': {'cached_tokens': 0}},
                'sampling': {'temperature': 1, 'top_p': 1, 'seed': 42+repeat, 'priority': 10, 'max_tokens': 512},
                'cache_salt_sha256': hashlib.sha256(f'{salt}-{name}-{repeat}-{i}'.encode()).hexdigest()})
        rows.append({'repeat': repeat, 'batch': batch, 'streams': streams,
            'spec_request_rounds': 1000000, 'accepted_tokens': round(3000000 + gain*400000),
            'server_decode_ms': 400000000 * drift * (1 - gain / 10), 'completed_requests': batch,
            'mixed_three_active_overlap': name == cells.CELLS[-1]})
    return {'cell': name, 'batch': batch, 'repetitions': rows}


def greedy_result() -> dict[str, Json]:
    return {'checks': [{'i': i, 'identical': True} for i in range(6)], 'identical': 6, 'near_tie': 0,
            'policy': cells.GREEDY_POLICY, 'near_tie_nats': cells.GREEDY_NEAR_TIE_NATS}


def starts(group: str, gain: float = .4) -> dict[str, Json]:
    plan = cells.plan_for(group)
    return {arm.name: {'status': 'COMPLETE', 'role': arm.role, 'greedy': greedy_result(),
        'cells': {name: measured(name, gain if arm.role != 'control' else 0, salt=arm.name, repeats=plan.repeats(name))
                  for name in plan.cells}}
        for arm in w.arms(group)}


@pytest.mark.parametrize('group', b.ARMS[1:])
def test_current_runtime_and_decision(group: str) -> None:
    for arm in w.arms(group):
        w.runtime(identity(None), identity(arm), arm)
        live = identity(arm)
        record(live['env'])['VLLM_ENGRAM_REMOTE_REQUIRED'] = '1'
        with pytest.raises(ValueError, match='env drift'):
            w.runtime(identity(None), live, arm)
    assert w.decision(group, starts(group))['status'] == 'CANDIDATE'
    # A 1 % gain is real but under the 2 % floor: INCONCLUSIVE under win2-v1, never a loss.
    assert w.decision(group, starts(group, .1))['status'] == 'INCONCLUSIVE'
    assert w.decision(group, starts(group, .201))['status'] == 'CANDIDATE'
    rows = starts(group)
    rows.pop(next(reversed(rows)))
    assert w.decision(group, rows)['status'] == 'NOT A CANDIDATE'


@pytest.mark.parametrize('field', ['image', 'engine', 'files', 'engram_identity', 'mounted', 'argv'])
def test_drift_is_rejected(field: str) -> None:
    arm = w.arms('chunk2048')[1]
    live = identity(arm)
    live[field] = 'changed' if field in ('image', 'engine', 'argv') else {}
    with pytest.raises(ValueError):
        w.runtime(identity(None), live, arm)


def worsen(rows: dict[str, Json], arm: str, name: str, repeats: int, *, metric: str = 'ttft_s',
           factor: float = 1.5) -> None:
    cell = record(record(record(rows[arm])['cells'])[name])
    for repeat in cast(list[Json], cell['repetitions'])[:repeats]:
        if metric == 'server_decode_ms':
            record(repeat)[metric] = number(record(repeat)[metric]) * factor
            continue
        for stream in cast(list[Json], record(repeat)['streams']):
            record(stream)[metric] = number(record(stream)[metric]) * factor


def test_decision_cell_loss_needs_two_of_three_repeats() -> None:
    # win2-v1: one slow repeat is noise (INCONCLUSIVE); the same slowdown in 2 of 3 repeats is a LOSS.
    rows = starts('chunk2048', 0)
    worsen(rows, 's2-chunk2048', '32k-b1', 1)
    assert w.decision('chunk2048', rows)['status'] == 'INCONCLUSIVE'
    rows = starts('chunk2048', 0)
    worsen(rows, 's2-chunk2048', '32k-b1', 2)
    assert w.decision('chunk2048', rows)['status'] == 'NOT A CANDIDATE'
    # A LOSS on one decision cell vetoes a WIN on the other.
    rows = starts('chunk2048')
    worsen(rows, 's2-chunk2048', '32k-b1', 2)
    result = w.decision('chunk2048', rows)
    assert result['status'] == 'NOT A CANDIDATE'
    verdicts = {c: record(v)['verdict'] for c, v in record(result['cells']).items() if record(v)['decides']}
    assert verdicts == {'32k-b1': 'LOSS', '150k-b1': 'WIN'}


def test_reported_cells_are_reported_never_decide() -> None:
    rows = starts('chunk2048')
    worsen(rows, 's2-chunk2048', 'mixed150k-2decode', 1, metric='p95_inter_token_delivery_gap_ms', factor=3)
    worsen(rows, 's2-chunk2048', 'short-b1', 1, metric='server_decode_ms', factor=2)
    # Decode metrics of a prefill decision cell are reported too.
    worsen(rows, 's2-chunk2048', '150k-b1', 3, metric='server_decode_ms', factor=2)
    result = w.decision('chunk2048', rows)
    assert result['status'] == 'CANDIDATE'
    reported = record(record(result['cells'])['mixed150k-2decode'])
    assert reported['decides'] is False and reported['verdict'] == 'REPORTED'
    gap = record(record(reported['metrics'])['decoders_p95_gap_ms'])
    assert gap['verdict'] == 'REPORTED' and number(gap['point_gain']) < -1
    server = record(record(result['cells'])['150k-b1:reported'])
    assert server['decides'] is False and number(record(record(server['metrics'])['ms_per_round'])['point_gain']) < -.5


@pytest.mark.parametrize('group', ['threshold2048', 'threshold512'])
def test_threshold_single_and_multi_user_verdicts_are_separate(group: str) -> None:
    arm = w.arms(group)[1].name
    rows = starts(group)
    worsen(rows, arm, 'short-b1', 2, metric='server_decode_ms')
    result = w.decision(group, rows)
    assert result['status'] == 'NOT A CANDIDATE'
    assert result['groups'] == {'single-user': 'NOT A CANDIDATE', 'multi-user': 'CANDIDATE'}
    short = record(record(record(result['cells'])['short-b1'])['metrics'])
    assert record(short['ms_per_round'])['verdict'] == 'LOSS'
    assert record(record(record(result['cells'])['short-b1'])['metrics']) == short and result['decision_cells'] == [
        'short-b1', 'mixed150k-2decode']
    # The mixed cell's 150K first-token guard vetoes a multi-user ms/round WIN.
    rows = starts(group)
    worsen(rows, arm, 'mixed150k-2decode', 3, metric='ttft_s', factor=1.3)
    result = w.decision(group, rows)
    mixed = record(record(result['cells'])['mixed150k-2decode'])
    assert record(record(mixed['metrics'])['ms_per_round'])['verdict'] == 'WIN'
    assert record(record(mixed['metrics'])['long_doc_ttft_s'])['verdict'] == 'LOSS' and mixed['verdict'] == 'LOSS'
    assert result['groups'] == {'single-user': 'CANDIDATE', 'multi-user': 'NOT A CANDIDATE'}
    # Acceptance-driven client tok/s noise in ONE repeat cannot veto an ms/round WIN (2 of 3 needed).
    rows = starts(group)
    worsen(rows, arm, 'short-b1', 1, metric='decode_tps', factor=.7)
    assert record(record(w.decision(group, rows)['cells'])['short-b1'])['verdict'] == 'WIN'


def test_decision_reports_brackets_rule_and_plan() -> None:
    result = w.decision('chunk16384', starts('chunk16384'))
    assert result['rule'] == 'win2-v1' and result['decision_cells'] == ['32k-b1', '150k-b1']
    assert result['groups'] == {'single-user': 'CANDIDATE'}
    bracket = record(record(result['brackets'])['150k-b1/ttft_s'])
    assert len(cast(list[Json], bracket['candidate'])) == 3
    assert all(number(v) == 0 for v in cast(list[Json], bracket['bracket_spread_per_repeat']))
    reported = record(record(result['brackets'])['short-b1/committed_tps'])
    assert len(cast(list[Json], reported['candidate'])) == 1


class Remote(base.FakeRemote):
    live: w.Live
    production_down = False  # a chained start: nothing answers on :8000
    placeholders_down = False
    hold_open_text = ''
    aborts: list[Path]

    def identity(self, rank: int) -> dict[str, Json]:
        return identity(cast(w.Arm | None, self.arm), rank, self.serial)

    def remote(self, label: str, host: str, command: str, stdin: str | None) -> str:
        if label.startswith('owned-'):
            return owner.FENCED if label.startswith('owned-fence-') else owner.ABSENT
        if label == 'prod-probe' and self.production_down:
            return ''
        if label.startswith('chain-quiet-'):
            return 'PC1 CHAIN QUIET\n'
        if label.startswith('placeholders-'):
            assert chain.PLACEHOLDER_PROBE in command
            return 'PLACEHOLDER ABSENT\n' if self.placeholders_down else 'PLACEHOLDER PRESENT\n'
        if label == 'hold-read':
            return self.hold
        if label == 'hold-open':
            self.hold_open_text = command + (stdin or '')
            self.hold = f'{base.iso(self.clock())} DSV41-OFFLOAD (session coordinator): PC1-WINDOW {self.live.tag}: isolated chunk2048'
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
            return ''
        if '-sources-' in label:
            role = label.split('-sources-')[0]
            return json.dumps(w.source_pins(role))
        if '-host-gate-' in label:
            rank = pf.HOSTS.index(host)
            script = str(self.identity(rank)['argv'])
            if self.fail == 'gate':
                script = script.replace('--long-prefill-token-threshold 1024', '--long-prefill-token-threshold 8192')
            result: dict[str, Json] = {'mode': 'experiment-settings', 'status': 'PARSE COMPLETE',
                'arm': self.live.group, 'rank': rank, 'container': self.live.ids[rank], 'identity_rechecked': True,
                'serve': host_gates.settings(script, rank, '8001', self.live.group)}
            return json.dumps(result)
        if '-logs-' in label:
            summary = base.log_summary(327680)
            if self.arm is not None:
                # Source-derived capacities; only chunk2048/control are captured hardware values.
                summary['kv_tokens'] = [capacity.expectation(cast(w.Arm, self.arm).role)['expected_tokens']]
            if self.arm is not None and cast(w.Arm, self.arm).role == self.live.group:
                if self.fail == 'capacity-wrong':
                    summary['kv_tokens'] = [2502600 if self.live.group == 'chunk2048' else 2933321]
                elif self.fail == 'capacity-empty':
                    summary['kv_tokens'] = []
                elif self.fail == 'capacity-mixed':
                    summary['kv_tokens'] = [2502600, 2933321]
            if self.fail == 'capacity-control' and self.arm is not None and cast(w.Arm, self.arm).role == 'control':
                summary['kv_tokens'] = [2933321]
            if host == 'sparkworker-wired':
                summary['kv_tokens'] = []  # Capacity is logged only by rank0.
            return json.dumps(summary)
        if self.fail == 'unfiltered-env' and '{{json .Config}}' in command and self.arm is not None and cast(w.Arm, self.arm).role == 'chunk2048':
            env_result = json.loads(super().remote(label, host, command, stdin))
            env_result['Env'].append('OMP_NUM_THREADS=999')
            return json.dumps(env_result)
        return super().remote(label, host, command, stdin)

    def call(self, label: str, argv: list[str], *, stdin: str | None = None, timeout: float = 120, inflight: int = 0) -> str:
        if label.startswith('owned-'):
            self.sup.poll()
            self.commands.append(label)
            return owner.FENCED if label.startswith('owned-fence-') else owner.ABSENT
        if argv[1:3] == ['-m', 'offload.pc1.cells']:
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
                output.write_text(json.dumps(result))
            elif '--cell' not in argv:
                output.write_text('{}')
            else:
                name = argv[argv.index('--cell') + 1]
                repeat = int(argv[argv.index('--repeat-index') + 1])
                salt = argv[argv.index('--salt') + 1]
                output.write_text(json.dumps(measured(name, .4 if cast(w.Arm, self.arm).role != 'control' else 0,
                                                      salt=salt, repeats=1, first=repeat)))
                self.rounds += 10
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
            return (1, 'JIT stuck') if self.fail == label else (0, 'PC1 EXPERIMENT ABSENT\n')
        if label == 'hold-reread':
            if self.fail == 'slow-restore':
                self.clock.now = self.live.restore_by - 20
            return 0, self.hold
        if label == 'experiment-stop':
            if self.fail == 'slow-restore':
                self.clock.now = self.live.restore_by + 1
            return 0, ''
        if label == 'chain-close':
            # The shared CHAINED_CLOSE under the -restore-close shim only records the unlink; commit-close removes HOLD.
            assert shlex.quote(chain.CHAINED_CLOSE) in argv[-1] and 'restore-close' in argv[-1]
            return (1, 'lost chain close reply') if self.fail == 'chain-close' else (0, chain.CHAINED_CLOSE_OK + '\n')
        if label == 'commit-close':
            self.hold = ''
            return 0, owner.CLOSED
        if label == 'cancel-close':
            return 0, 'SCOPED CLOSE COMMIT FENCED'
        if label == 'retain-close':
            self.hold = 'PC1-WINDOW tag: original retained'
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
            self.hold = 'PC1-WINDOW tag: failed restore'
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


def make_rehearsal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: str = '',
                   name: str = 'out') -> tuple[w.Live, Remote, pf.Supervisor]:
    """Real Live over the fake transport; several windows of one chain share tmp_path (campaign, chain marker)."""
    monitor = tmp_path / 'monitor.py'
    monitor.write_text(base.MONITOR)
    clock = base.Clock()
    out = tmp_path / name
    out.mkdir()
    sup = base.NoTunnel(out, clock=clock, monitor=lambda host: [os.environ.get('PC1_TEST_PYTHON', '/home/operator/miniconda3/bin/python3'), str(monitor), 'low', str(tmp_path / 'trip')], demand=lambda: 0.)
    remote = Remote(out, sup, clock, fail, trip=tmp_path / 'trip')
    live = w.Live(out, sup, group='chunk2048',
        remote='dsv41-deploy-offload', expected='a' * 40, data_dir=tmp_path, tag=name, call=remote.call, plain=remote.plain, clock=clock)
    remote.live = live
    import offload.prefill_trace.window_requests as wr
    monkeypatch.setattr(greedy_driver, 'greedy_inputs', lambda: (['public']*6, [{'i': i, 'text': 'public'} for i in range(6)]))
    monkeypatch.setattr(wr, 'corpus', lambda root: ('public fixture', {}))
    monkeypatch.setattr(pf, 'CAMPAIGN', tmp_path)
    # Never touch the campaign's real chain marker; chain mode is off unless a test turns it on.
    monkeypatch.setattr(chain, 'MARKER', tmp_path / 'CHAINED')
    monkeypatch.delenv('WINDOW_CHAIN', raising=False)
    monkeypatch.delenv('WINDOW_CHAIN_LAST', raising=False)
    remote.aborts = []
    def abort(out: Path) -> dict[str, Json]:
        # The shared chain.abort (strict W8 + watch under its own HOLD) is reviewed elsewhere; record the call only.
        remote.aborts.append(out)
        chain.clear()
        return {'chain': 'ABORTED', 'status': 'RESTORED: chain ended by abort; watch PASS; marker cleared',
                'restore': {'status': chain.ABORT_RESTORED}, 'post_close_watch': {'status': 'PASS'}}
    monkeypatch.setattr(chain, 'abort', abort)
    (tmp_path / 'D2-GEN.PAUSE').touch()
    return live, remote, sup


@pytest.fixture
def rehearsal(tmp_path: Path, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[w.Live, Remote]]:
    live, remote, sup = make_rehearsal(tmp_path, monkeypatch, str(getattr(request, 'param', '')), 'tag')
    try:
        yield live, remote
    finally:
        sup.close()


@pytest.mark.parametrize('group', b.ARMS[1:])
def test_full_live_sequence(rehearsal: tuple[w.Live, Remote], group: str) -> None:
    live, fake = rehearsal
    live.group = group
    result = w.window(live, group, lambda row: None)
    assert 'stop' not in result, result.get('stop')
    assert str(record(result['restore'])['status']).startswith('RESTORED'), result
    assert result['status'] == 'COMPLETE'
    assert record(result['decision'])['status'] == 'CANDIDATE'
    plan = cells.plan_for(group)
    for arm in w.arms(group):
        ran = record(record(record(result['starts'])[arm.name])['cells'])
        assert list(ran) == list(plan.cells)
        assert all(len(cast(list[Json], record(ran[n])['repetitions'])) == plan.repeats(n) for n in plan.cells)
    assert fake.restores == ['start', 'check', 'close']
    assert not fake.hold
    assert list(record(result['starts'])) == [a.name for a in w.arms(group)]
    for arm in w.arms(group):
        report = record(record(record(result['starts'])[arm.name])['capacity'])
        expected = capacity.expectation(arm.role)
        assert report == {**expected, 'reported_tokens': {
            'sparkhead': [expected['expected_tokens']], 'sparkworker-wired': []}}
    assert fake.commands.index('owned-absent-tag-s1-control-engine-sparkworker-wired') < fake.commands.index('start')
    assert fake.commands.index('owned-absent-tag-restore-start-2-sparkworker-wired') < fake.commands.index('check')
    assert fake.commands.index('owned-absent-tag-restore-check-3-sparkworker-wired') < fake.commands.index('close')
    deadlines = json.loads((live.out / 'deadlines.json').read_text())
    assert deadlines['restore_by'] - deadlines['opened'] == 85 * 60


@pytest.mark.parametrize('group', b.ARMS[1:])
@pytest.mark.parametrize('rehearsal', ['capacity-wrong', 'capacity-empty', 'capacity-mixed', 'capacity-control'], indirect=True)
def test_capacity_mismatch_stops_before_timing_and_restores(rehearsal: tuple[w.Live, Remote], group: str) -> None:
    live, fake = rehearsal
    live.group = group
    result = w.window(live, group, lambda row: None)
    assert 'KV capacity is not' in str(result['stop'])
    assert record(result['decision'])['status'] == 'NOT A CANDIDATE'
    assert str(record(result['restore'])['status']).startswith('RESTORED')
    failed_arm = w.arms(group)[0 if fake.fail == 'capacity-control' else 1]
    assert record(record(record(result['starts'])[failed_arm.name])['cells']) == {}
    reported = record(record(record(record(result['starts'])[failed_arm.name])['capacity'])['reported_tokens'])
    wrong = 2933321 if fake.fail == 'capacity-control' or group != 'chunk2048' else 2502600
    assert reported['sparkhead'] == ([] if fake.fail == 'capacity-empty' else
        [2502600, 2933321] if fake.fail == 'capacity-mixed' else [wrong])
    assert not any(c.startswith(f'{failed_arm.name}-{name}') for name in cells.CELLS for c in fake.commands)
    assert fake.restores == ['start', 'check', 'close']
    assert not fake.hold


@pytest.mark.parametrize('rehearsal', ['start', 'gate', 'experiment-greedy', 'signal', 'deadline', 'hold-open', 'STOP', 'memory', 'unfiltered-env'], indirect=True)
def test_stop_then_restore(rehearsal: tuple[w.Live, Remote]) -> None:
    live, fake = rehearsal
    result = w.window(live, live.group, lambda row: None)
    assert 'stop' in result
    assert str(record(result['restore'])['status']).startswith('RESTORED'), result
    assert record(result['decision'])['status'] == 'NOT A CANDIDATE'
    if fake.fail == 'gate':
        assert not any(c.startswith('s2-chunk2048-short-b1') for c in fake.commands)
    assert fake.restores[-2:] == ['check', 'close']


@pytest.mark.parametrize('rehearsal', ['cleanup', 'jit-wait-sparkhead', 'jit-wait-sparkworker-wired', 'greedy', 'slow-restore'], indirect=True)
def test_failed_cleanup_or_proof_keeps_hold(rehearsal: tuple[w.Live, Remote]) -> None:
    live, fake = rehearsal
    result = w.window(live, 'chunk2048', lambda row: None)
    assert record(result['restore'])['status'] == 'FAILED', result
    assert 'close' not in fake.restores
    assert fake.hold
    assert record(result['decision'])['status'] == 'NOT A CANDIDATE'


def test_unreviewed_head_refuses_before_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('WINDOW_COORDINATOR', 'Deepseek/coordinator')
    monkeypatch.setattr(pf, 'CAMPAIGN', tmp_path)
    assert w.main(['--execute', '--out', str(tmp_path / 'fresh'), '--expected-experiment-commit', 'a' * 40]) == 1
    assert not (tmp_path / 'fresh').exists()


@pytest.mark.parametrize('rehearsal', ['second-signal'], indirect=True)
def test_second_signal_cannot_abandon_recovery(rehearsal: tuple[w.Live, Remote]) -> None:
    live, fake = rehearsal
    result = w.window(live, 'chunk2048', lambda row: None)
    assert str(record(result['restore'])['status']).startswith('RESTORED'), result
    assert fake.restores == ['start', 'check', 'close']


@pytest.mark.parametrize('rehearsal', ['lost-close', 'owned-reap-tag-restore-close-sparkworker-wired'], indirect=True)
def test_ambiguous_close_recreates_hold(rehearsal: tuple[w.Live, Remote]) -> None:
    live, fake = rehearsal
    result = w.window(live, 'chunk2048', lambda row: None)
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
    label = 'owned-fence-PC1-k5-W-x-hold-sparkhead'
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
    for arm in b.ARMS:
        pins = w.source_pins(arm)
        package = {k: v for k, v in pins.items() if '/spec_decode/remote_dspark/' in k}
        assert package == mounted
