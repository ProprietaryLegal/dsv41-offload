"""CPU-only rehearsal: real TV1Live, PF1 Supervisor, SP1 trace collection; fake commands only."""
from __future__ import annotations

import gzip
import base64
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Iterator, cast
from dataclasses import FrozenInstanceError

import pytest

from offload.launch.windows import pf1_window as pf
from offload.launch.windows import sp1_window as sp
from offload.launch.windows import tv1_window as tv
from offload.launch.windows import post_close_watch as watchdog
from offload.launch.windows import scoped_remote as owner
from offload.launch.windows import up2_checks as ck
from offload.launch.windows.tests import test_pf1_window as base
from offload.launch.windows.tests import test_sp1_window as spbase
from offload.prefill_maxlen.evidence import Json, record
from offload.tv1 import cells, schedule, init_smoke, candidate_init_smoke, greedy, oracle_prestart
from offload.gb10_budget import window_cells as wc
from offload.gb10_budget import test_trace_budget as synth
from offload.tv1.window_config import FLAGS
from offload.tv1 import candidate_cases as geometry


def oracle_receipt() -> dict[str, Json]:
    return {'type': 'tv1-installed-oracle-v2', 'all_identical': True, 'classification': 'MEASURED SM121',
            'capability': [12, 1], 'exclusions': cast(Json, geometry.EXCLUSIONS),
            'cases': cast(Json, [{**case, 'eager_identical': True, 'replay_identical': True}
                                for case in geometry.case_matrix()])}


def engine_receipts() -> list[Json]:
    return [{'type': 'tv1-engine-oracle-v2', 'layer': layer, 'batch': batch, 'raw_context': context,
             'rows': batch*5, 'next_n': 1, 'stride': geometry.STRIDE, 'cache_offset': geometry.CACHE_OFFSET,
             'width': geometry.MAX_RAW, 'candidate_capacity': 2048, 'page': geometry.PAGE,
             'layout': 'flattened', 'backend': 'persistent', 'identical': True}
            for layer in geometry.LAYERS for batch, context in ((1,150000),(2,32000),(3,32000),(4,32000),(4,150000))]


class Fake(spbase.SP1Remote):
    # SP1's fake generates valid rank traces, metrics, prompt cuts and profile listings.
    def __init__(self, out: Path, sup: pf.Supervisor, clock: base.Clock, feature: str, fail: str = '') -> None:
        super().__init__(out, sup, clock)
        self.feature, self.failure = feature, fail
        self.live: tv.TV1Live | None = None
        self.position = 0
        self.current_out = out
        self.engine_events: list[str] = []

    def identity(self, rank: int) -> dict[str, Json]:
        if self.arm is None:
            return base.prod_identity(rank)
        assert self.live is not None
        arm = cast(tv.Arm, self.arm)
        live = base.live_identity(arm, 'r', rank)
        record(live['env'])['VLLM_ENGRAM_REMOTE_REQUIRED'] = '0'
        live['id'] = f'{arm.name}-{rank}-{self.serial}'
        live['argv'] = str(live['argv']) + ' --profiler-config ' + shlex.quote(sp.PROFILER_JSON)
        record(live['env']).update(cast(dict[str, Json], FLAGS[arm.role]))
        record(live['mounted']).update({pf.DIST + e['path']: e['sha256']
                                       for e in self.live.manifests[arm.role]['files']})
        if self.failure == 'identity' and self.position == 2:
            record(live['mounted'])[pf.DIST + 'vllm_exl3_moe/unknown.py'] = '0'*64
        return live

    def remote(self, label: str, host: str, command: str, stdin: str | None) -> str:
        tokens = shlex.split(command)
        if tokens[:1] == ['env'] and tokens[1].startswith('SCOPED_REMOTE_DEADLINE='):
            tokens = tokens[2:]
        if tokens[:2] == ['python3', '-c'] and tokens[3:4] == ['launch']:
            command = tokens[-1]
            tokens = shlex.split(command)
        if tokens[:2] == ['python3', '-c'] and tokens[3:4] == ['stop']:
            command = tokens[-1]
        if label.endswith('hold-read'):
            return self.hold
        if 'oracle-memory-' in label:
            return json.dumps({'type': 'tv1-oracle-memory-v1', 'page_size': 65536,
                'meminfo': 'MemAvailable: 12582912 kB\n',
                'buddyinfo': 'Node 0, zone Normal 0 0 0 0 0 0 0 0 0 0 200\n'})
        if 'TV1_ENGINE_ORACLE ' in command:
            rows = engine_receipts()
            if self.failure == 'engine-oracle' and host == pf.HOSTS[1]:
                rows = [r for r in rows if record(r)['batch'] != 3]
            return json.dumps({'records': rows})
        if label.endswith('hold-open'):
            self.hold = f'{base.iso(self.clock())} DSV41-OFFLOAD (session coordinator): TV1-WINDOW tag: test'
            if self.failure == 'hold':
                raise ValueError('STOP injected hold timeout after write')
            return 'open'
        if 'bash offload/launch/start_exp.sh' in command:
            self.engine_events.append('start-stop')
            self.position += 1
            self.arm = tv.arms(self.feature)[self.position-1]
            self.serial += 1
            self.started = {h: 0 for h in pf.HOSTS}
            if self.failure == 'start' and self.position == 2:
                raise ValueError('STOP injected start failure')
            return 'ready'
        if stdin and 'TV1 staged manifest drift' in stdin:
            return 'TV1 staged hashes verified'
        if stdin and 'TV1 runtime source hash mismatch' in stdin:
            if self.failure == 'hash' and self.position == 2:
                raise ValueError('STOP TV1 runtime source hash mismatch')
            return 'TV1 verified engine sources'
        if label.endswith('prestart-oracles'):
            if self.failure == 'oracle':
                raise ValueError('STOP installed DeepGEMM selected-index mismatch')
            receipt = oracle_receipt()
            if self.failure == 'oracle-geometry':
                record(cast(list[Json], receipt['cases'])[0])['stride'] = 8704
            return json.dumps({'type': 'tv1-prestart-oracles-v1', 'run': 'tag',
                'ranks': {h: {'image': oracle_prestart.IMAGE, 'server_absent_before': True,
                              'compute_absent_after': True, 'oracle': receipt} for h in pf.HOSTS}})
        if '{{json .Mounts}}' in command:
            assert self.live is not None and self.arm is not None
            arm = cast(tv.Arm, self.arm)
            return json.dumps([{'Type':'bind', 'Source':'/home/u/' + sp.PROFILE_DIR,
                                'Destination': '/sp1-profile', 'RW': True},
                *[{'Type': 'bind', 'Source': '/fake/file', 'Destination': pf.DIST + e['path'],
                   'RW': self.failure == 'writable' and self.position == 2}
                  for e in self.live.manifests[arm.role]['files']]])
        return super().remote(label, host, command, stdin)

    def call(self, label: str, argv: list[str], *, stdin: str | None = None, timeout: float = 120, inflight: int = 0) -> str:
        if argv[1:3] == ['-m', 'offload.tv1.qualification']:
            assert inflight == 4
            Path(argv[argv.index('--output')+1]).write_text(json.dumps({
                'type': 'tv1-qualification-traffic-v2', 'qualification_only': True}))
            return ''
        if argv[1:3] == ['-m', 'offload.tv1.cells']:
            assert self.arm is not None
            if 'greedy' in argv:
                if self.failure == 'greedy' and self.position == 2:
                    raise ValueError('STOP greedy DIVERGES')
                out = Path(argv[argv.index('--output')+1])
                out.write_text(json.dumps(greedy.registered({'checks':[
                    {'i':i, 'identical': True} for i in range(6)], 'identical':6, 'near_tie':0})))
                return ''
            # SP1 fake driver understands same interface but writes in its out dir.
            patched = list(argv)
            patched[2] = 'offload.gb10_budget.window_cells'
            previous = self.out
            self.out = Path(argv[argv.index('--output')+1]).parent.parent if 'run' in argv else self.current_out
            try:
                return super().call(label, patched, stdin=stdin, timeout=timeout, inflight=inflight)
            finally:
                self.out = previous
        return super().call(label, argv, stdin=stdin, timeout=timeout, inflight=inflight)

    def driver(self, argv: list[str], inflight: int) -> str:
        args = {argv[i]: argv[i + 1] for i in range(4, len(argv) - 1) if argv[i].startswith('--')}
        self.sup.poll()
        if 'build' in argv:
            Path(args['--output']).write_text(json.dumps({'prompts': {p.key: {'tokens': p.target - 3, 'cut': p.target * 3}
                for p in wc.PROMPTS.values()}}))
            return ''
        name = args['--cell']
        cell = cells.CELLS[name]
        assert inflight == cell.batch
        assert float(args['--deadline-utc']) == self.sup.abort_at - tv.ABORT_S + schedule.WINDOW_S - schedule.RECOVERY_S
        self.cells.append(name)
        captures: list[Json] = []
        for kind in cell.captures:
            captures.append({'kind': kind, 'status': 'TAKEN', 'start_call_rel_s': 5., 'start_return_rel_s': 5.01,
                             'stop_call_rel_s': 8., 'stop_return_rel_s': 8.5, 'mem_available_kb': {}})
            for rank, host in enumerate(pf.HOSTS):
                files = self.dirs[host]
                assert files is not None
                trace = synth.decode_trace(tid=11+rank)
                self.ns += 1
                files[f'dp0_pp0_tp{rank}_dcp0_ep{rank}_rank{rank}.{self.ns}.pt.trace.json.gz'] = gzip.compress(trace.text().encode(), mtime=0)
                self.started[host] += 1
        streams = [synth.stream(.1*i, 10., 30, 30.) for i in range(cell.batch)]
        Path(args['--output']).write_text(json.dumps({'type':'cell', 'cell':name, 'batch':cell.batch,
            'profiled':bool(cell.captures), 'warmup':cell.warmup, 'streams':streams, 'captures':captures,
            'spec':{'acceptance_fraction':.5, 'mean_accepted_per_round':1.}}))
        return ''

    def plain(self, step: str, command: list[str], timeout: float) -> tuple[int, str]:
        if step.startswith('engine-logs-'):
            self.engine_events.append(step)
            data = b'public startup root cause\n' + b'public engine log\n'*50
            return 0, json.dumps({'type': 'tv1-engine-logs-v1', 'name': 'dsv41-serve',
                'container_id': 'a'*64, 'returncode': 0, 'total_bytes': len(data), 'stored_bytes': len(data),
                'stream_sha256': hashlib.sha256(data).hexdigest(),
                'stored_sha256': hashlib.sha256(data).hexdigest(), 'truncated': False, 'complete': True,
                'data_b64': base64.b64encode(data).decode('ascii')})
        if step == 'retain-close':
            return 0, owner.RETAINED
        if step == 'cancel-close':
            return 0, 'SCOPED CLOSE COMMIT FENCED'
        if step == 'commit-close':
            return 0, owner.CLOSED
        if step.startswith(('fence-', 'reap-', 'absent-')):
            self.unsupervised.append((step, command[-2], command[-1]))
            return 0, owner.FENCED if step.startswith('fence-') else owner.ABSENT
        if step.startswith('jit-wait-'):
            return 0, tv.QUIET
        if step == 'experiment-stop':
            self.engine_events.append('final-stop')
            return 0, 'stopped'
        if step.startswith('profile-dir-'):
            return 0, 'CLEAN'
        if step in ('start', 'check', 'close'):
            tokens = shlex.split(command[-1])
            assert tokens[:1] == ['env'] and tokens[1].startswith('SCOPED_REMOTE_DEADLINE=')
            assert tokens[-3:] == ['bash', str(pf.RESTORE), step]
            command = ['bash', str(pf.RESTORE), step]
        if self.failure == 'restore' and step == 'check':
            self.restores.append(step)
            return 0, base.RESTORE_OK['check'].replace('5 IDENTICAL', '4 IDENTICAL')
        return super().plain(step, command, timeout)


@pytest.fixture
def rehearsal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Iterator[tuple[tv.TV1Live, Fake, base.Clock, pf.Supervisor]]:
    settings = cast(dict[str, str], getattr(request, 'param', {}))
    feature, fail = settings.get('feature', 'candidate'), settings.get('fail', '')
    monitor = tmp_path / 'monitor.py'
    monitor.write_text(base.MONITOR)
    out = tmp_path / 'window'
    out.mkdir()
    clock = base.Clock()
    sup = base.NoTunnel(out, clock=clock, monitor=lambda host: [sys.executable, str(monitor), 'ok', str(tmp_path/'trip')],
                        demand=lambda: 0.)
    fake = Fake(out, sup, clock, feature, fail)
    live = tv.TV1Live(out, sup, remote='dsv41-deploy-offload', expected='0'*40,
                      data_dir=tmp_path, tag='tag', stage_bundle=False, feature=feature,
                      call=fake.call, plain=fake.plain, clock=clock)
    fake.live = live
    import offload.prefill_trace.window_requests as wr
    monkeypatch.setattr(wr, 'corpus', lambda root: ('x', {}))
    try:
        yield live, fake, clock, sup
    finally:
        sup.close()


@pytest.mark.parametrize('rehearsal', [{'feature':'candidate'}, {'feature':'compact'}], indirect=True)
def test_live_all_starts_restore(rehearsal: tuple[tv.TV1Live, Fake, base.Clock, pf.Supervisor]) -> None:
    live, fake, _, _ = rehearsal
    result = tv.window(live, fake.feature, lambda row: None)
    assert result['status'] == 'COMPLETE', result.get('stop')
    assert fake.restores == ['start', 'check', 'close']
    assert fake.position == 3
    assert list(record(result['starts'])) == [a.name for a in tv.arms(fake.feature)]
    assert fake.cells == list(cells.order(fake.feature))*3
    smoke = live.root_out / 'compact-init-source-smoke.json'
    assert smoke.exists() == (fake.feature == 'compact')
    if smoke.exists():
        receipt = json.loads(smoke.read_text())
        assert receipt['classification'] == 'SOURCE/CPU ONLY'
        assert receipt['cuda_or_engine_init_qualified'] is False
    candidate_smoke = live.root_out / 'candidate-init-source-smoke.json'
    assert candidate_smoke.exists() == (fake.feature == 'candidate')
    if candidate_smoke.exists():
        receipt = json.loads(candidate_smoke.read_text())
        assert receipt['classification'] == 'SOURCE/CPU ONLY'
        assert receipt['bound_layers'] == [24, 28, 32, 36]
        assert receipt['cuda_initialized'] is False
        assert receipt['cuda_or_engine_init_qualified'] is False


@pytest.mark.parametrize('rehearsal', [{'feature': 'candidate'}], indirect=True)
@pytest.mark.parametrize('failed_host', pf.HOSTS)
def test_fragmented_oracle_memory_refuses_before_hold(
        rehearsal: tuple[tv.TV1Live, Fake, base.Clock, pf.Supervisor],
        monkeypatch: pytest.MonkeyPatch, failed_host: str) -> None:
    from offload.tv1.tests.test_oracle_memory import probe
    live, fake, _, _ = rehearsal
    original = fake.remote
    def remote(label: str, host: str, command: str, stdin: str | None) -> str:
        if 'oracle-memory-prehold-' in label and host == failed_host:
            return json.dumps(probe(failed=True))
        return original(label, host, command, stdin)
    monkeypatch.setattr(fake, 'remote', remote)
    result = tv.window(live, 'candidate', lambda row: None)
    assert result['status'] == 'INCOMPLETE' and 'pre-HOLD oracle memory refusal' in str(result.get('stop'))
    assert fake.position == 0 and not fake.hold and fake.restores == []
    assert live.open_admitted is None and not live.owner_tags
    assert str(record(result['restore'])['status']).startswith('NOT NEEDED')
    saved = ck.read(live.root_out / f'oracle-memory-prehold-{failed_host}.json')
    assert saved['largest_free_normal_order'] == 6 and saved['admitted'] is False
    assert saved['mem_available_kb'] == 9386836
    assert (live.root_out / f'oracle-memory-prehold-{failed_host}.probe.json').is_file()


@pytest.mark.parametrize('rehearsal', [{'feature': 'candidate'}], indirect=True)
def test_bad_candidate_binding_refuses_before_hold_or_remote_action(
        rehearsal: tuple[tv.TV1Live, Fake, base.Clock, pf.Supervisor], monkeypatch: pytest.MonkeyPatch) -> None:
    live, fake, _, _ = rehearsal
    calls: list[object] = []
    def broken(tree: Path) -> dict[str, Json]:
        raise AssertionError('Current vLLM config is not set')
    def no_remote(*args: object, **kwargs: object) -> str:
        calls.append((args, kwargs))
        raise AssertionError('smoke must run before any transport')
    monkeypatch.setattr(candidate_init_smoke, 'check', broken)
    monkeypatch.setattr(live, 'ssh', no_remote)
    result = tv.window(live, 'candidate', lambda row: None)
    assert result['status'] == 'INCOMPLETE' and 'Current vLLM config' in str(result['stop'])
    assert calls == [] and not fake.hold and fake.position == 0
    assert fake.restores == [] and live.open_admitted is None


@pytest.mark.parametrize('rehearsal', [{'feature': 'compact'}], indirect=True)
def test_bad_compact_source_refuses_before_hold_or_remote_action(
        rehearsal: tuple[tv.TV1Live, Fake, base.Clock, pf.Supervisor], monkeypatch: pytest.MonkeyPatch) -> None:
    live, fake, _, _ = rehearsal
    def broken(tree: Path) -> dict[str, Json]:
        raise NameError("name 'os' is not defined")
    monkeypatch.setattr(init_smoke, 'check', broken)
    result = tv.window(live, 'compact', lambda row: None)
    assert result['status'] == 'INCOMPLETE' and 'NameError' in str(result['stop'])
    assert not fake.hold and fake.position == 0
    assert fake.restores == [] and live.open_admitted is None


@pytest.mark.parametrize('rehearsal', [{'feature':'compact'}, {'feature':'candidate'}], indirect=True)
@pytest.mark.parametrize('margin', [.5, .75, 1.])
def test_real_nbw_greedy_diagnosis_survives_in_summary_and_strict_restore(
        rehearsal: tuple[tv.TV1Live, Fake, base.Clock, pf.Supervisor], margin: float,
        monkeypatch: pytest.MonkeyPatch) -> None:
    from offload.tv1.tests.test_greedy import public_references
    from offload.launch.windows.tests.test_nbw_window import GreedyApi
    live, fake, _, _ = rehearsal
    public_references(live.root_out, monkeypatch)
    original = live.call
    def call(label: str, argv: list[str], *, stdin: str | None = None,
             timeout: float = 120, inflight: int = 0) -> str:
        if argv[1:3] == ['-m','offload.tv1.cells'] and 'greedy' in argv:
            output = Path(argv[argv.index('--output')+1])
            greedy.run(GreedyApi(margin if fake.position == 2 else None), 1e12, output)
            return ''
        return original(label, argv, stdin=stdin, timeout=timeout, inflight=inflight)
    monkeypatch.setattr(live,'call',call)
    result = tv.window(live, fake.feature, lambda row: None)
    assert result['registration_id'] == 'tv1-r10-near-tie-0p9' and result['greedy_rule'] == greedy.RULE
    accepted = margin <= greedy.NEAR_TIE_NATS
    second = record(record(result['starts'])[tv.arms(fake.feature)[1].name])
    receipt = record(second['greedy'])
    row = ck.rows(receipt['checks'])[3]
    assert row['token_index'] == 2 and row['margin_nats'] == margin and row['gap_to_top_nats'] == 0
    assert row['near_tie'] is accepted
    assert receipt['registration_id'] == result['registration_id']
    if accepted:
        assert result['status'] == 'COMPLETE' and fake.position == 3
        assert receipt['identical'] == 5 and receipt['near_tie'] == 1
    else:
        assert result['status'] == 'INCOMPLETE' and 'not a near-tie' in str(result['stop'])
        assert fake.position == 2 and second['cells'] == {}
        assert 'not a near-tie' in str(receipt['diagnostic_error'])
    assert fake.restores == ['start','check','close']
    assert str(record(result['restore'])['status']).startswith('RESTORED: K1 START VERIFIED, greedy 6/6 IDENTICAL')
    encoded = json.dumps(result)
    assert all(text not in encoded for text in ('public prompt','public reference','public changed output'))


@pytest.mark.parametrize('rehearsal', [
    {'feature': 'compact'}, {'feature': 'compact', 'fail': 'hold'},
    {'feature': 'compact', 'fail': 'start'}, {'feature': 'compact', 'fail': 'greedy'},
], indirect=True)
def test_full_restore_real_local_logs_never_collide(
        rehearsal: tuple[tv.TV1Live, Fake, base.Clock, pf.Supervisor]) -> None:
    """Run the complete recovery, including both fencing passes, through real log IO.

    Only transport/W8 responses are fake: each response is emitted by an owned CPU
    child. TV1.local itself opens/writes/reads every log and waits/cleans the child.
    The hold failure is before any start; start failure follows the first OFF arm.
    """
    live, fake, _, _ = rehearsal
    labels: list[str] = []

    def transport(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        labels.append(label)
        rc, text = fake.plain(label, argv, timeout)
        return live.local(label, [sys.executable, '-c',
            'import sys;sys.stdout.write(sys.argv[1]);sys.exit(int(sys.argv[2]))', text, str(rc)], timeout)

    live.plain = transport
    result = tv.window(live, 'compact', lambda row: None)
    assert str(record(result['restore'])['status']).startswith('RESTORED'), result
    assert fake.restores == (['check', 'close'] if fake.failure == 'hold' else ['start', 'check', 'close'])
    assert fake.position == {'': 3, 'hold': 0, 'start': 2, 'greedy': 2}[fake.failure]
    # Repeated fence/reap/absence labels are intentional and must preserve both
    # observations, rather than overwriting the first or failing after W8 close.
    assert len(labels) > len(set(labels))
    logs = sorted((live.root_out / 'cmd').glob('restore-*.log'))
    assert len(logs) == len(labels)
    assert len({p.name for p in logs}) == len(labels)
    assert [int(p.name.split('-')[1]) for p in logs] == list(range(1, len(labels)+1))
    assert logs[-1].read_text() == 'SCOPED CLOSE COMMIT FENCED'


@pytest.mark.parametrize('rehearsal', [{'feature': 'compact'}, {'feature': 'compact', 'fail': 'start'}], indirect=True)
def test_both_rank_engine_logs_preserved_before_every_experiment_removal(
        rehearsal: tuple[tv.TV1Live, Fake, base.Clock, pf.Supervisor]) -> None:
    live, fake, _, _ = rehearsal
    result = tv.window(live, 'compact', lambda row: None)
    assert str(record(result['restore'])['status']).startswith('RESTORED'), result
    # First start stops production, so it has no experiment logs to preserve.
    # Every subsequent start and the final recovery stop must follow BOTH reads.
    events = fake.engine_events
    assert events[0] == 'start-stop'
    for index, event in enumerate(events[1:], 1):
        if event in ('start-stop', 'final-stop'):
            assert events[index-2].endswith('sparkhead')
            assert events[index-1].endswith('sparkworker-wired')
    receipts = list((live.root_out / 'engine-logs').glob('*/*.json'))
    assert len(receipts) == (4 if fake.failure else 6)
    for path in receipts:
        receipt = json.loads(path.read_text())
        data = path.with_suffix('.log').read_bytes()
        assert data.startswith(b'public startup root cause\n')  # absent from a tail40
        assert receipt['complete'] is True
        assert receipt['stream_sha256'] == receipt['stored_sha256'] == hashlib.sha256(data).hexdigest()
    assert bool(list((live.root_out / 'engine-logs').glob('*-start-failed'))) == bool(fake.failure)


@pytest.mark.parametrize('node', pf.HOSTS)
@pytest.mark.parametrize('rehearsal', [{'feature': 'compact'}], indirect=True)
def test_diagnostic_failure_stops_measurement_but_allows_w8_recovery(
        rehearsal: tuple[tv.TV1Live, Fake, base.Clock, pf.Supervisor], node: str) -> None:
    live, fake, _, _ = rehearsal
    original = live.plain
    def fail(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        if label.startswith('engine-logs-') and label.endswith(node):
            fake.engine_events.append(label)
            return 1, 'injected docker/transport failure'
        return original(label, argv, timeout)
    live.plain = fail
    result = tv.window(live, 'compact', lambda row: None)
    assert result['status'] == 'INCOMPLETE' and result['engine_log_errors']
    assert str(record(result['restore'])['status']).startswith('RESTORED'), result
    assert fake.restores == ['start', 'check', 'close'] and fake.position == 1
    assert len(list((live.root_out / 'engine-logs').glob('*/*.error.txt'))) == 2
    assert any(e.endswith('sparkworker-wired') for e in fake.engine_events)
    assert any(e.endswith('sparkhead') for e in fake.engine_events)


@pytest.mark.parametrize('rehearsal', [{'feature': 'compact'}, {'feature': 'candidate'}], indirect=True)
@pytest.mark.parametrize('fault', ['mkdir', 'error-write-sparkhead', 'error-write-sparkworker-wired'])
def test_diagnostic_filesystem_failure_preserves_real_local_w8_recovery(
        rehearsal: tuple[tv.TV1Live, Fake, base.Clock, pf.Supervisor],
        monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    """S69: only diagnostic storage fails; real restore-command log IO stays writable."""
    live, fake, clock, _ = rehearsal
    mkdir, write_text = Path.mkdir, Path.write_text
    failed_node = fault.removeprefix('error-write-')
    attempts: list[tuple[str, int, str]] = []

    def diagnostic_mkdir(path: Path, mode: int = 0o777, parents: bool = False,
                         exist_ok: bool = False) -> None:
        if fault == 'mkdir' and 'engine-logs' in path.parts:
            raise OSError('S69 diagnostic mkdir failure')
        mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

    def diagnostic_write(path: Path, data: str, encoding: str | None = None,
                         errors: str | None = None, newline: str | None = None) -> int:
        if 'engine-logs' in path.parts and path.name == failed_node + '.error.txt':
            raise OSError('S69 diagnostic error receipt failure')
        return write_text(path, data, encoding=encoding, errors=errors, newline=newline)

    def transport(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        assert live.budget is not None
        assert 0 < timeout <= live.budget.restore - clock.now
        rc, text = fake.plain(label, argv, timeout)
        if label.startswith('engine-logs-'):
            deadline = live.budget.restore if label.startswith('engine-logs-2-') else live.measure_by
            assert timeout <= min(30., deadline - clock.now)
            if fault != 'mkdir' and label.endswith(failed_node):
                rc, text = 1, 'S69 injected diagnostic transport failure'
        attempts.append((label, rc, text))
        return live.local(label, [sys.executable, '-c',
            'import sys;sys.stdout.write(sys.argv[1]);sys.exit(int(sys.argv[2]))', text, str(rc)], timeout)

    monkeypatch.setattr(Path, 'mkdir', diagnostic_mkdir)
    monkeypatch.setattr(Path, 'write_text', diagnostic_write)
    live.plain = transport
    result = tv.window(live, fake.feature, lambda row: None)
    restored = record(result['restore'])
    assert result['status'] == 'INCOMPLETE', result
    assert str(restored['status']).startswith('RESTORED'), result
    assert restored['restore_deadline_met'] is True
    assert result['engine_log_errors'] == restored['engine_log_errors'] == live.engine_log_errors
    if fault == 'mkdir':
        assert 'S69 diagnostic mkdir failure' in str(result)
    else:
        assert 'TV1 engine log transport rc1' in str(result)
        assert 'S69 diagnostic error receipt failure' in str(result)
    assert 'engine log preservation failed before next stop' in str(result['stop'])
    assert fake.position == 1 and fake.restores == ['start', 'check', 'close']
    assert list(record(result['starts'])) == ['s1-off']  # no further experiment submitted
    labels = [label for label, _, _ in attempts]
    assert [label for label in labels if label.startswith('engine-logs-')] == [
        f'engine-logs-{serial}-{host}' for serial in (1, 2) for host in pf.HOSTS]
    ordered = ['experiment-stop', *(f'jit-wait-{host}' for host in pf.HOSTS),
               'start', 'check', 'close', 'commit-close', 'cancel-close']
    assert [labels.index(label) for label in ordered] == sorted(labels.index(label) for label in ordered)
    assert max(i for i, label in enumerate(labels) if label.startswith('engine-logs-')) < labels.index('experiment-stop')
    logs = sorted((live.root_out / 'cmd').glob('restore-*.log'))
    assert len(logs) == len(attempts)
    for serial, (path, (label, _, text)) in enumerate(zip(logs, attempts), 1):
        assert path.name == f'restore-{serial:04d}-{label}.log' and path.read_text() == text
    if fault != 'mkdir':
        healthy = next(host for host in pf.HOSTS if host != failed_node)
        for directory in (live.root_out / 'engine-logs').iterdir():
            assert json.loads((directory / f'{healthy}.json').read_text())['complete'] is True
            assert not (directory / f'{failed_node}.error.txt').exists()


@pytest.mark.parametrize('rehearsal', [{'feature':'candidate'}, {'feature':'compact'}], indirect=True)
def test_remaining_schedule_admission_precedes_every_start(
        rehearsal: tuple[tv.TV1Live, Fake, base.Clock, pf.Supervisor], monkeypatch: pytest.MonkeyPatch) -> None:
    live, fake, clock, _ = rehearsal
    original = live.begin
    def late(arm: tv.Arm) -> None:
        if arm.role == fake.feature:
            assert live.budget is not None
            clock.now = live.budget.opened + schedule.WINDOW_S - schedule.remaining_s(fake.feature, 1) + .01
        original(arm)
    monkeypatch.setattr(live, 'begin', late)
    result = tv.window(live, fake.feature, lambda row: None)
    assert 'remaining starts plus recovery' in str(result['stop'])
    assert fake.position == 1  # no second stop/run command was submitted
    assert fake.restores == ['start', 'check', 'close']
    assert str(record(result['restore'])['status']).startswith('RESTORED')


def test_measurement_deadline_is_stricter_without_changing_outer_budget(
        rehearsal: tuple[tv.TV1Live, Fake, base.Clock, pf.Supervisor]) -> None:
    live, _, clock, sup = rehearsal
    live.budget = tv.HoldBudget(clock.now)
    live.submit = live.budget.submission
    sup.abort_at = live.budget.abort
    assert live.measure_by == live.budget.opened + 73*60
    assert live.submit == live.budget.opened + 85*60
    assert sup.abort_at == live.budget.opened + 86*60
    assert live.budget.restore == live.budget.opened + 98*60
    seen: list[float] = []
    def call(label: str, argv: list[str], *, stdin: str | None = None,
             timeout: float = 120, inflight: int = 0) -> str:
        seen.append(timeout)
        return 'ok'
    live.base_call = call
    clock.now = live.measure_by - 7
    assert live.tagged_call('probe', ['fake'], timeout=120) == 'ok'
    assert seen == [7]
    clock.now += 7
    with pytest.raises(ValueError, match='measurement deadline'):
        live.tagged_call('probe', ['fake'])
    assert seen == [7]


@pytest.mark.parametrize('rehearsal', [{'fail':x} for x in ['hold','start','hash','identity','writable','oracle',
    'oracle-geometry','engine-oracle','greedy','restore']], indirect=True)
def test_every_stop_restores(rehearsal: tuple[tv.TV1Live, Fake, base.Clock, pf.Supervisor]) -> None:
    live, fake, _, _ = rehearsal
    result = tv.window(live, fake.feature, lambda row: None)
    assert 'stop' in result or record(result['restore']).get('status') == 'FAILED'
    if fake.failure == 'restore':
        assert fake.restores == ['start', 'check']
        assert record(result['restore'])['status'] == 'FAILED'
    else:
        assert fake.restores == (['check', 'close'] if fake.failure == 'hold' else ['start','check','close'])


def test_dry_and_execute_refusal(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    assert tv.main([]) == 0
    assert 'DRY RUN' in capsys.readouterr().out
    monkeypatch.delenv('WINDOW_COORDINATOR', raising=False)
    assert tv.main(['--execute']) == 1
    assert 'before any remote action' in capsys.readouterr().err


@pytest.mark.parametrize(('node', 'mode'), [(node, mode) for node in tv.OWNER_HOSTS
    for mode in ('fence', 'reap', 'absent', 'jit-wait') if mode != 'jit-wait' or node in pf.HOSTS])
def test_cleanup_failure_prevents_w8_close(rehearsal: tuple[tv.TV1Live, Fake, base.Clock, pf.Supervisor],
                                         node: str, mode: str) -> None:
    live, fake, _, _ = rehearsal
    original = live.plain
    def failed(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        if label.startswith(mode + '-') and label.endswith(node):
            return 1, 'injected cleanup failure'
        return original(label, argv, timeout)
    live.plain = failed
    result = tv.window(live, fake.feature, lambda row: None)
    assert record(result['restore'])['status'] == 'FAILED'
    assert fake.restores == []
    assert fake.hold


@pytest.mark.parametrize('rehearsal', [{'feature': 'compact'}], indirect=True)
@pytest.mark.parametrize('step', ['start', 'check', 'close'])
def test_recovery_hops_also_require_absence(rehearsal: tuple[tv.TV1Live, Fake, base.Clock, pf.Supervisor],
                                          step: str) -> None:
    live, fake, _, _ = rehearsal
    original = live.plain
    def failed(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        if label.startswith(f'reap-tag-restore-{step}') and label.endswith('-sparkworker-wired'):
            return 1, 'lost cleanup reply'
        return original(label, argv, timeout)
    live.plain = failed
    result = tv.window(live, fake.feature, lambda row: None)
    assert record(result['restore'])['status'] == 'FAILED', result
    assert fake.restores == ['start','check','close'][:['start','check','close'].index(step)+1]
    assert fake.hold


def test_shared_fake_clock_deadline_stops_between_recovery_steps(
        rehearsal: tuple[tv.TV1Live, Fake, base.Clock, pf.Supervisor]) -> None:
    live, fake, clock, _ = rehearsal
    opened = clock.now - tv.ABORT_S
    live.open_admitted = opened
    live.budget = tv.HoldBudget(opened)
    live.hold_written = True
    live.owner_tags = ['tag-s1-off-start']
    fake.hold = 'TV1-WINDOW tag: owned'
    seen: list[tuple[str, float, float]] = []
    def slow(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        assert live.budget is not None
        assert 0 < timeout <= live.budget.restore - clock.now
        seen.append((label, clock.now, timeout))
        clock.now += timeout - .5 if label == 'start' else min(timeout, 5)
        if label.startswith('fence-'):
            return 0, owner.FENCED
        if label.startswith(('reap-', 'absent-')):
            return 0, owner.ABSENT
        if label.startswith('jit-wait-'):
            return 0, tv.QUIET
        return 0, base.RESTORE_OK.get(label, '')
    live.plain = slow
    with pytest.raises(ValueError, match='deadline'):
        live.restore(production_stopped=True)
    assert 'start' in [x[0] for x in seen]
    assert not {'check','close','commit-close'} & {x[0] for x in seen}
    assert clock.now <= opened + tv.RESTORE_S
    assert fake.hold
    assert live.budget.opened == opened


def test_expired_budget_admits_no_recovery_commands(
        rehearsal: tuple[tv.TV1Live, Fake, base.Clock, pf.Supervisor]) -> None:
    live, _, clock, _ = rehearsal
    live.budget = tv.HoldBudget(clock.now - tv.RESTORE_S)
    live.open_admitted = live.budget.opened
    live.owner_tags = ['tag-hold-open']
    called: list[str] = []
    def call(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        called.append(label)
        raise AssertionError('expired command submitted')
    live.plain = call
    with pytest.raises(ValueError, match='deadline exhausted'):
        live.restore(production_stopped=True)
    assert called == []


@pytest.mark.parametrize('rehearsal', [{'fail':'hold'}], indirect=True)
def test_lost_hold_reply_still_fixes_immutable_budget(
        rehearsal: tuple[tv.TV1Live, Fake, base.Clock, pf.Supervisor]) -> None:
    live, fake, _, _ = rehearsal
    result = tv.window(live, fake.feature, lambda row: None)
    assert live.budget is not None
    assert record(result['restore'])['restore_deadline_met'] is True
    data = json.loads((live.root_out/'deadlines.json').read_text())
    assert data['restore_by'] == data['opened'] + 98*60
    assert data['abort'] == data['opened'] + 86*60
    with pytest.raises(FrozenInstanceError):
        setattr(live.budget, 'opened', 0)
    with pytest.raises(ValueError, match='renewed'):
        live.open_hold()


@pytest.mark.parametrize('rehearsal', [{'feature':'compact'}], indirect=True)
@pytest.mark.parametrize('lost', ['commit-close', 'cancel-close'])
def test_lost_final_close_ack_requires_original_hold_retention(
        rehearsal: tuple[tv.TV1Live, Fake, base.Clock, pf.Supervisor], lost: str) -> None:
    live, fake, _, _ = rehearsal
    original = live.plain
    retained: list[str] = []
    def fail(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        if label == lost:
            return 124, 'injected lost final close reply'
        if label == 'retain-close':
            retained.append(label)
        return original(label, argv, timeout)
    live.plain = fail
    result = tv.window(live, fake.feature, lambda row: None)
    assert record(result['restore'])['status'] == 'FAILED'
    assert retained == ['retain-close'] and fake.hold


@pytest.mark.parametrize('remaining', [20, 90, 200])
def test_final_close_reserves_cancel_and_retention_at_shared_deadline(
        rehearsal: tuple[tv.TV1Live, Fake, base.Clock, pf.Supervisor], remaining: int) -> None:
    live, fake, clock, _ = rehearsal
    live.budget = tv.HoldBudget(clock.now - tv.RESTORE_S + remaining)
    live.open_admitted = live.budget.opened
    live.hold_written = True
    hold_present = [True]
    calls: list[tuple[str, float]] = []
    original = live.plain
    def lose(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        if label in ('commit-close', 'cancel-close', 'retain-close'):
            calls.append((label, timeout))
            clock.now += timeout  # use the entire permitted per-call allowance
            if label == 'commit-close':
                hold_present[0] = False  # actual unlink, lost reply
            if label == 'retain-close':
                hold_present[0] = True
                return 0, owner.RETAINED
            return 124, 'lost reply'
        return original(label, argv, timeout)
    live.plain = lose
    with pytest.raises(ValueError):
        live.restore(production_stopped=True)
    assert hold_present[0]
    if remaining <= 90:
        assert calls == [], 'no recovery scope or unlink begins without the final-close reserve'
    else:
        assert calls == [('commit-close',20),('cancel-close',20),('retain-close',20)]
        assert live.budget.restore - clock.now >= 30


@pytest.mark.parametrize('preflight_s', [120, 1080])
def test_opening_origin_covers_near_limit_drain_after_preflight(
        rehearsal: tuple[tv.TV1Live, Fake, base.Clock, pf.Supervisor], preflight_s: int) -> None:
    live, fake, clock, _ = rehearsal
    clock.now += preflight_s
    invoked = clock.now
    original = live.base_call
    def drain(label: str, argv: list[str], *, stdin: str | None = None,
              timeout: float = 120, inflight: int = 0) -> str:
        if label.endswith('hold-open'):
            clock.now += pf.DRAIN_S - 10
        return original(label, argv, stdin=stdin, timeout=timeout, inflight=inflight)
    live.base_call = drain
    live.open_hold()
    assert live.open_deadline == invoked + pf.DRAIN_S + 60
    assert live.budget is not None
    assert live.owner_deadlines['tag-hold'] == live.hold_scope_deadline >= live.budget.restore
    assert live.budget.opened == int(invoked + pf.DRAIN_S - 10)


def test_late_opening_never_starts_target(rehearsal: tuple[tv.TV1Live, Fake, base.Clock, pf.Supervisor]) -> None:
    live, fake, clock, _ = rehearsal
    original = live.base_call
    def late(label: str, argv: list[str], *, stdin: str | None = None,
             timeout: float = 120, inflight: int = 0) -> str:
        if label.endswith('hold-open'):
            clock.now += pf.DRAIN_S + 70
        return original(label, argv, stdin=stdin, timeout=timeout, inflight=inflight)
    live.base_call = late
    result = tv.window(live, fake.feature, lambda row: None)
    assert 'opening ceiling' in str(result['stop'])
    assert fake.position == 0
    assert str(record(result['restore'])['status']).startswith('RESTORED')


def test_watch_adapter_uses_shared_watch_and_scoped_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    def watch(health: watchdog.Health, recovery: watchdog.Recovery, *, log: object) -> watchdog.Result:
        assert isinstance(health, watchdog.ProductionHealth)
        assert isinstance(recovery, watchdog.ScopedRecovery)
        called.append('watch')
        return watchdog.Result('PASS', 30, 0, 0)
    monkeypatch.setattr(watchdog, 'watch', watch)
    assert tv.post_close(tmp_path, 'tag')['status'] == 'PASS'
    assert called == ['watch']


@pytest.mark.parametrize(('restored', 'watch_status', 'expected'), [
    ('RESTORED: verified', 'PASS', 0), ('RESTORED: verified', 'FAILED', 2),
    ('FAILED', 'PASS', 2), ('NOT NEEDED: no HOLD', 'PASS', 0)])
def test_cli_watch_runs_only_after_verified_close(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                restored: str, watch_status: str, expected: int) -> None:
    import signal
    import subprocess
    monkeypatch.setenv('WINDOW_COORDINATOR', 'Deepseek/coordinator')
    head = 'a'*40
    calls: list[str] = []
    def git(argv: list[str], **kwargs: object) -> str:
        assert argv[0] == 'git'
        return head if argv[1] == 'rev-parse' else ''
    class Supervisor:
        def __init__(self, out: Path) -> None:
            pass
        def close(self) -> None:
            calls.append('closed')
    def completed(*args: object) -> dict[str, Json]:
        return {'status': 'COMPLETE', 'restore': {'status': restored}}
    def post_close(out: Path, tag: str) -> dict[str, Json]:
        assert calls[-1] == 'closed'
        calls.append('watch')
        return {'status': watch_status}
    monkeypatch.setattr(subprocess, 'check_output', git)
    monkeypatch.setattr(pf, 'Supervisor', Supervisor)
    monkeypatch.setattr(tv, 'TV1Live', lambda *args, **kwargs: object())
    monkeypatch.setattr(tv, 'window', completed)
    monkeypatch.setattr(tv, 'post_close', post_close)
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        assert tv.main(['--execute', '--out', str(tmp_path/'run'), '--expected-experiment-commit', head]) == expected
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)
    assert ('watch' in calls) == restored.startswith('RESTORED')
