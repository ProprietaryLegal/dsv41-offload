"""CPU protocol and failure rehearsals. No GPU, host, unit or live listener."""
from __future__ import annotations

import faulthandler
import importlib
import json
import os
import platform
import runpy
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from offload.dspark.probabilistic.pp4 import server_probe as probe
from offload.dspark.probabilistic.pp4 import server_probe_child as child
from offload.dspark.probabilistic.pp4 import server_probe_frames as frames
from offload.dspark.probabilistic.pp4 import server_probe_trace as trace
from offload.dspark.probabilistic.pp4.pair_qualify import SOURCES
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import pp4_prerequisites as prereq
from offload.launch.windows import scoped_remote as owner
from offload.launch.windows.tests.test_pp4_pair import receipt


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> probe.Config:
    monkeypatch.setattr(probe, 'SCRATCH', tmp_path)
    monkeypatch.setattr(platform, 'node', lambda: 'x86host')
    monkeypatch.setattr(probe, 'check_memory', lambda: None)
    monkeypatch.setenv('WINDOW_COORDINATOR', 'Deepseek/coordinator')
    c = probe.Config(tmp_path/'base', tmp_path/'candidate', 'a'*40, tmp_path/'runtime.json',
                     tmp_path/'pair.json', tmp_path/'run', 'PP4-server-cpu', 18644,
                     Path('/opt/dsv41/x86host-data/dspark'))
    driver = c.candidate/'offload/dspark/probabilistic/pp4/server_probe.py'
    driver.parent.mkdir(parents=True)
    for source in probe.DRIVER.parent.glob('server_probe*.py'):
        (driver.parent/source.name).write_bytes(source.read_bytes())
    monkeypatch.setattr(probe, 'DRIVER', driver)
    for name in set(SOURCES)-{'offload/dspark/worker/server.py'}:
        path = c.baseline/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(subprocess.check_output(['git', 'show', probe.BASELINE+':'+name]))
        copied = c.candidate/name
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_bytes(path.read_bytes())
    (c.baseline/'offload/dspark/worker/server.py').write_bytes(subprocess.check_output(
        ['git', 'show', probe.BASELINE+':offload/dspark/worker/server.py']))
    (c.candidate/'offload/dspark/worker/server.py').write_bytes((pf.ROOT/'offload/dspark/worker/server.py').read_bytes())
    monkeypatch.setattr(probe, 'git', lambda tree, *args: (probe.BASELINE if tree == c.baseline else c.commit)
                        if args[0] == 'rev-parse' else '')
    physical = dict(receipt())
    physical.update(source_commit=probe.BASELINE, source_sha256={name: probe.digest(c.baseline/name)
                    for name in set(SOURCES)-{'offload/dspark/worker/server.py'}})
    probe.write(c.pair_receipt, physical)
    artifacts = {}
    for label in probe.FILES:
        path = tmp_path/'runtime'/label
        path.parent.mkdir(exist_ok=True)
        path.write_text('CPU fixture '+label)
        artifacts[label] = {'path': str(path), 'sha256': probe.digest(path)}
    key = pf.digest(pf.ROOT/'offload/dspark/worker/p2p/p2p.cu')[:16]
    staging = tmp_path/'staging.json'
    probe.write(staging, {'source_commit': c.commit, 'staging_verdict': 'REVIEWED',
        'runtime_python': probe.PYTHON, 'runtime_torch': '2.12.1+cu130', 'runtime_artifact_sha256': {
            prereq.STOCK: artifacts['stock']['sha256'], prereq.FACC: artifacts['facc']['sha256'],
            f'p2p/{key}/dspark_p2p.so': artifacts['p2p']['sha256']}})
    probe.write(c.runtime, {'schema': 'pp4-server-probe-runtime-v1', 'reviewed': True,
        'python': probe.PYTHON, 'torch': '2.12.1+cu130', 'files': artifacts,
        'staging_receipt': str(staging), 'staging_sha256': probe.digest(staging)})
    return c


def test_preflight_binds_actual_sources_and_qualified_runtime(config: probe.Config) -> None:
    assert probe.preflight(config)['files'] == probe.read(config.runtime)['files']
    assert not config.out.exists()


@pytest.mark.parametrize('damage', ['host', 'coordinator', 'baseline', 'server', 'kernel', 'runtime', 'pair', 'port', 'out'])
def test_preflight_refuses_before_mutation(config: probe.Config, monkeypatch: pytest.MonkeyPatch, damage: str) -> None:
    if damage == 'host': monkeypatch.setattr(platform, 'node', lambda: 'Controller')
    if damage == 'coordinator': monkeypatch.delenv('WINDOW_COORDINATOR')
    if damage == 'baseline': monkeypatch.setattr(probe, 'git', lambda *args: 'b'*40)
    if damage in ('server', 'kernel'):
        path = config.candidate/'offload/dspark/worker'/('server.py' if damage == 'server' else 'sampled_fused.py')
        path.write_text(path.read_text()+'# unreviewed\n')
    if damage == 'runtime': Path(probe.read(config.runtime)['files']['stock']['path']).write_text('different')
    if damage == 'pair':
        data = probe.read(config.pair_receipt); data['source_commit'] = 'b'*40; probe.write(config.pair_receipt, data)
    if damage == 'port': config = replace(config, port=18620)
    if damage == 'out': config.out.mkdir()
    existed = config.out.exists()
    with pytest.raises(ValueError, match='STOP:'):
        probe.preflight(config)
    assert config.out.exists() is existed


def test_dry_default_has_no_process_or_output(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, 'argv', ['server_probe'])
    monkeypatch.setattr(probe, 'execute', lambda *args: pytest.fail('dry execute'))
    probe.main()
    assert 'DRY:' in capsys.readouterr().out


@pytest.mark.parametrize('fault', ['none', 'lost-open', 'probe', 'close', 'fence', 'close-fence', 'before-open', 'signal', 'missing-primary'])
def test_shared_local_fence_and_close_every_exit(config: probe.Config, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    events: list[str] = []
    monkeypatch.setattr(probe, 'preflight', lambda c: probe.read(c.runtime))
    monkeypatch.setattr(owner, 'command', lambda mode, tag, *args, **kwargs: tag)
    def local(command: str, timeout: float, log: Path) -> tuple[int, str]:
        events.append(command)
        if command.endswith('-probe'):
            if fault != 'before-open': (config.out/'pair-attempted').touch()
            if fault == 'lost-open': raise subprocess.TimeoutExpired('owned open', 120)
            if fault == 'signal': raise KeyboardInterrupt('fixture signal')
            if fault not in ('missing-primary', 'before-open'):
                row: dict[str, object] = dict.fromkeys(('complete', 'pass', 'graphs', 'worker_exited', 'listener_absent'), True)
                row['signatures'] = ['CPU signature']
                probe.write(config.out/'comparison.json', {'reference': row, 'candidate': row, 'candidate_exact_reference': True})
            return (1, 'probe failed') if fault == 'probe' else (0, 'done')
        assert command.endswith('-close')
        assert events.index('fence-probe') < events.index(command)
        assert timeout == probe.RESTORE_S-90
        return (1, 'bad') if fault == 'close' else (0, 'PRODUCTION HOLD OK\n')
    def fence(tag: str, invoke: Any, *, hosts: tuple[str, ...]) -> None:
        assert hosts == ('localhost',)
        events.append('fence-'+('probe' if tag.endswith('-probe') else 'close'))
        if fault == 'fence' or (fault == 'close-fence' and tag.endswith('-close')):
            raise ValueError('fixture absence unproved')
    monkeypatch.setattr(probe, 'local', local)
    monkeypatch.setattr(owner, 'fence_all', fence)
    result = probe.execute(config)
    if fault in ('fence', 'before-open'):
        assert config.run+'-close' not in events
    else:
        assert config.run+'-close' in events
        assert 'fence-close' in events
    assert result['pair_restored'] is (fault not in ('fence', 'close', 'close-fence'))
    assert result['verdict'] == ('DIAGNOSTIC COMPLETE' if fault == 'none' else 'STOP')
    assert probe.read(config.out/'summary.json') == result


def test_private_empty_caches_and_fixed_environment(config: probe.Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('DSPARK_WORKER_TP_EMULATE', '1')
    monkeypatch.setenv('DSPARK_D2_ADAPTER', 'bad')
    monkeypatch.setenv('CUDA_MODULE_LOADING', 'EAGER')
    monkeypatch.setenv('TRITON_CACHE_DIR', '/forbidden/shared')
    runtime = probe.read(config.runtime)
    paths = []
    for variant, tree in (('baseline', config.baseline), ('candidate', config.candidate)):
        out = config.out/variant; out.mkdir(parents=True)
        env = probe.child_env(config, tree, out, runtime)
        assert env['DSPARK_SAMPLED_TOP32'] == '1' and env['CUDA_VISIBLE_DEVICES'] == pf.WORKER_GPUS
        assert 'DSPARK_WORKER_TP_EMULATE' not in env and 'DSPARK_D2_ADAPTER' not in env
        assert env.get('CUDA_MODULE_LOADING') is None
        assert str(tree/'offload/dspark') in env['PYTHONPATH'].split(os.pathsep)
        for key in ('TRITON_CACHE_DIR', 'CUDA_CACHE_PATH', 'TORCH_EXTENSIONS_DIR', 'TMPDIR'):
            cache = Path(env[key]); assert cache.is_relative_to(out) and not list(cache.iterdir())
        paths.append(env['TRITON_CACHE_DIR'])
    assert paths[0] != paths[1]


class SampleBackend:
    def __init__(self) -> None:
        self.steps: list[int] = []
        self.rows = 0
        self.drafts: list[tuple[int, int, list[int]]] = []
    def set_max_model_len(self, n: int) -> None: pass
    def reset(self, slot: int) -> None: pass
    def stage(self, slot: int, first: int, rows: Any) -> None: self.rows += len(rows)
    def discard(self) -> None: self.rows = 0
    def healthy(self) -> bool: return True
    def flush(self) -> None: pass
    def draft_sampled(self, requests: list[Any], depth: int, keys: Any, mask: list[int]) -> tuple[Any, Any, Any, Any]:
        self.steps.append(len(requests))
        self.drafts.append((len(requests), self.rows, mask.copy()))
        ids = np.full((len(requests), depth), 2, np.int32)
        conf = np.full(ids.shape, .5, np.float32)
        support = np.tile(np.arange(32, dtype=np.int32), (len(requests), depth, 1))
        q = np.zeros_like(support, dtype=np.float32)
        for j, bit in enumerate(mask):
            if bit: q[j, :, 2] = 1.
        return ids, conf, support, q


@pytest.mark.parametrize('depth', [4, 5])
def test_actual_frames_core_and_local_ofl_reconnect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, depth: int) -> None:
    # Actual staged namespace and DraftCore/codec; transport alone is a CPU fake.
    server = importlib.import_module('worker.server')
    p = importlib.import_module('worker.protocol')
    monkeypatch.setenv('DSPARK_SAMPLED_TOP32', '1')
    backend = SampleBackend()
    hello = server.expected_hello(b'a'*32, b'b'*32, hidden=2, row_bytes=12, sample_mode=2, num_spec=depth)
    core = server.DraftCore(hello, backend)
    observed: list[tuple[Any, Any, int, int]] = []
    session_drafts: dict[int, int] = {}
    handle = core.handle
    def record(frame: bytes) -> bytes:
        header = p.AppHeader.decode(frame)
        step = p.decode_step(frame[p.APP_HDR.size:], hello.row_bytes, hello.max_slots,
                             hello.max_rows_per_step) if header.op == p.OP_STEP else None
        before = len(backend.steps)
        response = handle(frame)
        assert isinstance(response, bytes)
        observed.append((header, step, before, len(backend.steps)))
        if header.op == p.OP_HELLO:
            session_drafts[header.session] = 0
        if step is not None and before < len(backend.steps):
            session_drafts[header.session] += 1
            if session_drafts[header.session] <= 2:
                assert all(core.slots[e.slot].rng.rows_for_anchor(e.anchor_pos) == hello.window for e in step.entries)
        return response
    monkeypatch.setattr(core, 'handle', record)
    direct = frames.direct(core, tmp_path)
    direct_frames = observed.copy()
    direct_drafts = backend.drafts.copy()
    observed.clear()
    backend.drafts.clear()
    frames.save(tmp_path/'hello.json', frames.hello_record(hello))
    calls: list[str] = []
    class Client:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            assert args == (1, '10.202.28.1', 18644, '10.202.28.1')
            calls.append('connect')
        def reg(self, response: bytearray) -> bytearray: return response
        def call(self, op: int, frame: bytes, regions: Any, **kwargs: Any) -> SimpleNamespace:
            assert kwargs['timeout_s'] == 30.
            response = core.handle(frame)
            regions[0][0][:len(response)] = response
            return SimpleNamespace(resp_len=len(response))
        def close(self) -> None: calls.append('close')
    monkeypatch.setattr(server, '_load_ofl', lambda: SimpleNamespace(Client=Client, BACKEND_TCP=1))
    over_ofl = frames.client(tmp_path, 18644)
    assert over_ofl['complete'] is True and over_ofl['signatures'] == direct
    assert calls == ['connect', 'close', 'connect', 'close']
    for traffic, drafts in ((direct_frames, direct_drafts), (observed, backend.drafts)):
        assert drafts[:2] == [(1, 2, [1]), (4, 28, [1]*4)]
        assert len(drafts) == 2*(2+3*4*10)
        for connection in (0, 1):
            session = frames.SESSION+connection
            events = [event for event in traffic if event[0].session == session]
            drafted = [i for i, (_, _, before, after) in enumerate(events) if after > before]
            first, second = drafted[:2]
            assert second == first+1  # No RESET, HELLO, other width or draft in between.
            assert [event[0].op for event in events[:first]] == [p.OP_HELLO, p.OP_STEP]
            setup = events[1][1]
            assert setup.n_req == 4 and setup.n_rows == 512
            assert all(e.req_flags == p.REQ_NEW_GEN and e.n_rows == 128 for e in setup.entries)
            assert setup.sample_meta == [(4103+i, 0) for i in range(4)]
            ends = {e.slot: e.anchor_pos for e in setup.entries}
            for index, batch, count in ((first, 1, 2), (second, 4, 28)):
                header, step, _, _ = events[index]
                assert header.op == p.OP_STEP and (step.n_req, step.n_rows) == (batch, count)
                assert step.sample_meta == [(4103+i, 1) for i in range(batch)]
                for entry in step.entries:
                    assert entry.generation == 1 and entry.req_flags == p.REQ_NEED_DRAFT
                    assert entry.first_pos == ends[entry.slot]
                    assert entry.anchor_pos == entry.first_pos+entry.n_rows
                    ends[entry.slot] = entry.anchor_pos
            # The original warm matrix starts only after the cold pair.
            assert events[second+1][0].op == p.OP_RESET
            widths = [events[i][1].n_req for i in drafted[2:]]
            assert widths == [batch for _ in frames.MODES for batch in frames.BATCHES for _ in range(frames.ROUNDS)]
            assert events[-2][0].op == p.OP_RESET and events[-1][0].op == p.OP_PING
    assert len(direct) == 2*(3+3*4*10)
    for batch in frames.BATCHES:
        body = frames.step_body(batch, 1, 'mixed', 12, depth)
        step = p.decode_step(body, 12, 4, 512)
        assert step.n_rows == (2 if batch == 1 else 7 if batch == 4 else 5)*batch
        assert step.sample_meta == [(4103+i, m) for i, m in enumerate(frames.masks(batch, 'mixed'))]
    # Position-defined synthetic taps are unchanged by chunking or another call.
    assert frames.rows(127, 2, 0, 12) == frames.rows(127, 1, 0, 12)+frames.rows(128, 1, 0, 12)


def test_model_initial_positive_widths_precede_warm_matrix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from offload.dspark.probabilistic.pp4 import full_worker

    monkeypatch.setattr(torch.Tensor, 'pin_memory', lambda self: self)
    calls: list[tuple[int, int, list[bool]]] = []
    def sampled(inputs: torch.Tensor, k: int, keys: torch.Tensor, mask: torch.Tensor,
                *, n_out: int) -> tuple[torch.Tensor, ...]:
        calls.append((inputs.shape[1], n_out, mask.tolist()))
        return (torch.zeros(len(mask), n_out, dtype=torch.int32), torch.zeros(len(mask), n_out),
                torch.zeros(len(mask), n_out, 32, dtype=torch.int32), torch.zeros(len(mask), n_out, 32))
    def greedy(inputs: torch.Tensor, k: int, *, n_out: int) -> tuple[torch.Tensor, ...]:
        return torch.zeros(inputs.shape[1], n_out, dtype=torch.int32), torch.zeros(inputs.shape[1], n_out), torch.empty(0)
    model = SimpleNamespace(s=SimpleNamespace(hidden=2), draft_sampled=sampled,
                            has_graph=lambda *args: True, draft_graphed=greedy)
    monkeypatch.setattr(full_worker, 'load_synthetic', lambda model, fixture: fixture[1])
    signatures = child.model_checks(model, tmp_path)
    assert calls[:4] == [(1, 4, [True]), (4, 4, [True]*4), (1, 5, [True]), (4, 5, [True]*4)]
    assert len(signatures) == len(calls) == 4+4*2*3*10
    assert calls[4:] == [(batch, depth, [bool(m) for m in frames.masks(batch, mode)])
                         for batch in frames.BATCHES for depth in (4, 5)
                         for mode in frames.MODES for _ in range(frames.ROUNDS)]


@pytest.mark.parametrize('fault', ['sticky', 'error', 'done', 'q', 'ids', 'epoch', 'repeat', 'proposed'])
def test_post_drain_peer_failures_refuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    monkeypatch.setattr(trace, 'OUT', tmp_path)
    monkeypatch.setattr(trace, 'EPOCHS', {})
    state = {r: {name: torch.zeros(2, dtype=torch.int32) for name in ('sticky', 'error', 'done', 'ids', 'q')}
             for r in (0, 1)}
    for rank in (0, 1): state[rank]['epoch'] = torch.ones(2, dtype=torch.int32)
    ctx = SimpleNamespace(state=state, batch=2, depth=5)
    ctx._pp4_probe_outputs = {0: torch.zeros(2, 5), 1: torch.zeros(2, 5)}
    ctx.model = SimpleNamespace(sampled_graphs={(2, 5, 5): {'context': ctx}})
    trace.snapshot(ctx)
    if fault == 'proposed': ctx._pp4_probe_outputs[1][0, 0] = 2
    elif fault != 'repeat': state[1][fault][0] = 2
    with pytest.raises(RuntimeError, match='after drain'):
        trace.snapshot(ctx)


def test_preimport_bootstrap_executes_real_cli_entry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[object] = []
    monkeypatch.setattr(child, 'guard', lambda: tmp_path)
    monkeypatch.setattr(child, 'install', lambda namespace, out: seen.append(namespace['__file__']))
    monkeypatch.setenv('PP4_SERVER_WEIGHTS', '/fixture-weights')
    monkeypatch.setenv('PP4_SERVER_PORT', '18644')
    monkeypatch.setattr(sys, 'argv', ['fixture'])
    for name in ('enable', 'register', 'dump_traceback_later', 'cancel_dump_traceback_later'):
        monkeypatch.setattr(faulthandler, name, lambda *args, _name=name, **kwargs: seen.append(_name))
    def module(name: str, **kwargs: object) -> None:
        assert name == 'worker.server' and kwargs == {'run_name': '__main__', 'alter_sys': True}
        assert sys.argv[sys.argv.index('--tp')+1] == '2'
        assert '--no-graphs' not in sys.argv and '--no-rdma' in sys.argv
        namespace = {'__package__': 'worker', '__file__': 'actual-server-fixture.py'}
        exec(compile('def main():\n    return 1\nmain()\n', 'fixture.py', 'exec'), namespace)  # noqa: S102 -- fixed local profile fixture
    monkeypatch.setattr(runpy, 'run_module', module)
    child.main()
    assert seen == ['enable', 'register', 'dump_traceback_later', 'actual-server-fixture.py', 'cancel_dump_traceback_later']
    assert sys.getprofile() is None


def test_real_owned_cpu_child_bound_dumps_and_exits(config: probe.Config, monkeypatch: pytest.MonkeyPatch) -> None:
    # Only this test's exact new-session CPU child is signalled. No user scope
    # or listener is created; real parent polling/terminate/wait are exercised.
    bootstrap = config.candidate/'offload/dspark/probabilistic/pp4/server_probe_child.py'
    bootstrap.write_text('''import faulthandler, signal, os, json, time
from pathlib import Path
faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=True)
faulthandler.dump_traceback_later(.05, repeat=True)
out=Path(os.environ['PP4_SERVER_OUT'])
(out/'bound.json').write_text(json.dumps({'phase':'CPU owned stalled frame','until':time.monotonic()+.15}))
print('BEGIN CPU owned stalled frame', flush=True)
time.sleep(10)
''')
    config.out.mkdir()
    files = [config.out/'lock0', config.out/'lock1']
    with files[0].open('w') as a, files[1].open('w') as b:
        result = probe.worker(config, 'baseline', config.baseline, probe.read(config.runtime), (a.fileno(), b.fileno()))
    assert result['complete'] is False and result['worker_exited'] is True and result['listener_absent'] is True
    assert 'bound exceeded at CPU owned stalled frame' in result['error']
    log = (config.out/'baseline/worker.log').read_text()
    assert 'BEGIN CPU owned stalled frame' in log and log.count('Timeout') >= 2
    assert str(bootstrap)+'", line 9 in <module>' in log
    assert result['elapsed_s'] < 5


@pytest.mark.parametrize('ram,swap,passed', [(48*1024*1024, 0, True), (48*1024*1024-1, 0, False),
                                          (64*1024*1024, 1, False)])
def test_memory_admission_uses_x86host_observation(tmp_path: Path, ram: int, swap: int, passed: bool) -> None:
    path = tmp_path/'meminfo'
    path.write_text(f'MemAvailable: {ram} kB\nSwapTotal: 1000 kB\nSwapFree: {1000-swap} kB\n')
    if passed:
        probe.check_memory(path)
    else:
        with pytest.raises(ValueError, match='RAM and zero used swap'):
            probe.check_memory(path)


def test_rank_records_flush_only_after_the_pair_returns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trace, 'OUT', tmp_path)
    monkeypatch.setattr(trace, 'PENDING', [])
    monkeypatch.setattr(trace, 'DEFER_EVENTS', False)
    with trace.phase('model._replay'):
        before = (tmp_path/'phases.jsonl').read_text()
        trace.DEFER_EVENTS = True
        trace.emit('BEGIN', phase='rank1')
        trace.emit('END', phase='rank1')
        trace.emit('BEGIN', phase='rank0')
        assert (tmp_path/'phases.jsonl').read_text() == before
        trace.emit('END', phase='rank0')
    assert not trace.DEFER_EVENTS and not trace.PENDING
    events = [json.loads(line) for line in (tmp_path/'phases.jsonl').read_text().splitlines()]
    assert [(r['event'], r['phase']) for r in events] == [('BEGIN', 'model._replay'), ('BEGIN', 'rank1'),
        ('END', 'rank1'), ('BEGIN', 'rank0'), ('END', 'rank0'), ('END', 'model._replay')]
