"""PF1-WINDOW CPU rehearsal: no SSH, HTTP, Spark, x86host or GPU. Every scenario must end in the W8 restore."""
from __future__ import annotations

import ast
import datetime
import hashlib
import shlex
import signal
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import pytest

from offload.prefill_maxlen.evidence import Json, record
from offload.launch.windows import pf1_window as w
from offload.launch.windows.up2_low import X86HOST_PROBE, ENGRAM_PROBE
from offload.prefill_trace import window_requests as wr
from offload.prefill_trace.emit_verify import render as verifier
from offload.prefill_trace.telemetry import FIELDS

TTFT = {(327680, 150000): 68.7, (327680, 300000): 140.2, (524288, 150000): 93.3, (524288, 300000): 189.0}
RESTORE_OK = {
    'start': 'abc1234 production\n2026-10-26T01:00:00Z START: K1 START VERIFIED\n',
    'check': ''.join(f'greedy K1-post vs restored: {i} IDENTICAL {100 + i}\n' for i in range(6)) +
             'ok: status file tp 2\nPRODUCTION HOLD OK\n2026-10-26T01:01:00Z CHECK: DONE\n',
    'close': '2026-10-26T01:01:05Z CLOSE: window closed, production K1-327680 serving\n'}


class Clock:
    def __init__(self) -> None:
        self.now = time.time()

    def __call__(self) -> float:
        return self.now


def iso(t: float) -> str:
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def cold_row(cap: int, target: int, rep: int, ttft: float, kind: str = 'cold', run: str = 'run') -> dict[str, Json]:
    budget = 64 if kind == 'warmup' else 1500
    return {'type': 'request', 'kind': kind, 'rep': rep, 'target_ctx': target, 'cap': cap,
            'trace_request': hashlib.sha256(f'{run}:{kind}:{target}:{rep}'.encode()).hexdigest()[:24],
            'cache_salt_sha256': hashlib.sha256(f'salt:{target}:{rep}:{kind}'.encode()).hexdigest(),
            'prompt_sha256': hashlib.sha256(f'prompt:{target}'.encode()).hexdigest(),
            'prompt_chars': target * 4, 'prompt_pad_x': 0, 'cached_tokens': 0, 'uncached_prompt_tokens': target,
            'sampling': {'temperature': 1, 'top_p': 1, 'seed': 42, 'priority': 10, 'max_tokens': budget},
            'usage': {'prompt_tokens': target, 'completion_tokens': budget}, 'finish': 'length',
            'ttft_s': ttft, 'total_s': ttft + 27}


def off_cells(scale524: float = 1.355, noise: float = 0.0) -> dict[str, list[dict[str, Json]]]:
    out: dict[str, list[dict[str, Json]]] = {}
    for arm in w.OFF_ARMS:
        out[arm.name] = [cold_row(arm.cap, n, rep, TTFT[(327680, n)] * (scale524 if arm.cap == 524288 else 1) *
                                  (1 + noise * rep)) for n in w.OFF_CONTEXTS for rep in range(2)]
    return out


# ---------------------------------------------------------------- pure checks

def test_gate_outcomes_and_pairing() -> None:
    g = w.gate(off_cells())
    assert g['status'] == 'REPRODUCED'
    points = cast(list[dict[str, float]], g['points'])
    assert points[0]['excess_ms_per_prompt_token'] == pytest.approx((68.7 * 1.355 - 68.7) * 1000 / 150000)
    assert w.gate(off_cells(1.02))['status'] == 'NOT_REPRODUCED'
    assert w.gate(off_cells(1.07))['status'] == 'AMBIGUOUS'
    assert w.gate(off_cells(1.355, noise=.04))['status'] == 'AMBIGUOUS'
    broken = off_cells()
    broken['off-524'][0]['prompt_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='paired'):
        w.gate(broken)
    short = off_cells()
    short['off-327'].pop()
    with pytest.raises(ValueError, match='both repetitions'):
        w.gate(short)


@pytest.mark.parametrize('step,text,message', [
    ('start', 'START: K1 START FAILED', 'K1 START VERIFIED'),
    ('check', RESTORE_OK['check'].replace('5 IDENTICAL', '5 DIVERGES at char 3'), '6/6'),
    ('check', RESTORE_OK['check'].replace('PRODUCTION HOLD OK', 'prod check failed'), 'PRODUCTION HOLD OK'),
    ('check', RESTORE_OK['check'].replace('CHECK: DONE', ''), 'check not done'),
    ('close', 'window stays open', 'close not confirmed')])
def test_restore_strings_are_enforced(step: str, text: str, message: str) -> None:
    w.restored(step, 0, RESTORE_OK[step])
    with pytest.raises(ValueError, match=message):
        w.restored(step, 0, text)
    with pytest.raises(ValueError, match='rc1'):
        w.restored(step, 1, RESTORE_OK[step])


def test_hold_must_be_freshly_written_by_this_window() -> None:
    now = time.time()
    text = f'{iso(now)} DSV41-OFFLOAD (session coordinator): PF1-WINDOW tagA: matched'
    assert w.hold_opened(text, 'tagA', now) == pytest.approx(now, abs=1)
    for bad, at in ((text.replace('tagA', 'tagB'), now), (text.replace('coordinator', 'other'), now), (text, now + 600)):
        with pytest.raises(ValueError):
            w.hold_opened(bad, 'tagA', at)


def prod_identity(host: int = 0) -> dict[str, Json]:
    mounted = {w.DIST + 'vllm/v1/worker/gpu/model_runner.py': 'a' * 64,
               w.DIST + 'vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py': 'b' * 64,
               w.DIST + 'vllm/models/deepseek_v41/attention.py': 'c' * 64, '/opt/dsv41-ofl/libofl.so': 'd' * 64,
               '/serve.sh': f'{host}' * 64}
    return {'id': f'prod{host}', 'state': 'running', 'image': w.IMAGE, 'engine': '0.31.1', 'files': {'/srv/dsv41/legal-v2a/config.json': 'f'},
            'mounted': cast(Json, mounted), 'env': {'VLLM_ENGRAM_REMOTE_REQUIRED': '0', 'VLLM_DSV41_KV_PAIR': '3', 'NCCL_PROTO': '^LL128',
                    'VLLM_EXL3_TF_PROMPT_TRANSIENT': '1', 'VLLM_EXL3_TF_PROMPT_ALIAS': '1'},
            'engram_identity': {'shim': 'e' * 64},
            'argv': 'vllm serve /srv/dsv41/legal-v2a --port 8000 --max-model-len 327680 \\\n --max-num-seqs 4 --enable-prompt-tokens-details\n'}


def live_identity(arm: w.Arm, run: str, host: int = 0) -> dict[str, Json]:
    live = prod_identity(host)
    live['id'] = f'{arm.name}{host}'
    mounted = record(live['mounted'])
    mounted['/serve.sh'] = '9' * 64
    if arm.trace:
        mounted.update({w.DIST + p: sha for p, sha in w.overlay_pins(arm.cause).items()})
    env = record(live['env'])
    env['VLLM_ENGRAM_REMOTE_REQUIRED'] = '1'
    if arm.trace:
        env.update({'VLLM_DSV41_PREFILL_TRACE': '1', 'VLLM_DSV41_PREFILL_TRACE_DIR': '/pf1-trace',
                    'VLLM_DSV41_PREFILL_TRACE_RUN': run})
    if arm.cause:
        env.update({'VLLM_EXL3_TF_PROMPT_TRANSIENT': str(arm.transient), 'VLLM_EXL3_TF_PROMPT_ALIAS': str(arm.transient)})
    live['argv'] = (str(live['argv']).replace('--port 8000', '--port 8001').replace('327680', str(arm.cap))
                    .replace('details', 'details --enforce-eager' if arm.eager else 'details'))
    return live


@pytest.mark.parametrize('arm', [*w.OFF_ARMS, *w.ON_ARMS, w.Arm('off-e', 524288, 0, 1)])
def test_runtime_identity_equals_production_except_declared(arm: w.Arm) -> None:
    w.runtime(prod_identity(), live_identity(arm, 'r'), arm, 'r')


@pytest.mark.parametrize('mutate,message', [
    (lambda d: record(d['mounted']).update({w.DIST + 'vllm/models/deepseek_v41/attention.py': '0' * 64}), 'mounted'),
    (lambda d: record(d['mounted']).update({w.DIST + 'vllm/v1/worker/gpu/extra.py': '0' * 64}), 'mounted'),
    (lambda d: record(d['env']).update({'NCCL_PROTO': 'LL'}), 'env drift'),
    (lambda d: record(d['env']).update({'VLLM_ENGRAM_REMOTE_REQUIRED': '0'}), 'REQUIRED'),
    (lambda d: d.update({'argv': str(d['argv']).replace('--max-num-seqs 4', '--max-num-seqs 2')}), 'argv'),
    (lambda d: d.update({'image': 'sha256:other'}), 'image'),
    (lambda d: d.update({'engram_identity': {'shim': '0'}}), 'Engram'),
    (lambda d: d.update({'files': {}}), 'target metadata')])
def test_identity_mismatches_stop(mutate: Callable[[dict[str, Json]], object], message: str) -> None:
    arm = w.OFF_ARMS[1]
    live = live_identity(arm, 'r')
    mutate(live)
    with pytest.raises(ValueError, match=message):
        w.runtime(prod_identity(), live, arm, 'r')


def test_trace_arm_must_carry_exact_declared_sources() -> None:
    arm = w.ON_ARMS[0]
    live = live_identity(arm, 'r')
    record(live['mounted'])[w.DIST + 'vllm/v1/worker/gpu/dsv41_prefill_trace.py'] = '0' * 64
    with pytest.raises(ValueError, match='mounted'):
        w.runtime(prod_identity(), live, arm, 'r')
    with pytest.raises(ValueError, match='VLLM_DSV41_PREFILL_TRACE_RUN'):
        w.runtime(prod_identity(), live_identity(arm, 'other'), arm, 'r')


def log_summary(cap: int, **over: Json) -> dict[str, Json]:
    base: dict[str, Json] = {'lines': 100, 'kv_tokens': [w.KV[cap]], 'proxy_pids': [123], 'async_on': 1,
                             'worker_ready': 1, 'engram_remote': 2, 'stop_hits': 0, 'stop_examples': []}
    return {**base, **over}


@pytest.mark.parametrize('over,message', [({'kv_tokens': [2502600]}, 'KV'), ({'stop_hits': 1}, 'STOP pattern'),
    ({'worker_ready': 0}, 'HELLO'), ({'proxy_pids': [1, 2]}, 'proxy'), ({'engram_remote': 0}, 'Engram')])
def test_log_gates(over: dict[str, Json], message: str) -> None:
    w.logs(log_summary(524288), 0, 524288)
    with pytest.raises(ValueError, match=message):
        w.logs(log_summary(524288, **over), 0, 524288)
    # A long-lived production log is a record: only its KV line binds.
    w.logs(log_summary(327680, stop_hits=3, worker_ready=0), 0, 327680, production=True)


def request_rows(arm: w.Arm, run: str, contexts: tuple[int, ...], reps: int) -> list[dict[str, Json]]:
    rows: list[dict[str, Json]] = [{'type': 'identity', 'cap': arm.cap, 'trace_run': run, 'corpus_sha256': {}, 'seed': 42}]
    rows.append(cold_row(arm.cap, 32000, 0, 15.0, 'warmup', run))
    rows += [cold_row(arm.cap, n, rep, TTFT.get((arm.cap, n), 70.0), run=run) for n in contexts for rep in range(reps)]
    return rows


def test_request_cells_fail_closed() -> None:
    arm = w.OFF_ARMS[0]
    rows = request_rows(arm, 'r', w.OFF_CONTEXTS, 2)
    assert len(w.cells(rows, arm, 'r', w.OFF_CONTEXTS, 2)) == 4
    for index, key, value in ((1, 'kind', 'cold'), (2, 'cached_tokens', 5), (3, 'rep', 0), (4, 'ttft_s', None),
                              (2, 'sampling', {'max_tokens': 64}), (0, 'cap', 524288)):
        bad = json.loads(json.dumps(rows))
        bad[index][key] = value
        with pytest.raises(ValueError):
            w.cells(bad, arm, 'r', w.OFF_CONTEXTS, 2)


def write_traces(directory: Path, rank: int, rows: list[dict[str, Json]], *, drop_total: bool = False,
                 hash_ids: bool = True) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in rows:
        if row.get('type') != 'request':
            continue
        target = int(cast(int, row['target_ctx']))
        key = str(row['trace_request']) if hash_ids else hashlib.sha256(str(row['trace_request']).encode()).hexdigest()[:24]
        for prefix in range(0, target, 4096):
            new = min(4096, target - prefix)
            member = {'request': key, 'chunk': f'{key}:{prefix}:{new}:0', 'new_tokens': new, 'prefix_tokens': prefix}
            base = {'rank': rank, 'run': 'r', 'max_model_len': row['cap'], 'members': [member]}
            lines.append({**base, 'phase': 'chunk.begin'})
            lines.append({**base, 'phase': 'indexer.topk', 'cuda_ms': 1.0, 'wall_ms': 1.0})
            if not (drop_total and prefix == 0):
                lines.append({**base, 'phase': 'chunk.total', 'cuda_ms': 1800.0, 'wall_ms': 1801.0})
    (directory / f'prefill-r{rank}-p77.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in lines))


def test_trace_completeness_and_correlation(tmp_path: Path) -> None:
    arm = w.ON_ARMS[1]
    rows = request_rows(arm, 'r', (150000,), 1)
    for rank in (0, 1):
        write_traces(tmp_path / f'ok{rank}', rank, rows)
    files = {rank: [tmp_path / f'ok{rank}/prefill-r{rank}-p77.jsonl'] for rank in (0, 1)}
    result = w.trace_check(files, rows)
    assert result['correlation'] == 'HASH'
    assert cast(dict[int, dict[str, int]], result['ranks'])[0]['requests'] == 2
    write_traces(tmp_path / 'id1', 1, rows, hash_ids=False)
    write_traces(tmp_path / 'id0', 0, rows, hash_ids=False)
    by_size = w.trace_check({0: [tmp_path / 'id0/prefill-r0-p77.jsonl'], 1: [tmp_path / 'id1/prefill-r1-p77.jsonl']}, rows)
    assert str(by_size['correlation']).startswith('BY_SIZE_ONLY')
    write_traces(tmp_path / 'bad1', 1, rows, drop_total=True)
    with pytest.raises(ValueError, match='incomplete'):
        w.trace_check({0: files[0], 1: [tmp_path / 'bad1/prefill-r1-p77.jsonl']}, rows)
    with pytest.raises(ValueError, match='rank1'):
        w.trace_check({0: files[0], 1: []}, rows)
    with pytest.raises(ValueError, match='unsent request of 150000 tokens'):
        w.trace_check(files, rows[:-1])


# ---------------------------------------------------------------- orchestration with a Backend fake

class FakeBackend:
    def __init__(self, fail: str = '', scale: float = 1.355, restore_fails: bool = False) -> None:
        self.calls: list[str] = []
        self.fail, self.scale, self.restore_fails = fail, scale, restore_fails

    def step(self, name: str) -> None:
        self.calls.append(name)
        if name == self.fail:
            raise ValueError(f'STOP: injected {name}')

    def preflight(self) -> None: self.step('preflight')
    def open_hold(self) -> float:
        self.step('open_hold')
        return 1.0
    def snapshot(self) -> None: self.step('snapshot')
    def stage_trace(self) -> None: self.step('stage_trace')
    def start(self, arm: w.Arm) -> None: self.step(f'start:{arm.name}')

    def requests(self, arm: w.Arm, contexts: tuple[int, ...], reps: int) -> list[dict[str, Json]]:
        self.step(f'requests:{arm.name}')
        scale = self.scale if arm.cap == 524288 else 1
        return [cold_row(arm.cap, n, rep, TTFT[(327680, n)] * scale) for n in contexts for rep in range(reps)]

    def collect(self, arm: w.Arm, rows: list[dict[str, Json]]) -> dict[str, Json]:
        self.step(f'collect:{arm.name}')
        return {}

    def restore(self, *, production_stopped: bool) -> dict[str, Json]:
        self.calls.append(f'restore:{production_stopped}')
        if self.restore_fails:
            raise ValueError('STOP: greedy is not 6/6 IDENTICAL')
        return {'status': 'RESTORED'}


def run_window(backend: FakeBackend, phase: str = 'all', prior: dict[str, Json] | None = None) -> dict[str, Json]:
    return w.window(backend, phase, lambda _row: None, prior_gate=prior)


def test_full_sequence_order_and_restore() -> None:
    b = FakeBackend()
    result = run_window(b)
    arms = [f'{s}:{a.name}' for a in (*w.OFF_ARMS,) for s in ('start', 'requests', 'collect')]
    arms += ['stage_trace'] + [f'{s}:{a.name}' for a in w.ON_ARMS for s in ('start', 'requests', 'collect')]
    assert b.calls == ['preflight', 'open_hold', 'snapshot', *arms, 'restore:True']
    assert result['status'] == 'COMPLETE' and record(result['gate'])['status'] == 'REPRODUCED'
    assert 'stop' not in result


@pytest.mark.parametrize('scale,status', [(1.0, 'NOT_REPRODUCED'), (1.07, 'AMBIGUOUS')])
def test_gate_stop_skips_on_arms_and_restores(scale: float, status: str) -> None:
    b = FakeBackend(scale=scale)
    result = run_window(b)
    assert not any(c.startswith(('stage', 'start:on')) for c in b.calls) and b.calls[-1] == 'restore:True'
    assert result['status'] == f'COMPLETE: gate {status}, ON skipped'


@pytest.mark.parametrize('fail,stopped', [('start:off-327', True), ('start:off-524', True), ('requests:off-524', True),
    ('start:on-327', True), ('collect:on-524', True), ('snapshot', False), ('open_hold', False)])
def test_every_failure_after_hold_restores(fail: str, stopped: bool) -> None:
    b = FakeBackend(fail=fail)
    result = run_window(b)
    assert b.calls[-1] == f'restore:{stopped}' and 'injected' in str(result['stop'])
    assert b.calls.index(fail) == len(b.calls) - 2


def test_preflight_failure_needs_no_restore() -> None:
    b = FakeBackend(fail='preflight')
    result = run_window(b)
    assert b.calls == ['preflight'] and str(record(result['restore'])['status']).startswith('NOT NEEDED')


def test_failed_restore_is_reported_not_hidden() -> None:
    result = run_window(FakeBackend(fail='requests:off-327', restore_fails=True))
    assert record(result['restore'])['status'] == 'FAILED' and 'IDENTICAL' in str(record(result['restore'])['error'])


def test_on_phase_needs_reproduced_prior_gate() -> None:
    b = FakeBackend()
    result = run_window(b, 'on', {'status': 'REPRODUCED'})
    assert 'start:off-327' not in b.calls and 'start:on-524' in b.calls and result['status'] == 'COMPLETE'
    b = FakeBackend()
    result = run_window(b, 'on', None)
    assert 'prior OFF window gate' in str(result['stop']) and b.calls[-1] == 'restore:False'


# ---------------------------------------------------------------- the real Live backend through a fake command layer

MONITOR = '''
import json, sys, time
import os
mode, trip = sys.argv[1], sys.argv[2]
while True:
    tripped = os.path.exists(trip)
    avail = 3000000 if mode == 'low' and tripped else 7000000
    swap = 1000000 - (900000 if mode == 'swap' and tripped else 0)
    print(json.dumps({'kind': 'memory', 'utc': time.time(), 'kb': {'MemAvailable': avail, 'SwapFree': swap, 'SwapTotal': 2000000}}), flush=True)
    if mode == 'gpu' and tripped:
        print(json.dumps({'kind': 'gpu_error', 'utc': time.time(), 'error': 'TimeoutExpired'}), flush=True)
    else:
        print(json.dumps({'kind': 'gpu', 'utc': time.time(), 'csv': 'GPU-x,2100,4000,50.1,100,60,90,0x0'}), flush=True)
    time.sleep(.02)
'''


class NoTunnel(w.Supervisor):
    def tunnel(self, command: list[str]) -> None:
        self.tunnel_command = command


class FakeRemote:
    """Answers every command the live backend issues, from the saved W13 formats; never opens a socket or SSH."""

    def __init__(self, out: Path, sup: w.Supervisor, clock: Clock, fail: str = '', demand: float = 0,
                 trip: Path | None = None) -> None:
        self.out, self.sup, self.clock, self.fail, self.trip = out, sup, clock, fail, trip
        self.arm: w.Arm | None = None
        self.hold = ''
        self.commands: list[str] = []
        self.restores: list[str] = []
        self.unsupervised: list[tuple[str, str, str]] = []
        self.busy_polls = 0
        self.steps = 100
        self.rounds = 10
        self.serial = 0
        self.demand = demand

    def identity(self, rank: int) -> dict[str, Json]:
        if self.arm is None:
            return prod_identity(rank)
        live = live_identity(self.arm, f'tag-{self.arm.name}', rank)
        live['id'] = f'{self.arm.name}-{rank}-{self.serial}'
        if self.fail == 'identity' and self.arm.name == 'off-524':
            record(live['mounted'])[w.DIST + 'vllm/models/deepseek_v41/attention.py'] = '0' * 64
        return live

    def call(self, label: str, argv: list[str], *, stdin: str | None = None, timeout: float = 120, inflight: int = 0) -> str:
        self.sup.poll()
        self.commands.append(label)
        if argv[0] == 'ssh':
            return self.remote(label, argv[-2], argv[-1], stdin)
        if argv[1:3] == ['-m', 'offload.prefill_trace.window_requests']:
            return self.requests(argv, inflight)
        if argv[0] == 'scp' and 'pf1-trace' in argv[-2]:
            arm = cast(w.Arm, self.arm)
            rank = w.HOSTS.index(argv[-2].split(':')[0])
            rows = [record(cast(Json, json.loads(x))) for x in (self.out / f'{arm.name}-requests.jsonl').read_text().splitlines()]
            write_traces(Path(argv[-1]), rank, rows)
        return ''

    def requests(self, argv: list[str], inflight: int) -> str:
        arm = cast(w.Arm, self.arm)
        args = {argv[i]: argv[i + 1] for i in range(3, len(argv) - 1) if argv[i].startswith('--')}
        contexts = tuple(int(x) for x in argv[argv.index('--contexts') + 1:argv.index('--reps')])
        reps = int(args['--reps'])
        deadline = float(args['--deadline-utc'])
        if self.trip is not None:
            self.trip.touch()   # the fake Spark monitors now report the injected memory/swap/GPU fault
        ck_end = time.monotonic() + .6
        self.sup.inflight = inflight
        try:
            while time.monotonic() < ck_end:   # supervision continues while the owned driver runs
                self.sup.poll()
        finally:
            self.sup.inflight = 0
        if self.clock() >= deadline:
            raise ValueError('STOP: requests: window submission deadline reached')
        rows = request_rows(arm, args['--trace-run'], contexts, reps)
        scale = 1.355 if arm.cap == 524288 else 1.0
        for row in rows[2:]:
            row['ttft_s'] = TTFT[(327680, int(cast(int, row['target_ctx'])))] * scale
        Path(args['--output']).write_text(''.join(json.dumps(r) + '\n' for r in rows))
        self.rounds += 50
        self.steps += 50
        return ''

    def remote(self, label: str, host: str, command: str, stdin: str | None) -> str:
        rank = w.HOSTS.index(host) if host in w.HOSTS else -1
        if 'window.sh open' in command:
            self.hold = f'{iso(self.clock())} DSV41-OFFLOAD (session coordinator): {command.split("PF1-WINDOW ")[1][:-1]}'
            return 'window open'
        if 'cat ~/dsv41-deploy/HOLD' in command:
            return self.hold
        if 'bash offload/launch/start_exp.sh' in command:
            head = command.split(' && ')[1].split(' bash ')[0].split()
            assert head[-2:] == ['timeout', str(w.START_S)]
            env = dict(x.split('=') for x in head[:-2])
            self.serial += 1
            cap, trace = int(env['PF1_CAP']), int(env['PF1_TRACE'])
            self.arm = next(a for a in (*w.OFF_ARMS, *w.ON_ARMS, *w.CAUSE_ARMS, w.CAUSE_T0)
                            if a.cap == cap and a.trace == trace and a.cause == (w.CAUSE_CONFIG in command)
                            and a.transient == int(env.get('PF1C_TF_TRANSIENT', '1')))
            if self.fail == 'start' and self.arm.name == 'off-524':
                raise ValueError('STOP: off-524-start failed rc1; see log')
            return 'ready'
        if host == 'x86host':
            assert shlex.quote(X86HOST_PROBE) in command
            self.steps += 1
            return json.dumps({'revision': w.WORKER, 'current': '/w/fb68c52', 'cwd': '/w/fb68c52/offload/dspark',
                'cmd': ['python', '-m', 'worker.server', '--tp', '2', '--max-model-len', '1048576'], 'gpus': w.WORKER_GPUS,
                'status': {'tp': 2, 'tp_comm_errors': 0, 'steps': self.steps, 'time': iso(self.clock())}})
        if command == 'bash -s':
            ident = self.identity(rank)
            container = {'id': ident['id'], 'image': ident['image'], 'state': ident['state']}
            files = {'engine': ident['engine'], 'files': ident['files'], 'mounted_code_sha256': ident['mounted'],
                     'generation_config': {'exists': False, 'sha256': None}}
            return json.dumps(container) + '\n' + json.dumps(files)
        if "{{json .Config}}" in command:
            config_env = record(self.identity(rank)['env'])
            return json.dumps({'Env': [f'{k}={v}' for k, v in config_env.items()] + ['HOME=/root']})
        if "{{.Id}}" in command:
            return str(self.identity(rank)['id']) + '\n'
        if 'cat /serve.sh' in command:
            return str(self.identity(rank)['argv'])
        if stdin == ENGRAM_PROBE:
            return json.dumps(self.identity(rank)['engram_identity'])
        if stdin in (verifier(False), verifier(True), verifier(True, cause=True)):
            count = 18 if stdin == verifier(True, cause=True) else 17 if stdin == verifier(True) else 13
            return f'PF1 verified {count} engine sources\n'
        if stdin == w.IMAGE_BASES:
            return json.dumps(w.image_pins())
        if stdin is not None and '--pid' in command:
            self.rounds += 1
            return json.dumps({'counters': {'rounds_ok': self.rounds}, 'policy': 'capacity', 'async': True,
                'async_proxy': {'mode': 'process'}, 'async_handoff': {'mlock': 'ok'},
                'async_watchdog': {'fires': 0, 'running': True}, 'proxy_torch_loaded': False})
        if 'kv_tokens' in command:
            cap = self.arm.cap if self.arm is not None else 327680
            return json.dumps(log_summary(cap))
        if command == 'curl -fsS --max-time 10 http://localhost:8000/metrics':
            busy = 1 if self.busy_polls > 0 else 0
            self.busy_polls -= 1
            return f'vllm:num_requests_running{{m="x"}} {busy}\nvllm:num_requests_waiting{{m="x"}} 0\n'
        if 'localhost:800' in command and 'v1/models' in command:
            cap = self.arm.cap if self.arm is not None and ':8001' in command else 327680
            return json.dumps({'data': [{'id': 'm', 'root': '/srv/dsv41/legal-v2a', 'max_model_len': cap}]})
        if 'sha256sum offload' in command:
            return '\n'.join(f'{w.digest(w.ROOT / p)}  {p}' for p in shlex.split(command.split('sha256sum ', 1)[1])) + '\n'
        if command == w.HOST_PARITY:
            return 'test-kernel\nMemAvailable: 7000000 kB\ncompact_success 0\n'
        if 'nvidia-smi' in command:
            return (json.dumps({'kind': 'memory', 'kb': {'MemAvailable': 7000000, 'SwapFree': 1, 'SwapTotal': 2}}) + '\n' +
                    json.dumps({'kind': 'gpu', 'csv': 'GPU-x,2100,4000,50.1,100,60,90,0x0'}) + '\n')
        if 'if [ -e generated ]' in command:
            return 'ABSENT\n'
        return ''

    def plain(self, step: str, command: list[str], timeout: float) -> tuple[int, str]:
        if command[0] == 'ssh':   # S40 FIX 1-2: unsupervised reap, JIT wait and HOLD re-read before any restore step
            self.unsupervised.append((step, command[-2], command[-1]))
            if step == 'reap':
                assert "reap 'window[.]sh open PF1-WINDOW tag:' 'HOLD opener'" in command[-1]
                return 0, 'reap done\n'
            if step.startswith('jit-wait-'):
                return 0, 'no JIT prebuild running\n'
            assert step == 'hold-reread'
            return (255, '') if self.fail == 'hold-unreadable' else (0, self.hold)
        assert command == ['bash', str(w.RESTORE), step]
        self.restores.append(step)
        if self.fail == 'restore-check' and step == 'check':
            return 0, RESTORE_OK['check'].replace('3 IDENTICAL', '3 DIVERGES at char 9')
        return 0, RESTORE_OK[step]


@pytest.fixture
def harness(tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[tuple[w.Live, FakeRemote, Clock, w.Supervisor]]:
    params = cast(dict[str, object], getattr(request, 'param', {}))
    monitor = tmp_path / 'monitor.py'
    monitor.write_text(MONITOR)
    out = tmp_path / 'PF1-WINDOW-rehearsal'
    out.mkdir()
    clock = Clock()
    remote: list[FakeRemote] = []
    trip = tmp_path / 'trip'
    sup = NoTunnel(out, clock=clock, monitor=lambda host: [sys.executable, str(monitor), str(params.get('monitor', 'ok')), str(trip)],
                   demand=lambda: remote[0].demand)
    fake = FakeRemote(out, sup, clock, str(params.get('fail', '')), trip=trip if 'monitor' in params else None)
    remote.append(fake)
    live = w.Live(out, sup, remote='dsv41-deploy-offload', expected='0' * 40, data_dir=tmp_path, tag='tag',
                  stage_bundle=False, call=fake.call, plain=fake.plain, clock=clock)
    import offload.prefill_trace.window_requests as requests_module
    original = requests_module.corpus
    requests_module.corpus = lambda root: ('x', {})
    try:
        yield live, fake, clock, sup
    finally:
        requests_module.corpus = original
        sup.close()


def go(live: w.Live, phase: str = 'all') -> dict[str, Json]:
    rows: list[dict[str, Json]] = []
    return w.window(live, phase, rows.append)


def test_live_full_sequence_rehearsal(harness: tuple[w.Live, FakeRemote, Clock, w.Supervisor]) -> None:
    live, fake, _clock, sup = harness
    result = go(live)
    assert 'stop' not in result, result.get('stop')
    assert result['status'] == 'COMPLETE' and record(result['gate'])['status'] == 'REPRODUCED'
    assert fake.restores == ['start', 'check', 'close']
    assert str(record(result['restore'])['status']).startswith('RESTORED')
    assert [c for c in fake.commands if c.endswith('-start')] == ['off-327-start', 'off-524-start', 'on-327-start', 'on-524-start']
    checks = record(record(record(result['arms'])['on-524'])['checks'])
    assert record(checks['trace'])['correlation'] == 'HASH'
    assert (live.out / 'rollback.json').is_file() and (live.out / 'deadlines.json').is_file()
    assert json.loads((live.out / 'deadlines.json').read_text())['submission'] - json.loads(
        (live.out / 'deadlines.json').read_text())['opened'] == 75 * 60
    telemetry = (live.out / 'telemetry-sparkhead.jsonl').read_text().splitlines()
    assert telemetry and all(json.loads(x)['kind'] in ('memory', 'gpu') for x in telemetry)


@pytest.mark.parametrize('harness', [{'fail': 'start'}], indirect=True)
def test_live_failed_start_restores(harness: tuple[w.Live, FakeRemote, Clock, w.Supervisor]) -> None:
    live, fake, _clock, _sup = harness
    result = go(live)
    assert 'off-524-start failed' in str(result['stop']) and fake.restores == ['start', 'check', 'close']
    assert 'off-524' not in record(result['arms']) and 'off-327' in record(result['arms'])


@pytest.mark.parametrize('harness', [{'monitor': 'low'}, {'monitor': 'swap'}, {'monitor': 'gpu'}], indirect=True)
def test_live_memory_swap_and_telemetry_guards_restore(harness: tuple[w.Live, FakeRemote, Clock, w.Supervisor]) -> None:
    live, fake, _clock, _sup = harness
    result = go(live)
    assert any(s in str(result['stop']) for s in ('below 4 GiB', 'new swap use', 'GPU telemetry failing')), result['stop']
    assert fake.restores == ['start', 'check', 'close'] and 'off-524' not in record(result['arms'])


def test_live_deadline_expiry_restores(harness: tuple[w.Live, FakeRemote, Clock, w.Supervisor]) -> None:
    live, fake, clock, _sup = harness
    original = fake.requests
    def late(argv: list[str], inflight: int) -> str:
        clock.now += 81 * 60
        return original(argv, inflight)
    fake.requests = late  # type: ignore[method-assign]
    result = go(live)
    assert '80-minute measurement bound' in str(result['stop']) and fake.restores == ['start', 'check', 'close']


def test_live_submission_bound_refuses_new_arm(harness: tuple[w.Live, FakeRemote, Clock, w.Supervisor]) -> None:
    live, fake, clock, _sup = harness
    original = fake.requests
    def slow(argv: list[str], inflight: int) -> str:
        clock.now += 68 * 60
        return original(argv, inflight)
    fake.requests = slow  # type: ignore[method-assign]
    result = go(live)
    assert 'off-524: not enough time' in str(result['stop']) and fake.restores == ['start', 'check', 'close']


@pytest.mark.parametrize('harness', [{'fail': 'identity'}], indirect=True)
def test_live_identity_mismatch_restores(harness: tuple[w.Live, FakeRemote, Clock, w.Supervisor]) -> None:
    live, fake, _clock, _sup = harness
    result = go(live)
    assert 'mounted engine bytes differ from production' in str(result['stop']) and fake.restores == ['start', 'check', 'close']


def test_live_foreign_demand_restores(harness: tuple[w.Live, FakeRemote, Clock, w.Supervisor]) -> None:
    live, fake, _clock, _sup = harness
    fake.demand = 2
    result = go(live)
    assert 'foreign demand' in str(result['stop']) and fake.restores == ['start', 'check', 'close']


@pytest.mark.parametrize('harness', [{'fail': 'restore-check'}], indirect=True)
def test_live_unverified_restore_never_closes(harness: tuple[w.Live, FakeRemote, Clock, w.Supervisor]) -> None:
    live, fake, _clock, _sup = harness
    result = go(live)
    assert fake.restores == ['start', 'check'] and record(result['restore'])['status'] == 'FAILED'


def test_live_external_stop_file_restores(harness: tuple[w.Live, FakeRemote, Clock, w.Supervisor]) -> None:
    live, fake, _clock, _sup = harness
    original = fake.requests
    def stop(argv: list[str], inflight: int) -> str:
        (live.out / 'STOP').write_text('coordinator\n')
        return original(argv, inflight)
    fake.requests = stop  # type: ignore[method-assign]
    result = go(live)
    assert 'external STOP file' in str(result['stop']) and fake.restores == ['start', 'check', 'close']


def test_failure_before_hold_touches_nothing(harness: tuple[w.Live, FakeRemote, Clock, w.Supervisor]) -> None:
    live, fake, _clock, _sup = harness
    original = fake.remote
    def busy(label: str, host: str, command: str, stdin: str | None) -> str:
        if 'window.sh open' in command:
            raise ValueError('STOP: hold-open failed rc1 (serve still busy after drain)')
        return original(label, host, command, stdin)
    fake.remote = busy  # type: ignore[method-assign]
    result = go(live)
    assert 'hold-open failed' in str(result['stop']) and fake.restores == []
    assert str(record(result['restore'])['status']).startswith('NOT NEEDED')


# ---------------------------------------------------------------- supervisor, CLI and driver additions

def test_supervisor_owned_commands_timeout_failure_and_deadline_text(tmp_path: Path) -> None:
    sup = w.Supervisor(tmp_path)
    try:
        with pytest.raises(ValueError, match='timeout'):
            sup.run('slow', [sys.executable, '-c', 'import time; time.sleep(5)'], timeout=.1)
        with pytest.raises(ValueError, match='failed rc3'):
            sup.run('fails', [sys.executable, '-c', 'raise SystemExit(3)'])
        with pytest.raises(ValueError, match='submission deadline'):
            sup.run('late', [sys.executable, '-c', 'print("RuntimeError: window submission deadline reached"); raise SystemExit(1)'])
        assert 'ok' in sup.run('fine', [sys.executable, '-c', 'print("ok")'])
    finally:
        sup.close()


def test_dry_run_and_refusals_make_no_remote_call(tmp_path: Path) -> None:
    env = {k: v for k, v in os.environ.items() if k != 'WINDOW_COORDINATOR'}
    dry = subprocess.run([sys.executable, '-m', 'offload.launch.windows.pf1_window'], cwd=w.ROOT, text=True,
                         capture_output=True, check=True, env=env)
    assert dry.stdout.startswith('DRY RUN') and 'restore.sh start' in dry.stdout
    for extra, message in ((['--execute'], 'WINDOW_COORDINATOR'),):
        result = subprocess.run([sys.executable, '-m', 'offload.launch.windows.pf1_window', *extra, '--out', str(tmp_path / 'x')],
                                cwd=w.ROOT, text=True, capture_output=True, env=env)
        assert result.returncode == 1 and message in result.stderr and not (tmp_path / 'x').exists()
    coord = {**env, 'WINDOW_COORDINATOR': 'Deepseek/coordinator'}
    for extra, message in ((['--out', str(tmp_path / 'y')], 'campaign directory'),
                           (['--out', str(w.CAMPAIGN / 'PF1-WINDOW-never-created'), '--remote-repo', 'dsv41-deploy',
                             '--expected-experiment-commit', 'a' * 40], 'never ~/dsv41-deploy'),
                           (['--out', str(w.CAMPAIGN / 'PF1-WINDOW-never-created'), '--expected-experiment-commit', 'abc'], 'full reviewed')):
        result = subprocess.run([sys.executable, '-m', 'offload.launch.windows.pf1_window', '--execute', *extra],
                                cwd=w.ROOT, text=True, capture_output=True, env=coord)
        assert result.returncode == 1 and message in result.stderr
    assert not (w.CAMPAIGN / 'PF1-WINDOW-never-created').exists()


class FakeResponse:
    def __init__(self, data: bytes, lines: list[bytes] | None = None) -> None:
        self.data, self.lines = data, lines or []
    def __enter__(self) -> FakeResponse: return self
    def __exit__(self, *args: object) -> None: pass
    def read(self) -> bytes: return self.data
    def __iter__(self) -> Iterator[bytes]: return iter(self.lines)


def driver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: list[str], output: Path,
           drift: bool = False) -> tuple[list[dict[str, Json]], list[int]]:
    import urllib.request
    tokenized: list[int] = []
    monkeypatch.setattr(wr, 'corpus', lambda _root: (wr.HEADER + ('public text ' * 60000) + ('drift' if drift else ''),
                                                    {'public': 'sha'}))
    def open_fake(req: str | urllib.request.Request, timeout: float = 0) -> FakeResponse:
        url = req if isinstance(req, str) else req.full_url
        if url.endswith('/metrics'):
            return FakeResponse(b'vllm:num_requests_running{} 0\nvllm:num_requests_waiting{} 0\n')
        if url.endswith('/v1/models'):
            return FakeResponse(json.dumps({'data': [{'id': 'm', 'max_model_len': 327680}]}).encode())
        assert isinstance(req, urllib.request.Request) and isinstance(req.data, bytes)
        body = record(cast(Json, json.loads(req.data)))
        text = str(record(cast(list[Json], body['messages'])[0])['content'])
        if url.endswith('/tokenize'):
            tokenized.append(len(text))
            return FakeResponse(json.dumps({'count': (len(text) + (500 if drift else 0)) // 4}).encode())
        n = int(cast(int, body['max_tokens']))
        usage = {'prompt_tokens': len(text) // 4, 'completion_tokens': n, 'prompt_tokens_details': {'cached_tokens': 0}}
        assert body.get('ignore_eos', False) is (n == 64)
        return FakeResponse(b'', [b'data: ' + json.dumps({'id': 'chatcmpl-x', 'choices': [{'delta': {'content': 'PRIVATE_TEXT'},
                                  'finish_reason': 'length'}], 'usage': usage}).encode(), b'data: [DONE]'])
    monkeypatch.setattr(urllib.request, 'urlopen', open_fake)
    monkeypatch.setattr(sys, 'argv', ['wr', '--execute', '--cap', '327680', '--contexts', '150000', '--reps', '1', '--modes', 'cold',
                                      '--trace-run', 'r', '--cache-salt-prefix', 'tag', '--deadline-utc', str(time.time() + 60),
                                      '--output', str(output), '--data-dir', str(tmp_path), *extra])
    wr.main()
    return [record(cast(Json, json.loads(x))) for x in output.read_text().splitlines()], tokenized


def test_driver_warmup_reuse_cuts_and_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = tmp_path / 'ledger.jsonl'
    first, calls = driver(tmp_path, monkeypatch, ['--warmup', '--ledger', str(ledger)], tmp_path / 'a.jsonl')
    assert [r.get('kind') for r in first] == [None, 'warmup', 'cold']
    assert record(first[1]['usage'])['completion_tokens'] == 64 and first[2]['target_ctx'] == 150000
    assert 'PRIVATE' not in (tmp_path / 'a.jsonl').read_text() and len(calls) > 10
    events = [json.loads(x)['event'] for x in ledger.read_text().splitlines()]
    assert events == ['send_intent', 'http_opened', 'completed'] * 2
    second, recount = driver(tmp_path, monkeypatch, ['--warmup', '--reuse-cuts', str(tmp_path / 'a.jsonl')], tmp_path / 'b.jsonl')
    assert len(recount) == 2 and [r['prompt_sha256'] for r in second[1:]] == [r['prompt_sha256'] for r in first[1:]]
    with pytest.raises(ValueError, match='no longer reproduces'):
        driver(tmp_path, monkeypatch, ['--reuse-cuts', str(tmp_path / 'a.jsonl')], tmp_path / 'c.jsonl', drift=True)


def test_remote_scripts_run_on_cpu_with_fake_tools(tmp_path: Path) -> None:
    """The exact Spark-side scripts, with fake docker/nvidia-smi on PATH: numbers only, streaming, fail visibly."""
    from offload.launch.windows import up2_checks as ck
    tools = tmp_path / 'bin'
    tools.mkdir()
    log = tmp_path / 'docker.log'
    log.write_text('GPU KV cache size: 2,820,756 tokens\nthrough proxy process pid 595\nasync mode on\n'
                   "remote DSpark worker 'e-dspark-3090x2' ready\nremote Engram rows attached\nPRIVATE-free line\n")
    (tools / 'docker').write_text(f'#!/bin/sh\ncat {log}\n')
    (tools / 'nvidia-smi').write_text('#!/bin/sh\necho "GPU-x, 2100, 4000, 50.1, 100.0, 60, 90, 0x0"\n')
    for tool in tools.iterdir():
        tool.chmod(0o755)
    env = {**os.environ, 'PATH': f'{tools}:{os.environ["PATH"]}'}
    def run(script: str, *args: str) -> list[dict[str, Json]]:
        done = subprocess.run([sys.executable, '-c', script, *args], env=env, text=True, capture_output=True, check=True)
        return [record(cast(Json, json.loads(x))) for x in done.stdout.splitlines()]
    summary = run(w.LOG_FILTER, ck.STOP_LOG.pattern)[0]
    assert w.logs(summary, 0, 524288)['lines'] == 6
    rows = run(w.TELEMETRY, FIELDS, '2')
    assert [r['kind'] for r in rows] == ['memory', 'gpu'] * 2
    w.gpu_row(rows[1])
    assert w.memory_row(rows[0])[0] > 0
    (tools / 'nvidia-smi').write_text('#!/bin/sh\nexit 6\n')
    assert [r['kind'] for r in run(w.TELEMETRY, FIELDS, '1')] == ['memory', 'gpu_error']
    with pytest.raises(ValueError, match='shape'):
        w.gpu_row({'csv': 'GPU-x, 2100'})
    sample = tmp_path / 'base.py'
    sample.write_bytes(b'print(1)\n' * 1000)
    assert run(w.IMAGE_BASES, str(sample))[0] == {str(sample): hashlib.sha256(sample.read_bytes()).hexdigest()}


# ---------------------------------------------------------------- S40 fixes (abort during drain, start race, signals, idle)

@pytest.mark.parametrize('abort', ['stop-file', 'interrupt'])
def test_abort_during_drain_finds_the_late_hold_and_closes_it(harness: tuple[w.Live, FakeRemote, Clock, w.Supervisor],
                                                              abort: str) -> None:
    """S40 FIX 1: the opener wrote the HOLD after the runner gave up; the unsupervised re-read finds it."""
    live, fake, _clock, _sup = harness
    original = fake.remote
    def drain(label: str, host: str, command: str, stdin: str | None) -> str:
        if 'window.sh open' in command:
            original(label, host, command, stdin)          # the remote opener writes the HOLD...
            if abort == 'stop-file':
                (live.out / 'STOP').write_text('coordinator\n')   # ...and the supervised hold-read is now refused
                return 'window open'
            raise KeyboardInterrupt
        return original(label, host, command, stdin)
    fake.remote = drain  # type: ignore[method-assign]
    result = go(live)
    # With a STOP file the supervised read in open_hold is refused, so the unsupervised re-read decides; on Ctrl-C the
    # supervised read in open_hold's finally still ran and already found the HOLD.
    expected = ['reap', 'jit-wait-sparkhead', 'jit-wait-sparkworker-wired'] + (['hold-reread'] if abort == 'stop-file' else [])
    assert [x[0] for x in fake.unsupervised] == expected
    assert fake.restores == ['check', 'close'] and str(record(result['restore'])['status']).startswith('RESTORED')


@pytest.mark.parametrize('harness', [{'fail': 'hold-unreadable'}], indirect=True)
def test_unreadable_hold_after_abort_goes_to_check_and_close(harness: tuple[w.Live, FakeRemote, Clock, w.Supervisor]) -> None:
    live, fake, _clock, _sup = harness
    original = fake.remote
    def drain(label: str, host: str, command: str, stdin: str | None) -> str:
        if 'window.sh open' in command:
            raise KeyboardInterrupt
        return original(label, host, command, stdin)
    fake.remote = drain  # type: ignore[method-assign]
    result = go(live)
    assert fake.restores == ['check', 'close'] and 'NOT NEEDED' not in str(record(result['restore'])['status'])


def test_every_restore_reaps_owned_sessions_and_waits_for_jit_first(harness: tuple[w.Live, FakeRemote, Clock, w.Supervisor]) -> None:
    """S40 FIX 2: an abandoned start_exp.sh is ended (by its PF1 config, whole session) before the W8 start."""
    live, fake, _clock, _sup = harness
    original = fake.remote
    def fail_start(label: str, host: str, command: str, stdin: str | None) -> str:
        if 'start_exp.sh' in command and 'PF1_CAP=524288' in command:
            raise KeyboardInterrupt
        return original(label, host, command, stdin)
    fake.remote = fail_start  # type: ignore[method-assign]
    result = go(live)
    reap = next(x for x in fake.unsupervised if x[0] == 'reap')
    assert reap[1] == 'sparkhead' and "start_exp[.]sh offload/launch/configs/PF1-window[.]env" in reap[2]
    assert 'docker kill' not in w.JIT_WAIT and 'docker rm' not in w.JIT_WAIT and 'pkill' not in w.JIT_WAIT
    assert fake.restores == ['start', 'check', 'close'] and str(record(result['restore'])['status']).startswith('RESTORED')


def test_reap_patterns_match_owned_command_lines_and_never_themselves() -> None:
    import re as regex
    owned = ['timeout 1200 offload/launch/window.sh open PF1-WINDOW tag: matched 327680/524288 prefill attribution',
             'timeout 900 bash offload/launch/start_exp.sh offload/launch/configs/PF1-window.env']
    patterns = ['window[.]sh open PF1-WINDOW tag:', 'start_exp[.]sh offload/launch/configs/PF1-window[.]env']
    script = w.REAP.replace('@TAG@', 'tag')
    for line, pattern in zip(owned, patterns):
        assert regex.search(pattern, line) and pattern in script
        assert not regex.search(pattern, 'bash -c ' + script)          # the reaper's own command line
    production = ['bash offload/launch/start.sh', 'offload/launch/window.sh close', 'bash offload/launch/start_exp.sh offload/launch/configs/D-kv6-async-pack.env']
    assert not any(regex.search(p, x) for p in patterns for x in production)


def test_signals_are_ignored_before_the_backend_restore_begins() -> None:
    """S40 FIX 3: no gap between the supervisor close and the restore's own signal handling."""
    seen: list[object] = []
    class Backend(FakeBackend):
        def restore(self, *, production_stopped: bool) -> dict[str, Json]:
            seen.append(signal.getsignal(signal.SIGINT))
            seen.append(signal.getsignal(signal.SIGTERM))
            return super().restore(production_stopped=production_stopped)
    before = signal.getsignal(signal.SIGINT)
    w.window(Backend(), 'all', lambda row: None)
    assert seen == [signal.SIG_IGN, signal.SIG_IGN] and signal.getsignal(signal.SIGINT) == before


def test_busy_production_waits_then_stops_before_any_stop(harness: tuple[w.Live, FakeRemote, Clock, w.Supervisor],
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """S40 FIX 4: production must be idle before the first start_exp.sh stops it; else STOP with production serving."""
    live, fake, _clock, _sup = harness
    monkeypatch.setattr(w, 'IDLE_WAIT_S', 0)
    fake.busy_polls = 10 ** 6
    result = go(live)
    assert 'production not idle' in str(result['stop'])
    assert not any(c.endswith('-start') for c in fake.commands) and fake.restores == ['check', 'close']


def test_production_idle_after_one_busy_poll_proceeds(harness: tuple[w.Live, FakeRemote, Clock, w.Supervisor],
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    live, fake, _clock, _sup = harness
    monkeypatch.setattr(time, 'sleep', lambda s: None)
    fake.busy_polls = 1
    result = go(live)
    assert 'stop' not in result and fake.commands.count('prod-idle') == 2


def test_trace_check_tolerates_tiny_engine_requests_but_not_large_unsent_ones(tmp_path: Path) -> None:
    """Window 1 (MEASURED): the trace also held engine startup requests of 2/7/7/7/7/15 tokens on both ranks."""
    arm = w.ON_ARMS[0]
    rows = request_rows(arm, 'r', (150000,), 1)
    def with_extra(name: str, sizes: tuple[int, ...]) -> dict[int, list[Path]]:
        files: dict[int, list[Path]] = {}
        for rank in (0, 1):
            write_traces(tmp_path / f'{name}{rank}', rank, rows)
            path = tmp_path / f'{name}{rank}/prefill-r{rank}-p77.jsonl'
            lines = path.read_text()
            for i, n in enumerate(sizes):
                member = {'request': f'engine{i}', 'chunk': f'engine{i}:0:{n}:0', 'new_tokens': n, 'prefix_tokens': 0}
                base = {'rank': rank, 'run': 'r', 'max_model_len': arm.cap, 'members': [member]}
                lines += json.dumps({**base, 'phase': 'chunk.begin'}) + '\n'
                lines += json.dumps({**base, 'phase': 'chunk.total', 'cuda_ms': 1.0, 'wall_ms': 1.0}) + '\n'
            path.write_text(lines)
            files[rank] = [path]
        return files
    result = w.trace_check(with_extra('tiny', (2, 7, 7, 7, 7, 15)), rows)
    assert cast(dict[int, dict[str, int]], result['ranks'])[0]['unsent_small_requests'] == 6
    with pytest.raises(ValueError, match='unsent request of 4096 tokens'):
        w.trace_check(with_extra('large', (4096,)), rows)
    with pytest.raises(ValueError, match='lacks the sent 150000-token request'):
        w.trace_check(with_extra('only', (2,)), [*rows, {**rows[-1], 'target_ctx': 150000, 'trace_request': 'missing'}])


def test_prior_window_from_an_ancestor_commit_needs_runner_only_changes() -> None:
    """S40 round 3: window 1 ran 56d3e91; an ON-only rerun from a runner-only descendant may use its OFF gate."""
    window1, trace_fix = '56d3e91025b2228e798cf3af4db653d6887599c1', '0811344'
    full = subprocess.run(['git', 'rev-parse', trace_fix], cwd=w.ROOT, capture_output=True, text=True, check=True).stdout.strip()
    assert w.same_sources(window1, window1)
    assert w.same_sources(window1, full)                       # diff touches only pf1_window.py and its test
    assert not w.same_sources(full, window1)                   # never a descendant's gate for an ancestor run
    base = '5a47cfe'
    base_full = subprocess.run(['git', 'rev-parse', base], cwd=w.ROOT, capture_output=True, text=True, check=True).stdout.strip()
    assert not w.same_sources(base_full, window1)              # the base -> window-1 diff changes launch/trace sources


@pytest.mark.parametrize('t0', [False, True])
def test_cause_sequence_and_live_identity(harness: tuple[w.Live, FakeRemote, Clock, w.Supervisor], t0: bool) -> None:
    live, fake, _clock, _sup = harness
    live.cause = True
    result = w.window(live, 'cause', lambda row: None, cause_t0=t0, prior_gate={'status': 'REPRODUCED'})
    assert 'stop' not in result, result.get('stop')
    names = ['cause-327', 'cause-524'] + (['cause-524-t0'] if t0 else [])
    assert list(record(result['arms'])) == names
    assert [c for c in fake.commands if c.endswith('-start')] == [n + '-start' for n in names]
    assert fake.restores == ['start', 'check', 'close']
    assert 'cause-copy' in fake.commands and 'trace-copy' in fake.commands
    for name in names:
        assert set(record(record(record(record(result['arms'])[name])['checks'])['mem_available_min_kb'])) == set(w.HOSTS)
    assert set(json.loads((live.out / 'arm-memory.json').read_text())) == set(names)
    assert 'PF1-cause[.]env' in fake.unsupervised[0][2]


def test_cause_verifier_has_exact_18_pins() -> None:
    script = verifier(True, cause=True)
    pins = ast.literal_eval(script.split('pins = ', 1)[1].splitlines()[0])
    assert len(pins) == 18
    for path, sha in w.overlay_pins(True).items():
        assert pins[w.DIST + path] == sha
    assert verifier(True) != script


def test_cause_lineage_accepts_actual_window_one_but_preserves_on_rule() -> None:
    baseline = '56d3e91025b2228e798cf3af4db653d6887599c1'
    current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=w.ROOT, text=True).strip()
    assert w.cause_sources(baseline, current)
    assert not w.same_sources(baseline, current)
    assert not w.cause_sources(current, baseline)


# ---------------------------------------------------------------- cause-specific STOP rehearsal (all external commands fake)

@pytest.mark.parametrize('fail', ['open_hold', 'snapshot', 'stage_trace', *[
    f'{step}:{arm.name}' for arm in (*w.CAUSE_ARMS, w.CAUSE_T0) for step in ('start', 'requests', 'collect')]])
def test_every_cause_backend_stop_ends_in_restore(fail: str) -> None:
    b = FakeBackend(fail=fail)
    result = w.window(b, 'cause', lambda row: None, cause_t0=True, prior_gate={'status': 'REPRODUCED'})
    assert 'injected' in str(result['stop'])
    assert b.calls[-1] == f'restore:{fail not in ("open_hold", "snapshot", "stage_trace")}'
    assert b.calls[-2] == fail


@pytest.mark.parametrize('harness', [{'monitor': 'low'}, {'monitor': 'swap'}, {'monitor': 'gpu'}], indirect=True)
def test_t0_guard_trip_uses_w8_restore_and_keeps_minima(harness: tuple[w.Live, FakeRemote, Clock, w.Supervisor]) -> None:
    live, fake, _clock, _sup = harness
    live.cause = True
    trip = fake.trip
    fake.trip = None
    original = fake.requests
    def t0_trip(argv: list[str], inflight: int) -> str:
        if fake.arm == w.CAUSE_T0:
            fake.trip = trip
        return original(argv, inflight)
    fake.requests = t0_trip  # type: ignore[method-assign]
    result = w.window(live, 'cause', lambda row: None, cause_t0=True, prior_gate={'status': 'REPRODUCED'})
    assert any(s in str(result['stop']) for s in ('below 4 GiB', 'new swap use', 'GPU telemetry failing')), result.get('stop')
    assert list(record(result['arms'])) == ['cause-327', 'cause-524']
    assert fake.restores == ['start', 'check', 'close']
    memory = json.loads((live.out / 'arm-memory.json').read_text())
    assert 'cause-524-t0' in memory
    if 'below 4 GiB' in str(result['stop']):
        assert min(memory['cause-524-t0'].values()) == 3000000


@pytest.mark.parametrize('fault,message', [('pin', 'pins not verified'), ('mount', 'mounted engine'),
    ('env', 'wrong experiment setting'), ('start-floor', 'below 5 GiB')])
def test_t0_pin_identity_and_start_floor_mismatch_restore(harness: tuple[w.Live, FakeRemote, Clock, w.Supervisor],
                                                           fault: str, message: str) -> None:
    live, fake, _clock, _sup = harness
    live.cause = True
    original_remote = fake.remote
    original_identity = fake.identity
    def identity(rank: int) -> dict[str, Json]:
        found = original_identity(rank)
        if fake.arm == w.CAUSE_T0 and rank == 1:
            if fault == 'mount':
                record(found['mounted'])[w.DIST + 'vllm_exl3_moe/tf_prompt.py'] = '0' * 64
            elif fault == 'env':
                record(found['env'])['VLLM_EXL3_TF_PROMPT_ALIAS'] = '1'
        return found
    def remote(label: str, host: str, command: str, stdin: str | None) -> str:
        if label.startswith('cause-524-t0-'):
            if fault == 'pin' and label.endswith('-verify-sparkworker-wired'):
                return 'PF1 verified 17 engine sources'
            if fault == 'start-floor' and command == w.HOST_PARITY:
                return 'MemAvailable: 5000000 kB\n'
        return original_remote(label, host, command, stdin)
    fake.identity = identity  # type: ignore[method-assign]
    fake.remote = remote  # type: ignore[method-assign]
    result = w.window(live, 'cause', lambda row: None, cause_t0=True, prior_gate={'status': 'REPRODUCED'})
    assert message in str(result['stop']), result.get('stop')
    assert fake.restores == ['start', 'check', 'close']
    assert 'cause-524-t0' not in record(result['arms'])
    if fault == 'start-floor':
        assert 5000000 in json.loads((live.out / 'arm-memory.json').read_text())['cause-524-t0'].values()


@pytest.mark.parametrize('fault', ['stop-file', 'interrupt', 'deadline', 'foreign-demand'])
def test_cause_t0_abort_paths_restore(harness: tuple[w.Live, FakeRemote, Clock, w.Supervisor], fault: str) -> None:
    live, fake, clock, _sup = harness
    live.cause = True
    original = fake.requests
    def abort(argv: list[str], inflight: int) -> str:
        if fake.arm == w.CAUSE_T0:
            if fault == 'stop-file':
                (live.out / 'STOP').write_text('CPU fake\n')
            elif fault == 'interrupt':
                raise KeyboardInterrupt
            elif fault == 'deadline':
                clock.now += 81 * 60
            else:
                fake.demand = 2
                live.sup.demand_due = 0
        return original(argv, inflight)
    fake.requests = abort  # type: ignore[method-assign]
    result = w.window(live, 'cause', lambda row: None, cause_t0=True, prior_gate={'status': 'REPRODUCED'})
    assert 'stop' in result and 'cause-524-t0' not in record(result['arms'])
    assert fake.restores == ['start', 'check', 'close']
    assert 'PF1-cause[.]env' in fake.unsupervised[0][2]


@pytest.mark.parametrize('state', ['REUSED', 'DIFFERENT'])
def test_cause_staging_never_overwrites_existing_different_sources(harness: tuple[w.Live, FakeRemote, Clock, w.Supervisor],
                                                                    state: str) -> None:
    live, fake, _clock, _sup = harness
    live.cause = True
    original = fake.remote
    def remote(label: str, host: str, command: str, stdin: str | None) -> str:
        if label == 'cause-remote':
            return state
        return original(label, host, command, stdin)
    fake.remote = remote  # type: ignore[method-assign]
    result = w.window(live, 'cause', lambda row: None, prior_gate={'status': 'REPRODUCED'})
    assert 'cause-copy' not in fake.commands
    if state == 'DIFFERENT':
        assert 'never overwrite' in str(result['stop']) and fake.restores == ['check', 'close']
        assert not any(c.endswith('-start') for c in fake.commands)
    else:
        assert 'stop' not in result and fake.restores == ['start', 'check', 'close']


@pytest.mark.parametrize('changed', [w.CONFIG, 'offload/prefill_trace/runtime.py', 'offload/prefill_trace/build_overlay.py',
    'offload/kvpack/MANIFEST.json', 'offload/launch/serve_exp.sh', 'offload/dspark/d2/capture_window_identity.sh'])
def test_cause_lineage_rejects_execution_source_drift(changed: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0, stdout=changed + '\n' if argv[1] == 'diff' else '')
    monkeypatch.setattr(subprocess, 'run', run)
    assert not w.cause_sources('a' * 40, 'b' * 40)


def test_actual_window_one_prior_gate_validates_locally() -> None:
    current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=w.ROOT, text=True).strip()
    gate = w.prior(w.CAMPAIGN / 'PF1-W1-20261025T234254Z', current, cause=True)
    assert gate['status'] == 'REPRODUCED'
