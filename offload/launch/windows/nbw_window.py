"""NBW attended K/I window. Dry by default. Coordinator only; always close pair and strictly verify W8 restore."""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from offload.launch.windows import scoped_remote as owner
from offload.launch.windows import spark_quiet, hold_admission
from offload.launch.windows import nbw_cells as cells
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import post_close_watch as watchdog
from offload.launch.windows import up2_checks as ck
from offload.launch.windows.up2_low import X86HOST_PROBE
from offload.prefill_maxlen.evidence import Json, number, record

CONFIG = 'offload/launch/configs/NBW-native.env'
PROD_COMMIT = '596ba1cf4e3b11be29b41a57e7f69a5d8efa4799'
PROD_TAG = 'prod-k1-327-ops1-20261026'
SUBMIT_S, ABORT_S, RESTORE_S = 72 * 60, 75 * 60, 90 * 60
K_BUDGET_S, I_BUDGET_S = 12 * 60, 18 * 60
KERNEL, DRIVER = '7.0.0-1019-nvidia', '580.178.04'
IDENTITY_KEYS = ('VLLM_DSPARK_REMOTE_IDENTITY', 'VLLM_DSPARK_REMOTE_MARKOV_IDENTITY')
# Round 2: private copy of the pinned E/production TP2 cache, explicitly reviewed before staging.
# Original W8 cache bytes/hash were not recovered; NBW-WINDOW.md documents the alternative's provenance/limits.
TUNE_CACHE = 'dsv41-offload/nbw-source/exl3_tune_cache-reviewed.bin'
TUNE_SHA256 = '937e9633334be0a5a7a97557e83f962bfbc33dce84a76b93e331b8017b0d405e'
PIN_FILES = (CONFIG, 'offload/launch/start_exp.sh', 'offload/launch/serve_exp.sh',
             'offload/launch/dsv41_tp2_offload.sh', 'offload/kvpack/MANIFEST.json',
             'offload/launch/windows/nbw_window.py', 'offload/launch/windows/nbw_cells.py',
             'offload/launch/windows/scoped_remote.py', 'offload/launch/windows/post_close_watch.py',
             'offload/launch/windows/spark_quiet.py', 'offload/launch/windows/production_health.py',
             'offload/launch/windows/hold_admission.py')


@dataclass(frozen=True)
class Arm(pf.Arm):
    role: str = 'K'

    @property
    def config(self) -> str:
        return CONFIG


ARMS = tuple(Arm(f's{i}-{role}', 327680, 0, role=role) for i, role in enumerate(('K', 'I', 'K', 'I', 'K'), 1))


@dataclass(frozen=True)
class Native:
    tree: str
    commit: str
    identity: str
    markov: str

    def validate(self) -> None:
        ck.require(re.fullmatch(r'dsv41-offload/[A-Za-z0-9_/-]+/repo', self.tree) is not None
                   and '..' not in self.tree.split('/'), 'durable separate native tree relative to x86host HOME required')
        ck.require(re.fullmatch('[0-9a-f]{40}', self.commit) is not None, 'full reviewed NB1 source commit required')
        ck.require(all(re.fullmatch('[0-9a-f]{64}', v) is not None for v in (self.identity, self.markov)),
                   'reviewed native drafter and Markov digests required')
        ck.require(not self.identity.startswith('5c3be5c7'), 'W8 native identity is stale for NB1')


def host_check(text: str) -> None:
    ck.require(text.splitlines()[:2] == [KERNEL, DRIVER], 'Spark kernel/driver differs from current production brief')
    available = re.findall(r'^MemAvailable:\s+(\d+) kB$', text, re.M)
    ck.require(len(available) == 1 and int(available[0]) >= pf.START_FLOOR_KB, 'MemAvailable below 5 GiB before start')


def runtime(prod: dict[str, Json], live: dict[str, Json], arm: Arm, native: Native) -> None:
    """Use PF1's full comparison after proving/normalising exactly the NBW env differences."""
    before, after = record(prod['env']), record(live['env'])
    expected = dict(before)
    if arm.role == 'I':
        expected.update(dict(zip(IDENTITY_KEYS, (native.identity, native.markov))))
    ck.require(after == expected, 'NBW runtime env drift beyond worker identity')
    # PF1's comparator encodes REQUIRED=0->1. Normalise its inputs ONLY after checking actual NBW equality above.
    a = {**prod, 'env': {**before, 'VLLM_ENGRAM_REMOTE_REQUIRED': '0'}}
    b = {**live, 'env': {**before, 'VLLM_ENGRAM_REMOTE_REQUIRED': '1'}}
    pf.runtime(a, b, arm, arm.name)


def decision(starts: dict[str, Json]) -> dict[str, Json]:
    """Brief's fixed absolute K-to-K spread rule, with complete matched evidence required."""
    groups: dict[str, list[dict[str, Json]]] = {'K': [], 'I': []}
    for value in starts.values():
        row = record(value)
        if row.get('status') == 'SKIPPED_TIME':
            continue
        if row.get('status') != 'COMPLETE' or row.get('role') not in groups:
            return {'status': 'NOT A CANDIDATE', 'reason': 'incomplete start'}
        groups[str(row['role'])].append(row)
    if any(len(g) < 2 for g in groups.values()):
        return {'status': 'NOT A CANDIDATE', 'reason': 'two complete K and I starts required'}
    points: dict[str, Json] = {}
    passes = True
    for name in cells.CELLS:
        rows = {role: [record(record(r['cells'])[name]) for r in blocks] for role, blocks in groups.items()}
        hashes = [[record(s)['prompt_sha256'] for s in cast(list[Json], row['streams'])]
                  for blocks in rows.values() for row in blocks]
        ck.require(all(h == hashes[0] for h in hashes), 'K/I public prompts differ')
        rates = {role: [number(r['median_decode_tps']) for r in blocks] for role, blocks in rows.items()}
        k, i = statistics.median(rates['K']), statistics.median(rates['I'])
        spread = max(rates['K']) - min(rates['K'])
        passed = i - k > spread if name in ('short-b1', '32k-b1') else k - i <= spread
        points[name] = {'K_median_tps': k, 'I_median_tps': i, 'K_spread_tps': spread, 'pass': passed}
        passes &= passed
    for row in groups['I']:
        checks = record(row['greedy'])
        ck.require(number(checks.get('identical', -1)) + number(checks.get('near_tie', 0)) == 6,
                   'six identical-or-near-tie native greedy checks required')
    return {'status': 'CANDIDATE' if passes else 'NOT A CANDIDATE', 'cells': points,
            'promotion': 'written package only; separate attended approval and fidelity review required'}


# Pure command builders. Commands are never run by a dry plan or by this lane.
def native_identity_script() -> str:
    return ('import json; from offload.dspark.worker.identity import compute as checkpoint; '
            'from offload.dspark.native.identity import compute; '
            'print(json.dumps(compute("native-hybrid-kx2", checkpoint("/opt/dsv41/x86host-data/dspark", '
            'check_extracted=False))))')


# Same frozen extension/cache lookup as N6, without N6's probe-only filesystem audit (the server writes its status).
# Missing compiled artifacts refuse; no compiler or fallback kernel is allowed in this window.
NATIVE_BOOTSTRAP = r'''
import importlib.util, os, runpy
from pathlib import Path
import torch.utils.cpp_extension as cpp
from triton import knobs
from triton.runtime.cache import FileCacheManager
runtime = Path.home() / 'dsv41-offload/n4'
def extension(name, **kwargs):
    path = runtime / 'build' / name / (name + '.so')
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError('missing N4 prebuilt extension: ' + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
cpp.load = extension
class ReadOnlyCache(FileCacheManager):
    def __init__(self, key, override=False, dump=False):
        if override or dump: raise RuntimeError('cache override/dump forbidden in NBW')
        self.key, self.lock_path = key, None
        roots = [runtime/'triton', runtime.parent/'n2/triton', runtime.parent/'n3/triton',
                 runtime.parent/'s13/triton', Path.home()/'.triton/cache']
        self.cache_dir = str(next((p/key for p in roots if (p/key).is_dir()), roots[0]/key))
    def put(self, data, filename, binary=True):
        raise RuntimeError('NBW read-only Triton cache miss: ' + self.key + '/' + filename)
    def put_group(self, filename, group):
        raise RuntimeError('NBW read-only Triton group miss: ' + filename)
knobs.cache.manager_class = ReadOnlyCache
runpy.run_module('offload.dspark.worker.server', run_name='__main__')
'''


def native_script(native: Native, tag: str) -> str:
    tree = shlex.quote(native.tree)
    directory = f'dsv41-offload-run/nbw/{tag}'
    return ('set -euo pipefail\n'
            f'printf \'%s  %s\\n\' {TUNE_SHA256} "$HOME/{TUNE_CACHE}" | sha256sum -c -\n'
            # owner.launch registers this group BEFORE pair-open can stop the production worker (S49).
            'timeout --signal=TERM --kill-after=15s --foreground 120 bash ~/dsv41-offload/prod_window.sh pair COORD\n'
            'test "$(head -1 ~/dsv41-offload/PAIR_OK)" = COORD\n'
            f'cd ~/{tree}\n'
            f'test "$(git rev-parse HEAD)" = {native.commit}\n'
            'test -z "$(git status --porcelain --untracked-files=no)"\n'
            'export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 '
            'OFL_PY_DIR=$HOME/dsv41-offload/n4/lib OFL_LIB=$HOME/dsv41-offload/n4/lib/libofl.so '
            'DSPARK_P2P_PREBUILT=$HOME/dsv41-offload/n4/p2p '
            f'CUDA_VISIBLE_DEVICES={pf.WORKER_GPUS} DSPARK_WORKER_ARM=native-hybrid-kx2 '
            f'DSPARK_SOURCE_COMMIT={native.commit} DSPARK_NATIVE_BATCH_INVARIANT=2 '
            f'EXLLAMAV3_TUNE_CACHE=$HOME/{TUNE_CACHE}\n'
            'unset DSPARK_V1_DUMP_DIR DSPARK_WORKER_TP_EMULATE DSPARK_D2_ADAPTER\n'
            'exec sh "$HOME/dsv41-offload/prod/dspark-worker/current/offload/dspark/worker/run/gpu_locked.sh" '
            '"$HOME/dsv41-offload/locks/gpu-c4e7a371.lock" "$HOME/dsv41-offload/locks/gpu-7744b44b.lock" -- '
            f'"$HOME/dsv41-offload/e/venv/bin/python" -u -c {shlex.quote(NATIVE_BOOTSTRAP)} '
            '--tp 2 --root /opt/dsv41/x86host-data/dspark --max-model-len 1048576 '
            f'--ofl 10.202.28.1:18620 --status-file "$HOME/{directory}/status.json"')


def native_probe_script(native: Native, tag: str) -> str:
    # Fail if the process, source, read-once mode or status freshness changed. No GPU imports.
    return f'''
import json, os
from pathlib import Path
root=Path.home()
status=json.loads((root/'dsv41-offload-run/nbw/{tag}/status.json').read_text())
p=Path('/proc')/str(status['pid'])
env=dict(x.split('=',1) for x in (p/'environ').read_text().split('\\0') if '=' in x)
print(json.dumps({{'status':status,'cwd':os.readlink(p/'cwd'),
 'cmd':(p/'cmdline').read_text().split('\\0'),
 'gpus':env.get('CUDA_VISIBLE_DEVICES'),'level':env.get('DSPARK_NATIVE_BATCH_INVARIANT'),
 'revision':env.get('DSPARK_SOURCE_COMMIT'),'expected_cwd':str(root/{native.tree!r})}}))
'''


def native_worker(probe: dict[str, Json], native: Native, now: float) -> dict[str, Json]:
    status = record(probe['status'])
    ck.require(probe.get('revision') == native.commit and probe.get('cwd') == probe.get('expected_cwd'), 'native source identity changed')
    ck.require(probe.get('gpus') == pf.WORKER_GPUS and probe.get('level') == '2', 'native GPU/level identity differs')
    ck.require(status.get('arm') == 'native-hybrid-kx2' and status.get('drafter_identity') == native.identity
               and status.get('markov_identity') == native.markov, 'native worker identity refusal')
    ck.require(status.get('tp') == 2 and status.get('tp_comm_errors') == 0, 'native TP failure')
    argv = ck.strings(probe['cmd'])
    ck.require(argv[argv.index('--tp') + 1] == '2' and argv[argv.index('--max-model-len') + 1] == '1048576', 'native HELLO bounds differ')
    stamp = datetime.datetime.fromisoformat(str(status['time']).replace('Z', '+00:00')).timestamp()
    ck.require(abs(now - stamp) < 30, 'native worker status stale')
    return status


FAILED_ROUNDS = ('rounds_failed', 'rounds_some_failed')


class Live(pf.Live):
    def __init__(self, out: Path, sup: pf.Supervisor, *, native: Native, remote: str, expected: str,
                 data_dir: Path, tag: str, call: pf.Call | None = None,
                 plain: Callable[[str, list[str], float], tuple[int, str]] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        super().__init__(out, sup, remote=remote, expected=expected, data_dir=data_dir, tag=tag,
                         stage_bundle=False, call=call, plain=plain, clock=clock)
        self.native = native
        self.native_tags: list[str] = []
        self.pair_open = False
        self.restore_by = self.clock() + RESTORE_S
        self.owner_tags: list[str] = []
        self.owner_deadlines: dict[str, float] = {}
        self.recovery_serial = 0
        # Provisional; open_hold() freezes both, once, immediately before the only HOLD submission (S55).
        self.open_deadline = self.clock() + pf.DRAIN_S + 60
        # The HOLD-opener scope also hosts the private placeholder server that keeps other Spark users out between arms,
        # so its guardian lease must outlive the whole window (drain, then HOLD+90), not just the opening drain.
        self.hold_scope_deadline = self.open_deadline + RESTORE_S
        self.open_admitted = False
        self.admission_token = uuid.uuid4().hex
        # Failed-round counters at the first measured cell of each start: the async proxy can record a failed
        # round while the engine warms up (live NBW-W-20261026T221846Z: rounds_failed=1 from round 1, unchanged
        # through every cell). Only growth after this baseline invalidates a measurement.
        self.round_baseline: dict[str, dict[str, float]] = {}

    @property
    def pin_files(self) -> tuple[str, ...]:
        return PIN_FILES

    def hold_purpose(self) -> str:
        return f'NBW-WINDOW {self.tag}: K1-327680 K/I native invariant drafter, worker and identity only'

    def remember(self, tag: str, deadline: float) -> None:
        ck.require(math.isfinite(deadline), 'finite scoped deadline required')
        ck.require(self.owner_deadlines.get(tag, deadline) == deadline, 'immutable phase deadline changed')
        if tag not in self.owner_tags:
            self.owner_tags.append(tag)  # BEFORE submitting any possibly ambiguous command
            self.owner_deadlines[tag] = deadline

    def owned(self, label: str, host: str, tag: str, script: str, *, timeout: float) -> str:
        deadline = self.hold_scope_deadline if tag.endswith('-hold') else self.sup.abort_at
        self.remember(tag, deadline)
        return self.ssh(label, host, owner.command('launch', tag, ['bash', '-c', script], deadline=deadline), timeout=timeout)

    def fence_phase(self, tag: str, *, guarded: bool = False) -> None:
        def invoke(host: str, command: str) -> tuple[int, str]:
            mode = command.rsplit(' ', 2)[-2]
            label = f'owned-{mode}-{tag}-{host}'
            argv = ['bash', '-c', command] if host == 'localhost' else [*pf.SSH, host, command]
            if guarded:
                return 0, self.call(label, argv, timeout=60)
            return self.plain(label, argv, 60)
        owner.fence_all(tag, invoke, (*owner.NODES, 'localhost'))

    def recovery(self, label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        # Unique attempt identities allow a retry without ever reopening a fence.
        self.recovery_serial += 1
        tag = f'{self.tag}-restore-{label}-{self.recovery_serial}'
        deadline = self.restore_by - 90  # reserve final commit/cancel/retention
        self.remember(tag, deadline)
        ck.require(self.clock() < deadline, 'NBW insufficient time for owned recovery; HOLD stays open')
        try:
            return self.plain(label, ['bash', '-c', owner.command('launch', tag, argv, deadline=deadline)], timeout)
        finally:
            self.fence_phase(tag)

    def preflight(self) -> None:
        self.sup.begin()
        ck.require(pf.digest(pf.RESTORE) == pf.RESTORE_SHA256, 'W8 restore changed')
        cells.greedy_inputs()
        from offload.prefill_trace.window_requests import corpus
        corpus(self.data_dir)
        ck.require((pf.CAMPAIGN / 'D2-GEN.PAUSE').exists() or subprocess.run(
            ['tmux', 'has-session', '-t', '=d2-gen'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0,
            'D2 generation active without pause')
        self.ssh('ledger', 'sparkhead', 'cd ~/dsv41-deploy && test ! -e HOLD && '
                 f'test "$(git rev-parse HEAD)" = {PROD_COMMIT} && '
                 f'test "$(git rev-parse HEAD)" = "$(git rev-list -n1 {PROD_TAG})" && '
                 'git diff --exit-code HEAD -- dsv41.env serve.sh start.sh '
                 'offload/launch/warmup.py offload/launch/warmup_serve.sh')
        checkout = (f'cd ~/{shlex.quote(self.remote)} && test "$(git rev-parse HEAD)" = {self.expected} && '
                    'test -z "$(git status --porcelain --untracked-files=no)"')
        self.ssh('checkout', 'sparkhead', checkout)
        remote = self.ssh('pins', 'sparkhead', f'cd ~/{shlex.quote(self.remote)} && sha256sum ' + shlex.join(PIN_FILES))
        ck.require(remote.strip() == '\n'.join(f'{pf.digest(pf.ROOT / p)}  {p}' for p in PIN_FILES), 'staged source pins differ')
        self.ssh('prod-health', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8000/health')
        models = ck.rows(self.json('prod-models', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8000/v1/models')['data'])
        ck.require(len(models) == 1 and models[0].get('max_model_len') == 327680
                   and models[0].get('root') == '/srv/dsv41/legal-v2a', 'production model/cap differs')
        pf.worker(self.json('worker', 'x86host', 'python3 -c ' + shlex.quote(X86HOST_PROBE)), self.clock())
        for host in pf.HOSTS:
            host_check(self.ssh(f'preflight-host-{host}', host, pf.HOST_PARITY))
        self.ssh('native-tree', 'x86host', f'cd ~/{self.native.tree} && '
                 f'test "$(git rev-parse HEAD)" = {self.native.commit} && '
                 f'test "$(cat SOURCE_COMMIT)" = {self.native.commit} && '
                 'test -z "$(git status --porcelain --untracked-files=no)" && '
                 'test -d ~/dsv41-offload/n4/build && test -d ~/dsv41-offload/n4/p2p && '
                 'test -f ~/dsv41-offload/n4/lib/libofl.so && '
                 f'printf \'%s  %s\\n\' {TUNE_SHA256} "$HOME/{TUNE_CACHE}" | sha256sum -c - && '
                 + ' && '.join('true' if arm.role == 'K' else
                     f'test ! -e ~/dsv41-offload-run/nbw/{self.tag}-{arm.name} && '
                     f'test ! -e ~/{owner.CONTROL}/{self.tag}-{arm.name}' for arm in ARMS))
        ident = self.json('native-reviewed-identity', 'x86host', f'cd ~/{self.native.tree} && '
            'PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 DSPARK_NATIVE_BATCH_INVARIANT=2 '
            '~/dsv41-offload/e/venv/bin/python -c ' + shlex.quote(native_identity_script()))
        ck.require(ident.get('drafter_identity') == self.native.identity and ident.get('markov_identity') == self.native.markov,
                   'reviewed native identity differs from source/receipt')
        (self.out / 'native-receipt.json').write_text(json.dumps(ident, allow_nan=False) + '\n')

    def open_hold(self) -> float:
        ck.require(not self.open_admitted, 'NBW HOLD cannot be reopened or renewed')
        self.open_admitted = True
        # Freeze the opening ceiling and the placeholder lease together AFTER preflight: a HOLD published by the ceiling
        # always has restore_by = HOLD + RESTORE_S <= the lease (S55 FIX 1).
        self.open_deadline = self.clock() + pf.DRAIN_S + 60
        self.hold_scope_deadline = self.open_deadline + RESTORE_S
        # Establish NBW bounds as soon as HOLD is read, including if open/tunnel fails.
        try:
            self.owned('hold-open', 'sparkhead', self.tag + '-hold',
                hold_admission.open_script(self.tag + '-hold', self.admission_token,
                    self.hold_purpose(), pf.DRAIN_S), timeout=pf.DRAIN_S + 60)
        finally:
            text = self.ssh('hold-read', 'sparkhead',
                hold_admission.command('check', self.tag + '-hold', self.admission_token))
            self.hold_written = f'NBW-WINDOW {self.tag}:' in text
            if self.hold_written:
                opened = pf.hold_opened(text, self.tag, self.clock())
                self.submit, self.sup.abort_at, self.restore_by = opened + SUBMIT_S, opened + ABORT_S, opened + RESTORE_S
                self.sup.bound = '75-minute NBW abort bound'
                (self.out / 'deadlines.json').write_text(json.dumps({'opened': opened, 'submit': self.submit,
                    'abort': self.sup.abort_at, 'restore_by': self.restore_by}) + '\n')
        ck.require(self.hold_written, 'NBW exact owned HOLD marker required')
        opened = pf.hold_opened(text, self.tag, self.clock())
        ck.require(opened <= self.open_deadline and self.restore_by <= self.hold_scope_deadline,
                   'NBW HOLD published after its opening ceiling; placeholder lease would not cover the window')
        self.sup.tunnel([*pf.SSH, '-N', '-o', 'ExitOnForwardFailure=yes', '-L', '127.0.0.1:18001:127.0.0.1:8001', 'sparkhead'])
        return opened

    def snapshot(self) -> None:
        super().snapshot()
        path = self.out / 'rollback.json'
        data = ck.read(path)
        data.update({'production_commit': PROD_COMMIT, 'tag': PROD_TAG})
        path.write_text(json.dumps(data, indent=1) + '\n')

    def fits(self, arm: Arm) -> bool:
        return self.clock() + (I_BUDGET_S if arm.role == 'I' else K_BUDGET_S) <= self.submit

    def probe_worker(self, label: str, arm: Arm) -> dict[str, Json]:
        if arm.role == 'K':
            return pf.worker(self.json(label, 'x86host', 'python3 -c ' + shlex.quote(X86HOST_PROBE)), self.clock())
        tag = f'{self.tag}-{arm.name}'
        return native_worker(self.json(label, 'x86host', 'python3 -c ' + shlex.quote(native_probe_script(self.native, tag))),
                             self.native, self.clock())

    def pair_close(self, *, guarded: bool) -> None:
        for tag in list(self.owner_tags):
            if guarded and tag == self.tag + '-hold':
                continue  # its private placeholder server protects the gaps between arms
            self.fence_phase(tag, guarded=guarded)
        script = 'PROD_WINDOW_WAIT_S=60 PROD_WINDOW_READY_S=180 bash ~/dsv41-offload/prod_window.sh close'
        self.recovery_serial += 1
        tag = f'{self.tag}-pair-close-{self.recovery_serial}'
        deadline = self.sup.abort_at if guarded else self.restore_by - 90
        self.remember(tag, deadline)
        command = [*pf.SSH, 'x86host', owner.command('launch', tag, ['bash', '-c', script], deadline=deadline)]
        try:
            if guarded:
                text = self.call('pair-close', command, timeout=300)
                rc = 0
            else:
                rc, text = self.plain('pair-close', command, 300)
        finally:
            self.fence_phase(tag, guarded=guarded)
        ck.require(rc == 0 and 'PRODUCTION HOLD OK' in text.splitlines(),
                   'production TP2 worker did not return: PRODUCTION HOLD OK missing')
        self.pair_open = False

    def start(self, base_arm: pf.Arm) -> None:
        ck.require(isinstance(base_arm, Arm), 'NBW arm required')
        arm = cast(Arm, base_arm)
        ck.require(self.fits(arm), f'{arm.name}: full block cannot fit before submission deadline')
        self.sup.unready()
        if arm.role == 'K' and self.pair_open:
            self.pair_close(guarded=True)
        for host in pf.HOSTS:
            host_check(self.ssh(f'{arm.name}-host-{host}', host, pf.HOST_PARITY))
        self.sup.arm_minima = {}
        self.sup.arm_memory[arm.name] = self.sup.arm_minima
        if arm.role == 'I':
            tag = f'{self.tag}-{arm.name}'
            self.native_tags.append(tag)  # record BEFORE any remote mutation/ambiguous SSH reply
            self.pair_open = True
            directory = f'dsv41-offload-run/nbw/{tag}'
            self.remember(tag, self.sup.abort_at)
            script = f'mkdir -p ~/{directory} && exec > ~/{directory}/worker.log 2>&1; ' + native_script(self.native, tag)
            launch = owner.command('launch', tag, ['bash', '-c', script], deadline=self.sup.abort_at)
            self.ssh(f'{arm.name}-native-launch', 'x86host', f'(nohup {launch} > /dev/null 2>&1 < /dev/null &)', timeout=30)
            until = self.clock() + 10 * 60
            while True:
                # A missing status while prepacking is expected; identity failures after JSON appears are fatal.
                text = self.ssh(f'{arm.name}-native-status', 'x86host', f'if [ -f ~/{directory}/status.json ]; then '
                    'python3 -c ' + shlex.quote(native_probe_script(self.native, tag)) + '; else echo WAIT; fi')
                if text.strip() != 'WAIT':
                    native_worker(record(cast(Json, json.loads(text))), self.native, self.clock())
                    break
                ck.require(self.clock() < until, 'native worker failed to start within ten minutes')
                self.sup.poll()
                time.sleep(1)
        env = f'NBW_ROLE={arm.role} NBW_NATIVE_IDENTITY={self.native.identity} NBW_NATIVE_MARKOV_IDENTITY={self.native.markov}'
        start = (f'cd ~/{self.remote} && {env} timeout {pf.START_S} '
                 f'bash offload/launch/start_exp.sh {CONFIG}')
        start = hold_admission.command('stop', self.tag + '-hold', self.admission_token,
                                       ['bash', '-c', start])
        self.owned(f'{arm.name}-start', 'sparkhead', f'{self.tag}-{arm.name}-engine', start,
                   timeout=pf.START_S + 60)
        self.ssh(f'{arm.name}-health', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8001/health')
        models = ck.rows(self.json(f'{arm.name}-models', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8001/v1/models')['data'])
        ck.require(len(models) == 1 and models[0].get('max_model_len') == 327680
                   and models[0].get('root') == '/srv/dsv41/legal-v2a', 'experiment model/cap differs')
        current = [self.capture(host, arm.name) for host in pf.HOSTS]
        for rank, host in enumerate(pf.HOSTS):
            runtime(self.production[rank], current[rank], arm, self.native)
            ck.require(str(current[rank]['id']) not in self.seen, 'experiment reused a container')
            self.verify(host, arm.name, 0)
            logs = pf.logs(self.json(f'{arm.name}-logs-{host}', host, 'python3 -c ' + shlex.quote(pf.LOG_FILTER)
                + ' ' + shlex.quote(ck.STOP_LOG.pattern)), rank, 327680)
            if rank == 0:
                self.proxy_pid = int(number(cast(list[Json], logs['proxy_pids'])[0]))
        self.ids = [str(row['id']) for row in current]
        self.seen.update(self.ids)
        self.probe_worker(f'{arm.name}-worker', arm)
        (self.out / f'{arm.name}-identity.json').write_text(json.dumps({'ranks': cast(Json, current)}, indent=1) + '\n')
        self.sup.mark_ready()

    def driver(self, mode: str, output: Path, *extra: str) -> list[str]:
        return [sys.executable, '-m', 'offload.launch.windows.nbw_cells', '--execute', '--mode', mode,
                '--data-dir', str(self.data_dir), '--output', str(output), '--deadline-utc', str(self.submit), *extra]

    def build(self) -> None:
        self.call('public-cuts', self.driver('build', self.out / 'cuts.json'), timeout=300)

    def greedy(self, arm: Arm) -> dict[str, Json]:
        output = self.out / f'{arm.name}-greedy.json'
        self.call(f'{arm.name}-greedy', self.driver('greedy', output), timeout=300, inflight=1)
        result = ck.read(output)
        checks = [record(r) for r in ck.rows(result['checks'])]
        ck.require([r.get('i') for r in checks] == list(range(6))
                   and all(r.get('identical') is True or r.get('near_tie') is True for r in checks)
                   and number(result.get('identical', -1)) + number(result.get('near_tie', -1)) == 6,
                   'native greedy divergence that is not a near-tie, or missing checks')
        return result

    def cell(self, arm: Arm, name: str) -> dict[str, Json]:
        before = self.stats(f'{arm.name}-{name}-before')
        steps = number(self.probe_worker(f'{arm.name}-{name}-worker-before', arm).get('steps', 0))
        output = self.out / f'{arm.name}-{name}.json'
        self.call(f'{arm.name}-{name}', self.driver('cell', output, '--cell', name, '--cuts', str(self.out / 'cuts.json'),
            '--salt', f'{self.tag}-{arm.name}', '--ledger', str(self.out / 'requests-ledger.jsonl')),
            timeout=cells.CELLS[name].estimate_s * 3 + 60, inflight=cells.CELLS[name].batch)
        result = ck.read(output)
        ck.require(result.get('cell') == name and result.get('batch') == cells.CELLS[name].batch, 'cell artifact identity mismatch')
        after = self.stats(f'{arm.name}-{name}-after')
        ck.counters(after, 0, before)
        self.round_baseline.setdefault(arm.name, {key: number(record(before['counters']).get(key, 0)) for key in FAILED_ROUNDS})
        for key in FAILED_ROUNDS:
            ck.require(number(record(after['counters']).get(key, 0)) == number(record(before['counters']).get(key, 0)),
                       f'{arm.name}: {key} grew during {name}')
        ck.require(number(record(after['counters'])['rounds_ok']) > number(record(before['counters'])['rounds_ok']), 'no drafting during cell')
        ck.require(number(self.probe_worker(f'{arm.name}-{name}-worker-after', arm).get('steps', 0)) > steps, 'worker took no steps')
        result['proxy_before'], result['proxy_after'] = before, after
        output.write_text(json.dumps(result, allow_nan=False) + '\n')
        return result

    def finish(self, arm: Arm) -> dict[str, Json]:
        checks: dict[str, Json] = {}
        stats = self.stats(f'{arm.name}-final-stats')
        ck.counters(stats)
        baseline = self.round_baseline.get(arm.name)
        ck.require(baseline is not None, f'{arm.name}: no measured cell recorded a failed-round baseline')
        for key in FAILED_ROUNDS:
            ck.require(number(record(stats['counters']).get(key, 0)) == cast(dict[str, float], baseline)[key],
                       f'{arm.name}: {key} grew after measurement began')
        checks['startup_failed_rounds'] = cast(Json, dict(cast(dict[str, float], baseline)))
        checks['proxy'] = stats
        for rank, host in enumerate(pf.HOSTS):
            ck.require(self.ssh(f'{arm.name}-final-id-{host}', host, "docker inspect --format '{{.Id}}' dsv41-serve").strip() == self.ids[rank],
                       'experiment container changed during measurement')
            checks[host] = pf.logs(self.json(f'{arm.name}-final-logs-{host}', host, 'python3 -c '
                + shlex.quote(pf.LOG_FILTER) + ' ' + shlex.quote(ck.STOP_LOG.pattern)), rank, 327680)
        minima = self.sup.arm_minima
        ck.require(minima is not None and set(minima) == set(pf.HOSTS), 'missing memory minima')
        checks['memory_min_kb'] = cast(Json, dict(cast(dict[str, int], minima)))
        return checks

    def restore(self, *, production_stopped: bool) -> dict[str, Json]:
        self.sup.close()
        original = self.plain
        # Clamp cleanup, W8 and close transactions to ONE original HOLD deadline.
        def bounded(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
            if label in ('cancel-close', 'retain-close'):
                return original(label, argv, min(30, timeout))
            left = self.restore_by - self.clock()
            ck.require(left > 0, 'NBW HOLD+90 exhausted; manual restore required; HOLD stays open')
            return original(label, argv, min(left, timeout))
        self.plain = bounded
        try:
            # Fence the HOLD writer BEFORE rereading it: a late opener must not
            # create a HOLD after recovery decided that production was untouched.
            for tag in list(self.owner_tags):
                self.fence_phase(tag)
            if not self.hold_written:
                rc, text = self.plain('nbw-hold-reread', [*pf.SSH, 'sparkhead',
                    hold_admission.command('check', self.tag + '-hold', self.admission_token)], 60)
                ck.require(rc == 0, 'NBW HOLD ownership unreadable; HOLD stays open')
                self.hold_written = f'NBW-WINDOW {self.tag}:' in text
                if self.hold_written:
                    opened = pf.hold_opened(text, self.tag, self.clock())
                    self.restore_by = opened + RESTORE_S
            if not self.hold_written:
                ck.require(not production_stopped, 'NBW production may be stopped without owned HOLD')
                return {'status': 'NOT NEEDED: no owned HOLD; all remote mutators fenced and absent'}
            ck.require(math.isfinite(self.restore_by), 'NBW missing immutable restore deadline')
            self.pair_close(guarded=False)
            if production_stopped:
                # dockerd children are outside cgroups: explicitly stop/remove
                # the same named experiment containers and require no GPU apps.
                rc, text = self.recovery('experiment-stop', [*pf.SSH, 'sparkhead',
                    f'cd ~/{self.remote} && ./stop.sh </dev/null'], 120)
                ck.require(rc == 0, 'NBW experiment stop failed; HOLD stays open')
                for host in pf.HOSTS:
                    rc, text = self.plain('jit-wait-' + host, [*pf.SSH, host,
                        spark_quiet.command('NBW EXPERIMENT ABSENT', deadline=min(self.restore_by - 90, self.clock() + 350))], 360)
                    ck.require(rc == 0 and 'NBW EXPERIMENT ABSENT' in text.splitlines(),
                               f'{host}: experiment/container/compute absence unproved; HOLD stays open')
            result: dict[str, Json] = {'pair_close': 'PRODUCTION HOLD OK'}
            steps = ('start', 'check', 'close') if production_stopped else ('check', 'close')
            for step in steps:
                # Close uses an exact suffix to install the HOLD-unlink shim on
                # every receiving host. W8 still performs its original checks.
                if step == 'close':
                    tag = f'{self.tag}-restore-close'
                    deadline = self.restore_by - 90
                    self.remember(tag, deadline)
                    ck.require(self.clock() < deadline, 'NBW final close budget exhausted; HOLD stays open')
                    try:
                        rc, text = self.plain(step, ['bash', '-c', owner.command('launch', tag,
                            ['bash', str(pf.RESTORE), step], deadline=deadline)], 120)
                    finally:
                        self.fence_phase(tag)
                else:
                    rc, text = self.recovery(step, ['bash', str(pf.RESTORE), step], 900 if step == 'start' else 600)
                pf.restored(step, rc, text)  # requires six distinct IDENTICAL, not near ties
                result[step] = 'PASS'
            for prior in list(self.owner_tags):
                self.fence_phase(prior)
            ck.require(self.restore_by - self.clock() >= 90, 'NBW final close needs 90s reserve; HOLD stays open')
            tag = f'{self.tag}-restore-close'
            try:
                try:
                    rc, text = self.plain('commit-close', [*pf.SSH, 'sparkhead', owner.command('commit-close', tag,
                        deadline=min(self.restore_by-5, self.clock()+25))], 30)
                    ck.require(rc == 0 and owner.CLOSED in text.splitlines(), 'NBW final owned HOLD close unconfirmed')
                finally:
                    rc, text = self.plain('cancel-close', [*pf.SSH, 'sparkhead', owner.command('cancel-close', tag)], 30)
                    ck.require(rc == 0 and 'SCOPED CLOSE COMMIT FENCED' in text.splitlines(), 'NBW final close fence unconfirmed')
                result['restore_deadline_met'] = self.clock() < self.restore_by
                ck.require(result['restore_deadline_met'] is True, 'NBW restore completed late')
            except (Exception, KeyboardInterrupt):
                rc, text = self.plain('retain-close', [*pf.SSH, 'sparkhead', owner.command('retain-close', tag)], 30)
                ck.require(rc == 0 and owner.RETAINED in text.splitlines(), 'NBW original HOLD retention unconfirmed')
                raise
            result['status'] = 'RESTORED: K1 START VERIFIED, greedy 6/6 IDENTICAL, PRODUCTION HOLD OK, window closed'
            return result
        finally:
            self.plain = original
            (self.out / 'arm-memory.json').write_text(json.dumps(self.sup.arm_memory) + '\n')

class Backend(Protocol):
    def preflight(self) -> None: ...
    def open_hold(self) -> float: ...
    def snapshot(self) -> None: ...
    def fits(self, arm: Arm) -> bool: ...
    def start(self, arm: Arm) -> None: ...
    def build(self) -> None: ...
    def greedy(self, arm: Arm) -> dict[str, Json]: ...
    def cell(self, arm: Arm, name: str) -> dict[str, Json]: ...
    def finish(self, arm: Arm) -> dict[str, Json]: ...
    def restore(self, *, production_stopped: bool) -> dict[str, Json]: ...


def window(b: Backend, checkpoint: Callable[[dict[str, Json]], None]) -> dict[str, Json]:
    result: dict[str, Json] = {'status': 'INCOMPLETE', 'starts': {}, 'decision': {'status': 'NOT A CANDIDATE', 'reason': 'not measured completely'}}
    starts = record(result['starts'])
    hold = stopped = False
    try:
        b.preflight()
        hold = True
        result['hold_opened'] = b.open_hold()
        b.snapshot()
        for i, arm in enumerate(ARMS):
            if i == 4 and not b.fits(arm):
                starts[arm.name] = {'status': 'SKIPPED_TIME', 'role': arm.role}
                checkpoint({'closing_K': 'SKIPPED_TIME'})
                break
            ck.require(b.fits(arm), f'{arm.name}: mandatory block cannot fit')
            stopped = True  # native pair operation or start_exp may mutate despite a lost reply
            row: dict[str, Json] = {'status': 'STARTED', 'role': arm.role, 'cells': {}}
            starts[arm.name] = row
            checkpoint({'start': arm.name, 'status': 'STARTED'})
            b.start(arm)
            if i == 0:
                b.build()
            if arm.role == 'I':
                row['greedy'] = b.greedy(arm)
            for name in cells.CELLS:
                record(row['cells'])[name] = b.cell(arm, name)
                checkpoint({'start': arm.name, 'cell': name})
            row['checks'] = b.finish(arm)
            row['status'] = 'COMPLETE'
            checkpoint({'start': arm.name, 'status': 'COMPLETE'})
        result['decision'] = decision(starts)
        result['status'] = 'COMPLETE'
    except (Exception, KeyboardInterrupt) as exc:
        result['stop'] = f'{type(exc).__name__}: {exc}'
    finally:
        saved = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
        try:
            if hold:
                try:
                    result['restore'] = b.restore(production_stopped=stopped)
                except (Exception, KeyboardInterrupt) as exc:
                    result['restore'] = {'status': 'FAILED', 'error': f'{type(exc).__name__}: {exc}'}
            else:
                result['restore'] = {'status': 'NOT NEEDED: preflight only'}
        finally:
            for sig, handler in saved.items():
                signal.signal(sig, handler)
    if record(result['restore']).get('status') == 'FAILED':
        result['decision'] = {'status': 'NOT A CANDIDATE', 'reason': 'production restore unverified'}
    return result


def plan() -> str:
    return ('DRY RUN: no command, network or file. Coordinator Deepseek/coordinator only.\n'
            'Current production 596ba1c / prod-k1-327-ops1-20261026; K1 327680, KV 2502600, TP2 K2 fb68c52.\n'
            'Preflight: reviewed clean Spark experiment checkout, current production env/mounts; NB1 durable tree, N4 prebuilt caches,\n'
            '  full native source commit + both solo-reviewed identities. W5 six prompts and K1 references; public corpus only.\n'
            'Drain, exclusive owned HOLD; snapshot both ranks; K I K I, optional closing K if its full 12 min fits before HOLD+72.\n'
            'I: enter cgroup-scoped fenced launcher, then prod_window.sh pair COORD and native worker, invariant=2, exact reviewed identities.\n'
            'Each I: six K1 greedy identical-or-diagnosed-near-tie checks before measurement; other divergence -> STOP.\n'
            'Each start: short B1, 32K B1/B2/B3, one 1500-token run/cell, T=1/top_p=1, seed42/high/priority10, cold public prompts.\n'
            '  TTFT, per-stream/median decode tok/s, acceptance/round, server request-decode ms/spec request-round; retain raw counters.\n'
            'Guards: >=5 GiB before start; >=4 GiB throughout, swap, telemetry, foreign demand, STOP, signals, HOLD+75 abort.\n'
            'ALWAYS: fence all hosts, kill/prove scopes and cgroups absent for every phase; x86host prod_window.sh close -> PRODUCTION HOLD OK;\n'
            '  W8/restore.sh start (if stopped), check (COUNT six distinct IDENTICAL 0..5), close. Failed gate keeps HOLD.\n'
            'ESTIMATED HOLD ~70 min K I K I incl restore, ~82 with closing K; restore bound HOLD+90.\n'
            'After successful close: 15min production watchdog, 30s model/priority20 one-token probes, at most one scoped recovery.\n'
            'CANDIDATE only if B1 gains exceed K spread and B2/B3 losses do not exceed K spread; written promotion only.')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--remote-repo', default='dsv41-deploy-offload')
    ap.add_argument('--expected-experiment-commit', default='')
    ap.add_argument('--native-tree', default='')
    ap.add_argument('--native-commit', default='')
    ap.add_argument('--native-identity', default='')
    ap.add_argument('--native-markov-identity', default='')
    ap.add_argument('--data-dir', type=Path, default=Path('/opt/dsv41/data/bench-data'))
    args = ap.parse_args(argv)
    if not args.execute:
        print(plan())
        return 0
    native = Native(args.native_tree, args.native_commit, args.native_identity, args.native_markov_identity)
    try:
        ck.require(os.environ.get('WINDOW_COORDINATOR') == 'Deepseek/coordinator', 'WINDOW_COORDINATOR=Deepseek/coordinator required')
        native.validate()
        ck.require(args.out is not None and args.out.resolve().parent == pf.CAMPAIGN and not args.out.exists(), 'fresh --out inside campaign required')
        ck.require(re.fullmatch('[0-9a-f]{40}', args.expected_experiment_commit) is not None, 'full reviewed experiment commit required')
        ck.require(re.fullmatch('[A-Za-z0-9_.-]+', args.remote_repo) is not None and args.remote_repo != 'dsv41-deploy', 'separate experiment checkout required')
        ck.require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=pf.ROOT, text=True).strip() == args.expected_experiment_commit,
                   'local HEAD differs from reviewed commit')
        ck.require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=pf.ROOT, text=True).strip(), 'tracked source dirty')
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f'STOP before any remote action: {exc}', file=sys.stderr)
        return 1
    out = cast(Path, args.out)
    out.mkdir()
    tag = re.sub('[^A-Za-z0-9_-]', '-', out.name)
    def interrupted(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(f'signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    def checkpoint(row: dict[str, Json]) -> None:
        with (out / 'checkpoints.jsonl').open('a') as handle:
            handle.write(json.dumps({'time': time.time(), **row}, allow_nan=False) + '\n')
    sup = pf.Supervisor(out)
    live = Live(out, sup, native=native, remote=args.remote_repo, expected=args.expected_experiment_commit,
                     data_dir=args.data_dir, tag=tag)
    result = window(live, checkpoint)
    sup.close()
    if str(record(result['restore']).get('status', '')).startswith('RESTORED'):
        def watch_log(message: str) -> None:
            print(message, flush=True)
            with (out / 'post-close-watch.log').open('a') as handle:
                handle.write(message + '\n')
        watched = watchdog.watch(watchdog.ProductionHealth(),
            watchdog.ScopedRecovery(tag + '-watch', out / 'post-close-commands'), log=watch_log)
        result['post_close_watch'] = cast(Json, dict(watched.__dict__))
        if watched.status != 'PASS':
            result['decision'] = {'status': 'NOT A CANDIDATE', 'reason': 'post-close production health unverified'}
    result.update({'experiment_commit': args.expected_experiment_commit, 'production_commit': PROD_COMMIT,
                   'restore_sha256': pf.RESTORE_SHA256, 'config_sha256': pf.digest(pf.ROOT / CONFIG),
                   'unavailable_traces': 'actual B, raw confidence, chosen depth, exposed draft wait not exported by unchanged production',
                   'artifact_sha256': {str(p.relative_to(out)): pf.digest(p) for p in sorted(out.rglob('*.json*')) if p.is_file()}})
    (out / 'summary.json').write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'stop': result.get('stop'), 'decision': result['decision'], 'restore': result['restore']}))
    if 'post_close_watch' in result and record(result['post_close_watch']).get('status') != 'PASS':
        print('POST-CLOSE PRODUCTION HEALTH FAILED; inspect watchdog log and HOLD before further work.', file=sys.stderr)
        return 2
    if str(record(result['restore']).get('status', '')).startswith(('RESTORED', 'NOT NEEDED')):
        return 0 if 'stop' not in result else 1
    print('PRODUCTION NOT VERIFIED; HOLD remains open. Inspect cmd/restore-*.log; restore pair first, then W8 start/check/close.', file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
