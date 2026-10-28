"""PF1-WINDOW coordinator runner: HOLD, stage, matched OFF arms, gate, traced ON arms, ALWAYS W8 restore.

Dry plan by default (no command, network or file). Only Deepseek / coordinator runs --execute, after review and
staging. Records numbers and hashes only: no prompt or generated text is saved. See
offload/prefill_maxlen/PF1-WINDOW.md (coordinator sheet), PF1-WINDOW-DESIGN.md and PF1-WINDOW-RECONCILE.md.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import selectors
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Protocol, cast

from offload.prefill_maxlen.evidence import Json, number, record
from offload.launch.windows import up2_checks as ck
from offload.launch.windows.up2_low import X86HOST_PROBE, ENGRAM_PROBE, terminate
from offload.prefill_trace.emit_verify import render as verifier
from offload.prefill_cause2.emit_verify import render as fix_verifier
from offload.prefill_trace.telemetry import FIELDS

CAMPAIGN = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD')
ROOT = Path(__file__).resolve().parents[3]
RESTORE = CAMPAIGN / 'W8/restore.sh'
RESTORE_SHA256 = 'd1c4479dbfbd90bc2036c7bf1561db5b564c2b162a7cde4a121e46bfb9f86a49'
CONFIG = 'offload/launch/configs/PF1-window.env'
CAUSE_CONFIG = 'offload/launch/configs/PF1-cause.env'
FIX_CONFIG = 'offload/launch/configs/PF1-fix.env'
PROD_COMMIT, PROD_TAG = '21bed31', 'prod-k1-327-20261023'
IMAGE = 'sha256:435a5236ce3a7db9d45f5a2c483817d60ea629c4ea0a0b3af33da2ef8122334b'
WORKER = 'fb68c5291792ab90c0f468bd94010917967799f3'
WORKER_GPUS = 'GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93,GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc'
KV = {327680: 2502600, 524288: 2820756}
DIST = '/usr/local/lib/python3.12/dist-packages/'
HOSTS = ('sparkhead', 'sparkworker-wired')
URL = 'http://127.0.0.1:18001'
MEM_FLOOR_KB = 4 * 1024**2          # brief: MemAvailable >= 4 GiB on both Sparks, always
START_FLOOR_KB = 5 * 1024**2        # before each start (model load dips about 1 GiB below idle)
SWAP_GROWTH_KB = 256 * 1024         # after readiness; model load itself swaps 3-4 GiB (S18)
SUBMIT_S, ABORT_S = 75 * 60, 80 * 60
ARM_BUDGET_S = 8 * 60               # start + identity + warm-up, ESTIMATED from W13/W6b
DRAIN_S = 20 * 60
START_S = 15 * 60                   # W13/W6b starts took 6.7-6.8 min
OFF_CONTEXTS, OFF_REPS = (150000, 300000), 2
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2']
PIN_FILES = ('offload/launch/configs/PF1-window.env', 'offload/launch/start_exp.sh', 'offload/launch/serve_exp.sh',
             'offload/launch/dsv41_tp2_offload.sh', 'offload/prefill_trace/OVERLAY.sha256',
             'offload/prefill_trace/BASES.json', 'offload/kvpack/MANIFEST.json')
CAUSE_PIN_FILES = (*PIN_FILES, CAUSE_CONFIG, 'offload/prefill_cause/BASES.json', 'offload/prefill_cause/OVERLAY.sha256')
MATCH_PIN_FILES = (*CAUSE_PIN_FILES, FIX_CONFIG, 'offload/prefill_cause2/BASES.json', 'offload/prefill_cause2/OVERLAY.sha256')
HOST_PARITY = ("uname -r && nvidia-smi --query-gpu=driver_version --format=csv,noheader && "
               "cat /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag "
               "/proc/buddyinfo && grep -h -E '^(compact_|MemAvailable:)' /proc/vmstat /proc/meminfo")
TELEMETRY = r'''
import json, subprocess, sys, time
fields, samples, n = sys.argv[1], int(sys.argv[2]), 0
while samples == 0 or n < samples:
    started = time.monotonic()
    mem = {}
    for line in open('/proc/meminfo'):
        key, _, value = line.partition(':')
        if key in ('MemAvailable', 'SwapFree', 'SwapTotal'):
            mem[key] = int(value.split()[0])
    print(json.dumps({'kind': 'memory', 'utc': time.time(), 'kb': mem}), flush=True)
    try:
        csv = subprocess.run(['nvidia-smi', '--query-gpu=' + fields, '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        print(json.dumps({'kind': 'gpu', 'utc': time.time(), 'csv': csv}), flush=True)
    except Exception as exc:
        print(json.dumps({'kind': 'gpu_error', 'utc': time.time(), 'error': type(exc).__name__}), flush=True)
    n += 1
    time.sleep(max(0, 1 - (time.monotonic() - started)))
'''
LOG_FILTER = r'''
import json, re, subprocess, sys
# Streams line by line (constant memory on the Spark); prints counts and capacity numbers only.
stop = re.compile(sys.argv[1], re.I)
kv, pids, hits = [], [], []
n = async_on = ready = engram = stops = 0
proc = subprocess.Popen(['docker', 'logs', 'dsv41-serve'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, errors='replace')
for line in proc.stdout:
    n += 1
    m = re.search(r'GPU KV cache size: ([\d,]+) tokens', line)
    if m: kv.append(int(m.group(1).replace(',', '')))
    m = re.search(r'through proxy process pid (\d+)', line)
    if m: pids.append(int(m.group(1)))
    async_on += 'async mode on' in line
    ready += re.search(r"remote DSpark worker '[^']+' ready", line) is not None
    engram += 'remote Engram' in line or 'remote table' in line or 'Engram remote' in line
    if stop.search(line):
        stops += 1
        if len(hits) < 5: hits.append(line[:160].rstrip())
if proc.wait():
    raise SystemExit('docker logs failed')
print(json.dumps({'lines': n, 'kv_tokens': kv, 'proxy_pids': pids, 'async_on': async_on, 'worker_ready': ready,
                  'engram_remote': engram, 'stop_hits': stops, 'stop_examples': hits}))
'''
IMAGE_BASES = r'''
import hashlib, json, subprocess, sys
out = {}
for path in sys.argv[1:]:
    proc = subprocess.Popen(['dd', 'if=' + path, 'iflag=direct', 'bs=4096', 'status=none'], stdout=subprocess.PIPE)
    try:
        out[path] = hashlib.file_digest(proc.stdout, 'sha256').hexdigest()
    finally:
        proc.stdout.close()
        if proc.wait():
            raise SystemExit('O_DIRECT read failed: ' + path)
print(json.dumps(out, sort_keys=True))
'''


@dataclass(frozen=True)
class Arm:
    name: str
    cap: int
    trace: int
    eager: int = 0
    cause: bool = False
    transient: int = 1
    fix: bool = False

    @property
    def config(self) -> str:
        return CAUSE_CONFIG if self.cause else FIX_CONFIG if self.fix else CONFIG


OFF_ARMS = (Arm('off-327', 327680, 0), Arm('off-524', 524288, 0))
ON_ARMS = (Arm('on-327', 327680, 1), Arm('on-524', 524288, 1))

CAUSE_ARMS = (Arm('cause-327', 327680, 1, cause=True), Arm('cause-524', 524288, 1, cause=True))
CAUSE_T0 = Arm('cause-524-t0', 524288, 1, cause=True, transient=0)
# PF1-CAUSE2 matched window: production code, the CW cause overlay and the fence fix, one HOLD, 524,288 only.
MATCH_ARMS = (Arm('off-524', 524288, 0), Arm('cause-524', 524288, 1, cause=True), Arm('fix-524', 524288, 0, fix=True))
MATCH_REPS = 2


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlay_pins(cause: bool = False, *, fix: bool = False) -> dict[str, str]:
    pins: dict[str, str] = {}
    for family in (('prefill_cause2',) if fix else ('prefill_trace', 'prefill_cause') if cause else ('prefill_trace',)):
        lines = (ROOT / f'offload/{family}/OVERLAY.sha256').read_text().splitlines()
        pins.update({path: sha for sha, path in (line.split() for line in lines)})
    return pins


def image_pins() -> dict[str, str]:
    from offload.prefill_trace.build_overlay import FILES
    bases = cast(dict[str, str], json.loads((ROOT / 'offload/prefill_trace/BASES.json').read_text()))
    return {DIST + path: bases[path] for path, source in FILES.items() if source == 'image'}


# ---------------------------------------------------------------- pure checks (CPU-tested)

def hold_opened(text: str, tag: str, now: float) -> float:
    ck.require('coordinator' in text and tag in text, 'HOLD is not this window\'s coordinator HOLD')
    opened = datetime.datetime.fromisoformat(text.split()[0].replace('Z', '+00:00')).timestamp()
    ck.require(now - 180 <= opened <= now + 5, 'HOLD timestamp is not the one just written; never adopt or renew a HOLD')
    return opened


def memory_row(row: dict[str, Json]) -> tuple[int, int]:
    kb = record(row['kb'])
    ck.require({'MemAvailable', 'SwapFree'} <= set(kb), 'missing memory telemetry fields')
    return int(number(kb['MemAvailable'])), int(number(kb['SwapFree']))


def gpu_row(row: dict[str, Json]) -> None:
    columns = str(row['csv']).splitlines()
    ck.require(bool(columns) and all(len(line.split(',')) == len(FIELDS.split(',')) for line in columns),
               'nvidia-smi telemetry shape differs from the reviewed fields')


def logs(summary: dict[str, Json], rank: int, cap: int, *, production: bool = False, kv: int | None = None) -> dict[str, Json]:
    """`kv` overrides the expected rank-0 pool for a non-production KV reservation (PF2-LOWKV); default KV[cap]."""
    if production:
        # Long-lived production logs are a record here; only the capacity line binds the snapshot.
        logged = cast(list[Json], summary['kv_tokens'])
        ck.require(rank == 1 or (bool(logged) and all(number(n) == KV[cap] for n in logged)), 'production KV is not 2,502,600')
        return summary
    ck.require(number(summary['stop_hits']) == 0, f'rank{rank} engine log STOP pattern: {summary.get("stop_examples")}')
    ck.require(number(summary['engram_remote']) > 0, f'rank{rank} remote Engram evidence missing')
    if rank == 0:
        want = KV[cap] if kv is None else kv
        found = cast(list[Json], summary['kv_tokens'])
        ck.require(bool(found) and all(number(n) == want for n in found), f'KV capacity is not {want} for cap {cap}')
        pids = cast(list[Json], summary['proxy_pids'])
        ck.require(len(pids) == 1 and number(summary['async_on']) > 0, 'async process proxy startup not proven')
        ck.require(number(summary['worker_ready']) > 0, 'remote DSpark worker never became ready (HELLO refused?)')
    return summary


def worker(probe: dict[str, Json], now: float) -> dict[str, Json]:
    ck.require(probe.get('revision') == WORKER, 'x86host worker revision differs; never overwrite it')
    ck.require(str(probe['cwd']).startswith(str(probe['current']) + '/'), 'running worker source differs from current')
    argv = ck.strings(probe['cmd'])
    ck.require(argv[argv.index('--tp') + 1] == '2' and argv[argv.index('--max-model-len') + 1] == '1048576',
               'worker is not TP2 with HELLO bound 1048576')
    ck.require(probe.get('gpus') == WORKER_GPUS, 'worker GPU identity differs')
    status = record(probe['status'])
    ck.require(status.get('tp') == 2 and status.get('tp_comm_errors') == 0, 'worker TP/communication error')
    stamp = datetime.datetime.fromisoformat(str(status['time']).replace('Z', '+00:00')).timestamp()
    ck.require(abs(now - stamp) < 30, 'worker status stale')
    return status


def runtime(prod: dict[str, Json], live: dict[str, Json], arm: Arm, run: str) -> None:
    """Experiment rank == saved production rank except port/cap/REQUIRED (+ declared trace sources/env, eager)."""
    ck.require(live.get('image') == prod.get('image') == IMAGE, 'image differs from production/pin')
    ck.require(live.get('engine') == prod.get('engine') and live.get('files') == prod.get('files'),
               'engine version or target metadata differs from production')
    ck.require(live.get('state') == 'running', 'experiment container not running')
    ck.require(bool(record(live['engram_identity'])) and live.get('engram_identity') == prod.get('engram_identity'),
               'Engram shim/source/JIT identity differs from production')
    expected = {k: v for k, v in record(prod['mounted']).items() if k != '/serve.sh'}
    if arm.trace:
        expected.update({DIST + path: sha for path, sha in overlay_pins(arm.cause).items()})
    if arm.fix:
        ck.require(not arm.trace and not arm.cause and not arm.eager, 'invalid fix arm')
        expected.update({DIST + path: sha for path, sha in overlay_pins(fix=True).items()})
    got = {k: v for k, v in record(live['mounted']).items() if k != '/serve.sh'}
    ck.require(got == expected, 'mounted engine bytes differ from production (plus declared trace sources)')
    allowed: dict[str, str] = {'VLLM_ENGRAM_REMOTE_REQUIRED': '1'}
    if arm.trace:
        allowed.update({'VLLM_DSV41_PREFILL_TRACE': '1', 'VLLM_DSV41_PREFILL_TRACE_DIR': '/pf1-trace',
                        'VLLM_DSV41_PREFILL_TRACE_RUN': run})
    if arm.cause:
        ck.require(arm.trace == 1 and arm.transient in (0, 1), 'invalid cause arm')
        for key in ('VLLM_EXL3_TF_PROMPT_TRANSIENT', 'VLLM_EXL3_TF_PROMPT_ALIAS'):
            ck.require(record(prod['env']).get(key) == '1', f'production {key} is not 1')
            allowed[key] = str(arm.transient)
    if arm.fix:
        ck.require('VLLM_DSV41_MOE_FENCE' not in record(prod['env']), 'production already sets VLLM_DSV41_MOE_FENCE')
        # the loop below only sees keys present on either side: the switch's absence must fail here
        ck.require(record(live['env']).get('VLLM_DSV41_MOE_FENCE') == '1', 'fix arm lacks VLLM_DSV41_MOE_FENCE=1')
        allowed['VLLM_DSV41_MOE_FENCE'] = '1'
    before, after = record(prod['env']), record(live['env'])
    ck.require(before.get('VLLM_ENGRAM_REMOTE_REQUIRED') == '0', 'production Engram REQUIRED is not the reconciled 0')
    for key in before.keys() | after.keys():
        if key in allowed:
            ck.require(after.get(key) == allowed[key], f'wrong experiment setting {key}')
        else:
            ck.require(before.get(key) == after.get(key), f'env drift from production: {key}')
    a = shlex.split(str(prod['argv']).replace('\\\n', ' '))
    b = shlex.split(str(live['argv']).replace('\\\n', ' '))
    for flag, old, new in (('--port', '8000', '8001'), ('--max-model-len', '327680', str(arm.cap))):
        ck.require(a[a.index(flag) + 1] == old, f'rollback argv {flag} is not production')
        a[a.index(flag) + 1] = new
    if arm.eager:
        ck.require(b.count('--enforce-eager') == 1, 'eager arm lacks --enforce-eager')
        b.remove('--enforce-eager')
    ck.require(a == b, 'launch argv differs from production beyond port/cap')


def cells(rows: list[dict[str, Json]], arm: Arm, run: str, contexts: tuple[int, ...], reps: int) -> list[dict[str, Json]]:
    ck.require(bool(rows) and rows[0].get('type') == 'identity' and rows[0].get('cap') == arm.cap and
               rows[0].get('trace_run') == run, 'request artifact identity/cap/run mismatch')
    requests = rows[1:]
    warm = [r for r in requests if r.get('kind') == 'warmup']
    ck.require(len(warm) == 1 and warm[0].get('target_ctx') == 32000 and warm[0].get('cached_tokens') == 0 and
               record(warm[0]['usage']).get('completion_tokens') == 64, 'warm-up missing or not a full 64-token cold request')
    cold = [r for r in requests if r.get('kind') == 'cold']
    ck.require(len(cold) == len(requests) - 1, 'unexpected request kind')
    ck.require([(r.get('target_ctx'), r.get('rep')) for r in cold] == [(n, i) for n in contexts for i in range(reps)],
               'cold cells incomplete or out of order')
    for row in cold:
        usage = record(row['usage'])
        ck.require(row.get('cap') == arm.cap and usage.get('prompt_tokens') == row['target_ctx'] and
                   row.get('cached_tokens') == 0 and row.get('uncached_prompt_tokens') == row['target_ctx'],
                   'exact prompt/cold cache/cap failure')
        ck.require(row.get('sampling') == {'temperature': 1, 'top_p': 1, 'seed': 42, 'priority': 10, 'max_tokens': 1500},
                   'sampling drift')
        ck.require(row.get('finish') in ('stop', 'length') and number(usage['completion_tokens']) > 0, 'incomplete cold request')
        ttft = row.get('ttft_s')
        ck.require(isinstance(ttft, (int, float)) and not isinstance(ttft, bool) and math.isfinite(ttft) and ttft > 0,
                   'cold request has no first-token time')
        for key in ('prompt_sha256', 'cache_salt_sha256'):
            ck.require(re.fullmatch('[0-9a-f]{64}', str(row.get(key))) is not None, 'missing pairing hash')
    return cold


def paired(arms: dict[str, list[dict[str, Json]]]) -> None:
    keys: dict[tuple[Json, Json], tuple[Json, Json]] = {}
    for rows in arms.values():
        for row in rows:
            pair = (row['prompt_sha256'], row['cache_salt_sha256'])
            ck.require(keys.setdefault((row['target_ctx'], row['rep']), pair) == pair, 'arms are not prompt/salt paired')


def gate(arms: dict[str, list[dict[str, Json]]]) -> dict[str, Json]:
    """Design gate (PF1-WINDOW-DESIGN.md): window continuation only, never a promotion decision."""
    paired(arms)
    points: list[Json] = []
    gaps: list[float] = []
    spreads: list[float] = []
    for n in OFF_CONTEXTS:
        medians: dict[str, float] = {}
        for name in ('off-327', 'off-524'):
            times = [float(number(r['ttft_s'])) for r in arms[name] if r['target_ctx'] == n]
            ck.require(len(times) == OFF_REPS, 'gate needs both repetitions of every OFF cell')
            medians[name] = statistics.median(times)
            spreads.append(abs(times[0] - times[1]) / statistics.mean(times))
        gap = medians['off-524'] / medians['off-327'] - 1
        gaps.append(gap)
        points.append({'context': n, 'median_ttft_s': cast(Json, medians), 'gap': gap,
                       'excess_ms_per_prompt_token': (medians['off-524'] - medians['off-327']) * 1000 / n,
                       's18_reference_ms_per_prompt_token': 0.163})
    quiet = max(spreads) <= .03
    status = ('REPRODUCED' if quiet and min(gaps) >= .10 else
              'NOT_REPRODUCED' if quiet and max(gaps) < .05 else 'AMBIGUOUS')
    return {'classification': 'MEASURED TTFT (prefill proxy)', 'status': status, 'points': points,
            'max_rep_spread': max(spreads), 'thresholds': {'reproduced_gap': .10, 'absent_gap': .05, 'rep_spread': .03}}


MATCH_THRESHOLDS = {'faster': .20, 'like_cause': .05, 'like_off': .05, 'rep_spread': .03}


def safety(off: dict[str, Json] | None, fix: dict[str, Json] | None) -> dict[str, Json]:
    """fix-524 vs off-524 of the same HOLD: greedy answer hashes 6/6 and the first-token top-20 floor (up2 rule)."""
    if off is None or fix is None:
        return {'status': 'NOT RUN'}
    def rows(checks: dict[str, Json], kind: str) -> list[dict[str, Json]]:
        return [record(r) for r in cast(list[Json], checks[kind])]
    a, b = rows(off, 'greedy'), rows(fix, 'greedy')
    ck.require([r.get('i') for r in a] == [r.get('i') for r in b] == list(range(6)), 'greedy checks incomplete')
    same = [x['i'] for x, y in zip(a, b) if x['text_sha256'] == y['text_sha256'] and
            x['completion_tokens'] == y['completion_tokens']]
    out: dict[str, Json] = {'greedy_identical': len(same), 'greedy_of': 6}
    la, lb = rows(off, 'logit'), rows(fix, 'logit')
    ck.require([r.get('salt_suffix') for r in la] == ['a', 'aa'] and [r.get('salt_suffix') for r in lb] == ['b'],
               'logit checks incomplete')
    ck.require(len({r['prompt_sha256'] for r in (*la, *lb)}) == 1, 'logit probes used different prompts')
    try:
        out['logit'] = ck.numerical(la[0], la[1], lb[0])
        logit_ok = True
    except ValueError as exc:
        out['logit'] = {'status': 'FAILED', 'error': str(exc)}
        logit_ok = False
    out['status'] = 'PASS' if len(same) == 6 and logit_ok else 'FAIL'
    return out


def match_reading(arms: dict[str, list[dict[str, Json]]], checks: dict[str, dict[str, Json] | None]) -> dict[str, Json]:
    """Pre-registered (PF1-CAUSE2 MATCH.md, written before any match window). Never a promotion decision."""
    paired(arms)
    t = {name: [float(number(r['ttft_s'])) for r in arms[name] if r['target_ctx'] == 150000] for name in
         (a.name for a in MATCH_ARMS)}
    ck.require(all(len(v) == MATCH_REPS for v in t.values()), 'match needs both repetitions of every arm')
    med = {k: statistics.median(v) for k, v in t.items()}
    spread = {k: abs(v[0] - v[1]) / statistics.mean(v) for k, v in t.items()}
    th = MATCH_THRESHOLDS
    off, cause = med['off-524'], med['cause-524']
    quiet = max(spread.values()) <= th['rep_spread']
    effect = all(x <= (1 - th['faster']) * off for x in t['cause-524'])
    like_cause = all(abs(x / cause - 1) <= th['like_cause'] for x in t['fix-524'])
    fast = all(x <= (1 - th['faster']) * off for x in t['fix-524'])
    like_off = all(x >= (1 - th['like_off']) * off for x in t['fix-524'])
    if not quiet:
        speed = 'AMBIGUOUS'
    elif not effect:
        speed = 'CAUSE_NOT_REPRODUCED'
    elif like_cause and fast:
        speed = 'FIX_REPRODUCES_CAUSE'
    elif like_off:
        speed = 'FIX_REFUTED'
    else:
        speed = 'AMBIGUOUS'
    safe = safety(checks.get('off-524'), checks.get('fix-524'))
    verdict = ('UNSAFE' if safe.get('status') == 'FAIL' else
               'CONFIRMED' if speed == 'FIX_REPRODUCES_CAUSE' and safe.get('status') == 'PASS' else
               'REFUTED' if speed == 'FIX_REFUTED' else
               'CAUSE_NOT_REPRODUCED' if speed == 'CAUSE_NOT_REPRODUCED' else 'AMBIGUOUS')
    return {'classification': 'MEASURED TTFT (prefill proxy), 150K cold x2, one HOLD', 'verdict': verdict,
            'speed': speed, 'safety': safe, 'ttft_s': cast(Json, t), 'median_ttft_s': cast(Json, med),
            'rep_spread': cast(Json, spread), 'fix_vs_cause': med['fix-524'] / cause - 1,
            'off_vs_cause': off / cause - 1, 'thresholds': cast(Json, dict(th))}


# The engine traces its own startup/readiness requests too (window 1 MEASURED: 2/7/7/7/7/15 tokens on both ranks
# beside the window's 32K warm-up and 150K request); they are reported, never matched to a sent request.
SMALL_TRACED_TOKENS = 1024


def trace_check(files: dict[int, list[Path]], rows: list[dict[str, Json]]) -> dict[str, Json]:
    """Both ranks, complete chunk events, every sent request traced once; extras only tiny engine requests."""
    sent = [r for r in rows if r.get('type') == 'request']
    per_rank: dict[int, dict[str, int]] = {}
    hashes: set[str] = set()
    for rank in (0, 1):
        ck.require(bool(files.get(rank)), f'trace files missing for rank{rank}')
        began: set[str] = set()
        total: set[str] = set()
        size: dict[str, int] = {}
        count = 0
        for path in files[rank]:
            for line in path.read_text().splitlines():
                row = record(cast(Json, json.loads(line)))
                count += 1
                ck.require(row.get('rank') == rank, 'trace row on the wrong rank file')
                ck.require(not row.get('failed'), 'a traced span failed')
                if row.get('phase') not in ('chunk.begin', 'chunk.total'):
                    continue
                members = [record(m) for m in cast(list[Json], row['members'])]
                for member in members:
                    chunk = str(member['chunk'])
                    if row['phase'] == 'chunk.begin':
                        began.add(chunk)
                        request = str(member['request'])
                        size[request] = max(size.get(request, 0), int(number(member['prefix_tokens']) + number(member['new_tokens'])))
                    else:
                        ck.require(row.get('cuda_ms') is not None, 'uncompleted CUDA chunk event')
                        total.add(chunk)
        ck.require(began == total and bool(began), f'rank{rank} chunk begin/total events incomplete')
        extra = sorted(size.values())
        for target in sorted(int(number(r['target_ctx'])) for r in sent):
            ck.require(target in extra, f'rank{rank} trace lacks the sent {target}-token request')
            extra.remove(target)
        ck.require(all(x < SMALL_TRACED_TOKENS for x in extra),
                   f'rank{rank} traced an unsent request of {max(extra, default=0)} tokens')
        per_rank[rank] = {'rows': count, 'chunks': len(began), 'requests': len(size), 'unsent_small_requests': len(extra)}
        if rank == 0:
            hashes = set(size)
        else:
            ck.require(set(size) == hashes, 'ranks traced different request identities')
    matched = sum(str(r['trace_request']) in hashes for r in sent)
    return {'ranks': cast(Json, per_rank), 'trace_request_hash_matches': matched, 'sent': len(sent),
            'correlation': 'HASH' if matched == len(sent) else 'BY_SIZE_ONLY (engine id differs from SSE id; report as such)'}


def restored(step: str, rc: int, text: str) -> None:
    ck.require(rc == 0, f'restore {step} rc{rc}')
    if step == 'start':
        ck.require('K1 START VERIFIED' in text, 'restore start lacks K1 START VERIFIED')
    elif step == 'check':
        same = re.findall(r'greedy K1-post vs restored: (\d+) IDENTICAL \d+', text)
        ck.require(sorted(same) == ['0', '1', '2', '3', '4', '5'] and 'DIVERGES' not in text, 'greedy is not 6/6 IDENTICAL')
        ck.require('PRODUCTION HOLD OK' in text and 'CHECK: DONE' in text, 'x86host PRODUCTION HOLD OK / check not done')
    else:
        ck.require('CLOSE: window closed' in text, 'window close not confirmed')


# ---------------------------------------------------------------- supervision of owned processes

class Supervisor:
    """Owned local process groups only; 1 s telemetry from both Sparks; STOP file, deadline and demand guards."""

    def __init__(self, out: Path, *, clock: Callable[[], float] = time.time,
                 monitor: Callable[[str], list[str]] | None = None,
                 demand: Callable[[], float] | None = None) -> None:
        self.out, self.clock = out, clock
        self.monitor = monitor or (lambda host: [*SSH, host, 'exec python3 -u -c ' + shlex.quote(TELEMETRY) +
                                                 ' ' + shlex.quote(FIELDS) + ' 0'])
        self.demand = demand or self.metrics
        self.abort_at = math.inf
        self.bound = '80-minute measurement bound'      # STOP text for abort_at (SP1 sets its own)
        self.select = selectors.DefaultSelector()
        self.procs: list[subprocess.Popen[str]] = []
        self.files: list[IO[str]] = []
        self.pending: dict[int, str] = {}
        self.last: dict[str, float] = {}
        self.swap: dict[str, int] = {}
        self.ready: dict[str, int] = {}
        self.gpu_errors: dict[str, int] = {}
        self.serving = False
        self.inflight = 0
        self.demand_due = 0.0
        self.demand_failures = 0
        self.active: subprocess.Popen[str] | None = None
        self.serial = 0
        self.closed = False
        self.arm_minima: dict[str, int] | None = None
        self.arm_memory: dict[str, dict[str, int]] = {}
        (out / 'cmd').mkdir()

    def begin(self) -> None:
        for host in HOSTS:
            err = (self.out / f'telemetry-{host}.stderr').open('x')
            log = (self.out / f'telemetry-{host}.jsonl').open('x')
            self.files.extend([err, log])
            proc = subprocess.Popen(self.monitor(host), stdout=subprocess.PIPE, stderr=err, text=True,
                                    start_new_session=True)
            self.procs.append(proc)
            assert proc.stdout is not None
            os.set_blocking(proc.stdout.fileno(), False)
            self.select.register(proc.stdout, selectors.EVENT_READ, (host, log))
            self.pending[proc.stdout.fileno()] = ''
        start = time.monotonic()
        while len(self.last) < len(HOSTS):
            ck.require(time.monotonic() - start < 15, 'initial Spark telemetry missing')
            self.poll()

    def tunnel(self, command: list[str]) -> None:
        with socket.socket() as probe:
            ck.require(probe.connect_ex(('127.0.0.1', 18001)) != 0, '127.0.0.1:18001 already in use; never reuse a foreign tunnel')
        err = (self.out / 'tunnel.stderr').open('x')
        self.files.append(err)
        self.procs.append(subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err,
                                           text=True, start_new_session=True))
        end = time.monotonic() + 15
        while True:
            self.poll()
            with socket.socket() as probe:
                if probe.connect_ex(('127.0.0.1', 18001)) == 0:
                    return
            ck.require(time.monotonic() < end, 'experiment tunnel did not come up')
            time.sleep(.2)

    def metrics(self) -> float:
        with urllib.request.urlopen(URL + '/metrics', timeout=3) as response:
            return request_gauges(response.read().decode(), 'experiment')

    def mark_ready(self) -> None:
        self.ready = dict(self.swap)
        self.serving = True

    def unready(self) -> None:
        self.ready, self.serving = {}, False

    def poll(self) -> None:
        ck.require(not (self.out / 'STOP').exists(), 'external STOP file')
        ck.require(self.clock() < self.abort_at, f'{self.bound} {self.abort_at:.0f} reached')
        for key, _ in self.select.select(timeout=.1):
            stream = cast(IO[str], key.fileobj)
            part = os.read(stream.fileno(), 65536).decode()
            ck.require(bool(part), 'Spark telemetry stream ended')
            self.pending[stream.fileno()] += part
            host, log = cast(tuple[str, IO[str]], key.data)
            while '\n' in self.pending[stream.fileno()]:
                line, self.pending[stream.fileno()] = self.pending[stream.fileno()].split('\n', 1)
                row = record(cast(Json, json.loads(line)))
                if row.get('kind') == 'memory':
                    available, swap = memory_row(row)
                    self.last[host], self.swap[host] = time.monotonic(), swap
                    if self.arm_minima is not None:
                        self.arm_minima[host] = min(self.arm_minima.get(host, available), available)
                    ck.require(available >= MEM_FLOOR_KB, f'{host} MemAvailable {available} kB below 4 GiB')
                    if host in self.ready:
                        ck.require(swap >= self.ready[host] - SWAP_GROWTH_KB, f'{host} new swap use after readiness')
                elif row.get('kind') == 'gpu':
                    gpu_row(row)
                    self.gpu_errors[host] = 0
                else:
                    self.gpu_errors[host] = self.gpu_errors.get(host, 0) + 1
                    ck.require(self.gpu_errors[host] < 10, f'{host} GPU telemetry failing; clocks/throttle unrecorded')
                log.write(json.dumps(row, allow_nan=False) + '\n')
                log.flush()
        ck.require(all(p.poll() is None for p in self.procs), 'telemetry or tunnel process exited')
        ck.require(all(time.monotonic() - t < 12 for t in self.last.values()), 'stale Spark telemetry')
        if self.serving and time.monotonic() >= self.demand_due:
            self.demand_due = time.monotonic() + 5
            try:
                busy = self.demand()
                self.demand_failures = 0
            except (OSError, ValueError):
                self.demand_failures += 1
                ck.require(self.demand_failures < 3, 'experiment metrics unavailable; demand guard blind')
                return
            ck.require(busy <= self.inflight, f'foreign demand on the experiment ({busy:g} > owned {self.inflight})')

    def run(self, name: str, command: list[str], *, stdin: str | None = None, timeout: float = 120,
            inflight: int = 0) -> str:
        self.poll()
        self.serial += 1
        path = self.out / 'cmd' / f'{self.serial:04d}-{name}.log'
        started = time.monotonic()
        with path.open('x') as log:
            source: IO[str]
            if stdin is not None:
                staged = path.with_suffix('.stdin')
                staged.write_text(stdin)
                source = staged.open()
            else:
                source = open(os.devnull)
            with source:
                proc = subprocess.Popen(command, stdin=source, stdout=log, stderr=subprocess.STDOUT, text=True,
                                        start_new_session=True, cwd=ROOT)
                self.active, self.inflight = proc, inflight
                try:
                    while proc.poll() is None:
                        self.poll()
                        ck.require(time.monotonic() - started < timeout, f'{name} timeout')
                    log.flush()
                    deadline = re.search(r'window submission deadline reached', path.read_text())
                    ck.require(deadline is None, f'{name}: window submission deadline reached')
                    ck.require(proc.returncode == 0, f'{name} failed rc{proc.returncode}; see {path.name}')
                finally:
                    terminate(proc)
                    self.active, self.inflight = None, 0
        return path.read_text()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.unready()
        if self.active is not None:
            terminate(self.active)
        for proc in self.procs:
            terminate(proc)
        self.select.close()
        for handle in self.files:
            handle.close()


def request_gauges(text: str, what: str) -> float:
    """running + waiting requests from a vLLM /metrics page; both gauges must be present."""
    lines = text.splitlines()
    found = {key: [float(x.split()[-1]) for x in lines if x.startswith((key + '{', key + ' '))]
             for key in ('vllm:num_requests_running', 'vllm:num_requests_waiting')}
    ck.require(all(found.values()), f'{what} metrics lack request gauges')
    return sum(sum(v) for v in found.values())


# S40 FIX 1-2: after an abort, this window's own remote sessions may still be running on sparkhead (a draining
# `window.sh open` would write a HOLD later; an abandoned start_exp.sh would race the W8 restart). Each owned command
# line is matched by this window's tag or experiment config (never a production process) and its WHOLE ssh session is
# terminated. The bracketed '.' keeps the patterns from matching this script's own command line.
REAP = r'''me=$(ps -o sid= -p $$ | tr -d ' ')
reap() {
  for pid in $(pgrep -f -- "$1"); do
    sid=$(ps -o sid= -p "$pid" 2>/dev/null | tr -d ' ')
    if [ -n "$sid" ] && [ "$sid" != "$me" ]; then pkill -TERM -s "$sid" && echo "reaped session $sid: $2"; fi
  done
}
reap 'window[.]sh open PF1-WINDOW @TAG@:' 'HOLD opener'
reap 'start_exp[.]sh offload/launch/configs/PF1-window[.]env' 'experiment start'
sleep 3
for pid in $(pgrep -f -- 'window[.]sh open PF1-WINDOW @TAG@:|start_exp[.]sh offload/launch/configs/PF1-window[.]env'); do
  sid=$(ps -o sid= -p "$pid" 2>/dev/null | tr -d ' ')
  if [ -n "$sid" ] && [ "$sid" != "$me" ]; then pkill -KILL -s "$sid" && echo "killed session $sid"; fi
done
echo reap done'''
# A JIT prebuild container is never killed (a torch-extension build killed mid-way can leave a stale lock in the
# shared ~/.cache/dsv41-serve that production's own start would then wait on): wait for it to finish instead.
JIT_WAIT = r'''for i in $(seq 1 60); do
  n=$(docker ps --no-trunc --format '{{.Command}}' | grep -c 'tf_experts' || true)
  if [ "$n" = 0 ]; then echo "no JIT prebuild running"; exit 0; fi
  sleep 5
done
echo "JIT prebuild still running after 300 s"; exit 1'''
IDLE_WAIT_S = 300


# ---------------------------------------------------------------- orchestration (shared by Live and fakes)

class Backend(Protocol):
    def preflight(self) -> None: ...
    def open_hold(self) -> float: ...
    def snapshot(self) -> None: ...
    def stage_trace(self) -> None: ...
    def start(self, arm: Arm) -> None: ...
    def requests(self, arm: Arm, contexts: tuple[int, ...], reps: int) -> list[dict[str, Json]]: ...
    def collect(self, arm: Arm, rows: list[dict[str, Json]]) -> dict[str, Json]: ...
    def restore(self, *, production_stopped: bool) -> dict[str, Json]: ...


def window(b: Backend, phase: str, checkpoint: Callable[[dict[str, Json]], None], *,
           on_contexts: tuple[int, ...] = (150000,), cause_t0: bool = False, prior_gate: dict[str, Json] | None = None) -> dict[str, Json]:
    """Whatever happens after HOLD is written, the W8 restore runs; a failed restore is never hidden."""
    result: dict[str, Json] = {'phase': phase, 'status': 'INCOMPLETE', 'arms': {}}
    arms = cast(dict[str, Json], result['arms'])
    hold = stopped = False
    try:
        b.preflight()
        hold = True    # from here a HOLD may exist (open can time out after writing it): always restore/close
        result['hold_opened'] = b.open_hold()
        b.snapshot()
        cold: dict[str, list[dict[str, Json]]] = {}
        if phase == 'match':
            # PF1-CAUSE2: off / cause / fix at 524,288 in this one HOLD; no prior gate (off-524 is the control)
            b.stage_trace()
            found: dict[str, dict[str, Json] | None] = {}
            for arm in MATCH_ARMS:
                stopped = True
                b.start(arm)
                cold[arm.name] = b.requests(arm, (150000,), MATCH_REPS)
                got = b.collect(arm, cold[arm.name])
                arms[arm.name] = {'cold': cast(Json, cold[arm.name]), 'checks': got}
                found[arm.name] = record(got['match']) if 'match' in got else None
                checkpoint({'arm': arm.name, 'cells': len(cold[arm.name])})
            result['match'] = match_reading(cold, found)
            checkpoint({'match': result['match']})
            result['status'] = 'COMPLETE'
        else:
            if phase in ('all', 'off'):
                for arm in OFF_ARMS:
                    stopped = True    # start_exp.sh stops production before anything else
                    b.start(arm)
                    cold[arm.name] = b.requests(arm, OFF_CONTEXTS, OFF_REPS)
                    arms[arm.name] = {'cold': cast(Json, cold[arm.name]), 'checks': b.collect(arm, cold[arm.name])}
                    checkpoint({'arm': arm.name, 'cells': len(cold[arm.name])})
                result['gate'] = gate(cold)
                checkpoint({'gate': result['gate']})
            else:
                ck.require(prior_gate is not None, 'ON-only phase needs the prior OFF window gate')
                result['gate'] = cast(Json, prior_gate)
            status = record(result['gate']).get('status')
            if phase in ('all', 'on', 'cause') and status == 'REPRODUCED':
                b.stage_trace()
                selected = (*CAUSE_ARMS, *((CAUSE_T0,) if cause_t0 else ())) if phase == 'cause' else ON_ARMS
                for arm in selected:
                    stopped = True
                    b.start(arm)
                    rows = b.requests(arm, (150000,) if phase == 'cause' else on_contexts, 1)
                    arms[arm.name] = {'cold': cast(Json, rows), 'checks': b.collect(arm, rows)}
                    checkpoint({'arm': arm.name, 'cells': len(rows)})
                paired({k: cast(list[dict[str, Json]], record(v)['cold']) for k, v in arms.items()})
            result['status'] = 'COMPLETE' if phase != 'all' or status == 'REPRODUCED' else f'COMPLETE: gate {status}, ON skipped'
    except (Exception, KeyboardInterrupt) as exc:
        result['stop'] = f'{type(exc).__name__}: {exc}'
    finally:
        # S40 FIX 3: ignore hangups/interrupts BEFORE the backend closes its supervisor; a second signal in that gap
        # would otherwise skip the W8 restore.
        saved = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
        try:
            if hold:
                try:
                    result['restore'] = b.restore(production_stopped=stopped)
                except (Exception, KeyboardInterrupt) as exc:
                    result['restore'] = {'status': 'FAILED', 'error': f'{type(exc).__name__}: {exc}'}
            else:
                result['restore'] = {'status': 'NOT NEEDED: no HOLD written, production untouched'}
        finally:
            for sig, handler in saved.items():
                signal.signal(sig, handler)
    return result


# ---------------------------------------------------------------- live backend (coordinator only)

Call = Callable[..., str]


class Live:
    def __init__(self, out: Path, sup: Supervisor, *, remote: str, expected: str, data_dir: Path, tag: str,
                 stage_bundle: bool, call: Call | None = None, plain: Callable[[str, list[str], float], tuple[int, str]] | None = None,
                 clock: Callable[[], float] = time.time, cause: bool = False, match: bool = False) -> None:
        self.cause, self.match = cause, match
        self.out, self.sup, self.remote, self.expected = out, sup, remote, expected
        self.data_dir, self.tag, self.stage_bundle, self.clock = data_dir, tag, stage_bundle, clock
        self.call: Call = call or sup.run
        self.plain = plain or self.local
        self.submit = math.inf
        self.production: list[dict[str, Json]] = []
        self.ids: list[str] = []
        self.seen: set[str] = set()
        self.proxy_pid = 0
        self.first_output: Path | None = None
        self.hold_written = False

    @property
    def pin_files(self) -> tuple[str, ...]:
        return MATCH_PIN_FILES if self.match else CAUSE_PIN_FILES if self.cause else PIN_FILES

    def ssh(self, label: str, host: str, command: str, stdin: str | None = None, timeout: float = 120,
            inflight: int = 0) -> str:
        return self.call(label, [*SSH, host, command], stdin=stdin, timeout=timeout, inflight=inflight)

    def json(self, label: str, host: str, command: str, stdin: str | None = None) -> dict[str, Json]:
        lines = [x for x in self.ssh(label, host, command, stdin).splitlines() if x.startswith('{')]
        ck.require(len(lines) == 1, f'{label}: expected one JSON line')
        return record(cast(Json, json.loads(lines[0])))

    def local(self, name: str, command: list[str], timeout: float) -> tuple[int, str]:
        """Restore path: no STOP-file/telemetry supervision may block restoring production."""
        log = self.out / 'cmd' / f'restore-{name}.log'
        with log.open('x') as handle:
            try:
                rc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True, timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                rc = 124
        return rc, log.read_text()

    def run_name(self, arm: Arm) -> str:
        return f'{self.tag}-{arm.name}'

    # -- preflight, read-only (production keeps serving)
    def preflight(self) -> None:
        self.sup.begin()
        ck.require(digest(RESTORE) == RESTORE_SHA256, 'W8/restore.sh changed; re-review its STOP strings before use')
        ck.require((CAMPAIGN / 'D2-GEN.PAUSE').exists() or subprocess.run(
            ['tmux', 'has-session', '-t', '=d2-gen'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0,
            'D2 generation active without pause')
        from offload.prefill_trace.window_requests import corpus
        corpus(self.data_dir)
        self.ssh('ledger', 'sparkhead', 'cd ~/dsv41-deploy && test ! -e HOLD && '
                 f'test "$(git rev-parse --short=7 HEAD)" = {PROD_COMMIT} && '
                 f'test "$(git rev-parse HEAD)" = "$(git rev-list -n1 {PROD_TAG})" && '
                 'git diff --exit-code HEAD -- dsv41.env serve.sh start.sh')
        checkout = (f'cd ~/{shlex.quote(self.remote)} && test "$(git rev-parse HEAD)" = {self.expected} && '
                    'test -z "$(git status --porcelain --untracked-files=no)"')
        try:
            self.ssh('checkout', 'sparkhead', checkout)
        except ValueError:
            ck.require(self.stage_bundle, 'experiment checkout is not at the reviewed commit; stage it (--stage-bundle)')
            self.bundle()
            self.ssh('checkout-staged', 'sparkhead', checkout)
        remote = self.ssh('pins', 'sparkhead', f'cd ~/{shlex.quote(self.remote)} && sha256sum ' + shlex.join(self.pin_files))
        ck.require(remote.strip() == '\n'.join(f'{digest(ROOT / p)}  {p}' for p in self.pin_files), 'staged sources differ')
        self.ssh('prod-health', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8000/health')
        models = ck.rows(self.json('prod-models', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8000/v1/models')['data'])
        ck.require(len(models) == 1 and models[0].get('max_model_len') == 327680 and
                   models[0].get('root') == '/srv/dsv41/legal-v2a', 'production is not K1 327680 legal-v2a')
        worker(self.json('worker', 'x86host', 'python3 -c ' + shlex.quote(X86HOST_PROBE)), self.clock())
        for host in HOSTS:
            rows = [record(cast(Json, json.loads(x))) for x in self.ssh(
                f'telemetry-probe-{host}', host, 'python3 -c ' + shlex.quote(TELEMETRY) + ' ' + shlex.quote(FIELDS) + ' 1'
            ).splitlines() if x.startswith('{')]
            ck.require([r.get('kind') for r in rows] == ['memory', 'gpu'], f'{host} nvidia-smi fields unsupported; review telemetry')
            gpu_row(rows[1])
            ck.require(memory_row(rows[0])[0] >= START_FLOOR_KB, f'{host} MemAvailable below 5 GiB before the window')

    def bundle(self) -> None:
        local = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        ck.require(local == self.expected, 'local checkout is not the reviewed commit')
        path = self.out / f'pf1-{self.expected[:7]}.bundle'
        self.call('bundle', ['git', 'bundle', 'create', str(path), 'HEAD'])
        name = f'pf1-window-{self.expected[:12]}.bundle'
        self.call('bundle-copy', ['scp', '-q', '-o', 'BatchMode=yes', str(path), f'sparkhead:{name}'])
        self.ssh('bundle-fetch', 'sparkhead', f'cd ~/{shlex.quote(self.remote)} && '
                 'test -z "$(git status --porcelain --untracked-files=no)" && '
                 f'git fetch -q ~/{name} HEAD && test "$(git rev-parse FETCH_HEAD)" = {self.expected} && '
                 f'git checkout -q --detach {self.expected} && rm -f ~/{name}')

    # -- HOLD and the saved production identity
    def hold_purpose(self) -> str:
        """HOLD text; its 'PF1-WINDOW <tag>:' prefix is what reap_script() matches (subclass hook)."""
        return (f'PF1-WINDOW {self.tag}: matched 524288 off/cause/fix (PF1-CAUSE2); no attention changes' if self.match
                else f'PF1-WINDOW {self.tag}: matched 327680/524288 prefill attribution; no attention changes')

    def open_hold(self) -> float:
        purpose = self.hold_purpose()
        try:
            # Remote timeout too: an abandoned window.sh must not write a HOLD after this runner gave up.
            self.ssh('hold-open', 'sparkhead', f'cd ~/dsv41-deploy && timeout {DRAIN_S} offload/launch/window.sh open '
                     + shlex.quote(purpose), timeout=DRAIN_S + 60)
        finally:
            # Even a timed-out open may have written the HOLD: restoration follows whatever is actually there.
            text = self.ssh('hold-read', 'sparkhead', 'cat ~/dsv41-deploy/HOLD 2>/dev/null || true')
            self.hold_written = self.tag in text
        opened = hold_opened(text, self.tag, self.clock())
        self.submit, self.sup.abort_at = opened + SUBMIT_S, opened + ABORT_S
        (self.out / 'deadlines.json').write_text(json.dumps({'opened': opened, 'submission': self.submit,
            'abort': self.sup.abort_at, 'restore_by': opened + 90 * 60}) + '\n')
        self.sup.tunnel([*SSH, '-N', '-o', 'ExitOnForwardFailure=yes', '-L', '127.0.0.1:18001:127.0.0.1:8001', 'sparkhead'])
        return opened

    def capture(self, host: str, label: str) -> dict[str, Json]:
        text = self.ssh(f'{label}-identity-{host}', host, 'bash -s', (ROOT / 'offload/dspark/d2/capture_window_identity.sh').read_text())
        objects = [record(cast(Json, json.loads(x))) for x in text.splitlines() if x.startswith('{')]
        ck.require(len(objects) == 2, 'incomplete rank identity capture')
        container, files = objects
        config = self.json(f'{label}-config-{host}', host, "docker inspect --format '{{json .Config}}' dsv41-serve")
        env: dict[str, Json] = {}
        for entry in ck.strings(config['Env']):
            key, _, value = entry.partition('=')
            if key.startswith(('VLLM_', 'EXL3_', 'NCCL_', 'PYTORCH_', 'CUDA_')) or key in ('PYTHONPATH', 'DSV41_ENGRAM_OVERLAY'):
                ck.require(key not in env, 'duplicate runtime environment key')
                env[key] = value
        target = {k: v for k, v in record(files['files']).items() if k.startswith('/srv/dsv41/legal-v2a/')}
        target['generation_config'] = files['generation_config']
        return {'id': container['id'], 'state': container['state'], 'image': container['image'],
                'engine': files['engine'], 'files': target, 'mounted': files['mounted_code_sha256'], 'env': env,
                'engram_identity': self.json(f'{label}-engram-{host}', host, 'docker exec -i dsv41-serve python3 -', ENGRAM_PROBE),
                'argv': self.ssh(f'{label}-argv-{host}', host, 'docker exec dsv41-serve cat /serve.sh')}

    def verify(self, host: str, label: str, trace: int, *, cause: bool = False, fix: bool = False) -> None:
        script = fix_verifier() if fix else verifier(bool(trace), cause=cause)
        text = self.ssh(f'{label}-verify-{host}', host, 'docker exec -i dsv41-serve python3 -', script)
        ck.require(f'PF1 verified {14 if fix else 18 if cause else 17 if trace else 13} engine sources' in text,
                   'in-container source pins not verified')

    def snapshot(self) -> None:
        pins = image_pins()
        for rank, host in enumerate(HOSTS):
            self.production.append(self.capture(host, 'prod'))
            self.verify(host, 'prod', 0)
            bases = self.json(f'prod-image-bases-{host}', host, 'docker exec -i dsv41-serve python3 - ' + shlex.join(pins), IMAGE_BASES)
            ck.require(bases == pins, 'image trace bases differ from BASES.json; rebuild/review, never force')
            logs(self.json(f'prod-logs-{host}', host, 'python3 -c ' + shlex.quote(LOG_FILTER) + ' ' + shlex.quote(ck.STOP_LOG.pattern)),
                 rank, 327680, production=True)
        a, b = self.production
        ck.require(a['image'] == b['image'] == IMAGE and a['files'] == b['files'] and
                   {k: v for k, v in record(a['mounted']).items() if k != '/serve.sh'} ==
                   {k: v for k, v in record(b['mounted']).items() if k != '/serve.sh'}, 'production ranks disagree')
        self.seen.update(str(r['id']) for r in self.production)
        self.production_idle()
        (self.out / 'rollback.json').write_text(json.dumps({'production_commit': PROD_COMMIT, 'tag': PROD_TAG,
            'checkout': '~/dsv41-deploy', 'worker_source': WORKER, 'restore_sha256': RESTORE_SHA256,
            'ranks': cast(Json, self.production)}, indent=1) + '\n')

    def production_idle(self) -> None:
        """S40 FIX 4 (as UP2): the first start_exp.sh stops production and kills in-flight work; wait for idle."""
        end = time.monotonic() + IDLE_WAIT_S
        while True:
            busy = request_gauges(self.ssh('prod-idle', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8000/metrics'),
                                  'production')
            if busy == 0:
                return
            ck.require(time.monotonic() < end, f'production not idle before the first stop ({busy:g} running+waiting)')
            time.sleep(5)

    def stage_trace(self) -> None:
        self.stage_overlay('prefill_trace', 'trace')
        if self.cause or self.match:
            self.stage_overlay('prefill_cause', 'cause')
        if self.match:
            self.stage_overlay('prefill_cause2', 'fix', 'build_fix')

    def stage_overlay(self, family: str, label: str, builder: str = 'build_overlay') -> None:
        source = self.out / f'{label}-src'
        self.call(f'{label}-build', [sys.executable, '-m', f'offload.{family}.{builder}', '--enabled',
                                  '--output-root', str(source)])
        self.call(f'{label}-hash', ['bash', '-c', f'cd {shlex.quote(str(source))} && sha256sum --quiet -c '
                                 + shlex.quote(str(ROOT / f'offload/{family}/OVERLAY.sha256'))])
        base = f'~/{shlex.quote(self.remote)}/offload/{family}'
        state = self.ssh(f'{label}-remote', 'sparkhead', f'cd {base} && if [ -e generated ]; then '
                         '(cd generated && sha256sum --quiet -c ../OVERLAY.sha256) && echo REUSED; else echo ABSENT; fi')
        if 'ABSENT' in state:
            self.call(f'{label}-copy', ['scp', '-rq', '-o', 'BatchMode=yes', str(source),
                                     f'sparkhead:{self.remote}/offload/{family}/generated'])
            self.ssh(f'{label}-verify', 'sparkhead', f'cd {base}/generated && sha256sum --quiet -c ../OVERLAY.sha256')
        else:
            ck.require('REUSED' in state, f'existing remote {label} sources differ; never overwrite')

    # -- arms
    def start(self, arm: Arm) -> None:
        ck.require(self.clock() + ARM_BUDGET_S <= self.submit, f'{arm.name}: not enough time before the submission bound')
        self.ssh(f'{arm.name}-hold', 'sparkhead', 'test -f ~/dsv41-deploy/HOLD')
        self.sup.unready()
        if arm.cause or self.match:
            self.sup.arm_minima = {}
            self.sup.arm_memory[arm.name] = self.sup.arm_minima
            for host in HOSTS:
                text = self.ssh(f'{arm.name}-host-parity-{host}', host, HOST_PARITY)
                available = re.findall(r'^MemAvailable:\s+(\d+) kB$', text, re.M)
                ck.require(len(available) == 1, f'{host} missing MemAvailable before {arm.name} start')
                self.sup.arm_minima[host] = min(self.sup.arm_minima.get(host, int(available[0])), int(available[0]))
                ck.require(int(available[0]) >= START_FLOOR_KB,
                           f'{host} MemAvailable below 5 GiB before {arm.name} start')
        env = f'PF1_CAP={arm.cap} PF1_TRACE={arm.trace} PF1_EAGER={arm.eager}' + self.arm_env(arm)
        if arm.cause:
            env += f' PF1C_TF_TRANSIENT={arm.transient}'
        if arm.trace:
            run = self.run_name(arm)
            for host in HOSTS:
                self.ssh(f'{arm.name}-trace-dir-{host}', host,
                         f'mkdir -p ~/dsv41-offload-run/pf1-trace && mkdir ~/dsv41-offload-run/pf1-trace/{run}')
            env += f' PF1_TRACE_RUN={run}'
        self.ssh(f'{arm.name}-start', 'sparkhead',
                 f'cd ~/{shlex.quote(self.remote)} && {env} timeout {START_S} bash offload/launch/start_exp.sh {arm.config}',
                 timeout=START_S + 60)
        self.ssh(f'{arm.name}-health', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8001/health')
        models = ck.rows(self.json(f'{arm.name}-models', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8001/v1/models')['data'])
        ck.require(len(models) == 1 and models[0].get('max_model_len') == arm.cap and
                   models[0].get('root') == '/srv/dsv41/legal-v2a', 'experiment model/cap identity failure')
        current = [self.capture(host, arm.name) for host in HOSTS]
        for rank, host in enumerate(HOSTS):
            self.runtime_check(rank, current[rank], arm)
            ck.require(str(current[rank]['id']) not in self.seen, 'start reused a previous container')
            self.verify(host, arm.name, arm.trace, cause=arm.cause, fix=arm.fix)
            summary = logs(self.json(f'{arm.name}-logs-{host}', host, 'python3 -c ' + shlex.quote(LOG_FILTER) + ' ' +
                                     shlex.quote(ck.STOP_LOG.pattern)), rank, arm.cap, kv=self.expected_kv(arm))
            if rank == 0:
                self.proxy_pid = int(number(cast(list[Json], summary['proxy_pids'])[0]))
        self.ids = [str(r['id']) for r in current]
        self.seen.update(self.ids)
        worker(self.json(f'{arm.name}-worker', 'x86host', 'python3 -c ' + shlex.quote(X86HOST_PROBE)), self.clock())
        (self.out / f'{arm.name}-identity.json').write_text(json.dumps({'ranks': cast(Json, current)}, indent=1) + '\n')
        self.sup.mark_ready()

    def runtime_check(self, rank: int, live: dict[str, Json], arm: Arm) -> None:
        """Experiment rank vs the saved production rank (subclass hook; PF1 arms use runtime() unchanged)."""
        runtime(self.production[rank], live, arm, self.run_name(arm))

    def arm_env(self, arm: Arm) -> str:
        """Extra start_exp.sh environment, ' KEY=value' form (subclass hook; PF1/SP1 add nothing)."""
        return ''

    def expected_kv(self, arm: Arm) -> int:
        """Rank-0 engine pool this arm must log (subclass hook; PF1/SP1: the production reservation at the arm's cap)."""
        return KV[arm.cap]

    def stats(self, label: str) -> dict[str, Json]:
        return self.json(label, 'sparkhead', f'docker exec -i dsv41-serve python3 - --pid {self.proxy_pid}',
                         (ROOT / 'offload/launch/windows/up2_proxy_stats.py').read_text())

    def requests(self, arm: Arm, contexts: tuple[int, ...], reps: int) -> list[dict[str, Json]]:
        before = self.stats(f'{arm.name}-stats-before')
        steps = number(worker(self.json(f'{arm.name}-worker-before', 'x86host', 'python3 -c ' + shlex.quote(X86HOST_PROBE)),
                              self.clock()).get('steps', 0))
        output = self.out / f'{arm.name}-requests.jsonl'
        command = [sys.executable, '-m', 'offload.prefill_trace.window_requests', '--execute', '--cap', str(arm.cap),
                   '--contexts', *map(str, contexts), '--reps', str(reps), '--modes', 'cold', '--warmup',
                   '--trace-run', self.run_name(arm), '--cache-salt-prefix', self.tag,
                   '--deadline-utc', f'{self.submit:.3f}', '--output', str(output), '--data-dir', str(self.data_dir),
                   '--ledger', str(self.out / 'requests-ledger.jsonl')]
        if self.first_output is not None:
            command += ['--reuse-cuts', str(self.first_output)]
        self.call(f'{arm.name}-requests', command, timeout=900 * (len(contexts) * reps + 1) + 600, inflight=1)
        rows = [record(cast(Json, json.loads(x))) for x in output.read_text().splitlines()]
        result = cells(rows, arm, self.run_name(arm), contexts, reps)
        self.first_output = self.first_output or output
        after = self.stats(f'{arm.name}-stats-after')
        ck.counters(after, 0, before)
        ck.require(number(record(after['counters']).get('rounds_ok', 0)) > number(record(before['counters']).get('rounds_ok', 0)),
                   'no remote drafting during the arm')
        status = worker(self.json(f'{arm.name}-worker-after', 'x86host', 'python3 -c ' + shlex.quote(X86HOST_PROBE)), self.clock())
        ck.require(number(status.get('steps', 0)) > steps, 'x86host worker took no steps during the arm')
        with (self.out / 'proxy-stats.jsonl').open('a') as handle:
            handle.write(json.dumps({'arm': arm.name, 'before': before['counters'], 'after': after['counters']}) + '\n')
        if self.match and not arm.trace:
            # after the timed cells (they stay exactly the CW protocol): greedy hashes and the logit floor
            checks = self.out / f'{arm.name}-checks.jsonl'
            self.call(f'{arm.name}-checks', [sys.executable, '-m', 'offload.prefill_cause2.match_requests', '--execute',
                      '--cap', str(arm.cap), '--role', 'fix' if arm.fix else 'off', '--cache-salt-prefix', self.tag,
                      '--deadline-utc', f'{self.submit:.3f}', '--output', str(checks), '--data-dir', str(self.data_dir),
                      '--ledger', str(self.out / 'requests-ledger.jsonl')], timeout=1800, inflight=1)
        return result

    def collect(self, arm: Arm, rows: list[dict[str, Json]]) -> dict[str, Json]:
        checks: dict[str, Json] = {}
        for rank, host in enumerate(HOSTS):
            identity = self.ssh(f'{arm.name}-id-{host}', host, "docker inspect --format '{{.Id}}' dsv41-serve").strip()
            ck.require(identity == self.ids[rank], 'experiment container changed during the arm')
            checks[f'logs_rank{rank}'] = logs(self.json(f'{arm.name}-logs-after-{host}', host, 'python3 -c ' +
                shlex.quote(LOG_FILTER) + ' ' + shlex.quote(ck.STOP_LOG.pattern)), rank, arm.cap)
        if arm.trace:
            run = self.run_name(arm)
            files: dict[int, list[Path]] = {}
            for rank, host in enumerate(HOSTS):
                local = self.out / 'traces' / run / host
                local.parent.mkdir(parents=True, exist_ok=True)
                self.call(f'{arm.name}-trace-copy-{host}', ['scp', '-rq', '-o', 'BatchMode=yes',
                                                             f'{host}:dsv41-offload-run/pf1-trace/{run}', str(local)])
                files[rank] = sorted(local.glob(f'prefill-r{rank}-p*.jsonl'))
            all_rows = [record(cast(Json, json.loads(x))) for x in (self.out / f'{arm.name}-requests.jsonl').read_text().splitlines()]
            checks['trace'] = trace_check(files, all_rows)
            (self.out / f'{arm.name}-trace-check.json').write_text(json.dumps(checks['trace'], indent=1) + '\n')
        if self.match and not arm.trace:
            found = [record(cast(Json, json.loads(x))) for x in (self.out / f'{arm.name}-checks.jsonl').read_text().splitlines()]
            ck.require(bool(found) and found[0].get('type') == 'identity' and found[0].get('role') == ('fix' if arm.fix else 'off')
                       and found[0].get('cap') == arm.cap, 'match check artifact identity mismatch')
            checks['match'] = {kind: [r for r in found if r.get('type') == kind] for kind in ('greedy', 'logit')}
        if arm.cause or self.match:
            ck.require(self.sup.arm_minima is not None and set(self.sup.arm_minima) == set(HOSTS), 'missing arm memory minima')
            checks['mem_available_min_kb'] = cast(Json, dict(cast(dict[str, int], self.sup.arm_minima)))
            self.sup.arm_minima = None
        return checks

    def reap_script(self) -> str:
        """REAP for this window's own HOLD opener and experiment start (subclass hook)."""
        reap = REAP.replace('@TAG@', self.tag)
        if self.cause:
            reap = reap.replace('PF1-window[.]env', 'PF1-cause[.]env')
        if self.match:
            reap = reap.replace('PF1-window[.]env', 'PF1-(window|cause|fix)[.]env')
        return reap

    def restore(self, *, production_stopped: bool) -> dict[str, Json]:
        self.sup.close()
        if self.cause or self.match:
            (self.out / 'arm-memory.json').write_text(json.dumps(self.sup.arm_memory, indent=1) + '\n')
        out: dict[str, Json] = {}
        # S40 FIX 1-2: unsupervised (a STOP file or dead telemetry must not block this): end this window's own remote
        # sessions, let any JIT prebuild finish, then decide from the HOLD actually on disk.
        rc, text = self.plain('reap', [*SSH, 'sparkhead', self.reap_script()], 120)
        out['reap'] = f'rc{rc}: ' + ' | '.join(x for x in text.splitlines() if x.startswith(('reaped', 'killed', 'reap done')))
        for host in HOSTS:
            rc, text = self.plain(f'jit-wait-{host}', [*SSH, host, JIT_WAIT], 360)
            out[f'jit_{host}'] = f'rc{rc}: {text.strip()[-120:]}'
        if not self.hold_written:
            rc, text = self.plain('hold-reread', [*SSH, 'sparkhead', 'cat ~/dsv41-deploy/HOLD 2>/dev/null || true'], 60)
            out['hold_reread'] = f'rc{rc}'
            # Doubt (an unreadable HOLD) goes to W8 check+close, never to "production untouched".
            self.hold_written = rc != 0 or self.tag in text
        if not self.hold_written:
            out['status'] = 'NOT NEEDED: this window wrote no HOLD; production untouched'
            return out
        steps = (('start', 15 * 60),) if production_stopped else ()
        # A hangup or a second Ctrl-C must not abandon production half-restored; each step has its own bound.
        saved = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
        try:
            for step, timeout in (*steps, ('check', 10 * 60), ('close', 120)):
                rc, text = self.plain(step, ['bash', str(RESTORE), step], timeout)
                restored(step, rc, text)
                out[step] = 'PASS'
        finally:
            for sig, handler in saved.items():
                signal.signal(sig, handler)
        out['status'] = 'RESTORED: K1 START VERIFIED, greedy 6/6 IDENTICAL, PRODUCTION HOLD OK, window closed'
        return out


# ---------------------------------------------------------------- entry point

def plan(phase: str, on_contexts: tuple[int, ...], cause_t0: bool = False) -> str:
    lines = ['DRY RUN: no command, network or file. Coordinator Deepseek/coordinator only (--execute).',
             'Preflight (read-only, production serving): restore.sh pin; D2 pause; corpus; production 21bed31/tag/clean, no HOLD;',
             '  experiment ~/dsv41-deploy-offload = reviewed commit, clean (or --stage-bundle); staged pins; :8000 health/K1 327680;',
             '  x86host TP2 worker fb68c52 probe; Spark telemetry probe (nvidia-smi fields, MemAvailable >= 5 GiB); 1 s telemetry both Sparks.',
             'HOLD: ssh sparkhead offload/launch/window.sh open (drain <= 20 min); deadlines from HOLD: submit +75, abort +80, restore by +90 min.',
             'Snapshot production both ranks: identity/mounted hashes/env/argv/Engram, 13-pin verify, image trace bases, KV/log record -> rollback.json.',
             f'Tunnel 127.0.0.1:18001 -> sparkhead 127.0.0.1:8001 (owned). Guards: MemAvailable >= 4 GiB, swap +{SWAP_GROWTH_KB} kB after ready,',
             '  telemetry liveness, foreign demand on :8001, STOP file, deadlines.']
    def arm_lines(arm: Arm, contexts: tuple[int, ...], reps: int) -> None:
        env = f'PF1_CAP={arm.cap} PF1_TRACE={arm.trace} PF1_EAGER={arm.eager}'
        if arm.cause:
            env += f' PF1C_TF_TRANSIENT={arm.transient}'
        lines.append(f'{arm.name}: {env} bash offload/launch/start_exp.sh {arm.config}')
        declared = '/trace sources' if arm.trace else '/fence sources + VLLM_DSV41_MOE_FENCE=1' if arm.fix else ''
        lines.append(f'  identity == production except port/cap/REQUIRED{declared}; KV {KV[arm.cap]}; '
                     f'{14 if arm.fix else 18 if arm.cause else 17 if arm.trace else 13}-pin verify both ranks; proxy/Engram/worker ready')
        lines.append(f'  window_requests --warmup (32000, 64 tokens) then cold {"/".join(map(str, contexts))} x{reps}, 1500 output, fixed salts')
        if arm.trace:
            lines.append('  collect both ranks\' trace files; chunk begin/total completeness; request correlation recorded')
    if phase in ('all', 'off'):
        for arm in OFF_ARMS:
            arm_lines(arm, OFF_CONTEXTS, OFF_REPS)
        lines.append('GATE: REPRODUCED (gap >= 10 % at 150K and 300K, rep spread <= 3 %) -> ON arms'
                     + ('' if phase == 'all' else ' in a later --phase on HOLD') + '; otherwise restore now')
    if phase in ('all', 'on'):
        lines.append('Stage hash-checked trace sources into the experiment checkout (never overwrite different files).')
        for arm in ON_ARMS:
            arm_lines(arm, on_contexts, 1)
    if phase == 'match':
        lines.append('No prior gate (off-524 is this HOLD\'s control); --cause-from: trace+cause pins == window CW\'s mounted bytes.')
        lines.append('Stage trace + cause + fix (offload/prefill_cause2) hash-checked sources (never overwrite different files).')
        for arm in MATCH_ARMS:
            arm_lines(arm, (150000,), MATCH_REPS)
            if not arm.trace:
                lines.append(f'  then match_requests --role {"fix" if arm.fix else "off"}: 6 W5 greedy hashes; 8,192-token '
                             f'top-20 first token x{2 if not arm.fix else 1} (fresh salts); numbers/hashes only')
        lines.append('Every arm: host parity + MemAvailable minima; >=5 GiB before EACH start, >=4 GiB always.')
        lines.append('READING (pre-registered, MATCH.md): CONFIRMED if both fix reps within 5 % of the cause median and '
                     '>=20 % below the off median, cause >=20 % below off, spreads <=3 %, greedy 6/6 and logit floor PASS.')
    if phase == 'cause':
        lines.append('Prior OFF gate required (window 1, ancestor with unchanged execution sources); stage trace + cause hash-checked sources.')
        for arm in (*CAUSE_ARMS, *((CAUSE_T0,) if cause_t0 else ())):
            arm_lines(arm, (150000,), 1)
        lines.append('Cause: per-rank env/mount identity, host parity and MemAvailable minima; >=5 GiB before EACH start, >=4 GiB always.')
    lines.append('ALWAYS (success or any STOP): close owned processes, then bash ' + str(RESTORE) + ' start (if production stopped),')
    lines.append('  check, close; require K1 START VERIFIED, greedy 6/6 IDENTICAL, PRODUCTION HOLD OK, CLOSE confirmed.')
    lines.append('ESTIMATED: full ~67 min HOLD; gate stop ~46 min (offload/prefill_maxlen/PF1-WINDOW-DESIGN.md).')
    if phase == 'cause':
        lines[-1] = 'ESTIMATED: ~27 min HOLD without T0; ~36.5 min with T0 (WINDOW.md, measured window-2 inputs).'
    if phase == 'match':
        lines[-1] = 'ESTIMATED: ~42-43 min HOLD, production down ~42 min (prefill_cause2/MATCH.md, measured W1/W3/CW inputs).'
    return '\n'.join(lines)


# A prior OFF window may come from an ancestor commit only when the descendant changed nothing but this runner and its
# tests (S40 round 3: window 1 ran 56d3e91; the trace-check fix is a runner-only change). Pinned config, launch, trace
# and K1 sources stay identical by construction, and the config hash is still compared below.
RUNNER_ONLY = {'offload/launch/windows/pf1_window.py', 'offload/launch/windows/tests/test_pf1_window.py'}


def same_sources(prior_commit: str, expected: str) -> bool:
    if prior_commit == expected:
        return True
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    ancestor = subprocess.run(['git', 'merge-base', '--is-ancestor', prior_commit, expected], cwd=ROOT, env=env)
    if ancestor.returncode != 0:
        return False
    changed = subprocess.run(['git', 'diff', '--name-only', prior_commit, expected], cwd=ROOT, env=env,
                             capture_output=True, text=True, check=True).stdout.split()
    return set(changed) <= RUNNER_ONLY


# Cause is a new opt-in experiment; these reviewed additions and non-executed analysis/D2 work may differ.
# Everything else (including production config, launchers, K1 and PF1-R2 runtime/build inputs) must be unchanged.
CAUSE_LINEAGE_FILES = RUNNER_ONLY | {
    CAUSE_CONFIG, 'offload/launch/windows/tests/test_pf1_cause_config.py',
    'offload/prefill_trace/emit_verify.py', 'offload/prefill_trace/fit_cost.py', 'offload/prefill_trace/test_window.py',
    'offload/prefill_maxlen/PF1-WINDOW.md',
    # PF1-CAUSE2 (match phase): the fix arm's config and its test; offload/prefill_cause2/ is admitted below. None of it
    # is mounted or sourced by a cause arm.
    FIX_CONFIG, 'offload/launch/windows/tests/test_pf1_fix_config.py', 'offload/launch/windows/tests/test_pf1_match.py',
    # SP1-WINDOW: the profiling arm's config, runner, precheck and their tests; offload/gb10_budget/ is admitted below.
    # A cause arm neither sources nor mounts any of it (it starts PF1-cause.env through the unchanged launchers).
    'offload/launch/configs/SP1-profile.env', 'offload/launch/windows/sp1_window.py', 'offload/launch/windows/sp1_window.sh',
    'offload/launch/windows/sp1_precheck.py', 'offload/launch/windows/sp1_precheck.sh',
    'offload/launch/windows/tests/test_sp1_config.py', 'offload/launch/windows/tests/test_sp1_window.py',
    'offload/launch/windows/tests/test_sp1_precheck.py',
    # PF2-LOWKV: its arm config, runner, precheck, sheet and their tests; offload/pf2_lowkv/ is admitted below. A cause arm
    # neither sources nor mounts any of it (PF2-lowkv.env is started only by pf2_window.py).
    'offload/launch/configs/PF2-lowkv.env', 'offload/launch/windows/pf2_window.py', 'offload/launch/windows/pf2_window.sh',
    'offload/launch/windows/pf2_precheck.py', 'offload/launch/windows/pf2_precheck.sh', 'offload/launch/windows/PF2-LOWKV.md',
    'offload/launch/windows/tests/test_pf2_config.py', 'offload/launch/windows/tests/test_pf2_window.py',
    'offload/launch/windows/tests/test_pf2_rehearsal.py', 'offload/launch/windows/tests/test_pf2_precheck.py',
}


def cause_sources(prior_commit: str, expected: str) -> bool:
    if prior_commit == expected:
        return True
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    ancestor = subprocess.run(['git', 'merge-base', '--is-ancestor', prior_commit, expected], cwd=ROOT, env=env)
    if ancestor.returncode != 0:
        return False
    changed = subprocess.run(['git', 'diff', '--name-only', prior_commit, expected], cwd=ROOT, env=env,
                             capture_output=True, text=True, check=True).stdout.splitlines()
    return all(path in CAUSE_LINEAGE_FILES or path == 'offload/__init__.py' or path.startswith(
        ('documents/', 'offload/prefill_cause/', 'offload/prefill_cause2/', 'offload/dspark/d2/train/', 'offload/dspark/d2/s37/',
         'offload/gb10_budget/', 'offload/pf2_lowkv/', 'offload/launch/maintenance/', 'offload/audit/',
         'offload/dspark/native/', 'offload/kl1/', 'offload/launch/windows/kl1_', 'offload/launch/windows/tests/test_kl1_',
         'offload/operations/'))
    # host maintenance, audit tools, native-drafter code, the KL1 microbenchmark and the Controller-side operations
    # registry/checker: never in a PF1 arm (the empty offload/__init__.py package marker is listed by name below)
        for path in changed)


def prior(directory: Path, expected: str, *, cause: bool = False) -> dict[str, Json]:
    summary = ck.read(directory / 'summary.json')
    ck.require((cause_sources if cause else same_sources)(str(summary.get('experiment_commit')), expected)
               and summary.get('config_sha256') == digest(ROOT / CONFIG), 'prior OFF window ran other sources')
    gated = record(summary['gate'])
    ck.require(gated.get('status') == 'REPRODUCED' and record(summary['restore']).get('close') == 'PASS',
               'prior OFF window did not REPRODUCE the gap or did not restore')
    for name, sha in record(summary['artifact_sha256']).items():
        ck.require(digest(directory / name) == sha, 'prior window evidence changed')
    return gated


def cause_window_pins(directory: Path) -> None:
    """--phase match: the trace + cause overlays to be mounted are byte-for-byte the ones window CW mounted (its saved
    cause-524 identity, both ranks), so cause-524 here repeats CW's code exactly."""
    summary = ck.read(directory / 'summary.json')
    ck.require(summary.get('phase') == 'cause' and summary.get('status') == 'COMPLETE' and
               record(summary['restore']).get('close') == 'PASS', 'cause window not COMPLETE and restored')
    for name, sha in record(summary['artifact_sha256']).items():
        ck.require(digest(directory / name) == sha, 'cause window evidence changed')
    ranks = cast(list[Json], ck.read(directory / 'cause-524-identity.json')['ranks'])
    pins = {DIST + path: sha for path, sha in overlay_pins(True).items()}
    ck.require(len(ranks) == 2 and all({k: record(record(r)['mounted']).get(k) for k in pins} == pins for r in ranks),
               'trace/cause overlay pins differ from the bytes window CW mounted')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    ap.add_argument('--phase', choices=('all', 'off', 'on', 'cause', 'match'), default='all')
    ap.add_argument('--cause-t0', action='store_true', help='--phase cause: add instrumented persistent-scratch 524288 arm')
    ap.add_argument('--on-contexts', type=int, nargs='+', choices=OFF_CONTEXTS, default=[150000])
    ap.add_argument('--gate-from', type=Path, help='--phase on/cause: the prior OFF window directory (REPRODUCED, restored)')
    ap.add_argument('--cause-from', type=Path, help='--phase match: window CW (its mounted trace/cause bytes must equal the pins)')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--remote-repo', default='dsv41-deploy-offload')
    ap.add_argument('--expected-experiment-commit', default='')
    ap.add_argument('--stage-bundle', action='store_true', help='bundle-stage the reviewed commit if the remote checkout differs')
    ap.add_argument('--data-dir', type=Path, default=Path('/opt/dsv41/data/bench-data'))
    args = ap.parse_args(argv)
    on_contexts = tuple(dict.fromkeys(args.on_contexts))
    if args.cause_t0 and args.phase != 'cause':
        ap.error('--cause-t0 requires --phase cause')
    if args.phase in ('cause', 'match') and on_contexts != (150000,):
        ap.error(f'--phase {args.phase} fixes the cold context to 150000')
    if not args.execute:
        print(plan(args.phase, on_contexts, args.cause_t0))
        return 0
    try:
        ck.require(os.environ.get('WINDOW_COORDINATOR') == 'Deepseek/coordinator', 'WINDOW_COORDINATOR=Deepseek/coordinator required')
        ck.require(args.out is not None and args.out.resolve().parent == CAMPAIGN and not args.out.exists(),
                   'fresh --out directly inside the campaign directory required')
        ck.require(re.fullmatch('[0-9a-f]{40}', args.expected_experiment_commit) is not None, 'full reviewed commit required')
        ck.require(re.fullmatch('[A-Za-z0-9_.-]+', args.remote_repo) is not None and args.remote_repo != 'dsv41-deploy',
                   'experiment checkout must be a separate relative checkout, never ~/dsv41-deploy')
        local = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        ck.require(local == args.expected_experiment_commit, 'local checkout is not the reviewed commit')
        ck.require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip(),
                   'local tracked source is dirty')
        ck.require((args.phase in ('on', 'cause')) == (args.gate_from is not None), '--gate-from is required with, and only with, --phase on/cause')
        gated = prior(args.gate_from, args.expected_experiment_commit, cause=args.phase == 'cause') if args.phase in ('on', 'cause') else None
        ck.require((args.phase == 'match') == (args.cause_from is not None), '--cause-from is required with, and only with, --phase match')
        if args.phase == 'match':
            cause_window_pins(cast(Path, args.cause_from))
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
    def checkpoint(data: dict[str, Json]) -> None:
        with (out / 'checkpoints.jsonl').open('a') as handle:
            handle.write(json.dumps({'time': time.time(), **data}, allow_nan=False) + '\n')
    sup = Supervisor(out)
    live = Live(out, sup, remote=args.remote_repo, expected=args.expected_experiment_commit, data_dir=args.data_dir,
                tag=tag, stage_bundle=args.stage_bundle, cause=args.phase == 'cause', match=args.phase == 'match')
    result = window(live, args.phase, checkpoint, on_contexts=on_contexts, cause_t0=args.cause_t0, prior_gate=gated)
    sup.close()
    if args.phase in ('cause', 'match'):
        result['arm_memory_min_kb'] = cast(Json, live.sup.arm_memory)
    if args.phase == 'cause':
        result['prior_off_config_sha256'] = digest(ROOT / CONFIG)
    if args.phase == 'match':
        result['arm_config_sha256'] = {arm.name: digest(ROOT / arm.config) for arm in MATCH_ARMS}
        result['fix_overlay_sha256'] = cast(Json, overlay_pins(fix=True))
    result.update({'experiment_commit': args.expected_experiment_commit, 'config_sha256': digest(ROOT / (CAUSE_CONFIG if args.phase == 'cause' else FIX_CONFIG if args.phase == 'match' else CONFIG)),
                   'restore_sha256': RESTORE_SHA256, 'worker_source': WORKER,
                   'artifact_sha256': {p.name: digest(p) for p in sorted(out.iterdir())
                                       if p.is_file() and p.suffix in ('.json', '.jsonl') and not p.name.startswith('telemetry-')}})
    (out / 'summary.json').write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    restore = record(result['restore'])
    print(json.dumps({'status': result['status'], 'stop': result.get('stop'), 'gate': record(result.get('gate') or {}).get('status'),
                      'restore': restore.get('status')}))
    if str(restore.get('status', '')).startswith('RESTORED') or str(restore.get('status', '')).startswith('NOT NEEDED'):
        return 0 if 'stop' not in result else 1
    print('PRODUCTION NOT VERIFIED; HOLD left open. Coordinator: inspect cmd/restore-*.log, then run\n'
          + '\n'.join(f'  bash {RESTORE} {s}' for s in ('start', 'check', 'close')), file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
