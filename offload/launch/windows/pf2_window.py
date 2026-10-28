"""PF2-LOWKV coordinator runner: HOLD, stage, four alternating 524,288 starts (H = 6 GiB KV, L = the checked LOW
reservation; H-L-H-L), per start two unprofiled cold 150K + one mid-prompt profiled 150K, one 480K needle on the last L,
bounded trace collection, ALWAYS the W8 restore.

Dry plan by default (no command, network or file). Only Deepseek / coordinator runs --execute, after review and staging.
Records numbers and hashes only: no prompt or generated text is saved. Reuses the reviewed SP1 runner (SP1Live: preflight
with the profile-directory and disk checks, trace listing/caps/copy/O_DIRECT hash/removal, restore and directory cleanup) and
through it PF1's (Supervisor, Live preflight/HOLD/snapshot/start/restore, runtime(), restored()). See
offload/pf2_lowkv/DESIGN.md (design, pre-registered reading) and offload/launch/windows/PF2-LOWKV.md (coordinator sheet).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Protocol, cast

from offload.launch.windows import pf1_window as pf
from offload.launch.windows import sp1_window as s
from offload.launch.windows import up2_checks as ck
from offload.launch.windows.up2_low import X86HOST_PROBE, terminate
from offload.pf2_lowkv import arith
from offload.pf2_lowkv import cells as pc
from offload.prefill_maxlen.evidence import Json, number, record
from offload.prefill_trace.telemetry import FIELDS

CONFIG = 'offload/launch/configs/PF2-lowkv.env'
PIN_FILES = ('offload/launch/configs/PF1-window.env', CONFIG, 'offload/launch/start_exp.sh', 'offload/launch/serve_exp.sh',
             'offload/launch/dsv41_tp2_offload.sh', 'offload/kvpack/MANIFEST.json')
CAP = arith.CAP
SUBMIT_S, ABORT_S = 62 * 60, 66 * 60            # from the HOLD timestamp (DESIGN.md section 7)
START_BLOCK_S = 14.5 * 60                       # restart 6.5 + one slow start's cells, MEASURED-based 14.2-14.5 min (S44 note 3)
ORDER = ('H', 'L', 'H', 'L')
KERNEL = re.compile(r'\d+\.\d+\.\d+-\d+-nvidia[A-Za-z0-9_.-]*')
DRIVER = re.compile(r'\d{3}\.\d+(?:\.\d+)?')
VMSTAT = ('compact_stall', 'compact_fail', 'compact_success', 'compact_migrate_scanned', 'compact_free_scanned',
          'pgscan_kswapd', 'pgscan_direct', 'pgsteal_kswapd', 'pgsteal_direct', 'thp_fault_alloc', 'thp_fault_fallback',
          'thp_collapse_alloc', 'pgmajfault', 'pgfault', 'pswpin', 'pswpout')
MEMINFO = ('MemAvailable', 'MemFree', 'Cached', 'SwapFree', 'SwapTotal', 'AnonHugePages', 'Shmem')
# PF1's 1 s telemetry plus meminfo detail, /proc/vmstat reclaim/compaction/THP/fault counters and PSI, in the SAME
# 'memory' row (PF1's Supervisor guards MemAvailable/SwapFree from it and counts any other kind as a GPU error).
TELEMETRY = r'''
import json, subprocess, sys, time
fields, samples, n = sys.argv[1], int(sys.argv[2]), 0
MEM = set(sys.argv[3].split(','))
VM = set(sys.argv[4].split(','))
while samples == 0 or n < samples:
    started = time.monotonic()
    mem, vm, psi = {}, {}, {}
    for line in open('/proc/meminfo'):
        key, _, value = line.partition(':')
        if key in MEM:
            mem[key] = int(value.split()[0])
    for line in open('/proc/vmstat'):
        key, value = line.split()
        if key in VM or key.startswith('allocstall_'):
            vm[key] = int(value)
    try:
        for line in open('/proc/pressure/memory'):
            kind, *parts = line.split()
            psi[kind] = {k: float(v) for k, v in (p.split('=') for p in parts)}
    except OSError:
        psi = None
    print(json.dumps({'kind': 'memory', 'utc': time.time(), 'kb': mem, 'vmstat': vm, 'psi': psi}), flush=True)
    try:
        csv = subprocess.run(['nvidia-smi', '--query-gpu=' + fields, '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        print(json.dumps({'kind': 'gpu', 'utc': time.time(), 'csv': csv}), flush=True)
    except Exception as exc:
        print(json.dumps({'kind': 'gpu_error', 'utc': time.time(), 'error': type(exc).__name__}), flush=True)
    n += 1
    time.sleep(max(0, 1 - (time.monotonic() - started)))
'''
# In the experiment container, 1 s: smaps_rollup + status of the API server, engine core and workers (numbers only; no
# command line is printed). smaps_rollup walks the mappings under the mmap lock: its read time is recorded per process.
PROCMEM = r'''
import json, os, time
ROLL = ('Rss', 'Pss', 'Pss_Anon', 'Pss_File', 'Pss_Shmem', 'Swap')
STAT = ('VmRSS', 'RssAnon', 'RssFile', 'VmHWM')
def role(cmd):
    if cmd.startswith('VLLM::EngineCore'): return 'engine_core'
    if cmd.startswith('VLLM::Worker'): return 'worker'
    if cmd.startswith('VLLM::APIServer') or 'vllm serve ' in cmd: return 'api_server'
    return None
while True:
    started = time.monotonic()
    procs = []
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            cmd = open('/proc/%s/cmdline' % pid, 'rb').read().replace(b'\0', b' ').decode(errors='replace')
        except OSError:
            continue
        kind = role(cmd)
        if kind is None:
            continue
        row = {'pid': int(pid), 'role': kind}
        try:
            t = time.monotonic()
            for line in open('/proc/%s/smaps_rollup' % pid):
                key, _, value = line.partition(':')
                if key in ROLL:
                    row[key] = int(value.split()[0])
            row['rollup_ms'] = round((time.monotonic() - t) * 1000, 3)
            for line in open('/proc/%s/status' % pid):
                key, _, value = line.partition(':')
                if key in STAT:
                    row[key] = int(value.split()[0])
        except OSError:
            row['error'] = 'gone'
        procs.append(row)
    print(json.dumps({'kind': 'procmem', 'utc': time.time(), 'procs': procs}), flush=True)
    time.sleep(max(0, 1 - (time.monotonic() - started)))
'''


def telemetry_command(samples: int) -> str:
    """Remote command for `samples` extended telemetry rows (0 = until terminated)."""
    return (f'exec python3 -u -c {shlex.quote(TELEMETRY)} {shlex.quote(FIELDS)} {samples} '
            f'{",".join(MEMINFO)} {",".join(VMSTAT)}')


def monitor(host: str) -> list[str]:
    return [*pf.SSH, host, telemetry_command(0)]


@dataclass(frozen=True)
class PF2Arm(pf.Arm):
    """PF1's OFF arm at 524,288 through PF2-lowkv.env with an explicit KV reservation."""
    kv: int = arith.H_KV
    role: str = 'H'

    @property
    def config(self) -> str:
        return CONFIG


def arms(low: int) -> tuple[PF2Arm, ...]:
    arith.check_low(low)
    return tuple(PF2Arm(f's{i}-{r}', CAP, 0, kv=arith.H_KV if r == 'H' else low, role=r) for i, r in enumerate(ORDER, 1))


@dataclass(frozen=True)
class Platform:
    kernel: dict[str, str] = field(default_factory=dict)
    driver: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        ck.require(set(self.kernel) == set(self.driver) == set(pf.HOSTS), 'kernel and driver required for both Sparks')
        for host in pf.HOSTS:
            ck.require(KERNEL.fullmatch(self.kernel[host]) is not None, f'{host} kernel parameter is not a kernel release')
            ck.require(DRIVER.fullmatch(self.driver[host]) is not None, f'{host} driver parameter is not a driver version')


# ---------------------------------------------------------------- pure checks (CPU-tested)

def strip_kv(argv: str, kv: int) -> str:
    """The arm's argv with exactly one --kv-cache-memory-bytes equal to `kv`, normalised to production's H value."""
    tokens = shlex.split(argv.replace('\\\n', ' '))
    ck.require(tokens.count('--kv-cache-memory-bytes') == 1, 'experiment argv lacks exactly one --kv-cache-memory-bytes')
    i = tokens.index('--kv-cache-memory-bytes')
    ck.require(i + 1 < len(tokens) and tokens[i + 1] == str(kv), f'experiment KV reservation is not the arm\'s {kv}')
    tokens[i + 1] = str(arith.H_KV)
    return shlex.join(tokens)


def host_identity(text: str, host: str, platform: Platform) -> dict[str, Json]:
    """pf.HOST_PARITY output: uname -r, driver, THP enabled, THP defrag, buddyinfo, compact_*, MemAvailable."""
    lines = [x.strip() for x in text.strip().splitlines()]
    ck.require(len(lines) >= 4, f'{host} host identity output too short')
    ck.require(lines[0] == platform.kernel[host], f'{host} kernel {lines[0]!r} is not the reviewed {platform.kernel[host]!r}')
    ck.require(lines[1] == platform.driver[host], f'{host} driver {lines[1]!r} is not the reviewed {platform.driver[host]!r}')
    available = re.findall(r'^MemAvailable:\s+(\d+) kB$', text, re.M)
    ck.require(len(available) == 1, f'{host} missing MemAvailable')
    compact = {m[1]: int(m[2]) for m in re.finditer(r'^(compact_\w+) (\d+)$', text, re.M)}
    return {'kernel': lines[0], 'driver': lines[1], 'thp_enabled': lines[2], 'thp_defrag': lines[3],
            'mem_available_kb': int(available[0]), 'compact': cast(Json, compact)}


def telemetry_probe(rows: list[dict[str, Json]]) -> str:
    """One extended sample: the guard fields, every vmstat counter and PSI must be readable before the HOLD."""
    ck.require([r.get('kind') for r in rows] == ['memory', 'gpu'], 'extended telemetry probe shape differs')
    pf.memory_row(rows[0])
    pf.gpu_row(rows[1])
    vm = record(rows[0]['vmstat'])
    missing = [k for k in VMSTAT if k not in vm] + ([] if any(k.startswith('allocstall_') for k in vm) else ['allocstall_*'])
    ck.require(not missing, f'vmstat lacks {missing}')
    psi = rows[0].get('psi')
    ck.require(isinstance(psi, dict) and {'some', 'full'} <= set(psi), '/proc/pressure/memory unreadable (PSI off?)')
    return f"vmstat {len(vm)} counters, PSI some/full, MemAvailable {number(record(rows[0]['kb'])['MemAvailable']) / 2**20:.1f} GiB"


# ---------------------------------------------------------------- live backend (coordinator only)

class PF2Live(s.SP1Live):
    def __init__(self, out: Path, sup: pf.Supervisor, *, remote: str, expected: str, data_dir: Path, tag: str,
                 stage_bundle: bool, low: int, platform: Platform, call: pf.Call | None = None,
                 plain: Callable[[str, list[str], float], tuple[int, str]] | None = None,
                 clock: Callable[[], float] = time.time,
                 spawn: Callable[[list[str], IO[str], IO[str]], subprocess.Popen[str]] | None = None) -> None:
        super().__init__(out, sup, remote=remote, expected=expected, data_dir=data_dir, tag=tag, stage_bundle=stage_bundle,
                         call=call, plain=plain, clock=clock)
        self.low, self.platform = low, platform
        self.spawn = spawn or self._spawn
        self.samplers: dict[str, tuple[subprocess.Popen[str], Path, list[IO[str]]]] = {}
        self.sampler_gaps: dict[str, list[str]] = {}
        self.hosts: dict[str, dict[str, Json]] = {}
        self.arm: PF2Arm | None = None

    @property
    def pin_files(self) -> tuple[str, ...]:
        return PIN_FILES

    def hold_purpose(self) -> str:
        return (f'PF2-LOWKV {self.tag}: 524288 H/L KV reservation A/B (6 vs {self.low / 2**30:g} GiB/rank), SP1 profiler; '
                'no attention/K1/cap/clock change')

    def reap_script(self) -> str:
        return (pf.REAP.replace('PF1-WINDOW', 'PF2-LOWKV').replace('PF1-window[.]env', 'PF2-lowkv[.]env')
                .replace('@TAG@', self.tag))

    def arm_env(self, arm: pf.Arm) -> str:
        return f' PF2_KV_BYTES={cast(PF2Arm, arm).kv}'

    def expected_kv(self, arm: pf.Arm) -> int:
        return arith.expected_tokens(cast(PF2Arm, arm).kv)

    def runtime_check(self, rank: int, live: dict[str, Json], arm: pf.Arm) -> None:
        argv = strip_kv(s.strip_profiler(str(live['argv'])), cast(PF2Arm, arm).kv)
        pf.runtime(self.production[rank], {**live, 'argv': argv}, arm, self.run_name(arm))

    # -- before the HOLD (read-only, production serving)
    def preflight(self) -> None:
        super().preflight()          # PF1 preflight + SP1 profile directory absent and home disk
        for host in pf.HOSTS:
            self.hosts[host] = host_identity(self.ssh(f'host-identity-{host}', host, pf.HOST_PARITY), host, self.platform)
            rows = [record(cast(Json, json.loads(x))) for x in self.ssh(
                f'telemetry-extended-{host}', host, telemetry_command(1)).splitlines() if x.startswith('{')]
            telemetry_probe(rows)
        (self.out / 'hosts-preflight.json').write_text(json.dumps(self.hosts, indent=1) + '\n')

    def open_hold(self) -> float:
        opened = pf.Live.open_hold(self)        # not SP1's 40/45-minute bounds
        self.submit, self.sup.abort_at = opened + SUBMIT_S, opened + ABORT_S
        self.sup.bound = f'{ABORT_S // 60}-minute PF2 abort bound'
        (self.out / 'deadlines.json').write_text(json.dumps({'opened': opened, 'submission': self.submit,
            'abort': self.sup.abort_at, 'restore_by': opened + ABORT_S + 8 * 60}) + '\n')
        return opened

    def fits_start(self) -> bool:
        return self.clock() + START_BLOCK_S <= self.submit

    # -- per-process memory sampler (owned; never a guard: a gap is recorded, the MemAvailable guard stays strict)
    @staticmethod
    def _spawn(command: list[str], out: IO[str], err: IO[str]) -> subprocess.Popen[str]:
        return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=out, stderr=err, text=True, start_new_session=True)

    def start_samplers(self, name: str) -> None:
        for host in pf.HOSTS:
            path = self.out / f'procmem-{name}-{host}.jsonl'
            out, err = path.open('x'), path.with_suffix('.stderr').open('x')
            proc = self.spawn([*pf.SSH, host, 'docker exec -i dsv41-serve python3 -u -'], out, err)
            if proc.stdin is not None:
                proc.stdin.write(PROCMEM)
                proc.stdin.close()
            self.samplers[host] = (proc, path, [out, err])
        self.sampler_gaps[name] = []

    def check_samplers(self, name: str, label: str) -> None:
        now = time.time()
        for host, (proc, path, _) in self.samplers.items():
            if proc.poll() is not None or not path.exists() or now - path.stat().st_mtime > 5:
                self.sampler_gaps.setdefault(name, []).append(f'{label}:{host}')

    def stop_samplers(self) -> None:
        for proc, _, files in self.samplers.values():
            terminate(proc)
            for handle in files:
                handle.close()
        self.samplers = {}

    # -- one start
    def host_check(self, arm: PF2Arm) -> dict[str, Json]:
        found: dict[str, Json] = {}
        for host in pf.HOSTS:
            row = host_identity(self.ssh(f'{arm.name}-host-{host}', host, pf.HOST_PARITY), host, self.platform)
            ck.require(row['thp_enabled'] == record(self.hosts[host])['thp_enabled'] and
                       row['thp_defrag'] == record(self.hosts[host])['thp_defrag'], f'{host} THP setting changed during the window')
            ck.require(int(number(row['mem_available_kb'])) >= pf.START_FLOOR_KB, f'{host} MemAvailable below 5 GiB before {arm.name}')
            found[host] = row
        return found

    def begin_start(self, arm: PF2Arm, first: bool) -> dict[str, Json]:
        self.stop_samplers()
        hosts = self.host_check(arm)
        for host in pf.HOSTS:
            if first:
                self.ssh(f'{arm.name}-profile-dir-{host}', host, f'mkdir -p ~/dsv41-offload-run && mkdir ~/{s.PROFILE_DIR}')
                self.created.add(host)
            files, free = s.listing(self.ssh(f'{arm.name}-profile-empty-before-{host}', host, self.find()))
            ck.require(not files and free >= s.DISK_FLOOR_KB, f'{host} profile directory not empty or disk low before {arm.name}')
        self.sup.arm_minima = {}
        self.sup.arm_memory[arm.name] = self.sup.arm_minima
        self.arm = arm
        pf.Live.start(self, arm)    # HOLD, start_exp.sh + PF2_KV_BYTES, /v1/models 524288, identity, 13 pins, KV log, worker
        for host in pf.HOSTS:
            mounts = ck.rows(json.loads(self.ssh(f'{arm.name}-mounts-{host}', host,
                                                 "docker inspect --format '{{json .Mounts}}' dsv41-serve")))
            s.trace_mount(mounts)
            ck.require(self.log_hits(host, s.PROFILER_STARTED) == 0, f'{host} profiler already started')
        ck.require(self.log_hits('sparkhead', s.PROFILER_ON) >= 1, 'API server did not register the profiler endpoints')
        self.start_samplers(arm.name)
        return {'hosts': cast(Json, hosts), 'kv_bytes': arm.kv, 'kv_tokens': self.expected_kv(arm), 'role': arm.role}

    def driver(self, *extra: str) -> list[str]:
        return [sys.executable, '-m', 'offload.pf2_lowkv.cells', '--execute', '--data-dir', str(self.data_dir), *extra]

    def probes(self, label: str) -> tuple[dict[str, Json], float]:
        before = self.stats(f'{label}-stats-before')
        steps = number(pf.worker(self.json(f'{label}-worker-before', 'x86host', 'python3 -c ' + shlex.quote(X86HOST_PROBE)),
                                 self.clock()).get('steps', 0))
        return before, steps

    def after(self, label: str, before: dict[str, Json], steps: float) -> None:
        after = self.stats(f'{label}-stats-after')
        ck.counters(after, 0, before)
        ck.require(number(record(after['counters']).get('rounds_ok', 0)) > number(record(before['counters']).get('rounds_ok', 0)),
                   f'{label}: no remote drafting')
        status = pf.worker(self.json(f'{label}-worker-after', 'x86host', 'python3 -c ' + shlex.quote(X86HOST_PROBE)), self.clock())
        ck.require(number(status.get('steps', 0)) > steps, f'{label}: x86host worker took no steps')
        with (self.out / 'proxy-stats.jsonl').open('a') as handle:
            handle.write(json.dumps({'cell': label, 'before': before['counters'], 'after': after['counters']}) + '\n')

    def run_cell(self, arm: PF2Arm, name: str) -> dict[str, Json]:
        cell = pc.CELLS[name]
        label = f'{arm.name}-{name}'
        ck.require(self.clock() + cell.estimate_s <= self.submit, f'{label}: does not fit before the submission bound')
        if cell.captures:
            ck.require(not any(self.log_hits(host, s.PROFILER_STARTED) for host in pf.HOSTS),
                       'a profiler started before the profiled cell')
        self.check_samplers(arm.name, label)
        before, steps = self.probes(label)
        output = self.out / 'cells' / f'{label}.json'
        output.parent.mkdir(exist_ok=True)
        self.call(f'{label}-cell', self.driver(
            'run', '--cell', name, '--cuts', str(self.cuts), '--output', str(output), '--deadline-utc', f'{self.submit:.3f}',
            '--telemetry', *(str(self.out / f'telemetry-{host}.jsonl') for host in pf.HOSTS),
            '--capture-floor-kb', str(s.CAPTURE_FLOOR_KB), '--cache-salt-prefix', self.tag,
            '--ledger', str(self.out / 'requests-ledger.jsonl')), timeout=3 * cell.estimate_s + 600, inflight=cell.batch)
        result = ck.read(output)
        ck.require(result.get('cell') == name and len(cast(list[Json], result['streams'])) == cell.batch,
                   'cell artifact identity mismatch')
        self.after(label, before, steps)
        traces = self.collect_traces(label, result) if cell.captures else []
        return {'captures': [{k: c.get(k) for k in ('kind', 'status')} for c in map(record, cast(list[Json], result['captures']))],
                'traces': cast(Json, traces), 'file': str(output.relative_to(self.out))}

    def cell(self, name: str) -> dict[str, Json]:          # SP1 Backend method; PF2 cells always know their start
        ck.require(self.arm is not None, 'no PF2 start is running')
        return self.run_cell(cast(PF2Arm, self.arm), name)

    def needle(self, arm: PF2Arm) -> dict[str, Json]:
        label = f'{arm.name}-needle-480k'
        if self.clock() + pc.NEEDLE_ESTIMATE_S > self.submit:
            return {'status': 'SKIPPED_TIME'}
        self.check_samplers(arm.name, label)
        before, steps = self.probes(label)
        output = self.out / 'cells' / f'{label}.json'
        self.call(label, self.driver('needle', '--output', str(output), '--deadline-utc', f'{self.submit:.3f}',
                                     '--cache-salt-prefix', self.tag, '--ledger', str(self.out / 'requests-ledger.jsonl')),
                  timeout=3 * pc.NEEDLE_ESTIMATE_S + 600, inflight=1)
        result = ck.read(output)
        ck.require(result.get('type') == 'needle', 'needle artifact identity mismatch')
        if result.get('status') == 'RAN':
            ck.require(result.get('cached_tokens') in (0, None) or number(result['cached_tokens']) == 0, 'needle hit the cache')
            self.after(label, before, steps)
        return {**{k: result.get(k) for k in ('status', 'recall', 'of', 'recalled_all', 'prompt_tokens', 'elapsed_s')},
                'file': str(output.relative_to(self.out))}

    def finish_start(self, arm: PF2Arm) -> dict[str, Json]:
        checks: dict[str, Json] = {}
        for rank, host in enumerate(pf.HOSTS):
            identity = self.ssh(f'{arm.name}-final-id-{host}', host, "docker inspect --format '{{.Id}}' dsv41-serve").strip()
            ck.require(identity == self.ids[rank], 'experiment container changed during the start')
            checks[f'logs_rank{rank}'] = pf.logs(self.json(f'{arm.name}-final-logs-{host}', host, 'python3 -c ' +
                shlex.quote(pf.LOG_FILTER) + ' ' + shlex.quote(ck.STOP_LOG.pattern)), rank, CAP, kv=self.expected_kv(arm))
            files, _ = s.listing(self.ssh(f'{arm.name}-final-list-{host}', host, self.find()))
            ck.require(not files, f'{host} profile directory still holds files after collection')
        minima = self.sup.arm_minima
        ck.require(minima is not None and set(minima) == set(pf.HOSTS), 'missing memory minima')
        checks['mem_available_min_kb'] = cast(Json, dict(cast(dict[str, int], minima)))
        checks['procmem_gaps'] = cast(Json, self.sampler_gaps.get(arm.name, []))
        return checks

    def finish(self) -> dict[str, Json]:
        self.stop_samplers()
        for host in pf.HOSTS:
            if host in self.created:
                self.ssh(f'final-rmdir-{host}', host, f'rmdir ~/{s.PROFILE_DIR}')
                self.created.discard(host)
        pf.worker(self.json('final-worker', 'x86host', 'python3 -c ' + shlex.quote(X86HOST_PROBE)), self.clock())
        return {'profile_dirs_removed': True}

    def build(self) -> dict[str, Json]:
        self.call('prompts-build', self.driver('build', '--output', str(self.cuts)), timeout=600)
        cuts = ck.read(self.cuts)
        return {key: record(v).get('tokens') for key, v in record(cuts['prompts']).items()}

    def restore(self, *, production_stopped: bool) -> dict[str, Json]:
        self.stop_samplers()
        return super().restore(production_stopped=production_stopped)


# ---------------------------------------------------------------- orchestration (shared by PF2Live and fakes)

class Backend(Protocol):
    def preflight(self) -> None: ...
    def open_hold(self) -> float: ...
    def snapshot(self) -> None: ...
    def fits_start(self) -> bool: ...
    def begin_start(self, arm: PF2Arm, first: bool) -> dict[str, Json]: ...
    def build(self) -> dict[str, Json]: ...
    def run_cell(self, arm: PF2Arm, name: str) -> dict[str, Json]: ...
    def needle(self, arm: PF2Arm) -> dict[str, Json]: ...
    def finish_start(self, arm: PF2Arm) -> dict[str, Json]: ...
    def finish(self) -> dict[str, Json]: ...
    def restore(self, *, production_stopped: bool) -> dict[str, Json]: ...


def window(b: Backend, plan: tuple[PF2Arm, ...], checkpoint: Callable[[dict[str, Json]], None]) -> dict[str, Json]:
    """Whatever happens after the HOLD may exist, the W8 restore runs; a failed restore is never hidden."""
    result: dict[str, Json] = {'status': 'INCOMPLETE', 'starts': {}}
    starts = cast(dict[str, Json], result['starts'])
    hold = stopped = False
    last_low = max(i for i, a in enumerate(plan) if a.role == 'L')
    try:
        b.preflight()
        hold = True    # from here a HOLD may exist (open can time out after writing it): always restore/close
        result['hold_opened'] = b.open_hold()
        b.snapshot()
        for i, arm in enumerate(plan):
            if i and not b.fits_start():
                for later in plan[i:]:
                    starts[later.name] = {'status': 'SKIPPED_TIME', 'role': later.role, 'kv_bytes': later.kv}
                checkpoint({'skipped_from': arm.name})
                break
            stopped = True    # start_exp.sh stops production (or the previous arm) before anything else
            row: dict[str, Json] = {'status': 'STARTED', 'position': i + 1, **b.begin_start(arm, i == 0)}
            starts[arm.name] = row
            if i == 0:
                result['prompts'] = b.build()
                checkpoint({'prompts': result['prompts']})
            done: dict[str, Json] = {}
            row['cells'] = done
            for name in pc.ORDER:
                done[name] = b.run_cell(arm, name)
                checkpoint({'start': arm.name, 'cell': name})
            row['checks'] = b.finish_start(arm)
            row['status'] = 'COMPLETE'     # classified before the needle: a needle STOP must not discard this start (S44 FIX 1)
            if i == last_low:
                done['needle-480k'] = b.needle(arm)
                checkpoint({'start': arm.name, 'cell': 'needle-480k'})
                row['needle_checks'] = b.finish_start(arm)     # finish_start only reads; safe to call twice
        result['finish'] = b.finish()
        result['status'] = 'COMPLETE'
    except (Exception, KeyboardInterrupt) as exc:
        result['stop'] = f'{type(exc).__name__}: {exc}'
    finally:
        # Ignore hangups/interrupts BEFORE the backend closes its supervisor (PF1 S40 FIX 3).
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


# ---------------------------------------------------------------- entry point

def plan_text(low: int) -> str:
    plan = arms(low)
    lines = ['DRY RUN: no command, network or file. Coordinator Deepseek/coordinator only (--execute).',
             'Preflight (read-only, production serving): PF1 preflight (restore.sh pin; D2 pause; corpus; production 21bed31/tag,',
             f'  clean, no HOLD; ~/dsv41-deploy-offload = reviewed commit or --stage-bundle; {len(PIN_FILES)} staged pins incl. {CONFIG};',
             '  :8000 K1 327680; x86host TP2 worker fb68c52 read-only probe; MemAvailable >= 5 GiB); SP1: ~/dsv41-offload-run/sp1-profile',
             '  absent and >= 10 GiB free home disk; host identity: uname -r and driver == --sparkhead-/--sparkworker-kernel/-driver;',
             '  extended telemetry probe (vmstat counters, PSI); 1 s telemetry both Sparks for the whole window.',
             f'HOLD: window.sh open "PF2-LOWKV <tag>: ..." (drain <= 20 min); deadlines from HOLD: submit +{SUBMIT_S // 60},'
             f' abort +{ABORT_S // 60}, restore by about +{ABORT_S // 60 + 8} min.',
             'Snapshot production both ranks (identity, mounted hashes, env, argv, Engram, 13 pins, image bases) -> rollback.json.',
             'Tunnel 127.0.0.1:18001 -> sparkhead 127.0.0.1:8001 (owned). Guards: MemAvailable >= 4 GiB always (incl. trace flush),',
             '  swap +256 MiB after ready, telemetry liveness, foreign demand on :8001, STOP file, deadlines.']
    for i, arm in enumerate(plan):
        lines.append(f'{arm.name}: {"" if i == 0 else f"if now + {START_BLOCK_S / 60:g} min <= submit: "}host identity + THP unchanged,'
                     f' MemAvailable >= 5 GiB; PF1_CAP={CAP} PF1_TRACE=0 PF1_EAGER=0 PF2_KV_BYTES={arm.kv}'
                     f' bash offload/launch/start_exp.sh {CONFIG}')
        lines.append(f'  identity == production except port/cap/REQUIRED, KV {arm.kv} and one reviewed --profiler-config;'
                     f' engine KV {arith.expected_tokens(arm.kv):,} tokens; 13-pin verify; /sp1-profile rw; profiler never started;'
                     ' in-container PSS/RSS sampler both ranks')
        if i == 0:
            lines.append('  prompts: cells build (32K warm-up + 150K public prompts, <= 4 /tokenize each; numeric cuts + sha256 only)')
        lines.append('  warmup 32K x64 (discarded); c150-a, c150-b cold 150K x1500 (unprofiled: fast/slow + decode sample);')
        lines.append(f'  p150 cold 150K with ONE mid-prompt capture at +{pc.MID_DELAY_S:g} s for 5 s on both ranks (floor 4.5 GiB);'
                     ' collect + O_DIRECT sha256 + remove; proxy/spec/worker counters per cell')
        if arm.role == 'L' and i == len(plan) - 1:
            lines.append(f'  needle-480k: {pc.NEEDLES} needles, {pc.NEEDLE_CTX:,} tokens, T=0, <= {pc.MAX_NEEDLE_COUNTS} /tokenize;'
                         ' recall/foreign counts + sha256 only; SKIPPED_TIME if it cannot end before submit')
        lines.append('  finish: container unchanged, clean logs + KV, profile directory empty, memory minima, sampler gaps')
    lines += ['ALWAYS (success or any STOP): stop samplers, close owned processes, then bash ' + str(pf.RESTORE) + ' start (if production',
              '  stopped), check, close; require K1 START VERIFIED, six "greedy K1-post vs restored: i IDENTICAL n", PRODUCTION HOLD OK,',
              '  CHECK: DONE, CLOSE: window closed. Then remove only the empty profile directories this window created.',
              'Analysis afterwards (Controller, CPU): python3 -m offload.pf2_lowkv.analyze --window <out> [--reference <SP1 rerun dir>].',
              'ESTIMATED: ~62.6-68.5 min HOLD, production down ~61.8-67.7 min (offload/pf2_lowkv/DESIGN.md section 8).']
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--remote-repo', default='dsv41-deploy-offload')
    ap.add_argument('--expected-experiment-commit', default='')
    ap.add_argument('--stage-bundle', action='store_true', help='bundle-stage the reviewed commit if the remote checkout differs')
    ap.add_argument('--data-dir', type=Path, default=Path('/opt/dsv41/data/bench-data'))
    ap.add_argument('--low-kv-bytes', type=int, default=arith.LOW_KV, help='L reservation per rank (arith.check_low)')
    for host, flag in (('sparkhead', 'sparkhead'), ('sparkworker-wired', 'sparkworker')):
        ap.add_argument(f'--{flag}-kernel', default='', help=f'{host} `uname -r` the window must find (no default)')
        ap.add_argument(f'--{flag}-driver', default='', help=f'{host} nvidia driver version the window must find (no default)')
    args = ap.parse_args(argv)
    try:
        arith.check_low(args.low_kv_bytes)
    except ValueError as exc:
        ap.error(str(exc))
    if not args.execute:
        print(plan_text(args.low_kv_bytes))
        return 0
    platform = Platform({'sparkhead': args.sparkhead_kernel, 'sparkworker-wired': args.sparkworker_kernel},
                        {'sparkhead': args.sparkhead_driver, 'sparkworker-wired': args.sparkworker_driver})
    try:
        platform.validate()
        ck.require(os.environ.get('WINDOW_COORDINATOR') == 'Deepseek/coordinator', 'WINDOW_COORDINATOR=Deepseek/coordinator required')
        ck.require(args.out is not None and args.out.resolve().parent == pf.CAMPAIGN and not args.out.exists(),
                   'fresh --out directly inside the campaign directory required')
        ck.require(re.fullmatch('[0-9a-f]{40}', args.expected_experiment_commit) is not None, 'full reviewed commit required')
        ck.require(re.fullmatch('[A-Za-z0-9_.-]+', args.remote_repo) is not None and args.remote_repo != 'dsv41-deploy',
                   'experiment checkout must be a separate relative checkout, never ~/dsv41-deploy')
        local = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=pf.ROOT, text=True).strip()
        ck.require(local == args.expected_experiment_commit, 'local checkout is not the reviewed commit')
        ck.require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=pf.ROOT,
                                               text=True).strip(), 'local tracked source is dirty')
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
    sup = pf.Supervisor(out, monitor=monitor)
    live = PF2Live(out, sup, remote=args.remote_repo, expected=args.expected_experiment_commit, data_dir=args.data_dir,
                   tag=tag, stage_bundle=args.stage_bundle, low=args.low_kv_bytes, platform=platform)
    plan = arms(args.low_kv_bytes)
    result = window(live, plan, checkpoint)
    sup.close()
    result.update({'experiment_commit': args.expected_experiment_commit, 'config_sha256': pf.digest(pf.ROOT / CONFIG),
                   'profiler_config': s.PROFILER_JSON, 'restore_sha256': pf.RESTORE_SHA256, 'worker_source': pf.WORKER,
                   'low_kv_bytes': args.low_kv_bytes, 'order': [a.name for a in plan],
                   'expected_kv_tokens': {a.name: arith.expected_tokens(a.kv) for a in plan},
                   'platform': {'kernel': cast(Json, platform.kernel), 'driver': cast(Json, platform.driver)},
                   'arm_memory_min_kb': cast(Json, sup.arm_memory),
                   'artifact_sha256': {str(p.relative_to(out)): pf.digest(p) for p in sorted(out.rglob('*.json*'))
                                       if p.is_file() and not p.name.startswith(('telemetry-', 'procmem-'))}})
    (out / 'summary.json').write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    restore = record(result['restore'])
    print(json.dumps({'status': result['status'], 'stop': result.get('stop'), 'restore': restore.get('status')}))
    if str(restore.get('status', '')).startswith(('RESTORED', 'NOT NEEDED')):
        return 0 if 'stop' not in result else 1
    print('PRODUCTION NOT VERIFIED; HOLD left open. Coordinator: inspect cmd/restore-*.log, then run\n'
          + '\n'.join(f'  bash {pf.RESTORE} {step}' for step in ('start', 'check', 'close')), file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
