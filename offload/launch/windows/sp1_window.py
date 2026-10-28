"""SP1-WINDOW coordinator runner: HOLD, stage, one profiled 327,680 arm (never-started controls, then torch-profiler
captures on both TP ranks), bounded trace collection, ALWAYS the W8 restore.

Dry plan by default (no command, network or file). Only Deepseek / coordinator runs --execute, after review and staging.
Records numbers and hashes only: no prompt or generated text is saved. Reuses the reviewed PF1 runner
(offload/launch/windows/pf1_window.py: Supervisor, Live preflight/HOLD/snapshot/start/restore, runtime(), restored()).
See offload/gb10_budget/PROFILE-WINDOW.md (coordinator sheet), PROFILE-WINDOW-DESIGN.md and PROFILE-WINDOW-RECONCILE.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from offload.gb10_budget import window_cells as wc
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import up2_checks as ck
from offload.launch.windows.up2_low import X86HOST_PROBE
from offload.prefill_maxlen.evidence import Json, number, record

CONFIG = 'offload/launch/configs/SP1-profile.env'
# Byte-identical to SP1-profile.env (CPU test). Fields verified against img vllm/config/profiler.py:39-158.
PROFILER_CONFIG: dict[str, Json] = {
    'profiler': 'torch', 'torch_profiler_dir': '/sp1-profile', 'torch_profiler_activities': ['CPU', 'CUDA'],
    'torch_profiler_with_stack': False, 'torch_profiler_with_flops': False, 'torch_profiler_record_shapes': False,
    'torch_profiler_with_memory': False, 'torch_profiler_use_gzip': True, 'torch_profiler_dump_cuda_time_total': False,
    'ignore_frontend': True, 'detailed_trace_annotation': True, 'max_iterations': 16}
PROFILER_JSON = json.dumps(PROFILER_CONFIG, separators=(',', ':'))
PIN_FILES = ('offload/launch/configs/PF1-window.env', CONFIG, 'offload/launch/start_exp.sh', 'offload/launch/serve_exp.sh',
             'offload/launch/dsv41_tp2_offload.sh', 'offload/kvpack/MANIFEST.json')
PROFILE_DIR = 'dsv41-offload-run/sp1-profile'      # relative to each node's home; must be absent before the window
MOUNT = '/sp1-profile'
TRACE_NAME = re.compile(r'(?:dp0_pp0_tp(?P<tp>[01])_dcp0_ep\d+_)?rank(?P<rank>[01])\.(?P<ns>\d+)\.pt\.trace\.json\.gz')
FILE_CAP = 128 * 1024**2                            # per trace file
DIR_CAP = 512 * 1024**2                             # profile directory per node at any check
DISK_FLOOR_KB = 8 * 1024**2                         # free home disk after every capture
PRE_DISK_KB = 10 * 1024**2                          # free home disk before the HOLD
CAPTURE_FLOOR_KB = int(4.5 * 1024**2)               # MemAvailable before every /start_profile (item 1 §4)
SUBMIT_S, ABORT_S = 40 * 60, 45 * 60                # from the HOLD timestamp (design: ~20 min of measurement)
PROFILER_ON = "Profiler with mode 'torch' is enabled in the API server"
PROFILER_STARTED = 'Torch profiling enabled. Traces will be saved to'


@dataclass(frozen=True)
class SP1Arm(pf.Arm):
    """PF1's OFF arm at 327,680 (PF1_CAP/PF1_TRACE/PF1_EAGER passed by Live.start) through SP1-profile.env."""

    @property
    def config(self) -> str:
        return CONFIG


ARM = SP1Arm('sp1-327', 327680, 0)


# ---------------------------------------------------------------- pure checks (CPU-tested)

def strip_profiler(argv: str) -> str:
    """The experiment's /serve.sh argv minus exactly one reviewed --profiler-config; anything else refuses."""
    tokens = shlex.split(argv.replace('\\\n', ' '))
    ck.require(tokens.count('--profiler-config') == 1, 'experiment argv lacks exactly one --profiler-config')
    i = tokens.index('--profiler-config')
    ck.require(i + 1 < len(tokens) and tokens[i + 1] == PROFILER_JSON, 'profiler config differs from the reviewed JSON')
    del tokens[i:i + 2]
    return shlex.join(tokens)


def trace_mount(mounts: list[dict[str, Json]]) -> dict[str, Json]:
    found = [m for m in mounts if m.get('Destination') == MOUNT]
    ck.require(len(found) == 1, f'expected exactly one {MOUNT} mount')
    m = found[0]
    ck.require(m.get('Type') == 'bind' and m.get('RW') is True and str(m.get('Source', '')).endswith('/' + PROFILE_DIR),
               f'{MOUNT} is not the read-write bind of ~/{PROFILE_DIR}')
    return {'source_tail': PROFILE_DIR, 'rw': True}


def listing(text: str) -> tuple[dict[str, int], int]:
    """`find -printf '%y %s %f'` + FREE line -> ({file: bytes}, free kB); refuses directories or odd names."""
    files: dict[str, int] = {}
    free = -1
    for line in text.splitlines():
        parts = line.split()
        if parts[:1] == ['FREE'] and len(parts) == 2:
            free = int(parts[1])
        elif len(parts) == 3 and parts[0] in ('f', 'd'):
            ck.require(parts[0] == 'f', f'unexpected directory in the profile directory: {parts[2]}')
            ck.require(TRACE_NAME.fullmatch(parts[2]) is not None, f'unexpected file in the profile directory: {parts[2][:80]}')
            files[parts[2]] = int(parts[1])
        elif line.strip():
            raise ValueError(f'unparsed profile listing line: {line[:80]}')
    ck.require(free >= 0, 'profile listing lacks the FREE line')
    return files, free


def new_traces(files: dict[str, int], seen: set[str], rank: int, expected: int, free_kb: int) -> list[tuple[str, int]]:
    """This capture's files on one node, oldest first, against the hard caps."""
    ck.require(sum(files.values()) <= DIR_CAP, f'rank{rank} profile directory exceeds {DIR_CAP >> 20} MiB')
    ck.require(free_kb >= DISK_FLOOR_KB, f'rank{rank} node free disk below {DISK_FLOOR_KB >> 20} GiB')
    new = sorted(((n, s) for n, s in files.items() if n not in seen), key=lambda x: int(_ns(x[0])))
    ck.require(len(new) == expected, f'rank{rank}: {len(new)} new trace files, expected {expected}')
    for name, size in new:
        match = TRACE_NAME.fullmatch(name)
        ck.require(match is not None and int(match['rank']) == rank and match['tp'] in (None, str(rank)),
                   f'trace {name[:60]} is not global rank {rank}')
        ck.require(0 < size <= FILE_CAP, f'trace {name[:60]} size {size} outside (0, {FILE_CAP >> 20} MiB]')
    return new


def _ns(name: str) -> str:
    match = TRACE_NAME.fullmatch(name)
    return match['ns'] if match is not None else '0'


def taken(cell: dict[str, Json]) -> list[dict[str, Json]]:
    """Captures that started a profiler (each wrote one file per rank); SKIPPED_MEMORY started none."""
    rows = [record(c) for c in cast(list[Json], cell['captures'])]
    for row in rows:
        ck.require(row.get('status') in ('TAKEN', 'MIXED', 'ENDED', 'SKIPPED_MEMORY'), 'capture status unknown')
        ck.require(row.get('status') == 'SKIPPED_MEMORY' or row.get('stop_return_rel_s') is not None,
                   'a started capture has no completed stop')
    return [r for r in rows if r.get('status') != 'SKIPPED_MEMORY']


def log_count(text: str) -> int:
    lines = [x for x in text.split() if x.isdigit()]
    ck.require(len(lines) >= 1, 'log count missing')
    return int(lines[-1])


# ---------------------------------------------------------------- live backend (coordinator only)

class SP1Live(pf.Live):
    def __init__(self, out: Path, sup: pf.Supervisor, *, remote: str, expected: str, data_dir: Path, tag: str,
                 stage_bundle: bool, call: pf.Call | None = None,
                 plain: Callable[[str, list[str], float], tuple[int, str]] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        super().__init__(out, sup, remote=remote, expected=expected, data_dir=data_dir, tag=tag,
                         stage_bundle=stage_bundle, call=call, plain=plain, clock=clock)
        self.created: set[str] = set()
        self.collected: dict[str, set[str]] = {host: set() for host in pf.HOSTS}   # Live.seen = container ids
        self.manifest: list[dict[str, Json]] = []
        self.cuts = out / 'cuts.json'
        self.never_started: dict[str, Json] | None = None

    @property
    def pin_files(self) -> tuple[str, ...]:
        return PIN_FILES

    def hold_purpose(self) -> str:
        return f'SP1-WINDOW {self.tag}: bounded torch profile of K1-327 production; no attention/K1/cap/clock change'

    def reap_script(self) -> str:
        return (pf.REAP.replace('PF1-WINDOW', 'SP1-WINDOW').replace('PF1-window[.]env', 'SP1-profile[.]env')
                .replace('@TAG@', self.tag))

    def runtime_check(self, rank: int, live: dict[str, Json], arm: pf.Arm) -> None:
        stripped = {**live, 'argv': strip_profiler(str(live['argv']))}
        pf.runtime(self.production[rank], stripped, arm, self.run_name(arm))

    # -- before the HOLD (read-only, production serving)
    def preflight(self) -> None:
        super().preflight()
        for host in pf.HOSTS:
            text = self.ssh(f'profile-dir-{host}', host, f'test ! -e ~/{PROFILE_DIR} && df -Pk ~ | tail -1')
            ck.require(int(text.split()[-3]) >= PRE_DISK_KB, f'{host} home has < {PRE_DISK_KB >> 20} GiB free')

    def open_hold(self) -> float:
        opened = super().open_hold()
        self.submit, self.sup.abort_at = opened + SUBMIT_S, opened + ABORT_S
        self.sup.bound = f'{ABORT_S // 60}-minute SP1 abort bound'
        (self.out / 'deadlines.json').write_text(json.dumps({'opened': opened, 'submission': self.submit,
            'abort': self.sup.abort_at, 'restore_by': opened + ABORT_S + 8 * 60}) + '\n')
        return opened

    def node_memory(self, label: str) -> None:
        for host in pf.HOSTS:
            text = self.ssh(f'{label}-memory-{host}', host, pf.HOST_PARITY)
            available = re.findall(r'^MemAvailable:\s+(\d+) kB$', text, re.M)
            ck.require(len(available) == 1, f'{host} missing MemAvailable')
            ck.require(int(available[0]) >= pf.START_FLOOR_KB, f'{host} MemAvailable below 5 GiB before the start')

    def log_hits(self, host: str, pattern: str) -> int:
        return log_count(self.ssh(f'log-count-{host}', host,
                                  f'docker logs dsv41-serve 2>&1 | grep -c -F -e {shlex.quote(pattern)} || true'))

    # -- the arm
    def start_arm(self) -> None:
        self.node_memory(ARM.name)
        for host in pf.HOSTS:
            self.ssh(f'{ARM.name}-profile-dir-{host}', host, f'mkdir -p ~/dsv41-offload-run && mkdir ~/{PROFILE_DIR}')
            self.created.add(host)
        self.sup.arm_minima = {}
        self.sup.arm_memory[ARM.name] = self.sup.arm_minima
        super().start(ARM)          # PF1: HOLD, start_exp.sh, /v1/models 327680, identity == production, 13 pins, logs, worker
        for host in pf.HOSTS:
            mounts = ck.rows(json.loads(self.ssh(f'{ARM.name}-mounts-{host}', host,
                                                 "docker inspect --format '{{json .Mounts}}' dsv41-serve")))
            trace_mount(mounts)
            files, free = listing(self.ssh(f'{ARM.name}-profile-empty-{host}', host, self.find()))
            ck.require(not files and free >= DISK_FLOOR_KB, f'{host} profile directory not empty or disk low')
            ck.require(self.log_hits(host, PROFILER_STARTED) == 0, f'{host} profiler already started')
        ck.require(self.log_hits('sparkhead', PROFILER_ON) >= 1, 'API server did not register the profiler endpoints')

    def find(self) -> str:
        return (f"cd ~/{PROFILE_DIR} && find . -mindepth 1 -maxdepth 1 -printf '%y %s %f\\n' && "
                "echo \"FREE $(df -Pk ~ | tail -1 | awk '{print $4}')\"")

    def driver(self, *extra: str) -> list[str]:
        return [sys.executable, '-m', 'offload.gb10_budget.window_cells', '--execute', '--data-dir', str(self.data_dir), *extra]

    def build(self) -> dict[str, Json]:
        self.call('prompts-build', self.driver('build', '--output', str(self.cuts)), timeout=600)
        cuts = ck.read(self.cuts)
        return {key: record(v).get('tokens') for key, v in record(cuts['prompts']).items()}

    def cell(self, name: str) -> dict[str, Json]:
        cell = wc.CELLS[name]
        ck.require(self.clock() + cell.estimate_s <= self.submit, f'{name}: does not fit before the submission bound')
        if cell.captures and self.never_started is None:
            # Every control so far ran with the profiler configured and never started: prove it before the first start.
            self.never_started = {host: self.log_hits(host, PROFILER_STARTED) for host in pf.HOSTS}
            ck.require(not any(self.never_started.values()), 'a profiler started before the profiled cells')
        before = self.stats(f'{name}-stats-before')
        steps = number(pf.worker(self.json(f'{name}-worker-before', 'x86host', 'python3 -c ' + shlex.quote(X86HOST_PROBE)),
                                 self.clock()).get('steps', 0))
        output = self.out / 'cells' / f'{name}.json'
        output.parent.mkdir(exist_ok=True)
        self.call(f'{name}-cell', self.driver(
            'run', '--cell', name, '--cuts', str(self.cuts), '--output', str(output), '--deadline-utc', f'{self.submit:.3f}',
            '--telemetry', *(str(self.out / f'telemetry-{host}.jsonl') for host in pf.HOSTS),
            '--capture-floor-kb', str(CAPTURE_FLOOR_KB), '--cache-salt-prefix', self.tag,
            '--ledger', str(self.out / 'requests-ledger.jsonl')), timeout=3 * cell.estimate_s + 600, inflight=cell.batch)
        result = ck.read(output)
        ck.require(result.get('cell') == name and len(cast(list[Json], result['streams'])) == cell.batch,
                   'cell artifact identity mismatch')
        after = self.stats(f'{name}-stats-after')
        ck.counters(after, 0, before)
        ck.require(number(record(after['counters']).get('rounds_ok', 0)) > number(record(before['counters']).get('rounds_ok', 0)),
                   'no remote drafting during the cell')
        status = pf.worker(self.json(f'{name}-worker-after', 'x86host', 'python3 -c ' + shlex.quote(X86HOST_PROBE)), self.clock())
        ck.require(number(status.get('steps', 0)) > steps, 'x86host worker took no steps during the cell')
        with (self.out / 'proxy-stats.jsonl').open('a') as handle:
            handle.write(json.dumps({'cell': name, 'before': before['counters'], 'after': after['counters']}) + '\n')
        traces = self.collect_traces(name, result) if cell.captures else []
        return {'captures': [{k: c.get(k) for k in ('kind', 'status')} for c in map(record, cast(list[Json], result['captures']))],
                'traces': cast(Json, traces)}

    def collect_traces(self, name: str, result: dict[str, Json]) -> list[dict[str, Json]]:
        started = taken(result)
        rows: list[dict[str, Json]] = []
        for rank, host in enumerate(pf.HOSTS):
            files, free = listing(self.ssh(f'{name}-list-{host}', host, self.find()))
            new = new_traces(files, self.collected[host], rank, len(started), free)
            if not new:
                continue
            local = self.out / 'traces' / name / host
            local.mkdir(parents=True, exist_ok=True)
            names = [n for n, _ in new]
            remote = self.json(f'{name}-hash-{host}', host, f'cd ~/{PROFILE_DIR} && python3 - ' + shlex.join(names),
                               pf.IMAGE_BASES)      # O_DIRECT sha256 on the Spark, no page-cache fill
            for (file, size), capture in zip(new, started):
                self.call(f'{name}-copy-{host}', ['scp', '-q', '-o', 'BatchMode=yes', f'{host}:{PROFILE_DIR}/{file}', str(local)])
                got = pf.digest(local / file)
                ck.require(remote.get(file) == got and (local / file).stat().st_size == size, f'{file[:40]} copy differs')
                rows.append({'cell': name, 'capture': capture['kind'], 'status': capture['status'], 'rank': rank,
                             'host': host, 'file': str((local / file).relative_to(self.out)), 'bytes': size, 'sha256': got})
            # Verified copies only; the directory is user-owned, so the container's root-owned files unlink normally.
            self.ssh(f'{name}-remove-{host}', host, f'cd ~/{PROFILE_DIR} && rm -f -- ' + shlex.join(names))
            self.collected[host].update(names)
        self.manifest.extend(rows)
        (self.out / 'traces-manifest.json').write_text(json.dumps(self.manifest, indent=1) + '\n')
        return rows

    def finish(self) -> dict[str, Json]:
        checks: dict[str, Json] = {}
        for rank, host in enumerate(pf.HOSTS):
            identity = self.ssh(f'final-id-{host}', host, "docker inspect --format '{{.Id}}' dsv41-serve").strip()
            ck.require(identity == self.ids[rank], 'experiment container changed during the window')
            checks[f'logs_rank{rank}'] = pf.logs(self.json(f'final-logs-{host}', host, 'python3 -c ' +
                shlex.quote(pf.LOG_FILTER) + ' ' + shlex.quote(ck.STOP_LOG.pattern)), rank, ARM.cap)
            files, _ = listing(self.ssh(f'final-list-{host}', host, self.find()))
            ck.require(not files, f'{host} profile directory still holds files after collection')
            self.ssh(f'final-rmdir-{host}', host, f'rmdir ~/{PROFILE_DIR}')
            self.created.discard(host)
        pf.worker(self.json('final-worker', 'x86host', 'python3 -c ' + shlex.quote(X86HOST_PROBE)), self.clock())
        checks['never_started_before_profiled_cells'] = cast(Json, self.never_started)
        minima = self.sup.arm_minima
        ck.require(minima is not None and set(minima) == set(pf.HOSTS), 'missing memory minima')
        checks['mem_available_min_kb'] = cast(Json, dict(cast(dict[str, int], minima)))
        return checks

    def restore(self, *, production_stopped: bool) -> dict[str, Json]:
        (self.out / 'arm-memory.json').write_text(json.dumps(self.sup.arm_memory, indent=1) + '\n')
        out = super().restore(production_stopped=production_stopped)
        # Only directories this window created; uncollected traces are evidence and stay for the coordinator.
        for host in sorted(self.created):
            rc, text = self.plain(f'profile-dir-{host}', [*pf.SSH, host, f'rmdir ~/{PROFILE_DIR} 2>/dev/null; '
                                  f'if [ -e ~/{PROFILE_DIR} ]; then echo "LEFT $(ls -1 ~/{PROFILE_DIR} | wc -l) files"; '
                                  'else echo CLEAN; fi'], 60)
            out[f'profile_dir_{host}'] = f'rc{rc}: {text.strip()[-80:]}'
        return out


# ---------------------------------------------------------------- orchestration (shared by SP1Live and fakes)

class Backend(Protocol):
    def preflight(self) -> None: ...
    def open_hold(self) -> float: ...
    def snapshot(self) -> None: ...
    def start_arm(self) -> None: ...
    def build(self) -> dict[str, Json]: ...
    def cell(self, name: str) -> dict[str, Json]: ...
    def finish(self) -> dict[str, Json]: ...
    def restore(self, *, production_stopped: bool) -> dict[str, Json]: ...


def window(b: Backend, checkpoint: Callable[[dict[str, Json]], None]) -> dict[str, Json]:
    """Whatever happens after the HOLD may exist, the W8 restore runs; a failed restore is never hidden."""
    result: dict[str, Json] = {'status': 'INCOMPLETE', 'cells': {}}
    cells = cast(dict[str, Json], result['cells'])
    hold = stopped = False
    try:
        b.preflight()
        hold = True    # from here a HOLD may exist (open can time out after writing it): always restore/close
        result['hold_opened'] = b.open_hold()
        b.snapshot()
        stopped = True    # start_exp.sh stops production before anything else
        b.start_arm()
        result['prompts'] = b.build()
        checkpoint({'prompts': result['prompts']})
        for name in wc.ORDER:
            cells[name] = b.cell(name)
            checkpoint({'cell': name})
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

def plan() -> str:
    lines = ['DRY RUN: no command, network or file. Coordinator Deepseek/coordinator only (--execute).',
             'Preflight (read-only, production serving): PF1 preflight (restore.sh pin; D2 pause; corpus; production 21bed31/tag,',
             f'  clean, no HOLD; ~/dsv41-deploy-offload = reviewed commit or --stage-bundle; {len(PIN_FILES)} staged pins incl. {CONFIG};',
             '  :8000 K1 327680; x86host TP2 worker fb68c52 read-only probe; nvidia-smi + MemAvailable >= 5 GiB); 1 s telemetry both Sparks;',
             f'  ~/{PROFILE_DIR} absent and >= {PRE_DISK_KB >> 20} GiB free home disk on both Sparks.',
             f'HOLD: window.sh open "SP1-WINDOW <tag>: ..." (drain <= 20 min); deadlines from HOLD: submit +{SUBMIT_S // 60},'
             f' abort +{ABORT_S // 60}, restore by about +{ABORT_S // 60 + 8} min.',
             'Snapshot production both ranks (identity, mounted hashes, env, argv, Engram, 13 pins, image bases) -> rollback.json.',
             'Tunnel 127.0.0.1:18001 -> sparkhead 127.0.0.1:8001 (owned). Guards: MemAvailable >= 4 GiB, swap +256 MiB after ready,',
             '  telemetry liveness, foreign demand on :8001 above owned in-flight requests, STOP file, deadlines.',
             f'{ARM.name}: mkdir ~/{PROFILE_DIR} both nodes; PF1_CAP=327680 PF1_TRACE=0 PF1_EAGER=0 bash offload/launch/start_exp.sh {CONFIG}',
             '  identity == production except port/REQUIRED and exactly one reviewed --profiler-config; KV 2502600; 13-pin verify;',
             f'  {MOUNT} rw bind on both ranks; profile directory empty; API "{PROFILER_ON[:40]}..." logged; profiler never started.',
             'Prompts: window_cells build (7 public prompts, <= 4 /tokenize each; numeric cuts + sha256 only).']
    for name in wc.ORDER:
        cell = wc.CELLS[name]
        what = ('+'.join(cell.captures) + ' capture(s)') if cell.captures else ('discarded' if cell.warmup else 'never-started control')
        lines.append(f'  {name}: {cell.batch} x {"/".join(cell.prompts)}, {cell.max_tokens} output; {what}; '
                     'proxy/spec/worker counters before+after')
        if cell.captures:
            lines.append(f'    capture floor MemAvailable >= {CAPTURE_FLOOR_KB / 2**20:.1f} GiB else SKIPPED_MEMORY; per node: one new rank file'
                         f' per capture <= {FILE_CAP >> 20} MiB, dir <= {DIR_CAP >> 20} MiB, free disk >= {DISK_FLOOR_KB >> 20} GiB;'
                         ' copy + O_DIRECT sha256 both sides; remove verified Spark copies')
    lines += ['Finish: containers unchanged, clean logs both ranks, worker probe, profile directories empty and removed.',
              'ALWAYS (success or any STOP): close owned processes, then bash ' + str(pf.RESTORE) + ' start (if production stopped),',
              '  check, close; require K1 START VERIFIED, six "greedy K1-post vs restored: i IDENTICAL n", PRODUCTION HOLD OK,',
              '  CHECK: DONE, CLOSE: window closed. Then remove only the empty profile directories this window created.',
              'Analysis afterwards (Controller, CPU): python3 -m offload.gb10_budget.trace_budget --window <out> (PROFILE-WINDOW.md).',
              'ESTIMATED: ~28 min HOLD, production down ~27.6 min (offload/gb10_budget/PROFILE-WINDOW-DESIGN.md).']
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
    args = ap.parse_args(argv)
    if not args.execute:
        print(plan())
        return 0
    try:
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
    sup = pf.Supervisor(out)
    live = SP1Live(out, sup, remote=args.remote_repo, expected=args.expected_experiment_commit, data_dir=args.data_dir,
                   tag=tag, stage_bundle=args.stage_bundle)
    result = window(live, checkpoint)
    sup.close()
    result.update({'experiment_commit': args.expected_experiment_commit, 'config_sha256': pf.digest(pf.ROOT / CONFIG),
                   'profiler_config': PROFILER_JSON, 'restore_sha256': pf.RESTORE_SHA256, 'worker_source': pf.WORKER,
                   'artifact_sha256': {str(p.relative_to(out)): pf.digest(p) for p in sorted(out.rglob('*.json*'))
                                       if p.is_file() and not p.name.startswith('telemetry-')}})
    (out / 'summary.json').write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    restore = record(result['restore'])
    print(json.dumps({'status': result['status'], 'stop': result.get('stop'), 'restore': restore.get('status')}))
    if str(restore.get('status', '')).startswith(('RESTORED', 'NOT NEEDED')):
        return 0 if 'stop' not in result else 1
    print('PRODUCTION NOT VERIFIED; HOLD left open. Coordinator: inspect cmd/restore-*.log, then run\n'
          + '\n'.join(f'  bash {pf.RESTORE} {s}' for s in ('start', 'check', 'close')), file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
