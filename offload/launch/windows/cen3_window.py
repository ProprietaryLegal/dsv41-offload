"""CEN3-WINDOW coordinator runner (documents/plans/lanes/CEN3-BYTES-AND-WATERFALL.md, CEN-RUN.plan.md): one HOLD, three
parts, each with its own function-only smoke; a part whose smoke fails is SKIPPED with the reason and the window goes
on (the parts are independent measurements); any other failure STOPs and restores. SINGLE window, never chained.

  production stop (owned, under the HOLD admission), both GPUs idle, MemAvailable >= 40 GiB, sources staged + sha256
  (c) per-Spark streaming bandwidth: read / write / copy of 2 GiB buffers, 20 reps, both Sparks in parallel
  (a) Nsight Compute: ncu found (image, else the host's Nsight Compute mounted read-only), metric names probed, one
      profiled mrow launch with a positive read count; then smoe B1/B4, dense mrow and LM-head calls, one profiled call
      per case, --cache-control all, --clock-control none; exact logical bytes per launch from the driver
  (b) one profiled production-config start (CEN3-profile.env): decode captures at short-b1, 32k-b1, 32k-b4; the
      drafter-down segment (prod_window.sh pair on x86host: target-only rounds, rounds_ok unchanged, STOP counters
      unchanged), prod_window.sh close (PRODUCTION HOLD OK), a recovery request (rounds_ok and worker steps grow);
      traces copied with sha256 both sides (SP1 layout)
Afterwards (CPU, Controller): offload.census.cen3_analyze (waterfall, byte ratios, bandwidth) and offload.census.cen4.
"""
from __future__ import annotations

import json
import re
import shlex
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

from offload.census import bytes_model as BM
from offload.census import cen3_cells as cells
from offload.launch.windows import cen_common as cc
from offload.launch.windows import hold_admission
from offload.launch.windows import nbw_window as n
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import scoped_remote as owner
from offload.launch.windows import sp1_window as sp1
from offload.launch.windows import up2_checks as ck
from offload.launch.windows.up2_low import X86HOST_PROBE
from offload.prefill_maxlen.evidence import Json, number, record

CONFIG = 'offload/launch/configs/CEN3-profile.env'
# Byte-identical to CEN3-profile.env (CPU test): SP1's profiler config with its own directory and 64 iterations.
PROFILER_CONFIG: dict[str, Json] = {**sp1.PROFILER_CONFIG, 'torch_profiler_dir': '/cen3-profile', 'max_iterations': 64}
PROFILER_JSON = json.dumps(PROFILER_CONFIG, separators=(',', ':'))
SOURCES = ('offload/__init__.py', 'offload/kl1/__init__.py', 'offload/kl1/shapes.py', 'offload/kl1/smoe_bench.py',
           'offload/census/__init__.py', 'offload/census/cen3_bench.py')
PIN_FILES = (*cc.SHARED_PINS, CONFIG, 'offload/launch/configs/NBW-native.env', 'offload/launch/windows/cen3_window.py',
             *SOURCES)
RUN = 'dsv41-offload-run/cen3'
MOUNT = '/cen3-profile'
PLAN = '/srv/dsv41/legal-v2a/exl3_moe_plan.json'
CONTAINER = 'dsv41-exp-cen3'                       # + '-<part>'; spark_quiet refuses any running dsv41-exp* container
IDLE = 'nvidia-smi --query-compute-apps=pid --format=csv,noheader'
GPU_QUERY = ('nvidia-smi --query-gpu=uuid,clocks.gr,clocks.max.gr,clocks.mem,power.draw,power.limit,temperature.gpu,'
             'clocks_event_reasons.active --format=csv,noheader,nounits')
PRE_MEMORY_KB = 40 * 1024**2
BENCH_FILE_CAP = 16 * 1024**2
TIMEOUT_S = {'bw-smoke': 180, 'bw': 420, 'ncu-smoke': 900, 'ncu': 1200}
ESTIMATE_S = {'c': 6 * 60, 'a': 25 * 60, 'b': 27 * 60}
NCU_GROUPS = ('smoe-b1', 'smoe-b4', 'dense', 'head')
CAPTURE_FLOOR_KB = sp1.CAPTURE_FLOOR_KB
START_BUDGET_S = 15 * 60
RESULT = re.compile(r'(?:bw-smoke|bw)\.json|bw-gpu-(?:before|after)\.csv|ncu-(?:path|version)\.txt|ncu-query\.txt|'
                    r'ncu-metrics\.json|ncu-failures\.txt|ncu-(?:smoke|smoe-b1|smoe-b4|dense|head)'
                    r'(?:\.csv|\.app\.txt|\.err|-target\.json)')


# ---------------------------------------------------------------- pure (CPU-tested)

def bench_script(part: str, rank: int) -> str:
    """In-container script for one part on one rank. Outputs to /cen3-out (numbers only); the ncu parts run as root
    with SYS_ADMIN (GPU performance counters) and hand every file back to the host user on exit."""
    py = 'python3 -m offload.census.cen3_bench'
    gpu = f'{GPU_QUERY} > /cen3-out/bw-gpu-{{when}}.csv'
    if part == 'bw-smoke':
        return f'set -euo pipefail\n{py} bw --mib 64 --reps 1 --rank {rank} --out /cen3-out/bw-smoke.json'
    if part == 'bw':
        return '\n'.join(['set -euo pipefail', gpu.format(when='before'),
                          f'{py} bw --mib 2048 --reps 20 --rank {rank} --out /cen3-out/bw.json', gpu.format(when='after')])
    find = ('NCU=; for c in "$(command -v ncu || true)" /usr/local/cuda/bin/ncu /opt/nvidia/nsight-compute/*/ncu '
            '/ncu-host/ncu; do if [ -n "$c" ] && [ -x "$c" ]; then NCU=$c; break; fi; done')
    head = ['set -euo pipefail',
            'trap \'chown -R "$HOST_UID:$HOST_GID" /cen3-out /ext 2>/dev/null || true\' EXIT',
            'export HOME=/ext TORCH_EXTENSIONS_DIR=/ext/jit', find]

    def profile(group: str) -> str:
        return (f'"$NCU" --profile-from-start off --cache-control all --clock-control none --metrics "$M" --csv '
                f'--page raw --print-units base --log-file /cen3-out/ncu-{group}.csv '
                f'{py} ncu-target --group {group} --rank {rank} --plan /plan.json --out /cen3-out/ncu-{group}-target.json '
                f'> /cen3-out/ncu-{group}.app.txt 2> /cen3-out/ncu-{group}.err')
    if part == 'ncu-smoke':
        return '\n'.join([*head,
                          'if [ -z "$NCU" ]; then echo ABSENT > /cen3-out/ncu-path.txt; exit 3; fi',
                          'echo "$NCU" > /cen3-out/ncu-path.txt',
                          '"$NCU" --version > /cen3-out/ncu-version.txt',
                          '"$NCU" --query-metrics > /cen3-out/ncu-query.txt',
                          "python3 -c 'from vllm_exl3_moe import smoe, mrow; smoe.ext(); mrow.ext()'",
                          f'M=$({py} pick-metrics --query /cen3-out/ncu-query.txt --out /cen3-out/ncu-metrics.json)',
                          profile('smoke')])
    if part == 'ncu':
        lines = [*head, 'test -n "$NCU"', 'M=$(python3 -c \'import json; print(",".join(json.load(open('
                 '"/cen3-out/ncu-metrics.json"))["metrics"]))\')', ': > /cen3-out/ncu-failures.txt']
        lines += [profile(g) + f' || echo "{g} $?" >> /cen3-out/ncu-failures.txt' for g in NCU_GROUPS]
        return '\n'.join(lines)
    raise ValueError(f'unknown CEN3 bench part {part!r}')


def docker_command(part: str, rank: int, run: str, ncu_host: str | None) -> str:
    """Host-side launch on one Spark: refuse a leftover container of this part, then one bounded --rm container."""
    name = f'{CONTAINER}-{part}'
    root = part.startswith('ncu')
    mounts = [f'type=bind,src=$HOME/{run}/src,dst=/cen3,readonly', f'type=bind,src=$HOME/{run}/out,dst=/cen3-out',
              f'type=bind,src=$HOME/{run}/ext,dst=/ext', f'type=bind,src={PLAN},dst=/plan.json,readonly']
    if root and ncu_host is not None:
        mounts.append(f'type=bind,src={ncu_host},dst=/ncu-host,readonly')
    who = '--cap-add SYS_ADMIN -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)"' if root else '--user "$(id -u):$(id -g)"'
    return (f'cd ~/{run} && test -z "$(docker ps -aq --filter name=^/{name}$)" && '
            f'exec timeout --signal=TERM --kill-after=15s --foreground {TIMEOUT_S[part]}s docker run --rm --init '
            f'--name {name} --gpus all --network none {who} --memory=32g --memory-swap=32g --shm-size=1g '
            '--pids-limit=1024 ' + ' '.join(f'--mount "{m}"' for m in mounts) + ' '
            '-e PYTHONPATH=/cen3 -e PYTHONDONTWRITEBYTECODE=1 -e MAX_JOBS=4 -e EXL3_SMOE_FLAGS=-DSMOE_RUNFLAGS=1 '
            '-e EXL3_MROW32_VARIANT=ring -e EXL3_MROW32_MIN_ROWS=9 '
            f'-w /cen3 --entrypoint bash {pf.IMAGE} -c {shlex.quote(bench_script(part, rank))}')


def bench_files(text: str) -> dict[str, int]:
    """`find -printf '%y %s %f'` of the out directory -> {file: bytes}; refuses directories, odd names or sizes."""
    files: dict[str, int] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        kind, size, name = line.split()
        ck.require(kind == 'f' and RESULT.fullmatch(name) is not None, f'unexpected bench output {name[:60]}')
        ck.require(0 <= int(size) <= BENCH_FILE_CAP and name not in files, f'bench output {name[:60]} size/duplicate')
        files[name] = int(size)
    return files


def ncu_host_dir(text: str) -> str | None:
    """The newest host Nsight Compute directory holding an `ncu` binary, from the preflight probe; None if none."""
    dirs = [line.strip() for line in text.splitlines() if line.strip().startswith('/')]
    for d in dirs:
        ck.require(re.fullmatch(r'/[A-Za-z0-9_./+-]+', d) is not None and '..' not in d, 'odd Nsight Compute path')
    return sorted(dirs)[-1] if dirs else None


def target_only(before: dict[str, Json], after: dict[str, Json]) -> dict[str, Json]:
    """The drafter-down cell: no remote round succeeded, target-only rounds were counted, no STOP counter moved."""
    ck.counters(after, 0, before)
    a, b = record(before['counters']), record(after['counters'])
    ck.require(number(b.get('rounds_ok', 0)) == number(a.get('rounds_ok', 0)),
               'remote rounds succeeded while the x86host drafter was down (pair window not in force)')
    grew = number(b.get('fallback_not_ready', 0)) - number(a.get('fallback_not_ready', 0))
    ck.require(grew > 0, 'no target-only round counted (fallback_not_ready did not grow)')
    return {'fallback_not_ready_delta': grew, 'rounds_ok_delta': 0,
            'fallback_timeout_delta': number(b.get('fallback_timeout', 0)) - number(a.get('fallback_timeout', 0))}


# ---------------------------------------------------------------- live backend (coordinator only)

class CEN3Live(cc.CenLive):
    MARKER = 'CEN3-WINDOW'
    TIMING = cc.Timing(58 * 60, 62 * 60, 82 * 60)

    def __init__(self, out: Path, sup: pf.Supervisor, *, remote: str, expected: str, data_dir: Path, tag: str,
                 production_commit: str = n.PROD_COMMIT, production_tag: str = n.PROD_TAG,
                 call: pf.Call | None = None,
                 plain: Callable[[str, list[str], float], tuple[int, str]] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        super().__init__(out, sup, remote=remote, expected=expected, data_dir=data_dir, tag=tag,
                         production_commit=production_commit, production_tag=production_tag, call=call, plain=plain,
                         clock=clock)
        self.created: set[str] = set()
        self.ncu_host: dict[str, str | None] = {}
        self.plan_sha = ''
        self.bench_collected: set[str] = set()
        self.collected: dict[str, set[str]] = {host: set() for host in pf.HOSTS}
        self.traces: list[dict[str, Json]] = []
        self.traces_done = False
        self.prompts = self.out / 'prompts.json'

    @property
    def pin_files(self) -> tuple[str, ...]:
        return PIN_FILES

    @property
    def run(self) -> str:
        return f'{RUN}/{self.tag}'

    def run_dirs(self) -> tuple[str, ...]:
        return (self.run,)

    def hold_purpose(self) -> str:
        return f'{self.hold_marker()} bandwidth, ncu bytes and one profiled production-config start; no production change'

    # -- before the HOLD (read-only)
    def extra_preflight(self) -> None:
        hashes: list[str] = []
        for host in pf.HOSTS:
            self.ssh(f'cen3-image-{host}', host, f'docker image inspect {pf.IMAGE} >/dev/null && '
                     f'test -z "$(docker ps -aq --filter name=^/{CONTAINER}-)"')
            sha = str(self.json(f'cen3-plan-{host}', host, 'python3 - ' + shlex.quote(PLAN), pf.IMAGE_BASES).get(PLAN, ''))
            ck.require(re.fullmatch('[0-9a-f]{64}', sha) is not None, f'{host} missing plan digest')
            hashes.append(sha)
            self.ncu_host[host] = ncu_host_dir(self.ssh(f'cen3-ncu-host-{host}', host,
                'for d in /opt/nvidia/nsight-compute/* /usr/local/cuda/nsight-compute*; do '
                '[ -x "$d/ncu" ] && echo "$d"; done; true'))
        ck.require(len(set(hashes)) == 1, 'Spark plan digests differ')
        self.plan_sha = hashes[0]
        self.ssh('cen3-x86host-window', 'x86host', 'test -f ~/dsv41-offload/prod_window.sh && test ! -e ~/dsv41-offload/PAIR_OK '
                 '&& test ! -e ~/dsv41-offload/d2/TRAIN_OK')
        (self.out / 'preflight.json').write_text(json.dumps({'plan_sha256': self.plan_sha,
                                                             'ncu_host': cast(Json, self.ncu_host)}) + '\n')

    # -- production down, sources staged
    def stop_production(self) -> dict[str, Json]:
        self.stopped = True           # BEFORE: a lost reply may still have stopped production
        script = hold_admission.command('stop', self.tag + '-hold', self.admission_token,
                                        ['bash', '-c', 'cd ~/dsv41-deploy && ./stop.sh </dev/null'])
        self.owned('cen3-stop', 'sparkhead', f'{self.tag}-cen3-stop', script, timeout=240)
        hosts: dict[str, Json] = {}
        for host in pf.HOSTS:
            ck.require(not self.ssh(f'cen3-idle-{host}', host, IDLE).strip(), f'{host} still has compute apps after stop')
            text = self.ssh(f'cen3-memory-{host}', host, pf.HOST_PARITY)
            available = re.findall(r'^MemAvailable:\s+(\d+) kB$', text, re.M)
            ck.require(len(available) == 1 and int(available[0]) >= PRE_MEMORY_KB, f'{host} MemAvailable below 40 GiB')
            hosts[host] = int(available[0])
        self.sup.arm_minima = {}
        self.sup.arm_memory['cen3-bench'] = self.sup.arm_minima
        return {'mem_available_kb': hosts}

    def stage(self) -> None:
        for host in pf.HOSTS:
            self.created.add(host)
            self.ssh(f'cen3-mkdir-{host}', host, f'mkdir -p ~/{RUN} && mkdir ~/{self.run} && mkdir -p '
                     f'~/{self.run}/src/offload/kl1 ~/{self.run}/src/offload/census ~/{self.run}/out ~/{self.run}/ext '
                     f'~/{self.run}/profile')
            for path in SOURCES:
                self.call(f'cen3-source-{host}', ['scp', '-q', '-o', 'BatchMode=yes', str(pf.ROOT / path),
                                                  f'{host}:{self.run}/src/{path}'])
            text = self.ssh(f'cen3-source-hash-{host}', host, f'cd ~/{self.run}/src && sha256sum ' + shlex.join(SOURCES))
            ck.require(text.strip() == '\n'.join(f'{pf.digest(pf.ROOT / p)}  {p}' for p in SOURCES),
                       f'{host} staged source hashes differ')

    # -- bench containers
    def bench(self, part: str) -> tuple[bool, str]:
        """One part's container on both Sparks at once, in one owned phase; (both exited 0, reason)."""
        tag = f'{self.tag}-cen3-{part}'
        self.remember(tag, self.sup.abort_at)
        commands = [shlex.join([*pf.SSH, host, owner.command('launch', tag, ['bash', '-c', docker_command(
            part, rank, self.run, self.ncu_host.get(host))], deadline=self.sup.abort_at)])
            for rank, host in enumerate(pf.HOSTS)]
        shell = (f'{commands[0]} & a=$!\n{commands[1]} & b=$!\nra=0; rb=0; wait "$a" || ra=$?; wait "$b" || rb=$?\n'
                 'echo "CEN3 BENCH rc $ra $rb"; test "$ra" = 0 && test "$rb" = 0')
        try:
            self.call(f'cen3-{part}', ['bash', '-c', shell], timeout=TIMEOUT_S[part] + 120)
            ok, reason = True, ''
        except ValueError as exc:
            # Only the containers' own non-zero exit is a part result; a supervisor STOP (STOP file, memory floor,
            # abort bound, local timeout) propagates and ends the window.
            if f'cen3-{part} failed rc' not in str(exc):
                raise
            ok, reason = False, str(exc)[-300:]
        self.benches_absent(part)
        return ok, reason

    def benches_absent(self, label: str) -> None:
        """No CEN3 container on either Spark (a killed client leaves dockerd's child), and no compute app."""
        for host in pf.HOSTS:
            text = self.ssh(f'cen3-absent-{label}-{host}', host,
                            f'ids=$(docker ps -aq --filter name=^/{CONTAINER}-); if [ -n "$ids" ]; then '
                            'docker rm -f $ids >/dev/null; fi; '
                            f'test -z "$(docker ps -aq --filter name=^/{CONTAINER}-)" && apps=$({IDLE}) && '
                            'test -z "$apps" && echo CEN3 BENCH ABSENT')
            ck.require(text.splitlines()[-1:] == ['CEN3 BENCH ABSENT'], f'{host}: CEN3 container or compute app remains')

    def fetch(self, names: tuple[str, ...]) -> dict[str, dict[str, Path]]:
        """Copy listed bench outputs (those present) from both Sparks with sha256 both sides; {host: {name: path}}."""
        got: dict[str, dict[str, Path]] = {}
        for host in pf.HOSTS:
            files = bench_files(self.ssh(f'cen3-out-list-{host}', host,
                                         f"cd ~/{self.run}/out && find . -mindepth 1 -maxdepth 1 -printf '%y %s %f\\n'"))
            want = sorted(n_ for n_ in names if n_ in files and f'{host}/{n_}' not in self.bench_collected)
            local = self.out / 'bench' / host
            local.mkdir(parents=True, exist_ok=True)
            got[host] = {n_: local / n_ for n_ in names if (local / n_).exists()}
            if not want:
                continue
            hashes = self.json(f'cen3-out-hash-{host}', host, f'cd ~/{self.run}/out && python3 - ' + shlex.join(want),
                               pf.IMAGE_BASES)
            for name in want:
                self.call(f'cen3-out-copy-{host}', ['scp', '-q', '-o', 'BatchMode=yes', f'{host}:{self.run}/out/{name}',
                                                    str(local)])
                ck.require(hashes.get(name) == pf.digest(local / name) and (local / name).stat().st_size == files[name],
                           f'{host}/{name} copy differs')
                self.bench_collected.add(f'{host}/{name}')
                got[host][name] = local / name
        manifest = [{'host': h, 'file': f'bench/{h}/{p.name}', 'bytes': p.stat().st_size, 'sha256': pf.digest(p)}
                    for h in pf.HOSTS for p in sorted((self.out / 'bench' / h).iterdir())]
        (self.out / 'bench-manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
        return got

    def part_c(self) -> dict[str, Json]:
        ok, reason = self.bench('bw-smoke')
        got = self.fetch(('bw-smoke.json',))
        if not ok or not all('bw-smoke.json' in got[h] for h in pf.HOSTS):
            return {'status': f'SKIPPED: bandwidth smoke failed: {reason or "no output"}'}
        for host in pf.HOSTS:
            ops = cast(list[Json], json.loads(got[host]['bw-smoke.json'].read_text())['ops'])
            if not all(number(record(o)['gbps_median']) > 0 for o in ops):
                return {'status': f'SKIPPED: bandwidth smoke on {host} measured no positive rate'}
        ok, reason = self.bench('bw')
        ck.require(ok, f'(c) bandwidth bench failed after a passing smoke: {reason}')
        got = self.fetch(('bw.json', 'bw-gpu-before.csv', 'bw-gpu-after.csv'))
        ck.require(all('bw.json' in got[h] for h in pf.HOSTS), '(c) bandwidth result missing')
        return {'status': 'DONE', 'files': {h: f'bench/{h}/bw.json' for h in pf.HOSTS}}

    def part_a(self) -> dict[str, Json]:
        ok, reason = self.bench('ncu-smoke')
        smoke = ('ncu-path.txt', 'ncu-version.txt', 'ncu-query.txt', 'ncu-metrics.json', 'ncu-smoke.csv',
                 'ncu-smoke.app.txt', 'ncu-smoke.err', 'ncu-smoke-target.json')
        got = self.fetch(smoke)
        for host in pf.HOSTS:
            path = got[host].get('ncu-path.txt')
            if path is None or path.read_text().strip() in ('', 'ABSENT'):
                return {'status': f'SKIPPED: Nsight Compute absent in the image and on {host}', 'smoke_rc': reason}
        if not ok:
            return {'status': f'SKIPPED: ncu smoke failed: {reason}'}
        picked: dict[str, Json] = {}
        for host in pf.HOSTS:
            launches = smoke_launches(got[host])
            m = record(cast(Json, json.loads(got[host]['ncu-metrics.json'].read_text())))
            read = str(m['read'])
            if not any(ln.family == 'mrow' and ln.metrics.get(read, 0) > 0 for ln in launches):
                return {'status': f'SKIPPED: ncu smoke on {host} recorded no mrow launch with a positive {read}'}
            picked[host] = m
        ok, reason = self.bench('ncu')
        ck.require(ok, f'(a) ncu bench failed after a passing smoke: {reason}')
        names = tuple(f'ncu-{g}{s}' for g in NCU_GROUPS for s in ('.csv', '.app.txt', '.err', '-target.json'))
        got = self.fetch((*names, 'ncu-failures.txt'))
        failures = {h: got[h]['ncu-failures.txt'].read_text().split() if 'ncu-failures.txt' in got[h] else ['missing']
                    for h in pf.HOSTS}
        return {'status': 'DONE', 'metrics': cast(Json, picked), 'failures': cast(Json, failures)}

    # -- part (b)
    def start_cen3(self) -> list[dict[str, Json]]:
        current = self.start_engine('cen3', f'CEN3_TAG={self.tag}', CONFIG, budget_s=START_BUDGET_S)
        for rank, host in enumerate(pf.HOSTS):
            cc.runtime_identity(self.production[rank], current[rank], env={}, eager=False,
                                strip=('--profiler-config', PROFILER_JSON))
            mounts = ck.rows(json.loads(self.ssh(f'cen3-mounts-{host}', host,
                                                 "docker inspect --format '{{json .Mounts}}' dsv41-serve")))
            found = [m for m in mounts if m.get('Destination') == MOUNT]
            ck.require(len(found) == 1 and found[0].get('Type') == 'bind' and found[0].get('RW') is True and
                       str(found[0].get('Source', '')).endswith(f'/{self.run}/profile'),
                       f'{MOUNT} is not the read-write bind of ~/{self.run}/profile')
            files, free = sp1.listing(self.ssh(f'cen3-profile-empty-{host}', host, self.find()))
            ck.require(not files and free >= sp1.DISK_FLOOR_KB, f'{host} profile directory not empty or disk low')
        self.accept_start('cen3', current)
        ck.require(self.log_hits('sparkhead', sp1.PROFILER_ON) >= 1, 'API server did not register the profiler endpoints')
        for host in pf.HOSTS:
            ck.require(self.log_hits(host, sp1.PROFILER_STARTED) == 0, f'{host} profiler already started')
        return current

    def log_hits(self, host: str, pattern: str) -> int:
        return sp1.log_count(self.ssh(f'log-count-{host}', host,
                                      f'docker logs dsv41-serve 2>&1 | grep -c -F -e {shlex.quote(pattern)} || true'))

    def find(self) -> str:
        return (f"cd ~/{self.run}/profile && find . -mindepth 1 -maxdepth 1 -printf '%y %s %f\\n' && "
                "echo \"FREE $(df -Pk ~ | tail -1 | awk '{print $4}')\"")

    def cen3_driver(self, *extra: str) -> list[str]:
        return [sys.executable, '-m', 'offload.census.cen3_cells', '--execute', '--data-dir', str(self.data_dir), *extra]

    def build_prompts(self) -> dict[str, Json]:
        self.call('prompts-build', self.cen3_driver('build', '--output', str(self.prompts)), timeout=600)
        return {k: record(v).get('tokens') for k, v in record(ck.read(self.prompts)['prompts']).items()}

    def worker_steps(self, label: str) -> float:
        return number(pf.worker(self.json(label, 'x86host', 'python3 -c ' + shlex.quote(X86HOST_PROBE)),
                                self.clock()).get('steps', 0))

    def run_cell(self, name: str, *, drafter: bool) -> dict[str, Json]:
        """One cell. With the drafter up: STOP counters unchanged, rounds_ok and worker steps grow (SP1's checks).
        With it down (the pair segment): target_only()'s checks; the x86host worker is not probed."""
        cell = cells.CELLS[name]
        ck.require(self.clock() + cell.estimate_s <= self.submit, f'{name}: does not fit before the submission bound')
        before = self.stats(f'{name}-stats-before')
        steps = self.worker_steps(f'{name}-worker-before') if drafter else 0.0
        output = self.out / 'cells' / f'{name}.json'
        output.parent.mkdir(exist_ok=True)
        self.call(f'{name}-cell', self.cen3_driver(
            'run', '--cell', name, '--cuts', str(self.prompts), '--output', str(output), '--deadline-utc',
            f'{self.submit:.3f}', '--telemetry', *(str(self.out / f'telemetry-{host}.jsonl') for host in pf.HOSTS),
            '--capture-floor-kb', str(CAPTURE_FLOOR_KB), '--cache-salt-prefix', self.tag,
            '--ledger', str(self.out / 'requests-ledger.jsonl')), timeout=3 * cell.estimate_s + 600, inflight=cell.batch)
        result = ck.read(output)
        ck.require(result.get('cell') == name and len(cast(list[Json], result['streams'])) == cell.batch,
                   'cell artifact identity mismatch')
        after = self.stats(f'{name}-stats-after')
        row: dict[str, Json] = {'captures': [{k: c.get(k) for k in ('kind', 'status')}
                                             for c in map(record, cast(list[Json], result['captures']))]}
        if drafter:
            ck.counters(after, 0, before)
            ck.require(number(record(after['counters']).get('rounds_ok', 0)) >
                       number(record(before['counters']).get('rounds_ok', 0)), 'no remote drafting during the cell')
            ck.require(self.worker_steps(f'{name}-worker-after') > steps, 'x86host worker took no steps during the cell')
        else:
            row['target_only'] = target_only(before, after)
        with (self.out / 'proxy-stats.jsonl').open('a') as handle:
            handle.write(json.dumps({'cell': name, 'before': before['counters'], 'after': after['counters']}) + '\n')
        if cell.captures:
            row['traces'] = cast(Json, self.collect_traces(name, result))
        return row

    def collect_traces(self, name: str, result: dict[str, Json]) -> list[dict[str, Json]]:
        started = sp1.taken(result)
        rows: list[dict[str, Json]] = []
        for rank, host in enumerate(pf.HOSTS):
            files, free = sp1.listing(self.ssh(f'{name}-list-{host}', host, self.find()))
            new = sp1.new_traces(files, self.collected[host], rank, len(started), free)
            if not new:
                continue
            local = self.out / 'traces' / name / host
            local.mkdir(parents=True, exist_ok=True)
            names = [n_ for n_, _ in new]
            remote = self.json(f'{name}-hash-{host}', host, f'cd ~/{self.run}/profile && python3 - ' + shlex.join(names),
                               pf.IMAGE_BASES)
            for (file, size), capture in zip(new, started):
                self.call(f'{name}-copy-{host}', ['scp', '-q', '-o', 'BatchMode=yes',
                                                  f'{host}:{self.run}/profile/{file}', str(local)])
                got = pf.digest(local / file)
                ck.require(remote.get(file) == got and (local / file).stat().st_size == size, f'{file[:40]} copy differs')
                rows.append({'cell': name, 'capture': capture['kind'], 'status': capture['status'], 'rank': rank,
                             'host': host, 'file': str((local / file).relative_to(self.out)), 'bytes': size, 'sha256': got})
            self.ssh(f'{name}-remove-{host}', host, f'cd ~/{self.run}/profile && rm -f -- ' + shlex.join(names))
            self.collected[host].update(names)
        self.traces.extend(rows)
        (self.out / 'traces-manifest.json').write_text(json.dumps(self.traces, indent=1) + '\n')
        return rows

    def drafter_down(self) -> dict[str, Json]:
        """prod_window.sh pair (x86host's TP2 production worker stopped: target-only rounds), one profiled target-only
        cell, prod_window.sh close (PRODUCTION HOLD OK), then a recovery cell with remote rounds growing again."""
        self.pair_used = True          # BEFORE: a lost reply may still have stopped the worker
        text = self.owned('cen3-pair', 'x86host', f'{self.tag}-cen3-pair',
                          'timeout --signal=TERM --kill-after=15s --foreground 120 bash ~/dsv41-offload/prod_window.sh '
                          'pair COORD && test "$(head -1 ~/dsv41-offload/PAIR_OK)" = COORD && echo CEN3 PAIR OPEN',
                          timeout=180)
        self.pair_open = True
        ck.require('CEN3 PAIR OPEN' in text.splitlines(), 'prod_window.sh pair did not confirm PAIR_OK for COORD')
        out: dict[str, Json] = {'pair': 'OPEN'}
        try:
            out[cells.TARGET_ONLY] = self.run_cell(cells.TARGET_ONLY, drafter=False)
        finally:
            self.pair_close(guarded=True)          # PRODUCTION HOLD OK, or a STOP (restore closes it again)
        out['pair'] = 'CLOSED: PRODUCTION HOLD OK'
        out[cells.RECOVERY] = self.run_cell(cells.RECOVERY, drafter=True)
        return out

    def finish_cen3(self) -> dict[str, Json]:
        checks: dict[str, Json] = {}
        for rank, host in enumerate(pf.HOSTS):
            ident = self.ssh(f'final-id-{host}', host, "docker inspect --format '{{.Id}}' dsv41-serve").strip()
            ck.require(ident == self.ids[rank], 'experiment container changed during the window')
            checks[f'logs_rank{rank}'] = pf.logs(self.json(f'final-logs-{host}', host, 'python3 -c ' +
                shlex.quote(pf.LOG_FILTER) + ' ' + shlex.quote(ck.STOP_LOG.pattern)), rank, 327680)
            files, _ = sp1.listing(self.ssh(f'final-list-{host}', host, self.find()))
            ck.require(not files, f'{host} profile directory still holds files after collection')
        pf.worker(self.json('final-worker', 'x86host', 'python3 -c ' + shlex.quote(X86HOST_PROBE)), self.clock())
        self.traces_done = True
        return checks

    def part_b(self, result: dict[str, Json]) -> dict[str, Json]:
        if self.clock() + ESTIMATE_S['b'] > self.submit:
            return {'status': 'SKIPPED_TIME: the profiled start and its cells do not fit before the submission bound'}
        self.start_cen3()
        result['clock'] = self.clock_cap('cen3')
        out: dict[str, Json] = {'prompts': self.build_prompts()}
        for name in cells.VERIFY:
            out[name] = self.run_cell(name, drafter=True)
        out['drafter_down'] = self.drafter_down()
        out['finish'] = self.finish_cen3()
        out['status'] = 'DONE'
        return out

    # -- end
    def before_restore(self) -> None:
        """Bench containers are dockerd children outside every owned scope: remove them before NBW's quiet proof."""
        if not self.created:
            return
        for host in pf.HOSTS:
            rc, text = self.recovery(f'cen3-rm-{host}', [*pf.SSH, host,
                f'ids=$(docker ps -aq --filter name=^/{CONTAINER}-); if [ -n "$ids" ]; then docker rm -f $ids; fi; '
                f'test -z "$(docker ps -aq --filter name=^/{CONTAINER}-)" && echo CEN3 BENCH ABSENT'], 90)
            ck.require(rc == 0 and 'CEN3 BENCH ABSENT' in text.splitlines(), f'{host}: CEN3 bench container remains')

    def after_restore(self) -> dict[str, Json]:
        """Run directories: sources, JIT build and collected outputs go; uncollected outputs or traces stay as evidence."""
        out: dict[str, Json] = {}
        for host in sorted(self.created):
            d = f'~/{self.run}'
            rc, text = self.plain(f'cen3-dir-clean-{host}', [*pf.SSH, host,
                f'rm -rf -- {d}/src {d}/ext; ' + ' '.join(f'rm -f -- {d}/out/{p.split("/", 1)[1]};'
                                                          for p in sorted(self.bench_collected) if p.startswith(host + '/'))
                + f' rmdir {d}/out {d}/profile 2>/dev/null; rmdir {d} 2>/dev/null; '
                f'if [ -e {d} ]; then echo "LEFT $(find {d} -type f | wc -l) files"; else echo CLEAN; fi'], 60)
            out[f'run_dir_{host}'] = f'rc{rc}: {text.strip()[-80:]}'
        return out


def smoke_launches(files: dict[str, Path]) -> list[BM.Launch]:
    """The ncu smoke's launches from whichever of its outputs holds the CSV report (--log-file, else stdout)."""
    for name in ('ncu-smoke.csv', 'ncu-smoke.app.txt'):
        path = files.get(name)
        if path is None:
            continue
        try:
            launches = BM.read_ncu_csv(path.read_text(errors='replace'))
        except (ValueError, IndexError):
            continue
        if launches:
            return launches
    return []


def body(live: cc.CenLive) -> Callable[[dict[str, Json]], None]:
    w = cast(CEN3Live, live)

    def run(result: dict[str, Json]) -> None:
        parts: dict[str, Json] = {}
        result['parts'] = parts
        result['production_stop'] = w.stop_production()
        w.stage()
        for name, fn in (('c', w.part_c), ('a', w.part_a)):
            if w.clock() + ESTIMATE_S[name] > w.submit:
                parts[name] = {'status': 'SKIPPED_TIME'}
                continue
            parts[name] = fn()
        w.benches_absent('before-b')
        parts['b'] = w.part_b(result)
    return run


def analyze(out: Path) -> dict[str, Json]:
    from offload.census import cen3_analyze, cen4
    result = cen3_analyze.analyze(out)
    result['cen4'] = cen4.analyze(out, analysis=result)
    (out / 'cen3-analysis.json').write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    return {'file': 'cen3-analysis.json', 'parts': result.get('parts'), 'cen4': record(result['cen4']).get('verdict')}


def plan() -> str:
    t = CEN3Live.TIMING
    lines = ['DRY RUN: no command, network or file. Coordinator Deepseek/coordinator only (--execute / --precheck).',
             'SINGLE window, never chained: WINDOW_CHAIN / WINDOW_CHAIN_LAST unset and no chain marker, else STOP.',
             'Preflight (read-only): W8 restore pin; D2 pause; corpus; production ledger (--production-commit/--production-tag,',
             '  default NBW 596ba1c / prod-k1-327-ops1-20261026), clean, no HOLD; experiment checkout = reviewed commit;',
             f'  {len(PIN_FILES)} staged pins incl. {CONFIG}; :8000 K1 327680; x86host worker probe, prod_window.sh present and no',
             f'  PAIR_OK/TRAIN_OK; both Sparks kernel/driver, image, same plan digest, no {CONTAINER}-* container, host Nsight',
             f'  Compute directory recorded; ~/{RUN}/<tag> absent and >= 10 GiB free home disk.',
             f'HOLD: owned admission "CEN3-WINDOW <tag>:" (drain); deadlines from HOLD: submit +{t.submit / 60:.0f}, abort'
             f' +{t.abort / 60:.0f}, restore by +{t.restore / 60:.0f} min (placeholder lease +90). Snapshot production -> rollback.json.',
             'Production stop: ./stop.sh in an owned scope under the HOLD admission; both GPUs idle; MemAvailable >= 40 GiB.',
             f'Stage: ~/{RUN}/<tag>/{{src,out,ext,profile}} on both Sparks; {len(SOURCES)} sources scp + sha256 both sides.',
             f'(c) {CONTAINER}-bw-smoke (64 MiB x1) both Sparks in parallel; failure -> part SKIPPED. Then {CONTAINER}-bw:',
             '  read/write/copy 2 GiB x20 + nvidia-smi clocks/power/temp/event reasons before and after.',
             f'(a) {CONTAINER}-ncu-smoke (root + SYS_ADMIN): ncu in the image, else the host directory read-only at /ncu-host;',
             '  --version, --query-metrics, metric pick (DRAM, else L2 sectors: said so), smoe+mrow JIT into the run directory,',
             '  one profiled mrow launch with a positive read count; failure -> part SKIPPED. Then ' + ', '.join(NCU_GROUPS) + ':',
             '  ncu --profile-from-start off --cache-control all --clock-control none --csv --page raw, one call per case.',
             '  Every part: containers --rm, bounded by timeout; afterwards no cen3 container and no compute app (else STOP).',
             f'(b) CEN3_TAG=<tag> start_exp.sh {CONFIG}: identity == production except port and one reviewed --profiler-config',
             f'  (max_iterations 64); {MOUNT} rw bind; profile directory empty; profiler registered, never started; clock cap',
             '  from the start log (2300 registered; else INCONCLUSIVE). Prompts built. Cells (fresh salt, decode capture each,',
             '  proxy STOP counters unchanged, rounds_ok + worker steps grow): ' + ', '.join(cells.VERIFY) + '.',
             '  Drafter down: x86host prod_window.sh pair COORD (owned) -> ' + cells.TARGET_ONLY + ' (rounds_ok unchanged,',
             '  fallback_not_ready grows, STOP counters unchanged) -> prod_window.sh close (PRODUCTION HOLD OK) -> ' + cells.RECOVERY,
             '  (rounds_ok and worker steps grow). Traces: one new file a rank a capture, copy + O_DIRECT sha256, Spark copy removed.',
             'ALWAYS: fence every owned phase; remove cen3 containers; NBW restore: prod_window.sh close if the pair was opened,',
             '  experiment stop, quiet proof, W8 start/check/close (six IDENTICAL, PRODUCTION HOLD OK), owned HOLD close, watch.',
             'After (CPU, Controller): offload.census.cen3_analyze (waterfall incl. target-only, byte ratios per Spark, bandwidth)',
             '  and offload.census.cen4 (rank imbalance attribution) -> cen3-analysis.json. Traces never published.',
             'ESTIMATED HOLD ~55-60 min (stop+stage 4, c 4, a 12, b start 8 + cells 10 + pair 6, W8 10); production down ~55 min.']
    return '\n'.join(lines)


ENTRY = cc.Entry(
    description=__doc__ or 'CEN3 window',
    plan=plan,
    make=lambda out, sup, args, tag: CEN3Live(out, sup, remote=args.remote_repo,
                                              expected=args.expected_experiment_commit, data_dir=args.data_dir, tag=tag,
                                              production_commit=args.production_commit,
                                              production_tag=args.production_tag),
    body=body, config=CONFIG, analyze=analyze)


def main(argv: list[str] | None = None) -> int:
    return cc.main(ENTRY, argv)


if __name__ == '__main__':
    raise SystemExit(main())
