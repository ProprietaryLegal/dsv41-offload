"""KL1 synthetic kernel window. Dry by default; coordinator only. Always remove bench containers, then W8 restore.

No serving experiment is started. PF1 owns preflight, production snapshot/drain, telemetry and exact W8 checks.
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
from pathlib import Path
from typing import Protocol, cast

from offload.kl1.smoe_bench import VARIANTS
from offload.launch.windows import kl1_remote as kr
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, record

SOURCES = ('offload/__init__.py', 'offload/kl1/__init__.py', 'offload/kl1/shapes.py', 'offload/kl1/smoe_bench.py')
PIN_FILES = (*pf.PIN_FILES, *SOURCES, 'offload/launch/windows/kl1_window.py', 'offload/launch/windows/kl1_remote.py',
             'offload/launch/windows/kl1_window.sh')
RUN_DIR = 'dsv41-offload-run/kl1'
CONTAINER = 'dsv41-kl1-bench'
PLAN = '/srv/dsv41/legal-v2a/exl3_moe_plan.json'
KERNEL, DRIVER = '7.0.0-1019-nvidia', '580.178.04'
ABORT_S, RESTORE_S, BENCH_S = 40 * 60, 55 * 60, 1500
PRE_MEMORY_KB = 40 * 1024**2
PRE_DISK_KB = 10 * 1024**2
FILE_CAP = 16 * 1024**2
RESULT_NAMES = {'prod-first.json', 'prod-last.json', 'fused.json', 'dense.json', 'failures.json',
                *(v.name + '.json' for v in VARIANTS if v.name != 'prod')}
IDLE = 'nvidia-smi --query-compute-apps=pid --format=csv,noheader'


def host_check(text: str, floor: int) -> dict[str, Json]:
    lines = text.splitlines()
    ck.require(lines[:2] == [KERNEL, DRIVER], 'KL1 requires post-FW1/FW2 kernel 7.0.0-1019-nvidia / driver 580.178.04')
    available = re.findall(r'^MemAvailable:\s+(\d+) kB$', text, re.M)
    ck.require(len(available) == 1 and int(available[0]) >= floor, f'MemAvailable below {floor / 1024**2:g} GiB')
    return {'kernel': lines[0], 'driver': lines[1], 'mem_available_kb': int(available[0])}


def bench_script(rank: int, tag: str) -> str:
    """In-container script. Separate processes read build flags once; failed variants are retained as evidence."""
    # The CUDA allocator cap (24 GiB) bounds GPU allocations as well as the Docker 32 GiB host-memory cgroup.
    py = ('import runpy,torch; torch.cuda.set_per_process_memory_fraction('
          '24*1024**3/torch.cuda.get_device_properties(0).total_memory); '
          'runpy.run_module("offload.kl1.smoe_bench",run_name="__main__")')
    lines = ['set -euo pipefail', "trap 'rm -rf -- /ext/*' EXIT", 'export HOME=/ext TORCH_EXTENSIONS_DIR=/ext/jit',
             "printf '[]\\n' > /kl1-out/failures.json",
             'failed() { python3 -c ' + shlex.quote('import json,sys; p="/kl1-out/failures.json"; '
                 'x=json.load(open(p)); x.append({"step":sys.argv[1],"rc":int(sys.argv[2])}); '
                 'json.dump(x,open(p,"w"))') + ' "$1" "$2"; }']
    # Bounded batches, with every child's status observed (plain `wait` can hide the first failed build).
    for start in range(0, len(VARIANTS), 3):
        batch = VARIANTS[start:start + 3]
        for i, v in enumerate(batch):
            lines.append(f'EXL3_SMOE_FLAGS={shlex.quote(v.flags)} python3 -c '
                         + shlex.quote('from vllm_exl3_moe import smoe; smoe.ext()') + f' & p{i}=$!')
        for i, v in enumerate(batch):
            lines.append(f'wait "$p{i}" || failed build-{v.name} "$?"')
    for module in ('mrow', 'router', 'l2pf'):
        lines.append('python3 -c ' + shlex.quote(f'from vllm_exl3_moe import {module}; {module}.ext()')
                     + f' || failed build-{module} "$?"')
    sequence = [('prod-first', VARIANTS[0]), *((v.name, v) for v in VARIANTS[1:])]
    for name, variant in sequence:
        lines.append(f'EXL3_SMOE_FLAGS={shlex.quote(variant.flags)} python3 -c {shlex.quote(py)} gpu '
                     f'--variant {variant.name} --plan /plan.json --rank {rank} --out /kl1-out/{name}.json '
                     f'|| failed {name} "$?"')
    lines += [f'EXL3_SMOE_FLAGS=-DSMOE_RUNFLAGS=1 python3 -c {shlex.quote(py)} gpu-fused --plan /plan.json '
              f'--rank {rank} --out /kl1-out/fused.json || failed fused "$?"',
              f'python3 -c {shlex.quote(py)} gpu-dense --rank {rank} --out /kl1-out/dense.json || failed dense "$?"',
              f'EXL3_SMOE_FLAGS=-DSMOE_RUNFLAGS=1 python3 -c {shlex.quote(py)} gpu --variant prod --plan /plan.json '
              f'--rank {rank} --out /kl1-out/prod-last.json || failed prod-last "$?"']
    stamp = ('import glob,json,os; '
             'meta={"window":os.environ["KL1_WINDOW"],"kernel":os.environ["KL1_KERNEL"],'
             '"driver":os.environ["KL1_DRIVER"],"rank":int(os.environ["KL1_RANK"])}; '
             '\nfor p in glob.glob("/kl1-out/*.json"):\n'
             ' if p.endswith("failures.json"): continue\n'
             ' d=json.load(open(p)); d.update(meta); '
             'd["sample"]="first" if p.endswith("prod-first.json") else "last" if p.endswith("prod-last.json") else "variant"; '
             'json.dump(d,open(p,"w"))')
    lines.append('python3 -c ' + shlex.quote(stamp))
    return '\n'.join(lines)


def remote_bench(rank: int, tag: str) -> str:
    script = bench_script(rank, tag)
    # $HOME is evaluated on EACH host; --mount refuses a missing source rather than creating a root-owned directory.
    # The persistent fence is outside the disposable run directory. Recheck immediately before Docker.
    fence = f'"$HOME/{kr.CONTROL}/{tag}/ABORT"'
    docker = (f'cd ~/{RUN_DIR} && test ! -e {fence} && exec docker run --rm --init '
            f'--name {CONTAINER} --gpus all --network none --user "$(id -u):$(id -g)" '
            '--memory=32g --memory-swap=32g --shm-size=1g --pids-limit=1024 '
            f'--mount "type=bind,src=$HOME/{RUN_DIR}/src,dst=/kl1,readonly" '
            f'--mount "type=bind,src=$HOME/{RUN_DIR}/out,dst=/kl1-out" '
            f'--mount "type=bind,src=$HOME/{RUN_DIR}/ext,dst=/ext" '
            f'--mount type=bind,src={PLAN},dst=/plan.json,readonly '
            '-e PYTHONPATH=/kl1 -e PYTHONDONTWRITEBYTECODE=1 -e MAX_JOBS=4 '
            f'-e KL1_WINDOW={shlex.quote(tag)} -e KL1_RANK={rank} -e KL1_KERNEL={KERNEL} -e KL1_DRIVER={DRIVER} '
            f'-w /kl1 --entrypoint bash {pf.IMAGE} -c {shlex.quote(script)}')
    # --foreground keeps timeout, bash and Docker in the registered group so remote abort can wait for all of them.
    return kr.command('launch', tag, ['timeout', '--signal=TERM', '--kill-after=15s', '--foreground', f'{BENCH_S}s',
                                     'bash', '-c', docker])


class KL1Live(pf.Live):
    def __init__(self, out: Path, sup: pf.Supervisor, *, remote: str, expected: str, data_dir: Path, tag: str,
                 stage_bundle: bool, call: pf.Call | None = None,
                 plain: Callable[[str, list[str], float], tuple[int, str]] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        super().__init__(out, sup, remote=remote, expected=expected, data_dir=data_dir, tag=tag,
                         stage_bundle=stage_bundle, call=call, plain=plain, clock=clock)
        self.created: set[str] = set()
        self.stopped = False
        self.restore_by = float('inf')
        self.hosts: dict[str, Json] = {}
        self.plan_sha = ''

    @property
    def pin_files(self) -> tuple[str, ...]:
        return PIN_FILES

    def hold_purpose(self) -> str:
        return f'KL1-WINDOW {self.tag}: bounded synthetic kernel bench; no serving/weight/production JIT changes'

    def reap_script(self) -> str:
        # No start_exp.sh in this window: never reap another window's experiment.
        pattern = f'window[.]sh open KL1-WINDOW {self.tag}:'
        return pf.REAP.replace('PF1-WINDOW @TAG@:', f'KL1-WINDOW {self.tag}:').replace(
            'start_exp[.]sh offload/launch/configs/PF1-window[.]env', pattern)

    def preflight(self) -> None:
        super().preflight()
        hashes: list[str] = []
        for host in pf.HOSTS:
            text = self.ssh(f'kl1-dir-{host}', host,
                f'test ! -e ~/{RUN_DIR} && test ! -e ~/{kr.CONTROL}/{self.tag} && '
                f'test -z "$(docker ps -aq --filter name=^/{CONTAINER}$)" && '
                'case "$(findmnt -n -o FSTYPE -T ~)" in ext4|xfs|btrfs|zfs) ;; *) exit 1;; esac && '
                'if [ -e ~/dsv41-offload-run ]; then test ! -L ~/dsv41-offload-run && '
                'case "$(findmnt -n -o FSTYPE -T ~/dsv41-offload-run)" in ext4|xfs|btrfs|zfs) ;; *) exit 1;; esac; fi && '
                'df -Pk ~ | tail -1')
            ck.require(int(text.split()[-3]) >= PRE_DISK_KB, f'{host} home has < 10 GiB free')
            self.ssh(f'kl1-image-{host}', host, f'docker image inspect {pf.IMAGE} >/dev/null')
            hashed = self.json(f'kl1-plan-{host}', host, 'python3 - ' + shlex.quote(PLAN), pf.IMAGE_BASES)
            sha = str(hashed.get(PLAN, ''))
            ck.require(re.fullmatch('[0-9a-f]{64}', sha) is not None, f'{host} missing plan digest')
            hashes.append(sha)
            self.hosts[host] = host_check(self.ssh(f'kl1-host-{host}', host, pf.HOST_PARITY), pf.START_FLOOR_KB)
        ck.require(len(set(hashes)) == 1, 'Spark K plan digests differ')
        self.plan_sha = hashes[0]
        (self.out / 'preflight.json').write_text(json.dumps({'hosts': self.hosts, 'plan_sha256': self.plan_sha}) + '\n')

    def open_hold(self) -> float:
        # Repeat the no-HOLD check immediately before open (window.sh itself would overwrite it).
        self.ssh('kl1-no-hold', 'sparkhead', 'test ! -e ~/dsv41-deploy/HOLD')
        # PF1 opens/drains and reads back the timestamp; this bench never needs an API tunnel.
        try:
            self.ssh('hold-open', 'sparkhead', f'cd ~/dsv41-deploy && test ! -e HOLD && timeout {pf.DRAIN_S} offload/launch/window.sh open '
                     + shlex.quote(self.hold_purpose()), timeout=pf.DRAIN_S + 60)
        finally:
            text = self.ssh('hold-read', 'sparkhead', 'cat ~/dsv41-deploy/HOLD 2>/dev/null || true')
            self.hold_written = self.tag in text
            if self.hold_written:
                opened = pf.hold_opened(text, self.tag, self.clock())
                self.sup.abort_at, self.restore_by = opened + ABORT_S, opened + RESTORE_S
                self.sup.bound = '40-minute KL1 abort bound'
                (self.out / 'deadlines.json').write_text(json.dumps({'opened': opened, 'abort': self.sup.abort_at,
                    'restore_by': self.restore_by}) + '\n')
        return pf.hold_opened(text, self.tag, self.clock())

    def stop(self) -> None:
        self.stopped = True   # a lost SSH reply may still have stopped production
        self.ssh('kl1-stop', 'sparkhead', 'cd ~/dsv41-deploy && ./stop.sh', timeout=120)
        for host in pf.HOSTS:
            ck.require(not self.ssh(f'kl1-idle-{host}', host, IDLE).strip(), f'{host} still has compute apps after stop')
            host_check(self.ssh(f'kl1-memory-{host}', host, pf.HOST_PARITY), PRE_MEMORY_KB)
        self.sup.arm_minima = {}
        self.sup.arm_memory['kl1-bench'] = self.sup.arm_minima

    def stage(self) -> None:
        for host in pf.HOSTS:
            self.ssh(f'kl1-mkdir-{host}', host, f'mkdir -p ~/dsv41-offload-run && mkdir ~/{RUN_DIR} && '
                     f'mkdir -p ~/{RUN_DIR}/src/offload/kl1 ~/{RUN_DIR}/out ~/{RUN_DIR}/ext')
            self.created.add(host)
            for path in SOURCES:
                target = f'{host}:{RUN_DIR}/src/{path}'
                self.call(f'kl1-source-{host}', ['scp', '-q', '-o', 'BatchMode=yes', str(pf.ROOT / path), target])
            text = self.ssh(f'kl1-source-hash-{host}', host, f'cd ~/{RUN_DIR}/src && sha256sum ' + shlex.join(SOURCES))
            ck.require(text.strip() == '\n'.join(f'{pf.digest(pf.ROOT / p)}  {p}' for p in SOURCES),
                       f'{host} staged source hashes differ')

    def bench(self) -> None:
        commands = [shlex.join([*pf.SSH, host, remote_bench(rank, self.tag)]) for rank, host in enumerate(pf.HOSTS)]
        # Wait on BOTH children and preserve either failure; Supervisor watches one owned local process group.
        shell = f'{commands[0]} & a=$!\n{commands[1]} & b=$!\nrc=0; wait "$a" || rc=$?; wait "$b" || rc=$?; exit "$rc"'
        self.call('kl1-bench', ['bash', '-c', shell], timeout=BENCH_S + 60)

    def collect_results(self) -> dict[str, Json]:
        manifest: list[Json] = []
        failures: dict[str, Json] = {}
        for host in pf.HOSTS:
            listed = self.ssh(f'kl1-results-{host}', host,
                f'cd ~/{RUN_DIR}/out && find . -mindepth 1 -maxdepth 1 -printf "%y %s %f\\n"')
            files: dict[str, int] = {}
            for line in listed.splitlines():
                kind, size_text, name = line.split()
                ck.require(kind == 'f' and name in RESULT_NAMES and 0 < int(size_text) <= FILE_CAP,
                           f'{host} unexpected or oversized result {name}')
                ck.require(name not in files, 'duplicate result filename')
                files[name] = int(size_text)
            ck.require(bool(files), f'{host} has no result JSONs')
            hashes = self.json(f'kl1-result-hashes-{host}', host,
                f'cd ~/{RUN_DIR}/out && python3 - ' + shlex.join(sorted(files)), pf.IMAGE_BASES)
            local = self.out / 'results' / host
            local.mkdir(parents=True)
            for name, size in sorted(files.items()):
                self.call(f'kl1-result-copy-{host}', ['scp', '-q', '-o', 'BatchMode=yes',
                    f'{host}:{RUN_DIR}/out/{name}', str(local)])
                sha = pf.digest(local / name)
                ck.require(hashes.get(name) == sha and (local / name).stat().st_size == size, f'{host}/{name} copy differs')
                manifest.append({'host': host, 'file': str((local / name).relative_to(self.out)), 'bytes': size, 'sha256': sha})
            # Delete nothing from this node until ALL its copies are verified.
            self.ssh(f'kl1-results-remove-{host}', host, f'cd ~/{RUN_DIR}/out && rm -- ' + shlex.join(sorted(files)))
            failures[host] = cast(Json, json.loads((local / 'failures.json').read_text())) if 'failures.json' in files else ['missing failures.json']
            if not {'prod-first.json', 'prod-last.json'} <= files.keys():
                failures[host] = [failures[host], 'missing production endpoint']
        (self.out / 'results-manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
        return {'files': manifest, 'failures': failures}

    def restore(self, *, production_stopped: bool) -> dict[str, Json]:
        # Local SSH group closure alone cannot end a remote SSH-server session (S47).
        # window() masks signals BEFORE this close (S40), including a second signal here.
        self.sup.close()
        cleanup: dict[str, Json] = {}
        plain = self.plain
        def bounded(label: str, command: list[str], timeout: float) -> tuple[int, str]:
            left = self.restore_by - self.clock()
            ck.require(left > 0, 'KL1 HOLD+55 restore deadline exhausted; manual restore required')
            return plain(label, command, min(timeout, left))
        errors: list[str] = []
        fences: dict[str, Json] = {}
        # Fence BOTH nodes before any reap/removal. Registration takes the same lock remotely; a late SSH session
        # either registers before ABORT (and is reaped) or observes ABORT and cannot launch. Never delete these fences.
        for host in pf.HOSTS:
            try:
                rc, text = bounded(f'kl1-fence-{host}', [*pf.SSH, host, kr.command('fence', self.tag)], 30)
                fences[host] = {'rc': rc, 'text': text.strip()[-300:]}
                ck.require(rc == 0 and text.splitlines()[-1:] == [kr.FENCED], f'{host} remote fence failed')
            except (Exception, KeyboardInterrupt) as exc:
                errors.append(f'{host}: {exc}')
        (self.out / 'fences.json').write_text(json.dumps(fences, indent=1) + '\n')
        # Attempt both nodes even if one fails. Fail closed before W8 start if GPU cleanup cannot be proved.
        marker = 'KL1 CLEAN GPU IDLE' if production_stopped or self.stopped else 'KL1 CLEAN BENCH ABSENT'
        for host in pf.HOSTS:
            command = ('set -e\n' + kr.command('reap', self.tag) + '\n' +
                       f'docker rm -f {CONTAINER} 2>&1 || true\n'
                       f'n=$(docker ps -aq --filter name=^/{CONTAINER}$) || exit; '
                       'test -z "$n" || exit 1; ' +
                       (f'apps=$({IDLE}) || exit; test -z "$apps" || exit 1; ' if production_stopped or self.stopped else '') +
                       f'echo "{marker}"')
            try:
                rc, text = bounded(f'kl1-rm-{host}', [*pf.SSH, host, command], 60)
                cleanup[host] = {'rc': rc, 'text': text.strip()[-300:]}
                ck.require(rc == 0 and kr.ABSENT in text.splitlines() and text.splitlines()[-1:] == [marker],
                           f'{host} launcher/container/GPU cleanup failed')
            except (Exception, KeyboardInterrupt) as exc:
                errors.append(f'{host}: {exc}')
        (self.out / 'cleanup.json').write_text(json.dumps(cleanup, indent=1) + '\n')
        ck.require(not errors, '; '.join(errors))
        self.plain = bounded
        try:
            result = super().restore(production_stopped=production_stopped or self.stopped)
        finally:
            self.plain = plain
        result['container_cleanup'] = cleanup
        result['remote_fences'] = fences
        result['restore_deadline_met'] = self.clock() <= self.restore_by
        ck.require(bool(result['restore_deadline_met']), 'KL1 restore completed after HOLD+55')
        # Isolated user-owned build cache only; never touch production cache. Keep uncollected result evidence.
        for host in sorted(self.created):
            rc, text = plain(f'kl1-dir-clean-{host}', [*pf.SSH, host,
                f'rm -rf -- ~/{RUN_DIR}/ext ~/{RUN_DIR}/src && rmdir ~/{RUN_DIR}/out ~/{RUN_DIR}'], 60)
            result[f'directory_{host}'] = f'rc{rc}: {text.strip()[-160:]}'
        (self.out / 'arm-memory.json').write_text(json.dumps(self.sup.arm_memory) + '\n')
        return result


class Backend(Protocol):
    def preflight(self) -> None: ...
    def open_hold(self) -> float: ...
    def snapshot(self) -> None: ...
    def stop(self) -> None: ...
    def stage(self) -> None: ...
    def bench(self) -> None: ...
    def collect_results(self) -> dict[str, Json]: ...
    def restore(self, *, production_stopped: bool) -> dict[str, Json]: ...


def window(b: Backend, checkpoint: Callable[[dict[str, Json]], None]) -> dict[str, Json]:
    result: dict[str, Json] = {'status': 'INCOMPLETE'}
    hold = stopped = False
    try:
        b.preflight()
        hold = True
        result['hold_opened'] = b.open_hold()
        b.snapshot()
        stopped = True
        b.stop()
        b.stage()
        b.bench()
        result['results'] = b.collect_results()
        checkpoint({'results_collected': True})
        ck.require(not any(record(record(result['results'])['failures']).values()), 'bench reported failures; inspect result JSONs')
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
                result['restore'] = {'status': 'NOT NEEDED: no HOLD written, production untouched'}
        finally:
            for sig, handler in saved.items():
                signal.signal(sig, handler)
    return result


def plan() -> str:
    return ('DRY RUN: no command, network or file. Coordinator Deepseek/coordinator only.\n'
            'Precheck: inherited PF1 preflight, no HOLD, reviewed clean checkout/pins, image, equal O_DIRECT plan hashes,\n'
            '  node-local home disk >=10 GiB; both post-FW1/FW2 7.0.0-1019-nvidia / 580.178.04.\n'
            'HOLD + drain; inherited snapshot + idle; explicit ./stop.sh; both GPUs idle, MemAvailable >=40 GiB.\n'
            'Stage sha256-checked sources on each node own disk. Parallel named dsv41-kl1-bench containers, timeout 1500s,\n'
            '  host 32 GiB/no swap, GPU allocator 24 GiB; read-only single-file plan; user-owned private build cache.\n'
            'Prebuild smoe/mrow/router/l2pf; prod first/last; copy results with size/sha256 checks before deleting.\n'
            'Guards throughout: MemAvailable >=4 GiB, STOP, telemetry liveness, abort HOLD+40; restore by HOLD+55.\n'
            'ALWAYS: fence BOTH remote launchers, reap/wait this run only, prove launchers absent;\n'
            '  remove named containers on BOTH nodes, prove GPU idle, W8/restore.sh start|check|close: \n'
            '  K1 START VERIFIED; six greedy K1-post vs restored: N IDENTICAL M; PRODUCTION HOLD OK; CHECK: DONE; CLOSE: window closed.\n'
            'Production down ~34–44 min ESTIMATED. --precheck is read-only and opens no HOLD.')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--precheck', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--remote-repo', default='dsv41-deploy-offload')
    ap.add_argument('--expected-experiment-commit', default='')
    ap.add_argument('--stage-bundle', action='store_true')
    ap.add_argument('--data-dir', type=Path, default=Path('/opt/dsv41/data/bench-data'))
    args = ap.parse_args(argv)
    if not (args.execute or args.precheck):
        print(plan())
        return 0
    try:
        ck.require(os.environ.get('WINDOW_COORDINATOR') == 'Deepseek/coordinator', 'WINDOW_COORDINATOR=Deepseek/coordinator required')
        ck.require(args.out is not None and args.out.resolve().parent == pf.CAMPAIGN and not args.out.exists(),
                   'fresh --out directly inside campaign directory required')
        ck.require(re.fullmatch('[0-9a-f]{40}', args.expected_experiment_commit) is not None, 'full reviewed commit required')
        ck.require(re.fullmatch('[A-Za-z0-9_.-]+', args.remote_repo) is not None and args.remote_repo != 'dsv41-deploy',
                   'experiment checkout must be separate, never ~/dsv41-deploy')
        ck.require(not (args.precheck and args.stage_bundle), '--precheck cannot stage a bundle')
        ck.require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=pf.ROOT, text=True).strip() ==
                   args.expected_experiment_commit, 'local checkout is not the reviewed commit')
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
    live = KL1Live(out, sup, remote=args.remote_repo, expected=args.expected_experiment_commit, data_dir=args.data_dir,
                   tag=tag, stage_bundle=args.stage_bundle)
    if args.precheck:
        try:
            live.preflight()
            result: dict[str, Json] = {'status': 'PRECHECK PASS', 'restore': {'status': 'NOT NEEDED: read-only precheck'}}
        except (Exception, KeyboardInterrupt) as exc:
            result = {'status': 'PRECHECK FAILED', 'stop': f'{type(exc).__name__}: {exc}',
                      'restore': {'status': 'NOT NEEDED: read-only precheck'}}
        finally:
            sup.close()
    else:
        result = window(live, checkpoint)
        sup.close()
    result.update({'experiment_commit': args.expected_experiment_commit, 'restore_sha256': pf.RESTORE_SHA256,
                   'artifact_sha256': {str(p.relative_to(out)): pf.digest(p) for p in sorted(out.rglob('*.json*'))
                                       if p.is_file() and not p.name.startswith('telemetry-')}})
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
