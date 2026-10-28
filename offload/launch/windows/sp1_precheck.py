"""SP1-WINDOW read-only precheck (coordinator Deepseek / coordinator only; the SP1-WINDOW lane never runs it).

Run on Controller from the reviewed checkout, with production serving, before `sp1_window.sh --execute`:

    bash offload/launch/windows/sp1_precheck.sh <reviewed 40-hex commit> [--out <planned window directory>]

Reads only. No HOLD, no start/stop, no request to :8000 except /health, /v1/models and /metrics GETs; no request to
:8001/:18001 or to the x86host worker (its status file and /proc only); `docker exec` runs hash-only or source-reading
scripts in the production containers (no vllm/torch import); nothing is written anywhere (stdout only). Every check runs
and prints PASS/FAIL/INFO; the exit status is 1 if any FAIL. The checks are the runner's own preflight checks
(sp1_window.SP1Live.preflight on top of PF1's) plus the image's profiler fields, run before the HOLD so a failure costs
no production time. Nsight is reported as INFO only (PROFILE-WINDOW.md section 6).
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

from offload.launch.windows import pf1_precheck as pc
from offload.launch.windows import pf1_window as w
from offload.launch.windows import sp1_window as s
from offload.launch.windows import up2_checks as ck
from offload.launch.windows.up2_low import X86HOST_PROBE
from offload.prefill_maxlen.evidence import Json, record
from offload.prefill_trace.emit_verify import render as verifier
from offload.prefill_trace.telemetry import FIELDS

# Reads vllm/config/profiler.py and the profile router as text (find_spec does not import vllm); versions from metadata.
PROFILER_PROBE = '''import ast,importlib.metadata as md,importlib.util,json,pathlib
root=pathlib.Path(list(importlib.util.find_spec('vllm').submodule_search_locations)[0])
tree=ast.parse((root/'config/profiler.py').read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ProfilerConfig')
fields=sorted(n.target.id for n in cls.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name))
router=(root/'entrypoints/serve/profile/api_router.py').read_text()
def version(name):
    try: return md.version(name)
    except md.PackageNotFoundError: return None
print(json.dumps({'fields':fields,'endpoints':[e for e in ('/start_profile','/stop_profile') if e in router],
                  'vllm':version('vllm'),'torch':version('torch')}))
'''
NSYS = "command -v nsys || echo none; docker exec dsv41-serve sh -c 'command -v nsys || ls /usr/local/cuda/bin/nsys 2>/dev/null || echo none'"


def precheck(expected: str, out: Path | None, remote: pc.Remote, *, root: Path = w.ROOT, now: Callable[[], float] = time.time,
             data_dir: Path = Path('/opt/dsv41/data/bench-data'), listening: Callable[[], str] = pc.sockets) -> int:
    r = pc.Report()
    git = ['git', '-C', str(root)]
    ok = pc.ok

    # -- Controller (local)
    def local_commit() -> str:
        head = subprocess.check_output([*git, 'rev-parse', 'HEAD'], text=True).strip()
        ck.require(head == expected, f'local HEAD {head} is not the reviewed commit')
        ck.require(not subprocess.check_output([*git, 'status', '--porcelain', '--untracked-files=no'], text=True).strip(),
                   'local tracked source is dirty')
        return head[:12]
    r.check('controller reviewed checkout', local_commit)

    def restore() -> str:
        ck.require(w.digest(w.RESTORE) == w.RESTORE_SHA256, 'restore.sh changed; re-review its STOP strings before use')
        return w.RESTORE_SHA256[:12]
    r.check('controller W8 restore.sh pin', restore)

    def d2() -> str:
        if (w.CAMPAIGN / 'D2-GEN.PAUSE').exists():
            return 'D2-GEN.PAUSE present'
        active = subprocess.run(['tmux', 'has-session', '-t', '=d2-gen'], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL).returncode == 0
        ck.require(not active, 'D2 generation (tmux d2-gen) active and not paused')
        return 'no d2-gen session'
    r.check('controller D2 paused or absent', d2)

    def corpus() -> str:
        from offload.prefill_trace.window_requests import corpus as read
        return ', '.join(f'{k} {v!s:.12}' for k, v in read(data_dir)[1].items())
    r.check('controller public corpus', corpus)

    def tunnel_port() -> str:
        ck.require(re.search(r':18001\s', listening()) is None, '127.0.0.1:18001 already listening (stale tunnel?)')
        return '18001 free'
    r.check('controller tunnel port', tunnel_port)

    def out_dir() -> str:
        target = cast(Path, out)
        ck.require(target.resolve().parent == w.CAMPAIGN and not target.exists(), f'{target} must be new, directly in the campaign')
        return str(target)
    if out is not None:
        r.check('controller --out', out_dir)

    # -- sparkhead (head, rank 0): production ledger and the separate experiment checkout
    r.check('sparkhead production ledger', lambda: ok(remote, 'sparkhead', 'cd ~/dsv41-deploy && test ! -e HOLD && '
        f'test "$(git rev-parse --short=7 HEAD)" = {w.PROD_COMMIT} && '
        f'test "$(git rev-parse HEAD)" = "$(git rev-list -n1 {w.PROD_TAG})" && '
        'git diff --exit-code --quiet HEAD -- dsv41.env serve.sh start.sh && echo "$(git rev-parse --short=12 HEAD) clean, no HOLD"',
        'production checkout is not 21bed31/tag, is dirty, or a HOLD exists').strip())

    staged = {'at_reviewed': False}
    def checkout() -> str:
        rc, text = remote('sparkhead', 'cd ~/dsv41-deploy-offload && echo "$(git rev-parse HEAD)" && '
                          'git status --porcelain --untracked-files=no | wc -l', '')
        ck.require(rc == 0, f'~/dsv41-deploy-offload unreadable: {text.strip()[-200:]}')
        head, dirty = text.split()[:2]
        ck.require(dirty == '0', 'experiment checkout has tracked changes; neither the runner nor --stage-bundle will use it')
        staged['at_reviewed'] = head == expected
        return 'at the reviewed commit' if head == expected else f'at {head[:12]}: run with --stage-bundle'
    r.check('sparkhead experiment checkout', checkout)

    def pins() -> str:
        text = ok(remote, 'sparkhead', 'cd ~/dsv41-deploy-offload && sha256sum ' + shlex.join(s.PIN_FILES), 'pin read')
        ck.require(text.strip() == '\n'.join(f'{w.digest(root / p)}  {p}' for p in s.PIN_FILES), 'staged sources differ')
        return f'{len(s.PIN_FILES)} files equal, incl. {s.CONFIG}'
    if staged['at_reviewed']:
        r.check('sparkhead staged pins', pins)
    else:
        r.line('INFO', 'sparkhead staged pins', 'checked by the runner after --stage-bundle')

    def production() -> str:
        ok(remote, 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8000/health', ':8000 /health')
        models = ck.rows(json.loads(ok(remote, 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8000/v1/models',
                                       ':8000 /v1/models'))['data'])
        ck.require(len(models) == 1 and models[0].get('max_model_len') == 327680 and
                   models[0].get('root') == '/srv/dsv41/legal-v2a', 'production is not K1 327680 legal-v2a')
        return 'K1 327680 legal-v2a'
    r.check('sparkhead production serving', production)

    def idle() -> str:
        text = ok(remote, 'sparkhead', 'curl -fsS --max-time 5 http://localhost:8000/metrics', ':8000 /metrics')
        busy = [float(x.split()[-1]) for x in text.splitlines()
                if x.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{'))]
        ck.require(len(busy) >= 2, 'request gauges missing')
        return f'running+waiting {sum(busy):g} (window.sh open drains; the runner waits at most 20 min)'
    r.check('sparkhead production load', idle)

    for host in w.HOSTS:
        def node(host: str = host) -> str:
            text = ok(remote, host, "docker inspect --format '{{.Image}} {{.State.Status}}' dsv41-serve && "
                      "(ss -Hltn | grep -c ':8001 ' || true) && (tmux has-session -t dsv41-hold 2>/dev/null && echo hold || echo nohold)",
                      'container inspect')
            image, state, port, hold = text.split()[:4]
            ck.require(image == w.IMAGE and state == 'running', f'dsv41-serve is {image[:19]} {state}')
            ck.require(port == '0', ':8001 already listening (an experiment is running?)')
            ck.require(hold == 'nohold', 'dsv41-hold placeholder session exists (stale window?)')
            return 'production image running; :8001 free; no hold placeholder'
        r.check(f'{host} container', node)

        def pins13(host: str = host) -> str:
            text = ok(remote, host, 'docker exec -i dsv41-serve python3 -', 'in-container pin verify', verifier(False))
            ck.require('PF1 verified 13 engine sources' in text, 'production in-container sources differ from the 13 pins')
            bases = json.loads(ok(remote, host, 'docker exec -i dsv41-serve python3 - ' + shlex.join(w.image_pins()),
                                  'image trace bases', w.IMAGE_BASES).strip().splitlines()[-1])
            ck.require(bases == w.image_pins(), 'image trace bases differ from BASES.json; rebuild/review, never force')
            return '13 pins; image trace bases equal BASES.json'
        r.check(f'{host} production sources', pins13)

        def profiler(host: str = host) -> str:
            found = record(cast(Json, json.loads(ok(remote, host, 'docker exec -i dsv41-serve python3 -', 'profiler source read',
                                                    PROFILER_PROBE).strip().splitlines()[-1])))
            missing = sorted(set(s.PROFILER_CONFIG) - set(cast(list[str], found['fields'])))
            ck.require(not missing, f'image ProfilerConfig lacks {missing}: the reviewed --profiler-config would fail')
            ck.require(found['endpoints'] == ['/start_profile', '/stop_profile'], 'profile endpoints missing from the image')
            return f"{len(s.PROFILER_CONFIG)} reviewed fields present; endpoints present; vllm {found['vllm']}, torch {found['torch']}"
        r.check(f'{host} image profiler', profiler)

        def telemetry(host: str = host) -> str:
            rows = pc.objects(ok(remote, host, 'python3 -c ' + shlex.quote(w.TELEMETRY) + ' ' + shlex.quote(FIELDS) + ' 1',
                                 'telemetry probe'))
            ck.require([x.get('kind') for x in rows] == ['memory', 'gpu'], 'nvidia-smi fields unsupported; review telemetry')
            w.gpu_row(rows[1])
            available, swap = w.memory_row(rows[0])
            ck.require(available >= w.START_FLOOR_KB, f'MemAvailable {available / 2**20:.1f} GiB < 5 GiB')
            return f'MemAvailable {available / 2**20:.1f} GiB, SwapFree {swap / 2**20:.1f} GiB'
        r.check(f'{host} memory and GPU telemetry', telemetry)

        def profile_dir(host: str = host) -> str:
            text = ok(remote, host, f'if [ -e ~/{s.PROFILE_DIR} ]; then echo PRESENT; else echo ABSENT; fi; df -Pk ~ | tail -1',
                      'profile directory and disk')
            state, *df = text.split()
            ck.require(state == 'ABSENT', f'~/{s.PROFILE_DIR} exists (an earlier window left traces?): the runner would STOP')
            free = int(df[3])
            ck.require(free >= s.PRE_DISK_KB, f'only {free / 2**20:.0f} GiB free in home (< {s.PRE_DISK_KB >> 20})')
            return f'~/{s.PROFILE_DIR} absent; {free / 2**20:.0f} GiB free'
        r.check(f'{host} profile directory and disk', profile_dir)

        rc, text = remote(host, NSYS, '')
        r.line('INFO', f'{host} Nsight', f'host/container: {" | ".join(text.split()) if rc == 0 else f"rc {rc}"} (not used)')

    # -- x86host: the TP2 drafter worker, read from its status file and /proc only (no request to the worker)
    def drafter() -> str:
        probe = record(cast(Json, json.loads(ok(remote, 'x86host', 'python3 -c ' + shlex.quote(X86HOST_PROBE), 'worker probe'))))
        return f'{w.WORKER[:7]} TP2, HELLO bound 1048576, steps {w.worker(probe, now()).get("steps")}'
    r.check('x86host TP2 worker', drafter)

    def drafter_gpus() -> str:
        found = record(cast(Json, json.loads(ok(remote, 'x86host', 'python3 -c ' + shlex.quote(pc.GPU_APPS), 'compute apps'))))
        ours = set(w.WORKER_GPUS.split(','))
        worker = {int(x) for x in cast(list[int], found['worker_pids'])}
        apps = [(str(gpu), int(pid)) for gpu, pid in cast(list[list[str]], found['apps'])]
        foreign = sorted({pid for gpu, pid in apps if gpu in ours} - worker)
        ck.require(not foreign, f'other compute processes on the drafter GPUs: pids {foreign}')
        return f'drafter GPUs used only by the worker tree ({len(worker)} pids); {len(apps)} compute processes on x86host'
    r.check('x86host drafter GPUs', drafter_gpus)

    print(('PRECHECK FAIL: ' + str(r.failed) + ' check(s); fix before the window') if r.failed else
          'PRECHECK PASS: run sp1_window.sh --dry-run, then --execute', flush=True)
    return 1 if r.failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('expected', help='the reviewed 40-hex commit')
    ap.add_argument('--out', type=Path, help='the planned sp1_window.sh --out directory (checked absent)')
    ap.add_argument('--data-dir', type=Path, default=Path('/opt/dsv41/data/bench-data'))
    args = ap.parse_args(argv)
    if re.fullmatch('[0-9a-f]{40}', args.expected) is None:
        print('full 40-hex reviewed commit required', file=sys.stderr)
        return 2
    return precheck(args.expected, args.out, pc.ssh, data_dir=args.data_dir)


if __name__ == '__main__':
    raise SystemExit(main())
