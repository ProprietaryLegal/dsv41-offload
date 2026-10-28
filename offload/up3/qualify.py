"""UP3 receipt collector and in-worker adapter boundary. Dry by default.

Host CLI only reads the exact experiment container and collects its worker's
receipt. It cannot load a worker, create a CUDA context, or turn supplied pass
booleans into evidence. See bootstrap.py for the explicit in-worker lifecycle.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from offload.prefill_maxlen.evidence import Json, number, record
from offload.up3 import build, gates
from offload.up3.engine_common import attr, require, sha

HERE = Path(__file__).resolve().parent
MARKER = 'UP3_ENGINE_GATE '


def package() -> dict[str, Json]:
    lock = record(cast(Json, json.loads((HERE/'qualification.lock.json').read_text())))
    for name, digest in record(lock['files']).items():
        require(sha(build.ROOT/name) == digest, 'qualification package source drift: '+name)
    return lock


@dataclass(frozen=True)
class Request:
    arm: str
    rank: int
    run: str
    container: str
    program_sha256: str
    since: float

    def validate(self) -> None:
        require(self.arm in gates.CASES and self.rank in (0, 1), 'known arm and TP rank required')
        require(re.fullmatch('[A-Za-z0-9_-]{1,180}', self.run) is not None, 'invalid run identity')
        require(re.fullmatch('[0-9a-f]{64}', self.container) is not None, 'full Docker container ID required')
        require(self.program_sha256 == sha(Path(__file__)), 'reviewed qualifier program hash differs')
        require(0 < self.since <= time.time()+2, 'invalid start time')
        package()


def engine_sources(arm: str) -> None:
    expected = record(record(package()['engine'])[arm])
    for name, digest in expected.items():
        module = importlib.import_module(name)
        filename = attr(module, '__file__')
        require(isinstance(filename, str) and sha(Path(filename)) == digest, 'installed engine source drift: '+name)


def worker_identity(request: Request, worker: object) -> list[Json]:
    """Check the initialized worker BEFORE native qualification work, and again at receipt."""
    torch = importlib.import_module('torch')
    dist = importlib.import_module('vllm.distributed')
    from offload.up3.engine_common import invoke, runner
    runner(worker)
    group = invoke(attr(dist, 'get_tp_group'))
    require(attr(group, 'rank_in_group') == request.rank and attr(group, 'world_size') == 2, 'worker TP rank differs')
    cuda = attr(torch, 'cuda')
    capability = list(cast(tuple[int, int], invoke(attr(cuda, 'get_device_capability'))))
    require(capability == [12, 1], 'actual GB10 SM121 required')
    require(Path('/etc/hostname').read_text().strip() == request.container[:12], 'worker container hostname/ID differs')
    return cast(list[Json], capability)


def receipt(request: Request, cases: dict[str, Json], worker: object) -> dict[str, Json]:
    """Only the live worker boundary may label scope actual-engine-inputs."""
    request.validate()
    engine_sources(request.arm)
    capability = worker_identity(request, worker)
    result: dict[str, Json] = {'schema': 'up3-gb10-gates-v1', 'arm': request.arm, 'rank': request.rank,
        'run': request.run, 'container': request.container, 'production': build.PRODUCTION,
        'program_sha256': request.program_sha256,
        'manifest_sha256': sha(HERE/'manifests'/f'{request.arm}.json'), 'device_capability': capability,
        'scope': 'actual-engine-inputs', 'time': time.time(), 'cases': cases,
        'worker_pid': os.getpid(), 'qualification_package_sha256': sha(HERE/'qualification.lock.json')}
    if request.arm == 'dual-ring':
        result['nccl_only_fallback'] = False  # run() already rejects fallback logs, missing ring and wrong routing
    gates.validate(result, arm=request.arm, rank=request.rank, run=request.run, container=request.container,
        since=request.since, now=time.time(), program_sha=request.program_sha256)
    return result


def collect(request: Request, inspect_data: list[Json], log: str, *, now: float) -> dict[str, Json]:
    request.validate()
    require(len(inspect_data) == 1, 'one exact experiment container required')
    identity = record(inspect_data[0])
    require(identity.get('Id') == request.container and record(identity['State']).get('Running') is True,
            'experiment container stopped/replaced')
    cmd = record(identity['Config']).get('Cmd')
    require(isinstance(cmd, list) and '--port' in cmd and cmd[cmd.index('--port')+1] == '8001',
            'collector refuses a production-port container')
    rows: list[dict[str, Json]] = []
    for line in log.splitlines():
        if MARKER not in line:
            continue
        item = record(cast(Json, json.loads(line.split(MARKER, 1)[1])))
        if item.get('run') == request.run and item.get('rank') == request.rank:
            rows.append(item)
    require(len(rows) == 1, 'missing/duplicate actual-worker receipt; bootstrap/probes incomplete')
    result = rows[0]
    require(number(result.get('worker_pid', 0)) > 0, 'worker PID evidence absent')
    require(result.get('qualification_package_sha256') == sha(HERE/'qualification.lock.json'), 'worker package differs')
    gates.validate(result, arm=request.arm, rank=request.rank, run=request.run, container=request.container,
        since=request.since, now=now, program_sha=request.program_sha256)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--check-package', action='store_true')
    parser.add_argument('--arm', choices=tuple(gates.CASES))
    parser.add_argument('--rank', type=int, choices=(0, 1))
    parser.add_argument('--run', default='')
    parser.add_argument('--container', default='')
    parser.add_argument('--program-sha256', default='')
    parser.add_argument('--since', type=float, default=0)
    args = parser.parse_args(argv)
    try:
        if args.check_package:
            package()
            print('UP3 qualification package hashes verified (CPU only; no engine import)')
            return 0
        if not args.execute:
            print('DRY: no host/engine action. Requires in-worker bootstrap on BOTH ranks, public probes and source review; see UP3-WINDOW.md.')
            return 0
        # The shared receiving-host scope must own this collector's lifetime.
        deadline = float(os.environ.get('SCOPED_REMOTE_DEADLINE', '0'))
        cgroup = Path('/proc/self/cgroup').read_text().strip()
        require(math.isfinite(deadline) and deadline > time.time()
                and cgroup.startswith('0::') and Path(cgroup[3:]).name.startswith('dsv41w-')
                and Path(cgroup[3:]).name.endswith('.scope'),
                'collector must run inside the shared registered scope/deadline')
        request = Request(args.arm or '', args.rank if args.rank is not None else -1,
            args.run, args.container, args.program_sha256, args.since)
        request.validate()
        inspected = subprocess.run(['docker', 'inspect', request.container], check=True, capture_output=True, text=True, timeout=10)
        logs = subprocess.run(['docker', 'logs', request.container], check=True, capture_output=True, text=True, timeout=20)
        rows = cast(list[Json], json.loads(inspected.stdout))
        result = collect(request, rows, logs.stdout+'\n'+logs.stderr, now=time.time())
        print(json.dumps(result, allow_nan=False))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError, KeyError, IndexError) as exc:
        print('STOP: '+str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    # Direct-file invocation used by the existing runner needs the checkout on PYTHONPATH.
    raise SystemExit(main())
