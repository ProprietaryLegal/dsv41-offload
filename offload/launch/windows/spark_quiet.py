"""Read-only, bounded Spark cleanup proof. Dockerd children outlive client scopes.

Wait for anonymous expert/Engram JIT containers even before their first CUDA
context. Never stop a compiler or remove an unrelated container. Any failed
enumeration, inspection or GPU query refuses restoration and retains HOLD.
The exact stdlib source is sent inline; no remote checkout dependency.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import cast


def checked(argv: list[str], timeout: float) -> str:
    return subprocess.check_output(argv, text=True, timeout=timeout)


def snapshot(timeout: float) -> bool:
    ids = checked(['docker', 'ps', '-a', '--no-trunc', '--format', '{{.ID}}'], timeout).split()
    if len(ids) != len(set(ids)) or any(re.fullmatch('[0-9a-f]{64}', cid) is None for cid in ids):
        raise ValueError('Docker container identity enumeration invalid; HOLD stays open')
    jit = False
    for cid in ids:
        value: object = json.loads(checked(['docker', 'inspect', cid], timeout))
        if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
            raise ValueError('Docker inspection invalid; HOLD stays open')
        row = cast(dict[str, object], value[0])
        config, state = row.get('Config'), row.get('State')
        if row.get('Id') != cid or not isinstance(config, dict) or not isinstance(state, dict):
            raise ValueError('Docker identity/config/state unproved; HOLD stays open')
        name = row.get('Name')
        if not isinstance(name, str) or not isinstance(row.get('Path'), str) or not isinstance(row.get('Args'), list):
            raise ValueError('Docker name/command unproved; HOLD stays open')
        status = state.get('Status')
        if status not in ('created', 'running', 'paused', 'restarting', 'removing', 'exited', 'dead'):
            raise ValueError('Docker lifecycle unproved; HOLD stays open')
        # A stopped (exited/dead) container cannot act; old renamed production containers such as
        # dsv41-serve-stopped-20261022 left DEP1-W-20261027T020407Z's restore refused with production down.
        if status not in ('exited', 'dead') and re.search(
                r'^(?:dsv41-serve(?:$|-)|dsv41-exp)|experiment|nbw|tv1|pp4|up3|dep1', name.lstrip('/')):
            raise ValueError('experiment/serve container remains: ' + name + '; HOLD stays open')
        # Inspect full argv AND config/env/mount identity. Random --rm names and
        # import/build-lock pauses have no relationship to compute-app presence.
        identity = json.dumps({key: row.get(key) for key in ('Path', 'Args', 'Config', 'Mounts')})
        is_jit = any(key in identity for key in
                     ('tf_experts', 'vllm_exl3_moe', '_load_engram_remote_ext',
                      'DSV41_ENGRAM_OVERLAY', 'offload/engram/spark', 'prebuild_jit.sh'))
        jit |= is_jit and status not in ('exited', 'dead')
    apps = checked(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], timeout)
    return not jit and not apps.strip()


def wait(deadline: float, marker: str) -> None:
    if not math.isfinite(deadline):
        raise ValueError('finite quiet-proof deadline required')
    while True:
        left = deadline - time.time()
        if left <= 0:
            raise ValueError('JIT/compute quiet wait exhausted; HOLD stays open')
        if snapshot(min(10, left)):
            if time.time() >= deadline:
                raise ValueError('quiet proof arrived after deadline; HOLD stays open')
            print(marker, flush=True)
            return
        time.sleep(min(5, max(0, deadline - time.time())))


def command(marker: str, *, deadline: float) -> str:
    return shlex.join(['python3', '-c', Path(__file__).read_text(), str(deadline), marker])


if __name__ == '__main__':
    wait(float(sys.argv[1]), sys.argv[2])
