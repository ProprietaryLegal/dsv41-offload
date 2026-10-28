"""PC1 production pins and registered settings. CPU only; no engine replacement."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import TypedDict, cast

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PRODUCTION = '596ba1cf4e3b11be29b41a57e7f69a5d8efa4799'
TAG = 'prod-k1-327-ops1-20261026'
DIST = '/usr/local/lib/python3.12/dist-packages/'
SETTINGS = {'control': (4096, 1024), 'chunk2048': (2048, 1024),
            'threshold2048': (4096, 2048), 'threshold512': (4096, 512),
            'chunk8192': (8192, 1024), 'chunk16384': (16384, 1024)}
ARMS = tuple(SETTINGS)
LARGE_ARMS = ('chunk8192', 'chunk16384')


class Lock(TypedDict):
    production: str
    inputs: dict[str, str]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def production_file(name: str) -> bytes:
    return subprocess.check_output(['git', 'show', f'{PRODUCTION}:{name}'], cwd=ROOT)


def verify_inputs() -> Lock:
    lock = cast(Lock, json.loads((HERE / 'inputs.json').read_text()))
    if lock['production'] != PRODUCTION or not lock['inputs']:
        raise ValueError('PC1 production lock invalid')
    for key, expected in lock['inputs'].items():
        if not key.startswith('git:') or digest(production_file(key[4:])) != expected:
            raise ValueError('PC1 production input drift: ' + key)
        if key.startswith('git:offload/kvpack/') and digest((ROOT / key[4:]).read_bytes()) != expected:
            raise ValueError('PC1 checked-out K1 differs: ' + key)
    return lock


def admit(arm: str) -> None:
    if arm not in SETTINGS:
        raise ValueError('PC1 unregistered arm')


def source_pins(arm: str) -> dict[str, str]:
    """Only files production actually mounts: stage_overlay mounts *.py, not hostflag.c."""
    admit(arm)
    lock = cast(Lock, json.loads((HERE / 'inputs.json').read_text()))
    pins: dict[str, str] = {}
    for key, sha in lock['inputs'].items():
        if key.startswith('git:offload/kvpack/overlay/'):
            pins[DIST + key.removeprefix('git:offload/kvpack/overlay/')] = sha
        elif key.startswith('git:offload/dspark/spark/remote_dspark/') and key.endswith('.py'):
            pins[DIST + 'vllm/v1/worker/gpu/spec_decode/remote_dspark/' + key.rsplit('/', 1)[1]] = sha
        elif key.startswith('git:offload/dspark/spark/overlay/'):
            pins[DIST + key.removeprefix('git:offload/dspark/spark/overlay/')] = sha
    return pins


if __name__ == '__main__':
    lock = verify_inputs()
    print(f'PC1: {len(lock["inputs"])} production inputs verified; {len(ARMS)} registered pairs; no replacement sources')
