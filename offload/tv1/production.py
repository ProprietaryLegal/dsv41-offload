"""TV1's OPS1 ledger and strict runtime reconciliation, independent of historical PF1 policy."""
from __future__ import annotations

import shlex
from typing import Protocol

from offload.launch.windows import pf1_window as pf
from offload.launch.windows import sp1_window as sp
from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, record
from offload.tv1 import build_overlay as overlay
from offload.tv1.window_config import FLAGS

PROD_COMMIT = '596ba1cf4e3b11be29b41a57e7f69a5d8efa4799'
PROD_TAG = 'prod-k1-327-ops1-20261026'


class Arm(Protocol):
    @property
    def role(self) -> str: ...
    @property
    def cap(self) -> int: ...
    @property
    def trace(self) -> int: ...
    @property
    def eager(self) -> int: ...
    @property
    def cause(self) -> bool: ...
    @property
    def fix(self) -> bool: ...


def ledger_script() -> str:
    """Read only; fail before HOLD unless HEAD/tag and all production launch files match."""
    return ('cd ~/dsv41-deploy && test ! -e HOLD && '
            f'test "$(git rev-parse HEAD)" = {PROD_COMMIT} && '
            f'test "$(git rev-parse HEAD)" = "$(git rev-list -n1 {PROD_TAG})" && '
            'git diff --exit-code HEAD -- dsv41.env serve.sh start.sh stop.sh offload/launch/window.sh '
            'offload/launch/warmup.py offload/launch/warmup_serve.sh')


def rollback_metadata() -> dict[str, Json]:
    return {'production_commit': PROD_COMMIT, 'tag': PROD_TAG, 'checkout': '~/dsv41-deploy',
            'worker_source': pf.WORKER, 'restore_sha256': pf.RESTORE_SHA256}


def runtime(prod: dict[str, Json], live: dict[str, Json], arm: Arm,
            manifest: overlay.Manifest, run: str) -> None:
    """Allow only TV1 source/flags, idle profiler and port differences; compare all other values exactly."""
    del run  # profiler/feature settings are fixed, never arbitrary per-run environment overrides
    ck.require(arm.role in FLAGS and arm.cap == 327680 and not any((arm.trace, arm.eager, arm.cause, arm.fix)),
               'TV1 unsupported arm settings')
    ck.require(manifest['switches'] == ([] if arm.role == 'off' else [arm.role]), 'TV1 manifest role mismatch')
    ck.require(live.get('image') == prod.get('image') == pf.IMAGE, 'image differs from production/pin')
    ck.require(live.get('engine') == prod.get('engine') and live.get('files') == prod.get('files'),
               'engine version or target metadata differs from production')
    ck.require(prod.get('state') == live.get('state') == 'running', 'production or experiment container not running')
    ck.require(bool(record(live['engram_identity'])) and live.get('engram_identity') == prod.get('engram_identity'),
               'Engram shim/source/JIT identity differs from production')
    expected = {k: v for k, v in record(prod['mounted']).items() if k != '/serve.sh'}
    expected.update({pf.DIST + e['path']: e['sha256'] for e in manifest['files']})
    got = {k: v for k, v in record(live['mounted']).items() if k != '/serve.sh'}
    ck.require(got == expected, 'TV1 mounted bytes differ from exact reviewed overlay plus production')
    before, after = record(prod['env']), record(live['env'])
    ck.require(before.get('VLLM_ENGRAM_REMOTE_REQUIRED') == '0', 'production Engram REQUIRED is not reconciled 0')
    ck.require(not any(key.startswith('VLLM_TV1_') for key in before), 'production already carries TV1 flags')
    expected_env = {**before, **FLAGS[arm.role]}
    ck.require(after == expected_env, 'TV1 runtime environment differs from production beyond exact feature flags')
    a = shlex.split(str(prod['argv']).replace('\\\n', ' '))
    b = shlex.split(sp.strip_profiler(str(live['argv'])).replace('\\\n', ' '))
    ck.require(a.count('--port') == 1 and a.count('--max-model-len') == 1, 'production port/cap argv malformed')
    index = a.index('--port') + 1
    ck.require(index < len(a) and a[index] == '8000', 'production port is not 8000')
    cap_index = a.index('--max-model-len') + 1
    ck.require(cap_index < len(a) and a[cap_index] == '327680', 'production cap is not 327680')
    a[index] = '8001'
    ck.require(a == b, 'TV1 launch argv differs from production beyond port and idle profiler')
