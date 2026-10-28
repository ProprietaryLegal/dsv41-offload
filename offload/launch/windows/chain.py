"""WIN2 back-to-back (chain) mode (NDM 2026-10-27: "We do not need to keep restoring production -- go window after window").

Coordinator-only. A chained window still opens and closes its own HOLD and keeps every fence, quiet and absence proof;
at a SUCCESSFUL end it leaves production DOWN (no W8 restore, no post-close watch) and writes MARKER. The next chained
window's preflight accepts "MARKER + production down + no HOLD" instead of "production serving". Any stop or failure,
or the last window of a chain (WINDOW_CHAIN_LAST=1), runs the normal strict W8 restore + watch and then clear().
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

from offload.prefill_maxlen.evidence import Json, record

MARKER = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD/scratch/CHAINED')
SCHEMA = 'dsv41-chain-v1'
PRODUCTION_MODELS = 'http://HEAD_IP:8000/v1/models'
K1_CAP = 327680


def enabled() -> bool:
    return os.environ.get('WINDOW_CHAIN') == '1'


def last() -> bool:
    return os.environ.get('WINDOW_CHAIN_LAST') == '1'


def skip_restore(stopped: bool) -> bool:
    """True only for a chained, non-last window that ended without any stop."""
    return enabled() and not last() and not stopped


def finish_chained(window_out: Path, quiet_proof: Mapping[str, Json]) -> None:
    if not (enabled() and not last()):
        raise ValueError('finish_chained outside a non-last chained window')
    if not quiet_proof:
        raise ValueError('chained finish requires the Spark quiet proof')
    data: dict[str, Json] = {'schema': SCHEMA, 'window': str(window_out), 'time': time.time(),
                             'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                             'quiet_proof': cast(Json, dict(quiet_proof))}
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    temp = MARKER.with_name(MARKER.name + '.new')
    temp.write_text(json.dumps(data, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temp, MARKER)


def read_marker() -> dict[str, Json] | None:
    if not MARKER.exists():
        return None
    data = record(cast(Json, json.loads(MARKER.read_text())))
    if data.get('schema') != SCHEMA:
        raise ValueError(f'{MARKER}: unknown chain marker schema')
    return data


def preflight(production_serving: bool, hold_exists: bool) -> str:
    """'chained' (marker, production down, no HOLD) or 'production' (no marker, production serving); else STOP."""
    if hold_exists:
        raise ValueError('STOP: a HOLD exists; no window may start')
    marker = read_marker()
    if marker is not None:
        if not enabled():
            raise ValueError('STOP: production is down in a chain (marker present); run with WINDOW_CHAIN=1 or restore W8')
        if production_serving:
            raise ValueError('STOP: chain marker present but production is serving; inspect before continuing')
        return 'chained'
    if not production_serving:
        raise ValueError('STOP: production not serving and no chain marker')
    return 'production'


def clear() -> None:
    """After a strict W8 restore (chain end or any failure)."""
    MARKER.unlink(missing_ok=True)


def production_serving(url: str = PRODUCTION_MODELS, timeout: float = 15.0) -> bool:
    """Production answers /v1/models with the K1 cap (the same test the coordinator's prev-ok gate uses)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = record(cast(Json, json.loads(response.read(1024 * 1024))))
    except (OSError, ValueError):
        return False
    rows = data.get('data')
    return isinstance(rows, list) and any(isinstance(r, dict) and r.get('max_model_len') == K1_CAP for r in rows)


def hold_exists() -> bool:
    """HOLD file on sparkhead (read-only ssh test); an ssh failure counts as 'exists' so nothing starts blind."""
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'sparkhead', 'test ! -e ~/dsv41-deploy/HOLD'],
                            capture_output=True, timeout=30, check=False)
    return result.returncode != 0


def gate(serving: Callable[[], bool] = production_serving, hold: Callable[[], bool] = hold_exists) -> str:
    """Between-window gate: 'chained' or 'production' (see preflight); raises otherwise."""
    return preflight(production_serving=serving(), hold_exists=hold())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cmd', choices=('gate', 'status', 'clear'))
    args = parser.parse_args(argv)
    if args.cmd == 'status':
        print(json.dumps({'enabled': enabled(), 'last': last(), 'marker': read_marker()}))
        return 0
    if args.cmd == 'clear':
        clear()
        print('chain marker cleared')
        return 0
    try:
        print(gate())
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
