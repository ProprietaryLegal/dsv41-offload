"""Emit stdlib-only two-rank hash verification for K1 plus the selected trace arm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from offload.prefill_maxlen.evidence import Json, record

HERE = Path(__file__).resolve().parent


def render(enabled: bool, *, cause: bool = False) -> str:
    manifest = record(cast(Json, json.loads((HERE.parent / 'kvpack/MANIFEST.json').read_text())))
    rows = manifest['files']
    if not isinstance(rows, list):
        raise ValueError('K1 manifest files missing')
    pins = {str(record(row)['container_path']): str(record(row)['overlay_sha256']) for row in rows}
    if enabled:
        for line in (HERE / 'OVERLAY.sha256').read_text().splitlines():
            sha, path = line.split()
            pins['/usr/local/lib/python3.12/dist-packages/' + path] = sha
    if cause:
        if not enabled:
            raise ValueError('cause verification requires trace sources')
        for line in (HERE.parent / 'prefill_cause/OVERLAY.sha256').read_text().splitlines():
            sha, path = line.split()
            pins['/usr/local/lib/python3.12/dist-packages/' + path] = sha
    return ('import hashlib, pathlib\n'
            f'pins = {pins!r}\n'
            'for path, expected in pins.items():\n'
            '    actual = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()\n'
            '    if actual != expected: raise RuntimeError(f"PF1 source mismatch: {path}: {actual}")\n'
            'print(f"PF1 verified {len(pins)} engine sources")\n')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--enabled', action='store_true')
    args = ap.parse_args()
    print(render(args.enabled), end='')


if __name__ == '__main__':
    main()
