"""Emit the stdlib-only in-container hash verification for the fix arm: K1's 13 sources with the 3 MoE files replaced
by the fence sources, plus the fence helper (14). Same script shape as offload/prefill_trace/emit_verify.py."""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from offload.prefill_maxlen.evidence import Json, record

HERE = Path(__file__).resolve().parent
DIST = '/usr/local/lib/python3.12/dist-packages/'


def pins() -> dict[str, str]:
    manifest = record(cast(Json, json.loads((HERE.parent / 'kvpack/MANIFEST.json').read_text())))
    rows = manifest['files']
    if not isinstance(rows, list):
        raise ValueError('K1 manifest files missing')
    out = {str(record(row)['container_path']): str(record(row)['overlay_sha256']) for row in rows}
    for line in (HERE / 'OVERLAY.sha256').read_text().splitlines():
        sha, path = line.split()
        out[DIST + path] = sha
    return out


def render() -> str:
    return ('import hashlib, pathlib\n'
            f'pins = {pins()!r}\n'
            'for path, expected in pins.items():\n'
            '    actual = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()\n'
            '    if actual != expected: raise RuntimeError(f"PF1 source mismatch: {path}: {actual}")\n'
            'print(f"PF1 verified {len(pins)} engine sources")\n')
