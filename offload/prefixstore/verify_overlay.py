#!/usr/bin/env python3
"""Verify K1+RP1 bytes locally or through light docker-exec reads (coordinator).

No container is created or restarted. `tree` checks saved image/base or overlay
copies without contacting a Spark; `running NODE...` checks mounted runtime bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any, cast

from build_overlay import ROOT, verify_checkout


def rows() -> list[dict[str, Any]]:
    rp1 = verify_checkout()
    k1 = json.loads((ROOT / 'offload/kvpack/MANIFEST.json').read_text())
    entries = k1['files'] + rp1['files']
    if len({e['path'] for e in entries}) != len(entries):
        raise ValueError('overlapping K1/RP1 mount targets require a fresh audit')
    return cast(list[dict[str, Any]], entries)


def verify_tree(tree: Path, base: bool = False) -> None:
    for e in rows():
        expected = e['base_sha256' if base else 'overlay_sha256']
        path = tree / e['path']
        if expected is None:
            if path.exists(): raise ValueError(f'base unexpectedly contains {path}')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'SHA mismatch: {path}')
    print(f'verified {len(rows())} {"base" if base else "overlay"} files')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('tree')
    p.add_argument('root', type=Path)
    p.add_argument('--base', action='store_true')
    p = sub.add_parser('running')
    p.add_argument('nodes', nargs='+')
    p.add_argument('--container', default='dsv41-serve')
    p.add_argument('--control', action='store_true')
    p.add_argument('--verify-stores', action='store_true')
    p.add_argument('--fraction', default='0.2')
    p.add_argument('--io-cpus', default='0-4,10-13')
    args = parser.parse_args()
    if args.mode == 'tree':
        verify_tree(args.root, args.base)
        return
    expected = {} if args.control else {
        'VLLM_DSV41_KV_CRC': '1', 'VLLM_DSV41_OFFLOAD_EXPANDABLE_OK': '1',
        'VLLM_DSV41_OFFLOAD_MULTINODE': '1',
        'VLLM_DSV41_OFFLOAD_MN_VERIFY': '1' if args.verify_stores else '0',
        'VLLM_DSV41_OFFLOAD_MN_STAGE_MB': '8',
        'VLLM_DSV41_OFFLOAD_LOAD_FRACTION': args.fraction,
        'VLLM_DSV41_OFFLOAD_IO_CPUS': args.io_cpus}
    # The script only reads the 20 reviewed source files and selected environment keys.
    script = ('import hashlib,json,os; rows=' + repr(rows()) + '; expected=' + repr(expected) + '; '
              'got={e["path"]:hashlib.sha256(open(e["container_path"],"rb").read()).hexdigest() for e in rows}; '
              'assert all(got[e["path"]]==e["overlay_sha256"] for e in rows), "runtime bytes differ"; '
              'keys=["VLLM_DSV41_KV_CRC","VLLM_DSV41_OFFLOAD_EXPANDABLE_OK","VLLM_DSV41_OFFLOAD_MULTINODE",'
              '"VLLM_DSV41_OFFLOAD_MN_VERIFY","VLLM_DSV41_OFFLOAD_MN_STAGE_MB",'
              '"VLLM_DSV41_OFFLOAD_LOAD_FRACTION","VLLM_DSV41_OFFLOAD_IO_CPUS"]; '
              'actual={k:os.environ[k] for k in keys if k in os.environ}; '
              'assert actual==expected, ("RP1 switches differ",actual,expected); '
              'print("verified 20 runtime files and RP1 switches")')
    cmd = shlex.join(['docker', 'exec', args.container, 'python3', '-c', script])
    for node in args.nodes:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', node, cmd], check=True,
                                capture_output=True, text=True, timeout=30)
        print(f'{node}: {result.stdout.strip()}')


if __name__ == '__main__':
    main()
