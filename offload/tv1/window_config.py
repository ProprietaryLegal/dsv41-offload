"""Compose TV1 mounts over the unchanged OPS1 production snapshot before SSH/stop."""
from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path, PurePosixPath
from typing import cast

from offload.tv1 import build_overlay as b

FLAGS = {'off': {'VLLM_TV1_CANDIDATE': '0', 'VLLM_TV1_COMPACT': '0', 'VLLM_TV1_COMPACT_TAU': '0.5'},
         'candidate': {'VLLM_TV1_CANDIDATE': '1', 'VLLM_TV1_COMPACT': '0', 'VLLM_TV1_COMPACT_TAU': '0.5'},
         'compact': {'VLLM_TV1_CANDIDATE': '0', 'VLLM_TV1_COMPACT': '1', 'VLLM_TV1_COMPACT_TAU': '0.5'}}


def compose(role: str, tree: str, raw: str, env: str, root: Path = b.ROOT) -> tuple[str, str, str]:
    if role not in FLAGS or not re.fullmatch(r'offload/tv1/generated-[A-Za-z0-9_-]+/(off|candidate|compact)', tree):
        raise ValueError('TV1 unsafe role or generated path')
    path = b.safe(root, tree)
    manifest = cast(b.Manifest, json.loads(path.with_suffix('.manifest.json').read_text()))
    expected = [] if role == 'off' else [role]
    if manifest['switches'] != expected or manifest['inputs_lock_sha256'] != b.digest(b.HERE / 'inputs.lock.json'):
        raise ValueError('TV1 manifest switch/input-lock mismatch')
    b.verify(path, manifest)
    targets = {PurePosixPath(b.PREFIX + e['path']) for e in manifest['files']}
    tokens = shlex.split(raw)
    kept: list[str] = []
    replaced: set[PurePosixPath] = set()
    seen: set[PurePosixPath] = set()
    for index in range(0, len(tokens), 2):
        if index + 1 >= len(tokens) or tokens[index] != '-v':
            raise ValueError('TV1 base mounts must be explicit -v source:target:mode pairs')
        spec = tokens[index + 1]
        parts = spec.split(':')
        if len(parts) != 3 or any(ch.isspace() for ch in spec):
            raise ValueError('TV1 malformed base mount')
        target = PurePosixPath(parts[1])
        if target in seen:
            raise ValueError('TV1 duplicate production mount destination')
        seen.add(target)
        if parts[2] != 'ro' and not (target == PurePosixPath('/sp1-profile') and parts[2] == 'rw'):
            raise ValueError('TV1 writable production mount')
        if target in targets:
            if target in replaced or parts[2] != 'ro':
                raise ValueError('TV1 duplicate/writable replacement mount')
            replaced.add(target)
            continue
        if any(target in wanted.parents for wanted in targets):
            raise ValueError('TV1 parent mount shadows a qualified file')
        kept.extend(tokens[index:index + 2])
    runner = PurePosixPath(b.PREFIX + 'vllm/v1/worker/gpu/model_runner.py')
    if runner not in replaced:
        raise ValueError('TV1 expected the production K1 runner mount exactly once')
    variables = shlex.split(env)
    keys = [v.partition('=')[0] for v in variables]
    if len(set(keys)) != len(keys) or any('=' not in v for v in variables):
        raise ValueError('TV1 duplicate/malformed production environment')
    if any(v.split('=')[0].startswith('VLLM_TV1_') for v in variables):
        raise ValueError('TV1 inherited/duplicate feature environment')
    extra = ' '.join(f'{k}={v}' for k,v in FLAGS[role].items())
    overlays = ' '.join(f'{tree}/{e["path"]}={b.PREFIX}{e["path"]}' for e in manifest['files'])
    return ' '.join(kept), overlays, env + ' ' + extra


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role', choices=tuple(FLAGS), required=True)
    p.add_argument('--tree', required=True)
    p.add_argument('--raw', required=True)
    p.add_argument('--env', required=True)
    p.add_argument('--root', type=Path, default=b.ROOT)
    a = p.parse_args()
    print('\n'.join(compose(a.role, a.tree, a.raw, a.env, a.root)))


if __name__ == '__main__':
    main()
