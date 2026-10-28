#!/usr/bin/env python3
"""MC1 composition audit: matter-cache mounts against K1, production overlays and pending packages.

Reads every package from git objects (never a checkout), so it can be re-run at window time against the
then-current refs. Reports (1) shared mounted container paths, which need an exact composition, and (2) import
adjacency: a mounted module of one side importing a mounted module of the other. Adjacency is not a conflict; it
names where a semantic interaction must be reviewed. Interface coupling through vLLM's connector API (core scheduler
<-> connector) is not an import and is listed in AUDIT.md by hand.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, cast

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DIST = '/usr/local/lib/python3.12/dist-packages/'
REMOTE_REPO = 'offload/dspark/spark/remote_dspark/'
REMOTE_MOUNT = 'vllm/v1/worker/gpu/spec_decode/remote_dspark/'


@dataclass
class Package:
    name: str
    ref: str
    # container-relative mounted paths (relative to dist-packages) -> git path of the mounted bytes at ref
    mounts: dict[str, str] = field(default_factory=dict)
    note: str = ''


def git(*args: str) -> str:
    return subprocess.run(['git', '-C', str(ROOT), *args], check=True, text=True, capture_output=True).stdout


def show(ref: str, path: str) -> str:
    return git('show', f'{ref}:{path}')


def mount_of_repo_path(path: str) -> str:
    """Container-relative target for a package's repo path (remote_dspark is mounted as a directory)."""
    if path.startswith(REMOTE_REPO):
        return REMOTE_MOUNT + path[len(REMOTE_REPO):]
    if path.startswith(('vllm/', 'vllm_exl3_moe/')):
        return path
    raise ValueError(f'unmapped package path {path}')


def mc1(ref: str) -> Package:
    manifest = cast(dict[str, Any], json.loads(show(ref, 'offload/prefixstore/MANIFEST.json')))
    entries = list(manifest['files']) + list(manifest.get('new_files', []))
    return Package('MC1 (RP1 Stage A + matter cache)', ref,
                   {e['path']: 'offload/prefixstore/overlay/' + e['path'] for e in entries})


def k1(ref: str) -> Package:
    manifest = cast(dict[str, Any], json.loads(show(ref, 'offload/kvpack/MANIFEST.json')))
    return Package('K1 packing (13 files)', ref,
                   {e['path']: 'offload/kvpack/overlay/' + e['path'] for e in manifest['files']})


def production_overlays(ref: str) -> Package:
    """remote_dspark + spec_decode/__init__.py as mounted by the K1 async base config (async proxy lives inside)."""
    text = show(ref, 'offload/launch/configs/D-kv6-async-pack.env')
    line = next(l for l in text.splitlines() if l.startswith('OVERLAYS="'))
    mounts: dict[str, str] = {}
    for pair in line[len('OVERLAYS="'):-1].split():
        source, _, target = pair.partition('=')
        rel = target.removeprefix(DIST)
        if source.startswith('offload/kvpack/'):
            continue  # K1 is its own package above
        if source.rstrip('/') == REMOTE_REPO.rstrip('/'):
            for path in git('ls-tree', '-r', '--name-only', ref, '--', REMOTE_REPO).split():
                if path.endswith('.py'):
                    mounts[REMOTE_MOUNT + path[len(REMOTE_REPO):]] = path
        else:
            mounts[rel] = source
    return Package('remote DSpark + async proxy + spec_decode (production overlays)', ref, mounts,
                   'Engram is loaded through PYTHONPATH/sitecustomize outside the vllm package (no vllm mount)')


def upstream_adopt(ref: str) -> Package:
    manifest = cast(dict[str, Any], json.loads(show(ref, 'offload/upstream_adopt/MANIFEST.json')))
    mounts: dict[str, str] = {}
    for e in manifest['files']:
        path = cast(str, e['path'])
        if path.startswith(REMOTE_REPO):
            mounts[mount_of_repo_path(path)] = path
        else:
            mounts[path] = 'offload/upstream_adopt/overlay/' + path
    return Package('0075 replay skip (UP2-ADOPT, upstream_adopt overlay)', ref, mounts)


def upfix(ref: str) -> Package:
    mounts: dict[str, str] = {}
    for name in ('offload/upfix/MANIFEST-0091.json', 'offload/upfix/topk/MANIFEST-topk.json'):
        manifest = cast(dict[str, Any], json.loads(show(ref, name)))
        if 'files' in manifest:
            for e in manifest['files']:
                mounts[e['path']] = e['overlay']
        else:
            overlay = cast(dict[str, Any], manifest['overlay'])
            path = cast(str, overlay['path'])
            mounts[path] = 'offload/upfix/topk/overlay/' + path
    return Package('0091 upstream fixes (R-UPF) + top-k overlay', ref, mounts)


def pp4(ref: str) -> Package:
    text = show(ref, 'offload/launch/windows/pp4_promote_switch.py')
    consts = dict(re.findall(r"^(RUNNER|UTILS|TRACE|REMOTE) = '([^']+)'", text, re.M))
    commit = re.search(r"^PP4_COMMIT = '([0-9a-f]{40})'", text, re.M)
    if set(consts) != {'RUNNER', 'UTILS', 'TRACE', 'REMOTE'} or commit is None:
        raise ValueError('PP4 switch constants changed; re-read pp4_promote_switch.py')
    mounts = {consts['RUNNER']: '(generated: K1 runner + PP3 verify mask + PP4 timing decorators)',
              consts['UTILS']: '(generated: PP3 rejection-sampler utilities)',
              consts['TRACE']: 'offload/launch/windows/pp4_round_trace.py@' + commit.group(1)}
    for path in git('ls-tree', '-r', '--name-only', commit.group(1), '--', REMOTE_REPO).split():
        if path.endswith('.py'):
            mounts[consts['REMOTE'].rstrip('/') + '/' + path[len(REMOTE_REPO):]] = path + '@' + commit.group(1)
    return Package('PP4 sampled K2 (PP4-PROMOTE)', ref, mounts,
                   'runner = K1 model_runner.py + verify-mask block + @execution/@sampling decorators only')


def clock_fold(ref: str, base: str) -> Package:
    changed = git('diff', '--name-only', f'{base}...{ref}').split()
    if changed != ['dsv41.env']:
        raise ValueError(f'clock fold now changes {changed}; re-audit')
    return Package('clock fold 2300 (dsv41.env only)', ref, {}, 'no container mount; dsv41.env GPU_CLOCK_CAP default')


def module_name(rel: str) -> str:
    return rel.removesuffix('.py').removesuffix('/__init__').replace('/', '.')


def imports(source: str, module: str) -> set[str]:
    out: set[str] = set()
    package = module.rsplit('.', 1)[0]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ''
            if node.level:
                parts = package.split('.')
                base = '.'.join(parts[:len(parts) - node.level + 1] + ([base] if base else []))
            out.add(base)
            out.update(f'{base}.{a.name}' for a in node.names)
    return out


def source_of(pkg: Package, rel: str) -> str | None:
    git_path = pkg.mounts[rel]
    if git_path.startswith('('):
        return None
    ref = pkg.ref
    if '@' in git_path:
        git_path, ref = git_path.split('@', 1)
    return show(ref, git_path)


def adjacency(a: Package, b: Package, base_tree: Path) -> list[str]:
    """a's mounted modules importing b's mounted modules (and vice versa)."""
    out: list[str] = []
    for x, y in ((a, b), (b, a)):
        targets = {module_name(rel): rel for rel in y.mounts if rel.endswith('.py')}
        for rel in sorted(x.mounts):
            if not rel.endswith('.py'):
                continue
            src = source_of(x, rel)
            if src is None:
                image = base_tree / rel
                if not image.exists():
                    continue
                src = image.read_text()
            hits = sorted(targets[m] for m in imports(src, module_name(rel)) if m in targets)
            out.extend(f'{x.name}: {rel} imports {y.name}: {h}' for h in hits)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mc1', default='HEAD')
    parser.add_argument('--production', default='596ba1c')
    parser.add_argument('--pp4', default='offload/pp4-promote')
    parser.add_argument('--up2', default='offload/up2-adopt')
    parser.add_argument('--upf', default='offload/upfix')
    parser.add_argument('--clock', default='prod/clock-fold-2300')
    parser.add_argument('--image-src', type=Path, default=Path('/opt/dsv41/data/src-image'))
    parser.add_argument('--json', type=Path, help='write the full result here')
    args = parser.parse_args(argv)
    mine = mc1(args.mc1)
    others = [k1(args.production), production_overlays(args.production), upstream_adopt(args.up2),
              upfix(args.upf), pp4(args.pp4), clock_fold(args.clock, args.production)]
    result: dict[str, Any] = {'mc1_ref': git('rev-parse', args.mc1).strip(), 'mc1_mounts': sorted(mine.mounts),
                              'packages': []}
    shared_total = 0
    for pkg in others:
        shared = sorted(set(mine.mounts) & set(pkg.mounts))
        shared_total += len(shared)
        adj = adjacency(mine, pkg, args.image_src)
        result['packages'].append({'name': pkg.name, 'ref': git('rev-parse', pkg.ref).strip(),
                                   'mounts': len(pkg.mounts), 'shared': shared, 'import_adjacency': adj,
                                   'note': pkg.note})
        print(f'{pkg.name} @ {pkg.ref}: {len(pkg.mounts)} mounts, shared {shared or "none"}')
        for line in adj:
            print(f'  adjacency: {line}')
        if pkg.note:
            print(f'  note: {pkg.note}')
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + '\n')
    print(f'MC1 mounts {len(mine.mounts)}; shared mounted paths with all packages: {shared_total}')
    return 1 if shared_total else 0


if __name__ == '__main__':
    sys.exit(main())
