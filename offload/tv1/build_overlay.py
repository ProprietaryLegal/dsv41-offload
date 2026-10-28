"""Pinned zero-fuzz TV1 overlay build. No host access, deployment, imports of engine code or donor execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import py_compile
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TypedDict, cast

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
IMAGE = Path('/opt/dsv41/data/src-image')
PREFIX = '/usr/local/lib/python3.12/dist-packages/'
SWITCHES = ('candidate', 'compact')
BASES = {
    'vllm/model_executor/layers/sparse_attn_indexer.py': IMAGE / 'vllm/model_executor/layers/sparse_attn_indexer.py',
    'vllm/v1/worker/gpu/model_runner.py': ROOT / 'offload/kvpack/overlay/vllm/v1/worker/gpu/model_runner.py',
}
HELPERS = {
    'candidate': ('candidate.py', 'candidate_kernel.py', 'candidate_cases.py', 'candidate_metadata.py'),
    'compact': ('compact.py',),
}


class Entry(TypedDict):
    path: str
    sha256: str


class Manifest(TypedDict):
    switches: list[str]
    files: list[Entry]
    inputs_lock_sha256: str
    qualification: str


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe(root: Path, name: str) -> Path:
    path = root / name
    if Path(name).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'unsafe overlay path: {name}')
    return path


def sources(switches: tuple[str, ...]) -> dict[str, Path]:
    if len(set(switches)) != len(switches) or set(switches) - set(SWITCHES):
        raise ValueError(f'unknown or duplicate TV1 switch: {switches}')
    found = dict(BASES)
    for flag in switches:
        for name in HELPERS[flag]:
            found['vllm_exl3_moe/tv1_' + name] = HERE / name
    return found


def build(output: Path, switches: tuple[str, ...] = ()) -> Manifest:
    if output.exists() and any(output.iterdir()):
        raise ValueError('TV1 build output must be empty')
    lock = cast(dict[str, str], json.loads((HERE / 'inputs.lock.json').read_text()))
    inputs = sources(switches)
    for name, path in inputs.items():
        if digest(path) != lock[name]:
            raise ValueError(f'TV1 source hash drift: {path}')
    for flag in switches:
        path = HERE / 'patches' / (flag + '.patch')
        if digest(path) != lock['patch:' + flag]:
            raise ValueError(f'TV1 patch hash drift: {flag}')
    output.mkdir(parents=True, exist_ok=True)
    for name, path in inputs.items():
        dst = safe(output, name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dst)
    # A pinned combined diff avoids fuzzy/offset patching when both independent
    # features touch the runner. Each standalone patch remains independently pinned.
    patch_flags = ('candidate-compact',) if set(switches) == set(SWITCHES) else switches
    for flag in patch_flags:
        if digest(HERE / 'patches' / (flag + '.patch')) != lock['patch:' + flag]:
            raise ValueError(f'TV1 patch hash drift: {flag}')
        result = subprocess.run(['patch', '--batch', '--forward', '--fuzz=0', '--no-backup-if-mismatch',
                                 '-p1', '-d', str(output), '-i', str(HERE / 'patches' / (flag + '.patch'))],
                                capture_output=True, text=True)
        if result.returncode or 'fuzz' in result.stdout.lower() or 'offset' in result.stdout.lower():
            raise ValueError(f'TV1 patch failed: {flag}\n{result.stdout}\n{result.stderr}')
    actual = sorted(p for p in output.rglob('*') if p.is_file())
    if any(p.suffix in ('.rej', '.orig') for p in actual):
        raise ValueError('TV1 build left rejects or backups')
    with tempfile.TemporaryDirectory(prefix='tv1-compile-') as tmp:
        for path in actual:
            py_compile.compile(str(path), cfile=str(Path(tmp) / 'check.pyc'), doraise=True)
    return {'switches': list(switches),
            'files': [{'path': str(p.relative_to(output)), 'sha256': digest(p)} for p in actual],
            'inputs_lock_sha256': digest(HERE / 'inputs.lock.json'),
            'qualification': 'Source composition only; GB10 bitwise top-k/greedy and speed gates required.'}


def verify(tree: Path, manifest: Manifest) -> None:
    names = {str(p.relative_to(tree)) for p in tree.rglob('*') if p.is_file()}
    if names != {e['path'] for e in manifest['files']}:
        raise ValueError('TV1 overlay file set mismatch')
    for entry in manifest['files']:
        if digest(safe(tree, entry['path'])) != entry['sha256']:
            raise ValueError(f'TV1 overlay hash mismatch: {entry["path"]}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--switch', action='append', choices=SWITCHES, default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('TV1 output already exists')
    sidecar = args.output.with_suffix('.manifest.json')
    if sidecar.exists():
        raise ValueError('TV1 manifest already exists')
    manifest = build(args.output, tuple(args.switch))
    verify(args.output, manifest)
    sidecar.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'TV1 built {len(manifest["files"])} pinned files; switches={manifest["switches"]}')


if __name__ == '__main__':
    main()
