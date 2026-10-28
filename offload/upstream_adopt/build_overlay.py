"""Rebuild reviewable UP1 files on CPU from pinned sources with patch --fuzz=0. Never deploy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import py_compile
import shutil
import subprocess
import tempfile
from typing import TypedDict, cast

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
IMAGE = Path('/opt/dsv41/data/src-image')
DONOR = Path('/opt/dsv41/data/upstream/henryous')
PREFIX = '/usr/local/lib/python3.12/dist-packages/'


class Step(TypedDict):
    kind: str
    path: str


class Lock(TypedDict):
    donor_sha: str
    files: dict[str, str]
    seed: dict[str, str]
    steps: list[Step]


class K1Entry(TypedDict):
    path: str
    base_sha256: str | None
    overlay_sha256: str


class K1Manifest(TypedDict):
    files: list[K1Entry]


class Entry(TypedDict):
    path: str
    sha256: str
    image_base_sha256: str | None


class Manifest(TypedDict):
    donor_sha: str
    inputs_lock_sha256: str
    steps: list[Step]
    fuzz: int
    files: list[Entry]
    qualification: str


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_path(root: Path, name: str) -> Path:
    path = root / name
    if Path(name).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'unsafe path: {name}')
    return path


def verify_inputs(image: Path = IMAGE, donor: Path = DONOR) -> Lock:
    lock = cast(Lock, json.loads((HERE/'inputs.lock.json').read_text()))
    revision = subprocess.check_output(['git','-C',str(donor),'rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(donor),'status','--porcelain'],text=True).strip()
    if revision != lock['donor_sha'] or dirty:
        raise ValueError('donor must be clean at pinned revision')
    roots = {'repo': ROOT, 'image': image, 'donor': donor}
    for key, expected in lock['files'].items():
        kind, name = key.split(':',1)
        path = safe_path(roots[kind], name)
        if sha(path) != expected:
            raise ValueError(f'input hash drift: {key}')
    return lock


def put(tree: Path, name: str, data: bytes) -> None:
    target = safe_path(tree, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def build(tree: Path, image: Path = IMAGE, donor: Path = DONOR) -> Manifest:
    lock = verify_inputs(image, donor)
    if tree.exists() and any(tree.iterdir()):
        raise ValueError('build destination must be empty')
    tree.mkdir(parents=True, exist_ok=True)
    roots = {'repo':ROOT, 'image':image, 'donor':donor}
    for name, kind in lock['seed'].items():
        put(tree, name, safe_path(roots[kind], name).read_bytes())
    k1 = cast(K1Manifest, json.loads((ROOT/'offload/kvpack/MANIFEST.json').read_text()))
    for entry in k1['files']:
        put(tree, entry['path'], (ROOT/'offload/kvpack/overlay'/entry['path']).read_bytes())
    for step in lock['steps']:
        patch = safe_path(roots[step['kind']],step['path'])
        result = subprocess.run(['patch','--batch','--forward','--fuzz=0','--no-backup-if-mismatch',
                                 '-p1','-d',str(tree),'-i',str(patch)],text=True,capture_output=True)
        if result.returncode or 'fuzz' in result.stdout.lower():
            raise ValueError(f"patch failed: {step['path']}\n{result.stdout}\n{result.stderr}")
        print(f"fuzz0: {step['path']}")
    if list(tree.rglob('*.rej')) or list(tree.rglob('*.orig')):
        raise ValueError('patch left rejects/backups')
    entries: list[Entry] = []
    with tempfile.TemporaryDirectory(prefix='up1-compile-') as compile_tmp:
        for path in sorted(tree.rglob('*')):
            if not path.is_file():
                continue
            name = str(path.relative_to(tree))
            if path.suffix == '.py':
                py_compile.compile(str(path),cfile=str(Path(compile_tmp)/'check.pyc'),doraise=True)
            if path.suffix == '.sh':
                subprocess.run(['bash','-n',str(path)],check=True)
            original = image / name
            entries.append({'path':name,'sha256':sha(path),
                            'image_base_sha256':sha(original) if original.is_file() else None})
    manifest: Manifest = {'donor_sha':lock['donor_sha'],'inputs_lock_sha256':sha(HERE/'inputs.lock.json'),
            'steps':lock['steps'],'fuzz':0,'files':entries,
            'qualification':'CPU source composition and syntax only; no CUDA/JIT/GB10 execution or speed claim'}
    return manifest


def verify_tree(tree: Path, manifest: Manifest) -> None:
    actual = {str(p.relative_to(tree)) for p in tree.rglob('*') if p.is_file()}
    expected = {e['path'] for e in manifest['files']}
    if actual != expected:
        raise ValueError(f'overlay file set differs: {sorted(actual ^ expected)}')
    for entry in manifest['files']:
        if sha(safe_path(tree,entry['path'])) != entry['sha256']:
            raise ValueError(f"overlay hash drift: {entry['path']}")


def mounts(manifest: Manifest) -> list[tuple[str, str]]:
    pairs: list[tuple[str,str]] = []
    for entry in manifest['files']:
        name = entry['path']
        if name.startswith(('vllm/','vllm_exl3_moe/')):
            pairs.append(('offload/upstream_adopt/overlay/'+name,PREFIX+name))
    pairs += [
        ('offload/upstream_adopt/overlay/offload/dspark/spark/remote_dspark',
         PREFIX+'vllm/v1/worker/gpu/spec_decode/remote_dspark'),
    ]
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--mounts',action='store_true')
    args = parser.parse_args()
    if args.mounts:
        manifest = cast(Manifest,json.loads((HERE/'MANIFEST.json').read_text()))
        verify_tree(HERE/'overlay',manifest)
        print(' '.join(f'{a}={b}' for a,b in mounts(manifest)))
        return
    with tempfile.TemporaryDirectory(prefix='up1-rebuild-') as tmp:
        tree = Path(tmp)/'tree'
        manifest = build(tree)
        body = json.dumps(manifest,indent=2)+'\n'
        if args.output:
            if args.output.exists():
                raise ValueError('--output must not already exist')
            sidecar = args.output.parent/(args.output.name+".manifest.json")
            if sidecar.exists():
                raise ValueError("output manifest already exists")
            shutil.copytree(tree,args.output)
            sidecar.write_text(body)
        elif args.write:
            # Remove only previous generated files with a known manifest, never a shared source directory.
            if (HERE/'MANIFEST.json').exists():
                prior = cast(Manifest,json.loads((HERE/'MANIFEST.json').read_text()))
                verify_tree(HERE/'overlay',prior)
                for entry in prior['files']:
                    safe_path(HERE/'overlay',entry['path']).unlink()
            shutil.copytree(tree,HERE/'overlay',dirs_exist_ok=True)
            (HERE/'MANIFEST.json').write_text(body)
        else:
            if (HERE/'MANIFEST.json').read_text() != body:
                raise ValueError('MANIFEST.json differs from rebuild')
            verify_tree(HERE/'overlay',manifest)
        print(f"verified {len(manifest['files'])} files, pins, fuzz0, py_compile and shell syntax")


if __name__ == '__main__':
    main()
