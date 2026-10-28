"""Emit a self-contained, stdlib-only live file checker. No SSH, CUDA imports or service actions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import TypedDict, cast


class FileEntry(TypedDict):
    path: str
    sha256: str


class RuntimeManifest(TypedDict):
    files: list[FileEntry]


EMBEDDED_EXPECTATIONS: RuntimeManifest | None = globals().get('UP1_PINNED_EXPECTATIONS')


def verify(manifest: RuntimeManifest, root: Path) -> int:
    dist = root/'usr/local/lib/python3.12/dist-packages'
    count = 0
    for entry in manifest['files']:
        name = entry['path']
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError(f'unsafe manifest path: {name}')
        if name.startswith(('vllm/','vllm_exl3_moe/')):
            targets = [dist/name]
        elif name.startswith('offload/dspark/spark/remote_dspark/'):
            targets = [dist/'vllm/v1/worker/gpu/spec_decode/remote_dspark'/Path(name).name]
        else:
            raise ValueError(f'unmounted manifest path: {name}')
        for path in targets:
            got = hashlib.sha256(path.read_bytes()).hexdigest()
            if got != entry['sha256']:
                raise ValueError(f'runtime hash mismatch: {path}: {got}, expected {entry["sha256"]}')
            count += 1
    print(f'UP1 runtime hashes verified: {count} files; no GPU/JIT behavior certified')
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emit',action='store_true')
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--root',type=Path,default=Path('/'))
    args = parser.parse_args()
    if args.emit:
        path = args.manifest or Path(__file__).with_name('MANIFEST.json')
        manifest = cast(RuntimeManifest,json.loads(path.read_text()))
        source = Path(__file__).read_text()
        # Future import must precede the injected literal. repr preserves Python syntax, no shell interpolation.
        source = source.replace('from __future__ import annotations\n',
                                'from __future__ import annotations\nUP1_PINNED_EXPECTATIONS = '+repr(manifest)+'\n',1)
        sys.stdout.write(source)
        return
    expected = EMBEDDED_EXPECTATIONS
    if args.manifest:
        expected = cast(RuntimeManifest,json.loads(args.manifest.read_text()))
    if expected is None:
        raise ValueError('use --emit before piping to container, or supply --manifest')
    verify(expected,args.root)


if __name__ == '__main__':
    main()
