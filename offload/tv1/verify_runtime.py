"""Emit a self-contained stdlib TV1 runtime hash verifier; performs no network/service work."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import TypedDict, cast


class Entry(TypedDict):
    path: str
    sha256: str


class Runtime(TypedDict):
    files: list[Entry]


EXPECTED: Runtime | None = globals().get('TV1_EXPECTED')


def verify(manifest: Runtime, root: Path) -> int:
    dist = root / 'usr/local/lib/python3.12/dist-packages'
    seen: set[str] = set()
    for e in manifest['files']:
        name = e['path']
        if (name in seen or not name.startswith(('vllm/', 'vllm_exl3_moe/'))
                or '..' in Path(name).parts or Path(name).is_absolute()):
            raise ValueError(f'TV1 unsafe/duplicate runtime path: {name}')
        seen.add(name)
        if hashlib.sha256((dist / name).read_bytes()).hexdigest() != e['sha256']:
            raise ValueError(f'TV1 runtime hash mismatch: {name}')
    if not seen:
        raise ValueError('TV1 empty runtime manifest')
    print(f'TV1 runtime hashes verified: {len(seen)} files; no GPU equivalence certified')
    return len(seen)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--emit', action='store_true')
    p.add_argument('--root', type=Path, default=Path('/'))
    a = p.parse_args()
    expected = EXPECTED
    if a.manifest:
        expected = cast(Runtime, json.loads(a.manifest.read_text()))
    if expected is None:
        raise ValueError('TV1 --manifest required before emitting a verifier')
    if a.emit:
        source = Path(__file__).read_text().replace('from __future__ import annotations\n',
                    'from __future__ import annotations\nTV1_EXPECTED = ' + repr(expected) + '\n', 1)
        sys.stdout.write(source)
    else:
        verify(expected, a.root)


if __name__ == '__main__':
    main()
