#!/usr/bin/env python3
"""Derive the hash-only excluded-namespace file from NDM's private exclusion list (Controller only).

Reads the exclusion list (one client folder name per line; '#' comments and blank lines ignored; a path is
reduced to its last component) and the HMAC key, both owner-only files under /opt/dsv41/data/private/.
The list must match its pinned sha256. Namespaces come from ``dsv41_matter_client.derive_ns`` (the definition the
harnesses' client helper uses). Writes, atomically and mode 0600, one 64-hex namespace per line (sorted, unique)
plus ``<out>.sha256`` (``sha256sum`` format), and prints only the entry count and the output paths. No name,
namespace or digest derived from a name is printed or logged; errors cite line numbers only. The output is exactly
the format the Spark-side gate accepts (``dsv41_matter.parse_excluded``). Stage it on the Sparks and pass the first
field of the ``.sha256`` file as PREFIX_MATTER_EXCLUDED_SHA256.

``--add-synthetic-ns HEX`` (repeatable) adds synthetic test namespaces for a window's exclusion test; it is
refused for the production output path.

    python3 offload/prefixstore/matter/derive_excluded_ns.py
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsv41_matter_client import parse_names
from ns import DEFAULT_KEY, InputError, derive_ns, load_key, read_private

PRIVATE = Path('/opt/dsv41/data/private')
DEFAULT_LIST = PRIVATE / 'matter-exclusions.txt'
DEFAULT_LIST_SHA256 = ''  # public release: pass --expect-sha256 (sha256 of your own reviewed list)
DEFAULT_OUT = PRIVATE / 'matter-cache-excluded-ns.txt'
HEX64 = re.compile(r'[0-9a-f]{64}')


def derive(key: bytes, data: bytes, synthetic: list[str]) -> list[str]:
    out: set[str] = set()
    for number, name in enumerate(parse_names(data), 1):
        try:
            out.add(derive_ns(key, name))
        except InputError as exc:
            raise InputError(f'exclusion entry {number}: {exc}') from None
    for value in synthetic:
        if HEX64.fullmatch(value) is None:
            raise InputError('synthetic namespace must be 64 lowercase hex')
        out.add(value)
    return sorted(out)


def write_private_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        os.fchmod(fd, 0o600)
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--list', type=Path, default=DEFAULT_LIST)
    parser.add_argument('--expect-sha256', default=DEFAULT_LIST_SHA256,
                        help='pinned sha256 of the exclusion list (change only with a new reviewed list)')
    parser.add_argument('--key', default=DEFAULT_KEY)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--add-synthetic-ns', action='append', default=[])
    args = parser.parse_args(argv)
    try:
        if args.add_synthetic_ns and args.out.resolve() == DEFAULT_OUT.resolve():
            raise InputError('synthetic namespaces never go into the production output; pass a window --out')
        data = read_private(str(args.list), 'exclusion list')
        if hashlib.sha256(data).hexdigest() != args.expect_sha256:
            raise InputError('exclusion list: sha256 differs from the pinned value')
        namespaces = derive(load_key(args.key), data, args.add_synthetic_ns)
        payload = ''.join(f'{ns}\n' for ns in namespaces).encode('ascii')
        digest = hashlib.sha256(payload).hexdigest()
        write_private_atomic(args.out, payload)
        write_private_atomic(args.out.with_name(args.out.name + '.sha256'), f'{digest}  {args.out.name}\n'.encode())
    except InputError as exc:
        print(f'derive_excluded_ns: {exc}', file=sys.stderr)
        return 2
    print(f'wrote {len(namespaces)} excluded namespaces (hashes only) to {args.out} and {args.out.name}.sha256')
    return 0


if __name__ == '__main__':
    sys.exit(main())
