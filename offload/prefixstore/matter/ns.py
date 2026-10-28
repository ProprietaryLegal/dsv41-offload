#!/usr/bin/env python3
"""Matter namespace derivation (Controller side): a CLI over the one definition in ``dsv41_matter_client.py``.

    matter_ns = HMAC-SHA256(K_matter, b"dsv41-matter-ns-v1\\0" + canonical(folder_name)).hexdigest()

``folder_name`` is the last component of a client-folder string; ``canonical`` = Unicode NFKC, casefold, every
run of whitespace collapsed to one space, stripped. The legal pipeline (through the client helper), the exclusion
derivation and purge-by-matter all use ``derive_ns`` with the same key, so a casing, spacing or path difference
between the exclusion list and a client folder name cannot defeat an exclusion. The key never leaves
Controller's private directory; names never reach a Spark, a log or git: only these 64-hex digests do.

CLI (names on stdin, one per line; prints only digests, in input order)::

    python3 offload/prefixstore/matter/ns.py --key /opt/dsv41/data/private/matter-ns-hmac.key < names
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsv41_matter_client import (  # noqa: E402  (the single definition; re-exported for the tools)
    DOMAIN,
    KEY_BYTES,
    MatterInputError as InputError,
    canonical,
    derive_ns,
    load_key,
    read_private,
)

__all__ = ['DEFAULT_KEY', 'DOMAIN', 'KEY_BYTES', 'InputError', 'canonical', 'derive_ns', 'load_key', 'read_private']

DEFAULT_KEY = '/opt/dsv41/data/private/matter-ns-hmac.key'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--key', default=DEFAULT_KEY)
    args = parser.parse_args(argv)
    try:
        key = load_key(args.key)
        for number, line in enumerate(sys.stdin, 1):
            try:
                print(derive_ns(key, line))
            except InputError as exc:
                raise InputError(f'stdin line {number}: {exc}') from None
    except InputError as exc:
        print(f'ns: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
