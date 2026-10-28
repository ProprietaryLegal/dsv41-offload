"""Offline validation of decode_copy's v1 receipt; no remote imports or access."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import cast

from .assembly_paths import AssemblyPaths


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def obj(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError('copy receipt requires metadata objects')
    return cast(dict[str, object], value)


def verify_copy(root: Path) -> str:
    """Hash every copied byte once per prepare/build; accept additive receipt fields.

    inventories identify exactly what was frozen; files bind each transferred
    payload's full digest (capture payload inventories deliberately omit hashes).
    Source inode/timestamps are historical, never compared to destination stats.
    """
    paths = AssemblyPaths(root, {})
    receipt = paths.inside(root / 'COPY_RECEIPT.json')
    receipt_hash = digest(receipt)
    check = obj(json.loads(receipt.read_text()))
    if check.get('schema') != 'd2.decode-copy.v1' or check.get('frozen') is not True:
        raise ValueError('copy receipt is not a frozen decode copy')
    nonce = check.get('nonce')
    if not isinstance(nonce, str) or not re.fullmatch('[A-Za-z0-9_-]{1,120}', nonce):
        raise ValueError('copy receipt nonce missing or unsafe')
    inventories = obj(check.get('inventories'))
    groups = {'capture', 'generator'} | ({'top20'} if check.get('top20') is not None else set())
    if set(inventories) != groups:
        raise ValueError('copy receipt inventory tree coverage mismatch')
    files = check.get('files')
    if not isinstance(files, list) or not files:
        raise ValueError('copy receipt file records missing')
    records: dict[tuple[str, str], dict[str, object]] = {}
    for raw in files:
        row = obj(raw)
        tree, relative = row.get('tree'), row.get('relative')
        if not isinstance(tree, str) or tree not in groups or not isinstance(relative, str):
            raise ValueError('copy receipt file identity missing')
        rel = Path(relative)
        if rel.is_absolute() or '..' in rel.parts or str(rel) != relative or relative == '.':
            raise ValueError('unsafe copy receipt relative path')
        key = tree, relative
        if key in records:
            raise ValueError('duplicate copy receipt file')
        records[key] = row
    expected: set[tuple[str, str]] = set()
    for tree in sorted(groups):
        inventory = obj(inventories[tree])
        if not inventory:
            raise ValueError('empty copy inventory')
        directory = paths.inside(root / tree)
        actual: set[str] = set()
        for path in directory.rglob('*'):
            paths.inside(path)
            if path.is_file():
                actual.add(str(path.relative_to(directory)))
            elif not path.is_dir():
                raise ValueError('nonregular copied file')
        if actual != set(inventory):
            raise ValueError('copy receipt file coverage mismatch: ' + tree)
        for relative, raw in inventory.items():
            key = tree, relative
            expected.add(key)
            record, original = records.get(key), obj(raw)
            if record is None or type(original.get('size')) is not int or record.get('size') != original['size']:
                raise ValueError('copy receipt size/record mismatch')
            sha = record.get('sha256')
            if not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{64}', sha):
                raise ValueError('copy receipt digest missing')
            if 'sha256' in original and original['sha256'] != sha:
                raise ValueError('copy receipt source/file digest mismatch')
            path = paths.inside(root / tree / relative)
            if path.stat().st_size != record['size'] or digest(path) != sha:
                raise ValueError('copy receipt file hash/size mismatch: ' + tree + '/' + relative)
    if set(records) != expected:
        raise ValueError('copy receipt extra file records')
    for name in ('FLUSH.json', 'FLUSHED.json'):
        flush = obj(json.loads((root / 'capture' / name).read_text()))
        if flush.get('nonce') != nonce:
            raise ValueError('copy receipt flush nonce mismatch')
    if digest(receipt) != receipt_hash:
        raise ValueError('copy receipt changed during verification')
    return receipt_hash
