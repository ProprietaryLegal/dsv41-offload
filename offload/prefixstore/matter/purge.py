#!/usr/bin/env python3
"""Purge CLI for the Tier 1 matter store (run on the Spark that holds it, as the owner of its files).

    purge.py list   --root R
    purge.py purge  --root R (--ns HEX | --all) --audit A
    purge.py expire --root R --ttl-days N --audit A

The store root is the host side of the container's matter root (default container path /root/.cache/kv_matter_mc1
= ~/.cache/dsv41-serve/kv_matter_mc1 on the host); its files are owned by the container user, so run it with sudo.

If no engine holds the root's lock, ``purge`` and ``expire`` delete at once and append to the audit log (rank -1 =
this CLI). While an engine runs, ``purge`` files a request (``<root>/purge-requests/<ns>.req``) that the engine's
janitor executes, refusing that namespace immediately, within its polling interval; ``expire`` is refused (the
engine expires by itself). Namespaces only: map a matter to its namespace on Controller with ``matter/ns.py``.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import importlib.util
import os
from pathlib import Path
import sys
import time
from collections.abc import Iterator
from types import ModuleType

GATE = Path(__file__).resolve().parents[1] / 'overlay/vllm/v1/kv_offload/tiering/fs/dsv41_matter.py'


def gate_module() -> ModuleType:
    """The same helpers the engine's tier uses (one implementation of layout, scan and purge)."""
    spec = importlib.util.spec_from_file_location('dsv41_matter_cli', GATE)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load {GATE}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


m = gate_module()


@contextlib.contextmanager
def offline(root: Path) -> Iterator[bool]:
    """Yields True while holding the root's lock (no engine), False if an engine holds it."""
    fd = os.open(root / m.LOCK_FILE, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def day(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime('%Y-%m-%d') if ts else '-'


def cmd_list(root: Path) -> int:
    total = 0
    for ns in m.namespaces(str(root)):
        blocks, other = m.scan_namespace(str(root), ns)
        size = sum(b.nbytes for b in blocks) + other
        total += size
        oldest = min((b.mtime for b in blocks), default=0.0)
        newest = max((b.mtime for b in blocks), default=0.0)
        print(f'{ns} blocks={len(blocks)} bytes={size} oldest_use={day(oldest)} newest_use={day(newest)}')
    pending = sorted(p.name for p in (root / m.PURGE_DIR).glob('*.req')) if (root / m.PURGE_DIR).is_dir() else []
    print(f'total bytes={total}; pending purge requests={len(pending)}')
    return 0


def cmd_purge(root: Path, target: str, audit_path: str) -> int:
    with offline(root) as free:
        if not free:
            requests = root / m.PURGE_DIR
            requests.mkdir(mode=0o700, exist_ok=True)
            with contextlib.suppress(FileExistsError):
                os.close(os.open(requests / f'{target}.req', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
            print(f'engine running: purge of {target} requested; its janitor refuses it now and deletes it shortly')
            return 0
        audit = m.AuditLog(audit_path, rank=-1)
        try:
            targets = m.namespaces(str(root)) if target == m.PURGE_ALL else [target]
            files = size = 0
            for ns in targets:
                f, s = m.purge_namespace(str(root), ns)
                files += f
                size += s
            audit.write('purge', None if target == m.PURGE_ALL else target,
                        scope='all' if target == m.PURGE_ALL else 'namespace', files=files, bytes=size, by='cli')
        finally:
            audit.close()
    print(f'purged {target}: {files} files, {size} bytes')
    return 0


def cmd_expire(root: Path, ttl_days: float, audit_path: str) -> int:
    if ttl_days <= 0:
        raise ValueError('--ttl-days must be positive')
    with offline(root) as free:
        if not free:
            print('engine running: it expires files itself; refusing an offline expiry', file=sys.stderr)
            return 3
        audit = m.AuditLog(audit_path, rank=-1)
        cutoff = time.time() - ttl_days * 86400.0
        try:
            for ns in m.namespaces(str(root)):
                blocks, _ = m.scan_namespace(str(root), ns)
                old = [b for b in blocks if b.mtime < cutoff]
                freed = sum(m.remove_block(b) for b in old)
                if len(old) == len(blocks):
                    f, s = m.purge_namespace(str(root), ns)
                    freed += s
                else:
                    m.prune_empty_dirs(m.ns_root(str(root), ns))
                if old:
                    audit.write('expire', ns, files=len(old), bytes=freed, by='cli')
                    print(f'{ns}: expired {len(old)} blocks, {freed} bytes')
        finally:
            audit.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('list')
    p.add_argument('--root', type=Path, required=True)
    p = sub.add_parser('purge')
    p.add_argument('--root', type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--ns')
    group.add_argument('--all', action='store_true')
    p.add_argument('--audit', required=True)
    p = sub.add_parser('expire')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--ttl-days', type=float, required=True)
    p.add_argument('--audit', required=True)
    args = parser.parse_args(argv)
    if not (args.root / m.NS_DIR).is_dir():
        print(f'{args.root} is not a matter store root (no {m.NS_DIR}/)', file=sys.stderr)
        return 2
    if args.cmd == 'list':
        return cmd_list(args.root)
    if args.cmd == 'purge':
        target = m.PURGE_ALL if args.all else args.ns
        if target != m.PURGE_ALL and m.NS_RE.fullmatch(target) is None:
            print('--ns must be 64 lowercase hex', file=sys.stderr)
            return 2
        return cmd_purge(args.root, target, args.audit)
    return cmd_expire(args.root, args.ttl_days, args.audit)


if __name__ == '__main__':
    sys.exit(main())
