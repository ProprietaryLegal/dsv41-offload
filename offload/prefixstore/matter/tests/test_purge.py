"""MC1 item 6: purge CLI offline and online, expiry, listing; StoreUsage and scan helpers (synthetic data)."""
from __future__ import annotations

import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import purge

m = purge.m


def ns() -> str:
    return secrets.token_hex(32)


def make_ns(root: Path, n: str, blocks: int, age_s: float = 0.0) -> list[Path]:
    base = Path(m.ns_root(str(root), n)) / 'model_abc_r0'
    out = []
    for i in range(blocks):
        d = base / f'{i:03x}' / f'{i:02x}_g0'
        d.mkdir(parents=True, exist_ok=True)
        data = d / f'{secrets.token_hex(32)}.bin'
        data.write_bytes(b'\0' * 4096)
        (d / (data.name + '.crc')).write_text('00000000')
        t = time.time() - age_s - i
        os.utime(data, (t, t))
        out.append(data)
    (base / 'config.json').write_text('{}')
    return out


class PurgeCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'store'
        (self.root / m.NS_DIR).mkdir(parents=True)
        self.audit = str(Path(self.tmp.name) / 'audit.jsonl')

    def run_cli(self, *args: str) -> tuple[int, str]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code = purge.main(list(args))
        return code, buf.getvalue()

    def audit_lines(self) -> list[dict[str, object]]:
        return [json.loads(line) for line in Path(self.audit).read_text().splitlines()]

    def test_offline_purge_namespace_and_all(self) -> None:
        a, b, c = ns(), ns(), ns()
        for x in (a, b, c):
            make_ns(self.root, x, 3)
        code, out = self.run_cli('purge', '--root', str(self.root), '--ns', a, '--audit', self.audit)
        self.assertEqual(code, 0, out)
        self.assertEqual(m.namespaces(str(self.root)), sorted([b, c]))
        code, out = self.run_cli('purge', '--root', str(self.root), '--all', '--audit', self.audit)
        self.assertEqual(code, 0, out)
        self.assertEqual(m.namespaces(str(self.root)), [])
        lines = self.audit_lines()
        self.assertEqual([(r['op'], r['ns'], r['rank'], r['by']) for r in lines],
                         [('purge', a, -1, 'cli'), ('purge', None, -1, 'cli')])
        self.assertEqual(lines[0]['files'], 7)  # 3 blocks + 3 sidecars + config
        self.assertEqual(os.stat(self.audit).st_mode & 0o777, 0o600)

    def test_online_purge_files_a_request_and_deletes_nothing(self) -> None:
        a = ns()
        make_ns(self.root, a, 2)
        fd = os.open(self.root / m.LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # an engine holds the root
            code, out = self.run_cli('purge', '--root', str(self.root), '--ns', a, '--audit', self.audit)
            self.assertEqual(code, 0)
            self.assertIn('requested', out)
            self.assertEqual(m.namespaces(str(self.root)), [a])
            self.assertEqual(m.take_purge_requests(str(self.root)), [a])
            code, out = self.run_cli('expire', '--root', str(self.root), '--ttl-days', '1', '--audit', self.audit)
            self.assertEqual(code, 3)
        finally:
            os.close(fd)
        self.assertFalse(Path(self.audit).exists())

    def test_offline_expire_and_list(self) -> None:
        old, mixed = ns(), ns()
        make_ns(self.root, old, 2, age_s=40 * 86400)
        stale = make_ns(self.root, mixed, 2, age_s=40 * 86400)
        fresh = Path(m.ns_root(str(self.root), mixed)) / 'model_abc_r0' / 'fff' / 'ff_g0'
        fresh.mkdir(parents=True)
        (fresh / 'k.bin').write_bytes(b'\0' * 4096)
        code, out = self.run_cli('list', '--root', str(self.root))
        self.assertEqual(code, 0)
        self.assertIn(f'{old} blocks=2', out)
        code, out = self.run_cli('expire', '--root', str(self.root), '--ttl-days', '30', '--audit', self.audit)
        self.assertEqual(code, 0, out)
        self.assertEqual(m.namespaces(str(self.root)), [mixed])
        self.assertFalse(any(p.exists() for p in stale))
        self.assertTrue((fresh / 'k.bin').exists())
        self.assertEqual(sorted(r['ns'] for r in self.audit_lines()), sorted([old, mixed]))

    def test_refusals(self) -> None:
        code, _ = self.run_cli('purge', '--root', str(self.root), '--ns', 'abc', '--audit', self.audit)
        self.assertEqual(code, 2)
        code, _ = self.run_cli('list', '--root', str(Path(self.tmp.name) / 'nope'))
        self.assertEqual(code, 2)


class UsageTests(unittest.TestCase):
    def test_admission_and_watermarks(self) -> None:
        u = m.StoreUsage(max_bytes=100, max_ns_bytes=60)
        self.assertEqual(u.admit('a', 10), 'scan-pending')
        u.ready = True
        self.assertIsNone(u.admit('a', 60))
        self.assertEqual(u.admit('a', 61), 'ns-cap')
        u.add('a', 55, 1.0)
        u.add('b', 40, 2.0)
        self.assertEqual(u.admit('b', 10), 'tier-cap')
        self.assertTrue(u.over_high())
        self.assertEqual(u.lru_namespaces(), ['a', 'b'])
        u.touch('a', 3.0)
        self.assertEqual(u.lru_namespaces(), ['b', 'a'])
        u.set_ns('b', 0, 0.0)
        self.assertEqual((u.namespaces(), u.total), (['a'], 55))
        with self.assertRaises(ValueError):
            m.StoreUsage(0, 1)
        with self.assertRaises(ValueError):
            m.StoreUsage(10, 10, high=0.5, low=0.6)

    def test_scan_orders_oldest_first_and_remove_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = ns()
            files = make_ns(root, a, 3)
            blocks, other = m.scan_namespace(str(root), a)
            self.assertEqual([b.path for b in blocks], [str(p) for p in reversed(files)])  # older = larger i
            self.assertEqual(blocks[0].nbytes, 4096 + 8)
            self.assertEqual(other, 2)  # config.json
            self.assertEqual(m.remove_block(blocks[0]), 4104)
            self.assertFalse(Path(blocks[0].path + '.crc').exists())
            m.prune_empty_dirs(m.ns_root(str(root), a))
            self.assertEqual(len(m.scan_namespace(str(root), a)[0]), 2)


if __name__ == '__main__':
    unittest.main()
