"""MC1 MatterGate unit tests (standard library module; synthetic namespaces and keys only)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
import time
from types import ModuleType
import unittest

HERE = Path(__file__).resolve().parents[1]
MODULE = HERE / 'overlay/vllm/v1/kv_offload/tiering/fs/dsv41_matter.py'


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location('dsv41_matter_under_test', MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


m = load()


def ns() -> str:
    return secrets.token_hex(32)


def write_private(path: Path, data: bytes, mode: int = 0o600) -> str:
    path.write_bytes(data)
    os.chmod(path, mode)
    return hashlib.sha256(data).hexdigest()


def excluded_bytes(*names: str) -> bytes:
    return ''.join(f'{n}\n' for n in sorted(set(names))).encode()


class TagTests(unittest.TestCase):
    def test_default_deny_matrix(self) -> None:
        good = ns()
        cases = [
            (None, m.Verdict.UNTAGGED), ({}, m.Verdict.UNTAGGED), ('x', m.Verdict.UNTAGGED),
            ({'max_load_tokens': 5}, m.Verdict.UNTAGGED),
            ({'matter_ns': good}, m.Verdict.NOT_PERSIST), ({'matter_ns': good, 'persist': False}, m.Verdict.NOT_PERSIST),
            ({'matter_ns': good, 'persist': 'true'}, m.Verdict.BAD_TAG), ({'matter_ns': good, 'persist': 1}, m.Verdict.BAD_TAG),
            ({'persist': True}, m.Verdict.BAD_TAG), ({'matter_ns': good.upper(), 'persist': True}, m.Verdict.BAD_TAG),
            ({'matter_ns': good[:-1], 'persist': True}, m.Verdict.BAD_TAG), ({'matter_ns': 7, 'persist': True}, m.Verdict.BAD_TAG),
            ({'matter_ns': good + '\n', 'persist': True}, m.Verdict.BAD_TAG),
            ({'matter_ns': good, 'persist': True}, m.Verdict.ALLOW),
        ]
        for params, verdict in cases:
            self.assertIs(m.parse_tag(params)[0], verdict, params)


class ExcludedFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.excluded = [ns(), ns(), ns()]
        self.path = self.dir / 'excluded.txt'
        self.pin = write_private(self.path, excluded_bytes(*self.excluded))

    def test_open_and_excludes(self) -> None:
        s = m.ExclusionSet(str(self.path), self.pin)
        self.assertFalse(s.closed)
        self.assertTrue(all(s.is_excluded(x) for x in self.excluded))
        self.assertFalse(s.is_excluded(ns()))

    def test_parse_strictness(self) -> None:
        a, b = sorted([ns(), ns()])
        for bad in (b'', a.encode(), f'{b}\n{a}\n'.encode(), f'{a}\n{a}\n'.encode(), f'{a.upper()}\n'.encode(),
                    f'# c\n{a}\n'.encode(), f'{a}\n\n'.encode(), f'{a} \n'.encode(), f'{a}\r\n'.encode()):
            with self.assertRaises((ValueError, UnicodeDecodeError), msg=repr(bad)):
                m.parse_excluded(bad)
        self.assertEqual(m.parse_excluded(f'{a}\n{b}\n'.encode()), frozenset({a, b}))

    def test_fail_closed_reasons(self) -> None:
        cases: list[tuple[str, str, str]] = []
        cases.append(('missing', str(self.dir / 'nope'), self.pin))
        cases.append(('pin-malformed', str(self.path), 'abc'))
        cases.append(('sha-mismatch', str(self.path), hashlib.sha256(b'x').hexdigest()))
        loose = self.dir / 'loose.txt'
        cases.append(('mode', str(loose), write_private(loose, excluded_bytes(ns()), 0o640)))
        empty = self.dir / 'empty.txt'
        cases.append(('malformed', str(empty), write_private(empty, b'')))
        link = self.dir / 'link.txt'
        link.symlink_to(self.path)
        cases.append(('symlink', str(link), self.pin))
        cases.append(('not-regular', str(self.dir), self.pin))
        for reason, path, pin in cases:
            s = m.ExclusionSet(path, pin)
            self.assertTrue(s.closed, reason)
            self.assertEqual(s.reason, reason)
            self.assertEqual(s.excluded, frozenset())

    def test_change_closes_sticky(self) -> None:
        s = m.ExclusionSet(str(self.path), self.pin, recheck_s=0.0)
        self.assertFalse(s.revalidate(time.monotonic() + 1))
        data = self.path.read_bytes()
        self.path.write_bytes(data)  # same bytes, new mtime: still a change of the pinned file object
        os.utime(self.path, ns=(1, 1))
        self.assertTrue(s.revalidate(time.monotonic() + 2))
        self.assertEqual(s.reason, 'changed')
        self.assertFalse(s.revalidate(time.monotonic() + 3))  # stays closed
        self.path.unlink()
        s2 = m.ExclusionSet(str(self.dir / 'gone'), self.pin, recheck_s=0.0)
        self.assertEqual(s2.reason, 'missing')


class GateAndAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.excluded = ns()
        path = self.dir / 'excluded.txt'
        pin = write_private(path, excluded_bytes(self.excluded))
        self.audit_path = self.dir / 'audit' / 'audit.jsonl'
        self.audit = m.AuditLog(str(self.audit_path), rank=0)
        self.gate = m.MatterGate(m.ExclusionSet(str(path), pin), self.audit)

    def test_decisions(self) -> None:
        allowed = ns()
        self.assertIs(self.gate.decide(self.gate.classify({'matter_ns': allowed, 'persist': True})), m.Verdict.ALLOW)
        self.assertIs(self.gate.decide(self.gate.classify({'matter_ns': self.excluded, 'persist': True})),
                      m.Verdict.EXCLUDED)
        self.assertIs(self.gate.decide(self.gate.classify(None)), m.Verdict.UNTAGGED)
        self.gate.tombstone(allowed, time.monotonic() + 60)
        self.assertIs(self.gate.decide(self.gate.classify({'matter_ns': allowed, 'persist': True})), m.Verdict.PURGING)
        self.assertEqual(self.gate.expire_tombstones(time.monotonic() + 61), [allowed])
        self.assertIs(self.gate.decide(self.gate.classify({'matter_ns': allowed, 'persist': True})), m.Verdict.ALLOW)
        self.gate.audit = None
        self.assertIs(self.gate.decide(self.gate.classify({'matter_ns': allowed, 'persist': True})), m.Verdict.GATE_CLOSED)

    def test_audit_is_append_only_0600_and_name_free(self) -> None:
        a = ns()
        self.audit.write('put', a, chunks=3, tokens=480, bytes=3 * 4096)
        self.audit.close()
        again = m.AuditLog(str(self.audit_path), rank=1)
        again.write('refuse', a, reason='excluded')
        again.close()
        self.assertEqual(os.stat(self.audit_path).st_mode & 0o777, 0o600)
        lines = [json.loads(line) for line in self.audit_path.read_text().splitlines()]
        self.assertEqual([r['op'] for r in lines], ['put', 'refuse'])
        self.assertEqual({tuple(sorted(r)) for r in lines},
                         {('bytes', 'chunks', 'ns', 'op', 'rank', 'tokens', 'ts'), ('ns', 'op', 'rank', 'reason', 'ts')})
        os.chmod(self.audit_path, 0o644)
        with self.assertRaises(ValueError):
            m.AuditLog(str(self.audit_path), rank=0)


class StoreHelperTests(unittest.TestCase):
    def test_purge_namespace_and_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = ns(), ns()
            for x in (a, b):
                d = Path(m.ns_root(str(root), x)) / 'model_r0' / 'abc'
                d.mkdir(parents=True)
                (d / 'k.bin').write_bytes(b'\0' * 4096)
                (d / 'k.bin.crc').write_text('00000000')
            self.assertEqual(m.namespaces(str(root)), sorted([a, b]))
            self.assertEqual(m.purge_namespace(str(root), a), (2, 4104))
            self.assertEqual(m.namespaces(str(root)), [b])
            self.assertEqual(m.purge_namespace(str(root), a), (0, 0))  # idempotent
            with self.assertRaises(ValueError):
                m.ns_root(str(root), '../x')
            outside = root / 'outside'
            outside.mkdir()
            (outside / 'keep.bin').write_bytes(b'1')
            c = ns()
            Path(m.ns_root(str(root), c)).symlink_to(outside)
            with self.assertRaises(ValueError):
                m.purge_namespace(str(root), c)
            self.assertTrue((outside / 'keep.bin').exists())
            req = root / m.PURGE_DIR
            req.mkdir()
            for name in (f'{b}.req', 'ALL.req', 'junk.req', f'{b}.txt'):
                (req / name).write_text('')
            self.assertEqual(sorted(m.take_purge_requests(str(root))), sorted([b, 'ALL']))
            self.assertEqual(sorted(os.listdir(req)), sorted(['junk.req', f'{b}.txt']))

    def test_private_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / 'root'
            d.mkdir(mode=0o755)
            m.ensure_private_dir(str(d))
            self.assertEqual(os.stat(d).st_mode & 0o777, 0o700)


if __name__ == '__main__':
    unittest.main()
