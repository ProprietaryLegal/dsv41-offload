"""MC1 item 3: namespace derivation and the hash-only exclusion file. Synthetic names and keys only."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import secrets
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import derive_excluded_ns as d
import ns as nsmod

GATE = HERE.parent / 'overlay/vllm/v1/kv_offload/tiering/fs/dsv41_matter.py'
SYNTHETIC = ['Testperson Alpha', 'Example  Beta  Gamma', 'Sample Delta']


def gate_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location('dsv41_matter_for_derive', GATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DeriveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        os.chmod(self.dir, 0o700)
        self.key = secrets.token_bytes(32)
        self.key_path = self.write('key', self.key)
        body = '# synthetic exclusion list, comment line\n\n' + '\n'.join(SYNTHETIC) + '\n'
        self.list_path = self.write('list.txt', body.encode())
        self.pin = hashlib.sha256(body.encode()).hexdigest()
        self.out = self.dir / 'out' / 'excluded.txt'

    def write(self, name: str, data: bytes, mode: int = 0o600) -> Path:
        path = self.dir / name
        path.write_bytes(data)
        os.chmod(path, mode)
        return path

    def run_main(self, *extra: str) -> tuple[int, str]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code = d.main(['--list', str(self.list_path), '--expect-sha256', self.pin, '--key', str(self.key_path),
                           '--out', str(self.out), *extra])
        return code, buf.getvalue()

    def assert_no_names(self, text: str) -> None:
        for name in SYNTHETIC:
            for word in name.split():
                self.assertNotIn(word.lower(), text.lower())

    def test_writes_hash_only_file_the_gate_accepts(self) -> None:
        code, output = self.run_main()
        self.assertEqual(code, 0, output)
        data = self.out.read_bytes()
        expected = sorted(nsmod.derive_ns(self.key, n) for n in SYNTHETIC)
        self.assertEqual(data, ''.join(f'{x}\n' for x in expected).encode())
        digest = hashlib.sha256(data).hexdigest()
        self.assertNotIn(digest, output)  # derived from names: only in the private .sha256 file
        self.assertTrue(all(x not in output for x in expected))
        self.assertEqual((self.out.parent / 'excluded.txt.sha256').read_text(), f'{digest}  excluded.txt\n')
        for path in (self.out, self.out.parent / 'excluded.txt.sha256'):
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
        self.assert_no_names(output + data.decode())
        gate = gate_module()
        exclusions = gate.ExclusionSet(str(self.out), digest)
        self.assertIsNone(exclusions.reason)
        self.assertTrue(all(exclusions.is_excluded(x) for x in expected))

    def test_canonicalization_defeats_spacing_and_case(self) -> None:
        base = nsmod.derive_ns(self.key, 'Testperson Alpha')
        for variant in ('  testperson   ALPHA ', 'TESTPERSON\tAlpha\n', 'Ｔｅｓｔｐｅｒｓｏｎ Alpha'):
            self.assertEqual(nsmod.derive_ns(self.key, variant), base)
        self.assertNotEqual(nsmod.derive_ns(self.key, 'Testperson Alphb'), base)
        self.assertNotEqual(nsmod.derive_ns(secrets.token_bytes(32), 'Testperson Alpha'), base)
        # domain-separated: not the bare HMAC of the name
        import hmac
        self.assertNotEqual(base, hmac.new(self.key, b'testperson alpha', hashlib.sha256).hexdigest())

    def test_refusals_never_print_names(self) -> None:
        cases = [('--expect-sha256', '0' * 64)]
        for flag, value in cases:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                code = d.main(['--list', str(self.list_path), flag, value, '--key', str(self.key_path),
                               '--out', str(self.out)])
            self.assertEqual(code, 2)
            self.assert_no_names(buf.getvalue())
        os.chmod(self.list_path, 0o644)
        code, output = self.run_main()
        self.assertEqual(code, 2)
        self.assertIn('0600 required', output)
        os.chmod(self.list_path, 0o600)
        self.key_path.write_bytes(self.key[:31])
        code, output = self.run_main()
        self.assertEqual(code, 2)
        self.assertIn('32 bytes', output)
        self.key_path.write_bytes(self.key)
        link = self.dir / 'link'
        link.symlink_to(self.list_path)
        self.list_path, original = link, self.list_path
        code, output = self.run_main()
        self.assertEqual(code, 2)
        self.list_path = original
        self.assertFalse(self.out.exists())

    def test_empty_list_and_synthetic_entries(self) -> None:
        body = b'# only comments\n\n'
        self.list_path.write_bytes(body)
        self.pin = hashlib.sha256(body).hexdigest()
        code, output = self.run_main()
        self.assertEqual(code, 2)
        self.assertIn('no entries', output)
        body = '\n'.join(SYNTHETIC).encode() + b'\n'
        self.list_path.write_bytes(body)
        self.pin = hashlib.sha256(body).hexdigest()
        extra = secrets.token_hex(32)
        code, output = self.run_main('--add-synthetic-ns', extra)
        self.assertEqual(code, 0, output)
        self.assertIn(extra, self.out.read_text().split())
        code, output = self.run_main('--add-synthetic-ns', 'XYZ')
        self.assertEqual(code, 2)
        with patch.object(d, 'DEFAULT_OUT', self.out):
            code, output = self.run_main('--add-synthetic-ns', extra)
        self.assertEqual(code, 2)
        self.assertIn('production output', output)

    def test_ns_cli_prints_only_digests(self) -> None:
        buf = io.StringIO()
        with patch('sys.stdin', io.StringIO('\n'.join(SYNTHETIC) + '\n')), contextlib.redirect_stdout(buf):
            self.assertEqual(nsmod.main(['--key', str(self.key_path)]), 0)
        lines = buf.getvalue().split()
        self.assertEqual(lines, [nsmod.derive_ns(self.key, n) for n in SYNTHETIC])
        self.assert_no_names(buf.getvalue())


if __name__ == '__main__':
    unittest.main()
