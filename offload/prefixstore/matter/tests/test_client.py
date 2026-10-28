"""The harness-agnostic client helper (dsv41_matter_client.py). Synthetic names and keys only."""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import logging
import os
from pathlib import Path
import secrets
import sys
import tempfile
from types import ModuleType
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import derive_excluded_ns as d
import dsv41_matter_client as mc
import ns as nsmod

GATE = HERE.parent / 'overlay/vllm/v1/kv_offload/tiering/fs/dsv41_matter.py'
SYNTHETIC = ['Testperson Alpha', 'Example  Beta  Gamma', 'Sample Delta']
OTHER = 'Placeholder Omega'


def gate_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location('dsv41_matter_for_client', GATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        os.chmod(self.dir, 0o700)
        self.key = secrets.token_bytes(32)
        self.key_path = self.write('key', self.key)
        self.list_body = ('# synthetic list\n\n' + '\n'.join(SYNTHETIC) + '\n').encode()
        self.list_path = self.write('list.txt', self.list_body)
        self.ns_path = self.dir / 'excluded.txt'
        code = d.main(['--list', str(self.list_path), '--expect-sha256', hashlib.sha256(self.list_body).hexdigest(),
                       '--key', str(self.key_path), '--out', str(self.ns_path)])
        self.assertEqual(code, 0)
        self.ns_pin = hashlib.sha256(self.ns_path.read_bytes()).hexdigest()

    def write(self, name: str, data: bytes, mode: int = 0o600) -> Path:
        path = self.dir / name
        path.write_bytes(data)
        os.chmod(path, mode)
        return path

    def tagger(self, **kw: object) -> mc.MatterTagger:
        args: dict[str, object] = {'exclusion_list': str(self.list_path), 'excluded_ns_file': str(self.ns_path)}
        args.update(kw)
        return mc.MatterTagger(str(self.key_path), **args)  # type: ignore[arg-type]

    def assert_no_names(self, text: str) -> None:
        for name in SYNTHETIC + [OTHER]:
            for word in name.split():
                self.assertNotIn(word.lower(), text.lower())

    # ---------------------------------------------------------------- namespace
    def test_one_definition_and_folder_forms(self) -> None:
        base = mc.derive_ns(self.key, OTHER)
        self.assertEqual(nsmod.derive_ns(self.key, OTHER), base)          # ns.py is the same function
        for form in ('placeholder   OMEGA', 'client folders/Placeholder Omega',
                     '/x/client folders/!Closed/Placeholder Omega/', 'X:\\share\\client-folders\\Placeholder Omega',
                     ' Placeholder Omega \n'):
            self.assertEqual(mc.derive_ns(self.key, form), base, form)
        for bad in ('', '   ', '/', '..', '.', '!Closed', 'client folders/!Closed/', 'C:\\x\\!lf'):
            with self.assertRaises(mc.MatterInputError):
                mc.derive_ns(self.key, bad)
        with self.assertRaises(mc.MatterInputError):
            mc.derive_ns(self.key[:31], OTHER)

    def test_persist_tag_is_what_the_server_gate_allows(self) -> None:
        tag = self.tagger().tag(OTHER)
        self.assertTrue(tag.persist)
        self.assertEqual(tag.reason, mc.PERSIST)
        fields = tag.body_fields()
        ns = mc.derive_ns(self.key, OTHER)
        self.assertEqual(fields, {'kv_transfer_params': {'matter_ns': ns, 'persist': True}, 'cache_salt': ns})
        gate = gate_module()
        self.assertEqual(gate.parse_tag(fields['kv_transfer_params']), (gate.Verdict.ALLOW, ns))
        salt = fields['cache_salt']
        self.assertTrue(0 < len(salt) <= 128 and not any(ch in salt for ch in '@/\\\0'))   # vLLM's cache_salt rules
        excluded = gate.ExclusionSet(str(self.ns_path), self.ns_pin)
        self.assertFalse(excluded.is_excluded(ns))

    # --------------------------------------------------------------- exclusions
    def test_excluded_matters_are_never_tagged_by_either_source(self) -> None:
        variants = ['testperson alpha', 'client folders/Example Beta Gamma', 'X:\\x\\!Closed\\SAMPLE  DELTA']
        for kw in ({}, {'excluded_ns_file': None}, {'exclusion_list': None}):
            tagger = self.tagger(**kw)
            for v in variants:
                tag = tagger.tag(v)
                self.assertEqual((tag.persist, tag.reason, tag.matter_ns, tag.body_fields()),
                                 (False, mc.EXCLUDED, None, {}), (kw, v))
            self.assertTrue(tagger.tag(OTHER).persist)
        gate = gate_module()
        hashes = gate.ExclusionSet(str(self.ns_path), self.ns_pin)
        self.assertTrue(all(hashes.is_excluded(mc.derive_ns(self.key, v)) for v in variants))  # server agrees

    def test_names_layer_is_independent_of_the_hash_file(self) -> None:
        # A hash file that does not list a matter (e.g. stale) does not let the name list's matter through.
        other_ns = sorted({mc.derive_ns(self.key, OTHER)})
        stale = self.write('stale.txt', ''.join(f'{x}\n' for x in other_ns).encode())
        tagger = self.tagger(excluded_ns_file=str(stale))
        self.assertEqual(tagger.tag(SYNTHETIC[0]).reason, mc.EXCLUDED)
        self.assertEqual(tagger.tag(OTHER).reason, mc.EXCLUDED)          # and the hash file's own entry too

    def test_fail_closed_at_start_and_at_runtime(self) -> None:
        for kw in ({'exclusion_list': None, 'excluded_ns_file': None},
                   {'exclusion_list_sha256': '0' * 64},
                   {'excluded_ns_sha256': '0' * 64},
                   {'exclusion_list': str(self.dir / 'missing')}):
            with self.assertRaises(mc.MatterInputError):
                self.tagger(**kw)
        os.chmod(self.list_path, 0o644)
        with self.assertRaises(mc.MatterInputError) as caught:
            self.tagger()
        self.assertIn('0600 required', str(caught.exception))
        os.chmod(self.list_path, 0o600)
        with self.assertRaises(mc.MatterInputError):
            mc.MatterTagger(str(self.write('short', self.key[:16])), exclusion_list=str(self.list_path))

        tagger = self.tagger(excluded_ns_sha256=self.ns_pin)
        self.assertTrue(tagger.tag(OTHER).persist)
        log = io.StringIO()
        handler = logging.StreamHandler(log)
        logging.getLogger('dsv41_matter_client').addHandler(handler)
        self.addCleanup(logging.getLogger('dsv41_matter_client').removeHandler, handler)
        self.ns_path.write_bytes(b'not hex\n')                           # replaced while the harness runs
        self.assertEqual(tagger.tag(OTHER).reason, mc.EXCLUSIONS_UNAVAILABLE)
        self.assertEqual(tagger.tag(OTHER).body_fields(), {})
        good = self.dir / 'good.txt'
        d.write_private_atomic(good, ''.join(f'{x}\n' for x in sorted(mc.derive_ns(self.key, n) for n in SYNTHETIC))
                               .encode())
        os.replace(good, self.ns_path)                                   # same bytes as the pinned file
        self.assertTrue(tagger.tag(OTHER).persist)
        self.list_path.write_bytes(self.list_body + b'Placeholder Omega\n')   # NDM adds a matter: honoured at once
        self.assertEqual(tagger.tag(OTHER).reason, mc.EXCLUDED)
        os.unlink(self.list_path)
        self.assertEqual(tagger.tag('Unlisted Person').reason, mc.EXCLUSIONS_UNAVAILABLE)
        self.assert_no_names(log.getvalue())
        self.assertIn('untagged', log.getvalue())

    # ----------------------------------------------------------- request shape
    def test_apply_merges_and_strips(self) -> None:
        tagger = self.tagger()
        body = {'model': 'm', 'messages': [], 'kv_transfer_params': {'other': 1, 'persist': True,
                                                                     'matter_ns': 'f' * 64}}
        out = tagger.tag(SYNTHETIC[1]).apply(body)                        # excluded: stale tag stripped
        self.assertEqual(out['kv_transfer_params'], {'other': 1})
        self.assertNotIn('cache_salt', out)
        self.assertEqual(body['kv_transfer_params'], {'other': 1, 'persist': True, 'matter_ns': 'f' * 64})
        self.assertNotIn('kv_transfer_params', mc.MatterTag(False, mc.DISABLED).apply({'kv_transfer_params': {
            'persist': True}}))
        tag = tagger.tag(OTHER)
        out = tag.apply({'model': 'm', 'kv_transfer_params': {'other': 1}})
        self.assertEqual(out['kv_transfer_params'], {'other': 1, 'matter_ns': tag.matter_ns, 'persist': True})
        self.assertEqual(out['cache_salt'], tag.matter_ns)
        self.assertEqual(tag.apply({'cache_salt': tag.matter_ns})['cache_salt'], tag.matter_ns)
        with self.assertRaises(mc.MatterInputError):
            tag.apply({'cache_salt': 'another'})
        with self.assertRaises(mc.MatterInputError):
            tag.apply({'kv_transfer_params': 'x'})

    def test_switches_and_invalid_folders(self) -> None:
        self.assertEqual(self.tagger().tag(OTHER, persist=False).reason, mc.NOT_REQUESTED)
        self.assertEqual(self.tagger().tag('!Closed').reason, mc.INVALID_FOLDER)
        off = mc.MatterTagger.from_env({})
        self.assertEqual((off.enabled, off.tag(OTHER).reason), (False, mc.DISABLED))
        with self.assertRaises(mc.MatterInputError):
            off.namespace(OTHER)
        env = {'DSV41_MATTER_CACHE': 'on', 'DSV41_MATTER_KEY_FILE': str(self.key_path),
               'DSV41_MATTER_EXCLUSION_LIST': str(self.list_path),
               'DSV41_MATTER_EXCLUSION_LIST_SHA256': hashlib.sha256(self.list_body).hexdigest(),
               'DSV41_MATTER_EXCLUDED_NS_FILE': str(self.ns_path), 'DSV41_MATTER_EXCLUDED_NS_SHA256': self.ns_pin}
        on = mc.MatterTagger.from_env(env)
        self.assertEqual(on.tag(OTHER).matter_ns, on.namespace(OTHER))
        self.assertEqual(on.tag(SYNTHETIC[2]).reason, mc.EXCLUDED)
        for bad in ({**env, 'DSV41_MATTER_CACHE': 'yes'}, {'DSV41_MATTER_CACHE': '1'},
                    {'DSV41_MATTER_CACHE': '1', 'DSV41_MATTER_KEY_FILE': str(self.key_path)}):
            with self.assertRaises(mc.MatterInputError):
                mc.MatterTagger.from_env(bad)

    # ----------------------------------------------------------------- the CLI
    def test_selfcheck_prints_counts_only(self) -> None:
        (self.dir / 'cf').mkdir()
        (self.dir / 'cf' / 'Testperson Alpha').mkdir()
        (self.dir / 'cf' / '!Closed').mkdir()
        (self.dir / 'cf' / '!Closed' / 'Example Beta Gamma').mkdir()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code = mc.main(['selfcheck', '--key', str(self.key_path), '--exclusion-list', str(self.list_path),
                            '--excluded-ns', str(self.ns_path), '--client-files', str(self.dir / 'cf'),
                            str(self.dir / 'cf' / '!Closed')])
        out = buf.getvalue()
        self.assertEqual(code, 1, out)                                    # Sample Delta has no folder
        self.assertIn('every listed matter is in it: yes', out)
        self.assertIn('2 of 3 exclusion entries match a folder name', out)
        self.assert_no_names(out)
        for x in (mc.derive_ns(self.key, n) for n in SYNTHETIC):
            self.assertNotIn(x, out)
        stale = self.write('stale.txt', f'{mc.derive_ns(self.key, OTHER)}\n'.encode())
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(mc.main(['selfcheck', '--key', str(self.key_path), '--exclusion-list',
                                      str(self.list_path), '--excluded-ns', str(stale)]), 1)
        self.assertIn('every listed matter is in it: NO', buf.getvalue())

    def test_standard_library_only(self) -> None:
        tree = ast.parse(Path(mc.__file__).read_text())
        roots = {(node.module or '').split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        roots |= {a.name.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for a in node.names}
        self.assertEqual(roots - set(sys.stdlib_module_names) - {'__future__'}, set())


if __name__ == '__main__':
    unittest.main()
