"""compose_audit helpers on synthetic sources, plus the MC1-vs-K1 audit on committed history."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compose_audit as ca


class ComposeAuditTests(unittest.TestCase):
    def test_imports_resolve_absolute_and_relative(self) -> None:
        src = 'import os\nfrom vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager\nfrom . import io\n'
        got = ca.imports(src, 'vllm.v1.kv_offload.tiering.fs.manager')
        self.assertIn('vllm.v1.kv_offload.tiering.async_lookup', got)
        self.assertIn('vllm.v1.kv_offload.tiering.fs.io', got)

    def test_remote_dspark_maps_to_its_mount(self) -> None:
        self.assertEqual(ca.mount_of_repo_path('offload/dspark/spark/remote_dspark/round.py'),
                         'vllm/v1/worker/gpu/spec_decode/remote_dspark/round.py')
        with self.assertRaises(ValueError):
            ca.mount_of_repo_path('offload/elsewhere/x.py')

    def test_adjacency_reports_both_directions(self) -> None:
        a = ca.Package('A', 'HEAD', {'vllm/a.py': '(x)'})
        b = ca.Package('B', 'HEAD', {'vllm/b.py': '(y)'})
        sources = {'vllm/a.py': 'from vllm.b import f\n', 'vllm/b.py': 'import vllm.a\n'}
        original = ca.source_of
        ca.source_of = lambda pkg, rel: sources[rel]
        try:
            got = ca.adjacency(a, b, Path('/nonexistent'))
        finally:
            ca.source_of = original
        self.assertEqual(got, ['A: vllm/a.py imports B: vllm/b.py', 'B: vllm/b.py imports A: vllm/a.py'])

    def test_mc1_and_k1_share_no_mount_at_head(self) -> None:
        mine, k1 = ca.mc1('HEAD'), ca.k1('596ba1c')
        self.assertEqual(len(k1.mounts), 13)
        self.assertFalse(set(mine.mounts) & set(k1.mounts))


if __name__ == '__main__':
    unittest.main()
