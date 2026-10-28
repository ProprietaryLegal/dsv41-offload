"""Run the real-vLLM CPU cases (tests/vllm_cases/cases_*.py) under the kvpack-build venv.

The cases import the actual image source with the K1 and RP1/MC1 overlays applied (vllm_tree.build). A missing
venv or source snapshot FAILS this test; it never skips.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vllm_tree

CASES = Path(__file__).resolve().parent / 'vllm_cases'


# 0091 (R-UPF) mounts async_lookup.py, which the matter tier and the fs tier import. Its ref is pending; at window
# time point MC1_UPF_REF at the ref being promoted. A missing ref FAILS (it is never skipped).
UPF_REF = os.environ.get('MC1_UPF_REF', 'offload/upfix')
ASYNC_LOOKUP = 'vllm/v1/kv_offload/tiering/async_lookup.py'


def git(*args: str) -> bytes:
    return subprocess.run(['git', '-C', str(vllm_tree.ROOT), *args], check=True, capture_output=True).stdout


class VllmCases(unittest.TestCase):
    def test_cases_pass_under_real_vllm(self) -> None:
        self.run_cases(compose_0091=False)

    def test_cases_pass_composed_with_0091_async_lookup(self) -> None:
        self.run_cases(compose_0091=True)

    def run_cases(self, compose_0091: bool) -> None:
        self.assertTrue(vllm_tree.VENV_PYTHON.exists(), f'venv interpreter missing: {vllm_tree.VENV_PYTHON}')
        files = sorted(str(p) for p in CASES.glob('cases_*.py'))
        self.assertTrue(files, 'no vLLM case files found')
        with tempfile.TemporaryDirectory(prefix='mc1-vllm-tree-') as tmp:
            tree = vllm_tree.build(Path(tmp) / 'tree')
            if compose_0091:
                manifest = json.loads(git('show', f'{UPF_REF}:offload/upfix/MANIFEST-0091.json'))
                (entry,) = [e for e in manifest['files'] if e['path'] == ASYNC_LOOKUP]
                data = git('show', f'{UPF_REF}:{entry["overlay"]}')
                self.assertEqual(hashlib.sha256(data).hexdigest(), entry['overlay_sha256'])
                vllm_tree.overlay_bytes(tree, ASYNC_LOOKUP, data)
            env = {k: v for k, v in os.environ.items() if not k.startswith(('VLLM_', 'PREFIX_'))}
            env.update({'PYTHONPATH': str(tree), 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
                        'VLLM_DSV41_KV_CRC': '1', 'MC1_SCRATCH': tmp})
            result = subprocess.run([str(vllm_tree.VENV_PYTHON), '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                                     '-o', 'python_files=cases_*.py', *files],
                                    cwd=tmp, env=env, text=True, capture_output=True, timeout=900)
        tail = '\n'.join((result.stdout + result.stderr).splitlines()[-60:])
        self.assertEqual(result.returncode, 0, tail)
        print(tail.splitlines()[-1] if tail else '')


if __name__ == '__main__':
    unittest.main()
