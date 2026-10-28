"""Supervise locally spawned fake SSH, never a GPU or remote host."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
FAKE_SSH = '''#!/usr/bin/env python3
import os, sys, time
mode = os.environ.get('FAKE_MODE', 'good')
if 'nvidia-smi' in sys.argv[-1]:
    uuids = ['GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93', 'GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc']
    while True:
        for uuid in uuids:
            print('now, ' + uuid + ', 4000, 100, 40', flush=True)
        time.sleep(.05)
else:
    for index in range(1000):
        if mode == 'closed': sys.exit(0)
        mem = 5242879 if mode == 'low' else 6000000
        swap = 1000 - (index if mode == 'swap' else 0)
        print(f'MemAvailable: {mem} kB;SwapFree: {swap} kB;', flush=True)
        time.sleep(.05)
'''


class Supervisor(unittest.TestCase):
    def invoke(self, root: Path, mode: str, command: list[str], timeout: float = 5) -> subprocess.CompletedProcess[str]:
        fake = root / 'ssh'; fake.write_text(FAKE_SSH); fake.chmod(0o755)
        env = dict(os.environ, PATH=str(root) + ':' + os.environ['PATH'], FAKE_MODE=mode)
        return subprocess.run([sys.executable, str(HERE / 'checks.py'), 'supervise', '--out', str(root),
                               '--name', 'trial', '--hosts', 'fake-rank0', 'fake-rank1', '--x86host', 'fake-x86host',
                               '--timeout', str(timeout), '--', *command], env=env, capture_output=True, text=True, timeout=15)

    def test_success_records_both_cards_and_memory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = self.invoke(root, 'good', [sys.executable, '-c', 'print("owned-command")'])
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn('owned-command', p.stdout)
            self.assertFalse((root / 'STOP').exists())
            self.assertIn('GPU-c4e7a371', (root / 'trial.gpu-fake-x86host.log').read_text())
            self.assertIn('GPU-7744b44b', (root / 'trial.gpu-fake-x86host.log').read_text())
            self.assertEqual(len(json.loads((root / 'swap-baseline.json').read_text())), 2)

    def test_low_or_closed_prevents_command_start(self) -> None:
        for mode in ('low', 'closed'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                p = self.invoke(root, mode, [sys.executable, '-c', f'open({str(root / "must-not-exist")!r}, "w").close()'])
                self.assertNotEqual(p.returncode, 0)
                self.assertTrue((root / 'STOP').exists())
                self.assertFalse((root / 'must-not-exist').exists())

    def test_command_failure_and_timeout(self) -> None:
        for code, timeout in [('raise SystemExit(29)', 5), ('import time; time.sleep(20)', .1)]:
            with self.subTest(code=code), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                p = self.invoke(root, 'good', [sys.executable, '-c', code], timeout)
                self.assertNotEqual(p.returncode, 0)
                self.assertTrue((root / 'STOP').exists())

    def test_swap_abort_preserves_unrelated_process(self) -> None:
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'])
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                p = self.invoke(root, 'swap', [sys.executable, '-c', 'import time; time.sleep(20)'])
                self.assertNotEqual(p.returncode, 0)
                self.assertIn('swap growth', (root / 'STOP').read_text())
                self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate(); unrelated.wait()


if __name__ == '__main__':
    unittest.main()
