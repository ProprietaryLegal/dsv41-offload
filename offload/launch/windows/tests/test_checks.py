"""Meaningful parser regressions on recorded aggregate logs plus malformed inputs."""
from __future__ import annotations

import contextlib
import io
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from typing import Any

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('window_checks', HERE / 'checks.py')
assert spec and spec.loader
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


class Checks(unittest.TestCase):
    def test_recorded_bench(self) -> None:
        rows = checks.bench(ROOT / 'offload/results/E/E-tp2-short.jsonl')
        self.assertGreater(len(rows), 0)
        self.assertTrue(all(r['accepted_per_round'] > 0 and r['ttft_s'] > 0 for r in rows))

    def test_recorded_kv(self) -> None:
        path = ROOT / 'offload/results/C/startup-C-shadow-rank0.txt'
        text = path.read_text()
        values = checks.re.findall(r'GPU KV cache size: ([\d,]+)', text)
        self.assertTrue(values)
        expected = int(values[0].replace(',', ''))
        self.assertEqual(checks.kv(text, expected), expected)
        with self.assertRaises(ValueError): checks.kv(text, expected + 1)
        with self.assertRaises(ValueError): checks.kv('startup succeeded', expected)

    def test_recorded_counters_and_failure(self) -> None:
        root = ROOT / 'offload/results/W4-E0'
        before = checks.load(root / 'rank0-stats-before.json')
        after = checks.load(root / 'rank0-stats-after.json')
        with self.assertRaises(ValueError): checks.counter_delta(before, after) # recorded startup had no stats yet
        before = checks.load(root / 'rank0-stats-after.json')
        checks.counter_delta(before, after)
        after['counters']['fallback_internal_error'] = 1
        with self.assertRaises(ValueError): checks.counter_delta(before, after)

    def test_strict_memory_and_prefix(self) -> None:
        checks.memory('MemAvailable: 5242880 kB\nSwapFree: 100 kB\n')
        for bad in ('MemAvailable: 5242879 kB\nSwapFree: 100 kB', 'MemAvailable: 9999999 kB', ''):
            with self.assertRaises(ValueError): checks.memory(bad)
        for rank in (0, 1):
            line = f'EXL3 RDMA all-reduce: rank {rank} of 2 on rocep1s0f1,roceP2p1s0f1, gid 3, proxy on cpu 5'
            checks.dual(line, rank)
            with self.assertRaises(ValueError): checks.dual('prefix ' + line, rank)
            with self.assertRaises(ValueError): checks.dual(line.replace('roceP2p1s0f1', 'roceP2p1s0f10'), rank)
            with self.assertRaises(ValueError): checks.dual(line + '\nRDMA all-reduce off x NCCL stays', rank)

    def test_replay_stop_strings(self) -> None:
        for text in ('invalid replay tap provenance', 'replay skip would remove taps'):
            with self.assertRaises(ValueError): checks.clean_logs(text)

    def test_receipt_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / '.run-start').touch()
            source = root / 'source.txt'; source.write_text('aggregate diagnostic evidence')
            data: dict[str, Any] = {'schema': 'wr1-gate-v1', 'gate': 'idle', 'arm': 'test', 'run_dir': str(root),
                    'checks': dict.fromkeys(checks.REQUIRED['idle'], True),
                    'evidence': [{'path': str(source), 'sha256': checks.hashlib.sha256(source.read_bytes()).hexdigest()}]}
            target = root / 'receipt.json'; target.write_text(json.dumps(data))
            checks.receipt(target, 'idle', 'test', root)
            source.write_text('changed')
            with self.assertRaises(ValueError): checks.receipt(target, 'idle', 'test', root)
            data['checks']['experiment_idle'] = 'true'
            target.write_text(json.dumps(data))
            with self.assertRaises(ValueError): checks.receipt(target, 'idle', 'test', root)

    def test_native_dry_run_no_transport_or_writes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            env = dict(checks.os.environ, WINDOW_OUT=td + '/must-not-exist')
            fake = Path(td) / 'ssh'
            fake.write_text('#!/bin/sh\nexit 99\n'); fake.chmod(0o755)
            env['PATH'] = td + ':' + env['PATH']
            for script, args in [('native_pair.sh', ['r000008_b0']), ('native.sh', ['control.env', '100', '100', '/missing-timings.json']), ('up1.sh', ['2502600']), ('rp1.sh', ['2502600', '2502600', 'a' * 64, '0-4,10-13']), ('depth.sh', ['2502600', 'control.env', 'sp2-fixed.env', 'sp2-adaptive.env'])]:
                proc = subprocess.run(['bash', str(HERE / script), '--dry-run', *args], env=env, capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn('STEP summary', proc.stdout)
            self.assertFalse(Path(env['WINDOW_OUT']).exists())

    def test_order_and_required_dry_run_steps(self) -> None:
        env = dict(checks.os.environ, WINDOW_OUT='/tmp/wr1-uncreated-test-plan')
        proc = subprocess.run(['bash', str(HERE / 'up1.sh'), '--dry-run', '2502600'], env=env, capture_output=True, text=True, check=True)
        labels = ['p0-r1-K1-start', 'p0-r1-C-start', 'p0-r1-K2-start', 'up1-split-r1-X-start',
                  'up1-split-pin-r1-X-start', 'up1-cpu-pin-r1-X-start', 'up1-replay-r1-X-start',
                  'up1-k5-r1-X-start', 'up1-dual-ring-r1-X-start', 'up1-vor-copy-r1-X-start',
                  'up1-vor-swa-r1-X-start', 'up1-vor-both-r1-X-start']
        positions = [proc.stdout.index('STEP ' + label + ':') for label in labels]
        self.assertEqual(positions, sorted(positions))
        for rank in (0, 1):
            self.assertIn(f'up1-split-r1-X-runtime-hash-rank{rank}', proc.stdout)
            self.assertIn(f'up1-dual-ring-r1-X-dual-rank{rank}', proc.stdout)
        self.assertIn('UP1_SPLIT=63', proc.stdout)
        self.assertIn('up1-replay-r1-X-stats-before', proc.stdout)
        self.assertIn('up1-replay-r1-X-stats-after', proc.stdout)
        self.assertNotIn('up1-all-', proc.stdout)

    def test_invalid_arguments_stop_before_transport(self) -> None:
        for script, args in [('native.sh', ['control.env', 'bad', '100', '/missing']),
                             ('rp1.sh', ['100', '100', 'bad-hash', '0-4']),
                             ('depth.sh', ['100', 'same.env', 'same.env'])]:
            proc = subprocess.run(['bash', str(HERE / script), '--dry-run', *args], capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertNotIn('STEP hold:', proc.stdout)

    def test_timing_requires_evidence_and_cost_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / 'costs.json'; source.write_text('aggregate timing evidence')
            data: dict[str, Any] = {'classification': 'MEASURED', 'arm': 'native-hybrid-kx2', 'draft_ms': 6.5,
                    'round_ms': [10, 11, 12, 13, 14, 15], 'overlap_audited': True, 'verify_only': True,
                    'sources': [{'path': str(source), 'sha256': checks.hashlib.sha256(source.read_bytes()).hexdigest()}]}
            path = Path(td) / 'timing.json'; path.write_text(json.dumps(data))
            with contextlib.redirect_stdout(io.StringIO()): checks.timing(path)
            data['round_ms'] = [10, 11, 12, 10, 14, 15]; path.write_text(json.dumps(data))
            with self.assertRaises(ValueError): checks.timing(path)
            data['round_ms'] = [10, 11, 12, 13, 14, 15]
            data['sources'] = []; path.write_text(json.dumps(data))
            with self.assertRaises(ValueError): checks.timing(path)

    def test_step_propagates_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            command = f'source "{HERE}/common.sh"; W_DRY=0; W_OUT="{td}"; w_step failure bash -c "exit 23"; touch "{td}/unreachable"'
            proc = subprocess.run(['bash', '-c', command], capture_output=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertFalse((Path(td) / 'unreachable').exists())
            self.assertIn('rc=23', (Path(td) / 'steps.log').read_text())


if __name__ == '__main__':
    unittest.main()
