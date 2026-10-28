"""SP1-WINDOW precheck on CPU with a fake remote: no SSH, HTTP, Spark, x86host or GPU."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from offload.launch.windows import pf1_window as w
from offload.launch.windows import sp1_precheck as sc
from offload.launch.windows import sp1_window as s
from offload.launch.windows.tests import test_pf1_precheck as base

SRC_IMAGE = Path('/opt/dsv41/data/src-image')
FIELDS = sorted({*s.PROFILER_CONFIG, 'delay_iterations', 'torch_profiler_with_stack'})


class FakeRemote(base.FakeRemote):
    """PF1's read-only fake plus the SP1 answers: pins, image profiler source, profile directory, Nsight."""

    def __call__(self, host: str, command: str, stdin: str) -> tuple[int, str]:
        for word in ('bash offload/launch/start_exp', 'stop.sh', 'window.sh', 'restore.sh', ':8001/', '18001', 'rm ',
                     'rmdir', 'mkdir', 'git checkout', 'git fetch', 'docker stop', 'docker rm', 'tee ', '> '):
            assert word not in command, f'precheck must stay read-only: {word}'
        bad = self.bad
        if stdin == sc.PROFILER_PROBE:
            self.calls.append((host, command))
            fields = [f for f in FIELDS if not (bad == 'profiler' and f == 'ignore_frontend')]
            return 0, json.dumps({'fields': fields, 'endpoints': ['/start_profile', '/stop_profile'],
                                  'vllm': '0.31.1.dev0', 'torch': '2.13.0+cu130'}) + '\n'
        if 'command -v nsys' in command:
            self.calls.append((host, command))
            return 0, 'none\nnone\n'
        if 'sha256sum offload/' in command:
            self.calls.append((host, command))
            return 0, '\n'.join(f'{"f" * 64 if bad == "pins" else w.digest(self.root / p)}  {p}' for p in s.PIN_FILES) + '\n'
        if s.PROFILE_DIR in command:
            self.calls.append((host, command))
            state = 'PRESENT' if bad == 'profile-dir' and host == 'sparkworker-wired' else 'ABSENT'
            free = 9 * base.GIB if bad == 'disk' and host == 'sparkhead' else 900 * base.GIB
            return 0, f'{state}\n/dev/nvme0n1p2 1000000000 100000000 {free} 10% /\n'
        assert 'prefill_trace' not in command and 'pf1-trace' not in command, 'SP1 needs no PF1 trace sources'
        return super().__call__(host, command, stdin)


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, str, Path]:
    root = tmp_path / 'repo'
    for p in s.PIN_FILES:
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(w.ROOT / p, root / p)
    git = ['git', '-C', str(root), '-c', 'user.name=t', '-c', 'user.email=t@t']
    subprocess.run([*git, 'init', '-q'], check=True)
    subprocess.run([*git, 'add', '.'], check=True)
    subprocess.run([*git, 'commit', '-qm', 'x'], check=True)
    data = tmp_path / 'data'
    data.mkdir()
    for name in ('opinions_cl.jsonl', 'filings_recap.jsonl'):
        (data / name).write_text(json.dumps({'text': 'public record'}) + '\n')
    return root, subprocess.check_output([*git, 'rev-parse', 'HEAD'], text=True).strip(), data


def run(repo: tuple[Path, str, Path], capsys: pytest.CaptureFixture[str], bad: str = '') -> tuple[int, str, FakeRemote]:
    root, expected, data = repo
    fake = FakeRemote(root, expected, bad)
    rc = sc.precheck(expected, None, fake, root=root, data_dir=data, listening=lambda: '')
    return rc, capsys.readouterr().out, fake


def test_all_checks_pass_read_only(repo: tuple[Path, str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    rc, out, fake = run(repo, capsys)
    assert rc == 0 and 'PRECHECK PASS: run sp1_window.sh' in out and 'FAIL' not in out, out
    assert {h for h, _ in fake.calls} == {'sparkhead', 'sparkworker-wired', 'x86host'}
    assert f'PASS sparkhead staged pins: {len(s.PIN_FILES)} files equal, incl. {s.CONFIG}' in out
    assert f'PASS sparkworker-wired image profiler: {len(s.PROFILER_CONFIG)} reviewed fields present' in out
    assert 'INFO sparkhead Nsight: host/container: none | none (not used)' in out
    assert 'PASS sparkhead profile directory and disk: ~/dsv41-offload-run/sp1-profile absent; 900 GiB free' in out


@pytest.mark.parametrize('bad, line', [
    ('hold', 'FAIL sparkhead production ledger'),
    ('pins', 'FAIL sparkhead staged pins'),
    ('models', 'FAIL sparkhead production serving'),
    ('port', 'FAIL sparkworker-wired container'),
    ('pins13', 'FAIL sparkhead production sources'),
    ('profiler', 'FAIL sparkhead image profiler'),
    ('memory', 'FAIL sparkhead memory and GPU telemetry'),
    ('profile-dir', 'FAIL sparkworker-wired profile directory and disk'),
    ('disk', 'FAIL sparkhead profile directory and disk'),
    ('worker', 'FAIL x86host TP2 worker'),
    ('gpus', 'FAIL x86host drafter GPUs'),
])
def test_each_failure_is_reported_and_every_check_still_runs(repo: tuple[Path, str, Path], capsys: pytest.CaptureFixture[str],
                                                             bad: str, line: str) -> None:
    rc, out, _ = run(repo, capsys, bad)
    failures = 2 if bad == 'profiler' else 1           # the image is the same on both ranks
    assert rc == 1 and line in out and f'PRECHECK FAIL: {failures} check(s)' in out, out
    assert 'x86host drafter GPUs' in out


def test_cli_refuses_short_commit() -> None:
    assert sc.main(['abc1234']) == 2


def test_profiler_probe_reads_the_reference_image_source_without_importing_vllm(tmp_path: Path) -> None:
    if not (SRC_IMAGE / 'vllm/config/profiler.py').is_file():
        pytest.skip('reference image source not present')
    env = {'PATH': os.environ['PATH'], 'PYTHONPATH': str(SRC_IMAGE), 'PYTHONDONTWRITEBYTECODE': '1'}
    probe = sc.PROFILER_PROBE + "import sys\nassert 'vllm' not in sys.modules and 'torch' not in sys.modules\n"
    done = subprocess.run([sys.executable, '-'], input=probe, text=True, capture_output=True, env=env, cwd=tmp_path,
                          timeout=60, check=True)
    found = json.loads(done.stdout.strip().splitlines()[-1])
    assert set(s.PROFILER_CONFIG) <= set(found['fields'])
    assert found['endpoints'] == ['/start_profile', '/stop_profile']
