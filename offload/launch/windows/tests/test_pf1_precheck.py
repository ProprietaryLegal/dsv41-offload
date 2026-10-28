"""PF1-WINDOW precheck on CPU with a fake remote: no SSH, HTTP, Spark, x86host or GPU."""
from __future__ import annotations

import datetime
import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from offload.launch.windows import pf1_precheck as pc
from offload.launch.windows import pf1_window as w
from offload.prefill_trace.telemetry import FIELDS

GIB = 2**20    # kB


class FakeRemote:
    """Answers the precheck's read-only commands; `bad` names the one answer to break."""

    def __init__(self, root: Path, expected: str, bad: str = '') -> None:
        self.root, self.expected, self.bad, self.calls = root, expected, bad, list[tuple[str, str]]()

    def __call__(self, host: str, command: str, stdin: str) -> tuple[int, str]:
        self.calls.append((host, command))
        assert host in ('sparkhead', 'sparkworker-wired', 'x86host')
        for word in ('bash offload/launch/start_exp', 'stop.sh', 'window.sh open', 'restore.sh', ':8001/', '18001', 'rm ',
                     'git checkout', 'git fetch', 'docker stop', 'docker rm', 'tee ', '> '):
            assert word not in command, f'precheck must stay read-only: {word}'
        bad = self.bad
        if host == 'x86host' and 'query-compute-apps' in command:
            g0, g1 = w.WORKER_GPUS.split(',')
            apps = [[g0, '101'], [g1, '102'], ['GPU-other', '900']] + ([[g1, '555']] if bad == 'gpus' else [])
            return 0, json.dumps({'worker_pids': [100, 101, 102], 'apps': apps})
        if 'cd ~/dsv41-deploy &&' in command:
            return (1, 'HOLD exists') if bad == 'hold' else (0, '21bed31abcde clean, no HOLD\n')
        if 'cd ~/dsv41-deploy-offload && echo' in command:
            head = '0' * 40 if bad == 'unstaged' else self.expected
            return 0, f'{head}\n{"3" if bad == "dirty" else "0"}\n'
        if 'sha256sum offload/' in command:
            return 0, '\n'.join(f'{"f" * 64 if bad == "pins" else w.digest(self.root / p)}  {p}' for p in w.PIN_FILES) + '\n'
        if 'prefill_trace && if [ -e generated ]' in command:
            assert stdin == (self.root / 'offload/prefill_trace/OVERLAY.sha256').read_text()
            return (1, 'generated/x: FAILED') if bad == 'generated' else (0, 'ABSENT\n')
        if '/health' in command:
            return 0, ''
        if '/v1/models' in command:
            cap = 524288 if bad == 'models' else 327680
            return 0, json.dumps({'data': [{'max_model_len': cap, 'root': '/srv/dsv41/legal-v2a'}]})
        if '/metrics' in command:
            return 0, 'vllm:num_requests_running{m="x"} 1.0\nvllm:num_requests_waiting{m="x"} 0.0\n'
        if 'docker inspect' in command:
            port = '1' if bad == 'port' and host == 'sparkworker-wired' else '0'
            return 0, f'{w.IMAGE} running\n{port}\nnohold\n'
        if command == 'docker exec -i dsv41-serve python3 -':
            return 0, '' if bad == 'pins13' and host == 'sparkhead' else 'PF1 verified 13 engine sources\n'
        if 'docker exec -i dsv41-serve python3 - ' in command:
            return 0, json.dumps(w.image_pins()) + '\n'
        if 'nvidia-smi' in stdin + command:
            avail = 4 * GIB if bad == 'memory' and host == 'sparkhead' else 60 * GIB
            return 0, (json.dumps({'kind': 'memory', 'kb': {'MemAvailable': avail, 'SwapFree': 8 * GIB, 'SwapTotal': 16 * GIB}})
                       + '\n' + json.dumps({'kind': 'gpu', 'csv': ','.join(['1'] * len(FIELDS.split(',')))}) + '\n')
        if 'pf1-trace' in command:
            return 0, '2\n/dev/nvme0n1p2 1000000000 100000000 900000000 10% /\n'
        if host == 'x86host':
            revision = 'e' * 40 if bad == 'worker' else w.WORKER
            stamp = datetime.datetime.fromtimestamp(time.time(), datetime.UTC).isoformat().replace('+00:00', 'Z')
            return 0, json.dumps({'revision': revision, 'current': '/w/fb68c52', 'cwd': '/w/fb68c52/offload/dspark',
                'cmd': ['python', '-m', 'worker.server', '--tp', '2', '--max-model-len', '1048576'], 'gpus': w.WORKER_GPUS,
                'status': {'tp': 2, 'tp_comm_errors': 0, 'steps': 7, 'time': stamp}})
        raise AssertionError(f'unexpected precheck command on {host}: {command[:120]}')


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, str, Path]:
    root = tmp_path / 'repo'
    for p in w.PIN_FILES:
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


def run(repo: tuple[Path, str, Path], capsys: pytest.CaptureFixture[str], bad: str = '',
        listening: str = '') -> tuple[int, str, FakeRemote]:
    root, expected, data = repo
    fake = FakeRemote(root, expected, bad)
    rc = pc.precheck(expected, None, fake, root=root, data_dir=data, listening=lambda: listening)
    return rc, capsys.readouterr().out, fake


def test_all_checks_pass_and_touch_only_read_only_commands(repo: tuple[Path, str, Path],
                                                           capsys: pytest.CaptureFixture[str]) -> None:
    rc, out, fake = run(repo, capsys)
    assert rc == 0 and 'PRECHECK PASS' in out and 'FAIL' not in out, out
    assert {h for h, _ in fake.calls} == {'sparkhead', 'sparkworker-wired', 'x86host'}
    assert 'PASS sparkhead staged pins: 7 files equal' in out
    assert 'running+waiting 1' in out and 'MemAvailable 60.0 GiB' in out


@pytest.mark.parametrize('bad, line', [
    ('hold', 'FAIL sparkhead production ledger'),
    ('dirty', 'FAIL sparkhead experiment checkout'),
    ('pins', 'FAIL sparkhead staged pins'),
    ('generated', 'FAIL sparkhead trace sources'),
    ('models', 'FAIL sparkhead production serving'),
    ('port', 'FAIL sparkworker-wired container'),
    ('pins13', 'FAIL sparkhead production sources'),
    ('memory', 'FAIL sparkhead memory and GPU telemetry'),
    ('worker', 'FAIL x86host TP2 worker'),
    ('gpus', 'FAIL x86host drafter GPUs'),
])
def test_each_failure_is_reported_and_every_check_still_runs(repo: tuple[Path, str, Path], capsys: pytest.CaptureFixture[str],
                                                             bad: str, line: str) -> None:
    rc, out, _ = run(repo, capsys, bad)
    assert rc == 1 and line in out and 'PRECHECK FAIL: 1 check(s)' in out, out
    assert 'x86host drafter GPUs' in out    # the last check ran: one failure never hides the rest


def test_unstaged_checkout_is_info_not_failure(repo: tuple[Path, str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    rc, out, fake = run(repo, capsys, 'unstaged')
    assert rc == 0 and 'run with --stage-bundle' in out and 'INFO sparkhead staged pins' in out
    assert not any('sha256sum offload/' in c for _, c in fake.calls)


def test_local_tunnel_port_and_commit(repo: tuple[Path, str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    rc, out, _ = run(repo, capsys, listening='LISTEN 0 128 127.0.0.1:18001 0.0.0.0:*\n')
    assert rc == 1 and 'FAIL controller tunnel port' in out
    root, _, data = repo
    rc = pc.precheck('1' * 40, None, FakeRemote(root, '1' * 40), root=root, data_dir=data, listening=lambda: '')
    assert rc == 1 and 'FAIL controller reviewed checkout' in capsys.readouterr().out


def test_cli_refuses_short_commit(capsys: pytest.CaptureFixture[str]) -> None:
    assert pc.main(['abc1234']) == 2


def test_embedded_remote_script_compiles() -> None:
    compile(pc.GPU_APPS, 'GPU_APPS', 'exec')    # compiled only: it reads x86host's /proc and nvidia-smi
