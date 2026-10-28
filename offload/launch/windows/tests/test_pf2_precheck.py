"""PF2-LOWKV precheck on CPU with a fake remote: no SSH, HTTP, Spark, x86host or GPU."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from offload.launch.windows import pf1_window as w
from offload.launch.windows import pf2_precheck as pp
from offload.launch.windows import pf2_window as p
from offload.launch.windows.tests import test_pf1_precheck as base
from offload.launch.windows.tests import test_sp1_precheck as sp1t
from offload.launch.windows.tests.test_pf2_rehearsal import extended
from offload.launch.windows.tests.test_pf2_window import PLATFORM, parity
from offload.pf2_lowkv import arith


class FakeRemote(sp1t.FakeRemote):
    """SP1's read-only fake plus PF2's pins, host identity and extended telemetry probe."""

    def __call__(self, host: str, command: str, stdin: str) -> tuple[int, str]:
        for word in ('bash offload/launch/start_exp', 'stop.sh', 'window.sh', 'restore.sh', ':8001/', '18001', 'rm ',
                     'rmdir', 'mkdir', 'git checkout', 'git fetch', 'docker stop', 'docker rm', 'tee ', '> ', 'echo never',
                     'drop_caches', 'nvidia-smi -lgc', 'nvidia-smi -pl'):
            assert word not in command, f'precheck must stay read-only: {word}'
        if command == w.HOST_PARITY:
            self.calls.append((host, command))
            kernel = '6.17.0-1026-nvidia' if self.bad == 'kernel' and host == 'sparkhead' else PLATFORM.kernel[host]
            return 0, parity(kernel, PLATFORM.driver[host])
        if command == p.telemetry_command(1):
            self.calls.append((host, command))
            text = extended()
            if self.bad == 'psi' and host == 'sparkworker-wired':
                text = text.replace('"psi": {"some"', '"psi": null, "x": {"some"')
            return 0, text
        if 'sha256sum offload/' in command:
            self.calls.append((host, command))
            return 0, '\n'.join(f'{"f" * 64 if self.bad == "pins" else w.digest(self.root / f)}  {f}' for f in p.PIN_FILES) + '\n'
        return super().__call__(host, command, stdin)


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, str, Path]:
    root = tmp_path / 'repo'
    for f in p.PIN_FILES:
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(w.ROOT / f, root / f)
    git = ['git', '-C', str(root), '-c', 'user.name=t', '-c', 'user.email=t@t']
    subprocess.run([*git, 'init', '-q'], check=True)
    subprocess.run([*git, 'add', '.'], check=True)
    subprocess.run([*git, 'commit', '-qm', 'x'], check=True)
    data = tmp_path / 'data'
    data.mkdir()
    for name in ('opinions_cl.jsonl', 'filings_recap.jsonl'):
        (data / name).write_text(json.dumps({'text': 'public record'}) + '\n')
    return root, subprocess.check_output([*git, 'rev-parse', 'HEAD'], text=True).strip(), data


def run(repo: tuple[Path, str, Path], capsys: pytest.CaptureFixture[str], tmp_path: Path, bad: str = '',
        low: int = arith.LOW_KV, platform: p.Platform = PLATFORM, inventory: str | None = None,
        reference: Path | None = None) -> tuple[int, str, FakeRemote]:
    root, expected, data = repo
    campaign = tmp_path / 'campaign'
    campaign.mkdir(exist_ok=True)
    if inventory is not None:
        (campaign / 'FW1').mkdir(exist_ok=True)
        (campaign / 'FW1' / 'inventory-post.txt').write_text(inventory)
    fake = FakeRemote(root, expected, bad)
    rc = pp.precheck(expected, None, fake, low=low, platform=platform, reference=reference, campaign=campaign, root=root,
                     data_dir=data, listening=lambda: '')
    return rc, capsys.readouterr().out, fake


INVENTORY = ('7.0.0-1004-nvidia\nDGX_OTA_VERSION="7.6.0"\n580.178.02\nMemTotal:       128000000 kB\n'
             'MemAvailable:   120000000 kB\nSwapTotal:      16777212 kB\n')


def test_all_checks_pass_read_only(repo: tuple[Path, str, Path], capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    rc, out, fake = run(repo, capsys, tmp_path)
    assert rc == 0 and 'PRECHECK PASS: run pf2_window.sh' in out and 'FAIL' not in out, out
    assert {h for h, _ in fake.calls} == {'sparkhead', 'sparkworker-wired', 'x86host'}
    assert f'PASS sparkhead staged pins: {len(p.PIN_FILES)} files equal, incl. {p.CONFIG}' in out
    assert 'PASS controller LOW reservation: 3221225472 B = 23,301 blocks, pool 1,410,348 tokens' in out
    assert 'PASS sparkhead kernel and driver: 7.0.0-1004-nvidia / 580.178.02' in out
    assert 'PASS sparkworker-wired kernel and driver: 6.17.0-1026-nvidia / 580.159.03' in out
    assert 'INFO controller FW1 post-update inventory' in out and 'absent' in out
    assert 'PASS sparkworker-wired memory and telemetry' in out and 'PSI some/full' in out


def test_fw1_inventory_must_match_the_parameters(repo: tuple[Path, str, Path], capsys: pytest.CaptureFixture[str],
                                                 tmp_path: Path) -> None:
    rc, out, _ = run(repo, capsys, tmp_path, inventory=INVENTORY)
    assert rc == 0 and 'PASS controller FW1 post-update inventory: kernel/driver match; MemTotal 122.07 GiB' in out
    rc, out, _ = run(repo, capsys, tmp_path, inventory=INVENTORY.replace('580.178.02', '580.159.03'))
    assert rc == 1 and 'FAIL controller FW1 post-update inventory' in out


@pytest.mark.parametrize('bad, line', [
    ('kernel', 'FAIL sparkhead kernel and driver'),
    ('psi', 'FAIL sparkworker-wired memory and telemetry'),
    ('pins', 'FAIL sparkhead staged pins'),
    ('hold', 'FAIL sparkhead production ledger'),
    ('profile-dir', 'FAIL sparkworker-wired profile directory and disk'),
    ('worker', 'FAIL x86host TP2 worker'),
])
def test_each_failure_is_reported_and_every_check_still_runs(repo: tuple[Path, str, Path], capsys: pytest.CaptureFixture[str],
                                                             tmp_path: Path, bad: str, line: str) -> None:
    rc, out, _ = run(repo, capsys, tmp_path, bad)
    assert rc == 1 and line in out and 'PRECHECK FAIL: 1 check(s)' in out, out
    assert 'x86host drafter GPUs' in out


def test_parameters_are_checked(repo: tuple[Path, str, Path], capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    rc, out, _ = run(repo, capsys, tmp_path, low=2 * arith.GIB)
    assert rc == 1 and 'FAIL controller LOW reservation' in out
    rc, out, _ = run(repo, capsys, tmp_path, platform=p.Platform({}, {}))
    assert rc == 1 and 'FAIL controller platform parameters' in out and 'FAIL sparkhead kernel and driver' in out


def test_reference_is_read(repo: tuple[Path, str, Path], capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    ref = tmp_path / 'SP1-W-post'
    (ref / 'cells').mkdir(parents=True)
    (ref / 'cells' / 'ctl-150k-b1.json').write_text(json.dumps({'cell': 'ctl-150k-b1', 'streams': [{'ttft_s': 68.5}]}))
    rc, out, _ = run(repo, capsys, tmp_path, reference=ref)
    assert rc == 0 and 'PASS controller 327K reference (analysis): 68.50 s from SP1-W-post/cells/ctl-150k-b1.json' in out


def test_cli_refuses_short_commit() -> None:
    assert pp.main(['abc1234']) == 2
    assert base.GIB > 0
