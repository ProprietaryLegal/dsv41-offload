"""WIN2 API: back-to-back (chain) mode helpers (CPU only; marker redirected to tmp)."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from offload.launch.windows import chain
from offload.prefill_maxlen.evidence import record


@pytest.fixture(autouse=True)
def marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / 'CHAINED'
    monkeypatch.setattr(chain, 'MARKER', path)
    monkeypatch.delenv('WINDOW_CHAIN', raising=False)
    monkeypatch.delenv('WINDOW_CHAIN_LAST', raising=False)
    return path


def test_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    assert not chain.enabled() and not chain.last() and not chain.skip_restore(False)
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    assert chain.enabled() and chain.skip_restore(False) and not chain.skip_restore(True)
    monkeypatch.setenv('WINDOW_CHAIN_LAST', '1')
    assert chain.last() and not chain.skip_restore(False)


def test_finish_and_chained_preflight(monkeypatch: pytest.MonkeyPatch, marker: Path, tmp_path: Path) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    chain.finish_chained(tmp_path / 'W1', {'sparkhead': 'QUIET', 'sparkworker-wired': 'QUIET'})
    data = chain.read_marker()
    assert data is not None and data['window'] == str(tmp_path / 'W1') and marker.exists()
    assert chain.preflight(production_serving=False, hold_exists=False, placeholders_present=True) == 'chained'
    with pytest.raises(ValueError, match='serving'):
        chain.preflight(production_serving=True, hold_exists=False, placeholders_present=True)
    with pytest.raises(ValueError, match='HOLD'):
        chain.preflight(production_serving=False, hold_exists=True, placeholders_present=True)
    chain.clear()
    assert not marker.exists()
    assert chain.preflight(production_serving=True, hold_exists=False) == 'production'


def test_marker_without_chain_mode_refuses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    chain.finish_chained(tmp_path / 'W1', {'sparkhead': 'QUIET'})
    monkeypatch.delenv('WINDOW_CHAIN')
    with pytest.raises(ValueError, match='WINDOW_CHAIN=1'):
        chain.preflight(production_serving=False, hold_exists=False)


def test_production_down_without_marker_refuses() -> None:
    with pytest.raises(ValueError, match='no chain marker'):
        chain.preflight(production_serving=False, hold_exists=False)


@pytest.mark.parametrize('env', [{}, {'WINDOW_CHAIN': '1', 'WINDOW_CHAIN_LAST': '1'}])
def test_finish_refused_outside_non_last_chain(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, env: dict[str, str]) -> None:
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ValueError, match='non-last'):
        chain.finish_chained(tmp_path / 'W', {'sparkhead': 'QUIET'})


def test_finish_requires_quiet_proof_and_schema_guard(monkeypatch: pytest.MonkeyPatch, marker: Path, tmp_path: Path) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    with pytest.raises(ValueError, match='quiet proof'):
        chain.finish_chained(tmp_path / 'W', {})
    marker.write_text('{"schema": "other"}\n')
    with pytest.raises(ValueError, match='schema'):
        chain.read_marker()


def test_gate_uses_injected_probes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert chain.gate(lambda: True, lambda: False) == 'production'
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    chain.finish_chained(tmp_path / 'W1', {'sparkhead': 'QUIET'})
    assert chain.gate(lambda: False, lambda: False, lambda: True) == 'chained'
    with pytest.raises(ValueError, match='HOLD'):
        chain.gate(lambda: False, lambda: True, lambda: True)
    with pytest.raises(ValueError, match='placeholders missing'):
        chain.gate(lambda: False, lambda: False, lambda: False)


def test_production_probe_refuses_unreachable() -> None:
    assert chain.production_serving('http://127.0.0.1:9/v1/models', timeout=0.5) is False


def test_cli_status_and_clear(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    chain.finish_chained(tmp_path / 'W1', {'sparkhead': 'QUIET'})
    assert chain.main(['status']) == 0 and '"enabled": true' in capsys.readouterr().out
    assert chain.main(['clear']) == 0 and chain.read_marker() is None



# ---------------------------------------------------------------- WIN2-R2: hold placeholders kept across chained windows

def test_chained_preflight_requires_placeholders(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    chain.finish_chained(tmp_path / 'W1', {'sparkhead': 'QUIET'})
    assert 'only the final strict W8 restore' in str(record(chain.read_marker() or {})['placeholders'])
    with pytest.raises(ValueError, match='must prove the hold placeholders'):
        chain.preflight(production_serving=False, hold_exists=False)          # older two-argument call refuses
    with pytest.raises(ValueError, match='placeholders missing'):
        chain.preflight(production_serving=False, hold_exists=False, placeholders_present=False)
    chain.clear()
    probed: list[int] = []

    def probe() -> bool:
        probed.append(1)
        return True
    assert chain.gate(lambda: True, lambda: False, probe) == 'production'
    assert probed == []                                         # no marker: the placeholders are not probed


def _fake_ssh(tmp_path: Path, rc: int) -> Path:
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    ssh = bin_dir / 'ssh'
    ssh.write_text(f'#!/bin/sh\necho "$@" >> {tmp_path}/ssh.log\nexit {rc}\n')
    ssh.chmod(0o755)
    return bin_dir


@pytest.mark.parametrize('rc,present', [(0, True), (1, False), (255, False)])
def test_placeholder_probe_needs_both_sparks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rc: int, present: bool) -> None:
    monkeypatch.setenv('PATH', f'{_fake_ssh(tmp_path, rc)}:{os.environ["PATH"]}')
    assert chain.placeholders_present() is present
    calls = (tmp_path / 'ssh.log').read_text().splitlines()
    assert calls[0].split()[2] == 'sparkhead' and (not present or calls[1].split()[2] == 'sparkworker-wired')
    assert all('tmux has-session -t dsv41-hold' in c and 'DSV41-OFFLOAD hold' in c for c in calls)


@pytest.mark.parametrize('ssh_rc', [0, 1])
def test_chained_close_keeps_placeholders_and_removes_only_hold(tmp_path: Path, ssh_rc: int) -> None:
    """The shared close-own-HOLD-without-production script: never kills the placeholders; refuses if one is missing."""
    assert 'kill-session' not in chain.CHAINED_CLOSE and '8000' not in chain.CHAINED_CLOSE
    deploy = tmp_path / 'dsv41-deploy'
    deploy.mkdir()
    (deploy / 'dsv41.env').write_text('HEAD=spark\nWORKER=sparkworker\n')
    (deploy / 'HOLD').write_text('2026-10-27T20:00:00Z DSV41-OFFLOAD (session coordinator): test\n')
    env = {**os.environ, 'HOME': str(tmp_path), 'PATH': f'{_fake_ssh(tmp_path, ssh_rc)}:{os.environ["PATH"]}'}
    r = subprocess.run(['bash', '-c', chain.CHAINED_CLOSE], capture_output=True, text=True, env=env, timeout=30)
    calls = (tmp_path / 'ssh.log').read_text().splitlines()
    if ssh_rc == 0:
        assert r.returncode == 0 and chain.CHAINED_CLOSE_OK in r.stdout.splitlines() and not (deploy / 'HOLD').exists()
        assert [c.split()[2] for c in calls] == ['spark', 'sparkworker']
    else:
        assert r.returncode != 0 and 'hold placeholder missing on spark' in r.stdout and (deploy / 'HOLD').exists()


# ---------------------------------------------------------------- WIN2-R2: production snapshot carried by the marker

def _rollback(window: Path, commit: str = '5' * 40) -> Path:
    window.mkdir(parents=True, exist_ok=True)
    path = window / 'rollback.json'
    path.write_text(json.dumps({'production_commit': commit, 'ranks': [{'id': 'a'}, {'id': 'b'}]}) + '\n')
    return path


def test_marker_carries_and_inherits_the_production_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    first = _rollback(tmp_path / 'PC1-W-1')                     # started from serving production: its own capture
    chain.finish_chained(tmp_path / 'PC1-W-1', {'s': 'QUIET'})
    path, data = chain.production_snapshot()
    assert path == first and data['production_commit'] == '5' * 40
    (tmp_path / 'NBW2-W-2').mkdir()                              # chained window: nothing captured, inherits
    chain.finish_chained(tmp_path / 'NBW2-W-2', {'s': 'QUIET'})
    marker = record(chain.read_marker() or {})
    assert marker['window'] == str(tmp_path / 'NBW2-W-2') and record(marker['production_snapshot'])['path'] == str(first)
    assert chain.production_snapshot()[0] == first
    first.write_text(first.read_text().replace('5', '6'))        # the capture changed after it was recorded
    with pytest.raises(ValueError, match='changed'):
        chain.production_snapshot()
    with pytest.raises(ValueError, match='changed'):
        chain.finish_chained(tmp_path / 'UP3-W-3', {'s': 'QUIET'})


def test_snapshot_explicit_missing_and_invalid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    chain.finish_chained(tmp_path / 'W-none', {'s': 'QUIET'})   # no capture anywhere: recorded null, refused on use
    with pytest.raises(ValueError, match='no production snapshot'):
        chain.production_snapshot()
    chain.clear()
    explicit = _rollback(tmp_path / 'elsewhere')
    chain.finish_chained(tmp_path / 'W-x', {'s': 'QUIET'}, production_snapshot=explicit)
    assert chain.production_snapshot()[0] == explicit
    with pytest.raises(ValueError, match='40-hex'):
        chain.snapshot_ref(_rollback(tmp_path / 'short', '596ba1c'))
    with pytest.raises(ValueError, match='absolute'):
        chain.snapshot_ref(Path('rollback.json'))
