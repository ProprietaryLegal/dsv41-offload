"""WIN2 API: back-to-back (chain) mode helpers (CPU only; marker redirected to tmp)."""
from __future__ import annotations

from pathlib import Path

import pytest

from offload.launch.windows import chain


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
    assert chain.preflight(production_serving=False, hold_exists=False) == 'chained'
    with pytest.raises(ValueError, match='serving'):
        chain.preflight(production_serving=True, hold_exists=False)
    with pytest.raises(ValueError, match='HOLD'):
        chain.preflight(production_serving=False, hold_exists=True)
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
    assert chain.gate(lambda: False, lambda: False) == 'chained'
    with pytest.raises(ValueError, match='HOLD'):
        chain.gate(lambda: False, lambda: True)


def test_production_probe_refuses_unreachable() -> None:
    assert chain.production_serving('http://127.0.0.1:9/v1/models', timeout=0.5) is False


def test_cli_status_and_clear(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    chain.finish_chained(tmp_path / 'W1', {'sparkhead': 'QUIET'})
    assert chain.main(['status']) == 0 and '"enabled": true' in capsys.readouterr().out
    assert chain.main(['clear']) == 0 and chain.read_marker() is None
