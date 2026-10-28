"""WIN2 back-to-back windows over the real Live and fake transport (CPU only; chain marker under tmp)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from offload.launch.windows import chain
from offload.pc1 import window as w
from offload.pc1.tests.test_window import make_rehearsal
from offload.prefill_maxlen.evidence import record


def test_chained_window_leaves_production_down_then_the_last_restores(tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    live, fake, sup = make_rehearsal(tmp_path, monkeypatch, name='PC1-threshold2048-W-a')
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    live.group = 'threshold2048'
    try:
        first = w.window(live, 'threshold2048', lambda row: None)
    finally:
        sup.close()
    assert first['status'] == 'COMPLETE' and first['chain'] == 'CHAINED', first.get('stop')
    restore = record(first['restore'])
    assert restore['status'] == 'CHAINED: production left down'
    assert restore['quiet_proof'] == {'sparkhead': 'PC1 EXPERIMENT ABSENT', 'sparkworker-wired': 'PC1 EXPERIMENT ABSENT'}
    assert fake.restores == [] and not fake.hold  # no W8 step; the owned HOLD is closed
    assert fake.commands.index('chain-close') < fake.commands.index('commit-close')
    assert record(first['decision'])['status'] == 'CANDIDATE'
    marker = chain.read_marker()
    assert marker is not None and marker['window'] == str(live.out)
    w.post_close(first, live.out, live.tag)
    assert 'post_close_watch' not in first  # production is down on purpose: no watch

    # Next window of the chain, the last one: production is down, so the snapshot is carried forward.
    monkeypatch.setenv('WINDOW_CHAIN_LAST', '1')
    second_live, second, sup = make_rehearsal(tmp_path, monkeypatch, name='PC1-chunk16384-W-b')
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    monkeypatch.setenv('WINDOW_CHAIN_LAST', '1')
    second.production_down = True
    second_live.group = 'chunk16384'
    try:
        result = w.window(second_live, 'chunk16384', lambda row: None)
    finally:
        sup.close()
    assert second_live.chain_mode == 'chained'
    assert result['status'] == 'COMPLETE' and 'chain' not in result, result.get('stop')
    assert 'prod-health' not in second.commands and 'prod-models' not in second.commands
    # S92 FIX 1: admission tolerates the intentionally stopped :8000 only in a chained start, and both Sparks are
    # proved quiet before the first start_exp.sh.
    assert 'allow-unreachable' in second.hold_open_text and 'require-serving' in fake.hold_open_text
    quiet = [second.commands.index('chain-quiet-' + host) for host in ('sparkhead', 'sparkworker-wired')]
    assert max(quiet) < second.commands.index('s1-control-start')
    assert 'chain-quiet-sparkhead' not in fake.commands
    rollback = json.loads((second_live.out / 'rollback.json').read_text())
    assert rollback['chained_from'] == str(live.out / 'rollback.json')
    assert rollback['ranks'] == json.loads((live.out / 'rollback.json').read_text())['ranks']
    assert str(record(result['restore'])['status']).startswith('RESTORED')
    assert second.restores == ['start', 'check', 'close']
    # The marker goes only after this window's own restore AND its 15-minute watch PASS.
    assert chain.read_marker() is not None
    w.finish_chain({**result, 'post_close_watch': {'status': 'FAILED'}})
    assert chain.read_marker() is not None
    result['post_close_watch'] = {'status': 'PASS'}
    w.finish_chain(result)
    assert chain.read_marker() is None and str(result['chain']).startswith('ENDED')


@pytest.mark.parametrize('fail', ['gate', 'chain-close'])
def test_any_stop_or_failed_chain_close_runs_strict_w8(fail: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    live, fake, sup = make_rehearsal(tmp_path, monkeypatch, fail, name='PC1-threshold2048-W-c')
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    try:
        result = w.window(live, live.group, lambda row: None)
    finally:
        sup.close()
    assert 'chain' not in result
    assert record(result['decision'])['status'] == 'NOT A CANDIDATE'
    assert fake.restores[:2] == ['start', 'check']
    assert chain.read_marker() is None
    if fail == 'chain-close':
        assert 'deferred unlink' in str(result['chain_close_error'])
        assert 'commit-close' not in fake.commands[:fake.commands.index('start')]
    else:
        assert 'stop' in result and 'chain-close' not in fake.commands


def test_chained_start_with_a_failure_restores_production(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Production was left down by an earlier chained window; this window stops at its gate: W8 starts production.
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    previous = tmp_path / 'PC1-previous'
    seed_live, seed, sup = make_rehearsal(tmp_path, monkeypatch, name='PC1-seed')
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    try:
        seed_live.preflight()
        seed_live.snapshot()
    finally:
        sup.close()
    previous.mkdir()
    (previous / 'rollback.json').write_text((seed_live.out / 'rollback.json').read_text())
    chain.finish_chained(previous, {'sparkhead': 'PC1 EXPERIMENT ABSENT'})
    live, fake, sup = make_rehearsal(tmp_path, monkeypatch, 'gate', name='PC1-threshold2048-W-d')
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    fake.production_down = True
    try:
        result = w.window(live, live.group, lambda row: None)
    finally:
        sup.close()
    assert live.chain_mode == 'chained' and 'stop' in result
    assert str(record(result['restore'])['status']).startswith('RESTORED')
    assert fake.restores == ['start', 'check', 'close'] and not fake.aborts  # its own HOLD: its own W8, no abort
    result['post_close_watch'] = {'status': 'PASS'}
    w.finish_chain(result)
    assert chain.read_marker() is None


@pytest.mark.parametrize('env,down,placeholders,match', [
    ({}, True, True, 'WINDOW_CHAIN=1'),
    ({'WINDOW_CHAIN': '1'}, False, True, 'production is serving'),
    ({'WINDOW_CHAIN': '1'}, True, False, 'placeholders missing'),
    ({}, True, True, 'no chain marker')])
def test_chain_preflight_refusal_before_any_hold(env: dict[str, str], down: bool, placeholders: bool, match: str,
                                                 tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    live, fake, sup = make_rehearsal(tmp_path, monkeypatch, name='PC1-threshold2048-W-e')
    (tmp_path / 'PC1-earlier').mkdir()
    if match != 'no chain marker':
        monkeypatch.setenv('WINDOW_CHAIN', '1')
        chain.finish_chained(tmp_path / 'PC1-earlier', {'sparkhead': 'PC1 EXPERIMENT ABSENT'})
        monkeypatch.delenv('WINDOW_CHAIN')
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    fake.production_down, fake.placeholders_down = down, not placeholders
    try:
        result = w.window(live, live.group, lambda row: None)
    finally:
        sup.close()
    assert match in str(result['stop'])
    assert 'hold-open' not in fake.commands and not fake.restores
    if match == 'no chain marker':
        assert record(result['restore'])['status'] == 'NOT NEEDED: preflight only' and not fake.aborts
    else:
        # S92 FIX 1: no own HOLD, but the chain left production down: the shared chain abort, never "not needed".
        assert fake.aborts == [live.out / 'chain-abort'] and result['chain'] == 'ABORT'
        assert str(record(result['restore'])['status']).startswith('RESTORED')
        assert record(result['decision'])['status'] == 'NOT A CANDIDATE'
