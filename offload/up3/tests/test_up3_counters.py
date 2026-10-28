"""S59 counterexamples against actual Live.warmup/cell/finish, with CPU snapshots."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from offload.launch.windows import pf1_window as pf, up3_window as w
from offload.prefill_maxlen.evidence import Json, record
from offload.up3.tests.test_up3_window import measured, Remote, rehearsal as rehearsal


def stats(rounds: int = 100, fires: int = 0, **counters: int) -> dict[str, Json]:
    return {'counters': {'rounds_ok': rounds, **counters}, 'policy': 'capacity', 'async': True,
        'async_proxy': {'mode': 'process'}, 'async_handoff': {'mlock': 'ok'},
        'async_watchdog': {'fires': fires, 'running': True}, 'proxy_torch_loaded': False}


def counter_live(root: Path, monkeypatch: pytest.MonkeyPatch, baseline: dict[str, Json],
                 snapshots: list[dict[str, Json]]) -> tuple[w.Live, w.Arm, list[str]]:
    # All methods under test are real. Only process/transport/stats dependencies are inert.
    live = object.__new__(w.Live)
    arm = w.arms('k5')[1]
    live.out, live.tag, live.submit, live.started_at = root, 'fixture', 1e12, 0.
    live.group = 'k5'   # Live.__init__ sets it; allowance/archiving read it
    live.clock = lambda: 1.
    live.remote = 'fixture'
    live.ids = ['id0', 'id1']
    live.round_baseline, live.counter_seen = {}, {}
    live.clock_caps = {}   # Live.__init__ sets it; start() fills it from start_exp.sh's clock-cap lines
    live.sup = cast(pf.Supervisor, SimpleNamespace(arm_minima={h: 7000000 for h in pf.HOSTS}))
    rows = iter([baseline, *snapshots])
    events: list[str] = []
    def snapshot(label: str) -> dict[str, Json]:
        events.append(label)
        return copy.deepcopy(next(rows))
    def call(label: str, argv: list[str], *, timeout: float = 120, inflight: int = 0) -> str:
        events.append(label)
        (root/(arm.name+'-short-b1.json')).write_text(json.dumps(measured('short-b1')))
        return ''
    monkeypatch.setattr(live, 'stats', snapshot)
    monkeypatch.setattr(live, 'owned', lambda *args, **kwargs: '')
    monkeypatch.setattr(live, 'driver', lambda *args: [])
    monkeypatch.setattr(live, 'call', call, raising=False)
    monkeypatch.setattr(live, 'ssh', lambda label, host, command: live.ids[pf.HOSTS.index(host)])
    monkeypatch.setattr(live, 'rank_logs', lambda *args: {})
    live.warmup(arm)  # actual one-time full baseline capture
    return live, arm, events


def test_prior_nonzero_totals_pass_cell_and_finish_and_are_retained(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = stats(fires=1, rounds_failed=1, rounds_some_failed=2, fallback_internal_error=1)
    rows = [copy.deepcopy(baseline) for _ in range(3)]
    record(rows[1]['counters'])['rounds_ok'] = 110
    record(rows[2]['counters'])['rounds_ok'] = 111
    live, arm, _ = counter_live(tmp_path, monkeypatch, baseline, rows)
    live.cell(arm, 'short-b1')
    final = live.finish(arm)
    assert final['startup_baseline'] == baseline
    assert final['proxy'] == rows[-1]
    assert 'fallback_internal_error' in cast(list[Json], final['observed_counter_keys'])


@pytest.mark.parametrize('stage', ['before', 'after', 'final'])
@pytest.mark.parametrize('fault', ['watchdog-growth', 'watchdog-reset', 'round-failed', 'some-failed',
    'stop-growth', 'stop-reset', 'stop-missing', 'ordinary-reset', 'ordinary-missing', 'unhealthy'])
def test_faults_refuse_at_every_snapshot(stage: str, fault: str, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = stats(fires=1, rounds_failed=1, rounds_some_failed=2, fallback_internal_error=1, payloads=4)
    rows = [copy.deepcopy(baseline) for _ in range(3)]
    record(rows[1]['counters'])['rounds_ok'] = 110
    record(rows[2]['counters'])['rounds_ok'] = 111
    index = ('before', 'after', 'final').index(stage)
    current = rows[index]
    counters = record(current['counters'])
    if fault == 'watchdog-growth': record(current['async_watchdog'])['fires'] = 2
    elif fault == 'watchdog-reset': record(current['async_watchdog'])['fires'] = 0
    elif fault == 'round-failed': counters['rounds_failed'] = 2
    elif fault == 'some-failed': counters['rounds_some_failed'] = 3
    elif fault == 'stop-growth': counters['fallback_internal_error'] = 2
    elif fault == 'stop-reset': counters['fallback_internal_error'] = 0
    elif fault == 'stop-missing': counters.pop('fallback_internal_error')
    elif fault == 'ordinary-reset': counters['payloads'] = 0
    elif fault == 'ordinary-missing': counters.pop('payloads')
    else: current['async_dead'] = True
    live, arm, events = counter_live(tmp_path, monkeypatch, baseline, rows)
    with pytest.raises(ValueError):
        live.cell(arm, 'short-b1')
        live.finish(arm)
    if stage == 'before':
        assert arm.name+'-short-b1' not in events


def test_sparse_zeros_allowed_initially_but_seen_keys_cannot_disappear_or_reset(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = stats()
    live, arm, _ = counter_live(tmp_path, monkeypatch, baseline, [])
    seen = stats(payloads=5, fallback_internal_error=0, rounds_failed=0)
    live.baseline_check(arm, seen)
    for key in ('payloads', 'fallback_internal_error', 'rounds_failed'):
        missing = copy.deepcopy(seen)
        record(missing['counters']).pop(key)
        with pytest.raises(ValueError, match='missing/reset'):
            live.baseline_check(arm, missing)
    reset = copy.deepcopy(seen)
    record(reset['counters'])['payloads'] = 1
    with pytest.raises(ValueError, match='missing/reset'):
        live.baseline_check(arm, reset)
    live.baseline_check(arm, copy.deepcopy(seen))
    with pytest.raises(ValueError, match='cannot be replaced'):
        live.warmup(arm)


def test_watchdog_zero_to_one_is_rejected_in_actual_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    live, arm, _ = counter_live(tmp_path, monkeypatch, stats(), [stats(), stats(rounds=110, fires=1)])
    with pytest.raises(ValueError, match='watchdog grew'):
        live.cell(arm, 'short-b1')


def test_baseline_once_per_start_after_greedy_and_warmup(rehearsal: tuple[w.Live, Remote]) -> None:
    live, fake = rehearsal
    result = w.window(live, 'k5', lambda row: None)
    assert result['status'] == 'COMPLETE', result
    assert len(live.round_baseline) == 3
    for arm in w.arms('k5'):
        assert fake.commands.count(arm.name+'-baseline') == 1
        assert fake.commands.index(arm.name+'-greedy') < fake.commands.index(arm.name+'-warmup')
        assert fake.commands.index(arm.name+'-warmup') < fake.commands.index(arm.name+'-baseline')
        assert fake.commands.index(arm.name+'-baseline') < fake.commands.index(arm.name+'-short-b1-before')
        checks = record(record(record(result['starts'])[arm.name])['checks'])
        assert checks['startup_baseline'] == live.round_baseline[arm.name]


def test_missing_baseline_cannot_be_synthesized_from_first_cell(rehearsal: tuple[w.Live, Remote]) -> None:
    live, _ = rehearsal
    with pytest.raises(ValueError, match='measurement baseline'):
        live.cell(w.arms('k5')[1], 'short-b1')


@pytest.mark.parametrize('fault', ['watchdog', 'missing-stop', 'stop-growth'])
def test_counter_fault_stops_candidate_before_timing_and_restores(fault: str,
        rehearsal: tuple[w.Live, Remote], monkeypatch: pytest.MonkeyPatch) -> None:
    live, fake = rehearsal
    original = live.stats
    def snapshot(label: str) -> dict[str, Json]:
        current = original(label)
        if label.startswith('s2-k5'):
            counters = record(current['counters'])
            counters.update(rounds_failed=1, rounds_some_failed=2, fallback_internal_error=1)
            if label.endswith('-short-b1-before'):
                if fault == 'watchdog': record(current['async_watchdog'])['fires'] = 1
                elif fault == 'missing-stop': counters.pop('fallback_internal_error')
                else: counters['fallback_internal_error'] = 2
        return current
    monkeypatch.setattr(live, 'stats', snapshot)
    result = w.window(live, 'k5', lambda row: None)
    assert 'stop' in result, result
    assert 's1-control-short-b1' in fake.commands
    assert 's2-k5-short-b1' not in fake.commands
    assert str(record(result['restore'])['status']).startswith('RESTORED')
    assert record(result['decision'])['status'] == 'NOT A CANDIDATE'


def test_warmup_waves_are_owned_demand(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # UP3-k5-W-20261027T041627Z: the demand guard saw the OPS1 warm-up's concurrency-3 wave as foreign (owned 0).
    baseline = stats()
    live, arm, _ = counter_live(tmp_path, monkeypatch, baseline, [copy.deepcopy(baseline)])
    seen: list[dict[str, object]] = []
    def owned(*args: object, **kwargs: object) -> str:
        seen.append(kwargs)
        return ''
    monkeypatch.setattr(live, 'owned', owned)
    live.round_baseline.clear()
    live.warmup(arm)
    assert seen and seen[0]['inflight'] == 4
