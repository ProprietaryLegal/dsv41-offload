"""S59 counterexamples against actual Live.warmup/cell/finish, with CPU snapshots."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from offload.launch.windows import pf1_window as pf
from offload.pc1 import window as w
from offload.prefill_maxlen.evidence import Json, number, record
from offload.pc1.tests.test_window import measured, Remote, rehearsal as rehearsal


def stats(rounds: int = 100, fires: int = 0, **counters: int) -> dict[str, Json]:
    return {'counters': {'rounds_ok': rounds, **counters}, 'policy': 'capacity', 'async': True,
        'async_proxy': {'mode': 'process'}, 'async_handoff': {'mlock': 'ok'},
        'async_watchdog': {'fires': fires, 'running': True}, 'proxy_torch_loaded': False}


def counter_live(root: Path, monkeypatch: pytest.MonkeyPatch, baseline: dict[str, Json],
                 snapshots: list[dict[str, Json]]) -> tuple[w.Live, w.Arm, list[str]]:
    # All methods under test are real. Only process/transport/stats dependencies are inert.
    live = object.__new__(w.Live)
    arm = w.arms('chunk2048')[1]
    live.out, live.tag, live.submit, live.started_at, live.group = root, 'fixture', 1e12, 0., 'chunk2048'
    live.clock = lambda: 1.
    live.remote = 'fixture'
    live.ids = ['id0', 'id1']
    live.round_baseline, live.counter_seen, live.fault_level, live.drafter_faults = {}, {}, {}, {}
    live.sup = cast(pf.Supervisor, SimpleNamespace(arm_minima={h: 7000000 for h in pf.HOSTS}))
    rows = iter([baseline, *snapshots])
    events: list[str] = []
    def snapshot(label: str) -> dict[str, Json]:
        events.append(label)
        return copy.deepcopy(next(rows))
    def call(label: str, argv: list[str], *, timeout: float = 120, inflight: int = 0) -> str:
        events.append(label)
        (root/(label+'.json')).write_text(json.dumps(measured('short-b1', repeats=1, salt=label)))
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


# Growth of a drafter fault counter INSIDE a measured repeat ('after') is attributed and re-run (tests below);
# everywhere else, and every other fault at every snapshot, still refuses.
ATTRIBUTED = ('watchdog-growth', 'round-failed', 'some-failed')


@pytest.mark.parametrize('stage,fault', [(stage, fault) for stage in ('before', 'after', 'final') for fault in (
    'watchdog-growth', 'watchdog-reset', 'round-failed', 'some-failed', 'stop-growth', 'stop-reset', 'stop-missing',
    'ordinary-reset', 'ordinary-missing', 'unhealthy') if not (stage == 'after' and fault in ATTRIBUTED)])
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
        assert arm.name+'-short-b1-r0' not in events


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


def test_watchdog_zero_to_one_outside_a_repeat_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    live, arm, events = counter_live(tmp_path, monkeypatch, stats(), [stats(fires=1)])
    with pytest.raises(ValueError, match='watchdog grew'):
        live.cell(arm, 'short-b1')
    assert arm.name + '-short-b1-r0' not in events


def disconnect(base: dict[str, Json], rounds: int, **delta: int) -> dict[str, Json]:
    # PC1-threshold512-W-20261027T181314Z s3-control 150K r0: one drafter disconnect, 39 failed rounds.
    row = copy.deepcopy(base)
    counters = record(row['counters'])
    counters['rounds_ok'] = rounds
    for key, value in {'disconnects': 1, 'hellos': 1, 'fallback_not_ready': 36, 'fallback_timeout': 1,
                       'fallback_resync_pending': 2, 'resync_rounds': 1, 'rounds_failed': 39, **delta}.items():
        counters[key] = int(number(counters.get(key, 0))) + value
    return row


def test_transient_fault_inside_a_repeat_is_attributed_and_rerun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = stats(rounds=100, hellos=1)
    faulted = disconnect(base, 110)
    retry_after = copy.deepcopy(faulted)
    record(retry_after['counters'])['rounds_ok'] = 120
    final = copy.deepcopy(retry_after)
    live, arm, events = counter_live(tmp_path, monkeypatch, base, [copy.deepcopy(base), faulted,
                                                                  copy.deepcopy(faulted), retry_after, final])
    result = live.cell(arm, 'short-b1')
    assert events[-6:] == [arm.name + '-short-b1-r0-before', arm.name + '-short-b1-r0', arm.name + '-short-b1-r0-after',
                           arm.name + '-short-b1-r0-retry1-before', arm.name + '-short-b1-r0-retry1',
                           arm.name + '-short-b1-r0-retry1-after']
    reps = cast(list[Json], result['repetitions'])
    bad = cast(list[Json], result['faulted_repetitions'])
    assert len(reps) == len(bad) == 1
    assert record(record(bad[0])['drafter_faults'])['rounds_failed'] == 39
    assert record(reps[0])['proxy_after'] == retry_after
    checks = live.finish(arm)
    event = record(cast(list[Json], checks['drafter_faults'])[0])
    assert event['class'] == 'transient' and event['cell'] == 'short-b1' and event['repeat'] == 0
    assert (tmp_path / 'drafter-faults.jsonl').read_text().count('"transient"') == 1


# Totals that replace the observed event's: 65 failed rounds, two disconnects, or a breaker trip.
@pytest.mark.parametrize('delta,match', [
    ({'rounds_failed': 65}, 'systemic'), ({'disconnects': 2}, 'systemic'),
    ({'rounds_failed': 0}, 'systemic')])
def test_large_fault_inside_a_repeat_stops(delta: dict[str, int], match: str, tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    base = stats(rounds=100)
    faulted = disconnect(base, 110, **delta)
    if delta == {'rounds_failed': 0}:
        record(faulted)['breaker'] = {'trips': 1}
    live, arm, _ = counter_live(tmp_path, monkeypatch, base, [copy.deepcopy(base), faulted])
    with pytest.raises(ValueError, match=match):
        live.cell(arm, 'short-b1')
    assert (tmp_path / 'drafter-faults.jsonl').read_text().count('"systemic"') == 1


def test_third_fault_in_one_start_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows: list[dict[str, Json]] = []
    level = stats(rounds=100)
    for n in range(3):
        rows.append(copy.deepcopy(level))
        level = disconnect(level, 110 + 10 * n)
        rows.append(level)
    live, arm, _ = counter_live(tmp_path, monkeypatch, stats(rounds=100), rows)
    with pytest.raises(ValueError, match='<= 2 per start'):
        live.cell(arm, 'short-b1')
    assert [record(e)['class'] for e in live.drafter_faults[arm.name]] == ['transient', 'transient', 'systemic']


def test_rerun_must_fit_the_submit_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = stats(rounds=100)
    live, arm, _ = counter_live(tmp_path, monkeypatch, base, [copy.deepcopy(base), disconnect(base, 110)])
    live.group = 'chunk2048'
    live.submit = 1. + sum(w.arm_seconds('chunk2048', a.role) for a in w.arms('chunk2048')[2:])
    with pytest.raises(ValueError, match='cannot be re-run'):
        live.cell(arm, 'short-b1')


def test_baseline_once_per_start_after_greedy_and_warmup(rehearsal: tuple[w.Live, Remote]) -> None:
    live, fake = rehearsal
    result = w.window(live, 'chunk2048', lambda row: None)
    assert result['status'] == 'COMPLETE', result
    assert len(live.round_baseline) == 3
    for arm in w.arms('chunk2048'):
        assert fake.commands.count(arm.name+'-baseline') == 1
        assert fake.commands.index(arm.name+'-greedy') < fake.commands.index(arm.name+'-warmup')
        assert fake.commands.index(arm.name+'-warmup') < fake.commands.index(arm.name+'-baseline')
        assert fake.commands.index(arm.name+'-baseline') < fake.commands.index(arm.name+'-short-b1-r0-before')
        checks = record(record(record(result['starts'])[arm.name])['checks'])
        assert checks['startup_baseline'] == live.round_baseline[arm.name]


def test_missing_baseline_cannot_be_synthesized_from_first_cell(rehearsal: tuple[w.Live, Remote]) -> None:
    live, _ = rehearsal
    with pytest.raises(ValueError, match='measurement baseline'):
        live.cell(w.arms('chunk2048')[1], 'short-b1')


@pytest.mark.parametrize('fault', ['watchdog', 'missing-stop', 'stop-growth'])
def test_counter_fault_stops_candidate_before_timing_and_restores(fault: str,
        rehearsal: tuple[w.Live, Remote], monkeypatch: pytest.MonkeyPatch) -> None:
    live, fake = rehearsal
    original = live.stats
    def snapshot(label: str) -> dict[str, Json]:
        current = original(label)
        if label.startswith('s2-chunk2048'):
            counters = record(current['counters'])
            counters.update(rounds_failed=1, rounds_some_failed=2, fallback_internal_error=1)
            if label.endswith('-short-b1-r0-before'):
                if fault == 'watchdog': record(current['async_watchdog'])['fires'] = 1
                elif fault == 'missing-stop': counters.pop('fallback_internal_error')
                else: counters['fallback_internal_error'] = 2
        return current
    monkeypatch.setattr(live, 'stats', snapshot)
    result = w.window(live, 'chunk2048', lambda row: None)
    assert 'stop' in result, result
    assert 's1-control-short-b1-r0' in fake.commands
    assert 's2-chunk2048-short-b1-r0' not in fake.commands
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
