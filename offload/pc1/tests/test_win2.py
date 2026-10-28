"""WIN2 plans, metric table and the cross-window control archive (CPU only; temp archive)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from offload.gb10_budget import window_cells as wc
from offload.launch.windows import decision_v2 as d2
from offload.pc1 import build as b, cells, metrics, window as w
from offload.pc1.tests.test_window import Remote, rehearsal as rehearsal, starts
from offload.prefill_maxlen.evidence import Json, number, record


def test_registered_plans_match_the_win2_brief() -> None:
    assert set(cells.PLANS) == set(b.ARMS[1:])
    for group, plan in cells.PLANS.items():
        assert 1 <= len(plan.decision) <= d2.MAX_DECISION_CELLS
        assert not set(plan.decision) & set(plan.reported)
        assert set(plan.cells) == set(plan.decision) | set(plan.reported) == set(cells.CELLS)
        assert all(set(members) <= set(plan.decision) for members in plan.groups.values())
        assert [plan.repeats(n) for n in plan.decision] == [3] * len(plan.decision)
        assert [plan.repeats(n) for n in plan.reported] == [1] * len(plan.reported)
        # The primary (first) metric of each cell decides: ms/round for decode cells, TTFT for prefill cells.
        assert all(metrics.METRICS[n][0].decides for n in plan.decision)
    assert cells.plan_for('threshold2048').decision == ('short-b1', 'mixed150k-2decode')
    assert dict(cells.plan_for('threshold2048').groups) == {'single-user': ('short-b1',),
                                                            'multi-user': ('mixed150k-2decode',)}
    assert cells.plan_for('chunk16384').decision == cells.plan_for('chunk8192').decision == ('32k-b1', '150k-b1')
    primary = {n: metrics.METRICS[n][0].name for n in cells.CELLS}
    assert primary == {'short-b1': 'ms_per_round', '32k-b1': 'ttft_s', '150k-b1': 'ttft_s',
                       'mixed150k-2decode': 'ms_per_round'}
    for name in cells.CELLS:
        names = [m.name for m in metrics.METRICS[name]]
        assert len(names) == len(set(names))
    assert [m.name for m in metrics.METRICS['short-b1'] if m.decides] == ['ms_per_round', 'accepted_per_round', 'decode_tps']
    assert [m.name for m in metrics.METRICS['mixed150k-2decode'] if m.decides] == [
        'ms_per_round', 'accepted_per_round', 'decode_tps', 'long_doc_ttft_s']
    assert [m.name for m in metrics.METRICS['150k-b1'] if m.decides] == ['ttft_s']
    with pytest.raises(ValueError, match='no registered window plan'):
        cells.plan_for('control')
    with pytest.raises(ValueError, match='outside this window plan'):
        cells.Plan(decision=('32k-b1',), reported=()).repeats('short-b1')


@pytest.mark.parametrize('indices', [(), (3,), (-1,), (0, 0)])
def test_repeated_cell_refuses_an_unregistered_repeat_index(tmp_path: Path, indices: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match='registered repeat indices'):
        cells.repeated(cast(wc.Api, None), '32k-b1', {}, deadline=1e12, salt='s', ledger=tmp_path / 'l',
                       output=tmp_path / 'o', indices=indices)


def test_mixed_completion_metric_is_the_mean_of_each_request_total() -> None:
    cell = record(record(record(starts('threshold2048')['s1-control'])['cells'])['mixed150k-2decode'])
    row = record(cast(list[Json], cell['repetitions'])[0])
    totals = [number(record(s)['total_s']) for s in cast(list[Json], row['streams'])]
    assert metrics.mean_completion_s(row) == pytest.approx(sum(totals) / 3)
    record(cast(list[Json], row['streams'])[2])['total_s'] = 0
    with pytest.raises(ValueError):
        metrics.mean_completion_s(row)


def complete(group: str) -> dict[str, Json]:
    rows = starts(group)
    return {'status': 'COMPLETE', 'starts': cast(Json, rows), 'decision': w.decision(group, rows)}


def window_dir(root: Path, name: str, caps: dict[str, int]) -> Path:
    """A window out dir whose start logs prove each start's clock cap on both Sparks, as start_exp.sh prints it."""
    out = root / name
    (out / 'cmd').mkdir(parents=True)
    for i, (start, cap) in enumerate(caps.items()):
        (out / 'cmd' / f'{i:04d}-{start}-start.log').write_text(f'spark clock cap {cap} MHz\nsparkworker clock cap {cap} MHz\n')
    return out


CONTROLS = ('s1-control', 's3-control')


def test_archive_appends_control_values_only_for_complete_windows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = tmp_path / 'control-archive.jsonl'
    monkeypatch.setattr(w, 'ARCHIVE', archive)
    w.archive_controls({'status': 'INCOMPLETE', 'starts': {}, 'decision': {'status': 'NOT A CANDIDATE'}}, 'chunk16384',
                       'x', tmp_path / 'x')
    assert not archive.exists()
    first = complete('chunk16384')
    out = window_dir(tmp_path, 'PC1-chunk16384-W-20261027T210000Z', dict.fromkeys(CONTROLS, 2300))
    w.archive_controls(first, 'chunk16384', out.name, out)
    lines = d2.archive_rows(archive)
    plan = cells.plan_for('chunk16384')
    expected = sum(len(metrics.METRICS[n]) * plan.repeats(n) for n in plan.cells) * 2
    assert len(lines) == expected
    assert {r['start'] for r in lines} == set(CONTROLS)
    assert {r['window'] for r in lines} == {out.name} and {r['commit'] for r in lines} == {b.PRODUCTION}
    assert {r['clock_cap_mhz'] for r in lines} == {2300} and {r['clock_source'] for r in lines} == {'start-log'}
    assert {r['date'] for r in lines} == {'20261027'} and {r['max_tokens'] for r in lines} == {512}
    assert sorted({cast(int, r['repeat']) for r in lines}) == [0, 1, 2]
    ttft = [number(r['value']) for r in lines if r['cell'] == '150k-b1' and r['metric'] == 'ttft_s']
    assert ttft == pytest.approx([1., 1.001, 1.002] * 2)
    report = record(first['archive'])
    assert record(report['head_before'])['rows'] == 0 and record(report['head_after'])['rows'] == expected
    # One archived window has no day-to-day spread yet; a second identical window has zero.
    assert record(record(first['decision'])['archive_spread']) == {'32k-b1/ttft_s': None, '150k-b1/ttft_s': None}
    second = complete('chunk16384')
    out_b = window_dir(tmp_path, 'PC1-chunk16384-W-20261027T220000Z', dict.fromkeys(CONTROLS, 2300))
    w.archive_controls(second, 'chunk16384', out_b.name, out_b)
    assert len(d2.archive_rows(archive)) == 2 * expected
    assert record(record(second['decision'])['archive_spread']) == {'32k-b1/ttft_s': 0., '150k-b1/ttft_s': 0.}
    # A window at another clock cap never pools with them.
    third = complete('chunk16384')
    out_c = window_dir(tmp_path, 'PC1-chunk16384-W-20261027T230000Z', dict.fromkeys(CONTROLS, 2150))
    w.archive_controls(third, 'chunk16384', out_c.name, out_c)
    assert record(record(third['decision'])['archive_spread']) == {'32k-b1/ttft_s': None, '150k-b1/ttft_s': None}
    # A start whose log does not prove its clock cap refuses the append (main records archive_error, loudly).
    out_d = window_dir(tmp_path, 'PC1-chunk16384-W-20261028T000000Z', {'s1-control': 2300})
    (out_d / 'cmd' / '0009-s3-control-start.log').write_text('spark clock cap 2300 MHz\n')
    with pytest.raises(ValueError, match='clock cap'):
        w.archive_controls(complete('chunk16384'), 'chunk16384', out_d.name, out_d)
    archive.write_text(archive.read_text().replace('"s1-control"', '"edited"', 1))
    out_e = window_dir(tmp_path, 'PC1-chunk16384-W-20261028T010000Z', dict.fromkeys(CONTROLS, 2300))
    with pytest.raises(ValueError, match='chain broken'):
        w.archive_controls(complete('chunk16384'), 'chunk16384', out_e.name, out_e)


def test_archive_keeps_completed_controls_of_a_stopped_window_and_ignores_other_max_tokens(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = tmp_path / 'control-archive.jsonl'
    monkeypatch.setattr(w, 'ARCHIVE', archive)
    # History at 1,500 tokens (NBW) under the same cell name must not pool with PC1's 512-token cell.
    d2.archive_append(archive, [{'window': f'NBW-{i}', 'cell': 'short-b1', 'metric': 'ms_per_round', 'value': v,
                                 'max_tokens': 1500, 'commit': b.PRODUCTION, 'clock_cap_mhz': 2300, 'date': '20261027',
                                 'start': 's1-K', 'repeat': 0} for i, v in enumerate((70., 90.))])
    rows = starts('threshold2048')
    record(rows['s3-control'])['status'] = 'STARTED'
    result: dict[str, Json] = {'status': 'INCOMPLETE', 'stop': 'ValueError: STOP', 'starts': cast(Json, rows),
                               'decision': {'status': 'NOT A CANDIDATE', 'reason': 'STOP'}}
    out = window_dir(tmp_path, 'PC1-threshold2048-W-20261027T210000Z', {'s1-control': 2300})
    w.archive_controls(result, 'threshold2048', out.name, out)
    archived = [r for r in d2.archive_rows(archive) if r['window'] == out.name]
    assert archived and {r['start'] for r in archived} == {'s1-control'}
    report = record(result['archive'])
    assert report['rows_appended'] == len(archived)
    assert record(report['spread_primary']) == {'short-b1/ms_per_round': None, 'mixed150k-2decode/ms_per_round': None}
    assert 'archive_spread' not in record(result['decision'])
    with pytest.raises(ValueError, match="mixes {'max_tokens'"):
        d2.archive_spread(archive, 'short-b1', 'ms_per_round')


@pytest.mark.parametrize('caps,status', [((2300, 2300, 2300), 'CANDIDATE'), ((2300, 2150, 2300), 'NOT A CANDIDATE')])
def test_starts_at_different_clock_caps_decide_nothing(tmp_path: Path, caps: tuple[int, int, int], status: str) -> None:
    result = complete('threshold2048')
    out = window_dir(tmp_path, 'PC1-threshold2048-W-20261027T210000Z',
                     dict(zip([a.name for a in w.arms('threshold2048')], caps)))
    w.clock_check(result, 'threshold2048', out)
    assert record(result['decision'])['status'] == status
    assert {record(v)['clock_cap_mhz'] for v in record(result['clock_caps_by_start']).values()} == set(caps)
    if status != 'CANDIDATE':
        assert record(record(result['decision'])['speed_decision'])['status'] == 'CANDIDATE'


# PC1-threshold512-W-20261027T181314Z s3-control 150K r0: one drafter disconnect, 39 failed rounds (cumulative counters).
EVENT = {'disconnects': 1, 'hellos': 1, 'fallback_not_ready': 36, 'fallback_timeout': 1, 'fallback_resync_pending': 2,
         'resync_rounds': 1, 'rounds_failed': 39}


def inject(live: w.Live, monkeypatch: pytest.MonkeyPatch, triggers: set[str]) -> None:
    """From each trigger snapshot on, that start's drafter counters carry the event (they never go back down)."""
    original = live.stats
    extra: dict[str, dict[str, int]] = {}
    def snapshot(label: str) -> dict[str, Json]:
        current = original(label)
        start = label.split('-')[0]
        if label in triggers:
            add = extra.setdefault(start, {})
            for key, value in EVENT.items():
                add[key] = add.get(key, 0) + value
        counters = record(current['counters'])
        for key, value in extra.get(start, {}).items():
            counters[key] = int(number(counters.get(key, 0))) + value
        return current
    monkeypatch.setattr(live, 'stats', snapshot)


@pytest.mark.parametrize('triggers,verdict,status', [
    ({'s3-control-150k-b1-r0-after'}, 'CONTROL-ONLY', 'CANDIDATE'),
    ({'s2-threshold2048-short-b1-r1-after'}, 'ARM-ONLY-1', 'CANDIDATE'),
    ({'s1-control-mixed150k-2decode-r2-after', 's2-threshold2048-short-b1-r0-after'}, 'BOTH', 'CANDIDATE'),
    ({'s2-threshold2048-short-b1-r0-after', 's2-threshold2048-mixed150k-2decode-r1-after'}, 'ARM-SUSPECT',
     'NOT A CANDIDATE')])
def test_drafter_fault_is_attributed_rerun_and_the_window_completes(
        rehearsal: tuple[w.Live, Remote], monkeypatch: pytest.MonkeyPatch,
        triggers: set[str], verdict: str, status: str) -> None:
    live, fake = rehearsal
    live.group = 'threshold2048'
    inject(live, monkeypatch, triggers)
    result = w.window(live, 'threshold2048', lambda row: None)
    assert result['status'] == 'COMPLETE' and 'stop' not in result, result.get('stop')
    for trigger in triggers:
        assert trigger.removesuffix('-after') + '-retry1' in fake.commands
    decision = record(result['decision'])
    assert record(decision['drafter_faults'])['verdict'] == verdict
    assert decision['status'] == status
    if verdict == 'ARM-SUSPECT':
        assert decision['speed_status'] == 'CANDIDATE'
        assert decision['groups'] == {'single-user': 'NOT A CANDIDATE', 'multi-user': 'NOT A CANDIDATE'}
    for start in record(result['starts']).values():
        for cell in record(record(start)['cells']).values():
            # Timed repeats are the clean ones; faulted attempts stay as evidence only.
            for rep in cast(list[Json], record(cell)['repetitions']):
                assert w.fault_levels(record(record(rep)['proxy_after'])) == w.fault_levels(record(record(rep)['proxy_before']))
    faulted = [record(r) for s in record(result['starts']).values() for c in record(record(s)['cells']).values()
               for r in cast(list[Json], record(c)['faulted_repetitions'])]
    assert len(faulted) == len(triggers)
    assert all(record(r['drafter_faults'])['rounds_failed'] == 39 for r in faulted)


@pytest.mark.parametrize('counts,verdict', [
    ({'s1-control': ('control', 0), 's2-x': ('x', 0), 's3-control': ('control', 0)}, 'NONE'),
    ({'s1-control': ('control', 1), 's2-x': ('x', 0), 's3-control': ('control', 2)}, 'CONTROL-ONLY'),
    ({'s1-control': ('control', 1), 's2-x': ('x', 2), 's3-control': ('control', 0)}, 'BOTH'),
    ({'s1-control': ('control', 0), 's2-x': ('x', 1), 's3-control': ('control', 0)}, 'ARM-ONLY-1'),
    ({'s1-control': ('control', 0), 's2-x': ('x', 2), 's3-control': ('control', 0)}, 'ARM-SUSPECT')])
def test_fault_attribution_table(counts: dict[str, tuple[str, int]], verdict: str) -> None:
    result = metrics.fault_attribution(counts)
    assert result['verdict'] == verdict
    if verdict == 'ARM-SUSPECT':
        assert 'p = 0.111' in str(result['note'])
