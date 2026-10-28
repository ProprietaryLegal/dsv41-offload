"""WIN2 API: fairer decision rule and control archive (CPU only)."""
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import cast

import pytest

from offload.prefill_maxlen.evidence import Json, number, record

from offload.launch.windows import decision_v2 as d

Series = d.Series


def test_t_quantiles_match_table_and_floor_fractional_df() -> None:
    assert d.t975(2) == 4.303 and d.t975(2.9) == 4.303 and d.t975(200) == 2.042
    with pytest.raises(ValueError):
        d.t975(0.5)


def test_clear_paired_win_lower_bound_above_zero() -> None:
    s = Series('short-b1', 'decode_tps', True, (60.0, 61.0, 60.5), (64.0, 65.2, 64.6))
    r = d.decide_cell(s)
    interval = r['interval95']
    assert isinstance(interval, list)
    lo = number(interval[0])
    assert r['verdict'] == 'WIN' and number(r['point_gain']) > .05 and lo > 0


def test_gain_below_min_is_inconclusive_even_with_tight_interval() -> None:
    s = Series('short-b1', 'decode_tps', True, (60.0, 60.0, 60.0), (60.9, 60.9, 60.9))
    assert d.decide_cell(s)['verdict'] == 'INCONCLUSIVE'


def test_noisy_five_percent_is_inconclusive_not_loss() -> None:
    s = Series('32k-b1', 'ttft_s', False, (15.0, 14.0, 15.4), (13.0, 15.2, 13.6))
    assert d.decide_cell(s)['verdict'] == 'INCONCLUSIVE'


def test_loss_needs_two_of_three_repeats_beyond_own_spread() -> None:
    worse = Series('short-b1', 'ms_per_round', False, (40.0, 40.4, 40.2), (43.0, 43.5, 40.1))
    assert d.decide_cell(worse)['verdict'] == 'LOSS'
    one_bad = Series('short-b1', 'ms_per_round', False, (40.0, 40.4, 40.2), (43.0, 40.3, 40.1))
    assert d.decide_cell(one_bad)['verdict'] != 'LOSS'


def test_orientation_lower_is_better() -> None:
    faster = Series('150k-b1', 'ttft_s', False, (69.0, 69.5, 69.2), (64.0, 64.3, 64.1))
    r = d.decide_cell(faster)
    assert r['verdict'] == 'WIN' and number(r['point_gain']) > 0


def test_unpaired_welch_interval() -> None:
    s = Series('mixed', 'decode_tps', True, (50.0, 51.0, 49.5, 50.5), (55.0, 56.0, 55.5), paired=False)
    r = d.decide_cell(s)
    assert r['paired'] is False and r['verdict'] == 'WIN' and 'Welch' in str(r['interval_method'])


@pytest.mark.parametrize('control,candidate', [((1.0, 2.0), (1.0, 2.0, 3.0)), ((1.0, 2.0, 3.0), (1.0, 2.0))])
def test_fewer_than_three_repeats_refused(control: tuple[float, ...], candidate: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match='at least 3'):
        d.decide_cell(Series('c', 'm', True, control, candidate))


def test_paired_needs_equal_lengths_and_positive_values() -> None:
    with pytest.raises(ValueError, match='equal repeats'):
        d.decide_cell(Series('c', 'm', True, (1.0, 2.0, 3.0), (1.0, 2.0, 3.0, 4.0)))
    with pytest.raises(ValueError, match='positive'):
        d.decide_cell(Series('c', 'm', True, (1.0, 0.0, 3.0), (1.0, 2.0, 3.0)))


def _win(cell: str) -> Series:
    return Series(cell, 'decode_tps', True, (60.0, 61.0, 60.5), (64.0, 65.2, 64.6))


def _loss(cell: str) -> Series:
    return Series(cell, 'decode_tps', True, (60.0, 60.2, 60.1), (55.0, 55.1, 60.0))


def _flat(cell: str) -> Series:
    return Series(cell, 'decode_tps', True, (60.0, 61.0, 60.5), (60.2, 61.1, 60.6))


def test_window_candidate_needs_a_win_and_no_loss() -> None:
    r = d.decide_window([_win('a'), _flat('b')], ['a', 'b'])
    assert r['rule'] == 'win2-v1' and r['status'] == 'CANDIDATE'
    assert d.decide_window([_win('a'), _loss('b')], ['a', 'b'])['status'] == 'NOT A CANDIDATE'
    assert d.decide_window([_flat('a'), _flat('b')], ['a', 'b'])['status'] == 'INCONCLUSIVE'


def test_reported_cells_never_decide_and_may_have_fewer_repeats() -> None:
    reported = Series('mixed', 'decode_tps', True, (60.0,), (40.0,))
    r = d.decide_window([_win('a'), reported], ['a'])
    assert r['status'] == 'CANDIDATE'
    cell = record(record(r['cells'])['mixed'])
    assert cell['decides'] is False and cell['verdict'] == 'REPORTED'


def test_guard_metric_can_only_cause_loss() -> None:
    primary = _win('a')
    guard_bad = Series('a', 'accepted_per_round', True, (1.30, 1.31, 1.30), (1.10, 1.12, 1.11))
    assert d.decide_window([primary, guard_bad], ['a'])['status'] == 'NOT A CANDIDATE'
    guard_win = Series('a', 'accepted_per_round', True, (1.30, 1.31, 1.30), (1.50, 1.52, 1.51))
    r = d.decide_window([_flat('a'), guard_win], ['a'])
    assert r['status'] == 'INCONCLUSIVE' and r['rule_revision'] == 2
    assert record(record(record(record(r['cells'])['a'])['metrics'])['accepted_per_round'])['verdict'] == 'INCONCLUSIVE'


def test_guard_loss_needs_whole_interval_below_zero() -> None:
    """S92 FIX 3: two of three guard repeats beyond the spread but the interval crosses zero -> no LOSS (rev 1: LOSS)."""
    noisy = Series('a', 'accepted_per_round', True, (1.30, 1.31, 1.30), (1.10, 1.12, 1.30))
    as_guard = d.decide_cell(noisy, guard=True)
    assert as_guard['loss_repeat_test'] is True and number(cast(list[Json], as_guard['interval95'])[1]) > 0
    assert as_guard['verdict'] == 'INCONCLUSIVE' and d.decide_cell(noisy)['verdict'] == 'LOSS'   # primary rule unchanged
    assert d.decide_window([_win('a'), noisy], ['a'])['status'] == 'CANDIDATE'


def test_paired_loss_threshold_is_sqrt2_spread() -> None:
    control = (40.0, 40.4, 40.2)          # s 0.2, c4(3) 0.8862: spread 0.949 %; paired threshold 1.342 %
    within = Series('c', 'ms_per_round', False, control, (40.45, 40.9, 40.2))      # -1.12 %, -1.24 %, 0 %
    r = d.decide_cell(within)
    assert abs(number(r['control_spread']) - 1.69 * 0.2 / d.c4(3) / 40.2) < 1e-12 and abs(d.c4(3) - 0.886227) < 1e-6
    assert abs(number(r['loss_threshold']) - 2 ** .5 * number(r['control_spread'])) < 1e-15
    assert r['loss_repeats'] == 0 and r['verdict'] != 'LOSS'        # single-value spread: 2 of 3 -> LOSS
    unpaired = d.decide_cell(Series('c', 'ms_per_round', False, control, (40.7, 40.9, 40.8), paired=False))
    assert unpaired['loss_threshold'] == unpaired['control_spread'] and unpaired['loss_repeats'] == 3


def test_control_spread_does_not_grow_with_control_repeats() -> None:
    """S92 FIX 3 item 5: the range of 8 controls is ~1.7 x the range of 3; the 1.69 x SD spread stays put."""
    rng = random.Random(9)
    def mean_of(n: int, key: str) -> float:
        values = []
        for _ in range(2000):
            control = tuple(40 * (1 + rng.gauss(0, .02)) for _ in range(n))
            values.append(number(d.decide_cell(Series('c', 'ms_per_round', False, control, (40.0, 40.0, 40.0),
                                                      paired=False))[key]))
        return statistics.fmean(values)
    assert mean_of(8, 'control_range') / mean_of(3, 'control_range') > 1.5
    assert 0.95 < mean_of(8, 'control_spread') / mean_of(3, 'control_spread') < 1.05
    assert abs(mean_of(3, 'control_spread') / mean_of(3, 'control_range') - 1.0) < 0.05    # = expected range of 3


def test_groups_give_separate_single_and_multi_user_verdicts() -> None:
    r = d.decide_window([_win('mixed'), _flat('short')], ['mixed', 'short'],
                        groups={'multi-user': ['mixed'], 'single-user': ['short']})
    assert r['groups'] == {'multi-user': 'CANDIDATE', 'single-user': 'INCONCLUSIVE'}
    with pytest.raises(ValueError, match='decision cells only'):
        d.decide_window([_win('a'), _flat('b')], ['a'], groups={'x': ['b']})


@pytest.mark.parametrize('decision', [[], ['a', 'b', 'c'], ['a', 'a']])
def test_one_or_two_distinct_decision_cells(decision: list[str]) -> None:
    with pytest.raises(ValueError):
        d.decide_window([_win('a'), _win('b'), _win('c')], decision)


def test_missing_decision_cell_refused() -> None:
    with pytest.raises(ValueError, match='without measurements'):
        d.decide_window([_win('a')], ['z'])


PROD = '596ba1cf4e3b11be29b41a57e7f69a5d8efa4799'


def _row(window: str, value: float, *, cell: str = 'short-b1', metric: str = 'decode_tps', **extra: object) -> dict[str, Json]:
    row: dict[str, Json] = {'window': window, 'cell': cell, 'metric': metric, 'value': value, 'max_tokens': 512,
                            'commit': PROD, 'clock_cap_mhz': 2150, 'date': '20261027', 'start': 's1-control',
                            'repeat': 0}
    row.update(cast(dict[str, Json], extra))
    return row


def test_archive_chain_spread_and_tamper(tmp_path: Path) -> None:
    path = tmp_path / 'archive.jsonl'
    assert d.archive_spread(path, 'short-b1', 'decode_tps') is None
    d.archive_append(path, [_row('W1', 60.0), _row('W1', 62.0, repeat=1)])
    assert d.archive_spread(path, 'short-b1', 'decode_tps') is None
    d.archive_append(path, [_row('W2', 55.0)])
    spread = d.archive_spread(path, 'short-b1', 'decode_tps')
    assert spread is not None and abs(spread - (61.0 - 55.0) / 58.0) < 1e-12
    assert d.archive_head(path)['rows'] == 3
    lines = path.read_text().splitlines()
    assert json.loads(lines[0])['prev_sha256'] == d.GENESIS
    tampered = json.loads(lines[1]); tampered['value'] = 99.0
    path.write_text('\n'.join([lines[0], json.dumps(tampered, sort_keys=True), lines[2]]) + '\n')
    with pytest.raises(ValueError, match='chain broken'):
        d.archive_spread(path, 'short-b1', 'decode_tps')


@pytest.mark.parametrize('field,bad', [
    ('max_tokens', 0), ('max_tokens', True), ('max_tokens', 512.0), ('max_tokens', '512'),
    ('commit', None), ('commit', '596ba1c'), ('commit', 'abc'), ('commit', PROD.upper()),
    ('clock_cap_mhz', None), ('clock_cap_mhz', [2150]), ('clock_cap_mhz', 'unknown'), ('clock_cap_mhz', 2150.0),
    ('clock_cap_mhz', 0), ('clock_source', 'guess'), ('clock_caps_mhz', 2150), ('clock_caps_mhz', 'Unknown'),
    ('date', '2026-10-27'), ('date', 20261027), ('start', ''), ('start', None),
    ('repeat', -1), ('repeat', True), ('repeat', 1.0), ('value', 'fast')])
def test_archive_row_requirements(tmp_path: Path, field: str, bad: object) -> None:
    """S92 FIX 2: rows without a 40-hex production commit, the start's clock cap, date, start and repeat are refused."""
    row = _row('w', 1.0)
    row[field] = cast(Json, bad)
    with pytest.raises(ValueError):
        d.archive_append(tmp_path / 'a.jsonl', [row])
    assert not (tmp_path / 'a.jsonl').exists() or d.archive_rows(tmp_path / 'a.jsonl') == []


@pytest.mark.parametrize('field', d.ARCHIVE_REQUIRED)
def test_archive_required_fields(tmp_path: Path, field: str) -> None:
    row = _row('w', 1.0)
    del row[field]
    with pytest.raises(ValueError, match='lacks'):
        d.archive_append(tmp_path / 'a.jsonl', [row])
    d.archive_append(tmp_path / 'a.jsonl', [_row('w', 1.0, clock_caps_mhz='unknown', clock_source='clock-log')])


def test_archive_spread_refuses_mixed_max_tokens_unless_filtered(tmp_path: Path) -> None:
    path = tmp_path / 'archive.jsonl'
    row = {'cell': '32k-b3', 'metric': 'ms_per_round'}
    d.archive_append(path, [_row('W1', 72.0, max_tokens=1500, **row), _row('W2', 73.0, max_tokens=1500, **row),
                            _row('W3', 81.0, **row), _row('W4', 80.0, **row)])
    with pytest.raises(ValueError, match='max_tokens'):
        d.archive_spread(path, '32k-b3', 'ms_per_round')
    spread = d.archive_spread(path, '32k-b3', 'ms_per_round', where={'max_tokens': 512})
    assert spread is not None and abs(spread - 1.0 / 80.5) < 1e-12
    assert d.archive_spread(path, '32k-b3', 'ms_per_round', where={'max_tokens': 1500}) is not None


def test_archive_spread_refuses_mixed_commits_and_clocks(tmp_path: Path) -> None:
    """S92 counterexamples: rows without a commit were accepted, the spread pooled two production commits, and 2150
    and 2300 MHz rows pooled silently (5.73 % pooled against 0.51 % at 2150)."""
    path = tmp_path / 'archive.jsonl'
    other = 'b' * 40
    d.archive_append(path, [_row('W1', 40.0), _row('W2', 40.4), _row('W3', 37.0, commit=other),
                            _row('W4', 41.0, clock_cap_mhz=2300), _row('W5', 37.5, clock_cap_mhz=2300)])
    with pytest.raises(ValueError, match='commit'):
        d.archive_spread(path, 'short-b1', 'decode_tps')
    with pytest.raises(ValueError, match='clock_cap_mhz'):
        d.archive_spread(path, 'short-b1', 'decode_tps', where={'commit': PROD})
    spread = d.archive_spread(path, 'short-b1', 'decode_tps', where={'commit': PROD, 'clock_cap_mhz': 2150})
    assert spread is not None and abs(spread - 0.4 / 40.2) < 1e-12
    per = d.archive_spreads(path, 'short-b1', 'decode_tps')
    assert set(per) == {f'max_tokens=512 clock_cap_mhz=2150 commit={PROD[:12]}', 'max_tokens=512 clock_cap_mhz=2300 commit=' + PROD[:12],
                        'max_tokens=512 clock_cap_mhz=2150 commit=bbbbbbbbbbbb'}
    assert per[f'max_tokens=512 clock_cap_mhz=2150 commit={PROD[:12]}'] == spread


def test_older_archive_is_refused_by_name(tmp_path: Path) -> None:
    path = tmp_path / 'a.jsonl'
    path.write_text(json.dumps({**_row('W1', 1.0), 'schema': 'dsv41-control-archive-v2', 'prev_sha256': d.GENESIS}) + '\n')
    with pytest.raises(ValueError, match="expected 'dsv41-control-archive-v3'"):
        d.archive_append(path, [_row('W2', 1.0)])


def _null_window_rates(paired: bool, n_control: int, trials: int) -> dict[str, float]:
    """S92 sim_corr's model: two decision cells, no real effect; ms/round cv 1.5 %, acceptance cv 12 %, tok/s derived."""
    rng = random.Random(5)
    def arm(n: int) -> tuple[list[float], list[float], list[float]]:
        ms = [40 * (1 + rng.gauss(0, .015)) for _ in range(n)]
        acc = [max(.3, 1.35 * (1 + rng.gauss(0, .12))) for _ in range(n)]
        return ms, acc, [(1 + a) * 1000 / m for a, m in zip(acc, ms)]
    out = {'CANDIDATE': 0, 'INCONCLUSIVE': 0, 'NOT A CANDIDATE': 0}
    for _ in range(trials):
        series = []
        for cell in 'ab':
            k, c = arm(n_control), arm(3)
            for i, (metric, hib) in enumerate((('ms_per_round', False), ('accepted_per_round', True), ('decode_tps', True))):
                series.append(Series(cell, metric, hib, tuple(k[i]), tuple(c[i]), paired))
        out[str(d.decide_window(series, ['a', 'b'])['status'])] += 1
    return {k: v / trials for k, v in out.items()}


@pytest.mark.parametrize('paired,n_control,reject_max', [(True, 3, .14), (False, 4, .17)])
def test_no_effect_window_rates(paired: bool, n_control: int, reject_max: float) -> None:
    """S92 FIX 3: with both guards a no-effect window was rejected 32.5 % (paired) / 16.0 % (unpaired) of the time.
    Revision 2 (10,000 windows, seed 5, WIN2.md): 11.4 % / 14.6 % rejected, 1.8 % / 1.9 % accepted."""
    rates = _null_window_rates(paired, n_control, 3000)
    assert rates['NOT A CANDIDATE'] < reject_max and rates['CANDIDATE'] > .01
