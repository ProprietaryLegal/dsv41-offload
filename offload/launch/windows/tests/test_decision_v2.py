"""WIN2 API: fairer decision rule and control archive (CPU only)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from offload.prefill_maxlen.evidence import number, record

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
    guard_bad = Series('a', 'accepted_per_round', True, (1.30, 1.31, 1.30), (1.10, 1.12, 1.30))
    assert d.decide_window([primary, guard_bad], ['a'])['status'] == 'NOT A CANDIDATE'
    guard_win = Series('a', 'accepted_per_round', True, (1.30, 1.31, 1.30), (1.50, 1.52, 1.51))
    assert d.decide_window([_flat('a'), guard_win], ['a'])['status'] == 'INCONCLUSIVE'


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


def test_archive_chain_spread_and_tamper(tmp_path: Path) -> None:
    path = tmp_path / 'archive.jsonl'
    assert d.archive_spread(path, 'short-b1', 'decode_tps') is None
    d.archive_append(path, [{'window': 'W1', 'cell': 'short-b1', 'metric': 'decode_tps', 'value': 60.0, 'max_tokens': 512},
                            {'window': 'W1', 'cell': 'short-b1', 'metric': 'decode_tps', 'value': 62.0, 'max_tokens': 512}])
    assert d.archive_spread(path, 'short-b1', 'decode_tps') is None
    d.archive_append(path, [{'window': 'W2', 'cell': 'short-b1', 'metric': 'decode_tps', 'value': 55.0, 'max_tokens': 512}])
    spread = d.archive_spread(path, 'short-b1', 'decode_tps')
    assert spread is not None and abs(spread - (61.0 - 55.0) / 58.0) < 1e-12
    lines = path.read_text().splitlines()
    assert json.loads(lines[0])['prev_sha256'] == d.GENESIS
    tampered = json.loads(lines[1]); tampered['value'] = 99.0
    path.write_text('\n'.join([lines[0], json.dumps(tampered, sort_keys=True), lines[2]]) + '\n')
    with pytest.raises(ValueError, match='chain broken'):
        d.archive_spread(path, 'short-b1', 'decode_tps')


def test_archive_row_requirements(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='lacks'):
        d.archive_append(tmp_path / 'a.jsonl', [{'cell': 'x', 'metric': 'm', 'value': 1.0, 'max_tokens': 512}])
    with pytest.raises(ValueError, match='lacks'):
        d.archive_append(tmp_path / 'a.jsonl', [{'window': 'w', 'cell': 'x', 'metric': 'm', 'value': 1.0}])
    with pytest.raises(ValueError):
        d.archive_append(tmp_path / 'a.jsonl', [{'window': 'w', 'cell': 'x', 'metric': 'm', 'value': 'fast',
                                                 'max_tokens': 512}])
    for bad in (0, True, 512.0, '512'):
        with pytest.raises(ValueError, match='positive integer'):
            d.archive_append(tmp_path / 'a.jsonl', [{'window': 'w', 'cell': 'x', 'metric': 'm', 'value': 1.0,
                                                     'max_tokens': bad}])
    assert not (tmp_path / 'a.jsonl').exists() or d.archive_rows(tmp_path / 'a.jsonl') == []


def test_archive_spread_refuses_mixed_max_tokens_unless_filtered(tmp_path: Path) -> None:
    path = tmp_path / 'archive.jsonl'
    row = {'cell': '32k-b3', 'metric': 'ms_per_round'}
    d.archive_append(path, [{**row, 'window': 'W1', 'value': 72.0, 'max_tokens': 1500},
                            {**row, 'window': 'W2', 'value': 73.0, 'max_tokens': 1500},
                            {**row, 'window': 'W3', 'value': 81.0, 'max_tokens': 512},
                            {**row, 'window': 'W4', 'value': 80.0, 'max_tokens': 512}])
    with pytest.raises(ValueError, match='mixes max_tokens'):
        d.archive_spread(path, '32k-b3', 'ms_per_round')
    spread = d.archive_spread(path, '32k-b3', 'ms_per_round', where={'max_tokens': 512})
    assert spread is not None and abs(spread - 1.0 / 80.5) < 1e-12
    assert d.archive_spread(path, '32k-b3', 'ms_per_round', where={'max_tokens': 1500}) is not None
