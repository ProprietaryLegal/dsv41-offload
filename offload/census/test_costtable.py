"""CPU tests for the depth cost-table fitter and the control-archive resolution: synthetic archive rows only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from offload.census import costtable as CT
from offload.prefill_maxlen.evidence import Json, number, record


def row(cell: str, setting: str, metric: str, value: float, repeat: int = 0, clock: int = 2300,
        start: str = 's1-cen2', window: str = 'CEN2-W-20990101T000000Z') -> dict[str, Json]:
    return {'schema': 'dsv41-control-archive-v3', 'cell': cell, 'setting': setting, 'metric': metric, 'value': value,
            'clock_cap_mhz': clock, 'clock_caps_mhz': [clock], 'commit': 'c' * 40, 'date': '20991231',
            'window': window, 'start': start, 'repeat': repeat, 'max_tokens': 300}


def fixed_rows(cell: str, ms: list[float], acc: list[float], n: int = 6, clock: int = 2300) -> list[dict[str, Json]]:
    out: list[dict[str, Json]] = []
    for d, (m, a) in enumerate(zip(ms, acc)):
        for i in range(n):
            jitter = (i - (n - 1) / 2) * 0.1
            out.append(row(cell, f'fixed:{d}', 'ms_per_round', m + jitter, i, clock))
            out.append(row(cell, f'fixed:{d}', 'accepted_per_round', a, i, clock))
    return out


def nums(value: Json) -> list[float]:
    assert isinstance(value, list)
    return [number(x) for x in value]


def lines(rows: list[dict[str, Json]]) -> list[str]:
    return [json.dumps(r) for r in rows]


SHORT_MS = [30.0, 32.5, 35.0, 37.5, 40.0, 42.5]
ACC = [0.0, 0.8, 1.4, 1.8, 2.0, 2.1]


def test_pav_is_the_closest_non_decreasing_fit() -> None:
    assert CT.pav([1.0, 3.0, 2.0, 4.0], [1, 1, 1, 1]) == [1.0, 2.5, 2.5, 4.0]
    assert CT.pav([5.0, 1.0], [3, 1]) == [4.0, 4.0]
    assert CT.pav([1.0, 2.0], [1, 1]) == [1.0, 2.0]


def test_fit_writes_round_ms_conf_and_ctx() -> None:
    rows = CT.parse(lines(fixed_rows('short-b1', SHORT_MS, ACC) +
                          fixed_rows('32k-b1', [m + 1.0 for m in SHORT_MS], ACC)))
    res = CT.fit(rows, 5, 'short-b1', {'short': 1024, '32k': 32768})
    assert res['clock_cap_mhz'] == 2300
    env = record(res['env'])
    assert env['VLLM_DSPARK_ROUND_MS'] == '30.0,32.5,35.0,37.5,40.0,42.5'
    assert env['VLLM_DSPARK_DRAFT_MS'] == '0'
    files = record(res['depth_file'])
    assert files['conf'] == 'conf:30.0,32.5,35.0,37.5,40.0,42.5'
    assert files['conf_cal'] == 'conf:30.0,32.5,35.0,37.5,40.0,42.5 cal=1'
    assert record(res['ctx'])['option'] == 'ctx=1024:0.00,32768:1.00'
    short = record(record(res['b1_tables'])['short-b1'])
    assert short['missing_depths'] == []
    assert nums(short['vs_default_ms']) == pytest.approx(
        [a - b for a, b in zip(SHORT_MS, CT.DEFAULT_ROUND_MS)])
    # tokens per ms (1 + acc) / ms: .0333 .0554 .0686 .0747 .0750 .0729 -> depth 4
    assert short['best_fixed_depth'] == 4
    depth0 = record(record(record(short['depths'])['0'])['ms_per_round'])
    assert depth0['n'] == 6 and depth0['low_n'] is False


def test_non_monotone_medians_are_pooled_and_raw_kept() -> None:
    ms = [30.0, 33.0, 32.0, 36.0, 38.0, 40.0]
    res = CT.fit(CT.parse(lines(fixed_rows('short-b1', ms, ACC))))
    t = record(record(res['b1_tables'])['short-b1'])
    assert nums(t['ms_raw']) == pytest.approx(ms)
    assert nums(t['ms_monotone']) == pytest.approx([30.0, 32.5, 32.5, 36.0, 38.0, 40.0])


def test_rows_table_from_streams_and_depths() -> None:
    rows = fixed_rows('32k-b1', [30.0, 33.0, 36.0, 39.0, 42.0, 45.0], ACC)
    rows += fixed_rows('32k-b4', [39.0, 50.0], ACC[:2])               # 4 and 8 live rows
    res = CT.fit(CT.parse(lines(rows)), 5, '32k-b1')
    t = record(record(res['rows_tables'])['32k'])
    assert t['measured_rows'] == [1, 2, 3, 4, 5, 6, 8]
    assert nums(t['ms_by_rows']) == pytest.approx([30.0, 33.0, 36.0, 39.0, 42.0, 45.0, 47.5, 50.0], abs=0.06)
    assert t['interpolated_rows'] == [7]
    assert str(t['confr']).startswith('confr:30.0,33.0,36.0,')


def test_missing_depth_and_mixed_clock_are_refused() -> None:
    rows = CT.parse(lines(fixed_rows('short-b1', SHORT_MS[:4], ACC[:4])))
    res = CT.fit(rows)
    assert res['env'] is None and 'lacks a full fixed-depth table' in str(res['refused'])
    assert record(record(res['b1_tables'])['short-b1'])['missing_depths'] == [4, 5]
    mixed = fixed_rows('short-b1', SHORT_MS, ACC)
    mixed[0]['clock_caps_mhz'] = [2300, 2150]
    with pytest.raises(ValueError, match='INCONCLUSIVE'):
        CT.fit(CT.parse(lines(mixed)))
    with pytest.raises(ValueError, match='no fixed-depth rows'):
        CT.fit(CT.parse(lines([row('short-b1', 'conf', 'ms_per_round', 38.0)])))
    with pytest.raises(ValueError, match='stream count'):
        CT.fit(CT.parse(lines([row('mixed150k-2decode', 'fixed:1', 'ms_per_round', 38.0)])))


def test_low_n_is_flagged() -> None:
    res = CT.fit(CT.parse(lines(fixed_rows('short-b1', SHORT_MS, ACC, n=3))))
    entry = record(record(record(record(res['b1_tables'])['short-b1'])['depths'])['2'])
    assert record(entry['ms_per_round'])['low_n'] is True


def test_resolution_pairs_consecutive_control_starts() -> None:
    rows: list[dict[str, Json]] = []
    for w in range(6):
        window = f'W{w}'
        for start, value in (('s1-control', 50.0), ('s3-control', 50.0 * (1 + 0.01 * (w - 2.5))),
                             ('s2-arm', 99.0)):
            r = row('short-b1', 'control', 'decode_tps', value, 0, 2300, start, window)
            if start == 's2-arm':
                r['setting'] = 'fixed:2'                         # a candidate arm never enters the resolution
            rows.append(r)
    rows.append(row('short-b1', 'control', 'decode_tps', 10.0, 0, 2150, 's1-control', 'W9'))
    res = CT.resolution(CT.parse(lines(rows)), 2300)
    cell = record(record(res['cells'])['short-b1'])
    assert cell['pairs'] == 6 and cell['status'] == 'OK'
    drifts = [0.01 * (w - 2.5) / (1 + 0.005 * (w - 2.5)) for w in range(6)]
    assert number(cell['max_abs']) == pytest.approx(max(abs(d) for d in drifts))
    assert number(cell['two_x_resolution']) == pytest.approx(2 * number(cell['sd']))
    assert CT.resolution(CT.parse(lines(rows)), 2200)['controls_at_clock'] == 0


def test_cli_fit_and_resolution(tmp_path: Path) -> None:
    src = tmp_path / 'rows.jsonl'
    src.write_text('\n'.join(lines(fixed_rows('short-b1', SHORT_MS, ACC))) + '\n')
    out = tmp_path / 'fit.json'
    assert CT.main(['fit', '--rows', str(src), '--output', str(out)]) == 0
    assert record(json.loads(out.read_text()))['type'] == 'cen2-cost-table'
    assert CT.main(['fit', '--rows', str(src), '--output', str(out)]) == 1
    res = tmp_path / 'res.json'
    assert CT.main(['resolution', '--archive', str(src), '--clock', '2300', '--output', str(res)]) == 0
