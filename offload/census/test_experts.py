"""CPU tests for the distinct-expert reader: a faithful re-implementation of the hook's summary lines drives it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from offload.census import experts as E
from offload.prefill_maxlen.evidence import Json, number, record


def simulate(calls: list[tuple[int, int, int]], pid: int = 7) -> list[str]:
    """vllm_exl3_moe/forward.py _record_stats: cumulative per m, one rounded summary line per 2,000 calls."""
    stats: dict[int, list[int]] = {}
    out: list[str] = []
    for m, live, uniq in calls:
        b = stats.setdefault(m, [0, 0, 0])
        b[0] += 1
        b[1] += uniq
        b[2] += 6 * live
        if sum(v[0] for v in stats.values()) % 2000 == 0:
            out.append(json.dumps({'pid': pid, 'calls_by_tokens': {k: v[0] for k, v in sorted(stats.items())},
                                   'mean_distinct_experts': {k: round(v[1] / v[0], 2) for k, v in sorted(stats.items())},
                                   'mean_routed_slots': {k: round(v[2] / v[0], 2) for k, v in sorted(stats.items())}}))
    return out


def depth_run(live: int, values: tuple[int, ...], n: int, m: int = 6) -> list[tuple[int, int, int]]:
    return [(m, live, values[i % len(values)]) for i in range(n)]


def test_uniform_matches_the_plan() -> None:
    assert E.uniform(6) == pytest.approx(34.6, abs=0.05)
    assert E.uniform(24) == pytest.approx(120.9, abs=0.05)
    assert E.uniform(1) == pytest.approx(6.0)


def test_curve_by_live_rows_and_depth(tmp_path: Path) -> None:
    calls = depth_run(2, (9, 12), 7000) + depth_run(4, (14, 15, 16, 17, 18, 19, 20), 9000) + \
        depth_run(1, (6,), 4000, m=33)
    path = tmp_path / 'rank0.jsonl'
    path.write_text('\n'.join(simulate(calls)) + '\n')
    whole = E.analyze([path], [{'cell': 'short-b1', 'requests': 1, 'file': 0, 'from_line': 0, 'to_line': None}])
    seg = record(record(whole['segments'])['short-b1:rank0'])
    rows = {(int(number(record(r)['m'])), int(number(record(r)['live_rows']))): record(r)
            for r in cast_list(seg['curve'])}
    # without per-depth cuts the straddling interval (1,000 calls at 2 rows + 1,000 at 4) passes for 3 rows
    assert set(rows) == {(6, 2), (6, 3), (6, 4), (33, 1)}
    res = E.analyze([path], [{'cell': 'd1', 'requests': 1, 'depth': 1, 'file': 0, 'from_line': 0, 'to_line': 3},
                             {'cell': 'd3', 'requests': 1, 'depth': 3, 'file': 0, 'from_line': 4, 'to_line': 10},
                             {'cell': 'all', 'requests': 1, 'file': 0, 'from_line': 0, 'to_line': None}])
    d1 = cast_list(record(record(res['segments'])['d1:rank0'])['curve'])
    d3 = record(record(res['segments'])['d3:rank0'])
    assert [record(x)['live_rows'] for x in d1] == [2]
    assert [record(x)['live_rows'] for x in cast_list(d3['curve'])] == [4]
    assert number(d3['off_depth_calls']) == 4000           # the one-row m=33 calls inside the d3 cut
    seg = record(record(res['segments'])['all:rank0'])
    rows = {(int(number(record(r)['m'])), int(number(record(r)['live_rows']))): record(r)
            for r in cast_list(seg['curve'])}
    rows = {**rows, (6, 2): record(d1[0]), (6, 4): record(cast_list(d3['curve'])[0])}
    two, four = rows[(6, 2)], rows[(6, 4)]
    assert abs(number(two['distinct_mean']) - 10.5) <= number(two['distinct_bound']) + 1e-9
    assert abs(number(four['distinct_mean']) - 17.0) <= number(four['distinct_bound']) + 1e-9
    assert two['depth'] == 1 and four['depth'] == 3
    assert number(four['uniform']) == pytest.approx(E.uniform(4))
    files = cast_list(res['files'])
    assert record(files[0])['status'] == 'OK'
    share = record(record(record(files[0])['bands'])['share'])
    assert number(share['33-63']) == pytest.approx(4000 / 20000)


def cast_list(value: Json) -> list[Json]:
    assert isinstance(value, list)
    return value


def test_cuts_keep_only_pairs_inside_the_cell(tmp_path: Path) -> None:
    lines = simulate(depth_run(2, (10,), 8000) + depth_run(3, (13,), 8000))
    path = tmp_path / 'r.jsonl'
    path.write_text('\n'.join(lines) + '\n')
    cuts: list[dict[str, Json]] = [{'cell': 'a', 'requests': 1, 'file': 0, 'from_line': 0, 'to_line': 4},
                                   {'cell': 'b', 'requests': 1, 'file': 0, 'from_line': 4, 'to_line': 8}]
    res = E.analyze([path], cuts)
    a = cast_list(record(record(res['segments'])['a:rank0'])['curve'])
    b = cast_list(record(record(res['segments'])['b:rank0'])['curve'])
    assert [record(x)['live_rows'] for x in a] == [2]
    assert [record(x)['live_rows'] for x in b] == [3]
    assert number(record(b[0])['calls']) == 6000            # the straddling pair (lines 4 -> 5) is not counted
    assert number(record(b[0])['distinct_mean']) == pytest.approx(13.0, abs=number(record(b[0])['distinct_bound']))


def test_two_ranks_agree(tmp_path: Path) -> None:
    lines = simulate(depth_run(2, (9, 12), 6000))
    p0, p1 = tmp_path / 'r0.jsonl', tmp_path / 'r1.jsonl'
    p0.write_text('\n'.join(lines) + '\n')
    p1.write_text('\n'.join(simulate(depth_run(2, (9, 12), 6000), pid=8)) + '\n')
    assert record(E.analyze([p0, p1])['cross_rank'])['verdict'] == 'CONSISTENT'
    p1.write_text('\n'.join(simulate(depth_run(2, (10, 12), 6000), pid=8)) + '\n')
    assert record(E.analyze([p0, p1])['cross_rank'])['verdict'] != 'CONSISTENT'


def test_empty_file_reports_no_decode_calls(tmp_path: Path) -> None:
    path = tmp_path / 'r.jsonl'
    path.write_text('')
    res = E.analyze([path])
    assert str(record(cast_list(res['files'])[0])['status']).startswith('EMPTY')


@pytest.mark.parametrize('mutate, message', [
    (lambda d: d.update(calls_by_tokens={'6': 1999}), 'multiple of 2000'),
    (lambda d: d.update(mean_distinct_experts={'6': 1.0}), 'impossible'),
    (lambda d: d.update(mean_routed_slots={'6': 99.0}), 'outside'),
    (lambda d: d.pop('pid'), 'missing'),
])
def test_malformed_lines_are_refused(mutate: object, message: str) -> None:
    line = record(json.loads(simulate(depth_run(2, (9,), 2000))[0]))
    assert callable(mutate)
    mutate(line)
    with pytest.raises(ValueError, match=message):
        E.parse([json.dumps(line)])


def test_cumulative_counts_never_fall() -> None:
    lines = simulate(depth_run(2, (9,), 4000))
    with pytest.raises(ValueError, match='went down'):
        E.parse([lines[1], lines[0]])


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert E.main(['--uniform', '6', '24']) == 0
    assert 'U(6) = 34.6' in capsys.readouterr().out
    path = tmp_path / 'r.jsonl'
    path.write_text('\n'.join(simulate(depth_run(2, (9, 12), 4000))) + '\n')
    out = tmp_path / 'o.json'
    assert E.main(['--stats', str(path), '--output', str(out)]) == 0
    assert E.main(['--stats', str(path), '--output', str(out)]) == 1
    assert record(json.loads(out.read_text()))['type'] == 'cen1-distinct-experts'
    cut = [{'cell': 'x', 'requests': 1, 'file': 0, 'from_line': 0, 'to_line': None}]
    for name, doc in (('list.json', cut), ('runner.json', {'type': 'cen1-cuts', 'cuts': cut})):
        (tmp_path / name).write_text(json.dumps(doc))
        target = tmp_path / f'o-{name}'
        assert E.main(['--stats', str(path), '--cuts', str(tmp_path / name), '--output', str(target)]) == 0
        assert set(record(record(json.loads(target.read_text()))['segments'])) == {'x:rank0'}
