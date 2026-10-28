"""WIN2 history re-scoring and archive backfill: synthetic windows plus the real 2026-10-26/27 windows when present."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from offload.launch.windows import decision_v2 as d
from offload.launch.windows import win2_history as h
from offload.prefill_maxlen.evidence import record

CAMPAIGN = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD')


def _rep(ms: float, rounds: float, acc: float, tps: float, ttft: float, max_tokens: int = 512) -> dict[str, object]:
    return {'server_decode_ms': ms, 'spec_request_rounds': rounds, 'acceptance_per_round': acc,
            'median_decode_tps': tps, 'median_ttft_s': ttft, 'streams': [{'sampling': {'max_tokens': max_tokens}}]}


def _window(tmp: Path, cand_reps: int) -> Path:
    w = tmp / 'PC1-x-W-20261027T000000Z'
    w.mkdir()
    starts = {'s1-control': {'role': 'control'}, 's2-x': {'role': 'x'}, 's3-control': {'role': 'control'}}
    (w / 'summary.json').write_text(json.dumps({'starts': starts, 'decision': {'status': 'NOT A CANDIDATE'},
                                                'experiment_commit': 'abc'}))
    for start, n, ms in (('s1-control', 3, 40.0), ('s2-x', cand_reps, 36.0), ('s3-control', 3, 40.4)):
        reps = [_rep(ms * 100 + i, 100, 1.3, 55.0, 1.0) for i in range(n)]
        (w / f'{start}-short-b1.json').write_text(json.dumps({'repetitions': reps}))
        (w / f'{start}-greedy.json').write_text('{}')
    return w


def test_extract_roles_and_metrics(tmp_path: Path) -> None:
    data = h.extract(_window(tmp_path, 3))
    assert set(data) == {'short-b1'}
    assert len(data['short-b1']['control']['ms_per_round']) == 6 and len(data['short-b1']['candidate']['ms_per_round']) == 3


def test_three_candidate_repeats_get_a_verdict(tmp_path: Path) -> None:
    r = h.reanalyze(_window(tmp_path, 3), ['short-b1'])
    assert r['win2_status'] == 'CANDIDATE' and r['historical_status'] == 'NOT A CANDIDATE'


def test_two_repeats_are_insufficient_not_a_loss(tmp_path: Path) -> None:
    r = h.reanalyze(_window(tmp_path, 2), ['short-b1'])
    assert str(r['win2_status']).startswith('INCONCLUSIVE')
    cell = record(record(r['cells'])['short-b1'])
    assert cell['verdict'] == 'INSUFFICIENT_REPEATS'


def test_backfill_appends_control_rows_only(tmp_path: Path) -> None:
    archive = tmp_path / 'archive.jsonl'
    empty = tmp_path / 'stopped'
    empty.mkdir()
    (empty / 'summary.json').write_text(json.dumps({'status': 'INCOMPLETE'}))
    n = h.backfill([_window(tmp_path, 3), empty], archive)
    rows = [json.loads(line) for line in archive.read_text().splitlines()]
    assert n == len(rows) == 6 * 4 and {r['window'] for r in rows} == {'PC1-x-W-20261027T000000Z'}
    assert {r['max_tokens'] for r in rows} == {512}
    assert d.archive_spread(archive, 'short-b1', 'ms_per_round') is None     # one window only


def test_flat_nbw_file_yields_ms_per_round(tmp_path: Path) -> None:
    w = tmp_path / 'NBW-W-20261026T000000Z'
    w.mkdir()
    (w / 'summary.json').write_text(json.dumps({'starts': {'s1-K': {'role': 'K'}}}))
    (w / 's1-K-short-b1.json').write_text(json.dumps({'server_decode_ms_per_spec_request_round': 39.1,
                                                      'spec_request_rounds': 678.0, 'acceptance_per_round': 1.21,
                                                      'streams': [{'sampling': {'max_tokens': 1500}}]}))
    assert h.extract(w)['short-b1']['control']['ms_per_round'] == [39.1]
    assert h.samples(w)[0][4] == 1500


def test_repeat_with_mixed_or_missing_max_tokens_refuses(tmp_path: Path) -> None:
    w = tmp_path / 'NBW-W-20261026T000000Z'
    w.mkdir()
    (w / 'summary.json').write_text(json.dumps({'starts': {'s1-K': {'role': 'K'}}}))
    for streams in ([{'sampling': {'max_tokens': 512}}, {'sampling': {'max_tokens': 1500}}], []):
        (w / 's1-K-32k-b2.json').write_text(json.dumps({'server_decode_ms_per_spec_request_round': 50.0,
                                                        'streams': streams}))
        with pytest.raises(ValueError, match='one recorded max_tokens'):
            h.samples(w)


def test_backfill_rerun_adds_only_missing_metrics(tmp_path: Path) -> None:
    archive = tmp_path / 'archive.jsonl'
    window = _window(tmp_path, 3)
    assert h.backfill([window], archive) == 24
    assert h.backfill([window], archive) == 0
    kept = [line for line in archive.read_text().splitlines() if json.loads(line)['metric'] != 'ttft_s']
    rechained = tmp_path / 'rechained.jsonl'
    d.archive_append(rechained, [{k: v for k, v in json.loads(line).items() if k not in ('schema', 'prev_sha256')}
                                 for line in kept])
    assert h.backfill([window], rechained) == 6
    assert len(d.archive_rows(rechained)) == 24


@pytest.mark.parametrize('name,cells', [
    ('PC1-threshold2048-W-20261027T153817Z', ['short-b1', 'mixed150k-2decode']),
    ('UP3-dual-ring-W-20261027T165101Z', ['32k-b3', '32k-b1']),
    ('NBW-W-20261026T231445Z', ['short-b1', '32k-b1'])])
def test_real_windows_were_inconclusive_not_lost_under_win2(name: str, cells: list[str]) -> None:
    window = CAMPAIGN / name
    if not (window / 'summary.json').exists():
        pytest.skip('historical window not on this host')
    r = h.reanalyze(window, cells)
    assert r['historical_status'] == 'NOT A CANDIDATE'
    assert str(r['win2_status']).startswith('INCONCLUSIVE')
