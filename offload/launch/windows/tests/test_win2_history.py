"""WIN2 history re-scoring and archive backfill: synthetic windows plus the real 2026-10-26/27 windows when present."""
from __future__ import annotations

import calendar
import json
import time
from pathlib import Path

import pytest

from offload.launch.windows import decision_v2 as d
from offload.launch.windows import win2_history as h
from offload.prefill_maxlen.evidence import number, record

CAMPAIGN = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD')
PROD = '596ba1cf4e3b11be29b41a57e7f69a5d8efa4799'


def _rep(ms: float, rounds: float, acc: float, tps: float, ttft: float, max_tokens: int = 512) -> dict[str, object]:
    return {'server_decode_ms': ms, 'spec_request_rounds': rounds, 'acceptance_per_round': acc,
            'median_decode_tps': tps, 'median_ttft_s': ttft, 'streams': [{'sampling': {'max_tokens': max_tokens}}]}


def _start_log(w: Path, serial: int, start: str, caps: tuple[int, ...] | None = (2150, 2150)) -> None:
    (w / 'cmd').mkdir(exist_ok=True)
    lines = ['spark down', 'sparkworker down']
    if caps is not None:
        lines += [f'{node} clock cap {mhz} MHz' for node, mhz in zip(('spark', 'sparkworker'), caps)]
    (w / 'cmd' / f'{serial:04d}-{start}-start.log').write_text('\n'.join([*lines, 'port 8001 restricted']) + '\n')


def _window(tmp: Path, cand_reps: int, **summary: object) -> Path:
    # PC1/UP3 layout: the production commit is under 'production'; experiment_commit must never be archived (S92 FIX 2).
    w = tmp / 'PC1-x-W-20261027T000000Z'
    w.mkdir()
    starts = {'s1-control': {'role': 'control'}, 's2-x': {'role': 'x'}, 's3-control': {'role': 'control'}}
    (w / 'summary.json').write_text(json.dumps({'starts': starts, 'decision': {'status': 'NOT A CANDIDATE'},
                                                'production': PROD, 'experiment_commit': 'e' * 40, **summary}))
    for serial, (start, n, ms) in enumerate((('s1-control', 3, 40.0), ('s2-x', cand_reps, 36.0), ('s3-control', 3, 40.4))):
        reps = [_rep(ms * 100 + i, 100, 1.3, 55.0, 1.0) for i in range(n)]
        (w / f'{start}-short-b1.json').write_text(json.dumps({'repetitions': reps}))
        (w / f'{start}-greedy.json').write_text('{}')
        _start_log(w, 30 + 60 * serial, start)
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
    assert {r['commit'] for r in rows} == {PROD} and {r['date'] for r in rows} == {'20261027'}
    assert {(r['clock_cap_mhz'], r['clock_source'], json.dumps(r['clock_caps_mhz'])) for r in rows} == {(2150, 'start-log', '[2150]')}
    assert sorted({(r['start'], r['repeat']) for r in rows}) == [(s, i) for s in ('s1-control', 's3-control') for i in range(3)]
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


# ---------------------------------------------------------------- WIN2-R2 / S92 FIX 2: provenance (S92 counterexamples, inverted)

def test_backfill_never_records_an_experiment_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    w = tmp_path / 'X-W-20261027T000000Z'
    w.mkdir()
    rep = {'server_decode_ms': 4000.0, 'spec_request_rounds': 100.0, 'streams': [{'sampling': {'max_tokens': 512}}]}
    (w / 's1-control-short-b1.json').write_text(json.dumps({'repetitions': [rep]}))
    (w / 'summary.json').write_text(json.dumps({'starts': {'s1-control': {'role': 'control'}}, 'experiment_commit': 'e' * 40}))
    with pytest.raises(ValueError, match='no production commit'):
        h.backfill([w], tmp_path / 'a.jsonl')
    assert not (tmp_path / 'a.jsonl').exists()
    (w / 'rollback.json').write_text(json.dumps({'commit': PROD, 'ranks': []}))       # PC1/UP3 rollback.json layout
    (w / 'checkpoints.jsonl').write_text(json.dumps({'time': 1793120000.0, 'start': 's1-control', 'status': 'STARTED'}) + '\n')
    (tmp_path / 'CLOCK.log').write_text(CLOCK_LOG_TEXT)
    monkeypatch.setattr(h, 'CLOCK_LOG', tmp_path / 'CLOCK.log')
    assert h.backfill([w], tmp_path / 'a.jsonl') == 1                                  # 2026-10-27T17:33Z: 2150 by CLOCK.log
    row = d.archive_rows(tmp_path / 'a.jsonl')[0]
    assert row['commit'] == PROD and (row['clock_cap_mhz'], row['clock_source']) == (2150, 'clock-log')


@pytest.mark.parametrize('summary,rollback,expected', [
    ({'production_commit': PROD, 'production': 'a' * 40}, None, PROD),
    ({'production': PROD}, {'production_commit': 'a' * 40}, PROD),
    ({}, {'production_commit': PROD, 'commit': 'a' * 40}, PROD),
    ({'production_commit': '596ba1c'}, None, None),
    ({'experiment_commit': PROD}, None, None)])
def test_production_commit_order(tmp_path: Path, summary: dict[str, object], rollback: dict[str, object] | None,
                                 expected: str | None) -> None:
    w = tmp_path / 'W'
    w.mkdir()
    (w / 'summary.json').write_text(json.dumps(summary))
    if rollback is not None:
        (w / 'rollback.json').write_text(json.dumps(rollback))
    if expected is None:
        with pytest.raises(ValueError):
            h.production_commit(w)
    else:
        assert h.production_commit(w) == expected


def test_backfill_one_call_naming_a_window_twice_appends_it_once(tmp_path: Path) -> None:
    window = _window(tmp_path, 3)
    archive = tmp_path / 'a.jsonl'
    assert h.backfill([window, window, Path(str(window))], archive) == 24 and len(d.archive_rows(archive)) == 24
    assert h.backfill([window], archive) == 0


def test_clock_caps_are_proved_per_start_from_both_nodes(tmp_path: Path) -> None:
    w = _window(tmp_path, 3)
    assert h.start_clock_cap(w, 's1-control') == (2150, 'start-log')
    assert h.start_clock_caps(w, 's1-control') == [2150] and h.clock_caps_mhz(w) == [2150] and h.clock_caps(w) == [2150]
    _start_log(w, 200, 's9-one-node', (2300,))                 # only the head printed its cap
    _start_log(w, 201, 's8-none', None)                        # GPU_CLOCK_CAP empty: nothing printed
    _start_log(w, 202, 's7-mixed', (2150, 2300))               # the Sparks disagree
    for start in ('s9-one-node', 's8-none', 's7-mixed'):
        with pytest.raises(ValueError, match='does not prove one clock cap'):
            h.start_clock_cap(w, start)
    assert h.start_clock_caps(w, 's9-one-node') == 'unknown' and h.start_clock_caps(w, 's7-mixed') == [2150, 2300]
    assert h.clock_caps_mhz(w) == 'unknown' and h.clock_caps(w) == []
    assert h.clock_caps_mhz(w, ['s1-control', 's3-control']) == [2150]
    (w / 'cmd' / 'restore-0100-start.log').write_text('spark clock cap 2300 MHz\nsparkworker clock cap 2300 MHz\n')
    assert h.start_clock_cap(w, 's1-control') == (2150, 'start-log')        # restore logs are not start logs


def _utc(text: str) -> float:
    return float(calendar.timegm(time.strptime(text, '%Y-%m-%dT%H:%M:%SZ')))


CLOCK_LOG_TEXT = '''sparkhead clock cap 2300 MHz 2026-10-27T13:27:18Z
2125 MHz, 2125 MHz, 50, 9.30 W, 0x0000000000000000
sparkworker clock cap 2300 MHz 2026-10-27T13:27:19Z
sparkhead clock cap 2150 MHz 2026-10-27T13:36:27Z
sparkworker clock cap 2150 MHz 2026-10-27T13:36:28Z
sparkhead clock cap 2300 MHz 2026-10-27T19:23:25Z env=2300
sparkworker clock cap 2300 MHz 2026-10-27T19:23:27Z env=2300
'''


@pytest.mark.parametrize('t0,t1,expected', [
    ('2026-10-26T22:00:00Z', '2026-10-26T23:00:00Z', 2150), ('2026-10-27T13:28:00Z', '2026-10-27T13:35:00Z', 2300),
    ('2026-10-27T18:13:14Z', '2026-10-27T19:23:20Z', 2150), ('2026-10-27T19:23:39Z', '2026-10-27T20:30:00Z', 2300),
    ('2026-10-27T13:20:00Z', '2026-10-27T13:30:00Z', None), ('2026-10-27T19:00:00Z', '2026-10-27T19:23:26Z', None),
    ('2026-10-27T13:27:18Z', '2026-10-27T13:27:18Z', None)])
def test_clock_log_classifies_starts_and_refuses_straddles(tmp_path: Path, t0: str, t1: str, expected: int | None) -> None:
    """S92 FIX 2 item 7: 2150, except 2300 from 13:27:18 to 13:36:27Z and from 19:23:25Z on; a straddle refuses."""
    log = tmp_path / 'CLOCK.log'
    log.write_text(CLOCK_LOG_TEXT)
    assert [c for _, _, c in h.clock_changes(log)] == [2300, 2150, 2300]
    if expected is None:
        with pytest.raises(ValueError, match='straddles'):
            h.clock_log_cap(_utc(t0), _utc(t1), log)
    else:
        assert h.clock_log_cap(_utc(t0), _utc(t1), log) == expected


def test_start_without_log_uses_clock_log_and_checkpoint_times(tmp_path: Path) -> None:
    w = _window(tmp_path, 3)
    log = tmp_path / 'CLOCK.log'
    log.write_text(CLOCK_LOG_TEXT)
    with (w / 'checkpoints.jsonl').open('w') as handle:
        for when, start in (('2026-10-27T19:30:00Z', 's4-late'), ('2026-10-27T19:40:00Z', 's4-late'),
                            ('2026-10-27T19:20:00Z', 's5-straddle'), ('2026-10-27T19:25:00Z', 's5-straddle')):
            handle.write(json.dumps({'time': _utc(when), 'arm': start, 'status': 'STARTED'}) + '\n')
    assert h.start_clock_cap(w, 's4-late', clock_log=log) == (2300, 'clock-log')
    with pytest.raises(ValueError, match='straddles'):
        h.start_clock_cap(w, 's5-straddle', clock_log=log)
    with pytest.raises(ValueError, match='no checkpoint times'):
        h.start_clock_cap(w, 's6-unknown', clock_log=log)


def test_live_clock_log_timeline() -> None:
    if not h.CLOCK_LOG.exists():
        pytest.skip('campaign CLOCK.log not on this host')
    changes = h.clock_changes()
    assert (changes[0][0], changes[0][2]) == (_utc('2026-10-27T13:27:18Z'), 2300)
    assert (changes[2][0], changes[2][2]) == (_utc('2026-10-27T19:23:25Z'), 2300)


def test_window_date_from_name_only(tmp_path: Path) -> None:
    assert h.window_date(Path('/c/PC1-threshold512-W-20261027T181314Z')) == '20261027'
    with pytest.raises(ValueError):
        h.window_date(Path('/c/CHAIN-ABORT-20261027T181314Z'))


REAL = ('NBW-W-20261026T221846Z', 'NBW-W-20261026T224001Z', 'NBW-W-20261026T231445Z', 'DEP1-W-20261027T020407Z',
        'UP3-dual-ring-W-20261027T051604Z', 'UP3-k5-W-20261027T063155Z', 'PC1-chunk2048-W-20261027T100908Z',
        'PC1-chunk2048-W-20261027T112828Z', 'PC1-threshold2048-W-20261027T153817Z', 'UP3-dual-ring-W-20261027T165101Z',
        'PC1-threshold512-W-20261027T181314Z')


@pytest.mark.parametrize('name', REAL)
def test_real_windows_prove_production_commit_and_2150_mhz(name: str, tmp_path: Path) -> None:
    """The control archive's windows: 596ba1c everywhere (PC1/UP3 under 'production'), every measured start at 2150 MHz
    from its own start log (production went to 2300 at ~19:23Z, after PC1-threshold512's last start)."""
    window = CAMPAIGN / name
    if not (window / 'summary.json').exists():
        pytest.skip('historical window not on this host')
    assert h.production_commit(window) == PROD
    starts = {x.start for x in h.repeat_samples(window)}
    assert starts and all(h.start_clock_cap(window, s) == (2150, 'start-log') for s in starts)
    archive = tmp_path / 'a.jsonl'
    n = h.backfill([window, window], archive)
    rows = d.archive_rows(archive)
    assert n == len(rows) > 0 and {r['commit'] for r in rows} == {PROD}
    assert len({(r['start'], r['repeat'], r['cell'], r['metric']) for r in rows}) == len(rows)


# ---------------------------------------------------------------- WIN2-R2 / S92 FIX 3: reanalyze uses decide_window's cell rule

def test_reanalyze_agrees_with_decide_window_on_a_guard_loss(tmp_path: Path) -> None:
    """S92: reanalyze judged the primary only and said CANDIDATE where decide_window said NOT A CANDIDATE."""
    w = tmp_path / 'Y-W-20261027T000000Z'
    w.mkdir()

    def rep(ms: float, acc: float) -> dict[str, object]:
        return {'server_decode_ms': ms * 100, 'spec_request_rounds': 100.0, 'acceptance_per_round': acc,
                'streams': [{'sampling': {'max_tokens': 512}}]}
    ctl = [rep(40.0, 1.50), rep(40.2, 1.52), rep(40.1, 1.51)]
    cand = [rep(38.0, 1.20), rep(38.1, 1.22), rep(38.2, 1.21)]
    (w / 's1-control-short-b1.json').write_text(json.dumps({'repetitions': ctl}))
    (w / 's2-x-short-b1.json').write_text(json.dumps({'repetitions': cand}))
    (w / 'summary.json').write_text(json.dumps({'starts': {'s1-control': {'role': 'control'}, 's2-x': {'role': 'x'}}}))
    old = h.reanalyze(w, ['short-b1'])
    data = h.extract(w)['short-b1']
    series = [d.Series('short-b1', m, hib, tuple(data['control'][m]), tuple(data['candidate'][m]), paired=False)
              for m, hib in (('ms_per_round', False), ('accepted_per_round', True))]
    new = d.decide_window(series, ['short-b1'])
    assert old['win2_status'] == new['status'] == 'NOT A CANDIDATE'
    cell = record(record(old['cells'])['short-b1'])
    assert cell['verdict'] == 'LOSS' and record(record(cell['metrics'])['accepted_per_round'])['verdict'] == 'LOSS'
    assert record(record(cell['metrics'])['ms_per_round'])['verdict'] == 'WIN'


def test_threshold2048_short_acceptance_stays_loss_with_a_third_repeat() -> None:
    """S92 guard_cx: the real short-b1 repeats plus a third candidate repeat at the candidate's own mean. The guard rule
    still catches consistent, significant harm: acceptance -28 % [-47, -9] is LOSS although ms/round WINS (+3.4 %)."""
    window = CAMPAIGN / 'PC1-threshold2048-W-20261027T153817Z'
    if not (window / 'summary.json').exists():
        pytest.skip('historical window not on this host')
    data = h.extract(window)['short-b1']
    series = []
    for metric, hib in (('ms_per_round', False), ('accepted_per_round', True), ('decode_tps', True)):
        cand = list(data['candidate'][metric])
        series.append(d.Series('short-b1', metric, hib, tuple(data['control'][metric]),
                               (*cand, sum(cand) / len(cand)), paired=False))
    r = d.decide_window(series, ['short-b1'])
    metrics = record(record(record(r['cells'])['short-b1'])['metrics'])
    acc = record(metrics['accepted_per_round'])
    interval = acc['interval95']
    assert isinstance(interval, list) and -0.48 < number(interval[0]) < -0.46 and -0.10 < number(interval[1]) < -0.09
    assert acc['verdict'] == 'LOSS' and record(metrics['ms_per_round'])['verdict'] == 'WIN'
    assert r['status'] == 'NOT A CANDIDATE'


def test_first_2300_window_proves_2300_from_its_start_log() -> None:
    window = CAMPAIGN / 'PP4-941-admission-W-20261027T192339Z'
    if not (window / 'cmd').is_dir():
        pytest.skip('historical window not on this host')
    assert h.start_clock_cap(window, 'admission-B') == (2300, 'start-log')


# ---------------------------------------------------------------- WIN2-R2 runner requests: reanalyze arms and roles

def _arms_window(tmp: Path, roles: dict[str, tuple[str, float]], reps: int = 3) -> Path:
    w = tmp / 'UP3-two-arm-W-20261027T210000Z'
    w.mkdir()
    for start, (_, ms) in roles.items():
        rows = [_rep(ms * 100 + i * 10, 100, 1.3, 55.0, 1.0) for i in range(reps)]
        (w / f'{start}-short-b1.json').write_text(json.dumps({'repetitions': rows}))
    (w / 'summary.json').write_text(json.dumps({'starts': {s: {'role': r} for s, (r, _) in roles.items()}, 'production': PROD}))
    return w


def test_reanalyze_refuses_more_than_two_decision_cells(tmp_path: Path) -> None:
    w = _window(tmp_path, 3)
    for cells in ([], ['a', 'b', 'c'], ['short-b1', 'short-b1']):
        with pytest.raises(ValueError, match='decision cells'):
            h.reanalyze(w, cells)


def test_reanalyze_keeps_candidate_arms_apart(tmp_path: Path) -> None:
    """R-UP3 runs dual ring and DEP1 fixed2 as two arms in one window: never pool them into one candidate."""
    w = _arms_window(tmp_path, {'s1-control': ('control', 40.0), 's2-dual-ring': ('dual-ring', 36.0),
                                's3-fixed2': ('fixed2', 44.0), 's4-control': ('control', 40.2)})
    r = h.reanalyze(w, ['short-b1'])
    arms = record(r['arms'])
    assert set(arms) == {'dual-ring', 'fixed2'} and 'cells' not in r
    assert record(arms['dual-ring'])['win2_status'] == 'CANDIDATE'
    assert record(arms['fixed2'])['win2_status'] == 'NOT A CANDIDATE'
    assert r['win2_status'] == 'PER ARM: dual-ring=CANDIDATE; fixed2=NOT A CANDIDATE'
    with pytest.raises(ValueError, match='must not be pooled'):
        h.extract(w)
    assert set(h.extract_arms(w)['short-b1']) == {'control', 'dual-ring', 'fixed2'}


@pytest.mark.parametrize('control,candidate', [('off', 'compact'), ('H', 'L'), ('K', 'I'), ('control', 'k5')])
def test_control_roles_include_tv1_off_and_pf2_h(tmp_path: Path, control: str, candidate: str) -> None:
    w = _arms_window(tmp_path, {f's1-{control}': (control, 40.0), f's2-{candidate}': (candidate, 36.0),
                                f's3-{control}': (control, 40.2)})
    r = h.reanalyze(w, ['short-b1'])
    assert r['arm'] == candidate and r['win2_status'] == 'CANDIDATE'
    assert {x.arm for x in h.repeat_samples(w) if x.role == control} == {'control'}
