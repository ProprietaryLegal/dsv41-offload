"""CPU-only: the 0085b battery arithmetic over synthetic telemetry and docker log text."""
from __future__ import annotations

import datetime
import json

import pytest

from offload.launch.windows import decision_v2 as dv
from offload.prefill_maxlen.evidence import Json, record
from offload.upx import configs
from offload.upx import measure_0085b as m
from offload.upx import windows as w

GIB = 1024**2   # kB


def log_line(utc: float, text: str) -> str:
    stamp = datetime.datetime.fromtimestamp(utc, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')
    return f'{stamp}123Z (Worker_TP0 pid=9) INFO 10-27 20:00:00 [gpu_worker.py:1290] {text}'


def release_text(utc: float, tokens: int = 150000, mib: int = 1748) -> str:
    return log_line(utc, f'dsv41 patch 0085b: idle, after {tokens} prefill tokens released {mib} MiB of cached memory '
                         f'(reserved 9000 -> {9000 - mib} MiB) in 41.3 ms')


def make(mode: str, t0: float, *, idle_gain: int = 0, low_gain: int = 0, release_at: float | None = None,
         ttft: float = 69.0) -> tuple[m.Battery, dict[str, list[tuple[float, int]]], dict[str, str]]:
    """A at t0 (70 s), idle 20 s, B (ttft + 1 s), 15 s after. MemAvailable: 6.6 GiB at idle, dips to 6.4 GiB in B."""
    a_done = t0 + 70
    b_sent = a_done + m.IDLE_WAIT_S
    b_first = b_sent + ttft
    b_done = b_first + 1
    battery = m.Battery(mode, t0, a_done, b_sent, b_first, b_done, 149000, 149500)
    telemetry: dict[str, list[tuple[float, int]]] = {}
    logs: dict[str, str] = {}
    for k, host in enumerate(m.HOSTS):
        base = 6_600_000 + 400_000 * k
        series = []
        t = t0
        while t < b_done + m.POST_B_IDLE_S:
            if b_sent <= t < b_first:
                kb = base - 200_000 + low_gain - int(1000 * (t - b_sent))
            elif a_done + 6 <= t < b_sent:
                kb = base + idle_gain
            else:
                kb = base - 100_000
            series.append((t, kb))
            t += 1.0
        telemetry[host] = series
        lines = ['2026-10-27T20:00:00.000000000Z unrelated line']
        if mode == 'on':
            lines += [release_text(release_at if release_at is not None else a_done + 5.2), release_text(b_done + 5.1)]
        logs[host] = '\n'.join(lines)
    return battery, telemetry, logs


def result(mode: str, t0: float, *, idle_gain: int = 0, low_gain: int = 0,
           release_at: float | None = None) -> dict[str, Json]:
    battery, telemetry, logs = make(mode, t0, idle_gain=idle_gain, low_gain=low_gain, release_at=release_at)
    return m.battery_result(battery, telemetry, {h: m.parse_log(t) for h, t in logs.items()})


def test_on_battery_releases_once_in_each_idle_and_reports_both_ranks() -> None:
    r = result('on', 1_800_000_000.0, idle_gain=1_800_000, low_gain=500_000)
    hosts = record(r['hosts'])
    head = record(hosts['sparkhead'])
    assert head['releases'] == 2 and head['released_mib'] == 1748 and head['idle_kb'] == 6_600_000 + 1_800_000
    assert head['next_low_kb'] == 6_600_000 - 200_000 + 500_000 - 68_000
    assert r['next_ttft_s'] == pytest.approx(69.0)
    json.dumps(r, allow_nan=False)


def test_off_and_image_batteries_must_not_release() -> None:
    assert record(record(result('off', 1e9)['hosts'])['sparkworker-wired'])['releases'] == 0
    battery, telemetry, logs = make('off', 1e9)
    logs['sparkhead'] += '\n' + release_text(battery.a_done + 6)
    with pytest.raises(ValueError, match='mode off expects 0'):
        m.battery_result(battery, telemetry, {h: m.parse_log(t) for h, t in logs.items()})


def test_on_battery_refuses_missing_early_or_failed_releases() -> None:
    battery, telemetry, logs = make('on', 1e9)
    logs['sparkworker-wired'] = 'x'
    with pytest.raises(ValueError, match='mode on expects 1'):
        m.battery_result(battery, telemetry, {h: m.parse_log(t) for h, t in logs.items()})
    with pytest.raises(ValueError, match='inside the 5 s grace'):
        result('on', 1e9, release_at=1e9 + 70 + 2.0)
    battery, telemetry, logs = make('on', 1e9)
    logs['sparkhead'] += '\n' + log_line(battery.a_done + 5.3, "dsv41 patch 0085b: release failed: RuntimeError('x')")
    with pytest.raises(ValueError, match='release failed'):
        m.battery_result(battery, telemetry, {h: m.parse_log(t) for h, t in logs.items()})


def test_floor_breach_and_telemetry_gaps_stop() -> None:
    with pytest.raises(ValueError, match='below the 3 GiB floor'):
        result('off', 1e9, low_gain=-3_600_000)
    battery, telemetry, logs = make('off', 1e9)
    telemetry['sparkhead'] = [p for p in telemetry['sparkhead'] if not battery.a_done + 10 <= p[0] < battery.a_done + 20]
    with pytest.raises(ValueError, match='telemetry gaps'):
        m.battery_result(battery, telemetry, {h: m.parse_log(t) for h, t in logs.items()})


def test_batteries_need_threshold_sized_prompts_and_a_full_idle() -> None:
    battery, telemetry, logs = make('off', 1e9)
    parsed = {h: m.parse_log(t) for h, t in logs.items()}
    with pytest.raises(ValueError, match='threshold'):
        m.battery_result(m.Battery('off', battery.a_sent, battery.a_done, battery.b_sent, battery.b_first, battery.b_done,
                                   100000, 149000), telemetry, parsed)
    with pytest.raises(ValueError, match='idle shorter'):
        m.battery_result(m.Battery('off', battery.a_sent, battery.a_done, battery.a_done + 10, battery.b_first,
                                   battery.b_done, 149000, 149000), telemetry, parsed)


def run(gains: list[tuple[int, int]]) -> list[dict[str, Json]]:
    out = []
    t = 1_800_000_000.0
    for i, (idle, low) in enumerate(gains):
        mode = 'on' if i % 2 == 0 else 'off'
        out.append(result(mode, t, idle_gain=idle if mode == 'on' else 0, low_gain=low if mode == 'on' else 0))
        t += 400
    return out


def test_paired_series_feed_decide_window_and_a_real_gain_is_a_candidate() -> None:
    results = run([(1_800_000, 500_000), (0, 0)] * 3)
    rows = m.series(results)
    assert {(s.cell, s.metric) for s in rows} == {(c, metric) for c, ms in w.GROUPS['UPX-0085B'].metrics.items()
                                                   for metric, _ in ms}
    assert all(s.paired for s in rows)
    decision = dv.decide_window(rows, w.GROUPS['UPX-0085B'].decision_cells, groups=w.GROUPS['UPX-0085B'].groups)
    assert decision['status'] == 'CANDIDATE'
    flat = m.series(run([(0, 0), (0, 0)] * 3))
    assert dv.decide_window(flat, w.GROUPS['UPX-0085B'].decision_cells)['status'] == 'INCONCLUSIVE'


def test_pairs_need_strict_alternation() -> None:
    results = run([(1, 1), (0, 0)] * 3)
    assert len(m.pairs(results)) == 3
    with pytest.raises(ValueError, match='alternate'):
        m.pairs([results[0], results[2], results[1], results[3]])
    with pytest.raises(ValueError, match='alternate'):
        m.pairs(results[:3])


def test_summary_reports_gain_floor_margin_and_mounted_off_against_image() -> None:
    results = run([(1_800_000, 500_000), (0, 0)] * 3)
    image = [result('image', 1_900_000_000.0), result('image', 1_900_000_400.0)]
    s = m.summary(results, image)
    head = record(record(s['hosts'])['sparkhead'])
    assert record(head['idle_kb'])['gain_kb'] == pytest.approx(1_800_000)
    assert record(head['next_low_kb'])['off_minus_image_kb'] == pytest.approx(0)
    assert head['floor_margin_kb'] == 6_600_000 - 200_000 - 68_000 - m.FLOOR_KB
    json.dumps(s, allow_nan=False)


def test_schedule_alternates_and_writes_the_threshold_on_both_ranks() -> None:
    assert m.schedule(3) == [('on', 131072), ('off', 0)] * 3
    assert m.schedule(3, start_with='off')[0] == ('off', 0)
    with pytest.raises(ValueError):
        m.schedule(2)
    assert configs.release_file_command(m.THRESHOLD).endswith(configs.RELEASE_FILE_NODE)


def test_unparsed_release_line_is_loud() -> None:
    with pytest.raises(ValueError, match='unparsed'):
        m.parse_log(log_line(1e9, 'dsv41 patch 0085b: idle, released something else'))
    with pytest.raises(ValueError, match='timestamps'):
        m.parse_log('dsv41 patch 0085b: release failed: x')
