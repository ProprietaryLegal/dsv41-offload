"""0085b memory measurement arithmetic (R-UPX item 3; design in MEASURE-0085B.md). Pure functions over what a runner
already records: the 1 s Spark telemetry rows ({'kind': 'memory', 'utc', 'kb': {'MemAvailable', ...}}, pf1_window
TELEMETRY), `docker logs --timestamps` text of each rank's container, and the battery's request times (wall clock).

A battery is two cold 150K B1 requests, A then B, with IDLE_WAIT_S of idle between them and POST_B_IDLE_S after B. In
the candidate start the threshold file alternates: ON (131,072: each rank releases once in A's idle and once after B)
and OFF (0: no release). Per rank it yields
  idle_kb      median MemAvailable over [A done + IDLE_WINDOW[0], A done + IDLE_WINDOW[1]): after the release (ON) or not
  next_low_kb  minimum MemAvailable while B prefills ([B sent, B first token]): the 16K-chunk floor question
and next_ttft_s (B's first token, a guard: the released cache is re-mapped during B). ON_i/OFF_i are adjacent pairs, so
decision_v2 runs paired (WIN2-R2: true adjacent pairs only). The control starts' 'image' batteries (no overlay) show the
mounted-but-off overlay equals the image (reported).
"""
from __future__ import annotations

import datetime
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from offload.launch.windows import decision_v2 as dv
from offload.prefill_maxlen.evidence import Json, number, record
from offload.upx import configs

FLOOR_KB = 3 * 1024**2      # NDM 2026-10-27 "Put safety floor to 3gb" (PC1 headroom.FLOOR_KB, stricter GiB reading)
THRESHOLD = configs.DEFAULT_THRESHOLD
IDLE_WAIT_S = 20.0          # A done -> B sent: 5 s grace + the release (MEASURED upstream 1,748 MiB per rank) + settle
IDLE_WINDOW = (10.0, 20.0)  # the idle sample window after A done (ten 1 s samples, all after the 5 s grace)
POST_B_IDLE_S = 15.0        # after B: ON releases again, so the next battery starts from the same state either way
MIN_IDLE_SAMPLES = 8
HOSTS = ('sparkhead', 'sparkworker-wired')                     # rank 0 (head, engine core), rank 1
CELL_OF = {'sparkhead': 'mem-r0', 'sparkworker-wired': 'mem-r1'}
RELEASE = re.compile(r'dsv41 patch 0085b: idle, after (\d+) prefill tokens released (-?\d+) MiB of cached memory '
                     r'\(reserved (\d+) -> (\d+) MiB\) in ([\d.]+) ms')
FAILED = 'dsv41 patch 0085b: release failed:'
STAMP = re.compile(r'^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?Z ')


@dataclass(frozen=True)
class Release:
    utc: float
    tokens: int
    released_mib: int
    reserved_before_mib: int
    reserved_after_mib: int
    ms: float


def stamp(line: str) -> float:
    """Epoch seconds of a `docker logs --timestamps` line (RFC 3339, UTC, up to nanoseconds)."""
    m = STAMP.match(line)
    if m is None:
        raise ValueError('log line without a docker --timestamps prefix')
    whole = datetime.datetime.strptime(m.group(1), '%Y-%m-%dT%H:%M:%S').replace(tzinfo=datetime.timezone.utc)
    return whole.timestamp() + (float('0.' + m.group(2)) if m.group(2) else 0.0)


def parse_log(text: str) -> tuple[list[Release], list[float]]:
    """0085b release lines and failure lines of one rank's container log."""
    releases: list[Release] = []
    failures: list[float] = []
    for line in text.splitlines():
        if FAILED in line:
            failures.append(stamp(line))
        elif 'dsv41 patch 0085b: idle' in line:
            m = RELEASE.search(line)
            if m is None:
                raise ValueError(f'unparsed 0085b release line: {line[:160]!r}')
            releases.append(Release(stamp(line), int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)),
                                    float(m.group(5))))
    return releases, failures


def memory_series(rows: Iterable[Mapping[str, Json]]) -> list[tuple[float, int]]:
    out = [(number(r['utc']), int(number(record(r['kb'])['MemAvailable']))) for r in rows if r.get('kind') == 'memory']
    if any(b[0] <= a[0] for a, b in zip(out, out[1:])):
        raise ValueError('telemetry timestamps are not increasing')
    return out


def window(series: Sequence[tuple[float, int]], start: float, end: float) -> list[int]:
    return [kb for t, kb in series if start <= t < end]


@dataclass(frozen=True)
class Battery:
    mode: str                   # 'on' | 'off' (candidate start, threshold file) | 'image' (control start, no overlay)
    a_sent: float
    a_done: float
    b_sent: float
    b_first: float
    b_done: float
    a_prompt_tokens: int
    b_prompt_tokens: int


def battery_result(b: Battery, telemetry: Mapping[str, Sequence[tuple[float, int]]],
                   logs: Mapping[str, tuple[Sequence[Release], Sequence[float]]]) -> dict[str, Json]:
    """One battery, both ranks. Refuses (ValueError) on any broken registration; a floor breach is a STOP."""
    if b.mode not in ('on', 'off', 'image'):
        raise ValueError(f'unknown battery mode {b.mode!r}')
    if set(telemetry) != set(HOSTS) or set(logs) != set(HOSTS):
        raise ValueError(f'battery needs telemetry and logs of {HOSTS}')
    if not (b.a_sent < b.a_done <= b.b_sent - IDLE_WAIT_S and b.b_sent < b.b_first <= b.b_done):
        raise ValueError('battery times out of order or idle shorter than IDLE_WAIT_S')
    if min(b.a_prompt_tokens, b.b_prompt_tokens) < THRESHOLD:
        raise ValueError(f'battery prompts must reach the {THRESHOLD}-token threshold each')
    per_host: dict[str, Json] = {}
    for host in HOSTS:
        releases, failures = logs[host]
        if any(b.a_sent <= t <= b.b_done + POST_B_IDLE_S for t in failures):
            raise ValueError(f'STOP: {host}: 0085b release failed during the battery')
        idle_releases = [r for r in releases if b.a_done <= r.utc < b.b_sent]
        after_b = [r for r in releases if b.b_done <= r.utc < b.b_done + POST_B_IDLE_S]
        during = [r for r in releases if b.a_sent <= r.utc < b.a_done or b.b_sent <= r.utc < b.b_done]
        expected = 1 if b.mode == 'on' else 0
        if len(idle_releases) != expected or len(after_b) != expected or during:
            raise ValueError(f'STOP: {host}: {len(idle_releases)} release(s) in the idle after A, {len(after_b)} after B, '
                             f'{len(during)} while a request ran; mode {b.mode} expects {expected}, {expected}, 0')
        if b.mode == 'on' and idle_releases[0].utc - b.a_done < 5.0:
            raise ValueError(f'STOP: {host}: released {idle_releases[0].utc - b.a_done:.1f} s after A, inside the 5 s grace')
        if b.mode == 'on' and idle_releases[0].tokens < THRESHOLD:
            raise ValueError(f'STOP: {host}: released after {idle_releases[0].tokens} tokens, below the threshold')
        idle = window(telemetry[host], b.a_done + IDLE_WINDOW[0], b.a_done + IDLE_WINDOW[1])
        low = window(telemetry[host], b.b_sent, b.b_first)
        if len(idle) < MIN_IDLE_SAMPLES or len(low) < max(1, int((b.b_first - b.b_sent) / 2)):
            raise ValueError(f'{host}: telemetry gaps ({len(idle)} idle samples, {len(low)} during B)')
        if min(low) < FLOOR_KB:
            raise ValueError(f'STOP: {host}: MemAvailable {min(low)} kB below the 3 GiB floor during B ({b.mode})')
        per_host[host] = {'idle_kb': int(statistics.median(idle)), 'next_low_kb': min(low),
                          'releases': len(idle_releases) + len(after_b),
                          'released_mib': idle_releases[0].released_mib if idle_releases else 0,
                          'release_ms': idle_releases[0].ms if idle_releases else None}
    return {'mode': b.mode, 'hosts': per_host, 'next_ttft_s': b.b_first - b.b_sent}


def pairs(results: Sequence[Mapping[str, Json]]) -> list[tuple[Mapping[str, Json], Mapping[str, Json]]]:
    """(ON, OFF) adjacent pairs in run order; the candidate start alternates strictly (either mode first)."""
    modes = [str(r['mode']) for r in results]
    if len(modes) % 2 or any(m not in ('on', 'off') for m in modes) or any(a == b for a, b in zip(modes, modes[1:])):
        raise ValueError(f'battery modes {modes} do not alternate ON/OFF in complete pairs')
    out: list[tuple[Mapping[str, Json], Mapping[str, Json]]] = []
    for first, second in zip(results[::2], results[1::2]):
        out.append((first, second) if first['mode'] == 'on' else (second, first))
    return out


def series(results: Sequence[Mapping[str, Json]]) -> list[dv.Series]:
    """decision_v2 series for the UPX-0085B decision cells (windows.GROUPS['UPX-0085B'].metrics), paired ON vs OFF."""
    matched = pairs(results)

    def values(metric: str, host: str | None) -> tuple[tuple[float, ...], tuple[float, ...]]:
        def get(r: Mapping[str, Json]) -> float:
            return number(r[metric] if host is None else record(record(r['hosts'])[host])[metric])
        return tuple(get(off) for _, off in matched), tuple(get(on) for on, _ in matched)

    out: list[dv.Series] = []
    for host in HOSTS:
        cell = CELL_OF[host]
        for metric in ('next_low_kb', 'idle_kb'):
            control, candidate = values(metric, host)
            out.append(dv.Series(cell, metric, True, control, candidate, paired=True))
        if host == HOSTS[0]:
            control, candidate = values('next_ttft_s', None)
            out.append(dv.Series(cell, 'next_ttft_s', False, control, candidate, paired=True))
    return out


def summary(results: Sequence[Mapping[str, Json]], image: Sequence[Mapping[str, Json]]) -> dict[str, Json]:
    """Means per mode and host, the floor margins, and mounted-off (OFF) against the image's own batteries (reported)."""
    def mean(rows: Sequence[Mapping[str, Json]], host: str, metric: str) -> float | None:
        vals = [number(record(record(r['hosts'])[host])[metric]) for r in rows]
        return statistics.fmean(vals) if vals else None

    by_mode = {m: [r for r in results if r['mode'] == m] for m in ('on', 'off')}
    hosts: dict[str, Json] = {}
    for host in HOSTS:
        row: dict[str, Json] = {}
        for metric in ('idle_kb', 'next_low_kb'):
            on, off, img = mean(by_mode['on'], host, metric), mean(by_mode['off'], host, metric), mean(image, host, metric)
            row[metric] = {'on': on, 'off': off, 'image': img,
                           'gain_kb': None if on is None or off is None else on - off,
                           'off_minus_image_kb': None if off is None or img is None else off - img}
        lows = [number(record(record(r['hosts'])[host])['next_low_kb']) for r in results]
        row['floor_margin_kb'] = min(lows) - FLOOR_KB if lows else None
        row['released_mib'] = cast(Json, [record(record(r['hosts'])[host])['released_mib'] for r in by_mode['on']])
        hosts[host] = row
    return {'hosts': hosts, 'floor_kb': FLOOR_KB, 'threshold': THRESHOLD,
            'pairs': len(results) // 2, 'image_batteries': len(image)}


def schedule(pairs_wanted: int, *, start_with: str = 'on') -> list[tuple[str, int]]:
    """The candidate start's battery order and the threshold the runner writes on BOTH ranks before each battery
    (configs.release_file_command)."""
    if pairs_wanted < dv.MIN_REPEATS or start_with not in ('on', 'off'):
        raise ValueError(f'at least {dv.MIN_REPEATS} ON/OFF pairs, starting with on or off')
    other = 'off' if start_with == 'on' else 'on'
    order = [start_with, other] * pairs_wanted
    return [(m, THRESHOLD if m == 'on' else 0) for m in order]
