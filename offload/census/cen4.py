"""CEN4: rank-imbalance attribution from a CEN3 window (CPU, Controller), F-CENSUS.plan.md §3 and CEN3-BYTES-AND-WATERFALL.md.

  PYTHONPATH=. python3 -m offload.census.cen4 --window <CEN3-W-dir> --output <json>

Imbalance: per verify-round capture of part (b), each rank's rank_wait per round (offload.census.waterfall); the rank
that waits less is the slower one. Causes, each judged against a rule PRE-REGISTERED here before any CEN3 data
(lane CEN-RUN, 2026-10-27) and fired only when it points at the slower rank:
  clock      median graphics clock while busy (telemetry, utilization >= 50 %) lower by > CLOCK_REL on the slower rank
  throttle   power/thermal clock-event reasons (SW power cap, HW slowdown, SW/HW thermal, power brake) in the slower
             rank's telemetry and not in the other's
  thermal    THERMAL.log hw/sw thermal slowdown Active on the slower Spark only, inside the window
  bandwidth  part (c) read or copy median rate lower by > BW_REL on the slower rank
  kernel     part (a) identical-work kernel time (median over a family's launches) longer by > KERNEL_REL
  firmware   not measured here: host_check enforces the same kernel/driver on both Sparks; FW3/FW3b state is in the
             coordinator ledger
|imbalance| < IMBALANCE_MS a round -> SYMMETRIC. A start off the registered 2300 MHz cap -> INCONCLUSIVE for timing. The
comparison window SP1-W-20261026T191247Z (2150 MHz, before FW3) gives the earlier imbalance; |change| >= IMBALANCE_MS
says that FW3/FW3b and 2300 MHz together moved it (the two cannot be separated with these two windows).
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from offload.census import cen3_analyze as A
from offload.census import waterfall
from offload.launch.windows import win2_history as wh
from offload.prefill_maxlen.evidence import Json, number, record

CAMPAIGN = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD')
THERMAL_LOG = CAMPAIGN / 'scratch' / 'THERMAL.log'
SP1_WINDOW = CAMPAIGN / 'SP1-W-20261026T191247Z'
REGISTERED_MHZ = 2300
IMBALANCE_MS = 0.5
CLOCK_REL = 0.02
BW_REL = 0.03
KERNEL_REL = 0.03
BUSY_UTIL = 50.0
THROTTLE_BITS = 0x4 | 0x8 | 0x20 | 0x40 | 0x80
THERMAL_HOST = {'sparkhead': 'sparkhead', 'sparkworker': 'sparkworker-wired'}
THERMAL_LINE = re.compile(r'^(\S+Z) (\S+) (\d+), (\d+), ([\d.]+), (Active|Not Active), (Active|Not Active)\s*$')


def gpu_rows(path: Path) -> list[dict[str, float]]:
    """Telemetry 'gpu' samples: utc, clock MHz, power W, temperature C, utilization %, event-reason bits."""
    out: list[dict[str, float]] = []
    for line in path.read_text(errors='replace').splitlines():
        if not line.startswith('{'):
            continue
        row = record(cast(Json, json.loads(line)))
        if row.get('kind') != 'gpu':
            continue
        f = [x.strip() for x in str(row['csv']).split(',')]
        if len(f) != 8 or not f[1].isdigit():
            continue

        def num(x: str) -> float:
            return float(x) if re.fullmatch(r'[\d.]+', x) else float('nan')
        out.append({'utc': number(row['utc']), 'clock': float(f[1]), 'power': num(f[3]), 'temp': num(f[5]),
                    'util': num(f[6]), 'reasons': float(int(f[7], 16)) if f[7].startswith('0x') else float('nan')})
    return out


def clocks(rows: Sequence[Mapping[str, float]]) -> dict[str, Json]:
    busy = [r for r in rows if r['util'] >= BUSY_UTIL]
    throttle = [r for r in rows if r['reasons'] == r['reasons'] and int(r['reasons']) & THROTTLE_BITS]
    return {'samples': len(rows), 'busy_samples': len(busy),
            'busy_clock_median': statistics.median(r['clock'] for r in busy) if busy else None,
            'clock_max': max((r['clock'] for r in rows), default=None),
            'busy_power_median': statistics.median(r['power'] for r in busy if r['power'] == r['power'])
            if any(r['power'] == r['power'] for r in busy) else None,
            'temp_max': max((r['temp'] for r in rows if r['temp'] == r['temp']), default=None),
            'throttle_samples': len(throttle),
            'throttle_reasons': cast(Json, sorted({hex(int(r['reasons']) & THROTTLE_BITS) for r in throttle}))}


def thermal(text: str, t0: float, t1: float) -> dict[str, Json]:
    """THERMAL.log lines inside [t0, t1] by Spark: samples, slowdown-active samples, max temperature."""
    out: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        m = THERMAL_LINE.match(line)
        if m is None:
            continue
        t = datetime.datetime.fromisoformat(m.group(1).replace('Z', '+00:00')).timestamp()
        if not t0 <= t <= t1:
            continue
        host = THERMAL_HOST.get(m.group(2), m.group(2))
        row = out.setdefault(host, {'samples': 0, 'slowdown_active': 0, 'temp_max': 0.0})
        row['samples'] += 1
        row['slowdown_active'] += int('Active' in (m.group(6), m.group(7)))
        row['temp_max'] = max(row['temp_max'], float(m.group(3)))
    return cast(dict[str, Json], out)


def imbalance(captures: Mapping[str, Json]) -> dict[str, Json]:
    """rank_wait per round of each rank per verify capture (target-only excluded: one row, no drafter); positive
    `rank0_minus_rank1_ms` means rank 0 waits longer, i.e. rank 1 is the slower rank."""
    rows: dict[str, Json] = {}
    diffs: list[float] = []
    for label, cap in captures.items():
        c = record(cap)
        if 'rank_ms' not in c or 'target-only' in label:
            continue
        r = record(c['rank_ms'])
        w0, w1 = number(record(r['0'])['rank_wait']), number(record(r['1'])['rank_wait'])
        rows[label] = {'rank0_wait_ms': w0, 'rank1_wait_ms': w1, 'rank0_minus_rank1_ms': w0 - w1}
        diffs.append(w0 - w1)
    mean = statistics.fmean(diffs) if diffs else None
    slower = None if mean is None or abs(mean) < IMBALANCE_MS else (1 if mean > 0 else 0)
    return {'captures': rows, 'mean_rank0_minus_rank1_ms': mean, 'slower_rank': slower}


def _gap(a: float, b: float) -> float:
    return (a - b) / max(a, b)


def causes(slower: int, clk: Sequence[Mapping[str, Json]], therm: Mapping[str, Json],
           bw: Sequence[Mapping[str, Json] | None], times: Sequence[Mapping[str, Json]]) -> dict[str, Json]:
    """Each pre-registered cause with its numbers and whether it fires for the slower rank."""
    fast = 1 - slower
    out: dict[str, Json] = {}
    cs, cf = clk[slower].get('busy_clock_median'), clk[fast].get('busy_clock_median')
    out['clock'] = ({'fires': _gap(number(cf), number(cs)) > CLOCK_REL, 'slower_mhz': cs, 'other_mhz': cf}
                    if cs is not None and cf is not None else {'fires': False, 'note': 'no busy telemetry'})
    out['throttle'] = {'fires': number(clk[slower]['throttle_samples']) > 0 and number(clk[fast]['throttle_samples']) == 0,
                       'slower_samples': clk[slower]['throttle_samples'], 'other_samples': clk[fast]['throttle_samples']}
    hs, hf = record(therm.get(A.HOSTS[slower]) or {}), record(therm.get(A.HOSTS[fast]) or {})
    out['thermal'] = {'fires': number(hs.get('slowdown_active', 0)) > 0 and number(hf.get('slowdown_active', 0)) == 0,
                      'slower_active': hs.get('slowdown_active', 0), 'other_active': hf.get('slowdown_active', 0)}
    bs, bf = bw[slower], bw[fast]
    if bs is not None and bf is not None:
        gaps = {op: _gap(number(record(bf[op])['gbps_median']), number(record(bs[op])['gbps_median']))
                for op in ('read', 'copy')}
        out['bandwidth'] = {'fires': max(gaps.values()) > BW_REL, 'relative_gap': cast(Json, gaps)}
    else:
        out['bandwidth'] = {'fires': False, 'note': 'part (c) not measured'}
    common = sorted(set(times[0]) & set(times[1]))
    if common:
        gaps = {k: _gap(number(record(times[slower][k])['median_us']), number(record(times[fast][k])['median_us']))
                for k in common}
        out['kernel'] = {'fires': statistics.median(gaps.values()) > KERNEL_REL, 'relative_gap': cast(Json, gaps)}
    else:
        out['kernel'] = {'fires': False, 'note': 'part (a) kernel times not measured on both Sparks'}
    out['firmware'] = {'fires': False, 'note': 'not measured here: same kernel/driver enforced by host_check; FW3/FW3b '
                                               'state from the coordinator ledger'}
    return out


def analyze(out: Path, *, thermal_log: Path = THERMAL_LOG, sp1: Path | None = SP1_WINDOW,
            analysis: Mapping[str, Json] | None = None) -> dict[str, Json]:
    res = dict(analysis) if analysis is not None else A.analyze(out)
    parts = record(res['parts'])
    b = record(parts.get('b') or {})
    caps = record(b.get('captures') or {})
    imb = imbalance(caps)
    rows = [gpu_rows(out / f'telemetry-{h}.jsonl') if (out / f'telemetry-{h}.jsonl').exists() else [] for h in A.HOSTS]
    clk = [clocks(r) for r in rows]
    stamps = [r['utc'] for rr in rows for r in rr]
    therm = thermal(thermal_log.read_text(errors='replace'), min(stamps), max(stamps)) \
        if stamps and thermal_log.exists() else {}
    a = record(parts.get('a') or {})
    times = [record(record(a.get('kernel_time') or {}).get(h) or {}) for h in A.HOSTS]
    c = record(parts.get('c') or {})
    bw = [record(record(c['bandwidth'])[h]) if 'bandwidth' in c else None for h in A.HOSTS]
    try:
        cap, source = wh.start_clock_cap(out, 'cen3')
        clock: dict[str, Json] = {'start_cap_mhz': cap, 'source': source}
    except (ValueError, OSError) as exc:
        clock = {'start_cap_mhz': None, 'source': f'unproved: {exc}'}
    result: dict[str, Json] = {'type': 'cen4-rank-imbalance', 'rules': {
        'imbalance_ms': IMBALANCE_MS, 'clock_rel': CLOCK_REL, 'bw_rel': BW_REL, 'kernel_rel': KERNEL_REL,
        'busy_util_pct': BUSY_UTIL, 'throttle_bits': hex(THROTTLE_BITS), 'registered_mhz': REGISTERED_MHZ},
        'clock': clock, 'imbalance': imb, 'telemetry': {h: clk[i] for i, h in enumerate(A.HOSTS)},
        'thermal_log': therm}
    if clock['start_cap_mhz'] != REGISTERED_MHZ:
        result['verdict'] = f'INCONCLUSIVE: start cap {clock["start_cap_mhz"]} MHz, registered {REGISTERED_MHZ}'
    elif imb['slower_rank'] is None:
        result['verdict'] = ('NO DATA: no verify-round capture' if imb['mean_rank0_minus_rank1_ms'] is None else
                             f'SYMMETRIC: |mean imbalance| < {IMBALANCE_MS} ms a round; nothing to attribute')
    else:
        slower = int(number(imb['slower_rank']))
        found = causes(slower, clk, therm, bw, times)
        result['causes'] = found
        fired = [k for k, v in found.items() if record(v).get('fires') is True]
        result['verdict'] = (f'ATTRIBUTED (rank {slower} = {A.HOSTS[slower]} slower): ' + ', '.join(fired) if fired else
                             f'UNATTRIBUTED: rank {slower} = {A.HOSTS[slower]} slower by '
                             f'{abs(number(imb["mean_rank0_minus_rank1_ms"])):.2f} ms a round; no measured cause fires')
    if sp1 is not None and (sp1 / 'traces-manifest.json').exists():
        then = imbalance(record(waterfall.window(sp1)['captures']))
        now_mean, then_mean = imb['mean_rank0_minus_rank1_ms'], then['mean_rank0_minus_rank1_ms']
        delta = None if now_mean is None or then_mean is None else number(now_mean) - number(then_mean)
        result['sp1_comparison'] = {
            'window': sp1.name, 'then': then, 'delta_ms': delta,
            'statement': 'no comparable data' if delta is None else
            (f'changed by {delta:+.2f} ms a round since SP1 (2150 MHz, before FW3): FW3/FW3b and 2300 MHz together moved '
             'it (not separable here)' if abs(delta) >= IMBALANCE_MS else
             f'unchanged within {IMBALANCE_MS} ms a round since SP1 (2150 MHz, before FW3)')}
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--window', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(argv)
    if args.output.exists():
        print(f'{args.output} exists; refusing to overwrite', file=sys.stderr)
        return 1
    args.output.write_text(json.dumps(analyze(args.window), indent=1, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
