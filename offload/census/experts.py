"""CEN0 item 2: distinct routed experts per MoE decode call from VLLM_EXL3_MOE_STATS files (CEN1), against uniform
routing U(r) = 384 (1 - (1 - 6/384)^r) and KL1's inferred count.

  PYTHONPATH=. python3 -m offload.census.experts --stats <rank0.jsonl> [<rank1.jsonl>] [--cuts cuts.json] --output o.json
  PYTHONPATH=. python3 -m offload.census.experts --uniform 1 6 24

What the hook writes (image vllm_exl3_moe/forward.py `_record_stats`, read 2026-10-27): per worker process (pid), a
CUMULATIVE summary line every 2,000 recorded calls: calls_by_tokens {m: calls}, mean_distinct_experts {m: mean} and
mean_routed_slots {m: mean}, keyed by the call's row count m (the verify window, padding rows included) and rounded to
two decimals. Routed slots count ids >= 0, so slots / 6 = live rows (rows past a request's depth carry id -1). There
is no layer field: every figure is a mean over the 40 MoE layers of one rank. It records only on the coop / fused
decode paths: a call that takes exl3_smoe (production's decode path, VLLM_EXL3_MOE_SMOE default 1) returns before
`_record_stats`, so an eager start with smoe on counts no decode call at all (`decode_calls` 0 below is that case).

Live rows per interval: consecutive snapshots of one pid are differenced per m; an interval is PURE at r live rows when
its slots per call are within the rounding bound (+ TOL_ROWS) of 6 r, else MIXED (the depth file changed inside it, or
requests ended) and left out of the curve. The rounding bound of a differenced mean is 0.005 (c0 + c1) / dc.
Counts only: eager timing is never read here. Output: numbers only (no ids, no text).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from offload.kl1 import shapes as S
from offload.prefill_maxlen.evidence import Json, number, record

ROUND = 0.005                       # half of the hook's 2-decimal rounding
TOL_ROWS = 0.05                     # live-row tolerance on top of the rounding bound
FLUSH = 2000                        # calls between summary lines (forward.py)
# KL1 stage B (KL1-W-20261026T194927Z analysis-rank0.json, SP1-W-20261026T160012Z rank 0, 45 in-range B1 steps):
# distinct experts per layer INFERRED by inverting the isolated smoe time-vs-distinct curve (ESTIMATED, not counted).
KL1_INFERRED_B1 = {'median': 13.0, 'range': (9.0, 22.0), 'prefetch_correction': 1.7,
                   'source': 'KL1-W-20261026T194927Z/analysis-rank0.json (ESTIMATED inversion)'}
# m bands of patch 0064 (smoe split for 33-63 rows) and the smoe/coop row limits
BANDS = ((1, 8), (9, 32), (33, 63), (64, 1 << 30))


def uniform(rows: float) -> float:
    """Expected distinct experts when `rows` rows each pick 6 of 384 independently and uniformly (shapes.py)."""
    return S.unique_uniform(rows)


@dataclass(frozen=True)
class Snapshot:
    pid: int
    line: int                       # 1-based line number in its file
    calls: dict[int, int]
    distinct: dict[int, float]
    slots: dict[int, float]

    @property
    def total(self) -> int:
        return sum(self.calls.values())


def _int_keys(d: Json, what: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for k, v in record(d).items():
        if not k.lstrip('-').isdigit():
            raise ValueError(f'{what}: key {k!r} is not a row count')
        out[int(k)] = number(v)
    return out


def parse(lines: Sequence[str]) -> list[Snapshot]:
    snaps: list[Snapshot] = []
    for i, text in enumerate(lines, 1):
        if not text.strip():
            continue
        row = record(cast(Json, json.loads(text)))
        missing = [k for k in ('pid', 'calls_by_tokens', 'mean_distinct_experts', 'mean_routed_slots') if k not in row]
        if missing:
            raise ValueError(f'line {i}: missing {missing}')
        calls = {k: int(v) for k, v in _int_keys(row['calls_by_tokens'], f'line {i} calls').items()}
        distinct = _int_keys(row['mean_distinct_experts'], f'line {i} distinct')
        slots = _int_keys(row['mean_routed_slots'], f'line {i} slots')
        if set(calls) != set(distinct) or set(calls) != set(slots):
            raise ValueError(f'line {i}: row-count keys differ between the three fields')
        for m, c in calls.items():
            if c <= 0 or m <= 0:
                raise ValueError(f'line {i}: m {m} calls {c}')
            if not 0 <= slots[m] <= S.TOP_K * m + ROUND:
                raise ValueError(f'line {i}: mean routed slots {slots[m]} outside 0..6 m for m {m}')
            # a call with L >= 1 live rows reads >= 6 = 6L / L >= slots / m and <= slots distinct experts
            if not slots[m] / m - ROUND <= distinct[m] <= min(S.E, slots[m]) + ROUND:
                raise ValueError(f'line {i}: mean distinct {distinct[m]} impossible for {slots[m]} slots')
        if sum(calls.values()) % FLUSH:
            raise ValueError(f'line {i}: {sum(calls.values())} calls is not a multiple of {FLUSH} (another hook?)')
        snaps.append(Snapshot(int(number(row['pid'])), i, calls, distinct, slots))
    by_pid: dict[int, list[Snapshot]] = {}
    for s in snaps:
        by_pid.setdefault(s.pid, []).append(s)
    for pid, seq in by_pid.items():
        for a, b in zip(seq, seq[1:]):
            if any(b.calls.get(m, 0) < c for m, c in a.calls.items()):
                raise ValueError(f'pid {pid} line {b.line}: cumulative calls went down (restart without a new pid?)')
    return snaps


@dataclass(frozen=True)
class Interval:
    pid: int
    m: int
    first_line: int                 # the earlier snapshot (0 = process start)
    last_line: int
    calls: int
    distinct_mean: float
    distinct_bound: float
    live_rows: float
    rows_bound: float

    @property
    def pure_rows(self) -> int | None:
        """The integer live-row count this interval sits on. A 50/50 mix of r - 1 and r + 1 also sits on r: only a
        cut per depth (the runner's line count at each depth-file change) separates depths reliably."""
        r = round(self.live_rows)
        return r if r >= 1 and abs(self.live_rows - r) <= self.rows_bound + TOL_ROWS else None


def intervals(snaps: Sequence[Snapshot], start_line: int = 0, end_line: int | None = None) -> list[Interval]:
    """Differences of consecutive snapshots per pid and m with both snapshots in (start_line, end_line]; a pid's first
    snapshot ever is differenced against process start. start_line / end_line are the file's line counts the runner
    recorded at a cell's start and end."""
    out: list[Interval] = []
    by_pid: dict[int, list[Snapshot]] = {}
    for s in snaps:
        by_pid.setdefault(s.pid, []).append(s)
    for pid, seq in by_pid.items():
        prev: Snapshot | None = None
        for s in seq:
            if end_line is not None and s.line > end_line:
                break
            # a pair straddling the cut start would mix the previous cell's calls in: only pairs wholly inside, or
            # a pid whose first snapshot is inside (differenced against process start)
            if s.line > start_line and (prev is None or prev.line > start_line):
                for m, c1 in s.calls.items():
                    c0 = prev.calls.get(m, 0) if prev is not None else 0
                    dc = c1 - c0
                    if dc <= 0:
                        continue
                    d0 = prev.distinct.get(m, 0.0) if prev is not None else 0.0
                    s0 = prev.slots.get(m, 0.0) if prev is not None else 0.0
                    bound = ROUND * (c0 + c1) / dc
                    out.append(Interval(pid, m, prev.line if prev is not None else 0, s.line, dc,
                                        (s.distinct[m] * c1 - d0 * c0) / dc, bound,
                                        (s.slots[m] * c1 - s0 * c0) / dc / S.TOP_K, bound / S.TOP_K))
            prev = s
    return out


def curve(ivs: Sequence[Interval], requests: int | None, depth: int | None = None) -> dict[str, Json]:
    """Calls-weighted mean distinct experts per (m, live rows) over PURE intervals, against uniform routing. With a
    cut's fixed depth, intervals off its live-row count (requests x (depth + 1)) are counted apart, not curved."""
    groups: dict[tuple[int, int], list[Interval]] = {}
    mixed = off_depth = 0
    expected = requests * (depth + 1) if requests and depth is not None else None
    for iv in ivs:
        r = iv.pure_rows
        if r is None:
            mixed += iv.calls
            continue
        if expected is not None and r != expected:
            off_depth += iv.calls
            continue
        groups.setdefault((iv.m, r), []).append(iv)
    rows: list[dict[str, Json]] = []
    for (m, r), g in sorted(groups.items()):
        calls = sum(iv.calls for iv in g)
        mean = sum(iv.distinct_mean * iv.calls for iv in g) / calls
        bound = sum(iv.distinct_bound * iv.calls for iv in g) / calls
        u = uniform(r)
        row: dict[str, Json] = {'m': m, 'live_rows': r, 'intervals': len(g), 'calls': calls,
                                'distinct_mean': mean, 'distinct_bound': bound,
                                'distinct_min_interval': min(iv.distinct_mean for iv in g),
                                'distinct_max_interval': max(iv.distinct_mean for iv in g),
                                'uniform': u, 'over_uniform': mean / u, 'per_row': mean / r}
        if requests:
            row_depth = r / requests - 1
            row['depth'] = int(row_depth) if row_depth == int(row_depth) and row_depth >= 0 else None
        rows.append(row)
    return {'requests': requests, 'depth': depth, 'pure_calls': sum(number(r['calls']) for r in rows),
            'mixed_calls': mixed, 'off_depth_calls': off_depth, 'curve': cast(Json, rows)}


def bands(snaps: Sequence[Snapshot]) -> dict[str, Json]:
    """Share of recorded calls by row band (last snapshot of each pid): patch 0064's 33-63-row band frequency."""
    last: dict[int, Snapshot] = {}
    for s in snaps:
        last[s.pid] = s
    totals: dict[str, int] = {}
    for s in last.values():
        for m, c in s.calls.items():
            lo, hi = next(b for b in BANDS if b[0] <= m <= b[1])
            key = f'{lo}-{hi}' if hi < 1 << 30 else f'{lo}+'
            totals[key] = totals.get(key, 0) + c
    n = sum(totals.values())
    return {'calls': n, 'share': cast(Json, {k: v / n for k, v in totals.items()} if n else {})}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(paths: Sequence[Path], cuts: Sequence[Mapping[str, Json]] = ()) -> dict[str, Json]:
    """One file per rank (rank order). `cuts`: [{cell, requests, depth?, file: index, from_line, to_line}] line ranges
    the window runner recorded at cell and depth-file boundaries; without cuts each file is one segment with unknown
    requests (and a straddling interval that averages two depths can pass for the depth between them)."""
    files: list[dict[str, Json]] = []
    segments: dict[str, Json] = {}
    parsed: list[list[Snapshot]] = []
    for i, p in enumerate(paths):
        snaps = parse(p.read_text().splitlines())
        parsed.append(snaps)
        last: dict[int, Snapshot] = {}
        for s in snaps:
            last[s.pid] = s
        decode_calls = sum(c for s in last.values() for c in s.calls.values())
        files.append({'file': p.name, 'sha256': _sha(p), 'snapshots': len(snaps), 'pids': cast(Json, sorted(last)),
                      'decode_calls': decode_calls, 'bands': bands(snaps),
                      'status': 'OK' if decode_calls else
                      'EMPTY: no recorded decode call (smoe decode path or graph replay records nothing)'})
    for c in cuts or [{'cell': 'all', 'requests': None, 'file': i, 'from_line': 0, 'to_line': None}
                      for i in range(len(paths))]:
        i = int(number(c['file']))
        req = None if c.get('requests') is None else int(number(c['requests']))
        end = None if c.get('to_line') is None else int(number(c['to_line']))
        depth = None if c.get('depth') is None else int(number(c['depth']))
        ivs = intervals(parsed[i], int(number(c.get('from_line') or 0)), end)
        segments[f"{c['cell']}:rank{i}"] = curve(ivs, req, depth)
    out: dict[str, Json] = {'type': 'cen1-distinct-experts',
                            'status': 'MEASURED counts (eager start; mean over 40 layers of one rank); uniform U(r) '
                                      'and KL1 inferred count ESTIMATED',
                            'per_layer': 'NOT AVAILABLE: the hook has no layer field',
                            'files': cast(Json, files), 'segments': segments,
                            'uniform': cast(Json, {str(r): uniform(r) for r in (1, 2, 3, 4, 5, 6, 8, 12, 16, 20, 24)}),
                            'kl1_inferred_b1': cast(Json, {k: (list(v) if isinstance(v, tuple) else v)
                                                           for k, v in KL1_INFERRED_B1.items()})}
    if len(paths) == 2:
        out['cross_rank'] = cross_rank(segments)
    return out


def cross_rank(segments: Mapping[str, Json]) -> dict[str, Json]:
    """Both ranks see the same router ids: their curves must agree within the rounding bound."""
    worst = 0.0
    compared = 0
    for name, seg in segments.items():
        if not name.endswith(':rank0'):
            continue
        other = segments.get(name.removesuffix(':rank0') + ':rank1')
        if other is None:
            continue
        mine = [record(x) for x in cast(list[Json], record(seg)['curve'])]
        a = {(int(number(x['m'])), int(number(x['live_rows']))): x for x in mine}
        for r in cast(list[Json], record(other)['curve']):
            row = record(r)
            key = (int(number(row['m'])), int(number(row['live_rows'])))
            if key in a:
                compared += 1
                diff = abs(number(a[key]['distinct_mean']) - number(row['distinct_mean']))
                tol = number(a[key]['distinct_bound']) + number(row['distinct_bound'])
                worst = max(worst, diff - tol)
    verdict = ('NOT COMPARED (no cell on both ranks)' if not compared else
               'CONSISTENT' if worst <= 1e-9 else 'RANKS DISAGREE (check cuts / pids)')
    return {'compared': compared, 'max_excess_over_bound': worst, 'verdict': verdict}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stats', type=Path, nargs='+', help='VLLM_EXL3_MOE_STATS files, rank order')
    ap.add_argument('--cuts', type=Path, help='JSON list of {cell, requests, depth, file, from_line, to_line}, or the '
                                              "CEN1 runner's cuts.json ({'type': 'cen1-cuts', 'cuts': [...]})")
    ap.add_argument('--output', type=Path)
    ap.add_argument('--uniform', type=float, nargs='+', help='print U(r) for these row counts and exit')
    args = ap.parse_args(argv)
    if args.uniform:
        for r in args.uniform:
            if not math.isfinite(r) or r < 0:
                ap.error('rows must be >= 0')
            print(f'U({r:g}) = {uniform(r):.1f}')
        return 0
    if not args.stats or args.output is None:
        ap.error('give --stats and --output (or --uniform)')
    if args.output.exists():
        print(f'{args.output} exists; refusing to overwrite', file=sys.stderr)
        return 1
    raw: Json = json.loads(args.cuts.read_text()) if args.cuts else []
    cuts = [record(c) for c in cast(list[Json], record(raw)['cuts'] if isinstance(raw, dict) else raw)]
    res = analyze(args.stats, cuts)
    args.output.write_text(json.dumps(res, indent=1, allow_nan=False) + '\n')
    for name, seg in record(res['segments']).items():
        for row in cast(list[Json], record(seg)['curve']):
            x = record(row)
            print(f"{name:24} m={x['m']} rows={x['live_rows']} calls={x['calls']} distinct="
                  f"{number(x['distinct_mean']):.2f}+-{number(x['distinct_bound']):.2f} uniform={number(x['uniform']):.1f}")
    for f in cast(list[Json], res['files']):
        if record(f)['status'] != 'OK':
            print(f"{record(f)['file']}: {record(f)['status']}", file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
