"""CEN0 item 3: DSpark depth cost tables from fixed-depth rows (CEN2), and the control archive's measurement resolution.

  # CEN2 rows -> VLLM_DSPARK_ROUND_MS / depth-file "conf:" and "confr:" strings (and a "ctx=" term)
  PYTHONPATH=. python3 -m offload.census.costtable fit --rows <rows.jsonl> [--k 5] [--base-cell short-b1] \
      [--context short=<tokens>,32k=32768,150k=153600] --output <json>
  # resolution of a decision cell for the 2x-resolution rule (F-CENSUS.plan.md §3 decision rules)
  PYTHONPATH=. python3 -m offload.census.costtable resolution --archive control-archive.jsonl --clock 2300 --output <json>

Rows are control-archive rows (schema dsv41-control-archive-v3: cell, metric, value, clock_cap_mhz, clock_caps_mhz,
commit, date, window, start, repeat) with one added field for CEN2: `setting`, the depth-file text of the row's
policy ("fixed:3", "conf", "conf:38,41,... cal=1", "confb", "confr:..."); a row without it is a production control.
Streams come from the cell name (`-b<N>`) or a `streams` field. Metrics used: ms_per_round, accepted_per_round,
decode_tps.

Policy semantics (image vllm/v1/worker/gpu/spec_decode/depth_policy.py, read 2026-10-27): the conf table is verify ms
by depth 0..k (1..k+1 live rows) plus VLLM_DSPARK_DRAFT_MS (production 0: the drafter is remote and async, so a
measured round already holds the drafter wait); "confr:" is ms by total live rows 1, 2, 3 ... of a step; tables past
their end continue with the last slope; "ctx=L:ms,..." adds per-request context ms (confb/confr only). Measured tables
are made non-decreasing (weighted pool-adjacent-violators) before they are written out; raw medians are kept beside.
Fitting refuses mixed clock caps (a dial-back mid-window makes its timing INCONCLUSIVE).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from offload.prefill_maxlen.evidence import Json, number, record

DEFAULT_ROUND_MS = (38.0, 41.0, 44.0, 46.5, 48.5, 50.5)      # depth_policy.py _DEFAULT_ROUND_MS (upstream stack)
MIN_N = 6                                                    # WIN2-R3: >= 6 paired requests per cell
FIXED = re.compile(r'^fixed:(\d+)$')
STREAMS = re.compile(r'-b(\d+)$')
START_NO = re.compile(r'^s(\d+)-')


@dataclass(frozen=True)
class Row:
    setting: str
    cell: str
    streams: int | None             # None: a cell without a stream count (only resolution reads it)
    metric: str
    value: float
    clock: int
    caps: tuple[int, ...]
    commit: str
    window: str
    start: str
    repeat: int
    max_tokens: int | None

    @property
    def depth(self) -> int | None:
        m = FIXED.match(self.setting.strip().lower().split()[0]) if self.setting.strip() else None
        return int(m.group(1)) if m else None

    @property
    def context(self) -> str:
        return STREAMS.sub('', self.cell)


def parse(lines: Sequence[str]) -> list[Row]:
    rows: list[Row] = []
    for i, text in enumerate(lines, 1):
        if not text.strip():
            continue
        d = record(cast(Json, json.loads(text)))
        missing = [k for k in ('cell', 'metric', 'value', 'clock_cap_mhz', 'commit', 'window', 'start', 'repeat')
                   if k not in d]
        if missing:
            raise ValueError(f'line {i}: missing {missing}')
        cell = str(d['cell'])
        m = STREAMS.search(cell)
        streams = int(number(d['streams'])) if 'streams' in d else int(m.group(1)) if m else None
        if streams is not None and streams < 1:
            raise ValueError(f'line {i}: stream count {streams} for cell {cell!r}')
        caps = tuple(int(number(x)) for x in cast(list[Json], d.get('clock_caps_mhz') or [d['clock_cap_mhz']]))
        rows.append(Row(str(d.get('setting') or 'control'), cell, streams, str(d['metric']), number(d['value']),
                        int(number(d['clock_cap_mhz'])), caps, str(d['commit']), str(d['window']), str(d['start']),
                        int(number(d['repeat'])),
                        None if d.get('max_tokens') is None else int(number(d['max_tokens']))))
    return rows


def pav(values: Sequence[float], weights: Sequence[float]) -> list[float]:
    """Weighted pool-adjacent-violators: the closest non-decreasing sequence (least squares)."""
    blocks: list[list[float]] = []
    for y, w in zip(values, weights):
        blocks.append([y, w, 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            y2, w2, c2 = blocks.pop()
            y1, w1, c1 = blocks.pop()
            blocks.append([(y1 * w1 + y2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    out: list[float] = []
    for y, _, c in blocks:
        out += [y] * int(c)
    return out


def _summary(xs: Sequence[float]) -> dict[str, Json]:
    q = statistics.quantiles(xs, n=4) if len(xs) >= 2 else [xs[0], xs[0], xs[0]]
    return {'n': len(xs), 'median': statistics.median(xs), 'mean': statistics.fmean(xs), 'q1': q[0], 'q3': q[2],
            'low_n': len(xs) < MIN_N}


def _fmt(xs: Sequence[float]) -> str:
    return ','.join(f'{x:.1f}' for x in xs)


def _one(values: Sequence[int], what: str) -> int:
    distinct = sorted(set(values))
    if len(distinct) != 1:
        raise ValueError(f'INCONCLUSIVE: mixed {what} {distinct}')
    return distinct[0]


def depth_tables(rows: Sequence[Row], k: int) -> dict[str, Json]:
    """Per B1 cell: ms/round, accepted/round and tokens per ms by fixed depth 0..k."""
    out: dict[str, Json] = {}
    for cell in sorted({r.cell for r in rows if r.streams == 1}):
        by: dict[str, Json] = {}
        for d in range(k + 1):
            ms = [r.value for r in rows if r.cell == cell and r.depth == d and r.metric == 'ms_per_round']
            acc = [r.value for r in rows if r.cell == cell and r.depth == d and r.metric == 'accepted_per_round']
            if not ms:
                continue
            entry: dict[str, Json] = {'ms_per_round': _summary(ms)}
            if acc:
                entry['accepted_per_round'] = _summary(acc)
                entry['tokens_per_ms'] = (1 + statistics.median(acc)) / statistics.median(ms)
            by[str(d)] = entry
        missing = [d for d in range(k + 1) if str(d) not in by]
        medians = [number(record(record(by[str(d)])['ms_per_round'])['median']) for d in range(k + 1)
                   if str(d) in by]
        weights = [number(record(record(by[str(d)])['ms_per_round'])['n']) for d in range(k + 1) if str(d) in by]
        cell_out: dict[str, Json] = {'depths': by, 'missing_depths': cast(Json, missing)}
        if not missing:
            mono = pav(medians, weights)
            cell_out['ms_raw'] = cast(Json, medians)
            cell_out['ms_monotone'] = cast(Json, mono)
            cell_out['vs_default_ms'] = cast(Json, [m - dflt for m, dflt in zip(mono, DEFAULT_ROUND_MS)])
            best = [d for d in range(k + 1) if 'tokens_per_ms' in record(by[str(d)])]
            if best:
                top = max(best, key=lambda d: number(record(by[str(d)])['tokens_per_ms']))
                cell_out['best_fixed_depth'] = top
        out[cell] = cell_out
    return out


def rows_tables(rows: Sequence[Row]) -> dict[str, Json]:
    """Per context class: step ms by total live rows (streams x (depth + 1)) from every fixed-depth cell of it."""
    out: dict[str, Json] = {}
    for ctx in sorted({r.context for r in rows}):
        pts: dict[int, list[float]] = {}
        for r in rows:
            if r.context == ctx and r.depth is not None and r.streams is not None and r.metric == 'ms_per_round':
                pts.setdefault(r.streams * (r.depth + 1), []).append(r.value)
        if not pts:
            continue
        xs = sorted(pts)
        med = [statistics.median(pts[x]) for x in xs]
        mono = pav(med, [float(len(pts[x])) for x in xs])
        table: list[float] = []
        interpolated: list[int] = []
        for n in range(1, xs[-1] + 1):
            if n in pts:
                table.append(mono[xs.index(n)])
            elif n < xs[0]:
                table.append(mono[0])
                interpolated.append(n)
            else:
                hi = next(x for x in xs if x > n)
                lo = max(x for x in xs if x < n)
                ylo, yhi = mono[xs.index(lo)], mono[xs.index(hi)]
                table.append(ylo + (yhi - ylo) * (n - lo) / (hi - lo))
                interpolated.append(n)
        out[ctx] = {'measured_rows': cast(Json, xs), 'n': cast(Json, [len(pts[x]) for x in xs]),
                    'ms_raw': cast(Json, med), 'ms_by_rows': cast(Json, table),
                    'interpolated_rows': cast(Json, interpolated), 'confr': 'confr:' + _fmt(table)}
    return out


def ctx_term(tables: Mapping[str, Json], base: str, context: Mapping[str, int]) -> dict[str, Json] | None:
    """Per-request context ms for "ctx=": the median over depths of (cell ms - base cell ms), at each cell's context
    length; None unless the base and >= 1 other B1 cell have full monotone tables and known context lengths."""
    if base not in tables or 'ms_monotone' not in record(tables[base]):
        return None
    b = [number(x) for x in cast(list[Json], record(tables[base])['ms_monotone'])]
    pts: list[tuple[int, float]] = []
    for cell, t in tables.items():
        ctx = STREAMS.sub('', cell)
        if cell == base or 'ms_monotone' not in record(t) or ctx not in context:
            continue
        ms = [number(x) for x in cast(list[Json], record(t)['ms_monotone'])]
        pts.append((context[ctx], statistics.median(m - x for m, x in zip(ms, b))))
    base_ctx = STREAMS.sub('', base)
    if not pts or base_ctx not in context:
        return None
    pts = sorted([(context[base_ctx], 0.0), *pts])
    return {'points': cast(Json, [list(p) for p in pts]), 'option': 'ctx=' + ','.join(f'{x}:{y:.2f}' for x, y in pts)}


def fit(rows: Sequence[Row], k: int = 5, base_cell: str = 'short-b1', context: Mapping[str, int] | None = None
        ) -> dict[str, Json]:
    fixed = [r for r in rows if r.depth is not None]
    if not fixed:
        raise ValueError('no fixed-depth rows (setting "fixed:<d>")')
    unknown = sorted({r.cell for r in fixed if r.streams is None})
    if unknown:
        raise ValueError(f'fixed-depth rows without a stream count: {unknown} (name the cell -b<N> or add streams)')
    clock = _one([c for r in fixed for c in r.caps] + [r.clock for r in fixed], 'clock caps')
    commits = sorted({r.commit for r in fixed})
    tables = depth_tables(fixed, k)
    out: dict[str, Json] = {'type': 'cen2-cost-table', 'clock_cap_mhz': clock, 'commits': cast(Json, commits),
                            'k': k, 'default_round_ms': cast(Json, list(DEFAULT_ROUND_MS)),
                            'status': 'MEASURED medians of fixed-depth rows; monotone fit and interpolation ESTIMATED',
                            'b1_tables': tables, 'rows_tables': rows_tables(fixed)}
    base = record(tables[base_cell]) if base_cell in tables else None
    if base is not None and 'ms_monotone' in base:
        ms = [number(x) for x in cast(list[Json], base['ms_monotone'])]
        out['env'] = {'VLLM_DSPARK_ROUND_MS': _fmt(ms), 'VLLM_DSPARK_DRAFT_MS': '0'}
        out['depth_file'] = {'conf': 'conf:' + _fmt(ms), 'conf_cal': 'conf:' + _fmt(ms) + ' cal=1',
                             'confb': 'confb:' + _fmt(ms)}
        term = ctx_term(tables, base_cell, context or {})
        if term is not None:
            out['ctx'] = term
            record(out['depth_file'])['confb_ctx'] = 'confb:' + _fmt(ms) + ' ' + str(term['option'])
    else:
        out['env'] = None
        out['refused'] = f'base cell {base_cell!r} lacks a full fixed-depth table 0..{k}'
    return out


# ------------------------------------------------------------------ resolution


def resolution(rows: Sequence[Row], clock: int, metric: str = 'decode_tps') -> dict[str, Json]:
    """Control-to-control drift inside a window: per cell, consecutive control starts (s1 -> s3 -> s5) of the same
    window, max_tokens and clock; drift = (later - earlier) / mean of the two start means. Resolution = SD of the
    drifts (MEASURED). Status OK at >= 6 pairs, else INSUFFICIENT (the plan's rule)."""
    chosen = [r for r in rows if r.metric == metric and r.setting == 'control' and r.caps == (clock,)]
    by: dict[tuple[str, str, int | None], dict[str, list[float]]] = {}
    for r in chosen:
        by.setdefault((r.cell, r.window, r.max_tokens), {}).setdefault(r.start, []).append(r.value)
    cells: dict[str, list[float]] = {}
    for (cell, _, _), starts in by.items():
        order = sorted(starts, key=lambda s: int(m.group(1)) if (m := START_NO.match(s)) else 1 << 30)
        means = [statistics.fmean(starts[s]) for s in order]
        for a, b in zip(means, means[1:]):
            cells.setdefault(cell, []).append((b - a) / ((a + b) / 2))
    out: dict[str, Json] = {}
    for cell, drifts in sorted(cells.items()):
        sd = statistics.stdev(drifts) if len(drifts) >= 2 else None
        out[cell] = {'pairs': len(drifts), 'sd': sd, 'median_abs': statistics.median(abs(x) for x in drifts),
                     'max_abs': max(abs(x) for x in drifts), 'two_x_resolution': None if sd is None else 2 * sd,
                     'status': 'OK' if len(drifts) >= MIN_N else 'INSUFFICIENT'}
    return {'type': 'cen-resolution', 'metric': metric, 'clock_cap_mhz': clock,
            'status': 'MEASURED control-to-control drift (relative)', 'cells': out,
            'controls_at_clock': len(chosen)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fit')
    f.add_argument('--rows', type=Path, nargs='+', required=True)
    f.add_argument('--k', type=int, default=5)
    f.add_argument('--base-cell', default='short-b1')
    f.add_argument('--context', default='', help='context=tokens,... e.g. short=1024,32k=32768,150k=153600')
    rz = sub.add_parser('resolution')
    rz.add_argument('--archive', type=Path, nargs='+', required=True)
    rz.add_argument('--clock', type=int, required=True)
    rz.add_argument('--metric', default='decode_tps')
    for p in (f, rz):
        p.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(argv)
    if args.output.exists():
        print(f'{args.output} exists; refusing to overwrite', file=sys.stderr)
        return 1
    if args.cmd == 'fit':
        rows = [r for p in args.rows for r in parse(p.read_text().splitlines())]
        context = {kv.split('=')[0]: int(kv.split('=')[1]) for kv in args.context.split(',') if '=' in kv}
        res = fit(rows, args.k, args.base_cell, context)
        if res.get('env'):
            print(json.dumps(res['env']), json.dumps(res['depth_file']))
        else:
            print(res.get('refused'), file=sys.stderr)
    else:
        rows = [r for p in args.archive for r in parse(p.read_text().splitlines())]
        res = resolution(rows, args.clock, args.metric)
        if not res['controls_at_clock']:
            print(f'no control rows at {args.clock} MHz for {args.metric}', file=sys.stderr)
        for cell, v in record(res['cells']).items():
            x = record(v)
            sd = x['sd']
            print(f"{cell:20} pairs={x['pairs']} sd={'n/a' if sd is None else f'{100 * number(sd):.2f} %'} "
                  f"{x['status']}")
    args.output.write_text(json.dumps(res, indent=1, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
