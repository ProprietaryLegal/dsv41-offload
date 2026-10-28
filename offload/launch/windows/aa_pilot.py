"""WIN2-R3 seeded A/A pilot recipe, an end-of-restore add-on (F-METHOD §3.1 pilot and §3.4 A/A band; no extra HOLD).

Two starts of the same production configuration, each running the identical seeded cells and the distributional probe:
- start A: the window's LAST control start (production configuration on the experiment port), after its own cells;
- start B: restored production, inside the window's HOLD, between W8 `check` (PASS) and W8 `close`. W8 itself and its
  greedy-identity gate are unchanged; the pilot only reads.
`run` measures one start; `compare(A, B)` gives the verdict and the numbers the other rules register:
- PASS (single profile): every B1 seeded cell has identical output digests AND identical per-request round counts in
  both starts (determinism of seeded speculative decoding at B1). B >= 2 cells are reported, not required (production
  kernels are not batch-invariant).
- per cell and metric: the paired-difference CV (for `resolution.py --diff-cv`) and the start difference (one A/A, so a
  1-df between-start SD estimate for `decide_paired_window(start_sd=...)`; accumulate over pilots before relying on it);
- the distributional A/A comparison (feed `dist_gate.register_band`).
COST (ESTIMATED): seeded cells ~5 min per start (short-b1 38 s, 32k-b1 56 s, 32k-b2 88 s, 32k-b3 120 s), distributional
probe ~6.5 min per start for the default `short,32k` subset (one cold prefill per prompt; F-METHOD's "+2 min" covers
only the short prompts). Coordinator-only; `run` without --execute prints the plan.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from offload.gb10_budget import window_cells as wc
from offload.launch.windows import decision_v2 as d
from offload.launch.windows import dist_gate as dg
from offload.prefill_maxlen.evidence import Json, record

SCHEMA = 'dsv41-aa-pilot-v1'
PILOT_CELLS = ('seeded-short-b1', 'seeded-32k-b1', 'seeded-32k-b2', 'seeded-32k-b3')
METRICS = {'decode_tps': True, 'ms_per_round': False, 'accepted_per_round': True}
DIST_SUBSET = ('short', '32k')
COORDINATOR = 'Deepseek/coordinator'


def run(seeded_api: wc.SeededApi, dist_api: dg.DistApi, texts: dict[str, tuple[str, int]], plan: wc.Plan,
        materials: str, dist_rows: Mapping[str, Json], *, cells: Sequence[str] = PILOT_CELLS) -> dict[str, Json]:
    """One start: every pilot seeded cell, then the distributional probe (same salt prefix as the plan)."""
    results: dict[str, Json] = {}
    for name in cells:
        results[name] = wc.run_seeded_cell(seeded_api, wc.SEEDED_CELLS[name], texts, plan)
    return {'schema': SCHEMA, 'type': 'start', 'cells': results,
            'dist': dg.probe(dist_api, materials, dist_rows, plan.salt_prefix)}


def _diff_stats(a: Mapping[str, Json], b: Mapping[str, Json], metric: str, higher: bool) -> dict[str, Json]:
    x, y = wc.seeded_values(a, metric), wc.seeded_values(b, metric)
    diffs = [((v - u) if higher else (u - v)) / u for u, v in zip(x, y)]
    sd = statistics.stdev(diffs)
    mean = statistics.fmean(diffs)
    return {'pairs': len(diffs), 'mean_diff': mean, 'diff_cv': sd, 'sign_flip_p': d.sign_flip_p(diffs),
            'start_sd_estimate_1df': abs(mean) / math.sqrt(2)}


def compare(a: Mapping[str, Json], b: Mapping[str, Json]) -> dict[str, Json]:
    ca, cb = record(a['cells']), record(b['cells'])
    if set(ca) != set(cb):
        raise ValueError('the two starts ran different pilot cells')
    cells: dict[str, Json] = {}
    single_pass = True
    for name in sorted(ca):
        x, y = record(ca[name]), record(cb[name])
        identical = wc.streams_identical(x, y)
        rounds = [record(p)['rounds'] == record(q)['rounds']
                  for p, q in zip(cast(list[Json], x['pairs']), cast(list[Json], y['pairs']))]
        digests = [record(p)['output_sha256'] == record(q)['output_sha256']
                   for p, q in zip(cast(list[Json], x['pairs']), cast(list[Json], y['pairs']))]
        profile = str(x['profile'])
        if profile == 'single' and not identical:
            single_pass = False
        cells[name] = {'profile': profile, 'identical': identical, 'pairs_same_rounds': sum(rounds),
                       'pairs_same_output': sum(digests), 'pairs': len(rounds),
                       'metrics': {m: _diff_stats(x, y, m, hib) for m, hib in METRICS.items()}}
    singles = [c for c in cells.values() if record(c)['profile'] == 'single']
    return {'schema': SCHEMA, 'type': 'compare',
            'verdict': 'PASS' if single_pass and singles else 'FAIL',
            'reason': ('B1 seeded streams and round counts identical across the two starts' if single_pass and singles
                       else 'a B1 seeded cell differs between the two starts (or no B1 cell ran): seeded pairing at B1 '
                            'cannot assume identical streams; report measured diff CVs only'),
            'cells': cells, 'dist': dg.compare(record(a['dist']), record(b['dist']))}


def plan_text(cells: Sequence[str]) -> str:
    return ('DRY RUN seeded A/A pilot (end-of-restore add-on, no extra HOLD). Start A = the last control start, start B = '
            'restored production between W8 check and W8 close. Each start: seeded cells ' + ', '.join(cells) +
            f' (6 seeds x {wc.SEEDED_MAX_TOKENS} tokens, registered seeds), then the distributional probe ('
            + ','.join(DIST_SUBSET) + ' subset, 8 fixed positions per prompt). PASS = identical B1 streams and round counts.')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='one start')
    r.add_argument('--execute', action='store_true')
    r.add_argument('--url', required=True, help='the start under test (tunnel 127.0.0.1:18001 or production :8000)')
    r.add_argument('--cuts', type=Path, required=True, help='window_cells build output (numeric cuts)')
    r.add_argument('--dist-cuts', type=Path, required=True, help='dist_gate cuts output')
    r.add_argument('--salt', required=True)
    r.add_argument('--deadline-utc', type=float, required=True)
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--data-dir', type=Path, default=Path('/opt/dsv41/data/bench-data'))
    c = sub.add_parser('compare')
    c.add_argument('a', type=Path)
    c.add_argument('b', type=Path)
    c.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == 'compare':
        result = compare(record(cast(Json, json.loads(args.a.read_text()))), record(cast(Json, json.loads(args.b.read_text()))))
        with args.out.open('x') as handle:
            handle.write(json.dumps(result, allow_nan=False) + '\n')
        print(json.dumps({'verdict': result['verdict'], 'reason': result['reason']}))
        return 0 if result['verdict'] == 'PASS' else 1
    if not args.execute:
        print(plan_text(PILOT_CELLS))
        return 0
    if os.environ.get('WINDOW_COORDINATOR') != COORDINATOR:
        print(f'STOP: the A/A pilot is coordinator-only (WINDOW_COORDINATOR={COORDINATOR})', file=sys.stderr)
        return 2
    if args.out.exists():
        print('STOP: fresh --out required', file=sys.stderr)
        return 2
    from offload.prefill_trace.window_requests import corpus
    materials, hashes = corpus(args.data_dir)
    texts = wc.texts_from(record(cast(Json, json.loads(args.cuts.read_text()))), materials, hashes)
    dist = record(cast(Json, json.loads(args.dist_cuts.read_text())))
    if dist.get('corpus_sha256') != hashes:
        raise ValueError('public corpus changed since the distributional cuts were built')
    rows = record(dist['prompts'])
    keys = {p.key for p in dg.selected(list(DIST_SUBSET))}
    plan = wc.Plan(args.deadline_utc, 4 * 1024 ** 2, [], args.salt)
    result = run(wc.SeededHttp(args.url), dg.Http(args.url), texts, plan, materials, {k: rows[k] for k in sorted(keys)})
    with args.out.open('x') as handle:
        handle.write(json.dumps(result, allow_nan=False) + '\n')
    print(f'aa_pilot run: wrote {args.out} (numbers, digests and logprob vectors only)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
