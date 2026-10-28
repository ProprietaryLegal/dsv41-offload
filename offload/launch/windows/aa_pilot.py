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
DP-2 bridging cell (98aa385; start A only, so it needs no extra HOLD): bench2 legal-short and bench2 summary-32K at 1,500
tokens, the corpus short and 32k-0 prompts at 512 and 1,500 tokens, three seeded pairs each at production settings (T 1,
thinking high, priority 10), report-only (`bridge_report`). It separates answer length from date on the corpus prompts and
bridges the bench2 prompts across 10-24..26; a summary-32K reading near 1.06 instead of DP-2's 1.589 would point to a
production change (H4). Every prompt is pinned by its sha256 (window_cells.PROMPT_SHA256); the bench2 prompts are rebuilt
with bench2's own algorithm (`bench2_texts`) and refused unless they hash to DP-2's registered values.
COST (ESTIMATED): seeded cells ~5 min per start (short-b1 38 s, 32k-b1 56 s, 32k-b2 88 s, 32k-b3 120 s), distributional
probe ~6.5 min per start for the default `short,32k` subset (one cold prefill per prompt; F-METHOD's "+2 min" covers
only the short prompts), bridge ~8.6 min of cells plus up to ~1 min of /tokenize calls rebuilding summary-32K (start A).
Coordinator-only; `run` without --execute prints the plan.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

from offload.gb10_budget import window_cells as wc
from offload.launch.windows import decision_v2 as d
from offload.launch.windows import dist_gate as dg
from offload.prefill_maxlen.evidence import Json, number, record

SCHEMA = 'dsv41-aa-pilot-v1'
PILOT_CELLS = ('seeded-short-b1', 'seeded-32k-b1', 'seeded-32k-b2', 'seeded-32k-b3')
METRICS = {'decode_tps': True, 'ms_per_round': False, 'accepted_per_round': True}
DIST_SUBSET = ('short', '32k')
COORDINATOR = 'Deepseek/coordinator'

# ---------------------------------------------------------------- DP-2 bridging cell (report-only, start A)
BENCH2 = Path(__file__).resolve().parents[2] / 'bench/bench2.py'
BENCH2_SEED = 42                  # bench2 --seed default; B1 prompt index 0 uses seed + 0 (bench2 main)
BENCH2_SUMMARY_CTX = 32000
BENCH2_TITLE = 'MATERIALS (public court documents)'     # bench2 make_prompt, filler 'real'
# (cell, prompt key, max_tokens, seed_base, ESTIMATED cold prefill s, ESTIMATED s per pair: 1,500 tokens at >= 51 tok/s
# B1 MEASURED ~30 s + overhead, 512 tokens ~11 s)
BRIDGE_CELLS = tuple(wc.SeededCell(name, (key,), seed_base=base, pairs=3, max_tokens=tokens, prefill_s=prefill,
                                   pair_s=32.0 if tokens == 1500 else 11.0, report_only=True)
                     for name, key, tokens, base, prefill in (
                         ('bridge-bench2-legal-1500', 'bench2-legal-short', 1500, 201_000, 1.0),
                         ('bridge-bench2-summary32k-1500', 'bench2-summary-32k', 1500, 202_000, 20.0),
                         ('bridge-short-512', 'short', 512, 203_000, 2.0),
                         ('bridge-short-1500', 'short', 1500, 204_000, 2.0),
                         ('bridge-32k0-512', '32k-0', 512, 205_000, 20.0),
                         ('bridge-32k0-1500', '32k-0', 1500, 206_000, 20.0)))
# DP-2's MEASURED accepted per round (server counters, B1 cold controls; 95 % CI): the readings the bridge is set against.
DP2_REFERENCE = {
    'bridge-bench2-legal-1500': (1.002, 0.986, 1.018, '10-22..24, n 77'),
    'bridge-bench2-summary32k-1500': (1.589, 1.563, 1.615, '10-22..24, n 70'),
    'bridge-short-512': (1.294, 1.232, 1.355, '10-27, n 40'),
    'bridge-short-1500': (1.146, 1.089, 1.202, '10-26, n 8'),
    'bridge-32k0-512': (1.048, 1.000, 1.096, '10-27, n 40'),
    'bridge-32k0-1500': (0.949, 0.861, 1.036, '10-26, n 8')}
# DP-2 §5: a summary-32K reading near 1.06 (corpus-like) instead of ~1.59 would mean a production change (H4).
H4_PRODUCTION_CHANGE = 1.06


class Counter(Protocol):
    def tokens(self, text: str) -> int: ...          # chat-template count (bench2 prompt_tokens; window_cells.Http)
    def raw_tokens(self, text: str) -> int: ...      # bare prompt count (bench2 tok_count)


class BridgeHttp(wc.SeededHttp):
    def raw_tokens(self, text: str) -> int:
        return int(number(self.post('/tokenize', {'model': self.model()['id'], 'prompt': text})['count']))


def bench2_workloads(path: Path = BENCH2) -> dict[str, str]:
    """bench2's WORKLOADS literal, read without importing bench2 (its telemetry import exists only on the hosts)."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'WORKLOADS' for t in node.targets):
            value = ast.literal_eval(node.value)
            if isinstance(value, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
                return dict(value)
    raise ValueError(f'{path}: no WORKLOADS literal')


def bench2_docs(data_dir: Path) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    for name in ('opinions_cl.jsonl', 'filings_recap.jsonl'):
        for line in (data_dir / name).read_text().splitlines():
            row = record(cast(Json, json.loads(line)))
            docs.append({k: str(row.get(k, '')) for k in ('title', 'court', 'date', 'text')})
    return docs


def bench2_record(api: Counter, docs: Sequence[Mapping[str, str]], target_tokens: int, seed: int) -> str:
    """bench2 build_real, step for step (its trim loop included): the text must come out byte-identical."""
    order = list(docs)
    random.Random(seed).shuffle(order)
    parts: list[str] = []
    chars = 0
    ratio = 3.3
    for d in order:
        parts.append(f"=== {d.get('title', '')} ({d.get('court', '')}, {d.get('date', '')}) ===\n{d['text']}")
        chars += len(parts[-1])
        if chars / ratio >= target_tokens * 1.02:
            n = api.raw_tokens('\n\n'.join(parts))
            ratio = chars / n
            if n >= target_tokens:
                break
    text = '\n\n'.join(parts)
    while api.raw_tokens(text) > target_tokens:
        text = text[: int(len(text) - (api.raw_tokens(text) - target_tokens) * ratio) - 50]
    return text


def bench2_prompt(api: Counter, docs: Sequence[Mapping[str, str]], ctx: int, instruction: str, seed: int) -> tuple[str, int]:
    """bench2 make_prompt (filler 'real'): the longest record prefix whose chat prompt fits ctx tokens."""
    rec = bench2_record(api, docs, ctx + 4096, seed)

    def wrap(length: int) -> str:
        return f'{BENCH2_TITLE}:\n{rec[:length]}\n\nEND OF MATERIALS.\n\n{instruction}'
    lo, hi = 0, len(rec)
    best, count = wrap(0), api.tokens(wrap(0))
    if count > ctx:
        raise ValueError(f'requested context {ctx} is smaller than instruction/template ({count})')
    while lo <= hi:
        middle = (lo + hi) // 2
        candidate = wrap(middle)
        n = api.tokens(candidate)
        if n <= ctx:
            best, count, lo = candidate, n, middle + 1
        else:
            hi = middle - 1
    if ctx - count > 16:
        raise ValueError(f'corpus cannot fill requested context {ctx}; measured {count}')
    return best, count


def bench2_texts(api: Counter, data_dir: Path) -> dict[str, tuple[str, int]]:
    """The two bench2 prompts DP-2 measured, rebuilt and refused unless they hash to the registered values."""
    workloads = bench2_workloads()
    legal = workloads['legal']
    out = {'bench2-legal-short': (legal, api.tokens(legal)),
           'bench2-summary-32k': bench2_prompt(api, bench2_docs(data_dir), BENCH2_SUMMARY_CTX, workloads['summary'],
                                               BENCH2_SEED)}
    for key, (text, _) in out.items():
        sha = hashlib.sha256(text.encode()).hexdigest()
        if sha != wc.PROMPT_SHA256[key]:
            raise ValueError(f'{key}: rebuilt prompt hashes to {sha[:12]}, not DP-2\'s registered {wc.PROMPT_SHA256[key][:12]} '
                             '(bench data, tokenizer or template changed); the bridge never runs another prompt')
    return out


def bridge_report(results: Mapping[str, Json]) -> dict[str, Json]:
    """Per bridge cell: MEASURED accepted/round (mean of the three seeded pairs, per pair) beside DP-2's reference, the
    512-vs-1,500 difference per corpus prompt on one day, and the H4 flag. Report only: it never decides anything."""
    cells: dict[str, Json] = {}
    means: dict[str, float] = {}
    for cell in BRIDGE_CELLS:
        if cell.name not in results:
            continue
        values = wc.seeded_values(record(results[cell.name]), 'accepted_per_round')
        means[cell.name] = statistics.fmean(values)
        ref, low, high, where = DP2_REFERENCE[cell.name]
        cells[cell.name] = {'prompt': cell.prompts[0], 'max_tokens': cell.max_tokens,
                            'accepted_per_round': means[cell.name], 'per_pair': cast(Json, values),
                            'dp2_reference': {'mean': ref, 'ci95': [low, high], 'when': where},
                            'within_dp2_ci': low <= means[cell.name] <= high}
    length: dict[str, Json] = {}
    for key, short, long in (('short', 'bridge-short-512', 'bridge-short-1500'), ('32k-0', 'bridge-32k0-512', 'bridge-32k0-1500')):
        if short in means and long in means:
            length[key] = {'same_day_512_minus_1500': means[short] - means[long],
                           'dp2_cross_day_512_minus_1500': DP2_REFERENCE[short][0] - DP2_REFERENCE[long][0]}
    summary = means.get('bridge-bench2-summary32k-1500')
    flag: Json = None
    if summary is not None:
        midpoint = (H4_PRODUCTION_CHANGE + DP2_REFERENCE['bridge-bench2-summary32k-1500'][0]) / 2
        flag = ('H4 FLAG: summary-32K reads {:.3f}, nearer 1.06 than DP-2\'s 1.589: a production change between 10-24 and '
                '10-26, not prompt identity'.format(summary) if summary < midpoint else
                'summary-32K reads {:.3f}: consistent with prompt identity (DP-2 H1), no production change'.format(summary))
    return {'schema': SCHEMA, 'type': 'bridge', 'cells': cells, 'length_vs_date': length, 'h4': flag,
            'status': 'MEASURED seeded pairs (3 per cell), report only'}


def run(seeded_api: wc.SeededApi, dist_api: dg.DistApi, texts: dict[str, tuple[str, int]], plan: wc.Plan,
        materials: str, dist_rows: Mapping[str, Json], *, cells: Sequence[str] = PILOT_CELLS,
        start: str = 'A', bridge_texts: Mapping[str, tuple[str, int]] | None = None) -> dict[str, Json]:
    """One start: every pilot seeded cell, (start A) the DP-2 bridging cell, then the distributional probe (same salt
    prefix as the plan). Start A must carry `bridge_texts` (the two bench2 prompts, `bench2_texts`); start B never runs
    the bridge."""
    if start not in ('A', 'B'):
        raise ValueError(f'pilot start must be A (last control start) or B (restored production), not {start!r}')
    if (start == 'A') != (bridge_texts is not None):
        raise ValueError('the DP-2 bridging cell runs in start A only, with its bench2 prompts (bench2_texts)')
    results: dict[str, Json] = {}
    for name in cells:
        results[name] = wc.run_seeded_cell(seeded_api, wc.SEEDED_CELLS[name], texts, plan)
    out: dict[str, Json] = {'schema': SCHEMA, 'type': 'start', 'start': start, 'cells': results}
    if bridge_texts is not None:
        merged = {**texts, **bridge_texts}
        bridge: dict[str, Json] = {c.name: wc.run_seeded_cell(seeded_api, c, merged, plan) for c in BRIDGE_CELLS}
        out['bridge'] = bridge
        out['bridge_report'] = bridge_report(bridge)
    out['dist'] = dg.probe(dist_api, materials, dist_rows, plan.salt_prefix)
    return out


def _diff_stats(a: Mapping[str, Json], b: Mapping[str, Json], metric: str, higher: bool) -> dict[str, Json]:
    x, y = wc.seeded_values(a, metric), wc.seeded_values(b, metric)
    diffs = [((v - u) if higher else (u - v)) / u for u, v in zip(x, y)]
    sd = statistics.stdev(diffs)
    mean = statistics.fmean(diffs)
    return {'pairs': len(diffs), 'mean_diff': mean, 'diff_cv': sd, 'sign_flip_p': d.sign_flip_p(diffs),
            'start_sd_estimate_1df': abs(mean) / math.sqrt(2)}


def compare(a: Mapping[str, Json], b: Mapping[str, Json]) -> dict[str, Json]:
    """Start A against start B on the pilot cells (the bridge ran in A only and is reported, never compared)."""
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
    bridge = sum(c.estimate_s for c in BRIDGE_CELLS)
    return ('DRY RUN seeded A/A pilot (end-of-restore add-on, no extra HOLD). Start A = the last control start, start B = '
            'restored production between W8 check and W8 close. Each start: seeded cells ' + ', '.join(cells) +
            f' (6 seeds x {wc.SEEDED_MAX_TOKENS} tokens, registered seeds), then the distributional probe ('
            + ','.join(DIST_SUBSET) + ' subset, 8 fixed positions per prompt). PASS = identical B1 streams and round counts. '
            'Start A also runs the DP-2 bridging cell (report only): ' + ', '.join(c.name for c in BRIDGE_CELLS)
            + f' (3 seeded pairs each, ESTIMATED {bridge / 60:.1f} min of cells plus the summary-32K rebuild).')


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
    r.add_argument('--start', choices=('A', 'B'), required=True,
                   help='A = the last control start (runs the DP-2 bridge too), B = restored production')
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
    api = BridgeHttp(args.url)
    bridge = bench2_texts(api, args.data_dir) if args.start == 'A' else None
    result = run(api, dg.Http(args.url), texts, plan, materials, {k: rows[k] for k in sorted(keys)},
                 start=args.start, bridge_texts=bridge)
    with args.out.open('x') as handle:
        handle.write(json.dumps(result, allow_nan=False) + '\n')
    print(f'aa_pilot run: wrote {args.out} (numbers, digests and logprob vectors only)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
