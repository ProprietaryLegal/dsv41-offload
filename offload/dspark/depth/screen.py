"""CPU-only aggregate depth sensitivity model. Never a measured static-depth fit.

Reads bench2 records, emits only aggregate counters/timings and file hashes.
No network, torch, tokens, prompts, telemetry or captured tensors are exported.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

VERIFY = (28.79, 31.86, 34.44, 37.76, 41.56, 43.68)  # historical CONFIG, not new measurements
# MEASURED: offload/results/n2-pair-w5/summary.json, k2-tp2, k5/n5 graph p50.
DRAFT = (2.0275, 2.5122, 2.8961, 3.7219)


def obj(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError('expected JSON object')
    return cast(dict[str, object], value)


def number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('expected finite number')
    return float(value)


def integer(value: object) -> int:
    v = number(value)
    if v != int(v):
        raise ValueError('expected integer')
    return int(v)


def vector(value: object, size: int) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f'expected {size} values')
    return [number(v) for v in value]


@dataclass(frozen=True)
class Run:
    rounds: float
    counts: tuple[float, ...]
    step_ms: float


def read_suite(paths: list[Path], suite: str) -> list[dict[str, object]]:
    cells: dict[tuple[int, int], list[Run]] = defaultdict(list)
    sources: dict[tuple[int, int], set[tuple[str, str]]] = defaultdict(set)
    for path in paths:
        records = [obj(json.loads(s)) for s in path.read_text().splitlines() if s.strip()]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for r in records:
            if r.get('type') != 'run':
                continue
            if r.get('error') is not None:
                raise ValueError(f'{path}: failed run')
            settings = obj(r['settings'])
            if number(settings['temperature']) != 1 or number(settings['top_p']) != 1:
                raise ValueError(f'{path}: requires T=1, top_p=1')
            context, batch = integer(r['target_ctx']), integer(r['concurrency'])
            if r.get('mixed') is True:
                continue  # explicitly excluded by the steady full-batch screen contract
            if not 1 <= batch <= 4 or r.get('mixed') is not False:
                raise ValueError(f'{path}: requires unmixed B1..B4')
            spec = obj(r['spec'])
            rounds = number(spec['rounds'])
            if rounds <= 0:
                raise ValueError(f'{path}: no request-rounds')
            counts = obj(spec['accepted_per_position'])
            # Missing positions are not interpreted as zero: evidence must be explicit.
            cs = tuple(number(counts[str(i)]) for i in range(5))
            if any(c < 0 or c > rounds for c in cs) or any(b > a for a, b in zip(cs, cs[1:])):
                raise ValueError(f'{path}: invalid prefix counters')
            if not math.isclose(sum(cs), number(spec['accepted_tokens']), abs_tol=1e-6):
                raise ValueError(f'{path}: position counters do not sum to accepted tokens')
            reqs = [q for q in records if q.get('type') == 'request' and q.get('run_id') == r['run_id']]
            if len(reqs) != batch or any(q.get('error') is not None for q in reqs):
                raise ValueError(f'{path}: missing/failed requests')
            step = statistics.median(number(obj(q['step_gap_ms'])['p50']) for q in reqs)
            if step <= 0:
                raise ValueError(f'{path}: nonpositive step time')
            key = context, batch
            cells[key].append(Run(rounds, cs, step))
            sources[key].add((path.name, digest))
    out: list[dict[str, object]] = []
    for (context, batch), runs in sorted(cells.items()):
        total = sum(r.rounds for r in runs)
        surv = [sum(r.counts[i] for r in runs) / total for i in range(5)]
        out.append(dict(suite=suite, context=context, batch=batch, runs=len(runs), request_rounds=total,
                        prefix_survival_MEASURED=surv,
                        step_p50_median_MEASURED=statistics.median(r.step_ms for r in runs),
                        step_p50_min_MEASURED=min(r.step_ms for r in runs),
                        step_p50_max_MEASURED=max(r.step_ms for r in runs),
                        per_run_survival_MEASURED=[[c/r.rounds for c in r.counts] for r in runs],
                        sources_MEASURED=[dict(file=p, sha256=h) for p, h in sorted(sources[context,batch])]))
    return out


def expected(survival: list[float], depth: int) -> float:
    if not 0 <= depth <= 5:
        raise ValueError('supported depth is 0..5; no extrapolation above trained maximum')
    if len(survival) != 5 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in survival):
        raise ValueError('invalid prefix survival')
    if any(b > a for a,b in zip(survival,survival[1:])):
        raise ValueError('prefix survival must not increase')
    return 1 + sum(survival[:depth])


def rate(survival: list[float], depth: int, batch: int, anchor_ms: float,
         slope_scale: float, overlap: float, overhead: float) -> float:
    """ESTIMATED nominal-cap anchor. See ESTIMATES.md for non-identifiability limits."""
    if not 1 <= batch <= 4 or anchor_ms <= 0 or not 0 <= overlap <= 1 or overhead < 0:
        raise ValueError('invalid timing assumptions')
    if not 0 <= depth <= 5:
        raise ValueError('supported depth is 0..5')
    cap = 5 if batch <= 2 else 4
    draft = DRAFT[batch-1]
    # The measured anchor already contains remote work. Remove central exposed cost ONCE.
    central_exposed = .25*draft + .3
    verify = anchor_ms - central_exposed + slope_scale*batch*(VERIFY[depth]-VERIFY[cap])
    cost = verify + (1-overlap)*draft + overhead
    if cost <= 0:
        raise ValueError('nonpositive model cost')
    return 1000*batch*expected(survival,depth)/cost


def candidates(cell: dict[str, object]) -> list[dict[str, object]]:
    batch = integer(cell['batch'])
    s = vector(cell['prefix_survival_MEASURED'],5)
    if not isinstance(cell['per_run_survival_MEASURED'],list):
        raise ValueError('expected per-run arrays')
    sruns = [vector(v,5) for v in cell['per_run_survival_MEASURED']]
    anchor = number(cell['step_p50_median_MEASURED'])
    rows: list[dict[str, object]] = []
    for k in range(1,9):
        if k > 5:
            rows.append(dict(k=k, status='UNSUPPORTED trained block / HELLO maximum five'))
            continue
        central = rate(s,k,batch,anchor,.5,.75,.3)
        sensitivities = [rate(v,k,batch,t,m,o,h) for v,t,m,o,h in itertools.product(
            sruns, (number(cell['step_p50_min_MEASURED']),number(cell['step_p50_max_MEASURED'])),
            (0.,1.), (0.,1.), (0.,.6))]
        cap = 5 if batch <= 2 else 4
        gains = [100*(rate(v,k,batch,t,m,o,h)/rate(v,cap,batch,t,m,o,h)-1)
                 for v,t,m,o,h in itertools.product(sruns,
                     (number(cell['step_p50_min_MEASURED']),number(cell['step_p50_max_MEASURED'])),
                     (0.,1.), (0.,1.), (0.,.6))]
        rows.append(dict(k=k, status='ESTIMATED censored-survival nominal-cap screen',
                         tokens_per_request_round_ESTIMATED=expected(s,k),
                         aggregate_tps_ESTIMATED=central,
                         sensitivity_min_ESTIMATED=min(sensitivities),
                         sensitivity_max_ESTIMATED=max(sensitivities),
                         vs_nominal_cap_gain_percent_ESTIMATED=100*(central/rate(s,cap,batch,anchor,.5,.75,.3)-1),
                         gain_sensitivity_min_ESTIMATED=min(gains),
                         gain_sensitivity_max_ESTIMATED=max(gains)))
    return rows


def markdown(cells: list[dict[str, object]]) -> str:
    lines = ['# SP2 aggregate depth screen (generated)', '',
             'MEASURED input summaries; ALL candidate rates are ESTIMATED. See ESTIMATES.md.', '',
             'Rates are ideal steady full-batch decode aggregate tokens/s; divide by B for per stream.',
             'Sensitivity ranges are assumption envelopes, not confidence intervals or achieved speedups.', '',
             '| Suite / context / B | Runs | MEASURED step p50 median, ms | MEASURED prefix survival, positions 1–5 |',
             '|---|---:|---:|---|']
    for c in cells:
        s = vector(c['prefix_survival_MEASURED'],5)
        lines.append(f"| {c['suite']} / {c['context']} / {c['batch']} | {c['runs']} | "
                     f"{number(c['step_p50_median_MEASURED']):.2f} | "+', '.join(f'{v:.3f}' for v in s)+' |')
    lines += ['', '| Suite / context / B | k1 | k2 | k3 | k4 | k5 | k6–8 |', '|---|---:|---:|---:|---:|---:|---|']
    for c in cells:
        cols = []
        for r in candidates(c)[:5]:
            cols.append(f"{number(r['aggregate_tps_ESTIMATED']):.1f} "
                        f"[{number(r['sensitivity_min_ESTIMATED']):.1f}, {number(r['sensitivity_max_ESTIMATED']):.1f}]")
        lines.append(f"| {c['suite']} / {c['context']} / {c['batch']} | "+' | '.join(cols)+' | unsupported |')
    return '\n'.join(lines)+'\n'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-root',type=Path,required=True)
    ap.add_argument('--coordinator-results-root',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    a = ap.parse_args()
    cells: list[dict[str,object]] = []
    for suite in ('W4-E1p','W5-E1p','W5-prod'):
        paths = sorted((a.results_root/suite).glob(f'{suite}-*.jsonl'))
        if not paths:
            raise ValueError(f'missing {suite} suite')
        cells.extend(read_suite(paths,suite))
    k1 = a.coordinator_results_root/'K1'
    paths = [k1/f'R1-{cell}.jsonl' for cell in ('short','c32','c150','cap4')]
    cells.extend(read_suite(paths,'K1-R1'))
    a.out_dir.mkdir(parents=True,exist_ok=True)
    payload = dict(status='ESTIMATED screening only; not a qualified recalibration',
                   measured_worker_source='offload/results/n2-pair-w5/summary.json: k2-tp2 draft_p50_us B1..4 k5_n5',
                   measured_worker_ms=list(DRAFT),
                   estimated_assumptions=dict(nominal_anchor_caps=[5,5,4,4],central_overlap_fraction=.75,
                       overlap_sensitivity=[0,1],central_fixed_overhead_ms=.3,overhead_sensitivity_ms=[0,.6],
                       central_row_slope_scale=.5,row_slope_scale_sensitivity=[0,1]),
                   cells=[dict(c,candidates_ESTIMATED=candidates(c)) for c in cells])
    (a.out_dir/'screen.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    (a.out_dir/'TABLES.md').write_text(markdown(cells))


if __name__ == '__main__':
    main()
