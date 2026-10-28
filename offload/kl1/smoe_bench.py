"""KL1 stage-B microbenchmark harness (WRITTEN IN STAGE A, NOT RUN): decode-size smoe, fused MoE and dense mrow
kernels on synthetic EXL3 banks with legal-v2a's real per-layer K plan, inside the production image on one Spark,
in a coordinator window with production down. Plan: offload/kl1/STAGE-B-PLAN.md.

Pure part (CPU, tested by test_kl1.py): variants, routing with an exact distinct-expert count, exact weight bytes of a
routing, inversion of a time-vs-distinct curve. GPU part (`gpu` subcommand): imports torch and vllm_exl3_moe from the
image only when run; one process per kernel build (EXL3_SMOE_FLAGS is read when the extension first loads).
Synthetic banks hold random trellis words (EXL3 decode is data-independent arithmetic, so time and bytes match real
weights); no proprietary tensor is loaded, copied or written. Output: numbers and sha256 digests of kernel outputs.

  # CPU: list what the window would run, nothing imported from the image
  python3 -m offload.kl1.smoe_bench plan
  # in the container (coordinator window only), one variant per process:
  EXL3_SMOE_FLAGS='-DSMOE_RUNFLAGS=1' python3 -m offload.kl1.smoe_bench gpu --variant prod \
      --plan /srv/dsv41/legal-v2a/exl3_moe_plan.json --out /kl1-out/prod.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from offload.kl1 import shapes as S

BENCH_KERNEL, BENCH_DRIVER = '7.0.0-1019-nvidia', '580.178.04'
SHARED_PREFETCH_EXPERTS = 1.7  # ESTIMATED S46: shared-expert L2 prefetch biases inferred distinct count low

PROD_FLAGS = '-DSMOE_RUNFLAGS=1'                 # serve.sh:52 / serve_exp.sh:55


@dataclass(frozen=True)
class Variant:
    name: str
    flags: str                                    # EXL3_SMOE_FLAGS for the process

    def check(self) -> None:
        for f in self.flags.split():
            if not f.startswith('-D') or '=' not in f:
                raise ValueError(f'{self.name}: flag {f!r} is not -DNAME=VALUE')
            key, value = f[2:].split('=', 1)
            v = int(value)
            if key == 'SMOE_PIPE' and (v < 2 or v % 2):
                raise ValueError(f'{self.name}: SMOE_PIPE must be even and >= 2 (exl3_smoe.cu static_assert)')
            if key == 'SMOE_OBI' and (v < 1 or (S.H // 128) % v):
                raise ValueError(f'{self.name}: SMOE_OBI must divide H/128 = {S.H // 128} (smoe_register check)')
            if key not in ('SMOE_PIPE', 'SMOE_OBI', 'SMOE_RUNFLAGS', 'SMOE_RUNFLAGS_SLEEP', 'SMOE_L2HINT',
                           'SMOE_EVICT_FIRST'):
                raise ValueError(f'{self.name}: {key} is not a tuning flag of exl3_smoe.cu')


# Build-flag sweep: every variant keeps each output column's single-block k loop and the pick-order sum, so all must
# be bitwise identical to `prod` (the oracle checks it).
VARIANTS: tuple[Variant, ...] = (
    Variant('prod', PROD_FLAGS),
    Variant('pipe4', PROD_FLAGS + ' -DSMOE_PIPE=4'),
    Variant('pipe12', PROD_FLAGS + ' -DSMOE_PIPE=12'),
    Variant('pipe16', PROD_FLAGS + ' -DSMOE_PIPE=16'),
    Variant('obi1', PROD_FLAGS + ' -DSMOE_OBI=1'),
    Variant('obi4', PROD_FLAGS + ' -DSMOE_OBI=4'),
    Variant('l2h256', PROD_FLAGS + ' -DSMOE_L2HINT=256'),
    Variant('sleep64', PROD_FLAGS + ' -DSMOE_RUNFLAGS_SLEEP=64'),
    Variant('norunflags', '-DSMOE_RUNFLAGS=0'),
)


@dataclass(frozen=True)
class Case:
    """One routing shape: `rows` live verify rows split over `requests`, each request's rows picking from `unique`
    distinct experts; `core` of them shared by every request (0 = independent requests)."""
    rows: int
    requests: int
    unique: int
    core: int = 0

    @property
    def label(self) -> str:
        return f'r{self.rows}q{self.requests}u{self.unique}c{self.core}'


def cases() -> list[Case]:
    """B1 rows 3/4/6 (MEASURED SP1 schedule) over a distinct-count sweep that brackets KL1 item 1's bounds (<= 14-19 at
    273 GB/s), and B4 16/20 rows, four requests, independent and with a shared core."""
    out = [Case(r, 1, u) for r in (3, 4, 6) for u in (6, 8, 10, 12, 14, 16, 18, 20, 24) if u <= S.TOP_K * r]
    out += [Case(r, 4, u, c) for r in (16, 20) for u in (6, 9, 12, 15) for c in (0, u // 2)]
    return out


def split_rows(rows: int, requests: int) -> list[int]:
    if requests < 1 or rows < requests:
        raise ValueError('need at least one row per request')
    return [rows // requests + (1 if i < rows % requests else 0) for i in range(requests)]


def routing(case: Case, seed: int, experts: int = S.E, top_k: int = S.TOP_K) -> list[list[int]]:
    """Logical expert ids [rows][top_k]: each row's picks distinct, each request covering exactly `unique` experts
    (the first `core` common to all requests). Deterministic in (case, seed)."""
    per = split_rows(case.rows, case.requests)
    if not top_k <= case.unique <= experts or case.core > case.unique:
        raise ValueError(f'{case.label}: need top_k <= unique <= experts and core <= unique')
    rng = random.Random(f'{case.label}:{seed}')
    core = rng.sample(range(experts), case.core)
    ids: list[list[int]] = []
    for rows_q in per:
        if case.unique > rows_q * top_k:
            raise ValueError(f'{case.label}: {rows_q} rows cannot cover {case.unique} experts')
        rest = rng.sample([e for e in range(experts) if e not in core], case.unique - case.core)
        pool = core + rest
        order = pool[:]
        rng.shuffle(order)
        cursor = 0
        for _ in range(rows_q):
            row: list[int] = []
            while len(row) < top_k:                  # cover the pool first, then refill from it
                e = order[cursor % len(order)] if cursor < len(order) else rng.choice(pool)
                cursor += 1
                if e not in row:
                    row.append(e)
            ids.append(row)
    return ids


def distinct(ids: Sequence[Sequence[int]]) -> int:
    return len({e for row in ids for e in row})


def layer_bytes(ids: Sequence[Sequence[int]], bits: Sequence[tuple[float, float]], rx: int = 8,
                shared_k: float = S.SHARED_K) -> float:
    """Exact weight bytes one smoe call streams for this routing: each expert once per run of <= rx rows (gate + up at
    k_gu, down at k_d), the shared expert once per run of rx live rows (exl3_smoe.cu build_runs, RX)."""
    rows_of: dict[int, int] = {}
    for row in ids:
        for e in row:
            rows_of[e] = rows_of.get(e, 0) + 1
    total = 0.0
    for e, n in rows_of.items():
        k_gu, k_d = bits[e]
        total += math.ceil(n / rx) * (S.expert_gu_bytes(k_gu) + S.expert_down_bytes(k_d))
    return total + S.shared_runs(len(ids), rx) * S.expert_bytes(shared_k)


def monotone_curve(curve: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Cumulative-max fit: noisy or flat wave times are valid; duplicate expert counts are not."""
    pts = sorted(curve)
    if len(pts) < 2 or any(not math.isfinite(x) or not math.isfinite(y) or x < 0 or y <= 0 for x, y in pts):
        raise ValueError('need two finite, positive-time curve points')
    if any(b[0] <= a[0] for a, b in zip(pts, pts[1:])):
        raise ValueError('distinct expert counts must be unique')
    fitted: list[tuple[float, float]] = []
    high = 0.0
    for u, t in pts:
        high = max(high, t)
        fitted.append((u, high))
    return fitted


def flat_stretches(curve: Sequence[tuple[float, float]]) -> list[dict[str, float]]:
    pts = monotone_curve(curve)
    out: list[dict[str, float]] = []
    for (u0, t0), (u1, t1) in zip(pts, pts[1:]):
        if t0 == t1:
            if out and out[-1]['u_high'] == u0 and out[-1]['us'] == t0:
                out[-1]['u_high'] = u1
            else:
                out.append({'u_low': u0, 'u_high': u1, 'us': t0})
    return out


def invert(curve: Sequence[tuple[float, float]], t: float) -> tuple[float, str]:
    """Invert a non-decreasing fit, clamp outside, report non-identifiable flat stretches without dividing by zero."""
    pts = monotone_curve(curve)
    if not math.isfinite(t) or t <= 0:
        raise ValueError('span must be finite and positive')
    if t < pts[0][1]:
        return pts[0][0], 'below'
    if t > pts[-1][1]:
        return pts[-1][0], 'above'
    for flat in flat_stretches(pts):
        if t == flat['us']:
            return (flat['u_low'] + flat['u_high']) / 2, 'flat'
    for (u0, t0), (u1, t1) in zip(pts, pts[1:]):
        if t0 <= t <= t1 and t1 > t0:
            return u0 + (u1 - u0) * (t - t0) / (t1 - t0), 'inside'
    raise AssertionError('unreachable')


def _interp(points: Sequence[tuple[float, float]], x: float) -> float:
    pts = sorted(points)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y0
    return pts[0][1] if x < pts[0][0] else pts[-1][1]


def production_selection(bandwidth: dict[str, Any], window: str, rank: int,
                         window_dir: Path) -> tuple[list[tuple[int, float]], dict[str, str]]:
    """One SP1 window/rank only. Read its saved host log, never infer stack compatibility from a date."""
    if re.fullmatch(r'SP1-W-\d{8}T\d{6}Z', window) is None or rank not in (0, 1) or window_dir.name != window:
        raise ValueError('one SP1 --window and one --rank required')
    host = ('sparkhead', 'sparkworker-wired')[rank]
    logs = sorted((window_dir / 'cmd').glob(f'*-sp1-327-memory-{host}.log'))
    if len(logs) != 1:
        raise ValueError(f'need exactly one saved stack log for {window} rank{rank}')
    lines = logs[0].read_text().splitlines()
    if lines[:2] != [BENCH_KERNEL, BENCH_DRIVER]:
        raise ValueError(f'{window} rank{rank} is cross-stack; only post-FW1/FW2 {BENCH_KERNEL} / {BENCH_DRIVER} qualifies')
    if window not in bandwidth['windows']:
        raise ValueError('selected window absent from bandwidth input')
    steps: list[tuple[int, float]] = []
    for cell, cap in bandwidth['windows'][window].items():
        if cell.endswith('-b1:decode'):
            if str(rank) not in cap['ranks']:
                raise ValueError(f'{cell} lacks selected rank{rank}')
            steps.extend((int(p['rows']), 1e3 * float(p['smoe.span_ms']) / S.LAYERS)
                         for p in cap['ranks'][str(rank)]['per_step'])
    if not steps:
        raise ValueError('no B1 decode steps for selected window/rank')
    return steps, {'window': window, 'rank': str(rank), 'kernel': lines[0], 'driver': lines[1],
                   'stack_log': str(logs[0]), 'stack_log_sha256': hashlib.sha256(logs[0].read_bytes()).hexdigest()}


def analyze(bench: Sequence[dict[str, Any]], production_steps: Sequence[tuple[int, float]], *,
            window: str, rank: int, production_stack: tuple[str, str],
            replicate: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    """Isolated rates at an inferred count, never a measurement of serving bandwidth.

    Inference uses one same-stack SP1 rank/window. The optional OTHER Spark is a hardware replicate evaluated over
    that same count range, not a pool of production steps. Both prod endpoints are required per measured Spark.
    Gate 2 uses medians over in-range steps; 3% gain AND twice the endpoint drift must be beaten on BOTH Sparks.
    """
    if not window or rank not in (0, 1) or production_stack != (BENCH_KERNEL, BENCH_DRIVER):
        raise ValueError('one production window/rank on the post-FW1/FW2 bench stack required')
    if not bench or not production_steps:
        raise ValueError('bench and selected production steps required')
    measured = [*bench, *replicate]
    if len({d.get('window') for d in measured}) != 1 or not measured[0].get('window'):
        raise ValueError('pooled bench windows or missing window identity')
    for docs, expected_rank in ((bench, rank), (replicate, 1 - rank)):
        if any(d.get('rank') != expected_rank for d in docs):
            raise ValueError('pooled bench ranks or wrong selected rank')
    for d in measured:
        if d.get('type') != 'kl1-smoe-bench' or (d.get('kernel'), d.get('driver')) != (BENCH_KERNEL, BENCH_DRIVER):
            raise ValueError('bench type or stack differs from reviewed post-FW1/FW2 stack')
        if any(d.get(k) is None or d.get(k) != measured[0].get(k) for k in ('plan', 'banks', 'seed')):
            raise ValueError('bench plans, bank layers or seeds differ or are missing')
        if d.get('variant') not in {v.name for v in VARIANTS} or d.get('flags') != next(
                v.flags for v in VARIANTS if v.name == d.get('variant')):
            raise ValueError('unrecognised variant or mismatched build flags')
    runs: dict[int, dict[str, dict[str, Any]]] = {}
    for d in measured:
        r = int(d['rank'])
        sample = str(d.get('sample'))
        if (d['variant'] == 'prod' and sample not in ('first', 'last')) or (d['variant'] != 'prod' and sample != 'variant'):
            raise ValueError('prod requires first/last sample, variants require variant sample')
        key = 'prod-' + sample if d['variant'] == 'prod' else str(d['variant'])
        if key in runs.setdefault(r, {}):
            raise ValueError('duplicate variant/production endpoint')
        rows = {str(row['case']): row for row in d['results']}
        if not rows or len(rows) != len(d['results']):
            raise ValueError('empty or duplicate bench cases')
        for row in rows.values():
            for field in ('us_per_layer_median', 'bytes_per_layer', 'distinct'):
                if not math.isfinite(float(row[field])) or float(row[field]) <= 0:
                    raise ValueError('invalid bench time, bytes or expert count')
        runs[r][key] = rows
    for r, by in runs.items():
        if not {'prod-first', 'prod-last'} <= by.keys():
            raise ValueError(f'rank{r}: prod first AND last required')
        first, last = by['prod-first'], by['prod-last']
        if first.keys() != last.keys() or any(first[c]['out_sha256'] != last[c]['out_sha256'] for c in first):
            raise ValueError(f'rank{r}: prod endpoints disagree on cases or output digest')
        for rows in by.values():
            if rows.keys() != first.keys() or any(any(rows[c][k] != first[c][k] for k in
                    ('rows', 'requests', 'distinct', 'bytes_per_layer')) for c in first):
                raise ValueError(f'rank{r}: variant cases/shapes/bytes differ from prod')
    prod = runs[rank]['prod-first']
    baseline = {c: (float(row['us_per_layer_median']) + float(runs[rank]['prod-last'][c]['us_per_layer_median'])) / 2
                for c, row in prod.items()}
    curves = {rows: [(float(r['distinct']), baseline[c]) for c, r in prod.items() if r['rows'] == rows and r['requests'] == 1]
              for rows, _ in production_steps}
    inferred: list[dict[str, Any]] = []
    unsupported: list[int] = []
    flats: dict[str, Any] = {}
    for live_rows, span_us in production_steps:
        pts = curves[live_rows]
        if len(pts) < 2:
            unsupported.append(live_rows)
            continue
        fitted = monotone_curve(pts)
        flats[str(live_rows)] = flat_stretches(fitted)
        u, where = invert(fitted, span_us)
        byte_pts = [(float(r['distinct']), float(r['bytes_per_layer'])) for r in prod.values()
                    if r['rows'] == live_rows and r['requests'] == 1]
        nbytes = _interp(byte_pts, u)
        corrected = u + SHARED_PREFETCH_EXPERTS
        inferred.append({'rows': live_rows, 'span_us': span_us, 'distinct': u, 'where': where,
                         'distinct_prefetch_corrected_estimate': corrected,
                         'prefetch_correction_above_curve': corrected > fitted[-1][0],
                         'isolated_gbps': nbytes / _interp(fitted, u) / 1e3})
    inside = [p for p in inferred if p['where'] == 'inside']
    def median_or_none(values: Sequence[float]) -> float | None:
        return statistics.median(values) if values else None
    oracle: dict[str, Any] = {}
    speed: dict[str, Any] = {}
    per_rank: dict[str, Any] = {}
    for r, by in runs.items():
        a, b = by['prod-first'], by['prod-last']
        oracle[str(r)] = {v: [c for c in a if rows[c]['out_sha256'] != a[c]['out_sha256']]
                          for v, rows in by.items() if not v.startswith('prod-')}
        speed[str(r)] = {v: {c: min(float(a[c]['us_per_layer_median']), float(b[c]['us_per_layer_median'])) /
                                    float(rows[c]['us_per_layer_median']) for c in a}
                        for v, rows in by.items() if not v.startswith('prod-')}
        reading: dict[str, Any] = {}
        for v, variant in by.items():
            if v.startswith('prod-'):
                continue
            gains: list[float] = []
            drifts: list[float] = []
            margins: list[float] = []
            clamped = 0
            for p in inside:
                keys = [c for c in a if a[c]['rows'] == p['rows'] and a[c]['requests'] == 1]
                # Evaluate both the inferred point and the +1.7-expert prefetch sensitivity estimate. Use the worse
                # gain/margin so a shared-expert cache assumption cannot turn a borderline variant into a pass.
                effects: list[tuple[float, float]] = []
                for u in (p['distinct'], p['distinct_prefetch_corrected_estimate']):
                    high = max(float(a[c]['distinct']) for c in keys)
                    if u > high:
                        clamped += 1
                    ts = [_interp([(float(a[c]['distinct']), float(run[c]['us_per_layer_median'])) for c in keys], u)
                          for run in (a, b, variant)]
                    drift = abs(ts[1] - ts[0]) / statistics.mean(ts[:2])
                    gain = 1 - ts[2] / min(ts[:2])  # beat BOTH controls, not only their average
                    effects.append((gain, drift))
                gains.append(min(g for g, _ in effects))
                drifts.append(max(d for _, d in effects))
                margins.append(min(g - 2 * d for g, d in effects))
            gain_med, margin_med = median_or_none(gains), median_or_none(margins)
            passed = (gain_med is not None and margin_med is not None and gain_med > .03 and margin_med > 0
                      and not oracle[str(r)][v] and clamped == 0 and not unsupported)
            reading[v] = {'median_gain_fraction': gain_med, 'median_prod_drift_fraction': median_or_none(drifts),
                          'median_gain_minus_twice_drift': margin_med, 'correction_clamped_evaluations': clamped,
                          'pass_on_this_spark': passed}
        per_rank[str(r)] = reading
    variants = sorted({v for reading in per_rank.values() for v in reading})
    joint = {v: set(per_rank) == {'0', '1'} and all(v in per_rank[r] and per_rank[r][v]['pass_on_this_spark']
                                                    for r in ('0', '1')) for v in variants}
    return {'production_window': window, 'production_rank': rank, 'bench_window': measured[0]['window'],
            'kernel': BENCH_KERNEL, 'driver': BENCH_DRIVER,
            'rate_label': "smoe isolated rate at production's inferred expert count",
            'rate_limit': 'Read from the synthetic bench curve; this is not independently measured production bandwidth.',
            'production': inferred, 'oracle_mismatches': oracle, 'speed_vs_prod': speed,
            'flat_stretches': flats, 'unsupported_step_rows': unsupported,
            'in_range_steps': len(inside), 'clamped_steps': sum(p['where'] in ('below', 'above') for p in inferred),
            'flat_steps': sum(p['where'] == 'flat' for p in inferred),
            'median_isolated_gbps_in_range': median_or_none([p['isolated_gbps'] for p in inside]),
            'shared_prefetch_bias': {'estimated_experts_low': SHARED_PREFETCH_EXPERTS,
                'treatment': 'Report +1.7-expert correction and require gain at both raw/corrected points; an estimate, not a measured bound.'},
            'gate2_per_rank': per_rank, 'gate2_variant_pass_both_sparks': joint,
            'gate2_status': 'EVALUATED' if set(per_rank) == {'0', '1'} and inside and not unsupported else 'INCOMPLETE',
            'status': 'MEASURED isolated synthetic microbench; ESTIMATED production expert counts (kineto overhead retained)'}


def plan_summary() -> dict[str, Any]:
    vs = list(VARIANTS)
    for v in vs:
        v.check()
    cs = cases()
    return {'variants': [asdict(v) for v in vs], 'cases': [c.label for c in cs], 'n_cases': len(cs),
            'dense_rows': [4, 16, 20], 'mrow32_variants': ['ring', 'lean', 'ring1'], 'fused_rows': list(range(2, 9))}


# ----------------------------------------------------------------------------------------------------------------- GPU


def _digest(tensors: Sequence[Any]) -> str:
    """sha256 over the raw bytes of kernel outputs: the bitwise oracle compares these across variants."""
    import torch
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().contiguous().view(-1).view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def _replay_us(graph: Any, warmup: int, reps: int, calls: int) -> list[float]:   # pragma: no cover - GPU
    import torch
    for _ in range(warmup):
        graph.replay()
    out: list[float] = []
    cuda: Any = torch.cuda
    for _ in range(reps):
        a, b = cuda.Event(enable_timing=True), cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b) * 1e3 / calls)
    return out


def _rand_linear(n_in: int, n_out: int, k: int, gen: Any) -> Any:             # pragma: no cover - GPU
    import types

    import torch
    dev = gen.device
    tr = torch.empty((n_in // 16, n_out // 16, 16 * k), dtype=torch.int16, device=dev)
    tr.random_(-32768, 32767, generator=gen)
    suh = torch.randint(0, 2, (n_in,), generator=gen, device=dev).half() * 2 - 1
    svh = torch.empty(n_out, dtype=torch.half, device=dev).normal_(0.0, 0.02, generator=gen)
    return types.SimpleNamespace(trellis=tr, suh=suh, svh=svh)


def gpu(args: argparse.Namespace) -> int:                            # pragma: no cover - needs the image and a GB10
    import torch
    BK = importlib.import_module('vllm_exl3_moe.bank')
    P = importlib.import_module('vllm_exl3_moe.plan')
    SM = importlib.import_module('vllm_exl3_moe.smoe')

    if os.environ.get('EXL3_SMOE_FLAGS', '') != next(v.flags for v in VARIANTS if v.name == args.variant):
        raise SystemExit(f'EXL3_SMOE_FLAGS does not match variant {args.variant}')
    if args.out.exists():
        raise SystemExit(f'{args.out} exists; refusing to overwrite')
    dev = torch.device('cuda')
    if args.plan:
        plan = P.Exl3MoEPlan.from_json_file(args.plan)
        plan_src = 'file sha256 ' + hashlib.sha256(Path(args.plan).read_bytes()).hexdigest()
    else:
        plan = P.synthetic_plan(counts={float(k): n for k, n in S.K_MIX.items()}, seed=0)
        plan_src = 'synthetic_plan(K_MIX, seed 0)'
    gen = torch.Generator(device=dev).manual_seed(1234)
    layer_ids = [round(i * (S.LAYERS - 1) / (args.banks - 1)) for i in range(args.banks)]
    layers: list[tuple[Any, Any, list[tuple[float, float]]]] = []
    for L in layer_ids:
        lp = plan.layers[L]
        bk = BK.Exl3Bank(lp, S.H, S.I_EXPERT, S.TP, args.rank)
        t = bk.allocate(dev)
        t['w13_trellis'].random_(-32768, 32767, generator=gen)
        t['w2_trellis'].random_(-32768, 32767, generator=gen)
        for n in ('w13_suh', 'w2_suh'):
            t[n].copy_(torch.randint(0, 2, t[n].shape, generator=gen, device=dev).half() * 2 - 1)
        for n in ('w13_svh', 'w2_svh'):
            t[n].normal_(0.0, 0.02, generator=gen)
        layer = SM.SmoeLayer(bk, S.TOP_K)
        layer.attach_shared(_rand_linear(S.H, S.I_LOCAL, S.SHARED_K, gen), _rand_linear(S.H, S.I_LOCAL, S.SHARED_K, gen),
                            _rand_linear(S.I_LOCAL, S.H, S.SHARED_K, gen))
        perm = torch.tensor(lp.perm, dtype=torch.int64, device=dev)
        layers.append((layer, perm, [(b.k_gu, b.k_d) for b in lp.bits]))
    results: list[dict[str, Any]] = []
    for case in cases():
        ids_l = routing(case, args.seed)
        ids = torch.tensor(ids_l, dtype=torch.int32, device=dev)
        w = torch.rand((case.rows, S.TOP_K), generator=gen, device=dev)
        w = (w / w.sum(-1, keepdim=True)).float()
        x = (torch.randn((case.rows, S.H), generator=gen, device=dev) * 0.1).to(torch.bfloat16)
        calls = [layers[i % len(layers)] for i in range(S.LAYERS)]
        outs = [lay(x, ids, w, perm, S.SWIGLU_LIMIT, shared=True) for lay, perm, _ in calls]   # eager: oracle
        digest = _digest(outs)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for lay, perm, _ in calls:
                lay(x, ids, w, perm, S.SWIGLU_LIMIT, shared=True)
        times = _replay_us(graph, args.warmup, args.reps, S.LAYERS)   # us per layer
        nbytes = sum(layer_bytes(ids_l, bits) for _, _, bits in calls) / S.LAYERS
        med = sorted(times)[len(times) // 2]
        results.append({'case': case.label, **asdict(case), 'distinct': distinct(ids_l), 'bytes_per_layer': nbytes,
                        'us_per_layer_median': med, 'us_per_layer_min': min(times), 'gbps': nbytes / med / 1e3,
                        'out_sha256': digest})
        del graph
    doc = {'type': 'kl1-smoe-bench', 'variant': args.variant, 'flags': os.environ['EXL3_SMOE_FLAGS'],
           'plan': plan_src, 'banks': layer_ids, 'rank': args.rank, 'seed': args.seed,
           'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'time': time.time(), 'results': results}
    args.out.write_text(json.dumps(doc, indent=1) + '\n')
    return 0


def gpu_fused(args: argparse.Namespace) -> int:                      # pragma: no cover - needs the image and a GB10
    """smoe.fused (router + top-k + routed + shared, one launch) against exl3_router.router_topk + smoe(shared=True)
    on the same inputs, 2..8 rows: bitwise equality of the layer outputs and of the routing, and time per layer."""
    import torch
    BK = importlib.import_module('vllm_exl3_moe.bank')
    P = importlib.import_module('vllm_exl3_moe.plan')
    R = importlib.import_module('vllm_exl3_moe.router')
    SM = importlib.import_module('vllm_exl3_moe.smoe')

    if args.out.exists():
        raise SystemExit(f'{args.out} exists; refusing to overwrite')
    dev = torch.device('cuda')
    gen = torch.Generator(device=dev).manual_seed(4321)
    plan = P.Exl3MoEPlan.from_json_file(args.plan) if args.plan else P.synthetic_plan(
        counts={float(k): n for k, n in S.K_MIX.items()}, seed=0)
    lp = plan.layers[args.layer]
    bk = BK.Exl3Bank(lp, S.H, S.I_EXPERT, S.TP, args.rank)
    t = bk.allocate(dev)
    for n in ('w13_trellis', 'w2_trellis'):
        t[n].random_(-32768, 32767, generator=gen)
    for n in ('w13_suh', 'w2_suh'):
        t[n].copy_(torch.randint(0, 2, t[n].shape, generator=gen, device=dev).half() * 2 - 1)
    for n in ('w13_svh', 'w2_svh'):
        t[n].normal_(0.0, 0.02, generator=gen)
    layer = SM.SmoeLayer(bk, S.TOP_K)
    layer.attach_shared(_rand_linear(S.H, S.I_LOCAL, S.SHARED_K, gen), _rand_linear(S.H, S.I_LOCAL, S.SHARED_K, gen),
                        _rand_linear(S.I_LOCAL, S.H, S.SHARED_K, gen))
    perm = torch.tensor(lp.perm, dtype=torch.int64, device=dev)
    gate = (torch.randn((S.E, S.H), generator=gen, device=dev) * 0.02).to(torch.bfloat16).contiguous()
    bias = (torch.randn(S.E, generator=gen, device=dev) * 0.01).float()
    empty_f, empty_l = torch.empty(0, dtype=torch.float32, device=dev), torch.empty(0, dtype=torch.int64, device=dev)
    rows: list[dict[str, Any]] = []
    for m in range(2, 9):
        x = (torch.randn((m, S.H), generator=gen, device=dev) * 0.1).to(torch.bfloat16).contiguous()
        logits = torch.empty((m, S.E), dtype=torch.float32, device=dev)
        tw = torch.empty((m, S.TOP_K), dtype=torch.float32, device=dev)
        ti = torch.empty((m, S.TOP_K), dtype=torch.int32, device=dev)
        ctr = torch.zeros(1, dtype=torch.int32, device=dev)
        ws = torch.empty(2 * m * S.E, dtype=torch.float32, device=dev)

        def split() -> Any:
            R.ext().router_topk(x, gate, logits, bias, empty_f, empty_l, 0, S.ROUTED_SCALE, tw, ti, ctr, ws)
            return layer(x, ti, tw, perm, S.SWIGLU_LIMIT, shared=True)

        def fused() -> Any:
            return layer.fused(x, gate, bias, perm, S.SWIGLU_LIMIT, S.ROUTED_SCALE, shared=True)

        a = split()
        ids_a, w_a = ti.clone(), tw.clone()
        b, ids_b, w_b = fused()
        same = (_digest([a]) == _digest([b]) and torch.equal(ids_a, ids_b.to(ids_a.dtype))
                and _digest([w_a]) == _digest([w_b.contiguous()]))
        timing: dict[str, float] = {}
        for name, fn in (('split', split), ('fused', fused)):
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(S.LAYERS):
                    fn()
            us = _replay_us(g, args.warmup, args.reps, S.LAYERS)
            timing[name] = sorted(us)[len(us) // 2]
            del g
        rows.append({'rows': m, 'bitwise_equal': same, 'split_us': timing['split'], 'fused_us': timing['fused'],
                     'fused_over_split': timing['fused'] / timing['split']})
    args.out.write_text(json.dumps({'type': 'kl1-fused-bench', 'layer': args.layer, 'rows': rows}, indent=1) + '\n')
    return 0


def gpu_dense(args: argparse.Namespace) -> int:                      # pragma: no cover - needs the image and a GB10
    """The four exl3_mrow groups of a decode layer (DECODE_DENSE shapes, K5, random trellis) at 4 / 16 / 20 rows, each
    EXL3_MROW32_VARIANT (read at registration) and the 16-row kernel up to 16 rows (EXL3_MROW32_MIN_ROWS=17) against
    production's 9: time, GB/s and output digests (exact only within one plan)."""
    import torch
    MR = importlib.import_module('vllm_exl3_moe.mrow')

    if args.out.exists():
        raise SystemExit(f'{args.out} exists; refusing to overwrite')
    dev = torch.device('cuda')
    gen = torch.Generator(device=dev).manual_seed(999)
    q_dim, wkv = S.Q_LORA, S.KV_DIM
    heads_local = S.HEADS * S.HEAD_DIM // S.TP
    wo_in = S.HEADS * S.HEAD_DIM // S.O_GROUPS
    groups_local = S.O_GROUPS // S.TP

    def layer_weights() -> dict[str, list[tuple[Any, int, int]]]:
        return {'wq_a+wkv': [(_rand_linear(S.H, q_dim, S.DENSE_K, gen), 0, 0), (_rand_linear(S.H, wkv, S.DENSE_K, gen), 0, q_dim)],
                'wq_b': [(_rand_linear(q_dim, heads_local, S.DENSE_K, gen), 0, 0)],
                'wo_a': [(_rand_linear(wo_in, S.O_LORA, S.DENSE_K, gen), g * wo_in, g * S.O_LORA) for g in range(groups_local)],
                'wo_b': [(_rand_linear(S.O_GROUPS * S.O_LORA // S.TP, S.H, S.DENSE_K, gen), 0, 0)]}

    banks = [layer_weights() for _ in range(args.banks)]
    widths = {'wq_a+wkv': (S.H, q_dim + wkv), 'wq_b': (q_dim, heads_local),
              'wo_a': (groups_local * wo_in, groups_local * S.O_LORA), 'wo_b': (S.O_GROUPS * S.O_LORA // S.TP, S.H)}
    # Reuse identical input bytes across every variant/min32 setting for a valid bitwise oracle.
    xs_by_m = {m: {n: (torch.randn((m, w[0]), generator=gen, device=dev) * 0.1).to(torch.bfloat16)
                   for n, w in widths.items()} for m in (4, 16, 20)}
    rows_out: list[dict[str, Any]] = []
    for variant in ('ring', 'lean', 'ring1'):
        for min32 in ('9', '17'):
            os.environ['EXL3_MROW32_VARIANT'] = variant
            os.environ['EXL3_MROW32_MIN_ROWS'] = min32
            reg = [{n: MR.MrowGroup([(lin.trellis, lin.suh, lin.svh, xc, oc) for lin, xc, oc in lins], S.DENSE_K)
                    for n, lins in bank.items()} for bank in banks]
            for m in (4, 16, 20):
                xs = xs_by_m[m]
                outs = {n: torch.empty((m, w[1]), dtype=torch.bfloat16, device=dev) for n, w in widths.items()}
                eager = []
                for r in reg:
                    for n, grp in r.items():
                        eager.append(grp(xs[n], outs[n]).clone())
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for i in range(S.LAYERS):
                        for n, grp in reg[i % len(reg)].items():
                            grp(xs[n], outs[n])
                us = _replay_us(g, args.warmup, args.reps, S.LAYERS)
                del g
                med = sorted(us)[len(us) // 2]
                nbytes = sum(lin.bytes for lin in S.DECODE_DENSE)
                rows_out.append({'variant': variant, 'min32': int(min32), 'rows': m, 'us_per_layer': med,
                                 'gbps': nbytes / med / 1e3, 'out_sha256': _digest(eager),
                                 'info32': [{n: list(MR.ext().info32(grp.handle)) for n, grp in r.items()}
                                            for r in reg]})
    args.out.write_text(json.dumps({'type': 'kl1-dense-bench', 'rows': rows_out}, indent=1) + '\n')
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('plan', help='print the variants and cases the window would run (CPU, no image imports)')
    an = sub.add_parser('analyze', help='after the window (CPU): oracle, speed and production inference')
    an.add_argument('--bench', type=Path, nargs='+', required=True, help='gpu outputs, one per variant')
    an.add_argument('--bandwidth', type=Path, required=True, help='KL1-bandwidth.json')
    an.add_argument('--out', type=Path, required=True)
    an.add_argument('--window', required=True, help='one SP1-W-UTC window, never pooled')
    an.add_argument('--rank', required=True, type=int, choices=(0, 1))
    an.add_argument('--window-dir', type=Path, help='saved SP1 directory; default bandwidth parent / window')
    an.add_argument('--replicate-bench', type=Path, nargs='+', default=[], help='other Spark, SAME bench window; required for Gate 2 pass')
    g = sub.add_parser('gpu', help='run one variant in the production image (coordinator window only)')
    g.add_argument('--variant', required=True, choices=[v.name for v in VARIANTS])
    g.add_argument('--plan', help='legal-v2a exl3_moe_plan.json (read only); default: synthetic plan with K_MIX')
    g.add_argument('--banks', type=int, default=8, help='distinct synthetic layers cycled over 40 calls (defeats L2)')
    g.add_argument('--rank', type=int, default=0, choices=(0, 1))
    g.add_argument('--seed', type=int, default=0)
    g.add_argument('--warmup', type=int, default=5)
    g.add_argument('--reps', type=int, default=30)
    g.add_argument('--out', type=Path, required=True)
    f = sub.add_parser('gpu-fused', help='smoe.fused vs router_topk + smoe, 2..8 rows (coordinator window only)')
    f.add_argument('--plan')
    f.add_argument('--layer', type=int, default=10)
    d = sub.add_parser('gpu-dense', help='exl3_mrow / mrow32 dense groups at 4 / 16 / 20 rows (coordinator window only)')
    d.add_argument('--banks', type=int, default=8)
    for p in (f, d):
        p.add_argument('--rank', type=int, default=0, choices=(0, 1))
        p.add_argument('--warmup', type=int, default=5)
        p.add_argument('--reps', type=int, default=30)
        p.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == 'plan':
        json.dump(plan_summary(), sys.stdout, indent=1)
        sys.stdout.write('\n')
        return 0
    if args.cmd == 'analyze':
        if args.out.exists():
            raise SystemExit(f'{args.out} exists; refusing to overwrite')
        bw = json.loads(args.bandwidth.read_text())
        try:
            steps, provenance = production_selection(bw, args.window, args.rank,
                args.window_dir or args.bandwidth.parent / args.window)
            result = analyze([json.loads(p.read_text()) for p in args.bench], steps,
                window=args.window, rank=args.rank, production_stack=(provenance['kernel'], provenance['driver']),
                replicate=[json.loads(p.read_text()) for p in args.replicate_bench])
            result['production_provenance'] = provenance
        except (ValueError, KeyError, OSError) as exc:
            raise SystemExit(f'analysis refused: {exc}') from exc
        args.out.write_text(json.dumps(result, indent=1) + '\n')
        return 0
    return {'gpu': gpu, 'gpu-fused': gpu_fused, 'gpu-dense': gpu_dense}[args.cmd](args)


if __name__ == '__main__':
    raise SystemExit(main())
