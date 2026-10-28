"""KL1 item 1: achieved bandwidth per kernel, per step, from both SP1 windows' traces (both ranks).

  PYTHONPATH=. python3 -m offload.kl1.bandwidth --window <SP1-W-dir> [--window ...] --output <json>

Every byte count is ESTIMATED (offload/kl1/shapes.py states the formula and each assumption); every duration is
MEASURED by kineto in a profiled window (profile-affected). GB/s = ESTIMATED bytes / MEASURED time. The routed experts'
bytes depend on how many distinct experts a step's rows pick, which no trace records: they are given as a function of
that count, at the uniform-independent count (an upper bound) and at the counts 273 GB/s and the source's 240 GB/s
would allow. Output: numbers, kernel families and hashes only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from offload.gb10_budget.trace_budget import rules, sweep
from offload.kl1 import shapes as S
from offload.kl1 import trace_kernels as T
from offload.prefill_maxlen.evidence import Json, number, record

DECODE_CAPTURES = ('prof-short-b1:decode', 'prof-32k-b1:decode', 'prof-150k-b1:decode', 'prof-32k-b4:decode')
PREFILL_CAPTURES = ('prof-32k-b1:prefill-early', 'prof-150k-b1:prefill-late')
FULL_LAYERS = 20                    # deepseek_v41 replay cut: layers 0-19 run the whole chunk, 20-39 the 128-row suffix
SUFFIX_ROWS = 128


def _mean(xs: Sequence[float]) -> float:
    if not xs:
        raise ValueError('mean of nothing')
    return statistics.fmean(xs)


def expert_mean_bytes() -> tuple[float, float]:
    """(gate|up, down) bytes of an average routed expert on one rank: the K mix weighted by expert counts (assumes a
    pick's K is distributed like the bank's; picks that favour high-K experts move more)."""
    total = sum(S.K_MIX.values())
    gu = sum(n * S.expert_gu_bytes(k) for k, n in S.K_MIX.items()) / total
    dn = sum(n * S.expert_down_bytes(k) for k, n in S.K_MIX.items()) / total
    return gu, dn


def shared_bytes() -> float:
    return S.expert_bytes(S.SHARED_K)


def smoe_bytes(unique: float, live_rows: int) -> float:
    """One layer's smoe weight bytes: `unique` routed experts read once each (<= 8 rows each, exl3_smoe.cu RX) plus the
    shared expert once per run of 8 live rows (no L2 reuse assumed between its runs)."""
    gu, dn = expert_mean_bytes()
    return unique * (gu + dn) + S.shared_runs(live_rows) * shared_bytes()


def unique_at(gbps: float, span_us: float, live_rows: int) -> float:
    """Routed experts a layer could have read in span_us at gbps (shared runs first)."""
    gu, dn = expert_mean_bytes()
    return (gbps * 1e3 * span_us - S.shared_runs(live_rows) * shared_bytes()) / (gu + dn)


def sp1_split(step: T.Step) -> dict[str, float]:
    """SP1's exclusive budget for this step (equal split among active buckets, trace_budget.sweep) twice: with SP1's
    rules (l2pf_kernel is 'routed_moe') and with l2pf as its own bucket. ms."""
    out: dict[str, float] = {}
    for tag, rename in (('sp1', False), ('l2pf_apart', True)):
        spans = [(k.start, k.end, 'l2pf' if rename and k.family == 'l2pf' else k.bucket) for k in step.kernels]
        excl, _, gaps = sweep(spans, step.lo, step.hi)
        for b, v in excl.items():
            out[f'{tag}.{b}'] = v / 1e3
        out[f'{tag}.idle'] = sum(g1 - g0 for g0, g1 in gaps) / 1e3
    # l2pf removed entirely: what is exclusive routed-MoE time when prefetch time on the side stream is not counted
    excl, _, gaps = sweep([(k.start, k.end, k.bucket) for k in step.kernels if k.family != 'l2pf'], step.lo, step.hi)
    for b, v in excl.items():
        out[f'no_l2pf.{b}'] = v / 1e3
    out['no_l2pf.idle'] = sum(g1 - g0 for g0, g1 in gaps) / 1e3
    return out


def moe_wall(step: T.Step) -> dict[str, float]:
    """Wall time of the MoE phase: the union of router, routing top-k and smoe/TF expert kernels, ms per step."""
    fams = ('router', 'route_topk', 'smoe_rot', 'smoe_gu', 'smoe_down', 'smoe_fused', 'tf_gu', 'tf_down', 'tf_aux')
    ks = [k for k in step.kernels if k.family in fams]
    experts = [k for k in ks if k.family not in ('router', 'route_topk')]
    return {'moe_union_ms': T.union(((k.start, k.end) for k in ks), step.lo, step.hi) / 1e3,
            'expert_union_ms': T.union(((k.start, k.end) for k in experts), step.lo, step.hi) / 1e3,
            'router_topk_union_ms': T.union(((k.start, k.end) for k in ks if k not in experts), step.lo, step.hi) / 1e3}


def dense_decode(step: T.Step) -> dict[str, Json]:
    """exl3_mrow launches: four a layer in forward order (wq_a+wkv, wq_b, wo_a, wo_b), K from the template."""
    ks = [k for k in step.kernels if k.family == 'mrow']
    if len(ks) % len(S.DECODE_DENSE):
        raise ValueError(f'step {step.index}: {len(ks)} mrow launches, not a multiple of 4')
    rows: dict[str, Json] = {}
    for j, lin in enumerate(S.DECODE_DENSE):
        mine = ks[j::len(S.DECODE_DENSE)]
        for k in mine:
            if T.template_k(k.name) != lin.k:
                raise ValueError(f'{lin.name}: template K {T.template_k(k.name)} != {lin.k}')
        us = sum(k.dur for k in mine)
        rows[lin.name] = {'launches': len(mine), 'us_per_step': us, 'mb_per_launch': lin.bytes / 1e6,
                          'gbps': S.gbps(lin.bytes * len(mine), us), 'grid': cast(Json, list(mine[0].grid))}
    us = sum(k.dur for k in ks)
    total = sum(lin.bytes for lin in S.DECODE_DENSE) * len(ks) / len(S.DECODE_DENSE)
    rows['all'] = {'launches': len(ks), 'us_per_step': us, 'mb_per_step': total / 1e6, 'gbps': S.gbps(total, us)}
    head = [k for k in step.kernels if k.family == 'mrow_head' or (k.family == 'exl3_gemm' and T.template_k(k.name) == 6)]
    if head:
        hus = sum(k.dur for k in head)
        rows['lm_head'] = {'launches': len(head), 'us_per_step': hus, 'mb_per_step': S.LM_HEAD.bytes / 1e6,
                           'gbps': S.gbps(S.LM_HEAD.bytes, hus), 'kernel': head[0].family}
    for fam, nbytes in (('router', S.ROUTER_BYTES), ('hc_prenorm', S.MHC_BYTES)):
        ks2 = [k for k in step.kernels if k.family == fam]
        if ks2:
            us2 = sum(k.dur for k in ks2)
            rows[fam] = {'launches': len(ks2), 'us_per_step': us2, 'mb_per_launch': nbytes / 1e6,
                         'gbps': S.gbps(nbytes * len(ks2), us2)}
    return rows


def smoe_decode(step: T.Step, requests: int) -> dict[str, Json]:
    layers = T.moe_layers(step)
    if len(layers) != S.LAYERS:
        raise ValueError(f'step {step.index}: {len(layers)} smoe layers')
    m = step.rows
    slots = layers[0].gu.grid[0] // (S.I_LOCAL // 128)
    span = sum(L.span for L in layers)
    gu_us = sum(L.gu.dur for L in layers)
    down_after = sum(L.down_after_gu for L in layers)
    rot_us = sum(L.rot.dur for L in layers if L.rot is not None)
    uni = S.unique_uniform(m)
    per_layer_span = span / len(layers)
    return {'rows': m, 'requests': requests, 'graph_slots': slots, 'graph_rows': slots // (S.TOP_K + 1),
            'span_ms': span / 1e3, 'gu_ms': gu_us / 1e3, 'down_after_gu_ms': down_after / 1e3, 'rot_ms': rot_us / 1e3,
            'shared_runs': S.shared_runs(m),
            'unique_uniform': uni,
            'gbps_if_uniform': smoe_bytes(uni, m) * len(layers) / span / 1e3,
            'gbps_if_6_per_request': smoe_bytes(min(uni, 6.0 * requests), m) * len(layers) / span / 1e3,
            'unique_max_at_273': unique_at(S.GB10_SPEC_GBPS, per_layer_span, m),
            'unique_max_at_240': unique_at(S.SOURCE_READ_GBPS[1], per_layer_span, m)}


def tf_prefill(step: T.Step) -> dict[str, Json]:
    layers = T.moe_layers(step)
    if len(layers) != S.LAYERS:
        raise ValueError(f'prefill step {step.index}: {len(layers)} TF layers')
    gu, dn = expert_mean_bytes()
    out: dict[str, Json] = {}
    for tag, sel, rows in (('full', layers[:FULL_LAYERS], step.rows), ('suffix', layers[FULL_LAYERS:], SUFFIX_ROWS)):
        uni = S.unique_uniform(rows)
        gus = sum(L.gu.dur for L in sel)
        dns = sum(L.down.dur for L in sel)
        out[tag] = {'layers': len(sel), 'rows': rows, 'unique_uniform': uni, 'gu_ms': gus / 1e3, 'down_ms': dns / 1e3,
                    'gu_gbps_if_uniform': S.gbps(uni * gu * len(sel), gus),
                    'down_gbps_if_uniform': S.gbps(uni * dn * len(sel), dns),
                    'span_ms': sum(L.span for L in sel) / 1e3}
    return out


def reconstruct_prefill(step: T.Step) -> dict[str, Json]:
    ks = [k for k in step.kernels if k.family == 'reconstruct']
    by: dict[str, list[T.Kernel]] = {}
    for k in ks:
        by.setdefault('x'.join(map(str, S.reconstruct_shape(k.grid))), []).append(k)
    out: dict[str, Json] = {}
    tb = tus = 0.0
    for shape, group in sorted(by.items()):
        kk = T.template_k(group[0].name)
        if kk is None:
            raise ValueError('reconstruct without template K')
        nbytes = S.reconstruct_bytes(group[0].grid, kk) * len(group)
        us = sum(k.dur for k in group)
        tb += nbytes
        tus += us
        out[shape] = {'launches': len(group), 'K': kk, 'us': us, 'gbps': S.gbps(nbytes, us)}
    if ks:
        out['all'] = {'launches': len(ks), 'us': tus, 'gb': tb / 1e9, 'gbps': S.gbps(tb, tus)}
    return out


def _fmean_rows(rows: Sequence[dict[str, Json]], keys: Sequence[str]) -> dict[str, float]:
    return {k: _mean([number(r[k]) for r in rows]) for k in keys if all(k in r for r in rows)}


def _flatten(d: dict[str, Json], prefix: str = '') -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f'{prefix}{k}.'))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f'{prefix}{k}'] = float(v)
    return out


def capture(paths: dict[int, Path], kind: str) -> dict[str, Json]:
    ranks: dict[str, Json] = {}
    for rank, path in sorted(paths.items()):
        trace = T.load(path)
        steps = [s for s in T.steps(trace) if s.phase == ('decode' if kind == 'decode' else 'prefill')]
        per: list[dict[str, float]] = []
        for s in steps:
            row: dict[str, Json] = {'period_ms': s.period / 1e3, 'rows': s.rows}
            row.update(cast(dict[str, Json], sp1_split(s)))
            row.update(cast(dict[str, Json], moe_wall(s)))
            fam = T.family_totals(s)
            row['family_union_ms'] = {f: v['union_us'] / 1e3 for f, v in fam.items()}
            row['family_sum_ms'] = {f: v['sum_us'] / 1e3 for f, v in fam.items()}
            if kind == 'decode':
                row['dense'] = dense_decode(s)
                row['smoe'] = smoe_decode(s, s.requests)
            else:
                if s.rows < 1024:       # a short tail chunk is not a full-chunk budget
                    continue
                row['tf'] = tf_prefill(s)
                row['reconstruct'] = reconstruct_prefill(s)
            per.append(_flatten(row))
        if not per:
            raise ValueError(f'{path.name}: no complete {kind} steps')
        keys = sorted(set().union(*per))
        ranks[str(rank)] = {'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'steps': len(per),
                            'rows_hist': cast(Json, _hist([p['rows'] for p in per])),
                            'mean': cast(Json, {k: _mean([p.get(k, 0.0) for p in per]) for k in keys}),
                            'per_step': cast(Json, per)}
    return {'kind': kind, 'ranks': ranks}


def _hist(xs: Sequence[float]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        out[str(int(x))] = out.get(str(int(x)), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def window(root: Path) -> dict[str, Json]:
    manifest = [record(x) for x in cast(list[Json], json.loads((root / 'traces-manifest.json').read_text()))]
    groups: dict[str, dict[int, Path]] = {}
    for row in manifest:
        label = f"{row['cell']}:{row['capture']}"
        path = (root / str(row['file'])).resolve()
        if root.resolve() not in path.parents:
            raise ValueError('manifest path leaves the window')
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'{path.name}: sha256 differs from the manifest')
        groups.setdefault(label, {})[int(number(row['rank']))] = path
    out: dict[str, Json] = {}
    for label in (*DECODE_CAPTURES, *PREFILL_CAPTURES):
        if label not in groups:
            continue
        out[label] = capture(groups[label], 'decode' if label.endswith(':decode') else 'prefill')
    return out


def constants() -> dict[str, Json]:
    gu, dn = expert_mean_bytes()
    return {'status': 'ESTIMATED bytes (offload/kl1/shapes.py); MEASURED kineto durations; UPSTREAM CLAIM bandwidths',
            'mean_k': S.mean_k(), 'expert_gu_mb': gu / 1e6, 'expert_down_mb': dn / 1e6,
            'shared_expert_mb': shared_bytes() / 1e6,
            'dense_mb_per_layer': {lin.name: lin.bytes / 1e6 for lin in S.DECODE_DENSE},
            'lm_head_mb': S.LM_HEAD.bytes / 1e6, 'router_mb': S.ROUTER_BYTES / 1e6, 'mhc_mb': S.MHC_BYTES / 1e6,
            'gb10_spec_gbps': S.GB10_SPEC_GBPS, 'source_read_gbps': cast(Json, list(S.SOURCE_READ_GBPS)),
            'source_decode_gbps': cast(Json, list(S.SOURCE_DECODE_GBPS))}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--window', type=Path, action='append', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(argv)
    if args.output.exists():
        print(f'{args.output} exists; refusing to overwrite', file=sys.stderr)
        return 1
    rules()                                         # fail early if buckets.json is unreadable
    result: dict[str, Json] = {'type': 'kl1-bandwidth', 'constants': constants(), 'windows': {}}
    for root in args.window:
        cast(dict[str, Json], result['windows'])[root.name] = window(root)
    args.output.write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

