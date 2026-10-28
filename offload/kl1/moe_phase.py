"""KL1 item 2: what fills the MoE phase of a decode step, from both SP1 windows' traces (both ranks).

  PYTHONPATH=. python3 -m offload.kl1.moe_phase --window <SP1-W-dir> [--window ...] --output <json>

Per layer (MEASURED kineto, profile-affected), from the end of the attention all-reduce's sum to the end of smoe_down:
  pre      sum end -> smoe_gu start: router GEMV, routing top-k, mHC pre-norm, smoe_rot (+ launch latency)
  gu       smoe_gu start -> end (its blocks wait for rot first: PDL, exl3_smoe.cu:462)
  tail     smoe_gu end -> smoe_down end: down work not hidden under gate|up (SMOE_RUNFLAGS=1 in production)
  idle     time inside [sum end, down end] with no kernel on any stream
Grid facts per launch: active smoe_gu blocks = runs x I_local/128 (blocks past the run count exit at once,
exl3_smoe.cu:464); resident capacity = 48 SMs x kineto's activeBlocksPerMultiprocessor.
Shared-expert re-reads (ESTIMATED): ceil(live rows / 8) runs each read its 11.1 MB (exl3_smoe.cu RX = 8).
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

from offload.kl1 import bandwidth as B
from offload.kl1 import shapes as S
from offload.kl1 import trace_kernels as T
from offload.prefill_maxlen.evidence import Json, number, record

SMS = 48                                         # GB10 (trace deviceProperties numSms)


def layer_parts(step: T.Step) -> list[dict[str, float]]:
    """Per smoe layer: pre / gu / tail / idle microseconds (see module doc)."""
    sums = [k for k in step.kernels if k.family == 'coll_sum']
    out: list[dict[str, float]] = []
    for L in T.moe_layers(step):
        before = [k for k in sums if k.end <= L.start]
        if not before:
            continue                                # first layer of a capture without its attention all-reduce
        t0 = before[-1].end
        t1 = max(L.down.end, L.gu.end)
        busy = T.union(((k.start, k.end) for k in step.kernels if k.end > t0 and k.start < t1), t0, t1)
        busy_no_l2pf = T.union(((k.start, k.end) for k in step.kernels
                                if k.family != 'l2pf' and k.end > t0 and k.start < t1), t0, t1)
        out.append({'pre_us': L.gu.start - t0, 'gu_us': L.gu.dur, 'tail_us': L.down_after_gu,
                    'phase_us': t1 - t0, 'idle_us': (t1 - t0) - busy, 'idle_no_l2pf_us': (t1 - t0) - busy_no_l2pf,
                    'rot_us': L.rot.dur if L.rot is not None else 0.0,
                    'router_us': L.router.dur if L.router is not None else 0.0,
                    'topk_us': L.topk.dur if L.topk is not None else 0.0,
                    'gu_grid': float(L.gu.grid[0]), 'down_grid': float(L.down.grid[0]),
                    'gu_resident': SMS * (L.gu.blocks_per_sm or 0.0),
                    'down_resident': SMS * (L.down.blocks_per_sm or 0.0)})
    return out


def step_row(step: T.Step) -> dict[str, float]:
    parts = layer_parts(step)
    if not parts:
        raise ValueError(f'step {step.index}: no MoE layer with a preceding all-reduce')
    row: dict[str, float] = {'period_ms': step.period / 1e3, 'rows': float(step.rows), 'layers': float(len(parts))}
    for key in ('pre_us', 'gu_us', 'tail_us', 'phase_us', 'idle_us', 'idle_no_l2pf_us', 'rot_us', 'router_us', 'topk_us'):
        row[key.replace('_us', '_ms')] = sum(p[key] for p in parts) / 1e3
    for key in ('gu_grid', 'down_grid', 'gu_resident', 'down_resident'):
        row[key] = statistics.fmean(p[key] for p in parts)
    runs_sh = S.shared_runs(step.rows)
    row['shared_runs'] = float(runs_sh)
    row['shared_reread_mb_per_step'] = (runs_sh - 1) * B.shared_bytes() * S.LAYERS / 1e6 if runs_sh > 1 else 0.0
    # distinct-run block counts the resident capacity allows in one wave of smoe_gu
    row['gu_runs_one_wave'] = row['gu_resident'] / (S.I_LOCAL // 128)
    return row


def b4_over_b1_bytes(u: float, rows_b1: float, rows_b4: float, requests: int, shared_l2_dedup: bool) -> float:
    """ESTIMATED ratio of a layer's smoe weight bytes at B4 to B1 when one request's verify rows pick `u` distinct
    experts at rows_b1 (scaled by the uniform model to rows_b4 / requests rows), requests route independently of each
    other, and the shared expert is read once per 8-row run (or once when its concurrent runs hit L2)."""
    gu, dn = B.expert_mean_bytes()
    per = gu + dn
    u4 = u * S.unique_uniform(rows_b4 / requests) / S.unique_uniform(rows_b1)
    union4 = S.E * (1 - (1 - u4 / S.E) ** requests)
    sh1 = S.shared_runs(round(rows_b1))
    sh4 = 1 if shared_l2_dedup else S.shared_runs(round(rows_b4))
    return (union4 * per + sh4 * B.shared_bytes()) / (u * per + sh1 * B.shared_bytes())


def capture(paths: dict[int, Path]) -> dict[str, Json]:
    ranks: dict[str, Json] = {}
    for rank, path in sorted(paths.items()):
        steps = [s for s in T.steps(T.load(path)) if s.phase == 'decode']
        rows = [step_row(s) for s in steps]
        if not rows:
            raise ValueError(f'{path.name}: no decode steps')
        keys = sorted(rows[0])
        ranks[str(rank)] = {'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'steps': len(rows),
                            'mean': cast(Json, {k: statistics.fmean(r[k] for r in rows) for k in keys}),
                            'per_step': cast(Json, rows)}
    return {'ranks': ranks}


def window(root: Path) -> dict[str, Json]:
    manifest = [record(x) for x in cast(list[Json], json.loads((root / 'traces-manifest.json').read_text()))]
    groups: dict[str, dict[int, Path]] = {}
    for row in manifest:
        if row['capture'] != 'decode':
            continue
        path = (root / str(row['file'])).resolve()
        if root.resolve() not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'{path.name}: outside the window or sha256 differs from the manifest')
        groups.setdefault(f"{row['cell']}:decode", {})[int(number(row['rank']))] = path
    return {label: capture(paths) for label, paths in sorted(groups.items())}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--window', type=Path, action='append', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(argv)
    if args.output.exists():
        print(f'{args.output} exists; refusing to overwrite', file=sys.stderr)
        return 1
    result: dict[str, Json] = {'type': 'kl1-moe-phase', 'status': 'MEASURED kineto spans (profile-affected); '
                               'shared re-read bytes ESTIMATED', 'windows': {}}
    for root in args.window:
        cast(dict[str, Json], result['windows'])[root.name] = window(root)
    args.output.write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
