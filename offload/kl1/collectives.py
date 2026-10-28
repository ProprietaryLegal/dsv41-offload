"""KL1 item 3: rank balance and the TP all-reduce / all-gather time per decode step, both ranks paired.

  PYTHONPATH=. python3 -m offload.kl1.collectives --window <SP1-W-dir> [--window ...] --output <json>

The two ranks' clocks are not aligned, so nothing here subtracts timestamps across ranks. Steps are paired by iteration
index and annotation; collectives by their order inside the step (both ranks launch the same graph). For each RDMA-ring
all-reduce (exl3_rdma.cu: stage_kernel copies the local partial out, sum_kernel waits for the peer's partial and adds):
  * intrinsic  = the shorter of the two ranks' sum_kernel durations: the late rank finds the peer's data already there,
                 so its sum is the add plus the flag check (an upper bound on the add, a lower bound on nothing);
  * wait       = |d0 - d1|: the early rank polling for the late rank (rank imbalance plus late-rank wire latency).
Compute segments (end of one sum to the next stage, per rank) are paired too: sum |s0 - s1| / 2 is the step time the
ranks lose to imbalance, split into attention and MoE halves. All MEASURED kineto durations (profile-affected).
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

from offload.kl1 import trace_kernels as T
from offload.prefill_maxlen.evidence import Json, number, record


def step_pair(a: T.Step, b: T.Step) -> dict[str, float]:
    ca, cb = T.collective_calls(a), T.collective_calls(b)
    if [k.family for k in ca] != [k.family for k in cb]:
        raise ValueError(f'step {a.index}: collective sequences differ between ranks')
    row: dict[str, float] = {'period_ms': a.period / 1e3, 'rows': float(a.rows)}
    for fam in T.COLLECTIVE:
        row[f'{fam}_r0_ms'] = sum(k.dur for k in ca if k.family == fam) / 1e3
        row[f'{fam}_r1_ms'] = sum(k.dur for k in cb if k.family == fam) / 1e3
        row[f'{fam}_n'] = float(sum(1 for k in ca if k.family == fam))
    sums = [(x.dur, y.dur) for x, y in zip(ca, cb) if x.family == 'coll_sum']
    row['sum_intrinsic_ms'] = sum(min(x, y) for x, y in sums) / 1e3
    row['sum_wait_ms'] = sum(abs(x - y) for x, y in sums) / 1e3
    row['sum_wait_r0_ms'] = sum(x - y for x, y in sums if x > y) / 1e3      # rank 0 waited for rank 1
    row['sum_wait_r1_ms'] = sum(y - x for x, y in sums if y > x) / 1e3
    row['sum_intrinsic_median_us'] = statistics.median(min(x, y) for x, y in sums) if sums else 0.0
    row['stage_mean_ms'] = (row['coll_stage_r0_ms'] + row['coll_stage_r1_ms']) / 2
    row['gather_min_ms'] = sum(min(x.dur, y.dur) for x, y in zip(ca, cb) if x.family == 'coll_gather') / 1e3
    sa, sb = T.compute_segments(a), T.compute_segments(b)
    if [k for k, _ in sa] != [k for k, _ in sb]:
        raise ValueError(f'step {a.index}: compute segments differ between ranks')
    for kind in ('attn', 'moe'):
        pairs = [(x, y) for (k, x), (_, y) in zip(sa, sb) if k == kind]
        row[f'{kind}_seg_r0_ms'] = sum(x for x, _ in pairs) / 1e3
        row[f'{kind}_seg_r1_ms'] = sum(y for _, y in pairs) / 1e3
        row[f'{kind}_imbalance_ms'] = sum(abs(x - y) for x, y in pairs) / 2e3
        row[f'{kind}_r1_slower_frac'] = sum(1 for x, y in pairs if y > x) / len(pairs) if pairs else 0.0
    for tag, s in (('r0', a), ('r1', b)):
        busy = T.union(((k.start, k.end) for k in s.kernels), s.lo, s.hi)
        row[f'idle_{tag}_ms'] = (s.period - busy) / 1e3
    return row


def capture(paths: dict[int, Path]) -> dict[str, Json]:
    if sorted(paths) != [0, 1]:
        raise ValueError('need both ranks')
    sa = [s for s in T.steps(T.load(paths[0])) if s.phase == 'decode']
    sb = [s for s in T.steps(T.load(paths[1])) if s.phase == 'decode']
    rows = [step_pair(a, b) for a, b in T.pair_steps(sa, sb)]
    if not rows:
        raise ValueError('no paired decode steps')
    keys = sorted(rows[0])
    return {'pairs': len(rows), 'sha256': {str(r): hashlib.sha256(p.read_bytes()).hexdigest() for r, p in paths.items()},
            'mean': cast(Json, {k: statistics.fmean(r[k] for r in rows) for k in keys}), 'per_step': cast(Json, rows)}


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
    result: dict[str, Json] = {'type': 'kl1-collectives', 'status': 'MEASURED kineto durations, ranks paired by order',
                               'windows': {}}
    for root in args.window:
        cast(dict[str, Json], result['windows'])[root.name] = window(root)
    args.output.write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
