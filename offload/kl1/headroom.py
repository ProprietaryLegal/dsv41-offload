"""KL1 item 4: ESTIMATED decode-step headroom per candidate at B1 and B4, from items 1-3's outputs only.

  PYTHONPATH=. python3 -m offload.kl1.headroom --bandwidth KL1-bandwidth.json --moe KL1-moe-phase.json \
      --collectives KL1-collectives.json --output KL1-headroom.json

Every range is ESTIMATED arithmetic on MEASURED (profile-affected) kineto times and ESTIMATED bytes; nothing here is a
donor or upstream number. Inputs are means over both ranks and every listed cell of both SP1 windows; each candidate's
range is [low, high] ms per step and the same as a fraction of the cell's step period. Ranges do not add.

Reference streaming rate `ref` = the LM head's exl3_mrow K6 launch (248 MB, one launch, DRAM-cold) in the B1 cells:
the fastest EXL3 weight stream production runs, MEASURED time / ESTIMATED bytes.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from offload.kl1 import bandwidth as B
from offload.kl1 import moe_phase as M
from offload.kl1 import shapes as S
from offload.prefill_maxlen.evidence import Json, number, record

B1_CELLS = ('prof-short-b1:decode', 'prof-32k-b1:decode', 'prof-150k-b1:decode')
B4_CELLS = ('prof-32k-b4:decode',)


def _rank_means(doc: dict[str, Json], cells: Sequence[str], key: str) -> list[float]:
    """The key's per-rank (or per-pair) means over every window and listed cell."""
    out: list[float] = []
    for caps in record(doc['windows']).values():
        for cell in cells:
            cap = record(record(caps).get(cell) or {})
            if not cap:
                continue
            if 'ranks' in cap:
                out += [number(record(record(r)['mean'])[key]) for r in record(cap['ranks']).values()]
            else:
                out.append(number(record(cap['mean'])[key]))
    if not out:
        raise ValueError(f'{key}: no values in {cells}')
    return out


def _m(doc: dict[str, Json], cells: Sequence[str], key: str) -> float:
    return statistics.fmean(_rank_means(doc, cells, key))


def cell_inputs(bw: dict[str, Json], moe: dict[str, Json], coll: dict[str, Json], cells: Sequence[str]) -> dict[str, float]:
    return {'period_ms': _m(bw, cells, 'period_ms'), 'rows': _m(bw, cells, 'rows'),
            'moe_phase_ms': _m(moe, cells, 'phase_ms'), 'prefix_ms': _m(moe, cells, 'pre_ms'),
            'stream_ms': _m(moe, cells, 'gu_ms') + _m(moe, cells, 'tail_ms'),
            'shared_reread_mb': _m(moe, cells, 'shared_reread_mb_per_step'),
            'dense_ms': _m(bw, cells, 'dense.all.us_per_step') / 1e3, 'dense_mb': _m(bw, cells, 'dense.all.mb_per_step'),
            'head_ms': _m(bw, cells, 'dense.lm_head.us_per_step') / 1e3,
            'coll_intrinsic_ms': _m(coll, cells, 'sum_intrinsic_ms') + _m(coll, cells, 'stage_mean_ms')
            + _m(coll, cells, 'gather_min_ms'),
            'imbalance_ms': _m(coll, cells, 'attn_imbalance_ms') + _m(coll, cells, 'moe_imbalance_ms'),
            'unique_240': _m(bw, cells, 'smoe.unique_max_at_240')}


def ep_imbalance_ms(cell: dict[str, float]) -> float:
    """ESTIMATED time EP loses per step to splitting a layer's U distinct experts between two ranks by expert id: the
    slower rank carries E|X - U/2| extra experts, X ~ Binomial(U, 1/2), E|X - U/2| ~ sqrt(U / (2 pi)). U is the
    bandwidth-consistent count at 240 GB/s (larger U, smaller relative imbalance: the optimistic side)."""
    u = max(cell['unique_240'], 1.0)
    return cell['stream_ms'] * math.sqrt(u / (2 * math.pi)) / (u / 2)


def reference(bw: dict[str, Json]) -> tuple[float, float]:
    xs = _rank_means(bw, B1_CELLS, 'dense.lm_head.gbps')
    return min(xs), max(xs)


def per_byte_slack_b1(b1: dict[str, float], b4: dict[str, float]) -> float:
    """1 - (B4/B1 streaming-time ratio) / (smallest B4/B1 byte ratio over distinct-expert counts 4..20 under independent
    per-request routing, shared runs deduped in L2): B1's per-byte slack against B4 if the four requests route
    independently. 0 when the requests share enough experts to explain the time ratio."""
    time_ratio = b4['stream_ms'] / b1['stream_ms']
    byte_ratio = min(M.b4_over_b1_bytes(u, b1['rows'], b4['rows'], 4, True) for u in range(4, 21))
    return max(0.0, 1 - time_ratio / byte_ratio)


def candidates(b1: dict[str, float], b4: dict[str, float], ref: tuple[float, float]) -> list[dict[str, Json]]:
    ref_lo, ref_hi = ref
    slack = per_byte_slack_b1(b1, b4)
    shared_stream_b1 = S.LAYERS * B.shared_bytes() / ref_lo / 1e6            # ms at the reference rate
    rows: list[dict[str, Json]] = []

    def add(cid: str, mech: str, lo1: float, hi1: float, lo4: float, hi4: float, exact: str, cost: str) -> None:
        rows.append({'id': cid, 'mechanism': mech,
                     'b1_ms': [lo1, hi1], 'b1_frac': [lo1 / b1['period_ms'], hi1 / b1['period_ms']],
                     'b4_ms': [lo4, hi4], 'b4_frac': [lo4 / b4['period_ms'], hi4 / b4['period_ms']],
                     'exactness': exact, 'cost': cost})

    add('smoe-sweep', 'PIPE / OBI / blocks a SM / run-flag sleep / CTA-tail grid shaping of smoe_gu and smoe_down. '
        'High B1 = expert-stream time x the per-byte slack B1 shows against B4 under independent routing; high B4 = '
        'stream time x the spread between production EXL3 streams (ref_lo/ref_hi).',
        0.0, b1['stream_ms'] * slack, 0.0, b4['stream_ms'] * (1 - ref_lo / ref_hi),
        'bit-exact if each output column keeps its single-block k loop (PIPE, OBI, occupancy, run order); split-K or '
        'a different per-token pick order is not', 'low: build flags exist (SMOE_PIPE, SMOE_OBI, SMOE_L2HINT, '
        'SMOE_RUNFLAGS_SLEEP); one microbenchmark window')
    add('grouped-multirow', 'Runs of 16 rows instead of 8 (the m16 fragment\'s zero upper half): one read of an expert '
        'for up to 16 rows. B1 has <= 6 rows a step, so every expert is already read once; B4 re-reads the shared '
        'expert ceil(rows/8) times (high = those bytes at ref_lo; 0 if L2 already dedupes the concurrent runs).',
        0.0, 0.0, 0.0, b4['shared_reread_mb'] / ref_lo,
        'bit-exact expected (mma rows are independent, per-row epilogue and pick-order sum unchanged); bitwise oracle '
        'required', 'moderate: act layout, epilogue row loops, staging; one build + one window')
    add('fused-moe-launch', 'smoe_fused (router GEMV + top-k + routed + shared in one persistent launch, <= 8 rows) lets '
        'the shared expert stream beside the router. High = min(serial prefix, shared-expert stream at ref_lo). Not '
        'reachable at B4 (> 8 rows). Kernel exists; deepseek_v41 does not call it (engine wiring needed).',
        0.0, min(b1['prefix_ms'], shared_stream_b1), 0.0, 0.0,
        'source claims bit-identical to router_topk + smoe (tests/test_smoe_fused_gpu.py, not in src-image); verify',
        'low kernel cost, small engine change (call site in deepseek_v41 MoE forward), hash layers excluded')
    add('dense-path', 'exl3_mrow / mrow32 dense launches and the LM head at the reference rate. B1 dense already '
        'streams at ~ref (helped by l2pf); B4 mrow32 at 16-20 rows runs well below it.',
        0.0, max(0.0, b1['dense_ms'] - b1['dense_mb'] / ref_hi),
        0.0, max(0.0, b4['dense_ms'] - b4['dense_mb'] / ref_hi) + max(0.0, b4['head_ms'] - S.LM_HEAD.bytes / ref_hi / 1e6),
        'exact only if the k-slice plan and slice-sum order are kept; switching 17-32 rows to another plan changes '
        'bits (distribution gate)', 'moderate: mrow32 variant work (EXL3_MROW32_VARIANT / KS / ITEMS knobs exist)')
    add('collective-boundary-fusion', 'Fuse stage into the producer epilogue and the sum into its consumer. High = the '
        'intrinsic stage + sum + gather time; rank-imbalance waits and rank 1\'s step-start NCCL broadcast wait are '
        'not addressable by fusion.',
        0.0, b1['coll_intrinsic_ms'], 0.0, b4['coll_intrinsic_ms'],
        'exact if the rank-order bf16/fp32 add and rounding are reproduced', 'high: RDMA ring, graph pointers, faults')
    add('ep-at-b4', 'Expert parallel MoE with TP attention. TP2 already reads each picked expert once (half on each '
        'rank), so EP moves the same bytes, adds dispatch/combine (low: half the intrinsic collective time more) and '
        'splits each layer\'s distinct experts unevenly between the ranks (ep_imbalance_ms).',
        -(ep_imbalance_ms(b1) + b1['coll_intrinsic_ms'] / 2), 0.0, -(ep_imbalance_ms(b4) + b4['coll_intrinsic_ms'] / 2),
        0.0,
        'changes the combine order and the partial-sum split: not bit-exact', 'very high: new loader/bank split, graphs')
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bandwidth', type=Path, required=True)
    ap.add_argument('--moe', type=Path, required=True)
    ap.add_argument('--collectives', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(argv)
    if args.output.exists():
        print(f'{args.output} exists; refusing to overwrite', file=sys.stderr)
        return 1
    bw, moe, coll = (record(cast(Json, json.loads(p.read_text()))) for p in (args.bandwidth, args.moe, args.collectives))
    b1 = cell_inputs(bw, moe, coll, B1_CELLS)
    b4 = cell_inputs(bw, moe, coll, B4_CELLS)
    ref = reference(bw)
    result: dict[str, Json] = {'type': 'kl1-headroom', 'status': 'ESTIMATED from MEASURED kineto times (items 1-3)',
                               'b1': cast(Json, b1), 'b4': cast(Json, b4), 'ref_gbps': [ref[0], ref[1]],
                               'b1_per_byte_slack_vs_b4': per_byte_slack_b1(b1, b4),
                               'candidates': cast(Json, candidates(b1, b4, ref))}
    args.output.write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    for c in cast(list[dict[str, Json]], result['candidates']):
        b1f, b4f = cast(list[float], c['b1_frac']), cast(list[float], c['b4_frac'])
        print(f"{c['id']:28} B1 {100 * b1f[0]:5.1f}..{100 * b1f[1]:5.1f} %   B4 {100 * b4f[0]:5.1f}..{100 * b4f[1]:5.1f} %")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
