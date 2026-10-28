#!/usr/bin/env python3
"""PF2-LOWKV item 1: KV pool and memory arithmetic for 524,288 at a smaller KV reservation (CPU only, no network).

Built on offload/kvpack/kv_arith.py (validated against five GB10/upstream pool figures and S8R2's engine-manager runs) and
the image's admission rule (vllm/v1/core/sched/scheduler.py:1263 scheduler_reserve_full_isl=True by default,
vllm/config/scheduler.py:191; kv_cache_manager.py:519-536): a waiting request is admitted only if its whole prompt fits the
free blocks, with every SWA group charged at its admission cap (SlidingWindowSpec.max_admission_blocks_per_request = 131).
Production K1 = 0060 packing (VLLM_DSV41_KV_PAIR=3), indexer factor 4 (VLLM_INDEXER_PREFILL_BUFFER_FACTOR=4, no pool
effect), remote drafter = 14 SWA groups, --max-num-seqs 4, num_speculative_tokens 5. The pool is ONE aliased buffer of
num_blocks x 138,240 B per rank (overlay kv_cache_utils.py get_kv_cache_config_from_groups: "Groups alias from byte 0"),
so lowering --kv-cache-memory-bytes returns (blocks_H - blocks_L) x 138,240 B per rank to the system (ESTIMATED: on GB10
the pool is ordinary system memory; K1's memprobe showed allocation deltas reaching MemAvailable within about +-10 %).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

from offload.kvpack import kv_arith as k

GIB = 2**30
CAP = 524_288
PAGES = 3                    # VLLM_DSV41_KV_PAIR=3 (0060 packing)
SWA_GROUPS = 14              # remote (capacity) drafter: no local draft SWA group
MAX_SEQS = 4                 # --max-num-seqs
SPEC_TOKENS = 5              # num_speculative_tokens (lookahead slots)
SHORT_SEAT = 32_768          # the two remaining seats are charged as 32K requests (the 32K regression size)
H_KV = 6 * GIB               # production KV_BYTES = the PF1 OFF arm = capacity control H
CANDIDATES = (2 * GIB, 3 * GIB, 4 * GIB)
LOW_KV = 3 * GIB             # the pick (pick() must agree; CPU test)
MIN_MARGIN = .25             # pool >= 1.25 x the two-long need
KV_TOKENS = {H_KV: 2_820_756, LOW_KV: 1_410_348}   # engine "GPU KV cache size" at 524,288 (H MEASURED in PF1 logs)


def step_blocks() -> int:
    """Extra blocks one decode step's lookahead can claim per request: one per full-attention group (ratio-2 at 128 tokens,
    ratio-1 at 64 x PAGES); 1 + 5 + 5 slots never span more than one new block in either group."""
    return k.cdiv(1 + 2 * SPEC_TOKENS, 128) + k.cdiv(1 + 2 * SPEC_TOKENS, 64 * PAGES)


@dataclass(frozen=True)
class Arm:
    kv_bytes: int
    blocks: int
    allocated_bytes: int
    unused_bytes: int
    pool_tokens: int                 # what the engine logs as "GPU KV cache size"
    concurrency: float               # engine-reported max concurrency at 524,288
    two_long_need: int               # blocks: 2 x admission-cap request + lookahead + null block
    two_long_held: int               # blocks two fully prefilled 524,288 requests hold (one step in flight)
    seats_need: int                  # two_long_need + the two other seats at 32K (admission cap)
    margin_blocks: int               # blocks - two_long_need
    margin_ratio: float              # blocks / two_long_need - 1
    fits_two: bool
    fits_seats: bool
    meets_margin: bool
    gain_vs_h_bytes: int             # ESTIMATED MemAvailable gain per rank against H


def arm(kv_bytes: int) -> Arm:
    tokens, conc, blocks = k.capacity(kv_bytes, CAP, PAGES, SWA_GROUPS)
    one = k.per_request(CAP, PAGES, SWA_GROUPS)
    need = 2 * (one + step_blocks()) + 1
    seats = need + (MAX_SEQS - 2) * k.per_request(SHORT_SEAT, PAGES, SWA_GROUPS)
    h_blocks = H_KV // k.STRIDE
    return Arm(kv_bytes=kv_bytes, blocks=blocks, allocated_bytes=blocks * k.STRIDE, unused_bytes=kv_bytes - blocks * k.STRIDE,
               pool_tokens=tokens, concurrency=conc, two_long_need=need,
               two_long_held=2 * k.held(CAP, PAGES, SWA_GROUPS, 1), seats_need=seats, margin_blocks=blocks - need,
               margin_ratio=blocks / need - 1, fits_two=blocks >= need, fits_seats=blocks >= seats,
               meets_margin=blocks >= (1 + MIN_MARGIN) * need, gain_vs_h_bytes=(h_blocks - blocks) * k.STRIDE)


def pick(candidates: tuple[int, ...] = CANDIDATES) -> int:
    """Smallest reservation that admits two 524,288-token requests with >= 25 % block margin AND still admits two 32K
    requests on the other seats (max_num_seqs 4)."""
    ok = [kv for kv in sorted(candidates) if (a := arm(kv)).fits_two and a.fits_seats and a.meets_margin]
    if not ok:
        raise ValueError('no candidate reservation holds two 524,288-token requests with margin')
    return ok[0]


def check_low(kv_bytes: int) -> Arm:
    """Runner/precheck gate for the --low-kv-bytes parameter: below H, a whole number of MiB, and the item-1 criteria."""
    if not 0 < kv_bytes < H_KV or kv_bytes % 2**20:
        raise ValueError(f'LOW reservation {kv_bytes} must be a whole number of MiB below H ({H_KV})')
    a = arm(kv_bytes)
    if not (a.fits_two and a.fits_seats and a.meets_margin):
        raise ValueError(f'LOW reservation {kv_bytes} does not hold two 524,288-token requests + two 32K seats with '
                         f'{MIN_MARGIN:.0%} margin ({a.blocks} blocks, need {a.two_long_need} / {a.seats_need})')
    return a


def expected_tokens(kv_bytes: int) -> int:
    """The engine's logged pool for an arm (rank 0 'GPU KV cache size'); computed, then cross-checked by tests."""
    return arm(kv_bytes).pool_tokens


def table() -> list[dict[str, object]]:
    return [asdict(arm(kv)) for kv in (*CANDIDATES, H_KV)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args(argv)
    rows = table()
    if args.json:
        print(json.dumps({'cap': CAP, 'pick': pick(), 'rows': rows}, indent=1))
        return 0
    one = k.per_request(CAP, PAGES, SWA_GROUPS)
    print(f'524,288 request at admission: {one:,} blocks = full attention {k.full_attention(CAP, PAGES):,} + ring 1 + '
          f'{SWA_GROUPS} SWA x 131; lookahead {step_blocks()} blocks/request; stride {k.STRIDE:,} B')
    print('KV GiB/rank  blocks   pool tokens  conc   2x524K need  +2x32K seats  margin blocks (ratio)  '
          'gain vs H (ESTIMATED)')
    for r in map(arm, (*CANDIDATES, H_KV)):
        verdict = ('PICK' if r.kv_bytes == pick() else 'ok' if r.fits_two and r.fits_seats and r.meets_margin else
                   'too small' if r.kv_bytes < H_KV else 'H control')
        print(f'{r.kv_bytes / GIB:>10.2f}  {r.blocks:>6,}  {r.pool_tokens:>11,}  {r.concurrency:>4.2f}  {r.two_long_need:>11,}  '
              f'{r.seats_need:>12,}  {r.margin_blocks:>7,} ({r.margin_ratio:>+6.1%})  '
              f'{r.gain_vs_h_bytes / GIB:>5.3f} GiB  {verdict}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
