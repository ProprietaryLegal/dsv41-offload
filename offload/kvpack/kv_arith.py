#!/usr/bin/env python3
"""K1: DeepSeek-V4.1 KV pool arithmetic, transcribed from the v5 allocator (vllm/v1/core/kv_cache_utils.py
get_max_concurrency_for_kv_cache_config + get_kv_cache_capacity; vllm/v1/kv_cache_interface.py
AttentionSpec / SlidingWindowSpec.max_admission_blocks_per_request / CircularBufferSpec) and patches 0057 / 0060.

Shared pool: num_blocks = floor(kv_bytes / stride), stride = the widest group's page sum = 138,240 B with fp8_ds_mla
records (ratio-2 group: 3 x (37,440 B KV page + 8,640 B indexer page) per 128 tokens; v5 and every packing mode).
Per request at max_model_len, blocks drawn from that pool:
  ratio-2 group (block 128 tokens):           cdiv(L, 128)
  ratio-1 group (block 64 | 128 | 192 tokens): cdiv(L, 64 * pages)   pages = 1 (v5), 2 (0057), 3 (0060)
  compressor ring group (CircularBufferSpec):  1
  each SWA group (block 64, window 128):       cdiv(min(127 + in_flight, L), 64) + 1 = 131 with in_flight 8,192
Reported "GPU KV cache size" = int(num_blocks / per_request * L).
Pool pressure (S8R2 FIX-1): the 131-block SWA figure is the admission cap; it sizes admission and the pool, but during a
chunked prefill remove_skipped_blocks frees every SWA block behind the window, so a request holds per SWA group only
cdiv(127 + 1,024 x (1 + steps in flight), 64) = 18 (sync) to 34 (one step in flight) blocks. Preemption is judged from
the full-attention blocks (ratio-2 + ratio-1 groups) plus the ring block plus that held window, checked against S8R2's
runs of the engine's own KVCacheManager (PRESSURE_CHECKS).
Validated against the GB10 logs of 2026-10-22: 989,460 (local, 3.75 GiB), 1,003,083 (capacity, 3.75 GiB),
1,604,926 (capacity, 6.0 GiB) and upstream's 1,346,928 (0057) / 1,531,258 (0060) at 3.75 GiB local.
"""
from math import ceil

STRIDE = 138_240
IN_FLIGHT = 8_192          # 2 x max_num_batched_tokens 4,096
CHUNK = 1_024              # --long-prefill-token-threshold: prompt tokens per request per step (needles at priority 10)
GIB = 2**30


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def full_attention(L: int, pages: int) -> int:
    return cdiv(L, 128) + cdiv(L, 64 * pages)


def per_request(L: int, pages: int, swa_groups: int) -> int:
    swa = cdiv(min(127 + IN_FLIGHT, L), 64) + 1
    return full_attention(L, pages) + 1 + swa_groups * swa


def held(L: int, pages: int, swa_groups: int, steps_in_flight: int) -> int:
    """Blocks one request holds at the end of its chunked prefill: every full-attention block, the ring block, and per
    SWA group only the window behind the processed tokens plus the chunks scheduled since (not the admission cap)."""
    window = cdiv(min(127 + CHUNK * (1 + steps_in_flight), L), 64)
    return full_attention(L, pages) + 1 + swa_groups * window


def pressure(n: int, L: int, pages: int, swa_groups: int, blocks: int) -> tuple[str, int, int, int]:
    """Verdict for n concurrent L-token prompts in a pool of `blocks`, with the full-attention total and the held range
    (sync .. one step in flight)."""
    full = n * full_attention(L, pages)
    lo, hi = n * held(L, pages, swa_groups, 0), n * held(L, pages, swa_groups, 1)
    if full > blocks:
        verdict = "preempts: full attention alone > pool"
    elif lo > blocks:
        verdict = "preempts"
    elif hi > blocks:
        verdict = "preempts with a step in flight"
    else:
        verdict = "fits: no preemption"
    return verdict, full, lo, hi


def capacity(kv_bytes: int, L: int, pages: int, swa_groups: int) -> tuple[int, float, int]:
    blocks = kv_bytes // STRIDE
    conc = blocks / per_request(L, pages, swa_groups)
    return int(conc * L), conc, blocks


CHECKS = [  # (kv_bytes, L, pages, swa_groups, expected tokens, source)
    (4_026_531_840, 327_680, 1, 15, 989_460, "GB10 log, A / C-shadow (local drafter)"),
    (4_026_531_840, 327_680, 1, 14, 1_003_083, "GB10 log, C-capacity"),
    (6_442_450_944, 327_680, 1, 14, 1_604_926, "GB10 log, D-cal1-kv6"),
    (4_026_531_840, 327_680, 2, 15, 1_346_928, "upstream 0057 (image v8)"),
    (4_026_531_840, 327_680, 3, 15, 1_531_258, "upstream 0060 (image v12)"),
]

PRESSURE_CHECKS = [  # (L, pages, steps in flight, preemptions, peak blocks) for 4 needles in 1.5 GiB = 11,650 blocks:
    # S8R2's kvsim runs of the engine's own KVCacheManager (S8R2-evidence/sim-results.txt, sim-240k.txt)
    (120_000, 3, 0, 0, 7_253), (120_000, 3, 1, 0, 8_149), (120_000, 1, 0, 2, None), (120_000, 1, 1, 4, None),
    (180_000, 3, 1, 0, 11_245), (200_000, 3, 1, 6, None), (220_000, 3, 1, 2, None), (220_000, 1, 1, 2, None),
    (240_000, 3, 0, 2, None), (240_000, 3, 1, 2, None), (240_000, 1, 0, 4, None), (240_000, 1, 1, 14, None),
]


def main() -> None:
    bad = 0
    for kv, L, pages, swa, want, src in CHECKS:
        got, _, _ = capacity(kv, L, pages, swa)
        ok = got == want
        bad += not ok
        print(f"check {'OK ' if ok else 'BAD'} {got:>10,} vs {want:>10,}  {src}")
    if bad:
        raise SystemExit(f"{bad} check(s) failed: the transcription is wrong")
    kv = int(1.5 * GIB)
    pool = kv // STRIDE
    for L, pages, steps, pre, peak in PRESSURE_CHECKS:
        got = 4 * held(L, pages, 14, steps)
        ok = (got > pool) == (pre > 0) and (peak is None or peak <= got <= peak * 1.01)
        bad += not ok
        print(f"check {'OK ' if ok else 'BAD'} 4 x {L // 1000}K {'packed' if pages == 3 else 'v5':<6} {steps} in flight: "
              f"holds {got:>6,} of {pool:,} blocks vs engine {pre} preemptions" + (f", peak {peak:,}" if peak else ""))
    if bad:
        raise SystemExit(f"{bad} pressure check(s) failed: the held-window model disagrees with the engine's manager")
    print()
    for pages, mode in ((1, "v5"), (3, "0060")):
        tok, conc, blocks = capacity(kv, 327_680, pages, 14)
        verdict, full4, lo, hi = pressure(4, 240_000, pages, 14, blocks)
        one = per_request(300_300, pages, 14)
        full = full_attention(300_300, pages)
        print(f"pressure R1b {mode:<5} capacity 1.50 GiB: {blocks:,} blocks, {tok:,} tokens; 4 x 240K needles hold {lo:,}-{hi:,} "
              f"blocks (full attention {full4:,} + ring 4 + window held {lo - full4 - 4:,}-{hi - full4 - 4:,}) ({verdict}); "
              f"one 300K request {one:,} blocks, its cached part {full:,}; "
              f"with A and filler 1 cached, filler 2 finds {blocks - 2 * full:,} free blocks for its {one:,} "
              f"({'evicts A (LRU)' if blocks - 2 * full < one else 'no eviction'})")
    print()
    print("mode       drafter   KV GiB/rank  max_len     tokens      x max-len requests  B/token charged")
    for swa, who in ((15, "local"), (14, "capacity")):
        for gib in (3.75, 6.0):
            kv = int(gib * GIB)
            for L in (327_680, 1_048_576):
                for pages, mode in ((1, "v5"), (2, "0057"), (3, "0060")):
                    tok, conc, _ = capacity(kv, L, pages, swa)
                    print(f"{mode:<10} {who:<9} {gib:>5.2f}        {L:>9,}  {tok:>11,}  {conc:>6.2f}              "
                          f"{kv / tok:>7.0f}")


if __name__ == "__main__":
    main()
