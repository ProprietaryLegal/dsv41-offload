# PF2-LOWKV item 1: 524,288 with a smaller KV reservation, the arithmetic

PF2-LOWKV lane, 2026-10-26. CPU only. Commands:

- `python3 -m offload.pf2_lowkv.arith` (the table below);
- `python3 -m offload.pf2_lowkv.baselines` (the memory table, which reads earlier windows' telemetry read-only).

Labels: **MEASURED** means read from the campaign records. **ESTIMATED** means arithmetic, not yet seen on GB10.

## Pick: LOW = 3 GiB per rank (`--kv-cache-memory-bytes 3221225472`)

| KV/rank | blocks | pool tokens (engine log) | concurrency @524,288 | 2 × 524K need | + two 32K seats | margin vs 2 × 524K | MemAvailable gain vs H |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 GiB | 15,534 | 940,232 | 1.79 | 17,329 | 21,853 | −1,795 (−10.4 %): **too small** | 4.000 GiB |
| **3 GiB** | **23,301** | **1,410,348** | **2.69** | 17,329 | 21,853 | **+5,972 blocks (+34.5 %)** | **3.000 GiB** |
| 4 GiB | 31,068 | 1,880,464 | 3.59 | 17,329 | 21,853 | +13,739 (+79.3 %) | 2.000 GiB |
| 6 GiB = H | 46,603 | 2,820,756 (MEASURED: PF1 rank-0 logs, all five windows) | 5.38 | 17,329 | 21,853 | +29,274 | 0 |

**Why 3 GiB.** It is the smallest candidate (2, 3, 4 GiB) that meets all three criteria below; 2 GiB fails the first.

1. **Two maximum-length requests are admitted at once.** Two 524,288-token requests need 17,329 blocks. The 2 GiB pool
   is 15,534 blocks, so it admits only one, and a second 524K prompt would wait.
2. **The other two seats still admit a 32K request each.** `--max-num-seqs` is 4. Charging the two other seats at 32K
   each brings the need to 21,853 blocks, which 3 GiB holds with 1,448 blocks spare.
3. **At least a 25 % block margin over the two-long need.** 3 GiB has 34.5 %. That margin, 5,972 blocks, equals about
   0.69 of a further 524K admission (8,662 blocks).

The reported pool falls from 2,820,756 to 1,410,348 tokens. That is 2.69 maximum-length requests instead of 5.38. The
engine's KV pool is ESTIMATED to return **3.000 GiB per rank** (23,302 blocks × 138,240 B = 3,221,268,480 B). Weights,
precision, attention and the request maximum are unchanged.

## How a request is charged (image source, read-only reference copy)

- **Pool.**
  - It is one aliased buffer of `num_blocks × 138,240 B` per rank.
  - Source: overlay `vllm/v1/core/kv_cache_utils.py` `get_kv_cache_config_from_groups`, "Groups alias from byte 0".
  - `num_blocks = floor(kv_bytes / 138,240)`. At 3 GiB, 64,512 B of the reservation is never allocated.
  - With `--kv-cache-memory-bytes` set, the worker skips memory profiling and reserves exactly that amount
    (`vllm/v1/worker/gpu_worker.py:615-638`).
- **Admission (`scheduler_reserve_full_isl=True` by default).**
  - Default: `vllm/config/scheduler.py:191`; checked at `vllm/v1/core/sched/scheduler.py:1263` and
    `kv_cache_manager.py:519-536`.
  - A waiting request is admitted only if its **whole prompt** fits the free blocks.
  - Each of the 14 SWA groups is charged at its admission cap of 131 blocks (8,192 tokens in flight).
  - One 524,288-token request is therefore charged
    `cdiv(L,128) + cdiv(L,192) + 1 ring + 14 × 131 = 4,096 + 2,731 + 1 + 1,834 = 8,662` blocks.
  - The ratio-1 group's block is 192 tokens because the K1 packing factor is 3 (`VLLM_DSV41_KV_PAIR=3`).
  - The indexer prefill buffer factor 4 does not touch the pool.
- **Output reserve.**
  - `max_model_len` bounds prompt + output, so a request charged at 524,288 already covers its output.
  - Lookahead: one decode step's 1 + 5 + 5 slots can open at most one new block in each of the two full-attention
    groups, so 2 blocks are added per request.
  - One null block is added per pool.
- **Held, not charged.**
  - After its prefill, a 524K request holds 7,080 to 7,304 blocks, because SWA blocks behind the window are freed. This
    is S8R2's held-window model in `kv_arith.held`.
  - The admission charge (8,662) is the binding number, so the pick uses it.
- **Validation.**
  - `kv_arith.py` reproduces the five GB10/upstream pool figures and S8R2's engine-manager pressure runs.
  - The new 524K figure (2,820,756 at 6 GiB) is reproduced exactly; it is the MEASURED value in every PF1 524K arm.
  - Tests: `offload/pf2_lowkv/test_arith.py`.

## Memory today (MEASURED, PF1 windows, 1 s telemetry during each arm's traffic)

| window | arm (position in HOLD) | 150K cold TTFT | sparkhead median / min GiB | sparkworker median / min GiB |
|---|---|---:|---:|---:|
| W1 | off-327 (1) | 70.38 s | 5.88 / 4.81 | 7.36 / 7.26 |
| W1 | **off-524 (2)** | **95.33 s** | 5.97 / 5.81 | **6.81 / 6.58** |
| W2 | **on-524 (2)** | **91.48 s** | 6.00 / 5.96 | **6.77 / 6.72** |
| W3 | off-327 (1) | 68.96 s | 4.99 / **4.17** | 7.56 / 7.44 |
| W3 | **off-524 (2)** | **91.34 s** | 5.91 / 5.82 | **6.78 / 6.25** |
| CW | cause-524 (2), MoE-span overlay | 69.20 s | 6.06 / 6.01 | 7.26 / 7.20 |
| CW | cause-524-t0 (3), persistent scratch | 69.77 s | 5.58 / 5.49 | 6.57 / 6.39 |
| MATCH | off-524 (1) | 70.18 s | 5.31 / 4.43 | 7.19 / 7.12 |

### Pre-window observation (MEASURED, correlational, n = 3 slow vs 2 fast plain 524K arms; not a cause)

- **sparkworker memory.** In the three slow 524K arms, sparkworker (rank 1) sat at a median of **6.77–6.81 GiB**. In the two fast
  ones without persistent scratch it sat at **7.19–7.26 GiB**, about 0.4 GiB higher. PF1-CAUSE located the slow time on
  rank 1's MoE half, so a headroom/allocation reading fits this pattern.
- **Position.** It is confounded: every slow arm was the second start in its HOLD, after a 327K arm, and the fast MATCH
  arm was the first. The design therefore records position and alternates arms.
- **T0 arm.** It ran fast at 6.57 GiB. Its scratch is persistent rather than allocated per call, which is consistent with
  the scratch-allocation lead (PF1-CAUSE H1), but this does not discriminate.

### What LOW should do to these numbers (ESTIMATED)

- **Steady level.** About +3.0 GiB on both ranks:
  - sparkhead 4.9–6.0 → ~7.9–9.0 GiB;
  - sparkworker 6.8–7.5 → ~9.8–10.5 GiB.
  K1's memprobe saw allocation deltas reach MemAvailable within about ±10 %, so read this as +2.7 to +3.3 GiB.
- **Dips.** The 4.17 GiB sparkhead dip came from prompt building in a 327K arm and should move by the same amount.
- **Long requests.** S18's single 2 × 500K sample at H put sparkhead at 3.09 GiB. That is why the design runs the one
  450–500K request on L only, and last in the HOLD.

## Baselines after FW1

`baselines.inventory()` parses `FW1/inventory-{pre,post}.txt` (fw1_sparkhead.sh: `uname -r`, driver, MemTotal,
MemAvailable, SwapTotal). It does not exist yet (2026-10-26 ~15:30Z). When it exists:

- the precheck compares its kernel and driver with the runner's `--sparkhead-kernel` / `--sparkhead-driver` parameters;
- it reports MemTotal/MemAvailable as the post-FW1 baseline (a display carveout of 0.5 GB would show as about +0.5 GiB
  MemTotal against `inventory-pre`).

The window's own 1 s telemetry is the binding memory record.
