# PF1-CAUSE item 1: where rank 1's extra 524,288 prefill time is, and what can cause it

Lane PF1-CAUSE, 2026-10-26, CPU only on Controller. Sources: PF1 window 2 (`PF1-W2-20261026T010854Z`: traces,
identity, telemetry, cmd timestamps), window 1 (`PF1-W1-20261025T234254Z`), the pinned image source
`/opt/dsv41/data/src-image` (= "image"), the K1 overlay `offload/kvpack/overlay` (= "K1", bytes equal
MANIFEST 559c5867) and production's remote_dspark at 21bed31 (hashes equal the window's mounted files). MEASURED =
read from a saved window file; INFERRED = follows from MEASURED values by the stated argument; ESTIMATED = arithmetic.

## Plain language

- The extra ~630 ms per 4,092-token chunk is **not spread over the step**. It sits entirely in the **MoE (FFN) half
  of each of the 20 decoder layers that run the full chunk** (layers 0-19), about 27-35 ms per layer, a stable
  amount for each layer. Attention, the indexer, metadata, the embedding and the 128-row replay tail (layers 20-39)
  are unchanged.
- Rank 0 only *waits* for it, at the all-reduce that ends each MoE half. Rank 1 is the one whose MoE half is slower.
- The host is not involved: rank 1's Python launches the same kernels at the same pace at both caps. The GPU runs
  the same MoE launches more slowly on rank 1.
- The source has **no** max_model_len-sized object or loop in the MoE half (router, shared expert, routed experts,
  mHC). Every per-step max_model_len-sized object (block tables, token-id table, RoPE caches, indexer workspace,
  sparse metadata) lives in phases that did **not** get slower. So the cause is not a cap-sized sweep. The best fit
  is that the same MoE kernels run slower on rank 1 because of **where their large per-call buffers land in memory**
  at 524,288 (the routed-expert prompt scratch, 750 MiB per call, allocated per call since K1 patch 0076d/e). This is
  a ranked hypothesis, **not proven**: no MoE-internal timing exists yet. Item 2 designs the spans that settle it.

Reproduce the localization (CPU, reads only the window-2 trace files; numbers only):

```bash
python3 -m offload.prefill_cause.localize \
  --traces /opt/dsv41/campaigns/DSV41-OFFLOAD/PF1-W2-20261026T010854Z/traces/*/*/*.jsonl \
  --output "$SCRATCH/localize-w2.json"        # sha256 de1a7db2... on this lane's run (empty cause_labels)
```

## 1. MEASURED localization (window 2, 150K cold ×1 per cap, 36 matched full-chunk prefixes per rank)

Each full chunk issues 81 GroupCoordinator all-reduces on each rank, in the same order at both caps:
1 embedding all-reduce, then for each full layer 0-19 an attention-output all-reduce (follows that layer's sparse
dispatch span) and an FFN-output all-reduce (no traced span in between), then 40 all-reduces of 128 rows for the
replay tail (layers 20-39 re-run only the final 128 rows: K1 `nvidia/model.py:813`, DIAGNOSIS.md).

| Group (median delta 524,288 − 327,680, summed over the group) | rank 0 | rank 1 |
|---|---:|---:|
| `ffn` all-reduce, 4,092 rows, layers 0-19 (20) | **+606.8 ms** | −1.6 ms |
| `attn` all-reduce, 4,092 rows (20) | +13.9 ms | +0.3 ms |
| embedding all-reduce (1) | +0.2 ms | 0.0 ms |
| `ffn` all-reduce, 128 rows, tail (20) | −0.1 ms | +0.2 ms |
| `attn` all-reduce, 128 rows, tail (20) | +0.2 ms | 0.0 ms |
| whole chunk (`chunk.total`) | +630.3 ms | +630.5 ms |

Per layer, rank 0's FFN all-reduce grows by 34.2, 34.5, 31.2, 32.9, 26.9, 27.8, 32.6, 30.1, 28.1, 28.2, 28.9, 28.4,
28.8, 28.6, 32.4, 30.8, 32.5, 27.7, 35.1, 27.1 ms (layers 0-19); interquartile ranges across the 36 prefixes are about
±1 ms, so each layer carries its own stable extra, independent of prefix. At 327,680 that all-reduce takes ~2.1 ms
on both ranks (no wait in either direction).

**Why this places the work on rank 1 (INFERRED).** Each all-reduce is a meeting point. Both ranks leave layer L's
attention all-reduce together (its span is ~2-3 ms on both ranks at both caps). Rank 0 then runs its MoE half for m0,
rank 1 for m1; rank 0's FFN all-reduce span is ≈ max(0, m1 − m0) + transfer. A +31 ms growth there with rank 1's own
span unchanged means m1 − m0 grew by ~31 ms. Rank 0's own chunk total grows by its waits (630.3 ms = 606.8 + 13.9 +
9.6 ms of other small span deltas), so m0 did not grow: m1 did. Rank 1's chunk grows by the same 630 ms with no traced
span growing (window-2 fit.json) — consistent.

**Host side (MEASURED).** Rank 1's host enqueues layers 0-9 at the same offsets at both caps (embedding all-reduce at
24-25 ms after `chunk.begin`, layer 9's FFN all-reduce at 128-130 ms, while the GPU needs roughly 0.8-1 s to get
there, ESTIMATED); the median MoE-half host interval (attention all-reduce issued → FFN all-reduce issued) is 7.9 vs
7.8 ms. Later layers are paced by
the GPU (launch-queue back-pressure). So no host-side work, host synchronization or extra kernel launches are added in
the MoE half; the same launches take longer on the GPU.

**Telemetry (MEASURED, 1 s samples, prefill window of each 150K request).** SM clock 2,125 MHz (sparkhead) / 2,119 MHz
(sparkworker) at both caps, throttle reasons 0x0, utilization median 96 % on both, temperature 65-70 °C, SwapFree flat
during prefill. Power median 47.9 → 39.5 W (sparkhead), 48.5 → 39.7 W (sparkworker). MemAvailable during prefill: sparkhead
5.87 → 6.00 GiB, sparkworker 7.48 → 6.77 GiB (sparkworker holds ~0.7 GiB more during the 524,288 prefill than at 327,680; sparkhead
does not). Rank 1 slower at lower power and equal clocks = its MoE kernels stall more (memory/latency-bound), they do
not compute more.

## 2. Constraints any explanation must meet

| # | Constraint (MEASURED unless marked) |
|---|---|
| C1 | Extra time only in the MoE half of the 20 full-chunk layers, 27-35 ms per layer, layer-specific, prefix-independent |
| C2 | None in the 128-row MoE halves of layers 20-39 (same weights, same TF prompt path: 128 ≥ MIN_ROWS 64) |
| C3 | Rank 1 only; rank 0 runs the same MoE half at both caps in the same time |
| C4 | No host-side change on rank 1 in the MoE half (same enqueue pace, no added sync) |
| C5 | Equal clocks, no throttling, lower power on both ranks at 524,288; GPU 96 % busy |
| C6 | Rank env/mounts identical except VLLM_HOST_IP; argv differs only by `--node-rank 1 --headless` (identity files) |
| C7 | Reproduced across fresh starts (S18 R1/R5 2026-10-23, W1 OFF, W2 ON): ~0.163-0.171 ms per prompt token, stable |
| C8 | KV pool bytes fixed (6 GiB); indexer plans identical at both caps (window-2 fit.json `indexer_plan`) |

## 3. Everything rank 1 does differently, and why each is excluded or kept

| Rank-asymmetric item | Evidence | In the MoE-half GPU window? | Verdict |
|---|---|---|---|
| Headless: no API server/engine core; scheduler output arrives through the broadcast message queue | image `entrypoints/cli/serve.py:181` (`run_headless`), `v1/executor/multiproc_executor.py:614,623` | No: host, before `execute_model` | Excluded by C1/C4 |
| Batch-sharded sampling: with one request, rank 1 owns none and contributes padding | K1 `v1/worker/gpu/model_runner.py:1660,1704` | No: after the forward | Excluded by C1 |
| Remote DSpark stream sync / proxy only on rank 0; drafts broadcast to rank 1 | 21bed31 `remote_dspark/speculator.py:138,142` | No: `propose` runs after the step; its broadcast is outside the layer stack | Excluded by C1 (would appear on rank 0 or between chunks) |
| Engram remote reads over each rank's own x86host link (rank 1: 10.202.29.x) | env `VLLM_ENGRAM_REMOTE`; K1 `nvidia/model.py:320` (layers listed in the Engram layout only) | Engram runs before attention of its own layers only | Excluded by C1 (all 20 layers affected) |
| EXL3 RDMA all-reduce/all-gather proxy (rank-pinned CPU 14) | image `vllm_exl3_moe/rdma.py:42,47` (≤256 KB AR, ≤3 MB AG) | No: a 4,092×5,120 bf16 all-reduce is 40 MiB → NCCL (traced) | Excluded |
| L2 prefetch side stream | image `vllm_exl3_moe/l2pf.py:127` (rows ≤ 16 only) | Decode only | Excluded |
| TP shard: rank 1 holds the other half of every expert's intermediate channels | image `vllm_exl3_moe/bank.py:13,45`, `plan.py:202-214` | Yes: different weight bytes and different physical buffers | Kept only as "placement differs per rank" (see H1/H2); shapes, codebooks, routing and kernel choices are identical by construction |
| Host machine (sparkworker vs sparkhead): its own memory state, page cache, driver allocator | identity `VLLM_HOST_IP`; telemetry MemAvailable differs per host | Yes (affects where allocations land) | Kept: the only rank-1-specific input to the MoE kernels besides the weight shard |

No source branch on `tp_rank`/`rank_in_group` exists in the MoE half (grep of K1 `nvidia/model.py`, image
`vllm_exl3_moe/{moe_method,forward,smoe,router,plan,bank}.py`, K1 `tf_prompt.py`/`tf_experts.py`): the only rank
use is the TP shard cut.

## 4. Every per-step object whose size or loop bound depends on max_model_len

| Cap-dependent object (ESTIMATED bytes at 327,680 → 524,288) | Evidence | Phase it lives in | Verdict |
|---|---|---|---|
| Request all-token-id table, max_num_reqs × cap int32, UVA host memory (5 → 8 MiB) | image `v1/worker/gpu/states.py:34-38` | input preparation (pre-forward); `input_ids` handed to the model are the scheduled rows | Excluded: C1/C4, outside the layer stack |
| Block tables padded to cap (per group ≤ 2×4×cap/64 int32) | K1 `model_runner.py:611-636` | metadata build (`metadata.prepare` span unchanged: 1.23 → 1.28 ms on rank 1) | Excluded by measurement |
| Indexer K/scale workspace view (factor 4 × cap × 132 B: 165 → 264 MiB) | K1 `attention.py:1507,1562`; DIAGNOSIS.md | attention half (`indexer.*` spans unchanged) | Excluded by measurement |
| Expanded indexer block table (80 → 128 MiB) | DIAGNOSIS.md | decode fill only | Excluded |
| RoPE caches, cap + 2048 positions (161 → 257 MiB) | image `models/deepseek_v41/common/rope.py:14-50` | attention half only (index by positions) | Excluded by C1 |
| Sparse/FlashMLA workspace formulas | DIAGNOSIS.md | inactive on SM121 / decode | Excluded |
| Remote drafter HELLO max_model_len (worker allocates for it) | 21bed31 `remote_dspark/speculator.py:96` | x86host; drafting after the step | Excluded by C1 |
| **Total persistent memory** of all of the above: larger at 524,288 | memprobe/S18: +0.1 GiB at load; sparkworker −0.7 GiB MemAvailable during prefill (MEASURED) | shifts the **placement** of every later allocation, incl. per-call MoE scratch | Kept: the only path from cap to the MoE half |

The MoE half itself (K1 `nvidia/model.py:613-637`: `mhc_shifted_post_pre`, then `self.ffn`) is sized only by rows,
hidden size, intermediate size and top-k: router (vLLM GateLinear, cuBLAS; EXL3 router only ≤ 8 rows, image
`vllm_exl3_moe/router.py:24-25`), shared expert (exllamav3 reconstruct + GEMM above 144 rows), routed experts
(image `vllm_exl3_moe/moe_method.py:285` → `forward.py:118-121` → K1 `tf_prompt.py:95-113` → K1
`tf_experts.py:357-480`). No argument carries max_model_len or a KV/block-table extent.

## 5. Candidates ranked by fit

| Rank | Hypothesis | Fits | Misses / open | Discriminating observation |
|---|---|---|---|---|
| **H1** | The TF prompt routed-expert kernels run slower on rank 1 because the **per-call transient scratch** (K1 0076d/e: `rot` 2·P·D fp16 = 480 MiB, `z_gu` 2·P·I fp32 = 216 MiB, `xd` 54 MiB, `out` 80 MiB at 4,092 rows × 6 slots; K1 `tf_prompt.py:109`, `tf_experts.py:384,420,450`) lands in memory that the GPU reads slowly at 524,288 (placement after the larger cap-dependent persistent buffers; e.g. smaller GPU pages/TLB reach, or a different allocator segment) | C1 (only full-chunk calls get the 750-MiB scratch; the 128-row tail's 23-MiB scratch reuses small cached blocks), C2, C4 (no host change), C5 (stalls at low power), C7 (deterministic startup ⇒ repeatable placement), C8 | why rank 1 and not rank 0 (sparkworker's memory state; MEASURED sparkworker −0.7 GiB vs sparkhead +0.1 GiB during prefill) — **unverified**; layer-to-layer spread (27-35 ms) needs the per-layer K bits to check | `tf.*` kernel spans on rank 1 grow at 524,288 with identical sizes; rank 0's do not; growth disappears with `VLLM_EXL3_TF_PROMPT_TRANSIENT=0` (persistent scratch, allocated at profile time) |
| H2 | Same as H1 but for the **shared expert / router** (exllamav3 reconstructs the shared expert's fp16 weights per call above 144 rows, image `vllm_exl3_moe/dense.py:54-62`; router = vLLM GateLinear on cuBLAS) | C1-C5 (the 128-row tail takes exllamav3's small-row path, so C2 does not test it) | magnitude: the router GEMM is ~16 GFLOP and the shared expert a few tens of MiB of weights per layer, ESTIMATED ≪ 31 ms even if slowed severalfold | `moe.ffn` − `moe.routed` grows on rank 1 |
| H3 | Persistent buffer/weight placement on rank 1 (expert banks, TF tables) | C3, C5, C7 | C2: the tail reads the same weights and is unchanged → weak | `tf.grouped_mma` grows at 128 rows too (it does not today) |
| H4 | mHC FFN pre-mix (`mhc_shifted_post_pre`, tilelang) slower on rank 1 | C1 | no cap or rank input; small tensors | `mhc.ffn` span grows |
| H5 | Anything that is a **wait** inside rank 1's MoE half for an external party (host flag, x86host, CPU thread) | C5 (spin at low power) | no such wait exists in the MoE source (no hostflag/RDMA/stream join beyond the same-rank aux-stream join `forward.py:120`); would not be cap-dependent | a span with zero kernel growth but a growing join |

Not candidates (excluded above): per-step cap-sized sweeps, block-table padding, positions/slot mapping, sampler/
logits, K1 KV packing (attention half), EXL3 RDMA/L2PF paths (decode sizes), Engram (2 layers), remote drafter
(after the step), the headless scheduler-output path (host, before the forward).

## 6. What would change this ranking

- A per-layer EXL3 bit plan (`/srv/dsv41/legal-v2a/exl3_moe_plan.json`, on the Sparks only) correlated with the per-layer
  extra: proportional to each layer's routed-expert decode work favours a fractional kernel slowdown (H1/H3), flat
  favours a per-call fixed cost.
- Host parity facts the window can read without touching the serve: `uname -r`, driver version, THP mode,
  `/proc/buddyinfo` (free-memory fragmentation) on both Sparks before each start (item 2 adds them as read-only
  preflight lines).
