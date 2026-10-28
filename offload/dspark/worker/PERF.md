# DSpark worker latency on the RTX 3090 (D5)

**In short.** One drafter round for one request takes 2.80 ms of GPU time on the 3090 (the CUDA graph of the draft
step) plus 0.27 ms to insert the newly accepted context rows. Over lane O1's RDMA transport the full round trip (rows
in, drafts out) is 3.14 ms. The Spark's own drafter takes about 3.6 ms per round, so the remote drafter is faster at one
request by about 0.46 ms before the Spark-side cost of a remote round (rank 0's gather / packing / broadcast, which only
the coordinator's shadow run can measure). With 3-4 requests the Spark sends `n_out = 4` (CONTRACT 1.3) and a round
costs 4.1 / 4.8 ms of worker time. About 70 % of the step is EXL3 kernels (routed experts, the vocabulary head, dense
matrices) that are bound by memory bandwidth, so a second card is the only large lever left: an estimated 38-42 % less
step time (numbers below).

Raw data: `offload/results/o2/d5_bench.json` (final), `d5_bench_baseline.json` (before the D5 work),
`d4_server_selftest.json` (end to end through the server and O1's transport).
Commands (on x86host; GPU `GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93` only):
```bash
cd ~/dsv41-offload/o2/repo/offload/dspark
export CUDA_VISIBLE_DEVICES=GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93
~/dsv41-offload/o2/venv/bin/python -m worker.tools.bench /opt/dsv41/x86host-data/dspark d5_bench.json 300
OFL_PY_DIR=~/dsv41-offload/o2/lib OFL_LIB=~/dsv41-offload/o2/lib/libofl.so \
  ~/dsv41-offload/o2/venv/bin/python -m worker.tools.server_selftest /opt/dsv41/x86host-data/dspark selftest.json 18625
```
Conditions: RTX 3090 in P2 (the driver's compute state: SM 1935-1950 MHz, memory 9501 MHz), 55-65 °C, nothing else on
the card. Inputs: real English tokens (anchors), synthetic taps, full 128-position windows; 2 of the 5 Markov inputs of
the B = 1 case are outside the cached rows, as about 20 % of live ones are (D3b). Timing: CUDA events around each
replay, 300 replays after warm-up, p50 (p10-p90); kernel time and shares from CUPTI (torch profiler) over 20 replays.

## Draft step (ms)

`k` = block length the body runs (the engine always runs 5; k = 4 rows are for comparison only), `n_out` = positions
the head / Markov / confidence produce (CONTRACT 1.3).

| requests | k / n_out | graph p50 (p10-p90) | eager p50 | kernel time | experts | head | Markov | dense K5 | wo_a | rest |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 / 5 | **2.80** (2.79-2.81) | 4.70 | 2.70 | 0.74 (28 %) | 0.61 (23 %) | 0.19 (7 %) | 0.59 (22 %) | 0.14 (5 %) | 0.42 (16 %) |
| 1 | 5 / 4 | 2.76 (2.75-2.77) | 5.43 | 2.61 | 0.74 (29 %) | 0.60 (23 %) | 0.10 (4 %) | 0.59 (23 %) | 0.15 (6 %) | 0.42 (16 %) |
| 1 | 4 / 4 | 2.66 (2.65-2.67) | 5.34 | 2.52 | 0.67 (27 %) | 0.61 (24 %) | 0.10 (4 %) | 0.59 (23 %) | 0.14 (6 %) | 0.41 (16 %) |
| 2 | 5 / 5 | **3.49** (3.47-3.50) | 5.60 | 3.35 | 1.20 (36 %) | 0.62 (19 %) | 0.28 (8 %) | 0.65 (19 %) | 0.15 (5 %) | 0.45 (13 %) |
| 2 | 5 / 4 | 3.40 (3.40-3.41) | 5.47 | 3.24 | 1.19 (37 %) | 0.62 (19 %) | 0.19 (6 %) | 0.64 (20 %) | 0.15 (5 %) | 0.45 (14 %) |
| 2 | 4 / 4 | 3.22 (3.22-3.25) | 5.41 | 3.09 | 1.07 (35 %) | 0.62 (20 %) | 0.19 (6 %) | 0.62 (20 %) | 0.15 (5 %) | 0.44 (14 %) |
| 3 | 5 / 5 | 3.99 (3.98-4.03) | 5.72 | 3.85 | 1.60 (41 %) | 0.63 (16 %) | 0.28 (7 %) | 0.69 (18 %) | 0.15 (4 %) | 0.49 (13 %) |
| 3 | 5 / 4 | **3.92** (3.89-3.95) | 5.56 | 3.75 | 1.59 (42 %) | 0.62 (17 %) | 0.20 (5 %) | 0.69 (18 %) | 0.15 (4 %) | 0.49 (13 %) |
| 3 | 4 / 4 | 3.61 (3.59-3.64) | 5.52 | 3.48 | 1.37 (39 %) | 0.63 (18 %) | 0.20 (6 %) | 0.67 (19 %) | 0.15 (4 %) | 0.47 (14 %) |
| 4 | 5 / 5 | 5.42 (5.41-5.50) | 6.63 | 5.31 | 1.88 (35 %) | 1.26 (24 %) | 0.46 (9 %) | 0.95 (18 %) | 0.22 (4 %) | 0.54 (10 %) |
| 4 | 5 / 4 | **4.67** (4.67-4.69) | 5.90 | 4.51 | 1.85 (41 %) | 0.63 (14 %) | 0.37 (8 %) | 0.93 (21 %) | 0.22 (5 %) | 0.53 (12 %) |
| 4 | 4 / 4 | 4.19 (4.18-4.20) | 5.81 | 4.03 | 1.68 (42 %) | 0.63 (16 %) | 0.37 (9 %) | 0.70 (17 %) | 0.16 (4 %) | 0.49 (12 %) |

Bold: what the live Spark config asks for (n_out = 5 at 1-2 requests, 4 at 3-4). "experts" = routed coop MoE kernels
+ the shared expert's activation; "dense K5" = the 7 EXL3 K5 matrices per stage (wq_a, wkv, wq_b, wo_b, shared
w1/w3/w2); "rest" = mHC (0.11-0.16), attention (0.07-0.09), norms / RoPE / KV record (0.04-0.05), routing (0.02) and
casts / copies. Eager runs dispatch ~1,000 launches from Python; they are CPU-bound and noisy, so take the eager column
as an order of magnitude: graphs save 1.2-2.7 ms per step.

Why B = 4 x n_out 5 is slow: 20 head rows take two passes over the 474 MiB head (the head runs on one fixed 16-row
kernel shape so that positions < n_out are bit-identical for every n_out, CONTRACT 1.3); 20 body rows also move the
dense GEMMs to 32-row tiles. The Spark only sends n_out = 5 at 4 requests in shadow mode or with the depth policy off.

## Context insert and the full round

| rows inserted (main_proj + 3 window-KV stages) | 1 | 6 | 12 | 20 | 24 | 64 | 128 |
|---|---|---|---|---|---|---|---|
| graph p50, ms (before D5) | 0.27 (0.33) | 0.27 (0.33) | 0.29 (0.34) | 0.35 (0.41) | 0.35 (0.41) | 0.57 (0.61) | 0.89 (0.95) |

A decode round inserts 1-6 rows per request (accepted drafts + bonus). Worker time and round trip per STEP, through
`server.py` (`d4_server_selftest.json`; rows, generations, windows checked by the contract state machine; ofl on
10.202.28.1, client on x86host itself, so the wire is the NIC's loopback):

| STEP | in-process handle() p50 | RDMA: worker / round trip p50 (p99) | TCP: round trip p50 | request bytes |
|---|---|---|---|---|
| 1 request, 6 rows | 3.03 ms | 3.02 / **3.14** (3.39) ms | 3.20 ms | 184 KB |
| 2 requests | 3.61 ms | | | |
| 3 requests, n_out 4 | 4.12 ms | | | |
| 4 requests, n_out 4 | 4.79 ms | 4.77 / **5.01** (5.27) ms | 5.09 ms | 738 KB |
| 4 requests, n_out 5 | 5.50 ms | 5.50 / 5.74 ms | 5.84 ms | 738 KB |
| 1 request, WANT_MARGIN (shadow mode) | 3.44 ms | | | |
| 4 requests, n_out 4, WANT_MARGIN | 5.13 ms | | | |

WANT_MARGIN steps (the shadow policy asks for margins) run through their own CUDA graphs: the margin path computes the
full bf16 (base + Markov) logits per position for top1 - top2, +0.35-0.4 ms over the plain graph; latency measured
in shadow mode therefore overstates the production round by that much.

## Compared with the Spark's drafter (~3.6 ms per round, upstream profile)

| requests (live n_out) | worker round trip | vs ~3.6 ms local |
|---|---|---|
| 1 (5) | 3.14 ms | -0.46 ms before the Spark-side remote overhead |
| 2 (5) | ~3.73 ms (3.61 worker + ~0.12 transport) | about even (the local drafter also grows with the batch) |
| 3 (4) | ~4.24 ms | slower than 3.6 ms; the local drafter at 3 requests is not in the upstream profile |
| 4 (4) | 5.01 ms | slower |

The remote round is serial on the Spark's timeline in the synchronous mode; O3's async mode
(`VLLM_DSPARK_REMOTE_ASYNC=1`) removes the Spark CPU block but not the GPU-timeline wait. Independently of latency, the
remote drafter frees about 5 GiB per Spark (coordinator: 5-7 GiB available with the local drafter, 11-12 GiB without
speculation), and draft agreement with the local drafter is at the near-tie floor (PARITY.md D3b).

## What D5 changed (draft step graph p50, ms, baseline -> final)

| requests | k 5 | k 4 |
|---|---|---|
| 1 | 3.21 -> 2.80 | 3.07 -> 2.66 |
| 2 | 4.10 -> 3.49 | 3.82 -> 3.22 |
| 3 | 4.59 -> 3.99 (n_out 4: 3.92) | 4.31 -> 3.61 |
| 4 | 5.83 -> 5.42 (n_out 4: 4.67) | 4.69 -> 4.19 |

1. Block attention as one Triton kernel reading the window ring in place (no per-step gather, no fp32 cuBLAS sgemm
   that went SIMT at B >= 2): 70-155 -> 15-21 us per stage.
2. wo_a as a W8A16 Triton kernel on the FP8 bytes (exact in-register dequantization through the fp16 bit pattern):
   95 -> 51 us per stage (32 instead of 64 MiB read).
3. Markov: two-to-three launches per sequential step instead of five to six, and all 8,192 listed rows cached (the
   serve caches 4,096; same bits): 80 % instead of 74 % of live Markov inputs hit the cache; a miss is a 66 MB GEMV
   (~85 us) that no kernel can avoid.
4. Context insert: main_proj's prescale in two kernels (bit-identical to the plugin's formula), the scaled main_norm,
   and one norm + RoPE + V4-record + ring-store kernel per stage: 40 -> 17 launches.
5. CONTRACT 1.3 n_out (proposed here, accepted by O3): -0.75 ms at 4 requests.
6. Cost added on purpose: main_proj on fp32 accumulation (+12 us per insert) for main_x precision (PARITY.md D3b).

Tried and dropped: running exllamav3 GEMMs concurrently on side streams inside the graph (they are cooperative
launches that share one device-wide lock / barrier buffer: unsafe); the whole step on fp32 accumulation (+0.13 ms for
no draft-agreement gain).

## Headroom on one card

At one request 70 % of the step is EXL3 GEMMs: the head reads 474 MiB per step at ~820 GB/s (the 3090 peaks at
936 GB/s), the routed experts read up to 15 experts (8.5 MiB each) per stage at about half of peak, and the 21 dense K5 GEMMs carry a
fixed ~18 us each (cooperative launch, grid barrier, in-kernel Hadamard passes) on top of their bytes, so the five
smallest per stage (wq_a 3.9 MiB, wkv 1.6, shared w1/w3/w2 7.0 each) run at 80-300 GB/s. Left on one card, roughly:
exllamav3's multi-matrix GEMM for the fused pairs (wq_a + wkv, shared w1 + w3) at ~0.1 ms per step; a custom EXL3
GEMV for <= 16 rows without the cooperative grid barrier, maybe 0.2-0.3 ms; the P0 memory clock (9751 instead of
9501 MHz in P2; needs root) ~2 %. None changes the picture at 3-4 requests.

## Numbers for a two-card (NVLink pair) variant (estimate; not built)

One card holds the worker safely (8.1 GiB of 24 GiB, MEMORY_BUDGET.md); the reason for a second card would be
latency at 3-4 requests, where one 3090 is slower than the Spark's local drafter. Split the bandwidth-bound parts the
way the Spark's TP = 2 splits them (head and Markov by vocabulary shard with a local argmax, routed experts by expert
id, wq_b / wo_b / shared expert column-row parallel, wo_a by head group), keep mHC / attention / norms replicated, and
add 6 NVLink all-reduces of <= 200 KB per step (~12 us each) plus one argmax exchange per Markov step (~6 us):

| config | 1 card, kernel time | of which splittable | 2 cards, estimate | estimated round trip (+ insert, transport) |
|---|---|---|---|---|
| 1 request, n_out 5 | 2.70 ms | 2.28 ms | ~1.66 ms (-38 %) | ~2.1 ms |
| 2 requests, n_out 5 | 3.35 ms | 2.90 ms | ~2.00 ms (-40 %) | ~2.5 ms |
| 3 requests, n_out 4 | 3.75 ms | 3.26 ms | ~2.22 ms (-41 %) | ~2.7 ms |
| 4 requests, n_out 4 | 4.51 ms | 3.99 ms | ~2.62 ms (-42 %) | ~3.2 ms |

That would put every live batch size under the ~3.6 ms local drafter. Assumptions to check before building: the
second card is the NVLink partner (`GPU-7744b44b`, not this lane's), the per-shard EXL3 kernels keep their per-byte
efficiency at half the width, and the all-reduce latency holds inside CUDA graphs on two processes or one.
