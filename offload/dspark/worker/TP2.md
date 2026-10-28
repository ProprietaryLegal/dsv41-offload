# The two-card (TP = 2) DSpark worker on x86host's NVLink pair (lane E)

**In short.** `tp2.py` runs the same drafter as the one-card worker (same checkpoint tensors, same contract, same
server, same port) split over the NVLink pair `GPU-c4e7a371` (rank 0) + `GPU-7744b44b` (rank 1), selected with
`--tp 2` / `DSPARK_WORKER_TP=2`; one card stays the default. Each card does half of the
bandwidth-bound work (vocabulary head + Markov, routed and shared experts, wq_b / attention / wo_a / wo_b) and
replicates the small rest; the halves meet in 6 all-reduces and one argmax exchange per drafted position per step,
through lane E's own NVLink peer-to-peer kernels inside the CUDA graphs. **Measured on the pair** (2026-10-23
00:32-00:42Z, both cards at NDM's 325 W cap): the draft step takes 1.94 ms at 1 request (one card: 2.94 ms, -34 %) and
3.17 ms at 4 requests with n_out 4 (one card: 4.89 ms, -35 %); a whole STEP over lane O1's RDMA transport takes
2.34 ms at 1 request (one card: 3.14 ms) and 3.67 ms at 4 requests (one card: 5.01 ms). The real pair computes the
same bits as the single-card emulation, and its drafts agree with the Spark's own drafter exactly as often as the
one-card worker's do (93.4 % of live blocks; every difference a numeric near-tie).

## Layout

| | rank r owns (r = 0, 1) | replicated on both |
|---|---|---|
| head + Markov bias | vocabulary columns [0, 64768) / [64768, 129280) (253 / 252 tiles of 256: the head's kernel tile; rank 0 holds the lower ids so a tie of the global argmax goes to the lower id, as on one card) | Markov embedding (by previous token), token -> cached-row map |
| attention | wq_b columns (heads 32r..32r+31), the heads' sinks, block attention, wo_a groups 4r..4r+3 (FP8 bytes + scales), wo_b input rows 4096r.. | embedding, mHC mixes / collapse / post, norms, wq_a, wkv, RoPE, the window-KV ring (inserted on both), block KV |
| FFN | routed experts: intermediate channels 1152r.. (Exl3Bank tp_size 2, the Spark's own cut); shared expert w1 / w3 columns, w2 rows | routing (gate, top-3) |
| other | | main_proj + main_norm (context insert), confidence head (rank 0 only) |

Every cut is at a 128-channel boundary of the EXL3 Hadamard blocks, so a shard's decoded weights are the full tensor's
decoded weights bit for bit (`tests/test_tp2.py`: 98 matrices incl. experts of every stage; FP8 wo_a bytes, scales,
sinks; Markov head rows and cached bias rows == the one-card model's columns).

Per step: per stage one all-reduce of the attention partials (wo_b) and one of the FFN partials (routed + shared),
fp32 sums rounded to bf16 — identical bits on both ranks because a + b is commutative, so the replicated state never
diverges; per drafted position one exchange of (shard max, index) folded into the Markov finish kernel
(`xchg_markov`); with WANT_MARGIN a (top1, top2, index) exchange.

## Collectives (`p2p/p2p.cu`)

One process drives both GPUs (`CUDA_VISIBLE_DEVICES=GPU-c4e7a371...,GPU-7744b44b...`, peer access enabled both ways).
One-shot push: each rank writes its payload straight into the peer's receive buffer over NVLink (UVA stores), signals
with `st.release.sys` on the peer's flag, waits with `ld.acquire.sys` on its own, then combines. Epoch counter per
rank (read at kernel start, published by the last block), double-buffered by epoch parity, per-block flags. Waits are
bounded by `%globaltimer` (default 2 s, `DSPARK_TP_TIMEOUT_MS`): a timeout sets the rank's error word (mirrored into
the peer's), the STEP fails loudly (INTERNAL, slots resynced), both devices are synchronized and both states reset.
No allocation, static arguments: captured inside the CUDA graphs. All kernels are launched once at start-up
(`p2p.preload`): CUDA lazy loading may synchronize a context on a kernel's first launch, which deadlocks against a
collective spinning on the same device (CUDA programming guide, lazy loading / concurrent execution). Test:
`tests/test_p2p.py` (single-card mode: both ranks on one card, two streams; two-card mode: the pair).

## Execution

Single controller: every op is issued for each active rank in turn under that rank's device, so an eager step is
lockstep and each collective's two halves are issued back to back. CUDA graphs are captured per rank by running the
same code with only that rank active (capture records, nothing runs, so a collective needs no partner while
capturing); a step replays rank 1's graph, then rank 0's (rank 0's outputs and its error word come back in one D2H).
Graphs: (B 1-4) x (n_out 1-5) x (plain, WANT_MARGIN), inserts of 8-128 rows, per rank.
Modes for development on one card: emulation (both ranks on one device, collectives as torch adds / compares — the
same combine bit for bit), solo (one rank alone, collectives replaced by local stand-ins: one rank's compute time),
tp = 1 (the same code unsplit on one card: separates what the second card adds from what the code adds).

Every EXL3 GEMM runs with a forced (shape, num_sms) from `tp2_kernels.json` (`tools/tp2_tune.py`, timed on the 3090):
exllamav3's timing-based autotuner could pick different kernels on the two cards (breaking the bit-identity of the
replicated activations) and cannot run inside a capture. The head keeps one kernel for 1-16 rows (n_out bit-identity,
CONTRACT 1.3).

Small replicated chains fused in `fused_e.py` (tested against the ops they replace, `tests/test_fused_e.py`): routing
in 2 launches (was ~10; same expert sets on 195/195 test rows, weights within 3.5e-7), SwiGLU -> fp16, norms with
fp16 outputs, shared expert through the coop kernel's `sh_out` (bit-identical). Tried and kept off:
`Exl3Pair` (wq_a + wkv and shared w1 + w3 in one `exl3_mgemm` launch): -70 to -100 us per step, but its rounding is
0.12-0.13 % from fp64 instead of 0.10 % and draft agreement with the engine on the captures fell from 93.4 % to
91.3 % of blocks (`results/e/e6_parity_emulated_v3_fused_pairs.json`, ~1.8 standard errors). The drafter is ~3-8 % of
a decode round, so 0.1 ms of drafter time is worth less than 1-2 points of acceptance; `DSPARK_TP_FUSE_PAIRS=1` turns
it on.

## Timing on one card (solo shards; `results/e/e3_bench_solo_v3.json`)

RTX 3090 `GPU-7744b44b`, 325 W cap (NDM), P2, SM 1965 MHz / mem 9501 MHz at the end of the run; graph replay p50 over
300 replays, inputs as O2's D5 bench (full 128-row windows, real English anchors, synthetic taps). "one card" = O2's
deployed model.py on this card; "E tp=1" = this code unsplit on this card; rank 0 / rank 1 = one rank's share of the
TP = 2 step with the collectives replaced by local stand-ins (rank 1's Markov and expert times are inflated by the
stand-in: its local argmax feeds different previous tokens / routing than a real pair would). Bold = the live
config's n_out.

| requests, n_out | one card (O2) | E tp=1 | TP2 rank 0 | TP2 rank 1 |
|---|---|---|---|---|
| 1, 5 | **2.94 ms** | 2.83 | **1.76** | 1.85 |
| 1, 4 | 2.89 | 2.75 | 1.70 | 1.79 |
| 2, 5 | **3.67** | 3.50 | **2.23** | 2.32 |
| 2, 4 | 3.57 | 3.43 | 2.16 | 2.26 |
| 3, 5 | 4.19 | 4.05 | 2.51 | 2.66 |
| 3, 4 | **4.10** | 3.97 | **2.45** | 2.59 |
| 4, 5 | 5.71 | 5.62 | 3.31 | 3.37 |
| 4, 4 | **4.89** | 4.78 | **2.87** | 2.97 |

Where the time goes at 1 request (rank 0, kernel us): experts 383 (one card 812), head 310 (651), dense K5 455 (656:
21 GEMMs with a fixed ~15 us each, so halving their bytes saves little), Markov 147 (202), wo_a 91 (153), mHC 114
(replicated), attention 44, norms / RoPE / KV 44, routing 33, other 79. The context insert is replicated: 286 us for
1-6 rows on each card (incl. the taps' upload), the same as one card.

## Pair (2026-10-23 00:32-00:42Z, `results/e/e7_*`)

Conditions: both cards at 325 W (NDM), P2 / SM 1965-1980 MHz / mem 9501 MHz at the end, the live one-card worker
idle on GPU-c4e7a371 (8.3 GiB, no traffic) during the window; power watchdog on the coordinator's 1 s power log (no
anomaly; both pair cards reached their 325 W cap, at most 710 W for all four); NVLink replay / recovery / CRC errors
0 before and after. "one card" = O2's model.py on this card model at 325 W (`e3_bench_solo_v3.json`); TP2 = the real
pair (`e7_bench_pair.json`; the repeat `e7_bench_pair_run2.json` is within 0.5 %). Graph replay p50 over 300 replays,
rank 0's CUDA events (input copies of both ranks + both replays); bold = the live config's n_out.

| requests, n_out | one card | TP2 pair | change | TP2 host wall | rank 0 / rank 1 kernel time | of which collectives (rank 0) |
|---|---|---|---|---|---|---|
| **1, 5** | 2.94 ms | **1.94** | -34 % | 1.99 | 1.80 / 1.81 | 0.07 |
| 1, 4 | 2.89 | 1.92 | -33 % | 1.91 | 1.77 / 1.78 | 0.07 |
| **2, 5** | 3.67 | **2.42** | -34 % | 2.42 | 2.28 / 2.29 | 0.09 |
| 2, 4 | 3.57 | 2.37 | -34 % | 2.36 | 2.23 / 2.24 | 0.09 |
| 3, 5 | 4.19 | 2.72 | -35 % | 2.73 | 2.59 / 2.60 | 0.12 |
| **3, 4** | 4.10 | **2.67** | -35 % | 2.66 | 2.53 / 2.54 | 0.13 |
| 4, 5 | 5.71 | 3.60 | -37 % | 3.60 | 3.50 / 3.51 | 0.14 |
| **4, 4** | 4.89 | **3.17** | -35 % | 3.17 | 3.09 / 3.10 | 0.14 |

O2's estimate (PERF.md "two-card variant") was -38 to -42 % of kernel time; measured -33 to -37 % of graph time:
the 21 dense GEMMs per step keep their fixed ~15 us each at half the bytes, and the replicated mHC / norms /
routing / attention-side work does not shrink.

**Collectives** (p2p.cu inside CUDA graphs, 20 back-to-back per replay, `collective_bench`): all-reduce of
5 / 10 / 15 / 20 rows x 5120 fp32 (100-400 KB each way) 12.9 / 15.2 / 19.6 / 22.5 us; argmax exchange 9.2 us for 1-4
requests. In the step they cost 0.07 ms (1 request) to 0.14 ms (4 requests) of rank 0's timeline: 3.6-4.7 % of the
step. NVLink traffic of one bench run: 9.3 GB each way.

**Context insert** (both ranks, incl. the taps' upload): 395 us for 1-6 rows (one card 276 us), 478 us for 20 rows,
1,026 us for 128 rows (one card 843). Slower than one card although it is the same replicated work: rank 0's graph
waits while the host issues rank 1's small copies and replay (no collective in the insert, so the order is free);
packing each rank's inputs into one upload and replaying rank 0 first should take ~0.1 ms off (not done yet).

**A whole STEP through the server** (`tools/server_selftest.py ... 2`, `e7_selftest_pair.json`; ofl server on
10.202.28.1:18627 and client on x86host, i.e. the NIC's loopback; contract checks: identity / config refusal, 41/41 rounds
bit-identical to direct model calls, n_out prefix, rollback, partial window, bad entry, stale generation, RDMA + TCP):

| STEP | TP2 in-process handle | TP2 RDMA round trip p50 (p99) | TP2 TCP | one card RDMA (PERF.md) |
|---|---|---|---|---|
| 1 request, 6 rows | 2.25 ms | **2.34** (2.55) | 2.40 | 3.14 |
| 2 requests | 2.64 | | | ~3.73 |
| 3 requests, n_out 4 | 3.01 | | | ~4.24 |
| 4 requests, n_out 4 | 3.46 | **3.67** (3.79) | 3.74 | 5.01 |
| 4 requests, n_out 5 | 3.88 | 4.09 (4.21) | 4.16 | 5.74 |

**Memory per card** (bench process, `nvidia-smi`): 5,726 MiB on GPU-c4e7a371, 5,050 MiB on GPU-7744b44b (torch
reserved 5,316 / 4,656 MiB; all CUDA graphs captured in 1.5 s); the one-card worker: 8,152 MiB on one card.

**Correctness on the pair**: `test_pair_equals_emulation` (real pair == emulation bit for bit: drafts, confidences,
margins, base logits, head hidden; graph == eager); capture parity on the pair (`e7_parity_pair.json`): TP2 vs the
engine's drafter 92.6 % (A) / 93.4 % (B) of live blocks exact — the emulated numbers exactly; all 53 live first
mismatches against the one-card reference numeric near-ties. (The one-card reference itself scored 92.6 / 91.9 % here
instead of 91.5 / 93.4 % on 7744b44b alone: model.py autotunes its EXL3 kernels by timing per card, so the same model
picks other kernels on another card or in another process and flips other near-ties. TP2 forces its kernels and gives
the same bits on either card.)

## When two cards beat one (decision data)

* The draft step is 34-37 % shorter at every live batch size (1.0 ms at 1 request, 1.7 ms at 4 requests).
* A whole remote round (rows in, drafts out, over RDMA): -0.80 ms at 1 request (2.34 vs 3.14 ms, -25 %), -1.34 ms
  at 4 requests with n_out 4 (3.67 vs 5.01 ms, -27 %). That is under the Spark's own ~3.6 ms local drafter (upstream
  profile at 1 request) at 1 request, at 2-3 requests too (in-process 2.64 / 3.01 ms + ~0.1 ms transport; their round
  trips were not measured) and level with it at 4.
* What that is worth end to end (estimate; the coordinator's in-engine runs decide it). The production
  configuration (dsv41.env: capacity policy, synchronous rounds) prices a round as the verify forward
  (REMOTE_ROUND_MS: 43.68 ms at depth 5, 41.56 ms at depth 4) plus the remote draft (REMOTE_DRAFT_MS 4.4 ms, calibrated on
  one card). 0.80 ms less at 1 request is ~1.7 % more rounds per second (48.1 -> 47.3 ms); 1.34 ms less at 4 requests is
  ~3 %. Against the upstream single-stream profile (38.9 ms with the local drafter) the same savings are 2-3.5 %. That
  holds if acceptance is unchanged, and the parity shows it is: same agreement with the engine as the one-card worker.
  Against NDM's promotion threshold (>= ~5 %, repeatable) TP2 alone is below it at 1 request and short of it at 4.
* With TP2 the depth policy's REMOTE_DRAFT_MS (4.4 ms, one card) overprices the draft by ~0.8 ms at 1 request, so a
  TP2 configuration should be recalibrated (or the policy keeps choosing as if the drafter were slower).
* Cost: the second 3090 fully (5 GB, ~260 W under load, both cards reach the 325 W cap in sustained benches), the
  pair cannot also host a second independent worker, and the failure surface doubles (a dead card stops the worker;
  the collectives fail fast and the worker exits for systemd after 3 failed steps in a row).
* Accuracy beats small speed wins here: the drafter is ~3-8 % of a round, so a 1-point drop in acceptance costs more
  than 0.1 ms of drafter time saves (why `Exl3Pair` stays off).

## Open items

* Insert host overhead on the pair (above, ~0.1 ms per round).
* S10 P-3 mixed-model build/capture qualification: a process-wide implicit capture stream on the wrong device was
  reproduced and fixed by passing model.py's owning stream explicitly. Fixed full/swapped/recreated pair lifecycles
  pass. Native peer state also removes dependence on Torch's runtime expandable allocator. The original illegal-address
  kernel remains unlocalized: corrected baseline/fixed cross-device allocator controls and a fresh full pair/emulation
  comparison are still pending an authorized RX pair window. Do not mark P-3 closed from capture sensitivity alone.
* Only the coordinator can measure the Spark-side effect (in-engine test).
* --max-model-len (main's 88a31ed, ported in E18; the TP2 unit runs 1048576 like the production unit): the RoPE
  table is replicated, 269 MB per rank at 1,048,576 (84 MB at 327,680), so both cards hold ~0.18 GB more than the pair
  numbers above (measured at 327,680). test_tp2's agreement test at positions near 900,000 has not run on a GPU yet.

## Memory per card in detail (rank shards, MiB)

| item | rank 0 | rank 1 | one card (O2) |
|---|---|---|---|
| embedding (bf16, replicated) | 1,262.5 | 1,262.5 | 1,262.5 |
| head shard (EXL3 K6) | 237.3 | 236.4 | 473.7 |
| routed experts (EXL3 K2, half the channels) | 1,633.8 | 1,633.8 | 3,256.3 |
| wq_b / wo_b / shared shards + wq_a / wkv / main_proj | 106.9 + 16.5 + 46.9 | same | 230.2 + 46.9 |
| wo_a FP8 shard | 48.2 | 48.2 | 96.4 |
| Markov embedding + head shard + cached rows | 63.1 + 31.6 + 1,012.5 | 63.1 + 31.5 + 1,008.5 | 126.2 + 2,020.5 |
| RoPE, norms, windows | 80.5 + 18.8 + 3.4 | same | 80.5 + 18.8 + 3.4 |
| torch allocated (weights + state) | 4,772 | 4,767 | 7,639 |

## Parity (`tools/tp2_parity.py`, live captures A-capture-20261022, 453 rounds, 542 live request-blocks)

Emulated TP2 (identical bits to the pair: `test_pair_equals_emulation`) against O2's one-card worker and the Spark's
drafter, blocks exact (all 5 drafts equal), `results/e/e6_parity_emulated_v2.json`:

| window | TP2 vs one card | one card vs engine | TP2 vs engine |
|---|---|---|---|
| A (captured main_x) | 93.9 % | 91.5 % | 92.6 % |
| B (main_x from taps: production) | 96.1 % | 93.4 % | 93.4 % |

Every first mismatch between TP2 and the one-card worker (54 live) is a numeric near-tie (the two tokens within
max(2 bf16 steps, 3 sigma of the cross-model logit noise) in one model's logits); no routing cascade, nothing
unexplained. TP2 agrees with the engine exactly as often as the one-card worker does.

## Files

`tp2.py` (model + `TP2Backend`), `p2p/` (collectives), `fused_e.py`, `tp2_kernels.json`, `server.py` (`--tp`),
`tools/{tp2_tune,tp2_bench,tp2_parity}.py`, `tools/server_selftest.py` (tp argument), `tests/{test_p2p,test_tp2,
test_fused_e}.py`, `run/{dsv41-dspark-worker-tp2.service,start_tp2.sh,stop_tp2.sh,status_tp2.sh,e_job.sh}`,
`requirements.e.lock`. Results: `offload/results/e/`.

## RX-TP2SAFE collective safety revision (2026-10-23)

The worker's arithmetic and channel split are unchanged. Each push now publishes a per-block consumption acknowledgment
only after all consumer reads and mirrored error stores finish. A successful block waits for that acknowledgment before
completing. Together with double buffering and stream order this prevents reuse before consumption; it also makes a bad-id
refusal on rank 1 visible to the rank-0 STEP check in the same round (S10 P-2). Both data publication and acknowledgment use
system-scope release/acquire ordering. Timers compare only a local GPU timer against its own start time.

Error bits (both headers): 1 local timeout, 2 peer timeout, 4 invalid received ID (mirrored), 8 cancellation, 16 epoch
exhaustion. Epoch zero is never published: at uint32 exhaustion the operation fails closed, with no peer payload access.
The old claim that exact-equality alone makes wrap safe was incomplete: an unused block's initialized zero flag could match
wrapped epoch zero. A drained reset clears flags/acks and starts a fresh generation. Cancellation uses pre-created independent
streams, so it can interrupt a collective parked on another stream. Outputs of failed/cancelled operations are refused by
STEP and must never be interpreted as complete sums or proposals.

`P2PPair.headers/reset` track every stream used by eager collective launches as well as current per-device streams. They wait
for bounded idle completion before reading/clearing state; a timed-out idle wait or CUDA exception permanently poisons the
pair. A poisoned pair refuses launches, graph replay and reset; `TP2Backend.healthy` reports false, causing the existing fatal
worker exit/restart policy. This host latch never clears a CUDA sticky error. The native peer-enable path checks pending CUDA
errors before handling the benign AlreadyEnabled status. Graphs retain the same state pointers throughout their lifetime;
callers must keep the drafter alive until its graph replays and peer accesses drain.

Debug oracle: `worker.p2p.reference.all_reduce` uses supported PyTorch device copies with explicit producer events, FP32
addition and optional BF16 rounding. It is eager-only and does not use the push flags or raw peer pointers. `test_p2p.py`
compares both implementations bit for bit, including delayed streams, varying block counts and mixed collective types.
`python -m worker.p2p.lifecycle --full --cycles 2` isolates the S10 mixed-model build/capture order, checks eager/graph IDs
and confidence equality, then destroys/recreates. `--swapped` reverses logical devices; `--legacy-capture` reinstates the
implicit one-card capture stream for diagnosis. Run each failure reproduction in a fresh process.

Lane-only runner: `worker/p2p/run_owned.sh solo|pair LOG COMMAND...` enforces UUIDs and both locks, caps execution at 14 minutes,
records per-card telemetry, and terminates its own test process group if coordinator `PAIR_OK` stops naming RX-TP2SAFE. It
never creates PAIR_OK or changes a service, power limit or clock. Production launchers remain the coordinator's responsibility.

Peer state now uses an owning native `cudaMalloc` tensor with a guarded `cudaFree` deleter, allocated outside capture on
its exact device. It is independent of Torch's caching/VMM allocator: exllamav3's Python import can enable expandable
segments through a runtime setter without changing either allocator environment variable. `cudaDeviceEnablePeerAccess`
does not establish access to arbitrary VMM allocations. The small native peer blocks therefore remain conventional CUDA
allocations even when the model uses expandable segments. At max_words=163840 each state is 1,314,816 bytes (about
1.254 MiB/card), outside Torch memory accounting. `test_runtime_expandable_allocator_cannot_change_peer_state_allocation`
checks that native state pointers lie outside all Torch-managed segments and runs randomized exact collectives after the
runtime allocator toggle. cudaMallocAsync remains explicitly unqualified/refused. A failed native free warns and requires
process restart; it never clears a poisoned context.

The model-free lifecycle's pair capacity matches `tests.test_p2p.MAXW` (163840 words), so every randomized payload fits;
`test_pair_lifecycle_capacity_covers_sequence` pins that contract. `--allocator-import` imports the installed exllamav3
Python package, allocates a probe and reports Torch segment types before constructing either peer state. Compare fresh
baseline processes with and without `EXL3_EXPANDABLE_SEGMENTS=0`, then the native-allocation implementation. Single-card
mode checks lifecycle/arithmetic only; cross-device raw-pointer mappings require the authorized pair. Use asynchronous
launches for successful collective controls: `CUDA_LAUNCH_BLOCKING=1` serializes the controller's rank launches and can
prevent the peer from joining a spin-wait collective.
