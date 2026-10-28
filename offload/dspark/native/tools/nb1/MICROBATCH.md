# NB1 item 3 — drafter-only microbatching vs packed drafting (MEASURED per shard, 2026-10-26)

**Answer.** Running each request as its own B1 sub-batch on the worker costs **1.48–1.49× the packed draft at B2,
1.90× at B3 and 1.83–2.08× at B4** per shard (+1.3, +2.8–2.9 and +3.7–4.1 ms of worker time per round). It buys the
same draft invariance the batch-invariant mode (item 2) gets for +0.7–2.8 % at B2/B3. On worker time the invariant
mode is the better route, so the window arm (item 4) uses it. **This is not the deciding number.** The worker drafts
asynchronously, overlapped with the target's verification. What decides is the time exposed at the consumer
deadline, i.e. how long the consumer waits for the draft under async overlap, and only a window on the production
pair measures that.

## Measurement

`latency_probe.py` (same run as item 2's cost table, `evidence/item2/LATENCY.md` / `latency.json`): per TP2 shard
(`solo_rank`, collectives are local stand-ins), the native kx2 default arm's B1 CUDA graph replayed B times back to
back, once per request, with each request's IDs/confidence copied out before the next replay (rank 0 owns the
outputs). That is what a worker-side microbatcher would launch. "Packed" is the same arm's B-row graph. Median of 5
interleaved runs × 200 replays, slower shard, µs, MEASURED:

|B|n_out|packed (native default)|microbatched (B × B1)|ratio|extra per round|packed invariant mode (item 2)|
|---|---|---|---|---|---|---|
|2|4|2646|3911|1.48|+1265|2663|
|2|5|2703|4034|1.49|+1331|2751|
|3|4|3078|5853|1.90|+2775|3165|
|3|5|3184|6036|1.90|+2852|3251|
|4|4|3759|7832|2.08|+4073|4010|
|4|5|4423|8084|1.83|+3661|4690|

Microbatched time is B × the B1 time (B1: 1953/2026 µs) to within 1 %: the B1 graph does not fill the card, so
packing four requests costs 1.9–2.2× one request rather than 4×. Microbatching gives that saving back.

## What the number leaves out

- **Pair cost.** Per-shard compute only. On the pair each sub-batch also pays its TP2 collectives and a graph
  launch, so the pair-level ratio is expected to be at least the per-shard one (ESTIMATED direction, not measured).
- **Insertion.** The draft is invariant by construction: each sub-batch is the request's own B1 graph on its own
  slot. That was not separately measured; item 2 measured B1 graph outputs equal to frozen B1. Window insertion
  stays packed unless it is also split per request. Item 2 found packed insertion batch-dependent at 5 rows per
  request and at 32 rows B2. Splitting it costs one insert graph per request: B2 × 5 rows becomes 2 × 216 µs (8-row bucket)
  instead of one 325 µs 16-row insert.
- **Implementation.** No worker microbatcher exists. This measures the launch pattern one would use, not a
  serving path.

## Why a window decides

The draft is consumed at the next round's deadline. W8 measured the drafter round trip at 2.90 ms p50 in every arm,
against 60.4 / 73.7 ms target steps at 32K B2 / B3. If the extra worker time is fully hidden behind verification,
microbatching costs nothing. If it is fully exposed, it adds at most about +1.3 / +2.9 ms per round: ≈ +2.2 % /
+3.9 % of step time (ESTIMATED upper bound, simple ratio of the two MEASURED numbers). That is the same size as W8's
B2/B3 losses (−4.44 % / −2.57 % per request). The invariant mode's exposed bound is ≤ 0.09 ms per round for the
draft plus ≤ 0.35 ms per steady-decode insertion flush. Only the audited async-overlap counter at the first draft
consumer can say how much of either is actually exposed (N6 `NEXT-WINDOW.md`, item 3 counters; NB1 arm).
