# NB1 item 2 — default-OFF batch-invariant dispatch for native kx2 (MEASURED 2026-10-26)

**Answer.** One environment switch, `DSPARK_NATIVE_BATCH_INVARIANT`, read when the native drafter is built:

|value|behaviour|
|---|---|
|unset / `0` (default)|unchanged. Byte-identical to the frozen N4 runtime on every probe output (below).|
|`1`|draft path: every draft-body shape B×5 (B = 1..4) runs the dispatch B1 uses — FP8 with B1's split-K count at M5/M10/M15/M20 (no cuBLAS BF16 fall-back at B4, no paired BF16 shared gate/up at T20), and B1's fused-expert variant at every row count.|
|`2`|`1` plus window insertion: `main_proj` and `wkv` run in fixed 8-row sub-batches (the bucket-8 arithmetic) whatever the flush size or companions.|
|anything else|refuses at build (fail loudly). Only the kx2 dispatch implements it; other arms refuse.|

With `1`, IDs **and** confidence are bit-identical alone vs packed at B2, B3 and B4 on every captured and
synthetic state, eager and graph, n_out 4 and 5 (3,456 / 3,456 alone-vs-packed request comparisons), no operation diverges anywhere in the
draft, and every B1 output is bit-identical to today's. With `2` the same holds for the draft, and every packed
window insertion (1, 2, 5, 32 rows per request; B2–B4) equals the alone insertion; B1 inserts above 8 rows
(prefill chunks) change, and their distance to FP64 is not worse (below). Latency costs: see "Cost".

## What the switch changes (smallest change that removes every item-1 divergence)

`native/kx.py`: `KXSelectiveDense(..., splits=)` pins one split-K count for every selected M (row tiles stay per
M — they partition rows, not the K reduction; item 1 confirmed tiles are arithmetic-neutral); `KXFusedBank(...,
variant_rows=)` selects every row count's expert variant as for the B1 block (131072 on all stages). Item 1 found the
stage-1 mode 256 bit-identical on these states, so this is a code-path simplification rather than a measured fix.
`native/tp2_native.py`: builds those for M ∈ {5, 10, 15, 20}, drops the T20 paired GEMM, and (level 2) routes the
insert projections through `_fixed_rows`. `native/arms.py`: the torch-free switch parser and `kx2_dispatch`.
`native/identity.py`: the identity recipe names the level when on and is unchanged when off (production's K2
identity covers checkpoint files only and is unaffected; the native identity digest changes with any source edit).
Tests: `native/tests/test_batch_invariant.py` (11 CPU tests: parser, B1 split preserved, scratch shapes, identity,
sub-batch helper).

## Gate (MEASURED; `evidence/item2/`)

Same probe and fixtures as item 1 (12 captured N4/V1 states, 12 synthetic stress states, every ordered B2 pair and
B3/B4 permutation), plus insertion at 32 and 128 rows per request and FP64 accuracy of every B1 insert projection.
Runs on bc62df2a at 225 W under the lane lock, all rc 0 (`evidence/item2/f-*.json`, reduced by `collect.py`;
gate `evidence/item2/gate.json` by `gate.py`, PASSED at both levels):

|run|code|switch|output digest (all eager/graph outputs + windows)|isolated batch-dependent op cells|
|---|---|---|---|---|
|f-native-frozen|frozen N4 runtime (`~/dsv41-offload/n4`)|n/a|`014d0ec685dfc08c…`|118|
|f-native-off|NB1 branch|unset|`014d0ec685dfc08c…` (identical)|118 (identical)|
|f-native-on1|NB1 branch|`1`|`f109961e38de7bdd…`|0|
|f-native-on2|NB1 branch|`2`|`215fc62ccab324a5…`|0|
|f-k2-off|NB1 branch, K2 arm|unset|`7e8e9c8de246bffb…` = item 1 / frozen K2|42 (unchanged; K2 is not touched)|

1. **OFF is inert.** The output digest, all 24 B1 digests and every aggregate in the evidence (first divergence,
   per-op gaps, FP64 labels, depth transitions, insertion rings, dense configs) equal the frozen runtime's, with no
   differing key. The switch is read once at build; unset and `0` are the same code path.
2. **Level 1 is batch-invariant for the draft.** All 18 cells (captured/synthetic × B2/B3/B4 × eager n5, graph n4,
   graph n5): IDs at every position and confidence bit-identical (`conf_max_abs` 0.0) in 3,456 / 3,456 comparisons.
   End-to-end first-divergent op: none in all 360 placements; isolated replay: no batch-dependent op anywhere
   (default: 118 cells, starting with stage-0 `wo_a`). Depth transitions alone→packed are all k→k (default: 3→2 on
   6 / 72 captured B2 and 18 / 216 captured B3 placements — real states drafting one token shorter when packed).
3. **Level 1 keeps B1.** All 24 alone (B1) digests equal the frozen runtime's.
4. **Insertion (level 2).** The default and level 1 do not batch-invariantly insert the window: alone vs packed
   rings differ at 5 rows per request (B2 60/96, B3 180/288, B4 144/384 slot-positions exact) and at 32 rows B2
   (0/48) — `main_proj` is FP8 only at 8 rows and cuBLAS picks per-M kernels. At level 2 every ring is exact:
   1, 2, 5 and 32 rows per request at B2/B3/B4, 2,688 / 2,688. This matters for production because flushes pack
   several requests' accepted tokens; level 1 alone leaves that path batch-dependent.
5. **Level 2 and B1.** Level 2's draft is level 1's (all 3,456 exact, no divergent op). Its B1 insertion arithmetic
   changes when a request inserts more than 8 rows at once (prefill chunks and the 32/128-row cases): 12 of 24 B1
   digests differ from frozen, as expected. The pre-registered fallback (`gate.py`, bound fixed before the
   numbers were read): mean |output − FP64| of every insert projection (main_proj, wkv × 3 stages) at every
   rows-per-request (1, 2, 5, 32, 128; captured and synthetic) not worse than 1.05 × the default's. Measured
   worst ratio **1.0011** (stage-2 wkv, 32 rows); main_proj ratios 1.0000; max-abs errors identical in all 32
   cells. B1 at level 2 is therefore not less accurate than today.

## Cost

Per-shard CUDA-graph replay on bc62df2a (`latency_probe.py`, one process per TP2 rank, `solo_rank` with local
collective stand-ins): median of 5 interleaved runs, each the median of 200 replays after 40 warm-up, every arm and
cell in one rotated order per run. Table = the slower shard. Full table incl. microbatching and per-rank values:
`evidence/item2/LATENCY.md`, `latency.json`. MEASURED 2026-10-26 16:22–16:43Z.

|B|n_out|K2 µs|native default µs|native invariant µs|invariant − default|
|---|---|---|---|---|---|
|1|4|2217|1953|1956|+0.2 %|
|1|5|2310|2026|2029|+0.1 %|
|2|4|2870|2646|2663|+0.7 %|
|2|5|2883|2703|2751|+1.8 %|
|3|4|3279|3078|3165|+2.8 %|
|3|5|3377|3184|3251|+2.1 %|
|4|4|3934|3759|4010|+6.7 % (+251 µs)|
|4|5|4450|4423|4690|+6.0 % (+267 µs)|

- **Draft.** B1 unchanged (same dispatch; differences ≤ 0.2 %). B2/B3 cost +0.7–2.8 % (≤ 87 µs per shard per
  round), below the 4.4–6.6 % spread between runs of one cell, so "at most a few per cent". B4 costs 6–7 % on both
  ranks and both n_out — real: at M20 the default falls back to cuBLAS BF16 (and the paired shared gate/up GEMM), and
  the invariant mode keeps FP8 with B1's split. The invariant mode stays faster than K2 at B1–B3 and is 1.9–5.4 %
  slower than K2 at B4.
- **Insertion (level 2 only).** Fixed 8-row sub-batches serialise the insert projections:

  |rows per flush (graph bucket)|8|16|32|64|128|
  |---|---|---|---|---|---|
  |native default µs|216|325|361|404|620|
  |level 2 µs|225|388|715|1361|2723|
  |ratio|1.04|1.19|1.98|3.37|4.40|

  A steady-decode flush packs each request's accepted tokens (≤ 6 rows each, ≤ 18 at B3: bucket ≤ 32, at most
  +0.35 ms per flush); prefill-chunk inserts pay ≈ +2.1 ms per 128 rows. Level 1 has no insertion cost but leaves
  window insertion batch-dependent (gate item 4). One launch over all 8-row tiles would remove most of this cost; it
  needs a kernel change and was not attempted.
- The card is capped at 225 W and throttles under this load (SM 1050–1740 MHz across runs, recorded per run in
  `latency.json`); interleaving shares that across arms. Per-shard compute only: TP2 collectives, transport and
  async overlap are not in these numbers (W8's measured drafter round trip was 2.90 ms p50 in every arm). Whether
  the extra microseconds are exposed at the consumer deadline is a window measurement (item 4).

## Limits

Solo-card emulation of both TP2 ranks; no physical-pair numerics and no target acceptance are claimed. The
invariance holds for B ≤ 4 (the fused experts' 20-row limit and production's slot cap). Insertion invariance at level
2 covers graph buckets (multiples of 8) and eager inserts (zero-padded). The switch makes packed drafts equal B1
drafts; it does not make B1 more accurate — whether that recovers W8's B2/B3 loss is a window question (item 4).
