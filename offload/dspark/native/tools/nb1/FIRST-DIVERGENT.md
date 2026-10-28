# NB1 item 1 — first divergent operation, alone vs packed (MEASURED 2026-10-26)

**Answer.** In native kx2 the first operation whose output changes when a request is packed with others is the
stage-0 attention output projection **`wo_a`** (grouped FP8, `KXSelectiveDense`): its split-K count is 8 at M5
(B1) and 4 at M10/M15 (B2/B3); at M20 (B4) it is not in the measured FP8 set at all and falls back to a BF16
`bmm`. Its gap is one BF16 rounding step on a few elements (max abs 0.00098 at B2/B3, 0.0078 at B4) and lies
**within the operation's own numerical error** (mean gap / mean error against FP64 ≤ 0.0008). The same family —
FP8 split-K 8→4 at M10/M15 for `wo_b`, `sh_w2` and `wo_a`, and the B4 fall-back of every draft projection to cuBLAS
BF16 (plus the paired BF16 shared gate/up at T20) — is the **only** batch-dependent arithmetic in the native draft
body. Fused experts (including stage 1's mode 256 at ten rows), routing, attention, norms, mHC, act QDQ, the K6
head, Markov update and the confidence matmul are **bit-identical** at the packed shape on identical inputs.

The tiny projection gaps then cross discontinuities downstream: act-QDQ (e4m3 step) and top-3 routing (expert ids
change on some rows from stage 1 on), so the request's hidden state, logits and confidence move far more than any
single operation's error. That is why confidence moves up to .052 (captured) with the projection gaps at 1 ulp.

At B4 the cuBLAS BF16 fall-back for `wo_b` is **beyond** its own error: gap ≈ 3–3.6 × the FP8 path's mean FP64
error and the packed result is itself 3.1–3.8 × less accurate (`sh_w2` at B4: 1.48–1.71 ×, around the √2 rule).
At B2/B3 every gap is within error (ratios 0.85–1.0 for the split-K changes).

## Method

`stage_probe.py` ran on x86host's solo `GPU-bc62df2a` (225 W, own `flock -n`, nice 19, CPUs 12–15/28–31, 48 GiB
scope), both TP2 ranks emulated with local collectives, reusing N6's frozen N4 d1dfe13 runtime read-only
(116 runtime source hashes; prebuilt `kx_fp8dense`/`kx_fusedmoe`/`kx_mxfp4_regfed`; no Triton compile was needed).
Fixtures, all with four requests in slots 0–3:

* 12 captured N4/V1 states (the N6 set) as three groups of four;
* N6's nine synthetic stress states (identical draws, seed 20261024) plus one NB1 state per case
  (last position 31 / 65 535 / 262 143, seed 20261026) — the 4-request case N6 did not cover.

Each request is drafted alone (B1), then packed in every ordered pair (B2), and every permutation of three (B3)
and four (B4): 60 placements per fixture. Every top-level operation of both ranks is recorded (201 per native draft, 165 per K2 draft).
For each operation the probe reports

* **end to end** — the request's rows of the packed output against its alone output; the first inexact operation
  in program order is "the first divergent operation";
* **isolated** — the operation re-run at the packed shape on the requests' *alone* inputs, so only the
  operation's own dispatch can differ;
* **FP64** — for every isolated gap (first two placements per op/kind/B), the operation's reference on the same
  alone input: exact-decoded FP8/BF16 weights with an FP64 dot; the fused experts with the official rounding points
  (BF16 g/u, SwiGLU with route weight, BF16 h, act QDQ) and FP64 dots; EXL3 (K2) from the reconstructed trellis
  with FP64 Hadamard rotations. A gap is **within the operation's own error** when
  mean|alone − packed| ≤ √2 × mean|alone − FP64| (two evaluations with independent, equally large rounding errors
  differ by about √2 × that error); the probe also reports max values and packed-vs-alone error ratios.

Window insertion was traced the same way (graph-bucket padded eager inserts, 1/2/5 rows per request, synthetic
embedding taps and captured taps).

## Results (MEASURED; evidence `evidence/item1/*.json`, tables in `evidence/item1/SUMMARY.md`)

Final outputs, graph path, n_out 5, request alone vs packed (positions 1–5 equal / comparisons):

|arm|kind|B2|B3|B4|conf max abs B2 / B3 / B4|
|---|---|---|---|---|---|
|native kx2|captured|72 all|216 all|288 all|.0524 / .0524 / .0432|
|native kx2|synthetic|60,60,54,54,54 /72|180,180,162,162,162 /216|216,168,144,120,120 /288|.182 / .182 / .206|
|K2|captured|72 all|216 all|288 all|.0076 / .0062 / .0059|
|K2|synthetic|54,54,42,36,30 /72|180,162,144,90,72 /216|288,264,264,240,192 /288|.140 / .257 / .088|

Captured B2/B3 confidence maxima reproduce N6 exactly (.05238831 native; .00758338/.00623515 K2). Graph = eager in
all 384 (native) and 384 (K2) checks; same-B companion/order changes are bit-exact in every case (B2–B4).
ESTIMATED inherited depth policy (production caps B2 n5, B3/B4 n4): native 3→2 on 6/72 B2 and 18/216 B3
placements (one captured state, as N6), none at B4; K2 none on captured states.

First divergent operation (every placement where anything diverged):

|arm|B2|B3|B4|
|---|---|---|---|
|native kx2|stage-0 `wo_a` (FP8 split-K 8→4)|stage-0 `wo_a` (split-K 8→4)|stage-0 `wo_a` (FP8→BF16 `bmm`)|
|K2|stage-0 shared `sh_w2` (EXL3 plan by M)|stage-0 `wq_b` (EXL3 plan by M)|stage-0 `wq_b`|

Native isolated batch-dependent operations (every other one of the 84 per-rank operation families is bit-identical):

|operation|B2/B3|B4|FP64 label|
|---|---|---|---|
|`wo_a` (FP8, groups 4), 3 stages|split-K 8→4: ≤ 1 BF16 ulp, 66–100 % of comparisons|BF16 `bmm`|within (gap/err ≤ 8e-4) at all B|
|`wo_b` (FP8, FP32 out), 3 stages|split-K 8→4: max 7.6e-6, gap/err 0.94–1.01|cuBLAS BF16: max 5.3e-5|B2/B3 within; **B4 beyond: gap/err 2.9–3.6, packed error 3.1–3.8 ×**|
|`sh_w2` (FP8, FP32 out), 3 stages|split-K 8→4: gap/err 0.85–0.92|cuBLAS BF16|B2/B3 within; B4 1.48–1.71 (around √2), packed error 1.49–1.71 ×|
|`wq_a`, stages 1–2|exact|cuBLAS BF16, 25–42 % of comparisons|within|
|shared gate/up|exact|paired BF16 `mm` at T20, 25–42 %|within|

K2's batch-dependent operations are its EXL3 GEMMs whose kernel plan is keyed by row count: `sh_w2` at every
B, `wq_b`/`wo_b` at B3/B4; gaps 1.05–1.36 × the op's own (FP16-rotation) error, within √2; `wo_b`'s M15/M20 plan
is 1.43–1.5 × less accurate. K2's coop MoE, head and Markov are bit-identical.

Propagation (captured, native, end to end): `wo_a` 1 ulp → stage-0 attention all-reduce max .125 → act QDQ and
routing: from stage 1 on, the routed expert ids of some rows change (`route_fused` ids differ) → final hidden
state max .48–.99, Markov logits max ≈ 4, confidence max .052. K2 shows the same mechanism (routing changes from
stage 1/2) with smaller confidence movement on the captured states.

Insertion: 1 or 2 rows per request stay in graph bucket 8 at every B and are bit-exact in both arms. At 5 rows per
request the packed flush crosses into bucket 16/32: native's first divergent op is `main_proj` (FP8 at M8, cuBLAS
BF16 at M16/M32; isolated gap max .0625, within its own error) and `wkv` (cuBLAS at M8 vs M32); 48/96 (B2),
144/288 (B3), 96/384 (B4) request windows stay bit-exact, ≤ 41 KV elements differ. K2: EXL3 `main_proj` by bucket
(gap ≤ 1.1e-5, within), no request window exact at 5 rows, ≤ 251 KV elements differ.

## What this rules in and out

* N6's three suspects: FP8 split-K 8→4 **confirmed** as the first divergence; stage-1 fused-expert mode 256 at M10
  **ruled out** (bit-identical to the B1 mode on every comparison); insertion `main_proj` FP8 at M8 vs BF16
  **confirmed** for packed flushes that cross bucket 8. New: B4 (not covered by N6) falls back to cuBLAS BF16
  for every draft projection, the one place where the batch gap exceeds the operation's own error.
* Every B2/B3 gap is no bigger than the operation's own rounding error; the visible effect comes from the model's
  discontinuities (act QDQ, top-3 routing), not from an inaccurate kernel. The defensible fix is therefore to make
  the dispatch independent of the batch (item 2), not to make any one kernel "more exact".

Limits: twelve early states of one captured request and twelve synthetic stress states; repeated placements are
coverage, not independent samples. No target acceptance is measured. Physical-pair numerics are not claimed.
