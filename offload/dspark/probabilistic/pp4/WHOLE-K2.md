# PASS — whole K2 timings and N4 greedy identity qualified

The sampled path replaces the selector in the complete production-weight K2
worker. On the assigned solo GPU, its measured incremental median is 0.288–0.791 ms
per round across B1–B4 and output depths 4/5. All required greedy identities pass,
including the older runtime’s archived N4 proposals and confidence. Production
is unchanged. Physical-pair overhead and live target performance remain unmeasured.

## MEASURED whole-worker cost

Both logical TP2 ranks execute on one SM86 card at 225 W. The timing fixture
contains four real N4 request states, with fixed query block 5 and output depth 4/5.
The table reports medians of five interleaved runs, each with 20 warmups and 100
individually event-timed rounds. Each p95 is the maximum per-run p95; neither
that statistic nor the median difference is a live-pair upper bound.

| Batch | Output depth | OFF median ms | ON median ms | Increment ms | OFF p95 ms | ON p95 ms |
|---|---|---|---|---|---|---|
|1|4|3.666944|3.954976|0.288032|3.777536|4.332960|
|1|5|3.686400|3.999392|0.312992|3.767296|4.157408|
|2|4|4.331520|4.822768|0.491248|4.396032|4.921664|
|2|5|4.350976|4.892960|0.541984|4.452352|4.949344|
|3|4|4.888576|5.417216|0.528640|4.979712|5.526976|
|3|5|4.898816|5.484656|0.585840|4.968448|5.576160|
|4|4|5.779456|6.264480|0.485024|5.894144|6.401984|
|4|5|6.790144|7.581072|0.790928|6.918144|8.055680|

These measurements include the full K2 body, head, Markov conditioning, proposal
selection, graph replay and request-metadata handling. They exclude physical
NVLink exchange, proxy transport, target q expansion/correction and consumer wait.
Both ranks run sequentially on the solo; this is not production-pair latency.
The new peer exchange still needs the coordinator window.

## MEASURED identity and runtime qualification

- All 120 identity cells pass: 48 all-T0 and 72 mixed-T0/T1 cells, totaling 228
  T0 request comparisons across 24 unique N4/NB1 states. Mixed masks alternate
  both ways, covering every packed row offset. IDs and confidence are exact;
  meaningful q payload on T0 rows is zero.
- All 48 archived B1 proposal digests and all 48 confidence digests match the
  NB1 K2 receipt under the older CPython 3.12/Torch 2.10 runtime. This detects
  no arithmetic drift on those 24 states at either output depth. It does not
  prove identity for every possible state or the untested physical pair.
- Production checkpoint and Markov identities are verified against their
  source/extracted-payload receipts before the model loads.
- Peak host RSS was 6.95 GiB; peak Torch GPU allocation was 9.47 GiB. The full
  corrected run completed in 172.65 seconds. The card remained at 225 W; observed
  active clocks varied within the cap.
- The required CPython 3.13/Torch 2.12 CUDA 13 environment successfully imports
  rebuilt stock EXL3, FP32-accumulation EXL3 and P2P together. P2P has only CPU
  build/import qualification; no production-pair GPU was used.

[full-worker.json](full-worker.json) contains the complete aggregate receipt,
per-run medians, clock readings and source/binary hashes.
[runtime-build.json](runtime-build.json) and [RUNTIME.md](RUNTIME.md) record the
dependency qualification and reproducible build instructions. No token IDs,
confidence values, capture tensors, prompts, weights or generated text appear
in these receipts. The archived reference contains digests only.

## Harness correction and closeout

An initial loader attempt set `max_rows=128`, exceeding the production expert
scratch bound of 256 slots at `top_k=3`. It exited on the explicit guard. The
corrected `max_rows=32` covers B4 × block 5; insertion buckets remain independently
captured through 128 rows. No production assertion or test was weakened.

The final source, runner and archived-reference hashes reconcile with the
receipt. The solo returned to 1 MiB at 225 W, its lock is free, and the private
runtime tmux server is absent. Build/failed-attempt logs and the full public
object-reuse certificate are preserved as compressed campaign artifacts.
All runtime scratch remains scheduled for removal by the root lane.
