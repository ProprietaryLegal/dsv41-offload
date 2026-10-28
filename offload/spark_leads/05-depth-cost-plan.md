# Depth-cost table and legal calibration

Design pin: sudoingX f8f20b370e41a3a220bc7d1ab96a61e4daa84964 confidence-feature/depth work is a lead;
our fitter is new stdlib CPU code, not its weights/cost constants. Local source base 21bc3ab, CONTRACT1.8.
Affected op: Spark choice of per-request verification depth; numerator is expected committed target tokens, denominator
is paired full round cost including uncovered prep/queue/CPU gaps. No target sampler/reasoning change.

Executable: `depth_cost.py INPUT.jsonl --out TABLE.json` (Python3, no GPU/dependency). Exit2 writes an explicitly
unqualified table if fit/held-out group coverage is insufficient. Invalid data/build mixture/leakage raises.
Existing bench2 JSONL request/run records lack paired verify/prep/depth rows and are deliberately refused; R1's
coarse histogram subtraction cannot supply these fields. No numerical calibration claim is made without new
COORD-owned paired captures. Existing legal prompts/captures remain on the four authorized machines.

Input schema `dsv41.depth-round.v1`, one engine step:

| Field | Meaning / bound |
|---|---|
| sample_id, group_id, partition, build_id | Unique round id; stable hash of original legal source/task FAMILY (not run id); fit/heldout; engine+overlay+worker+weights+transport/calibration signature. A group cannot occur in both partitions; a mode cannot mix builds. |
| mode | local, remote_sync, remote_async, two_card_tp, two_card_affine (not just “2 GPUs”) |
| n_requests, depth_vector | 1..4, one chosen depth0..5 per decode request; do not infer from requested maximum k |
| planned_verify_rows, graph_rows | sum(1+depth) and actual captured/padded target shape (eager: actual rows); keep context insertion/draft block rows separately in raw capture |
| context_tokens_max, cache | Actual maximum occupied context in this batch; short<=8192, <=32768, <=150000, <=256000, <=300000, over300k; cold/prefix_hit/steady. Keep per-request contexts and identity in protected raw capture. |
| verify_ms | Target verify GPU interval from paired events; excludes remote prep and queued-before-event time |
| prep_ms | Matched extraction + row/meta D2H + pack + RPC + worker insert/draft + reply H2D/broadcast COMPLETION, per paired trace; no queued target wait hidden inside gather wall time |
| overlap_ms, queue_ms | Measured interval intersection of verify/prep, and service/host queue wait. No invented residual from different medians. |
| round_ms, committed_tokens | Direct actual round start-to-next-verify boundary; actual emitted/committed tokens across this batch (0..planned rows), not proposed/rejected tokens or acceptance-only proxy. |

Output `dsv41.depth-table.v1`: exact cells indexed by mode × request count × planned verify rows × graph rows × context
bucket × cache category × depth vector. Train round mean/p95 and committed mean, separate phase means, sample/group
counts, per-heldout-round signed/relative error and actual-vs-predicted commits. No sparse-cell nearest-neighbour
fallback. Unknown cell uses unchanged current policy; do not silently use local or another mode's costs.

Estimator: empirical conditional mean on fixed shape cells, with p95 as tail penalty input; score =
E[committed] / (E[full_round_ms] + lambda*max(p95_round-E[round],0)). Queue cost already in full round; do NOT add prep
or queue again. Phase fields diagnose where cost moved. Per-round errors remain inspectable, including negatives.
The fitter does not wire a policy into production. The existing six-value ROUND_MS+DRAFT_MS interface cannot express
this multi-dimensional table: a separate reviewed adapter must select a measured cell per batch, never collapse and
claim equivalent calibration. Keep existing adaptive/conf policy as control.

Coordinator run plan:
1. Inventory protected capture task identities from existing JSONL prompt_sha256 and .pt round manifests; group ALL
   variants/repeats/prefixes from the same matter/source/task together. Split groups before fitting (e.g. salted
   SHA256 task-family, first20% heldout). Keep enough independent legal reasoning and prose tasks; no new export.
2. Use a small legal screening set, fixed depths0..5 first at B1 and B3/4, short/32K/150K steady context; expand only
   winning shapes to256K/300K and B2. Record effort75, T1, priority10; legal reasoning/prose strata explicit. Mark
   terminal/EOS/mixed/prefill-emission rounds separately; do not reconstruct commits from acceptance ratios.
3. Paired markers from A2/R1 for verify/prep/overlap and direct rounds; n_out held fixed compatible with planned
   depths. Randomize/interleave current local, remote-sync/async, conditional TP2/affine2 with same power policy.
   Network helpers paused. Log source/build ids, graph mode, inserted rows, n_out, Markov hits and concurrency.
4. Need >=3 independent fit groups and >=2 heldout groups per supported cell by default. Fit cost-only table first;
   heldout signed/p95 absolute relative errors by mode/ctx/shape; stop qualification on >10% median absolute cost
   error or >20% p95, systematic bias, insufficient coverage, negative speed/acceptance tradeoff or fairness loss.
5. Shadow policy choices against current policy on untouched heldout legal tasks, log each chosen depth vector,
   cell, predicted/actual cost/commits, score, error and unknown-cell fallback. A fitted confidence acceptance model
   (margin + top1 + confidence) is a separate future fit using the same heldout split, not donor coefficients.
6. Only after heldout numerical/quality/concurrency and ~5% useful end-to-end gain: reviewed runtime adapter and
   rollback to current conf/adaptive policy. Never tune on heldout prompts then call them heldout again.
