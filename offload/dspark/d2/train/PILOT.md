# Matched pilot with Amendment-4 sample and Amendment-6 acceptance rule

NDM approved the matched pilot at 16:57Z on 2026-10-23. Amendment 4 (17:28Z)
fixes its evaluation population before any real pilot training. The binding plan
is `documents/plans/2026-10-23-d2-plan.md`. Amendment 6 supersedes the old retention/spread rule before real pilot data. Reference arithmetic and final G2/packaging
on the full held-out set are unchanged.
Old failed arithmetic-agreement receipts remain failed and cannot authorize FAST.

`pilot prepare` writes the initial immutable-settings plan. CPU `pilot sample`
then makes its only permitted update: freeze a seed-20261024 random sample of at
most 1,000 complete fixed-K2 replay rounds from EACH of dev and heldout_legal (amendment 7).
It records the ordered key lists and SHA256, full population counts/hashes,
MANIFEST hash and exact K2 source digests. Splits use independent RNGs with that
fixed seed; keys retain their original traversal order. If a split has fewer
than 1,000 complete rounds, all its rounds are used. Partial five-target tails
are excluded. It reads tokens rather than materializing every round's taps.

Sampling is once-only, before training: any existing sample refuses redraw, even
when the caller repeats the same inputs. A nonblocking companion lock protects
its atomic/fsynced plan update. Every arm refuses an unsampled plan before model
load. Do not rerun prepare or sample after an interruption; resume only the
interrupted arm with the same sampled plan. Keep all real plans/receipts private:
they contain the sampled keys. Evaluation JSON contains only aggregates/hashes.

All three arms use the same first 20,000 blocks of the seed-23 shuffled training
population, zero-init LoRA B and identical initialized A/other trainables, r32,
raw taps, batch ceiling 128, vocabulary chunk 8192, AdamW lr .0002, weight decay
.01, clip norm 1, one matched cosine schedule and the unchanged KL objective.
REF/FAST have identical order; FAST-B reorders the SAME population with seed 24
without changing initialization seed 23. Batches split at exact 2,500/5,000/
10,000/20,000 boundaries (159 updates including partial batches). No projection
cache, confidence stage, initial adapter or hyperparameter override is accepted.
Changed source, plan, inputs, initialization or resume contract refuses.

Every step records private loss/target agreement; the evaluator also compares
REF/FAST evolving matched trajectories for position-1/block agreement and loss
differences. FAST-B's different-order diagnostics compare to its own targets.
These are informational and do not gate outcomes or claim fixed-weight parity.

`pilot_eval` scores exactly the saved sample on every baseline/checkpoint pass,
using the reference model, G2's `bucket_kl`/`match_length` and the fixed K2
trajectory. It refuses missing, extra, duplicate, reordered or miskeyed scored
rounds. Before loading a model, each job recomputes the seed-20261024 draw from
both complete held-out populations and refuses any difference from the frozen
sample, including a self-consistently rehashed alternative key list. This
verification never rewrites or redraws the plan. Candidate predictions never
choose the population. Each split can run
on its own solo card with `--split`; CPU `--merge-parts` verifies both split
results share the plan, sample, K2, checkpoint and step-evidence identities and
writes ONE aggregate JSON. Missing/foreign parts refuse. All thirteen passes
(baseline plus twelve checkpoints) use the same sampled keys.

`pilot_verdict` implements Amendment 6 (`amendment-6ab-source-prompt-bootstrap-v2`).
Primary E[A] is the G2 mean accepted draft prefix on the fixed replay trajectory,
excluding bonus/correction. It is a finite-sample estimate, not an acceptance
forecast on other trajectories. It retains per-request totals, source/prompt cluster identities and separate task labels.
Intervals pair the same observations across arms: 2,000 hierarchical bootstrap
resamples, seed 20261024, sample source families then whole requests within each
family, with all rounds of each selected request intact. Percentiles 5/95 give
the 90% interval; round weighting matches the point estimate. Coverage counts accompany each interval. Amendment 6a defines clusters as union-find
components of `source_docs` within each split, across all shard sidecars. Empty-source
A-captures cluster by `prompt_sha256`: acap0 and acap1 share a cluster (three clusters
in the historical acap fixture population). Task labels remain separate reported strata,
with per-task means and request/round counts. Every verdict receipt states the
cluster key and clusters per split; fewer than five clusters carries the explicit
caveat that the interval cannot establish between-family uncertainty.

At 10K and 20K on BOTH development splits: REF and FAST gain at least 0.05
accepted draft tokens/round over baseline, with interval lower bound >0;
FAST-minus-REF interval lower bound >=-0.03; FAST mean deficit to baseline <=0.01;
every position's median KL deficit to REF <=0.01 nats. REF without useful gain
is INCONCLUSIVE. NaN/inf/mismatched coverage refuses. FAST-B spread is diagnostic
only. Amendment 6b permits one further seed only when EVERY failed check
straddles its bound; there is no automatic extra training. Definitive statistical
failure or a hard KL/baseline regression yields FAIL even alongside a straddle.
REF without useful gain remains INCONCLUSIVE (adaptation not demonstrated),
including a missing FAST useful gain in that same cell; that exception cannot
excuse a definitive NI, KL, baseline or another cell's FAST-gain failure. A passing pilot never requests another seed.
Early checkpoints remain development data.

The real amendment-7 population is `dev` (reported as `dev`) and
`heldout_legal` (reported as `dev_legal`). Historical acap fixtures remain only
for regression/history and cannot admit production training. Both current
splits have been used for selection. A NEW reserved public TEST family set is
required for the frozen final candidate, never for checkpoint or hyperparameter
selection. Existing G2 ledgers do not make these development families fresh.

The versioned receipt binds the rule, sample, intervals, source, MANIFEST,
initialization, settings, parity/G1 and all twelve checkpoint hashes. Production
recomputes the decision, intervals and population report. Evaluator
`G2-reference-top20-tail-fixed-K2-amend6ab-v2` refuses prior evaluations. Synthetic
detection honours top-level `synthetic`, `checks.synthetic`, `taps_source`
(including checks metadata), and `generation.synthetic`; any true marker wins.
Old rule/evaluator versions, synthetic, FAIL and
INCONCLUSIVE receipts cannot authorize FAST. A real PASS authorizes only FAST
training, not final G2/G3, packaging or deployment.

## Prerequisites and leases

Run only on X86host after genuine G1(a), amendment-7b binding **dev** G1(b)
tripwire PASS, VALID live reference (foreign share <=2%), genuine 50-round
reference parity and exact K2 coverage. The current MANIFEST is
`/opt/dsv41/x86host-data/d2/decode-d2pass2-r1-c2/shards/MANIFEST.json`, SHA256
`450bb71653f3def68cd95ee9579a456c74c469d08a016940cceda24de48b4f30`.
G1 receipts and K2 files are under `/opt/dsv41/x86host-data/d2/g1-d2pass2-r1/`.
Historical acap G1 cannot admit this dataset. Do not use the first incomplete
copy attempt or earlier replay/amend6 pilot namespaces.

The coordinator's exact dry-default stage / prepare / train / evaluate / finish
procedure is [PILOT-STAGE.md](PILOT-STAGE.md). Use its immutable snapshot
`~/dsv41-offload/d2/pilot/decode-d2pass2-r1/repo`, its SOURCE_COMMIT and its output
`/opt/dsv41/x86host-data/d2/pilot/decode-d2pass2-r1`. Do not change that snapshot
while any pilot phase is running. The named solos remain GPU-240876ad and
GPU-bc62df2a, each at 225 W with its individual nonblocking flock and <=128 GiB
RSS/no-swap supervision. Neither production-pair card is authorized here.

## Exact coordinator steps

1. Stage the reviewed pilot snapshot using PILOT-STAGE.md; preserve its full
   SOURCE_COMMIT and all receipt sources at their original absolute paths.
2. Prepare then freeze the once-only sample before training.
3. Run REF and FAST/FAST-B, resuming only interrupted arms in the same namespace.
4. Evaluate both sampled development splits, merge and judge. The ONLY receipt
   that can authorize the full run is a genuine real PASS at
   `/opt/dsv41/x86host-data/d2/pilot/decode-d2pass2-r1/receipt.json`.
5. After that real PASS, use the separate dry-default **full_stage** procedure
   in [FULL-RUN.md](FULL-RUN.md). It uses the same native base and -c2 MANIFEST,
   amendment-7b dev G1, exact original pilot source and PASS receipt, plus a NEW
   reviewed immutable full snapshot and `/opt/dsv41/x86host-data/d2/full/decode-d2pass2-r1/`
   output namespace. `--pilot-source` verifies the byte-identical original
   pilot through its unchanged verifier; do not prepare a new pilot or change
   its source map to accommodate the full-run admission fix. Train fresh FAST
   for three epochs at B128/chunk8192/r32, then score **all complete rounds** of
   dev, heldout_legal and heldout_general in the reference path (general gated as D2-TRAIN G2).
   See [FULL-ADMISSION.md](FULL-ADMISSION.md) for source/gate refusals and the
   full-run cost sheet (about 2.70 FAST GPU-h plus up to 50.95 full-G2 GPU-h,
   ESTIMATED). The pilot sample restriction never applies to full G2.

On FAIL report to NDM; no new arithmetic round, altered sample or changed pass
rule is authorized. Never commit real sample plans, receipts, shards or weights.

## Estimates and validation limits

At the measured reference rate ~0.50 s/round, evaluation is capped at
13 passes * 2,000 rounds * 0.5 s ≈ **3.6 GPU-hours**, or **1.8 hours wall time**
with the two split jobs on the two solos. The prior measured median 0.5046 s gives
3.64 GPU-hours; initialization, source/shard validation, checkpoint application
and CPU merge add overhead. Smaller held-out populations cost less. REF training
still projects to about 1.9–2.2 GPU-hours (20K / 2.5–2.9 blocks/s), plus I/O;
concurrent FAST/FAST-B are shorter. Budget roughly four hours wall time for
pilot training plus parallel evaluation, subject to real-shard measurements.
This bound applies ONLY to the pilot; final G2/packaging still use full sets.

The earlier d4f03bf synthetic rehearsal completed all three 200-block arms and
13 full reference passes; synthetic FAIL was preserved/refused. The original
50 captured reference outputs were bitwise unchanged, and B128 with diagnostics
peaked at 20.154 GB reserved. That historical receipt predates Amendment 4 and
cannot qualify the Amendment-5 policy. This follow-up validates sampling, exact
scored-key coverage, split merge and receipt refusal on CPU; it claims no new GPU
benchmark or real G1/pilot outcome. Synthetic labels/targets are not legal-quality
evidence. Any future synthetic rehearsal must also prepare/sample BEFORE training.

## Matched units and complete cost accounting (Amendment 6)

The plan's `matched_units` records each arm at each boundary: blocks, VALID
supervised tokens (sequence tails count only their unmasked positions), optimizer
updates, accumulation=1, B128 ceiling, the exact applied-LR list and official
QDQ/STE semantics. Boundaries 2500/5000/10000/20000 mean 20/40/80/159 optimizer
updates, not 2500/5000/10000/20000 updates. FAST-B now permutes WITHIN each
boundary interval, so its checkpoint populations and valid-token totals exactly
match REF/FAST. Initialization remains shared. Checkpoint counters and actual
applied learning rates are checked against the frozen plan; production rereads
the dataset and recomputes those units. Old plans must be prepared anew.

ESTIMATED from the prior synthetic uncached rates (REF 2.7 blocks/s, FAST and
FAST-B 190 blocks/s; these are not real-data rates): 60,000 blocks, 477 updates
and <=300,000 supervised tokens across three arms cost about 2.116 GPU-hours.
The actual token count is the plan's metadata count, not blocks times five.
MEASURED historical reference evaluation ~0.505 seconds/round (D2-PILOT report),
so 13 passes over <=2000 rounds cost <=3.647 GPU-hours. Budget additionally
ESTIMATED 1–1.5 GPU-hours for replay (30–45 minutes on TWO target ranks in a
coordinator window) and 1 GPU-hour for adapter serving qualification (two solo
ranks for 30 minutes, including graph capture/proxy checks). Combined budget
is approximately 7.76–8.26 GPU-hours plus model load, I/O, queueing and final
full G2/fresh-test work. The two-solo ideal balanced lower bound for training+
pilot evaluation+worker checks is ~3.38 hours; the target replay adds a separate
30–45 minute coordinator window and is not parallel solo capacity. REF alone
needs ~2.06 hours, so schedules must account for that lower bound too.

`pilot_units.estimate` requires explicit positive replay and serving costs,
accepts measured/declared arm rates and the plan's actual units, and reports
components separately. No four-request acceptance forecast is used.

## Pre-pilot audits

`python -m offload.dspark.d2.train.audits --synthetic --out AUDIT.json` exercises
the complete tiny CPU drafter and writes EVERY parameter name with nonzero,
zero or missing gradient after one AdamW step. LoRA A must be nonzero and B
zero initially; initial A gradients being zero is expected, B must receive a
signal. The confidence-only head is explicitly excluded from the KL arm.
Run the real-size version with `--ckpt` and `--manifest` under the solo lease
before real training, preserving its private aggregate report. The synthetic
result is a plumbing audit, not evidence of a real-data adaptation gain.

Projection caches are now refused, including detached outputs: the old tensor
API cannot prove both weights and upstream inputs stayed frozen. The pilot
continues with raw taps. `DependencyGuard` invalidates Markov/feature state and
graphs on any adapted tensor version change and requires a fresh worker.
CPU audits compare analytic CE/KL/TV backward with independent unfused full-vocab
softmax/bucket expressions, repeated previous IDs, masked positions, a label
outside top-k and intentional identity-STE QDQ. Numerical rounding does not waive
gradient checks. GPU precision and real-worker outcome checks remain separate.

The isolated real-worker evaluator and solo-card GPU qualification commands are
in [WORKER-EVAL.md](WORKER-EVAL.md). CPU checks do not authorize deployment.
