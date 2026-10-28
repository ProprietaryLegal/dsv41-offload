# OBJ1: D2 objective and conditioning audit

2026-10-23, CPU-only Controller, branch offload/obj1. No serving changes or private captures/weights used.
The KL pilot remains the controlled first arm. This document distinguishes source facts, CPU MEASURED tests,
proposed experiments and unvalidated serving assumptions.

## 1. Existing objective and corrections for every arm

Let S be the teacher's retained support, p_i = exp(top_logp_i), p_T = 1-sum_S p_i.
The replay parser keeps the returned global normalization and records p_T separately (`d2/replay.py::parse_logprobs`).
For student logits z, compute logZ over ALL vocabulary chunks and logZ_T over the complement of S.
Then log q_i = z_i-logZ and log q_T = logZ_T-logZ. Neither distribution is renormalized within S.

The objective is coarse forward KL:

    sum_{i in S} p_i (log p_i - log q_i) + p_T (log p_T - log q_T)

or coarse TV = (sum_S |p_i-q_i| + |p_T-q_T|)/2. Both are lower bounds on the corresponding full-vocabulary
divergence; within-tail allocation is unobserved. Zero teacher buckets contribute zero. A positive teacher tail
with an impossible student tail should fail, never receive an artificial probability floor.

Positions use weights (1,.8,.65,.5,.4), divided by the sum of active position weights across the batch.
Missing-top20 active positions receive .1 times sampled-token CE, using the same FULL student logZ.
That sampled token need not belong to S. The lookup never removes it from S's complementary bucket.
With top20 present, no extra hard CE is used, even for a sampled token outside S.
D2's normative reader currently sets present=position_mask; the fallback is supported but not normally exercised
by complete valid D2 shards. `tok_lp` outside S is preserved by the shard but not consumed as an additional singleton
loss/confidence label. No fabricated zero teacher probability is assigned outside S.

CPU MEASURED defects and fixes (`tests/test_objective_edges.py`):

- Zero-probability retained entries previously evaluated 0*(-inf) and produced NaN in both paths and the evaluator.
  Shared `bucket_divergence` uses xlogy and masks zero-mass tail log terms.
- Full retained support with p_T=0 previously evaluated 0*(-inf). Empty complementary chunks could also merge two
  -inf values with undefined autograd derivatives. `complementary_logaddexp` makes the empty merge constant.
- FAST's KL derivative assumed sum(p)+p_T=1 exactly. Inputs permit 1e-4 normalization rounding error; REF differentiates
  the supplied mass. FAST now uses q_j*(sum(p)+p_T), preserving the actual unrenormalized target in every arm.

The analytic KL gradient before the weighted mean is:

    dL/dz_j = q_j * M - p_j                       j in S
              q_j * M - p_T*exp(z_j-logZ_T)      j outside S
    M = sum_S p_i + p_T

For TV, set a_i=.5*sign(q_i-p_i), a_T=.5*sign(q_T-p_T), and a_j=a_T outside S.
Then dL/dz_j=q_j*(a_j-sum_S a_i*q_i-a_T*q_T). Missing-target CE is .1*(q-onehot(label)).
All derivatives receive the same active-position weight/denominator and incoming autograd scale. Frozen base head
receives no gradient; hidden states and both Markov matrices do, including accumulation for repeated previous-token IDs.

MEASURED verification: original focused suite 9 passed; initial new edge matrix 5 failed/11 passed before fixes.
After fixes, full trainer suite 194 passed (Torch 2.12.1+cu130, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=2).
The edge matrix covers KL/TV, REF/analytic, chunk sizes 1/2/7, zero buckets, full support, rounded mass, missing labels
outside S, partial masks, repeated embedding IDs and non-unit incoming gradients against an independent dense oracle.
148 existing upstream Torch resize warnings in native/formats.py were reported, not suppressed.
These are synthetic CPU correctness checks, not GPU performance, private-data accuracy or arm qualification.

Both loss source hashes change. Apply these correctness fixes to REF/FAST/FAST-B together in new immutable snapshots;
rebind source identities and rerun required comparisons. Do not mutate a live snapshot or relabel old receipts.
`evaluate.py::bucket_kl` shares the safe formula; coordinate its small import/return change with D2-PILOT-R3.

Memory (ESTIMATED from tensor shapes): reference chunk checkpointing recomputes vocabulary logits; FAST retains
FP32 logits across all chunks, B*5*129280*4 bytes = 315.625 MiB at B128 and 631.25 MiB at B256, excluding parameter,
optimizer, hidden and backward workspace. It bounds each GEMM, but is not constant-memory in vocabulary size.
A recompute-backward version could remove that cache, at an additional head forward; no speedup is claimed.

## 2. Official DeepSpec comparison and a bounded later arm

Public reference pinned at `005e03b81cec38b7da6399833d609ee89a2587f2`; downloaded file hashes are recorded in
[deepspec-source.json](deepspec-source.json). Only public source files were downloaded; no code was executed.

| Component | Official public reference (UPSTREAM) | D2 source facts |
|---|---|---|
| Position weights | exp(-j/gamma), active-token weighted mean | Explicit (1,.8,.65,.5,.4), same mask-weighted denominator |
| CE | Hard next-token CE on every evaluated position | .1 CE only for missing top20 |
| Distribution term | Full-vocabulary L1, without factor 1/2 | Top20+tail forward KL; optional bucket TV includes factor 1/2 |
| Confidence | BCE-with-logits against detached 1-TV(p,q) | Separate MSE stage against known p(argmax q) |
| Distributed mean | Globally reduced denominators and DDP world-size correction | Single training device per arm; batch weighted mean |

The upstream loss requires aligned full teacher logits for L1/confidence and accumulates prefix products for
its diagnostic accepted length. These are architecture/policy-specific labels.
[Official loss at the pinned revision](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/loss.py).

The example config has block size 7, five draft layers, Markov rank 256, gamma 4, CE coefficient .1, L1 coefficient
.9, confidence coefficient 1 and Markov-conditioned confidence. It is a Qwen3 example, not our MoE recipe.
[Official example](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/config/dspark/dspark_qwen3_4b.py).

`draft_ops.py` constructs a noisy block, uses noncausal draft-backbone attention, samples sequential proposals,
conditions confidence on the preceding sampled tokens and truncates at the first below-threshold confidence.
It returns the probabilities for the retained proposal support to verification.
[Official draft operations](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/dspark/draft_ops.py).

TA-OPD v1 studies reverse KL on student-generated trajectories, preserving the teacher top-k and complementary
mass rather than renormalizing retained tokens. Its correction for sampled tokens assumes the specified sampling
law; it cannot be copied onto deterministic proposals or committed D2 samples without changing the estimator.
Its results do not establish an expected acceptance gain here.
[Paper, v1, especially §4/§6/Appendix D.2](https://arxiv.org/html/2608.14728v1).

Recommendation (PROPOSED; no pilot settings changed): finish matched KL first. If teacher-forced KL improves
without useful fixed-depth survival/E[A] gain, and the full-vocab audit shows probability overlap is the issue,
run ONE additional r32 arm using the existing `divergence="tvd"` on the same top20+tail labels. Keep initialization,
data order, effective blocks, active tokens, updates, LR, QDQ, confidence stage and position weights fixed.
Use the amendment-6 family-clustered useful-gain/no-regression rule against zero-adapter and the selected KL arm;
report position 1 separately and positions 2–5 conditional on reaching them. Do not add a CE mixture, reweight
positions or adopt upstream .1/.9 simultaneously. A bucket-TV gain needs full-TV/actual proposal checks because
within-tail mismatch is hidden. For point-mass proposals the direct acceptance target remains p(d), so TV is a
surrogate, not an equality. No new objective arm is justified before those diagnostics and the KL pilot outcome.

## 3. Confidence and sequential adaptation

For the current GREEDY draft token d, q is a point mass and ratio verification accepts with p_target(d),
where p_target is the effective target distribution after the request's actual processors/temperature/masks.
For a greedy TARGET request the label instead is the indicator of target argmax agreement; do not mix these modes.
For sampled q, mean acceptance is sum min(p,q)=1-TV(p,q). This last distributional quantity is not the greedy-draft label.

Source audit: `loss.py::confidence_loss` recomputes each student argmax with teacher-forced `previous`, looks up
its ORIGINAL teacher top20 probability, and minimizes squared sigmoid error only at known, active positions.
An argmax outside support is skipped; the complementary mass does not identify p(d). `head_x` is detached and
`loop.py::_train(stage="confidence")` freezes all parameters except confidence_head.proj.weight. The confidence
function itself does not freeze Markov embeddings; the loop's parameter gate is required. Confidence-stage MSE
is an unweighted mean over known positions, not the distillation position-weighted mean. That is a deliberate
separate stage, not a claim of matching upstream BCE.

`model.py::Drafter` registers BOTH last-stage Markov embed and Markov head as trainable. Distill mode updates them
alongside the LoRA/norms and freezes the confidence projection. Confidence mode freezes those adapted dependencies
and updates only its projection. `pilot.py::run_settings` chooses stage=distill; it does not automatically execute
confidence training. Thus the pilot tests Markov adaptation but not recalibrated confidence. The Markov component
is rank-256, conditioned on one previous token, not a recurrent model of the entire proposed prefix. No expert,
router, frozen vocabulary head, target embedding or main_proj adaptation is introduced.

The serving path (`worker/tp2.py::heads`, also `worker/model.py::heads`) rolls greedy tokens sequentially, then
uses the unnormalized pre-head hidden state concatenated with those actual preceding proposal embeddings for
confidence. D2 matches the feature definition but confidence labels use committed preceding tokens. A suffix
whose proposals diverged from the committed chain is outside that label's exact conditioning. After Markov
adaptation the old confidence projection is not demonstrated calibrated; run the separately identified confidence
stage and real-worker checks before a confidence-policy claim. Packaging includes Markov/confidence and regenerates
identities; adapted Markov-cache contents must be rebuilt by that worker identity. Graph-safe live adapters remain
D2-PILOT-R3's owner task, not established by this source audit.

Image reference `vllm/v1/worker/gpu/spec_decode/depth_policy.py::choose_depth/record` interprets confidence as
CONDITIONAL immediate acceptance, computes cumulative products as prefix survivals, and chooses depth maximizing
(1+sum survival)/round_cost. Optional calibration uses only reached observations, separate sampled/greedy target
bins, after its warmup count. It does not correct arbitrary off-support labels or a changed proposal policy.
For a deterministic proposal chain, computing all confidences before choosing depth has no sampled-draft selection
randomness; still, predictive calibration is empirical. A sampled-q arm must use PP1's non-anticipating stopping
rule, not transplant this retrospective all-depth chooser.

Required aggregate calibration screen (PROPOSED): per family, segment, absolute-position bucket and draft position,
record total active/known/off-support counts, known-subset MSE/Brier, reliability bins and reached-prefix survival.
Report unknown-label coverage alongside calibration; known-only accuracy can hide the worst predictions. Refit on
DEV only; freeze the model/calibration for TEST. Fixed-depth verification removes depth-policy censoring. Use actual
worker proposals and reached flags, not unrelated committed-prefix p(d) after divergence. Do not call marginal
accepted-per-position counts conditional probabilities or multiply those survival counts again.

MEASURED CPU tests in `tests/test_confidence_semantics.py` verify the actual p(argmax) MSE, exclude unknown/masked
rows and confirm nonzero gradients reach both Markov matrices while the confidence projection is frozen in distill.
No private calibration coverage or real-worker calibration result is available in this CPU-only lane.

## 4. Prefix alignment, causality and valid survival scoring

Source audit of `data.py::D2Blocks.__getitem__`, `collate`, `model.py::hidden/attention`, and replay parsing:

| Quantity | Actual positions / semantics |
|---|---|
| Target feature context | max(0,P-127)..P; absolute row_pos checked against the requested slice |
| Known anchor | committed/sampled token P+1; its target tap is NOT supplied |
| Draft query inputs | anchor followed by four noise tokens, at P+1..P+5 |
| Supervised predictions | token P+2+i, i=0..4, only while within the recorded generated span |
| Markov teacher forcing | token P+1+i as previous token; this is the true prefix for that target row |
| Logprob row | prompt_logprobs entry at the predicted token's index, conditioned on tokens before that index |

The reader verifies token IDs/hashes and shard row positions; the replay parser verifies the echoed raw token IDs.
The first generated token is used as the initial anchor rather than a supervised draft label. Tail positions are
masked; collate left-pads context with mask=false. No prompt prediction is included by these block offsets.
Raw token IDs, not re-rendered text, define alignment. Reasoning/answer-span bookkeeping and source identity gates
remain D2-FIX1's responsibility; this audit does not establish their private dataset coverage.

Draft attention deliberately allows all five query states to see each other; later states contain only noise
and transformations of the allowed prefix. This is the model's parallel-block architecture, not teacher-token
leakage. Do not add a triangular query mask or reduce backbone block length as a "fix." Target taps themselves
must come from causal target execution; source text and synthetic input-boundary tests do not prove prefill/decode
tap equivalence on the running build. Preserve G1/equivalence checks and tap-aware replay patch provenance.
The low-level hidden() API trusts its supplied positions/mask; the audited D2 reader is what excludes future rows.
Any alternative input producer must enforce that same boundary.

MEASURED: three CPU tests in `tests/test_prefix_semantics.py` pass, plus strict typing. Changing all stored taps
after P and later teacher distributions leaves every backbone input identical. Perturbing masked context taps
leaves both hidden outputs bit-identical. Existing data tests cover exact label offsets and the partial final block.

Teacher-forced loss is valid on each stored committed prefix. It is not a rollout loss: at depth j after an earlier
proposal differs, p(. | stored prefix) cannot label p(. | proposed prefix). `pilot_eval.score` uses teacher-forced
KL and independently obtains the candidate's sequential greedy proposals; its `match_length` stops at the FIRST
mismatch. It does not multiply off-chain target probabilities. Keep this distinction in the reported metrics.

A valid deterministic-chain estimator: at an anchor X chosen before unseen future target samples, freeze the
candidate and generate d_1..d_k. Draw a fresh target continuation Y from the same processed target distribution.
Set I_j=1{Y_1..Y_j=d_1..d_j}. Then

    E[I_j | X,d] = product_{i=1..j} p(d_i | X,d_<i)
    E[A | X,d] = sum_j E[I_j | X,d].

Thus prefix-match length on target-sampled continuations estimates accepted DRAFT tokens for point-mass proposals;
the exhaustive three-token-chain CPU test checks the identity. Mismatches after the first rejection add nothing.
Independent requests/families, not individual overlapping anchors, are the uncertainty units. Existing K2-selected
anchors are a fixed paired replay population, not the candidate's eventual stationary serving distribution.
Complete-round filtering near EOS, selection of successful traces and model-dependent anchor selection can further
restrict that population; do not label its observed mean a universal unbiased deployment gain. For new unbiased
sampling, preselect anchors using only visible prefix information and retain EOS/budget outcomes explicitly.

Alternative with less sampling variance: in a coordinator-owned window, append the deterministic candidate chain
to the EXACT committed prefix, score its teacher distributions in a causal target pass (teacher-forcing ON THAT
candidate chain), and compute sum of prefix products p(d_i | X,d_<i). Record actual stop/grammar state, target
processors, temperature, tool parsing, absolute positions and caches. Never retain target KV beyond rejection as
committed state. Gather only selected token probabilities when full logits are unnecessary.
For sampled proposals, independent target-token equality estimates sum p*q, not sum min(p,q). PP1 must evaluate
its real ratio verifier and residual sampler, with fixed depth first. Non-anticipating admission is essential for
that policy; see the [DSpark v1 Appendix A counterexample](https://arxiv.org/html/2607.05147v1#A1).
No conditioned target calls or Spark requests were made here.

## 5. Bounded full-vocabulary teacher audit

Status: the CPU audit implementation is ready; a served-head equivalence check and real-data audit are UNVALIDATED.
`vocab_audit.py::audit_logits` is an opt-in, side-effect-free two-pass utility, not a new trainer or engine hook.
It accepts two immutable CPU chunk providers, captured teacher support, committed/sample IDs and proposal IDs.
It rejects invalid IDs, duplicate support, malformed/nonfinite chunks and all-masked distributions. Per-chunk
hashes refuse providers that change between passes. It supports hard -inf masks, zero teacher buckets, padding
and exact outside-support proposal lookups; a genuine support mismatch remains infinite KL, never a floor.

Pass 1 computes global FP64 logZ for each teacher/student row, selected logits and separate complementary
logsumexp. Pass 2 recomputes chunks and accumulates full forward KL, full TV, sample CE and exact p(d)/q(d).
It also returns matching coarse KL/TV and teacher/student tail masses. Workspace is O(rows*chunk + rows*k), plus
small chunk fingerprints; no rows-by-vocabulary tensor is retained. "Exact" here means the entire supplied
vocabulary with one global normalizer, subject to FP64 floating-point rounding. It says nothing about whether
the supplied teacher is the served model. This utility does not backpropagate or change REF/FAST training.

MEASURED CPU tests compare every returned statistic to independent dense calculations for chunk sizes 1/3/11,
with logits shifted by +/-900; cover full support, padded/zero-probability tokens, hard support mismatch, invalid
teacher distributions and between-pass mutation. The existing analytic student bucket backward is tested
separately in §1; this full-teacher audit intentionally introduces no second backward implementation.

Proposed bounded coordinator procedure (no Spark access performed by OBJ1):

1. Freeze a stratified DEV-only draw of at most 64 anchors / 320 predicted rows, with request/family, reasoning
   versus answer, absolute-position/context-length buckets, and draft positions recorded privately. Include
   long-position anchors already available, not just short contexts; report empty strata rather than inventing
   examples. Fixed depth avoids policy censoring. Do not draw from final TEST during objective selection.
2. In an owner-controlled window, obtain the actual served target head's logits for those exact prefix/position
   rows, before normalization but AFTER the effective target processors. Preserve a separate raw-head stream if
   processors matter. TP shards must be joined by global token ID; use one global max/sumexp, never independent
   shard softmaxes. Hard grammar masks and target temperature/penalties must match verification. At T=1/top_p=1
   with no other processors, raw prompt-logprob normalization can be checked directly. Record every identity.
3. Stream vocabulary chunks (e.g. 4096 tokens) for a small row batch (e.g. 16), processing/replaying the same
   immutable rows on two passes. Save only private row summaries and source hashes; if an engine hook already
   materializes logits, measure and report that memory separately. A host-side chunk utility alone does not
   prove the GPU head was streamed. No integration hook or listener is provided or enabled by this lane.
4. Compare full versus bucket KL/TV, teacher retained mass quantiles, student tail mismatch, full-minus-bucket
   gaps and p(d) outside top20. Aggregate by family/segment/position, preserve per-request aggregates and use
   family-clustered uncertainty. Tie diagnostics back to actual survival and whole-round cost, not just loss.
5. Keep all raw logits/taps/token IDs local and private; commit only aggregates. Stop after the fixed budget.
   At 320 rows, a hypothetical FP32 teacher dense dump alone is ESTIMATED 157.8125 MiB at vocab 129280;
   chunking avoids persisting that dump. Scratch remains in the assigned campaign directory, not /tmp.

Target-hidden reconstruction is NOT established equivalent to served EXL3-head probabilities. The D2 shards
contain INPUT-layer taps 37/38/39, not the final target hidden state. A projection/lens from those taps cannot
be asserted to recover the final hidden/head distribution. The reference trainer loads the official BF16
`head.weight`; the image source's `ParallelLMHead(quant_config=...)` can use the checkpoint's quantized head and
`compute_logits` routes through its logits processor. Matching dimensions or draft argmax is insufficient.

To validate an actual final-hidden reconstruction, compare identical rows at the identical final norm/collapse
boundary against the installed served head: exact model/head manifest and tensor/shard IDs; TP/global vocab
ordering and padding; input/output dtypes, transforms/QDQ, dequantization/scales and GEMM accumulation; gathered
raw logits and post-processor probabilities. Check full-vocab max/quantile absolute and ULP errors, global logZ,
probability error, retained/tail mass and p(d), with absolute diagnostics at near-zero logits. Include prefill and
decode rows and near ties. Any tolerance and attribution must be declared against a matched served-head control,
not chosen after an observed mismatch. Until that experiment passes, label reconstruction a proxy and use served
logits as the audit teacher. Existing draft-reference parity receipts do not validate this separate claim.

Reproduce CPU verification from the worktree:

```bash
source ~/.config/ndm/paths.sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 "$HARNESS_PYTHON" -m pytest -q offload/dspark/d2/train/tests
CUDA_VISIBLE_DEVICES='' "$HARNESS_PYTHON" -m mypy --strict --explicit-package-bases --follow-imports=silent \
  offload/dspark/d2/train/loss.py offload/dspark/d2/train/fast_loss.py \
  offload/dspark/d2/train/evaluate.py offload/dspark/d2/train/vocab_audit.py \
  offload/dspark/d2/train/tests/test_objective_edges.py \
  offload/dspark/d2/train/tests/test_confidence_semantics.py \
  offload/dspark/d2/train/tests/test_prefix_semantics.py \
  offload/dspark/d2/train/tests/test_vocab_audit.py
```

Final OBJ1 verification (MEASURED, synthetic CPU): 208 trainer tests passed in 15.64 s; 163 pre-existing
native/formats.py Torch resize warnings were emitted. Strict mypy passed all eight changed Python files.
A vocabulary-shape smoke with five rows, vocab 129280 and chunk 4096 used at most 20480 elements per provider
callback and finished in 0.0956 s on this host. Its artificial sine/cosine logits and declared support are only a
shape/stability check, not a teacher top20 measurement or inference/training performance estimate.

The released 1017 confidence head's original training labels have not been independently established here;
the public DeepSpec example is a comparison, not proof of that private runtime's pretrained calibration.
