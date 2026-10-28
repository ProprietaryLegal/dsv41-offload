# SP2: acceptance improvements that preserve the target

Research checked 2026-10-23; stack compatibility checked against the read-only image source and this branch's
remote_dspark package. No paper's speedup is a measurement of this deployment. Performance quantities are labelled
MEASURED (local file), ESTIMATED (our inference) or UPSTREAM CLAIM. Algorithm/configuration literals identify the
method, not a performance observation. Engine-change sizes are ESTIMATED engineering effort, not delivery dates.

Repo throughput convention: `R = (1 + E[A]) / C`, A = accepted drafts (bonus excluded), C = complete round time.
DSpark paper v1's accepted length/rate includes the bonus (§4.1 footnote 4); its tau cannot be compared directly
with our A. See [the full unit audit](../../evidence/THROUGHPUT.md).

## What “exact” permits

At target T=1, current remote drafts are deterministic ids. The point-mass rejection test accepts d with p(d).
A general stochastic proposal q instead has expected single-token acceptance `sum_x min(p(x),q(x)) = 1-TV(p,q)`;
the exact rejection correction uses normalized positive residual `(p-q)+`. This follows the original
[Leviathan et al. algorithm](https://proceedings.mlr.press/v202/leviathan23a.html). In particular q=p can accept
with certainty, whereas a deterministic argmax draft accepts only with max p. A GREEDY-DRAFTER ceiling is not a
ceiling for all exact stochastic/tree algorithms. For current fixed deterministic chains, block-level acceptance
cannot beat the prefix probability bound. This last statement is an inference from point-mass support, not a
claim that block verification never helps stochastic proposals.

Greedy target requests preserve target decisions within the diagnosed numerical floor; sampled requests preserve
the target distribution and can consume different random draws. Neither confidence nor a better drafter authorizes
accepting a token without target verification. The current
[vLLM losslessness documentation](https://docs.vllm.ai/en/latest/features/speculative_decoding/#lossless-guarantees-of-speculative-decoding)
distinguishes rejection-sampler guarantees from finite-precision and batch variability. Its newer relaxed verifiers
are outside this exact-output task. The installed image source, rather than current generic docs, controls compatibility.

## Recommended order and concrete changes

| option | current stack fit / required work | acceptance or speed expectation |
|---|---|---|
| Paired depth-cost recalibration | Config only after measured fixed-depth traces; no wire/target change. Existing confidence head and scheduler already adapt. | Increases useful rate by avoiding unproductive rows; does not improve p(d). ESTIMATED incremental planning budget zero to three percent, unqualified; ESTIMATES.md. |
| Refit/retrain confidence | First test existing per-position sampled/greedy calibration alone. Fitted margin/top-prob features need worker reductions, reply fields, Spark policy and new identities/version. ESTIMATED small-to-medium, one to three engineering days plus qualification. | Better depth ranking, not intrinsically better draft tokens. Gain unmeasured on this workload. |
| D2 self-distillation | Existing staged trainer/data/package route; keep target unchanged, new drafter identities and confidence calibration. No new verifier algorithm. | Gain unknown; four-request oracle diagnostics do not forecast training outcomes. Require amendment-6 DEV evaluation and a matched GB10 A/B. |
| Exact stochastic q + optional block verification | Medium/large: worker distributions/sampling, new wire contract, Spark q buffers and sampler integration, numerical and long-context tests. ESTIMATED three to seven engineering days minimum. | Could exceed deterministic ceiling, but relevant upstream prose gain was absent; current remote adapter explicitly refuses it. |
| Multi-candidate/tree verification | Large: branch-aware target attention/KV/Engram/state, scheduler, masks, graph buckets, verifier and protocol. ESTIMATED one to three engineering weeks minimum. | Can cover more target probability mass; no measured DSpark TP2 gain, especially at B4 row pressure. |
| Draft context beyond trained window | Architecture/worker/KV/protocol and training/validation change. ESTIMATED multi-day investigation, potentially weeks for retraining. | No evidence it raises this drafter's acceptance. Maintain the exact trained window first. |
| Lower target T or target top-p | Changes the requested distribution. Not a candidate. | Artificially easier acceptance is not an improvement at the required T=1. |

## Trees and multiple candidates: useful, but not a flag

[Sequoia](https://arxiv.org/html/2402.12374v3) optimizes token-tree shape under a hardware cost budget and uses
sampling without replacement plus residual correction to preserve the target distribution. Its guarantees and
speedups concern its own Transformers implementations and hardware; they do not demonstrate support in this
DSpark verifier. This stack returns one `[request, position]` id chain and confidences. It has no parent indices,
branch-conditioned target KV or tree mask in the remote contract. Repeating identical greedy proposals creates no
new coverage. Distinct branches need distinct Markov histories and verification states.

A bounded initial design could expose head top alternatives and verify a very small tree: the first-token target
coverage is the sum of probabilities of distinct candidates. Exact target-sample-and-match traversal can keep target
sampling intact; an optimized Sequoia-style traversal needs its full proposal/residual semantics. This is our design
inference, not an implemented feature. Never flatten siblings into a causal sequence: it conditions later tokens on
rejected siblings. Target sparse/SWA attention, compressed states, Engram n-gram hashing, commits/cancel/reset and TP
broadcast must all follow the selected ancestor path. Capture graph/row costs at B1 then B3/B4; cap the tree budget
before it crosses the existing target kernel thresholds. Retain only if accepted-token gain beats verification cost.

## Dynamic/confidence-based length: mostly already here

[DISCO](https://arxiv.org/abs/2405.04304) learns to choose lookahead per iteration; UPSTREAM CLAIM: average ten-percent
speedup over its best static baseline on its tested datasets. It supports dynamic length as a method, not a ten-percent
prediction for this already-adaptive installation. Current confidence chooses verification depth AFTER the parallel
five-query draft pass. It can save routed target work and head tail via safe `n_out`; it cannot retroactively save
that backbone pass. Actual early-stop generation is more useful for autoregressive drafters. Do not shorten the
noncausal DSpark query block to mimic that behavior.

The particularly close donor is
[sudoingX patch 0036 at f8f20b3](https://github.com/sudoingX/dgx-spark-deepseek-v41/blob/f8f20b370e41a3a220bc7d1ab96a61e4daa84964/patches/0036-dsv41-dspark-conf-feat-head.patch):
a per-position logistic model of head logit, top-two margin and global top-one probability. UPSTREAM CLAIM: fitted
on 11,357 fixed-depth rounds, replay gain around 1.2% overall / 1.6% prose; a larger head refit only .2–.3%.
Its parser adds `conf+feat`, which THIS image does not parse. Remote v1 replies do not carry those full features.
Porting means fitting our own fixed-depth, held-out task families at T=1, adding global reduction features without
changing argmax ties, a versioned optional reply field, and explicit Spark policy support. Do not copy donor weights
or set `conf+feat` in a ready config. Existing `CONF_CALIB=1` is lower effort, but test it separately from cost changes;
per-position EMA at reached positions cannot recover wholly unverified tail observations.

## Drafter context window

The [official DSpark reference](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/inference/model.py)
constructs attention indices from the target-derived sliding window plus the whole draft block. Local reference
`/opt/dsv41/data/ref-ds41/inference/model.py` and CONTRACT identify the trained 128-position window and the
three target feature taps. Our remote state preserves the same window incrementally, with explicit resync. A long
150K target prompt does not require a 150K draft KV window: the latest target taps already encode target-context
information. This does not prove unlimited sufficiency, but it explains why simply enlarging the ring is not a
validated acceptance fix. Full-window row/position correctness, after rejection and prefix reuse, is the first check.

Larger draft context can change its predictions while exact target verification still protects the target
distribution. It would nevertheless depart from trained architecture/parity and require a distinct arm/identity,
new shape/cache budgets and held-out long-context evaluation. Preserve five query tokens while trimming output
head work. No context-window increase or draft truncation is included in SP2's ready candidates.

## Temperature and probabilistic proposals

For fixed logits, reducing TARGET temperature increases the maximum softmax probability, often easing deterministic
acceptance; it also changes the target output law. Keep target T=1/top_p=1. For deterministic drafts, a positive
DRAFT softmax temperature does not alter argmax. It can change confidence calibration, or it can change sampled q if
we implement probabilistic proposals. Any draft q is allowed with exact residual correction; match its actual
normalization, temperature, vocabulary support and filters on both worker and verifier. Reweighting an acceptance
threshold or relaxing verification is not the same operation.

[Block Verification](https://arxiv.org/html/2403.10444v3) jointly verifies stochastic blocks with a lossless
correction; UPSTREAM CLAIM: modest five-to-eight-percent wall-clock improvement on its own evaluations. With current
point-mass proposals, prefix probability is already saturated by ordinary rejection, so merely switching to block
verification adds no accepted-prefix probability. Image source `rejection_sampler_utils.py` supports draft logits,
but remote `_validate` refuses non-greedy sampling and top-k distributions. No probability tensor is currently sent.

Closest hardware-specific check:
[HeNryous's probabilistic/block results](https://github.com/HeNryous/deepseek-v41-1017-dgx-spark#readme), pinned locally
at afe4d5e41ec6fa4db17c5cc88758af439a106913, README “Not adopted.” UPSTREAM CLAIM: probabilistic drafts give .8% more
tokens/round but cost about one millisecond and reduce T=1 throughput 2.8%; adding block verification gives 3.95%
more tokens/round, prose unchanged, net throughput about flat. These are separate from our local measurements.
The donor also records an old long-context stochastic corruption incident; reuse the established long-context
correctness probes before qualification, rather than assuming current engine code shares that defect.

Our ESTIMATED payload arithmetic for full FP32 q/logits: five positions × vocabulary 129280 × four bytes =
2,585,600 bytes per request, or 10,342,400 bytes at B4. At theoretical 40 Gb/s this alone serializes in .517 / 2.068 ms,
excluding head materialization, synchronization, transport and TP distribution. Sparse normalized top-k q could
reduce traffic while retaining EXACT target sampling if residual correction uses that actual truncated q. It must
not pretend omitted proposal mass is present. Confidence scores cannot replace proposal probabilities. A new sparse
sampler/protocol is preferable to shipping dense tensors only if measured useful throughput supports the complexity.

## D2: measured headroom, realistic expectations, no qualified improvement yet

MEASURED local source: P4 `code/dsv41-exact-acceptance-ceiling/code/dspark/d1/results/exact_ceiling.json` (aggregate-only
copy under `/opt/dsv41/data/publish/p4/`). On 529 rounds from four requests, oracle target-greedy expectation
is 3.151 accepted drafts/round; captured K2 expectation lower bound 1.831; reported gap 1.320. The lower bound treats
unreturned draft-token probabilities as zero, so that gap is an upper estimate of the gap to captured expectation.
The oracle uses exact conditional target probabilities plus committed-prefix survival observations; it is a finite
trajectory estimator, not a population ceiling or a multiply-the-mean-probabilities calculation. No exact-replay
bootstrap interval was supplied. Different request/context mixes will differ.

MEASURED target-versus-native-reference top20-plus-tail KL rises from .6505 nats at the first position to 3.3583 at
the fifth (same aggregate file); the drafter cannot observe future target states. That motivates later-position
training and also limits recoverability. “Legal-v2a” here is a legal-weighted requantization, not a substantive
fine-tune; adaptation mainly targets workload/quantization mismatch, not a wholly different base model.

No supported training-payoff forecast follows from these four requests. D1's former low/central/high
projection is withdrawn as an expectation for this deployment. The oracle gap motivates a bounded experiment;
no benefit remains possible. Use [EV1's population reconciliation](../../evidence/ACCEPTANCE-RECONCILIATION.md)
and [family-clustered diagnostic design](../../evidence/DIAGNOSTIC-DESIGN.md) before generalizing. Accepted-depth
changes can also increase complete round cost, so their hypothetical rate effects cannot be added to depth-policy
estimates.

[DistillSpec](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8766fbc68e1ed1cdef712ce273e0a363-Abstract-Conference.html)
uses draft-on-policy data and task/decoding-specific divergence; UPSTREAM CLAIM: ten-to-forty-five-percent gains over
standard speculation in its experiments. D2 instead captures TARGET-generated trajectories and teacher-forces the
Markov prefix; it is related, not an identical replication. Compare forward-KL and the implemented TVD surrogate
on held-out data rather than assuming lower KL always maximizes deterministic acceptance.
[Online Speculative Decoding](https://proceedings.mlr.press/v235/liu24y.html) supports adapting to query distribution,
but its UPSTREAM CLAIM of .1–.65 acceptance-rate increase is not automatically our accepted-drafts/round metric.
[FastMTP](https://arxiv.org/html/2509.18362v1) self-distills a recursive shared head; UPSTREAM CLAIM: first-position
acceptance rises roughly 70% to 81%, with larger later-position changes. Its architecture differs from the trained
parallel DSpark block, so transfer the training motivation, not its effect size.

The binding D2 plan and amendments govern training/data access. As read here, D2-PILOT.report.md says real
G1-approved shards are absent and its synthetic rehearsal FAIL remains preserved; no learned acceptance gain is
qualified. D2 needs G1 data validity, real matched reference/fast pilot outcome, held-out G2 (legal gain at least
+.15 accepted/round with request-clustered interval above zero; general no worse than −.03 and per-position KL
improvement), native/EXL3 packaging identity/fidelity checks retaining at least 75% of gain, then G3 matched GB10 A/B
with useful rate gain at least five percent and no B3/B4 regression. Follow current amendments rather than the
older estimated training wall times. Recalibrate the confidence/depth policy for the packaged candidate and its own
identity AFTER acceptance changes; S20 M3 specifically forbids treating inherited K2 prices as native-arm measurements.
