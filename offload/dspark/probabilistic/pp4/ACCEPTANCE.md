# Emitted-q acceptance requalification

PASS: emitted dyadic24 q preserves PP2's acceptance benefit. MEASURED expected
retained tokens per nonterminal round rise from **3.091813 to 3.305515**, a
6.912% increase before added time. Paired source-cluster 95% interval for the
gain is **0.175370–0.254435 tokens/round**. This is acceptance qualification,
not an observed throughput gain or permission for a production experiment.

The complete [replay receipt](heldout-replay.json) and
[paired summary](heldout-summary.json) bind the 1,710 frozen anchors,
24 requests/source clusters and both heldout task families to the exact
production weights and consumed capture payloads. Actual fused support and q
match PP3's reference producer on **8,550/8,550 rows**. Own-greedy-path full-logit
argmax parity passes on **8,550/8,550 positions**. Current greedy source/context
prefix aggregates equal frozen PP2; that aggregate check alone cannot prove
per-anchor identity across the two runtime versions.

| View | Greedy retained/round | Sampled retained/round | Delta / paired 95% source interval |
|---|---:|---:|---|
| All 1,710 anchors / 24 sources |3.091813|3.305515|0.213702 /0.175370–0.254435|
| Chronology 802 / 12 |3.133416|3.369527|0.236111 /0.163163–0.301578|
| Motion outline 908 / 12 |3.055066|3.248976|0.193909 /0.162188–0.233094|
|16–32K 545 / 12 |3.223853|3.376376|0.152523 /0.103830–0.200527|

Every view with an available PP2 interval remains inside that interval.
Pooled change from PP2's float64 nominal q estimate is +0.000000613 retained
tokens/round. The >=32K view still has only 66 anchors from one source and no
interval. Maximum anchor context remains 47,084; no 150K result is claimed.

MEASURED on x86host's bc62df2a at 225 W, under its exclusive flock, eight reserved
threads/nice 19: full scoring 50.260 s after loading, peak RSS 8.69 GiB and Torch GPU
allocation 9.35 GiB. This duration includes diagnostics and is not serving
latency. Required miniconda CPython 3.13.12, Torch 2.12.1/cu130, Triton 3.7.1 used
the separately rebuilt stock/facc extensions recorded in the receipt. The
solo was verified 225 W / 1 MiB with a free lock before handing it to the runtime
sub-item. No serving request or physical-pair execution occurred.

The primary policy remains top32/tau 0.7, fixed depth 5, as frozen in PP2
`e1befc9aaa081a4f6cccbb11423a747fcd392503`. `acceptance_frozen.py` checks the
SHA256 of PP2's selection, heldout aggregates, summary, numeric scorer and
source-bootstrap implementation before using them. It can extract those
public-code/aggregate files with `git show` only into PP4 scratch. There is no
refit or new anchor selection.

`acceptance_replay.py` reloads the same production K2 weights and captured
heldout shards, verifying checkpoint tensor payloads, source/weight identity,
manifest, sidecars, token histories and every consumed shard. Both logical
TP2 shards run on x86host's assigned solo GPU. All 1,710 scheduled anchors from
the 24 chronology/motion_outline requests are required. Their grouping,
source connectivity, context bins and counts must match the frozen PP2
aggregate receipt before a summary is admitted. Greedy prefix aggregate drift
is reported explicitly, with separate paired comparisons to frozen and current
greedy baselines. Aggregate equality cannot prove per-anchor numeric identity.
The current CPython 3.13/Torch 2.12 environment differs from PP2's
CPython 3.12/Torch 2.10 environment. Terminal suffix
censoring, 128-token schedule and available 128-row windows are unchanged.

For each anchor, the new `FusedContext` GPU producer emits dyadic24 q on all
five rows from production base logits and forced target-history previous IDs.
It computes the actual fused Markov bias and candidate selection. PP3's
`Top32` producer runs separately on materialized BF16 conditioned rows as a
diagnostic, counting any differing support/q rows. The scorer consumes
those exact FP32 probabilities, checks integer ticket mass and positive draw
membership, and never renormalizes or reapplies temperature. The replay hashes
the transient sparse IDs/q and saves no logits, captured tensor or generated
text. Source hashes distinguish both producers. This qualifies the fused
selector on these heldout rows; it does not replace whole-worker graph,
physical-pair or lifecycle qualification.

The frozen target normalization/storage-error model is reused verbatim.
Prefix survival is `cumprod(min(1,q(c)/p(c)))` on the captured target-sampled
continuation. Expected retained tokens per nonterminal round are 1 plus the
sum of those survival values; this is a one-round counterfactual on fixed
target-history anchors, conditional on captured teacher/generation agreement.
It does not measure changed future round starts or throughput.

The exact PP2 bootstrap uses 2,000 paired connected-source-cluster resamples,
seed 4102, separately reporting both heldout task families and context views.
Each new point and paired 95% delta interval is compared to PP2's original
point/interval. The >=32K view retains its unavailable interval because it has
one source. Neither 150K nor broad task-family generalization is supported.

Reproduction uses `/home/operator/miniconda3/bin/python3` throughout. Stage the
code and frozen aggregate files only under PP4's approved scratch paths. Set
`PP4_SOURCE_ROOT` to staged PP4 code and `PP4_EXL3_PATH` to separately qualified
CPython 3.13 stock/facc modules. Then run `acceptance_run_locked.sh --out ...`
inside PP4's named detached x86host tmux. The wrapper waits for the sole assigned
UUID's exclusive flock, enforces 225 W, taskset 12–15,28–31, nice 19 and eight
threads; the replay guards resident memory after every anchor. No fallback
interpreter, serving endpoint or service operation is present.

CPU validation: 20 tests pass, covering sparse/dense frozen scoring equality,
the reverse-prefix dependence, invalid q refusal, exact frozen-bootstrap parity,
source/population drift, missing full replay and frozen-byte tampering. Strict
mypy passes for all four new Python files, ruff and shell syntax pass. These
checks establish the estimator/admission logic and reconcile committed
source/receipt identities. The separate GPU receipt establishes the heldout
result above.
