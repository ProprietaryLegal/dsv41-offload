# PP1: probabilistic-proposal correctness oracle

**OFF; CPU reference only.** No serving imports, launch flags, service changes,
checkpoints or GPU code. This directory does not enable sampled remote proposals.
See [AUDIT.md](AUDIT.md) for the current point-mass q path.
The [transport design](TRANSPORT.md) defines q precision, sparse/TP2 ownership
and non-anticipating depth. The [GB10 A/B plan](GB10-AB-PLAN.md) identifies
the remaining implementation and coordinator-only hardware gates.
Synthetic measured counts are in [cpu_results.json](cpu_results.json).

Run from the repository root on Controller:

```bash
source ~/.config/ndm/paths.sh
"$HARNESS_PYTHON" -m pytest -q -s offload/dspark/probabilistic/tests
"$HARNESS_PYTHON" -m mypy --strict --follow-imports=skip offload/dspark/probabilistic/reference.py
"$HARNESS_PYTHON" -m ruff check offload/dspark/probabilistic
```

No dependency was installed. The reference uses only Python's standard library;
pytest, mypy and ruff were already available in HARNESS_PYTHON.

`Categorical.dense(weights)` normalizes a full q. `q.sparse_topk(k)` selects
support deterministically (lowest global token ID on ties), then normalizes
on that support; all other IDs have **exactly zero** probability. Sample only
after this operation. The full target vector is retained for residual sampling.
`fp32_weights` rounds weights before constructing q, permitting sender/receiver
equality tests. It refuses overflow, non-finite values and zero total mass.
Weights that underflow individually to zero become zero in the defined q.

For any proposed x with q(x)>0, `acceptance` returns min(1,p(x)/q(x)).
`residual` normalizes max(p(v)−q(v),0) over every target vocabulary ID.
The mass accepted at v is min(p(v),q(v)). The total rejection mass equals
sum_v (p(v)−q(v))+, so adding the correction mass recovers p(v) exactly.
The no-rejection case p=q never constructs a zero-mass residual.
This is the standard tokenwise algorithm in [Leviathan et al., 2023,
algorithm 1 and appendix A.1](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf).

Fractions and integer tickets make the oracle exact for its supplied finite
distributions. Float inputs mean their exact binary64 values; FP32 inputs mean
the exact rational normalization of the rounded FP32 weights. This does not
claim that a GPU's finite-precision softmax, RNG or reduction order is identical.
Use the oracle to qualify those implementations separately. It is intentionally
unsuitable for full-vocabulary production throughput or timing forecasts.

`propose` commits depth in 1..5 before drawing and conditions each q on the
actual prior proposals. Only a terminal token can shorten that block.
`Verifier` evaluates target and processors on the matching proposed prefix,
stops at the first rejection and discards its suffix. It commits only accepted
drafts and one correction (or a target bonus if all drafts survive). Accepted
EOS stops without a bonus; correction/bonus EOS ends the request too.

`RandomStreams` binds proposal, acceptance and correction streams to request,
generation and round. Streams use independent deterministic seeds derived from
explicit domains; global random state is untouched. The verifier refuses wrong
owners, stale prefixes, changed context/stop policies, reused rounds and malformed
proposals before drawing. A failure during target/processor evaluation leaves
the prefix uncommitted and poisons the session until explicitly reconstructed.
This CPU RNG scheme is a test oracle, not a proposed production RNG ABI.

Target processors are pure `(prefix, distribution) -> distribution` callbacks.
Tests exercise a prefix-dependent grammar and forbidden draft tokens. Effective
p must include all actual target constraints; q need not share p's support.
The caller supplies a context identity covering target/model/processor policy.
An engine must attest that identity and own snapshots of mutable grammar,
penalty, stop, Markov and KV state. Unsupported processors must fail or select
target-only **before** any proposal draw; this reference cannot validate arbitrary
external processor objects or import the real engine on CPU.

The tests combine exact enumeration (225 p/q pairs on a three-token grid),
normalization/transport checks, frequency tests, conditional joint frequencies
at fixed depths 2 and 5, and state/failure regressions. Frequency tolerances are
predeclared six binomial standard deviations plus one count; exact algebra,
not a finite histogram, establishes the distribution identity. All data are
synthetic numbers. No claim about DeepSeek quality or speed follows from them.
