# PP1 probability-path audit — 2026-10-23

Status: CPU source inspection only; no Spark/x86host access, traffic or service changes.
The reference experiment is OFF: nothing in serving imports this directory.

## What q the current remote verifier uses

**q is the point mass at the proposed token d: q(d)=1, q(v≠d)=0.**
For target T=1 with ordinary stochastic verification, acceptance is p(d),
and a rejection samples the full processed target distribution with d removed
and renormalized. A confidence score is not q(d).

Evidence at lane base `cc9515fdd4e26197ef13b797c06bcf9cc39a085c`:

* `spark/remote_dspark/speculator.py:52–60` refuses non-greedy proposal mode;
  its capacity speculator sets `draft_logits=None` (line 272). HELLO declares
  `SAMPLE_GREEDY`. `wire.py:379–438` carries IDs, confidence and optional
  margins, with no categorical proposal distribution.
* `worker/model.py:402–435` applies sequential Markov bias to each head row,
  then argmax. Confidence is computed separately. The native/TP2 worker
  implementations retain the same greedy wire contract.
* The installed image's **GPU runner** `vllm/v1/worker/gpu/model_runner.py:1711–1718`
  passes `self.speculator.draft_logits` to its rejection sampler.
  `gpu/spec_decode/rejection_sampler.py:244–287` applies target sampling
  parameters before verification. Grammar masks are applied by the runner
  before this call (around line 1680).
* `gpu/spec_decode/rejection_sampler_utils.py:630–665` uses a log probability
  ratio in standard verification. Without draft logits, the draft logprob is
  zero. Lines 833–887 implement direct target sampling for bonus/placeholders,
  the positive residual for general q, or target logits with d masked for
  point-mass q. Target greedy requests instead compare target argmax.

The separate older `v1/worker/gpu_model_runner.py` caches `take_last_draft_probs()`
by request ID and calls `v1/sample/rejection_sampler.py` with dense probabilities.
That file's missing-request path warns and returns None. It is **not** the
remote overlay's integration path. Reusing that fallback for sampled proposals
would silently substitute point-mass q and be incorrect; a future sampled path
must refuse missing q or discard the entire proposal and run target-only.

## The installed optional local sampled path

`gpu/spec_decode/speculator.py:178–191` allocates full-vocabulary cached
pre-temperature logits only for `draft_sample_method=probabilistic` (default
greedy in `config/speculative.py:601`). DSpark `_sample_logits` applies Markov
bias, scatters a reduced draft vocabulary into target IDs if applicable, then
uses `gumbel_sample`. That routine caches the pre-temperature logits and uses
separate draft noise. Verification reloads those logits, upcasts to FP32,
divides by the sampling temperature and computes global logsumexp. Thus the
local sampled q is softmax of those exact cached, temperature-scaled logits,
not a confidence estimate or an independently truncated reconstruction.
The default head cache dtype comes from model `head_dtype`.

Standard ratio verification and block verification are distinct algorithms.
The image also supports block verification (prefix joint ratios and a scaled
residual); PP1 implements **standard tokenwise ratio verification only**.
No claim of block-verifier parity is made.

## HeNryous 0055: an existing negative prose experiment

Pinned donor: [HeNryous afe4d5e41ec6fa4db17c5cc88758af439a106913,
patch 0055](https://github.com/HeNryous/deepseek-v41-1017-dgx-spark/blob/afe4d5e41ec6fa4db17c5cc88758af439a106913/patches/0055-dsv41-hybrid-draft.patch).
Historical `b1726644b89fbb57b6468d33880cddac55a3c527` does not contain 0055;
it must not be used as that patch's citation.

UPSTREAM CLAIM (patch preamble, not our measurements): sampled proposals
helped some code/JSON cases, but dense probability handling added about 1 ms
per round and prose did not improve. The hybrid mode samples requests already
in the full-window regime; other requests use argmax and cache one-hot logits
(0 at d, −inf elsewhere). It requires probabilistic drafting and local argmax
disabled; their test used block verification. The reported aggregate gain was
0.75%, with single-stream prose −1.6% and four-stream JSON −5.1%; it was not
adopted. This supports a bounded new experiment, not an enablement decision.

## Provenance and limits

MEASURED (local file inspection): image `_version.py` identifies
`0.31.1.dev0+g6e517b15.dsv41gb10`. Read-only image snapshot hashes:

| File under `vllm/v1/worker/gpu/` | SHA256 |
|---|---|
| `spec_decode/dspark/speculator.py` | f319e29d7b5492f8504d141474ef0bb58d39971249a24749fa8d464a808e8708 |
| `spec_decode/rejection_sampler.py` | 25759f98e8aace9a556bc4252c206f246535f0536ebc919f97e097f92a72066a |
| `spec_decode/rejection_sampler_utils.py` | ef1d40bf31140eb644a8f3b7a003373209d6a8e297f059e70ea66cf4972d8d90 |

0055 SHA256: `9affd2a869ef344be901ed5fa9b444544ebff48f714b78c930101392467a05f5`.
No live revision attestation was attempted: this lane forbids Spark access.
Coordinator records identify the production K2 TP2 pair and process proxy;
that operational record does not make native source-faithful serving qualified.
Local refs at inspection: main `21bed313ac803d38327c4de1ff2e5235f329c716`,
offload-x86host `26ef0a3ab3df9760ad48e73d16523a2ee1316e5e` (moving refs).

Algorithm references: [Leviathan et al., ICML 2023, algorithm 1 and appendix A.1](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf),
[vLLM rejection_sampler.py pinned e4db46c79bca354fef07197465faea14ca563c09](https://github.com/vllm-project/vllm/blob/e4db46c79bca354fef07197465faea14ca563c09/vllm/v1/sample/rejection_sampler.py).
The latter HEAD was resolved via git ls-remote on 2026-10-23; the local image
above is authoritative for this deployment, not upstream main.
