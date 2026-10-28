# PP4 Round 12: what the greedy margins establish

Source/CPU evidence only. The historical admission remains NOT QUALIFIED.
The proposed registration is `pp4-r12-logit-grid-tie`, for NEW PP4 experiments
only. NBW, TV1 and W8 retain their existing rules.

## Primary observations and their limits

Read-only campaign `PP4-429-admission-W-20261027T133630Z` stopped at
`cmd/0156-admission-B-greedy.log`, after three divergent outputs:

| Reference | First differing token index | Margin nats | Better-token gap to top |
|---|---:|---:|---:|
|0|97|0.0|0.0|
|1|78|0.5|0.0|
|2|63|1.0|0.0|

`admission-B-greedy.json` SHA256:
`ace208cbb3135015577c163417f4504da9ab95ebefd8910e97d2dd7c78ef93eb`.
These are NBW diagnostics rounded to four decimal places. Its gap uses the
BETTER of the two tokens, so zero does not prove that the experiment token
was the top. No top-20 values, token IDs, absolute logits or runtime dtype
were saved. Prompts 3–5 were never checked. This artifact cannot retroactively
pass the new rule. Its observed margins can supply regression cases only
when supplemented by explicitly synthetic top-20 evidence.

The binding Round 12 brief also reports DEP1's prompt-2 margin of 1.0. The
named DEP1.report.md available during this review contains the Round 3 budget
and earlier reports, without that flip's primary values. Treat the flip as
coordinator-provided context, not an independently recovered top-20 receipt.

Read-only X86host `PP4-server-R11-20261027T132720Z/summary.json` records candidate
42928552727ee85561600d886a704d3342ad969a, DIAGNOSTIC COMPLETE, no stop and
pair_restored=true. Its admission field explicitly says NOT QUALIFIED. The
coordinator's baseline-reproduction/graph-reference result in the brief is
separate from this summary's narrower fields. Round 12 changes no worker byte.

## Dtype and API source chain

Image-source paths below are relative to
`/opt/dsv41/data/src-image/`; they were read, never executed as servers.

1. `vllm/models/deepseek_v41/nvidia/model.py:1601–1609,1638–1649`
   constructs ParallelLMHead and LogitsProcessor, and passes hidden states
   straight through that processor. The production-mounted K1 model is
   `offload/kvpack/overlay/vllm/models/deepseek_v41/nvidia/model.py`, SHA256
   bc3aa8b06ff40c65e93f31e11c3cad38fcb4fb6425946e61829f041bac283d3f,
   exactly matching BOTH saved admission identities. Its two compute_logits
   method source spans are identical to the image's. The image file's complete
   hash differs because K1 has other changes; do not confuse the two files.
2. `vllm/model_executor/layers/logits_processor.py:96–101,153–200,211–223`
   uses the head's quant_method.apply unless head_dtype explicitly differs
   from hidden dtype. Generation defaults to model dtype at
   `vllm/config/model.py:2487–2506`. The V4.1 constructor supplies neither
   scaling nor soft-cap overrides, so defaults scale=1/soft_cap=None apply.
   TP gathering and vocabulary slicing do not restore lost mantissa bits.
3. `vllm_exl3_moe/config.py:234–241` selects Exl3HeadMethod for an EXL3 head.
   `head.py:125–144` selects the BF16 mrow output for small BF16 inputs, or
   performs EXL3 FP32 accumulation and **casts back to x.dtype**. The saved
   admission env has VLLM_EXL3_HEAD_MROW=1 and EXL3_HEAD_MROW_MAX_ROWS=8 on
   both ranks. Thus changed verify/prefill row counts can change head kernels;
   the ordinary BF16 activation path returns BF16 logits on either branch.
   Unquantized heads similarly follow activation dtype. An explicit FP32 head
   override is a different configuration, and must not be presumed absent
   solely from a config hash. No live tensor dtype was measured in this lane.
4. `vllm/v1/sample/sampler.py:80–103,307–309,312–360` computes raw logprobs
   with logits.log_softmax(dtype=torch.float32), promotes sampling logits to
   FP32 before processors, and gathers top-k plus the sampled token. ModelConfig
   defaults to raw_logprobs (`config/model.py:285`). For speculative rows,
   `rejection_sampler.py:145–180,203–212,248–265` promotes model logits to
   FP32, keeps raw values separately from processors, reassembles accepted and
   bonus logits, then uses the same compute_logprobs/gather_logprobs path.
   PP4's probability verifier overlay does not increase head precision.
5. `vllm/entrypoints/openai/completion/serving.py:650–735` serializes those
   values under token_id placeholders, clamping only below -9999. The grid
   gate refuses clamped/nonfinite/malformed data. Logprobs are FP32 values,
   **not BF16 log-softmax outputs**. Subtraction cancels the common log-partition
   up to FP32 rounding, exposing the already rounded head-logit differences.

Relevant read-only image SHA256:

```text
model.py          66005a8edde825612bbb9e041a5a218132cf97123bba9b401890122de6314a89
logits_processor  35cf8235cb529ba2d37358fe9b9e5a29f40b5c3cbcc9e6750c8480c51bd7da83
sampler.py        a67286eb78aac98185b6922ac33d8c754ca5ff9788d893965ba52f2b5a0e18d1
rejection_sampler 9acaefa7ff3ff02649adf3a2edc4a7281d0089c23566fe5ff140c5f820d88ac4
exl3 head.py      295faadd33db3431e3ddaa54c7b2a410cd91b62e9b1957efcd68e5f8add1b744
```

## Grid interpretation and the next window

BF16 has eight significant binary digits. Within a binade, spacing is
2^(floor(log2(abs(logit)))-7): 0.5 at magnitudes [64,128), 1 at [128,256),
and 2 at [256,512). Spacing changes at binade boundaries, including on the
lower side of an exact power of two. FP32 promotion cannot undo that rounding.
CPU synthetic BF16 logits followed by the actual torch FP32 log_softmax show
0.5 gaps represented as 0.5000000596046448 and 1.0 gaps as 1.0; this explains
why comparisons need FP32 rounding bounds rather than decimal equality.

The observed 0/.5/1 margins are CONSISTENT with this mechanism. Their
logprob magnitudes do not reveal the absolute head-logit exponent: log-softmax
is invariant to a common shift. Nor does a top-20 lattice prove its fundamental
spacing; the finite sample could omit intermediate representable values.
The registered checker therefore uses a conservative observable step: the
smallest positive top-20 gap must itself be a binary power of two <=1, and
EVERY pairwise gap must be an integer multiple within FP32 subtraction error.
It never computes a fine GCD to rescue an off-grid set, chooses a larger
tolerance from the divergent margin, or infers a grid from an all-equal set.
Sparse or mixed-binade evidence may conservatively STOP.

Changed batch/reduction/EXL3 paths, changed attention/cache states, wrong token
alignment, stale KV, and an actual verifier/failure-mask defect can ALL change
the winning token. Prefix-cache reuse by the old diagnostic could also alter
its shape. Rounded margins alone distinguish none of these.

The next coordinator window obtains a fresh salt for EVERY greedy generation
and a DIFFERENT salt for its teacher-prefix probe; it saves all top-20 IDs and
unrounded values, both compared IDs, exact experiment-to-top gap, derived step,
all-pairs residual/tolerance and token counts. Only an experiment token actually
at the top, both tokens present, <=one observable step, equal token lengths and
at most three divergent prompts can pass. Every failure STOPs before cells;
W8 still requires six distinct exact IDENTICAL results.

Fresh full-top20 evidence passing that rule supports only the registered narrow
numerical allowance, not correctness of every verifier decision. A non-top
experiment token, off-grid set, >one step or >3 flips falsifies that allowance
and stops the window. To establish causation rather than compatibility, a
separately authorized diagnostic must compare the SAME token prefix with fresh
salts across target-only and controlled verify shapes, record pre-softmax dtype/
absolute top logits, and inspect argmax versus emitted verifier token before
later tokens diverge. Shape-dependent neighboring BF16 bins support rounding;
emission disagreeing with that shape's own argmax supports a verifier/state bug.
That additional runtime instrumentation is not added or run by Round 12.
