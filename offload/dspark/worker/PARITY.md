# DSpark worker parity (D3)

## D3a: fast SM86 path vs a straightforward fp32 reference (done 2026-10-22)

`reference.py` is written independently of `model.py`/`ops.py`/`fused.py`, from DeepSeek's reference structure
(`Block.hc_mixes / hc_pre / hc_post`, `Gate`, per-expert `Expert` loop with the routing weight applied before `w2`,
`DSparkAttention` per query row, complex-number RoPE) with the engine's choices where they differ (NOTES.md). Every EXL3
matrix is applied as `H(H(x*suh) @ W) * svh` in fp32 on its exact decoded weight (`reconstruct`, bit-identical to the
CPU decoder, D1); experts are decoded on demand. None of the fast path's kernels (exllamav3 GEMM, coop MoE, Triton,
batched attention) is used.

Inputs: an English legal-style paragraph through the checkpoint's tokenizer (400 tokens; anchors are the real next
tokens) and synthetic target taps (embedding x a gain per tap layer + noise, 8 channels x400 so `main_proj`'s
power-of-two prescale runs). The target does not run on x86host; the live captures are D3b.

### 1. Plumbing, identical inputs, one step at a time (`tools/parity_substeps.py`, `results/o2/d3a_substeps.json`)
The fast path with its EXL3 matrices and routed experts swapped for exact fp32 math, stage 0, every sub-step fed the
same tensors on both sides:

| sub-step | fast vs reference |
|---|---|
| attn mixes pre / post / comb (split-K Triton + Sinkhorn vs torch) | 1.2e-7 / 2.0e-7 / 4.4e-8 rel |
| collapse + attn_norm | bit-identical |
| q_norm(wq_a), q RoPE (vLLM YaRN-1 fp32 cache vs DeepSeek complex polar) | bit-identical |
| block KV (kv_norm, RoPE, V4 record) | bit-identical |
| attention core (ring kernel, masked, sink; probabilities rounded to bf16 for P @ V as in the engine's FlashInfer kernel; the reference keeps fp32 P) | 1.7e-3 rel |
| inverse RoPE (follows the core) / wo_a (FP8 W8A16 kernel vs fp32 einsum on the same input) | 1.3e-3 / 2.3e-5 rel |
| hc_post, ffn mixes, collapse + ffn_norm | bit-identical / 1.8e-7 / bit-identical |
| routing (sqrtsoftplus, bias, top-3, renorm x1.5) and MoE output | identical experts, bit-identical output |
| shared expert | bit-identical |
| stage 0 output stream | bit-identical |

With exact weights end to end (`precise_plumbing_check` in `d3a_parity_ref.json`): `main_x` 2e-5 rel, window KV
bit-identical, head hidden 1.1-1.4 % rel (bf16 activation rounding amplified through three MoE stages) except where a
routing near-tie flips (below).

### 2. The real fast path (exllamav3 kernels) vs the reference (`tools/parity_ref.py`, `results/o2/d3a_parity_ref.json`)
24 cases, batch 1-4, k = 5 and 4, 60 requests, 292 draft positions; reference given the fast path's own window KV
("shared context") and, separately, its own (end to end). Numbers of the final worker (rerun after the D5 kernels:
ring attention with bf16 P, FP8 wo_a, fixed-shape head, fp32-accumulation main_proj; the first run, before them, gave
91.4 % / 95.3 % / 77.1 % for the three agreement rows and 9 + 16 + 0 mismatch classes).

| measure | value |
|---|---|
| main_x / window KV (same main_x) / window KV (end to end) | 0.13 % / 0.35-0.38 % / 0.70-0.74 % rel; 99.0 % of NoPE record values identical |
| head hidden, median (max) | 1.7 % (39.8 %) rel |
| expert sets per token per stage | 94.7 % identical (830 / 876) |
| requests with a routing flip | 21 / 60; **every root flip is a gate near-tie**: the reference's 3rd- vs 4th-choice gap (0.007-0.93 % of the score) is within twice the score shift the input perturbation causes |
| teacher-forced draft agreement (each position given the fast path's previous tokens) | 90.1 % overall; **97.9 %** in requests without a routing flip |
| mismatch classes | 4 numeric near-ties (no flip in the block), 25 routing-tie cascades, **0 unexplained** |
| end-to-end draft prefix agreement | 76.7 % (positions before the first mismatch) |
| confidence | identical where drafts match (max abs diff 0.0 in most requests) |

Reading: the fast path computes the same function; its differences are rounding, amplified by the drafter's own
sensitivity: synthetic taps give flat routing (3rd/4th gaps of 0.03-1 %), so a bf16-level change in the attention
output flips an expert about once per three requests. The Spark's drafter has its own kernels (exl3_mrow, exl3_smoe,
FlashInfer SM120 sparse attention on the fp8 record), so token-level agreement with it is bounded the same way. A
different draft never changes target output (the Spark verifies every draft); only acceptance can change, and that is
what C vs A measures.

## D3b: against the live Spark drafter (done 2026-10-22)

**Data.** The coordinator's capture of the served engine, rank 0, production settings (DSpark k = 5, local argmax,
confidence head, thinking on, T = 1): 453 `propose()` rounds, three request sets (a short legal prompt, an 8K summary,
two 8K prompts at once) plus the server's warm-up requests. Format: `offload/dspark/capture/speculator.py` on
offload-x86host. Per round: target positions, main_x rows (= `combine_hidden_states(taps)`), num_rejected, query ids /
positions, the drafted ids [n_req, 5] and confidences, and the raw taps when the round has <= 64 tokens.

**Replay** (`tools/parity_capture.py`). Every round rebuilds each request's window exactly as the engine's
`_prepare_dflash_inputs_kernel` defines it: the first `num_ctx - num_rejected` target rows of the request are valid
context (the rejected suffix writes no KV), the query block is [anchor, noise x 4] at P + 1 .. P + 5 with
P = last valid position, and the window is positions max(0, P - 127) .. P. The worker inserts the request's valid rows
(last <= 128) and drafts the round's whole batch at once, as the engine does, through the production graph path (each
round also runs eagerly with a trace; graph == eager in 453 / 453 rounds). Two windows are kept: **A** from the
captured main_x (isolates the drafter body), **B** from main_x the worker computes from the captured taps where the
round carries them (decode rounds; prefill rows fall back to captured main_x): the production path, the Spark sends
taps. Plumbing checks (anchor position = last valid position + 1, query ids = [anchor, noise x 4], contiguous query
positions, no request first seen mid-sequence): 0 errors.

**Classification** of every first mismatch (equal prefix, so equal Markov input), with the independent fp32
reference (`reference.py`) on the worker's own window KV: *numeric near-tie* if the engine's token is within
max(2 bf16 steps, 3 sigma of worker-vs-reference logit noise) of the worker's pick under the worker's or the
reference's logits (or the reference prefers it); *routing-tie cascade* if not, but some block token has a routing
near-tie (3rd / 4th expert score gap within 2x the worker-vs-reference score shift, floor 1e-3); else *unexplained*.

**Result, final worker** (commit 1d46bcc; `results/o2/d3b_parity_capture_final.json`), live rounds (drafts the engine
used), 542 request-rounds:

| | pos 0 | pos 1 | pos 2 | pos 3 | pos 4 | block exact |
|---|---|---|---|---|---|---|
| A, captured main_x: match | 98.7 % | 97.2 % | 96.3 % | 95.9 % | 94.7 % | 91.5 % |
| A: match given equal prefix | 98.7 % | 98.3 % | 98.3 % | 98.7 % | 97.3 % | |
| A: teacher-forced (engine's previous token as Markov input) | 98.7 % | 98.3 % | 98.3 % | 98.7 % | 97.4 % | |
| B, worker main_x from taps: match | 98.3 % | 97.4 % | 97.2 % | 95.9 % | 95.8 % | 92.8 % |
| B: match given equal prefix | 98.3 % | 98.7 % | 99.1 % | 98.3 % | 98.2 % | |

* First mismatches (A, live): 46 = **40 numeric near-ties + 6 routing-tie cascades, 0 unexplained**. In the near-tie
  cases the engine's token is the worker's rank 0 (an exact bf16 tie the worker broke toward the lower id) or rank 1,
  one or two bf16 steps (0.125-0.25 at |logit| 16-32) below. Mid-prefill drafts (the engine discards them): 2 first
  mismatches, both near-ties. Warm-up requests: no mismatch past an equal prefix.
* Confidence |worker - engine| over positions up to the first mismatch: mean 0.0029, p99 0.024, max 0.106 (n = 2,756).
* main_x from taps (440 rounds, 2,749 rows): worker vs engine 0.131 % relative (max 0.146 %); against an fp32
  computation of the same taps the worker is at 0.125 %, the engine at 0.106 %.

**Configurations compared** (same replay; live rounds, blocks exact A / B):

| worker | A | B | main_x vs fp32 | note |
|---|---|---|---|---|
| stock exllamav3 1.6.0 sm_86 build everywhere (cd48fd9) | 93.2 % | 91.7 % | 0.372 % | `d3b_parity_capture_h16acc.json` |
| whole step on fp32 accumulation | 92.8 % | 93.5 % | 0.125 % | +130-140 us per draft step; `..._facc.json` |
| final: fp32 accumulation for main_proj only, FP8 wo_a kernel, fixed-shape head | 91.5 % | 92.8 % | 0.125 % | `..._final.json` |

The one systematic effect is main_x precision. exllamav3 1.6.0 accumulates `exl3_gemm` in fp16 on sm_86 only, and
main_proj's cancelling 15,360-wide input made that 0.37 %; with fp32 accumulation for main_proj the production path
(B) gains 1.1-1.8 points of whole-block agreement. The remaining differences between configurations (A 91.5-93.2 %) are
within the binomial noise of 542 blocks (standard error ~1.1 points): every configuration's mismatches are near-ties,
and each numeric change reshuffles which near-ties flip. A different draft never changes target output (the Spark
verifies every draft); near-tie flips trade one almost equally likely draft token for another.

Per-round drafts, token ids and the per-mismatch token pairs stay on x86host (`~/dsv41-offload/o2/results/
d3b_parity_capture*.json`): they are decodable legal output (COMMON rule 5). The repo copies hold the aggregates.
