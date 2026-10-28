# DSpark drafter of the running build: what the worker must compute (traced 2026-10-22)

Sources read (engine as it runs in `dsv41-serve`, copy in `/opt/dsv41/data/src-image/`):
`vllm/models/deepseek_v41/nvidia/dspark.py`, `nvidia/model.py` (`DeepseekV4DecoderLayer`), `attention.py`,
`nvidia/flashinfer_sparse.py` (SM120 class), `vllm/models/deepseek_v4/nvidia/{model.py (MoE), ops/o_proj.py}`,
`vllm/v1/worker/gpu/spec_decode/{dflash,dspark}/speculator.py`, `vllm/v1/attention/backends/mla/sparse_swa.py`
(non-causal DSpark indices), `model_executor/kernels/mhc/torch.py`, `models/deepseek_v41/common/rope.py`,
`nvidia/ops/markov_cache.py`, `model_executor/models/qwen3_dspark.py`, the EXL3 plugin `vllm_exl3_moe/`
(`config.py`, `dense.py`, `head.py`, `forward.py`, `kernels.py`, `sparse_attn.py`), and DeepSeek's reference
`inference/model.py` + `kernel.py` of DeepSeek-V4.1-Flash-1017 @ 1bae0e10 (code only, `/opt/dsv41/data/ref-ds41/`).
Live launch: `--speculative-config {"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":
[[1,2,5],[3,4,4]]}`, no `--attention-backend` (GB10 -> `DeepseekV4FlashInferSM120Attention`), env
`VLLM_DSPARK_LOCAL_ARGMAX=1`, `VLLM_DSPARK_MARKOV_CACHE=/root/.cache/markov_top_tokens.json`,
`VLLM_DSPARK_DEPTH_POLICY=conf`, `VLLM_DSV41_ADAPTIVE_K=1`, `EXL3_INT8_GEMV=0`.

## Shapes (config.json, checked by `checkpoint.DrafterShape`)
hidden 5120, vocab 129,280, 64 heads x 512 (rope 64, NoPE 448), q_lora 1280, o_lora 1024 in 8 groups, 3 stages
(`mtp.0..2` = engine layers 40..42), compress ratio 0 (sliding window 128), MoE 128 experts top-3 + 1 shared expert,
inter 2304, sqrtsoftplus / noaux_tc / normalized, routed scale 1.5, SwiGLU limit 10, mHC hc_mult 4, 20 Sinkhorn
iterations, hc_eps 1e-6, rms eps 1e-20, rope theta 10000, block 5, noise token 128799, Markov rank 256, taps = target
layers 37/38/39.

## One draft round (engine semantics)

Let P = the last position whose target features are known (last accepted position); the anchor is the token the target
just sampled at position P+1 (the "bonus"; during chunked prefill the next prompt token). k = 5 at 1-2 running requests,
4 at 3-4 (`num_speculative_tokens_per_batch_size`).

**Context insert** (every position p whose target forward ran and was not rejected, i.e. accepted positions only):
`taps[p] = concat(mean_hc(h_37), mean_hc(h_38), mean_hc(h_39))` (the hc-mean of each tap layer's *input* stream,
15,360 wide, bf16); `main_x[p] = main_norm(main_proj(taps[p]))` (main_proj EXL3 K5; above 32 rows the plugin divides
the input by a power of two so |x| fits fp16 and multiplies the output back); per stage s:
`kv = rope_p(kv_norm(wkv_s(main_x[p])))` (wkv = the second part of the fused wq_a|wkv EXL3 K5 module), stored in the
stage's sliding-window cache as the V4 fp8_ds_mla record on SM12x: 448 NoPE dims fp8 e4m3 with one UE8M0 scale per 64,
64 RoPE dims bf16. Rejected rows write nothing.

**Query block**: ids `[anchor, noise x (k-1)]` at positions P+1..P+k. `x = embed(ids)` (bf16, the target's embedding),
stream = 4 copies. For each stage (pre_mix carried in, `None` = copy 0 for stage 0):
1. attn mixes from the stream (`hc_attn_fn/scale/base`: fp32 GEMV over the flattened 4x5120 stream x rsqrt(mean sq +
   1e-20), pre = sigmoid+eps, post = 2 sigmoid, comb = softmax rows + eps then 20 Sinkhorn column/row normalisations);
   `x = attn_norm(collapse(stream, pre_mix_in))` (collapse = sum pre_mix_i * stream_i in fp32 -> bf16).
2. attention: `q = rope(wq_b(q_norm(wq_a(x))))` [k, 64, 512]; block KV = `rope(kv_norm(wkv(x)))`, written as records
   at P+1..P+k. Every block row attends to the same set: context positions max(0, P-127)..P plus all k block positions
   (non-causal inside the block; `ComputeDSparkNoncausalSWAIndicesKernel`: start = max(prefix_len - 128, 0),
   end = seq_len); K = V = the decoded 512-vector (MQA); scale 512^-0.5; a per-head sink logit joins the softmax
   denominator only. Output rope dims get the inverse rotation at the row's own position.
3. `wo_a`: FP8 e4m3 [8192, 4096] with 32x32 UE8M0 block scales, block-diagonal over 8 groups of 8 heads; on SM12x a
   Marlin MXFP8 W8A16 GEMM on the bf16 output (no activation quantization on this branch; the 1017 reference's
   act_quant of o is applied only on the EXL3 `grouped_bmm` branch, which the drafter's FP8 wo_a does not take;
   serve log 15:52:45: "Using Sm12xMarlinMxfp8BmmLinearKernel for MXFP8 GEMM" right after the draft model loads).
   `wo_b` EXL3 K5 [8192 -> 5120].
4. `stream = post * attn_out + comb^T stream` (fp32 -> bf16), ffn mixes from the new stream, `x = ffn_norm(collapse(
   stream, attn_pre))`; gate logits fp32 [128]; scores = sqrt(softplus(logits)); top-3 of scores + bias; weights =
   scores / sum * 1.5; routed experts EXL3 K2 (all 128) through the coop kernel (fp16 in, fp16 routing weights) +
   shared expert EXL3 K5 (w1/w3 -> clamp/silu -> w2); `stream = post * ffn_out + comb^T stream`; pre_mix_out = ffn_pre.
After stage 2: `head_hidden = collapse(stream, ffn_pre of stage 2)` [k, 5120] (pre-norm).

**Heads**: `base = head(norm(head_hidden))` (EXL3 K6 head, bf16 logits). Sequential Markov, greedy (temperature 0 /
local argmax, as deployed): `prev = anchor`; for i in 0..k-1: `e_i = markov_embed[prev]` (bf16 [256]);
`bias_i = bf16(sum_r fp32(markov_head[v, r]) * fp32(e_i[r]))` (rows of the 4,096 most frequent tokens precomputed by
the same kernel: same bits); `d_i = argmax(bf16(base_i + bias_i))`, ties -> lowest id; `prev = d_i`.
Confidence: `c_i = sigmoid(w_conf . fp32(concat(head_hidden_i, e_i)))` (w_conf bf16 -> fp32, no bias).
The Spark then verifies d_0..d_{k-1}; the confidence feeds `VLLM_DSPARK_DEPTH_POLICY=conf` and the 0047 adaptive verify
window (the verifier side; the worker only reports c_i).

## Differences between the engine and DeepSeek's reference that the worker follows the engine on
* KV record: reference fake-quantizes all 512 dims to fp8 per 32 (UE8M0); the engine on SM12x stores the V4 record
  (448 NoPE fp8 per-64 UE8M0 + 64 bf16 rope). The worker emulates the engine record (exact values, stored in bf16).
* wo_a input: reference act_quant(o, 32) before the einsum; the engine's FP8 branch keeps o in bf16 (W8A16).
* Routing weights reach the coop kernel as fp16 (plugin), the reference keeps fp32.
* EXL3 everywhere the plugin uses it (dense K5 incl. wkv and main_proj, experts K2, head K6) vs the reference's FP8/FP4.

## Representation changes on the 3090 (draft-only; target weights untouched)
* EXL3 tensors: unchanged bytes, exllamav3 1.6.0 kernels (`exl3_gemm`, `exl3_moe_coop`) instead of the plugin's
  sm_90+ `exl3_mrow` / `exl3_smoe`; same decoded weights bit for bit (D1), different accumulation.
* `wo_a`: the checkpoint's FP8 bytes and UE8M0 block scales stay as they are (32 MiB per stage); a Triton W8A16
  kernel (`fused.wo_a_fp8`) dequantizes in registers (the e4m3 byte placed as fp16 bits is the value / 2^8, subnormals
  included; x 2^8 x scale is exact in bf16) and accumulates bf16 x bf16 products in fp32, like the engine's Marlin
  MXFP8 W8A16 kernel but in its own order. The unfused path (`fused=False`) expands to bf16 at load and uses a bf16
  batched GEMM.
* Accumulation: exllamav3 1.6.0's `exl3_gemm` accumulates in fp16 on sm_86 (folded into fp32 per k-slice; ENV.md);
  the worker keeps that for the draft step's dense / head GEMMs and runs main_proj on the fp32-accumulation build
  (`exllamav3_ext_facc`). The coop MoE kernel accumulates in fp16 with periodic fp32 folds on every arch (its design).
* Kernel selection: the head runs with a fixed `exl3_gemm` shape (3: 16 x 32 x 256 tiles, the autotuner's own pick at
  1-16 rows) in chunks of <= 16 rows instead of exllamav3's timing-based autotune, so each row's logits are the same
  for any row count and in every process (CONTRACT 1.3 n_out); 17-20-row heads (4 requests x 5) take two passes.
* main_proj below the plugin's mrow threshold: the engine runs `exl3_mrow` (sm_90+, bf16 in, no prescale) at <= 32
  rows; the worker always runs the plugin's prompt-size path (power-of-two prescale, fp16 in, fp32 out).
* Draft KV: bf16 tensors holding the exact dequantized V4-record values (no packed bytes).
* Block attention (fused path, `fused.block_attention_ring`): one Triton program per (request, 16 or 32 query rows,
  256 output dims) reads the window rows straight from the request's ring slot (mask = ring position in [lo, last]) as
  a 128-key tile and the k block rows as a 16-key tile; fp32 scores (bf16 x bf16 MMA, fp32 accumulate), single-pass
  softmax with the global row max (all <= 133 keys at once), sink in the denominator, probabilities rounded to bf16
  for P @ V, fp32 denominator, bf16 out. The engine's FlashInfer SM120 kernel also feeds bf16 probabilities to the
  P @ V MMA but normalizes per key tile (online softmax), so its P roundings are taken against tile maxima; both are
  within bf16 rounding of the exact result (measured 1.78e-3 rel vs fp64 on real stage-0 tensors for both this
  kernel and ops.block_attention).
