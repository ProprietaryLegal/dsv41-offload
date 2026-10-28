Status 2026-10-23: CPU diagnostic design/tool delivered; full-logit fixed-prefix capture remains coordinator follow-up, not completed equivalence evidence. See [live roadmap](../../documents/plans/roadmap.md) and [measured record](../results/COMPARISON.md); campaign review decisions are in `LANES.md` / lane reports outside the repo.

# Teacher forcing and full-vocabulary comparison

Builds on `offload/engram/tests/logit_ab.py` at X1 base21bc3ab (prompt selection/API fixtures); no donor model/code
substitution. Affected op: diagnostic observation of target logits at fixed prefixes and identical shapes.
`fixed_prefix.py A/manifest.json B/manifest.json --out AB.json` is a CPU-only comparison tool, not a Spark runner.

The old logit_ab records token STRINGS/top20 only and follows each configuration's own greedy continuation. After
first disagreement it no longer measures the same conditional distribution. Top-k delta is not KL; absent prompt
logprobs/top-k is explicitly not full-vocabulary capture. Do not convert those historical files into dense logits.

Capture schema `dsv41.fixed-prefix.v1`:
- Top-level run_id, build_id, exact target_identity (legal-v2a+head), tokenizer_identity, numerical_config, vocab_size
  (129280 in current build), vocab_order=`token_id`, rows. Numerical_config fingerprints arithmetic/quantization/
  RoPE/cache-format settings that must stay matched; compared optimization flag/build_id remain separate.
- Each row: case_id (stable hashed task), absolute predicted position, forced_next_id, prefix_token_sha256 (all actual
  input ids through the predicting prefix), batch_token_sha256 (ALL batch ids incl. pad ids/order), schedule_sha256
  (complete preceding chunk schedule/prefix-hit/replay and cache allocation metadata), shape object, logprobs_file
  relative path and array_sha256. Shape MUST contain n_requests, query_lens, graph_mode, graph_rows, spec_depths,
  n_out, chunk_tokens. Add KV group/page fingerprints, padded offsets and backend dispatch to shape in real captures.
- Array is a `.npy` length-vocab vector of normalized logprobs in canonical token-id order0..129279. Float32 storage,
  float64 CPU reduction. No strings-to-token reconstruction, rank-local subset, top-k renormalization, or padding
  vocabulary. TP LM head shards must be gathered in correct vocabulary order before normalization/capture.
- Tool refuses missing rows, prefixes/forced ids/shapes/schedules/build identity mismatch, array hash/path escape,
  NaN/+infinity, wrong vector length or mass !=1 within1e-5. True zero probability is -inf; support mismatch yields
  explicit infinite KL. It does not silently replace infinities/NaNs or threshold outputs as “PASS.”

Reports both directional full KL, total variation, top1 token IDs/equality, support mismatches, max finite delta,
forced-token logprobs at every aligned row. KL sums over EVERY vocabulary id. A-A floor and A-B row tables remain
separate; compare matching cases/positions, distributions of shifts and outliers, not aggregate maximum alone.

Coordinator execution design:
1. Reuse logit_ab public-opinion selection for plumbing, and existing protected legal reasoning/prose prompts for
   qualification. Build one authoritative token-id fixture plus teacher-forced answer continuation from an EXISTING
   capture. Keep tool/reasoning tokens verbatim; no re-templating between builds. Store fixture only locally.
2. Query capability first on the maintenance endpoint: prompt_logprobs + complete vocabulary for selected prompt
   positions if supported. Never retry without them and label that full KL. Serve top20 API can supply diagnostics
   but not requested KL. Production speculation may refuse prompt_logprobs: explicit capability failure, use a
   bounded coordinator-owned logits capture hook, do not swap serving stack or assume API support.
3. Capture hook at target compute_logits BEFORE sample/rejection; for a bounded list of absolute positions, gather
   TP vocabulary logits in token-id order, log_softmax in FP32 (same normalization convention across builds), copy
   only selected rows into <=8-row pinned staging, persist .npy with manifest+hash. At129280*4 each row is517120B;
   eight rows ~3.95MiB, separately budgeted. Never dump complete150K-position distributions (~77GB).
4. Teacher force by prefill of fixed prompt+answer ids (score next forced token), or explicit input replay at model
   runner level. Do not feed each build's newly sampled tokens back. Exact prefill scoring is a separate shape
   diagnostic from production decode: prompt_logprobs disables replay trimming in v5/0075. To qualify 0075, ALSO
   capture final logits through normal trimmed prefill; to qualify decoding, replay fixed anchors/verified shapes
   through a matching decode path. Label these lanes and avoid comparing prefill GEMM with decode graph GEMM.
5. Lock n_requests, request ordering/context, batch/padded rows, prefix-cache hit state/salt, allocator/KV group/page
   structure, chunk boundaries, graph descriptor, forced depth vector, n_out, tokenization and sampling settings.
   Start cold for deterministic cache allocation, then warm-prefix tests independently. Include B1..4, mixed decode
   batches, final remainders1/127/128/129, cold/prefix replay, 32K/150K and winning256K/300K. Log actual shapes.
6. A1/A2/A3 repeated same configuration, then B1/B2, then A4; no helper traffic. Compare A-A paired floors by row and
   A-B outliers. All captures ephemeral/coordinator-owned, listener unchanged, output under campaign private path.
7. Stop on identity/shape/alignment mismatch, stale buffers, unnormalized/truncated distributions, unexplained
   support shift, large KL/logprob outliers above matched floor, or changed target source data. Investigate exact
   row/layer before calling it nondeterminism. Teacher forcing does not prove free-running stochastic identity;
   full task legal checks/acceptance/failure tests remain separate gates.

Mechanism of improvement: isolates arithmetic/state defects from divergent continuations; no throughput benefit
claimed. Measurement: full KL at aligned prefixes, same-shape A-A floor and per-row diagnostics. Stop conditions
above apply before any speed/promotion inference. CPU tests cover known KL, zeros/infinite support, truncation,
prefix/shape mismatch and corrupt array provenance. No serve calls or GPU work were performed by X1.
