# R-SP — prompt-only sequence-parallel prefill on K1 (UPS1 #3)

Status: CPU-qualified overlay, default OFF, not deployed, never run on a GPU. Window design: [WINDOW.md](WINDOW.md).

## What it is

At TP2 both Sparks repeat the row-wise work of every decoder layer on every prompt token: the mHC residual carry
([T, 4, 5120] bf16), mHC pre / post / post_pre, both RMSNorms and the Engram gate. With `VLLM_DSV41_PREFILL_SP=1`, a
large prompt forward keeps half the rows on each rank between collectives and gathers them back only where all rows are
needed. The idea is spark-ds41f patches 0014–0018 (fetched fresh at 7ec52302 into
`$C/scratch/RSP/donor`); their code targets a different model tree (`deepseek_v4_1`, CED, b12x) and does not apply, so
this is a re-implementation on our composed K1 `nvidia/model.py`.

| Where | K1 | R-SP (forward with T ≥ threshold) |
|---|---|---|
| embedding | [T, 5120], all-reduced | same, then this rank's L = ceil(T/2) rows |
| residual carry, mHC, norms | T rows on both ranks | L rows per rank |
| attention (projections, indexer, compressor, KV writes, sparse MLA, wo) | T rows; wo_b all-reduce | all-gather the normed input to T rows, same kernels, wo_b unreduced, reduce-scatter |
| MoE (gate, top-k, routed, shared) | T rows; runner's final all-reduce | all-gather to T rows, runner with `skip_final_all_reduce`, reduce-scatter |
| Engram | lookup + wkv + gate on T rows | lookup + wkv on T rows (column-parallel wkv needs equal rows), gate on L rows, pad rows pass through |
| replay (patch 0034, layers 20–39 on 128-row windows) | runs on the replay rows | a trimming step leaves SP: the cut layer's KV write gathers its input; the replay rows are fetched exactly from their owners; replay layers unsplit as K1 |
| final collapse + norm, aux states, MTP buffer | T rows | L rows, then gathered (as K1's static SP does) |

Communication volume is unchanged (all-gather + reduce-scatter carry what one ring all-reduce carries); SP adds two
collective launches per full-row layer and halves the row-wise compute and the residual memory of layers 0–19.

## Exactness

A reduce-scatter adds the same two partials as the all-reduce (one addition, commutative), gathers are copies, and
attention and the MoE see the same T rows as K1. The remaining condition is that the row-wise kernels compute each
row the same way for L rows as for T rows. They do on the image's dispatch rules once L clears a floor
(`exact_local_floor`): the fused mHC post + pre-norm GEMM serves ≤ 32 rows only; RMSNorm switches block size at 256
rows; DeepGEMM's mHC pre-norm GEMM picks its split-k from `n_sms // cdiv(rows, 64)` (buckets 1/4/16), which is 1 for
both L and T once `cdiv(L, 64) > n_sms / 2`. On GB10 (48 SMs) that is L ≥ 1,537, so **SP runs from 3,073 tokens at
TP2**: every full 4,096-token chunk (and PC1's 8K/16K chunks), never the short tail chunk of a prompt or any decode
step. Not covered by this argument: DeepGEMM's `mega_mhc` kernel (if the image's DeepGEMM has it, `can_use_mega_mhc`
takes it for any row count); the window's greedy check and the RS == AR bitwise microbenchmark gate cover it.

Threshold = max(one more than any decode forward, 0018's RDMA one-shot rule, the exactness floor,
`VLLM_DSV41_PREFILL_SP_MIN_ROWS`); it can only be raised. SP is off (logged once) with TP 1, PP, DP, prefill context
parallelism, microbatching, static SP, native mega MoE, the fused mHC all-reduce or GEMM-RS. Every rank shares T, so
the per-forward decision needs no agreement; nothing runs while a stream captures.

## Proven on CPU (Controller, no GPU)

`tests/` (64 tests, strict mypy):

- **Dataflow, bit for bit** (`test_dataflow.py`): the real K1 classes and the overlay's classes (lifted from model.py by
  AST) run on a fake TP2 group (two threads, collectives at a barrier, NCCL's two-rank bf16 sum) with row-wise torch
  stand-ins of the mHC kernels, a row-parallel attention, a MoE runner honouring `skip_final_all_reduce`, an Engram
  with the post-wkv kernel's semantics and the image's real `DecoderReplayLayers`. Final hidden states, aux hidden
  states, every KV write and the MTP buffer are identical between K1, the overlay with SP off and the overlay with SP
  on — odd T with a pad row, even T, no replay, MTP buffer, two Engram layers, one request, a replay step without a
  batch, and below the threshold. Four deliberate bugs (Engram offset, replay row fetch, double MoE reduction, missing
  reduce-scatter) are each caught.
- **Rows / threshold** (`test_rows.py`): layout covers every row once for TP 2–4; select is exact; 0018's byte rule
  reproduces the donor's 205-token TP3 example; the floor keeps every lifted image dispatch rule identical for 48, 82,
  132 and 148 SMs and is tight to the bucket; off reasons; the override only raises.
- **Overlay** (`test_overlay.py`): base = K1 MANIFEST bytes = UP1's copy (bc3aa8b0); OFF build is byte-identical; ON
  output pinned in `OVERLAY.sha256`; only the seven dataflow methods change; anchors refuse drift; the Engram kernel
  launch equals `Engram.forward`'s argument for argument in both the image and the O1 copies; mirrored image files
  sha-pinned.

Not proven: any CUDA kernel, NCCL/RoCE behaviour, speed, memory on GB10, startup.

## Composition (every shared file)

| File | Owner | R-SP |
|---|---|---|
| `vllm/models/deepseek_v41/nvidia/model.py` | K1 (= UP1 bytes, bc3aa8b0) | **replaced** by the build (K1 + anchored edits, 3c139086); K1 bytes untouched in the repo |
| `vllm/models/deepseek_v41/sp_prefill.py` | R-SP | **new** (20344b89) |
| `vllm/models/deepseek_v41/attention.py` | K1 | unchanged; relies on `wo_b.reduce_results` being read per call (RowParallelLinear) and `forward_kv` |
| `vllm/models/deepseek_v41/common/engram.py` / O1 `offload/engram/spark/engram.py` | image / O1 | unchanged; `engram_local` calls `embed`, `wkv` and the module's `_fused_engram_post_wkv_kernel` (launch pinned by test) |
| `fused_moe/runner/moe_runner.py` | image | unchanged; `skip_final_all_reduce` is its late-all-reduce switch (vLLM's Inkling uses it) |
| `vllm/models/deepseek_v41/decoder_replay_layers.py` | image (0034) | unchanged; mirrored by `sp_prefill.replay` (sha-pinned) |
| `nvidia/model_state.py`, `replay_skip.py`, `sparse_swa.py`, scheduler, `input_batch.py` | UP1 / UP2 (0067, 0075) | disjoint; they choose the replay rows R-SP fetches |
| `sparse_attn_indexer.py`, K1 `model_runner.py`, `vllm_exl3_moe/tv1_*` | TV1 (decode only) | disjoint |
| `models/common/ops/sequence_parallel.py` | image | reused for the collectives (NCCL at these sizes; EXL3 RDMA takes ≤ 256 KB only) |
| `vllm_exl3_moe` (rdma, tf_prompt, smoe fused ≤ 8 rows) | image / K1 | unchanged; the one-launch MoE path is below any SP forward |
| PF1-R2 / PF1-CAUSE trace builds | diagnostic | built on K1 model.py too: mutually exclusive with R-SP |

At their current heads PP4-PROMOTE, UP2-ADOPT, UPFIX, NBW, UP3, PC1, TV1 and PROMO1 change none of model.py,
engram.py, moe_runner.py, decoder_replay_layers.py or attention.py. Any later change to K1's model.py makes this build
refuse (BASES.json pin) until it is re-anchored.

## Build

```
export TMPDIR=/opt/dsv41/campaigns/DSV41-OFFLOAD/scratch/RSP/tmp CUDA_VISIBLE_DEVICES=
/home/operator/miniconda3/bin/python3 -m offload.sp_prefill.build_overlay --output offload/sp_prefill/generated
/home/operator/miniconda3/bin/python3 -m pytest -q offload/sp_prefill
```

`--off` emits K1's bytes. The window config `offload/launch/configs/RSP-window.env` mounts `generated/` after checking
it against `OVERLAY.sha256`.

## Expected value (ESTIMATED)

SP1's attribution puts mHC / norm / Engram projection and residuals at ~0.12 s of ~1.9 s per 4,096-token chunk; SP
halves the share in the 20 full-row layers: **~0.04–0.06 s per chunk, +1–3 % TTFT**, near the 2 % WIN bar, so
INCONCLUSIVE is a likely outcome. Memory: the residual carry halves, **80 MiB per rank at a 4,096 chunk, 320 MiB at
16,384** — more relevant to PC1's chunk16384 under NDM's 3 GiB floor than to TTFT.
