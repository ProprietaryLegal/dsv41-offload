"""Census kernel buckets: finer than SP1's buckets.json, with mHC, Engram, the indexer, its top-k and the collectives'
cross-rank waits split out (F-CENSUS.plan.md §3).

Ordered: the first rule whose regex (re.search) matches a GPU activity name wins. Patterns come from SP1's buckets.json
(offload/gb10_budget) and from the kernel names the three SP1 windows recorded; the source of each split is noted.
`rank_wait` and `bcast_wait` are never matched by name: offload.census.waterfall carves them out of paired collective
kernels (the early rank's sum_kernel / NCCL broadcast polls for the peer; KL1 item 3's pairing).
"""

from __future__ import annotations

import hashlib
import re

RULES: tuple[tuple[str, str], ...] = (
    ('prefetch', r'\bl2pf_kernel\b'),
    # exl3_rdma.cu ring: stage_kernel copies the local partial out, sum_kernel waits for the peer and adds
    ('allreduce', r'\b(stage_kernel|sum_kernel|gather_kernel)\b'),
    # NCCL broadcast at step start (one a step on each rank; 2.7-2.9 ms on rank 1, KL1-collectives.json)
    ('bcast', r'nccl'),
    ('allreduce', r'cross_device_reduce|custom_ar|all_?reduce|all_?gather|reduce_?scatter|AllReduce|AllGather|'
                  r'ReduceScatter'),
    ('engram', r'engram|\b_hash_ids_kernel\b|\b_write_hash_cache_kernel\b'),
    ('drafter', r'markov|dspark|eagle|dflash|\b_suffix_match_kernel\b|\b_selector_walk_kernel\b|\b_predict_kernel\b|'
                r'\b_trace_replay_kernel\b|\b_update_draft_inputs_kernel\b|\b_pad_trailing_draft_slots_kernel\b'),
    ('sampler', r'rejection|sample|gumbel|\b_topk_topp_kernel\b|\b_topp_sb_|\b_min_p_kernel\b|\b_temperature_kernel\b|'
                r'\b_penalties_kernel\b|\b_bad_words_kernel\b|logprob|\b_argmax_finish(_vi)?_kernel\b|'
                r'\b_resample_kernel\b|\b_thinking_budget_kernel\b|\b_compute_local_logits_stats_kernel\b|'
                r'\b_local_max_sumexp_kernel\b|\b_compute_local_residual_mass_kernel\b|'
                r'\b_compute_cumulative_log_p_kernel\b|\b_bias_kernel\b|\b_repeated_context_mask_kernel\b|'
                r'\b_compact_sampling_mask_kernel\b|\b_bincount_kernel\b|\b_ranks_kernel\b|'
                r'\b_cache_draft_logits_kernel\b'),
    # MoE routing: router GEMV and the routing top-k (_dsv4_topk_kernel), not the indexer's top-k
    ('moe_router', r'\brouter_kernel\b|\b_dsv4_topk_kernel\b|\b_route_(partial|finish)_kernel\b|\b_moe_prep_kernel\b|'
                   r'topk_softmax'),
    # routed + shared expert weight streaming (smoe decode; TensorFold grouped prefill kernels)
    ('smoe', r'\bsmoe_(rot|gu|down|fused)\b|\bgrouped_\w*kernel\b|\bgroup_(count|place)?_?kernel\b|'
             r'\bgateup_epilogue_kernel\b|\bdown_(epilogue|combine)_kernel\b|\bcombine(_y)?_kernel\b|'
             r'\bnative_combine_kernel\b|\bdecode_prep_kernel\b|\bwork_list_kernel\b|\brot_in_kernel\b|'
             r'\bdequant_kernel\b|\b_sum_groups_kernel\b|\b_moe_w(13|2)_kernel\b|mega_moe|fused_moe|moe_align'),
    # mHC: DeepGEMM tf32 pre-norm GEMM (24 x 20480 fp32 mix weight), TileLang pre/post fuse, exl3 post-collapse
    ('mhc', r'hc_prenorm|\bmhc_|post_collapse|\b_hc_|\bhc_head|sinkhorn'),
    ('lm_head', r'\bmrow(32)?_kernel<6\b|\bexl3_gemm_kernel<6\b'),
    # indexer top-k selection (sparse_attn_indexer: persistent_topk / top_k_per_row / deep_select; torch topk)
    ('topk', r'persistent_topk|top_k_per_row|deep_select|cooperative_topk|\bmbtopk::|radixSort|gatherTopK|'
             r'\b_store_candidates_kernel\b'),
    # DSA indexer: FP8 paged MQA logits and candidate-block selection (kernels/attention/dsa/candidate_blocks.py)
    ('indexer', r'paged_mqa_logits|\b_block_scores_kernel\b|\b_candidate_flags_kernel\b|\b_mask_candidates_kernel\b|'
                r'\b_indexer_\w*kernel\b|indexer'),
    ('dense', r'\bmrow(32)?_kernel\b|\bexl3_gemm_kernel\b|\b_dense_kernel\b|\blinear_kernel\b|\b_wo_a_fp8_kernel\b|'
              r'\b_mxfp8_wo_a_bmm_kernel\b|gemm|Gemm|GEMM|cutlass|cublas|nvjet|xmma|deep_gemm|scaled_mm|[Mm]arlin|'
              r'\breconstruct_had_kernel\b'),
    # vLLM's VllmTritonJitKernel launches carry the bare name 'Kernel'; every subclass in the image is an attention,
    # indexer or KV-cache op (fused_inv_rope_fp8_quant, fused_qk_rmsnorm, sparse_swa, compressor_utils, ...)
    ('attention', r'^Kernel$'),
    ('attention', r'attn|attention|flash|fmha|\bmla|mqa|_sparse_|swa|topk_(indices|lens)|'
                  r'\b_compute_topk_lens_kernel\b|\b_pack_global_topk_ragged_kernel\b|'
                  r'\b_prefill_topk_global_slots_kernel\b|\b_count_valid_row_entries_kernel\b|'
                  r'\b_compact_dense_row_to_ragged_kernel\b|rope|rotary|kv_cache|\b_kv_store_kernel\b|'
                  r'reshape_and_cache|_and_cache|_insert_k|_gather_k|\b_fused_save_compress_norm_kernel\b|kpool|'
                  r'slot_mapping|\b_block_attn_kernel\b|\b_gather_lookback_kernel\b|\b_gather_replay_batch_kernel\b|'
                  r'\b_pad_replayed_slots_kernel\b|\b_qdq_ue8m0_kernel\b|QNormRope|KVRope|\b_gather_kernel\b'),
    ('other', r'collapse|rms_norm|layer_norm|\b_prescale_kernel\b|swiglu|silu|\bactivation_kernel\b|'
              r'\b_act_qdq_kernel\b|\b_amax_kernel\b|elementwise|vectorized|reduce_kernel|index|scatter|gather|copy|'
              r'fill|\bcat\b|CatArray|embedding|_prepare_|\b_cache_inputs_kernel\b|\bexpand_kernel\b|_flatten|'
              r'_pack_|_unpack_|_shift_input|\b_accumulate_kernel\b|\b_refit_kernel\b|\b_build_shard_plan_kernel\b|'
              r'\b_insert_resampled_kernel\b|\b_update_committed_marker_cache_kernel\b|\b_post_update_kernel\b|'
              r'^memcpy32_post$|^memset32$|DeviceScan|^Memcpy|^Memset'),
)

# Not name-matched: carved out of paired collectives, or idle time classified by the activity that ends it.
WAITS = ('rank_wait', 'bcast_wait')
IDLE = ('drafter_wait', 'engram_wait', 'device_wait', 'host_gap', 'unrecorded_wait')
# SP1 wait-rule bucket -> census idle bucket (buckets.json wait_rules)
WAIT_RULE_NAMES = {'drafter': 'drafter_wait', 'engram': 'engram_wait'}

REPORT: tuple[str, ...] = (
    'smoe', 'moe_router', 'prefetch', 'dense', 'lm_head', 'mhc', 'attention', 'indexer', 'topk',
    'allreduce', 'rank_wait', 'bcast', 'bcast_wait', 'engram', 'engram_wait', 'drafter', 'drafter_wait', 'sampler',
    'other', 'unmatched', 'device_wait', 'host_gap', 'unrecorded_wait')

# Waterfall lines of F-CENSUS.plan.md §3 (each census bucket in exactly one line).
GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ('weights: routed + shared experts (smoe)', ('smoe',)),
    ('MoE routing (router GEMV + routing top-k)', ('moe_router',)),
    ('weights: dense projections (mrow K5 and other GEMMs)', ('dense',)),
    ('weights: LM head (mrow K6)', ('lm_head',)),
    ('l2pf prefetch side stream (overlapped)', ('prefetch',)),
    ('mHC (pre-norm GEMM, pre/post fuse, collapse)', ('mhc',)),
    ('attention', ('attention',)),
    ('indexer (MQA logits, candidate blocks)', ('indexer',)),
    ('indexer top-k', ('topk',)),
    ('all-reduce intrinsic', ('allreduce',)),
    ('all-reduce rank wait', ('rank_wait',)),
    ('step-start broadcast (intrinsic)', ('bcast',)),
    ('step-start broadcast wait (rank 1 waits for rank 0)', ('bcast_wait',)),
    ('drafter wait (idle before the drafts H2D)', ('drafter_wait',)),
    ('drafter kernels', ('drafter',)),
    ('Engram (kernels + idle before Engram)', ('engram', 'engram_wait')),
    ('sampler', ('sampler',)),
    ('launch / idle gaps (device wait, host gap, unrecorded)', ('device_wait', 'host_gap', 'unrecorded_wait')),
    ('other (norms, elementwise, copies) + unmatched', ('other', 'unmatched')),
)

COMPILED: tuple[tuple[str, re.Pattern[str]], ...] = tuple((b, re.compile(p)) for b, p in RULES)
SHA256 = hashlib.sha256(repr((RULES, REPORT, GROUPS)).encode()).hexdigest()

_CACHE: dict[str, str] = {}


def bucket(name: str) -> str:
    if name not in _CACHE:
        _CACHE[name] = next((b for b, p in COMPILED if p.search(name)), 'unmatched')
    return _CACHE[name]


def check() -> None:
    """Every bucket a rule, a wait or an idle class can produce is reported, and every reported bucket is in one
    waterfall line exactly once."""
    named = {b for b, _ in RULES} | set(WAITS) | set(IDLE) | {'unmatched'}
    if set(REPORT) != named:
        raise ValueError(f'REPORT differs from the rules: {sorted(set(REPORT) ^ named)}')
    grouped = [b for _, bs in GROUPS for b in bs]
    if sorted(grouped) != sorted(REPORT):
        raise ValueError('GROUPS must hold every reported bucket exactly once')


check()
