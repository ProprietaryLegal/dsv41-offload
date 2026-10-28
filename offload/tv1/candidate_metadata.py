"""The patch's metadata adapter and bounded actual-forward selected-index guard.

No donor expression is imported. Ragged packed Q and varlen indices use dense
fallback in the patch. All installed-oracle cases call this same adapter.
"""
from __future__ import annotations

import json
import re
from typing import cast

import torch

try:
    from vllm_exl3_moe.tv1_candidate import candidate_logits  # type: ignore[import-not-found]
    from vllm_exl3_moe.tv1_candidate_cases import CACHE_OFFSET, LAYERS, MAX_RAW, PAGE, STRIDE  # type: ignore[import-not-found]
except ModuleNotFoundError:
    from offload.tv1.candidate import candidate_logits
    from offload.tv1.candidate_cases import CACHE_OFFSET, LAYERS, MAX_RAW, PAGE, STRIDE

_CHECKED: set[tuple[int, int, int]] = set()


def consumer_layer(op: object) -> int:
    """Resolve the actual owning attention layer; shared K cache prefix is layer20."""
    cached = getattr(op, '_tv1_layer', None)
    if cached is not None:
        return int(cached)
    # The config context ends after model load. Profile/dummy, capture and
    # serving forwards instead bind this same map in set_forward_context().
    # See image vllm/forward_context.py:create_forward_context and CANDIDATE-INIT.md.
    from vllm.forward_context import get_forward_context  # type: ignore[import-not-found]
    context = get_forward_context().no_compile_layers
    matches = [name for name, module in context.items()
               if getattr(getattr(module, 'indexer', None), 'indexer_op', None) is op]
    if len(matches) != 1:
        raise ValueError('TV1 cannot uniquely identify actual candidate consumer')
    match = re.search(r'(?:^|\.)layers\.(\d+)\.', matches[0])
    if match is None or int(match[1]) not in LAYERS:
        raise ValueError('TV1 candidate consumer is outside qualified layers24/28/32/36')
    setattr(op, '_tv1_layer', int(match[1]))
    return int(match[1])


def map_decode(q: torch.Tensor, seq_lens: torch.Tensor, block_table: torch.Tensor,
               candidates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map installed native/flattened metadata to scorer rows without guessing lengths."""
    if q.ndim != 4:
        raise ValueError('TV1 padded query must be [metadata_batch,next_n,heads,dim]')
    batch, next_n = q.shape[:2]
    rows = batch * next_n
    if (next_n not in (1, 2) or seq_lens.shape != (batch, next_n)
            or block_table.ndim != 2 or block_table.shape[0] != batch
            or candidates.ndim != 2 or candidates.shape[0] < rows):
        raise ValueError('TV1 unqualified metadata expansion (native1/2 or flattened rows only)')
    return (q.reshape(rows, *q.shape[2:]), seq_lens.reshape(-1),
            block_table.repeat_interleave(next_n, dim=0, output_size=rows), candidates[:rows])


def score_decode(q: torch.Tensor, cache: torch.Tensor, weights: torch.Tensor,
                 seq_lens: torch.Tensor, block_table: torch.Tensor, candidates: torch.Tensor,
                 width: int, block_size: int) -> torch.Tensor:
    flat_q, lens, pages, blocks = map_decode(q, seq_lens, block_table, candidates)
    if (cache.shape[1] != PAGE or cache.stride(0) != STRIDE
            or cache.storage_offset() % STRIDE != CACHE_OFFSET
            or width != MAX_RAW or blocks.shape[1] != 2048 or flat_q.shape[1:] != (32, 128)):
        raise ValueError('TV1 loaded K1 consumer geometry differs from qualification matrix')
    return cast(torch.Tensor, candidate_logits(flat_q, cache, weights[:flat_q.shape[0]], lens, pages, blocks, width, block_size))


def check_actual(logits: torch.Tensor, q: torch.Tensor, cache: torch.Tensor,
                 weights: torch.Tensor, seq_lens: torch.Tensor, block_table: torch.Tensor,
                 candidates: torch.Tensor, schedule: torch.Tensor, width: int,
                 max_seq_len: int, layer: int, request_batch: int, topk_tokens: int,
                 topk_backend: str) -> None:
    """Bounded eager guard: at most 4*4*4=64 shadow forwards per worker.

    Metadata has CPU-known max_seq_len; no per-forward device read on checked
    buckets or capture. Qualification traffic must produce required receipts on
    both ranks before timing; absence/duplicates cannot waive equality.
    """
    bucket = context_bucket(max_seq_len)
    key = (layer, request_batch, bucket)
    if torch.cuda.is_current_stream_capturing() or key in _CHECKED:
        return
    if layer not in LAYERS or request_batch not in range(1, 5) or len(_CHECKED) >= 64:
        raise ValueError('TV1 actual comparison coverage exceeded qualified bounds')
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import apply_candidate_mask  # type: ignore[import-not-found]
    from vllm.model_executor.layers.indexer_topk import get_indexer_topk  # type: ignore[import-not-found]
    from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits  # type: ignore[import-not-found]
    select = get_indexer_topk(topk_backend)
    backend = select.resolve_backend(logits, topk_tokens, logits.shape[0])
    if backend != 'persistent' or topk_tokens != 512:
        raise ValueError('TV1 actual top-k backend differs from qualified persistent/512')
    dense = fp8_fp4_paged_mqa_logits((q, None), cache, weights[:logits.shape[0]],
                                   seq_lens, block_table, schedule,
                                   max_model_len=width, clean_logits=False, indices=None)
    apply_candidate_mask(dense, None, seq_lens.reshape(-1), candidates[:logits.shape[0]], 8, 1)
    a = torch.full((logits.shape[0], topk_tokens), -1, dtype=torch.int32, device=q.device)
    b = torch.full_like(a, -1)
    select(dense, seq_lens, q.shape[1], a, topk_tokens, max_seq_len)
    select(logits, seq_lens, q.shape[1], b, topk_tokens, max_seq_len)
    if not torch.equal(a, b):
        raise ValueError('TV1 actual metadata selected-index mismatch: STOP')
    lens = seq_lens.reshape(-1)
    positive = lens[lens > 0]
    raw_context = int(positive.min().item()) if positive.numel() else 0
    if raw_context == 0:
        # Initialization/capture warmups are not evidence about an occupied context.
        return
    _CHECKED.add(key)
    print('TV1_ENGINE_ORACLE ' + json.dumps({
        'type': 'tv1-engine-oracle-v2', 'layer': layer, 'batch': request_batch,
        'raw_context': raw_context, 'rows': logits.shape[0], 'next_n': q.shape[1],
        'stride': cache.stride(0), 'cache_offset': cache.storage_offset(), 'width': width,
        'candidate_capacity': candidates.shape[1], 'page': cache.shape[1],
        'layout': 'native2' if q.shape[1] == 2 else 'single' if q.shape[0] == request_batch else 'flattened',
        'backend': backend, 'identical': True}), flush=True)


def context_bucket(max_seq_len: int) -> int:
    return 0 if max_seq_len < 30000 else 1 if max_seq_len < 140000 else 2 if max_seq_len < 300000 else 3


def needs_eager_qualification(request_batch: int, max_seq_len: int) -> bool:
    """Force only unchecked real decode buckets eager; captured timing resumes afterwards."""
    if max_seq_len < 30000:
        # Short all-selected consumers may bypass this op entirely. Never
        # force every short decode eager while waiting for an impossible receipt.
        return False
    if request_batch not in range(1, 5):
        raise ValueError('TV1 qualification request batch outside B1..B4')
    bucket = context_bucket(max_seq_len)
    return any((layer, request_batch, bucket) not in _CHECKED for layer in LAYERS)
