"""FP8 paged candidate scoring; own implementation, donor mechanism in EXPLORE.md.

Keep the existing dense-column top-k contract. Only candidate keys are read;
non-candidates are -inf. Numerical identity to GB10 DeepGEMM is a WINDOW gate,
not implied by the mathematical oracle or the Ampere tests.
"""
from __future__ import annotations

import torch


def validate(q: torch.Tensor, cache: torch.Tensor, weights: torch.Tensor,
             lengths: torch.Tensor, pages: torch.Tensor, candidates: torch.Tensor,
             width: int, block_size: int) -> None:
    if q.ndim != 3:
        raise ValueError("TV1 queries must have three dimensions")
    rows, heads, dim = q.shape
    if dim != 128 or heads not in (32, 64):
        raise ValueError('TV1 requires [rows,32|64,128] FP8 queries')
    if q.dtype != torch.float8_e4m3fn or cache.dtype != torch.uint8:
        raise ValueError('TV1 candidate scoring requires FP8 Q and uint8 segregated FP8 KV')
    if cache.ndim != 4 or cache.shape[2:] != (1, 132) or cache.stride(-1) != 1:
        raise ValueError('TV1 cache must be [pages,page_tokens,1,132] with byte columns')
    if cache.shape[1] not in (64, 128, 192) or cache.stride(0) < cache.shape[1] * 132:
        raise ValueError('TV1 unsupported packed-page geometry')
    if (weights.shape != (rows, heads) or weights.dtype != torch.float32
            or lengths.numel() != rows or lengths.dtype != torch.int32
            or pages.ndim != 2 or pages.shape[0] != rows or pages.dtype != torch.int32
            or candidates.ndim != 2 or candidates.shape[0] != rows or candidates.dtype != torch.int32):
        raise ValueError('TV1 per-row weights/lengths/pages/candidate layout mismatch')
    if width <= 0 or width > pages.shape[1] * cache.shape[1] or block_size != 8:
        raise ValueError('TV1 width exceeds page table or candidate block is not eight')
    if any(t.device != q.device for t in (cache, weights, lengths, pages, candidates)):
        raise ValueError('TV1 operands must share one device')
    if any(t.stride(-1) != 1 for t in (q, weights, pages, candidates)):
        raise ValueError('TV1 requires unit last-dimension stride')


def candidate_logits(q: torch.Tensor, cache: torch.Tensor, weights: torch.Tensor,
                     lengths: torch.Tensor, pages: torch.Tensor, candidates: torch.Tensor,
                     width: int, block_size: int = 8) -> torch.Tensor:
    """Score real candidate positions, with graph-safe device lengths and int64 page offsets."""
    validate(q, cache, weights, lengths, pages, candidates, width, block_size)
    if not q.is_cuda:
        raise ValueError('TV1 Triton scorer requires CUDA')
    # Lazy import: the OFF path never imports Triton or this module.
    from vllm_exl3_moe.tv1_candidate_kernel import score  # type: ignore[import-not-found]
    out = torch.full((q.shape[0], width), -float('inf'), dtype=torch.float32, device=q.device)
    score(q, cache, weights, lengths.reshape(-1), pages, candidates, out, block_size)
    return out
