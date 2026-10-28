# SPDX-License-Identifier: Apache-2.0
"""dsv41 patch 0059 (VLLM_DSV41_KV_NVFP4=1, default off): FlashInfer's 384-byte NVFP4 DSV4 cache record on SM12x.

FlashInfer's SM120 sparse-MLA kernels read a group-16 NVFP4 record for the sliding-window and the compressed KV
caches (``trtllm_batch_decode_sparse_mla_dsv4(kv_cache_format="nvfp4")``): 448 NoPE values as e2m1 pairs with one
e4m3 scale per 16 values, the 64 RoPE values copied as BF16, 384 bytes per token instead of the 584 of fp8_ds_mla.
Inside a page the data rows (352 B) come first and the scale rows (32 B) after them, so the layout is FlashInfer's
own: rows are written with its ``nvfp4_quantize_append_sparse_mla_cache`` after GPT-J RoPE on dims 448..511 here,
which keeps writer and reader bit-compatible by construction.
"""

import os

import torch

from vllm.triton_utils import tl, triton

KV_NVFP4 = os.environ.get("VLLM_DSV41_KV_NVFP4", "0") == "1"
NVFP4_RECORD_BYTES = 384


@triton.jit
def _rope_rows_kernel(
    rows,
    positions,
    cos_sin,
    out,
    ROW_STRIDE: tl.constexpr,
    COS_STRIDE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
):
    # Same rotation as fused_compress_quant_cache.py's insert kernels: GPT-J pairs, NoPE pairs load (cos, sin) =
    # (1, 0), the compressed state takes the position of its group's first token.
    t = tl.program_id(0)
    position = tl.load(positions + t)
    d = tl.arange(0, 512)
    x = tl.load(rows + t.to(tl.int64) * ROW_STRIDE + d).to(tl.float32)
    even, odd = tl.split(tl.reshape(x, (256, 2)))
    pair = tl.arange(0, 256) - 224
    cs = cos_sin + (position // COMPRESS_RATIO * COMPRESS_RATIO) * COS_STRIDE
    c = tl.load(cs + tl.maximum(pair, 0), pair >= 0, other=1.0).to(tl.float32)
    s = tl.load(cs + 32 + tl.maximum(pair, 0), pair >= 0, other=0.0).to(tl.float32)
    rotated = tl.interleave(even * c - odd * s, odd * c + even * s)
    tl.store(out + t.to(tl.int64) * 512 + d, rotated.to(tl.bfloat16))


_module = None


def prebuild() -> None:
    """Build/load FlashInfer's NVFP4 module before CUDA graphs are captured (the writer runs inside captured regions)."""
    global _module
    if _module is None:
        from flashinfer.mla._sparse_mla_nvfp4_sm120 import get_sparse_mla_nvfp4_sm120_module

        _module = get_sparse_mla_nvfp4_sm120_module()


def nvfp4_rope_insert(
    rows: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    compress_ratio: int = 1,
) -> None:
    """GPT-J RoPE on the RoPE dims of BF16 latent rows, then FlashInfer's NVFP4 quantize-append by cache slot.

    ``rows`` holds one 512-wide BF16 row per slot (KV already normalized); slots < 0 are padding. With
    ``compress_ratio`` > 1 only rows ending a compress group are written, as in rope_quant_insert.
    """
    n = slot_mapping.numel()
    if n == 0:
        return
    assert rows.dtype == torch.bfloat16 and rows.shape[-1] == 512 and rows.stride(-1) == 1
    assert cache.dtype == torch.uint8 and cache.shape[-1] == NVFP4_RECORD_BYTES
    roped = torch.empty((n, 512), dtype=torch.bfloat16, device=rows.device)
    _rope_rows_kernel[(n,)](
        rows,
        positions,
        cos_sin_cache,
        roped,
        ROW_STRIDE=rows.stride(0),
        COS_STRIDE=cos_sin_cache.stride(0),
        COMPRESS_RATIO=compress_ratio,
        num_warps=4,
    )
    slots = slot_mapping
    if compress_ratio > 1:
        slots = torch.where(
            (positions[:n] + 1) % compress_ratio == 0, slot_mapping, torch.full_like(slot_mapping, -1)
        )
    nvfp4_append(roped, slots, cache)


def nvfp4_append(rows: torch.Tensor, slot_mapping: torch.Tensor, cache: torch.Tensor) -> None:
    """FlashInfer's NVFP4 quantize-append of already-roped BF16 rows (one per slot; slots < 0 are skipped)."""
    from flashinfer.mla._sparse_mla_nvfp4_sm120 import nvfp4_quantize_append_sparse_mla_cache

    m = slot_mapping.numel()
    if m == 0:
        return
    assert rows.shape[0] >= m and rows.shape[-1] == 512 and rows.dtype == torch.bfloat16
    nvfp4_quantize_append_sparse_mla_cache(rows[:m].contiguous(), slot_mapping.contiguous(), cache)
