"""Exact native post-TP combine into caller-owned BF16 output.

Channel partials must already have been reduced across both ranks. No routing
weight is applied here: it was applied before the down-input QDQ. Arithmetic
matches native.ops.moe_combine_official including ascending-ID FP32 additions.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...native.ops import _bf16_rne


@triton.jit
def _combine(P, IDS, O, H: tl.constexpr, PS: tl.constexpr, OS: tl.constexpr):
    row = tl.program_id(0)
    c = tl.program_id(1)*256 + tl.arange(0, 256)
    a = tl.load(IDS + row*3)
    b = tl.load(IDS + row*3 + 1)
    d = tl.load(IDS + row*3 + 2)
    i, j, k = tl.full((), 0, tl.int32), tl.full((), 1, tl.int32), tl.full((), 2, tl.int32)
    swap = a > b
    a, b = tl.minimum(a, b), tl.maximum(a, b)
    i, j = tl.where(swap, j, i), tl.where(swap, i, j)
    swap = b > d
    b, d = tl.minimum(b, d), tl.maximum(b, d)
    j, k = tl.where(swap, k, j), tl.where(swap, j, k)
    swap = a > b
    i, j = tl.where(swap, j, i), tl.where(swap, i, j)
    x = _bf16_rne(tl.load(P + row*PS + i*H + c, c < H, other=0).to(tl.float32))
    y = _bf16_rne(tl.load(P + row*PS + j*H + c, c < H, other=0).to(tl.float32))
    z = _bf16_rne(tl.load(P + row*PS + k*H + c, c < H, other=0).to(tl.float32))
    shared = _bf16_rne(tl.load(P + row*PS + 3*H + c, c < H, other=0).to(tl.float32))
    total = _bf16_rne(((x + y) + z) + shared)
    tl.store(O + row*OS + c, total.to(tl.bfloat16), c < H)


def combine_official_into(parts: torch.Tensor, ids: torch.Tensor, out: torch.Tensor) -> None:
    if parts.ndim != 3 or parts.shape[1] != 4 or parts.dtype != torch.float32:
        raise ValueError('TP-reduced FP32 parts [T,4,H] required')
    rows, _, channels = parts.shape
    if ids.shape != (rows, 3) or ids.dtype not in (torch.int32, torch.int64) or not ids.is_contiguous():
        raise ValueError('contiguous int32/int64 IDs [T,3] required')
    if out.shape != (rows, channels) or out.dtype != torch.bfloat16 or out.stride(1) != 1:
        raise ValueError('BF16 output [T,H] with contiguous columns required')
    if parts.device != ids.device or parts.device != out.device:
        raise ValueError('combine buffers must be on the same device')
    if parts.stride(2) != 1 or parts.stride(1) != channels or parts.stride(0) < 4*channels or out.stride(0) < channels:
        raise ValueError('nonoverlapping parts/output rows and contiguous slots required')
    if out.untyped_storage().data_ptr() in (parts.untyped_storage().data_ptr(), ids.untyped_storage().data_ptr()):
        raise ValueError('combine output must not alias inputs')
    _combine[(rows, triton.cdiv(channels, 256))](parts, ids, out, channels, parts.stride(0), out.stride(0),
                                               num_warps=4, enable_fp_fusion=False)
