"""Original FP8 paged scorer, portable SM86/SM121 Triton (no TMEM/UMMA).

An independent dot for each (query, candidate key) uses FP32 accumulation.
The dense mathematical oracle is in tests; production DeepGEMM top-k equality
is mandatory in the GB10 qualification window.
"""
from __future__ import annotations

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]


@triton.jit  # type: ignore[untyped-decorator]
def _e4m3(bits: tl.tensor) -> tl.tensor:
    # Exact e4m3fn conversion also works on SM86, which has no native e4m3fn.
    exponent = (bits >> 3) & 15
    mantissa = (bits & 7).to(tl.float32)
    value = tl.where(exponent == 0, mantissa * (2. ** -9),
                     (1. + mantissa * 0.125) * tl.exp2(exponent.to(tl.float32) - 7.))
    value = tl.where((exponent == 15) & (mantissa == 7), float('nan'), value)
    return tl.where((bits & 128) != 0, -value, value).to(tl.float16)


@triton.jit  # type: ignore[untyped-decorator]
def _score(Q: tl.tensor, K: tl.tensor, W: tl.tensor, L: tl.tensor, P: tl.tensor,
           C: tl.tensor, O: tl.tensor, QS: tl.constexpr, QH: tl.constexpr,
           KS: tl.constexpr, WS: tl.constexpr, PS: tl.constexpr, CS: tl.constexpr,
           OS: tl.constexpr, PAGE: tl.constexpr, NC: tl.constexpr, WIDTH: tl.constexpr,
           HEADS: tl.constexpr, BLOCK: tl.constexpr, TILE: tl.constexpr) -> None:
    row = tl.program_id(0)
    ci = tl.program_id(1) * TILE + tl.arange(0, TILE)
    cb = tl.load(C + row * CS + ci // BLOCK, ci // BLOCK < NC, other=-1)
    pos = cb * BLOCK + ci % BLOCK
    visible = tl.minimum(tl.maximum(tl.load(L + row), 0), WIDTH)
    good = (ci < NC * BLOCK) & (cb >= 0) & (pos < visible)
    physical = tl.load(P + row * PS + pos // PAGE, mask=good, other=0).to(tl.int64)
    base = physical * KS
    d = tl.arange(0, 128)
    # FP8 values are segregated from fp32 scales within each page.
    kv = _e4m3(tl.load(K + base[:, None] + (pos % PAGE)[:, None] * 128 + d[None, :],
                       mask=good[:, None], other=0))
    scale_bytes = K + base + PAGE * 128 + (pos % PAGE) * 4
    scale = tl.load(scale_bytes.to(tl.pointer_type(tl.float32)), mask=good, other=0)
    h = tl.arange(0, HEADS)
    query = _e4m3(tl.load(Q + row * QS + h[:, None] * QH + d[None, :]))
    dots = tl.dot(query, tl.trans(kv), input_precision='ieee')
    weight = tl.load(W + row * WS + h)
    reduced = tl.sum(tl.maximum(dots, 0.) * weight[:, None], axis=0) * scale
    tl.store(O + row * OS + pos, reduced, mask=good)


def score(q: torch.Tensor, cache: torch.Tensor, weights: torch.Tensor,
          lengths: torch.Tensor, pages: torch.Tensor, candidates: torch.Tensor,
          out: torch.Tensor, block: int) -> None:
    if q.shape[0] == 0 or candidates.shape[1] == 0:
        return
    with torch.cuda.device(q.device):
        _score[(q.shape[0], triton.cdiv(candidates.shape[1] * block, 32))](
            q.view(torch.uint8), cache, weights, lengths, pages, candidates, out,
            q.stride(0), q.stride(1), cache.stride(0), weights.stride(0), pages.stride(0),
            candidates.stride(0), out.stride(0), cache.shape[1], candidates.shape[1],
            out.shape[1], q.shape[1], block, 32, num_warps=4, enable_fp_fusion=False)
