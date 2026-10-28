"""Official online-64 attention for native arms, reading the existing ring.

The caller owns output storage. No allocation or host readback in this path.
See N4-official-arithmetic.md for the pinned source and arithmetic boundaries.
"""
from __future__ import annotations

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from triton.language.extra import libdevice  # type: ignore[import-untyped]


@triton.jit  # type: ignore[untyped-decorator]
def _online64(qp, rp, pp, sp, lp, hp, bp, skp, op, scale, rs, ps,  # type: ignore[no-untyped-def]
              K: tl.constexpr, H: tl.constexpr, D: tl.constexpr, W: tl.constexpr,
              BM: tl.constexpr, DO: tl.constexpr):
    b = tl.program_id(0)
    rows = tl.program_id(1) * BM + tl.arange(0, BM)
    od = tl.program_id(2) * DO + tl.arange(0, DO)
    rm = rows < K * H
    slot = tl.load(sp + b).to(tl.int64)
    lo = tl.load(lp + b)
    hi = tl.load(hp + b)
    # Official get_dspark_topk_idxs: physical ring order, compact short prefix.
    nc = tl.minimum(W, hi + 1)
    mx = tl.full((BM,), -1e30, tl.float32)
    den = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, DO), tl.float32)
    qi = qp + b.to(tl.int64) * K * H * D + rows[:, None] * D
    for t in range(tl.cdiv(nc + K, 64)):
        j = t * 64 + tl.arange(0, 64)
        context = j < nc
        block = (j >= nc) & (j < nc + K)
        pos = tl.load(pp + slot * ps + j, mask=context, other=-1)
        valid = (context & (pos >= lo) & (pos <= hi)) | block
        keys = tl.where(context, rp + slot * rs + j * D,
                        bp + b.to(tl.int64) * K * D + (j - nc) * D)
        score = tl.zeros((BM, 64), tl.float32)
        # Match a GEMM accumulating across the whole head dimension, rather
        # than separately rounding partial dots then adding their results.
        for d0 in tl.static_range(0, D, 64):
            d = d0 + tl.arange(0, 64)
            q = tl.load(qi + d[None, :], mask=rm[:, None], other=0.)
            v = tl.load(keys[:, None] + d[None, :], mask=valid[:, None], other=0.)
            score = tl.dot(q, tl.trans(v), score)
        score = tl.where(valid[None, :], score * scale, float("-inf"))
        new_mx = tl.maximum(mx, tl.max(score, 1))
        alpha = libdevice.exp(mx - new_mx)
        p = libdevice.exp(score - new_mx[:, None])
        # The pinned TileLang CUDA uses expf without --use_fast_math and its
        # default nvcc contraction on sum_exp * scale + scores_sum.
        den = tl.fma(den, alpha, tl.sum(p, 1))
        acc = acc * alpha[:, None]
        vout = tl.load(keys[:, None] + od[None, :], mask=valid[:, None], other=0.)
        acc = tl.dot(p.to(tl.bfloat16), vout, acc)
        mx = new_mx
    sink = tl.load(skp + rows % H, mask=rm, other=0.)
    den = den + libdevice.exp(sink - mx)
    dst = op + b.to(tl.int64) * K * H * D + rows[:, None] * D + od[None, :]
    tl.store(dst, tl.div_rn(acc, den[:, None]).to(tl.bfloat16), mask=rm[:, None])


def attention_into(q: torch.Tensor, ring: torch.Tensor, positions: torch.Tensor,
                   slots: torch.Tensor, lo: torch.Tensor, last: torch.Tensor,
                   block: torch.Tensor, sink: torch.Tensor, scale: float,
                   out: torch.Tensor) -> torch.Tensor:
    b, k, h, d = q.shape
    if q.dtype != torch.bfloat16 or out.dtype != q.dtype or out.shape != q.shape:
        raise ValueError("online64 requires matching BF16 query/output")
    if not q.is_contiguous() or not block.is_contiguous() or not out.is_contiguous():
        raise ValueError("online64 requires contiguous query/block/output")
    if d % 64 or ring.stride(1) != d or ring.stride(2) != 1 or positions.stride(1) != 1:
        raise ValueError("online64 requires 64-aligned heads and contiguous ring rows")
    do = min(128, d)
    _online64[(b, triton.cdiv(k * h, 16), triton.cdiv(d, do))](
        q, ring, positions, slots, lo, last, block, sink, out, scale, ring.stride(0), positions.stride(0),
        K=k, H=h, D=d, W=ring.shape[1], BM=16, DO=do, num_warps=4, num_stages=1, enable_fp_fusion=False)
    return out
