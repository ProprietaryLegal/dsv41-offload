"""Packed MXFP4 forward, decoded only in registers, FP64 reference sums.

Backward alone materializes one selected expert matrix in BF16. This kernel
does not expand a matrix/bank in device memory and has no expert gradients.
"""
from __future__ import annotations

import torch
from torch import Tensor
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]


@triton.jit  # type: ignore[untyped-decorator]
def _packed(X, W, S, Y, K: tl.constexpr, N: tl.constexpr,  # type: ignore[no-untyped-def]
            BN: tl.constexpr, BK: tl.constexpr) -> None:
    m = tl.program_id(0)
    n = tl.program_id(1)*BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    total = tl.full((BN,), 0., tl.float64)
    for start in range(tl.cdiv(K, BK)):
        k = start*BK+kk
        a = tl.load(X+m*K+k, k < K, 0).to(tl.float32)
        packed = tl.load(W+n[:, None]*(K//2)+k[None, :]//2,
                         (n[:, None] < N) & (k[None, :] < K), 0).to(tl.int32)
        code = (packed >> ((k[None, :] % 2)*4)) & 15
        mag = code & 7
        value = tl.where(mag < 2, mag.to(tl.float32)*.5,
                         (2+(mag & 1)).to(tl.float32)*tl.exp2(((mag >> 1)-2).to(tl.float32)))
        value = tl.where((code & 8) != 0, -value, value)
        scale = tl.load(S+n[:, None]*(K//32)+k[None, :]//32,
                        (n[:, None] < N) & (k[None, :] < K), 127).to(tl.int32)
        weight = value*tl.exp2((scale-127).to(tl.float32))
        # QDQ activations and FP4 values have <=4 and <=3 significant bits;
        # their product is exact in FP32. Only the sum needs FP64.
        total += tl.sum((a[None, :]*weight).to(tl.float64), 1)
    tl.store(Y+m*N+n, total.to(tl.float32), n < N)


def packed_matmul(x: Tensor, weight: Tensor, scale: Tensor) -> Tensor:
    K = x.shape[-1]
    n = weight.shape[0]
    x2 = x.reshape(-1, K).contiguous()
    out = torch.empty(x2.shape[0], n, device=x.device, dtype=torch.float32)
    _packed[(x2.shape[0], triton.cdiv(n, 4))](x2, weight, scale, out, K, n, BN=4, BK=128,
                                           num_warps=4, enable_fp_fusion=False)
    return out.reshape(*x.shape[:-1], n)
