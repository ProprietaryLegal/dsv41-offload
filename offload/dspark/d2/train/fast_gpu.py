"""Training-only tensor-core products with FP32 outputs and frozen-weight dX.

Packed routed weights are decoded per tile in registers. No decoded expert
bank, CPU routing list, or host activation copy is used. This backend has its
own agreement receipt; it never replaces the qualified reference evaluator.
"""
from __future__ import annotations

from typing import Any, Callable, cast

import torch
from torch import Tensor, nn
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from triton.language.extra import libdevice  # type: ignore[import-untyped]

from .ops import PackedLinear, check_cancelled


@triton.jit  # type: ignore[untyped-decorator]
def _qdq(X, Y, N: tl.constexpr, GROUPS: tl.constexpr):  # type: ignore[no-untyped-def]
    g = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
    k = tl.arange(0, 32)
    v = tl.load(X + g[:, None] * 32 + k[None, :], g[:, None] * 32 + k[None, :] < N, 0).to(tl.float32)
    a = tl.maximum(tl.max(tl.abs(v), 1), 1.e-4) * (1. / 448.)
    bits = a.to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 255) - 127 + ((bits & 0x7fffff) != 0).to(tl.int32)
    scale = ((e + 127) << 23).to(tl.float32, bitcast=True)
    q = tl.minimum(tl.maximum(v / scale[:, None], -448.), 448.)
    qb = q.to(tl.int32, bitcast=True)
    qe = tl.maximum(((qb >> 23) & 255) - 127, -6)
    quantum = ((qe - 3 + 127) << 23).to(tl.float32, bitcast=True)
    q = libdevice.rint(q / quantum) * quantum
    tl.store(Y + g[:, None] * 32 + k[None, :], q * scale[:, None], g[:, None] * 32 + k[None, :] < N)


class _QDQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor) -> Tensor:
        x = x.contiguous()
        if x.shape[-1] % 32:
            raise ValueError("fast QDQ requires a multiple of 32 channels")
        y = torch.empty_like(x)
        _qdq[(triton.cdiv(x.numel(), 512),)](x, y, x.numel(), GROUPS=16, num_warps=4,
                                          enable_fp_fusion=False)
        return y

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> Tensor:
        return grad


def fast_qdq(x: Tensor) -> Tensor:
    return cast(Callable[[Tensor], Tensor], _QDQ.apply)(x)


@triton.jit  # type: ignore[untyped-decorator]
def _decode(W, S, e, n, k, N: tl.constexpr, K: tl.constexpr):  # type: ignore[no-untyped-def]
    packed = tl.load(W + e.to(tl.int64) * N * (K // 2) + n * (K // 2) + k // 2,
                     (n < N) & (k < K), 0).to(tl.int32)
    code = (packed >> (k % 2 * 4)) & 15
    mag = code & 7
    val = tl.where(mag < 2, mag.to(tl.float32) * .5,
                   (2 + (mag & 1)).to(tl.float32) * tl.exp2(((mag >> 1) - 2).to(tl.float32)))
    val = tl.where((code & 8) != 0, -val, val)
    scale = tl.load(S + e.to(tl.int64) * N * (K // 32) + n * (K // 32) + k // 32,
                    (n < N) & (k < K), 127).to(tl.int32)
    return val * tl.exp2((scale - 127).to(tl.float32))


@triton.jit  # type: ignore[untyped-decorator]
def _grouped(X, W, S, OFF, TILES, Y, E: tl.constexpr, N: tl.constexpr, K: tl.constexpr,  # type: ignore[no-untyped-def]
             TRANSPOSE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
             EP: tl.constexpr):
    task = tl.program_id(0)
    es = tl.arange(0, EP)
    ends = tl.load(TILES + es, es < E, 2147483647)
    e = tl.sum((ends <= task).to(tl.int32), 0)
    if e >= E:
        return
    first_tile = tl.load(TILES + e - 1, e > 0, 0)
    lo = tl.load(OFF + e)
    hi = tl.load(OFF + e + 1)
    m = lo + (task - first_tile) * BM + tl.arange(0, BM)
    out = tl.program_id(1) * BN + tl.arange(0, BN)
    r = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    if TRANSPOSE:
        for start in range(tl.cdiv(N, BK)):
            n = start * BK + r
            a = tl.load(X + m[:, None] * N + n[None, :], (m[:, None] < hi) & (n[None, :] < N), 0)
            b = _decode(W, S, e, n[:, None], out[None, :], N, K)
            acc += tl.dot(a.to(tl.float32), b, input_precision="tf32x3")
        tl.store(Y + m[:, None] * K + out[None, :], acc, (m[:, None] < hi) & (out[None, :] < K))
    else:
        for start in range(tl.cdiv(K, BK)):
            k = start * BK + r
            a = tl.load(X + m[:, None] * K + k[None, :], (m[:, None] < hi) & (k[None, :] < K), 0)
            b = _decode(W, S, e, out[None, :], k[:, None], N, K)
            acc += tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
        tl.store(Y + m[:, None] * N + out[None, :], acc, (m[:, None] < hi) & (out[None, :] < N))


def grouped_product(x: Tensor, weight: Tensor, scale: Tensor, offsets: Tensor, tiles: Tensor,
                    transpose: bool = False) -> Tensor:
    check_cancelled()
    e, n, k2 = weight.shape
    k = k2 * 2
    x = x.contiguous()
    width = k if transpose else n
    out = torch.empty(x.shape[0], width, device=x.device, dtype=torch.float32)
    bm = 16
    # Sum ceil(count/BM) <= ceil(total/BM)+E-1, including empty experts.
    grid = (triton.cdiv(x.shape[0], bm) + e, triton.cdiv(width, 64))
    _grouped[grid](x, weight, scale, offsets, tiles, out, e, n, k, transpose,
                   BM=bm, BN=64, BK=32, EP=triton.next_power_of_2(e), num_warps=4)
    return out


class _Grouped(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, weight: Tensor, scale: Tensor, offsets: Tensor, tiles: Tensor) -> Tensor:
        ctx.save_for_backward(weight, scale, offsets, tiles)
        return grouped_product(x, weight, scale, offsets, tiles)

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> tuple[Tensor, None, None, None, None]:
        w, s, offsets, tiles = ctx.saved_tensors
        return grouped_product(grad, w, s, offsets, tiles, transpose=True), None, None, None, None


class ExpertBank(nn.Module):
    """Packed banks share storage with the original per-expert buffer views."""

    def __init__(self, experts: nn.ModuleList):
        super().__init__()
        for name in ("w1", "w2", "w3"):
            layers = [cast(PackedLinear, cast(nn.ModuleDict, ex)[name]) for ex in experts]
            ws = torch.stack([ex.weight for ex in layers])
            ss = torch.stack([ex.scale for ex in layers])
            self.register_buffer(name + "_weight", ws)
            self.register_buffer(name + "_scale", ss)
            for i, ex in enumerate(layers):
                ex.weight, ex.scale = ws[i], ss[i]

    def product(self, name: str, x: Tensor, offsets: Tensor, tiles: Tensor) -> Tensor:
        return cast(Callable[..., Tensor], _Grouped.apply)(
            x, getattr(self, name + "_weight"), getattr(self, name + "_scale"), offsets, tiles)


def linear_product(x: Tensor, w: Tensor, quantized: bool) -> Tensor:
    check_cancelled()
    flat = x.reshape(-1, x.shape[-1])
    if quantized:
        # QDQ outputs and decoded frozen weights are exactly representable in
        # BF16. CUDA mm keeps FP32 output instead of rounding the dense result.
        out = torch.mm(flat.bfloat16(), w.bfloat16().T, out_dtype=torch.float32)
    else:
        out = torch.mm(flat.float(), w.float().T)
    return out.reshape(*x.shape[:-1], w.shape[0])


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, w: Tensor, quantized: bool) -> Tensor:
        ctx.save_for_backward(w)
        return linear_product(x, w, quantized)

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> tuple[Tensor, None, None]:
        (w,) = ctx.saved_tensors
        check_cancelled()
        dx = torch.mm(grad.reshape(-1, grad.shape[-1]).float(), w.float())
        return dx.reshape(*grad.shape[:-1], w.shape[1]), None, None


def fast_linear(x: Tensor, w: Tensor, quantized: bool = False) -> Tensor:
    return cast(Callable[..., Tensor], _Linear.apply)(x, w, quantized)
