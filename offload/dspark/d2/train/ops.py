"""Official forward values; STE QDQ and recomputed frozen-weight input gradients.

No decoded routed matrix survives a call. Packed expert bytes alone are saved
for backward. BF16 decoding is exact for the pinned 1017 tensors (N1 audit).
FP64 accumulation is an oracle option, FP32 is the training path; TF32 is off.
"""
from __future__ import annotations

from typing import Any, Callable, cast
from contextlib import contextmanager
from collections.abc import Iterator
import threading

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ...native.formats import act_qdq, decode_fp4

_STOP: threading.Event | None = None


class TrainingCancelled(Exception):
    """Discard the current, uncommitted optimizer step and checkpoint its boundary."""


@contextmanager
def cancellation(stop: threading.Event) -> Iterator[None]:
    global _STOP
    old, _STOP = _STOP, stop
    try:
        yield
    finally:
        _STOP = old


def check_cancelled() -> None:
    if _STOP is not None and _STOP.is_set():
        raise TrainingCancelled()


class _QDQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor) -> Tensor:
        return act_qdq(x)

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> Tensor:
        return grad


def qdq(x: Tensor, fast: bool = False) -> Tensor:
    if fast:
        from .fast_gpu import fast_qdq
        return fast_qdq(x)
    return cast(Callable[[Tensor], Tensor], _QDQ.apply)(x)


class _FrozenLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, w: Tensor, scale: Tensor | None, acc: torch.dtype) -> Tensor:
        check_cancelled()
        ctx.save_for_backward(w, *(() if scale is None else (scale,)))
        ctx.acc, ctx.input_dtype, ctx.input_device = acc, x.dtype, x.device
        if scale is not None and w.is_cuda:
            from .packed_forward import packed_matmul
            result = packed_matmul(x.to(w.device), w, scale)
            return result.to(device=x.device, dtype=x.dtype)
        decoded = w if scale is None else decode_fp4(w, scale).to(torch.bfloat16)
        return F.linear(x.to(device=w.device, dtype=acc), decoded.to(acc)).to(device=x.device, dtype=x.dtype)

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> tuple[Tensor, None, None, None]:
        check_cancelled()
        saved = ctx.saved_tensors
        decoded = saved[0] if len(saved) == 1 else decode_fp4(*saved).to(torch.bfloat16)
        # dX uses FP32 accumulation on CUDA; this changes no forward value.
        gradient_acc = torch.float32 if decoded.is_cuda else ctx.acc
        dx = (grad.to(device=decoded.device, dtype=gradient_acc) @ decoded.to(gradient_acc)).to(device=ctx.input_device, dtype=ctx.input_dtype)
        return dx, None, None, None


def frozen_linear(x: Tensor, w: Tensor, scale: Tensor | None = None,
                  acc: torch.dtype = torch.float32) -> Tensor:
    return cast(Callable[[Tensor, Tensor, Tensor | None, torch.dtype], Tensor], _FrozenLinear.apply)(x, w, scale, acc)


def head_product(a: Tensor, b: Tensor) -> Tensor:
    """FP32 head output from a two-part BF16 activation and exact BF16 weights.

    No QDQ follows the vocabulary head, so its small input decomposition error
    cannot cross any internal quantizer. Whole-model KL/loss/gradient gates
    explicitly include this arithmetic. CPU is the FP32 oracle for tests.
    """
    if not a.is_cuda or b.dtype != torch.bfloat16:
        return a.float() @ b.float()
    high = a.bfloat16()
    low = (a.float() - high.float()).bfloat16()
    return torch.mm(high, b, out_dtype=torch.float32) + torch.mm(low, b, out_dtype=torch.float32)


class _HeadLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, w: Tensor) -> Tensor:
        ctx.save_for_backward(w)
        return head_product(x.reshape(-1, x.shape[-1]), w.T).reshape(*x.shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> tuple[Tensor, None]:
        (w,) = ctx.saved_tensors
        return head_product(grad.reshape(-1, grad.shape[-1]), w).reshape(*grad.shape[:-1], w.shape[1]), None


def fast_head(x: Tensor, w: Tensor) -> Tensor:
    return cast(Callable[[Tensor, Tensor], Tensor], _HeadLinear.apply)(x, w)


class _GroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, w: Tensor) -> Tensor:
        ctx.save_for_backward(w)
        expanded = w.float()
        if x.device.type == "cpu" and x.shape[0] > 1:
            return torch.cat([torch.einsum("bsgd,grd->bsgr", x[b:b+1], expanded) for b in range(x.shape[0])], 0)
        return torch.einsum("bsgd,grd->bsgr", x, expanded)

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> tuple[Tensor, None]:
        (w,) = ctx.saved_tensors
        return torch.einsum("bsgr,grd->bsgd", grad, w.float()), None


def grouped_linear(x: Tensor, w: Tensor) -> Tensor:
    # model.py's unquantized wo_a is specifically an FP32 grouped einsum,
    # including in the FP64 reference. Widening that sum changes QDQ later.
    return cast(Callable[[Tensor, Tensor], Tensor], _GroupedLinear.apply)(x, w)


class Dense(nn.Module):
    """Frozen decoded BF16 master, optional r32/64 LoRA, official input QDQ.

LoRA is added before the linear output rounding. This is a floating adapter;
the merged FP8 path is deliberately a separate packaging/evaluation gate.
"""
    weight: Tensor

    def __init__(self, weight: Tensor, rank: int = 0, quant_input: bool = True,
                 acc: torch.dtype = torch.float32):
        super().__init__()
        self.register_buffer("weight", weight.to(torch.bfloat16))
        self.quant_input, self.acc = quant_input, acc
        self.fast_training = False
        self.rank = rank
        if rank:
            if rank not in (32, 64):
                raise ValueError("Phase 2a supports only LoRA r32 or r64")
            self.a = nn.Parameter(torch.empty(rank, weight.shape[1], device=weight.device, dtype=torch.float32))
            self.b = nn.Parameter(torch.zeros(weight.shape[0], rank, device=weight.device, dtype=torch.float32))
            nn.init.kaiming_uniform_(self.a, a=5 ** .5)

    def forward(self, x: Tensor) -> Tensor:
        x = qdq(x, self.fast_training) if self.quant_input else x
        if self.fast_training:
            from .fast_gpu import fast_linear
            base = fast_linear(x, self.weight, self.quant_input)
        else:
            base = frozen_linear(x, self.weight, acc=self.acc)
        if self.rank:
            base = base + F.linear(F.linear(x.to(device=self.a.device, dtype=torch.float32), self.a), self.b).to(x.device)
        return base.to(x.dtype)

    def merged(self) -> Tensor:
        w = self.weight.float()
        return w + self.b @ self.a if self.rank else w


class PackedLinear(nn.Module):
    weight: Tensor
    scale: Tensor

    def __init__(self, weight: Tensor, scale: Tensor, acc: torch.dtype):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("scale", scale)
        self.acc = acc

    def forward(self, x: Tensor) -> Tensor:
        return frozen_linear(qdq(x), self.weight, self.scale, self.acc)


def rms(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    z = x.float()
    z = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + eps)
    return (weight * z).to(x.dtype)


def rope(x: Tensor, positions: Tensor, rd: int, theta: float, inverse: bool = False) -> Tensor:
    # Pure sliding-window layers explicitly disable YaRN in official model.py.
    freq = 1 / theta ** (torch.arange(0, rd, 2, device=x.device, dtype=torch.float32) / rd)
    angles = positions.float().unsqueeze(-1) * freq
    phase = torch.polar(torch.ones_like(angles), -angles if inverse else angles)
    if x.ndim == 4:
        phase = phase.unsqueeze(-2)
    tail = torch.view_as_complex(x[..., -rd:].float().reshape(*x.shape[:-1], rd//2, 2).contiguous())
    rotated = torch.view_as_real(tail * phase).flatten(-2).to(x.dtype)
    return torch.cat((x[..., :-rd], rotated), -1)


def mixes(x: Tensor, fn: Tensor, scale: Tensor, base: Tensor, hc: int, iters: int,
          norm_eps: float, eps: float, acc: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
    z = x.flatten(-2).float()
    z = F.linear(z, fn) * torch.rsqrt(z.square().mean(-1, keepdim=True) + norm_eps)
    z, scale, base = z.to(acc), scale.to(acc), base.to(acc)
    pre = torch.sigmoid(z[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(z[..., hc:2*hc] * scale[1] + base[hc:2*hc])
    comb = (z[..., 2*hc:] * scale[2] + base[2*hc:]).reshape(*z.shape[:-1], hc, hc)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre.float(), post.float(), comb.float()


def collapse(x: Tensor, pre: Tensor) -> Tensor:
    return (pre.unsqueeze(-1) * x.float()).sum(-2).to(x.dtype)


def expand(x: Tensor, residual: Tensor, post: Tensor, comb: Tensor) -> Tensor:
    return (post.unsqueeze(-1) * x.unsqueeze(-2)
            + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(-3)).to(x.dtype)
