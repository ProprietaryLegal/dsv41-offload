"""Standalone native packed MXFP4 expert kernels (experimental; no worker dispatch)."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import triton as tr

from . import kernels as k
from .reference import admit


def _cuda(t: torch.Tensor, device: torch.device, dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    if device.type != 'cuda' or device.index is None:
        raise ValueError('kernel arguments require an explicit indexed CUDA device')
    if t.device != device or t.dtype != dtype or tuple(t.shape) != shape or not t.is_contiguous():
        raise ValueError(f'expected contiguous {dtype} {shape} on {device}, got {t.dtype} {tuple(t.shape)} {t.device}')


@dataclass(frozen=True)
class PackedExperts:
    codes: torch.Tensor  # [E,K/2,N]
    scales: torch.Tensor  # [E,K/32,N]
    experts: int
    outputs: int
    inputs: int

    @classmethod
    def from_checkpoint(cls, codes: torch.Tensor, scales: torch.Tensor,
                        device: torch.device) -> PackedExperts:
        """CPU validation and lossless transpose. Call outside graph capture.

        Input tensors must already be the desired channel shard. Native values
        outside finite normal BF16 are refused, not clipped or rounded.
        """
        admit(codes, scales)
        if device.type != 'cuda' or device.index is None:
            raise ValueError('explicit indexed CUDA device required')
        if torch.cuda.get_device_capability(device) != (8, 6):
            raise ValueError('this kernel is qualified only for SM86')
        e, n, half_k = codes.shape
        return cls(codes.transpose(1, 2).contiguous().to(device),
                   scales.transpose(1, 2).contiguous().to(device), e, n, half_k*2)

    def linear_into(self, x: torch.Tensor, ids: torch.Tensor, out: torch.Tensor,
                    status: torch.Tensor, *, per_slot: bool = False,
                    up: PackedExperts | None = None, up_out: torch.Tensor | None = None,
                    mode: str = 'mma') -> None:
        """Graph-safe; prewarm each shape/mode before capture. No allocation.

        Caller zeros status before a pipeline and checks it after completion:
        invalid device IDs set status bit 0 and write NaNs (never OOB/zero fallback).
        Inputs/outputs/weights/status must be disjoint and retained until completion.
        """
        if mode not in ('mma', 'simt'):
            raise ValueError('mode must be mma or simt')
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= 3 or ids.shape[0] < 1:
            raise ValueError('IDs must be nonempty [rows,top_k<=3]')
        rows, top = ids.shape
        d = self.codes.device
        _cuda(ids, d, torch.int32, (rows, top))
        _cuda(x, d, torch.bfloat16, (rows*top if per_slot else rows, self.inputs))
        _cuda(out, d, torch.float32, (rows, top, self.outputs))
        _cuda(status, d, torch.int32, (1,))
        _cuda(self.codes, d, torch.uint8, (self.experts, self.inputs//2, self.outputs))
        _cuda(self.scales, d, torch.uint8, (self.experts, self.inputs//32, self.outputs))
        if up is not None:
            if (up.experts, up.inputs, up.outputs) != (self.experts, self.inputs, self.outputs):
                raise ValueError('gate and up banks must have identical dimensions')
            _cuda(up.codes, d, torch.uint8, tuple(self.codes.shape))
            _cuda(up.scales, d, torch.uint8, tuple(self.scales.shape))
            if up_out is None:
                raise ValueError('fused gate/up requires preallocated up_out')
            _cuda(up_out, d, torch.float32, tuple(out.shape))
        elif up_out is not None:
            raise ValueError('up_out supplied without an up bank')
        with torch.cuda.device(d):
            k.linear_kernel[(tr.cdiv(self.outputs, 32), rows*top)](
                x, ids, self.codes, self.scales,
                self.codes if up is None else up.codes, self.scales if up is None else up.scales,
                out, out if up_out is None else up_out, status,
                self.outputs, self.inputs, self.experts, top, per_slot, up is not None,
                mode == 'mma', 32, 64, num_warps=4, enable_fp_fusion=False)


def activation_into(gate: torch.Tensor, up: torch.Tensor, weights: torch.Tensor,
                    out: torch.Tensor, limit: float) -> None:
    """BF16 projection rounding, clamp/SwiGLU/routing, BF16 down input.

    Official activation QDQ before the down GEMM remains the caller's responsibility.
    Use unit weights for combine_into when routing was applied here.
    """
    if gate.ndim != 3 or not math.isfinite(limit) or limit <= 0:
        raise ValueError('expected [rows,top,N] intermediates and finite positive clamp')
    _cuda(gate, gate.device, torch.float32, tuple(gate.shape))
    _cuda(up, gate.device, torch.float32, tuple(gate.shape))
    _cuda(weights, gate.device, torch.float32, (gate.shape[0], gate.shape[1]))
    _cuda(out, gate.device, torch.bfloat16, (gate.shape[0]*gate.shape[1], gate.shape[2]))
    with torch.cuda.device(gate.device):
        k.activation_kernel[(tr.cdiv(gate.numel(), 256),)](
            gate, up, weights, out, gate.numel(), gate.shape[2], limit, 256, enable_fp_fusion=False)


def combine_into(parts: torch.Tensor, weights: torch.Tensor, out: torch.Tensor) -> None:
    """Deterministic FP32 sum in top-k slot order; no atomics or BF16 combine."""
    if parts.ndim != 3 or not 1 <= parts.shape[1] <= 3:
        raise ValueError('parts must be [rows,top_k<=3,N]')
    rows, top, n = parts.shape
    _cuda(parts, parts.device, torch.float32, tuple(parts.shape))
    _cuda(weights, parts.device, torch.float32, (rows, top))
    _cuda(out, parts.device, torch.float32, (rows, n))
    with torch.cuda.device(parts.device):
        k.combine_kernel[(tr.cdiv(n, 256), rows)](parts, weights, out, n, top, 256, enable_fp_fusion=False)


def combine_after_tp_into(parts: torch.Tensor, ids: torch.Tensor, out: torch.Tensor) -> None:
    """Native expert-loop ordering: TP-reduced w2 -> BF16 -> FP32 ascending-ID sum.

    Routing must have been applied before down-input QDQ. Never round a rank's
    channel partial here before the TP per-route reduction is complete.
    """
    if parts.ndim != 3 or not 1 <= parts.shape[1] <= 3:
        raise ValueError('parts must be [rows,top_k<=3,N]')
    rows, top, n = parts.shape
    _cuda(parts, parts.device, torch.float32, tuple(parts.shape))
    _cuda(ids, parts.device, torch.int32, (rows, top))
    _cuda(out, parts.device, torch.float32, (rows, n))
    with torch.cuda.device(parts.device):
        k.native_combine_kernel[(tr.cdiv(n, 256), rows)](parts, ids, out, n, top, 256, enable_fp_fusion=False)
