"""Allocation-free shared-expert SwiGLU and per-32 activation QDQ.

The arithmetic is native.ops._swiglu_qdq_kernel, with explicit FP32 widening
of BF16 inputs and row strides (also supports paired gate/up projections).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...native.ops import INTERP, _bf16_rne, _e4m3_qdq_group, _exp


@triton.jit
def _shared_activation(G, U, R, O, I: tl.constexpr, GS: tl.constexpr, US: tl.constexpr,
                       INTERPRET: tl.constexpr):
    row = tl.program_id(0)
    channel = tl.program_id(1)*64 + tl.arange(0, 64)
    gate = tl.minimum(tl.load(G + row*GS + channel).to(tl.float32), 10.)
    up = tl.minimum(tl.maximum(tl.load(U + row*US + channel).to(tl.float32), -10.), 10.)
    weight = tl.load(R + row)
    hidden = _bf16_rne(weight * (gate / (1. + _exp(-gate, INTERPRET)) * up))
    quantized = _e4m3_qdq_group(tl.reshape(hidden, (2, 32)), INTERPRET)
    tl.store(O + row*I + channel, tl.reshape(quantized, (64,)).to(tl.bfloat16))


def shared_activation_into(gate: torch.Tensor, up: torch.Tensor,
                           ones: torch.Tensor, out: torch.Tensor) -> None:
    rows, channels = gate.shape
    if gate.dtype != torch.bfloat16 or up.shape != gate.shape or up.dtype != gate.dtype or channels % 64:
        raise ValueError('BF16 gate/up matrices, identical shape and 64-aligned width required')
    if ones.shape != (rows, 1) or ones.dtype != torch.float32 or out.shape != gate.shape or out.dtype != gate.dtype:
        raise ValueError('caller-owned FP32 ones [M,1] and BF16 output [M,I] required')
    if gate.device != up.device or gate.device != ones.device or gate.device != out.device:
        raise ValueError('all shared activation buffers must be on the same device')
    if not out.is_contiguous() or not ones.is_contiguous() or gate.stride(1) != 1 or up.stride(1) != 1:
        raise ValueError('shared columns/output/ones must be contiguous')
    if min(gate.stride(0), up.stride(0)) < channels:
        raise ValueError('shared input rows must not overlap')
    if out.untyped_storage().data_ptr() in (gate.untyped_storage().data_ptr(), up.untyped_storage().data_ptr(),
                                          ones.untyped_storage().data_ptr()):
        raise ValueError('shared output must not alias projection inputs or route weights')
    # Ones are initialized at construction; never inspect a device value here.
    _shared_activation[(rows, channels//64)](gate, up, ones, out, channels,
                                            gate.stride(0), up.stride(0), INTERP,
                                            num_warps=2, enable_fp_fusion=False)
