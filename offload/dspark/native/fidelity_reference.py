"""CPU arithmetic oracles transcribed from pinned official 1017 inference.

FP64 mode changes dot accumulation only; softmax state and rounding boundaries
remain FP32/BF16, just as in V1's mirror64 control. No checkpoint is required.
"""
from __future__ import annotations

import torch


def online_attention(q: torch.Tensor, kv: torch.Tensor, sink: torch.Tensor,
                     indices: torch.Tensor, scale: float, *, fp64: bool = False) -> torch.Tensor:
    """Official kernel.py:311–389, including invalid indices and sink last."""
    if q.device.type != "cpu" or kv.device.type != "cpu":
        raise ValueError("CPU oracle requires CPU tensors")
    b, m, h, d = q.shape
    valid = indices >= 0
    gathered = kv[torch.arange(b).view(b, 1, 1), indices.long().clamp_min(0)]
    gathered = torch.where(valid[..., None], gathered, 0).float()
    mx = torch.full((b, m, h), -1e30, dtype=torch.float32)
    den = torch.zeros_like(mx)
    out = torch.zeros(b, m, h, d, dtype=torch.float32)
    acc_dtype = torch.float64 if fp64 else torch.float32
    for start in range(0, indices.shape[-1], 64):
        v = gathered[:, :, start:start + 64]
        scores = torch.einsum("bmhd,bmtd->bmht", q.to(acc_dtype), v.to(acc_dtype)).float()
        scores = scores.masked_fill(~valid[:, :, None, start:start + 64], float("-inf")) * scale
        new_mx = torch.maximum(mx, scores.amax(-1))
        alpha = torch.exp(mx - new_mx)
        p = torch.exp(scores - new_mx[..., None])
        den = den * alpha + p.sum(-1)
        # GEMM accumulates into the scaled old accumulator, not a separately
        # rounded block result followed by an addition.
        out = out * alpha[..., None]
        out = (out.to(acc_dtype) + torch.einsum("bmht,bmtd->bmhd",
               p.bfloat16().to(acc_dtype), v.to(acc_dtype))).float()
        mx = new_mx
    den = den + torch.exp(sink.float().view(1, 1, h) - mx)
    return (out / den[..., None]).to(q.dtype)


def head_logits(hidden: torch.Tensor, weight: torch.Tensor, *, fp64: bool = False) -> torch.Tensor:
    """ParallelHead: FP32 inputs/weights/output; FP64 dot is an error oracle."""
    dt = torch.float64 if fp64 else torch.float32
    return (hidden.to(dt) @ weight.to(dt).T).float()


def markov_logits(base: torch.Tensor, prev: torch.Tensor, embed: torch.Tensor,
                  weight: torch.Tensor, *, fp64: bool = False) -> torch.Tensor:
    """DSparkBlock.forward_head: FP32 Markov bias added to FP32 base."""
    return base.float() + head_logits(embed[prev], weight, fp64=fp64)
