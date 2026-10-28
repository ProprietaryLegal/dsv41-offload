"""Reference ops with the OFFICIAL 1017 semantics (inference/model.py + kernel.py @ 1bae0e10), device-agnostic torch.

These are the oracles for N1's loader / integration and for the KX kernels (NATIVE.md "Kernel contract"). They are
written for clarity and exactness, not speed. Accumulation: ``acc="f64"`` sums exact products in float64 and rounds
once (the real-number result rounded to the output dtype: the best any kernel can do); ``acc="blocked"`` is the
official kernels' order: each 32-wide K block summed (float64, then fp32, standing in for the MMA's fp32 block sum),
scaled, and accumulated in fp32 block by block (kernel.fp8_gemm / fp4_gemm C_local_accum). A correct kernel lands
within fp32 summation-order noise of both; anything further is a bug.

Rounding points the official model has and every native path must keep (NATIVE.md):
  * every FP8/FP4 linear quantizes its input with act_qdq (per 32, power-of-two scale, e4m3) and returns bf16;
  * Expert: gate = bf16(w1 x).float(), up = bf16(w3 x).float(); up clamped to [-10, 10], gate to <= 10;
    h = silu(gate) * up (fp32), h = route_w * h (fp32), bf16(h) -> act_qdq -> w2 -> bf16 per expert;
  * MoE: y = 0 (fp32); experts in ASCENDING expert id: y[rows] += bf16 expert output; y += bf16 shared output;
    out = bf16(y);
  * Gate: scores = fp32(x) @ fp32(W)^T (no TF32) -> sqrt(softplus); top-3 of (scores + bias) with the fp32 bias;
    weights = scores at those ids / (sum + 1e-20) * 1.5 (bias only selects).
"""
from __future__ import annotations

import torch
import torch.nn.functional as Fn

from . import formats as F

SWIGLU_LIMIT = 10.0
ROUTE_SCALE = 1.5
TOPK = 3


def _matmul_acc(a: torch.Tensor, w: torch.Tensor, sa: torch.Tensor | None, sw: torch.Tensor | None, acc: str
                ) -> torch.Tensor:
    """sum_k a[m,k] * w[n,k] with optional per-32 scales already folded (sa/sw None) or blocked (acc='blocked':
    a, w are CODES, sa [M, K/32], sw [N, K/32] the per-block scales). Returns fp32 [M, N]."""
    if acc == "f64":
        return (a.double() @ w.double().t()).float()
    if acc != "blocked":
        raise ValueError(acc)
    M, K = a.shape
    N = w.shape[0]
    nb = K // F.BLOCK
    ab = a.double().view(M, nb, F.BLOCK)
    wb = w.double().view(N, nb, F.BLOCK)
    out = torch.zeros(M, N, dtype=torch.float32, device=a.device)
    for b in range(nb):
        part = (ab[:, b] @ wb[:, b].t()).float()                     # exact block sum, rounded to fp32
        if sa is not None:
            part = part * sa[:, b:b + 1] * sw[:, b].unsqueeze(0)     # (C * sa) * sb, powers of two: exact
        out += part
    return out


def linear_fp4(x: torch.Tensor, w_packed: torch.Tensor, w_scale: torch.Tensor, acc: str = "f64") -> torch.Tensor:
    """model.linear for a float4_e2m1fn_x2 weight: act_quant(x) then fp4_gemm; bf16 [M, N]."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if acc == "blocked":
        q, s = F.act_quant(x2)
        codes = F._fp4_lut(x.device)[F.fp4_codes(w_packed).long()]
        y = _matmul_acc(q, codes, s, F.e8m0_to_f32(w_scale), acc)
    else:
        y = _matmul_acc(F.act_qdq(x2).float(), F.decode_fp4(w_packed, w_scale), None, None, acc)
    return y.to(torch.bfloat16).view(*shape[:-1], -1)


def linear_fp8(x: torch.Tensor, w_u8: torch.Tensor, w_scale: torch.Tensor, acc: str = "f64",
               rows: tuple[int, int] | None = None, cols: tuple[int, int] | None = None) -> torch.Tensor:
    """model.linear for a float8_e4m3fn weight (32x32 E8M0 blocks): act_quant(x) then fp8_gemm; bf16 [M, N].
    ``rows`` / ``cols``: the window of the full tensor ``w_u8`` holds (w_scale is always the full tensor's)."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if acc == "blocked":
        q, s = F.act_quant(x2)
        r0 = rows[0] if rows else 0
        c0 = cols[0] if cols else 0
        N, K = w_u8.shape
        sw = F.e8m0_to_f32(w_scale)[r0 // 32:(r0 + N + 31) // 32, c0 // 32:(c0 + K + 31) // 32]
        sw = sw.repeat_interleave(32, 0)[:N]
        y = _matmul_acc(q, F.e4m3_codes_to_f32(w_u8), s, sw, acc)
    else:
        y = _matmul_acc(F.act_qdq(x2).float(), F.decode_fp8_block(w_u8, w_scale, rows, cols), None, None, acc)
    return y.to(torch.bfloat16).view(*shape[:-1], -1)


def linear_bf16(x: torch.Tensor, w: torch.Tensor, acc: str = "f64") -> torch.Tensor:
    """F.linear(x, w) for bf16 x, w: fp32 accumulation, bf16 out."""
    shape = x.shape
    y = _matmul_acc(x.reshape(-1, shape[-1]).float(), w.float(), None, None, "f64" if acc == "blocked" else acc)
    return y.to(torch.bfloat16).view(*shape[:-1], -1)


def swiglu(g16: torch.Tensor, u16: torch.Tensor, route_w: torch.Tensor | None, limit: float = SWIGLU_LIMIT
           ) -> torch.Tensor:
    """Expert.forward between w1/w3 and w2: bf16 in (the linears' outputs), bf16 out (x.to(dtype))."""
    gate = g16.float()
    up = u16.float()
    if limit > 0:
        up = torch.clamp(up, min=-limit, max=limit)
        gate = torch.clamp(gate, max=limit)
    h = Fn.silu(gate) * up
    if route_w is not None:
        h = route_w * h
    return h.to(torch.bfloat16)


def expert(x: torch.Tensor, w1, s1, w3, s3, w2, s2, route_w: torch.Tensor | None, fmt: str = "fp4",
           acc: str = "f64") -> torch.Tensor:
    """One official Expert on rows x (bf16 [M, 5120]); route_w fp32 [M, 1] or None (shared expert). bf16 out."""
    lin = linear_fp4 if fmt == "fp4" else linear_fp8
    h = swiglu(lin(x, w1, s1, acc), lin(x, w3, s3, acc), route_w)
    return lin(h, w2, s2, acc)


def gate(x: torch.Tensor, gate_w: torch.Tensor, bias: torch.Tensor, topk: int = TOPK,
         route_scale: float = ROUTE_SCALE) -> tuple[torch.Tensor, torch.Tensor]:
    """Gate.forward (text tokens): (weights fp32 [T, k], indices int64 [T, k])."""
    scores = (x.float().double() @ gate_w.float().double().t()).float()       # F.linear fp32, no TF32
    scores = Fn.softplus(scores).sqrt()
    idx = (scores + bias.float()).topk(topk, dim=-1)[1]
    w = scores.gather(1, idx)
    if topk > 1:
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return w * route_scale, idx


def moe(x: torch.Tensor, gate_w: torch.Tensor, bias: torch.Tensor, get_expert, shared: tuple, acc: str = "f64"
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MoE.forward, world_size 1. get_expert(e) -> (w1, s1, w3, s3, w2, s2) packed FP4; shared = the FP8 shared
    expert's (w1, s1, w3, s3, w2, s2). Returns (bf16 out [T, 5120], weights, indices)."""
    w, idx = gate(x, gate_w, bias)
    y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for e in sorted(set(idx.flatten().tolist())):
        rows, slot = torch.where(idx == e)
        y[rows] += expert(x[rows], *get_expert(e), w[rows, slot, None], "fp4", acc).float()
    y += expert(x, *shared, None, "fp8", acc).float()
    return y.to(x.dtype), w, idx
