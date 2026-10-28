"""Lane E fusions of the draft step's small replicated chains (used by tp2.py at tp = 1 and 2; O2's fused.py is
untouched). Each has a plain-torch twin it is tested against (tests/test_fused_e.py):

  route_fused       gate logits (split-K fp32 dot, fixed order) -> sqrt(softplus) -> + bias -> top-k -> renormalized
                    weights x route_scale -> logical ids, physical slots (perm), fp16 weights: 2 launches instead of
                    ~10 per stage (cuBLAS sgemm + split-K reduce, softplus, sqrt, add, sort, topk, gather, sum, div,
                    mul, index, cast). The logits differ from cuBLAS in rounding only (a different summation order),
                    so a routing decision can differ only at a 3rd / 4th near-tie (the class O2 already measured
                    between the 3090 and the engine); ties of equal scores go to the lower expert id.
  swiglu_fp16       silu(min(g, lim)) * clamp(u, -lim, lim) in fp32 -> fp16 (torch's silu formula x / (1 + exp(-x))):
                    1 launch instead of 5
  collapse_norm2    fused.collapse_norm that also stores the fp16 conversion of its bf16 output (the next GEMM's input)
  rms_norm_fp16     fused.rms_norm storing fp16(bf16 result) (wq_a -> q_norm -> wq_b)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# --------------------------------------------------------------------------------------------- routing

_ROUTE_SPLITS = 80


@triton.jit
def _route_partial_kernel(x_ptr, w_ptr, part_ptr, T, H, CH, NS, E: tl.constexpr, BK: tl.constexpr):
    """Split sp: partial logits of all T tokens over columns [sp * CH, (sp + 1) * CH) (CH == BK): the gate tile is
    loaded once (coalesced rows) and kept in registers for every token; fp32 products, fixed-order sums."""
    sp = tl.program_id(0)
    e = tl.arange(0, E)
    cols = sp * CH + tl.arange(0, BK)
    w = tl.load(w_ptr + e[:, None].to(tl.int64) * H + cols[None, :])
    for t in range(T):
        x = tl.load(x_ptr + t * H + cols).to(tl.float32)
        tl.store(part_ptr + (t * NS + sp) * E + e, tl.sum(w * x[None, :], axis=1))


@triton.jit
def _route_finish_kernel(part_ptr, bias_ptr, perm_ptr, ids_ptr, sel_ptr, w32_ptr, w16_ptr, NS, scale,
                         E: tl.constexpr, TOPK: tl.constexpr, NSP: tl.constexpr, KP: tl.constexpr):
    t = tl.program_id(0)
    e = tl.arange(0, E)
    sidx = tl.arange(0, NSP)
    p = tl.load(part_ptr + (t * NS + sidx[:, None]) * E + e[None, :], mask=sidx[:, None] < NS, other=0.0)
    logits = tl.sum(p, axis=0)
    sp = tl.where(logits > 20.0, logits, libdevice.log1p(libdevice.exp(logits)))     # torch softplus, threshold 20
    scores = tl.sqrt_rn(sp)
    sel_s = scores + tl.load(bias_ptr + e)
    k = tl.arange(0, KP)
    ids = tl.zeros([KP], tl.int32)
    ws = tl.zeros([KP], tl.float32)
    for j in tl.static_range(TOPK):
        m = tl.max(sel_s, axis=0)
        i = tl.min(tl.where(sel_s == m, e, E), axis=0)                 # lowest id among equal scores
        ids = tl.where(k == j, i, ids)
        ws = tl.where(k == j, tl.sum(tl.where(e == i, scores, 0.0), axis=0), ws)
        sel_s = tl.where(e == i, float("-inf"), sel_s)
    kmask = k < TOPK
    w0 = tl.sum(tl.where(k == 0, ws, 0.0), axis=0)
    w1 = tl.sum(tl.where(k == 1, ws, 0.0), axis=0)
    w2 = tl.sum(tl.where(k == 2, ws, 0.0), axis=0)
    denom = (w0 + w1) + w2 + 1e-20
    wn = ws / denom * scale
    phys = tl.load(perm_ptr + ids, mask=kmask, other=0)
    tl.store(ids_ptr + t * TOPK + k, ids.to(tl.int64), mask=kmask)
    tl.store(sel_ptr + t * TOPK + k, phys, mask=kmask)
    tl.store(w32_ptr + t * TOPK + k, wn, mask=kmask)
    tl.store(w16_ptr + t * TOPK + k, wn.to(tl.float16), mask=kmask)


def route_fused(x: torch.Tensor, gate_w: torch.Tensor, bias: torch.Tensor, perm: torch.Tensor, top_k: int,
                route_scale: float):
    """x bf16 [T, H], gate_w fp32 [E, H], bias fp32 [E], perm int64 [E] (logical -> physical slot).
    Returns (ids int64 [T, k], sel int64 [T, k] = perm[ids], w fp32 [T, k], w fp16 [T, k])."""
    T, H = x.shape
    E = gate_w.shape[0]
    assert top_k == 3 and x.is_contiguous() and gate_w.is_contiguous() and E == triton.next_power_of_2(E)
    ns = _ROUTE_SPLITS
    assert H % ns == 0 and (H // ns) == triton.next_power_of_2(H // ns)
    dev = x.device
    part = torch.empty(T, ns, E, dtype=torch.float32, device=dev)
    _route_partial_kernel[(ns,)](x, gate_w, part, T, H, H // ns, ns, E=E, BK=H // ns, num_warps=4)
    ids = torch.empty(T, top_k, dtype=torch.int64, device=dev)
    sel = torch.empty(T, top_k, dtype=torch.int64, device=dev)
    w32 = torch.empty(T, top_k, dtype=torch.float32, device=dev)
    w16 = torch.empty(T, top_k, dtype=torch.float16, device=dev)
    _route_finish_kernel[(T,)](part, bias, perm, ids, sel, w32, w16, ns, float(route_scale), E=E, TOPK=top_k,
                               NSP=triton.next_power_of_2(ns), KP=4, num_warps=4, enable_fp_fusion=False)
    return ids, sel, w32, w16


# --------------------------------------------------------------------------------------------- SwiGLU


@triton.jit
def _swiglu_kernel(g_ptr, u_ptr, out_ptr, N, lim, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    g = tl.minimum(tl.load(g_ptr + offs, mask=m, other=0.0), lim)
    u = tl.minimum(tl.maximum(tl.load(u_ptr + offs, mask=m, other=0.0), -lim), lim)
    a = g / (1.0 + libdevice.exp(-g)) * u
    tl.store(out_ptr + offs, a.to(tl.float16), mask=m)


def swiglu_fp16(g: torch.Tensor, u: torch.Tensor, lim: float) -> torch.Tensor:
    """fp16(silu(g.clamp(max=lim)) * u.clamp(-lim, lim)) for fp32 g, u (contiguous, same shape)."""
    g, u = g.contiguous(), u.contiguous()
    out = torch.empty(g.shape, dtype=torch.float16, device=g.device)
    N = g.numel()
    _swiglu_kernel[(triton.cdiv(N, 1024),)](g, u, out, N, float(lim), BLOCK=1024, num_warps=4,
                                            enable_fp_fusion=False)
    return out


# --------------------------------------------------------------------------------------------- norms with fp16 out


@triton.jit
def _collapse_norm2_kernel(s_ptr, pre_ptr, w_ptr, out_ptr, out16_ptr, H, eps, HAS_PRE: tl.constexpr,
                           BH: tl.constexpr):
    t = tl.program_id(0)
    base = s_ptr + t.to(tl.int64) * 4 * H
    ss = tl.zeros([BH], tl.float32)
    if HAS_PRE:
        p0 = tl.load(pre_ptr + t * 4)
        p1 = tl.load(pre_ptr + t * 4 + 1)
        p2 = tl.load(pre_ptr + t * 4 + 2)
        p3 = tl.load(pre_ptr + t * 4 + 3)
    for h0 in range(0, H, BH):
        h = h0 + tl.arange(0, BH)
        if HAS_PRE:
            y = (p0 * tl.load(base + h).to(tl.float32) + p1 * tl.load(base + H + h).to(tl.float32)
                 + p2 * tl.load(base + 2 * H + h).to(tl.float32) + p3 * tl.load(base + 3 * H + h).to(tl.float32))
            yb = y.to(tl.bfloat16)
        else:
            yb = tl.load(base + h)
        yf = yb.to(tl.float32)
        ss += yf * yf
        tl.store(out_ptr + t.to(tl.int64) * H + h, yb)
    r = 1.0 / tl.sqrt(tl.sum(ss, axis=0) / H + eps)
    for h0 in range(0, H, BH):
        h = h0 + tl.arange(0, BH)
        yf = tl.load(out_ptr + t.to(tl.int64) * H + h).to(tl.float32)
        w = tl.load(w_ptr + h).to(tl.float32)
        ob = (w * (yf * r)).to(tl.bfloat16)
        tl.store(out_ptr + t.to(tl.int64) * H + h, ob)
        tl.store(out16_ptr + t.to(tl.int64) * H + h, ob.to(tl.float32).to(tl.float16))


def collapse_norm2(stream: torch.Tensor, pre: torch.Tensor | None, w: torch.Tensor, eps: float):
    """(fused.collapse_norm(stream, pre, w, eps), its fp16 conversion): bf16 [T, H] and fp16 [T, H]."""
    T, hc, H = stream.shape
    assert hc == 4 and stream.is_contiguous()
    out = torch.empty(T, H, dtype=torch.bfloat16, device=stream.device)
    out16 = torch.empty(T, H, dtype=torch.float16, device=stream.device)
    _collapse_norm2_kernel[(T,)](stream, pre if pre is not None else stream, w, out, out16, H, eps,
                                 HAS_PRE=pre is not None, BH=1024, num_warps=4, enable_fp_fusion=False)
    return out, out16


@triton.jit
def _rms_norm_fp16_kernel(x_ptr, w_ptr, out_ptr, D, eps, BD: tl.constexpr):
    r_ = tl.program_id(0).to(tl.int64)
    ss = tl.zeros([BD], tl.float32)
    for d0 in range(0, D, BD):
        d = d0 + tl.arange(0, BD)
        m = d < D
        x = tl.load(x_ptr + r_ * D + d, mask=m, other=0.0).to(tl.bfloat16).to(tl.float32)
        ss += x * x
    r = 1.0 / tl.sqrt(tl.sum(ss, axis=0) / D + eps)
    for d0 in range(0, D, BD):
        d = d0 + tl.arange(0, BD)
        m = d < D
        x = tl.load(x_ptr + r_ * D + d, mask=m, other=0.0).to(tl.bfloat16).to(tl.float32)
        w = tl.load(w_ptr + d, mask=m, other=0.0).to(tl.float32)
        tl.store(out_ptr + r_ * D + d, (w * (x * r)).to(tl.bfloat16).to(tl.float32).to(tl.float16), mask=m)


def rms_norm_fp16(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """fused.rms_norm(x, w, eps).to(float16) for fp32 or bf16 x [N, D] (rounded to bf16 first, like rms_norm)."""
    x = x.contiguous()
    N, D = x.shape
    out = torch.empty(N, D, dtype=torch.float16, device=x.device)
    _rms_norm_fp16_kernel[(N,)](x, w, out, D, eps, BD=1024 if D >= 1024 else 512, num_warps=4,
                                enable_fp_fusion=False)
    return out
