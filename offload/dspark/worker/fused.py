"""Fused Triton kernels for the draft step's elementwise chains (SM86). Each one has a plain-torch twin in ops.py and a
test (tests/test_fused.py) that holds it to that twin: bit-identical where the math allows, else <= 1 bf16 step.

Unfused, the step ran ~1,400 kernels (the 20-iteration Sinkhorn alone ~480): ~2 ms of launch-bound work per round.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .ops import _markov_bias_kernel  # noqa: F401  (the serve's kernel; markov_argmax repeats its arithmetic)

# --------------------------------------------------------------------------------------------- mHC mixes + Sinkhorn


@triton.jit
def _hc_mix_partial_kernel(x_ptr, fn_ptr, part_ptr, K, CH, NS, BK: tl.constexpr):
    """Split-K partial dot products of one token's flattened stream with the 24 mix rows, plus its sum of squares."""
    t = tl.program_id(0)
    sp = tl.program_id(1)
    r32 = tl.arange(0, 32)
    rmask = r32 < 24
    acc = tl.zeros([32], tl.float32)
    ss = tl.zeros([BK], tl.float32)
    for k0 in range(sp * CH, sp * CH + CH, BK):
        cols = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + t.to(tl.int64) * K + cols).to(tl.float32)
        ss += x * x
        w = tl.load(fn_ptr + r32[:, None] * K + cols[None, :], mask=rmask[:, None], other=0.0)
        acc += tl.sum(w * x[None, :], axis=1)
    acc = tl.where(r32 == 24, tl.sum(ss, axis=0), acc)
    tl.store(part_ptr + (t * NS + sp) * 32 + r32, acc)


@triton.jit
def _hc_mix_finish_kernel(part_ptr, scale_ptr, base_ptr, pre_ptr, post_ptr, comb_ptr, K, NS, rms_eps, hc_eps,
                          ITERS: tl.constexpr, NSP: tl.constexpr):
    t = tl.program_id(0)
    sidx = tl.arange(0, NSP)
    smask = sidx < NS
    r4 = tl.arange(0, 4)
    r16 = tl.arange(0, 16)
    rows = part_ptr + (t * NS + sidx) * 32
    acc_pre = tl.sum(tl.load(rows[:, None] + r4[None, :], mask=smask[:, None], other=0.0), axis=0)
    acc_post = tl.sum(tl.load(rows[:, None] + 4 + r4[None, :], mask=smask[:, None], other=0.0), axis=0)
    acc_comb = tl.sum(tl.load(rows[:, None] + 8 + r16[None, :], mask=smask[:, None], other=0.0), axis=0)
    ss = tl.sum(tl.load(rows + 24, mask=smask, other=0.0), axis=0)
    r = 1.0 / tl.sqrt(ss / K + rms_eps)
    s0 = tl.load(scale_ptr)
    s1 = tl.load(scale_ptr + 1)
    s2 = tl.load(scale_ptr + 2)
    pre = 1.0 / (1.0 + tl.exp(-(acc_pre * r * s0 + tl.load(base_ptr + r4)))) + hc_eps
    post = 2.0 / (1.0 + tl.exp(-(acc_post * r * s1 + tl.load(base_ptr + 4 + r4))))
    c = tl.reshape(acc_comb * r * s2 + tl.load(base_ptr + 8 + r16), (4, 4))
    c = tl.exp(c - tl.max(c, axis=1)[:, None])
    c = c / tl.sum(c, axis=1)[:, None] + hc_eps
    c = c / (tl.sum(c, axis=0)[None, :] + hc_eps)
    for _ in range(ITERS - 1):
        c = c / (tl.sum(c, axis=1)[:, None] + hc_eps)
        c = c / (tl.sum(c, axis=0)[None, :] + hc_eps)
    tl.store(pre_ptr + t * 4 + r4, pre)
    tl.store(post_ptr + t * 4 + r4, post)
    tl.store(comb_ptr + t * 16 + r16, tl.reshape(c, (16,)))


_HC_SPLITS = 40          # 20480 / 40 = 512 columns a program: T x 40 programs (a 3090 has 82 SMs)


def hc_mixes(stream: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, rms_eps: float,
             hc_eps: float, iters: int):
    """ops.hc_mixes: split-K GEMV of the flattened stream with the 24 mix rows (fixed-order sums, deterministic),
    then sigmoid / softmax / Sinkhorn per token."""
    T, hc, H = stream.shape
    K = hc * H
    assert hc == 4 and stream.is_contiguous() and fn.is_contiguous() and fn.shape == (24, K)
    ns = _HC_SPLITS
    assert K % (ns * 128) == 0
    part = torch.empty(T, ns, 32, dtype=torch.float32, device=stream.device)
    _hc_mix_partial_kernel[(T, ns)](stream, fn, part, K, K // ns, ns, BK=128, num_warps=4)
    pre = torch.empty(T, 4, dtype=torch.float32, device=stream.device)
    post = torch.empty(T, 4, dtype=torch.float32, device=stream.device)
    comb = torch.empty(T, 4, 4, dtype=torch.float32, device=stream.device)
    _hc_mix_finish_kernel[(T,)](part, scale, base, pre, post, comb, K, ns, rms_eps, hc_eps, ITERS=iters,
                                NSP=triton.next_power_of_2(ns), num_warps=1)
    return pre, post, comb


# --------------------------------------------------------------------------------------------- collapse + RMSNorm


@triton.jit
def _collapse_norm_kernel(s_ptr, pre_ptr, w_ptr, out_ptr, H, eps, HAS_PRE: tl.constexpr, BH: tl.constexpr):
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
        tl.store(out_ptr + t.to(tl.int64) * H + h, (w * (yf * r)).to(tl.bfloat16))


def collapse_norm(stream: torch.Tensor, pre: torch.Tensor | None, w: torch.Tensor, eps: float) -> torch.Tensor:
    """rms_norm(hc_collapse(stream, pre), w): collapse with pre-mix (or copy 0), round to bf16, RMSNorm."""
    T, hc, H = stream.shape
    assert hc == 4 and stream.is_contiguous()
    out = torch.empty(T, H, dtype=torch.bfloat16, device=stream.device)
    _collapse_norm_kernel[(T,)](stream, pre if pre is not None else stream, w, out, H, eps,
                                HAS_PRE=pre is not None, BH=1024, num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit
def _collapse_kernel(s_ptr, pre_ptr, out_ptr, H, BH: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    m = h < H
    base = s_ptr + t.to(tl.int64) * 4 * H
    y = (tl.load(pre_ptr + t * 4) * tl.load(base + h, mask=m).to(tl.float32)
         + tl.load(pre_ptr + t * 4 + 1) * tl.load(base + H + h, mask=m).to(tl.float32)
         + tl.load(pre_ptr + t * 4 + 2) * tl.load(base + 2 * H + h, mask=m).to(tl.float32)
         + tl.load(pre_ptr + t * 4 + 3) * tl.load(base + 3 * H + h, mask=m).to(tl.float32))
    tl.store(out_ptr + t.to(tl.int64) * H + h, y.to(tl.bfloat16), mask=m)


def collapse(stream: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    T, hc, H = stream.shape
    out = torch.empty(T, H, dtype=torch.bfloat16, device=stream.device)
    _collapse_kernel[(T, triton.cdiv(H, 1024))](stream, pre, out, H, BH=1024, num_warps=4, enable_fp_fusion=False)
    return out


# --------------------------------------------------------------------------------------------- mHC post


@triton.jit
def _hc_post_kernel(y_ptr, s_ptr, post_ptr, comb_ptr, out_ptr, H, BH: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    m = h < H
    base = t.to(tl.int64) * 4 * H
    y = tl.load(y_ptr + t.to(tl.int64) * H + h, mask=m).to(tl.float32)
    s0 = tl.load(s_ptr + base + h, mask=m).to(tl.float32)
    s1 = tl.load(s_ptr + base + H + h, mask=m).to(tl.float32)
    s2 = tl.load(s_ptr + base + 2 * H + h, mask=m).to(tl.float32)
    s3 = tl.load(s_ptr + base + 3 * H + h, mask=m).to(tl.float32)
    for j in tl.static_range(4):
        c0 = tl.load(comb_ptr + t * 16 + j)
        c1 = tl.load(comb_ptr + t * 16 + 4 + j)
        c2 = tl.load(comb_ptr + t * 16 + 8 + j)
        c3 = tl.load(comb_ptr + t * 16 + 12 + j)
        v = (c0 * s0 + c1 * s1 + c2 * s2 + c3 * s3) + tl.load(post_ptr + t * 4 + j) * y
        tl.store(out_ptr + base + j * H + h, v.to(tl.bfloat16), mask=m)


def hc_post(y: torch.Tensor, stream: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    T, hc, H = stream.shape
    out = torch.empty_like(stream)
    _hc_post_kernel[(T, triton.cdiv(H, 1024))](y, stream, post, comb, out, H, BH=1024, num_warps=4,
                                               enable_fp_fusion=False)
    return out


# --------------------------------------------------------------------------------------------- RMSNorm


@triton.jit
def _rms_norm_kernel(x_ptr, w_ptr, out_ptr, D, eps, BD: tl.constexpr):
    r_ = tl.program_id(0).to(tl.int64)
    ss = tl.zeros([BD], tl.float32)
    for d0 in range(0, D, BD):
        d = d0 + tl.arange(0, BD)
        m = d < D
        x = tl.load(x_ptr + r_ * D + d, mask=m, other=0.0).to(tl.float32)
        ss += x * x
    r = 1.0 / tl.sqrt(tl.sum(ss, axis=0) / D + eps)
    for d0 in range(0, D, BD):
        d = d0 + tl.arange(0, BD)
        m = d < D
        x = tl.load(x_ptr + r_ * D + d, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + d, mask=m, other=0.0).to(tl.float32)
        tl.store(out_ptr + r_ * D + d, (w * (x * r)).to(tl.bfloat16), mask=m)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """x [N, D] bf16 (or fp32: rounded to bf16 first, as ops.rms_norm(x.to(bf16), ...))."""
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    x = x.contiguous()
    N, D = x.shape
    out = torch.empty_like(x)
    _rms_norm_kernel[(N,)](x, w, out, D, eps, BD=1024 if D >= 1024 else 512, num_warps=4, enable_fp_fusion=False)
    return out


# --------------------------------------------------------------------------------------------- RoPE (+ KV record)


@triton.jit
def _rope_kernel(x_ptr, out_ptr, pos_ptr, cs_ptr, ROWS_PER_TOKEN, D: tl.constexpr, RD: tl.constexpr, INV: tl.constexpr,
                 QDQ: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    tok = row // ROWS_PER_TOKEN
    p = tl.load(pos_ptr + tok)
    NOPE: tl.constexpr = D - RD
    i = tl.arange(0, RD // 2)
    c = tl.load(cs_ptr + p * RD + i)
    s = tl.load(cs_ptr + p * RD + RD // 2 + i)
    if INV:
        s = -s
    x0 = tl.load(x_ptr + row * D + NOPE + 2 * i).to(tl.float32)
    x1 = tl.load(x_ptr + row * D + NOPE + 2 * i + 1).to(tl.float32)
    tl.store(out_ptr + row * D + NOPE + 2 * i, (x0 * c - x1 * s).to(tl.bfloat16))
    tl.store(out_ptr + row * D + NOPE + 2 * i + 1, (x1 * c + x0 * s).to(tl.bfloat16))
    g = tl.arange(0, 64)
    for b in tl.static_range(NOPE // 64):
        v = tl.load(x_ptr + row * D + b * 64 + g)
        if QDQ:
            vf = v.to(tl.float32)
            amax = tl.maximum(tl.max(tl.abs(vf), axis=0) * (1.0 / 448.0), 2.0 ** -126)
            bits = amax.to(tl.uint32, bitcast=True)
            e = ((bits >> 23) & 0xFF).to(tl.int32) - 127 + tl.where((bits & 0x7FFFFF) != 0, 1, 0)
            sc = ((e + 127) << 23).to(tl.uint32).to(tl.float32, bitcast=True)
            q = tl.minimum(tl.maximum(vf / sc, -448.0), 448.0)
            # e4m3 round-to-nearest-even by hand (SM86 has no fp8 conversion): quantum 2^(max(e, -6) - 3)
            qb = q.to(tl.uint32, bitcast=True)
            qe = tl.maximum(((qb >> 23) & 0xFF).to(tl.int32) - 127, -6)
            quant = ((qe - 3 + 127) << 23).to(tl.uint32).to(tl.float32, bitcast=True)
            q = libdevice.rint(q / quant) * quant
            v = (q * sc).to(tl.bfloat16)
        tl.store(out_ptr + row * D + b * 64 + g, v)


def rope(x: torch.Tensor, pos: torch.Tensor, cos_sin: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """ops.apply_rope for bf16 x [N, D] or [N, heads, D] (rotates the last 64 channels)."""
    x = x.contiguous()
    D = x.shape[-1]
    rows = x.numel() // D
    per = rows // x.shape[0]
    out = torch.empty_like(x)
    _rope_kernel[(rows,)](x, out, pos, cos_sin, per, D=D, RD=cos_sin.shape[-1], INV=inverse, QDQ=False,
                          num_warps=1, enable_fp_fusion=False)
    return out


def kv_rope_qdq(kv: torch.Tensor, pos: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
    """ops.kv_record_qdq(ops.apply_rope(kv, pos)) for kv [N, 512] bf16: the V4 record values."""
    kv = kv.contiguous()
    out = torch.empty_like(kv)
    _rope_kernel[(kv.shape[0],)](kv, out, pos, cos_sin, 1, D=kv.shape[-1], RD=cos_sin.shape[-1], INV=False,
                                 QDQ=True, num_warps=1, enable_fp_fusion=False)
    return out


# --------------------------------------------------------------------------------------------- Markov bias + argmax


@triton.jit
def _markov_argmax_kernel(ids_ptr, slot_ptr, cache_ptr, emb_ptr, w2_ptr, base_ptr, vals_ptr, idx_ptr, V,
                          cache_stride, w2_stride, emb_stride, base_stride, NBLK, ids_stride, R: tl.constexpr,
                          BLOCK_V: tl.constexpr, EMB_BY_TOKEN: tl.constexpr):
    b = tl.program_id(0)
    blk = tl.program_id(1)
    cols = blk * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = cols < V
    tok = tl.load(ids_ptr + b * ids_stride).to(tl.int64)
    slot = tl.load(slot_ptr + tok)
    if slot >= 0:
        bias = tl.load(cache_ptr + slot.to(tl.int64) * cache_stride + cols, mask=mask, other=0.0)
    else:
        r = tl.arange(0, R)
        if EMB_BY_TOKEN:        # emb_ptr = the embedding table: row of the previous token
            e = tl.load(emb_ptr + tok * emb_stride + r).to(tl.float32)
        else:
            e = tl.load(emb_ptr + b * emb_stride + r).to(tl.float32)
        w = tl.load(w2_ptr + cols[:, None].to(tl.int64) * w2_stride + r[None, :], mask=mask[:, None], other=0.0)
        bias = tl.sum(w.to(tl.float32) * e[None, :], axis=1).to(tl.bfloat16)
    lg = tl.load(base_ptr + b.to(tl.int64) * base_stride + cols, mask=mask, other=0.0)
    li = (lg.to(tl.float32) + bias.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    li = tl.where(mask, li, float("-inf"))
    v, i = tl.max(li, axis=0, return_indices=True, return_indices_tie_break_left=True)
    tl.store(vals_ptr + b * NBLK + blk, v)
    tl.store(idx_ptr + b * NBLK + blk, (blk * BLOCK_V + i).to(tl.int64))


def markov_argmax(base: torch.Tensor, prev: torch.Tensor, emb: torch.Tensor, mb) -> torch.Tensor:
    """argmax_v bf16(base[b, v] + bias(prev_b)[v]) per row, ties -> lowest id. base [B, V] bf16 (any row stride)."""
    B, V = base.shape
    assert base.stride(1) == 1
    BLOCK_V = 64
    nblk = triton.cdiv(V, BLOCK_V)
    vals = torch.empty(B, nblk, dtype=torch.float32, device=base.device)
    idx = torch.empty(B, nblk, dtype=torch.int64, device=base.device)
    emb = emb.contiguous()
    _markov_argmax_kernel[(B, nblk)](prev, mb.slot_of_token, mb.cache, emb, mb.w2, base, vals, idx, V,
                                     mb.cache.stride(0), mb.w2.stride(0), emb.stride(0), base.stride(0), nblk,
                                     prev.stride(0), R=emb.shape[1], BLOCK_V=BLOCK_V, EMB_BY_TOKEN=False,
                                     num_warps=4)
    best = vals.argmax(dim=-1, keepdim=True)
    return idx.gather(1, best).squeeze(1)


@triton.jit
def _markov_cached_kernel(ids_ptr, slot_ptr, cache_ptr, base_ptr, vals_ptr, idx_ptr, V, cache_stride, base_stride,
                          NBLK, ids_stride, BLOCK_V: tl.constexpr):
    """Cached previous token: block max of bf16(base + cache row) (first maximum = lowest id). Exits otherwise."""
    b = tl.program_id(0)
    blk = tl.program_id(1)
    tok = tl.load(ids_ptr + b * ids_stride).to(tl.int64)
    slot = tl.load(slot_ptr + tok)
    if slot >= 0:
        cols = blk * BLOCK_V + tl.arange(0, BLOCK_V)
        mask = cols < V
        bias = tl.load(cache_ptr + slot.to(tl.int64) * cache_stride + cols, mask=mask, other=0.0)
        lg = tl.load(base_ptr + b.to(tl.int64) * base_stride + cols, mask=mask, other=0.0)
        li = (lg.to(tl.float32) + bias.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        li = tl.where(mask, li, float("-inf"))
        v, i = tl.max(li, axis=0, return_indices=True, return_indices_tie_break_left=True)
        tl.store(vals_ptr + b * NBLK + blk, v)
        tl.store(idx_ptr + b * NBLK + blk, (blk * BLOCK_V + i).to(tl.int64))


@triton.jit
def _markov_uncached_kernel(ids_ptr, slot_ptr, emb_ptr, w2_ptr, base_ptr, vals_ptr, idx_ptr, V, w2_stride,
                            emb_stride, base_stride, NBLK, ids_stride, R: tl.constexpr, BLOCK_V: tl.constexpr,
                            SUB: tl.constexpr):
    """Uncached previous token: the serve's bias arithmetic (_markov_bias_kernel: [BLOCK_V, R] tile, fp32 products,
    one bf16 rounding) over SUB consecutive BLOCK_V-column blocks, running max with strict > (lowest id on ties).
    Exits for cached tokens."""
    b = tl.program_id(0)
    pb = tl.program_id(1)
    tok = tl.load(ids_ptr + b * ids_stride).to(tl.int64)
    slot = tl.load(slot_ptr + tok)
    if slot < 0:
        r = tl.arange(0, R)
        e = tl.load(emb_ptr + tok * emb_stride + r).to(tl.float32)
        best_v = float("-inf")
        best_i = (pb * SUB * BLOCK_V).to(tl.int64)
        for sb in range(SUB):
            cols = (pb * SUB + sb) * BLOCK_V + tl.arange(0, BLOCK_V)
            mask = cols < V
            w = tl.load(w2_ptr + cols[:, None].to(tl.int64) * w2_stride + r[None, :], mask=mask[:, None], other=0.0)
            bias = tl.sum(w.to(tl.float32) * e[None, :], axis=1).to(tl.bfloat16)
            lg = tl.load(base_ptr + b.to(tl.int64) * base_stride + cols, mask=mask, other=0.0)
            li = (lg.to(tl.float32) + bias.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
            li = tl.where(mask, li, float("-inf"))
            v, i = tl.max(li, axis=0, return_indices=True, return_indices_tie_break_left=True)
            take = v > best_v
            best_i = tl.where(take, ((pb * SUB + sb) * BLOCK_V + i).to(tl.int64), best_i)
            best_v = tl.where(take, v, best_v)
        tl.store(vals_ptr + b * NBLK + pb, best_v)
        tl.store(idx_ptr + b * NBLK + pb, best_i)


@triton.jit
def _argmax_finish_kernel(ids_ptr, slot_ptr, vals_c_ptr, idx_c_ptr, vals_u_ptr, idx_u_ptr, out_ptr, NBLK_C, NBLK_U,
                          ids_stride, out_stride, col, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    tok = tl.load(ids_ptr + b * ids_stride).to(tl.int64)
    cached = tl.load(slot_ptr + tok) >= 0
    j = tl.arange(0, BLOCK)
    vc = tl.load(vals_c_ptr + b * NBLK_C + j, mask=j < NBLK_C, other=float("-inf"))
    vu = tl.load(vals_u_ptr + b * NBLK_U + j, mask=j < NBLK_U, other=float("-inf"))
    v = tl.where(cached, vc, vu)
    _, jm = tl.max(v, axis=0, return_indices=True, return_indices_tie_break_left=True)
    if cached:
        d = tl.load(idx_c_ptr + b * NBLK_C + jm)
    else:
        d = tl.load(idx_u_ptr + b * NBLK_U + jm)
    tl.store(out_ptr + b * out_stride + col, d)


def markov_argmax_into(base: torch.Tensor, prev: torch.Tensor, mb, out: torch.Tensor, col: int) -> None:
    """out[:, col] = markov_argmax(base, prev, mb.embed(prev), mb) (same values; ties -> lowest id). Three launches:
    cached previous tokens take a wide-block kernel (2048 columns per program), uncached ones the serve's bias tile
    looped over 8 blocks per program (each kernel exits for the other kind), and a finish kernel reduces the block
    maxima straight into the output column. The Markov embedding row is read by token. prev may be a strided column
    (out[:, col - 1])."""
    B, V = base.shape
    assert base.stride(1) == 1 and out.dtype == torch.int64
    BC, BU, SUB = 2048, 64, 8
    nc, nu = triton.cdiv(V, BC), triton.cdiv(V, BU * SUB)
    vals_c = torch.empty(B, nc, dtype=torch.float32, device=base.device)
    idx_c = torch.empty(B, nc, dtype=torch.int64, device=base.device)
    vals_u = torch.empty(B, nu, dtype=torch.float32, device=base.device)
    idx_u = torch.empty(B, nu, dtype=torch.int64, device=base.device)
    _markov_cached_kernel[(B, nc)](prev, mb.slot_of_token, mb.cache, base, vals_c, idx_c, V, mb.cache.stride(0),
                                   base.stride(0), nc, prev.stride(0), BLOCK_V=BC, num_warps=8)
    _markov_uncached_kernel[(B, nu)](prev, mb.slot_of_token, mb.w1, mb.w2, base, vals_u, idx_u, V, mb.w2.stride(0),
                                     mb.w1.stride(0), base.stride(0), nu, prev.stride(0), R=mb.w1.shape[1],
                                     BLOCK_V=BU, SUB=SUB, num_warps=4)
    _argmax_finish_kernel[(B,)](prev, mb.slot_of_token, vals_c, idx_c, vals_u, idx_u, out, nc, nu, prev.stride(0),
                                out.stride(0), col, BLOCK=triton.next_power_of_2(max(nc, nu)), num_warps=4)


# --------------------------------------------------------------------------------------------- block attention


@triton.jit
def _block_attn_kernel(q_ptr, ring_ptr, ring_pos_ptr, slots_ptr, lo_ptr, last_ptr, blk_ptr, sink_ptr, out_ptr,
                       scale, k, ring_slot_stride, ring_pos_stride, W: tl.constexpr, D: tl.constexpr, HH: tl.constexpr,
                       BM: tl.constexpr, KB: tl.constexpr, BD: tl.constexpr, DO: tl.constexpr):
    """Program (b, m, n): BM query rows (row = t * HH + h) of request b, output dims [n * DO, (n + 1) * DO).
    Scores over the W window ring rows of b's slot (masked to positions in [lo, last]) and its k block rows (padded
    to KB), sink in the denominator, single-pass softmax, probabilities rounded to bf16 for P @ V. Keys = values."""
    b = tl.program_id(0)
    rows = tl.program_id(1) * BM + tl.arange(0, BM)
    d_out = tl.program_id(2) * DO + tl.arange(0, DO)
    nrows = k * HH
    rmask = rows < nrows
    slot = tl.load(slots_ptr + b).to(tl.int64)
    lo = tl.load(lo_ptr + b)
    hi = tl.load(last_ptr + b)
    jc = tl.arange(0, W)
    jb = tl.arange(0, KB)
    bmask = jb < k
    cpos = tl.load(ring_pos_ptr + slot * ring_pos_stride + jc)
    cvalid = (cpos >= lo) & (cpos <= hi)
    ring_base = ring_ptr + slot * ring_slot_stride
    blk_base = blk_ptr + b.to(tl.int64) * k * D
    q_base = q_ptr + b.to(tl.int64) * nrows * D
    sc = tl.zeros([BM, W], tl.float32)
    sb = tl.zeros([BM, KB], tl.float32)
    for d0 in tl.static_range(0, D, BD):
        d = d0 + tl.arange(0, BD)
        qc = tl.load(q_base + rows[:, None] * D + d[None, :], mask=rmask[:, None], other=0.0)
        kc = tl.load(ring_base + jc[:, None] * D + d[None, :])
        kb = tl.load(blk_base + jb[:, None] * D + d[None, :], mask=bmask[:, None], other=0.0)
        sc += tl.dot(qc, tl.trans(kc))
        sb += tl.dot(qc, tl.trans(kb))
    sc = tl.where(cvalid[None, :], sc * scale, float("-inf"))
    sb = tl.where(bmask[None, :], sb * scale, float("-inf"))
    sk = tl.load(sink_ptr + rows % HH, mask=rmask, other=0.0)
    mx = tl.maximum(tl.maximum(tl.max(sc, axis=1), tl.max(sb, axis=1)), sk)
    pc = tl.exp(sc - mx[:, None])
    pb = tl.exp(sb - mx[:, None])
    denom = tl.sum(pc, axis=1) + tl.sum(pb, axis=1) + tl.exp(sk - mx)
    vc = tl.load(ring_base + jc[:, None] * D + d_out[None, :])
    vb = tl.load(blk_base + jb[:, None] * D + d_out[None, :], mask=bmask[:, None], other=0.0)
    acc = tl.dot(pc.to(tl.bfloat16), vc) + tl.dot(pb.to(tl.bfloat16), vb)
    o_base = out_ptr + b.to(tl.int64) * nrows * D
    tl.store(o_base + rows[:, None] * D + d_out[None, :], (acc / denom[:, None]).to(tl.bfloat16), mask=rmask[:, None])


def block_attention_ring(q: torch.Tensor, ring_stage: torch.Tensor, ring_pos: torch.Tensor, slots: torch.Tensor,
                         ctx_lo: torch.Tensor, last_pos: torch.Tensor, blk_kv: torch.Tensor, sink: torch.Tensor,
                         scale: float, BM: int | None = None, DO: int = 256, BD: int = 64,
                         num_warps: int | None = None) -> torch.Tensor:
    """ops.block_attention with the context read straight from the window ring of each request's slot.

    q [B, k, Hh, D] bf16; ring_stage [S, W, D] (one stage's ring, any slot stride); ring_pos [S, W] int64;
    slots / ctx_lo / last_pos int64 [B]; blk_kv [B, k, D] bf16; sink [Hh] fp32. Returns bf16 [B, k, Hh, D].
    Default tiles (3090, profiler kernel time per stage): <= 640 query rows BM 16 / 8 warps (14.7 us at 1x5x64),
    else BM 32 / 4 warps (21.0 us at 4x5x64); the gathered torch path took 70-155 us."""
    B, k, Hh, D = q.shape
    if BM is None:
        BM, num_warps = (16, 8) if B * k * Hh <= 640 else (32, 4)
    W = ring_stage.shape[1]
    KB = 16
    assert k <= KB and ring_stage.stride(2) == 1 and ring_stage.stride(1) == D and ring_pos.stride(1) == 1
    assert W == triton.next_power_of_2(W) and D % DO == 0 and D % BD == 0
    q = q.contiguous()
    blk_kv = blk_kv.contiguous()
    out = torch.empty_like(q)
    _block_attn_kernel[(B, triton.cdiv(k * Hh, BM), D // DO)](
        q, ring_stage, ring_pos, slots, ctx_lo, last_pos, blk_kv, sink, out, scale, k, ring_stage.stride(0),
        ring_pos.stride(0), W=W, D=D, HH=Hh, BM=BM, KB=KB, BD=BD, DO=DO, num_warps=num_warps, num_stages=1)
    return out


# --------------------------------------------------------------------------------------------- context insert


@triton.jit
def _amax_kernel(x_ptr, amax_ptr, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs, mask=offs < N, other=0.0).to(tl.float32)
    tl.atomic_max(amax_ptr, tl.max(tl.abs(v), axis=0))


@triton.jit
def _prescale_kernel(x_ptr, amax_ptr, out_ptr, sc_ptr, N, BLOCK: tl.constexpr):
    # the plugin's sc = exp2(clamp_min(ceil(log2(clamp_min(amax, 1) / 16384)), 0)) with the libdevice functions torch
    # calls; x / sc is exact (a power of two), then fp16
    a = tl.maximum(tl.load(amax_ptr), 1.0) / 16384.0
    sc = libdevice.exp2(tl.maximum(libdevice.ceil(libdevice.log2(a)), 0.0))
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    v = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, (v / sc).to(tl.float16), mask=m)
    if pid == 0:
        tl.store(sc_ptr, sc)


def prescale_fp16(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """main_proj's input scaling (vllm_exl3_moe dense.py): x (any float) -> (fp16 x / sc, fp32 sc [1]), sc the power
    of two >= 1 that brings max |x| under 2^14. Two launches (+ a memset) instead of the ~12 torch kernels."""
    x = x.contiguous()
    N = x.numel()
    amax = torch.zeros(1, dtype=torch.float32, device=x.device)
    out = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    sc = torch.empty(1, dtype=torch.float32, device=x.device)
    BLOCK = 4096
    grid = (triton.cdiv(N, BLOCK),)
    _amax_kernel[grid](x, amax, N, BLOCK=BLOCK, num_warps=8)
    _prescale_kernel[grid](x, amax, out, sc, N, BLOCK=BLOCK, num_warps=8, enable_fp_fusion=False)
    return out, sc


@triton.jit
def _scaled_rms_norm_kernel(y_ptr, sc_ptr, w_ptr, out_ptr, D, eps, BD: tl.constexpr):
    r_ = tl.program_id(0).to(tl.int64)
    sc = tl.load(sc_ptr)
    ss = tl.zeros([BD], tl.float32)
    for d0 in range(0, D, BD):
        d = d0 + tl.arange(0, BD)
        m = d < D
        x = (tl.load(y_ptr + r_ * D + d, mask=m, other=0.0) * sc).to(tl.bfloat16).to(tl.float32)
        ss += x * x
    r = 1.0 / tl.sqrt(tl.sum(ss, axis=0) / D + eps)
    for d0 in range(0, D, BD):
        d = d0 + tl.arange(0, BD)
        m = d < D
        x = (tl.load(y_ptr + r_ * D + d, mask=m, other=0.0) * sc).to(tl.bfloat16).to(tl.float32)
        w = tl.load(w_ptr + d, mask=m, other=0.0).to(tl.float32)
        tl.store(out_ptr + r_ * D + d, (w * (x * r)).to(tl.bfloat16), mask=m)


def scaled_rms_norm(y: torch.Tensor, sc: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """rms_norm(y * sc, w) for fp32 y [N, D] and a device scalar sc: the product rounded to bf16 first, as
    rms_norm(x.to(bf16)) after the plugin's y * sc."""
    y = y.contiguous()
    N, D = y.shape
    out = torch.empty(N, D, dtype=torch.bfloat16, device=y.device)
    _scaled_rms_norm_kernel[(N,)](y, sc, w, out, D, eps, BD=1024 if D >= 1024 else 512, num_warps=4,
                                  enable_fp_fusion=False)
    return out


@triton.jit
def _kv_store_kernel(y_ptr, w_ptr, normed_ptr, pos_ptr, cs_ptr, slot_ptr, ring_ptr, ring_pos_ptr, eps,
                     ring_slot_stride, ring_pos_stride, W, D: tl.constexpr, RD: tl.constexpr,
                     STORE_POS: tl.constexpr):
    """Row r: kv_norm (as _rms_norm_kernel) -> RoPE + V4 record (as _rope_kernel QDQ) -> ring[slot, pos % W]."""
    row = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, D)
    x = tl.load(y_ptr + row * D + d).to(tl.bfloat16).to(tl.float32)
    ss = x * x
    r = 1.0 / tl.sqrt(tl.sum(ss, axis=0) / D + eps)
    w = tl.load(w_ptr + d).to(tl.float32)
    tl.store(normed_ptr + row * D + d, (w * (x * r)).to(tl.bfloat16))
    tl.debug_barrier()
    p = tl.load(pos_ptr + row)
    slot = tl.load(slot_ptr + row).to(tl.int64)
    dst = ring_ptr + slot * ring_slot_stride + (p % W) * D
    NOPE: tl.constexpr = D - RD
    i = tl.arange(0, RD // 2)
    c = tl.load(cs_ptr + p * RD + i)
    s = tl.load(cs_ptr + p * RD + RD // 2 + i)
    x0 = tl.load(normed_ptr + row * D + NOPE + 2 * i).to(tl.float32)
    x1 = tl.load(normed_ptr + row * D + NOPE + 2 * i + 1).to(tl.float32)
    tl.store(dst + NOPE + 2 * i, (x0 * c - x1 * s).to(tl.bfloat16))
    tl.store(dst + NOPE + 2 * i + 1, (x1 * c + x0 * s).to(tl.bfloat16))
    g = tl.arange(0, 64)
    for b in tl.static_range(NOPE // 64):
        vf = tl.load(normed_ptr + row * D + b * 64 + g).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(vf), axis=0) * (1.0 / 448.0), 2.0 ** -126)
        bits = amax.to(tl.uint32, bitcast=True)
        e = ((bits >> 23) & 0xFF).to(tl.int32) - 127 + tl.where((bits & 0x7FFFFF) != 0, 1, 0)
        sc = ((e + 127) << 23).to(tl.uint32).to(tl.float32, bitcast=True)
        q = tl.minimum(tl.maximum(vf / sc, -448.0), 448.0)
        qb = q.to(tl.uint32, bitcast=True)
        qe = tl.maximum(((qb >> 23) & 0xFF).to(tl.int32) - 127, -6)
        quant = ((qe - 3 + 127) << 23).to(tl.uint32).to(tl.float32, bitcast=True)
        q = libdevice.rint(q / quant) * quant
        tl.store(dst + b * 64 + g, (q * sc).to(tl.bfloat16))
    if STORE_POS:
        tl.store(ring_pos_ptr + slot * ring_pos_stride + p % W, p)


def kv_store(y: torch.Tensor, w: torch.Tensor, eps: float, pos: torch.Tensor, cos_sin: torch.Tensor,
             ring_stage: torch.Tensor, slots: torch.Tensor, ring_pos: torch.Tensor | None = None) -> None:
    """ring_stage[slots, pos % W] = kv_rope_qdq(rms_norm(y, w), pos) for fp32 y [N, D] (wkv output); also
    ring_pos[slots, pos % W] = pos when ring_pos is given. Callers pass distinct (slot, pos % W) per call."""
    y = y.contiguous()
    N, D = y.shape
    W = ring_stage.shape[1]
    assert ring_stage.stride(2) == 1 and ring_stage.stride(1) == D and (ring_pos is None or ring_pos.stride(1) == 1)
    normed = torch.empty(N, D, dtype=torch.bfloat16, device=y.device)
    _kv_store_kernel[(N,)](y, w, normed, pos, cos_sin, slots, ring_stage, ring_pos if ring_pos is not None else pos,
                           eps, ring_stage.stride(0), ring_pos.stride(0) if ring_pos is not None else 0, W, D=D,
                           RD=cos_sin.shape[-1], STORE_POS=ring_pos is not None, num_warps=4, enable_fp_fusion=False)


# --------------------------------------------------------------------------------------------- wo_a (FP8 W8A16)


@triton.jit
def _wo_a_fp8_kernel(o_ptr, w_ptr, sc_ptr, out_ptr, T, o_stride, out_stride, C: tl.constexpr, R: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """out[t, g*R + r] = sum_c o[t, g*C + c] * (w[g*R + r, c] * sc[(g*R + r) // 32, c // 32]); the dequantized weight is
    exact in bf16, products accumulate in fp32 (bf16 MMA), bf16 out. BN <= 32 and BN | 32: one scale row per tile.
    e4m3 -> value: the byte's sign and 7 magnitude bits placed as fp16 bits (exponent field = e4m3 exponent, mantissa
    top bits = e4m3 mantissa) are the value / 2^8 exactly, subnormals included; x (2^8 * scale) is exact."""
    g = tl.program_id(0)
    r0 = g * R + tl.program_id(1) * BN
    t = tl.arange(0, BM)
    r = r0 + tl.arange(0, BN)
    tmask = t < T
    acc = tl.zeros([BM, BN], tl.float32)
    for k0 in range(0, C, BK):
        c = k0 + tl.arange(0, BK)
        a = tl.load(o_ptr + t[:, None].to(tl.int64) * o_stride + g * C + c[None, :], mask=tmask[:, None], other=0.0)
        b16 = tl.load(w_ptr + r[:, None].to(tl.int64) * C + c[None, :]).to(tl.int16)
        h = (((b16 & 0x80) << 8) | ((b16 & 0x7F) << 7)).to(tl.float16, bitcast=True)
        scv = tl.load(sc_ptr + (r0 // 32) * (C // 32) + c // 32) * 256.0
        w = (h.to(tl.float32) * scv[None, :]).to(tl.bfloat16)
        acc += tl.dot(a, tl.trans(w))
    tl.store(out_ptr + t[:, None].to(tl.int64) * out_stride + r[None, :], acc.to(tl.bfloat16), mask=tmask[:, None])


def wo_a_fp8(o: torch.Tensor, w8: torch.Tensor, sc: torch.Tensor, groups: int, BN: int = 32, BK: int = 128,
             num_warps: int = 2, num_stages: int | None = None) -> torch.Tensor:
    """o [T, groups * C] bf16 (rows may be strided), w8 [groups * R, C] uint8 (e4m3fn bytes, no NaN),
    sc [groups*R/32, C/32] fp32 (UE8M0 values) -> [T, groups * R] bf16: the engine's W8A16 wo_a
    (Sm12xMarlinMxfp8Bmm) on SM86, reading the FP8 bytes (32 MiB per stage instead of 64 MiB of expanded bf16)."""
    T = o.shape[0]
    GR, C = w8.shape
    R = GR // groups
    assert o.shape[1] == groups * C and o.stride(1) == 1 and w8.is_contiguous() and sc.is_contiguous()
    assert 32 % BN == 0 and R % 32 == 0 and C % BK == 0
    out = torch.empty(T, GR, dtype=torch.bfloat16, device=o.device)
    BM = max(16, triton.next_power_of_2(T))
    if num_stages is None:          # 3090, profiler: 51 us/stage at <= 16 rows (bf16 bmm 95), 80 us at 17-32 rows
        num_stages = 3 if BM == 16 else 2
    _wo_a_fp8_kernel[(groups, R // BN)](o, w8, sc, out, T, o.stride(0), out.stride(0), C=C, R=R, BM=BM, BN=BN,
                                         BK=BK, num_warps=num_warps, num_stages=num_stages)
    return out
