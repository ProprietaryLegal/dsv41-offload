"""Instrument N1's exact w13 reduction order, without changing its source.

The kernel below is copied verbatim from native.ops._moe_w13_kernel up to
its BF16 rounding, replacing the epilogue with FP32 projection stores.
"""
from __future__ import annotations
import triton
import triton.language as tl
from offload.dspark.native.ops import _fp4_val, _e8m0_val

@triton.jit
def n1_projection_oracle(x_ptr, ids_ptr, rw_ptr, w_ptr, s_ptr, g_ptr, u_ptr, I, H, lim,
                    PACKED: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, INTERP: tl.constexpr):
    """Pair p = (row t, slot j), output channels [n0, n0 + BN) of the rank's I: g/u = bf16(sum_k x * w1/w3[e]),
    official clamps + silu * up * rw -> bf16 -> act QDQ per 32 -> h[p, n] (bf16)."""
    p = tl.program_id(0)
    n0 = tl.program_id(1) * BN
    t = p // 3
    e = tl.load(ids_ptr + p).to(tl.int64)
    rw = tl.load(rw_ptr + p)
    n = n0 + tl.arange(0, BN)
    acc_g = tl.zeros([BN], tl.float32)
    acc_u = tl.zeros([BN], tl.float32)
    xrow = x_ptr + t.to(tl.int64) * H
    if PACKED:
        HB = H // 2
        gbase = w_ptr + (e * 2 * I + n[:, None].to(tl.int64)) * HB
        ubase = gbase + I.to(tl.int64) * HB
        sg = s_ptr + (e * 2 * I + n[:, None].to(tl.int64)) * (H // 32)
        su = sg + I.to(tl.int64) * (H // 32)
        for k0 in range(0, H, BK):
            j = k0 // 2 + tl.arange(0, BK // 2)
            xe = tl.load(xrow + 2 * j).to(tl.float32)
            xo = tl.load(xrow + 2 * j + 1).to(tl.float32)
            grp = (2 * j // 32)[None, :]
            bg = tl.load(gbase + j[None, :]).to(tl.int32)
            scg = _e8m0_val(tl.load(sg + grp).to(tl.int32))
            acc_g += tl.sum((_fp4_val(bg & 15) * xe[None, :] + _fp4_val(bg >> 4) * xo[None, :]) * scg, axis=1)
            bu = tl.load(ubase + j[None, :]).to(tl.int32)
            scu = _e8m0_val(tl.load(su + grp).to(tl.int32))
            acc_u += tl.sum((_fp4_val(bu & 15) * xe[None, :] + _fp4_val(bu >> 4) * xo[None, :]) * scu, axis=1)
    else:
        gbase = w_ptr + (e * 2 * I + n[:, None].to(tl.int64)) * H
        ubase = gbase + I.to(tl.int64) * H
        for k0 in range(0, H, BK):
            k = k0 + tl.arange(0, BK)
            xk = tl.load(xrow + k).to(tl.float32)
            acc_g += tl.sum(tl.load(gbase + k[None, :]).to(tl.float32) * xk[None, :], axis=1)
            acc_u += tl.sum(tl.load(ubase + k[None, :]).to(tl.float32) * xk[None, :], axis=1)
    tl.store(g_ptr + p.to(tl.int64) * I + n, acc_g)
    tl.store(u_ptr + p.to(tl.int64) * I + n, acc_u)
