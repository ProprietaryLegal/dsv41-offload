"""Native-format GPU ops for the TP2 drafter (SM86, Triton): the official activation QDQ, FP8 block-scaled dense
GEMMs, and the routed-expert pair kernels for packed MXFP4 and expanded BF16 banks. These are N1's reference
implementations of the kernel contract (NATIVE.md §4): exact decoding, the official rounding points, fp32
accumulation, graph-safe (no host sync, no allocation inside a capture beyond torch's graph pool, device-side expert
ids). The KX lanes' kernels replace them behind the same calls once they beat them and pass the same tests.

Every kernel here is held to formats.py / reference.py by tests/test_ops.py (bit-identical where the math allows:
act_qdq, decoding, the g/u bf16 roundings up to fp32 order ties; fp32 summation order otherwise).
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

SWIGLU_LIMIT = 10.0
# the Triton CPU interpreter (tests without a GPU) has no libdevice: kernels take INTERP and use exact stand-ins there
INTERP = os.environ.get("TRITON_INTERPRET") == "1"


# --------------------------------------------------------------------------------------------- QDQ helpers


@triton.jit
def _rint(y, INTERP: tl.constexpr):
    """Round half to even. Interpreter: the 1.5 * 2^23 trick (exact for |y| < 2^22; here |y| <= 16)."""
    if INTERP:
        r = (y + 12582912.0) - 12582912.0
        return tl.where(r == 0.0, y * 0.0, r)                  # keep the sign of a zero (-0.0), as rint does
    else:
        return libdevice.rint(y)


@triton.jit
def _exp(y, INTERP: tl.constexpr):
    """Full-precision expf (torch's silu); the interpreter's numpy exp stands in for it there."""
    if INTERP:
        return tl.exp(y)
    else:
        return libdevice.exp(y)


@triton.jit
def _bf16_rne(x):
    """fp32 -> the nearest bf16 value (ties to even), still fp32. Explicit bits: the CPU interpreter's own bf16 cast
    truncates; on the GPU this is cvt.rn.bf16.f32 exactly (finite inputs)."""
    b = x.to(tl.uint32, bitcast=True)
    b = (b + (((b >> 16) & 1) + 0x7FFF)) & 0xFFFF0000
    return b.to(tl.float32, bitcast=True)


@triton.jit
def _e4m3_qdq_group(vf, INTERP: tl.constexpr):
    """Official act_quant of fp32 values [G, 32] (each row one 32-element block): returns q * s (fp32, exact)."""
    amax = tl.maximum(tl.max(tl.abs(vf), axis=1), 1e-4) * (1.0 / 448.0)
    bits = amax.to(tl.uint32, bitcast=True)
    e = ((bits >> 23) & 0xFF).to(tl.int32) - 127 + tl.where((bits & 0x7FFFFF) != 0, 1, 0)
    sc = ((e + 127) << 23).to(tl.uint32).to(tl.float32, bitcast=True)
    q = tl.minimum(tl.maximum(vf / sc[:, None], -448.0), 448.0)
    qb = q.to(tl.uint32, bitcast=True)
    qe = tl.maximum(((qb >> 23) & 0xFF).to(tl.int32) - 127, -6)
    quant = ((qe - 3 + 127) << 23).to(tl.uint32).to(tl.float32, bitcast=True)
    q = _rint(q / quant, INTERP) * quant
    return q * sc[:, None]


@triton.jit
def _act_qdq_kernel(x_ptr, out_ptr, K, x_stride, out_stride, G: tl.constexpr, INTERP: tl.constexpr):
    """Row r, G blocks of 32 starting at block pid(1) * G."""
    r = tl.program_id(0).to(tl.int64)
    b0 = tl.program_id(1) * G
    gi = tl.arange(0, G)
    ci = tl.arange(0, 32)
    col = (b0 + gi)[:, None] * 32 + ci[None, :]
    m = col < K
    v = tl.load(x_ptr + r * x_stride + col, mask=m, other=0.0).to(tl.float32)
    tl.store(out_ptr + r * out_stride + col, _e4m3_qdq_group(v, INTERP).to(tl.bfloat16), mask=m)


def act_qdq(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """formats.act_qdq for bf16 x [..., K] (K % 32 == 0) on the GPU, bit for bit. Rows may be strided."""
    if x.dtype != torch.bfloat16:
        raise TypeError(f"act_qdq takes bf16, got {x.dtype}")
    K = x.shape[-1]
    if K % 32:
        raise ValueError(f"act_qdq: {K} not a multiple of 32")
    x2 = x.reshape(-1, K)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    if out is None:
        out = torch.empty(x2.shape, dtype=torch.bfloat16, device=x.device)
    o2 = out.view(-1, K)
    G = 8
    _act_qdq_kernel[(x2.shape[0], triton.cdiv(K // 32, G))](x2, o2, K, x2.stride(0), o2.stride(0), G=G, INTERP=INTERP,
                                                              num_warps=2, enable_fp_fusion=False)
    return out.view(x.shape)


# --------------------------------------------------------------------------------------------- dense GEMMs


@triton.jit
def _e4m3_val(b):
    """e4m3fn byte (int32 0..255, no NaN) -> fp32 value, exact: e = (b >> 3) & 15, m = b & 7; e > 0:
    (1 + m/8) * 2^(e - 7), e == 0: m * 2^-9; bit 7 = sign."""
    e = (b >> 3) & 15
    m = b & 7
    normal = (((e - 7 + 127) << 23) | (m << 20)).to(tl.uint32).to(tl.float32, bitcast=True)
    v = tl.where(e == 0, m.to(tl.float32) * 0.001953125, normal)
    return tl.where((b & 0x80) != 0, -v, v)


@triton.jit
def _dense_kernel(a_ptr, w_ptr, sc_ptr, out_ptr, M, N, K, a_stride, out_stride, sc_stride,
                  FP8: tl.constexpr, OUT_F32: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  INTERP: tl.constexpr):
    """out[m, n] = sum_k a[m, k] * W[n, k]; W FP8 e4m3 bytes with E8M0 32x32 block scales (as fp32 values ``sc``
    [ceil(N/32), K/32]) or exact bf16. bf16 x bf16 MMA, fp32 accumulate. BN | 32 (one scale row per tile)."""
    n0 = tl.program_id(0) * BN
    m = tl.arange(0, BM)
    n = n0 + tl.arange(0, BN)
    mm = m < M
    acc = tl.zeros([BM, BN], tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + m[:, None].to(tl.int64) * a_stride + k[None, :], mask=mm[:, None], other=0.0)
        if FP8:
            b = tl.load(w_ptr + n[:, None].to(tl.int64) * K + k[None, :]).to(tl.int32)
            scv = tl.load(sc_ptr + (n0 // 32) * sc_stride + k // 32)
            w = (_e4m3_val(b) * scv[None, :]).to(tl.bfloat16)
        else:
            w = tl.load(w_ptr + n[:, None].to(tl.int64) * K + k[None, :])
        if INTERP:          # the interpreter's dot mishandles bf16 operands; fp32 (exact values) instead
            acc += tl.dot(a.to(tl.float32), tl.trans(w.to(tl.float32)), input_precision="ieee")
        else:
            acc += tl.dot(a, tl.trans(w))
    optr = out_ptr + m[:, None].to(tl.int64) * out_stride + n[None, :]
    if OUT_F32:
        tl.store(optr, acc, mask=mm[:, None])
    else:
        tl.store(optr, _bf16_rne(acc).to(tl.bfloat16), mask=mm[:, None])


class NativeDense:
    """One dense matrix (or a rank's block-aligned shard of one) in native form: ``fp8`` (official e4m3 bytes +
    E8M0 block scales held as fp32 values) or ``bf16`` (exactly decoded). __call__ takes the QDQ'd bf16 input and
    returns bf16 (the official linear's output) or fp32 (a TP partial sum, rounded only after the all-reduce)."""

    def __init__(self, name: str, kind: str, w: torch.Tensor, sc: torch.Tensor | None, backend: str | None = None):
        if kind not in ("fp8", "bf16"):
            raise ValueError(kind)
        self.name, self.kind = name, kind
        # bf16 weights: cuBLAS (fp32 accumulation; torch.mm out_dtype=fp32 for TP partials) is the measured choice
        # for every rank-local shape (KX-GROUPED dense screen); fp8 bytes: N1's Triton W8A16 kernel.
        self.backend = backend or ("cublas" if kind == "bf16" and w.is_cuda else "triton")
        if self.backend not in ("triton", "cublas") or (self.backend == "cublas" and kind != "bf16"):
            raise ValueError(f"{name}: backend {self.backend} for {kind}")
        self.w = w.contiguous()
        self.sc = sc.contiguous() if sc is not None else None
        self.out_features, self.in_features = self.w.shape
        if kind == "fp8" and (self.w.dtype != torch.uint8 or self.sc is None or self.sc.dtype != torch.float32):
            raise TypeError(f"{name}: fp8 needs uint8 bytes + fp32 scale values")
        if kind == "bf16" and self.w.dtype != torch.bfloat16:
            raise TypeError(f"{name}: bf16 kind with {self.w.dtype}")
        if self.in_features % 128:
            raise ValueError(f"{name}: K {self.in_features} not a multiple of 128")

    def nbytes(self) -> int:
        return self.w.numel() * self.w.element_size() + (self.sc.numel() * 4 if self.sc is not None else 0)

    def __call__(self, xq: torch.Tensor, out_f32: bool = False, out: torch.Tensor | None = None) -> torch.Tensor:
        M, K = xq.shape
        if K != self.in_features or xq.dtype != torch.bfloat16 or xq.stride(1) != 1:
            raise ValueError(f"{self.name}: input {tuple(xq.shape)} {xq.dtype}")
        N = self.out_features
        if self.backend == "cublas":
            if torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction:
                raise RuntimeError(f"{self.name}: cuBLAS reduced-precision reduction is enabled")
            if out_f32:
                if out is None:
                    return torch.mm(xq, self.w.t(), out_dtype=torch.float32)
                return torch.mm(xq, self.w.t(), out_dtype=torch.float32, out=out)
            y = torch.nn.functional.linear(xq, self.w)
            if out is not None:
                out.copy_(y)
                return out
            return y
        if out is None:
            out = torch.empty(M, N, dtype=torch.float32 if out_f32 else torch.bfloat16, device=xq.device)
        if M > 128:                         # eager callers only (context inserts of several requests): row chunks
            for m0 in range(0, M, 128):
                self(xq[m0:m0 + 128], out_f32, out[m0:m0 + 128])
            return out
        BM = max(16, triton.next_power_of_2(M))
        BN = 32 if N >= 32 * 82 else 16
        _dense_kernel[(triton.cdiv(N, BN),)](xq, self.w, self.sc if self.sc is not None else self.w, out, M, N, K,
                                             xq.stride(0), out.stride(0), self.sc.stride(0) if self.sc is not None
                                             else 0, FP8=self.kind == "fp8", OUT_F32=out_f32, BM=BM, BN=BN, BK=128, INTERP=INTERP,
                                             num_warps=4, num_stages=3 if BM <= 32 else 2)
        return out


# --------------------------------------------------------------------------------------------- routed experts


@triton.jit
def _fp4_val(c):
    """E2M1 code (int32 0..15) -> fp32 value, exact: mag = c & 7; e = mag >> 1, m = mag & 1;
    e == 0: m * 0.5, else (1 + m / 2) * 2^(e - 1); bit 3 = sign."""
    mag = c & 7
    e = mag >> 1
    mb = mag & 1
    normal = (((e - 1 + 127) << 23) | (mb << 22)).to(tl.uint32).to(tl.float32, bitcast=True)
    v = tl.where(e == 0, mb.to(tl.float32) * 0.5, normal)
    return tl.where((c & 8) != 0, -v, v)


@triton.jit
def _e8m0_val(s):
    """E8M0 byte (int32 1..254) -> 2^(s - 127) fp32 (loader refuses 0 and 255)."""
    return (s << 23).to(tl.uint32).to(tl.float32, bitcast=True)


@triton.jit
def _moe_w13_kernel(x_ptr, ids_ptr, rw_ptr, w_ptr, s_ptr, h_ptr, I, H, lim,
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
    g = _bf16_rne(acc_g)
    u = _bf16_rne(acc_u)
    u = tl.minimum(tl.maximum(u, -lim), lim)
    g = tl.minimum(g, lim)
    h = g / (1.0 + _exp(-g, INTERP)) * u
    h = _bf16_rne(rw * h)
    hq = _e4m3_qdq_group(tl.reshape(h, (BN // 32, 32)), INTERP)
    tl.store(h_ptr + p.to(tl.int64) * I + n, tl.reshape(hq, (BN,)).to(tl.bfloat16))


@triton.jit
def _moe_w2_kernel(h_ptr, ids_ptr, w_ptr, s_ptr, out_ptr, I, H, out_tstride, PACKED: tl.constexpr,
                   BN: tl.constexpr, BK: tl.constexpr):
    """Pair p = (t, j), hidden outputs [n0, n0 + BN): out[t, j, n] = sum over the rank's I inputs of h[p] * w2[e]
    (fp32 partial; out rows t strided by out_tstride, slots j by H)."""
    p = tl.program_id(0)
    n0 = tl.program_id(1) * BN
    e = tl.load(ids_ptr + p).to(tl.int64)
    n = n0 + tl.arange(0, BN)
    acc = tl.zeros([BN], tl.float32)
    hrow = h_ptr + p.to(tl.int64) * I
    if PACKED:
        IB = I // 2
        wb = w_ptr + (e * H + n[:, None].to(tl.int64)) * IB
        sb = s_ptr + (e * H + n[:, None].to(tl.int64)) * (I // 32)
        for k0 in range(0, I, BK):
            j = k0 // 2 + tl.arange(0, BK // 2)
            he = tl.load(hrow + 2 * j).to(tl.float32)
            ho = tl.load(hrow + 2 * j + 1).to(tl.float32)
            b = tl.load(wb + j[None, :]).to(tl.int32)
            sc = _e8m0_val(tl.load(sb + (2 * j // 32)[None, :]).to(tl.int32))
            acc += tl.sum((_fp4_val(b & 15) * he[None, :] + _fp4_val(b >> 4) * ho[None, :]) * sc, axis=1)
    else:
        wb = w_ptr + (e * H + n[:, None].to(tl.int64)) * I
        for k0 in range(0, I, BK):
            k = k0 + tl.arange(0, BK)
            hk = tl.load(hrow + k).to(tl.float32)
            acc += tl.sum(tl.load(wb + k[None, :]).to(tl.float32) * hk[None, :], axis=1)
    tl.store(out_ptr + (p // 3).to(tl.int64) * out_tstride + (p % 3) * H + n, acc)


class ExpertBank:
    """One stage's 128 routed experts on one rank, native. ``packed``: w13 uint8 [E, 2, I, H/2] (w1 rows then w3
    rows of the rank's I output channels, official bytes), s13 uint8 [E, 2, I, H/32], w2 uint8 [E, H, I/2] (the
    rank's I inputs), s2 uint8 [E, H, I/32]. ``expanded``: w13 bf16 [E, 2, I, H], w2 bf16 [E, H, I] (exact)."""

    def __init__(self, name: str, kind: str, E: int, H: int, I: int, dev: torch.device):
        if kind not in ("packed", "expanded"):
            raise ValueError(kind)
        self.name, self.kind, self.E, self.H, self.I, self.dev = name, kind, E, H, I, dev
        if kind == "packed":
            self.w13 = torch.empty(E, 2, I, H // 2, dtype=torch.uint8, device=dev)
            self.s13 = torch.empty(E, 2, I, H // 32, dtype=torch.uint8, device=dev)
            self.w2 = torch.empty(E, H, I // 2, dtype=torch.uint8, device=dev)
            self.s2 = torch.empty(E, H, I // 32, dtype=torch.uint8, device=dev)
        else:
            self.w13 = torch.empty(E, 2, I, H, dtype=torch.bfloat16, device=dev)
            self.s13 = None
            self.w2 = torch.empty(E, H, I, dtype=torch.bfloat16, device=dev)
            self.s2 = None

    def tensors(self) -> list[torch.Tensor]:
        return [t for t in (self.w13, self.s13, self.w2, self.s2) if t is not None]

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors())

    def forward_slots(self, xq: torch.Tensor, ids: torch.Tensor, rw: torch.Tensor, h: torch.Tensor | None = None,
                      out: torch.Tensor | None = None) -> torch.Tensor:
        """xq bf16 [T, H] (QDQ'd), ids int [T, 3], rw fp32 [T, 3] -> fp32 [T, 3, H] per-slot partials (SLOT mode)."""
        T = xq.shape[0]
        P = T * 3
        if xq.shape[1] != self.H or ids.shape != (T, 3) or rw.shape != (T, 3):
            raise ValueError(f"{self.name}: x {tuple(xq.shape)} ids {tuple(ids.shape)} rw {tuple(rw.shape)}")
        ids = ids.contiguous()
        rw = rw.contiguous().float()
        if h is None:
            h = torch.empty(P, self.I, dtype=torch.bfloat16, device=xq.device)
        if out is None:
            out = torch.empty(T, 3, self.H, dtype=torch.float32, device=xq.device)
        if out.shape != (T, 3, self.H) or out.stride(1) != self.H or out.stride(2) != 1 or out.dtype != torch.float32:
            raise ValueError(f"{self.name}: out {tuple(out.shape)} strides {out.stride()} {out.dtype}")
        packed = self.kind == "packed"
        dummy = self.w13
        BN1 = 64
        _moe_w13_kernel[(P, self.I // BN1)](xq, ids, rw, self.w13, self.s13 if packed else dummy, h, self.I, self.H,
                                            SWIGLU_LIMIT, PACKED=packed, BN=BN1, INTERP=INTERP, BK=256 if packed else 128,
                                            num_warps=4, enable_fp_fusion=False)
        BN2 = 64
        _moe_w2_kernel[(P, self.H // BN2)](h, ids, self.w2, self.s2 if packed else dummy, out, self.I, self.H,
                                           out.stride(0), PACKED=packed, BN=BN2, BK=128 if packed else 64, num_warps=4,
                                           enable_fp_fusion=False)
        return out


# --------------------------------------------------------------------------------------------- combines


def ordered_slot_sum(parts: torch.Tensor, ids: torch.Tensor, round_each: bool) -> torch.Tensor:
    """fp32 [T, H] = the 3 slot rows of ``parts`` [T, 3, H] summed in ascending expert-id order (the official MoE
    loop order), each rounded to bf16 first when ``round_each`` (the official per-expert bf16 output)."""
    order = ids.argsort(dim=1, stable=True)
    p = parts.gather(1, order.unsqueeze(-1).expand(-1, -1, parts.shape[-1]))
    if round_each:
        p = p.to(torch.bfloat16).float()
    return (p[:, 0] + p[:, 1]) + p[:, 2]


def moe_combine_official(slots_sum: torch.Tensor, shared_sum: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """After the fp32 all-reduce of the slot partials [T, 3, H] and the shared partial [T, H]: per-expert bf16,
    ascending-id fp32 sum, + bf16 shared, -> bf16 (MoE.forward, world size 1)."""
    y = ordered_slot_sum(slots_sum, ids, True)
    return (y + shared_sum.to(torch.bfloat16).float()).to(torch.bfloat16)


def swiglu_bf16(g16: torch.Tensor, u16: torch.Tensor, limit: float = SWIGLU_LIMIT) -> torch.Tensor:
    """The shared expert's SwiGLU (no route weight) on the bf16 linear outputs: reference.swiglu, torch ops."""
    gate = torch.clamp(g16.float(), max=limit)
    up = torch.clamp(u16.float(), min=-limit, max=limit)
    return (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)


# --------------------------------------------------------------------------------------------- head arm: exact Markov


@triton.jit
def _markov_logits_fp32_kernel(prev_ptr, slot_ptr, cache_ptr, w1_ptr, w2_ptr, base_ptr, out_ptr, V, base_stride,
                               out_stride, cache_stride, R: tl.constexpr, BLOCK_V: tl.constexpr,
                               HAS_CACHE: tl.constexpr):
    """out[b, v] = base[b, v] + sum_r fp32(w1[prev[b], r]) * fp32(w2[v, r]) over the shard's vocabulary rows v:
    DSparkMarkovHead with its fp32 head (model.py keeps ParallelHead weights fp32) added to fp32 logits. HAS_CACHE:
    previous tokens with a cached row (slot >= 0) read the fp32 bias row instead; the rows were written by this
    kernel at load (same block size, same reduction), so a cached row is the computed row bit for bit."""
    b = tl.program_id(0)
    v = tl.program_id(1) * BLOCK_V + tl.arange(0, BLOCK_V)
    m = v < V
    p = tl.load(prev_ptr + b).to(tl.int64)
    lb = tl.load(base_ptr + b * base_stride + v, mask=m, other=0.0)
    sl = -1
    if HAS_CACHE:
        sl = tl.load(slot_ptr + p)
    if sl >= 0:
        bias = tl.load(cache_ptr + sl.to(tl.int64) * cache_stride + v, mask=m, other=0.0)
    else:
        r = tl.arange(0, R)
        e = tl.load(w1_ptr + p * R + r).to(tl.float32)
        w = tl.load(w2_ptr + v[:, None].to(tl.int64) * R + r[None, :], mask=m[:, None], other=0.0).to(tl.float32)
        bias = tl.sum(w * e[None, :], axis=1)
    tl.store(out_ptr + b * out_stride + v, lb + bias, mask=m)


MARKOV_BV = 128


def markov_logits_fp32(base: torch.Tensor, prev: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
                       cache: "MarkovRowsFP32 | None" = None) -> torch.Tensor:
    """fp32 [B, V_shard] = base (fp32 head logits of the shard) + the exact fp32 Markov bias of ``prev``."""
    B, V = base.shape
    if base.dtype != torch.float32 or base.stride(1) != 1 or w2.shape[0] != V:
        raise ValueError(f"markov_logits_fp32: base {tuple(base.shape)} {base.dtype}, w2 {tuple(w2.shape)}")
    out = torch.empty(B, V, dtype=torch.float32, device=base.device)
    hc = cache is not None and cache.rows > 0
    _markov_logits_fp32_kernel[(B, triton.cdiv(V, MARKOV_BV))](
        prev, cache.slot_of_token if hc else prev, cache.cache if hc else base, w1, w2, base, out, V,
        base.stride(0), out.stride(0), cache.cache.stride(0) if hc else 0, R=w1.shape[1], BLOCK_V=MARKOV_BV,
        HAS_CACHE=hc, num_warps=4, enable_fp_fusion=False)
    return out


class MarkovRowsFP32:
    """Hot-row cache of the exact fp32 Markov bias over one vocabulary shard: rows for the first ``rows`` distinct ids
    of the serve's frequent-token list, computed once at load by the same kernel (never stale: the weights are fixed;
    a hit returns exactly the bits a miss computes). fp32: 4 B x shard width per row."""

    def __init__(self, w1: torch.Tensor, w2: torch.Tensor, token_list, rows: int):
        V, dev = w1.shape[0], w2.device
        ids: list[int] = []
        seen: set[int] = set()
        for t in token_list or []:
            t = int(t)
            if 0 <= t < V and t not in seen:
                seen.add(t)
                ids.append(t)
        ids = ids[:rows]
        self.rows = len(ids)
        self.slot_of_token = torch.full((V,), -1, dtype=torch.int32, device=dev)
        self.cache = torch.zeros((max(self.rows, 1), w2.shape[0]), dtype=torch.float32, device=dev)
        if ids:
            tok = torch.tensor(ids, dtype=torch.int64, device=dev)
            zero = torch.zeros(256, w2.shape[0], dtype=torch.float32, device=dev)
            for s0 in range(0, len(ids), 256):
                t = tok[s0:s0 + 256]
                self.cache[s0:s0 + len(t)] = markov_logits_fp32(zero[:len(t)], t, w1, w2)
            self.slot_of_token[tok] = torch.arange(len(ids), dtype=torch.int32, device=dev)

    def nbytes(self) -> int:
        return self.cache.numel() * 4 + self.slot_of_token.numel() * 4


# --------------------------------------------------------------------------------------------- SwiGLU + QDQ (KX path)


@triton.jit
def _swiglu_qdq_kernel(g_ptr, u_ptr, rw_ptr, h_ptr, I, lim, BN: tl.constexpr, INTERP: tl.constexpr):
    """Pair p, channels [n0, n0 + BN): the same epilogue as _moe_w13_kernel on fp32 projection outputs g/u
    [P, I]: bf16 rounding of the linear outputs, clamps, silu (full-precision expf) * up * route weight, bf16, act QDQ."""
    p = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    g = _bf16_rne(tl.load(g_ptr + p.to(tl.int64) * I + n))
    u = _bf16_rne(tl.load(u_ptr + p.to(tl.int64) * I + n))
    rw = tl.load(rw_ptr + p)
    u = tl.minimum(tl.maximum(u, -lim), lim)
    g = tl.minimum(g, lim)
    h = _bf16_rne(rw * (g / (1.0 + _exp(-g, INTERP)) * u))
    hq = _e4m3_qdq_group(tl.reshape(h, (BN // 32, 32)), INTERP)
    tl.store(h_ptr + p.to(tl.int64) * I + n, tl.reshape(hq, (BN,)).to(tl.bfloat16))


def swiglu_qdq(g: torch.Tensor, u: torch.Tensor, rw: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """g, u fp32 [T, 3, I] (routed projections), rw fp32 [T, 3] -> bf16 [T * 3, I]: the down projection's QDQ'd input."""
    T, top, I = g.shape
    if u.shape != g.shape or rw.shape != (T, top) or I % 64 or not (g.is_contiguous() and u.is_contiguous()):
        raise ValueError(f"swiglu_qdq: g {tuple(g.shape)} u {tuple(u.shape)} rw {tuple(rw.shape)}")
    if out is None:
        out = torch.empty(T * top, I, dtype=torch.bfloat16, device=g.device)
    _swiglu_qdq_kernel[(T * top, I // 64)](g, u, rw.contiguous().float(), out, I, SWIGLU_LIMIT, BN=64, INTERP=INTERP,
                                           num_warps=2, enable_fp_fusion=False)
    return out
