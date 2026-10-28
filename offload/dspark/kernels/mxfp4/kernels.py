"""Original SM86 native-packed W4A16 kernels. No native FP4/FP8 MMA or atomics on results."""
from __future__ import annotations

import triton as tr
import triton.language as tl


@tr.jit
def bf16_decode(code, scale):
    # E2M1 magnitude as BF16 fields, then exact exponent adjustment. Integer
    # construction also handles scale-0 subnormals; no FTZ multiplication here.
    mag = code & 7
    exp = tl.where(mag < 2, -1, tl.where(mag < 4, 0, tl.where(mag < 6, 1, 2)))
    mant = tl.where((mag == 3) | (mag == 5) | (mag == 7), 64, 0)
    biased = exp + scale
    normal = (biased << 7) | mant
    sub = (128 | mant) >> (1 - biased)
    bits = tl.where(biased > 0, normal, sub)
    bits = tl.where(biased >= 255, 0x7f80, bits)
    bits = tl.where(mag == 0, 0, bits) | ((code & 8) << 12)
    bits = tl.where(scale == 255, 0x7fc0, bits)
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@tr.jit
def decode_kernel(C, S, O, COUNT: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    c = tl.load(C+i, i < COUNT, 0).to(tl.int32)
    s = tl.load(S+i, i < COUNT, 127).to(tl.int32)
    tl.store(O+i, bf16_decode(c, s), i < COUNT)


@tr.jit
def _weights(W, S, e, k, n, N: tl.constexpr, K: tl.constexpr):
    p = tl.load(W + (e * (K//2) + k[:, None]//2) * N + n[None, :],
                (k[:, None] < K) & (n[None, :] < N), 0).to(tl.int32)
    c = (p >> ((k[:, None] & 1) * 4)) & 15
    s = tl.load(S + (e * (K//32) + k[:, None]//32) * N + n[None, :],
                (k[:, None] < K) & (n[None, :] < N), 127).to(tl.int32)
    return bf16_decode(c, s)


@tr.jit
def linear_kernel(X, IDS, W, S, U, US, O, OU, STATUS,
                  N: tl.constexpr, K: tl.constexpr, E: tl.constexpr,
                  TOP: tl.constexpr, PER_SLOT: tl.constexpr,
                  FUSED: tl.constexpr, MMA: tl.constexpr,
                  BN: tl.constexpr, BK: tl.constexpr):
    route = tl.program_id(1)
    expert = tl.load(IDS+route)
    valid = (expert >= 0) & (expert < E)
    if not valid:
        tl.atomic_or(STATUS, 1)
    expert = tl.minimum(tl.maximum(expert, 0), E-1)
    row = route if PER_SLOT else route // TOP
    n = tl.program_id(0)*BN + tl.arange(0, BN)
    if MMA:
        m = tl.arange(0, 16)
        acc = tl.full((16, BN), 0, tl.float32)
        if FUSED:
            up = tl.full((16, BN), 0, tl.float32)
        for start in range(tr.cdiv(K, BK)):
            k = start*BK + tl.arange(0, BK)
            x = tl.load(X + row*K + k[None, :], (m[:, None] == 0) & (k[None, :] < K), 0)
            w = _weights(W, S, expert, k, n, N, K)
            acc = tl.dot(x, w, acc, out_dtype=tl.float32)
            if FUSED:
                u = _weights(U, US, expert, k, n, N, K)
                up = tl.dot(x, u, up, out_dtype=tl.float32)
        value = tl.sum(tl.where(m[:, None] == 0, acc, 0), axis=0)
        if FUSED:
            uvalue = tl.sum(tl.where(m[:, None] == 0, up, 0), axis=0)
    else:
        # One/few row SIMT control: FP32 product/reduction, decoded BF16 values.
        # Only BK independent partials persist, rather than the whole matrix.
        acc = tl.full((BK, BN), 0, tl.float32)
        if FUSED:
            up = tl.full((BK, BN), 0, tl.float32)
        for start in range(tr.cdiv(K, BK)):
            k = start*BK + tl.arange(0, BK)
            x = tl.load(X+row*K+k, k < K, 0).to(tl.float32)
            w = _weights(W, S, expert, k, n, N, K).to(tl.float32)
            acc += x[:, None] * w
            if FUSED:
                u = _weights(U, US, expert, k, n, N, K).to(tl.float32)
                up += x[:, None] * u
        value = tl.sum(acc, axis=0)
        if FUSED:
            uvalue = tl.sum(up, axis=0)
    tl.store(O+route*N+n, tl.where(valid, value, float('nan')), n < N)
    if FUSED:
        tl.store(OU+route*N+n, tl.where(valid, uvalue, float('nan')), n < N)


@tr.jit
def activation_kernel(G, U, R, A, COUNT: tl.constexpr, N: tl.constexpr,
                      LIMIT: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0)*B + tl.arange(0, B)
    # Official linear outputs round to BF16 before clamp/SwiGLU; routing is
    # applied before the BF16 down input, not moved through its trained QDQ.
    gate = tl.minimum(tl.load(G+i, i < COUNT, 0).to(tl.bfloat16).to(tl.float32), LIMIT)
    up = tl.minimum(tl.maximum(tl.load(U+i, i < COUNT, 0).to(tl.bfloat16).to(tl.float32), -LIMIT), LIMIT)
    route = tl.load(R+i//N, i < COUNT, 0)
    act = gate / (1 + tl.exp(-gate)) * up * route
    tl.store(A+i, act.to(tl.bfloat16), i < COUNT)


@tr.jit
def combine_kernel(P, R, O, N: tl.constexpr, TOP: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(1)
    n = tl.program_id(0)*B + tl.arange(0, B)
    total = tl.full((B,), 0, tl.float32)
    for slot in tl.static_range(TOP):
        weight = tl.load(R+row*TOP+slot)
        part = tl.load(P+(row*TOP+slot)*N+n, n < N, 0)
        total = total + part * weight
    tl.store(O+row*N+n, total, n < N)


@tr.jit
def native_combine_kernel(P, IDS, O, N: tl.constexpr, TOP: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(1)
    n = tl.program_id(0)*B + tl.arange(0, B)
    total = tl.full((B,), 0, tl.float32)
    # Official expert-loop order. Stable slot order also handles duplicate IDs
    # in diagnostic inputs. TP reduction must already be complete per route.
    for order in tl.static_range(TOP):
        value = tl.full((B,), 0, tl.float32)
        for slot in tl.static_range(TOP):
            eid = tl.load(IDS+row*TOP+slot)
            rank = tl.full((), 0, tl.int32)
            for other in tl.static_range(TOP):
                other_id = tl.load(IDS+row*TOP+other)
                rank += ((other_id < eid) | ((other_id == eid) & (other < slot))).to(tl.int32)
            part = tl.load(P+(row*TOP+slot)*N+n, n < N, 0).to(tl.bfloat16).to(tl.float32)
            value = tl.where(rank == order, part, value)
        total = total + value
    tl.store(O+row*N+n, total, n < N)
