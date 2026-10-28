# SPDX-License-Identifier: Apache-2.0
"""One-launch version of RowGatherer.enqueue's device work (Triton).

The torch path issues ~40 small ops per round (index builds, compaction, three tap
gathers, a concat, two index_copy, metadata), each an eager kernel launch on the engine
stream right after sampling. This kernel does the same in one launch with no index upload:
the per-request facts (candidate count, span start, slot) are scalar launch arguments
(at most 4 requests), and the compaction offsets are recomputed per program from them and
num_rejected. Values move as 16-bit words (bf16 bits), so the result is bit-identical to
the torch path. tests/test_fused_gather.py checks that under TRITON_INTERPRET=1 on CPU.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

MAX_REQ = 4

if triton is not None:

    @triton.jit(do_not_specialize=["n_req", "nc0", "nc1", "nc2", "nc3", "st0", "st1", "st2",
                                   "st3", "sl0", "sl1", "sl2", "sl3", "dump_stage",
                                   "dump_ring"])
    def _gather_kernel(
        tap0, tap1, tap2,          # int16 views [T, H] (row stride s0/s1/s2)
        s0, s1, s2,
        positions,                 # int64 [T]
        num_rejected,              # int32 [n]
        num_sampled,               # int32 [n]
        last_sampled,              # int64 [max_reqs] (flat)
        next_prefill,              # int32 [max_reqs] (row 0)
        stage,                     # int16 [max_rows+1, 3H]
        ring,                      # int16 [slots*W+1, 3H]
        meta,                      # int64 [4, max_slots]: valid, first_pos, anchor, sampled
        meta_stride,
        n_req,
        nc0, nc1, nc2, nc3,        # candidate rows per request
        st0, st1, st2, st3,        # first candidate row (batch row index)
        sl0, sl1, sl2, sl3,        # slot
        dump_stage, dump_ring,
        H: tl.constexpr,
        W: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)     # candidate row 0..R-1
        cb = tl.program_id(1)      # column block within one tap
        t = tl.program_id(2)       # tap 0..2
        # which request owns this candidate row
        c1 = nc0
        c2 = c1 + nc1
        c3 = c2 + nc2
        req = tl.where(row < c1, 0, tl.where(row < c2, 1, tl.where(row < c3, 2, 3)))
        base = tl.where(req == 0, 0, tl.where(req == 1, c1, tl.where(req == 2, c2, c3)))
        r = row - base
        nc = tl.where(req == 0, nc0, tl.where(req == 1, nc1, tl.where(req == 2, nc2, nc3)))
        st = tl.where(req == 0, st0, tl.where(req == 1, st1, tl.where(req == 2, st2, st3)))
        sl = tl.where(req == 0, sl0, tl.where(req == 1, sl1, tl.where(req == 2, sl2, sl3)))
        # valid rows per request and the compacted offset of this request
        rej0 = tl.load(num_rejected + 0).to(tl.int64)
        rej1 = tl.load(num_rejected + 1, mask=n_req > 1, other=0).to(tl.int64)
        rej2 = tl.load(num_rejected + 2, mask=n_req > 2, other=0).to(tl.int64)
        v0 = tl.minimum(tl.maximum(nc0 - rej0, 0), nc0)
        v1 = tl.minimum(tl.maximum(nc1 - rej1, 0), nc1)
        v2 = tl.minimum(tl.maximum(nc2 - rej2, 0), nc2)
        off = tl.where(req == 0, 0, tl.where(req == 1, v0, tl.where(req == 2, v0 + v1,
                                                                    v0 + v1 + v2)))
        rej = tl.load(num_rejected + req).to(tl.int64)
        valid = tl.minimum(tl.maximum(nc - rej, 0), nc)
        is_valid = r < valid
        src = st + r
        pos = tl.load(positions + src)
        dest = tl.where(is_valid, off + r, dump_stage)
        rdest = tl.where(is_valid, sl * W + pos % W, dump_ring)
        cols = cb * BLOCK + tl.arange(0, BLOCK)
        m = cols < H
        if t == 0:
            x = tl.load(tap0 + src * s0 + cols, mask=m)
        elif t == 1:
            x = tl.load(tap1 + src * s1 + cols, mask=m)
        else:
            x = tl.load(tap2 + src * s2 + cols, mask=m)
        out_cols = t * H + cols
        tl.store(stage + dest * (3 * H) + out_cols, x, mask=m)
        tl.store(ring + rdest * (3 * H) + out_cols, x, mask=m)
        # per-request metadata from the request's first candidate row
        if (r == 0) & (cb == 0) & (t == 0):
            first = tl.load(positions + st)
            ns = tl.load(num_sampled + req)
            a = tl.where(ns > 0, tl.load(last_sampled + sl),
                         tl.load(next_prefill + sl).to(tl.int64))
            tl.store(meta + req, valid)
            tl.store(meta + meta_stride + req, first)
            tl.store(meta + 2 * meta_stride + req, a)
            tl.store(meta + 3 * meta_stride + req, (ns > 0).to(tl.int64))


def available() -> bool:
    return triton is not None


def fused_enqueue(g, bv, taps, positions, num_rejected, num_sampled, last_sampled,
                  next_prefill_tokens, nc, cand_start, R: int) -> None:
    """Device work of RowGatherer.enqueue in one launch (same buffers, same results)."""
    n = bv.n
    if n > MAX_REQ:
        raise ValueError(f"fused gather supports at most {MAX_REQ} requests")
    H = g.hidden
    t16 = []
    strides = []
    for t in taps:
        if t.dtype != torch.bfloat16 or t.shape[1] != H or t.stride(1) != 1:
            raise TypeError("fused gather needs bf16 taps [T, H] with unit column stride")
        t16.append(t.view(torch.int16))
        strides.append(t.stride(0))
    pad = [0] * (MAX_REQ - n)
    ncs = [int(x) for x in nc] + pad
    sts = [int(x) for x in cand_start] + pad
    sls = [int(x) for x in bv.slots] + pad
    BLOCK = 1024
    grid = (R, triton.cdiv(H, BLOCK), 3)
    _gather_kernel[grid](
        t16[0], t16[1], t16[2], strides[0], strides[1], strides[2],
        positions, num_rejected, num_sampled, last_sampled.reshape(-1), next_prefill_tokens[0],
        g.stage.view(torch.int16), g.ring.view(torch.int16), g.meta_dev, g.meta_dev.stride(0),
        n, *ncs, *sts, *sls, g.max_rows, g.ring_dump,
        H=H, W=g.window, BLOCK=BLOCK,
    )
