"""OFF-by-default top32 proposal primitive. No serving import on the greedy path.

q is dyadic24-floor-first-v1: FP32 exp, FP64 normalization, floor 2**24
tickets, remaining tickets to the largest logit. The integer draw samples
exactly the emitted q, including zero tickets. This is not unrounded softmax.
Two launches: parallel tile selection, then fused global selection/tau/q/draw.
Inputs are already Markov-conditioned logits; never transfer full logits to host.
"""
from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

K = 32
MASS = 1 << 24
SEMANTIC = "dyadic24-floor-first-v1"


@triton.jit
def _pack(x: tl.tensor, ids: tl.tensor) -> tl.tensor:
    bits = x.to(tl.uint32, bitcast=True)
    ordered = tl.where((bits >> 31) != 0, ~bits, bits ^ 0x80000000)
    return (ordered.to(tl.uint64) << 32) | (0xffffffff - ids.to(tl.uint32)).to(tl.uint64)


@triton.jit
def _unpack(key: tl.tensor) -> tl.tensor:
    ordered = (key >> 32).to(tl.uint32)
    bits = tl.where((ordered >> 31) != 0, ordered ^ 0x80000000, ~ordered)
    return bits.to(tl.float32, bitcast=True), (0xffffffff - key.to(tl.uint32)).to(tl.int32)


@triton.jit
def _tiles(X: tl.tensor, C: tl.tensor, E: tl.tensor, V: tl.constexpr, S: tl.constexpr, NT: tl.constexpr, TILE: tl.constexpr) -> None:
    b, t = tl.program_id(0), tl.program_id(1)
    j = tl.arange(0, TILE)
    ids = t * TILE + j
    x = tl.load(X + b * S + ids, ids < V, float("-inf")).to(tl.float32)
    bad = tl.sum((tl.extra.cuda.libdevice.isnan(x) | (x == float("inf"))).to(tl.int32), 0)
    tl.store(E + b * NT + t, bad)
    # Canonicalize signed zero for deterministic lower-ID ties.
    x = tl.where(x == 0, 0., x)
    keys = tl.topk(_pack(x, ids), 32)
    r = tl.arange(0, 32)
    tl.store(C + (b * NT + t) * 32 + r, keys)


@triton.jit
def _finish(C: tl.tensor, KEYS: tl.tensor, TEMP: tl.tensor, TOK: tl.tensor, IDS: tl.tensor, Q: tl.tensor, ERR: tl.tensor, FLAGS: tl.tensor,
            NC: tl.constexpr, PAD: tl.constexpr) -> None:
    b = tl.program_id(0)
    j = tl.arange(0, PAD)
    key = tl.load(C + b * NC + j, j < NC, 0)
    top = tl.topk(key, 32)
    r = tl.arange(0, 32)
    x, ids = _unpack(top)
    tau = tl.load(TEMP + b)
    valid = (tau > 0) & (tau < float("inf")) & (tl.max(x, 0) > float("-inf"))
    mx = tl.max(x, 0)
    w = tl.exp((x - mx) / tl.where(valid, tau, 1.)).to(tl.float64)
    counts = tl.floor(w / tl.sum(w, 0) * 16777216.).to(tl.int32)
    left = 16777216 - tl.sum(counts, 0)
    flags = tl.load(FLAGS + b * (NC // 32) + j, j < NC // 32, 0)
    valid = valid & (left >= 0) & (left < 32) & (tl.sum(flags, 0) == 0)
    counts += tl.where(r == 0, left, 0)
    # Canonical support ID order; ticket intervals use exactly these counts.
    packed = (ids.to(tl.uint64) << 32) | counts.to(tl.uint32).to(tl.uint64)
    packed = tl.sort(packed, descending=False)
    ids = (packed >> 32).to(tl.int32)
    counts = packed.to(tl.int32)
    token = tl.full((), -1, tl.int32)
    if valid:
        seed = tl.load(KEYS + b * 3).to(tl.uint64)
        rnd = tl.load(KEYS + b * 3 + 1).to(tl.uint64)
        pos = tl.load(KEYS + b * 3 + 2).to(tl.uint64)
        u, _, _, _ = tl.philox(seed, rnd.to(tl.uint32), (rnd >> 32).to(tl.uint32),
                              pos.to(tl.uint32), (pos >> 32).to(tl.uint32) ^ 0x50503301, 10)
        ticket = (u & 0xffffff).to(tl.int32)
        cdf = tl.cumsum(counts, 0)
        chosen = tl.min(tl.where(cdf > ticket, r, 32), 0)
        token = tl.sum(tl.where(r == chosen, ids, 0), 0)
    tl.store(TOK + b, tl.where(valid, token, -1))
    tl.store(IDS + b * 32 + r, ids)
    tl.store(Q + b * 32 + r, tl.where(valid, counts.to(tl.float32) / 16777216., 0.))
    tl.store(ERR + b, tl.where(valid, 0, 1))


class Top32:
    """Reusable fixed-shape buffers, CUDA-graph safe. KEYS [B,3] int64 holds
    request-generation seed, round, absolute position. Caller owns uniqueness.
    Error/token -1 must be checked before accepting an entire proposal reply.
    This primitive does not implement pair communication or Markov evaluation.
    """

    def __init__(self, batch: int, vocab: int, device: torch.device, tile: int = 1024):
        if not 1 <= batch <= 4 or vocab < K or tile not in (256, 512, 1024, 2048):
            raise ValueError("top32 needs B1..B4, vocab>=32 and a supported tile")
        self.batch, self.vocab, self.tile = batch, vocab, tile
        self.tiles = triton.cdiv(vocab, tile)
        self.candidates = torch.empty(batch, self.tiles * K, dtype=torch.uint64, device=device)
        self.tokens = torch.empty(batch, dtype=torch.int32, device=device)
        self.ids = torch.empty(batch, K, dtype=torch.int32, device=device)
        self.q = torch.empty(batch, K, dtype=torch.float32, device=device)
        self.error = torch.zeros(batch, dtype=torch.int32, device=device)
        self.flags = torch.empty(batch, self.tiles, dtype=torch.int32, device=device)

    def __call__(self, logits: torch.Tensor, keys: torch.Tensor,
                 temperatures: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (logits.shape != (self.batch, self.vocab) or logits.stride(1) != 1
                or logits.dtype not in (torch.bfloat16, torch.float32)
                or keys.shape != (self.batch, 3) or keys.dtype != torch.int64 or not keys.is_contiguous()
                or temperatures.shape != (self.batch,) or temperatures.dtype != torch.float32
                or not temperatures.is_contiguous()
                or any(t.device != self.tokens.device for t in (logits, keys, temperatures))):
            raise ValueError("top32 tensor ABI mismatch")
        _tiles[(self.batch, self.tiles)](logits, self.candidates, self.flags, self.vocab,
                                       logits.stride(0), self.tiles, self.tile, num_warps=4)
        _finish[(self.batch,)](self.candidates, keys, temperatures, self.tokens, self.ids,
                              self.q, self.error, self.flags, self.tiles * K,
                              triton.next_power_of_2(self.tiles * K), num_warps=4)
        return self.tokens, self.ids, self.q


def select_position(model: Any, ranks: Any, logits: dict[int, torch.Tensor],
                    previous: dict[int, torch.Tensor], outputs: dict[int, torch.Tensor],
                    position: int, context: dict[str, Any]) -> None:
    """Eager TP2 integration. Peer-copy completion uses CUDA events, never host
    logits. This deliberately charges full-logit GPU peer staging; the candidate
    exchange optimization requires pair qualification before a serving trial.
    """
    if tuple(ranks) != model.ALL:
        raise RuntimeError("PP3 requires both logical ranks; per-rank graph capture is unsupported")
    parts = []
    for r in ranks:
        rk = model.ranks[r]
        with torch.cuda.device(rk.dev):
            pv = previous[r].contiguous()
            bias = rk.markov.bias(pv, rk.markov.embed(pv))
            part = (logits[r][:, position] + bias).to(torch.bfloat16)
            ready = torch.cuda.Event()
            ready.record()
        with torch.cuda.device(model.dev):
            torch.cuda.current_stream().wait_event(ready)
            parts.append(part.to(model.dev, non_blocking=True))
    with torch.cuda.device(model.dev):
        keys = context["keys"].clone()
        keys[:, 2] += position
        # Zero-temperature requests retain their original greedy proposal and
        # do not enter the proposal RNG branch, even for a mixed batch.
        temperatures = torch.where(context["mask"], .7, 0.).float()
        token, ids, q = context["sampler"](torch.cat(parts, -1), keys, temperatures)
        chosen = torch.where(context["mask"], token.long(), outputs[0][:, position])
        context["ids"].append(ids.clone())
        context["q"].append(q.clone() * context["mask"][:, None])
        context["errors"].append(context["sampler"].error.bool().clone())
        ready = torch.cuda.Event()
        ready.record()
    for r in ranks:
        with torch.cuda.device(model.ranks[r].dev):
            torch.cuda.current_stream().wait_event(ready)
            outputs[r][:, position].copy_(chosen, non_blocking=True)
