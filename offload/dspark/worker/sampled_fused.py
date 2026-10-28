"""Single-launch Markov/top32/dyadic24 selector for the explicit experiment.

Each tile publishes 32 candidates. The last retiring tile reduces them and
draws, so no block waits for unscheduled blocks. On TP2 each last tile exchanges
only 32 candidates with its peer, with bounded release/acquire and consumption
acknowledgment. Emulation reduces both shards in the same launch. No conditioned
vocabulary tensor is allocated. The original head logits remain BF16.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
import triton
import triton.language as tl

from .sampled import _pack, _unpack


@triton.jit
def _clock() -> tl.tensor:
    return tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", constraints="=l", args=[],
                                     dtype=tl.uint64, is_pure=False, pack=1)


@triton.jit
def _acquire(p: tl.tensor) -> tl.tensor:
    return tl.inline_asm_elementwise("ld.acquire.sys.global.u32 $0, [$1];", constraints="=r,l", args=[p],
                                     dtype=tl.uint32, is_pure=False, pack=1)


@triton.jit
def _release(p: tl.tensor, v: tl.tensor) -> None:
    tl.inline_asm_elementwise("st.release.sys.global.u32 [$1], $2; mov.u32 $0, 0;",
                               constraints="=r,l,r", args=[p, v.to(tl.uint32)], dtype=tl.uint32,
                               is_pure=False, pack=1)


@triton.jit
def _wait(p: tl.tensor, epoch: tl.tensor, error: tl.tensor, timeout: tl.constexpr) -> tl.tensor:
    start = _clock()
    value = _acquire(p)
    err = _acquire(error)
    while (value != epoch) & (err == 0) & (_clock() - start < timeout):
        value = _acquire(p)
        err = _acquire(error)
    return (value == epoch) & (err == 0)


@triton.jit
def _kernel(L0: tl.tensor, L1: tl.tensor, P0: tl.tensor, P1: tl.tensor,
            S0: tl.tensor, S1: tl.tensor, C0: tl.tensor, C1: tl.tensor,
            W10: tl.tensor, W11: tl.tensor, W20: tl.tensor, W21: tl.tensor,
            KEYS: tl.tensor, MASK: tl.tensor, CAND: tl.tensor, BAD: tl.tensor,
            DONE: tl.tensor, EPOCH: tl.tensor, FLAGS: tl.tensor, ACK: tl.tensor, STICKY: tl.tensor,
            INBOX: tl.tensor, PEER_INBOX: tl.tensor, PEER_FLAGS: tl.tensor,
            PEER_ACK: tl.tensor, PEER_ERROR: tl.tensor,
            OUT0: tl.tensor, OUT1: tl.tensor, IDS: tl.tensor, Q: tl.tensor, ERROR: tl.tensor,
            V: tl.constexpr, V_SECOND: tl.constexpr, V0: tl.constexpr, V_SECOND_OFFSET: tl.constexpr,
            LS: tl.constexpr, LS_SECOND: tl.constexpr, PS: tl.constexpr,
            CS: tl.constexpr, CS_SECOND: tl.constexpr, WS: tl.constexpr, RANK_DIM: tl.constexpr,
            DEPTH: tl.constexpr, POS: tl.constexpr, NT: tl.constexpr,
            PAD: tl.constexpr, EMULATED: tl.constexpr, TILE: tl.constexpr,
            TIMEOUT: tl.constexpr) -> None:
    b, tile = tl.program_id(0), tl.program_id(1)
    shard = tile // NT
    t = tile % NT
    if EMULATED and shard == 1:
        base, prev, slots, cache, w1, w2 = L1, P1, S1, C1, W11, W21
        vocabulary, base_stride, cache_stride, offset = V_SECOND, LS_SECOND, CS_SECOND, V_SECOND_OFFSET
    else:
        base, prev, slots, cache, w1, w2 = L0, P0, S0, C0, W10, W20
        vocabulary, base_stride, cache_stride, offset = V, LS, CS, V0
    tok = tl.load(prev + b * PS).to(tl.int64)
    slot = tl.load(slots + tok).to(tl.int64)
    j = tl.arange(0, TILE)
    cols = t * TILE + j
    if slot >= 0:
        lg = tl.load(base + b * base_stride + cols, cols < vocabulary, 0.).to(tl.float32)
        bias = tl.load(cache + slot * cache_stride + cols, cols < vocabulary, 0.).to(tl.float32)
        cached_x = (lg + bias).to(tl.bfloat16).to(tl.float32)
        bad = tl.sum(((tl.extra.cuda.libdevice.isnan(cached_x) | (cached_x == float("inf"))) & (cols < vocabulary)).to(tl.int32), 0)
        cached_x = tl.where(cols < vocabulary, cached_x, float("-inf"))
        cached_x = tl.where(cached_x == 0, 0., cached_x)
        top = tl.topk(_pack(cached_x, cols + offset), 32)
    else:
        # Preserve the production per-column FP32 tree reduction and BF16 bias.
        # A small sub-tile bounds register use even on the uncached path.
        rr = tl.arange(0, RANK_DIM)
        emb = tl.load(w1 + tok * RANK_DIM + rr).to(tl.float32)
        top = tl.full((32,), 0, tl.uint64)
        bad = tl.full((), 0, tl.int32)
        sj = tl.arange(0, 64)
        for sub in range(TILE // 64):
            scols = t * TILE + sub * 64 + sj
            weight = tl.load(w2 + scols[:, None].to(tl.int64) * WS + rr[None, :],
                             scols[:, None] < vocabulary, 0.).to(tl.float32)
            ubias = tl.sum(weight * emb[None, :], 1).to(tl.bfloat16).to(tl.float32)
            ul = tl.load(base + b * base_stride + scols, scols < vocabulary, 0.).to(tl.float32)
            ux = (ul + ubias).to(tl.bfloat16).to(tl.float32)
            bad += tl.sum(((tl.extra.cuda.libdevice.isnan(ux) | (ux == float("inf"))) & (scols < vocabulary)).to(tl.int32), 0)
            ux = tl.where(scols < vocabulary, ux, float("-inf"))
            ux = tl.where(ux == 0, 0., ux)
            partial = tl.topk(_pack(ux, scols + offset), 32)
            top = tl.topk(tl.reshape(tl.join(top, partial), (64,)), 32)
    r = tl.arange(0, 32)
    count: tl.constexpr = NT * (2 if EMULATED else 1)
    tl.store(CAND + (b * count + tile) * 32 + r, top)
    tl.store(BAD + b * count + tile, bad)
    # Every tile's payload precedes its acq_rel retirement. Only the final
    # retiring CTA continues: no grid barrier or assumption about scheduling.
    tl.inline_asm_elementwise("fence.sc.gpu; mov.u64 $0, $1;", constraints="=l,l", args=[top],
                               dtype=tl.uint64, is_pure=False, pack=1)
    tl.debug_barrier()
    retired = tl.atomic_add(DONE + b, 1, sem="acq_rel", scope="gpu")
    if retired == count - 1:
        jj = tl.arange(0, PAD)
        keys = tl.load(CAND + b * count * 32 + jj, jj < count * 32, 0)
        local = tl.topk(keys, 32)
        errs = tl.load(BAD + b * count + jj, jj < count, 0)
        valid = tl.sum(errs, 0) == 0
        if not EMULATED:
            epoch = tl.load(EPOCH + b).to(tl.uint32)
            peer_inbox = tl.cast(PEER_INBOX, tl.pointer_type(tl.uint64))
            peer_flags = tl.cast(PEER_FLAGS, tl.pointer_type(tl.uint32))
            peer_ack = tl.cast(PEER_ACK, tl.pointer_type(tl.uint32))
            peer_error = tl.cast(PEER_ERROR, tl.pointer_type(tl.uint32))
            if not valid:
                tl.atomic_or(STICKY + b, 2, sem="relaxed", scope="sys")
                tl.atomic_or(peer_error + b, 2, sem="relaxed", scope="sys")
            healthy = (_acquire(STICKY + b) == 0) & (epoch != 0xffffffff)
            if healthy:
                tl.store(peer_inbox + b * 32 + r, local)
                # Release flag follows all payload lanes' system fences.
                tl.inline_asm_elementwise("fence.sc.sys; mov.u64 $0, $1;", constraints="=l,l", args=[local],
                                           dtype=tl.uint64, is_pure=False, pack=1)
                tl.debug_barrier()
                _release(peer_flags + b, epoch)
                healthy = _wait(FLAGS + b, epoch, STICKY + b, TIMEOUT)
            if healthy:
                merge_j = tl.arange(0, 64)
                own = tl.gather(local, merge_j % 32, 0)
                peer = tl.load(INBOX + b * 32 + merge_j % 32, volatile=True)
                local = tl.topk(tl.where(merge_j < 32, own, peer), 32)
            else:
                tl.atomic_or(STICKY + b, 1, sem="relaxed", scope="sys")
                tl.atomic_or(peer_error + b, 1, sem="relaxed", scope="sys")
            valid = valid & healthy
        selected_logits, support = _unpack(local)
        positive = tl.load(MASK + b)
        valid = valid & (tl.max(selected_logits, 0) > float("-inf"))
        mx = tl.max(selected_logits, 0)
        w = tl.exp((selected_logits - mx) / .7).to(tl.float64)
        tickets = tl.floor(w / tl.sum(w, 0) * 16777216.).to(tl.int32)
        left = 16777216 - tl.sum(tickets, 0)
        valid = valid & (left >= 0) & (left < 32)
        tickets += tl.where(r == 0, left, 0)
        token = tl.sum(tl.where(r == 0, support, 0), 0)
        packed = tl.sort((support.to(tl.uint64) << 32) | tickets.to(tl.uint32).to(tl.uint64), descending=False)
        support = (packed >> 32).to(tl.int32)
        tickets = packed.to(tl.int32)
        if positive & valid:
            seed = tl.load(KEYS + b * 3).to(tl.uint64)
            rnd = tl.load(KEYS + b * 3 + 1).to(tl.uint64)
            pos = tl.load(KEYS + b * 3 + 2).to(tl.uint64) + POS
            u, _, _, _ = tl.philox(seed, rnd.to(tl.uint32), (rnd >> 32).to(tl.uint32),
                                  pos.to(tl.uint32), (pos >> 32).to(tl.uint32) ^ 0x50503301, 10)
            ticket = (u & 0xffffff).to(tl.int32)
            chosen = tl.min(tl.where(tl.cumsum(tickets, 0) > ticket, r, 32), 0)
            token = tl.sum(tl.where(r == chosen, support, 0), 0)
        # Invalid rows retain an error flag and are discarded by the host.
        # Use an in-range internal continuation so a later sequential Markov
        # position cannot dereference token -1 before that host check.
        tl.store(OUT0 + b * DEPTH + POS, tl.where(positive & ~valid, 0, token))
        if EMULATED:
            tl.store(OUT1 + b * DEPTH + POS, tl.where(positive & ~valid, 0, token))
        tl.store(IDS + (b * DEPTH + POS) * 32 + r, support)
        tl.store(Q + (b * DEPTH + POS) * 32 + r,
                 tl.where(positive & valid, tickets.to(tl.float32) / 16777216., 0.))
        tl.store(ERROR + b * DEPTH + POS, (positive & ~valid).to(tl.int32))
        if not EMULATED:
            if healthy:
                tl.debug_barrier()
                _release(peer_ack + b, epoch)
                healthy = _wait(ACK + b, epoch, STICKY + b, TIMEOUT)
            if not healthy:
                tl.atomic_or(STICKY + b, 1, sem="relaxed", scope="sys")
                tl.atomic_or(peer_error + b, 1, sem="relaxed", scope="sys")
                tl.store(ERROR + b * DEPTH + POS, 1)
            tl.store(EPOCH + b, epoch + 1)
        tl.store(DONE + b, 0)


class FusedContext:
    """Fixed-address per-shape state. Own it for one worker, never concurrently.

    Peer access comes from the worker's existing qualified P2P communicator.
    Sticky peer errors require worker teardown; they are never cleared on retry.
    """

    def __init__(self, model: Any, batch: int, depth: int, tile: int = 1024):
        if tuple(model.ALL) != (0, 1) or not 1 <= batch <= 4 or depth not in (4, 5):
            raise ValueError("fused sampled selection requires full K2 TP2, B1..4, depth4/5")
        if tile not in (512, 1024, 2048):
            raise ValueError("sampled tile must be 512, 1024 or 2048")
        self.model, self.batch, self.depth, self.tile = model, batch, depth, tile
        self.state: dict[int, dict[str, torch.Tensor]] = {}
        self.primed: set[tuple[int, int, int, int]] = set()
        self.capture_expected: frozenset[int] = frozenset()
        v = max(rk.v1 - rk.v0 for rk in model.ranks)
        self.nt = triton.cdiv(v, tile)
        for rank in model.ALL:
            dev = model.devs[rank]
            count = self.nt * (2 if model.emulated else 1)
            self.state[rank] = {
                "keys": torch.zeros(batch, 3, dtype=torch.int64, device=dev),
                "mask": torch.ones(batch, dtype=torch.bool, device=dev),
                "cand": torch.empty(batch, count * 32, dtype=torch.uint64, device=dev),
                "bad": torch.empty(batch, count, dtype=torch.int32, device=dev),
                "done": torch.zeros(batch, dtype=torch.int32, device=dev),
                "epoch": torch.ones(batch, dtype=torch.int32, device=dev),
                "flags": torch.zeros(batch, dtype=torch.int32, device=dev),
                "ack": torch.zeros(batch, dtype=torch.int32, device=dev),
                "sticky": torch.zeros(batch, dtype=torch.int32, device=dev),
                "inbox": torch.empty(batch, 32, dtype=torch.uint64, device=dev),
                "ids": torch.zeros(batch, depth, 32, dtype=torch.int32, device=dev),
                "q": torch.zeros(batch, depth, 32, dtype=torch.float32, device=dev),
                "error": torch.zeros(batch, depth, dtype=torch.int32, device=dev),
            }

    @contextmanager
    def capturing(self, ranks: tuple[int, ...]) -> Iterator[None]:
        """Declare capture intent independently of CUDA's current-stream state.

        A graph accidentally capturing another device must never launch a peer
        kernel eagerly: its absent partner would poison both selector states.
        """
        if self.capture_expected:
            raise RuntimeError("nested sampled capture is not supported")
        self.capture_expected = frozenset((0,) if self.model.emulated else ranks)
        try:
            yield
        finally:
            self.capture_expected = frozenset()

    def update(self, keys: torch.Tensor, mask: torch.Tensor) -> None:
        if keys.shape != (self.batch, 3) or keys.dtype != torch.int64 or mask.shape != (self.batch,) or mask.dtype != torch.bool:
            raise ValueError("fused sampled RNG/mask ABI mismatch")
        for r, state in self.state.items():
            with torch.cuda.device(self.model.devs[r]):
                state["keys"].copy_(keys, non_blocking=True)
                state["mask"].copy_(mask, non_blocking=True)

    def select(self, ranks: Any, logits: dict[int, torch.Tensor], previous: dict[int, torch.Tensor],
               outputs: dict[int, torch.Tensor], position: int) -> None:
        model = self.model
        active = (0,) if model.emulated else ranks
        if self.capture_expected:
            if frozenset(active) != self.capture_expected:
                raise RuntimeError("sampled capture rank ownership differs")
            for r in active:
                with torch.cuda.device(model.devs[r]):
                    if not torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(f"sampled rank {r}: expected capture on its current device stream; refusing eager peer launch")
        prepared: list[tuple[int, tuple[Any, ...], tuple[int, int, int, int]]] = []
        for r in active:
            rk, other = model.ranks[r], model.ranks[1 if model.emulated else r]
            state, peer = self.state[r], self.state[1 - r]
            v = rk.v1 - rk.v0
            second = 1 if model.emulated else r
            base = logits[r][:, position]
            args = (
                    base, logits[second][:, position], previous[r], previous[second],
                    rk.markov.slot_of_token, other.markov.slot_of_token, rk.markov.cache, other.markov.cache,
                    rk.markov.w1, other.markov.w1, rk.markov.w2, other.markov.w2,
                    state["keys"], state["mask"], state["cand"], state["bad"], state["done"], state["epoch"],
                    state["flags"], state["ack"], state["sticky"], state["inbox"],
                    peer["inbox"].data_ptr(), peer["flags"].data_ptr(), peer["ack"].data_ptr(), peer["sticky"].data_ptr(),
                    outputs[r], outputs[second], state["ids"], state["q"], state["error"],
                    v, other.v1 - other.v0, rk.v0, other.v0, base.stride(0), logits[second][:, position].stride(0),
                    previous[r].stride(0), rk.markov.cache.stride(0), other.markov.cache.stride(0), rk.markov.w2.stride(0),
                    rk.markov.w1.shape[1], self.depth, position, self.nt,
                    triton.next_power_of_2(self.nt * (2 if model.emulated else 1) * 32),
                    model.emulated, self.tile, 50_000_000)
            prepared.append((r, args, (r, position, base.stride(0), previous[r].stride(0))))
        grid = (self.batch, self.nt * (2 if model.emulated else 1))
        # A first rank must never wait on the GPU while the other rank JITs.
        # Compile both specializations before launching either. Warmups before
        # _capture_one prime every position; capture itself cannot compile.
        kernel: Any = _kernel
        if not model.emulated:
            for r, args, key in prepared:
                if key not in self.primed:
                    with torch.cuda.device(model.devs[r]):
                        if torch.cuda.is_current_stream_capturing() or tuple(ranks) != tuple(model.ALL):
                            raise RuntimeError("paired sampled kernels must be primed together before capture")
                        compiled = kernel.warmup(*args, grid=grid, num_warps=4, enable_fp_fusion=False)
                        # HYPOTHESIS (PP4 Round8): compile-only warmup leaves
                        # Triton's CUDA handles lazy. Resolve both devices'
                        # launchers before either peer can wait on the GPU.
                        # Reading CompiledKernel.run loads, but never launches.
                        if compiled is None or not callable(compiled.run):
                            raise RuntimeError("paired sampled warmup did not load CUDA launch handles")
                    self.primed.add(key)
        for r, args, _ in prepared:
            with torch.cuda.device(model.devs[r]):
                kernel[grid](*args, num_warps=4, enable_fp_fusion=False)

    def result(self) -> tuple[torch.Tensor, torch.Tensor]:
        for state in self.state.values() if not self.model.emulated else (self.state[0],):
            if torch.any(state["error"]).item() or torch.any(state["sticky"]).item():
                raise RuntimeError("fused sampled selector failed; discard reply and tear down worker")
        return self.state[0]["ids"], self.state[0]["q"]
