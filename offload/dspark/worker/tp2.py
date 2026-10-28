"""The DSpark drafter split over x86host's NVLink pair (TP = 2; lane E). Same checkpoint tensors, same semantics as the
one-card model.py (NOTES.md); the split follows the Spark's own TP = 2 layout where the work is bandwidth bound:

  rank r owns                                   replicated on both ranks
  ------------------------------------------    ---------------------------------------------------------------
  head + Markov bias: vocabulary [v_r, v_r+1)   embedding, mHC mixes / collapse / post, all norms, RoPE
    (253 / 252 tiles of 256 columns)            wq_a, wkv, window-KV ring (inserted on both), block KV
  wq_b + attention + sink: heads 32r..32r+31    routing (gate, top-3), Markov embedding, confidence head (rank 0)
  wo_a: groups 4r..4r+3 (FP8 bytes, scales)     main_proj + main_norm (context insert)
  wo_b: input rows 4096r..4096r+4095
  routed experts: intermediate channels 1152r.. (Exl3Bank tp_size 2, the Spark's own cut)
  shared expert: w1/w3 columns, w2 rows (1152r..)

Collectives per draft step (p2p.cu, one-shot push over NVLink): per stage one all-reduce of the attention output
partials (wo_b) and one of the FFN partials (routed + shared), fp32 sums rounded to bf16 (identical bits on both
ranks: a + b is commutative); per Markov step one argmax exchange (ties -> lower id = rank 0). Every split is at a
128-channel boundary of the EXL3 Hadamard blocks, so each shard's decoded weights are the full tensor's decoded
weights bit for bit (tests/test_tp2.py); only accumulation order changes (draft-only, like the 3090 kernels already).

Execution: one process, single controller. Every op is issued for each active rank in turn (under that rank's device),
so eager runs are lockstep and a collective's two halves are issued back to back. CUDA graphs are captured per rank by
running the same code with only that rank active (capture records, nothing executes, so the collectives need no
partner at capture time); a step replays rank 0's and rank 1's graph. With both ranks on ONE device (emulation, for
development and parity on a single card) the collectives are plain torch adds / compares (bit-identical to p2p.cu's
combine) and one graph holds both ranks.

EXL3 kernels: every exl3_gemm runs with a forced (shape, num_sms) (tp2_kernels.json, from tools/tp2_tune.py; default
exllamav3's static shape heuristic): the timing-based autotuner could pick different kernels per card and break the
bit-identity of the replicated activations, and it cannot run inside a graph capture.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from . import env  # noqa: F401
from . import fused as FU
from . import fused_e as FE
from .checkpoint import Checkpoint, DrafterShape
from .exl3 import kernels as K
from .exl3.bank import Exl3Bank
from .exl3.plan import PROJ_DOWN, PROJ_GATE, PROJ_UP, Exl3MoEPlan
from .model import dequant_wo_a, facc_ext
from .ops import MarkovBias, markov_bias_rows, rope_cos_sin

TP = 2
GPU_UUIDS = ("GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93", "GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc")
HEAD_TILE = 256                      # the head's kernel shape 3 tile width; vocabulary shards are whole tiles
KERNEL_PLAN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tp2_kernels.json")
SOLO_UUID = GPU_UUIDS[1]             # lane E's own card until the coordinator frees the pair


def check_visible(pair: bool | None = None) -> str:
    """COMMON rule 3: lane E runs on GPU-7744b44b alone, or on the NVLink pair (c4e7a371 first) once freed. Anything
    else (including an unset CUDA_VISIBLE_DEVICES, which a detached tmux job silently gets) refuses to start.
    pair: True = require the pair, False = require the single card, None = either. Returns the value."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ok = {False: (SOLO_UUID,), True: (",".join(GPU_UUIDS),), None: (SOLO_UUID, ",".join(GPU_UUIDS))}[pair]
    if vis not in ok:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={vis!r}: lane E runs only on {' or '.join(ok)} (COMMON rule 3)")
    return vis


# --------------------------------------------------------------------------------------------- EXL3 matrices


def m_bucket(m: int) -> int:
    """Row-count key of the kernel plan: exact up to 32 rows (the draft body runs 5 x B rows), then powers of two."""
    if m <= 32:
        return m
    for b in (64, 128):
        if m <= b:
            return b
    return 256


class KernelPlan:
    """(shape, num_sms) per (in, out, K, row bucket): forced on both ranks. Missing entries: (0, all SMs), i.e.
    exllamav3's static select_gemm_shape heuristic without the autotuner (deterministic)."""

    def __init__(self, path: str | None = KERNEL_PLAN, n_sms: int = 82):
        self.n_sms = n_sms
        self.table: dict = {}
        self.path = path
        if path and os.path.exists(path):
            with open(path) as f:
                self.table = json.load(f).get("kernels", {})
        self.used: dict = {}

    @staticmethod
    def key(k: int, n: int, K_: int) -> str:
        return f"{k}x{n}xK{K_}"

    def pick(self, k: int, n: int, K_: int, m: int, force_shape: int = 0, bucket: str | None = None
             ) -> tuple[int, int]:
        key, b = self.key(k, n, K_), (bucket or str(m_bucket(m)))
        e = self.table.get(key, {}).get(b)
        sh, ns = (int(e[0]), int(e[1])) if e else (force_shape, self.n_sms)
        self.used[f"{key}@{b}"] = [sh, ns]
        return sh, ns


class Exl3Mat:
    """One EXL3 matrix or a 128-aligned shard of one: fp16 in, fp32 out through exl3_gemm with a fixed kernel."""

    def __init__(self, trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, k: int, dev: torch.device,
                 plan: KernelPlan, ext=None, force_shape: int = 0, name: str = "", fixed_bucket: str | None = None):
        self.in_features, self.out_features, self.k = trellis.shape[0] * 16, trellis.shape[1] * 16, k
        self.trellis = trellis.contiguous().to(dev)
        self.suh = suh.contiguous().to(dev)
        self.svh = svh.contiguous().to(dev)
        if ext is None:
            import exllamav3_ext as ext
        self.ext = ext
        self.plan = plan
        self.force_shape = force_shape
        self.name = name
        self.fixed_bucket = fixed_bucket      # one kernel for every row count (the head: n_out bit-identity)
        self._pick: dict = {}

    def gemm(self, x16: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        m = x16.shape[0]
        b = self.fixed_bucket or m_bucket(m)
        p = self._pick.get(b)
        if p is None:
            p = self.plan.pick(self.in_features, self.out_features, self.k, m, self.force_shape,
                               self.fixed_bucket)
            self._pick[b] = p
        if out is None:
            out = torch.empty(m, self.out_features, dtype=torch.float32, device=x16.device)
        self.ext.exl3_gemm(x16, self.trellis, out, self.suh, torch.empty_like(x16), self.svh, p[0], False, True,
                           p[1])
        return out

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        if x.dtype != torch.float16:
            x = x.to(torch.float16)
        return self.gemm(x.contiguous(), out)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.trellis, self.suh, self.svh))


class Exl3Pair:
    """Two EXL3 matrices on the same input in ONE exllamav3 exl3_mgemm launch (one input, two outputs; per-matrix
    widths through size_n_list / c_ptrs): wq_a + wkv and the shared expert's w1 + w3. Saves 15-21 us per pair against
    two exl3_gemm launches (their fixed cooperative-launch cost); a different kernel, so different rounding: 0.12-0.13 %
    from fp64 instead of 0.10 % for the tuned single GEMMs (both inside the 0.05-0.30 % O2 measured for exl3_gemm;
    TP2.md). Forced shape (no autotune): 2 up to 16 rows, 5 above (32-row tiles). Outputs live in per-row-count
    buffers owned by the pair (their pointers are in the c_ptrs table, so they cannot move): valid until the pair's
    next call with the same row count, which stream order keeps after every consumer of this one."""

    def __init__(self, a: Exl3Mat, b: Exl3Mat):
        if a.in_features != b.in_features or a.k != b.k or a.trellis.device != b.trellis.device:
            raise ValueError(f"{a.name} / {b.name}: not one input, one K, one device")
        self.a, self.b = a, b
        self.dev = a.trellis.device
        self.ext = a.ext
        mk = lambda *ts: torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=self.dev)
        self.B = mk(a.trellis, b.trellis)
        self.S = mk(a.suh, b.suh)
        self.V = mk(a.svh, b.svh)
        self.widths = torch.tensor([a.out_features, b.out_features], dtype=torch.int32, device=self.dev)
        self._mk = mk
        self.bufs: dict = {}

    @staticmethod
    def shape_for(m: int) -> int:
        return 2 if m <= 16 else 5

    def __call__(self, x16: torch.Tensor):
        m = x16.shape[0]
        buf = self.bufs.get(m)
        if buf is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f"{self.a.name}+{self.b.name}: {m} rows first seen during graph capture")
            ya = torch.empty(m, self.a.out_features, dtype=torch.float32, device=self.dev)
            yb = torch.empty(m, self.b.out_features, dtype=torch.float32, device=self.dev)
            c = torch.empty(2, m, max(self.a.out_features, self.b.out_features), dtype=torch.float32, device=self.dev)
            xh = torch.empty(2, m, self.a.in_features, dtype=torch.float16, device=self.dev)
            buf = self.bufs[m] = (ya, yb, c, xh, self._mk(ya, yb))
        ya, yb, c, xh, cp = buf
        self.ext.exl3_mgemm(x16.view(1, m, self.a.in_features), self.B, c, self.S, xh, self.V, None, None,
                            float(self.a.k), self.shape_for(m), False, True, -1, -1, 0, 1, self.widths, cp)
        return ya, yb


def col_shard(tr: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, c0: int, c1: int):
    """Output columns [c0, c1) of an EXL3 matrix: trellis tile columns, svh; suh whole (128-aligned: exact)."""
    if c0 % 128 or c1 % 128:
        raise ValueError(f"column shard [{c0}, {c1}) not on the 128-wide Hadamard blocks")
    return tr[:, c0 // 16:c1 // 16], suh, svh[c0:c1]


def row_shard(tr: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, r0: int, r1: int):
    """Input rows [r0, r1): trellis tile rows, suh; svh whole (128-aligned: exact)."""
    if r0 % 128 or r1 % 128:
        raise ValueError(f"row shard [{r0}, {r1}) not on the 128-wide Hadamard blocks")
    return tr[r0 // 16:r1 // 16], suh[r0:r1], svh


def vocab_split(vocab: int, tp: int = TP) -> list[tuple[int, int]]:
    if tp == 1:
        return [(0, vocab)]
    tiles = vocab // HEAD_TILE
    if tiles * HEAD_TILE != vocab:
        raise ValueError(f"vocab {vocab} is not a whole number of {HEAD_TILE}-column head tiles")
    cuts = [round(tiles * r / tp) * HEAD_TILE for r in range(tp + 1)]
    cuts[0], cuts[-1] = 0, vocab
    # rank 0 takes the larger half (253 / 252 tiles): ties of the global argmax go to rank 0 = the lower ids
    if tp == 2:
        cuts[1] = (tiles + 1) // 2 * HEAD_TILE
    return [(cuts[r], cuts[r + 1]) for r in range(tp)]


# --------------------------------------------------------------------------------------------- Markov shard


class MarkovShard:
    """Markov bias over vocabulary columns [v0, v1): w2 / cached rows for those columns; the embedding (rows by
    previous token) and the token -> cached-row map whole. Cached rows are filled by the serve's kernel on the shard
    (per-column fp32 dot over the rank: the same bits as the full cache's columns, tests/test_tp2.py)."""

    def __init__(self, w1: torch.Tensor, w2_full: torch.Tensor, v0: int, v1: int, token_list, rows: int):
        self.w1 = w1
        self.w2 = w2_full[v0:v1].contiguous()
        self.v0, self.v1 = v0, v1
        full = MarkovBias.__new__(MarkovBias)          # reuse MarkovBias's cache fill on the shard's w2
        MarkovBias.__init__(full, w1, self.w2, token_list, rows=rows)
        self.slot_of_token, self.cache, self.cached_rows = full.slot_of_token, full.cache, full.cached_rows

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.w1[token_ids]

    def bias(self, token_ids: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        out = torch.empty((emb.shape[0], self.w2.shape[0]), dtype=torch.bfloat16, device=emb.device)
        return markov_bias_rows(token_ids, self.slot_of_token, self.cache, emb.contiguous(), self.w2, out)


@triton.jit
def _argmax_finish_vi_kernel(ids_ptr, slot_ptr, vals_c_ptr, idx_c_ptr, vals_u_ptr, idx_u_ptr, val_out, idx_out,
                             NBLK_C, NBLK_U, ids_stride, BLOCK: tl.constexpr):
    """fused._argmax_finish_kernel, writing this shard's (max value, local index) instead of the output column."""
    b = tl.program_id(0)
    tok = tl.load(ids_ptr + b * ids_stride).to(tl.int64)
    cached = tl.load(slot_ptr + tok) >= 0
    j = tl.arange(0, BLOCK)
    vc = tl.load(vals_c_ptr + b * NBLK_C + j, mask=j < NBLK_C, other=float("-inf"))
    vu = tl.load(vals_u_ptr + b * NBLK_U + j, mask=j < NBLK_U, other=float("-inf"))
    v = tl.where(cached, vc, vu)
    vm, jm = tl.max(v, axis=0, return_indices=True, return_indices_tie_break_left=True)
    if cached:
        d = tl.load(idx_c_ptr + b * NBLK_C + jm)
    else:
        d = tl.load(idx_u_ptr + b * NBLK_U + jm)
    tl.store(val_out + b, vm)
    tl.store(idx_out + b, d)


def markov_partials(base: torch.Tensor, prev: torch.Tensor, mb: MarkovShard):
    """The block maxima of fused.markov_argmax_into on one vocabulary shard (cached and uncached previous tokens)."""
    B, V = base.shape
    assert base.stride(1) == 1
    BC, BU, SUB = 2048, 64, 8
    nc, nu = triton.cdiv(V, BC), triton.cdiv(V, BU * SUB)
    dev = base.device
    vals_c = torch.empty(B, nc, dtype=torch.float32, device=dev)
    idx_c = torch.empty(B, nc, dtype=torch.int64, device=dev)
    vals_u = torch.empty(B, nu, dtype=torch.float32, device=dev)
    idx_u = torch.empty(B, nu, dtype=torch.int64, device=dev)
    FU._markov_cached_kernel[(B, nc)](prev, mb.slot_of_token, mb.cache, base, vals_c, idx_c, V, mb.cache.stride(0),
                                      base.stride(0), nc, prev.stride(0), BLOCK_V=BC, num_warps=8)
    FU._markov_uncached_kernel[(B, nu)](prev, mb.slot_of_token, mb.w1, mb.w2, base, vals_u, idx_u, V,
                                        mb.w2.stride(0), mb.w1.stride(0), base.stride(0), nu, prev.stride(0),
                                        R=mb.w1.shape[1], BLOCK_V=BU, SUB=SUB, num_warps=4)
    return vals_c, idx_c, vals_u, idx_u


def markov_local_argmax(base: torch.Tensor, prev: torch.Tensor, mb: MarkovShard) -> tuple[torch.Tensor, torch.Tensor]:
    """fused.markov_argmax_into on one vocabulary shard: (max of bf16(base + bias(prev)) over the shard, its local
    index; ties -> lowest) per row, as fp32 [B] / int64 [B]."""
    B, V = base.shape
    assert base.stride(1) == 1
    BC, BU, SUB = 2048, 64, 8
    nc, nu = triton.cdiv(V, BC), triton.cdiv(V, BU * SUB)
    dev = base.device
    vals_c = torch.empty(B, nc, dtype=torch.float32, device=dev)
    idx_c = torch.empty(B, nc, dtype=torch.int64, device=dev)
    vals_u = torch.empty(B, nu, dtype=torch.float32, device=dev)
    idx_u = torch.empty(B, nu, dtype=torch.int64, device=dev)
    FU._markov_cached_kernel[(B, nc)](prev, mb.slot_of_token, mb.cache, base, vals_c, idx_c, V, mb.cache.stride(0),
                                      base.stride(0), nc, prev.stride(0), BLOCK_V=BC, num_warps=8)
    FU._markov_uncached_kernel[(B, nu)](prev, mb.slot_of_token, mb.w1, mb.w2, base, vals_u, idx_u, V,
                                        mb.w2.stride(0), mb.w1.stride(0), base.stride(0), nu, prev.stride(0),
                                        R=mb.w1.shape[1], BLOCK_V=BU, SUB=SUB, num_warps=4)
    val = torch.empty(B, dtype=torch.float32, device=dev)
    idx = torch.empty(B, dtype=torch.int64, device=dev)
    _argmax_finish_vi_kernel[(B,)](prev, mb.slot_of_token, vals_c, idx_c, vals_u, idx_u, val, idx, nc, nu,
                                   prev.stride(0), BLOCK=triton.next_power_of_2(max(nc, nu)), num_warps=4)
    return val, idx


def coop_with_shared(x16: torch.Tensor, sel: torch.Tensor, rw16: torch.Tensor, groups: list, sc, act_limit: float,
                     sh: torch.Tensor, clone: bool) -> torch.Tensor:
    """kernels.run_decode_coop with the shared expert's output as the first group's sh_out: the coop kernel adds it
    to every stored row (exl3_moe_coop_kernel.cuh: o += 1.0f * sh after the expert sum), so with one K group this is
    bit-identical to routed + sh and saves the add. Returns fp32 [M, H]: a view of the coop scratch, valid until the
    next coop call on this device (the collective reads it first, in stream order); ``clone`` when two ranks share one
    device (emulation: they share the scratch too)."""
    m, hidden = x16.shape
    e = K.ext()
    prev = sh.contiguous()
    for i, g in enumerate(groups):
        out = sc.out[i % 2]
        e.exl3_moe_coop(x16, sel, rw16, g.lo, g.hi, hidden, g.g.trellis, g.g.suh, g.g.svh, g.u.trellis, g.u.suh,
                        g.u.svh, g.d.trellis, g.d.suh, g.d.svh, None, None, None, float(g.k_gu), float(g.k_gu),
                        float(g.k_d), False, True, K.MOE_COOP_ACT_SILU, float(act_limit), True, sc.had_g, sc.had_u,
                        sc.gu_g, sc.gu_u, sc.act_out, sc.d_out, sc.ctr, out, prev, None)
        prev = out
    return prev[:m].clone() if clone else prev[:m]


# --------------------------------------------------------------------------------------------- shards


@dataclass
class StageShard:
    wq_a: Exl3Mat
    wkv: Exl3Mat
    wq_b: Exl3Mat                 # heads [h0, h1)
    wo_b: Exl3Mat                 # input rows of the rank's wo_a groups
    wo_a8: torch.Tensor           # uint8 [groups_r * 1024, 4096]
    wo_a_sc: torch.Tensor         # fp32 [groups_r * 32, 128]
    attn_sink: torch.Tensor       # fp32 [heads_r]
    q_norm: torch.Tensor
    kv_norm: torch.Tensor
    attn_norm: torch.Tensor
    ffn_norm: torch.Tensor
    hc_attn: tuple
    hc_ffn: tuple
    gate_w: torch.Tensor
    gate_bias: torch.Tensor
    sh_w1: Exl3Mat                # columns [i0, i1)
    sh_w3: Exl3Mat
    sh_w2: Exl3Mat                # rows [i0, i1)
    bank: Exl3Bank
    groups: list
    perm: torch.Tensor
    qkv: Exl3Pair | None = None   # wq_a + wkv, one launch (draft body; the insert runs wkv alone)
    gu: Exl3Pair | None = None    # shared w1 + w3, one launch


class Rank:
    """Everything one rank holds on its device: shards, the replicated tensors, windows, graph inputs."""

    def __init__(self, r: int, dev: torch.device, ck: Checkpoint, plan: KernelPlan, slots: int, max_model_len: int,
                 max_rows: int, markov_cache_rows: int, tokens, tp: int = TP, fuse_pairs: bool = False):
        s: DrafterShape = ck.shape
        self.r, self.dev, self.s, self.tp = r, dev, s, tp
        k_dense = ck.drafter_k
        xplan = Exl3MoEPlan.from_dict(ck.plan_json)
        heads_r = s.n_heads // tp
        self.h0, self.h1 = r * heads_r, (r + 1) * heads_r
        g_r = s.o_groups // tp
        self.g0, self.g1 = r * g_r, (r + 1) * g_r
        self.vrange = vocab_split(s.vocab, tp)[r]
        self.v0, self.v1 = self.vrange
        hd = s.head_dim
        rows_per_group = s.o_lora                                   # wo_a output rows per group (1024)
        cin = s.n_heads // s.o_groups * hd                          # wo_a input per group (4096)
        i_loc = s.inter // tp
        self.i0, self.i1 = r * i_loc, (r + 1) * i_loc

        def exl3(name: str, want_k: int | None):
            tr, suh, svh, k = ck.exl3(name)
            if want_k is not None and k != want_k:
                raise RuntimeError(f"{name}: trellis K{k}, config says K{want_k}")
            return tr, suh, svh, k

        def mat(parts, k, name, ext=None, force_shape=0, fixed_bucket=None):
            tr, suh, svh = parts
            return Exl3Mat(tr, suh, svh, k, dev, plan, ext, force_shape, name, fixed_bucket)

        def g(name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
            t = ck.get(name)
            return t.to(dev, dtype=dtype or t.dtype).contiguous()

        self.embed = g("embed.weight")
        tr, suh, svh, hk = exl3("head", ck.head_k)
        if hk != ck.head_k:
            raise RuntimeError("head K")
        self.head = mat(col_shard(tr, suh, svh, self.v0, self.v1), hk, f"head[{self.v0}:{self.v1}]",
                        force_shape=3, fixed_bucket="head")
        tr, suh, svh, mk = exl3("mtp.0.main_proj", k_dense["main_proj"])
        self.main_proj = mat((tr, suh, svh), mk, "main_proj", ext=facc_ext())
        self.main_norm = g("mtp.0.main_norm.weight")
        last = f"mtp.{s.n_stages - 1}."
        self.norm = g(last + "norm.weight")
        self.conf_w = g(last + "confidence_head.proj.weight", torch.float32).view(-1)
        self.markov = MarkovShard(g(last + "markov_head.embed.weight"), g(last + "markov_head.head.weight"), self.v0,
                                  self.v1, tokens, markov_cache_rows)
        self.stages: list[StageShard] = []
        for i in range(s.n_stages):
            p = f"mtp.{i}."
            L = s.n_target_layers + i
            lp = xplan.drafter_layers[L]
            if lp.num_experts != s.n_experts:
                raise RuntimeError(f"plan layer {L}: {lp.num_experts} experts, config {s.n_experts}")
            bank = Exl3Bank(lp, s.hidden, s.inter, tp, r)
            bank.allocate(dev)
            for e in range(s.n_experts):
                for proj, wn in ((PROJ_GATE, "w1"), (PROJ_UP, "w3"), (PROJ_DOWN, "w2")):
                    tr, suh, svh, _ = ck.exl3(f"{p}ffn.experts.{e}.{wn}")
                    bank.load_piece("trellis", proj, e, tr)
                    bank.load_piece("suh", proj, e, suh)
                    bank.load_piece("svh", proj, e, svh)
            groups, _ = bank.tables()
            wq_a = exl3(p + "attn.wq_a", k_dense["attn.wq_a"])
            wkv = exl3(p + "attn.wkv", k_dense["attn.wq_a"])
            wq_b = exl3(p + "attn.wq_b", k_dense["attn.wq_b"])
            wo_b = exl3(p + "attn.wo_b", k_dense["attn.wo_b"])
            w1 = exl3(p + "ffn.shared_experts.w1", k_dense["ffn.shared_experts.w1"])
            w3 = exl3(p + "ffn.shared_experts.w3", k_dense["ffn.shared_experts.w1"])
            w2 = exl3(p + "ffn.shared_experts.w2", k_dense["ffn.shared_experts.w2"])
            woa = ck.get(p + "attn.wo_a.weight")
            woa_sc = ck.get(p + "attn.wo_a.scale")
            r0, r1 = self.g0 * rows_per_group, self.g1 * rows_per_group
            if woa.shape[1] != cin or woa_sc.shape[0] * 32 != woa.shape[0]:
                raise RuntimeError(f"{p}attn.wo_a: {tuple(woa.shape)} / scale {tuple(woa_sc.shape)}")
            st = StageShard(
                wq_a=mat(wq_a[:3], wq_a[3], p + "wq_a"), wkv=mat(wkv[:3], wkv[3], p + "wkv"),
                wq_b=mat(col_shard(*wq_b[:3], self.h0 * hd, self.h1 * hd), wq_b[3], p + "wq_b"),
                wo_b=mat(row_shard(*wo_b[:3], r0, r1), wo_b[3], p + "wo_b"),
                wo_a8=woa[r0:r1].contiguous().to(dev).view(torch.uint8),
                wo_a_sc=woa_sc[r0 // 32:r1 // 32].to(dev, dtype=torch.float32).contiguous(),
                attn_sink=g(p + "attn.attn_sink", torch.float32)[self.h0:self.h1].contiguous(),
                q_norm=g(p + "attn.q_norm.weight"), kv_norm=g(p + "attn.kv_norm.weight"),
                attn_norm=g(p + "attn_norm.weight"), ffn_norm=g(p + "ffn_norm.weight"),
                hc_attn=(g(p + "hc_attn_fn"), g(p + "hc_attn_scale"), g(p + "hc_attn_base")),
                hc_ffn=(g(p + "hc_ffn_fn"), g(p + "hc_ffn_scale"), g(p + "hc_ffn_base")),
                gate_w=g(p + "ffn.gate.weight", torch.float32), gate_bias=g(p + "ffn.gate.bias", torch.float32),
                sh_w1=mat(col_shard(*w1[:3], self.i0, self.i1), w1[3], p + "sh_w1"),
                sh_w3=mat(col_shard(*w3[:3], self.i0, self.i1), w3[3], p + "sh_w3"),
                sh_w2=mat(row_shard(*w2[:3], self.i0, self.i1), w2[3], p + "sh_w2"),
                bank=bank, groups=groups, perm=torch.tensor(lp.perm, dtype=torch.int64, device=dev))
            if fuse_pairs:
                st.qkv = Exl3Pair(st.wq_a, st.wkv)
                st.gu = Exl3Pair(st.sh_w1, st.sh_w3)
            self.stages.append(st)
        self.coop = K.get_coop_scratch(dev, max_rows, s.top_k, s.hidden, i_loc)
        with torch.cuda.device(dev):
            K.prime(dev, self.stages[0].groups, self.coop, None, s.hidden, s.top_k, s.swiglu_limit, False, True)
        self.cos_sin = rope_cos_sin(max_model_len + 2048, s.rope_dim, s.rope_theta).to(dev)
        self.q_clamp = torch.tensor(max_model_len - 1, dtype=torch.int64, device=dev)
        self.ring_kv = torch.zeros(slots + 1, s.n_stages, s.window, hd, dtype=torch.bfloat16, device=dev)
        self.ring_pos = torch.full((slots + 1, s.window), -1, dtype=torch.int64, device=dev)
        self.arange_block = torch.arange(s.block + 1, dtype=torch.int64, device=dev)
        self.pool = None
        self.cap_stream = torch.cuda.Stream(dev)

    def nbytes(self) -> dict:
        def nb(t):
            return t.numel() * t.element_size()
        d = {
            "embed_bf16": nb(self.embed), "head_shard_exl3_k6": self.head.nbytes(),
            "main_proj_exl3_k5": self.main_proj.nbytes(),
            "stages_wq_a_wkv_replicated": sum(st.wq_a.nbytes() + st.wkv.nbytes() for st in self.stages),
            "stages_wq_b_wo_b_shared_shards": sum(st.wq_b.nbytes() + st.wo_b.nbytes() + st.sh_w1.nbytes()
                                                  + st.sh_w3.nbytes() + st.sh_w2.nbytes() for st in self.stages),
            "stages_wo_a_fp8_shard": sum(nb(st.wo_a8) + nb(st.wo_a_sc) for st in self.stages),
            "stages_experts_exl3_k2_shard": sum(sum(nb(t) for t in st.bank.t.values()) for st in self.stages),
            "stages_small_fp32": sum(nb(t) for st in self.stages for t in (st.attn_sink, st.q_norm, st.kv_norm,
                                                                           st.attn_norm, st.ffn_norm, st.gate_w,
                                                                           st.gate_bias, *st.hc_attn, *st.hc_ffn)),
            "markov_embed_bf16": nb(self.markov.w1), "markov_head_shard_bf16": nb(self.markov.w2),
            "markov_cache_shard_bf16": nb(self.markov.cache) + nb(self.markov.slot_of_token),
            "rope_cos_sin_fp32": nb(self.cos_sin), "windows_total": nb(self.ring_kv) + nb(self.ring_pos),
        }
        return d


# --------------------------------------------------------------------------------------------- collectives


class EmuComm:
    """Both ranks on one device: the collectives as torch ops (the same combine as p2p.cu, bit for bit)."""
    real = False

    def all_reduce(self, R, parts: dict, out_dtype=torch.bfloat16) -> dict:
        if tuple(R) != (0, 1):
            raise RuntimeError("emulated collectives need both ranks active")
        s = (parts[0] + parts[1]).to(out_dtype)
        return {0: s, 1: s}

    def argmax(self, R, ranks, vals: dict, idxs: dict, outs: dict, col: int) -> None:
        if tuple(R) != (0, 1):
            raise RuntimeError("emulated collectives need both ranks active")
        pick = torch.where(vals[0] >= vals[1], idxs[0] + ranks[0].v0, idxs[1] + ranks[1].v0)
        for r in R:
            outs[r][:, col] = pick

    def margin(self, R, ranks, top2: dict, idxs: dict, outs: dict, mgs: dict, col: int) -> None:
        a1, a2, b1, b2 = top2[0][:, 0], top2[0][:, 1], top2[1][:, 0], top2[1][:, 1]
        take_a = a1 >= b1
        pick = torch.where(take_a, idxs[0] + ranks[0].v0, idxs[1] + ranks[1].v0)
        second = torch.where(take_a, torch.maximum(a2, b1), torch.maximum(b2, a1))
        mg = torch.maximum(a1, b1) - second
        for r in R:
            outs[r][:, col] = pick
            mgs[r][:, col] = mg

    def error_word(self):
        return None

    def check(self) -> None:
        pass

    def reset(self) -> None:
        pass


class SoloComm:
    """Benchmark only: one rank alone, collectives replaced by local stand-ins of the same shapes (x + x, the local
    argmax). Times one rank's compute; the real step adds the p2p.cu collectives. Drafts are not meaningful."""
    real = False

    def all_reduce(self, R, parts: dict, out_dtype=torch.bfloat16) -> dict:
        return {r: (parts[r] + parts[r]).to(out_dtype) for r in R}

    def argmax(self, R, ranks, vals: dict, idxs: dict, outs: dict, col: int) -> None:
        for r in R:
            outs[r][:, col] = idxs[r] + ranks[r].v0

    def margin(self, R, ranks, top2: dict, idxs: dict, outs: dict, mgs: dict, col: int) -> None:
        for r in R:
            outs[r][:, col] = idxs[r] + ranks[r].v0
            mgs[r][:, col] = top2[r][:, 0] - top2[r][:, 1]

    def error_word(self):
        return None

    def check(self) -> None:
        pass

    def reset(self) -> None:
        pass


class P2PComm:
    """The real pair: p2p.cu collectives, one launch per active rank on its own device."""
    real = True

    def __init__(self, devs, max_words: int, vocab: int):
        from . import p2p
        self.pair = p2p.P2PPair(devs, max_words, vocab=vocab)

    @staticmethod
    def nblocks(n: int) -> int:
        return max(8, min(64, -(-n // 2048)))

    def all_reduce(self, R, parts: dict, out_dtype=torch.bfloat16) -> dict:
        out = {}
        for r in R:
            x = parts[r].contiguous()
            with torch.cuda.device(x.device):
                y = torch.empty(x.shape, dtype=out_dtype, device=x.device)
                self.pair.ar_sum(r, x.view(-1), y.view(-1), self.nblocks(x.numel()))
            out[r] = y
        return out

    def argmax(self, R, ranks, vals: dict, idxs: dict, outs: dict, col: int) -> None:
        for r in R:
            with torch.cuda.device(ranks[r].dev):
                self.pair.xchg_argmax(r, vals[r], idxs[r], ranks[r].v0, outs[r], col)

    def margin(self, R, ranks, top2: dict, idxs: dict, outs: dict, mgs: dict, col: int) -> None:
        for r in R:
            with torch.cuda.device(ranks[r].dev):
                self.pair.xchg_margin(r, top2[r].contiguous(), idxs[r], ranks[r].v0, outs[r], mgs[r], col)

    def error_word(self) -> torch.Tensor:
        """Rank 0's error word (int32 [1] on device 0; rank 1's timeouts are mirrored into it)."""
        return self.pair.st[0].hdr[2:3]

    def headers(self):
        return self.pair.headers()

    def check(self) -> None:
        h = self.pair.headers()
        if h[0][2] or h[1][2] or h[0][0] != h[1][0]:
            raise RuntimeError(f"TP collectives failed: rank headers {h} (epoch, done, error, error epoch, count)")

    def reset(self) -> None:
        self.pair.reset()


# --------------------------------------------------------------------------------------------- the drafter


class LocalComm:
    """tp = 1: nothing to combine (the same code path on one card, for the one-card comparison)."""
    real = False

    def all_reduce(self, R, parts: dict, out_dtype=torch.bfloat16) -> dict:
        return {r: parts[r].to(out_dtype) for r in R}

    def argmax(self, R, ranks, vals: dict, idxs: dict, outs: dict, col: int) -> None:
        for r in R:
            outs[r][:, col] = idxs[r]

    def margin(self, R, ranks, top2: dict, idxs: dict, outs: dict, mgs: dict, col: int) -> None:
        for r in R:
            outs[r][:, col] = idxs[r]
            mgs[r][:, col] = top2[r][:, 0] - top2[r][:, 1]

    def error_word(self):
        return None

    def check(self) -> None:
        pass

    def reset(self) -> None:
        pass


class TPDrafter:
    """DSparkDrafter's surface over two ranks. ``devs``: two devices (the NVLink pair) or the same device twice
    (emulation). Inputs to the public methods are CPU tensors (pinned for the graph paths); outputs are rank 0's.
    ``tp=1`` (devs = [card]): the same code unsplit on one card (comparison: what the second card adds)."""

    HEAD_CHUNK = 16

    def __init__(self, ck: Checkpoint, devs: list[torch.device], slots: int = 8, max_model_len: int = 327680,
                 max_rows: int = 32, markov_cache_rows: int = 8192, kernel_plan: str | None = KERNEL_PLAN,
                 solo_rank: int | None = None, tp: int = TP, fuse_pairs: bool | None = None):
        if not ck.verified:
            raise RuntimeError("checkpoint not identity-verified")
        if tp not in (1, 2) or len(devs) != tp:
            raise ValueError(f"tp {tp} needs {tp} device(s) (tp 2: or one device twice), got {devs}")
        t0 = time.time()
        self.tp = tp
        self.s: DrafterShape = ck.shape
        s = self.s
        if s.n_heads % (tp * (s.n_heads // s.o_groups)) or s.inter % (tp * 128):
            raise RuntimeError(f"drafter shape does not split at tp {tp} on whole groups / Hadamard blocks")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        self.devs = devs
        self.emulated = tp == 1 or devs[0] == devs[1]
        self.dev = devs[0]
        self.max_model_len = max_model_len
        self.max_rows = max_rows
        self.n_slots = slots
        self.eps = s.rms_eps
        self.scale = s.head_dim ** -0.5
        n_sms = torch.cuda.get_device_properties(devs[0]).multi_processor_count
        if any(torch.cuda.get_device_properties(d).multi_processor_count != n_sms for d in devs):
            raise RuntimeError("the two cards differ in SM count")
        self.plan = KernelPlan(kernel_plan, n_sms)
        tokens = None
        tok_path = os.path.join(ck.root, "markov_top_tokens.json")
        if os.path.exists(tok_path):
            with open(tok_path) as f:
                tokens = json.load(f)
        # solo_rank (benchmarks): only that rank is built and run, collectives are local stand-ins (SoloComm)
        self.solo = solo_rank
        # paired GEMMs in one exl3_mgemm launch (Exl3Pair): OFF by default. -70..-100 us per step, but on the live
        # captures it lowered draft agreement with the engine (blocks exact, production taps) from 93.4 % to 91.3 %
        # (e6_parity_emulated_v3.json, ~1.8 standard errors); the drafter is ~3-8 % of a round, so that much speed is
        # worth less than that much acceptance (TP2.md). DSPARK_TP_FUSE_PAIRS=1 turns it on.
        self.fuse_pairs = (os.environ.get("DSPARK_TP_FUSE_PAIRS", "0") == "1") if fuse_pairs is None else fuse_pairs
        self.ALL = tuple(range(tp)) if solo_rank is None else (solo_rank,)
        self.ranks: list = [None] * tp
        for r in self.ALL:
            with torch.cuda.device(devs[r]):       # Triton launches on the current device, not the tensors'
                self.ranks[r] = self._make_rank(r, devs[r], ck, slots, max_model_len, max_rows, markov_cache_rows,
                                                tokens, tp)
        from .exl3.hostcopy import release_stage
        release_stage()
        if tp == 1:
            self.comm = LocalComm()
        elif solo_rank is not None:
            self.comm = SoloComm()
            self.emulated = True                    # one graph holding the active rank
        else:
            self.comm = EmuComm() if self.emulated else P2PComm(devs, self.comm_words(), s.vocab)
        self.shared_scratch = len(self.ALL) == 2 and devs[0] == devs[1]   # emulation: one coop scratch, two ranks
        self.graphs: dict = {}
        self.margin_graphs: dict = {}
        self.insert_graphs: dict = {}
        self.g_in = None
        for d in devs:
            torch.cuda.synchronize(d)
        self.load_seconds = time.time() - t0

    # ------------------------------------------------------------------------------------------ helpers
    def _make_rank(self, r: int, dev: torch.device, ck: Checkpoint, slots: int, max_model_len: int, max_rows: int,
                   markov_cache_rows: int, tokens, tp: int):
        """Rank r's shards (subclasses with other weight representations override this; native/tp2_native.py)."""
        return Rank(r, dev, ck, self.plan, slots, max_model_len, max_rows, markov_cache_rows, tokens, tp,
                    self.fuse_pairs)

    def comm_words(self) -> int:
        """Largest all-reduce payload (fp32 words) the p2p buffers are sized for."""
        return 32 * self.s.hidden

    def _each(self, R, fn) -> dict:
        out = {}
        for r in R:
            rk = self.ranks[r]
            with torch.cuda.device(rk.dev):
                out[r] = fn(rk)
        return out

    def to_ranks(self, R, t: torch.Tensor) -> dict:
        """A CPU tensor on every active rank's device (non_blocking from pinned memory)."""
        return self._each(R, lambda rk: t.to(rk.dev, non_blocking=True))

    def set_query_clamp(self, max_model_len: int) -> None:
        if not 0 < max_model_len <= self.max_model_len:
            raise ValueError(f"max_model_len {max_model_len} outside 1..{self.max_model_len}")
        self._each(self.ALL, lambda rk: rk.q_clamp.fill_(max_model_len - 1))

    def reset_slots(self, slots: list[int]) -> None:
        idx = torch.tensor(list(slots), dtype=torch.int64)

        def one(rk):
            i = idx.to(rk.dev)
            rk.ring_pos[i] = -1
            rk.ring_kv[i] = 0
        self._each(self.ALL, one)

    # ------------------------------------------------------------------------------------------ context side
    def _main_x(self, rk: Rank, taps: torch.Tensor) -> torch.Tensor:
        x16, sc = FU.prescale_fp16(taps)
        return FU.scaled_rms_norm(rk.main_proj.gemm(x16), sc, rk.main_norm, self.eps)

    def _insert_rank(self, rk: Rank, slot_rows: torch.Tensor, pos_rows: torch.Tensor, taps: torch.Tensor) -> None:
        mx16 = self._main_x(rk, taps).to(torch.float16)
        for i, st in enumerate(rk.stages):
            FU.kv_store(st.wkv(mx16), st.kv_norm, self.eps, pos_rows, rk.cos_sin, rk.ring_kv[:, i], slot_rows,
                        rk.ring_pos if i == 0 else None)

    def insert_taps(self, R, slot_rows: dict, pos_rows: dict, taps: dict) -> None:
        """main_x + window KV of rows on every active rank (replicated; no collective)."""
        self._each(R, lambda rk: self._insert_rank(rk, slot_rows[rk.r], pos_rows[rk.r], taps[rk.r]))

    def main_x(self, taps: torch.Tensor, rank: int = 0) -> torch.Tensor:
        rk = self.ranks[rank]
        with torch.cuda.device(rk.dev):
            return self._main_x(rk, taps.to(rk.dev))

    # ------------------------------------------------------------------------------------------ draft step
    def _block_inputs(self, rk: Rank, inp: torch.Tensor, k: int):
        """inp int64 [4, B] on the rank's device: slots, anchors, last positions, window lows."""
        s = self.s
        slots, anchors, last, lo = inp[0], inp[1], inp[2], inp[3]
        ar = rk.arange_block[:k]
        ids = torch.where(ar.unsqueeze(0) == 0, anchors.unsqueeze(1), torch.full_like(ar, s.noise_token).unsqueeze(0))
        pos = torch.minimum(last.unsqueeze(1) + 1 + ar.unsqueeze(0), rk.q_clamp)
        return ids, pos, slots, anchors, last, lo

    def _attn_part(self, rk: Rank, i: int, st: StageShard, stream, pre_in, pos, slots, lo, last, B, k):
        s = self.s
        T = B * k
        pre, post, comb = FU.hc_mixes(stream, *st.hc_attn, self.eps, s.hc_eps, s.hc_iters)
        _, x16 = FE.collapse_norm2(stream, pre_in, st.attn_norm, self.eps)       # bf16 result, as fp16 for EXL3
        qa, kv = st.qkv(x16) if st.qkv is not None else (st.wq_a.gemm(x16), st.wkv.gemm(x16))
        qr16 = FE.rms_norm_fp16(qa, st.q_norm, self.eps)
        kvb = FU.kv_rope_qdq(FU.rms_norm(kv, st.kv_norm, self.eps), pos, rk.cos_sin)
        hr = rk.h1 - rk.h0
        q = FU.rope(st.wq_b.gemm(qr16).to(torch.bfloat16).view(T, hr, s.head_dim), pos, rk.cos_sin)
        o = FU.block_attention_ring(q.view(B, k, hr, s.head_dim), rk.ring_kv[:, i], rk.ring_pos, slots, lo, last,
                                    kvb.view(B, k, s.head_dim), st.attn_sink, self.scale)
        o = FU.rope(o.view(T, hr, s.head_dim), pos, rk.cos_sin, inverse=True)
        z = FU.wo_a_fp8(o.view(T, -1), st.wo_a8, st.wo_a_sc, rk.g1 - rk.g0)
        return st.wo_b(z), (pre, post, comb)

    def _ffn_part(self, rk: Rank, st: StageShard, stream, pre, trace=None, tag=""):
        s = self.s
        fpre, fpost, fcomb = FU.hc_mixes(stream, *st.hc_ffn, self.eps, s.hc_eps, s.hc_iters)
        xin, x16 = FE.collapse_norm2(stream, pre, st.ffn_norm, self.eps)
        ids, sel, w, w16 = FE.route_fused(xin, st.gate_w, st.gate_bias, st.perm, s.top_k, s.route_scale)
        if trace is not None and rk.r == 0:
            trace[tag + ".experts"] = ids
            trace[tag + ".route_w"] = w
            trace[tag + ".ffn_in"] = xin
        lim = s.swiglu_limit
        g, u = st.gu(x16) if st.gu is not None else (st.sh_w1.gemm(x16), st.sh_w3.gemm(x16))
        a16 = FE.swiglu_fp16(g, u, lim)
        sh = st.sh_w2.gemm(a16)
        part = coop_with_shared(x16, sel, w16, st.groups, rk.coop, lim, sh, self.shared_scratch)
        return part, (fpre, fpost, fcomb)

    def forward_block(self, R, inp: dict, k: int, trace: dict | None = None):
        """The three stages over the query blocks of every active rank. Returns (head_hidden per rank, block
        inputs per rank)."""
        s = self.s
        bi = self._each(R, lambda rk: self._block_inputs(rk, inp[rk.r], k))
        B = next(iter(bi.values()))[0].shape[0]
        T = B * k
        stream = self._each(R, lambda rk: rk.embed[bi[rk.r][0].reshape(T)].unsqueeze(1)
                            .expand(T, s.hc_mult, s.hidden).contiguous())
        pre_in = {r: None for r in R}
        for i in range(s.n_stages):
            res = self._each(R, lambda rk: self._attn_part(rk, i, rk.stages[i], stream[rk.r], pre_in[rk.r],
                                                           bi[rk.r][1].reshape(T), bi[rk.r][2], bi[rk.r][5],
                                                           bi[rk.r][4], B, k))
            a = self.comm.all_reduce(R, {r: res[r][0] for r in R})
            mix = {r: res[r][1] for r in R}
            stream = self._each(R, lambda rk: FU.hc_post(a[rk.r], stream[rk.r], mix[rk.r][1], mix[rk.r][2]))
            res = self._each(R, lambda rk: self._ffn_part(rk, rk.stages[i], stream[rk.r], mix[rk.r][0], trace,
                                                          f"stage{i}"))
            y = self.comm.all_reduce(R, {r: res[r][0] for r in R})
            fmix = {r: res[r][1] for r in R}
            stream = self._each(R, lambda rk: FU.hc_post(y[rk.r], stream[rk.r], fmix[rk.r][1], fmix[rk.r][2]))
            pre_in = {r: fmix[r][0] for r in R}
            if trace is not None and 0 in R:
                trace[f"stage{i}.attn_out"] = a[0]
                trace[f"stage{i}.ffn_out"] = y[0]
                trace[f"stage{i}.stream"] = stream[0]
        hh = self._each(R, lambda rk: FU.collapse(stream[rk.r], pre_in[rk.r]))
        return hh, bi

    def heads(self, R, hh: dict, bi: dict, B: int, k: int, n_out: int | None = None, margin: bool = False,
              trace: dict | None = None):
        """Sequential greedy Markov drafting over the vocabulary shards (one exchange per position) and the
        confidence head (rank 0). Positions >= n_out: draft 0, confidence 0 (CONTRACT 1.3; positions < n_out are
        bit-identical for every n_out: fixed-shape 16-row head chunks, per-row Markov and exchanges)."""
        s = self.s
        n = k if n_out is None else n_out
        if not 1 <= n <= k:
            raise ValueError(f"n_out {n} outside 1..{k}")

        def logits_of(rk):
            rows = FU.rms_norm(hh[rk.r], rk.norm, self.eps).view(B, k, s.hidden)[:, :n].reshape(B * n, s.hidden)
            rows = rows.to(torch.float16)
            l32 = torch.empty(B * n, rk.v1 - rk.v0, dtype=torch.float32, device=rk.dev)
            for c in range(0, B * n, self.HEAD_CHUNK):
                rk.head.gemm(rows[c:c + self.HEAD_CHUNK], out=l32[c:c + self.HEAD_CHUNK])
            return l32.to(torch.bfloat16).view(B, n, rk.v1 - rk.v0)
        logits = self._each(R, logits_of)
        out = self._each(R, lambda rk: torch.zeros(B, k, dtype=torch.int64, device=rk.dev))
        mg = self._each(R, lambda rk: torch.zeros(B, k, dtype=torch.float32, device=rk.dev)) if margin else None
        anchors = {r: bi[r][3] for r in R}
        if trace is not None:
            trace["base_logits"] = torch.cat([logits[r].to(self.dev) for r in R], -1) if tuple(R) == self.ALL \
                else None
        for i in range(n):
            prev = {r: anchors[r] if i == 0 else out[r][:, i - 1] for r in R}
            if not margin and self.comm.real:      # block maxima -> one fused finish + exchange (p2p.cu)
                def step(rk):
                    parts = markov_partials(logits[rk.r][:, i], prev[rk.r], rk.markov)
                    self.comm.pair.xchg_markov(rk.r, prev[rk.r], rk.markov.slot_of_token, parts, rk.v0, out[rk.r], i)
                self._each(R, step)
            elif not margin:
                loc = self._each(R, lambda rk: markov_local_argmax(logits[rk.r][:, i], prev[rk.r], rk.markov))
                self.comm.argmax(R, self.ranks, {r: loc[r][0] for r in R}, {r: loc[r][1] for r in R}, out, i)
            else:
                def local_margin(rk):
                    pv = prev[rk.r].contiguous()
                    li = logits[rk.r][:, i] + rk.markov.bias(pv, rk.markov.embed(pv))
                    return li.float().topk(2, dim=-1).values.contiguous(), li.argmax(dim=-1)
                loc = self._each(R, local_margin)
                self.comm.margin(R, self.ranks, {r: loc[r][0] for r in R}, {r: loc[r][1] for r in R}, out, mg, i)
        conf = None
        if 0 in R:
            rk = self.ranks[0]
            with torch.cuda.device(rk.dev):
                o = out[0]
                emb = rk.markov.embed(torch.cat((anchors[0].unsqueeze(1), o[:, :n - 1]), 1))
                if n < k:
                    emb = torch.cat((emb, emb.new_zeros(B, k - n, emb.shape[-1])), 1)
                feat = torch.cat((hh[0].view(B, k, s.hidden), emb), dim=-1).float()
                conf = torch.sigmoid(feat @ rk.conf_w)
                if n < k:
                    conf[:, n:] = 0.0
        return out, conf, mg

    def _draft(self, R, inp: dict, k: int, n_out: int | None, margin: bool, trace: dict | None = None):
        hh, bi = self.forward_block(R, inp, k, trace)
        B = next(iter(bi.values()))[0].shape[0]
        if trace is not None and 0 in R:
            trace["head_hidden"] = hh[0]
        out, conf, mg = self.heads(R, hh, bi, B, k, n_out, margin, trace)
        return out, conf, mg

    def draft(self, inp_host: torch.Tensor, k: int, n_out: int | None = None, margin: bool = False,
              trace: dict | None = None):
        """Eager lockstep step on both ranks. inp_host int64 [4, B] (slots, anchors, last positions, window lows).
        Returns rank 0's (ids [B, k], conf [B, k], margin [B, k] or None) and rank 1's ids (consistency checks)."""
        inp = self.to_ranks(self.ALL, inp_host)
        out, conf, mg = self._draft(self.ALL, inp, k, n_out, margin, trace)
        return out.get(0), conf, (mg.get(0) if mg is not None else None), out.get(1)

    # ------------------------------------------------------------------------------------------ CUDA graphs
    def capture(self, batch_sizes=(1, 2, 3, 4), ks=(5,), insert_rows=(8, 16, 32, 64, 128), n_outs=(1, 2, 3, 4, 5),
                margins: bool = True) -> float:
        """One graph per rank (one in emulation) per (B, k, n_out) [x margin] and per insert row bucket."""
        t0 = time.time()
        bmax, rmax = max(batch_sizes), max(insert_rows)
        s = self.s
        for r in self.ALL:
            if self.ranks[r].pool is None:
                self.ranks[r].pool = torch.cuda.graph_pool_handle()
        self.g_in = self._each(self.ALL, lambda rk: torch.zeros(4, bmax, dtype=torch.int64, device=rk.dev))
        for r in self.ALL:
            self.g_in[r][0].fill_(self.n_slots)              # scratch slot
        self.gi_slot = self._each(self.ALL, lambda rk: torch.full((rmax,), self.n_slots, dtype=torch.int64,
                                                                  device=rk.dev))
        self.gi_pos = self._each(self.ALL, lambda rk: torch.zeros(rmax, dtype=torch.int64, device=rk.dev))
        self.gi_taps = self._each(self.ALL, lambda rk: torch.zeros(rmax, s.hidden * len(s.taps),
                                                                   dtype=torch.bfloat16, device=rk.dev))
        for B in batch_sizes:
            for k in ks:
                for n in sorted({x for x in n_outs if x <= k} | {k}):
                    for mgn in ((False, True) if margins else (False,)):
                        def fn(R, B=B, k=k, n=n, mgn=mgn):
                            out, conf, mg = self._draft(R, {r: self.g_in[r][:, :B] for r in R}, k, n, mgn)
                            return {"ids": out.get(0), "conf": conf, "mg": mg.get(0) if mg else None}
                        g = self._capture_one(fn)
                        (self.margin_graphs if mgn else self.graphs)[(B, k, n)] = g
        for nrow in insert_rows:
            def fn(R, nrow=nrow):
                self.insert_taps(R, {r: self.gi_slot[r][:nrow] for r in R}, {r: self.gi_pos[r][:nrow] for r in R},
                                 {r: self.gi_taps[r][:nrow] for r in R})
                return {}
            self.insert_graphs[nrow] = self._capture_one(fn)
        self.reset_slots([self.n_slots])
        for d in self.devs:
            torch.cuda.synchronize(d)
        self.comm.check()
        return time.time() - t0

    def _capture_one(self, fn) -> dict:
        """Warm up eagerly on side streams (both ranks, lockstep, real collectives), then capture. Two devices: one
        graph per rank with only that rank active; emulation: one graph with both."""
        side = {r: torch.cuda.Stream(self.devs[r]) for r in self.ALL} if not self.emulated else \
            {self.ALL[0]: torch.cuda.Stream(self.devs[self.ALL[0]])}
        for r, st in side.items():
            st.wait_stream(torch.cuda.current_stream(self.devs[r]))
        ctxs = [torch.cuda.stream(st) for st in side.values()]
        for c in ctxs:
            c.__enter__()
        try:
            for _ in range(2):
                fn(self.ALL)
        finally:
            for c in reversed(ctxs):
                c.__exit__(None, None, None)
        for r, st in side.items():
            torch.cuda.current_stream(self.devs[r]).wait_stream(st)
        for d in self.devs:
            torch.cuda.synchronize(d)
        rec = {}
        if self.emulated:
            rk = self.ranks[self.ALL[0]]
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.device(rk.dev):
                with torch.cuda.graph(gr, pool=rk.pool, stream=rk.cap_stream):
                    o = fn(self.ALL)
            rec = {"graphs": {rk.r: gr}, **o}
        else:
            graphs = {}
            for r in self.ALL:
                rk = self.ranks[r]
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.device(rk.dev):
                    with torch.cuda.graph(gr, pool=rk.pool, stream=rk.cap_stream):
                        o = fn((r,))
                graphs[r] = gr
                if r == 0:
                    rec.update(o)
            rec["graphs"] = graphs
        return rec

    def _replay(self, g: dict) -> None:
        order = (1, 0) if not self.emulated else (self.ALL[0],)
        try:
            if self.comm.real:
                self.comm.pair._usable()
            for r in order:
                with torch.cuda.device(self.devs[r]):
                    g["graphs"][r].replay()
        except RuntimeError as error:
            if self.comm.real:
                self.comm.pair.poison(str(error))
            raise

    def has_graph(self, B: int, k: int, n_out: int | None = None, margin: bool = False) -> bool:
        return (B, k, k if n_out is None else n_out) in (self.margin_graphs if margin else self.graphs)

    def draft_graphed(self, inp_host: torch.Tensor, k: int, n_out: int | None = None, margin: bool = False):
        """Replay (B, k, n_out) on both ranks. inp_host: pinned int64 [4, B]. Returns views of rank 0's graph
        outputs (ids, conf, margin or None); copy before the next replay."""
        B = inp_host.shape[1]
        n = k if n_out is None else n_out
        g = (self.margin_graphs if margin else self.graphs).get((B, k, n))
        if g is None:
            raise KeyError(f"no CUDA graph for batch {B}, depth {k}, n_out {n}, margin {margin}")
        for r in self.ALL:
            with torch.cuda.device(self.devs[r]):
                self.g_in[r][:, :B].copy_(inp_host, non_blocking=True)
        self._replay(g)
        return g["ids"], g["conf"], g["mg"]

    def insert_taps_graphed(self, slot_rows: torch.Tensor, pos_rows: torch.Tensor, taps: torch.Tensor) -> None:
        """main_x + insert for n rows (host pinned tensors) through the smallest bucket >= n on both ranks; padding
        rows hit the scratch slot with zero taps (no stale rows in main_proj's batch-wide prescale)."""
        n = slot_rows.shape[0]
        sizes = sorted(self.insert_graphs)
        b = next((x for x in sizes if x >= n), None)
        if b is None:
            raise ValueError(f"{n} insert rows exceed the largest graph bucket {sizes[-1]}")
        for r in self.ALL:
            with torch.cuda.device(self.devs[r]):
                self.gi_slot[r][:n].copy_(slot_rows, non_blocking=True)
                self.gi_slot[r][n:b].fill_(self.n_slots)
                self.gi_pos[r][:n].copy_(pos_rows, non_blocking=True)
                self.gi_pos[r][n:b].zero_()
                self.gi_taps[r][:n].copy_(taps, non_blocking=True)
                self.gi_taps[r][n:b].zero_()
        self._replay(self.insert_graphs[b])

    def insert_mx_eager(self, slot_rows: torch.Tensor, pos_rows: torch.Tensor, mx: torch.Tensor) -> None:
        """Window KV of rows from given main_x rows (bf16 [n, H]; tools replaying captured main_x)."""
        R = self.ALL
        sr, pr, mxr = self.to_ranks(R, slot_rows), self.to_ranks(R, pos_rows), self.to_ranks(R, mx)

        def one(rk):
            mx16 = mxr[rk.r].to(torch.float16)
            for i, st in enumerate(rk.stages):
                FU.kv_store(st.wkv(mx16), st.kv_norm, self.eps, pr[rk.r], rk.cos_sin, rk.ring_kv[:, i], sr[rk.r],
                            rk.ring_pos if i == 0 else None)
        self._each(R, one)

    def insert_eager(self, slot_rows: torch.Tensor, pos_rows: torch.Tensor, taps: torch.Tensor) -> None:
        R = self.ALL
        self.insert_taps(R, self.to_ranks(R, slot_rows), self.to_ranks(R, pos_rows), self.to_ranks(R, taps))

    # ------------------------------------------------------------------------------------------ accounting
    def memory_table(self) -> dict:
        out = {}
        for rk in self.ranks:
            if rk is None:
                continue
            d = rk.nbytes()
            d["torch_allocated"] = torch.cuda.memory_allocated(rk.dev)
            d["torch_reserved"] = torch.cuda.memory_reserved(rk.dev)
            d["torch_max_allocated"] = torch.cuda.max_memory_allocated(rk.dev)
            out[f"rank{rk.r}"] = d
            if self.emulated:
                out[f"rank{rk.r}"]["note"] = "emulation: both ranks on one device; torch_* are the device totals"
        out["kernel_plan_used"] = dict(self.plan.used)
        return out


# --------------------------------------------------------------------------------------------- server backend


MAX_COMM_ERRORS_IN_A_ROW = 3


class TP2Backend:
    """server.GpuBackend's interface over TPDrafter: rows -> both ranks' windows, drafts through both ranks' graphs.
    Host buffers are pinned and reused; rank 0's ids / confidences (/ margins) and its collective error word (which
    carries rank 1's timeouts) come back in one D2H. A set error word fails the STEP loudly (the core answers INTERNAL
    and resyncs the entries' slots) after both devices are synchronized and the collective state is reset."""

    def __init__(self, model: TPDrafter, max_rows: int, row_bytes: int, graphs: bool = True, max_slots: int = 4):
        import logging

        import numpy as np
        self.log = logging.getLogger("dspark_worker")
        self.np = np
        self.m = model
        self.dev = model.dev
        self.row_bytes = row_bytes
        self.hidden3 = row_bytes // 2
        self.stage_buf = torch.empty((max_rows, row_bytes), dtype=torch.uint8, pin_memory=True)
        self.slot_buf = torch.empty(max_rows, dtype=torch.int64, pin_memory=True)
        self.pos_buf = torch.empty(max_rows, dtype=torch.int64, pin_memory=True)
        self.inp_buf = {B: torch.empty(4, B, dtype=torch.int64, pin_memory=True) for B in range(1, max_slots + 1)}
        self.pending: list = []
        self.copy_done = {}
        for r in model.ALL:
            with torch.cuda.device(model.devs[r]):
                ev = torch.cuda.Event()
                ev.record()
                self.copy_done[r] = ev
        self.graphs = graphs
        self.n_flush_rows = 0
        self.comm_errors = 0
        self.comm_errors_in_a_row = 0            # >= MAX_COMM_ERRORS_IN_A_ROW: unhealthy (systemd restarts the worker)
        self.max_model_len = model.max_model_len

    def set_max_model_len(self, n: int) -> None:
        self.m.set_query_clamp(n)

    def reset(self, slot: int) -> None:
        self.pending = [p for p in self.pending if p[0] != slot]
        self.m.reset_slots([slot])

    def stage(self, slot: int, first_pos: int, rows) -> None:
        self.pending.append((slot, first_pos, rows))

    def discard(self) -> None:
        self.pending = []

    def healthy(self) -> bool:
        """Both devices go idle within a bounded time and answer a trivial kernel; a collective error left by a
        failed step is cleared (the pair works again once both sides are idle and reset). Unhealthy after
        MAX_COMM_ERRORS_IN_A_ROW failed steps in a row (a degraded link or a wedged peer: restart, not retry)."""
        from .p2p import wait_idle
        if self.comm_errors_in_a_row >= MAX_COMM_ERRORS_IN_A_ROW:
            self.log.critical("%d TP collective failures in a row: the pair is unusable", self.comm_errors_in_a_row)
            return False
        try:
            # The collective boundary refuses poisoned contexts before any CUDA call and drains every tracked
            # stream. Keep the health interface independent of the concrete pair implementation (S10 fake/NCCL).
            h = self.m.comm.headers() if self.m.comm.real else None
            if h is None:
                wait_idle(self.m.devs, 30.0)
            for d in dict.fromkeys(self.m.devs):
                with torch.cuda.device(d):
                    if float((torch.ones(1, device=d) + 1).item()) != 2.0:
                        return False
            if self.m.comm.real:
                if h[0][2] or h[1][2] or h[0][0] != h[1][0]:
                    self.log.error("TP collectives left in error state %s: resetting both ranks", h)
                    self.m.comm.reset()
            return True
        except Exception:           # noqa: BLE001  (TimeoutError included)
            self.log.exception("GPU health probe failed")
            return False

    def flush(self) -> None:
        np = self.np
        if not self.pending:
            return
        n = sum(r.shape[0] for _, _, r in self.pending)
        for ev in self.copy_done.values():
            ev.synchronize()
        sb = self.stage_buf[:n].numpy()
        sl = self.slot_buf[:n].numpy()
        ps = self.pos_buf[:n].numpy()
        k = 0
        for slot, p0, rows in self.pending:
            c = rows.shape[0]
            sb[k:k + c] = rows
            sl[k:k + c] = slot
            ps[k:k + c] = np.arange(p0, p0 + c)
            k += c
        self.pending = []
        taps = self.stage_buf[:n].view(torch.bfloat16).view(n, self.hidden3)
        if self.graphs and self.m.insert_graphs:
            top = max(self.m.insert_graphs)
            for a in range(0, n, top):            # chunks: main_proj's prescale per chunk, as the eager path
                b = min(n, a + top)
                self.m.insert_taps_graphed(self.slot_buf[a:b], self.pos_buf[a:b], taps[a:b])
        else:
            for a in range(0, n, 128):
                b = min(n, a + 128)
                self.m.insert_eager(self.slot_buf[a:b], self.pos_buf[a:b], taps[a:b])
        for r, ev in self.copy_done.items():
            with torch.cuda.device(self.m.devs[r]):
                ev.record()
        self.n_flush_rows += n

    def draft(self, reqs, N: int, want_margin: bool):
        np = self.np
        B = len(reqs)
        n = max(k for *_r, k in reqs)
        host = self.inp_buf[B]
        hv = host.numpy()
        for j, (slot, tok, apos, rng, _k) in enumerate(reqs):
            hv[:, j] = (slot, tok, apos - 1, rng.read_lo(apos))
        if self.graphs and self.m.has_graph(B, N, n, margin=want_margin):
            ids, conf, mg = self.m.draft_graphed(host, N, n_out=n, margin=want_margin)
        else:
            ids, conf, mg, _ = self.m.draft(host, N, n_out=n, margin=want_margin)
        with torch.cuda.device(self.dev):
            parts = [ids.to(torch.float64), conf.to(torch.float64)] + ([mg.to(torch.float64)] if want_margin else [])
            err = self.m.comm.error_word()
            if err is not None:
                parts.append(err.view(1, 1).to(torch.float64).expand(B, 1))
            packed = torch.cat(parts, 1).cpu().numpy()                     # one D2H + sync
        if err is not None and packed[0, -1] != 0:
            self.comm_errors += 1
            self.comm_errors_in_a_row += 1
            h = self.m.comm.headers()             # bounded waits (TimeoutError -> the core classifies the backend)
            self.m.comm.reset()
            raise RuntimeError(f"TP collective failure in this step (rank headers {h}: epoch, done, error bits "
                               f"1 timeout / 2 peer timeout / 4 bad id / 8 cancelled / 16 epoch exhausted, "
                               f"error epoch, count); both ranks reset")
        self.comm_errors_in_a_row = 0
        margin = packed[:, 2 * N:3 * N].astype(np.float32) if want_margin else None
        return packed[:, :N].astype(np.int32), packed[:, N:2 * N].astype(np.float32), margin

    def stats(self) -> dict:
        # host-side counters only: the 5 s status snapshot never waits on a device
        return {"tp": self.m.tp, "gpu_mem_bytes": [torch.cuda.memory_reserved(d) for d in self.m.devs],
                "rows_inserted": self.n_flush_rows, "tp_comm_errors": self.comm_errors,
                "tp_comm_errors_in_a_row": self.comm_errors_in_a_row}
