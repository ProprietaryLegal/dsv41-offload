# SPDX-License-Identifier: Apache-2.0
"""Thin wrappers over the exllamav3 v1.6.0 CUDA extension (exllamav3_ext, MIT, turboderp-org).

Nothing from exllamav3 is copied here: we call its pybind entry points (exllamav3_ext/bindings.cpp)
with tensors laid out the way its kernels expect. Citations are exllamav3 v1.6.0 @ 151539c7 paths.

Entry points used:
  exl3_moe_coop      bindings.cpp:292   fused decode MoE, 2-3 launches (rot / gate+up / down),
                                        bsz * top_k <= 256 slots, no grid barrier, deterministic,
                                        no host sync (quant/exl3_moe_coop.cu:315-364,
                                        quant/exl3_moe_coop_kernel.cuh:1-21)
  exl3_moe           bindings.cpp:290   grouped MoE over expert-sorted rows, 16/32/64-row tiles,
                                        experts with more than R rows are skipped (quant/exl3_moe.cu:189-572)
  exl3_moe_max_concurrency  :289        groups the exl3_moe launch can keep co-resident
  reconstruct        bindings.cpp:170   trellis -> fp16 weight in the rotated basis
  had_r_128          bindings.cpp:175   128-block Hadamard with optional pre (suh) / post (svh) scale
  hgemm              bindings.cpp:209   fp16 GEMM, fp16 or fp32 output

Per-expert bitrates: every launch takes ONE (K_gate == K_up, K_down) pair
(exl3_moe.cu:367-378, exl3_moe_coop.cu:193-194, modules/multilinear.py:25-26), so a layer with
mixed K runs one launch per K group. Experts are stored K-sorted (plan.LayerPlan.perm) so a group
is a contiguous physical range [lo, hi) and the coop kernel's expert-range filter
(MoeCoopParams.min_expert/max_expert, exl3_moe_coop_kernel.cuh:211-223) masks the other groups.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

MOE_ACT_SILU = 0            # exl3_moe_common.cuh:7
MOE_COOP_ACT_SILU = 0       # exl3_moe_coop.cuh:13
COOP_MAX_SLOTS = 256        # exl3_moe_coop_kernel.cuh:46, exl3_moe_coop.cu:226
MAX_K_GROUPS = 8            # scratch sets for concurrent K groups (a layer has one group per distinct (k_gu, k_d))
MTILE_T1, MTILE_T2 = 16, 32  # row-tile tiers, as exllamav3 uses them (block_sparse_mlp.py:37-38)

_EXT = None


def ext():
    """The compiled extension. Prefer the prebuilt top-level module the exllamav3 wheel installs
    (setup.py: extension_name = "exllamav3_ext"); fall back to exllamav3's JIT loader (needs nvcc)."""
    global _EXT
    if _EXT is None:
        try:
            import exllamav3_ext as m  # type: ignore[import-not-found]
        except ImportError:
            try:
                from exllamav3.ext import exllamav3_ext as m  # type: ignore[import-not-found,no-redef]
            except Exception as e:  # noqa: BLE001
                raise ImportError(
                    "vllm_exl3_moe needs exllamav3 v1.6.0's compiled extension (exllamav3_ext). Build the "
                    "wheel for sm_121 (TORCH_CUDA_ARCH_LIST=12.1) and install it with --no-deps."
                ) from e
        missing = [n for n in ("exl3_moe", "exl3_moe_coop", "exl3_moe_max_concurrency", "reconstruct",
                               "had_r_128", "hgemm") if not hasattr(m, n)]
        if missing:
            raise ImportError(f"exllamav3_ext lacks {missing}: v1.6.0 or newer is required")
        _EXT = m
    return _EXT


def have_ext() -> bool:
    try:
        ext()
        return True
    except ImportError:
        return False


def coop_ctr_len(slots_max: int, rows_max: int, inter: int, out_dim: int) -> int:
    """int32 scratch of the coop kernels: completion counters + run table
    (exl3_moe_coop.cuh:154-158, exl3_moe_coop_ctr_len)."""
    return slots_max * (inter // 128) + rows_max * (out_dim // 128) + 2 + (slots_max + 1) + slots_max


# ---------------------------------------------------------------------------------------------
# Pointer tables


@dataclass
class ProjTables:
    trellis: torch.Tensor   # int64 [n] device: data_ptr of each expert's trellis
    suh: torch.Tensor       # int64 [n]
    svh: torch.Tensor       # int64 [n]

    def slice(self, lo: int, hi: int) -> "ProjTables":
        # 1-D slices of contiguous tensors stay contiguous (the kernels check it)
        return ProjTables(self.trellis[lo:hi], self.suh[lo:hi], self.svh[lo:hi])


@dataclass
class GroupTables:
    k_gu: float
    k_d: float
    lo: int
    hi: int
    g: ProjTables   # gate (w1)
    u: ProjTables   # up (w3)
    d: ProjTables   # down (w2)

    @property
    def size(self) -> int:
        return self.hi - self.lo


# ---------------------------------------------------------------------------------------------
# Scratch, shared by every layer of the process (all routed layers have the same shapes and run
# in stream order; the coop counters reset each other between calls, exl3_moe_coop_kernel.cuh:695-698,
# 833-835, so one zeroed buffer serves every layer)


@dataclass
class CoopScratch:
    rows_max: int
    slots_max: int
    had_g: torch.Tensor     # fp16 [slots_max, H]
    had_u: torch.Tensor     # fp16 [slots_max, H]
    gu_g: torch.Tensor      # fp16 [slots_max, I]
    gu_u: torch.Tensor      # fp16 [slots_max, I]
    act_out: torch.Tensor   # fp16 [slots_max, I]
    d_out: torch.Tensor     # fp32 [slots_max, H]
    ctr: torch.Tensor       # int32 [coop_ctr_len], zero once
    out: tuple[torch.Tensor, torch.Tensor]   # fp32 [rows_max, H] ping-pong (K-group chaining)


@dataclass
class FusedScratch:
    rows: int               # R: rows per expert the fused exl3_moe kernel takes
    concurrency: int        # C
    tsg: torch.Tensor       # fp16 [C, R, H]
    tsu: torch.Tensor
    tig: torch.Tensor       # fp16 [C, R, I]
    tiu: torch.Tensor


_COOP: dict[tuple, CoopScratch] = {}
_FUSED: dict[tuple, FusedScratch] = {}
_WBUF: dict[tuple, torch.Tensor] = {}


def get_coop_scratch(device: torch.device, rows_max: int, top_k: int, hidden: int, inter: int) -> CoopScratch:
    key = (str(device), rows_max, top_k, hidden, inter)
    sc = _COOP.get(key)
    if sc is None:
        sc = _new_coop_scratch(device, rows_max, top_k, hidden, inter)
        _COOP[key] = sc
    return sc


_PAR: dict[tuple, tuple[list[CoopScratch], list[torch.cuda.Stream]]] = {}


def get_parallel_coop(device: torch.device, rows_max: int, top_k: int, hidden: int, inter: int, n: int):
    """n independent scratch sets + n side streams, shared by every layer of the process (layers run in stream order,
    and each call joins its streams back before returning), for run_decode_coop_parallel."""
    key = (str(device), rows_max, top_k, hidden, inter, n)
    v = _PAR.get(key)
    if v is None:
        v = ([_new_coop_scratch(device, rows_max, top_k, hidden, inter) for _ in range(n)],
             [torch.cuda.Stream(device=device) for _ in range(n)])
        _PAR[key] = v
    return v


def _new_coop_scratch(device: torch.device, rows_max: int, top_k: int, hidden: int, inter: int) -> CoopScratch:
    if True:
        slots_max = rows_max * top_k
        if slots_max > COOP_MAX_SLOTS:
            raise ValueError(f"coop path holds at most {COOP_MAX_SLOTS} slots; rows_max {rows_max} x top_k {top_k}")
        f16 = dict(dtype=torch.float16, device=device)
        f32 = dict(dtype=torch.float32, device=device)
        sc = CoopScratch(
            rows_max=rows_max,
            slots_max=slots_max,
            had_g=torch.empty((slots_max, hidden), **f16),
            had_u=torch.empty((slots_max, hidden), **f16),
            gu_g=torch.empty((slots_max, inter), **f16),
            gu_u=torch.empty((slots_max, inter), **f16),
            act_out=torch.empty((slots_max, inter), **f16),
            d_out=torch.empty((slots_max, hidden), **f32),
            ctr=torch.zeros((coop_ctr_len(slots_max, rows_max, inter, hidden),), dtype=torch.int32, device=device),
            out=(torch.zeros((rows_max, hidden), **f32), torch.zeros((rows_max, hidden), **f32)),
        )
    return sc


def get_fused_scratch(device: torch.device, rows: int, hidden: int, inter: int) -> FusedScratch:
    key = (str(device), rows, hidden, inter)
    fs = _FUSED.get(key)
    if fs is None:
        c = int(ext().exl3_moe_max_concurrency(torch.device(device).index or 0))
        f16 = dict(dtype=torch.float16, device=device)
        fs = FusedScratch(
            rows=rows,
            concurrency=c,
            tsg=torch.empty((c, rows, hidden), **f16),
            tsu=torch.empty((c, rows, hidden), **f16),
            tig=torch.empty((c, rows, inter), **f16),
            tiu=torch.empty((c, rows, inter), **f16),
        )
        _FUSED[key] = fs
    return fs


def _wbuf(device: torch.device, numel: int) -> torch.Tensor:
    key = (str(device),)
    b = _WBUF.get(key)
    if b is None or b.numel() < numel:
        b = torch.empty((numel,), dtype=torch.float16, device=device)
        _WBUF[key] = b
    return b[:numel]


# ---------------------------------------------------------------------------------------------
# Decode: exl3_moe_coop, one launch set per K group, chained through the shared-expert input


def run_decode_coop(
    x16: torch.Tensor,          # fp16 [M, H], contiguous
    sel: torch.Tensor,          # int64 [M, top_k], physical (K-sorted) expert slots
    rw16: torch.Tensor,         # fp16 [M, top_k], 0 = inactive slot (exl3_moe_coop_kernel.cuh:213-215)
    groups: list[GroupTables],
    sc: CoopScratch,
    act_limit: float,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    """Routed partial sum of this TP rank, fp32 [M, H].

    Group g writes out_g = (its experts' weighted outputs) + out_{g-1}: the kernel adds its optional
    shared-expert input sh_out to every row it stores (exl3_moe_coop_kernel.cuh:572-610, 929-947) and
    rows with no active slot get sh_out alone, so chaining gives the sum over groups with no extra
    kernel. `out` is always passed whole so rows_max (and the counter layout derived from it,
    exl3_moe_coop.cu:228-238) never changes between calls.

    TODO(untested on sm_121): numerics vs the reconstruct reference, every K in the plan, bsz 1..16.
    """
    m, hidden = x16.shape
    e = ext()
    prev = None
    for i, g in enumerate(groups):
        out = sc.out[i % 2]
        e.exl3_moe_coop(
            x16, sel, rw16,
            g.lo, g.hi,             # expert-range filter: slots outside [lo, hi) are inactive
            hidden,                 # Hi (5120, a multiple of 128: no padding)
            g.g.trellis, g.g.suh, g.g.svh,
            g.u.trellis, g.u.suh, g.u.svh,
            g.d.trellis, g.d.suh, g.d.svh,
            None, None, None,       # no expert biases in DeepSeek-V4.1
            float(g.k_gu), float(g.k_gu), float(g.k_d),
            mcg, mul1,
            MOE_COOP_ACT_SILU, float(act_limit),
            True,                   # gated
            sc.had_g, sc.had_u, sc.gu_g, sc.gu_u, sc.act_out, sc.d_out, sc.ctr,
            out,
            prev,                   # sh_out: running sum of the previous groups
            None,                   # sh_gate_w
        )
        prev = out
    assert prev is not None
    return prev[:m]


def run_decode_coop_parallel(
    x16: torch.Tensor, sel: torch.Tensor, rw16: torch.Tensor, groups: list[GroupTables],
    scs: list[CoopScratch], streams: list[torch.cuda.Stream], act_limit: float, mcg: bool, mul1: bool,
    reduce: bool = True,
):
    """run_decode_coop with the K groups concurrent instead of chained: group i runs on side stream i with its own
    scratch (counters included) and writes only its own experts' sum (no sh_out), then the main stream joins every
    stream and adds the partial outputs in fp32. At decode sizes one layer is up to 8 small launches in sequence
    (2 per group); running the groups side by side lets one group's kernels fill the SMs another group's tail
    leaves idle. The fork / join is legal under CUDA-graph capture (side streams wait on the capturing stream and
    are joined before the call returns). VLLM_EXL3_MOE_PARALLEL_GROUPS=1 selects it."""
    m, hidden = x16.shape
    e = ext()
    main = torch.cuda.current_stream()
    outs = []
    for g, sc, st in zip(groups, scs, streams):
        st.wait_stream(main)
        with torch.cuda.stream(st):
            out = sc.out[0]
            e.exl3_moe_coop(
                x16, sel, rw16, g.lo, g.hi, hidden,
                g.g.trellis, g.g.suh, g.g.svh, g.u.trellis, g.u.suh, g.u.svh, g.d.trellis, g.d.suh, g.d.svh,
                None, None, None,
                float(g.k_gu), float(g.k_gu), float(g.k_d), mcg, mul1,
                MOE_COOP_ACT_SILU, float(act_limit), True,
                sc.had_g, sc.had_u, sc.gu_g, sc.gu_u, sc.act_out, sc.d_out, sc.ctr,
                out, None, None,
            )
        outs.append(out)
    for st in streams[:len(groups)]:
        main.wait_stream(st)
    if not reduce:
        return outs                  # per-group fp32 [rows_max, hidden] scratch; the caller sums rows [:m]
    total = outs[0][:m].clone()
    for o in outs[1:]:
        total.add_(o[:m])
    return total


# ---------------------------------------------------------------------------------------------
# Prefill / large batches: exl3_moe per K group (fused tier) + reconstruct for hot experts


def bucket_rows(phys_flat: torch.Tensor, g: GroupTables, flat_tok: torch.Tensor, flat_w16: torch.Tensor):
    """Sort the batch's assignments by group-local expert id; picks outside the group go to the
    sentinel bin E (exl3_moe.cu:200-203: the last count is ignored). Counts come from scatter_add,
    not bincount: torch.bincount reads max() back to the host, which a CUDA graph cannot take."""
    n = g.size
    loc = phys_flat - g.lo
    inb = (loc >= 0) & (loc < n)
    loc = torch.where(inb, loc, torch.full_like(loc, n))
    order = torch.argsort(loc)
    counts = torch.zeros((n + 1,), dtype=torch.long, device=loc.device)
    counts.scatter_add_(0, loc, torch.ones_like(loc))
    return counts, flat_tok[order].contiguous(), flat_w16[order].contiguous()


def run_exl3_moe(
    x16: torch.Tensor, out32: torch.Tensor, g: GroupTables, counts: torch.Tensor,
    token_sorted: torch.Tensor, weight_sorted: torch.Tensor, fs: FusedScratch,
    act_limit: float, mcg: bool, mul1: bool,
    num_active: int = -1, count_lo: int = 1, count_hi: int | None = None, m_tile: int = 16,
    output_scratch: torch.Tensor | None = None, fused_base: torch.Tensor | None = None,
) -> None:
    """One exl3_moe launch over group g: accumulates into out32 with atomics (not bit-reproducible), or, with
    output_scratch [slots, H] fp32 + fused_base [n] int64, writes every assignment's weighted output into its own slot
    (row r of expert e -> fused_base[e] + r) for exl3_moe_gather to sum in k order (exl3_moe.cu:575-697).
    num_active = -1 under CUDA graphs (exl3_moe.cu:254-257)."""
    ext().exl3_moe(
        x16, out32, counts, token_sorted, weight_sorted,
        fs.tsg, fs.tsu, fs.tig, fs.tiu,
        MOE_ACT_SILU,
        float(g.k_gu), float(g.k_gu), float(g.k_d),
        g.g.trellis, g.g.suh, g.g.svh,
        g.u.trellis, g.u.suh, g.u.svh,
        g.d.trellis, g.d.suh, g.d.svh,
        mcg, mul1, mcg, mul1, mcg, mul1,
        float(act_limit),
        int(num_active),
        output_scratch, fused_base,
        int(count_lo), int(fs.rows if count_hi is None else count_hi), int(m_tile),
    )


def exl3_linear_reconstruct(
    x16: torch.Tensor, trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor,
    k: float, mcg: bool, mul1: bool,
) -> torch.Tensor:
    """y = x @ W for one EXL3 matrix, fp32 out: rotate x by suh + Hadamard, dequantize the trellis in
    the rotated basis (exact in fp16), GEMM, rotate back with svh. Same arithmetic as
    LinearEXL3.reconstruct_hgemm's unfused branch (modules/quant/exl3.py:166-223)."""
    e = ext()
    k_in, n_out = trellis.shape[0] * 16, trellis.shape[1] * 16
    xh = torch.empty_like(x16)
    e.had_r_128(x16, xh, suh, None, 1.0)
    w = _wbuf(x16.device, k_in * n_out).view(k_in, n_out)
    e.reconstruct(w, trellis, float(k), mcg, mul1)
    y = torch.empty((x16.shape[0], n_out), dtype=torch.float32, device=x16.device)
    e.hgemm(xh, w, y)
    e.had_r_128(y, y, None, svh, 1.0)
    return y


def swiglu_limit(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """silu(min(gate, limit)) * clamp(up, -limit, limit): DeepSeek-V4 / vLLM semantics
    (vllm fused_moe/activation.py:296-310) and exllamav3's act_limit (activation_kernels.cuh:178-182)."""
    if limit:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return torch.nn.functional.silu(gate) * up


def prime(device: torch.device, groups: list[GroupTables], sc: CoopScratch, fs: FusedScratch | None,
          hidden: int, top_k: int, act_limit: float, mcg: bool, mul1: bool) -> None:
    """Eager warm-up before any CUDA graph capture: the first launch of each kernel instance sets its
    shared-memory attribute (exl3_moe_coop.cu:116-125, exl3_moe.cu:495-500) and exl3_moe allocates its
    lock buffer with cudaMalloc on first use (exl3_devctx.cu:89-106), which is illegal while a stream
    is capturing. One single-token call per group activates exactly that group's instances.
    Every serve since Oct 9 captures its graphs after this priming without an allocation error."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("vllm_exl3_moe.prime() must run before CUDA graph capture")
    x = torch.zeros((1, hidden), dtype=torch.float16, device=device)
    for g in groups:
        # distinct picks inside the group (bsz 1 assumes distinct top-k, exl3_moe_coop_kernel.cuh:224-227);
        # only slot 0 carries weight, the rest are inactive (w == 0)
        sel = (g.lo + torch.arange(top_k, device=device) % g.size).to(torch.long).view(1, top_k)
        rw = torch.zeros((1, top_k), dtype=torch.float16, device=device)
        rw[0, 0] = 1.0
        run_decode_coop(x, sel, rw, [g], sc, act_limit, mcg, mul1)
        if fs is not None:
            flat = sel.reshape(-1)
            counts, tok, w = bucket_rows(flat, g, torch.zeros_like(flat), rw.reshape(-1))
            out = torch.zeros((1, hidden), dtype=torch.float32, device=device)
            run_exl3_moe(x, out, g, counts, tok, w, fs, act_limit, mcg, mul1, num_active=-1)
    torch.cuda.synchronize(device)


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default
