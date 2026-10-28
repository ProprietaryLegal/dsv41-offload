"""The DSpark drafter of the served checkpoint on one RTX 3090 (SM86): weights, per-request window state, draft step.

Semantics: NOTES.md (the engine's DSpark round, traced from the running image). Tensors: the exact checkpoint bytes
(checkpoint.Checkpoint refuses anything else). Representation changes are draft-only and listed in NOTES.md.

Public surface (all device tensors, no host syncs inside, so the draft step and the insert capture into CUDA graphs):
  * ``main_x(taps)``                      [n, 15360] bf16 target taps -> [n, 5120] bf16 conditioning features
  * ``insert(slot_rows, pos_rows, mx)``  write the context KV of accepted positions into request windows
  * ``draft(slots, anchors, last_pos, k)`` -> (draft ids [B, k] int64, confidences [B, k] fp32)
  * ``reset_slots(slots)``                empty request windows
Graphs: ``capture(batch_sizes, ks, n_outs)`` records one graph per (B, k, n_out) (and a margin twin);
``draft_graphed`` replays it.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import torch

from . import env  # noqa: F401
from . import fused as FU
from .checkpoint import Checkpoint, DrafterShape
from .exl3 import kernels as K
from .exl3.bank import Exl3Bank
from .exl3.plan import PROJ_DOWN, PROJ_GATE, PROJ_UP, Exl3MoEPlan
from .ops import (MarkovBias, apply_rope, block_attention, hc_collapse, hc_mixes, hc_post, kv_record_qdq, rms_norm,
                  route, rope_cos_sin)

MUL1 = 0x83DCD12D - (1 << 32)       # the mul1 codebook marker as int32


@dataclass
class Window:
    """What a draft block attends to: the ring rows of ``slots`` whose position lies in [lo, last] (int64 [B] each).
    ``kv`` [B, n_stages, W, D] / ``valid`` [B, W] are the gathered rows, filled only for the unfused (ops) path; the
    fused path reads the ring in place (fused.block_attention_ring)."""
    slots: torch.Tensor
    lo: torch.Tensor
    last: torch.Tensor
    kv: torch.Tensor | None = None
    valid: torch.Tensor | None = None


def facc_ext():
    """exllamav3_ext_facc: the same exllamav3 1.6.0 source built with fp32 MMA accumulation on sm_86 (stock 1.6.0
    accumulates exl3_gemm in fp16 on sm_86 only, EXL3_GEMM_H_ACC; ENV.md). Raises if it is not installed."""
    import exllamav3_ext_facc  # type: ignore[import-not-found]
    return exllamav3_ext_facc


class Exl3Linear:
    """One EXL3 matrix on exllamav3 1.6.0: fp16 input, fp32 output (the plugin's prompt-size path; NOTES.md).

    ``prescale``: divide the input by the power of two that brings max |x| under 2^14 and multiply the output back
    (exact; the plugin does this for main_proj, whose input is the target's raw residual stream).
    ``ext`` / ``force_shape``: call ``ext.exl3_gemm`` directly (``ext`` defaults to the stock exllamav3_ext) with a
    fixed kernel shape instead of LinearEXL3's timing-based autotune: the kernel, and so each row's result, is then
    the same for every row count the shape covers (1..16 for the 16-row shapes) and in every process."""

    def __init__(self, trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, k: int, dev: torch.device,
                 prescale: bool = False, ext=None, force_shape: int = 0):
        from exllamav3.modules.quant.exl3 import LinearEXL3

        self.in_features, self.out_features, self.k = trellis.shape[0] * 16, trellis.shape[1] * 16, k
        self.trellis = trellis.to(dev).contiguous()
        self.suh = suh.to(dev).contiguous()
        self.svh = svh.to(dev).contiguous()
        self.mul1 = torch.tensor(MUL1, dtype=torch.int32, device=dev)
        self.lin = LinearEXL3(None, self.in_features, self.out_features, suh=self.suh, svh=self.svh,
                              trellis=self.trellis, mul1=self.mul1)
        self.prescale = prescale
        self.direct = ext is not None or force_shape > 0
        if self.direct:
            if ext is None:
                import exllamav3_ext as ext
            self.ext = ext
        self.force_shape = force_shape

    def _gemm(self, x16: torch.Tensor, out: torch.Tensor | None) -> torch.Tensor:
        if not self.direct:
            if out is not None:
                raise ValueError("out= needs the direct exl3_gemm path")
            return self.lin.forward(x16, {}, out_dtype=torch.float32)
        if out is None:
            out = torch.empty(x16.shape[0], self.out_features, dtype=torch.float32, device=x16.device)
        self.ext.exl3_gemm(x16, self.trellis, out, self.suh, torch.empty_like(x16), self.svh, self.force_shape, False,
                           True, 0)
        return out

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """x [m, in] (bf16 / fp32 are converted to fp16; fp16 is used as is) -> fp32 [m, out]."""
        if self.prescale:
            sc = torch.exp2(torch.ceil(torch.log2(x.abs().amax().float().clamp_min(1.0) / 16384.0)).clamp_min(0.0))
            y = self._gemm((x.float() / sc).to(torch.float16).contiguous(), out)
            return y.mul_(sc)
        if x.dtype != torch.float16:
            x = x.to(torch.float16)
        return self._gemm(x.contiguous(), out)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.trellis, self.suh, self.svh))


@dataclass
class Stage:
    wq_a: Exl3Linear
    wkv: Exl3Linear
    wq_b: Exl3Linear
    wo_b: Exl3Linear
    wo_a8: torch.Tensor           # uint8 [groups * 1024, 4096]: wo_a's e4m3fn bytes (fused path: fused.wo_a_fp8)
    wo_a_sc: torch.Tensor         # fp32 [256, 128]: its UE8M0 32 x 32 block scales
    wo_a_t: torch.Tensor | None   # bf16 [groups, 4096, 1024]: wo_a dequantized exactly, transposed per group (unfused
                                  # path and tools; None when fused, see DSparkDrafter.wo_a_bf16)
    attn_sink: torch.Tensor       # fp32 [heads]
    q_norm: torch.Tensor
    kv_norm: torch.Tensor
    attn_norm: torch.Tensor
    ffn_norm: torch.Tensor
    hc_attn: tuple[torch.Tensor, torch.Tensor, torch.Tensor]   # fn fp32 [24, 4H], scale [3], base [24]
    hc_ffn: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    gate_w: torch.Tensor          # fp32 [E, H]
    gate_bias: torch.Tensor       # fp32 [E]
    sh_w1: Exl3Linear
    sh_w3: Exl3Linear
    sh_w2: Exl3Linear
    bank: Exl3Bank
    groups: list
    perm: torch.Tensor            # int64 [E] logical -> physical slot


def dequant_wo_a(weight: torch.Tensor, scale: torch.Tensor, groups: int) -> torch.Tensor:
    """FP8 e4m3 [G*R, C] x UE8M0 block scales [G*R/bs, C/bs] -> bf16 [G, C, R] (exact: e4m3 x 2^k is a bf16 value)."""
    rows, cols = weight.shape
    br, bc = rows // scale.shape[0], cols // scale.shape[1]
    s = scale.float().repeat_interleave(br, 0).repeat_interleave(bc, 1)
    w = (weight.float() * s).to(torch.bfloat16)
    if not torch.equal(w.float(), weight.float() * s):
        raise RuntimeError("wo_a dequantization is not exact in bf16")
    return w.view(groups, rows // groups, cols).transpose(1, 2).contiguous()


class DSparkDrafter:
    """All weights resident on one GPU; per-request windows in ``slots`` fixed slots (slot ``slots`` = scratch)."""

    HEAD_SHAPE = 3      # exl3_gemm shape (16, 32, 256): what the autotuner picks for the head at 1-16 rows (689 us)
    HEAD_CHUNK = 16     # head rows per GEMM call: every call stays on the 16-row shape (n_out bit-identity)

    def __init__(self, ck: Checkpoint, dev: torch.device, slots: int = 16, max_model_len: int = 327680,
                 max_rows: int = 32, kv_format: str = "v4", fused: bool = True, main_proj_fp32acc: bool = True,
                 markov_cache_rows: int = 8192):
        if not ck.verified:
            raise RuntimeError("checkpoint not identity-verified")
        t0 = time.time()
        self.dev = dev
        self.s: DrafterShape = ck.shape
        s = self.s
        self.kv_format = kv_format          # "v4": the SM12x fp8_ds_mla record values; "bf16": no quantization
        self.fused = fused                  # Triton kernels of fused.py (False: the plain torch ops of ops.py)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        self.max_model_len = max_model_len
        self.max_pos = max_model_len + 2048          # RoPE rows, like the engine's cap (patch 0033a)
        self.max_rows = max_rows
        self.eps = s.rms_eps
        self.scale = s.head_dim ** -0.5
        k_dense = ck.drafter_k
        plan = Exl3MoEPlan.from_dict(ck.plan_json)

        def lin(name: str, want_k: int | None, prescale: bool = False, ext=None, force_shape: int = 0) -> Exl3Linear:
            tr, suh, svh, k = ck.exl3(name)
            if want_k is not None and k != want_k:
                raise RuntimeError(f"{name}: trellis K{k}, config says K{want_k}")
            return Exl3Linear(tr, suh, svh, k, dev, prescale, ext, force_shape)

        def g(name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
            t = ck.get(name)
            return t.to(dev, dtype=dtype or t.dtype).contiguous()

        self.embed = g("embed.weight")                                     # bf16 [V, H]
        self.head = lin("head", ck.head_k, force_shape=self.HEAD_SHAPE)
        # main_proj with fp32 accumulation: its 15360-wide input (three nearly equal target residual streams) cancels
        # heavily, and exllamav3's sm_86 fp16 accumulators put main_x 0.37 % from fp32 (the engine: 0.11 %; D3b)
        self.main_proj_fp32acc = main_proj_fp32acc
        self.main_proj = lin("mtp.0.main_proj", k_dense["main_proj"], prescale=True,
                             ext=facc_ext() if main_proj_fp32acc else None)
        self.main_norm = g("mtp.0.main_norm.weight")
        last = f"mtp.{s.n_stages - 1}."
        self.norm = g(last + "norm.weight")
        self.conf_w = g(last + "confidence_head.proj.weight", torch.float32).view(-1)
        tokens = None
        tok_path = os.path.join(ck.root, "markov_top_tokens.json")
        if os.path.exists(tok_path):
            with open(tok_path) as f:
                tokens = json.load(f)
        # cached Markov bias rows: the serve caches the first 4,096 ids of markov_top_tokens.json; the worker caches all
        # 8,192 (same kernel, same bits; 80 % instead of 74 % of the live captures' Markov inputs hit the cache, each
        # miss costs a 66 MB GEMV, ~85 us on the 3090). +1 GB.
        self.markov = MarkovBias(g(last + "markov_head.embed.weight"), g(last + "markov_head.head.weight"), tokens,
                                 rows=markov_cache_rows)

        self.stages: list[Stage] = []
        for i in range(s.n_stages):
            p = f"mtp.{i}."
            L = s.n_target_layers + i
            lp = plan.drafter_layers[L]
            if lp.num_experts != s.n_experts:
                raise RuntimeError(f"plan layer {L}: {lp.num_experts} experts, config {s.n_experts}")
            bank = Exl3Bank(lp, s.hidden, s.inter, 1, 0)
            bank.allocate(dev)
            for e in range(s.n_experts):
                for proj, wn in ((PROJ_GATE, "w1"), (PROJ_UP, "w3"), (PROJ_DOWN, "w2")):
                    tr, suh, svh, _ = ck.exl3(f"{p}ffn.experts.{e}.{wn}")
                    bank.load_piece("trellis", proj, e, tr)
                    bank.load_piece("suh", proj, e, suh)
                    bank.load_piece("svh", proj, e, svh)
            groups, _ = bank.tables()
            st = Stage(
                wq_a=lin(p + "attn.wq_a", k_dense["attn.wq_a"]), wkv=lin(p + "attn.wkv", k_dense["attn.wq_a"]),
                wq_b=lin(p + "attn.wq_b", k_dense["attn.wq_b"]), wo_b=lin(p + "attn.wo_b", k_dense["attn.wo_b"]),
                wo_a8=g(p + "attn.wo_a.weight").view(torch.uint8), wo_a_sc=g(p + "attn.wo_a.scale", torch.float32),
                wo_a_t=None if fused else dequant_wo_a(ck.get(p + "attn.wo_a.weight").to(dev),
                                                       ck.get(p + "attn.wo_a.scale").to(dev), s.o_groups),
                attn_sink=g(p + "attn.attn_sink", torch.float32), q_norm=g(p + "attn.q_norm.weight"),
                kv_norm=g(p + "attn.kv_norm.weight"), attn_norm=g(p + "attn_norm.weight"),
                ffn_norm=g(p + "ffn_norm.weight"),
                hc_attn=(g(p + "hc_attn_fn"), g(p + "hc_attn_scale"), g(p + "hc_attn_base")),
                hc_ffn=(g(p + "hc_ffn_fn"), g(p + "hc_ffn_scale"), g(p + "hc_ffn_base")),
                gate_w=g(p + "ffn.gate.weight", torch.float32), gate_bias=g(p + "ffn.gate.bias", torch.float32),
                sh_w1=lin(p + "ffn.shared_experts.w1", k_dense["ffn.shared_experts.w1"]),
                sh_w3=lin(p + "ffn.shared_experts.w3", k_dense["ffn.shared_experts.w1"]),
                sh_w2=lin(p + "ffn.shared_experts.w2", k_dense["ffn.shared_experts.w2"]),
                bank=bank, groups=groups, perm=torch.tensor(lp.perm, dtype=torch.int64, device=dev),
            )
            self.stages.append(st)
        from .exl3.hostcopy import release_stage
        release_stage()

        self.coop = K.get_coop_scratch(dev, max_rows, s.top_k, s.hidden, s.inter)
        K.prime(dev, self.stages[0].groups, self.coop, None, s.hidden, s.top_k, s.swiglu_limit, False, True)
        self.cos_sin = rope_cos_sin(self.max_pos, s.rope_dim, s.rope_theta).to(dev)
        # query positions are clamped to max_model_len - 1 (the engine's clamp); a device scalar so CUDA graphs read
        # the current value (set_query_clamp)
        self.q_clamp = torch.tensor(max_model_len - 1, dtype=torch.int64, device=dev)

        # per-request windows: slot `slots` is a scratch slot that padded rows write to
        self.n_slots = slots
        self.ring_kv = torch.zeros(slots + 1, s.n_stages, s.window, s.head_dim, dtype=torch.bfloat16, device=dev)
        self.ring_pos = torch.full((slots + 1, s.window), -1, dtype=torch.int64, device=dev)
        self.arange_block = torch.arange(s.block + 1, dtype=torch.int64, device=dev)
        self.graphs: dict[tuple[int, int, int], dict] = {}
        self.margin_graphs: dict[tuple[int, int, int], dict] = {}
        self.reply_graphs: dict[tuple[int, int, int, bool], dict] = {}
        self.insert_graphs: dict[int, dict] = {}
        self.pool = None
        torch.cuda.synchronize(dev)
        self.load_seconds = time.time() - t0

    def wo_a_bf16(self, i: int) -> torch.Tensor:
        """Stage i's wo_a as bf16 [groups, 4096, 1024] (exact dequantization; for tools on the fused model)."""
        st = self.stages[i]
        if st.wo_a_t is not None:
            return st.wo_a_t
        return dequant_wo_a(st.wo_a8.view(torch.float8_e4m3fn), st.wo_a_sc, self.s.o_groups)

    # ------------------------------------------------------------------------------------------ op selection
    def _norm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return FU.rms_norm(x, w, self.eps) if self.fused else rms_norm(x.to(torch.bfloat16), w, self.eps)

    def _rope(self, x: torch.Tensor, pos: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        return FU.rope(x, pos, self.cos_sin, inverse) if self.fused else apply_rope(x, pos, self.cos_sin, inverse)

    def _kv(self, kv: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """rope + (V4 record) of normed KV rows."""
        if self.kv_format != "v4":
            return self._rope(kv, pos)
        if self.fused:
            return FU.kv_rope_qdq(kv, pos, self.cos_sin)
        return kv_record_qdq(apply_rope(kv, pos, self.cos_sin))

    def _mixes(self, stream, fn, scale, base):
        f = FU.hc_mixes if self.fused else hc_mixes
        return f(stream, fn, scale, base, self.eps, self.s.hc_eps, self.s.hc_iters)

    def _collapse_norm(self, stream, pre, w):
        if self.fused:
            return FU.collapse_norm(stream, pre, w, self.eps)
        return rms_norm(hc_collapse(stream, pre), w, self.eps)

    def _post(self, y, stream, post, comb):
        return (FU.hc_post if self.fused else hc_post)(y, stream, post, comb)

    def _collapse(self, stream, pre):
        return FU.collapse(stream, pre) if self.fused else hc_collapse(stream, pre)

    # ------------------------------------------------------------------------------------------ context side
    def main_x(self, taps: torch.Tensor) -> torch.Tensor:
        """[n, 3H] bf16 taps of target layers 37/38/39 -> main_x [n, H] bf16."""
        if self.fused and isinstance(self.main_proj, Exl3Linear):
            x16, sc = FU.prescale_fp16(taps)
            return FU.scaled_rms_norm(self.main_proj._gemm(x16, None), sc, self.main_norm, self.eps)
        return self._norm(self.main_proj(taps), self.main_norm)

    def context_kv(self, mx: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Stage context KV [n_stages, n, D] (rope at pos, record-quantized) of main_x rows."""
        out = []
        for st in self.stages:
            out.append(self._kv(self._norm(st.wkv(mx), st.kv_norm), pos))
        return torch.stack(out)

    def insert(self, slot_rows: torch.Tensor, pos_rows: torch.Tensor, mx: torch.Tensor) -> None:
        """Write the context KV of rows (slot, position, main_x) into the windows. Callers pass at most the last
        `window` positions of a request per call (older ones would be overwritten anyway; duplicates are refused
        upstream). Padded rows use slot == n_slots (scratch)."""
        if self.fused and self.kv_format == "v4":
            mx16 = mx.to(torch.float16)
            for i, st in enumerate(self.stages):
                FU.kv_store(st.wkv(mx16), st.kv_norm, self.eps, pos_rows, self.cos_sin, self.ring_kv[:, i], slot_rows,
                            self.ring_pos if i == 0 else None)
            return
        kv = self.context_kv(mx, pos_rows)                          # [S, n, D]
        idx = pos_rows % self.s.window
        self.ring_kv[slot_rows, :, idx] = kv.transpose(0, 1)
        self.ring_pos[slot_rows, idx] = pos_rows

    def reset_slots(self, slots: torch.Tensor) -> None:
        self.ring_pos[slots] = -1
        self.ring_kv[slots] = 0

    # ------------------------------------------------------------------------------------------ draft step
    def _attention(self, st: Stage, i: int, x: torch.Tensor, pos: torch.Tensor, win: Window, B: int, k: int
                   ) -> torch.Tensor:
        s = self.s
        T = B * k
        x16 = x.to(torch.float16)
        qr = self._norm(st.wq_a(x16), st.q_norm)
        kvb = self._kv(self._norm(st.wkv(x16), st.kv_norm), pos)
        q = self._rope(st.wq_b(qr).to(torch.bfloat16).view(T, s.n_heads, s.head_dim), pos)
        qb, kb = q.view(B, k, s.n_heads, s.head_dim), kvb.view(B, k, s.head_dim)
        if win.kv is None:
            o = FU.block_attention_ring(qb, self.ring_kv[:, i], self.ring_pos, win.slots, win.lo, win.last, kb,
                                        st.attn_sink, self.scale)
        else:
            o = block_attention(qb, win.kv[:, i], win.valid, kb, st.attn_sink, self.scale)
        o = self._rope(o.view(T, s.n_heads, s.head_dim), pos, inverse=True)
        if st.wo_a_t is None:
            z = FU.wo_a_fp8(o.view(T, -1), st.wo_a8, st.wo_a_sc, s.o_groups)  # [T, G*1024] bf16
        else:
            og = o.view(T, s.o_groups, -1).transpose(0, 1)                   # [G, T, 4096]
            z = torch.bmm(og, st.wo_a_t).transpose(0, 1).reshape(T, -1)      # [T, G*1024] bf16
        return st.wo_b(z).to(torch.bfloat16)

    def _moe(self, st: Stage, x: torch.Tensor, trace: dict | None = None, tag: str = "") -> torch.Tensor:
        s = self.s
        ids, w = route(x, st.gate_w, st.gate_bias, s.top_k, s.route_scale)
        if trace is not None:
            trace[tag + ".experts"] = ids
            trace[tag + ".route_w"] = w
        x16 = x.to(torch.float16).contiguous()
        sel = st.perm[ids].contiguous()
        routed = K.run_decode_coop(x16, sel, w.to(torch.float16).contiguous(), st.groups, self.coop,
                                   s.swiglu_limit, False, True)
        g = st.sh_w1(x16)
        u = st.sh_w3(x16)
        lim = s.swiglu_limit
        a = torch.nn.functional.silu(g.clamp(max=lim)) * u.clamp(min=-lim, max=lim)
        sh = st.sh_w2(a.to(torch.float16))
        return (routed + sh).to(torch.bfloat16)

    def forward_block(self, ids: torch.Tensor, pos: torch.Tensor, win: Window, trace: dict | None = None
                      ) -> torch.Tensor:
        """The three stages over a query block. ids/pos [B, k]; win: the context window (block_inputs).
        Returns head_hidden [B*k, H] bf16 (pre-norm). ``trace`` (eager only) collects per-stage tensors."""
        s = self.s
        B, k = ids.shape
        T = B * k
        pos_f = pos.reshape(T)
        x = self.embed[ids.reshape(T)]
        stream = x.unsqueeze(1).expand(T, s.hc_mult, s.hidden).contiguous()
        pre_in = None
        for i, st in enumerate(self.stages):
            pre, post, comb = self._mixes(stream, *st.hc_attn)
            xin = self._collapse_norm(stream, pre_in, st.attn_norm)
            a = self._attention(st, i, xin, pos_f, win, B, k)
            stream = self._post(a, stream, post, comb)
            fpre, fpost, fcomb = self._mixes(stream, *st.hc_ffn)
            xin = self._collapse_norm(stream, pre, st.ffn_norm)
            y = self._moe(st, xin, trace, f"stage{i}")
            stream = self._post(y, stream, fpost, fcomb)
            pre_in = fpre
            if trace is not None:
                trace[f"stage{i}.attn_out"] = a
                trace[f"stage{i}.ffn_in"] = xin
                trace[f"stage{i}.ffn_out"] = y
                trace[f"stage{i}.stream"] = stream
        return self._collapse(stream, pre_in)

    def heads(self, hh: torch.Tensor, anchors: torch.Tensor, B: int, k: int, trace: dict | None = None,
              n_out: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy sequential Markov drafting + confidence over head_hidden [B*k, H], for the first ``n_out`` (default
        k) positions of every request; later positions return draft 0 and confidence 0. Positions < n_out are
        bit-identical for every n_out (CONTRACT 1.3): head rows go through the GEMM in chunks of <= HEAD_CHUNK rows on
        one fixed kernel shape, Markov and argmax are per row, and the confidence product keeps its [B, k] shape
        (zero Markov embeddings past n_out)."""
        s = self.s
        n = k if n_out is None else n_out
        if not 1 <= n <= k:
            raise ValueError(f"n_out {n} outside 1..{k}")
        rows = self._norm(hh, self.norm).view(B, k, s.hidden)[:, :n].reshape(B * n, s.hidden).to(torch.float16)
        l32 = torch.empty(B * n, s.vocab, dtype=torch.float32, device=self.dev)
        for c in range(0, B * n, self.HEAD_CHUNK):
            self.head(rows[c:c + self.HEAD_CHUNK], out=l32[c:c + self.HEAD_CHUNK])
        logits = l32.to(torch.bfloat16).view(B, n, s.vocab)
        out = torch.zeros(B, k, dtype=torch.int64, device=self.dev)
        if self.fused and trace is None:
            for i in range(n):
                FU.markov_argmax_into(logits[:, i], anchors if i == 0 else out[:, i - 1], self.markov, out, i)
            emb = self.markov.embed(torch.cat((anchors.unsqueeze(1), out[:, :n - 1]), 1))      # [B, n, R]
        else:
            prev = anchors
            embs = []
            for i in range(n):
                e = self.markov.embed(prev)
                li = logits[:, i] + self.markov(prev, e)
                d = li.argmax(dim=-1)
                if trace is not None:
                    trace[f"markov{i}.logits"] = li
                out[:, i] = d
                embs.append(e)
                prev = d
            emb = torch.stack(embs, 1)
        if n < k:
            emb = torch.cat((emb, emb.new_zeros(B, k - n, emb.shape[-1])), 1)
        feat = torch.cat((hh.view(B, k, s.hidden), emb), dim=-1).float()
        conf = torch.sigmoid(feat @ self.conf_w)
        if n < k:
            conf[:, n:] = 0.0
        if trace is not None:
            trace["base_logits"] = logits
        return out, conf

    def block_inputs(self, slots: torch.Tensor, anchors: torch.Tensor, last_pos: torch.Tensor, k: int,
                     ctx_lo: torch.Tensor | None = None, gather: bool | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor, Window]:
        """Query ids/positions and the window: ring rows whose position is in [ctx_lo, last_pos]. ctx_lo defaults to
        max(last_pos - W + 1, 0); the server passes max(lo, anchor_pos - W) with the slot's contiguous lower bound.
        ``gather`` (default: not fused) also materializes the window rows and their mask."""
        s = self.s
        ar = self.arange_block[:k]
        ids = torch.where(ar.unsqueeze(0) == 0, anchors.unsqueeze(1), torch.full_like(ar, s.noise_token).unsqueeze(0))
        pos = torch.minimum(last_pos.unsqueeze(1) + 1 + ar.unsqueeze(0), self.q_clamp)
        if ctx_lo is None:
            ctx_lo = (last_pos - (s.window - 1)).clamp_min(0)
        win = Window(slots, ctx_lo, last_pos)
        if gather is None:
            gather = not self.fused
        if gather:
            cpos = self.ring_pos[slots]
            win.valid = (cpos >= ctx_lo.unsqueeze(1)) & (cpos <= last_pos.unsqueeze(1))
            win.kv = self.ring_kv[slots]
        return ids, pos, win

    def draft(self, slots: torch.Tensor, anchors: torch.Tensor, last_pos: torch.Tensor, k: int,
              trace: dict | None = None, ctx_lo: torch.Tensor | None = None, n_out: int | None = None
              ) -> tuple[torch.Tensor, torch.Tensor]:
        """Eager draft step. slots/anchors/last_pos (/ctx_lo) int64 [B] on the device. The body always runs the
        whole k-token block; ``n_out`` limits the head / Markov / confidence to the first n_out positions."""
        B = slots.shape[0]
        ids, pos, win = self.block_inputs(slots, anchors, last_pos, k, ctx_lo)
        hh = self.forward_block(ids, pos, win, trace)
        if trace is not None:
            trace["head_hidden"] = hh
        return self.heads(hh, anchors, B, k, trace, n_out)

    def set_query_clamp(self, max_model_len: int) -> None:
        if not 0 < max_model_len <= self.max_model_len:
            raise ValueError(f"max_model_len {max_model_len} outside 1..{self.max_model_len}")
        self.q_clamp.fill_(max_model_len - 1)

    def draft_with_margin(self, slots: torch.Tensor, anchors: torch.Tensor, last_pos: torch.Tensor, k: int,
                          ctx_lo: torch.Tensor | None = None, n_out: int | None = None
                          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Eager draft that also returns top1 - top2 of (base + Markov) logits per position (CONTRACT WANT_MARGIN;
        0 past n_out). Same drafts as draft(): the argmax runs on the same bf16 logits."""
        tr: dict = {}
        ids, conf = self.draft(slots, anchors, last_pos, k, trace=tr, ctx_lo=ctx_lo, n_out=n_out)
        n = k if n_out is None else n_out
        mg = torch.zeros(slots.shape[0], k, dtype=torch.float32, device=self.dev)
        for i in range(n):
            v = tr[f"markov{i}.logits"].float().topk(2, dim=-1).values
            mg[:, i] = v[:, 0] - v[:, 1]
        return ids, conf, mg

    # ------------------------------------------------------------------------------------------ CUDA graphs
    def capture(self, batch_sizes=(1, 2, 3, 4), ks=(4, 5), insert_rows=(8, 16, 32, 64, 128), n_outs=(4, 5),
                margins: bool = True, pack_reply: bool = False) -> float:
        """One graph per (B, k, n_out) (n_out in n_outs and <= k; n_out = k always) for the draft step, the same with
        top1 - top2 margins (CONTRACT WANT_MARGIN; shadow mode asks for them) when ``margins``, and one per row
        bucket for the insert. ``pack_reply`` adds optional float64 reply twins;
        default False, so normal capture count and outputs remain unchanged. Returns seconds."""
        t0 = time.time()
        dev = self.dev
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        bmax = max(batch_sizes)
        self.g_slots = torch.full((bmax,), self.n_slots, dtype=torch.int64, device=dev)
        self.g_anchor = torch.zeros(bmax, dtype=torch.int64, device=dev)
        self.g_last = torch.zeros(bmax, dtype=torch.int64, device=dev)
        self.g_lo = torch.zeros(bmax, dtype=torch.int64, device=dev)
        rmax = max(insert_rows)
        self.gi_slot = torch.full((rmax,), self.n_slots, dtype=torch.int64, device=dev)
        self.gi_pos = torch.zeros(rmax, dtype=torch.int64, device=dev)
        self.gi_taps = torch.zeros(rmax, self.s.hidden * len(self.s.taps), dtype=torch.bfloat16, device=dev)
        s1 = torch.cuda.Stream(dev)
        for B in batch_sizes:
            for k in ks:
                for n in sorted({x for x in n_outs if x <= k} | {k}):
                    args = (self.g_slots[:B], self.g_anchor[:B], self.g_last[:B], k)
                    lo = self.g_lo[:B]
                    s1.wait_stream(torch.cuda.current_stream(dev))
                    with torch.cuda.stream(s1):
                        for _ in range(2):
                            self.draft(*args, ctx_lo=lo, n_out=n)   # warm-up: kernel attributes, buffers, Triton JIT
                    torch.cuda.current_stream(dev).wait_stream(s1)
                    torch.cuda.synchronize(dev)
                    gr = torch.cuda.CUDAGraph()
                    # A process-wide default capture stream can belong to another device (e.g. TP2 rank 0).
                    # Keep the pool on this model's stream, regardless of which model captured first.
                    with torch.cuda.graph(gr, pool=self.pool, stream=s1):
                        ids, conf = self.draft(*args, ctx_lo=lo, n_out=n)
                    self.graphs[(B, k, n)] = {"graph": gr, "ids": ids, "conf": conf}
                    if margins:
                        s1.wait_stream(torch.cuda.current_stream(dev))
                        with torch.cuda.stream(s1):
                            for _ in range(2):
                                self.draft_with_margin(*args, ctx_lo=lo, n_out=n)
                        torch.cuda.current_stream(dev).wait_stream(s1)
                        torch.cuda.synchronize(dev)
                        gr = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(gr, pool=self.pool, stream=s1):
                            ids, conf, mg = self.draft_with_margin(*args, ctx_lo=lo, n_out=n)
                        self.margin_graphs[(B, k, n)] = {"graph": gr, "ids": ids, "conf": conf, "mg": mg}
                    if pack_reply:
                        for margin in ((False, True) if margins else (False,)):
                            gr = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(gr, pool=self.pool, stream=s1):
                                values = (self.draft_with_margin(*args, ctx_lo=lo, n_out=n) if margin
                                          else self.draft(*args, ctx_lo=lo, n_out=n))
                                packed = torch.cat([v.to(torch.float64) for v in values], 1)
                            self.reply_graphs[(B, k, n, margin)] = {"graph": gr, "packed": packed}
        for n in insert_rows:
            args = (self.gi_slot[:n], self.gi_pos[:n])
            s1.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(s1):
                for _ in range(2):
                    self.insert(*args, self.main_x(self.gi_taps[:n]))
            torch.cuda.current_stream(dev).wait_stream(s1)
            torch.cuda.synchronize(dev)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, pool=self.pool, stream=s1):
                self.insert(*args, self.main_x(self.gi_taps[:n]))
            self.insert_graphs[n] = {"graph": gr}
        # the warm-ups wrote into the scratch slot only; keep it empty
        self.reset_slots(torch.tensor([self.n_slots], device=dev))
        torch.cuda.synchronize(dev)
        return time.time() - t0

    def has_graph(self, B: int, k: int, n_out: int | None = None, margin: bool = False) -> bool:
        return (B, k, k if n_out is None else n_out) in (self.margin_graphs if margin else self.graphs)

    def draft_graphed(self, slots: torch.Tensor, anchors: torch.Tensor, last_pos: torch.Tensor, k: int,
                      ctx_lo: torch.Tensor | None = None, n_out: int | None = None, margin: bool = False,
                      packed_reply: bool = False
                      ) -> tuple[torch.Tensor, ...]:
        """Replay the (B, k, n_out) graph; returns views of the graph's output buffers (ids, conf[, margin]; copy
        before the next replay). ``margin``: the draft_with_margin graph (same ids and confidences).
        ``packed_reply`` returns a one-tuple holding the captured float64 wire staging tensor;
        requires capture(pack_reply=True), including a margin twin when margin=True."""
        B = slots.shape[0]
        n = k if n_out is None else n_out
        g = (self.reply_graphs.get((B, k, n, margin)) if packed_reply
             else (self.margin_graphs if margin else self.graphs).get((B, k, n)))
        if g is None:
            raise KeyError(f"no CUDA graph for batch {B}, depth {k}, n_out {n}, margin {margin}, packed {packed_reply}")
        self.g_slots[:B].copy_(slots, non_blocking=True)
        self.g_anchor[:B].copy_(anchors, non_blocking=True)
        self.g_last[:B].copy_(last_pos, non_blocking=True)
        if ctx_lo is None:
            ctx_lo = (last_pos - (self.s.window - 1)).clamp_min(0)
        self.g_lo[:B].copy_(ctx_lo, non_blocking=True)
        g["graph"].replay()
        if packed_reply:
            return (g["packed"],)
        return (g["ids"], g["conf"], g["mg"]) if margin else (g["ids"], g["conf"])

    def insert_taps_graphed(self, slot_rows: torch.Tensor, pos_rows: torch.Tensor, taps: torch.Tensor) -> None:
        """main_x + insert for n rows through the smallest captured bucket >= n (padding rows hit the scratch slot)."""
        n = slot_rows.shape[0]
        sizes = sorted(self.insert_graphs)
        b = next((x for x in sizes if x >= n), None)
        if b is None:
            raise ValueError(f"{n} insert rows exceed the largest graph bucket {sizes[-1]}")
        self.gi_slot[:n].copy_(slot_rows, non_blocking=True)
        self.gi_slot[n:b].fill_(self.n_slots)
        self.gi_pos[:n].copy_(pos_rows, non_blocking=True)
        self.gi_pos[n:b].zero_()
        self.gi_taps[:n].copy_(taps, non_blocking=True)
        self.gi_taps[n:b].zero_()           # padding rows: no stale taps in main_proj's batch-wide prescale
        self.insert_graphs[b]["graph"].replay()

    # ------------------------------------------------------------------------------------------ accounting
    def memory_table(self) -> dict:
        def nb(t: torch.Tensor) -> int:
            return t.numel() * t.element_size()

        st_dense = sum(sum(l.nbytes() for l in (st.wq_a, st.wkv, st.wq_b, st.wo_b, st.sh_w1, st.sh_w3, st.sh_w2))
                       for st in self.stages)
        st_woa = sum(nb(st.wo_a8) + nb(st.wo_a_sc) + (nb(st.wo_a_t) if st.wo_a_t is not None else 0)
                     for st in self.stages)
        st_exp = sum(sum(nb(t) for t in st.bank.t.values()) for st in self.stages)
        st_small = sum(nb(t) for st in self.stages for t in (st.attn_sink, st.q_norm, st.kv_norm, st.attn_norm,
                                                               st.ffn_norm, st.gate_w, st.gate_bias, *st.hc_attn,
                                                               *st.hc_ffn))
        return {
            "embed_bf16": nb(self.embed),
            "head_exl3_k6": self.head.nbytes(),
            "main_proj_exl3_k5": self.main_proj.nbytes(),
            "stages_dense_exl3_k5": st_dense,
            "stages_wo_a_fp8": st_woa,
            "stages_experts_exl3_k2": st_exp,
            "stages_norms_hc_gate_fp32": st_small,
            "markov_w1_w2_bf16": nb(self.markov.w1) + nb(self.markov.w2),
            "markov_cache_rows_bf16": nb(self.markov.cache) + nb(self.markov.slot_of_token),
            "confidence_and_norms": nb(self.conf_w) + nb(self.norm) + nb(self.main_norm),
            "rope_cos_sin_fp32": nb(self.cos_sin),
            "windows_per_slot": nb(self.ring_kv[0]) + nb(self.ring_pos[0]),
            "windows_total": nb(self.ring_kv) + nb(self.ring_pos),
            "coop_scratch": sum(nb(t) for t in (self.coop.had_g, self.coop.had_u, self.coop.gu_g, self.coop.gu_u,
                                                self.coop.act_out, self.coop.d_out, self.coop.ctr, *self.coop.out)),
            "torch_allocated": torch.cuda.memory_allocated(self.dev),
            "torch_reserved": torch.cuda.memory_reserved(self.dev),
            "torch_max_allocated": torch.cuda.max_memory_allocated(self.dev),
        }
