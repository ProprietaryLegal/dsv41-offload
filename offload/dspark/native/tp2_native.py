"""Native 1017 weights in the TP2 drafter: NativePacked / NativeExpanded / NativeHybrid arms over lane E's
``TPDrafter`` (same collectives, graphs, contract and server), with the deployed K2 TP2 as the control.

Arms (``ArmConfig``; presets in ARMS):
  experts   "k2" (EXL3 K2 coop kernel, the control) | "packed" (official MXFP4 bytes) | "expanded" (exact BF16)
            ``expanded_stages``: stages whose experts are expanded when experts == "packed" (NativeHybrid)
  dense     "k5" (EXL3 K5, the control: E's attention / shared-expert path, engine semantics) | "fp8" (official
            bytes + scales) | "bf16" (exact decode); with fp8/bf16 the attention and shared expert follow the
            OFFICIAL semantics (act QDQ before every linear, bf16 linear outputs, the official KV QDQ)
  kv_qdq    "official" (act_quant per 32 over all 512 KV dims after RoPE, model.py) | "engine" (the V4 fp8_ds_mla
            record the Spark's engine stores: 448 NoPE per-64 + 64 bf16 RoPE dims)        [native dense only]
  wo_a_qdq  True: act_quant(o) before the wo_a einsum (model.py) | False: bf16 o (the engine's W8A16 branch)
  combine   "official": per-slot fp32 partials all-reduced, each expert rounded to bf16, ascending-id fp32 sum, +
            bf16 shared, bf16 (MoE.forward) | "fused": one fp32 sum per rank, one all-reduce, one bf16 rounding
  head      "k6" only so far (the target-shared head stays as deployed: NDM's first condition); the BF16 head arm
            is a separate step (NATIVE.md §5)

Everything not listed (embedding, mHC, norms, routing, attention kernel, Markov, confidence) is E's code unchanged.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

import torch

from ..worker import fused as FU
from ..worker import fused_e as FE
from ..worker.checkpoint import Checkpoint
from ..worker.exl3 import kernels as K
from ..worker.exl3.bank import Exl3Bank
from ..worker.exl3.plan import PROJ_DOWN, PROJ_GATE, PROJ_UP, Exl3MoEPlan
from ..worker.model import facc_ext
from ..worker.ops import rope_cos_sin
from ..worker.tp2 import (KERNEL_PLAN, Exl3Mat, MarkovShard, Rank, TPDrafter, coop_with_shared, col_shard, row_shard,
                          vocab_split)
from . import ops as NO
from .loader import LoadStats, load_bf16, load_dense, load_expert_banks
from .official import OfficialShards

NATIVE_ROOT = os.environ.get("NATIVE_1017_ROOT", "/opt/dsv41/x86host-data/native-1017")


@dataclass(frozen=True)
class ArmConfig:
    name: str
    experts: str = "k2"
    dense: str = "k5"
    kv_qdq: str = "engine"
    wo_a_qdq: bool = False
    combine: str = "fused"
    head: str = "k6"
    expanded_stages: tuple = ()
    expert_kernel: str = "n1"         # packed experts: "n1" (ops.ExpertBank Triton pair kernels) | "kx-regfed"
    markov: str = "factorized"        # head bf16: "factorized" (exact fp32 bias computed per position) | "hot"
    markov_rows: int = 4096           #   (+ exact fp32 rows cached for the serve's most frequent previous tokens)

    def check(self) -> None:
        if self.experts not in ("k2", "packed", "expanded") or self.dense not in ("k5", "fp8", "bf16"):
            raise ValueError(f"{self.name}: experts {self.experts} / dense {self.dense}")
        if self.kv_qdq not in ("official", "engine") or self.combine not in ("official", "fused"):
            raise ValueError(f"{self.name}: kv_qdq {self.kv_qdq} / combine {self.combine}")
        if self.expert_kernel not in ("n1", "kx-regfed"):
            raise ValueError(f"{self.name}: expert kernel {self.expert_kernel}")
        if self.head not in ("k6", "bf16") or self.markov not in ("factorized", "hot"):
            raise ValueError(f"{self.name}: head {self.head} / markov {self.markov}")
        if self.combine == "official" and self.experts == "k2":
            raise ValueError(f"{self.name}: the K2 coop kernel sums its experts itself (combine must be fused)")
        if self.dense == "k5" and (self.kv_qdq != "engine" or self.wo_a_qdq):
            raise ValueError(f"{self.name}: K5 dense runs E's attention (engine KV record, no wo_a QDQ)")

    @property
    def native(self) -> bool:
        return self.experts != "k2" or self.dense != "k5" or self.head != "k6"

    def stage_expert_kind(self, i: int) -> str:
        if self.experts == "packed" and i in self.expanded_stages:
            return "expanded"
        return self.experts


OFFICIAL = dict(kv_qdq="official", wo_a_qdq=True, combine="official")
ARMS = {
    "k2-tp2": ArmConfig("k2-tp2"),                                                      # the deployed control
    "native-experts-packed": ArmConfig("native-experts-packed", experts="packed", combine="official"),
    "native-experts-expanded": ArmConfig("native-experts-expanded", experts="expanded", combine="official"),
    "native-packed": ArmConfig("native-packed", experts="packed", dense="fp8", **OFFICIAL),
    "native-expanded": ArmConfig("native-expanded", experts="expanded", dense="bf16", **OFFICIAL),
    "native-hybrid": ArmConfig("native-hybrid", experts="packed", dense="bf16", **OFFICIAL),
    "native-hybrid-s0x": ArmConfig("native-hybrid-s0x", experts="packed", dense="bf16", expanded_stages=(0,),
                                   **OFFICIAL),
    # NDM §6 second step: the draft-only official BF16 head (fp32 logits) with the exact fp32 Markov bias
    "native-hybrid-head": ArmConfig("native-hybrid-head", experts="packed", dense="bf16", head="bf16", **OFFICIAL),
    "native-expanded-head": ArmConfig("native-expanded-head", experts="expanded", dense="bf16", head="bf16",
                                      **OFFICIAL),
    "native-hybrid-head-hot": ArmConfig("native-hybrid-head-hot", experts="packed", dense="bf16", head="bf16",
                                        markov="hot", **OFFICIAL),
    "k2-head": ArmConfig("k2-head", head="bf16"),            # the head change alone on the control
    # KX-MXFP4's register-fed kernel for the packed experts (same math; their schedule, MMA and reuse)
    "native-hybrid-kx": ArmConfig("native-hybrid-kx", experts="packed", dense="bf16", expert_kernel="kx-regfed",
                                  **OFFICIAL),
    "native-packed-kx": ArmConfig("native-packed-kx", experts="packed", dense="fp8", expert_kernel="kx-regfed",
                                  **OFFICIAL),
}


@dataclass
class NativeStage:
    """StageShard's fields (so E's attention / FFN code can run on a stage whose dense part is K5) plus the native
    parts. Dense fields hold Exl3Mat (dense k5) or NativeDense (fp8 / bf16)."""
    wq_a: object
    wkv: object
    wq_b: object
    wo_b: object
    wo_a8: torch.Tensor | None
    wo_a_sc: torch.Tensor | None
    wo_a_n: object                    # NativeDense of the rank's wo_a groups (native dense) or None
    attn_sink: torch.Tensor
    q_norm: torch.Tensor
    kv_norm: torch.Tensor
    attn_norm: torch.Tensor
    ffn_norm: torch.Tensor
    hc_attn: tuple
    hc_ffn: tuple
    gate_w: torch.Tensor
    gate_bias: torch.Tensor
    sh_w1: object
    sh_w3: object
    sh_w2: object
    bank: Exl3Bank | None
    groups: list
    perm: torch.Tensor
    nbank: NO.ExpertBank | None = None
    qkv: object = None
    gu: object = None


class NativeRank:
    """Rank r's tensors for an arm: the replicated / shared parts exactly as lane E's Rank builds them, the expert
    banks and dense matrices in the arm's representation."""

    def __init__(self, r: int, dev: torch.device, ck: Checkpoint, sh: OfficialShards, cfg: ArmConfig, plan,
                 slots: int, max_model_len: int, max_rows: int, markov_cache_rows: int, tokens, tp: int,
                 stats: LoadStats):
        s = ck.shape
        self.r, self.dev, self.s, self.tp, self.cfg = r, dev, s, tp, cfg
        k_dense = ck.drafter_k
        xplan = Exl3MoEPlan.from_dict(ck.plan_json)
        heads_r = s.n_heads // tp
        self.h0, self.h1 = r * heads_r, (r + 1) * heads_r
        g_r = s.o_groups // tp
        self.g0, self.g1 = r * g_r, (r + 1) * g_r
        self.vrange = vocab_split(s.vocab, tp)[r]
        self.v0, self.v1 = self.vrange
        hd = s.head_dim
        rows_per_group = s.o_lora
        i_loc = s.inter // tp
        self.i0, self.i1 = r * i_loc, (r + 1) * i_loc
        nat = cfg.dense != "k5"

        def exl3(name, want_k):
            tr, suh, svh, k = ck.exl3(name)
            if want_k is not None and k != want_k:
                raise RuntimeError(f"{name}: trellis K{k}, config says K{want_k}")
            return tr, suh, svh, k

        def mat(parts, k, name, ext=None, force_shape=0, fixed_bucket=None):
            tr, suh, svh = parts
            return Exl3Mat(tr, suh, svh, k, dev, plan, ext, force_shape, name, fixed_bucket)

        def g(name, dtype=None):
            t = ck.get(name)
            return t.to(dev, dtype=dtype or t.dtype).contiguous()

        def dense(name, rows=None, cols=None):
            return load_dense(sh, name, dev, cfg.dense, stats, rows, cols)

        self.embed = g("embed.weight")
        self.head, self.head_bf16 = None, None
        self.mrows = None
        hot_tokens = tokens
        if cfg.head == "bf16":
            hw = load_bf16(sh, "head.weight", dev, stats, rows=(self.v0, self.v1))
            self.head_bf16 = NO.NativeDense(f"head[{self.v0}:{self.v1}]", "bf16", hw, None)
            tokens = None                     # no bf16 bias rows: the fp32 bias is exact (factorized or hot rows)
        else:
            tr, suh, svh, hk = exl3("head", ck.head_k)
            self.head = mat(col_shard(tr, suh, svh, self.v0, self.v1), hk, f"head[{self.v0}:{self.v1}]",
                            force_shape=3, fixed_bucket="head")
        if nat:
            self.main_proj = dense("mtp.0.main_proj")
        else:
            tr, suh, svh, mk = exl3("mtp.0.main_proj", k_dense["main_proj"])
            self.main_proj = mat((tr, suh, svh), mk, "main_proj", ext=facc_ext())
        self.main_norm = g("mtp.0.main_norm.weight")
        last = f"mtp.{s.n_stages - 1}."
        self.norm = g(last + "norm.weight")
        self.conf_w = g(last + "confidence_head.proj.weight", torch.float32).view(-1)
        self.markov = MarkovShard(g(last + "markov_head.embed.weight"), g(last + "markov_head.head.weight"), self.v0,
                                  self.v1, tokens, markov_cache_rows)
        if cfg.head == "bf16" and cfg.markov == "hot":
            with torch.cuda.device(dev):
                self.mrows = NO.MarkovRowsFP32(self.markov.w1, self.markov.w2, hot_tokens, cfg.markov_rows)
        self.stages: list[NativeStage] = []
        any_k2 = False
        for i in range(s.n_stages):
            p = f"mtp.{i}."
            L = s.n_target_layers + i
            lp = xplan.drafter_layers[L]
            kind = cfg.stage_expert_kind(i)
            bank, groups, nbank = None, [], None
            if kind == "k2":
                any_k2 = True
                bank = Exl3Bank(lp, s.hidden, s.inter, tp, r)
                bank.allocate(dev)
                for e in range(s.n_experts):
                    for proj, wn in ((PROJ_GATE, "w1"), (PROJ_UP, "w3"), (PROJ_DOWN, "w2")):
                        tr, suh, svh, _ = ck.exl3(f"{p}ffn.experts.{e}.{wn}")
                        bank.load_piece("trellis", proj, e, tr)
                        bank.load_piece("suh", proj, e, suh)
                        bank.load_piece("svh", proj, e, svh)
                groups, _ = bank.tables()
            elif kind == "packed" and cfg.expert_kernel == "kx-regfed":
                from .kx import KXRegisterBank
                host = load_expert_banks(sh, i, [(r, torch.device("cpu"))], tp, kind, stats)[r]
                nbank = KXRegisterBank(host, dev)
                del host
            else:
                nbank = load_expert_banks(sh, i, [(r, dev)], tp, kind, stats)[r]
            r0, r1 = self.g0 * rows_per_group, self.g1 * rows_per_group
            if nat:
                woa = dense(p + "attn.wo_a", rows=(r0, r1))
                st_dense = dict(
                    wq_a=dense(p + "attn.wq_a"), wkv=dense(p + "attn.wkv"),
                    wq_b=dense(p + "attn.wq_b", rows=(self.h0 * hd, self.h1 * hd)),
                    wo_b=dense(p + "attn.wo_b", cols=(r0, r1)),
                    wo_a8=woa.w if woa.kind == "fp8" else None, wo_a_sc=woa.sc if woa.kind == "fp8" else None,
                    wo_a_n=woa,
                    sh_w1=dense(p + "ffn.shared_experts.w1", rows=(self.i0, self.i1)),
                    sh_w3=dense(p + "ffn.shared_experts.w3", rows=(self.i0, self.i1)),
                    sh_w2=dense(p + "ffn.shared_experts.w2", cols=(self.i0, self.i1)))
            else:
                wq_a = exl3(p + "attn.wq_a", k_dense["attn.wq_a"])
                wkv = exl3(p + "attn.wkv", k_dense["attn.wq_a"])
                wq_b = exl3(p + "attn.wq_b", k_dense["attn.wq_b"])
                wo_b = exl3(p + "attn.wo_b", k_dense["attn.wo_b"])
                w1 = exl3(p + "ffn.shared_experts.w1", k_dense["ffn.shared_experts.w1"])
                w3 = exl3(p + "ffn.shared_experts.w3", k_dense["ffn.shared_experts.w1"])
                w2 = exl3(p + "ffn.shared_experts.w2", k_dense["ffn.shared_experts.w2"])
                woa = ck.get(p + "attn.wo_a.weight")
                woa_sc = ck.get(p + "attn.wo_a.scale")
                st_dense = dict(
                    wq_a=mat(wq_a[:3], wq_a[3], p + "wq_a"), wkv=mat(wkv[:3], wkv[3], p + "wkv"),
                    wq_b=mat(col_shard(*wq_b[:3], self.h0 * hd, self.h1 * hd), wq_b[3], p + "wq_b"),
                    wo_b=mat(row_shard(*wo_b[:3], r0, r1), wo_b[3], p + "wo_b"),
                    wo_a8=woa[r0:r1].contiguous().to(dev).view(torch.uint8),
                    wo_a_sc=woa_sc[r0 // 32:r1 // 32].to(dev, dtype=torch.float32).contiguous(), wo_a_n=None,
                    sh_w1=mat(col_shard(*w1[:3], self.i0, self.i1), w1[3], p + "sh_w1"),
                    sh_w3=mat(col_shard(*w3[:3], self.i0, self.i1), w3[3], p + "sh_w3"),
                    sh_w2=mat(row_shard(*w2[:3], self.i0, self.i1), w2[3], p + "sh_w2"))
            st = NativeStage(
                **st_dense,
                attn_sink=g(p + "attn.attn_sink", torch.float32)[self.h0:self.h1].contiguous(),
                q_norm=g(p + "attn.q_norm.weight"), kv_norm=g(p + "attn.kv_norm.weight"),
                attn_norm=g(p + "attn_norm.weight"), ffn_norm=g(p + "ffn_norm.weight"),
                hc_attn=(g(p + "hc_attn_fn"), g(p + "hc_attn_scale"), g(p + "hc_attn_base")),
                hc_ffn=(g(p + "hc_ffn_fn"), g(p + "hc_ffn_scale"), g(p + "hc_ffn_base")),
                gate_w=g(p + "ffn.gate.weight", torch.float32), gate_bias=g(p + "ffn.gate.bias", torch.float32),
                bank=bank, groups=groups, perm=torch.tensor(lp.perm, dtype=torch.int64, device=dev), nbank=nbank)
            self.stages.append(st)
        self.coop = K.get_coop_scratch(dev, max_rows, s.top_k, s.hidden, i_loc) if any_k2 else None
        if any_k2:
            k2 = next(st for st in self.stages if st.bank is not None)
            with torch.cuda.device(dev):
                K.prime(dev, k2.groups, self.coop, None, s.hidden, s.top_k, s.swiglu_limit, False, True)
        self.cos_sin = rope_cos_sin(max_model_len + 2048, s.rope_dim, s.rope_theta).to(dev)
        self.q_clamp = torch.tensor(max_model_len - 1, dtype=torch.int64, device=dev)
        self.ring_kv = torch.zeros(slots + 1, s.n_stages, s.window, hd, dtype=torch.bfloat16, device=dev)
        self.ring_pos = torch.full((slots + 1, s.window), -1, dtype=torch.int64, device=dev)
        self.arange_block = torch.arange(s.block + 1, dtype=torch.int64, device=dev)
        self.pool = None
        self.cap_stream = torch.cuda.Stream(dev)

    def nbytes(self) -> dict:
        def nb(t):
            return 0 if t is None else t.numel() * t.element_size()

        def mb(m):
            return m.nbytes() if m is not None else 0
        d = {"embed_bf16": nb(self.embed), "head_shard": mb(self.head) + mb(self.head_bf16),
             "main_proj": mb(self.main_proj)}
        d["stages_dense_replicated"] = sum(mb(st.wq_a) + mb(st.wkv) for st in self.stages)
        d["stages_dense_shards"] = sum(mb(st.wq_b) + mb(st.wo_b) + mb(st.sh_w1) + mb(st.sh_w3) + mb(st.sh_w2)
                                       for st in self.stages)
        d["stages_wo_a"] = sum((mb(st.wo_a_n) if st.wo_a_n is not None else nb(st.wo_a8) + nb(st.wo_a_sc))
                               for st in self.stages)
        d["stages_experts_k2"] = sum(sum(nb(t) for t in st.bank.t.values()) for st in self.stages
                                     if st.bank is not None)
        d["stages_experts_native"] = sum(st.nbank.nbytes() for st in self.stages if st.nbank is not None)
        d["stages_small"] = sum(nb(t) for st in self.stages for t in (st.attn_sink, st.q_norm, st.kv_norm,
                                                                       st.attn_norm, st.ffn_norm, st.gate_w,
                                                                       st.gate_bias, *st.hc_attn, *st.hc_ffn))
        d["markov_embed_bf16"] = nb(self.markov.w1)
        d["markov_head_shard_bf16"] = nb(self.markov.w2)
        d["markov_cache_shard_bf16"] = nb(self.markov.cache) + nb(self.markov.slot_of_token)
        d["markov_rows_fp32"] = self.mrows.nbytes() if self.mrows is not None else 0
        d["rope_cos_sin_fp32"] = nb(self.cos_sin)
        d["windows_total"] = nb(self.ring_kv) + nb(self.ring_pos)
        return d


class NativeTPDrafter(TPDrafter):
    """TPDrafter with an arm's weight representation. ``arm``: an ArmConfig or a key of ARMS."""

    def __init__(self, ck: Checkpoint, devs: list[torch.device], arm: ArmConfig | str = "native-hybrid",
                 native_root: str = NATIVE_ROOT, **kw):
        self.cfg = ARMS[arm] if isinstance(arm, str) else arm
        self.cfg.check()
        self.sh = OfficialShards(native_root)
        self.load_stats = LoadStats()
        super().__init__(ck, devs, **kw)

    # ------------------------------------------------------------------------------------------ construction
    def _make_rank(self, r, dev, ck, slots, max_model_len, max_rows, markov_cache_rows, tokens, tp):
        if not self.cfg.native:
            return super()._make_rank(r, dev, ck, slots, max_model_len, max_rows, markov_cache_rows, tokens, tp)
        return NativeRank(r, dev, ck, self.sh, self.cfg, self.plan, slots, max_model_len, max_rows,
                          markov_cache_rows, tokens, tp, self.load_stats)

    def comm_words(self) -> int:
        # official combine all-reduces [rows, 3 slots + shared, hidden] fp32 per stage
        return 32 * 4 * self.s.hidden if self.cfg.combine == "official" else 32 * self.s.hidden

    def _native_dense(self, rk) -> bool:
        return isinstance(rk, NativeRank) and self.cfg.dense != "k5"

    # ------------------------------------------------------------------------------------------ context side
    def _main_x(self, rk, taps: torch.Tensor) -> torch.Tensor:
        if not self._native_dense(rk):
            return super()._main_x(rk, taps)
        y = rk.main_proj(NO.act_qdq(taps.to(torch.bfloat16)))                 # bf16, the official linear
        return FU.rms_norm(y, rk.main_norm, self.eps)

    def _kv_official(self, rk, kv16: torch.Tensor, w: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """model.py: kv_norm -> RoPE (bf16) -> act_quant(kv, 32) in place, over all 512 dims."""
        return NO.act_qdq(FU.rope(FU.rms_norm(kv16, w, self.eps), pos, rk.cos_sin))

    def _insert_rank(self, rk, slot_rows: torch.Tensor, pos_rows: torch.Tensor, taps: torch.Tensor) -> None:
        if not self._native_dense(rk):
            return super()._insert_rank(rk, slot_rows, pos_rows, taps)
        self._insert_mx(rk, slot_rows, pos_rows, self._main_x(rk, taps))

    def insert_mx_eager(self, slot_rows: torch.Tensor, pos_rows: torch.Tensor, mx: torch.Tensor) -> None:
        if not self.cfg.native or self.cfg.dense == "k5":
            return super().insert_mx_eager(slot_rows, pos_rows, mx)
        R = self.ALL
        sr, pr, mxr = self.to_ranks(R, slot_rows), self.to_ranks(R, pos_rows), self.to_ranks(R, mx)
        self._each(R, lambda rk: self._insert_mx(rk, sr[rk.r], pr[rk.r], mxr[rk.r].to(torch.bfloat16)))

    def _insert_mx(self, rk, slot_rows: torch.Tensor, pos_rows: torch.Tensor, mx: torch.Tensor) -> None:
        """Window KV of main_x rows (bf16): wkv (official linear) -> kv_norm -> RoPE -> KV QDQ -> ring."""
        mxq = NO.act_qdq(mx)
        W = self.s.window
        for i, st in enumerate(rk.stages):
            kv = st.wkv(mxq)
            if self.cfg.kv_qdq == "engine":
                FU.kv_store(kv.float(), st.kv_norm, self.eps, pos_rows, rk.cos_sin, rk.ring_kv[:, i], slot_rows,
                            rk.ring_pos if i == 0 else None)
            else:
                rk.ring_kv[slot_rows, i, pos_rows % W] = self._kv_official(rk, kv, st.kv_norm, pos_rows)
                if i == 0:
                    rk.ring_pos[slot_rows, pos_rows % W] = pos_rows

    # ------------------------------------------------------------------------------------------ draft step
    def _attn_part(self, rk, i, st, stream, pre_in, pos, slots, lo, last, B, k):
        if not self._native_dense(rk):
            return super()._attn_part(rk, i, st, stream, pre_in, pos, slots, lo, last, B, k)
        s = self.s
        T = B * k
        pre, post, comb = FU.hc_mixes(stream, *st.hc_attn, self.eps, s.hc_eps, s.hc_iters)
        x = FU.collapse_norm(stream, pre_in, st.attn_norm, self.eps)
        xq = NO.act_qdq(x)
        qr = FU.rms_norm(st.wq_a(xq), st.q_norm, self.eps)
        kv = st.wkv(xq)
        if self.cfg.kv_qdq == "engine":
            kvb = FU.kv_rope_qdq(FU.rms_norm(kv, st.kv_norm, self.eps), pos, rk.cos_sin)
        else:
            kvb = self._kv_official(rk, kv, st.kv_norm, pos)
        hr = rk.h1 - rk.h0
        q = FU.rope(st.wq_b(NO.act_qdq(qr)).view(T, hr, s.head_dim), pos, rk.cos_sin)
        o = FU.block_attention_ring(q.view(B, k, hr, s.head_dim), rk.ring_kv[:, i], rk.ring_pos, slots, lo, last,
                                    kvb.view(B, k, s.head_dim), st.attn_sink, self.scale)
        o = FU.rope(o.view(T, hr, s.head_dim), pos, rk.cos_sin, inverse=True).view(T, -1)
        if self.cfg.wo_a_qdq:
            o = NO.act_qdq(o)
        wa = st.wo_a_n
        if wa.kind == "fp8":
            z = FU.wo_a_fp8(o, wa.w, wa.sc, rk.g1 - rk.g0)
        else:
            G = rk.g1 - rk.g0
            z = torch.bmm(o.view(T, G, -1).transpose(0, 1), wa.w.view(G, s.o_lora, -1).transpose(1, 2))
            z = z.transpose(0, 1).reshape(T, -1)
        return st.wo_b(NO.act_qdq(z), out_f32=True), (pre, post, comb)

    def _ffn_part(self, rk, st, stream, pre, trace=None, tag=""):
        if not isinstance(rk, NativeRank) or (st.nbank is None and self.cfg.dense == "k5"):
            return super()._ffn_part(rk, st, stream, pre, trace, tag)
        s = self.s
        fpre, fpost, fcomb = FU.hc_mixes(stream, *st.hc_ffn, self.eps, s.hc_eps, s.hc_iters)
        xin, x16 = FE.collapse_norm2(stream, pre, st.ffn_norm, self.eps)
        ids, sel, w, w16 = FE.route_fused(xin, st.gate_w, st.gate_bias, st.perm, s.top_k, s.route_scale)
        if trace is not None and rk.r == 0:
            trace[tag + ".experts"] = ids
            trace[tag + ".route_w"] = w
            trace[tag + ".ffn_in"] = xin
        lim = s.swiglu_limit
        xq = NO.act_qdq(xin) if (st.nbank is not None or self.cfg.dense != "k5") else None
        # shared expert partial (fp32 [T, H] over the rank's 1152 channels)
        if self.cfg.dense == "k5":
            g, u = st.sh_w1.gemm(x16), st.sh_w3.gemm(x16)
            sh = st.sh_w2.gemm(FE.swiglu_fp16(g, u, lim))
        else:
            h = NO.swiglu_bf16(st.sh_w1(xq), st.sh_w3(xq), lim)
            sh = st.sh_w2(NO.act_qdq(h), out_f32=True)
        if st.nbank is None:                       # K2 routed experts + native shared: the coop kernel adds sh
            return coop_with_shared(x16, sel, w16, st.groups, rk.coop, lim, sh, self.shared_scratch), \
                (fpre, fpost, fcomb)
        T = xin.shape[0]
        if self.cfg.combine == "official":
            buf = torch.empty(T, 4, s.hidden, dtype=torch.float32, device=xin.device)
            st.nbank.forward_slots(xq, ids, w, out=buf[:, :3])
            buf[:, 3] = sh
            return (buf, ids), (fpre, fpost, fcomb)
        slots = st.nbank.forward_slots(xq, ids, w)
        return NO.ordered_slot_sum(slots, ids, False) + sh, (fpre, fpost, fcomb)

    def forward_block(self, R, inp: dict, k: int, trace: dict | None = None):
        if not self.cfg.native:
            return super().forward_block(R, inp, k, trace)
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
            first = res[next(iter(R))][0]
            if isinstance(first, tuple):           # official combine: fp32 all-reduce of [T, 4, H], then epilogue
                tot = self.comm.all_reduce(R, {r: res[r][0][0] for r in R}, out_dtype=torch.float32)
                y = self._each(R, lambda rk: NO.moe_combine_official(tot[rk.r][:, :3], tot[rk.r][:, 3],
                                                                    res[rk.r][0][1]))
            else:
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


def _heads_bf16(self, R, hh: dict, bi: dict, B: int, k: int, n_out: int | None = None, margin: bool = False,
                trace: dict | None = None):
    """TPDrafter.heads with the official BF16 head (fp32 logits over the rank's vocabulary shard) and the exact fp32
    Markov bias (no cache): per position the shard's (max, lowest index) or top-2 meet in E's exchanges. Positions
    >= n_out: draft 0, confidence 0 (CONTRACT 1.3)."""
    s = self.s
    n = k if n_out is None else n_out
    if not 1 <= n <= k:
        raise ValueError(f"n_out {n} outside 1..{k}")

    def logits_of(rk):
        rows = FU.rms_norm(hh[rk.r], rk.norm, self.eps).view(B, k, s.hidden)[:, :n].reshape(B * n, s.hidden)
        return rk.head_bf16(rows.contiguous(), out_f32=True).view(B, n, rk.v1 - rk.v0)
    logits = self._each(R, logits_of)
    out = self._each(R, lambda rk: torch.zeros(B, k, dtype=torch.int64, device=rk.dev))
    mg = self._each(R, lambda rk: torch.zeros(B, k, dtype=torch.float32, device=rk.dev)) if margin else None
    anchors = {r: bi[r][3] for r in R}
    if trace is not None:
        trace["base_logits"] = torch.cat([logits[r].to(self.dev) for r in R], -1) if tuple(R) == self.ALL else None
    for i in range(n):
        prev = {r: anchors[r] if i == 0 else out[r][:, i - 1] for r in R}
        full = self._each(R, lambda rk: NO.markov_logits_fp32(logits[rk.r][:, i].contiguous(),
                                                              prev[rk.r].contiguous(), rk.markov.w1, rk.markov.w2,
                                                              rk.mrows))
        if not margin:
            loc = self._each(R, lambda rk: full[rk.r].max(dim=1))
            self.comm.argmax(R, self.ranks, {r: loc[r][0].contiguous() for r in R},
                             {r: loc[r][1].contiguous() for r in R}, out, i)
        else:
            loc = self._each(R, lambda rk: (full[rk.r].topk(2, dim=-1).values.contiguous(),
                                            full[rk.r].argmax(dim=-1)))
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


def _heads(self, R, hh, bi, B, k, n_out=None, margin=False, trace=None):
    if self.cfg.head == "bf16":
        return _heads_bf16(self, R, hh, bi, B, k, n_out, margin, trace)
    return TPDrafter.heads(self, R, hh, bi, B, k, n_out, margin, trace)


NativeTPDrafter.heads = _heads


def build(arm: str | ArmConfig, devs: list[torch.device], ck: Checkpoint, **kw) -> TPDrafter:
    """The drafter for an arm: the plain TPDrafter for the control, NativeTPDrafter otherwise."""
    cfg = ARMS[arm] if isinstance(arm, str) else arm
    if not cfg.native:
        return TPDrafter(ck, devs, **kw)
    return NativeTPDrafter(ck, devs, cfg, **kw)
