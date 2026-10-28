"""Stateless Phase 2a native drafter. All windows use absolute positions.

Forward semantics follow the sha-pinned official model.py read by V1. Inputs
are committed target taps or explicitly marked engine main_x fallback rows.
The stages never detach/reuse a conditioning cache across optimizer steps.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Protocol, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ...native.formats import decode_fp8_block
from ...native.official import OfficialShards
from .ops import Dense, PackedLinear, collapse, expand, frozen_linear, grouped_linear, mixes, qdq, rms, rope


class Source(Protocol):
    def get(self, name: str) -> Tensor: ...
    def info(self, name: str) -> dict[str, object]: ...
    def verify(self, name: str) -> None: ...


@dataclass(frozen=True)
class Config:
    dim: int = 5120
    inter: int = 2304
    stages: int = 3
    vocab: int = 129280
    block: int = 5
    noise: int = 128799
    hc: int = 4
    heads: int = 64
    hd: int = 512
    rd: int = 64
    qr: int = 1280
    groups: int = 8
    outr: int = 1024
    experts: int = 128
    topk: int = 3
    markov: int = 256
    window: int = 128
    eps: float = 1e-20
    hc_eps: float = 1e-6
    sinkhorn: int = 20
    theta: float = 10000.0
    route_scale: float = 1.5
    gate_temp: float = 1.0
    limit: float = 10.0


def key(name: str) -> str:
    return name.replace(".", "__")


class Tensors(nn.Module):
    def __init__(self, source: Source, device: torch.device, reference_math: bool = False):
        super().__init__()
        self.source, self.device = source, device
        self.reference_math = reference_math

    def read(self, name: str) -> Tensor:
        self.source.verify(name)
        return self.source.get(name).to(self.device)

    def add(self, name: str, train: bool = False, bf16: bool = False) -> Tensor:
        t = self.read(name).to(torch.bfloat16 if bf16 else torch.float32)
        if self.reference_math and not bf16:
            t = t.cpu()
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"nonfinite {name}")
        if train:
            self.register_parameter(key(name), nn.Parameter(t))
        else:
            self.register_buffer(key(name), t)
        return t

    def __getitem__(self, name: str) -> Tensor:
        return cast(Tensor, getattr(self, key(name)))

    def dense(self, name: str, rank: int, acc: torch.dtype, quant_input: bool = True) -> Dense:
        w = self.read(name + ".weight")
        if self.source.info(name + ".weight")["dtype"] == "F8_E4M3":
            w = decode_fp8_block(w, self.read(name + ".scale"))
        return Dense(w, rank, quant_input, acc)


class Stage(nn.Module):
    wo_a_base: Tensor

    def __init__(self, c: Config, index: int, source: Source, device: torch.device, rank: int, acc: torch.dtype,
                 reference_math: bool = False):
        super().__init__()
        self.c, self.index, self.acc = c, index, acc
        self.prefix = p = f"mtp.{index}."
        self.fast_training = False
        self.t = t = Tensors(source, device, reference_math)
        self.dense = nn.ModuleDict()
        for name in ("attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_b",
                     "ffn.shared_experts.w1", "ffn.shared_experts.w2", "ffn.shared_experts.w3"):
            self.dense[key(name)] = t.dense(p + name, rank, acc)
        # wo_a is grouped, so each output group has its own low-rank update.
        full = t.dense(p + "attn.wo_a", 0, acc, False).weight
        self.register_buffer("wo_a_base", (full.cpu() if reference_math else full).reshape(c.groups, c.outr, -1))
        self.wo_a = nn.ModuleList(Dense(w, rank, False, acc) for w in full.chunk(c.groups, 0))
        for name in ("attn_norm", "ffn_norm", "attn.q_norm", "attn.kv_norm"):
            t.add(p + name + ".weight", train=True)
        for name in ("hc_attn_fn", "hc_attn_scale", "hc_attn_base", "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base",
                     "attn.attn_sink", "ffn.gate.weight", "ffn.gate.bias"):
            t.add(p + name)
        self.experts = nn.ModuleList()
        for e in range(c.experts):
            ex = nn.ModuleDict()
            for name in ("w1", "w2", "w3"):
                ep = f"{p}ffn.experts.{e}.{name}"
                ex[name] = PackedLinear(t.read(ep + ".weight"), t.read(ep + ".scale"), acc)
            self.experts.append(ex)

    def weight(self, name: str) -> Tensor:
        return self.t[self.prefix + name]

    def norm(self, x: Tensor, name: str) -> Tensor:
        return rms(x, self.weight(name + ".weight"), self.c.eps)

    def linear(self, name: str, x: Tensor) -> Tensor:
        return cast(Tensor, self.dense[key(name)](x))

    def kv(self, x: Tensor, positions: Tensor) -> Tensor:
        z = self.norm(self.linear("attn.wkv", x), "attn.kv_norm")
        return qdq(rope(z, positions, self.c.rd, self.c.theta), self.fast_training)

    def attention(self, x: Tensor, mx: Tensor, positions: Tensor, mask: Tensor, P: Tensor
                  ) -> tuple[Tensor, Tensor]:
        c = self.c
        kv_ctx = self.kv(mx, positions)
        # Match the reference's ring-slot order (important for summation bands).
        slots = positions.remainder(c.window).argsort(-1)
        ctx = kv_ctx.gather(1, slots.unsqueeze(-1).expand_as(kv_ctx))
        valid_ctx = mask.gather(1, slots)
        query_pos = P[:, None] + 1 + torch.arange(c.block, device=x.device)
        q = self.linear("attn.wq_b", self.norm(self.linear("attn.wq_a", x), "attn.q_norm"))
        q = rope(q.reshape(*q.shape[:-1], c.heads, c.hd), query_pos, c.rd, c.theta)
        kv = torch.cat((ctx, self.kv(x, query_pos)), 1)
        valid = torch.cat((valid_ctx, torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)), 1)
        scores = torch.einsum("bshd,btd->bsht", q.to(self.acc), kv.to(self.acc)) * c.hd ** -.5
        scores = scores.masked_fill(~valid[:, None, None, :], -torch.inf)
        sink = self.weight("attn.attn_sink").to(self.acc)[None, None, :, None].expand(*scores.shape[:-1], 1)
        prob = torch.cat((scores, sink), -1).softmax(-1)[..., :-1]
        o = torch.einsum("bsht,btd->bshd", prob, kv.to(self.acc)).to(x.dtype)
        o = rope(o, query_pos, c.rd, c.theta, inverse=True).reshape(x.shape[0], c.block, c.groups, -1)
        o = qdq(o, self.fast_training)
        projected = grouped_linear(o, self.wo_a_base)
        updates = [F.linear(F.linear(o[:, :, g].to(cast(Dense, layer).a.device), cast(Dense, layer).a), cast(Dense, layer).b).to(o.device)
                   for g, layer in enumerate(self.wo_a) if cast(Dense, layer).rank]
        if updates:
            projected = projected + torch.stack(updates, -2)
        return self.linear("attn.wo_b", projected.flatten(-2)), kv_ctx

    def expert(self, x: Tensor, e: nn.ModuleDict, w: Tensor | None = None) -> Tensor:
        gate, up = e["w1"](x).float(), e["w3"](x).float()
        h = F.silu(gate.clamp(max=self.c.limit)) * up.clamp(-self.c.limit, self.c.limit)
        if w is not None:
            h = h * w
        return cast(Tensor, e["w2"](h.to(x.dtype)))

    def moe(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        c = self.c
        shape = x.shape
        x = x.reshape(-1, c.dim)
        # The official router uses F.linear FP32 even in the FP64 oracle mode.
        if self.t.reference_math:
            raw = torch.cat([F.linear(x[b:b+c.block].float(), self.weight("ffn.gate.weight"))
                             for b in range(0, x.shape[0], c.block)], 0)
        else:
            raw = F.linear(x.float(), self.weight("ffn.gate.weight"))
        score = F.softplus(raw / c.gate_temp).sqrt()
        biased = score + self.weight("ffn.gate.bias")
        idx = biased.topk(c.topk, -1).indices
        w = score.gather(1, idx)
        w = w / (w.sum(-1, keepdim=True) + 1e-20) * c.route_scale
        y = torch.zeros_like(x, dtype=torch.float32)
        if self.fast_training:
            y = self.fast_moe(x, idx, w)
        else:
            for eid in sorted(set(idx.detach().flatten().tolist())):
                rows, col = torch.where(idx == eid)
                # Per selected expert only; no full-bank dequantization in either pass.
                out = self.expert(x[rows], self.experts[eid], w[rows, col, None])
                y = y.index_add(0, rows, out.float())
        shared = nn.ModuleDict({n: self.dense[key("ffn.shared_experts." + n)] for n in ("w1", "w2", "w3")})
        y = y + self.expert(x, shared)
        return y.reshape(shape).to(x.dtype), idx.reshape(*shape[:-1], c.topk), biased

    def fast_moe(self, x: Tensor, idx: Tensor, w: Tensor) -> Tensor:
        from .fast_gpu import ExpertBank
        bank = cast(ExpertBank, self.expert_bank)
        order = idx.flatten().argsort(stable=True)
        counts = torch.bincount(idx.flatten(), minlength=self.c.experts)
        offsets = F.pad(counts.cumsum(0), (1, 0))
        tiles = ((counts + 15) // 16).cumsum(0)
        rows = order // self.c.topk
        xq = qdq(x, True)[rows]
        gate = bank.product("w1", xq, offsets, tiles)
        up = bank.product("w3", xq, offsets, tiles)
        h = F.silu(gate.clamp(max=self.c.limit)) * up.clamp(-self.c.limit, self.c.limit)
        h = qdq(h * w.flatten()[order, None], True)
        out = bank.product("w2", h, offsets, tiles)
        # Ascending expert order, as in the reference. CUDA index_add accumulates
        # only top-k terms; agreement is measured, never presumed bit-exact.
        return torch.zeros_like(x).index_add(0, rows, out)

    def hc_mix(self, x: Tensor, kind: str) -> tuple[Tensor, Tensor, Tensor]:
        c = self.c
        if self.t.reference_math and x.shape[0] > 1:
            parts = [self.hc_mix(x[b:b+1], kind) for b in range(x.shape[0])]
            return (torch.cat([p[0] for p in parts], 0), torch.cat([p[1] for p in parts], 0), torch.cat([p[2] for p in parts], 0))
        return mixes(x, self.weight(f"hc_{kind}_fn"), self.weight(f"hc_{kind}_scale"),
                     self.weight(f"hc_{kind}_base"), c.hc, c.sinkhorn, c.eps, c.hc_eps, self.acc)

    def forward(self, h: Tensor, pre: Tensor, mx: Tensor, positions: Tensor, mask: Tensor, P: Tensor,
                trace: dict[str, object] | None = None) -> tuple[Tensor, Tensor]:
        original_h, original_pre = h, pre
        if trace is None and bool((P == 0).all()):
            # Official DSparkBlock(start_pos=0) seeds context only; no block
            # transformation occurs. Stateless callers need no mutable seed.
            return h, pre
        apre, apost, acomb = self.hc_mix(h, "attn")
        aout, kv = self.attention(self.norm(collapse(h, pre), "attn_norm"), mx, positions, mask, P)
        h = expand(aout, h, apost, acomb)
        fpre, fpost, fcomb = self.hc_mix(h, "ffn")
        fin = self.norm(collapse(h, apre), "ffn_norm")
        fout, idx, biased = self.moe(fin)
        h = expand(fout, h, fpost, fcomb)
        h = torch.where((P == 0)[:, None, None, None], original_h, h)
        fpre = torch.where((P == 0)[:, None, None], original_pre, fpre)
        if trace is not None:
            trace.update(attn_out=aout.detach(), ffn_in=fin.detach(), ffn_out=fout.detach(), h_out=h.detach(),
                         kv=kv.detach(), route={"idx": idx.detach(), "biased": biased.detach()})
        return h, fpre


class Drafter(nn.Module):
    def __init__(self, source: Source, config: Config = Config(), device: str = "cpu", rank: int = 32,
                 acc: torch.dtype = torch.float32, checkpoint_stages: bool = True, reference_math: bool = False):
        super().__init__()
        torch.backends.cuda.matmul.allow_tf32 = False
        self.c, self.acc, self.checkpoint_stages = config, acc, checkpoint_stages
        self.reference_math = reference_math
        self.fast_training = False
        self.activation_device = "cpu" if reference_math else device
        identity = source.manifest if isinstance(source, OfficialShards) else getattr(source, "fingerprint", None)
        if identity is None:
            raise ValueError("tensor source must provide a stable frozen-weight identity")
        self.frozen_identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.t = t = Tensors(source, torch.device(device), reference_math)
        self.main_proj = t.dense("mtp.0.main_proj", 0, acc)
        t.add("mtp.0.main_norm.weight", train=True)
        t.add("embed.weight", bf16=True)
        t.add("head.weight", bf16=True)
        last = f"mtp.{config.stages-1}."
        for name in ("norm.weight", "markov_head.embed.weight", "markov_head.head.weight", "confidence_head.proj.weight"):
            t.add(last + name, train=True)
        self.stages = nn.ModuleList(Stage(config, i, source, t.device, rank, acc, reference_math) for i in range(config.stages))

    def enable_fast_training(self) -> None:
        if self.reference_math or self.t.device.type != "cuda":
            raise ValueError("fast training requires a separate all-GPU model")
        if self.fast_training:
            return
        from .fast_gpu import ExpertBank
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        for module_stage in self.stages:
            stage = cast(Stage, module_stage)
            stage.expert_bank = ExpertBank(stage.experts)
            stage.fast_training = True
        for module in self.modules():
            if isinstance(module, Dense):
                module.fast_training = True
        self.fast_training = True

    def last(self, name: str) -> Tensor:
        return self.t[f"mtp.{self.c.stages-1}." + name]

    def conditioning(self, taps: Tensor, engine_mx: Tensor, has_taps: Tensor,
                     main_projection: Tensor | None = None) -> Tensor:
        if main_projection is not None:
            raise ValueError("projection cache refused: detached output does not prove frozen weights AND frozen inputs; use raw taps")
        # FP64 quantized GEMM sums retain the same per-row QDQ inputs when batched.
        flat = taps.reshape(-1, taps.shape[-1])
        selected = has_taps.flatten().nonzero().flatten()
        mx = engine_mx.float().reshape(-1, self.c.dim)
        if selected.numel():
            projected = self.main_proj(flat[selected].float())
            native = rms(projected, self.t["mtp.0.main_norm.weight"], self.c.eps)
            mx = mx.index_copy(0, selected, native)
        return mx.reshape(*taps.shape[:-1], self.c.dim)

    def hidden(self, taps: Tensor, engine_mx: Tensor, has_taps: Tensor, positions: Tensor, mask: Tensor,
               anchor: Tensor, P: Tensor, trace: dict[str, Any] | None = None,
               main_projection: Tensor | None = None) -> tuple[Tensor, Tensor]:
        c = self.c
        mx = self.conditioning(taps, engine_mx, has_taps, main_projection)
        ids = torch.full((anchor.numel(), c.block), c.noise, dtype=torch.long, device=anchor.device)
        ids[:, 0] = anchor
        h = F.embedding(ids.to(self.t["embed.weight"].device), self.t["embed.weight"]).to(self.activation_device).float().unsqueeze(-2).repeat(1, 1, c.hc, 1)
        pre = torch.zeros(*h.shape[:-1], device=h.device, dtype=torch.float32)
        pre[..., 0] = 1
        for i, stage in enumerate(self.stages):
            if self.checkpoint_stages and self.training and torch.is_grad_enabled() and trace is None:
                h, pre = checkpoint(stage, h, pre, mx, positions, mask, P, use_reentrant=False)
            else:
                st: dict[str, object] = {}
                h, pre = stage(h, pre, mx, positions, mask, P, st if trace is not None else None)
                if trace is not None:
                    trace[f"s{i}"] = st
        head_x = collapse(h, pre)
        hidden = rms(head_x, self.last("norm.weight"), c.eps)
        if trace is not None:
            trace.update(head_x=head_x.detach(), head_hidden=hidden.detach(), main_x=mx.detach())
        return hidden, head_x

    def hidden_from(self, inp: Mapping[str, Tensor], trace: dict[str, Any] | None = None) -> tuple[Tensor, Tensor]:
        return self.hidden(inp["taps"], inp["engine_mx"], inp["has_taps"], inp["positions"], inp["mask"],
                           inp["anchor"], inp["P"], trace, inp.get("main_projection"))

    def logit_chunk(self, hidden: Tensor, previous: Tensor, start: int, end: int) -> Tensor:
        if self.fast_training:
            from .ops import fast_head
            base = fast_head(hidden.float(), self.t["head.weight"][start:end])
        else:
            base = frozen_linear(hidden.float(), self.t["head.weight"][start:end], acc=torch.float32)
        emb = F.embedding(previous, self.last("markov_head.embed.weight"))
        return base + F.linear(emb.float(), self.last("markov_head.head.weight")[start:end])

    def confidence(self, head_x: Tensor, previous: Tensor) -> Tensor:
        emb = F.embedding(previous, self.last("markov_head.embed.weight"))
        return F.linear(torch.cat((head_x, emb), -1).float(), self.last("confidence_head.proj.weight")).squeeze(-1)

    @torch.no_grad()
    def greedy(self, hidden: Tensor, head_x: Tensor, anchor: Tensor, chunk: int = 4096) -> tuple[Tensor, Tensor]:
        prev = anchor
        drafts, conf = [], []
        for pos in range(self.c.block):
            best = torch.full_like(prev, -torch.inf, dtype=torch.float32)
            pick = torch.zeros_like(prev)
            for lo in range(0, self.c.vocab, chunk):
                logits = self.logit_chunk(hidden[:, pos], prev, lo, min(lo + chunk, self.c.vocab))
                val, idx = logits.max(-1)
                better = val > best  # equal ties retain the lower vocabulary id
                pick = torch.where(better, idx + lo, pick)
                best = torch.maximum(best, val)
            conf.append(self.confidence(head_x[:, pos], prev).sigmoid())
            drafts.append(pick)
            prev = pick
        return torch.stack(drafts, -1), torch.stack(conf, -1)


def load(root: str, device: str = "cpu", rank: int = 32, oracle: bool = False, reference_math: bool = False,
         native_manifest: str | None = None) -> Drafter:
    source = OfficialShards(root) if native_manifest is None else OfficialShards(root, native_manifest)
    return Drafter(source, device=device, rank=rank,
                   acc=torch.float64 if oracle or reference_math else torch.float32, reference_math=reference_math)
