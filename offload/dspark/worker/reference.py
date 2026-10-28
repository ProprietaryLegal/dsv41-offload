"""Straightforward reference of the same drafter for parity checks (D3a). Slow, fp32, one request at a time.

Independent of ops.py / model.py on purpose: written from DeepSeek's reference inference/model.py (DSparkBlock,
Block.hc_mixes / hc_pre / hc_post, Gate, Expert, DSparkAttention, forward_head) with the running engine's choices
where they differ (NOTES.md): the SM12x V4 KV record, wo_a without activation quantization, EXL3 weights. Every EXL3
matrix is decoded once to its exact rotated-basis fp16 weight (exllamav3 reconstruct, bit-identical to the CPU
decoder, D1) and applied in fp32 as y = H(H(x * suh) @ W) * svh; experts are decoded on demand. No exllamav3 GEMM,
coop MoE, Triton or batched kernel of the fast path is used here.
"""

from __future__ import annotations

import json
import math
import os

import torch
import torch.nn.functional as F

from .checkpoint import Checkpoint


def _hadamard128(dev: torch.device) -> torch.Tensor:
    h = torch.ones((1, 1), dtype=torch.float64)
    while h.shape[0] < 128:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(128)).to(dev, torch.float32)


class RefLinear:
    def __init__(self, ck: Checkpoint, name: str, dev: torch.device, H: torch.Tensor, keep: bool = True):
        import exllamav3_ext as ext

        tr, suh, svh, k = ck.exl3(name)
        self.name, self.k, self.dev, self.H = name, k, dev, H
        self.in_f, self.out_f = tr.shape[0] * 16, tr.shape[1] * 16
        self.suh = suh.to(dev).float()
        self.svh = svh.to(dev).float()
        self._tr = tr
        self.w = None
        if keep:
            self.w = self.decode()
        self._ext = ext

    def decode(self) -> torch.Tensor:
        import exllamav3_ext as ext
        w = torch.empty(self.in_f, self.out_f, dtype=torch.float16, device=self.dev)
        ext.reconstruct(w, self._tr.to(self.dev), self.k, False, True)
        return w

    def _had(self, x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        return (x.view(m, n // 128, 128) @ self.H).view(m, n)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """fp32 [m, in] -> fp32 [m, out] on the exact decoded weight."""
        w = self.w if self.w is not None else self.decode()
        y = self._had(x.float() * self.suh) @ w.float()
        return self._had(y) * self.svh


class RefDrafter:
    def __init__(self, ck: Checkpoint, dev: torch.device, kv_format: str = "v4"):
        s = ck.shape
        self.s, self.dev, self.ck, self.kv_format = s, dev, ck, kv_format
        H = _hadamard128(dev)
        self.H = H
        g = lambda n: ck.get(n).to(dev)  # noqa: E731
        self.embed = g("embed.weight")
        self.head = RefLinear(ck, "head", dev, H)
        self.main_proj = RefLinear(ck, "mtp.0.main_proj", dev, H)
        self.main_norm = g("mtp.0.main_norm.weight")
        last = f"mtp.{s.n_stages - 1}."
        self.norm = g(last + "norm.weight")
        self.markov_embed = g(last + "markov_head.embed.weight")
        self.markov_head = g(last + "markov_head.head.weight")
        self.conf_w = g(last + "confidence_head.proj.weight").float()
        self.stages = []
        for i in range(s.n_stages):
            p = f"mtp.{i}."
            wo_a = ck.get(p + "attn.wo_a.weight").to(dev).float()
            sc = ck.get(p + "attn.wo_a.scale").to(dev).float()
            wo_a = wo_a * sc.repeat_interleave(32, 0).repeat_interleave(32, 1)
            self.stages.append({
                "wq_a": RefLinear(ck, p + "attn.wq_a", dev, H), "wkv": RefLinear(ck, p + "attn.wkv", dev, H),
                "wq_b": RefLinear(ck, p + "attn.wq_b", dev, H), "wo_b": RefLinear(ck, p + "attn.wo_b", dev, H),
                "wo_a": wo_a.view(s.o_groups, s.o_lora, -1),
                "attn_sink": g(p + "attn.attn_sink").float(), "q_norm": g(p + "attn.q_norm.weight"),
                "kv_norm": g(p + "attn.kv_norm.weight"), "attn_norm": g(p + "attn_norm.weight"),
                "ffn_norm": g(p + "ffn_norm.weight"),
                "hc_attn_fn": g(p + "hc_attn_fn"), "hc_attn_scale": g(p + "hc_attn_scale"),
                "hc_attn_base": g(p + "hc_attn_base"), "hc_ffn_fn": g(p + "hc_ffn_fn"),
                "hc_ffn_scale": g(p + "hc_ffn_scale"), "hc_ffn_base": g(p + "hc_ffn_base"),
                "gate_w": g(p + "ffn.gate.weight").float(), "gate_bias": g(p + "ffn.gate.bias").float(),
                "sh": [RefLinear(ck, p + f"ffn.shared_experts.{w}", dev, H) for w in ("w1", "w3", "w2")],
                "prefix": p,
            })
        # DeepSeek's precompute_freqs_cis without YaRN (original_seq_len = 0 for pure sliding-window layers)
        freqs = 1.0 / (s.rope_theta ** (torch.arange(0, s.rope_dim, 2, dtype=torch.float32) / s.rope_dim))
        self.freqs = freqs.to(dev)

    # ------------------------------------------------------------------ pieces in the reference's own form
    def norm_(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.s.rms_eps)
        return (w.float() * x).to(dtype)

    def rope_(self, x: torch.Tensor, pos: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """x [n, (h,) d]: rotate the last rope_dim channels (adjacent pairs as complex numbers)."""
        rd = self.s.rope_dim
        ang = pos.to(torch.float32).unsqueeze(-1) * self.freqs           # [n, rd/2]
        fc = torch.polar(torch.ones_like(ang), ang)
        if inverse:
            fc = fc.conj()
        y = x.clone()
        z = torch.view_as_complex(x[..., -rd:].float().contiguous().unflatten(-1, (-1, 2)))
        if z.dim() == 3:
            fc = fc.unsqueeze(1)
        y[..., -rd:] = torch.view_as_real(z * fc).flatten(-2).to(x.dtype)
        return y

    def kv_quant_(self, kv: torch.Tensor) -> torch.Tensor:
        if self.kv_format != "v4":
            return kv
        out = kv.clone()
        for gi in range(7):
            blk = kv[..., gi * 64:(gi + 1) * 64].float()
            amax = blk.abs().amax(-1, keepdim=True).clamp_min(448.0 * 2.0 ** -126)
            sc = 2.0 ** torch.ceil(torch.log2(amax / 448.0))
            out[..., gi * 64:(gi + 1) * 64] = ((blk / sc).clamp(-448, 448).to(torch.float8_e4m3fn).float() * sc
                                               ).to(kv.dtype)
        return out

    def hc_mixes_(self, x: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor):
        hc, eps = self.s.hc_mult, self.s.hc_eps
        xf = x.flatten(1).float()
        mixes = F.linear(xf, fn) * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.s.rms_eps)
        pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + eps
        post = 2 * torch.sigmoid(mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc])
        comb = mixes[:, 2 * hc:].unflatten(-1, (hc, hc)) * scale[2] + base[2 * hc:].unflatten(-1, (hc, hc))
        comb = comb.softmax(-1) + eps
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
        for _ in range(self.s.hc_iters - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + eps)
            comb = comb / (comb.sum(-2, keepdim=True) + eps)
        return pre, post, comb

    @staticmethod
    def hc_pre_(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        return torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=1).to(x.dtype)

    @staticmethod
    def hc_post_(y: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        out = post.unsqueeze(-1) * y.float().unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.float().unsqueeze(-2), dim=1)
        return out.to(y.dtype)

    # ------------------------------------------------------------------ context
    def main_x(self, taps: torch.Tensor) -> torch.Tensor:
        return self.norm_(self.main_proj(taps).to(torch.bfloat16), self.main_norm)

    def context_kv(self, mx: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        out = []
        for st in self.stages:
            kv = self.norm_(st["wkv"](mx).to(torch.bfloat16), st["kv_norm"])
            out.append(self.kv_quant_(self.rope_(kv, pos)))
        return torch.stack(out)                                         # [stages, n, 512]

    # ------------------------------------------------------------------ block
    def attention_(self, st: dict, x: torch.Tensor, pos: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        s = self.s
        k = x.shape[0]
        qr = self.norm_(st["wq_a"](x).to(torch.bfloat16), st["q_norm"])
        q = st["wq_b"](qr).to(torch.bfloat16).unflatten(-1, (s.n_heads, s.head_dim))
        q = self.rope_(q, pos)
        kv = self.norm_(st["wkv"](x).to(torch.bfloat16), st["kv_norm"])
        kv = self.kv_quant_(self.rope_(kv, pos))
        keys = torch.cat([ctx, kv], 0).float()                          # [n_ctx + k, 512]
        o = torch.empty(k, s.n_heads, s.head_dim, dtype=torch.float32, device=x.device)
        for t in range(k):
            sc = (q[t].float() @ keys.t()) * (s.head_dim ** -0.5)       # [heads, n]
            sc = torch.cat([sc, st["attn_sink"].unsqueeze(1)], 1)
            p = sc.softmax(-1)[:, :-1]
            o[t] = p @ keys
        o = self.rope_(o.to(torch.bfloat16), pos, inverse=True)
        o = o.view(k, s.o_groups, -1)
        z = torch.einsum("sgd,grd->sgr", o.float(), st["wo_a"]).to(torch.bfloat16)
        return st["wo_b"](z.flatten(1)).to(torch.bfloat16)

    def moe_(self, st: dict, x: torch.Tensor) -> torch.Tensor:
        s = self.s
        scores = F.softplus(F.linear(x.float(), st["gate_w"])).sqrt()
        idx = (scores + st["gate_bias"]).topk(s.top_k, dim=-1)[1]
        w = scores.gather(1, idx)
        w = w / (w.sum(-1, keepdim=True) + 1e-20) * s.route_scale
        y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        lim = s.swiglu_limit
        for t in range(x.shape[0]):
            for j in range(s.top_k):
                e = int(idx[t, j])
                p = st["prefix"] + f"ffn.experts.{e}."
                w1, w3, w2 = (RefLinear(self.ck, p + n, self.dev, self.H) for n in ("w1", "w3", "w2"))
                gt = w1(x[t:t + 1]).clamp(max=lim)
                up = w3(x[t:t + 1]).clamp(-lim, lim)
                a = F.silu(gt) * up * w[t, j]
                y[t] += w2(a.to(x.dtype))[0]
        w1, w3, w2 = st["sh"]
        a = F.silu(w1(x).clamp(max=lim)) * w3(x).clamp(-lim, lim)
        y += w2(a.to(x.dtype))
        return y.to(x.dtype), idx

    def block(self, anchor: int, last_pos: int, ctx: list[torch.Tensor], k: int, trace: dict | None = None):
        """One request's query block. ctx: per stage [n_ctx, 512] bf16 context KV (the valid window)."""
        s = self.s
        ids = torch.full((k,), s.noise_token, dtype=torch.int64, device=self.dev)
        ids[0] = anchor
        pos = torch.arange(last_pos + 1, last_pos + 1 + k, device=self.dev)
        x = self.embed[ids].unsqueeze(1).repeat(1, s.hc_mult, 1)
        pre_mix = torch.zeros(k, s.hc_mult, dtype=torch.float32, device=self.dev)
        pre_mix[:, 0] = 1.0
        for i, st in enumerate(self.stages):
            residual = x
            a_pre, a_post, a_comb = self.hc_mixes_(x, st["hc_attn_fn"], st["hc_attn_scale"], st["hc_attn_base"])
            h = self.norm_(self.hc_pre_(x, pre_mix), st["attn_norm"])
            h = self.attention_(st, h, pos, ctx[i])
            x = self.hc_post_(h, residual, a_post, a_comb)
            residual = x
            f_pre, f_post, f_comb = self.hc_mixes_(x, st["hc_ffn_fn"], st["hc_ffn_scale"], st["hc_ffn_base"])
            h = self.norm_(self.hc_pre_(x, a_pre), st["ffn_norm"])
            y, idx = self.moe_(st, h)
            x = self.hc_post_(y, residual, f_post, f_comb)
            pre_mix = f_pre
            if trace is not None:
                trace[f"stage{i}.ffn_in"] = h
                trace[f"stage{i}.ffn_out"] = y
                trace[f"stage{i}.stream"] = x
                trace[f"stage{i}.experts"] = idx
        return self.hc_pre_(x, pre_mix)                                 # head_hidden [k, H] bf16

    def heads(self, hh: torch.Tensor, anchor: int, k: int, trace: dict | None = None):
        """Greedy Markov drafting on bf16 logits + bias (the engine's dtypes); fp32 values kept for diagnostics."""
        base32 = self.head(self.norm_(hh, self.norm))                    # [k, V] fp32
        prev = anchor
        ids, embs, logits32 = [], [], []
        for i in range(k):
            e = self.markov_embed[prev]
            bias32 = self.markov_head.float() @ e.float()
            l16 = base32[i].to(torch.bfloat16) + bias32.to(torch.bfloat16)
            d = int(l16.argmax())
            ids.append(d)
            embs.append(e)
            logits32.append(base32[i] + bias32)
            prev = d
        feat = torch.cat([hh, torch.stack(embs)], -1).float()
        conf = torch.sigmoid(feat @ self.conf_w.view(-1))
        if trace is not None:
            trace["logits32"] = torch.stack(logits32)
            trace["base32"] = base32
        return ids, conf
