"""Small native-format fixtures; no downloaded or proprietary tensors."""
from __future__ import annotations

import torch
from torch import Tensor

from .model import Config


class SyntheticSource:
    def __init__(self, c: Config, seed: int = 11):
        self.fingerprint = f"synthetic-native-v1:{c}:{seed}"
        self.tensors: dict[str, Tensor] = {}
        self.dtypes: dict[str, str] = {}
        g = torch.Generator().manual_seed(seed)

        def add(name: str, shape: tuple[int, ...], kind: str = "BF16", ones: bool = False) -> None:
            value = torch.ones(shape) if ones else torch.randn(shape, generator=g) * .025
            if kind == "F8_E4M3":
                value = (value * 128).to(torch.float8_e4m3fn).view(torch.uint8)
                self.tensors[name.replace("weight", "scale")] = torch.full(
                    ((shape[0]+31)//32, (shape[1]+31)//32), 120, dtype=torch.uint8)
                self.dtypes[name.replace("weight", "scale")] = "F8_E8M0"
            self.tensors[name] = value if kind == "F8_E4M3" else value.to(torch.float32 if kind == "F32" else torch.bfloat16)
            self.dtypes[name] = kind

        add("embed.weight", (c.vocab, c.dim))
        add("head.weight", (c.vocab, c.dim))
        add("mtp.0.main_proj.weight", (c.dim, c.dim * 3), "F8_E4M3")
        add("mtp.0.main_norm.weight", (c.dim,), ones=True)
        last = f"mtp.{c.stages-1}."
        add(last + "norm.weight", (c.dim,), ones=True)
        add(last + "markov_head.embed.weight", (c.vocab, c.markov))
        add(last + "markov_head.head.weight", (c.vocab, c.markov))
        add(last + "confidence_head.proj.weight", (1, c.dim+c.markov))
        for i in range(c.stages):
            p = f"mtp.{i}."
            for name, shape in {
                "attn.wq_a": (c.qr, c.dim), "attn.wq_b": (c.heads*c.hd, c.qr),
                "attn.wkv": (c.hd, c.dim), "attn.wo_a": (c.groups*c.outr, c.heads*c.hd//c.groups),
                "attn.wo_b": (c.dim, c.groups*c.outr),
                "ffn.shared_experts.w1": (c.inter, c.dim), "ffn.shared_experts.w2": (c.dim, c.inter),
                "ffn.shared_experts.w3": (c.inter, c.dim),
            }.items():
                add(p + name + ".weight", shape, "F8_E4M3")
            for name, size in {"attn_norm": c.dim, "ffn_norm": c.dim, "attn.q_norm": c.qr, "attn.kv_norm": c.hd}.items():
                add(p+name+".weight", (size,), ones=True)
            for kind in ("attn", "ffn"):
                add(p+f"hc_{kind}_fn", ((2+c.hc)*c.hc, c.hc*c.dim), "F32")
                add(p+f"hc_{kind}_base", ((2+c.hc)*c.hc,), "F32")
                add(p+f"hc_{kind}_scale", (3,), "F32")
            add(p+"attn.attn_sink", (c.heads,), "F32")
            add(p+"ffn.gate.weight", (c.experts, c.dim))
            add(p+"ffn.gate.bias", (c.experts,), "F32")
            for e in range(c.experts):
                for name, (n, k) in {"w1": (c.inter, c.dim), "w2": (c.dim, c.inter), "w3": (c.inter, c.dim)}.items():
                    ep = f"{p}ffn.experts.{e}.{name}"
                    self.tensors[ep+".weight"] = torch.randint(0, 256, (n, k//2), generator=g, dtype=torch.uint8)
                    self.tensors[ep+".scale"] = torch.full((n, k//32), 120, dtype=torch.uint8)
                    self.dtypes[ep+".weight"], self.dtypes[ep+".scale"] = "I8", "F8_E8M0"

    def get(self, name: str) -> Tensor:
        return self.tensors[name].clone()

    def info(self, name: str) -> dict[str, object]:
        return {"dtype": self.dtypes[name], "shape": tuple(self.tensors[name].shape)}

    def verify(self, name: str) -> None:
        if name not in self.tensors:
            raise KeyError(name)


def tiny_config() -> Config:
    return Config(dim=64, inter=64, stages=3, vocab=96, noise=95, heads=2, hd=32, rd=16, qr=32,
                  groups=2, outr=32, experts=4, topk=3, markov=32, window=8)


def inputs(c: Config, batch: int = 2, P: int = 12) -> dict[str, Tensor]:
    g = torch.Generator().manual_seed(37)
    pos = torch.arange(max(0, P-c.window+1), P+1)[None].expand(batch, -1).clone()
    shape = pos.shape
    return {"taps": torch.randn(*shape, c.dim*3, generator=g).to(torch.bfloat16),
            "engine_mx": torch.zeros(*shape, c.dim), "has_taps": torch.ones(shape, dtype=torch.bool),
            "positions": pos, "mask": torch.ones(shape, dtype=torch.bool),
            "anchor": torch.arange(batch) + 3, "P": torch.full((batch,), P, dtype=torch.long)}
