"""Evaluation-only native base plus FP32 low-rank delta. Never merges weights.

TP convention: column/output sharding slices B rows; row/input sharding slices
A columns, with partials reduced by the existing worker. Install once BEFORE
capture. Every trained norm/Markov/confidence tensor is consumed or we refuse.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Protocol, cast

import torch
from torch import Tensor
from torch.nn import functional as F

from ...native.arms import ArmConfig
from .cache_guard import DependencyGuard
from .model import Config


class NativeLinear(Protocol):
    in_features: int
    out_features: int
    def __call__(self, x: Tensor, out_f32: bool = False, out: Tensor | None = None) -> Tensor: ...
    def nbytes(self) -> int: ...


class DeltaLinear:
    def __init__(self, base: NativeLinear, a: Tensor, b: Tensor):
        if a.ndim != 2 or b.shape != (base.out_features, a.shape[0]) or a.shape[1] != base.in_features:
            raise ValueError("adapter shard shape differs from native base")
        self.base, self.a, self.b = base, a.detach().float().contiguous(), b.detach().float().contiguous()
        self.in_features, self.out_features = base.in_features, base.out_features

    def __call__(self, x: Tensor, out_f32: bool = False, out: Tensor | None = None) -> Tensor:
        # x has already undergone the model's input QDQ. Add BEFORE output
        # rounding and, for row-parallel projections, before the all-reduce.
        value = self.base(x, out_f32=True).float()+F.linear(F.linear(x.float(), self.a), self.b)
        value = value if out_f32 else value.to(x.dtype)
        if out is not None:
            out.copy_(value)
            return out
        return value

    def nbytes(self) -> int:
        return self.base.nbytes()+4*(self.a.numel()+self.b.numel())


class GroupedDelta:
    def __init__(self, weight: Tensor, factors: list[tuple[Tensor, Tensor]]):
        if weight.ndim != 3 or len(factors) != weight.shape[0]:
            raise ValueError("grouped adapter ownership mismatch")
        self.weight, self.factors = weight, factors
        if any(a.ndim != 2 or a.shape[1] != weight.shape[2] or b.shape != (weight.shape[1], a.shape[0]) for a, b in factors):
            raise ValueError("grouped adapter factor shape mismatch")

    def __call__(self, x: Tensor) -> Tensor:
        grouped = x.reshape(x.shape[0], len(self.factors), -1).float()
        base = torch.einsum("tgd,god->tgo", grouped, self.weight.float())
        delta = torch.stack([F.linear(F.linear(grouped[:, g], a), b) for g, (a, b) in enumerate(self.factors)], 1)
        return (base+delta).reshape(x.shape[0], -1).to(x.dtype)


class Stage(Protocol):
    wq_a: NativeLinear
    wq_b: NativeLinear
    wkv: NativeLinear
    wo_b: NativeLinear
    sh_w1: NativeLinear
    sh_w2: NativeLinear
    sh_w3: NativeLinear
    wo_a_n: object
    adapter_wo_a: Callable[[Tensor], Tensor] | None
    shared_pair: object
    attn_norm: Tensor
    ffn_norm: Tensor
    q_norm: Tensor
    kv_norm: Tensor


class Markov(Protocol):
    w1: Tensor
    w2: Tensor
    cache: Tensor
    slot_of_token: Tensor


class Rank(Protocol):
    g0: int
    g1: int
    h0: int
    h1: int
    i0: int
    i1: int
    v0: int
    v1: int
    dev: torch.device
    stages: list[Stage]
    main_norm: Tensor
    norm: Tensor
    conf_w: Tensor
    markov: Markov
    mrows: object
    ring_kv: Tensor
    ring_pos: Tensor


class Worker(Protocol):
    cfg: ArmConfig
    ALL: tuple[int, ...]
    ranks: list[Rank]
    graphs: dict[object, object]
    margin_graphs: dict[object, object]
    insert_graphs: dict[object, object]
    adapter_guard: DependencyGuard


CacheFactory = Callable[[Tensor, Tensor, list[int], int], object]


def install(model: Worker, values: Mapping[str, Tensor], config: Config, rank: int,
            cache_factory: CacheFactory, hot_tokens: list[int]) -> None:
    if (model.cfg.dense != "bf16" or model.cfg.head != "bf16" or model.cfg.combine != "official"
            or not model.cfg.wo_a_qdq or model.cfg.kv_qdq != "official" or model.cfg.dense_kernel != "n1"):
        raise ValueError("adapter evaluation requires native BF16 dense/head and official QDQ/combine")
    if model.graphs or model.margin_graphs or model.insert_graphs or hasattr(model, "adapter_guard"):
        raise ValueError("install a single immutable adapter before graph capture; recreate worker")
    expected: dict[str, tuple[int, ...]] = {}
    projections = {"attn.wq_a": (config.qr, config.dim), "attn.wq_b": (config.heads*config.hd, config.qr),
                   "attn.wkv": (config.hd, config.dim), "attn.wo_b": (config.dim, config.groups*config.outr),
                   "ffn.shared_experts.w1": (config.inter, config.dim), "ffn.shared_experts.w3": (config.inter, config.dim),
                   "ffn.shared_experts.w2": (config.dim, config.inter)}
    def tensor_key(name: str) -> str:
        return name.replace(".", "__")
    expected["t.mtp__0__main_norm__weight"] = (config.dim,)
    for name, shape in {"norm.weight": (config.dim,), "markov_head.embed.weight": (config.vocab, config.markov),
                        "markov_head.head.weight": (config.vocab, config.markov),
                        "confidence_head.proj.weight": (1, config.dim+config.markov)}.items():
        expected["t."+tensor_key(f"mtp.{config.stages-1}."+name)] = shape
    for i in range(config.stages):
        for name, (out, width) in projections.items():
            stem = f"stages.{i}.dense.{tensor_key(name)}."
            expected[stem+"a"], expected[stem+"b"] = (rank, width), (out, rank)
        for group in range(config.groups):
            stem = f"stages.{i}.wo_a.{group}."
            expected[stem+"a"] = (rank, config.heads//config.groups*config.hd)
            expected[stem+"b"] = (config.outr, rank)
        for name, width in (("attn_norm", config.dim), ("ffn_norm", config.dim), ("attn.q_norm", config.qr), ("attn.kv_norm", config.hd)):
            expected[f"stages.{i}.t."+tensor_key(f"mtp.{i}.{name}.weight")] = (width,)
    if set(expected) != set(values) or rank not in (32, 64):
        raise ValueError("adapter trainable coverage/rank mismatch; no ignored tensors")
    for name, shape in expected.items():
        if values[name].shape != shape or not bool(torch.isfinite(values[name]).all()):
            raise ValueError(f"adapter shape/nonfinite mismatch: {name}")
    # Build all replacements before mutation. LoRA never changes any base bytes.
    changes: list[tuple[object, str, object]] = []
    dependencies: dict[str, Tensor] = {}
    for r in model.ALL:
        rk = model.ranks[r]
        def on_device(name: str) -> Tensor:
            tensor = values[name].detach().to(device=rk.dev, dtype=torch.float32).clone().contiguous()
            dependencies[f"{r}/{name}"] = tensor
            return tensor
        def replace(owner: object, attribute: str, name: str, selection: slice | None = None, flatten: bool = False) -> None:
            t = on_device(name)
            if selection is not None:
                t = t[selection].contiguous()
            if flatten:
                t = t.flatten()
            changes.append((owner, attribute, t))
        replace(rk, "main_norm", "t.mtp__0__main_norm__weight")
        last = "t."+tensor_key(f"mtp.{config.stages-1}.")
        replace(rk, "norm", last+"norm__weight")
        replace(rk, "conf_w", last+"confidence_head__proj__weight", flatten=True)
        w1 = on_device(last+"markov_head__embed__weight")
        w2 = on_device(last+"markov_head__head__weight")[rk.v0:rk.v1].contiguous()
        # Rebuild exact adapted Markov hot rows; discard all original BF16 rows.
        changes.extend(((rk.markov, "w1", w1), (rk.markov, "w2", w2),
                        (rk, "mrows", cache_factory(w1, w2, hot_tokens, model.cfg.markov_rows))))
        for i, stage in enumerate(rk.stages):
            for op, attribute in (("attn.wq_a", "wq_a"), ("attn.wq_b", "wq_b"), ("attn.wkv", "wkv"),
                                  ("attn.wo_b", "wo_b"), ("ffn.shared_experts.w1", "sh_w1"),
                                  ("ffn.shared_experts.w2", "sh_w2"), ("ffn.shared_experts.w3", "sh_w3")):
                stem = f"stages.{i}.dense.{tensor_key(op)}."
                a, b = on_device(stem+"a"), on_device(stem+"b")
                if op == "attn.wq_b":
                    b = b[rk.h0*config.hd:rk.h1*config.hd]
                elif op == "attn.wo_b":
                    a = a[:, rk.g0*config.outr:rk.g1*config.outr]
                elif op in ("ffn.shared_experts.w1", "ffn.shared_experts.w3"):
                    b = b[rk.i0:rk.i1]
                elif op == "ffn.shared_experts.w2":
                    a = a[:, rk.i0:rk.i1]
                changes.append((stage, attribute, DeltaLinear(cast(NativeLinear, getattr(stage, attribute)), a, b)))
            # Native BF16 wo_a is grouped (not a dense matrix over all groups).
            weight = cast(Tensor, getattr(stage.wo_a_n, "w"))
            factors = [(on_device(f"stages.{i}.wo_a.{g}.a"), on_device(f"stages.{i}.wo_a.{g}.b")) for g in range(rk.g0, rk.g1)]
            changes.append((stage, "adapter_wo_a", GroupedDelta(weight.reshape(rk.g1-rk.g0, config.outr, -1), factors)))
            changes.append((stage, "shared_pair", None))  # prevent a fused base-only bypass
            for op, attribute in (("attn_norm", "attn_norm"), ("ffn_norm", "ffn_norm"), ("attn.q_norm", "q_norm"), ("attn.kv_norm", "kv_norm")):
                replace(stage, attribute, f"stages.{i}.t."+tensor_key(f"mtp.{i}.{op}.weight"))
    with torch.no_grad():
        for owner, attribute, value in changes:
            setattr(owner, attribute, value)
    # Resolve installed bindings on each pre-replay check, detecting both
    # in-place updates and replacement of a tensor/factor object.
    def current_dependencies() -> dict[str, Tensor]:
        current = dict(dependencies)
        for i, (owner, attribute, _) in enumerate(changes):
            value = getattr(owner, attribute)
            if isinstance(value, Tensor):
                current[f"installed/{i}"] = value
            elif isinstance(value, DeltaLinear):
                current[f"delta/{i}/a"], current[f"delta/{i}/b"] = value.a, value.b
            elif isinstance(value, GroupedDelta):
                for g, (a, b) in enumerate(value.factors):
                    current[f"group/{i}/{g}/a"], current[f"group/{i}/{g}/b"] = a, b
        return current
    def invalidate() -> None:
        model.graphs.clear()
        model.margin_graphs.clear()
        model.insert_graphs.clear()
        for r in model.ALL:
            rk = model.ranks[r]
            rk.ring_kv.zero_()
            rk.ring_pos.fill_(-1)
            rk.markov.cache.zero_()
            rk.markov.slot_of_token.fill_(-1)
            rk.mrows = None
    # All feature windows are invalidated before the first adapted insertion.
    for r in model.ALL:
        model.ranks[r].ring_kv.zero_()
        model.ranks[r].ring_pos.fill_(-1)
        model.ranks[r].markov.cache.zero_()
        model.ranks[r].markov.slot_of_token.fill_(-1)
    model.adapter_guard = DependencyGuard(current_dependencies(), invalidate, current_dependencies)


def adapted_identity(base: Mapping[str, object], checkpoint_sha256: str, config: Config) -> dict[str, object]:
    from .pilot import sources
    from .data import sha256
    runtime_sources = sources()
    proxy_root = Path(__file__).resolve().parents[2]/"spark/remote_dspark"
    runtime_sources.update({"process_proxy/"+p.name: sha256(p) for p in sorted(proxy_root.iterdir())
                            if p.is_file() and p.suffix in (".py", ".c")})
    def identity(role: str) -> str:
        return hashlib.sha256(json.dumps({"schema": "dsv41.d2.native-adapter.v1", "role": role, "base": dict(base),
            "adapter_sha256": checkpoint_sha256, "config": asdict(config), "runtime_sources": runtime_sources,
            "semantics": "native-base-plus-fp32-delta-before-rounding;immutable;official-QDQ"}, sort_keys=True).encode()).hexdigest()
    return {"schema": "dsv41.d2.adapted-native-profile.v1", "source_receipt": dict(base),
            "native_base_profile": "native-hybrid-head-hot", "arm": "d2-adapted-native", "profile_label": "adapted; not unchanged source-faithful",
            "adapter_sha256": checkpoint_sha256, "draft_head": "official BF16 head with FP32 logits", "drafter_identity": identity("drafter"), "markov_identity": identity("markov")}
