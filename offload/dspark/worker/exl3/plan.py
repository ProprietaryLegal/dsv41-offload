# SPDX-License-Identifier: Apache-2.0
"""Per-expert EXL3 bit plan, bank layout and TP slicing math (pure Python, no torch).

Everything here is importable and testable without torch, vLLM or exllamav3, so the
layout logic can be checked on any machine before a GPU run.

Facts this module encodes (exllamav3 v1.6.0, turboderp-org/exllamav3 @ 151539c7, MIT):
  * One EXL3 linear = trellis int16 [in/16, out/16, 16*K] + suh fp16 [in] + svh fp16 [out]
    + a scalar codebook marker tensor ("mul1" or "mcg", int32)
    (modules/quant/exl3.py:16-116, quantize.py:1825-1856, multilinear.py:92).
  * Weights are stored row-major as (in_features, out_features), the transpose of torch's
    nn.Linear (quantize.py:1636-1640).
  * Hadamard rotations are block-diagonal with 128-wide blocks on both sides
    (quantize.py:15 had_k = had_n = 128; exl3.py:239-249), every 16x16 trellis tile is coded
    on its own, so slicing at multiples of 128 on either axis is exact. exllamav3's own MoE
    tensor-parallel split does exactly this (block_sparse_mlp.py:1517-1554, 1640-1648;
    exl3.py:296-342).
  * Bitrate K comes from the tile width: trellis.shape[-1] == 16 * K (exl3.py:66-68);
    half-integer K (1.5, 2.5, 3.5) needs the mul1 codebook (exl3.py:81).
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

HAD_BLOCK = 128          # exllamav3 had_k = had_n = 128
TILE = 16                # trellis tile edge
ALIGN_WORDS = 128        # each trellis starts on a 256-byte boundary inside a bank
FP16_BYTES = 2
WORD_BYTES = 2           # trellis element = int16

# Codebook marker values exllamav3 writes (quantize.py:18-19, stored as int32 views)
CODEBOOK_MCG = 0xCBAC1FED
CODEBOOK_MUL1 = 0x83DCD12D

INTEGER_K = (1, 2, 3, 4, 5, 6, 7, 8)
HALF_K = (1.5, 2.5, 3.5)
# One character per expert in the compact plan string: '1'..'8' integer K, 'a'/'b'/'c' = 1.5/2.5/3.5
_HALF_CODES = {"a": 1.5, "b": 2.5, "c": 3.5}
_HALF_CODES_REV = {v: k for k, v in _HALF_CODES.items()}

# Checkpoint tensor names, same keys exllamav3 uses for DeepSeek-V4 experts
# (architecture/deepseek_v4.py:144, 177, 182-184): layers.{L}.ffn.experts.{E}.w{1,2,3}.<suffix>
EXPERT_TENSOR_SUFFIXES = ("trellis", "suh", "svh", "mul1", "mcg")
PROJ_GATE, PROJ_DOWN, PROJ_UP = "w1", "w2", "w3"


class PlanError(ValueError):
    pass


def parse_k(v: Any) -> float:
    """Normalise one bitrate value (int, float, or a one-character code)."""
    if isinstance(v, str):
        if v in _HALF_CODES:
            return _HALF_CODES[v]
        if v.isdigit():
            v = int(v)
        else:
            raise PlanError(f"bad K code {v!r}")
    k = float(v)
    if k.is_integer() and int(k) in INTEGER_K:
        return float(int(k))
    if k in HALF_K:
        return k
    raise PlanError(f"unsupported EXL3 bitrate {v!r} (allowed: {INTEGER_K} and {HALF_K})")


def k_code(k: float) -> str:
    k = parse_k(k)
    return str(int(k)) if k.is_integer() else _HALF_CODES_REV[k]


def trellis_words(rows: int, cols: int, k: float) -> int:
    """int16 words of an EXL3 trellis for a (rows x cols) matrix at bitrate k."""
    if rows % TILE or cols % TILE:
        raise PlanError(f"matrix {rows}x{cols} is not a multiple of {TILE}")
    per_tile = 16 * parse_k(k)
    if not float(per_tile).is_integer():
        raise PlanError(f"bitrate {k}: tile is not a whole number of int16 words")
    return (rows // TILE) * (cols // TILE) * int(per_tile)


def trellis_shape(rows: int, cols: int, k: float) -> tuple[int, int, int]:
    return rows // TILE, cols // TILE, int(16 * parse_k(k))


def _align(n: int, a: int = ALIGN_WORDS) -> int:
    return (n + a - 1) // a * a


@dataclass(frozen=True)
class ExpertBits:
    k_gu: float   # gate (w1) and up (w3) share one rate (exllamav3 requires it: exl3_moe_coop.cu:194)
    k_d: float    # down (w2)

    @property
    def avg(self) -> float:
        return (2 * self.k_gu + self.k_d) / 3


@dataclass(frozen=True)
class KGroup:
    """Experts of one layer that share (k_gu, k_d): one kernel launch per group."""
    k_gu: float
    k_d: float
    lo: int                     # physical slot range [lo, hi)
    hi: int
    experts: tuple[int, ...]    # logical expert ids, in physical order

    @property
    def size(self) -> int:
        return self.hi - self.lo


@dataclass
class LayerPlan:
    layer: int
    bits: tuple[ExpertBits, ...]            # indexed by logical expert id
    groups: tuple[KGroup, ...] = ()
    perm: tuple[int, ...] = ()              # logical id -> physical slot
    inv_perm: tuple[int, ...] = ()          # physical slot -> logical id

    @classmethod
    def build(cls, layer: int, bits: Sequence[ExpertBits]) -> "LayerPlan":
        bits = tuple(bits)
        order = sorted(range(len(bits)), key=lambda e: (bits[e].k_gu, bits[e].k_d, e))
        perm = [0] * len(bits)
        for p, e in enumerate(order):
            perm[e] = p
        groups: list[KGroup] = []
        start = 0
        for p in range(1, len(order) + 1):
            if p == len(order) or (bits[order[p]].k_gu, bits[order[p]].k_d) != (
                bits[order[start]].k_gu, bits[order[start]].k_d
            ):
                b = bits[order[start]]
                groups.append(KGroup(b.k_gu, b.k_d, start, p, tuple(order[start:p])))
                start = p
        return cls(layer, bits, tuple(groups), tuple(perm), tuple(order))

    @property
    def num_experts(self) -> int:
        return len(self.bits)

    def group_of_slot(self, p: int) -> KGroup:
        for g in self.groups:
            if g.lo <= p < g.hi:
                return g
        raise IndexError(p)


@dataclass
class BankLayout:
    """Flat per-layer, per-rank banks (int16 words), indexed by physical slot.

    w13 bank: [w1 shard | w3 shard] per slot; w2 bank: [w2 shard] per slot. Pointer tables for
    the kernels are base + offset, so experts of different K can share one allocation.
    """
    hidden: int
    inter_local: int
    words_w1: tuple[int, ...]     # per slot; w3 has the same size
    off_w1: tuple[int, ...]
    off_w3: tuple[int, ...]
    total13: int
    words_w2: tuple[int, ...]
    off_w2: tuple[int, ...]
    total2: int

    @classmethod
    def build(cls, lp: LayerPlan, hidden: int, inter_local: int) -> "BankLayout":
        words_w1, off_w1, off_w3, words_w2, off_w2 = [], [], [], [], []
        c13 = c2 = 0
        for p in range(lp.num_experts):
            b = lp.bits[lp.inv_perm[p]]
            w = trellis_words(hidden, inter_local, b.k_gu)
            words_w1.append(w)
            off_w1.append(c13)
            c13 = _align(c13 + w)
            off_w3.append(c13)
            c13 = _align(c13 + w)
            w2 = trellis_words(inter_local, hidden, b.k_d)
            words_w2.append(w2)
            off_w2.append(c2)
            c2 = _align(c2 + w2)
        return cls(hidden, inter_local, tuple(words_w1), tuple(off_w1), tuple(off_w3), c13,
                   tuple(words_w2), tuple(off_w2), c2)

    @property
    def trellis_bytes(self) -> int:
        return (self.total13 + self.total2) * WORD_BYTES

    def sign_bytes(self, num_experts: int) -> int:
        # w13: suh [2, H] + svh [2, I_loc]; w2: suh [I_loc] + svh [H]; fp16
        return num_experts * FP16_BYTES * (3 * self.hidden + 3 * self.inter_local)


def tp_shard(inter: int, tp_size: int, tp_rank: int) -> tuple[int, int]:
    """Intermediate-channel range [i0, i1) of one TP rank (the vLLM MoE TP split)."""
    if inter % tp_size:
        raise PlanError(f"intermediate {inter} not divisible by tp {tp_size}")
    loc = inter // tp_size
    if loc % HAD_BLOCK:
        raise PlanError(
            f"intermediate per rank {loc} is not a multiple of the {HAD_BLOCK}-wide Hadamard block: "
            f"an EXL3 expert cannot be channel-split at tp={tp_size} (use tp in 1, 2, 3, 6, 9, 18 for 2304, "
            f"or expert parallel)")
    if not 0 <= tp_rank < tp_size:
        raise PlanError(f"bad tp rank {tp_rank}")
    return tp_rank * loc, (tp_rank + 1) * loc


@dataclass(frozen=True)
class ShardSlices:
    """Index ranges into the full exllamav3 tensors that one TP rank keeps."""
    trellis_dim: int              # axis of the trellis that is cut (tile units)
    trellis_range: tuple[int, int]
    suh_range: tuple[int, int] | None    # None = keep all
    svh_range: tuple[int, int] | None


def shard_slices(proj: str, inter: int, tp_size: int, tp_rank: int) -> ShardSlices:
    """w1/w3 (in = hidden, out = inter) are cut along out; w2 (in = inter, out = hidden) along in."""
    i0, i1 = tp_shard(inter, tp_size, tp_rank)
    t = (i0 // TILE, i1 // TILE)
    if proj in (PROJ_GATE, PROJ_UP):
        return ShardSlices(1, t, None, (i0, i1))
    if proj == PROJ_DOWN:
        return ShardSlices(0, t, (i0, i1), None)
    raise PlanError(f"bad projection {proj!r}")


def _parse_layer_bits(entry: Any, num_experts: int) -> tuple[ExpertBits, ...]:
    if isinstance(entry, (str, list, tuple)):
        k_gu = entry
        k_d = None
    elif isinstance(entry, dict):
        k_gu = entry.get("k", entry.get("k_gu"))
        k_d = entry.get("k_down", entry.get("k_d"))
    else:
        raise PlanError(f"bad layer entry type {type(entry).__name__}")

    def expand(v: Any) -> list[float]:
        if isinstance(v, str):
            vals = [parse_k(c) for c in v]
        else:
            vals = [parse_k(x) for x in v]
        if len(vals) != num_experts:
            raise PlanError(f"plan has {len(vals)} experts for a layer, expected {num_experts}")
        return vals

    gu = expand(k_gu)
    d = expand(k_d) if k_d is not None else gu
    return tuple(ExpertBits(a, b) for a, b in zip(gu, d))


@dataclass
class Exl3MoEPlan:
    num_layers: int
    num_experts: int
    hidden_size: int
    moe_intermediate_size: int
    codebook: str = "mul1"
    layers: dict[int, LayerPlan] = field(default_factory=dict)
    base_model: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    # DSpark drafter MoE layers (absolute index >= num_layers, e.g. 40-42 = mtp.0-2), with their own expert count
    # (dspark_n_routed_experts): kept apart so the backbone's bpw / bytes / histogram summaries stay backbone-only
    drafter_num_experts: int = 0
    drafter_layers: dict[int, LayerPlan] = field(default_factory=dict)

    # ---- construction -------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Exl3MoEPlan":
        if d.get("format", "exl3_moe_plan") != "exl3_moe_plan":
            raise PlanError(f"not an exl3_moe_plan: format={d.get('format')!r}")
        if int(d.get("version", 1)) != 1:
            raise PlanError(f"unsupported plan version {d.get('version')}")
        codebook = d.get("codebook", "mul1")
        if codebook not in ("mul1", "mcg"):
            raise PlanError(f"codebook must be mul1 or mcg (the fused MoE kernels take no other), got {codebook!r}")
        plan = cls(
            num_layers=int(d["num_layers"]),
            num_experts=int(d["num_experts"]),
            hidden_size=int(d["hidden_size"]),
            moe_intermediate_size=int(d["moe_intermediate_size"]),
            codebook=codebook,
            base_model=d.get("base_model"),
            meta={k: v for k, v in d.items() if k in ("stats", "calibration", "notes")},
        )
        for key, entry in (d.get("layers") or {}).items():
            L = int(key)
            if not 0 <= L < plan.num_layers:
                raise PlanError(f"layer {L} outside 0..{plan.num_layers - 1}")
            bits = _parse_layer_bits(entry, plan.num_experts)
            if codebook != "mul1" and any(not (b.k_gu.is_integer() and b.k_d.is_integer()) for b in bits):
                raise PlanError(f"layer {L}: half-integer bitrates need the mul1 codebook")
            plan.layers[L] = LayerPlan.build(L, bits)
        dr = d.get("drafter") or {}
        if dr:
            plan.drafter_num_experts = int(dr["num_experts"])
            for key, entry in (dr.get("layers") or {}).items():
                L = int(key)
                if L < plan.num_layers:
                    raise PlanError(f"drafter layer {L} is a backbone layer (0..{plan.num_layers - 1})")
                bits = _parse_layer_bits(entry, plan.drafter_num_experts)
                if codebook != "mul1" and any(not (b.k_gu.is_integer() and b.k_d.is_integer()) for b in bits):
                    raise PlanError(f"drafter layer {L}: half-integer bitrates need the mul1 codebook")
                plan.drafter_layers[L] = LayerPlan.build(L, bits)
        return plan

    @classmethod
    def from_json_file(cls, path: str | os.PathLike) -> "Exl3MoEPlan":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_dict(self, compact: bool = True) -> dict[str, Any]:
        def entries(layer_plans: dict[int, LayerPlan]) -> dict[str, Any]:
            res: dict[str, Any] = {}
            for L, lp in sorted(layer_plans.items()):
                gu = [b.k_gu for b in lp.bits]
                dn = [b.k_d for b in lp.bits]
                if compact:
                    entry: dict[str, Any] = {"k": "".join(k_code(k) for k in gu)}
                    if dn != gu:
                        entry["k_down"] = "".join(k_code(k) for k in dn)
                else:
                    entry = {"k": gu}
                    if dn != gu:
                        entry["k_down"] = dn
                res[str(L)] = entry
            return res

        layers = entries(self.layers)
        out = {
            "format": "exl3_moe_plan",
            "version": 1,
            "base_model": self.base_model,
            "codebook": self.codebook,
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "hidden_size": self.hidden_size,
            "moe_intermediate_size": self.moe_intermediate_size,
            "layers": layers,
        }
        if self.drafter_layers:
            out["drafter"] = {"num_experts": self.drafter_num_experts, "layers": entries(self.drafter_layers)}
        out.update(self.meta)
        return out

    # ---- queries ------------------------------------------------------------------------

    def has_layer(self, layer: int) -> bool:
        return layer in self.layers or layer in self.drafter_layers

    def layer(self, layer: int) -> LayerPlan:
        return self.layers[layer] if layer in self.layers else self.drafter_layers[layer]

    def bpw(self, layers: Iterable[int] | None = None) -> float:
        """Parameter-weighted trellis bits per weight (static memory)."""
        ls = list(self.layers) if layers is None else list(layers)
        tot = sum(b.avg for L in ls for b in self.layers[L].bits)
        n = sum(len(self.layers[L].bits) for L in ls)
        return tot / n if n else 0.0

    def usage_weighted_bpw(self, usage: dict[int, Sequence[float]]) -> float:
        """Bits per weight READ per token at decode: sum_e p(e) * K_e. Hot experts at high K cost
        bandwidth on every token even when the static average looks small."""
        num = den = 0.0
        for L, lp in self.layers.items():
            p = usage.get(L)
            if p is None:
                continue
            for e, b in enumerate(lp.bits):
                num += p[e] * b.avg
                den += p[e]
        return num / den if den else 0.0

    def bank_layout(self, layer: int, tp_size: int) -> BankLayout:
        i0, i1 = tp_shard(self.moe_intermediate_size, tp_size, 0)
        return BankLayout.build(self.layer(layer), self.hidden_size, i1 - i0)

    def bytes_per_rank(self, tp_size: int) -> int:
        total = 0
        for L in self.layers:
            bl = self.bank_layout(L, tp_size)
            total += bl.trellis_bytes + bl.sign_bytes(self.num_experts)
        return total

    def group_histogram(self) -> dict[tuple[float, float], int]:
        h: dict[tuple[float, float], int] = {}
        for lp in self.layers.values():
            for b in lp.bits:
                h[(b.k_gu, b.k_d)] = h.get((b.k_gu, b.k_d), 0) + 1
        return dict(sorted(h.items()))


def synthetic_plan(
    num_layers: int = 40,
    num_experts: int = 384,
    hidden_size: int = 5120,
    moe_intermediate_size: int = 2304,
    counts: dict[float, int] | None = None,
    uniform_k: float | None = None,
    seed: int = 0,
    base_model: str | None = None,
) -> Exl3MoEPlan:
    """A plan for plumbing / dummy-load tests, never for serving quality.

    counts: total experts per K over the whole model (e.g. {2: 7279, 3: 6299, 5: 1782}),
    shuffled across layers and experts with a fixed seed. uniform_k: every expert at one K.
    """
    total = num_layers * num_experts
    if uniform_k is not None:
        ks = [parse_k(uniform_k)] * total
    else:
        if not counts or sum(counts.values()) != total:
            raise PlanError(f"counts must sum to {total}")
        ks = [parse_k(k) for k, n in sorted(counts.items()) for _ in range(n)]
        random.Random(seed).shuffle(ks)
    d = {
        "format": "exl3_moe_plan",
        "version": 1,
        "base_model": base_model,
        "codebook": "mul1",
        "num_layers": num_layers,
        "num_experts": num_experts,
        "hidden_size": hidden_size,
        "moe_intermediate_size": moe_intermediate_size,
        "layers": {
            str(L): {"k": "".join(k_code(k) for k in ks[L * num_experts:(L + 1) * num_experts])}
            for L in range(num_layers)
        },
        "notes": "synthetic plan (tests only)",
    }
    return Exl3MoEPlan.from_dict(d)
