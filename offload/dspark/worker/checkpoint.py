"""Identity-checked loader for the DSpark drafter tensors that lane S2 copied to x86host.

Source of truth: ``TENSORS.json`` written by S2 next to the extracted safetensors files. Every entry records the
tensor's name, dtype, shape, the destination file and byte range, and the sha256 of the *source* byte range on the
Spark (``/srv/dsv41/legal-v2a``, hardlinked from HeNryous EXL3 2.75 bpw @ 86f85253). Loading re-hashes each
destination byte range and refuses the checkpoint on any difference, on a missing or unexpected tensor, or on a
header that disagrees with the manifest. Nothing is converted here: tensors come back as zero-copy CPU views of the
mapped files, in the checkpoint's own dtypes.

The tensor set is derived from config.json (three ``mtp.{0,1,2}`` stages, 128 drafter experts, the target's embedding
and EXL3 head), not from the manifest, so a manifest that lost or gained tensors is caught too.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch

DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I16": torch.int16,
    "I32": torch.int32,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
}

EXL3_PARTS = ("trellis", "suh", "svh", "mul1")


class IdentityError(RuntimeError):
    """The files on disk are not the tensors the manifest promises."""


@dataclass(frozen=True)
class DrafterShape:
    """The numbers of config.json the worker depends on (checked, never assumed)."""
    hidden: int
    vocab: int
    n_heads: int
    head_dim: int
    rope_dim: int
    q_lora: int
    o_lora: int
    o_groups: int
    inter: int
    n_experts: int
    top_k: int
    n_stages: int
    n_target_layers: int
    window: int
    block: int
    noise_token: int
    markov_rank: int
    hc_mult: int
    hc_iters: int
    hc_eps: float
    rms_eps: float
    swiglu_limit: float
    route_scale: float
    rope_theta: float
    taps: tuple[int, ...]

    @classmethod
    def from_config(cls, cfg: dict) -> "DrafterShape":
        t = cfg["text_config"]
        n_layers = int(t["num_hidden_layers"])
        ratios = t["compress_ratios"]
        n_stages = int(t["num_nextn_predict_layers"])
        if [int(r) for r in ratios[n_layers:n_layers + n_stages]] != [0] * n_stages:
            raise IdentityError(f"drafter stages are not pure sliding-window layers: {ratios[n_layers:]}")
        if t.get("scoring_func") != "sqrtsoftplus" or t.get("topk_method") != "noaux_tc" or not t.get("norm_topk_prob"):
            raise IdentityError("routing is not sqrtsoftplus / noaux_tc / normalized top-k")
        if t.get("hidden_act") != "silu":
            raise IdentityError(f"hidden_act {t.get('hidden_act')!r}")
        return cls(
            hidden=int(t["hidden_size"]), vocab=int(t["vocab_size"]), n_heads=int(t["num_attention_heads"]),
            head_dim=int(t["head_dim"]), rope_dim=int(t["qk_rope_head_dim"]), q_lora=int(t["q_lora_rank"]),
            o_lora=int(t["o_lora_rank"]), o_groups=int(t["o_groups"]), inter=int(t["moe_intermediate_size"]),
            n_experts=int(t["dspark_n_routed_experts"]), top_k=int(t["dspark_num_experts_per_tok"]),
            n_stages=n_stages, n_target_layers=n_layers, window=int(t["sliding_window"]),
            block=int(t["dspark_block_size"]), noise_token=int(t["dspark_noise_token_id"]),
            markov_rank=int(t["dspark_markov_rank"]), hc_mult=int(t["hc_mult"]), hc_iters=int(t["hc_sinkhorn_iters"]),
            hc_eps=float(t["hc_eps"]), rms_eps=float(t["rms_norm_eps"]), swiglu_limit=float(t["swiglu_limit"]),
            route_scale=float(t["routed_scaling_factor"]), rope_theta=float(t["rope_theta"]),
            taps=tuple(int(x) for x in t["dspark_target_layer_ids"]),
        )


def required_names(s: DrafterShape) -> list[str]:
    """Every tensor the worker reads (plus ffn.gate.bias_vl, checked though text drafts never use it)."""
    names = ["embed.weight"] + [f"head.{p}" for p in EXL3_PARTS]
    for i in range(s.n_stages):
        pre = f"mtp.{i}."
        for lin in ("attn.wq_a", "attn.wkv", "attn.wq_b", "attn.wo_b",
                    "ffn.shared_experts.w1", "ffn.shared_experts.w2", "ffn.shared_experts.w3"):
            names += [f"{pre}{lin}.{p}" for p in EXL3_PARTS]
        names += [f"{pre}attn.wo_a.weight", f"{pre}attn.wo_a.scale", f"{pre}attn.attn_sink",
                  f"{pre}attn.q_norm.weight", f"{pre}attn.kv_norm.weight", f"{pre}attn_norm.weight",
                  f"{pre}ffn_norm.weight", f"{pre}ffn.gate.weight", f"{pre}ffn.gate.bias", f"{pre}ffn.gate.bias_vl"]
        names += [f"{pre}hc_{s_}_{k}" for s_ in ("attn", "ffn") for k in ("fn", "base", "scale")]
        for e in range(s.n_experts):
            for w in ("w1", "w2", "w3"):
                names += [f"{pre}ffn.experts.{e}.{w}.{p}" for p in EXL3_PARTS]
    names += [f"mtp.0.main_proj.{p}" for p in EXL3_PARTS] + ["mtp.0.main_norm.weight"]
    last = f"mtp.{s.n_stages - 1}."
    names += [f"{last}norm.weight", f"{last}markov_head.embed.weight", f"{last}markov_head.head.weight",
              f"{last}confidence_head.proj.weight"]
    return names


def _read_header(path: str) -> tuple[int, dict]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 64 << 20:
            raise IdentityError(f"{path}: implausible safetensors header length {n}")
        return 8 + n, json.loads(f.read(n))


class Checkpoint:
    """The verified drafter tensors of one S2 delivery directory. ``get(name)`` returns a CPU view (no copy)."""

    def __init__(self, root: str, verify: bool = True, threads: int = 16):
        self.root = root
        with open(os.path.join(root, "config.json")) as f:
            self.config = json.load(f)
        with open(os.path.join(root, "TENSORS.json")) as f:
            self.manifest = json.load(f)
        with open(os.path.join(root, "exl3_moe_plan.json")) as f:
            self.plan_json = json.load(f)
        self.shape = DrafterShape.from_config(self.config)
        self._check_quant_config()
        entries = {e["name"]: e for e in self.manifest["tensors"]}
        need = required_names(self.shape)
        missing = [n for n in need if n not in entries]
        if missing:
            raise IdentityError(f"{len(missing)} drafter tensors missing from TENSORS.json, e.g. {missing[:5]}")
        # norm.weight (the target's final norm) is delivered too; the drafter uses mtp.2.norm, so it is ignored
        extra = sorted(set(entries) - set(need) - {"norm.weight"})
        if extra:
            raise IdentityError(f"unexpected tensors in TENSORS.json: {extra[:5]} ({len(extra)})")
        self.entries = {n: entries[n] for n in need}
        self._maps: dict[str, mmap.mmap] = {}
        self._headers: dict[str, tuple[int, dict]] = {}
        from .identity import IdentityMismatch, resolve_inside
        for e in self.entries.values():
            fn = e["destination_file"]
            if fn not in self._maps:
                try:                                  # never open a file outside the delivery (S5R fix 1)
                    path = resolve_inside(root, fn)
                except IdentityMismatch as err:
                    raise IdentityError(f"{e['name']}: {err}") from err
                self._headers[fn] = _read_header(path)
                with open(path, "rb") as f:
                    self._maps[fn] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        for name, e in self.entries.items():
            self._check_header(name, e)
        self.verified = False
        if verify:
            self.verify(threads)

    def _check_quant_config(self) -> None:
        q = self.config.get("quantization_config", {})
        x = q.get("exl3_moe", {})
        if q.get("quant_method") != "deepseek_v41_exl3" or x.get("exllamav3_version") != "1.6.0":
            raise IdentityError(f"quantization_config is not deepseek_v41_exl3 / exllamav3 1.6.0: {q.get('quant_method')}")
        if x.get("head") != "exl3":
            raise IdentityError(f"head is {x.get('head')!r}, the worker implements the EXL3 head only")
        if self.manifest.get("codebook") != "mul1" or self.plan_json.get("codebook") != "mul1":
            raise IdentityError("codebook is not mul1")
        self.head_k = int(x["head_k"])
        self.drafter_k = {k: int(v) for k, v in x["dense"]["drafter"].items()}

    def _check_header(self, name: str, e: dict) -> None:
        fn = e["destination_file"]
        hlen, hdr = self._headers[fn]
        h = hdr.get(name)
        if h is None:
            raise IdentityError(f"{name}: not in {fn}'s header")
        if h["dtype"] != e["dtype"] or list(h["shape"]) != list(e["shape"]):
            raise IdentityError(f"{name}: header {h['dtype']} {h['shape']} vs manifest {e['dtype']} {e['shape']}")
        a, b = h["data_offsets"]
        if [hlen + a, hlen + b] != list(e["destination_byte_range"]):
            raise IdentityError(f"{name}: header range {[hlen + a, hlen + b]} vs manifest {e['destination_byte_range']}")
        if e["dtype"] not in DTYPES:
            raise IdentityError(f"{name}: dtype {e['dtype']} not supported")
        numel = 1
        for d in e["shape"]:
            numel *= int(d)
        if numel * DTYPES[e["dtype"]].itemsize != b - a:
            raise IdentityError(f"{name}: {b - a} bytes for {e['shape']} {e['dtype']}")
        sa, sb = e["source_byte_range"]
        if sb - sa != b - a:
            raise IdentityError(f"{name}: source range {sb - sa} B vs destination {b - a} B")

    def _bytes(self, name: str) -> memoryview:
        e = self.entries[name]
        a, b = e["destination_byte_range"]
        return memoryview(self._maps[e["destination_file"]])[a:b]

    def verify(self, threads: int = 16) -> dict:
        """sha256 of every destination byte range == the manifest's source-range sha256. Raises on any mismatch."""
        def one(name: str) -> tuple[str, bool]:
            return name, hashlib.sha256(self._bytes(name)).hexdigest() == self.entries[name]["source_byte_range_sha256"]
        # big tensors first so the pool stays busy
        order = sorted(self.entries, key=lambda n: -int(self.entries[n]["resident_packed_bytes"]))
        with ThreadPoolExecutor(threads) as ex:
            bad = [n for n, ok in ex.map(one, order) if not ok]
        if bad:
            raise IdentityError(f"sha256 mismatch for {len(bad)} tensors, e.g. {bad[:5]}")
        self.verified = True
        return {"tensors": len(order), "bytes": sum(int(self.entries[n]["resident_packed_bytes"]) for n in order)}

    def get(self, name: str) -> torch.Tensor:
        """Zero-copy CPU view of the tensor bytes in the mapped file (read-only memory; copy before writing)."""
        if not self.verified:
            raise IdentityError("checkpoint not verified")
        e = self.entries[name]
        buf = self._bytes(name)
        dt = DTYPES[e["dtype"]]
        if len(buf) == 0:
            return torch.empty(e["shape"], dtype=dt)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # torch warns that the mmap is not writable; nothing writes it
            t = torch.frombuffer(buf, dtype=torch.uint8)
        return t.view(dt).view(*e["shape"]) if e["shape"] else t.view(dt).reshape(())

    def exl3(self, prefix: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """(trellis int16 [in/16, out/16, 16K], suh fp16 [in], svh fp16 [out], K) of an EXL3 linear; mul1 checked."""
        trellis, suh, svh, mul1 = (self.get(f"{prefix}.{p}") for p in EXL3_PARTS)
        words = trellis.shape[-1]
        if trellis.dim() != 3 or words % 16:
            raise IdentityError(f"{prefix}.trellis shape {tuple(trellis.shape)}")
        if int(mul1.reshape(-1)[0].item()) & 0xFFFFFFFF != 0x83DCD12D:
            raise IdentityError(f"{prefix}.mul1 marker {int(mul1.reshape(-1)[0].item()) & 0xFFFFFFFF:#x} is not mul1")
        if suh.shape[0] != trellis.shape[0] * 16 or svh.shape[0] != trellis.shape[1] * 16:
            raise IdentityError(f"{prefix}: suh/svh {tuple(suh.shape)}/{tuple(svh.shape)} vs trellis {tuple(trellis.shape)}")
        return trellis, suh, svh, words // 16

    def close(self) -> None:
        for m in self._maps.values():
            m.close()
        self._maps.clear()
