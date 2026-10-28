"""Offline native-arm packaging. Separate sibling hardlink tree, never in place.

Merge Phase 2a LoRA only into attention/shared dense tensors. Routed FP4,
routers, target embedding/head and main_proj are immutable. This prepares a
candidate, not a deployment. Quantization/identity gates still must pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any
import uuid
from collections import Counter

import torch
from torch import Tensor
from safetensors import safe_open
from safetensors.torch import save_file

from ...native.formats import decode_fp8_block
from ...native.official import OfficialShards
from ...spark.tools.identity import identities
from .data import D2Blocks, sha256
from .replay import read_checkpoint


def quantize_fp8(w: Tensor) -> tuple[Tensor, Tensor]:
    """Official 32x32 E4M3FN values with power-of-two UE8M0 block scales."""
    if w.ndim != 2 or any(n % 32 for n in w.shape) or not torch.isfinite(w).all():
        raise ValueError("FP8 weight must be finite, 2D and block32 aligned")
    n, k = w.shape
    blocks = w.float().reshape(n//32, 32, k//32, 32)
    amax = blocks.abs().amax((1, 3)).clamp_min(1e-30)
    exponent = torch.ceil(torch.log2(amax.double()/448)).clamp(-126, 127).int()
    scale = torch.ldexp(torch.ones_like(amax), exponent)
    q = (blocks/scale[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.reshape(n, k).view(torch.uint8), (exponent+127).to(torch.uint8)


def hot_tokens(counts: dict[int, int], size: int = 8192, vocab: int = 129280) -> list[int]:
    if not counts or size < 1 or any(t < 0 or t >= vocab or n < 0 for t, n in counts.items()):
        raise ValueError("invalid Markov input counts")
    return sorted(counts, key=lambda t: (-counts[t], t))[:size]


def training_counts(manifest: Path) -> dict[int, int]:
    """Count actual teacher-forced Markov inputs; only the train split."""
    data = D2Blocks(manifest, "train")
    counts: Counter[int] = Counter()
    for path, row, _ in data.sequences:
        tok_off, length, gen_start = row[:3]
        with safe_open(path, framework="pt", device="cpu") as f:
            previous = f.get_slice("tokens")[tok_off+gen_start:tok_off+length-1].tolist()
        for j, token in enumerate(previous):
            # Overlapping five-target blocks reuse each committed input up to
            # five times. Include that multiplicity, including the first tail.
            counts[int(token)] += min(5, j+1)
    return dict(counts)


def phase2a_updates(source: OfficialShards, state: dict[str, Tensor]) -> dict[str, Tensor]:
    updates: dict[str, Tensor] = {}
    consumed: set[str] = set()
    for name, a in state.items():
        dense = re.fullmatch(r"stages\.(\d+)\.dense\.(.+)\.a", name)
        group = re.fullmatch(r"stages\.(\d+)\.wo_a\.(\d+)\.a", name)
        if dense or group:
            b = state[name[:-1]+"b"]
            if a.ndim != 2 or a.shape[0] not in (32, 64) or b.ndim != 2 or b.shape[1] != a.shape[0]:
                raise ValueError("Phase 2a adapter rank must be 32 or 64")
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError("nonfinite LoRA adapter")
        if dense:
            stage, op = dense.groups()
            op = op.replace("__", ".")
            if op not in ("attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_b", "ffn.shared_experts.w1", "ffn.shared_experts.w2", "ffn.shared_experts.w3"):
                raise ValueError(f"not a Phase 2a dense adapter: {name}")
            p = f"mtp.{stage}.{op}"
            source.verify(p+".weight")
            source.verify(p+".scale")
            base = decode_fp8_block(source.get(p+".weight"), source.get(p+".scale"))
            bn = name[:-1]+"b"
            updates[p+".weight"] = base + state[bn].float() @ a.float()
            consumed.update((name, bn))
        elif group:
            stage, index = group.groups()
            p = f"mtp.{stage}.attn.wo_a"
            if p+".weight" not in updates:
                source.verify(p+".weight")
                source.verify(p+".scale")
                updates[p+".weight"] = decode_fp8_block(source.get(p+".weight"), source.get(p+".scale"))
            bn = name[:-1]+"b"
            rows = state[bn].shape[0]
            start = int(index)*rows
            updates[p+".weight"][start:start+rows] += state[bn].float() @ a.float()
            consumed.update((name, bn))
    for name, value in state.items():
        if name in consumed:
            continue
        plain = re.fullmatch(r"(?:t|stages\.\d+\.t)\.(mtp__\d+__.+)", name)
        if plain is None:
            raise ValueError(f"unrecognised or unapproved Phase 2a parameter: {name}")
        original = plain[1].replace("__", ".")
        if not any(part in original for part in ("_norm.weight", ".norm.weight", ".markov_head.", ".confidence_head.")):
            raise ValueError(f"frozen tensor in update: {original}")
        source.verify(original)
        if source.get(original).shape != value.shape or not torch.isfinite(value).all():
            raise ValueError(f"invalid adapted shape/value: {original}")
        updates[original] = value.detach().cpu().float()
    return updates


def hardlink_tree(source: Path, destination: Path) -> None:
    source, destination = source.resolve(), destination.absolute()
    if destination.exists() or destination.parent.resolve() != source.parent:
        raise ValueError("candidate must be a new sibling beside the source tree")
    destination.mkdir(mode=0o700)
    for p in source.rglob("*"):
        if p.is_symlink():
            raise ValueError(f"source tree symlink refused: {p}")
        rel = p.relative_to(source)
        if p.is_dir():
            (destination/rel).mkdir(mode=0o700, exist_ok=True)
        elif p.is_file():
            os.link(p, destination/rel)


def build(source_root: Path, output: Path, adapter: Path, counts: dict[int, int], native_manifest: Path) -> dict[str, Any]:
    source_root, output = source_root.resolve(), output.absolute()
    if output.exists() or output.parent.resolve() != source_root.parent:
        raise ValueError("candidate must be a new sibling; original is never overwritten")
    sh = OfficialShards(str(source_root), str(native_manifest))
    checkpoint, adapter_digest = read_checkpoint(adapter)
    identity = hashlib.sha256(json.dumps(sh.manifest, sort_keys=True).encode()).hexdigest()
    if checkpoint.get("schema") != "dsv41.d2.train-checkpoint.v1" or checkpoint.get("frozen_identity") != identity:
        raise ValueError("adapter checkpoint does not identify these frozen source tensors")
    changed = phase2a_updates(sh, checkpoint["trainable"])
    for name in list(changed):
        if sh.info(name)["dtype"] == "F8_E4M3":
            changed[name], changed[name.replace(".weight", ".scale")] = quantize_fp8(changed[name])
        else:
            dt = sh.get(name).dtype
            changed[name] = changed[name].to(dt)
    stage = output.with_name(output.name+".building-"+uuid.uuid4().hex[:8])
    hardlink_tree(source_root, stage)
    touched = sorted({sh.wmap[n] for n in changed})
    for shard in touched:
        values: dict[str, Tensor] = {}
        with safe_open(source_root/shard, framework="pt", device="cpu") as f:
            for n in f.keys():
                original = f.get_tensor(n)
                values[n] = changed[n].view(original.dtype) if n in changed else original
        temp = stage/(shard+".new")
        save_file(values, temp)
        os.chmod(temp, 0o600)
        os.replace(temp, stage/shard)  # breaks the hardlink; source inode stays immutable
        del values
    # Never mutate a hardlinked manifest/Markov file, even when one already exists.
    manifest = json.loads(native_manifest.read_text())
    entries = []
    for record in manifest["tensors"]:
        n = record["name"]
        if n in changed:
            t = changed[n].contiguous().view(torch.uint8)
            entries.append({"name": n, "file": sh.wmap[n], "dtype": sh.info(n)["dtype"], "shape": list(changed[n].shape),
                            "sha256": hashlib.sha256(t.numpy().tobytes()).hexdigest(), "nbytes": t.numel()})
        else:
            entries.append({k: record[k] for k in ("name", "file", "dtype", "shape", "sha256", "nbytes")})
    manifest = {"schema": "n1-native-manifest/1", "root": str(output), "n_tensors": len(entries), "tensors": entries,
                "adaptation": {"phase": "2a", "adapter_sha256": adapter_digest, "source_manifest_sha256": sha256(native_manifest)}}
    for name, text in (("NATIVE_MANIFEST.json", json.dumps(manifest, separators=(",", ":"))+"\n"),
                       ("markov_top_tokens.json", json.dumps(hot_tokens(counts))+"\n")):
        temp = stage/(name+".new")
        temp.write_text(text)
        os.chmod(temp, 0o600)
        os.replace(temp, stage/name)
    packaged = OfficialShards(str(stage), str(stage/"NATIVE_MANIFEST.json"))
    for name in changed:
        packaged.verify(name)
    files = sorted({"config.json", "model.safetensors.index.json", "NATIVE_MANIFEST.json"}
                   | {sh.wmap[r["name"]] for r in entries})
    receipt: dict[str, Any] = {"schema": "d2.native-package.v1", "arm": "native-fp8", "publish": False, "deployed": False,
             "source": str(source_root), "destination": str(output), "adapter_sha256": adapter_digest,
             "changed_tensors": sorted(changed), "files": [{"file": n, "sha256": sha256(stage/n)} for n in files+["markov_top_tokens.json"]]}
    drafter, markov_id, _ = identities(receipt, tuple(files))
    receipt.update(drafter_identity=drafter, markov_identity=markov_id, identity_files=files,
                   exl3="Assessed only: requires calibration/re-encoding and separate K2 identity/parity; not built.")
    receipt_temp = stage/"PACKAGE_RECEIPT.json.new"
    receipt_temp.write_text(json.dumps(receipt, indent=2)+"\n")
    os.chmod(receipt_temp, 0o600)
    os.replace(receipt_temp, stage/"PACKAGE_RECEIPT.json")
    os.rename(stage, output)
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser(__doc__)
    for name in ("source", "output", "adapter", "native-manifest"):
        ap.add_argument("--"+name, required=True, type=Path)
    counts_group = ap.add_mutually_exclusive_group(required=True)
    counts_group.add_argument("--counts", type=Path)
    counts_group.add_argument("--data-manifest", type=Path, help="derive Markov input counts from train only")
    a = ap.parse_args()
    counts = training_counts(a.data_manifest) if a.data_manifest else {int(k): int(v) for k, v in json.loads(a.counts.read_text()).items()}
    receipt = build(a.source, a.output, a.adapter, counts, a.native_manifest)
    print(json.dumps({k: receipt[k] for k in ("arm", "drafter_identity", "markov_identity", "deployed")}))


if __name__ == "__main__":
    main()
