"""Native-arm identity: source provenance, exact arm policy, K6 head and runtime sources.

Pure Python, shared by the worker and the Spark identity tool. The worker verifies
the official receipt's whole files before loading. The existing checkpoint identity
still verifies every retained checkpoint tensor (embedding, norms, Markov, K6 head).
Neither path trusts an advertised digest in place of checking its own bytes.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .arms import ARMS, batch_invariant_from_env

from ..worker.arm_selection import SERVING_ARMS, selected_arm

ROOT = Path(__file__).resolve().parents[1]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def digest(recipe: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def official_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    pairs = re.findall(r"^([0-9a-f]{64})  ([^\n]+)$", manifest["receipt"], re.M)
    hashes = {name: h for h, name in pairs}
    required = {"config.json", "model.safetensors.index.json"} | {t["file"] for t in manifest["tensors"]}
    if not required <= hashes.keys():
        raise ValueError(f"official receipt lacks {sorted(required - hashes.keys())}")
    return {n: hashes[n] for n in sorted(required)}


def runtime_hashes(root: Path = ROOT) -> dict[str, str]:
    """Conservative coverage, including shared kernels; excludes tools/tests/private data."""
    paths = []
    for subdir in ("worker", "native", "kernels"):
        for p in (root / subdir).rglob("*"):
            rel = p.relative_to(root)
            if any(x in rel.parts for x in ("tests", "tools", "__pycache__", "results")):
                continue
            if p.is_file() and p.suffix in (".py", ".cu", ".cuh", ".cpp", ".h", ".c", ".json"):
                paths.append(p)
    if not paths:
        raise ValueError(f"no runtime sources under {root}")
    return {str(p.relative_to(root)): sha256(p) for p in sorted(paths)}


def compute(arm: str, checkpoint: dict[str, Any], *, native_root: str | None = None,
            source_root: Path = ROOT) -> dict[str, Any]:
    """native_root present = startup verification; absent = offline matching identity."""
    selected_arm(arm)
    if arm == "k2-tp2":
        return checkpoint
    cfg = ARMS[arm]
    cfg.check()
    manifest_path = source_root / "native/NATIVE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    hashes = official_hashes(manifest)
    if native_root is not None:
        root = Path(native_root).resolve(strict=True)
        for n, want in hashes.items():
            p = root / n
            if p.resolve().parent != root or not p.is_file():
                raise ValueError(f"{arm}: missing or redirected official shard/config {p}")
            got = sha256(p)
            if got != want:
                raise ValueError(f"{arm}: official {n} sha256 {got} != receipt {want}")
    recipe = {
        "schema": "dspark.native-identity.v1",
        "arm": asdict(cfg),
        "draft_head": "target-shared EXL3 K6 (not official BF16 head)",
        "official_repo": manifest["official_repo"],
        "official_revision": manifest["official_revision"],
        "official_files": hashes,
        "official_tensor_manifest_sha256": sha256(manifest_path),
        # Retained tensors are still loaded from the verified delivery. Conservative
        # coverage includes unused K2 files, but the native digest is never the K2 id.
        "retained_checkpoint_files": {n: h.lower() for n, h in checkpoint["files"].items()},
        "runtime_sources": runtime_hashes(source_root),
    }
    level = batch_invariant_from_env()      # NB1 switch: absent from the recipe when off
    if level:
        if not cfg.kx2_dispatch:
            raise ValueError(f"{arm}: batch-invariant dispatch is implemented for the kx2 dispatch only")
        recipe["batch_invariant_dispatch"] = ("nb1-draft-v1: B1 split-K / FP8 / expert variant at every B<=4" +
                                              ("; insert projections in 8-row sub-batches" if level == 2 else ""))
    return {"drafter_identity": digest(recipe), "markov_identity": checkpoint["markov_identity"],
            "arm": arm, "draft_head": recipe["draft_head"], "recipe": recipe}
