"""CPU-only P2P ABI build/import and compact PP4 runtime receipt."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import cast

import torch
from torch.utils.cpp_extension import load


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/opt/dsv41/x86host-data/pp4/runtime"))
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or cast(Callable[[], bool], torch.cuda.is_initialized)():
        raise RuntimeError("P2P build requires CUDA hidden and no initialized GPU context")
    if os.sched_getaffinity(0) != {12, 13, 14, 15, 28, 29, 30, 31} or os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise RuntimeError("P2P build requires reserved affinity and nice19")
    root, source = args.root.resolve(), args.source.resolve()
    if root != Path("/opt/dsv41/x86host-data/pp4/runtime") or not source.is_relative_to(root):
        raise RuntimeError("P2P sources and outputs must remain inside assigned runtime scratch")
    key = digest(source)[:16]
    build = root / "p2p" / key
    build.mkdir(parents=True, exist_ok=True)
    module = cast(ModuleType, load(name="dspark_p2p", sources=[str(source)], build_directory=str(build), verbose=True,
                                  extra_cflags=["-O2"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_86,code=sm_86"]))
    if module.__file__ is None:
        raise RuntimeError("compiled P2P module lacks an artifact path")
    compiled = Path(module.__file__)
    compiled.with_suffix(".so.sha256").write_text(digest(compiled) + "\n")
    modules: dict[str, Path] = {"dspark_p2p": compiled}
    for name in ("exllamav3_ext", "exllamav3_ext_facc"):
        path = importlib.import_module(name).__file__
        if path is None:
            raise RuntimeError(f"{name} lacks an artifact path")
        modules[name] = Path(path)
    manifest = json.loads((root / "reuse-manifest.json").read_text())
    source_trees: dict[str, str] = {}
    for name in ("exllamav3", "exllamav3-facc"):
        base = root / name / "exllamav3/exllamav3_ext"
        files = sorted(path for path in base.rglob("*") if path.is_file())
        source_trees[name] = hashlib.sha256("".join(
            f"{path.relative_to(base)} {digest(path)}\n" for path in files).encode()).hexdigest()
    result = {"label": "MEASURED", "scope": "CPU build and ABI import only; no pair/GPU execution",
              "python": sys.version, "torch": torch.__version__, "triton": importlib.import_module("triton").__version__,
              "modules": {name: {"path": str(path), "sha256": digest(path)} for name, path in modules.items()},
              "p2p_source_sha256": digest(source), "source_tree_sha256": source_trees,
              "reuse_manifest_sha256": digest(root / "reuse-manifest.json"),
              "reused_objects": len(manifest["approved"]), "rebuilt_objects": len(manifest["rejected"]),
              "compiler_wheels": {path.name: digest(path) for path in sorted((root / "wheels").glob("*.whl"))}}
    (root / "runtime-build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
