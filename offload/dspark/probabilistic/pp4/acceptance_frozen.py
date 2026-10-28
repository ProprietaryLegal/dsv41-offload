"""Read-only, hash-pinned PP2 inputs; materialize only in lane scratch."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

COMMIT = "e1befc9aaa081a4f6cccbb11423a747fcd392503"
PREFIX = "offload/dspark/probabilistic/pp2/"
HASHES = {
    "dev-selection.json": "72989e3c7c2bd3ab9d489d8457e02e89a1bd7fbdc1c298f52fa21715f70615be",
    "heldout-replay.json": "36435b607ca8c56de4ffc539c49926a50085ef4ec33071f645387c2561be0a99",
    "numeric.py": "c47cb3dfae376b8ffed48ca6c283a0a7dd9d1cfbc5d1b6e1a91b45aa8b28dcb2",
    "heldout-summary.json": "e95d0b35095db74b6886a2c52e84589391fa87144eb3e8f61414dbbe88311197",
    "summarize.py": "070ff608f389b107386f1020731a169d7c5785e2b01adac922b4109dae5f62e7",
}


def read(root: Path, name: str) -> bytes:
    data = (root / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != HASHES[name]:
        raise ValueError("frozen PP2 input differs: " + name)
    return data


def value(root: Path, name: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(read(root, name)))


def module(root: Path, name: str) -> ModuleType:
    read(root, name + ".py")
    spec = importlib.util.spec_from_file_location("pp4_frozen_pp2_" + name, root / (name + ".py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned PP2 source")
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


def materialize(root: Path) -> None:
    if not (str(root.resolve()).startswith("/opt/dsv41/campaigns/DSV41-OFFLOAD/scratch/PP4/")
            or str(root.resolve()).startswith("/opt/dsv41/x86host-data/pp4/")):
        raise ValueError("PP4 scratch destination required")
    root.mkdir(parents=True, exist_ok=True)
    for name, expected_hash in HASHES.items():
        data = subprocess.run(["git", "show", COMMIT + ":" + PREFIX + name],
                              check=True, capture_output=True).stdout
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError("git frozen-input identity mismatch: " + name)
        (root / name).write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    materialize(args.directory)
    print("PASS: five PP2 inputs hash-verified at " + COMMIT)


if __name__ == "__main__":
    main()
