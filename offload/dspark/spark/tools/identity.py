#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Drafter / Markov identities of CONTRACT.md 6.7 from lane S2's copy receipt.

    python3 offload/dspark/spark/tools/identity.py \
        /opt/dsv41/campaigns/DSV41-OFFLOAD/S2.DSPARK_COPY_DONE.json

Prints VLLM_DSPARK_REMOTE_IDENTITY=<hex> and VLLM_DSPARK_REMOTE_MARKOV_IDENTITY=<hex>.
The drafter identity covers exactly the files the worker reads tensors or configuration
from (DRAFTER_FILES); the sha256 values are the Spark-side source hashes S2 recorded
(equal to the deployment's manifest.json). Lane O2 must compute the same digest.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any
import argparse
from pathlib import Path

DRAFTER_FILES = (
    "config.json",
    "exl3_moe_plan.json",
    "model-00002-of-00046.safetensors",
    "model-00043-of-00046.safetensors",
    "model-00044-of-00046.safetensors",
    "model-00045-of-00046.safetensors",
    "model-00046-of-00046.safetensors",
    "model-exl3-head.safetensors",
    "model-exl3-mtp.safetensors",
    "model-exl3-mtp-experts.safetensors",
)
MARKOV_FILE = "markov_top_tokens.json"


def identities(receipt: dict[str, Any], files: tuple[str, ...] = DRAFTER_FILES) -> tuple[str, str, dict[str, str]]:
    hashes = {os.path.basename(f["source"] if "source" in f else f["file"]): f["sha256"]
              for f in receipt["files"]}
    if not files or len(files) != len(set(files)) or any(os.path.basename(n) != n for n in files):
        raise ValueError("identity files must be distinct basenames")
    missing = [n for n in files if n not in hashes]
    if missing:
        raise SystemExit(f"receipt lacks {missing}")
    lines = "".join(f"{n} {hashes[n].lower()}\n" for n in sorted(files))
    drafter = hashlib.sha256(lines.encode()).hexdigest()
    markov = hashes.get(MARKOV_FILE, "0" * 64).lower()
    return drafter, markov, {n: hashes[n] for n in sorted(files)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("receipt")
    ap.add_argument("--arm", default="k2-tp2")
    ap.add_argument("--json-out", type=Path, help="write the identity recipe (no weights or capture values)")
    a = ap.parse_args()
    with open(a.receipt) as f:
        receipt = json.load(f)
    drafter, markov, used = identities(receipt)
    result: dict[str, Any] = {"drafter_identity": drafter, "markov_identity": markov, "files": used}
    if a.arm != "k2-tp2":
        # Also works when invoked as a script from outside the checkout. No torch.
        sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
        from offload.dspark.native.identity import compute
        if MARKOV_FILE not in {os.path.basename(f.get("source", f.get("file", ""))) for f in receipt["files"]}:
            raise SystemExit("native identity requires an explicit Markov file receipt")
        result = compute(a.arm, result)
        drafter, markov = result["drafter_identity"], result["markov_identity"]
    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(f"VLLM_DSPARK_REMOTE_IDENTITY={drafter}")
    print(f"VLLM_DSPARK_REMOTE_MARKOV_IDENTITY={markov}")
    if "-v" in os.environ.get("IDENTITY_VERBOSE", ""):
        for n, h in used.items():
            print(f"  {n} {h}")


if __name__ == "__main__":
    main()
