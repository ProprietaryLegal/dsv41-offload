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


def identities(receipt: dict) -> tuple[str, str, dict[str, str]]:
    hashes = {os.path.basename(f["source"] if "source" in f else f["file"]): f["sha256"]
              for f in receipt["files"]}
    missing = [n for n in DRAFTER_FILES if n not in hashes]
    if missing:
        raise SystemExit(f"receipt lacks {missing}")
    lines = "".join(f"{n} {hashes[n].lower()}\n" for n in sorted(DRAFTER_FILES))
    drafter = hashlib.sha256(lines.encode()).hexdigest()
    markov = hashes.get(MARKOV_FILE, "0" * 64).lower()
    return drafter, markov, {n: hashes[n] for n in sorted(DRAFTER_FILES)}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    with open(sys.argv[1]) as f:
        receipt = json.load(f)
    drafter, markov, used = identities(receipt)
    print(f"VLLM_DSPARK_REMOTE_IDENTITY={drafter}")
    print(f"VLLM_DSPARK_REMOTE_MARKOV_IDENTITY={markov}")
    if "-v" in os.environ.get("IDENTITY_VERBOSE", ""):
        for n, h in used.items():
            print(f"  {n} {h}")


if __name__ == "__main__":
    main()
