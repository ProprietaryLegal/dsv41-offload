"""Run the pinned, unmodified official TileLang attention on bounded inputs.

Heads are independent; 16-head launches fit SM86 shared memory. This is a
component source-fidelity check, not a full official model serving result.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, cast

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--official", type=Path, required=True)
    ap.add_argument("--dependencies", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if hashlib.sha256(a.official.read_bytes()).hexdigest() != "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455":
        raise ValueError("official kernel source hash differs from V1 pin")
    sys.path.insert(0, str(a.dependencies))
    spec = importlib.util.spec_from_file_location("n4_official_kernel", a.official)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned official kernel")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    official = cast(Any, module).sparse_attn
    rows = []
    a.out.mkdir(parents=True, exist_ok=False)
    for p in sorted(a.inputs.glob("*-s0.pt")):
        d = torch.load(p, weights_only=False)
        q, kv, sink, idx = (d[k].cuda() for k in ("q", "kv", "sink", "indices"))
        parts = []
        for h in range(0, q.shape[2], 16):
            parts.append(official(q[:, :, h:h + 16].contiguous(), kv, sink[h:h + 16].contiguous(),
                                  idx.int(), d["scale"]))
        got = torch.cat(parts, dim=2).cpu()
        want = d["out"]
        rows.append(dict(sample=p.stem, unequal=int((got != want).sum()), elements=got.numel(),
                         rel=float((got.float() - want.float()).norm() / want.float().norm())))
        torch.save(got, a.out / p.name)
        print(json.dumps(rows[-1]), flush=True)
    (a.out / "aggregate.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
