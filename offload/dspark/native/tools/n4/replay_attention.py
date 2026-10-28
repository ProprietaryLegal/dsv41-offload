"""Replay private raw attention inputs through the current native kernel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ...attention import attention_into


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for p in sorted(a.inputs.glob("attention-*.pt")):
        d = torch.load(p, weights_only=False)
        t = d["tensors"]
        args = (t["q"].cuda(), t["ring"].cuda(), t["positions"].cuda(), t["slots"].cuda(),
                t["lo"].cuda(), t["last"].cuda(), t["block"].cuda(), t["sink"].cuda())
        out = torch.empty_like(args[0])
        attention_into(*args, d["scale"], out)
        t["out"] = out.cpu()
        torch.save(d, a.out / p.name)
        rows.append(dict(name=p.name, elements=out.numel()))
    (a.out / "aggregate.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
