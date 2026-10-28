"""Aggregate stage-error localization on private N1 fixtures (no updates)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .model import load
from .replay import relative
from .safety import assert_gpu_lease


def main() -> None:
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fixtures", type=Path, required=True)
    ap.add_argument("--mode", choices=("fast", "gpu32", "gpu64"), default="fast")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    assert_gpu_lease()
    torch.set_num_threads(8)
    torch.manual_seed(23)
    m = load(a.ckpt, "cuda", oracle=a.mode == "gpu64")
    if a.mode == "fast":
        m.enable_fast_training()
    m.eval()
    rows = []
    for p in sorted(a.fixtures.glob("n1-*.pt")):
        d = torch.load(p, map_location="cpu", weights_only=False)
        inp = {k: v.cuda() for k, v in d["inputs"].items()}
        trace: dict[str, Any] = {}
        with torch.no_grad():
            h, hx = m.hidden_from(inp, trace)
            picks, _ = m.greedy(h, hx, inp["anchor"])
        row = {"key": p.stem, "drafts_equal": picks[0].tolist() == d["dump"]["drafts"],
               "main_x_P_relative": relative(trace["main_x"][0, -1].cpu(), d["dump"]["main_x_P"]),
               "hidden_relative": relative(h[0].cpu(), d["dump"]["head_hidden"]),
               "stages": [{name: relative(trace[f"s{i}"][name][0].cpu(), d["dump"][f"s{i}"][name])
                           for name in ("attn_out", "ffn_in", "ffn_out", "h_out")}
                          for i in range(m.c.stages)]}
        rows.append(row)
        print(json.dumps(row), flush=True)
    a.out.write_text(json.dumps({"mode": a.mode, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
