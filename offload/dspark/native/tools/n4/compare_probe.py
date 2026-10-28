"""Aggregate private identical-input attention probes without exporting values."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ...fidelity_reference import online_attention


def error(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | int]:
    return dict(rel=float((a.double() - b.double()).norm() / b.double().norm()),
                unequal=int((a != b).sum()), elements=a.numel())


def compare(actual: Path, reference: Path) -> list[dict[str, Any]]:
    rows = []
    for i, p in enumerate(sorted(actual.glob("attention-*.pt"))):
        payload = torch.load(p, weights_only=False)
        a = payload["tensors"]
        q, ring, slots, last = a["q"], a["ring"], a["slots"], a["last"]
        if q.shape[0] != 1:
            raise ValueError("bounded B1 diagnostic expected")
        k = q.shape[1]
        nc = min(128, int(last[0]) + 1)
        idx = torch.cat((torch.arange(nc), 128 + torch.arange(k))).view(1, 1, -1).expand(1, k, -1)
        kv = torch.cat((ring[slots], a["block"]), dim=1)
        own = online_attention(q, kv, a["sink"], idx, payload["scale"])
        sample = ("r000008_b0", "r000011_b0")[i // 6]
        stage, rank = i % 6 // 2, i % 2
        r = torch.load(reference / f"{sample}-s{stage}.pt", weights_only=False)
        heads = slice(rank * 32, (rank + 1) * 32)
        want = r["out"][:, :, heads]
        rows.append(dict(sample=sample, stage=stage, rank=rank,
            gpu_vs_cpu_same_inputs=error(a["out"], own),
            gpu_vs_official=error(a["out"], want),
            cpu_own_vs_official=error(own, want),
            q_vs_official=error(q, r["q"][:, :, heads]),
            kv_vs_official=error(kv, r["kv"])))
    if len(rows) != 12:
        raise ValueError(f"incomplete raw probe: {len(rows)} calls")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--actual", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.write_text(json.dumps(compare(a.actual, a.reference), indent=2) + "\n")


if __name__ == "__main__":
    main()
