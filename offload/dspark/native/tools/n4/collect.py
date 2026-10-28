"""Strict aggregate-only comparison of all 48 same-card acquisitions."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

UUID = "GPU-bc62df2a-59cf-9828-7d43-558d1355bab7"
ARMS = ("native-packed", "native-expanded", "native-hybrid-kx", "native-hybrid-kx2")


def span(values: list[float]) -> dict[str, float]:
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def collect(root: Path) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    records: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for side in ("before", "repo"):
        for arm in ARMS:
            for rank in (0, 1):
                rows = []
                for rep in (1, 2, 3):
                    name = f"repeat{rep}-{side}-{arm}-r{rank}.json"
                    d = json.loads((root / name).read_text())
                    if d["devices"] != [UUID] or d["arm"] != arm or d["rank"] != rank or d["mode"] != "solo":
                        raise ValueError(f"wrong device/arm/rank/mode: {name}")
                    if d["iters"] != 200:
                        raise ValueError(f"sample count mismatch: {name}")
                    if side in sources and sources[side] != d["source_sha256"]:
                        raise ValueError(f"source drift within {side}: {name}")
                    sources[side] = d["source_sha256"]
                    expected = {f"B{b}_k5_n{n}" for b in range(1, 5) for n in (4, 5)}
                    if set(d["draft"]) != expected:
                        raise ValueError(f"incomplete timing cells: {name}")
                    for cell in d["draft"].values():
                        if cell["settle_seconds"] != .5 or not cell["breakdown"]:
                            raise ValueError(f"missing settling/kernel maps: {name}")
                        # Check cap at every recorded case boundary, both policies.
                        for field in ("gpu_before", "gpu_after"):
                            if "225.00 W" not in str(cell[field][UUID]):
                                raise ValueError(f"power cap mismatch: {name}")
                    rows.append(d)
                records[side, arm, rank] = rows
    comparisons = []
    memory = []
    for arm in ARMS:
        for rank in (0, 1):
            old, new = records["before", arm, rank], records["repo", arm, rank]
            for key in old[0]["draft"]:
                before = span([float(d["draft"][key]["p50_us"]) for d in old])
                after = span([float(d["draft"][key]["p50_us"]) for d in new])
                comparisons.append(dict(arm=arm, rank=rank, cell=key, before_us=before, after_us=after,
                                        change_percent=100 * (after["median"] / before["median"] - 1)))
            for side, rows in (("before", old), ("after", new)):
                memory.append(dict(arm=arm, rank=rank, policy=side,
                    load_seconds=span([float(d["load_seconds"]) for d in rows]),
                    load_allocated_bytes=max(v["max_allocated"] for d in rows for v in d["load_peak"].values()),
                    process_allocated_highwater_through_serving_bytes=max(
                        v["max_allocated"] for d in rows for v in d["serving_peak"].values()),
                    post_serving_smi=[d["serving_nvidia_smi"] for d in rows]))
    return dict(schema="n4.same-card-arithmetic-comparison.v1", gpu=UUID, cap_watts=225,
                repetitions=3, acquisitions=48, source_sha256=sources, comparisons=comparisons, memory=memory,
                scope="solo channel shards with local collective stand-ins; not physical pair or GB10")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.write_text(json.dumps(collect(a.root), indent=2) + "\n")


if __name__ == "__main__":
    main()
