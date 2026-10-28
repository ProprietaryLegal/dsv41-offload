"""Aggregate eight real solo-shard/full40-graph/1M-cap receipts."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from .collect import ARMS, UUID


def sampled_peaks(root: Path, name: str) -> dict[str, float]:
    phases = [json.loads(line) for line in (root / f"{name}.phase.jsonl").read_text().splitlines()]
    if [p["phase"] for p in phases] != ["load", "prewarm", "serve", "done"]:
        raise ValueError(f"incomplete phase chronology: {name}")
    if any(b["time"] <= a["time"] for a, b in zip(phases, phases[1:])):
        raise ValueError(f"phase time moved backwards: {name}")
    peaks: dict[str, float] = {}
    for line in (root / f"{name}.smi.csv").read_text().splitlines():
        fields = [s.strip() for s in line.split(",")]
        if fields[1] != UUID or float(fields[4].split()[0]) != 225:
            raise ValueError(f"wrong device or cap: {name}")
        timestamp = dt.datetime.strptime(fields[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
        phase = next((p["phase"] for p in reversed(phases) if timestamp >= p["time"]), "startup")
        peaks[phase] = max(peaks.get(phase, 0.), float(fields[2].split()[0]))
    if not {"load", "prewarm", "serve"} <= peaks.keys():
        raise ValueError(f"missing phase memory samples: {name}")
    return peaks


def collect(root: Path, expected_path: Path, prefix: str = "qualified") -> dict[str, Any]:
    if prefix not in ("qualified", "baseline-qualified"):
        raise ValueError("unknown qualification policy prefix")
    expected = json.loads(expected_path.read_text())
    rows = []
    cells = {f"B{b}_n{n}_margin{m}" for b in range(1, 5) for n in range(1, 6) for m in (0, 1)}
    for arm in ARMS:
        for rank in (0, 1):
            name = f"{prefix}-{arm}-r{rank}"
            d = json.loads((root / f"{name}.json").read_text())
            if d["arm"] != arm or d["rank"] != rank or d["devices"] != UUID:
                raise ValueError(f"wrong scope: {name}")
            if d["identity"] != expected[arm]["drafter_identity"]:
                raise ValueError(f"stale arithmetic/source identity: {name}")
            if d["rounds"] < 2000 or set(d["latency_ms"]) != cells:
                raise ValueError(f"incomplete graph/replay coverage: {name}")
            if sum(c["n"] for c in d["latency_ms"].values()) != d["rounds"] or any(
                    c["n"] < 50 for c in d["latency_ms"].values()):
                raise ValueError(f"incomplete per-shape replay counts: {name}")
            if d["rank0_output_checks"] is not (rank == 0):
                raise ValueError(f"incorrect output-check scope label: {name}")
            rows.append({k: d[k] for k in ("arm", "rank", "identity", "rounds", "load_s", "capture_s",
                "load_memory", "prewarm_memory", "serving_memory", "rank0_output_checks")} |
                {"graph_cases": 40, "memory_peak_mib_1s_observed": sampled_peaks(root, name)})
    return dict(schema="n4.solo-shard-qualification.v1", gpu=UUID, cap_watts=225,
                policy="before" if prefix == "baseline-qualified" else "after",
                max_model_len=1048576, graph_cases_per_rank=40, shards=rows,
                scope="local collective stand-ins, not physical TP2; rank1 output checks absent as labelled")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--expected", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prefix", choices=("qualified", "baseline-qualified"), default="qualified")
    a = ap.parse_args()
    a.out.write_text(json.dumps(collect(a.root, a.expected, a.prefix), indent=2) + "\n")


if __name__ == "__main__":
    main()
