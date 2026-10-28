"""CPU-only safety and reduction helpers for the isolated W1 profiler."""

from __future__ import annotations

import os
from pathlib import Path
from statistics import median
from typing import Mapping, Sequence

GPU_UUID = "GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc"
PROCESS_LIMIT_MIB = 20 * 1024


def require_gpu_window(gate: Path | None = None) -> None:
    """Call before importing CUDA libraries, each capture, and each measured sample."""
    marker = gate if gate is not None else Path.home() / "dsv41-offload/W1_GPU_OK"
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
        raise RuntimeError(f"W1 requires CUDA_VISIBLE_DEVICES={GPU_UUID}")
    if not marker.is_file():
        raise RuntimeError(f"W1 GPU window closed: required marker absent: {marker}")


def summarize(samples: Sequence[Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """Reduce paired samples; never subtract independently computed percentiles.

    CPU wall minus GPU timeline is a residual, NOT a pure CPU execution measurement:
    it includes event waits, device scheduling and event submission overhead.
    """
    if not samples:
        raise ValueError("cannot summarize an empty profile")
    rows = [dict(s) for s in samples]
    for row in rows:
        row["wall_minus_gpu_us"] = row["wall_us"] - row["gpu_us"]
    result: dict[str, dict[str, float]] = {}
    for key in rows[0]:
        values = sorted(row[key] for row in rows)
        result[key] = {"p50": median(values), "p10": values[int((len(values) - 1) * .1)],
                       "p90": values[int((len(values) - 1) * .9)], "mean": sum(values) / len(values)}
    return result


def expert_counts(ids: Sequence[int], experts: int = 128) -> list[int]:
    counts = [0] * experts
    for expert in ids:
        if not 0 <= expert < experts:
            raise ValueError(f"expert {expert} outside 0..{experts - 1}")
        counts[expert] += 1
    return counts
