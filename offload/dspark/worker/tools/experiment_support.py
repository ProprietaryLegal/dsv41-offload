"""Paired graph microbenchmarks for owned W1b experiments."""
from __future__ import annotations

from collections.abc import Callable
from statistics import median
from typing import TYPE_CHECKING
import time

from .profile_support import require_gpu_window
if TYPE_CHECKING:
    import torch


def paired_graphs(baseline: Callable[[], object], candidate: Callable[[], object],
                  repeats: int = 30) -> dict[str, object]:
    """Capture independent pools, alternate order, reduce within-pair deltas.

    Retain outputs for graph allocation lifetime. GPU events measure each replay;
    CPU wall includes submission and the completion wait. No percentile subtraction.
    """
    require_gpu_window()
    import torch
    graphs: list[torch.cuda.CUDAGraph] = []
    outputs: list[object] = []
    for fn in (baseline, candidate):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs.append(fn())
        graphs.append(graph)
    pairs: list[dict[str, float]] = []
    for i in range(repeats + 5):
        require_gpu_window()
        row: dict[str, float] = {}
        for j in ((0, 1) if i % 2 == 0 else (1, 0)):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter_ns()
            a.record()
            graphs[j].replay()
            b.record()
            b.synchronize()
            row[f"{'baseline' if j == 0 else 'candidate'}_gpu_us"] = a.elapsed_time(b)*1000
            row[f"{'baseline' if j == 0 else 'candidate'}_wall_us"] = (time.perf_counter_ns()-t0)/1000
        if i >= 5:
            row["saving_gpu_us"] = row["baseline_gpu_us"] - row["candidate_gpu_us"]
            row["saving_wall_us"] = row["baseline_wall_us"] - row["candidate_wall_us"]
            pairs.append(row)
    return {"pairs": pairs, "median": {k: median(r[k] for r in pairs) for k in pairs[0]}}
