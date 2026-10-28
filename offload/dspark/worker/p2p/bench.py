"""Model-free eager collective A/B. Times complete local host calls plus device completion, not engine rounds.

Use --backend push|copy|both; copy is the independent event/peer-copy oracle. Acquire the lane's GPU locks first.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from . import P2PPair
from .reference import all_reduce


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=("push", "copy", "both"), default="both")
    ap.add_argument("--repeats", type=int, default=100)
    args = ap.parse_args()
    if torch.cuda.device_count() != 2 or args.repeats < 10:
        raise ValueError("benchmark needs the authorized pair and at least 10 repeats")
    devs = [torch.device("cuda", i) for i in (0, 1)]
    pair = P2PPair(devs, 32 * 5120)
    gen = torch.Generator().manual_seed(63)
    report: dict[str, object] = {"conditions": "resident FP32 inputs, BF16 outputs, eager host wall + both-device sync",
                                "repeats": args.repeats, "warmups": 5, "results": []}
    rows = []
    for nrows in (5, 10, 15, 20):
        cpu = [torch.randn(nrows * 5120, generator=gen) for _ in (0, 1)]
        xs = (cpu[0].to(devs[0]), cpu[1].to(devs[1]))
        want = (cpu[0] + cpu[1]).bfloat16()
        push_out = tuple(torch.empty_like(x, dtype=torch.bfloat16) for x in xs)
        for d in devs:
            torch.cuda.synchronize(d)
        for backend in (("push", "copy") if args.backend == "both" else (args.backend,)):
            times = []
            for rep in range(args.repeats + 5):
                start = time.perf_counter_ns()
                if backend == "copy":
                    result = all_reduce(xs)
                else:
                    for r in (0, 1):
                        with torch.cuda.device(devs[r]):
                            pair.ar_sum(r, xs[r], push_out[r], max(8, min(64, -(-xs[r].numel() // 2048))))
                    result = push_out
                for d in devs:
                    torch.cuda.synchronize(d)
                elapsed = (time.perf_counter_ns() - start) / 1000
                if rep >= 5:
                    times.append(elapsed)
            assert all(torch.equal(y.cpu(), want) for y in result), (nrows, backend)
            h = pair.headers()
            assert all(v[2] == 0 for v in h), h
            times.sort()
            row = {"rows": nrows, "bytes_each_way": nrows * 5120 * 4, "backend": backend,
                   "p50_us": round(statistics.median(times), 2), "p10_us": round(times[len(times) // 10], 2),
                   "p90_us": round(times[len(times) * 9 // 10], 2), "exact_both_ranks": True}
            rows.append(row)
            print(json.dumps(row), flush=True)
    report["results"] = rows
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
