"""Solo-only synthetic top32 qualification and graph timing; no serve requests."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
from collections.abc import Callable
from pathlib import Path

import torch

from ..sampled import MASS, Top32

UUID = "GPU-bc62df2a-59cf-9828-7d43-558d1355bab7"


def timing(fn: Callable[[], object], repetitions: int = 100) -> dict[str, float]:
    for _ in range(3):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            fn()
    samples: list[float] = []
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(repetitions):
        start.record()
        graph.replay()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) / 10)
    samples.sort()
    return {"median_ms": samples[len(samples)//2], "p95_ms": samples[int(.95*len(samples))]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != UUID:
        raise RuntimeError("PP3 solo UUID required")
    import fcntl
    fd = int(os.environ["PP3_LOCK_FD"])
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-") != UUID.removeprefix("GPU-"):
        raise RuntimeError("wrong device")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.manual_seed(4103)
    dev = torch.device("cuda:0")
    out: dict[str, object] = {"schema": "pp3.producer.v1", "classification": "MEASURED solo synthetic logits",
                             "complete": False, "torch": torch.__version__, "uuid": UUID,
                             "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out["producer_sha256"] = hashlib.sha256((Path(__file__).parents[1]/'sampled.py').read_bytes()).hexdigest()
    rows: list[dict[str, object]] = []
    out["rows"] = rows

    def save() -> None:
        out["maxrss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        out["gpu_peak_bytes"] = torch.cuda.max_memory_allocated()
        a.out.write_text(json.dumps(out, indent=2)+"\n")

    for batch in range(1, 5):
        x = torch.randn(batch, 129280, device=dev).to(torch.bfloat16)
        keys = torch.tensor([[4103+j, 1, 32768] for j in range(batch)], device=dev, dtype=torch.int64)
        tau = torch.full((batch,), .7, device=dev)
        sampler = Top32(batch, 129280, dev)
        tok, ids, q = sampler(x, keys, tau)
        torch.cuda.synchronize()
        if sampler.error.any().item():
            raise AssertionError("valid proposal refused")
        expected = torch.argsort(x.float(), descending=True, stable=True)[:, :32].sort(-1).values
        torch.testing.assert_close(ids.long(), expected, rtol=0, atol=0)
        if not torch.equal((q.double()*MASS).sum(-1), torch.full((batch,), MASS, device=dev, dtype=torch.float64)):
            raise AssertionError("nonconserving mass")
        if not ((ids == tok[:, None]) & (q > 0)).any(-1).all().item():
            raise AssertionError("proposal outside support")
        for depth in (4, 5):
            def sampled(depth: int = depth, sampler: Top32 = sampler, x: torch.Tensor = x,
                        keys: torch.Tensor = keys, tau: torch.Tensor = tau) -> None:
                for _ in range(depth):
                    sampler(x, keys, tau)

            def greedy(depth: int = depth, x: torch.Tensor = x) -> None:
                for _ in range(depth):
                    x.argmax(-1)

            row: dict[str, object] = {"batch": batch, "depth": depth, "sampled": timing(sampled),
                   "argmax": timing(greedy), "scope": "logit selection only; excludes Markov, TP exchange, wire"}
            rows.append(row)
            save()
            print(json.dumps(row), flush=True)
    out["complete"] = True
    save()


if __name__ == "__main__":
    main()
