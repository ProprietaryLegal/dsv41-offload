"""Sustained synthetic N1 training benchmark, including backward and fused AdamW.

No captures, real training shards, or adapted weight exports. Cached projection
mode is explicitly labeled; its setup cost and memory are recorded separately.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import time
from typing import Callable, cast

import torch

from .loss import distillation_loss
from .model import load
from .perf import n1_inputs, synthetic_targets
from .safety import GPU, assert_gpu_lease
from .fast_policy import BACKEND


def main() -> None:
    os.umask(0o077)
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fixtures", type=Path, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--cache-projection", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.steps < 1 or a.warmup < 1:
        ap.error("positive steps/warmup required; >=200 measured steps for a sustained claim")
    if a.cache_projection:
        raise ValueError("projection cache refused: frozen weights AND inputs are unproven; use raw taps")
    assert_gpu_lease()
    torch.set_num_threads(8)
    torch.manual_seed(23)
    begin = time.perf_counter()
    model = load(a.ckpt, "cuda")
    model.enable_fast_training()
    torch.cuda.empty_cache()
    inp = n1_inputs(a.fixtures, 4 if a.cache_projection else a.batch, "cuda")
    setup = time.perf_counter() - begin
    projection_seconds = 0.
    target = synthetic_targets(model, inp)
    parameters = [p for n, p in model.named_parameters() if "confidence_head" not in n]
    optimizer = torch.optim.AdamW(parameters, lr=2e-4, foreach=False, fused=True)
    rows = []
    model.train()
    for step in range(a.warmup + a.steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        begin = time.perf_counter()
        h, _ = model.hidden_from(inp)
        loss, _ = distillation_loss(model, h, chunk=a.chunk, **target)
        cast(Callable[[], None], loss.backward)()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True, foreach=True)
        optimizer.step()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - begin
        row = {"step": step, "warmup": step < a.warmup, "seconds": seconds, "loss": float(loss.detach())}
        rows.append(row)
        print(json.dumps(row), flush=True)
    measured = rows[a.warmup:]
    seconds = sum(float(row["seconds"]) for row in measured)
    rate = a.batch * a.steps / seconds
    peak = torch.cuda.max_memory_reserved()
    report = {"gpu_uuid": GPU, "backend": BACKEND, "batch": a.batch, "chunk": a.chunk,
              "sustained": a.steps >= 200, "measured_steps": a.steps, "warmup_steps": a.warmup,
              "seconds": seconds, "blocks_per_second": rate, "setup_seconds": setup,
              "cached_projection": a.cache_projection, "projection_setup_seconds": projection_seconds,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": peak,
              "memory_le22gb": peak <= 22_000_000_000, "throughput_ge200": rate >= 200,
              "projected_2p4m_block_pilot_hours": 2_400_000 / rate / 3600,
              "projected_3m_block_hours": 3_000_000 / rate / 3600,
              "host_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "frozen_identity": model.frozen_identity, "rows": rows,
              "scope": "Four repeated synthetic N1 windows; GPU compute including optimizer; excludes dataset I/O. "
                       "Projection cache, when selected, remains before trainable main_norm. No adapted weights saved."}
    a.out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
