"""One real native TP2 shard: startup, memory, all graph shapes and >=2000 replays.

This measures each rank independently; local collective stand-ins are not a
physical pair or a complete server/HELLO qualification.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast

import torch

from ...identity import compute
from ...tp2_native import build
from ....worker.checkpoint import Checkpoint
from ....worker.identity import compute as checkpoint_identity
from ....worker.tp2 import check_visible
from ....worker.tools.tp2_bench import setup_inputs, fill_windows_tp, host_inp
from ....worker.tools.parity_ref import make_taps


def memory(dev: torch.device) -> dict[str, int]:
    return {"allocated": torch.cuda.memory_allocated(dev), "reserved": torch.cuda.memory_reserved(dev),
            "peak_allocated": torch.cuda.max_memory_allocated(dev), "peak_reserved": torch.cuda.max_memory_reserved(dev)}


def phase(path: Path, name: str) -> None:
    line = json.dumps({"phase": name, "time": time.time()}) + "\n"
    path.write_text(line)
    with path.with_suffix('.jsonl').open('a') as log:
        log.write(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm")
    ap.add_argument("--rank", type=int, choices=(0, 1), required=True)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", default="/opt/dsv41/x86host-data/dspark")
    ap.add_argument("--native-root", default="/opt/dsv41/x86host-data/native-1017")
    a = ap.parse_args()
    if a.rounds < 2000:
        raise ValueError("at least 2000 graph replays required")
    vis = check_visible(pair=False)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    marker = a.out.with_suffix(".phase.json")
    phase(marker, "load")
    start = time.monotonic()
    ck = Checkpoint(a.root)
    ident = compute(a.arm, checkpoint_identity(a.root, ck.manifest), native_root=a.native_root)
    dev = torch.device("cuda", 0)
    torch.empty(1, device=dev)
    torch.cuda.reset_peak_memory_stats(dev)
    m = build(a.arm, [dev, dev], ck, native_root=a.native_root, solo_rank=a.rank,
              slots=8, max_model_len=1048576)
    rep = {"arm": a.arm, "rank": a.rank, "mode": "solo-shard-local-collectives", "devices": vis,
           "identity": ident["drafter_identity"], "head": "EXL3 K6", "load_s": time.monotonic() - start,
           "load_memory": memory(dev)}
    print(json.dumps(rep), flush=True)
    phase(marker, "prewarm")
    # The legacy helpers are duck-typed around .embed, despite their one-card annotation.
    setup = cast(Callable[[torch.device, str], torch.Tensor], setup_inputs)
    make = cast(Callable[[Any, torch.Tensor, torch.Generator], torch.Tensor], make_taps)
    toks = setup(dev, a.root)
    taps = make(SimpleNamespace(embed=m.ranks[a.rank].embed), toks, torch.Generator(device=dev).manual_seed(5)).cpu()
    fill_windows_tp(m, taps)
    rep["capture_s"] = m.capture(batch_sizes=(1, 2, 3, 4), ks=(5,), n_outs=(1, 2, 3, 4, 5),
                                  insert_rows=(8, 16, 32, 64, 128), margins=True)
    rep["prewarm_memory"] = memory(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    phase(marker, "serve")
    times: dict[str, list[float]] = {}
    start = time.monotonic()
    for i in range(a.rounds):
        B, n, margin = 1 + (i // 10) % 4, 1 + i % 5, bool((i // 5) % 2)
        inp = host_inp(B, toks.cpu())
        t = time.monotonic()
        ids, conf, mg = m.draft_graphed(inp, 5, n_out=n, margin=margin)
        torch.cuda.synchronize(dev)
        if a.rank == 0:
            assert bool(torch.isfinite(conf).all())
            assert bool(((ids[:, :n] >= 0) & (ids[:, :n] < m.s.vocab)).all())
            assert bool((ids[:, n:] == 0).all() & (conf[:, n:] == 0).all())
        times.setdefault(f"B{B}_n{n}_margin{int(margin)}", []).append((time.monotonic() - t) * 1000)
    rep["rounds"] = a.rounds
    rep["serve_s"] = time.monotonic() - start
    rep["serving_memory"] = memory(dev)
    rep["latency_ms"] = {k: {"n": len(v), "p50": sorted(v)[len(v)//2], "p99": sorted(v)[int(len(v)*.99)]}
                         for k, v in times.items()}
    rep["rank0_output_checks"] = a.rank == 0
    phase(marker, "done")
    a.out.write_text(json.dumps(rep, indent=2) + "\n")
    print(f"DONE {a.arm} rank{a.rank}: {a.rounds} replays, {rep['serve_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
