"""N1: memory (serving + load peak) and draft-step time of an arm, per card.

  solo  one rank of the arm alone on GPU-7744b44b (TPDrafter solo_rank: the rank's own tensors and kernels, the
        collectives replaced by local stand-ins of the same shapes): that card's real serving footprint and load peak,
        and the rank's compute time. Run rank 0 and rank 1 (their shards differ only in the vocabulary / head / group
        columns they own).
  pair  both ranks on the NVLink pair (PAIR window only): real collectives, both cards' footprints.
Per arm: load seconds, torch peak allocated / reserved during the load and after graph capture, nvidia-smi process
memory, GPU clocks / power before and after, graph replay p50 for B = 1-4 at k = 5, n_out 5 and 4, insert 8-128
rows. Same inputs as lane E's tp2_bench (O2's D5 bench), so the K2 control numbers are comparable.
Usage (x86host): python -m offload.dspark.native.tools.native_bench ARM solo|pair OUT.json [RANK] [ITERS]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import torch

from ...worker.checkpoint import Checkpoint
from ...worker.tools.bench import time_fn
from ...worker.tools.parity_ref import make_taps
from ...worker.tools.tp2_bench import _EmbedOnly, breakdown, fill_windows_tp, gpu_state, host_inp, setup_inputs
from ...worker.tp2 import check_visible
from ..tp2_native import ARMS, build

ROOT = "/opt/dsv41/x86host-data/dspark"


def smi_process_mib() -> list:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    me = str(os.getpid())
    return [l.strip() for l in out.splitlines() if l.strip().startswith(me + ",")]


def main() -> None:
    arm, mode, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    rank = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    iters = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    if arm not in ARMS:
        raise SystemExit(f"arm {arm!r} not in {sorted(ARMS)}")
    vis = check_visible(pair=(mode == "pair"))
    uuids = vis.split(",")
    t0 = time.time()
    rep: dict = {"arm": arm, "cfg": ARMS[arm].__dict__, "mode": mode, "rank": rank if mode == "solo" else None,
                 "devices": uuids, "iters": iters, "gpu_before": {u: gpu_state(u) for u in uuids}}
    ck = Checkpoint(ROOT)
    d0 = torch.device("cuda", 0)
    devs = [d0, torch.device("cuda", 1)] if mode == "pair" else [d0, d0]
    for d in set(devs):
        torch.empty(1, device=d)                    # the CUDA context and allocator exist before the counters reset
        torch.cuda.reset_peak_memory_stats(d)
    tl0 = time.time()
    kw = {"slots": 8}
    if mode == "solo":
        kw["solo_rank"] = rank
    m = build(arm, devs, ck, **kw)
    rep["load_seconds"] = round(time.time() - tl0, 1)
    rep["load_peak"] = {str(d): {"max_allocated": torch.cuda.max_memory_allocated(d),
                                 "max_reserved": torch.cuda.max_memory_reserved(d),
                                 "allocated": torch.cuda.memory_allocated(d)} for d in sorted(set(devs), key=str)}
    rep["load_nvidia_smi"] = smi_process_mib()
    if getattr(m, "load_stats", None) is not None:
        rep["load_stats"] = m.load_stats.__dict__
    r0 = m.ALL[0]
    toks = setup_inputs(devs[r0], ROOT)
    taps = make_taps(_EmbedOnly(m.ranks[r0].embed), toks, torch.Generator(device=devs[r0]).manual_seed(5))
    taps_cpu, toks_cpu = taps.cpu(), toks.cpu()
    fill_windows_tp(m, taps_cpu)
    rep["capture_seconds"] = round(m.capture(batch_sizes=(1, 2, 3, 4), ks=(5,), insert_rows=(8, 16, 32, 64, 128),
                                             n_outs=(4, 5), margins=False), 2)
    rep["draft"] = {}
    rep["insert"] = {}
    with torch.cuda.device(devs[r0]):
        for B in (1, 2, 3, 4):
            inp = host_inp(B, toks_cpu)
            for n in (5, 4):
                g = time_fn(lambda: m.draft_graphed(inp, 5, n_out=n), iters)
                rep["draft"][f"B{B}_k5_n{n}"] = g
                print(f"{arm} B{B} n_out{n}: p50 {g['p50_us']} us", flush=True)
                if n == 5 and B in (1, 4):
                    bd = breakdown(lambda: m.draft_graphed(inp, 5, n_out=n))
                    rep["draft"][f"B{B}_k5_n{n}"]["breakdown"] = bd
                    print(json.dumps(bd), flush=True)
        for nrow in (8, 32, 128):
            sr = torch.full((nrow,), 7, dtype=torch.int64).pin_memory()
            pr = torch.arange(2000, 2000 + nrow).pin_memory()
            tt = taps_cpu[:nrow].clone().pin_memory()
            rep["insert"][f"rows{nrow}"] = time_fn(lambda: m.insert_taps_graphed(sr, pr, tt), iters)
    rep["memory"] = m.memory_table()
    rep["serving_nvidia_smi"] = smi_process_mib()
    rep["serving_peak"] = {str(d): {"max_allocated": torch.cuda.max_memory_allocated(d),
                                    "max_reserved": torch.cuda.max_memory_reserved(d)}
                           for d in sorted(set(devs), key=str)}
    rep["gpu_after"] = {u: gpu_state(u) for u in uuids}
    rep["seconds"] = round(time.time() - t0, 1)
    with open(out_path, "w") as f:
        json.dump(rep, f, indent=1, default=str)
    print(json.dumps({k: rep[k] for k in ("arm", "rank", "load_seconds", "load_peak", "load_nvidia_smi",
                                          "serving_nvidia_smi", "capture_seconds")}, indent=1, default=str))
    print("WROTE", out_path, rep["seconds"], "s")


if __name__ == "__main__":
    main()
