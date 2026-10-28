"""Lane E timing: the TP = 2 draft step against the one-card worker, same card class, same inputs (O2's D5 bench).

Modes
  solo  (one card, GPU-7744b44b): the one-card model (model.py), the TP code unsplit on the card (tp = 1: separates
        what the second card adds from what the code adds), and each TP rank's shard alone (TPDrafter
        solo_rank 0 / 1: the rank's own kernels with the collectives replaced by local stand-ins of the same shapes).
        A rank's solo time + the measured collectives is the TP = 2 step; the slower rank sets it.
  pair  (the NVLink pair): the real TP = 2 worker, both ranks' graphs replayed per step (rank 0's CUDA events and host
        wall time), per-device kernel breakdown, NVLink collective microbenchmarks (p2p.cu inside CUDA graphs) and
        the NVLink data counters of both cards (nvidia-smi nvlink -gt d) around the run.
For every batch size 1-4: k = 5 at n_out 5 and 4 (CONTRACT 1.3; the live Spark sends 5 at 1-2 requests, 4 at 3-4),
graph replay p50 (p10-p90) over `iters` replays after warm-up, kernel time per category from CUPTI (torch profiler);
the context insert at 1-128 rows. Inputs: the parity text's tokens, synthetic taps (parity_ref.make_taps), full
128-position windows in slots 0-3 (O2's bench, so the one-card numbers are comparable with PERF.md).
Usage: python -m worker.tools.tp2_bench solo|pair ROOT OUT.json [iters]
"""

from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import time

import torch

from ..checkpoint import Checkpoint
from ..tp2 import TPDrafter, check_visible
from .bench import categorize, time_fn
from .parity_ref import _parity_text, make_taps

LASTS = (300, 500, 700, 900)
CASES = ((5, 5), (5, 4))
INSERT_ROWS = (1, 6, 12, 20, 64, 128)
CATS_EXTRA = (("p2p", ("ar_sum_kernel", "xchg_argmax", "xchg_margin")),)


def cat2(name: str) -> str:
    for c, keys in CATS_EXTRA:
        if any(k in name for k in keys):
            return c
    if "_argmax_finish_vi" in name:
        return "markov"
    return categorize(name)


def gpu_state(uuid: str) -> dict:
    q = "clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit,pstate"
    out = subprocess.run(["nvidia-smi", f"--id={uuid}", f"--query-gpu={q}", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True)
    return dict(zip(q.split(","), [v.strip() for v in out.stdout.strip().split(",")]))


def nvlink_kib(uuid: str) -> dict:
    out = subprocess.run(["nvidia-smi", "nvlink", "-gt", "d", "-i", uuid], capture_output=True, text=True).stdout
    tx = sum(int(l.split(":")[-1].split()[0]) for l in out.splitlines() if "Data Tx" in l)
    rx = sum(int(l.split(":")[-1].split()[0]) for l in out.splitlines() if "Data Rx" in l)
    return {"tx_kib": tx, "rx_kib": rx}


def breakdown(fn, reps: int = 20) -> dict:
    """Kernel time per step and device from CUPTI, by category (+ the top kernels)."""
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(reps):
            fn()
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)
    agg: dict = collections.defaultdict(collections.Counter)
    kern: dict = collections.defaultdict(collections.Counter)
    for e in p.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            dev = getattr(e, "device_index", 0)
            agg[dev][cat2(e.name)] += e.device_time_total
            kern[dev][e.name[:90]] += e.device_time_total
    out = {}
    for dev, a in sorted(agg.items()):
        tot = sum(a.values())
        out[f"device{dev}"] = {"kernel_us_per_step": round(tot / reps, 1),
                               "us": {k: round(v / reps, 1) for k, v in a.most_common()},
                               "top_kernels_us": {k: round(v / reps, 1) for k, v in kern[dev].most_common(10)}}
    return out


def setup_inputs(dev, root):
    from tokenizers import Tokenizer
    ids = Tokenizer.from_file(os.path.join(root, "tokenizer.json")).encode(_parity_text()).ids
    ids = (ids * (1 + 1200 // len(ids)))[:1200]
    return torch.tensor(ids, dtype=torch.int64, device=dev)


class _EmbedOnly:
    def __init__(self, embed):
        self.embed = embed


def fill_windows_tp(m: TPDrafter, taps_cpu: torch.Tensor) -> None:
    W = m.s.window
    for b, lp in enumerate(LASTS):
        rows = torch.arange(lp - W + 1, lp + 1)
        m.insert_eager(torch.full_like(rows, b), rows, taps_cpu[rows])


def host_inp(B: int, toks_cpu: torch.Tensor) -> torch.Tensor:
    last = torch.tensor(LASTS[:B])
    return torch.stack((torch.arange(B), toks_cpu[last + 1], last, (last - 127).clamp_min(0))).pin_memory()


def bench_tp(m: TPDrafter, toks_cpu: torch.Tensor, taps_cpu: torch.Tensor, iters: int, wall: bool) -> dict:
    rep: dict = {"draft": {}, "insert": {}}
    dev0 = m.devs[m.ALL[0]]
    with torch.cuda.device(dev0):
        for B in (1, 2, 3, 4):
            inp = host_inp(B, toks_cpu)
            for k, n in CASES:
                g = time_fn(lambda: m.draft_graphed(inp, k, n_out=n), iters)
                r = {"graph": g, "graph_breakdown": breakdown(lambda: m.draft_graphed(inp, k, n_out=n))}
                if wall:
                    ts = []
                    for _ in range(iters):
                        t0 = time.perf_counter()
                        m.draft_graphed(inp, k, n_out=n)
                        torch.cuda.synchronize(dev0)
                        ts.append((time.perf_counter() - t0) * 1e6)
                    ts.sort()
                    r["host_wall_p50_us"] = round(ts[len(ts) // 2], 1)
                rep["draft"][f"B{B}_k{k}_n{n}"] = r
                bd = r["graph_breakdown"]
                print(f"B{B} k{k} n_out{n}: graph p50 {g['p50_us']} us; " + "; ".join(
                    f"{d}: {v['kernel_us_per_step']} us ({', '.join(f'{c} {x}' for c, x in v['us'].items())})"
                    for d, v in bd.items()), flush=True)
        for nrow in INSERT_ROWS:
            sr = torch.full((nrow,), 7, dtype=torch.int64).pin_memory()
            pr = torch.arange(2000, 2000 + nrow).pin_memory()
            tt = taps_cpu[:nrow].clone().pin_memory()
            g = time_fn(lambda: m.insert_taps_graphed(sr, pr, tt), iters)
            rep["insert"][f"rows{nrow}"] = g
            print(f"insert {nrow} rows: {g['p50_us']} us", flush=True)
    return rep


def bench_one_card(root: str, ck: Checkpoint, dev, iters: int) -> dict:
    from ..model import DSparkDrafter
    m = DSparkDrafter(ck, dev, slots=8)
    toks = setup_inputs(dev, root)
    taps = make_taps(m, toks, torch.Generator(device=dev).manual_seed(5))
    W = m.s.window
    for b, lp in enumerate(LASTS):
        rows = torch.arange(lp - W + 1, lp + 1, device=dev)
        m.insert(torch.full_like(rows, b), rows, m.main_x(taps[rows]))
    m.capture(batch_sizes=(1, 2, 3, 4), ks=(5,), insert_rows=(8, 16, 32, 64, 128), n_outs=(4, 5), margins=False)
    rep: dict = {"draft": {}, "insert": {}}
    for B in (1, 2, 3, 4):
        slots = torch.arange(B, device=dev)
        last = torch.tensor(LASTS[:B], device=dev)
        anch = toks[last + 1]
        for k, n in CASES:
            g = time_fn(lambda: m.draft_graphed(slots, anch, last, k, n_out=n), iters)
            bd = breakdown(lambda: m.draft_graphed(slots, anch, last, k, n_out=n))
            rep["draft"][f"B{B}_k{k}_n{n}"] = {"graph": g, "graph_breakdown": bd}
            print(f"one-card B{B} k{k} n_out{n}: graph p50 {g['p50_us']} us, kernels "
                  f"{bd['device0']['kernel_us_per_step']} us", flush=True)
    for nrow in INSERT_ROWS:
        sr = torch.full((nrow,), 7, dtype=torch.int64, device=dev)
        pr = torch.arange(2000, 2000 + nrow, device=dev)
        tt = taps[:nrow].clone()
        rep["insert"][f"rows{nrow}"] = time_fn(lambda: m.insert_taps_graphed(sr, pr, tt), iters)
    rep["memory"] = m.memory_table()
    del m
    torch.cuda.empty_cache()
    return rep, taps.cpu(), toks.cpu()


def collective_bench(m: TPDrafter, iters: int) -> dict:
    """p2p.cu inside CUDA graphs on both ranks: 20 back-to-back collectives per replay (rank 1 replayed first)."""
    pair = m.comm.pair
    out = {}
    H = m.s.hidden
    for kind, sizes in (("ar_sum_bf16", (5, 10, 15, 20)), ("xchg_argmax", (1, 2, 3, 4))):
        for sz in sizes:
            bufs, graphs = {}, {}
            for r in (0, 1):
                d = m.devs[r]
                with torch.cuda.device(d):
                    if kind == "ar_sum_bf16":
                        x = torch.randn(sz * H, device=d)
                        y = torch.empty(sz * H, dtype=torch.bfloat16, device=d)
                        nb = m.comm.nblocks(sz * H)
                        fn = (lambda r=r, x=x, y=y, nb=nb: pair.ar_sum(r, x, y, nb))
                    else:
                        v = torch.randn(sz, device=d)
                        i = torch.randint(0, 60000, (sz,), device=d)
                        o = torch.zeros(sz, 5, dtype=torch.int64, device=d)
                        fn = (lambda r=r, v=v, i=i, o=o: pair.xchg_argmax(r, v, i, m.ranks[r].v0, o, 0))
                    bufs[r] = fn
            # warm-up lockstep, then capture each rank alone
            for _ in range(3):
                for r in (0, 1):
                    with torch.cuda.device(m.devs[r]):
                        bufs[r]()
            for r in (0, 1):
                torch.cuda.synchronize(m.devs[r])
            for r in (0, 1):
                with torch.cuda.device(m.devs[r]):
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, stream=torch.cuda.Stream(m.devs[r])):
                        for _ in range(20):
                            bufs[r]()
                    graphs[r] = g
            ts = []
            for it in range(iters // 4 + 5):
                with torch.cuda.device(m.devs[0]):
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                with torch.cuda.device(m.devs[1]):
                    graphs[1].replay()
                with torch.cuda.device(m.devs[0]):
                    graphs[0].replay()
                    b.record()
                b.synchronize()
                torch.cuda.synchronize(m.devs[1])
                if it >= 5:
                    ts.append(a.elapsed_time(b) * 1000.0 / 20)
            ts.sort()
            out[f"{kind}_{sz}"] = {"p50_us": round(ts[len(ts) // 2], 2), "p10_us": round(ts[len(ts) // 10], 2),
                                   "p90_us": round(ts[int(len(ts) * 0.9)], 2),
                                   "bytes_each_way": (sz * H * 4) if kind == "ar_sum_bf16" else sz * 16}
            print(f"{kind} {sz}: {out[f'{kind}_{sz}']['p50_us']} us", flush=True)
    m.comm.check()
    return out


def main() -> None:
    mode = sys.argv[1]
    root = sys.argv[2] if len(sys.argv) > 2 else "/opt/dsv41/x86host-data/dspark"
    out_path = sys.argv[3] if len(sys.argv) > 3 else f"e_bench_{mode}.json"
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    vis = check_visible(pair=(mode == "pair"))
    uuids = vis.split(",")
    t0 = time.time()
    ck = Checkpoint(root)
    rep: dict = {"mode": mode, "devices": uuids, "iters": iters,
                 "gpu_before": {u: gpu_state(u) for u in uuids}}
    if mode == "solo":
        dev = torch.device("cuda", 0)
        one, taps_cpu, toks_cpu = bench_one_card(root, ck, dev, iters)
        rep["one_card"] = one
        for r in ("tp1", 0, 1):
            if r == "tp1":                     # the same code unsplit on this card: what the second card adds
                m = TPDrafter(ck, [dev], slots=8, tp=1)
            else:
                m = TPDrafter(ck, [dev, dev], slots=8, solo_rank=r)
            fill_windows_tp(m, taps_cpu)
            m.capture(batch_sizes=(1, 2, 3, 4), ks=(5,), insert_rows=(8, 16, 32, 64, 128), n_outs=(4, 5),
                      margins=False)
            key = "e_tp1_one_card" if r == "tp1" else f"solo_rank{r}"
            print(f"--- {key}", flush=True)
            rep[key] = bench_tp(m, toks_cpu, taps_cpu, iters, wall=False)
            rep[key]["memory"] = m.memory_table()
            del m
            torch.cuda.empty_cache()
    elif mode == "pair":
        devs = [torch.device("cuda", 0), torch.device("cuda", 1)]
        nv0 = {u: nvlink_kib(u) for u in uuids}
        m = TPDrafter(ck, devs, slots=8)
        toks = setup_inputs(devs[0], root)
        taps = make_taps(_EmbedOnly(m.ranks[0].embed), toks, torch.Generator(device=devs[0]).manual_seed(5))
        taps_cpu, toks_cpu = taps.cpu(), toks.cpu()
        fill_windows_tp(m, taps_cpu)
        rep["capture_seconds"] = round(m.capture(), 2)
        rep["tp2"] = bench_tp(m, toks_cpu, taps_cpu, iters, wall=True)
        rep["collectives"] = collective_bench(m, iters)
        rep["memory"] = m.memory_table()
        rep["memory_nvidia_smi_mib"] = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True).stdout
        m.comm.check()
        rep["comm_headers"] = m.comm.headers()
        nv1 = {u: nvlink_kib(u) for u in uuids}
        rep["nvlink_kib_during_run"] = {u: {k: nv1[u][k] - nv0[u][k] for k in nv0[u]} for u in uuids}
    else:
        raise SystemExit("mode: solo | pair")
    rep["gpu_after"] = {u: gpu_state(u) for u in uuids}
    rep["seconds"] = round(time.time() - t0, 1)
    with open(out_path, "w") as f:
        json.dump(rep, f, indent=1)
    print("WROTE", out_path, rep["seconds"], "s")


if __name__ == "__main__":
    main()
