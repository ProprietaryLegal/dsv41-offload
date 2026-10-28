"""D5: draft-step latency of the 3090 worker, and where the time goes.

For B = 1..4 requests: the draft step with k = 5 (the served block) at n_out 5 and 4 (CONTRACT 1.3: the Spark sends 4
at 3-4 requests), and k = 4 for reference, through its CUDA graph and eagerly (CUDA-event wall time over `iters`
replays after warm-up), and a per-kernel GPU-time breakdown of the graph replay (torch profiler / CUPTI) grouped
into: experts (exl3_moe_coop + shared expert's silu), head (EXL3 K6 GEMM), dense (EXL3 K5 GEMMs), wo_a, markov
(bias + argmax), mhc (mixes / collapse / post), attention, norm/rope/kv, routing, other. Also the context insert
(main_proj + window KV) for 1..128 rows through its graphs. Inputs: the parity text's tokens (real English; the
Markov inputs of this text include uncached ids, as about 20 % of live ones are), synthetic taps, full 128-position
windows. GPU clocks / temperature are recorded before and after (nvidia-smi).
Usage: python -m worker.tools.bench ROOT OUT.json [iters]
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
from ..model import DSparkDrafter
from .parity_ref import _parity_text, make_taps

CATEGORIES = (
    ("experts", ("exl3_moe_coop", "silu_kernel", "clamp_scalar")),
    ("head_k6", ("exl3_gemm_kernel<6",)),
    ("dense_k5", ("exl3_gemm_kernel<5", "exl3_gemv")),
    ("wo_a", ("_wo_a_fp8", "s16816gemm_bf16", "bf16_s1688gemm", "cutlass")),
    ("markov", ("_markov", "ArgMax", "gather_kernel")),
    ("mhc", ("_hc_mix", "_collapse", "_hc_post")),
    ("norm_rope_kv", ("_rms_norm", "_rope", "_kv_store", "_scaled_rms_norm", "_prescale", "_amax")),
    ("attention", ("_block_attn", "sgemm", "softmax", "exp_kernel", "max_kernel")),
    ("routing", ("topk", "TopK", "sort", "softplus", "_route")),
)


def categorize(name: str) -> str:
    for cat, keys in CATEGORIES:
        if any(k in name for k in keys):
            return cat
    return "other"


def gpu_state() -> dict:
    q = "clocks.sm,clocks.mem,clocks.max.sm,temperature.gpu,power.draw,pstate"
    out = subprocess.run(["nvidia-smi", f"--id={os.environ.get('CUDA_VISIBLE_DEVICES', '0')}",
                          f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True, check=True)
    return dict(zip(q.split(","), [v.strip() for v in out.stdout.strip().split(",")]))


def time_fn(fn, iters: int) -> dict:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0)
    ts.sort()
    return {"p50_us": round(ts[len(ts) // 2], 1), "p10_us": round(ts[len(ts) // 10], 1),
            "p90_us": round(ts[int(len(ts) * 0.9)], 1), "mean_us": round(sum(ts) / len(ts), 1)}


def breakdown(fn, reps: int = 20) -> dict:
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    agg: collections.Counter = collections.Counter()
    kern: collections.Counter = collections.Counter()
    n = 0
    for e in p.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            agg[categorize(e.name)] += e.device_time_total
            kern[e.name[:90]] += e.device_time_total
            n += 1
    total = sum(agg.values())
    return {"kernel_us_per_step": round(total / reps, 1), "kernels_per_step": n / reps,
            "share": {k: round(v / total, 4) for k, v in agg.most_common()},
            "us": {k: round(v / reps, 1) for k, v in agg.most_common()},
            "top_kernels_us": {k: round(v / reps, 1) for k, v in kern.most_common(12)}}


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "/opt/dsv41/x86host-data/dspark"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "d5_bench.json"
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    dev = torch.device("cuda", 0)
    ck = Checkpoint(root)
    m = DSparkDrafter(ck, dev, slots=8)
    from tokenizers import Tokenizer
    ids = Tokenizer.from_file(os.path.join(root, "tokenizer.json")).encode(_parity_text()).ids
    ids = (ids * (1 + 1200 // len(ids)))[:1200]
    toks = torch.tensor(ids, dtype=torch.int64, device=dev)
    taps = make_taps(m, toks, torch.Generator(device=dev).manual_seed(5))
    W = m.s.window
    lasts = [300, 500, 700, 900]
    for b, lp in enumerate(lasts):
        rows = torch.arange(lp - W + 1, lp + 1, device=dev)
        m.insert(torch.full_like(rows, b), rows, m.main_x(taps[rows]))
    t0 = time.time()
    cap_s = m.capture(batch_sizes=(1, 2, 3, 4), ks=(4, 5), insert_rows=(8, 16, 32, 64, 128), n_outs=(4, 5))
    rep: dict = {"device": torch.cuda.get_device_name(0), "capture_seconds": round(cap_s, 2), "iters": iters,
                 "gpu_before": gpu_state(), "draft": {}, "insert": {}}
    for B in (1, 2, 3, 4):
        slots = torch.arange(B, device=dev)
        last = torch.tensor(lasts[:B], device=dev)
        anch = toks[last + 1]
        for k, n in ((5, 5), (5, 4), (4, 4)):
            g = time_fn(lambda: m.draft_graphed(slots, anch, last, k, n_out=n), iters)
            e = time_fn(lambda: m.draft(slots, anch, last, k, n_out=n), max(20, iters // 10))
            bd = breakdown(lambda: m.draft_graphed(slots, anch, last, k, n_out=n))
            rep["draft"][f"B{B}_k{k}_n{n}"] = {"graph": g, "eager": e, "graph_breakdown": bd}
            print(f"B{B} k{k} n_out{n}: graph p50 {g['p50_us']} us, eager p50 {e['p50_us']} us, kernels "
                  f"{bd['kernel_us_per_step']} us: " + ", ".join(f"{c} {v}" for c, v in bd["us"].items()),
                  flush=True)
    for n in (1, 6, 12, 20, 24, 64, 128):
        sr = torch.full((n,), 7, dtype=torch.int64, device=dev)
        pr = torch.arange(2000, 2000 + n, device=dev)
        tt = taps[:n].clone()
        g = time_fn(lambda: m.insert_taps_graphed(sr, pr, tt), iters)
        rep["insert"][f"rows{n}"] = g
        print(f"insert {n} rows: {g['p50_us']} us", flush=True)
    rep["gpu_after"] = gpu_state()
    rep["memory"] = m.memory_table()
    with open(out_path, "w") as f:
        json.dump(rep, f, indent=1)
    print("WROTE", out_path, "total", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
