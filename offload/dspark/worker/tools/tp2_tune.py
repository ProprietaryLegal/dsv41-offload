"""Kernel plan for the TP = 2 worker's EXL3 matrices (tp2_kernels.json): every exl3_gemm launch of tp2.py runs with a
forced (shape, num_sms) so both cards run the same kernel on replicated matrices (bit-identical activations) and no
timing-based autotune runs at start-up or inside a graph capture.

For every matrix the TP = 2 model uses (both ranks' shards; replicated matrices once; and the unsplit tp = 1 matrices
of the one-card comparison) and every row count it runs at
(draft body 5 / 10 / 15 / 20 rows, head fixed for 1-16 rows, context insert 8-128 rows), time each sm_86 tile shape
(exllamav3 exl3_kernel_map.cuh) that divides the matrix, at a ladder of grid sizes, as CUDA-graph replays of 20
back-to-back GEMMs that cycle through enough distinct copies of the weights (>= 48 MiB) that no call finds its matrix
in the 6 MiB L2; keep the fastest. The default (shape 0 = exllamav3's static heuristic, all SMs) is timed too.
Usage: python -m worker.tools.tp2_tune ROOT OUT.json        (one GPU visible)
"""

from __future__ import annotations

import json
import sys
import time

import torch

from ..checkpoint import Checkpoint
from ..tp2 import KernelPlan, TPDrafter, check_visible

SHAPES = {1: (16, 16, 128), 2: (16, 32, 128), 3: (16, 32, 256), 4: (16, 16, 512), 5: (32, 32, 128), 6: (48, 32, 128),
          7: (64, 32, 128)}
NSMS = (82, 80, 76, 72, 68, 64, 56, 48, 41, 32, 24, 16)
BODY_M = (5, 10, 15, 20)
INSERT_M = (8, 16, 32, 64, 128)


def time_graph(fn, reps: int = 5) -> float:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    best = 1e30
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        best = min(best, a.elapsed_time(b) * 1000.0)
    del g
    return best


def tune_matrix(mat, ms, bucket_of, calls: int = 20) -> dict:
    nbytes = mat.nbytes()
    copies = max(1, -(-(48 << 20) // nbytes))
    trs = [mat.trellis.clone() for _ in range(copies)]
    out = {}
    for m in ms:
        x = torch.randn(m, mat.in_features, device=mat.trellis.device).to(torch.float16)
        y = torch.empty(m, mat.out_features, dtype=torch.float32, device=x.device)
        xh = torch.empty_like(x)
        k, n = mat.in_features, mat.out_features
        cands = [(0, 82)]
        for sh, (tm, tk, tn) in SHAPES.items():
            if k % tk or n % tn or (tm > 16 and m <= tm // 2):
                continue
            slices = (k // tk) * (n // tn)
            for ns in sorted({min(v, slices) for v in NSMS}, reverse=True):
                cands.append((sh, ns))
        res = []
        for sh, ns in cands:
            def fn(sh=sh, ns=ns):
                for c in range(calls):
                    mat.ext.exl3_gemm(x, trs[c % copies], y, mat.suh, xh, mat.svh, sh, False, True, ns)
            try:
                us = time_graph(fn) / calls
            except RuntimeError as e:          # a shape the kernel refuses (shared memory): skip, recorded
                res.append({"shape": sh, "num_sms": ns, "error": str(e)[:120]})
                continue
            res.append({"shape": sh, "num_sms": ns, "us": round(us, 2)})
        ok = [r for r in res if "us" in r]
        best = min(ok, key=lambda r: r["us"])
        default = next(r for r in ok if r["shape"] == 0)
        out[bucket_of(m)] = {"best": [best["shape"], best["num_sms"]], "us": best["us"], "default_us": default["us"],
                             "candidates": len(ok)}
        print(f"  m={m:3d} -> {bucket_of(m)}: shape {best['shape']} sms {best['num_sms']} {best['us']} us "
              f"(default {default['us']} us, {len(ok)} candidates)", flush=True)
    del trs
    torch.cuda.empty_cache()
    return out


def main() -> None:
    check_visible(pair=False)
    root = sys.argv[1] if len(sys.argv) > 1 else "/opt/dsv41/x86host-data/dspark"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "tp2_kernels.json"
    dev = torch.device("cuda", 0)
    ck = Checkpoint(root)
    t0 = time.time()
    plan: dict = {}
    seen = set()
    for tp, r in ((2, 0), (2, 1), (1, 0)):
        if tp == 2:
            m = TPDrafter(ck, [dev, dev], slots=1, solo_rank=r, kernel_plan=None)
        else:                                  # tp = 1: the same code unsplit (one-card comparison)
            m = TPDrafter(ck, [dev], slots=1, kernel_plan=None, tp=1)
        rk = m.ranks[r]
        mats = [(rk.head, ("head",), lambda _m: "head")]
        mats.append((rk.main_proj, INSERT_M, str))
        st = rk.stages[0]                      # stages share shapes: tune stage 0's matrices
        mats.append((st.wkv, INSERT_M + BODY_M, str))
        for mt in (st.wq_a, st.wq_b, st.wo_b, st.sh_w1, st.sh_w3, st.sh_w2):
            mats.append((mt, BODY_M, str))
        for mt, ms, bucket_of in mats:
            key = KernelPlan.key(mt.in_features, mt.out_features, mt.k)
            if key in seen:
                continue
            seen.add(key)
            print(f"{key} ({mt.name})", flush=True)
            if ms == ("head",):
                plan[key] = tune_matrix(mt, (5,), bucket_of)
            else:
                plan[key] = tune_matrix(mt, ms, bucket_of)
        del m, rk, st, mats
        torch.cuda.empty_cache()
    kernels = {k: {b: v["best"] for b, v in d.items()} for k, d in plan.items()}
    with open(out_path, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(dev), "seconds": round(time.time() - t0, 1),
                   "note": "tools/tp2_tune.py: [shape, num_sms] per matrix key (in x out x K) and row bucket",
                   "kernels": kernels, "detail": plan}, f, indent=1)
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
