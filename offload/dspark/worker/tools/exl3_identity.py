"""D1: the drafter's EXL3 tensors decode and run on SM86 (exllamav3 1.6.0) identically to a CPU reference.

Checks, on the exact checkpoint tensors (identity-verified by checkpoint.Checkpoint):
  1. trellis decode: exllamav3_ext.reconstruct on the GPU vs the plain-torch CPU decoder (exl3/exl3_ref.py, written from
     exllamav3's CUDA and checked bit-exact against it by the plugin's authors) -> must be bit-identical (fp16 bits).
  2. Hadamard: had_r_128 on the GPU vs a CPU fp64 128-block Hadamard -> fp16-rounding level.
  3. linear: LinearEXL3.forward (exl3_gemm path at <= 144 rows, reconstruct + hgemm path above) vs the CPU fp64 linear
     y = H(H(x * suh) @ W) * svh on the same fp16 input -> relative error at fp16 level, reported per tensor.
  4. routed experts: exllamav3's exl3_moe_coop (the kernel the Spark uses when exl3_smoe is off) vs a CPU fp64
     SwiGLU-with-limit reference on the same experts and routing weights.
Writes a JSON report (argv[2], default offload/results/o2/d1_exl3_identity.json).
Usage: CUDA_VISIBLE_DEVICES=GPU-c4e7a371-... python -m worker.tools.exl3_identity /opt/dsv41/x86host-data/dspark out.json
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import torch

from ..checkpoint import Checkpoint
from ..exl3 import exl3_ref
from ..exl3 import kernels as K


def cpu_decode(trellis: torch.Tensor, k: int, col_tiles_per_slice: int = 512) -> torch.Tensor:
    """exl3_ref.decode in column slices (whole 16-wide tiles), so the int64 temporaries stay small."""
    kt, nt, _ = trellis.shape
    out = torch.empty(kt * 16, nt * 16, dtype=torch.float16)
    for a in range(0, nt, col_tiles_per_slice):
        b = min(nt, a + col_tiles_per_slice)
        out[:, a * 16:b * 16] = exl3_ref.decode(trellis[:, a:b, :].contiguous(), k)
    return out


def had_cpu(x: torch.Tensor) -> torch.Tensor:
    H = exl3_ref.hadamard128(dtype=torch.float64)
    m, n = x.shape
    return (x.double().view(m, n // 128, 128) @ H).view(m, n)


def linear_ref64(x16: torch.Tensor, w: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor) -> torch.Tensor:
    """fp64 y = H(H(x * suh) @ W) * svh; w is the decoded rotated-basis weight (fp16 or already fp64)."""
    xh = had_cpu(x16.double() * suh.double())
    return had_cpu(xh @ w.double()) * svh.double()


def rel(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.double().cpu(), b.double().cpu()
    d = (a - b)
    return {"rel_fro": float(d.norm() / b.norm().clamp_min(1e-30)),
            "max_abs": float(d.abs().max()), "ref_max_abs": float(b.abs().max())}


def check_linear(ck: Checkpoint, name: str, dev: torch.device, gen: torch.Generator, decode_all: bool) -> dict:
    from exllamav3.modules.quant.exl3 import LinearEXL3

    trellis, suh, svh, k = ck.exl3(name)
    in_f, out_f = trellis.shape[0] * 16, trellis.shape[1] * 16
    r: dict = {"name": name, "K": k, "in": in_f, "out": out_f}
    t0 = time.time()
    # 1. decode, bit for bit (all columns, or the first 4096 for the 129,280-wide head when decode_all is False)
    tr_d = trellis.to(dev)
    w_gpu = torch.empty(in_f, out_f, dtype=torch.float16, device=dev)
    K.ext().reconstruct(w_gpu, tr_d, k, False, True)
    if decode_all or out_f <= 32768:
        w_cpu = cpu_decode(trellis, k)
        w_cmp = w_gpu.cpu()
    else:
        nt = 4096 // 16
        w_cpu = cpu_decode(trellis[:, :nt, :].contiguous(), k)
        w_cmp = w_gpu[:, :nt * 16].cpu()
    same = torch.equal(w_cpu.view(torch.int16), w_cmp.view(torch.int16))
    r["decode_bit_identical"] = bool(same)
    r["decode_columns_checked"] = int(w_cpu.shape[1])
    if not same:
        r["decode_mismatches"] = int((w_cpu.view(torch.int16) != w_cmp.view(torch.int16)).sum())
    # 2./3. linear on both exllamav3 paths, fp16 input, fp32 output
    lin = LinearEXL3(None, in_f, out_f, suh=suh.to(dev).contiguous(), svh=svh.to(dev).contiguous(), trellis=tr_d,
                     mul1=torch.tensor(0x83DCD12D - (1 << 32), dtype=torch.int32, device=dev))
    w_full = (w_cpu if w_cpu.shape[1] == out_f else w_gpu.cpu()).double()   # decode bit-identity checked above
    for rows in (1, 5, 20, 160):
        x = (torch.randn(rows, in_f, generator=gen) * 0.5).to(torch.float16)
        y = lin.forward(x.to(dev).contiguous(), {}, out_dtype=torch.float32)
        ref = linear_ref64(x, w_full, suh, svh)
        r[f"linear_rows{rows}"] = rel(y, ref)
        # the GPU input Hadamard alone (fp16 out) vs fp64
        if rows == 5:
            xh = torch.empty_like(x.to(dev))
            K.ext().had_r_128(x.to(dev), xh, suh.to(dev), None, 1.0)
            r["had_r_128_rows5"] = rel(xh.float(), had_cpu(x.double() * suh.double()))
    r["seconds"] = round(time.time() - t0, 2)
    return r


def check_experts(ck: Checkpoint, stage: int, dev: torch.device, gen: torch.Generator) -> dict:
    """exl3_moe_coop over a handful of experts of one drafter layer vs CPU fp64."""
    from ..exl3.bank import Exl3Bank
    from ..exl3.plan import Exl3MoEPlan, PROJ_DOWN, PROJ_GATE, PROJ_UP

    s = ck.shape
    plan = Exl3MoEPlan.from_dict(ck.plan_json)
    L = s.n_target_layers + stage
    lp = plan.drafter_layers[L]
    bank = Exl3Bank(lp, s.hidden, s.inter, 1, 0)
    bank.allocate(dev)
    names = {PROJ_GATE: "w1", PROJ_UP: "w3", PROJ_DOWN: "w2"}
    for e in range(s.n_experts):
        for proj, w in names.items():
            tr, suh, svh, k = ck.exl3(f"mtp.{stage}.ffn.experts.{e}.{w}")
            bank.load_piece("trellis", proj, e, tr)
            bank.load_piece("suh", proj, e, suh)
            bank.load_piece("svh", proj, e, svh)
    groups, _ = bank.tables()
    rows = 5
    sc = K.get_coop_scratch(dev, 32, s.top_k, s.hidden, s.inter)
    x = (torch.randn(rows, s.hidden, generator=gen) * 0.3).to(torch.float16)
    ids = torch.stack([torch.randperm(s.n_experts, generator=gen)[: s.top_k] for _ in range(rows)])
    w = torch.rand(rows, s.top_k, generator=gen) + 0.2
    w = (w / w.sum(-1, keepdim=True) * s.route_scale).to(torch.float16)
    perm = torch.tensor(lp.perm, dtype=torch.long)
    sel = perm[ids].to(dev)
    K.prime(dev, groups, sc, None, s.hidden, s.top_k, s.swiglu_limit, False, True)
    out = K.run_decode_coop(x.to(dev), sel, w.to(dev), groups, sc, s.swiglu_limit, False, True).clone()
    ref = torch.zeros(rows, s.hidden, dtype=torch.float64)
    cache: dict = {}
    for t in range(rows):
        for j in range(s.top_k):
            e = int(ids[t, j])
            if e not in cache:
                ws = {}
                for wn in ("w1", "w3", "w2"):
                    tr, suh, svh, k = ck.exl3(f"mtp.{stage}.ffn.experts.{e}.{wn}")
                    ws[wn] = (cpu_decode(tr, k), suh, svh)
                cache[e] = ws
            ws = cache[e]
            xt = x[t:t + 1]
            g = linear_ref64(xt, *ws["w1"])
            u = linear_ref64(xt, *ws["w3"])
            lim = s.swiglu_limit
            a = torch.nn.functional.silu(g.clamp(max=lim)) * u.clamp(-lim, lim)
            # the coop kernel feeds the down projection an fp16 activation (act_out is fp16 scratch)
            y = linear_ref64(a.to(torch.float16), *ws["w2"])
            ref[t] += float(w[t, j]) * y[0]
    return {"stage": stage, "K_groups": [(g.k_gu, g.k_d, g.lo, g.hi) for g in groups], "rows": rows,
            "coop_vs_fp64": rel(out[:rows], ref)}


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "/opt/dsv41/x86host-data/dspark"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "offload/results/o2/d1_exl3_identity.json"
    dev = torch.device("cuda", 0)
    t0 = time.time()
    ck = Checkpoint(root)
    report: dict = {"device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": torch.__version__,
                    "verify_seconds": round(time.time() - t0, 1), "tensors_verified": len(ck.entries)}
    import exllamav3
    report["exllamav3"] = exllamav3.__version__ if hasattr(exllamav3, "__version__") else "1.6.0"
    gen = torch.Generator().manual_seed(1017)
    names = ["mtp.0.attn.wkv", "mtp.0.attn.wq_a", "mtp.0.attn.wq_b", "mtp.0.attn.wo_b", "mtp.0.main_proj",
             "mtp.1.ffn.shared_experts.w1", "mtp.1.ffn.shared_experts.w2", "mtp.2.ffn.shared_experts.w3",
             "mtp.2.attn.wq_b", "mtp.0.ffn.experts.0.w1", "mtp.1.ffn.experts.77.w2", "mtp.2.ffn.experts.127.w3",
             "head"]
    report["linears"] = []
    for n in names:
        r = check_linear(ck, n, dev, gen, decode_all=(n != "head"))
        print(json.dumps(r), flush=True)
        report["linears"].append(r)
    report["experts"] = [check_experts(ck, st, dev, gen) for st in (0, 2)]
    print(json.dumps(report["experts"]), flush=True)
    report["all_decode_bit_identical"] = all(r["decode_bit_identical"] for r in report["linears"])
    report["max_linear_rel_fro"] = max(v["rel_fro"] for r in report["linears"] for k_, v in r.items()
                                       if k_.startswith("linear_rows"))
    report["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1)
    print("WROTE", out_path, "all_decode_bit_identical", report["all_decode_bit_identical"],
          "max_linear_rel_fro", f"{report['max_linear_rel_fro']:.3e}")
    if not report["all_decode_bit_identical"] or not math.isfinite(report["max_linear_rel_fro"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
