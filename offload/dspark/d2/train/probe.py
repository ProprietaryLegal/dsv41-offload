"""Full-weight synthetic backward/memory screen, with N1 benchmark taps.

No captured/test sequence is ever trained on by this probe. Weights/checkpoints
are not exported. N1 window fixtures stay local for independent CPU parity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Callable, cast

import torch

from .loss import distillation_loss
from .model import load
from .replay import dump_sample
from .safety import GPU, assert_gpu_lease


def main() -> None:
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n1-repo", required=True)
    ap.add_argument("--worker-root", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--reference-math", action="store_true")
    a = ap.parse_args()
    assert_gpu_lease()
    if a.batch < 1 or a.steps < 2:
        ap.error("batch must be positive and steps >= 2")
    telemetry = subprocess.run(["nvidia-smi", "--id="+GPU,
        "--query-gpu=uuid,power.limit,pcie.link.gen.max,pcie.link.width.max", "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout.strip()
    sys.path.append(a.n1_repo)
    from offload.dspark.worker.tools.parity_ref import make_taps
    from offload.dspark.worker.tools.tp2_bench import LASTS, setup_inputs
    torch.set_num_threads(8)
    torch.cuda.reset_peak_memory_stats()
    begin = time.monotonic()
    model = load(a.ckpt, "cuda", reference_math=a.reference_math)
    load_seconds = time.monotonic()-begin
    load_peak = torch.cuda.max_memory_allocated()
    toks = cast(Callable[[torch.device, str], torch.Tensor], setup_inputs)(torch.device("cuda"), a.worker_root)
    wrapper = type("Embedding", (), {"embed": model.t["embed.weight"]})()
    taps = make_taps(wrapper, toks, torch.Generator(device="cuda").manual_seed(5)).cpu()
    toks = toks.cpu()
    positions = torch.stack([torch.arange(LASTS[b%len(LASTS)]-127, LASTS[b%len(LASTS)]+1) for b in range(a.batch)])
    P = positions[:, -1]
    act_dev = model.activation_device
    inp = {"taps": taps[positions].to(act_dev), "engine_mx": torch.zeros(a.batch, 128, 5120, device=act_dev),
           "has_taps": torch.ones(a.batch, 128, dtype=torch.bool, device=act_dev), "positions": positions.to(act_dev),
           "mask": torch.ones(a.batch, 128, dtype=torch.bool, device=act_dev), "P": P.to(act_dev),
           "anchor": toks[P+1].to(act_dev)}
    a.out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    model.eval()
    with torch.no_grad():
        d = dump_sample(model, inp, {}, {"arm": "D2-N1-synthetic"}) if a.batch == 1 else None
        if d is not None:
            torch.save({"inputs": {k: v.cpu() for k, v in inp.items()}, "dump": d}, a.out.with_suffix(".fixture.pt"))
    model.train()
    params = [p for n, p in model.named_parameters() if "confidence_head" not in n]
    optimizer = torch.optim.AdamW(params, lr=2e-4, foreach=False)
    previous = torch.stack([toks[p+1:p+6] for p in P.tolist()]).to(act_dev)
    ids = torch.arange(20, device=act_dev)[None, None].expand(a.batch, 5, -1)
    lp = torch.full((a.batch, 5, 20), .04, device=act_dev).log()
    mask = torch.ones(a.batch, 5, dtype=torch.bool, device=act_dev)
    measured = []
    for step in range(a.steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        begin = time.monotonic()
        hidden, hx = model.hidden(inp["taps"], inp["engine_mx"], inp["has_taps"], inp["positions"], inp["mask"], inp["anchor"], inp["P"])
        loss, _ = distillation_loss(model, hidden, previous, ids, lp, torch.full((a.batch, 5), .2, device=act_dev),
                                   previous, mask, mask)
        cast(Callable[[], None], loss.backward)()
        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        seconds = time.monotonic()-begin
        measured.append({"step": step, "seconds": seconds, "blocks_per_second": a.batch/seconds,
                         "loss": float(loss.detach())})
        print(json.dumps(measured[-1]), flush=True)
    report: dict[str, Any] = {"batch": a.batch, "reference_math": a.reference_math, "gpu_uuid": GPU, "gpu_telemetry": telemetry, "load_seconds": load_seconds,
              "load_peak_bytes": load_peak, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "host_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "steps": measured, "memory_le20gb": torch.cuda.max_memory_reserved() <= 20_000_000_000,
              "estimated_3epoch_1m_hours": 3_000_000/(measured[-1]["blocks_per_second"]*3600)}
    a.out.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
