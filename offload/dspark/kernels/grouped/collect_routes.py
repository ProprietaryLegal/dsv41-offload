"""Replay captured real legal-v2a rounds through the K2 control and retain MoE inputs.

All output stays local on x86host. This measures control reuse histograms, not future
native routing choices. No engine requests or Spark access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

from build import require_lock


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("worker_root", type=Path)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("captures", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--rounds", type=int, default=32)
    a = p.parse_args()
    require_lock()
    sys.path.insert(0, str(a.worker_root))
    from worker.checkpoint import Checkpoint
    from worker.model import DSparkDrafter
    from worker.tools.parity_capture import Slots
    torch.set_num_threads(16)
    m = DSparkDrafter(Checkpoint(str(a.checkpoint)), torch.device("cuda:0"), slots=4)
    slots = Slots(0, 4)
    cases: list[dict[str, object]] = []
    reports: list[dict[str, object]] = []
    files = sorted(a.captures.glob("round*.pt"))
    if len(files) < a.rounds:
        raise RuntimeError("too few captured rounds")
    for index, path in enumerate(files[:a.rounds]):
        r = torch.load(path, map_location="cpu", weights_only=False)
        reqs, rnd = r["req_ids"], int(r["round"])
        qids = r["query_input_ids"].view(len(reqs), -1).long()
        qpos = r["query_positions"].view(len(reqs), -1).long()
        k = int(r["num_speculative_steps"])
        if not r["sample_from_anchor"] or qids.shape[1] != k:
            raise RuntimeError("unsupported capture contract")
        category = "warmup" if any("warmup" in req for req in reqs) else (
            "live" if bool((r["num_sampled"]>0).any()) else "mid_prefill")
        qsl, nrej = r["query_start_loc"], r["num_rejected"]
        sl, sr, pr, mx = [], [], [], []
        for b, req in enumerate(reqs):
            slot, new = slots.get(req, rnd, set(reqs))
            if new:
                m.reset_slots(torch.tensor([slot], device=m.dev))
                if int(r["target_positions"][int(qsl[b])]) != 0:
                    raise RuntimeError("cannot replay a request missing its initial window")
            sl.append(slot)
            lo, hi = int(qsl[b]), int(qsl[b+1])-int(nrej[b])
            lo = max(lo, hi-128)
            if hi>lo:
                sr.append(torch.full((hi-lo,), slot, dtype=torch.int64))
                pr.append(r["target_positions"][lo:hi].long())
                mx.append(r["main_x"][lo:hi])
        if sr:
            m.insert(torch.cat(sr).to(m.dev), torch.cat(pr).to(m.dev), torch.cat(mx).to(m.dev))
        trace: dict[str, torch.Tensor] = {}
        m.draft(torch.tensor(sl, device=m.dev), qids[:, 0].to(m.dev),
                (qpos[:, 0]-1).to(m.dev), k, trace=trace)
        for stage in range(3):
            ids = trace[f"stage{stage}.experts"].cpu().int()
            x = trace[f"stage{stage}.ffn_in"].cpu().bfloat16()
            weights = trace[f"stage{stage}.route_w"].cpu().float()
            counts = torch.bincount(ids.flatten().long(), minlength=128)
            cases.append({"round": rnd, "stage": stage, "category": category, "x": x, "ids": ids, "weights": weights})
            reports.append({"round": rnd, "stage": stage, "category": category, "rows": x.shape[0],
                            "active_experts": int((counts>0).sum()), "counts": counts.tolist(),
                            "capture_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        if index%16==0 or index==a.rounds-1:
            torch.save(cases, a.out)
            a.out.with_suffix(".json").write_text(json.dumps(reports, indent=2)+"\n")
        print("captured round", rnd, "rows", len(reqs)*k, flush=True)
    a.out.with_suffix(".complete.json").write_text(json.dumps({"rounds": a.rounds, "moe_inputs": len(cases),
        "capture_root": str(a.captures), "complete": True})+"\n")
    print("complete", len(cases), "MoE inputs", flush=True)


if __name__ == "__main__":
    main()
