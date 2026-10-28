"""Full TP2-emulated factory check of FP32-head output-prefix invariance."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast

import torch

from ....worker.server import build_core_tp2
from ....worker.tools.parity_ref import make_taps
from ....worker.tools.tp2_bench import fill_windows_tp, host_inp, setup_inputs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", default="/opt/dsv41/x86host-data/dspark")
    a = ap.parse_args()
    if os.environ.get("DSPARK_WORKER_TP_EMULATE") != "1":
        raise RuntimeError("bounded solo TP2 emulation required")
    os.environ["DSPARK_WORKER_ARM"] = "native-packed"
    os.environ.pop("DSPARK_V1_DUMP_DIR", None)
    core, model = build_core_tp2(a.root, max_model_len=1048576)
    m = cast(Any, model)
    dev = torch.device("cuda", 0)
    setup = cast(Callable[[torch.device, str], torch.Tensor], setup_inputs)
    make = cast(Callable[[Any, torch.Tensor, torch.Generator], torch.Tensor], make_taps)
    host = cast(Callable[[int, torch.Tensor], torch.Tensor], host_inp)
    tokens = setup(dev, a.root)
    taps = make(SimpleNamespace(embed=m.ranks[0].embed), tokens, torch.Generator(device=dev).manual_seed(5)).cpu()
    fill_windows_tp(m, taps)
    cells = []
    for b in range(1, 5):
        inp = host(b, tokens.cpu())
        ids, confidence, _ = m.draft_graphed(inp, 5, n_out=5, margin=False)
        torch.cuda.synchronize(dev)
        expected_ids, expected_conf = ids.cpu().clone(), confidence.cpu().clone()
        for n in range(1, 6):
            for margin in (False, True):
                ids, confidence, mg = m.draft_graphed(inp, 5, n_out=n, margin=margin)
                torch.cuda.synchronize(dev)
                got, conf = ids.cpu(), confidence.cpu()
                if not torch.equal(got[:, :n], expected_ids[:, :n]) or not torch.equal(conf[:, :n], expected_conf[:, :n]):
                    raise RuntimeError(f"FP32 head prefix changed: B{b}/n{n}/margin{margin}")
                if bool((got[:, n:] != 0).any() | (conf[:, n:] != 0).any()):
                    raise RuntimeError("unconsumed output tail is not zero")
                if margin and (mg is None or not bool(torch.isfinite(mg).all())):
                    raise RuntimeError("missing or nonfinite requested margins")
                cells.append(dict(batch=b, n_out=n, margin=margin, exact=True))
    result = dict(arm="native-packed", identity=core.expected.drafter_identity.hex(),
                  mode="complete TP2 emulation on assigned solo card", cases=cells,
                  devices=os.environ["CUDA_VISIBLE_DEVICES"], exact_prefixes=True,
                  scope="shared FP32 K6/Markov head path; not physical pair protocol")
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)


if __name__ == "__main__":
    main()
