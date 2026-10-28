"""N1: CUDA-graph replay == eager for an arm (emulated TP2 on one card, or the pair): drafts and confidences for
B = 1-4 at k = 5, n_out 5 and 4, after O2's bench windows. Usage: native_graphcheck ARM emu|pair OUT.json
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace

import torch

from ...worker.checkpoint import Checkpoint
from ...worker.tools.parity_ref import make_taps
from ...worker.tools.tp2_bench import _EmbedOnly, fill_windows_tp, host_inp, setup_inputs
from ...worker.tp2 import check_visible
from ..tp2_native import ARMS, build

ROOT = "/opt/dsv41/x86host-data/dspark"


def main() -> None:
    arm, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
    check_visible(pair=(mode == "pair"))
    d0 = torch.device("cuda", 0)
    devs = [d0, torch.device("cuda", 1)] if mode == "pair" else [d0, d0]
    cfg = ARMS[arm]
    if len(sys.argv) > 4:
        cfg = replace(cfg, dense_profile=sys.argv[4])
    m = build(cfg, devs, Checkpoint(ROOT), slots=8)
    toks = setup_inputs(d0, ROOT)
    taps = make_taps(_EmbedOnly(m.ranks[0].embed), toks, torch.Generator(device=d0).manual_seed(5))
    fill_windows_tp(m, taps.cpu())
    m.capture(batch_sizes=(1, 2, 3, 4), ks=(5,), insert_rows=(8,), n_outs=(4, 5), margins=False)
    rep = {"arm": arm, "cfg": cfg.__dict__, "mode": mode, "cases": {}}
    ok = True
    for B in (1, 2, 3, 4):
        inp = host_inp(B, toks.cpu())
        for n in (5, 4):
            e_ids, e_conf, _, e_ids1 = m.draft(inp, 5, n_out=n)
            g_ids, g_conf, _ = m.draft_graphed(inp, 5, n_out=n)
            same = bool(torch.equal(e_ids.cpu(), g_ids[:B].cpu())) and bool(torch.equal(e_conf.cpu(), g_conf[:B].cpu()))
            ranks_agree = e_ids1 is None or bool(torch.equal(e_ids.cpu(), e_ids1.cpu()))
            rep["cases"][f"B{B}_n{n}"] = {"graph_equals_eager": same, "rank_ids_agree": ranks_agree}
            ok &= same and ranks_agree
    for st in m.ranks[0].stages:
        if hasattr(st.nbank, "check"):
            st.nbank.check()
    rep["all_ok"] = ok
    with open(out, "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))
    if not ok:
        raise SystemExit("graph replay differs from eager")


if __name__ == "__main__":
    main()
