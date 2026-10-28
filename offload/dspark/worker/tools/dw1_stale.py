"""DW1 item 6 (F-DWAIT option C): offline upper bound for pre-drafting on a stale window.

Option C would draft round t+1's proposals while the target verifies round t, so the drafter would lack exactly the rows
that verify produces (the rows the engine sends in round t). This measures, on the V1 replay set, the acceptance of
drafts made WITHOUT those rows against drafts made with them, with everything else perfect: the true anchor (the token
the target sampled) and the true accepted count. Real pre-drafting must also guess both, so this is an upper bound.

Two windows per request on one model (one card, GPU-bc62df2a): slots 0-3 insert the round's rows, then draft (fresh,
as served); slots 4-7 draft first, then insert the same rows (stale). Both end every round with the same window.
Rows: main_x from the captured taps where the round has them, the captured main_x otherwise (parity_capture mode B).
Score: leading drafts equal to the target's realized tokens (native_parity.accept_len; fixed depth 5, the offline proxy
V1-VERIFY used), on live blocks (parity_capture.category) of requests seen in an earlier round, whose acceptance is
determined for both arms, as a paired difference.
Usage (from offload/dspark): python -m worker.tools.dw1_stale ROOT OUT.json [capture_dir]
"""

from __future__ import annotations

import collections
import glob
import json
import math
import os
import sys

import torch

from .dw1_stages import build_one_card
from .parity_capture import Slots, category


# load_rounds / realized_tokens / accept_len: copies of offload/dspark/native/tools/native_parity.py's helpers (that
# module imports the native TP2 stack, which this one-card tool does not need)
def load_rounds(cap: str) -> list[dict]:
    files = sorted(glob.glob(os.path.join(cap, "round*.pt")))
    if not files:
        raise FileNotFoundError(f"no round*.pt under {cap}")
    return [torch.load(f, map_location="cpu", weights_only=False) for f in files]


def realized_tokens(rounds: list[dict]) -> dict:
    """req -> {position: token the target produced there} from the accepted target rows and the anchors."""
    tok: dict = collections.defaultdict(dict)
    for r in rounds:
        req_ids = list(r["req_ids"])
        qsl = [int(x) for x in r["query_start_loc"]]
        nrej = r["num_rejected"].tolist()
        tids = r["target_input_ids"].long()
        tpos = r["target_positions"].long()
        nq = int(r["num_query_per_req"])
        qids = r["query_input_ids"].view(len(req_ids), nq).long()
        qpos = r["query_positions"].view(len(req_ids), nq).long()
        for b, req in enumerate(req_ids):
            st, en = qsl[b], qsl[b + 1]
            nv = en - st - nrej[b]
            for j in range(st, st + max(nv, 0)):
                p, t = int(tpos[j]), int(tids[j])
                if tok[req].get(p, t) != t:
                    raise RuntimeError(f"{req}: position {p} seen with two target tokens")
                tok[req][p] = t
            tok[req][int(qpos[b, 0])] = int(qids[b, 0])
    return tok


def accept_len(draft: list[int], first_pos: int, real: dict) -> tuple[int, bool]:
    """(leading drafts equal to the target's tokens at first_pos.., determined)."""
    for i, d in enumerate(draft):
        t = real.get(first_pos + i)
        if t is None:
            return i, False
        if t != d:
            return i, True
    return len(draft), True


def main() -> None:
    root, out_path = sys.argv[1], sys.argv[2]
    cap = sys.argv[3] if len(sys.argv) > 3 else os.path.join(root, "captures/A-capture-20261022/dspark-captures")
    _core, m = build_one_card(root)
    dev, W = m.dev, m.s.window
    rounds = load_rounds(cap)
    real = realized_tokens(rounds)
    slots = Slots(0, 4)
    seen_before: set[str] = set()
    fresh_L: list[int] = []
    stale_L: list[int] = []
    same_drafts = 0
    for r in rounds:
        req_ids = list(r["req_ids"])
        nr = len(req_ids)
        k = int(r["num_speculative_steps"])
        nq = int(r["num_query_per_req"])
        qsl = [int(x) for x in r["query_start_loc"]]
        nrej = r["num_rejected"].tolist()
        nsam = r["num_sampled"].tolist()
        tpos = r["target_positions"].long()
        qids = r["query_input_ids"].view(nr, nq).long()
        qpos = r["query_positions"].view(nr, nq).long()
        mx = r["main_x"].to(dev)
        if r["taps"] is not None:
            mx = m.main_x(torch.cat([t.to(dev) for t in r["taps"]], dim=-1))
        live = set(req_ids)
        sl, ins_s, ins_p, ins_x = [], [], [], []
        for b, req in enumerate(req_ids):
            slot, new = slots.get(req, int(r["round"]), live)
            if new:
                m.reset_slots(torch.tensor([slot, slot + 4], device=dev))
                seen_before.discard(req)
            sl.append(slot)
            st, en = qsl[b], qsl[b + 1]
            nv = en - st - nrej[b]
            if nv > 0:
                lo = max(st, st + nv - W)
                ins_s.append(torch.full((st + nv - lo,), slot, dtype=torch.int64))
                ins_p.append(tpos[lo:st + nv])
                ins_x.append(mx[lo:st + nv])
        anc, lst = qids[:, 0].to(dev), (qpos[:, 0] - 1).to(dev)
        s_fresh = torch.tensor(sl, device=dev)
        s_stale = s_fresh + 4
        d_stale, _ = m.draft_graphed(s_stale, anc, lst, k)          # before this round's rows
        d_stale = d_stale[:nr].cpu()
        if ins_s:
            sr = torch.cat(ins_s).to(dev)
            pr = torch.cat(ins_p).to(dev)
            xr = torch.cat(ins_x)
            m.insert(sr, pr, xr)
            m.insert(sr + 4, pr, xr)
        d_fresh, _ = m.draft_graphed(s_fresh, anc, lst, k)
        d_fresh = d_fresh[:nr].cpu()
        for b, req in enumerate(req_ids):
            ok_cat = category(req, nsam[b]) == "live"
            if ok_cat and req in seen_before:
                p0 = int(qpos[b, 0]) + 1
                lf, df = accept_len(d_fresh[b].tolist(), p0, real[req])
                ls, ds = accept_len(d_stale[b].tolist(), p0, real[req])
                if df and ds:
                    fresh_L.append(lf)
                    stale_L.append(ls)
                    same_drafts += int(torch.equal(d_fresh[b], d_stale[b]))
            seen_before.add(req)
    n = len(fresh_L)
    d = [a - b for a, b in zip(fresh_L, stale_L)]
    mean = sum(d) / max(n, 1)
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / max(n - 1, 1))
    rep = {"device": torch.cuda.get_device_name(0), "blocks": n,
           "fresh_accepted_mean": round(sum(fresh_L) / max(n, 1), 4),
           "stale_accepted_mean": round(sum(stale_L) / max(n, 1), 4),
           "loss_mean": round(mean, 4), "loss_se": round(sd / math.sqrt(max(n, 1)), 4),
           "blocks_identical_drafts": same_drafts,
           "note": "fixed depth 5; upper bound: true anchor and accepted count; live blocks of requests seen in an earlier round"}
    with open(out_path, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
