"""N1: an arm's drafts on the live Spark captures — against the engine's drafts, against what the TARGET actually
produced (realized continuation), and the routed-expert reuse the kernels see.

Replay = lane E's tp2_parity (O2's D3b): one slot per live request, the round's valid rows inserted (window A: the
captured main_x rows; B: main_x computed from the captured taps on decode rounds, the production path), the whole
batch drafted at once with k = 5 (eager, with a trace for the routing ids).
Realized continuation: every round carries the target's own input ids at its target positions (the accepted rows)
and the anchor (the token the target sampled next); together they give the target's token at every position the
replay reaches. A block's realized acceptance L = number of leading drafts equal to the target's tokens at P+2..
(counted only where the target's token is known; exact acceptance for greedy rounds, the realized-sample match
otherwise). The engine's own drafts get the same L, so arms compare on identical ground.
Expert reuse: per stage and round, unique experts among the T x 3 (row, slot) pairs, by T.

Outputs: OUT.json (aggregates only: committed-safe) and OUT_drafts.pt (per-round drafts + routing: token ids are
decodable legal output, x86host only, COMMON rule 5). One process, one arm; compare arms with native_compare.py.
Usage (x86host): python -m offload.dspark.native.tools.native_parity ARM emu|pair OUT.json [CAPTURE_DIR] [ROOT]
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys
import time

import torch

from ...worker.checkpoint import Checkpoint
from ...worker.tools.parity_capture import Slots, category
from ...worker.tp2 import check_visible
from ..tp2_native import ARMS, build

ROOT = "/opt/dsv41/x86host-data/dspark"
CAPS = os.path.join(ROOT, "captures/A-capture-20261022/dspark-captures")


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
    """(leading drafts equal to the target's tokens at first_pos.., determined). Determined when the first mismatch
    lies at a position whose target token is known, or every draft matches; undetermined when an unknown position
    comes first (the end of a request's replayed text)."""
    for i, d in enumerate(draft):
        t = real.get(first_pos + i)
        if t is None:
            return i, False
        if t != d:
            return i, True
    return len(draft), True


def stage_counts(freq: collections.Counter, stage: int) -> list[int]:
    """Use counts of one stage's experts, largest first."""
    return sorted((c for (j, _), c in freq.items() if j == stage), reverse=True)


def main() -> None:
    arm = sys.argv[1]
    mode = sys.argv[2]
    out_path = sys.argv[3]
    cap = sys.argv[4] if len(sys.argv) > 4 else CAPS
    root = sys.argv[5] if len(sys.argv) > 5 else ROOT
    if arm not in ARMS:
        raise SystemExit(f"arm {arm!r} not in {sorted(ARMS)}")
    vis = check_visible(pair=(mode == "pair"))
    t0 = time.time()
    rounds = load_rounds(cap)
    real = realized_tokens(rounds)
    ck = Checkpoint(root)
    d0 = torch.device("cuda", 0)
    devs = [d0, torch.device("cuda", 1)] if mode == "pair" else [d0, d0]
    for d in set(devs):
        torch.empty(1, device=d)                    # the CUDA context and allocator exist before the counters reset
        torch.cuda.reset_peak_memory_stats(d)
    tl0 = time.time()
    m = build(arm, devs, ck, slots=8)
    load_s = time.time() - tl0
    mem = m.memory_table()
    W = m.s.window
    slots = {"A": Slots(0, 4), "B": Slots(4, 4)}
    st = {w: collections.defaultdict(collections.Counter) for w in "AB"}
    reuse = collections.defaultdict(list)          # (stage, T) -> unique expert counts (window B, live rounds)
    freq = collections.Counter()
    saved = []
    for r in rounds:
        req_ids = list(r["req_ids"])
        nr = len(req_ids)
        k = int(r["num_speculative_steps"])
        nq = int(r["num_query_per_req"])
        if not r["sample_from_anchor"] or nq != k:
            raise RuntimeError(f"round {r['round']}: sample_from_anchor {r['sample_from_anchor']} nq {nq} k {k}")
        qsl = [int(x) for x in r["query_start_loc"]]
        nrej = r["num_rejected"].tolist()
        nsam = r["num_sampled"].tolist()
        tpos = r["target_positions"]
        qids = r["query_input_ids"].view(nr, nq).long()
        qpos = r["query_positions"].view(nr, nq).long()
        eng = r["draft_tokens"].long()
        taps = r["taps"]
        taps_cat = torch.cat([t for t in taps], dim=-1) if taps is not None else None
        live = set(req_ids)
        anchors = qids[:, 0]
        lasts = qpos[:, 0] - 1
        rec = {"round": int(r["round"]), "req": [q[-12:] for q in req_ids]}
        for w in "AB":
            sl = []
            ins = []
            for b, req in enumerate(req_ids):
                slot, new = slots[w].get(req, int(r["round"]), live)
                if new:
                    m.reset_slots([slot])
                sl.append(slot)
                s_, e_ = qsl[b], qsl[b + 1]
                nv = e_ - s_ - nrej[b]
                if nv <= 0:
                    continue
                lo = max(s_, s_ + nv - W)
                ins.append((slot, tpos[lo:s_ + nv].long(), lo, s_ + nv))
            if ins:
                sr = torch.cat([torch.full((len(p),), s2, dtype=torch.int64) for s2, p, _, _ in ins])
                pr = torch.cat([p for _, p, _, _ in ins])
                rows = torch.cat([torch.arange(a, z) for _, _, a, z in ins])
                if w == "B" and taps_cat is not None:
                    m.insert_eager(sr, pr, taps_cat[rows])
                else:
                    m.insert_mx_eager(sr, pr, r["main_x"][rows])
            slt = torch.tensor(sl)
            lo_ = (lasts - (W - 1)).clamp_min(0)
            inp = torch.stack((slt, anchors, lasts, lo_)).pin_memory()
            tr: dict = {}
            d, conf, _, _ = m.draft(inp, k, trace=tr)
            d = d[:nr].cpu()
            rec[w] = d.tolist()
            if w == "B":
                rec["experts"] = [tr[f"stage{i}.experts"].cpu().tolist() for i in range(m.s.n_stages)]
                if any(category(q, n) == "live" for q, n in zip(req_ids, nsam)):
                    T = nr * k
                    for i in range(m.s.n_stages):
                        ids = tr[f"stage{i}.experts"].cpu()
                        reuse[(i, T)].append(int(ids.unique().numel()))
                        freq.update((i, e) for e in ids.flatten().tolist())
            for b, req in enumerate(req_ids):
                cat = category(req, nsam[b])
                S = st[w][cat]
                S["blocks"] += 1
                S["block_exact_vs_engine"] += int(bool((d[b] == eng[b]).all()))
                for i in range(k):
                    S[f"pos{i}_vs_engine"] += int(d[b, i] == eng[b, i])
                first = int(qpos[b, 0]) + 1
                La, oka = accept_len(d[b].tolist(), first, real[req])
                Le, oke = accept_len(eng[b].tolist(), first, real[req])
                if oka and oke:                          # both acceptance lengths determined by the target's tokens
                    S["full_known"] += 1
                    S["accept_arm"] += La
                    S["accept_engine"] += Le
                    S[f"accept_arm_hist_{La}"] += 1
                    S["arm_better"] += int(La > Le)
                    S["arm_worse"] += int(La < Le)
        saved.append(rec)
        if int(r["round"]) % 100 == 0:
            print(f"round {r['round']} ({time.time() - t0:.0f} s)", flush=True)

    def table(S):
        n = max(S["blocks"], 1)
        f = max(S["full_known"], 1)
        out = {"blocks": S["blocks"], "block_exact_vs_engine": round(S["block_exact_vs_engine"] / n, 4),
               "pos_vs_engine": [round(S[f"pos{i}_vs_engine"] / n, 4) for i in range(5)],
               "realized_blocks": S["full_known"],
               "realized_accept_mean_arm": round(S["accept_arm"] / f, 4),
               "realized_accept_mean_engine": round(S["accept_engine"] / f, 4),
               "arm_better_blocks": S["arm_better"], "arm_worse_blocks": S["arm_worse"],
               "accept_hist_arm": [S[f"accept_arm_hist_{j}"] for j in range(6)]}
        return out

    temps = collections.Counter(round(float(t), 3) for r in rounds for t in r["temperature"].tolist())
    rep = {"arm": arm, "cfg": ARMS[arm].__dict__, "mode": mode, "devices": vis, "captures": cap,
           "request_temperatures": {str(k_): v for k_, v in sorted(temps.items())},
           "rounds": len(rounds), "load_seconds": round(load_s, 1), "seconds": round(time.time() - t0, 1),
           "summary": {w: {c: table(S) for c, S in st[w].items()} for w in "AB"},
           "expert_reuse_unique_by_stage_T": {f"stage{i}_T{T}": {"n": len(v), "mean": round(sum(v) / len(v), 2),
                                                                  "max": max(v), "min": min(v)}
                                              for (i, T), v in sorted(reuse.items())},
           "experts_used_per_stage": {f"stage{i}": len(stage_counts(freq, i)) for i in range(m.s.n_stages)},
           "expert_top8_share_per_stage": {f"stage{i}": round(sum(stage_counts(freq, i)[:8]) /
                                                              max(1, sum(stage_counts(freq, i))), 3)
                                           for i in range(m.s.n_stages)},
           "memory": mem,
           "load_stats": getattr(m, "load_stats", None).__dict__ if getattr(m, "load_stats", None) else None}
    with open(out_path, "w") as fh:
        json.dump(rep, fh, indent=1, default=str)
    torch.save(saved, out_path.replace(".json", "_drafts.pt"))
    print(json.dumps({k_: rep[k_] for k_ in ("arm", "load_seconds", "seconds", "summary",
                                             "expert_reuse_unique_by_stage_T")}, indent=1))
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
