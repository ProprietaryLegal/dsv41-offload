"""Lane E parity: the TP = 2 drafter against the one-card worker (model.py) on the live Spark drafter captures.

The same replay as O2's D3b (tools/parity_capture.py: windows rebuilt from target positions and num_rejected, query
block [anchor, noise x 4], one slot per live request) drives both models; two windows per model: A from the
captured main_x, B from main_x the model computes itself from the captured taps (decode rounds; the production path).
Per round and request: TP2 vs one-card drafts (free-running, as served), and each of them vs the engine's drafts.
Every first TP2-vs-one-card mismatch (equal prefix, so the same Markov input) is classified from both models' own
logits (base + Markov bias of the common previous token):
  numeric_near_tie     one model's gap between the two tokens is within max(2 bf16 steps, 3 sigma of the
                       TP2-vs-one-card logit noise over the top-64 tokens)
  routing_tie_cascade  not a logit near-tie, but a token of the block changed its expert set between the models and
                       one model's 3rd / 4th expert scores are within 2x the cross-model score shift (floor 1e-3)
  unexplained          neither (a candidate bug; listed with details)
Two visible GPUs (the pair): TP2 runs on both cards, the one-card model on card 1 (GPU-7744b44b: card 0 also hosts the
live one-card worker during the pair window); one GPU: TP2 emulated on it, the one-card model next to it.
Per-round token ids stay on x86host (decodable legal output, COMMON rule 5); the summary carries aggregates only.
Usage: python -m worker.tools.tp2_parity ROOT CAPTURE_DIR OUT.json [OUT_DETAIL.json]
"""

from __future__ import annotations

import collections
import glob
import json
import os
import sys
import time

import torch

from ..checkpoint import Checkpoint
from ..model import DSparkDrafter
from ..tp2 import TPDrafter, check_visible
from .parity_capture import Slots, category


def routing_scores(st_gate_w, st_gate_bias, x):
    return torch.nn.functional.softplus(x.float() @ st_gate_w.float().t()).sqrt() + st_gate_bias.float()


def classify(one: DSparkDrafter, t1: dict, t2: dict, b: int, i0: int, prev: int, tok1: int, tok2: int, B: int,
             k: int) -> dict:
    dev = one.dev
    pv = torch.tensor([prev], device=dev)
    bias = one.markov(pv, one.markov.embed(pv))[0]
    l1 = (t1["base_logits"][b, i0].to(dev) + bias).float()
    l2 = (t2["base_logits"].view(B, -1, t2["base_logits"].shape[-1])[b, i0].to(dev) + bias).float()
    top = torch.topk(l1, 64).indices
    noise = float((l1[top] - l2[top]).std())
    mag = float(max(l1[tok1].abs(), l2[tok2].abs()))
    step = 2.0 ** (int(torch.floor(torch.log2(torch.tensor(max(mag, 1e-6))))) - 7)
    band = max(2 * step, 3 * noise)
    gap1 = float(l1[tok1] - l1[tok2])
    gap2 = float(l2[tok2] - l2[tok1])
    min_gap, shift, flips = 1.0, 0.0, 0
    for i, st in enumerate(one.stages):
        x1 = t1[f"stage{i}.ffn_in"].view(B, k, -1)[b].to(dev)
        x2 = t2[f"stage{i}.ffn_in"].view(B, k, -1)[b].to(dev)
        c1, c2 = routing_scores(st.gate_w, st.gate_bias, x1), routing_scores(st.gate_w, st.gate_bias, x2)
        for c in (c1, c2):
            v = c.topk(4, dim=-1).values
            min_gap = min(min_gap, float(((v[:, 2] - v[:, 3]) / v[:, 2].abs()).min()))
        v1 = c1.topk(4, dim=-1).values
        shift = max(shift, float(((c1 - c2).abs().amax(-1) / v1[:, 2].abs()).max()))
        e1 = t1[f"stage{i}.experts"].view(B, k, -1)[b].sort(-1).values.to(dev)
        e2 = t2[f"stage{i}.experts"].view(B, k, -1)[b].sort(-1).values.to(dev)
        flips += int((~(e1 == e2).all(-1)).sum())
    if gap1 <= band or gap2 <= band:
        cls = "numeric_near_tie"
    elif flips and min_gap <= max(2 * shift, 1e-3):
        cls = "routing_tie_cascade"
    else:
        cls = "unexplained"
    return {"class": cls, "gap_one_card": round(gap1, 4), "gap_tp2": round(gap2, 4), "band": round(band, 4),
            "logit_noise": round(noise, 4), "expert_set_flips": flips, "route_min_gap_rel": min_gap,
            "route_shift_rel": shift}


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "/opt/dsv41/x86host-data/dspark"
    cap = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "captures/A-capture-20261022/dspark-captures")
    out_path = sys.argv[3] if len(sys.argv) > 3 else "e_parity.json"
    detail_path = sys.argv[4] if len(sys.argv) > 4 else out_path.replace(".json", "_detail.json")
    vis = check_visible()
    pair = "," in vis
    t0 = time.time()
    ck = Checkpoint(root)
    d0 = torch.device("cuda", 0)
    devs = [d0, torch.device("cuda", 1)] if pair else [d0, d0]
    d1c = devs[-1]                              # the one-card model's device
    on1 = lambda: torch.cuda.device(d1c)        # model.py launches Triton on the current device
    tp = TPDrafter(ck, devs, slots=8)
    tp.capture(batch_sizes=(1, 2, 3, 4), ks=(5,), insert_rows=(8,), n_outs=(5,), margins=False)
    with on1():
        one = DSparkDrafter(ck, d1c, slots=8)
        # model.py captures without a stream: torch.cuda.graph's class-wide default capture stream is created on the
        # device current at its first use (TP2 rank 0's card here) and would steer these captures to card 0
        torch.cuda.graphs.graph.default_capture_stream = torch.cuda.Stream(d1c)
        one.capture(batch_sizes=(1, 2, 3, 4), ks=(5,), insert_rows=(8,), n_outs=(5,), margins=False)
    s = one.s
    W = s.window
    files = sorted(glob.glob(os.path.join(cap, "round*.pt")))
    if not files:
        raise FileNotFoundError(f"no round*.pt under {cap}")
    slots = {"A": Slots(0, 4), "B": Slots(4, 4)}
    st = {m: collections.defaultdict(collections.Counter) for m in "AB"}
    first_mm: list = []
    detail: list = []
    graph_eq = collections.Counter()
    for f in files:
        r = torch.load(f, map_location="cpu", weights_only=False)
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
        mx_cap = r["main_x"]
        taps_cat = torch.cat([t for t in taps], dim=-1) if taps is not None else None
        live = set(req_ids)
        anchors = qids[:, 0]
        lasts = qpos[:, 0] - 1
        rec = {"round": int(r["round"]), "B": nr, "reqs": []}
        for mode in "AB":
            sl = []
            ins = []
            for b, req in enumerate(req_ids):
                slot, new = slots[mode].get(req, int(r["round"]), live)
                if new:
                    with on1():
                        one.reset_slots(torch.tensor([slot], device=d1c))
                    tp.reset_slots([slot])
                sl.append(slot)
                st_, en = qsl[b], qsl[b + 1]
                nv = en - st_ - nrej[b]
                if nv <= 0:
                    continue
                lo = max(st_, st_ + nv - W)
                ins.append((slot, tpos[lo:st_ + nv].long(), lo, st_ + nv))
            if ins:
                sr = torch.cat([torch.full((len(p),), s_, dtype=torch.int64) for s_, p, _, _ in ins])
                pr = torch.cat([p for _, p, _, _ in ins])
                rows = torch.cat([torch.arange(a, z) for _, _, a, z in ins])
                if mode == "B" and taps_cat is not None:
                    tt = taps_cat[rows]
                    with on1():
                        one.insert(sr.to(d1c), pr.to(d1c), one.main_x(tt.to(d1c)))
                    tp.insert_eager(sr, pr, tt)
                else:
                    mx = mx_cap[rows]
                    with on1():
                        one.insert(sr.to(d1c), pr.to(d1c), mx.to(d1c))
                    tp.insert_mx_eager(sr, pr, mx)
            slt = torch.tensor(sl)
            lo_ = (lasts - (W - 1)).clamp_min(0)
            with on1():
                d1, _ = one.draft_graphed(slt.to(d1c), anchors.to(d1c), lasts.to(d1c), k)
                d1 = d1[:nr].cpu()
            inp = torch.stack((slt, anchors, lasts, lo_)).pin_memory()
            d2 = tp.draft_graphed(inp, k)[0][:nr].cpu()
            need = not torch.equal(d1, d2)
            t1: dict = {}
            t2: dict = {}
            if need:
                with on1():
                    one.draft(slt.to(d1c), anchors.to(d1c), lasts.to(d1c), k, trace=t1)
                e2, _, _, _ = tp.draft(inp, k, trace=t2)
                graph_eq[(mode, torch.equal(e2.cpu(), d2))] += 1
            for b, req in enumerate(req_ids):
                cat = category(req, nsam[b])
                S = st[mode][cat]
                S["blocks"] += 1
                same = d1[b] == d2[b]
                S["tp2_vs_one_block_exact"] += int(bool(same.all()))
                S["one_vs_engine_block_exact"] += int(bool((d1[b] == eng[b]).all()))
                S["tp2_vs_engine_block_exact"] += int(bool((d2[b] == eng[b]).all()))
                for i in range(k):
                    S[f"pos{i}_n"] += 1
                    S[f"pos{i}_tp2_one"] += int(same[i])
                    S[f"pos{i}_one_eng"] += int(d1[b, i] == eng[b, i])
                    S[f"pos{i}_tp2_eng"] += int(d2[b, i] == eng[b, i])
                if not bool(same.all()):
                    i0 = int((~same).nonzero()[0])
                    prev = int(anchors[b]) if i0 == 0 else int(d1[b, i0 - 1])
                    with on1():
                        c = classify(one, t1, t2, b, i0, prev, int(d1[b, i0]), int(d2[b, i0]), nr, k)
                    c.update({"mode": mode, "cat": cat, "round": int(r["round"]), "pos": i0, "B": nr})
                    first_mm.append(c)
                    S["first_mm_" + c["class"]] += 1
                if mode == "A":
                    rec["reqs"].append({"req": req[-12:], "cat": cat, "engine": eng[b].tolist(),
                                        "one_A": d1[b].tolist(), "tp2_A": d2[b].tolist()})
                else:
                    rec["reqs"][b].update({"one_B": d1[b].tolist(), "tp2_B": d2[b].tolist()})
        detail.append(rec)
        if int(r["round"]) % 100 == 0:
            print(f"round {r['round']} ({time.time() - t0:.0f} s)", flush=True)

    def table(S):
        n = max(S["blocks"], 1)
        out = {"blocks": S["blocks"],
               "tp2_vs_one_card_block_exact": round(S["tp2_vs_one_block_exact"] / n, 4),
               "one_card_vs_engine_block_exact": round(S["one_vs_engine_block_exact"] / n, 4),
               "tp2_vs_engine_block_exact": round(S["tp2_vs_engine_block_exact"] / n, 4),
               "first_mismatch_classes": {c[9:]: v for c, v in S.items() if c.startswith("first_mm_")}}
        for i in range(5):
            m = S[f"pos{i}_n"]
            if m:
                out[f"pos{i}"] = {"tp2_vs_one": round(S[f"pos{i}_tp2_one"] / m, 4),
                                  "one_vs_engine": round(S[f"pos{i}_one_eng"] / m, 4),
                                  "tp2_vs_engine": round(S[f"pos{i}_tp2_eng"] / m, 4)}
        return out

    summary = {mode: {cat: table(S) for cat, S in st[mode].items()} for mode in "AB"}
    rep = {"captures": cap, "rounds": len(files), "pair": pair, "devices": vis, "seconds": round(time.time() - t0, 1),
           "summary": summary, "tp2_graph_equals_eager_on_mismatch_rounds": {f"{a}_{b}": v for (a, b), v in
                                                                             graph_eq.items()},
           "first_mismatch_count": len(first_mm),
           "first_mismatch_by_class": dict(collections.Counter((c["mode"], c["cat"], c["class"]) for c in first_mm)
                                           .most_common()),
           "first_mismatches_unexplained": [c for c in first_mm if c["class"] == "unexplained"],
           "first_mismatch_gaps": [{k_: c[k_] for k_ in ("mode", "cat", "class", "pos", "B", "gap_one_card", "gap_tp2",
                                                        "band", "expert_set_flips")} for c in first_mm]}
    rep["first_mismatch_by_class"] = {"/".join(k_): v for k_, v in rep["first_mismatch_by_class"].items()}
    with open(out_path, "w") as fh:
        json.dump(rep, fh, indent=1)
    with open(detail_path, "w") as fh:
        json.dump({"rounds": detail, "first_mismatches": first_mm}, fh)
    print(json.dumps({k_: rep[k_] for k_ in ("summary", "first_mismatch_by_class",
                                             "tp2_graph_equals_eager_on_mismatch_rounds")}, indent=1))
    print("WROTE", out_path, detail_path)


if __name__ == "__main__":
    main()
