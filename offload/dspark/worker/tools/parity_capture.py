"""D3b: the worker's drafts against live drafter captures from the Spark (rank 0 of the served legal-v2a engine).

Capture format: offload/dspark/capture/speculator.py on offload-x86host (one torch.save dict per real propose() call,
recorded after DSparkSpeculator.propose returns). Engine semantics used to rebuild each round (dflash/speculator.py
_prepare_dflash_inputs_kernel): per request, the first num_ctx - num_rejected target rows are valid context (their
main_x rows get KV at target_positions; the rejected suffix writes no KV), the query block is
[bonus/next-prefill token, noise x (N-1)] at last_valid_pos + 1 .. + N, and the window is positions
max(0, P - 127) .. P with P = last_valid_pos.

Replay: one slot per live request id; every round inserts the request's valid rows (last <= 128) and drafts the
round's whole batch at once, as the engine does. Two windows are kept per request:
  A  the captured main_x rows (isolates the drafter body: attention, MoE, mHC, heads, Markov)
  B  main_x computed by the worker from the raw taps where the round carries them (num_tokens <= 64, i.e. decode
     rounds), captured rows otherwise (prefill): the production path, the Spark sends taps.
Drafts are compared per position: free-running (as served), conditional on an equal prefix, and teacher-forced
(the worker's logits at position i with the engine's own previous draft as Markov input). Every first mismatch
(equal prefix, so the same Markov input) is classified with the independent fp32 reference (reference.py) on the
same window KV:
  numeric_near_tie     the engine's token is within the measured cross-implementation logit noise of the worker's
                       choice (max(2 bf16 steps, 3 sigma) of worker-vs-reference logits), or the reference prefers it
  routing_tie_cascade  not a logit near-tie, but some token of the block has a routing near-tie (relative gap of its
                       3rd/4th expert choice scores within 2x the worker-vs-reference score shift, floor 1e-3)
  unexplained          neither (a candidate bug; listed for hand inspection)
Drafts of requests in mid-prefill chunks (num_sampled 0: the engine discards them; classified too) and of the
server's warm-up requests are counted separately. Rounds with taps also compare the worker's and the engine's main_x
with the fp32 reference main_x of the same taps.
Usage: python -m worker.tools.parity_capture ROOT CAPTURE_DIR OUT.json
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
from ..reference import RefDrafter


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


class Slots:
    """req_id -> model slot within [base, base + n); least recently seen request evicted when full."""

    def __init__(self, base: int, n: int):
        self.base, self.n = base, n
        self.map: dict[str, int] = {}
        self.seen: dict[str, int] = {}

    def get(self, req: str, rnd: int, live: set[str]) -> tuple[int, bool]:
        new = req not in self.map
        if new:
            used = set(self.map.values())
            free = [s for s in range(self.base, self.base + self.n) if s not in used]
            if not free:
                old = min((r for r in self.map if r not in live), key=lambda r: self.seen[r])
                free = [self.map.pop(old)]
                self.seen.pop(old)
            self.map[req] = free[0]
        self.seen[req] = rnd
        return self.map[req], new


def category(req: str, num_sampled: int) -> str:
    if "warmup" in req:
        return "warmup"
    return "live" if num_sampled > 0 else "mid_prefill"


def routing_scores(m: DSparkDrafter, stage: int, x: torch.Tensor) -> torch.Tensor:
    st = m.stages[stage]
    return torch.nn.functional.softplus(x.float() @ st.gate_w.float().t()).sqrt() + st.gate_bias.float()


def classify(m: DSparkDrafter, ref: RefDrafter, slot: int, anchor: int, last: int, k: int, b: int, i0: int,
             prev: int, f_tok: int, e_tok: int, tr: dict, B: int) -> dict:
    """First mismatch at position i0 of request b (equal prefix): worker vs reference on the worker's window KV."""
    s = m.s
    rp = m.ring_pos[slot]
    sel = (rp >= max(0, last - s.window + 1)) & (rp <= last)
    ctx = [m.ring_kv[slot, i][sel] for i in range(s.n_stages)]
    rtr: dict = {}
    hh_r = ref.block(anchor, last, ctx, k, trace=rtr)
    ref.heads(hh_r, anchor, k, trace=rtr)
    pv = torch.tensor([prev], device=m.dev)
    e = m.markov.embed(pv)
    l_f = (tr["base_logits"][b, i0] + m.markov(pv, e)[0]).float()
    bias32 = ref.markov_head.float() @ ref.markov_embed[prev].float()
    l_r = rtr["base32"][i0] + bias32
    top = torch.topk(l_r, 64).indices
    noise = float((l_f[top] - l_r[top]).std())
    mag = float(l_f[f_tok].abs())
    bf16_step = 2.0 ** (int(torch.floor(torch.log2(torch.tensor(max(mag, 1e-6))))) - 7)
    band = max(2 * bf16_step, 3 * noise)
    gap_f = float(l_f[f_tok] - l_f[e_tok])
    gap_r = float(l_r[f_tok] - l_r[e_tok])
    rank_e = int((l_f > l_f[e_tok]).sum())
    # routing near-ties of the worker's block (all tokens: the block attends to itself at every later stage)
    route_min_gap, route_shift, flips = 1.0, 0.0, 0
    for i in range(s.n_stages):
        xf = tr[f"stage{i}.ffn_in"].view(B, k, -1)[b]
        xr = rtr[f"stage{i}.ffn_in"]
        cf, cr = routing_scores(m, i, xf), routing_scores(m, i, xr)
        v = cf.topk(4, dim=-1).values
        route_min_gap = min(route_min_gap, float(((v[:, 2] - v[:, 3]) / v[:, 2].abs()).min()))
        route_shift = max(route_shift, float(((cf - cr).abs().amax(-1) / v[:, 2].abs()).max()))
        ef = tr[f"stage{i}.experts"].view(B, k, -1)[b].sort(-1).values
        er = rtr[f"stage{i}.experts"].sort(-1).values
        flips += int((~(ef == er).all(-1)).sum())
    route_tie = route_min_gap <= max(2 * route_shift, 1e-3)
    if gap_f <= band or gap_r <= band:
        cls = "numeric_near_tie"
    elif route_tie:
        cls = "routing_tie_cascade"
    else:
        cls = "unexplained"
    return {"class": cls, "gap_worker": round(gap_f, 4), "gap_ref32": round(gap_r, 4), "band": round(band, 4),
            "logit_noise": round(noise, 4), "bf16_step": bf16_step, "engine_token_rank_worker": rank_e,
            "route_min_gap_rel": route_min_gap, "route_shift_rel": route_shift, "expert_set_flips_vs_ref": flips,
            "ref_pick": int(l_r.argmax()), "head_hidden_rel_vs_ref": rel(tr["head_hidden"].view(B, k, -1)[b], hh_r)}


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "/opt/dsv41/x86host-data/dspark"
    cap = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "captures/A-capture-20261022/dspark-captures")
    out_path = sys.argv[3] if len(sys.argv) > 3 else "d3b_parity_capture.json"
    dev = torch.device("cuda", 0)
    t0 = time.time()
    ck = Checkpoint(root)
    m = DSparkDrafter(ck, dev, slots=8)
    ref = RefDrafter(ck, dev)
    s = m.s
    W = s.window
    m.capture(batch_sizes=(1, 2, 3, 4), ks=(5,), insert_rows=(8,))
    files = sorted(glob.glob(os.path.join(cap, "round*.pt")))
    if not files:
        raise FileNotFoundError(f"no round*.pt under {cap}")
    slots = {"A": Slots(0, 4), "B": Slots(4, 4)}
    stats = {mode: collections.defaultdict(lambda: collections.Counter()) for mode in "AB"}
    conf_err = {mode: [] for mode in "AB"}
    first_mm: list[dict] = []
    plumbing: list[str] = []
    rounds_out = []
    mainx = {"rows": 0, "rows_identical": 0, "rel_max": 0.0, "rel_sum": 0.0, "n": 0,
             "worker_vs_fp32_sum": 0.0, "engine_vs_fp32_sum": 0.0}
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
        conf_e = r["confidence"]
        taps = r["taps"]
        mx_cap = r["main_x"].to(dev)
        mx_own = None
        if taps is not None:
            mx_own = m.main_x(torch.cat([t.to(dev) for t in taps], dim=-1))
            mainx["n"] += 1
            mainx["rows"] += mx_own.shape[0]
            mainx["rows_identical"] += int((mx_own == mx_cap).all(-1).sum())
            rr = rel(mx_own, mx_cap)
            mainx["rel_max"] = max(mainx["rel_max"], rr)
            mainx["rel_sum"] += rr
            # both against the fp32 reference main_x (decoded weights, fp32 GEMM) of the same taps
            mx_32 = ref.main_x(torch.cat([t.to(dev) for t in taps], dim=-1))
            mainx["worker_vs_fp32_sum"] += rel(mx_own, mx_32)
            mainx["engine_vs_fp32_sum"] += rel(mx_cap, mx_32)
        live = set(req_ids)
        anchors = qids[:, 0]
        lasts = qpos[:, 0] - 1
        for b in range(nr):
            exp_q = torch.cat((anchors[b:b + 1], torch.full((nq - 1,), s.noise_token)))
            if not torch.equal(qids[b], exp_q):
                plumbing.append(f"round {r['round']} req {b}: query ids {qids[b].tolist()}")
            if not torch.equal(qpos[b], lasts[b] + 1 + torch.arange(nq)):
                plumbing.append(f"round {r['round']} req {b}: query positions {qpos[b].tolist()}")
        rec = {"round": int(r["round"]), "B": nr, "k": k, "reqs": []}
        for mode in "AB":
            sl = []
            ins_slot, ins_pos, ins_rows = [], [], []
            for b, req in enumerate(req_ids):
                slot, new = slots[mode].get(req, int(r["round"]), live)
                st, en = qsl[b], qsl[b + 1]
                if new:
                    m.reset_slots(torch.tensor([slot], device=dev))
                    if int(tpos[st]) != 0:
                        plumbing.append(f"round {r['round']} req {req}: first seen at position {int(tpos[st])} "
                                        f"(window incomplete)")
                sl.append(slot)
                nv = en - st - nrej[b]
                if nv <= 0:
                    continue
                if int(tpos[st + nv - 1]) != int(lasts[b]):
                    plumbing.append(f"round {r['round']} req {b}: last valid pos {int(tpos[st + nv - 1])} != "
                                    f"anchor pos - 1 {int(lasts[b])}")
                lo = max(st, st + nv - W)
                ins_slot.append(torch.full((st + nv - lo,), slot, dtype=torch.int64))
                ins_pos.append(tpos[lo:st + nv].long())
                src = mx_own if (mode == "B" and mx_own is not None) else mx_cap
                ins_rows.append(src[lo:st + nv])
            if ins_slot:
                m.insert(torch.cat(ins_slot).to(dev), torch.cat(ins_pos).to(dev), torch.cat(ins_rows))
            slt = torch.tensor(sl, device=dev)
            anc = anchors.to(dev)
            lst = lasts.to(dev)
            d_g, c_g = m.draft_graphed(slt, anc, lst, k)
            d_g, c_g = d_g[:nr].cpu(), c_g[:nr].cpu()
            tr: dict = {}
            d_f, c_f = m.draft(slt, anc, lst, k, trace=tr)
            d_f, c_f = d_f.cpu(), c_f.cpu()
            graph_eq[(mode, torch.equal(d_g, d_f) and torch.equal(c_g, c_f))] += 1
            for b, req in enumerate(req_ids):
                cat = category(req, nsam[b])
                S = stats[mode][cat]
                S["blocks"] += 1
                same = (d_g[b] == eng[b])
                S["block_exact"] += int(bool(same.all()))
                i0 = int((~same).nonzero()[0]) if not bool(same.all()) else k
                for i in range(k):
                    S[f"pos{i}_n"] += 1
                    S[f"pos{i}_same"] += int(same[i])
                    if i <= i0 - 1 or (i == i0 and i0 < k):
                        S[f"pos{i}_cond_n"] += 1
                        S[f"pos{i}_cond_same"] += int(same[i])
                # teacher-forced: the worker's logits with the engine's previous draft as Markov input
                prevs = torch.cat((anchors[b:b + 1], eng[b, :-1])).to(dev)
                e = m.markov.embed(prevs)
                lt = tr["base_logits"][b] + m.markov(prevs, e)
                tf = lt.argmax(-1).cpu()
                for i in range(k):
                    S[f"pos{i}_tf_same"] += int(tf[i] == eng[b, i])
                if conf_e is not None:
                    upto = min(i0 + 1, k)
                    conf_err[mode].extend((c_g[b, :upto] - conf_e[b, :upto]).abs().tolist())
                rq = {"req": req[-12:], "cat": cat, f"slot_{mode}": sl[b], "last_pos": int(lasts[b]),
                      "anchor": int(anchors[b])}
                if mode == "A":
                    rq["engine"] = eng[b].tolist()
                rq[f"worker_{mode}"] = d_g[b].tolist()
                if i0 < k and mode == "A" and cat != "warmup":
                    prev = int(anchors[b]) if i0 == 0 else int(eng[b, i0 - 1])
                    c = classify(m, ref, sl[b], int(anchors[b]), int(lasts[b]), k, b, i0, prev, int(d_g[b, i0]),
                                 int(eng[b, i0]), tr, nr)
                    c.update({"round": int(r["round"]), "req": req[-12:], "cat": cat, "pos": i0,
                              "worker_tok": int(d_g[b, i0]),
                              "engine_tok": int(eng[b, i0]), "B": nr})
                    first_mm.append(c)
                    S["first_mm_" + c["class"]] += 1
                if mode == "A":
                    rec["reqs"].append(rq)
                else:
                    rec["reqs"][b].update(rq)
        rounds_out.append(rec)
        if int(r["round"]) % 50 == 0:
            print(f"round {r['round']} ({time.time() - t0:.0f} s)", flush=True)

    def table(S: collections.Counter, k: int = 5) -> dict:
        out = {"blocks": S["blocks"], "block_exact_rate": round(S["block_exact"] / max(S["blocks"], 1), 4)}
        for i in range(k):
            n = S[f"pos{i}_n"]
            if not n:
                continue
            out[f"pos{i}"] = {"match": round(S[f"pos{i}_same"] / n, 4),
                              "match_given_prefix": round(S[f"pos{i}_cond_same"] / max(S[f"pos{i}_cond_n"], 1), 4),
                              "prefix_n": S[f"pos{i}_cond_n"],
                              "teacher_forced": round(S[f"pos{i}_tf_same"] / n, 4)}
        out["first_mismatch_classes"] = {c[9:]: v for c, v in S.items() if c.startswith("first_mm_")}
        return out

    summary = {mode: {cat: table(S) for cat, S in stats[mode].items()} for mode in "AB"}
    for mode in "AB":
        ce = torch.tensor(conf_err[mode]) if conf_err[mode] else torch.zeros(1)
        summary[mode]["confidence_absdiff"] = {"n": len(conf_err[mode]), "mean": float(ce.mean()),
                                               "p99": float(ce.quantile(0.99)), "max": float(ce.max())}
    rep = {"captures": cap, "rounds": len(files), "seconds": round(time.time() - t0, 1),
           "summary": summary,
           "main_x_from_taps": {**mainx, "rel_mean": mainx["rel_sum"] / max(mainx["n"], 1),
                                "worker_vs_fp32_mean": mainx["worker_vs_fp32_sum"] / max(mainx["n"], 1),
                                "engine_vs_fp32_mean": mainx["engine_vs_fp32_sum"] / max(mainx["n"], 1)},
           "graph_equals_eager": {f"{a}_{b}": v for (a, b), v in graph_eq.items()},
           "plumbing_errors": plumbing, "first_mismatches": first_mm, "rounds_detail": rounds_out}
    with open(out_path, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps({"summary": summary, "main_x_from_taps": rep["main_x_from_taps"],
                      "graph_equals_eager": rep["graph_equals_eager"], "plumbing_errors": plumbing[:10]}, indent=1))
    cls = collections.Counter((c["cat"], c["class"]) for c in first_mm)
    print("first mismatches (mode A):", dict(cls))
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
