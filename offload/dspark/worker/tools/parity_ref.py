"""D3a: the fast SM86 drafter (model.py) against the straightforward fp32 reference (reference.py), per layer.

Inputs are the same for both: a token sequence (an English text through the checkpoint's tokenizer; anchors are real
next tokens) and synthetic target taps (the target does not run on x86host; the coordinator's live captures are D3b).
Taps: per tap layer, the token's embedding times a layer gain plus Gaussian noise, with a few large channels, so
main_proj's power-of-two input prescale is exercised.

Two modes per case:
  shared context  - the reference attends over the fast path's own window KV (isolates the query-block forward);
  independent     - the reference computes main_x and the window KV itself from the taps (end to end).
Reported per case: rel. errors of main_x, window KV (and the fraction of identical NoPE record values), per-stage
activations, head hidden and base logits; expert-set agreement per stage with the reference's 3rd/4th score margin
when they differ; drafts end-to-end and teacher-forced (each position scored with the fast path's previous tokens) with
every mismatch classified as near-tie or not from the reference's fp32 logit margin and the measured logit noise.
Usage: python -m worker.tools.parity_ref ROOT OUT.json [cases]
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

from ..checkpoint import Checkpoint
from ..model import DSparkDrafter
from ..reference import RefDrafter

# The reference paragraph is not shipped with the public source (prompt-bearing text stays outside the PLI Labs
# data boundary). Point DSV41_PARITY_TEXT_FILE at any UTF-8 paragraph of about 300 tokens.
def _parity_text() -> str:
    path = os.environ.get("DSV41_PARITY_TEXT_FILE")
    if not path:
        raise SystemExit("set DSV41_PARITY_TEXT_FILE to a UTF-8 text file; the reference paragraph is not shipped")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


class PreciseLinear:
    """Exact-decoded fp32 stand-in for an Exl3Linear (same tensors): y = H(H(x * suh) @ W) * svh in fp32."""

    def __init__(self, lin, H: torch.Tensor):
        import exllamav3_ext as ext
        self.lin, self.H = lin, H
        self.w = torch.empty(lin.in_features, lin.out_features, dtype=torch.float16, device=lin.trellis.device)
        ext.reconstruct(self.w, lin.trellis, lin.k, False, True)

    def _had(self, x):
        m_, n = x.shape
        return (x.view(m_, n // 128, 128) @ self.H).view(m_, n)

    def __call__(self, x, out=None):
        y = self._had(self._had(x.float() * self.lin.suh.float()) @ self.w.float()) * self.lin.svh.float()
        if out is not None:
            out.copy_(y)
            return out
        return y


def precise_moe(m: DSparkDrafter, ref: RefDrafter):
    """m._moe with the routed experts computed like reference.moe_ (exact weights, fp32, routing weight before w2)."""
    from ..ops import route

    def moe(st, x, trace=None, tag=""):
        s_ = m.s
        ids, w = route(x, st.gate_w, st.gate_bias, s_.top_k, s_.route_scale)
        if trace is not None:
            trace[tag + ".experts"] = ids
        si = m.stages.index(st)
        rst = ref.stages[si]
        y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        lim = s_.swiglu_limit
        from ..reference import RefLinear
        for t in range(x.shape[0]):
            for j in range(s_.top_k):
                e = int(ids[t, j])
                pfx = rst["prefix"] + f"ffn.experts.{e}."
                w1, w3, w2 = (RefLinear(ref.ck, pfx + n, ref.dev, ref.H) for n in ("w1", "w3", "w2"))
                a = torch.nn.functional.silu(w1(x[t:t + 1]).clamp(max=lim)) * w3(x[t:t + 1]).clamp(-lim, lim) * w[t, j]
                y[t] += w2(a.to(x.dtype))[0]
        g = st.sh_w1(x)
        u = st.sh_w3(x)
        a = torch.nn.functional.silu(g.clamp(max=lim)) * u.clamp(min=-lim, max=lim)
        y += st.sh_w2(a.to(x.dtype))
        return y.to(torch.bfloat16)
    return moe


def make_precise(m: DSparkDrafter, ref: RefDrafter) -> None:
    """Swap every EXL3 matrix and the routed experts of m for exact fp32 math (in place, for this process)."""
    H = ref.H
    m.head = PreciseLinear(m.head, H)
    m.main_proj = PreciseLinear(m.main_proj, H)
    for st in m.stages:
        for n in ("wq_a", "wkv", "wq_b", "wo_b", "sh_w1", "sh_w3", "sh_w2"):
            setattr(st, n, PreciseLinear(getattr(st, n), H))
    m._moe = precise_moe(m, ref)


def make_taps(m: DSparkDrafter, toks: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    e = m.embed[toks].float()
    parts = []
    for gain in (40.0, 70.0, 110.0):
        noise = torch.randn(e.shape, generator=gen, device=e.device) * 0.02 * gain
        parts.append(e * gain + noise)
    t = torch.cat(parts, -1)
    big = torch.randint(0, t.shape[1], (8,), generator=gen, device=e.device)
    t[:, big] *= 400.0                                   # a few channels in the 1e2-1e4 range
    return t.to(torch.bfloat16)


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "/opt/dsv41/x86host-data/dspark"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "d3a_parity_ref.json"
    n_cases = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    dev = torch.device("cuda", 0)
    ck = Checkpoint(root)
    m = DSparkDrafter(ck, dev)
    ref = RefDrafter(ck, dev)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(root, "tokenizer.json"))
    ids = tok.encode(_parity_text()).ids
    ids = (ids * (1 + 400 // len(ids)))[:400]
    toks = torch.tensor(ids, dtype=torch.int64, device=dev)
    gen = torch.Generator(device=dev).manual_seed(1017)
    taps = make_taps(m, toks, gen)
    W, s = m.s.window, m.s
    report: dict = {"cases": [], "tokens": len(ids)}

    # main_x / context KV over the whole sequence once (per position)
    pos_all = torch.arange(len(ids), device=dev)
    mx_f = m.main_x(taps)
    mx_r = ref.main_x(taps)
    kv_f = m.context_kv(mx_f, pos_all)                  # [S, n, 512]
    kv_r = ref.context_kv(mx_r, pos_all)
    kv_r_shared = ref.context_kv(mx_f, pos_all)          # same main_x: isolates wkv / norm / rope / record
    nope_same = float((kv_f[..., :448] == kv_r_shared[..., :448]).float().mean())
    report["main_x_rel"] = rel(mx_f, mx_r)
    report["context_kv_rel_independent"] = [rel(kv_f[i], kv_r[i]) for i in range(s.n_stages)]
    report["context_kv_rel_same_main_x"] = [rel(kv_f[i], kv_r_shared[i]) for i in range(s.n_stages)]
    report["context_kv_nope_identical_fraction_same_main_x"] = nope_same
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}), flush=True)

    totals = {"pos": 0, "e2e_same": 0, "tf_same": 0, "expert_sets_same": 0, "expert_sets": 0, "requests": 0,
              "requests_with_flip": 0}
    lastp = list(range(140, 140 + 10 * n_cases, 10))
    for ci, P in enumerate(lastp):
        P = min(P, len(ids) - 2)
        B = 1 + ci % 4
        k = 5 if ci % 6 else 4
        slots = torch.arange(B, device=dev)
        # each request of the batch: the same text, windows ending at P - b (distinct contexts)
        lasts = torch.tensor([P - b for b in range(B)], device=dev)
        m.reset_slots(slots)
        for b in range(B):
            lp = int(lasts[b])
            rows = torch.arange(max(0, lp - W + 1), lp + 1, device=dev)
            m.insert(torch.full_like(rows, b), rows, mx_f[rows])
        anchors = toks[lasts + 1]
        tr: dict = {}
        d_fast, c_fast = m.draft(slots, anchors, lasts, k, trace=tr)
        hh_f = tr["head_hidden"].view(B, k, -1)
        base_f = tr["base_logits"]                       # bf16 [B, k, V]
        case = {"P": P, "B": B, "k": k, "requests": []}
        for b in range(B):
            lp = int(lasts[b])
            rows = torch.arange(max(0, lp - W + 1), lp + 1, device=dev)
            rq: dict = {"last_pos": lp}
            rp = m.ring_pos[b]
            sel = (rp >= max(0, lp - W + 1)) & (rp <= lp)
            ring_ctx = [m.ring_kv[b, i][sel] for i in range(s.n_stages)]     # the fast path's own window KV
            for mode, ctx in (("shared", ring_ctx),
                              ("independent", [kv_r[i][rows] for i in range(s.n_stages)])):
                rtr: dict = {}
                hh_r = ref.block(int(anchors[b]), lp, ctx, k, trace=rtr)
                d_ref, c_ref = ref.heads(hh_r, int(anchors[b]), k, trace=rtr)
                r = {"head_hidden_rel": rel(hh_f[b], hh_r), "drafts_fast": d_fast[b].tolist(), "drafts_ref": d_ref,
                     "conf_absdiff_max": float((c_fast[b].cpu() - c_ref.cpu()).abs().max())}
                for i in range(s.n_stages):
                    r[f"stage{i}_stream_rel"] = rel(tr[f"stage{i}.stream"].view(B, k, s.hc_mult, -1)[b],
                                                    rtr[f"stage{i}.stream"])
                    ef = tr[f"stage{i}.experts"].view(B, k, -1)[b].sort(-1).values
                    er = rtr[f"stage{i}.experts"].sort(-1).values
                    same = (ef == er).all(-1)
                    r[f"stage{i}_expert_sets_same"] = int(same.sum())
                    if mode == "shared":
                        totals["expert_sets"] += k
                        totals["expert_sets_same"] += int(same.sum())
                r["base_logits_rel"] = rel(base_f[b], rtr["base32"])
                # routing flips: first stage whose expert sets differ, and for each differing token the reference's
                # gap between its 3rd and 4th choice score (relative) against how far the two ffn inputs differ
                flip = None
                for i in range(s.n_stages):
                    ef = tr[f"stage{i}.experts"].view(B, k, -1)[b].sort(-1).values
                    er = rtr[f"stage{i}.experts"].sort(-1).values
                    bad = (~(ef == er).all(-1)).nonzero().flatten().tolist()
                    if bad:
                        xr = rtr[f"stage{i}.ffn_in"].float()
                        xf = tr[f"stage{i}.ffn_in"].view(B, k, -1)[b].float()
                        st_ = m.stages[i]
                        ch_r = torch.nn.functional.softplus(xr @ st_.gate_w.t()).sqrt() + st_.gate_bias
                        ch_f = torch.nn.functional.softplus(xf @ st_.gate_w.t()).sqrt() + st_.gate_bias
                        roots = []
                        for t in bad:
                            v = ch_r[t].topk(4).values
                            gap = float((v[2] - v[3]) / v[2].abs())
                            # how much the ffn-input perturbation moves the choice scores of this token
                            dscore = float((ch_f[t] - ch_r[t]).abs().max() / v[2].abs())
                            roots.append({"token": t, "ref_gap_3rd_4th_rel": gap, "score_shift_rel": dscore,
                                          "input_rel": rel(xf[t], xr[t]), "tie": gap <= 2 * dscore})
                        flip = {"stage": i, "roots": roots, "all_ties": all(x["tie"] for x in roots)}
                        break
                r["routing_flip"] = flip
                # teacher-forced: score every position with the fast path's previous tokens
                tf = []
                prev = int(anchors[b])
                for i in range(k):
                    e = ref.markov_embed[prev]
                    bias32 = ref.markov_head.float() @ e.float()
                    l32 = rtr["base32"][i] + bias32
                    l16 = rtr["base32"][i].to(torch.bfloat16) + bias32.to(torch.bfloat16)
                    pick_r = int(l16.argmax())
                    pick_f = int(d_fast[b, i])
                    fast_l = (base_f[b, i].float() + bias32.to(torch.bfloat16).float())
                    top = torch.topk(l32, 64).indices
                    noise = float((fast_l[top] - l32[top]).std())
                    margin = float(l32[pick_r] - l32[pick_f])
                    mag = float(l32[pick_r].abs())
                    bf16_step = 2.0 ** (int(torch.floor(torch.log2(torch.tensor(max(mag, 1e-6))))) - 7)
                    near = margin <= max(2 * bf16_step, 3 * noise)
                    if pick_r == pick_f:
                        cls = None
                    elif flip is None:
                        cls = "numeric_near_tie" if near else "unexplained"
                    else:
                        cls = "routing_tie_cascade" if flip["all_ties"] else "unexplained"
                    tf.append({"i": i, "same": pick_r == pick_f, "margin_ref32": margin, "logit_noise": noise,
                               "bf16_step": bf16_step, "class": cls})
                    if mode == "shared":
                        totals["pos"] += 1
                        if pick_r == pick_f:
                            totals["tf_same"] += 1
                        else:
                            totals["tf_" + cls] = totals.get("tf_" + cls, 0) + 1
                        if flip is None:
                            totals["pos_no_flip"] = totals.get("pos_no_flip", 0) + 1
                            totals["tf_same_no_flip"] = totals.get("tf_same_no_flip", 0) + int(pick_r == pick_f)
                    prev = pick_f
                r["teacher_forced"] = tf
                if mode == "shared":
                    totals["requests"] += 1
                    totals["requests_with_flip"] += int(flip is not None)
                    first = next((i for i in range(k) if d_ref[i] != int(d_fast[b, i])), None)
                    totals["e2e_same"] += k if first is None else first
                rq[mode] = r
            case["requests"].append(rq)
            print(json.dumps({"case": ci, "B": B, "k": k, "req": b, "shared_drafts_same":
                              rq["shared"]["drafts_fast"] == rq["shared"]["drafts_ref"],
                              "shared_hh_rel": round(rq["shared"]["head_hidden_rel"], 5),
                              "indep_hh_rel": round(rq["independent"]["head_hidden_rel"], 5),
                              "flip": None if rq["shared"]["routing_flip"] is None else
                              [(x["token"], round(x["ref_gap_3rd_4th_rel"], 5), round(x["score_shift_rel"], 5))
                               for x in rq["shared"]["routing_flip"]["roots"]],
                              "tf": [(t["same"], round(t["margin_ref32"], 3), t["class"])
                                     for t in rq["shared"]["teacher_forced"]]}), flush=True)
        report["cases"].append(case)
    # ---- plumbing check: the same fast-path code with exact fp32 linears / experts against the reference
    make_precise(m, ref)
    prec = []
    for ci, P in enumerate(lastp[:8]):
        P = min(P, len(ids) - 2)
        slots = torch.arange(1, device=dev)
        lp = P
        m.reset_slots(slots)
        rows = torch.arange(max(0, lp - W + 1), lp + 1, device=dev)
        m.insert(torch.zeros_like(rows), rows, m.main_x(taps[rows]))
        anchors = toks[torch.tensor([lp + 1], device=dev)]
        tr = {}
        d_p, c_p = m.draft(slots, anchors, torch.tensor([lp], device=dev), 5, trace=tr)
        rp = m.ring_pos[0]
        sel = (rp >= max(0, lp - W + 1)) & (rp <= lp)
        ring_ctx = [m.ring_kv[0, i][sel] for i in range(s.n_stages)]
        rtr = {}
        hh_r = ref.block(int(anchors[0]), lp, ring_ctx, 5, trace=rtr)
        d_r, c_r = ref.heads(hh_r, int(anchors[0]), 5, trace=rtr)
        kv_ref_same = ref.context_kv(m.main_x(taps[rows]), rows)
        e = {"P": lp, "main_x_vs_ref_rel": rel(m.main_x(taps[rows]), ref.main_x(taps[rows])),
             "ring_kv_vs_ref_rel": [rel(ring_ctx[i], kv_ref_same[i][rp[sel] - rows[0]]) for i in range(s.n_stages)],
             "stage_stream_rel": [rel(tr[f"stage{i}.stream"], rtr[f"stage{i}.stream"]) for i in range(s.n_stages)],
             "head_hidden_rel": rel(tr["head_hidden"], hh_r), "base_logits_rel": rel(tr["base_logits"][0], rtr["base32"]),
             "drafts_same": d_p[0].tolist() == d_r, "conf_absdiff": float((c_p[0].cpu() - c_r.cpu()).abs().max())}
        prec.append(e)
        print("PRECISE", json.dumps(e), flush=True)
    report["precise_plumbing_check"] = prec
    report["totals_shared_context"] = totals
    report["teacher_forced_agreement"] = totals["tf_same"] / max(totals["pos"], 1)
    report["teacher_forced_agreement_no_flip"] = totals.get("tf_same_no_flip", 0) / max(totals.get("pos_no_flip", 0), 1)
    report["end_to_end_prefix_agreement"] = totals["e2e_same"] / max(totals["pos"], 1)
    report["expert_set_agreement"] = totals["expert_sets_same"] / max(totals["expert_sets"], 1)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1)
    print("WROTE", out_path, json.dumps(totals), "tf_agree", round(report["teacher_forced_agreement"], 4),
          "tf_agree_no_flip", round(report["teacher_forced_agreement_no_flip"], 4),
          "e2e_prefix_agree", round(report["end_to_end_prefix_agreement"], 4), "expert_sets",
          round(report["expert_set_agreement"], 4))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print("seconds", round(time.time() - t0, 1))
