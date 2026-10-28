"""N1: realized acceptance of saved drafts (native_parity *_drafts.pt) against the target's own tokens in the
captures, paired with the engine's drafts of the same blocks, per window (A / B) and category. CPU only. Aggregates
only. Several drafts files: each arm is paired with the engine and with the first file (the reference arm).
Usage (x86host): python -m offload.dspark.native.tools.native_accept OUT.json DRAFTS.pt [DRAFTS.pt ...]
"""
from __future__ import annotations

import collections
import json
import math
import sys

import torch

from ...worker.tools.parity_capture import category
from .native_parity import CAPS, accept_len, load_rounds, realized_tokens


def paired(a: list[int], b: list[int]) -> dict:
    n = len(a)
    if not n:
        return {"n": 0}
    d = [x - y for x, y in zip(a, b)]
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / max(n - 1, 1))
    return {"n": n, "mean_a": round(sum(a) / n, 4), "mean_b": round(sum(b) / n, 4), "mean_diff": round(m, 4),
            "se_diff": round(sd / math.sqrt(n), 4), "a_better": sum(x > 0 for x in d), "b_better": sum(x < 0 for x in d)}


def main() -> None:
    out_path, files = sys.argv[1], sys.argv[2:]
    rounds = load_rounds(CAPS)
    real = realized_tokens(rounds)
    by_round = {int(r["round"]): r for r in rounds}
    runs = {f.rsplit("/", 1)[-1].replace("_drafts.pt", ""): torch.load(f, weights_only=False) for f in files}
    names = list(runs)
    L: dict = collections.defaultdict(list)        # (name|engine, window) -> per-block L (determined for all)
    keys = []
    for idx, recs in enumerate(zip(*runs.values())):
        rnd = recs[0]["round"]
        if any(r["round"] != rnd for r in recs):
            raise SystemExit("drafts files replayed different rounds")
        r = by_round[rnd]
        req_ids = list(r["req_ids"])
        nq = int(r["num_query_per_req"])
        qpos = r["query_positions"].view(len(req_ids), nq).long()
        eng = r["draft_tokens"].long()
        nsam = r["num_sampled"].tolist()
        for b, req in enumerate(req_ids):
            if category(req, nsam[b]) != "live":
                continue
            first = int(qpos[b, 0]) + 1
            le, oke = accept_len(eng[b].tolist(), first, real[req])
            for w in "AB":
                vals = [accept_len(rec[w][b], first, real[req]) for rec in recs]
                if not oke or not all(ok for _, ok in vals):
                    continue
                L[("engine", w)].append(le)
                for name, (la, _) in zip(names, vals):
                    L[(name, w)].append(la)
    rep = {"captures": CAPS, "arms": names, "category": "live"}
    for w in "AB":
        rep[w] = {"blocks": len(L[("engine", w)]),
                  "engine_mean_accept": round(sum(L[("engine", w)]) / max(1, len(L[("engine", w)])), 4)}
        for name in names:
            rep[w][name] = {"vs_engine": paired(L[(name, w)], L[("engine", w)]),
                            "vs_first_arm": paired(L[(name, w)], L[(names[0], w)]) if name != names[0] else None}
    with open(out_path, "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
