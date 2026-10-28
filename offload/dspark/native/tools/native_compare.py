"""N1: two arms' drafts on the same captures (native_parity *_drafts.pt): block / per-position agreement per window,
and how often their routing (stage expert sets, window B) differs. Aggregates only.
Usage (x86host): python -m offload.dspark.native.tools.native_compare A_drafts.pt B_drafts.pt OUT.json
"""
from __future__ import annotations

import collections
import json
import sys

import torch


def main() -> None:
    a = torch.load(sys.argv[1], weights_only=False)
    b = torch.load(sys.argv[2], weights_only=False)
    if [r["round"] for r in a] != [r["round"] for r in b]:
        raise SystemExit("the two runs replayed different rounds")
    c = collections.Counter()
    for ra, rb in zip(a, b):
        for w in "AB":
            for da, db in zip(ra[w], rb[w]):
                c[f"{w}_blocks"] += 1
                c[f"{w}_block_equal"] += int(da == db)
                for i, (x, y) in enumerate(zip(da, db)):
                    c[f"{w}_pos{i}_equal"] += int(x == y)
                    if x != y:
                        c[f"{w}_first_diff_pos{i}"] += 1
                        break
        for i, (ea, eb) in enumerate(zip(ra["experts"], rb["experts"])):
            for ta, tb in zip(ea, eb):
                c[f"stage{i}_rows"] += 1
                c[f"stage{i}_expert_set_equal"] += int(sorted(ta) == sorted(tb))
    out = {"a": sys.argv[1].rsplit("/", 1)[-1], "b": sys.argv[2].rsplit("/", 1)[-1], "rounds": len(a)}
    for w in "AB":
        n = max(c[f"{w}_blocks"], 1)
        out[w] = {"blocks": c[f"{w}_blocks"], "block_equal": round(c[f"{w}_block_equal"] / n, 4),
                  "first_diff_pos": [c[f"{w}_first_diff_pos{i}"] for i in range(5)]}
    out["routing_rows_same_expert_set"] = {f"stage{i}": round(c[f"stage{i}_expert_set_equal"] /
                                                              max(c[f"stage{i}_rows"], 1), 4) for i in range(3)}
    with open(sys.argv[3], "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
