"""DP-5 phase 1: tabulate dp5_clocks sweep runs (stdlib only; runs anywhere).

Usage: python3 dp5_report.py RUN1.json RUN2.json [...]
Prints, per clock setting and live shape, handle p50/p95 (us) per run, draft-graph p50, the SM clock just before the STEP,
and the p50/p95 difference from the default idle-gap setting (0,1950) measured in the same run.
"""

from __future__ import annotations

import json
import sys

SHAPES = ("1req_nout5", "2req_nout5", "3req_nout4", "4req_nout4")


def label(s: dict) -> str:
    return f"{s['lgc']}{'' if s['gap'] else ' (no gap)'}"


def load(path: str) -> dict[str, list[dict]]:
    rep = json.load(open(path))
    out: dict[str, list[dict]] = {}
    for _key, s in sorted(rep["settings"].items()):
        out.setdefault(label(s), []).append(s["shapes"])
    return out


def main() -> None:
    runs = [load(p) for p in sys.argv[1:]]
    settings = list(dict.fromkeys(k for r in runs for k in r))
    base = "0,1950"
    print("| setting | shape | handle p50 (us) per run | handle p95 (us) per run | Δp50 vs 0,1950 | Δp95 vs 0,1950 |"
          " draft_gpu p50 | SM MHz before STEP p50 |")
    print("|---|---|---|---|---|---|---|---|")
    for st in settings:
        for sh in SHAPES:
            p50s, p95s, d50, d95, dg, clk = [], [], [], [], [], []
            for r in runs:
                for shapes in r.get(st, []):
                    x = shapes[sh]
                    p50s.append(x["handle"]["p50"])
                    p95s.append(x["handle"]["p95"])
                    dg.append(x["draft_gpu"]["p50"])
                    clk.append(x["sm_mhz_before"]["p50"])
                    b = r[base][0][sh]["handle"]     # first default occurrence in the same run
                    d50.append(x["handle"]["p50"] - b["p50"])
                    d95.append(x["handle"]["p95"] - b["p95"])
            if not p50s:
                continue
            f = lambda xs: " / ".join(f"{v:.0f}" for v in xs)   # noqa: E731
            print(f"| {st} | {sh} | {f(p50s)} | {f(p95s)} | {f(d50)} | {f(d95)} | {f(dg)} | {f(clk)} |")


if __name__ == "__main__":
    main()
