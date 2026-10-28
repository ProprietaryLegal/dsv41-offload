#!/usr/bin/env python3
"""Markdown tables from logs/bench-*.jsonl (median and range per group)."""
import json, statistics as st, sys, collections
def load(f): return [json.loads(l) for l in open(f)]
def cap(r):
    t = r["ts"][11:19]
    c = r["gpu_clock_med"]["spark"]
    return "stock (~2450)" if c and c > 2400 else ("2350" if c and c > 2300 else ("2250" if c and c > 2200 else "2150"))
groups = collections.OrderedDict()
for f in sys.argv[1:]:
    for r in load(f):
        if r.get("error") or not r.get("decode_tps"): continue
        k = (f.split("/")[-1], r.get("filler", "synthetic") if r["target_ctx"] else "-", r["target_ctx"], r["workload"], r["thinking"], cap(r))
        groups.setdefault(k, []).append(r)
print("| file | filler | target ctx | workload | thinking | clock cap | n | actual prompt tokens | TTFT s (med) | prefill tok/s (med) | **decode tok/s median** (range) | tokens/round (med) | min MemAvail head / worker GiB | max GPU °C head / worker |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
m = lambda xs: st.median(xs)
for k, rs in groups.items():
    d = [r["decode_tps"] for r in rs]
    tpr = [r["spec"]["tokens_per_round"] for r in rs if r.get("spec") and r["spec"].get("tokens_per_round")]
    print(f"| {k[0]} | {k[1]} | {k[2]:,} | {k[3]} | {'on' if k[4] else 'off'} | {k[5]} | {len(rs)} | "
          f"{min(r['prompt_tokens'] for r in rs):,}–{max(r['prompt_tokens'] for r in rs):,} | {m([r['ttft_s'] for r in rs]):.1f} | "
          f"{m([r['prefill_tps'] for r in rs]):,.0f} | **{m(d):.1f}** ({min(d):.1f}–{max(d):.1f}) | {m(tpr):.2f} | "
          f"{min(r['min_avail_gib']['spark'] for r in rs):.1f} / {min(r['min_avail_gib']['sparkworker'] for r in rs):.1f} | "
          f"{max(r['max_temp_c']['spark'] for r in rs)} / {max(r['max_temp_c']['sparkworker'] for r in rs)} |")
