"""Markdown tables for TP2.md / the lane report from lane E's result files (no GPU).
Usage: python -m worker.tools.tp2_report RESULTS_DIR   (e3_bench_solo_v3.json, e7_bench_pair.json, e7_selftest_pair.json,
       e7_parity_pair.json, e7_pair_power.jsonl; missing files are skipped)
"""

from __future__ import annotations

import json
import os
import sys

CASES = ((1, 5), (1, 4), (2, 5), (2, 4), (3, 5), (3, 4), (4, 5), (4, 4))
LIVE = {(1, 5), (2, 5), (3, 4), (4, 4)}


def load(d, name):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def ms(us):
    return f"{us / 1000:.2f}"


def main() -> None:
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    solo = load(d, "e3_bench_solo_v3.json")
    pair = load(d, "e7_bench_pair.json")
    if pair:
        t = pair["tp2"]["draft"]
        print("### Draft step on the pair (graph replay p50, ms; rank 0's CUDA events incl. the input copies)\n")
        print("| requests, n_out | one card (O2, same bench) | TP2 pair | TP2 host wall | change | rank 0 kernels | "
              "rank 1 kernels | of which collectives (rank 0) |")
        print("|---|---|---|---|---|---|---|---|")
        for B, n in CASES:
            k = f"B{B}_k5_n{n}"
            r = t[k]
            one = solo["one_card"]["draft"][k]["graph"]["p50_us"] if solo else None
            g = r["graph"]["p50_us"]
            bd = r["graph_breakdown"]
            k0 = bd.get("device0", {}).get("kernel_us_per_step")
            k1 = bd.get("device1", {}).get("kernel_us_per_step")
            p2p = bd.get("device0", {}).get("us", {}).get("p2p")
            star = "**" if (B, n) in LIVE else ""
            ch = f"{(g - one) / one * 100:+.0f} %" if one else ""
            print(f"| {star}{B}, {n}{star} | {ms(one) if one else ''} | {star}{ms(g)}{star} | "
                  f"{ms(r.get('host_wall_p50_us', 0))} | {ch} | {ms(k0) if k0 else ''} | {ms(k1) if k1 else ''} | "
                  f"{ms(p2p) if p2p else ''} |")
        print("\n### Context insert (graph p50, us, both ranks, incl. the taps' upload)\n")
        print("| rows | " + " | ".join(r_[4:] for r_ in pair["tp2"]["insert"]) + " |")
        print("|---|" + "---|" * len(pair["tp2"]["insert"]))
        print("| TP2 | " + " | ".join(str(v["p50_us"]) for v in pair["tp2"]["insert"].values()) + " |")
        if solo:
            print("| one card | " + " | ".join(str(v["p50_us"]) for v in solo["one_card"]["insert"].values()) + " |")
        print("\n### NVLink collectives inside CUDA graphs (us per collective, p50 of 20 back-to-back)\n")
        print("| collective | size | p50 | p10-p90 | bytes each way |")
        print("|---|---|---|---|---|")
        for k, v in pair["collectives"].items():
            name, sz = k.rsplit("_", 1)
            unit = "rows x 5120 fp32" if name.startswith("ar") else "requests"
            print(f"| {name} | {sz} {unit} | {v['p50_us']} | {v['p10_us']}-{v['p90_us']} | {v['bytes_each_way']} |")
        print("\nNVLink data during the run (KiB):", json.dumps(pair.get("nvlink_kib_during_run")))
        print("\nComm headers at the end:", pair.get("comm_headers"))
        print("\nGPU before / after:", json.dumps(pair.get("gpu_before")), json.dumps(pair.get("gpu_after")))
        mem = pair.get("memory", {})
        for r in ("rank0", "rank1"):
            if r in mem:
                m = mem[r]
                print(f"\n{r}: torch allocated {m['torch_allocated'] / 2**20:.0f} MiB, reserved "
                      f"{m['torch_reserved'] / 2**20:.0f} MiB, peak allocated {m['torch_max_allocated'] / 2**20:.0f} MiB")
        print("\nnvidia-smi apps:\n" + pair.get("memory_nvidia_smi_mib", ""))
    st = load(d, "e7_selftest_pair.json")
    if st:
        print("### Server selftest on the pair\n")
        print({k: st[k] for k in ("hello_live", "exact_vs_shadow", "n_out4_vs_full", "rollback_n0", "partial_window",
                                  "bad_entry", "stale_new_gen", "ping_rdma", "ping_tcp") if k in st})
        print("\n| STEP | in-process handle p50 | RDMA rtt p50 (p99) | RDMA worker p50 | TCP rtt p50 |")
        print("|---|---|---|---|---|")
        tm = st["timing"]
        for nreq, tag in ((1, ""), (2, ""), (3, ""), (4, ""), (3, "_nout4"), (4, "_nout4")):
            ip = tm.get(f"in_process_{nreq}req{tag}_us", {})
            rd = tm.get(f"ofl_rdma_{nreq}req{tag}_us", {})
            tc = tm.get(f"ofl_tcp_{nreq}req{tag}_us", {})
            print(f"| {nreq} req{tag} | {ip.get('p50', '')} | {rd.get('rtt_p50', '')} ({rd.get('rtt_p99', '')}) | "
                  f"{rd.get('worker_us_p50', '')} | {tc.get('rtt_p50', '')} |")
    par = load(d, "e7_parity_pair.json")
    if par:
        print("\n### Parity on the pair\n")
        for mode in "AB":
            t = par["summary"][mode]["live"]
            print(mode, {k: v for k, v in t.items() if not k.startswith("pos")})
        print(par["first_mismatch_by_class"])
    pw = os.path.join(d, "e7_pair_power.jsonl")
    if os.path.exists(pw):
        rows = [json.loads(l) for l in open(pw) if l.strip()]
        tot = [r["total"] for r in rows if "total" in r]
        per = {c: max(r["w"][c] for r in rows if "w" in r) for c in ("0", "1", "2", "3")} if tot else {}
        print(f"\npower: {len(tot)} samples, total max {max(tot) if tot else 0} W, per card max {per}; verdicts "
              f"{[r for r in rows if 'verdict' in r]}")


if __name__ == "__main__":
    main()
