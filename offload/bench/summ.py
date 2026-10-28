#!/usr/bin/env python3
"""Compact per-run view of bench2 JSONL files (coordinator's quick look; report.py makes the comparison tables).

  python3 offload/bench/summ.py offload/results/A/*.jsonl
"""
import json
import statistics
import sys


def fmt(x: object, nd: int = 1) -> str:
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main() -> None:
    print("file | ctx | conc/mixed | cache | rep | per-request decode tok/s | ttft s | first visible s | reasoning/answer | finish | agg tok/s | acc/round | acc frac | step p50/p90 ms (events) | err")
    for path in sys.argv[1:]:
        reqs: dict[str, list[dict]] = {}
        runs = []
        for line in open(path):
            r = json.loads(line)
            if r.get("type") == "request":
                reqs.setdefault(r["run_id"], []).append(r)
            elif r.get("type") == "run":
                runs.append(r)
        for run in runs:
            rr = sorted(reqs.get(run["run_id"], []), key=lambda x: x.get("request_index", 0))
            spec = run.get("spec") or {}
            dec = "/".join(fmt(x.get("decode_tps")) for x in rr)
            ttft = "/".join(fmt(x.get("ttft_s"), 2) for x in rr)
            vis = "/".join(fmt(x.get("first_visible_s"), 1) for x in rr)
            ra = "/".join(f"{x.get('reasoning_tokens')}+{x.get('answer_tokens')}" for x in rr)
            fin = "/".join(str(x.get("finish")) for x in rr)
            steps = "/".join(f"{fmt(x['step_gap_ms']['p50'])}/{fmt(x['step_gap_ms']['p90'])} ({x['step_gap_ms']['n'] + 1})"
                             if x.get("step_gap_ms") else "-" for x in rr)
            print(f"{path.split('/')[-1]} | {run.get('target_ctx')} | {run.get('concurrency')}{'M' if run.get('mixed') else ''} | "
                  f"{rr[0].get('cache_state') if rr else '-'} | {run.get('rep')} | {dec} | {ttft} | {vis} | {ra} | {fin} | "
                  f"{fmt(run.get('aggregate_tps'))} | {fmt(spec.get('mean_accepted_per_round'), 2)} | "
                  f"{fmt(spec.get('acceptance_fraction'), 3)} | {steps} | {run.get('error') or ''}")


if __name__ == "__main__":
    main()
