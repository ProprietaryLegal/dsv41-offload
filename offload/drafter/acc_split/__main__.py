"""ACC-SPLIT CLI: index every bench2 run and window cell, write acc-split.json and acc-split.md (numbers and hashes only).

    python3 -m offload.drafter.acc_split --campaign /opt/dsv41/campaigns/DSV41-OFFLOAD \
        --bench2-root offload/results --bench2-root /opt/dsv41/campaigns/DSV41-OFFLOAD --out <dir>

CPU only and read-only on its inputs; it refuses to overwrite an existing output file.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from offload.drafter.acc_split import analysis, sources
from offload.prefill_maxlen.evidence import Json, record

# the three shifts DP-2 must explain: (old prompt, new prompt, old budget, new budget); prompt names as prompt_names()
SHIFTS = {'32K': ('bench2:summary-32000', 'corpus:32k-b1', 1500, 512),
          'short': ('bench2:legal-0', 'corpus:short-b1', 1500, 512),
          '150K': ('bench2:synth-150000', 'corpus:150k-b1', 1500, 512)}


def _fmt(s: Json) -> str:
    r = record(s)
    ci = cast(list[float], r['ci95'])
    return f"{r['mean']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] n={r['n']}"


def markdown(rep: dict[str, Json]) -> str:
    lines = ['# ACC-SPLIT tables (MEASURED unless marked ESTIMATED)', '',
             f"Counts: {json.dumps(rep['counts'])}", '', '## Acceptance by prompt and answer budget (B1, controls)', '',
             '| prompt | prompt tokens | max_tokens | accepted/round (95 % CI, n) | depth | per-position rates 0..4 | dates |',
             '|---|---|---|---|---|---|---|']
    for row in cast(list[Json], rep['by_prompt']):
        r = record(row)
        lines.append(f"| {r['prompt']} | {r['prompt_tokens']} | {r['max_tokens']} | {_fmt(r['accepted_per_round'])} | "
                     f"{r['depth']} | {r['per_position']} | {', '.join(cast(list[str], r['dates']))} |")
    lines += ['', '## Shift decomposition', '', '| shift | from | via | to | prompt step | length/date step | total |',
              '|---|---|---|---|---|---|---|']
    for name, d in record(rep['shifts']).items():
        r = record(d)
        lines.append(f"| {name} | {_fmt(r['from'])} | {_fmt(r['via'])} | {_fmt(r['to'])} | {r['prompt_step']} | "
                     f"{r['length_step']} | {r['total']} |")
    lines += ['', '## Metric reconciliation (bench2 B1, means)', '',
              '| prompt | n | server | tok/s x p50 gap - 1 | tok/s x mean gap - 1 | tokens/SSE event - 1 | rounds/event |',
              '|---|---|---|---|---|---|---|']
    for row in cast(list[Json], rep['metric_reconciliation']):
        r = record(row)
        lines.append(f"| {r['prompt']} | {r['n']} | {r['server']} | {r['tps_x_p50_gap']} | {r['tps_x_mean_gap']} | "
                     f"{r['tokens_per_sse_event']} | {r['rounds_per_sse_event']} |")
    seg = record(rep['segment_split'])
    lines += ['', '## Segment split (ESTIMATED)', '', f"SSE validation on {seg['validation_runs']} reasoning-only runs: "
              f"SSE minus server {json.dumps(seg['validation_sse_minus_server'])}", '',
              '| prompt | run | method | reasoning tokens | answer tokens | server (all) | reasoning | answer |',
              '|---|---|---|---|---|---|---|---|']
    for row in cast(list[Json], seg['mixed_runs']):
        r = record(row)
        lines.append(f"| {r['prompt']} | {r['label']} {r['date']} | {r['method']} | {r['reasoning_tokens']} | "
                     f"{r['answer_tokens']} | {r['server_all']} | {r['reasoning_estimated']} | {r['answer_estimated']} |")
    lines += ['', '## Batch split (window controls)', '', '| cell | max_tokens | batch | accepted/round | depth | per-position |',
              '|---|---|---|---|---|---|']
    for row in cast(list[Json], rep['batch_split']):
        r = record(row)
        lines.append(f"| {r['cell']} | {r['max_tokens']} | B{r['batch']} | {_fmt(r['accepted_per_round'])} | {r['depth']} | "
                     f"{r['per_position']} |")
    lines += ['', '## Per window (B1 controls; drift within and across days)', '',
              '| prompt | max_tokens | date | window | mean (n) |', '|---|---|---|---|---|']
    for row in cast(list[Json], rep['by_window']):
        r = record(row)
        lines.append(f"| {r['prompt']} | {r['max_tokens']} | {r['date']} | {r['window']} | {r['mean']} ({r['n']}) |")
    return '\n'.join(lines) + '\n'


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--campaign', type=Path, required=True, help='folder holding the window directories')
    ap.add_argument('--bench2-root', type=Path, action='append', required=True, help='searched recursively for bench2 jsonl')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    obs = (sources.load_bench2(sources.bench2_files(args.bench2_root))
           + sources.load_windows(sources.window_dirs(args.campaign)))
    rep = analysis.report(obs, SHIFTS)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in (('acc-split.json', json.dumps(rep, indent=1, sort_keys=True) + '\n'), ('acc-split.md', markdown(rep))):
        path = args.out / name
        if path.exists():
            raise SystemExit(f'{path} exists; refusing to overwrite')
        path.write_text(text)
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
