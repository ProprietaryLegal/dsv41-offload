"""TV1 numeric-only cells using SP1 prompt/stream/profile helpers. Dry by default."""
from __future__ import annotations

import argparse
import json
import time
import threading
from dataclasses import replace
from pathlib import Path
from typing import cast

from offload.gb10_budget import window_cells as wc
from offload.gb10_budget.capture import load_bench
from offload.prefill_maxlen.evidence import Json, record
from offload.prefill_trace.window_requests import corpus
from offload.tv1 import metrics, greedy

CELLS = {name: wc.Cell(name, row.prompts, ('decode',) if name.startswith('prof-') else (),
                       max_tokens=512 if not row.warmup else 64, warmup=row.warmup,
                       estimate_s={'warmup': 30, 'ctl-short-b1': 15, 'ctl-32k-b1': 30,
                                   'ctl-150k-b1': 85, 'ctl-32k-b4': 100,
                                   'prof-short-b1': 20, 'prof-32k-b1': 40,
                                   'prof-150k-b1': 95, 'prof-32k-b4': 115}[name])
         for name, row in wc.CELLS.items()}
ORDER = tuple(CELLS)


def controls(feature: str) -> tuple[str, str]:
    if feature not in ('candidate', 'compact'):
        raise ValueError('TV1 one known feature per HOLD')
    return ('short-b1', '150k-b1' if feature == 'candidate' else '32k-b4')


def order(feature: str) -> tuple[str, ...]:
    short, target = controls(feature)
    return ('warmup', 'ctl-'+short, 'ctl-'+target, 'prof-'+short, 'prof-'+target)


class Http(wc.Http):
    def stream(self, body: dict[str, Json], first: threading.Event) -> dict[str, Json]:
        # A full fixed output budget keeps all four streams alive for the decode
        # capture. This is identical in controls/ON and does not affect greedy checks.
        return super().stream({**body, 'ignore_eos': True}, first)


def repetitions(cell: wc.Cell) -> int:
    return 1 if cell.warmup or cell.captures else 4 // cell.batch


def measure(api: wc.Api, cell: wc.Cell, texts: dict[str, tuple[str, int]], plan: wc.Plan) -> dict[str, Json]:
    count = repetitions(cell)
    if time.time() + count*cell.estimate_s > plan.deadline:
        raise ValueError('TV1 complete repeated cell does not fit submission deadline')
    trials: list[dict[str, Json]] = []
    bench = cast(wc.Counters, load_bench())
    for index in range(count):
        trial_plan = replace(plan, salt_prefix=plan.salt_prefix+f'-rep{index}')
        before = bench.parse_metrics(api.metrics())
        result = wc.run_cell(api, cell, texts, trial_plan)
        after = bench.parse_metrics(api.metrics())
        for stream in cast(list[Json], result['streams']):
            row = record(stream)
            if row.get('finish') != 'length' or record(row['usage']).get('completion_tokens') != cell.max_tokens:
                raise ValueError('TV1 timed cell did not complete its forced output budget')
        result['forced_output_tokens'] = cell.max_tokens
        if not cell.warmup and not cell.captures:
            result['server_counters'] = metrics.counters(before, after, cell.batch)
            result['repetition'] = index
            result['cache_salt_prefix'] = trial_plan.salt_prefix
            trials.append(result)
    if trials:
        # Keep the original per-batch streams/captures shape for SP1 collection;
        # all independent control trials and raw snapshots travel alongside it.
        return {**result, 'type': 'tv1-repeated-cell-v3', 'trials': cast(Json, trials),
                'pooled': metrics.pool(trials), 'request_count': count*cell.batch}
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--data-dir', type=Path, default=Path('/opt/dsv41/data/bench-data'))
    sub = p.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build')
    build.add_argument('--output', type=Path, required=True)
    run = sub.add_parser('run')
    run.add_argument('--cell', choices=ORDER, required=True)
    run.add_argument('--cuts', type=Path, required=True)
    run.add_argument('--deadline-utc', type=float, required=True)
    run.add_argument('--telemetry', type=Path, nargs=2, required=True)
    run.add_argument('--capture-floor-kb', type=int, required=True)
    run.add_argument('--cache-salt-prefix', required=True)
    run.add_argument('--ledger', type=Path)
    run.add_argument('--output', type=Path, required=True)
    checks = sub.add_parser('greedy')
    checks.add_argument('--deadline-utc', type=float, required=True)
    checks.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    if not a.execute:
        print(json.dumps({'executed': False, 'cells': list(CELLS), 'greedy_references': 6,
                          'registration_id': greedy.REGISTRATION_ID, 'greedy_rule': greedy.RULE}))
        return 0
    if a.output.exists():
        raise ValueError('TV1 output must be new')
    api = Http(wc.URL)
    result: dict[str, Json]
    if a.command == 'greedy':
        greedy.run(api, a.deadline_utc, a.output)
        return 0  # NBW writes each partial check and the registered final receipt.
    else:
        material, hashes = corpus(a.data_dir)
        if a.command == 'build':
            result = wc.build(api, material, hashes)
        else:
            if a.deadline_utc <= time.time() or not 4*1024**2 <= a.capture_floor_kb <= 8*1024**2:
                raise ValueError('TV1 past deadline or invalid capture floor')
            cuts = record(cast(Json, json.loads(a.cuts.read_text())))
            texts = wc.texts_from(cuts, material, hashes)
            plan = wc.Plan(a.deadline_utc, a.capture_floor_kb, list(a.telemetry), a.cache_salt_prefix, a.ledger)
            result = measure(api, CELLS[a.cell], texts, plan)
    with a.output.open('x') as handle:
        handle.write(json.dumps(result, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
