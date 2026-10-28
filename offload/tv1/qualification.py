"""Coordinator-only bounded candidate shadow traffic; public existing cuts, numeric output.

Dry by default. Runner must validate both workers' TV1_ENGINE_ORACLE receipts
AFTER this command and BEFORE warmup/timing. No profiling or promotion timing.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import cast

from offload.gb10_budget import window_cells as wc
from offload.prefill_maxlen.evidence import Json, record
from offload.prefill_trace.window_requests import corpus
from offload.tv1.cells import Http

CELLS = (
    wc.Cell('qual-150k-b1', ('150k',), max_tokens=64, warmup=True, estimate_s=120),
    wc.Cell('qual-32k-b2', ('32k-0', '32k-1'), max_tokens=128, warmup=True, estimate_s=75),
    wc.Cell('qual-32k-b3', ('32k-0', '32k-1', '32k-2'), max_tokens=128, warmup=True, estimate_s=90),
    wc.Cell('qual-32k-b4', ('32k-0', '32k-1', '32k-2', '32k-3'), max_tokens=128, warmup=True, estimate_s=90),
    wc.Cell('qual-150k-b4', ('150k',) * 4, max_tokens=512, warmup=True, estimate_s=360),
)
ESTIMATE_S = sum(c.estimate_s for c in CELLS)


def run(api: wc.Api, texts: dict[str, tuple[str, int]], plan: wc.Plan) -> dict[str, Json]:
    for key in ('150k', '32k-0', '32k-1', '32k-2', '32k-3'):
        minimum = 147000 if key == '150k' else 31360
        if key not in texts or texts[key][1] < minimum:
            raise ValueError('TV1 qualification public cuts do not meet occupied-context bound')
    observations: list[Json] = []
    for cell in CELLS:
        if time.time() + cell.estimate_s > plan.deadline:
            raise ValueError('TV1 qualification cell does not fit remaining submission budget')
        observations.append(cast(Json, wc.run_cell(api, cell, texts, plan)))
    return {'type': 'tv1-qualification-traffic-v2', 'observations': observations,
            'qualification_only': True, 'estimate_s': ESTIMATE_S}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--data-dir', type=Path, default=Path('/opt/dsv41/data/bench-data'))
    p.add_argument('--cuts', type=Path)
    p.add_argument('--deadline-utc', type=float)
    p.add_argument('--telemetry', type=Path, nargs=2)
    p.add_argument('--capture-floor-kb', type=int, default=4 * 1024**2)
    p.add_argument('--cache-salt-prefix')
    p.add_argument('--output', type=Path)
    a = p.parse_args(argv)
    if not a.execute:
        print(json.dumps({'executed': False, 'estimate_s': ESTIMATE_S,
                          'cells': [{'name': c.name, 'batch': c.batch, 'prompts': c.prompts,
                                     'max_tokens': c.max_tokens} for c in CELLS]}))
        return 0
    if (a.cuts is None or a.deadline_utc is None or a.telemetry is None
            or a.cache_salt_prefix is None or a.output is None):
        raise ValueError('TV1 qualification execute requires cuts/deadline/telemetry/salt/output')
    if a.output.exists() or not 4 * 1024**2 <= a.capture_floor_kb <= 8 * 1024**2:
        raise ValueError('TV1 qualification refuses existing output or invalid memory floor')
    material, hashes = corpus(a.data_dir)
    texts = wc.texts_from(record(cast(Json, json.loads(a.cuts.read_text()))), material, hashes)
    plan = wc.Plan(a.deadline_utc, a.capture_floor_kb, list(a.telemetry), a.cache_salt_prefix, None)
    result = run(Http(wc.URL), texts, plan)
    with a.output.open('x') as f:
        f.write(json.dumps(result, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
