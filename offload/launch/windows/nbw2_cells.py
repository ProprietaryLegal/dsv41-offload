"""NBW v2 KL driver (WIN2 greedy refinement), reported only. Dry by default.

The I starts' greedy checks run through the shared `nbw_cells` command line at NDM's 0.9-nat limit with the divergence
top-20 recorded (`--near-tie-nats 0.901 --record-distribution --probe-salt-prefix`, WIN2-R2). This driver runs on the
FOLLOWING K start and re-probes those prefixes with a fresh cache salt (`greedy_kl.compare_recorded`): REPORTED ONLY,
never a stop, because NBW has no measured A/A floor yet; `KL_NOMINAL_BOUND` is UP2's floor, used only to label rows.
"""
from __future__ import annotations

import argparse
import os
import json
import time
from pathlib import Path

from offload.gb10_budget import window_cells as wc
from offload.launch.windows import greedy_kl
from offload.launch.windows import nbw_cells as cells
from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, record

KL_NOMINAL_BOUND = 0.005


def kl(api: wc.Http, candidate: Path, salt: str) -> dict[str, Json]:
    rows = [record(r) for r in ck.rows(ck.read(candidate)['checks'])]
    ck.require([r.get('i') for r in rows] == list(range(6)), 'candidate greedy checks missing')
    out: list[Json] = []
    for row in rows:
        try:  # one divergence at a time: a probe or estimator refusal is recorded, never a window stop
            out += [dict(r) for r in greedy_kl.compare_recorded(api, [row], salt, KL_NOMINAL_BOUND)]
        except (ValueError, OSError) as exc:
            out.append({'i': row.get('i'), 'error': f'{type(exc).__name__}: {exc}'})
    return {'mode': 'kl', 'status': 'REPORTED ONLY (no NBW A/A floor; never a stop)', 'nominal_bound': KL_NOMINAL_BOUND,
            'divergences': sum(1 for r in rows if r.get('identical') is not True), 'rows': out}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--mode', choices=('kl',), required=True)
    ap.add_argument('--deadline-utc', type=float, default=0)
    ap.add_argument('--salt', default='')
    ap.add_argument('--candidate', type=Path, required=True, help='the I start greedy output (record_distribution)')
    args = ap.parse_args(argv)
    if not args.execute:
        print('DRY RUN: re-probe the recorded I-start greedy divergences on this K start (KL, reported only); '
              'no network or file')
        return 0
    ck.require(os.environ.get('WINDOW_COORDINATOR') == 'Deepseek/coordinator', 'coordinator required')
    ck.require(not args.output.exists(), 'fresh output required')
    ck.require(time.time() < args.deadline_utc, 'submission deadline reached')
    ck.require(bool(args.salt.strip()), 'probe cache salt required')
    result = kl(cells.Http(wc.URL), args.candidate, args.salt)
    args.output.write_text(json.dumps(result, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
