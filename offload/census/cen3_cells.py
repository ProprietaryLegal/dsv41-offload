"""CEN3 part (b) cell driver: window_cells' build / run_cell with CEN3's cells (coordinator only; dry plan by default).

  python3 -m offload.census.cen3_cells --execute build --output prompts.json
  python3 -m offload.census.cen3_cells --execute run --cell cen3-32k-b4 --cuts prompts.json --output cell.json \
      --deadline-utc <t> --telemetry <telemetry-sparkhead.jsonl> <telemetry-sparkworker-wired.jsonl> --capture-floor-kb <kb> \
      --cache-salt-prefix <tag>

offload/launch/windows/cen3_window.py calls it inside the HOLD through the owned tunnel 127.0.0.1:18001. The cells are
SP1's decode captures (offload/gb10_budget/window_cells.py run_cell: fresh cache salt, profile start/stop around the
decode, numbers and hashes only) at B1 short, B1 32K and B4 32K, plus the target-only capture taken while the x86host
drafter is down (prod_window.sh pair) and a recovery request after it returns. The profiler config stops each capture
after 64 iterations (CEN3-profile.env); window_cells stops it explicitly after DECODE_S.
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

CELLS = {c.name: c for c in (
    wc.Cell('warmup', ('warm-32k',), max_tokens=64, warmup=True, estimate_s=60),
    wc.Cell('cen3-short-b1', ('short',), ('decode',), estimate_s=120),
    wc.Cell('cen3-32k-b1', ('32k-0',), ('decode',), estimate_s=150),
    wc.Cell('cen3-32k-b4', wc.B4, ('decode',), estimate_s=240),
    # drafter down: every round is target-only (one live row), the profiler records >= 50 of them in its 64 iterations
    wc.Cell('cen3-target-only-short-b1', ('short',), ('decode',), estimate_s=150),
    # drafter back: the engine reconnects within its 5 s backoff cap, then remote rounds resume (rounds_ok and worker
    # steps grow); 1,500 tokens leave ~30 s of rounds after the reconnect; no capture
    wc.Cell('cen3-recovery-short-b1', ('short',), estimate_s=90))}
VERIFY = ('warmup', 'cen3-short-b1', 'cen3-32k-b1', 'cen3-32k-b4')
TARGET_ONLY, RECOVERY = 'cen3-target-only-short-b1', 'cen3-recovery-short-b1'


def plan() -> dict[str, Json]:
    return {'executed': False, 'verify': list(VERIFY), 'target_only': TARGET_ONLY, 'recovery': RECOVERY,
            'cells': {n: {'prompts': list(c.prompts), 'captures': list(c.captures), 'max_tokens': c.max_tokens,
                          'estimate_s': c.estimate_s} for n, c in CELLS.items()}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--url', default=wc.URL)
    ap.add_argument('--data-dir', type=Path, default=Path('/opt/dsv41/data/bench-data'))
    sub = ap.add_subparsers(dest='command')
    b = sub.add_parser('build')
    b.add_argument('--output', type=Path, required=True)
    r = sub.add_parser('run')
    r.add_argument('--cell', choices=tuple(CELLS), required=True)
    r.add_argument('--cuts', type=Path, required=True)
    r.add_argument('--output', type=Path, required=True)
    r.add_argument('--deadline-utc', type=float, required=True)
    r.add_argument('--telemetry', type=Path, nargs=2, required=True)
    r.add_argument('--capture-floor-kb', type=int, required=True)
    r.add_argument('--cache-salt-prefix', required=True)
    r.add_argument('--ledger', type=Path)
    args = ap.parse_args(argv)
    if not args.execute or args.command is None:
        print(json.dumps(plan()))
        return 0
    if args.url != wc.URL:
        ap.error('coordinator must use the reserved experiment tunnel at 127.0.0.1:18001')
    if args.output.exists():
        ap.error('refusing an existing output; use a fresh window directory')
    materials, hashes = corpus(args.data_dir)
    api = wc.Http(args.url)
    if args.command == 'build':
        result = wc.build(api, materials, hashes)
    else:
        if args.deadline_utc <= time.time():
            ap.error('--deadline-utc is in the past')
        if not 4 * 1024**2 <= args.capture_floor_kb <= 8 * 1024**2:
            ap.error('capture floor must lie in 4..8 GiB')
        texts = wc.texts_from(record(cast(Json, json.loads(args.cuts.read_text()))), materials, hashes)
        plan_ = wc.Plan(args.deadline_utc, args.capture_floor_kb, list(args.telemetry), args.cache_salt_prefix,
                        args.ledger)
        result = wc.run_cell(api, CELLS[args.cell], texts, plan_)
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, allow_nan=False) + '\n')
    print(f'{args.command}: wrote {args.output.name} (numbers and hashes only)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
