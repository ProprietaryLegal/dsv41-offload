"""R-UPX gates and band cells for the UP3 runner (coordinator 2026-10-27), run as a subprocess over the experiment tunnel
like offload.up3.cells. Every gate and cell is offload/upx band.py's own function (imported, never copied); this module
only reads the window's own files, calls them and writes one JSON result. Dry by default; coordinator only.

Modes (registration upx-0p9: NDM's 0.9-nat greedy screen plus UP2's registered KL bound, replacing the never-built
resident-worker bitwise harness that parked 0064 and 0067 before HOLD):
  greedy       six W5/K1 prompts, nbw_cells.greedy at 0.9+1e-3 nats with every divergence's top-20 and prefix recorded
  regions      band.regions (public token regions); later starts must reproduce the first start's pins
  band-probes  band.kl_probes (one-token top-20 probes whose final prefill step has 32/33/42/43/63/64 rows)
  band-greedy  band.token_greedy on the 33-63-token prompts (no reference = record, the first control start)
  long-greedy  band.token_greedy on the 32,000- and 150,000-token prompts (0067)
  band-cell    band.run_band_cell x repeats (refuses a thin band census)
  kl           greedy_kl.compare_recorded over the candidate start's recorded divergences, at a registered bound
  needle       up2_request.needle at 150,000 tokens (0067)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import cast

from offload.gb10_budget import window_cells as wc
from offload.gb10_budget.capture import load_bench
from offload.launch.windows import greedy_kl as gk
from offload.launch.windows import nbw_cells as greedy_driver
from offload.launch.windows import up2_checks as ck
from offload.launch.windows import up2_request
from offload.prefill_maxlen.evidence import Json, number, record
from offload.prefill_trace.window_requests import corpus
from offload.up3 import cells
from offload.upx import band

REGISTRATION = 'upx-0p9'
MODES = ('greedy', 'regions', 'band-probes', 'band-greedy', 'long-greedy', 'band-cell', 'kl', 'needle')
NEEDLE_TARGET = 150000


def regions_of(path: Path) -> dict[str, list[int]]:
    return {name: band.token_ids(ids) for name, ids in record(ck.read(path)['regions']).items()}


def run(mode: str, api: wc.Http, args: argparse.Namespace) -> dict[str, Json] | None:
    output, salt, deadline = cast(Path, args.output), str(args.salt), float(args.deadline)
    ck.require(bool(salt.strip()), 'a cache salt prefix is required')
    if mode == 'greedy':
        result = greedy_driver.greedy(api, deadline, *greedy_driver.greedy_inputs(), output,
                                      near_tie_nats=band.NEAR_TIE_NATS, record_distribution=True, probe_salt_prefix=salt)
        result.update(near_tie_nats=band.NEAR_TIE_NATS, greedy_registration=REGISTRATION)
        return result
    if mode == 'needle':
        up2_request.needle(NEEDLE_TARGET, salt, output, cast(Path, args.ledger), deadline, cast(Path, args.data_dir))
        return None    # needle writes its own exclusive result file
    if mode == 'kl':
        rows = [r for path in cast(list[Path], args.candidate) for r in ck.rows(ck.read(path)['checks'])]
        bound = float(args.bound)
        ck.require(bound >= band.KL_FLOOR, 'KL bound below the registered floor')
        compared = gk.compare_recorded(api, rows, salt, bound)
        return {'rows': cast(Json, compared), 'bound': bound, 'compared': len(compared),
                'pass': all(r['pass'] is True for r in compared), 'registration': REGISTRATION}
    materials, hashes = corpus(cast(Path, args.data_dir))
    if mode == 'regions':
        built = band.regions(api, materials)
        pins = band.region_pins(built)
        if args.pins is not None:
            band.verify_regions(built, record(ck.read(cast(Path, args.pins))['pins']))
        return {'pins': pins, 'regions': cast(Json, built), 'corpus_sha256': hashes}
    regions = regions_of(cast(Path, args.regions))
    if mode == 'band-probes':
        return {'probes': band.kl_probes(api, regions['probe'], salt), 'lengths': cast(Json, list(band.PROBE_LENGTHS))}
    if mode in ('band-greedy', 'long-greedy'):
        prompts = band.greedy_prompts(regions['greedy']) if mode == 'band-greedy' else band.long_prompts(regions['long'])
        reference = None if args.reference is None else band.references(ck.read(cast(Path, args.reference)))
        return band.token_greedy(api, prompts, reference, salt)
    ck.require(mode == 'band-cell', f'unknown mode {mode}')
    cell = band.BAND_CELLS[str(args.cell)]
    texts = cells.texts(ck.read(cast(Path, args.cuts)), materials, hashes)
    counters = cast(band.Counters, load_bench())
    repetitions: list[Json] = []
    for repeat in range(int(args.repeats)):
        row = band.run_band_cell(cast(band.Api, api), counters, cell, texts, regions['tiny'],
                                 band.Plan(deadline, f'{salt}-repeat{repeat}'))
        row['repeat'] = repeat
        repetitions.append(row)
        output.write_text(json.dumps({'cell': cell.name, 'repetitions': repetitions}, allow_nan=False) + '\n')
    return {'cell': cell.name, 'max_tokens': cell.archive_max_tokens, 'repetitions': repetitions}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--salt', required=True)
    parser.add_argument('--regions', type=Path)
    parser.add_argument('--pins', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--candidate', type=Path, nargs='+')
    parser.add_argument('--bound', type=float)
    parser.add_argument('--cell', choices=tuple(band.BAND_CELLS))
    parser.add_argument('--repeats', type=int)
    parser.add_argument('--cuts', type=Path)
    parser.add_argument('--ledger', type=Path)
    args = parser.parse_args()
    needs = {'band-probes': ('regions',), 'band-greedy': ('regions',), 'long-greedy': ('regions',),
             'band-cell': ('regions', 'cell', 'repeats', 'cuts'), 'kl': ('candidate', 'bound'), 'needle': ('ledger',)}
    missing = [n for n in needs.get(args.mode, ()) if getattr(args, n) is None]
    ck.require(not missing, f'{args.mode}: missing {missing}')
    ck.require(args.repeats is None or 1 <= int(args.repeats) <= cells.MAX_REPEATS, 'repeats 1..12')
    if not args.execute:
        print(f'DRY: upx {args.mode} ({REGISTRATION}); no network or file')
        return 0
    ck.require(os.environ.get('WINDOW_COORDINATOR') == 'Deepseek/coordinator', 'coordinator required')
    ck.require(not args.output.exists() and time.time() < args.deadline, 'fresh output and live deadline required')
    result = run(args.mode, wc.Http(wc.URL), args)
    if result is not None:
        args.output.write_text(json.dumps(result, allow_nan=False) + '\n')
    else:
        ck.require(args.output.exists() and number(record(ck.read(args.output)).get('recall', 0)) == 6, 'needle result missing')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
