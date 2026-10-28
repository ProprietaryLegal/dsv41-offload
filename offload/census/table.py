"""CEN0 deliverable: the re-analysed SP1 / KL1 census table from existing evidence (no window).

  PYTHONPATH=. python3 -m offload.census.table --campaign /opt/dsv41/campaigns/DSV41-OFFLOAD \
      --sp1 SP1-W-20261026T144153Z SP1-W-20261026T160012Z SP1-W-20261026T191247Z --kl1w KL1-W-20261026T194927Z \
      --output CEN0-census.json --markdown CEN0-census.md

Runs, read only: the census waterfall (offload/census/waterfall.py) on every SP1 window's decode captures; the
logical-bytes floor and per-round bytes (bytes_model.py); KL1-W's isolated rates as logical-byte rates and, with
KL1-bandwidth.json present, KL1's own smoe_bench.analyze re-run to reproduce its 226.6 GB/s; the control archive's
resolution at 2150 and 2300 MHz (costtable.py). Each window's clock cap is read from its saved start log. Output
paths are relative to --campaign and never overwritten. Numbers, kernel-family names and hashes only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from offload.census import bytes_model as BM
from offload.census import costtable as CT
from offload.census import rules as C
from offload.census import waterfall as W
from offload.kl1 import shapes as S
from offload.kl1 import smoe_bench as SB
from offload.prefill_maxlen.evidence import Json, number, record

CAP = re.compile(r'clock cap (\d+) MHz')
KL1_INFERRED = 13.0                  # KL1-W analysis-rank0.json median inferred distinct experts per layer at B1
KL1_PRODUCTION = ('SP1-W-20261026T160012Z', 0)   # the production window/rank KL1's analysis inverted
# Platform state of each window (LANES.md 2026-10-26 15:11 / 16:25 / 19:39 COORD rows)
PLATFORM = {'SP1-W-20261026T144153Z': 'pre-FW1 (both Sparks 6.17 / 580.159)',
            'SP1-W-20261026T160012Z': 'post-FW1 (sparkhead 7.0 / 580.178, sparkworker 6.17 / 580.159)',
            'SP1-W-20261026T191247Z': 'post-FW2 (both 7.0 / 580.178), pre-FW3/FW3b'}


def clock_caps(root: Path) -> list[int]:
    caps: set[int] = set()
    for log in sorted((root / 'cmd').glob('*-start.log')):
        caps.update(int(m.group(1)) for m in CAP.finditer(log.read_text(errors='replace')))
    return sorted(caps)


def kl1_reproduction(campaign: Path, kl1w: Path) -> dict[str, Json]:
    """KL1's own analysis (offload.kl1.smoe_bench.analyze) on the same inputs: sparkhead = rank 0, sparkworker replicate."""
    bw_path = campaign / 'KL1-bandwidth.json'
    if not bw_path.exists():
        return {'status': 'SKIPPED: KL1-bandwidth.json absent'}
    bench: list[dict[str, Any]] = []
    replicate: list[dict[str, Any]] = []
    for host, sink in (('sparkhead', bench), ('sparkworker-wired', replicate)):
        for p in sorted((kl1w / 'results' / host).glob('*.json')):
            doc = json.loads(p.read_text())
            if isinstance(doc, dict) and doc.get('type') == 'kl1-smoe-bench':
                sink.append(doc)
    bw = json.loads(bw_path.read_text())
    window, rank = KL1_PRODUCTION
    steps, prov = SB.production_selection(bw, window, rank, campaign / window)
    res = SB.analyze(bench, steps, window=window, rank=rank, production_stack=(prov['kernel'], prov['driver']),
                     replicate=replicate)
    inferred = [float(p['distinct']) for p in res['production'] if p['where'] == 'inside']
    return {'bandwidth_sha256': hashlib.sha256(bw_path.read_bytes()).hexdigest(),
            'median_isolated_gbps_in_range': res['median_isolated_gbps_in_range'],
            'in_range_steps': res['in_range_steps'],
            'inferred_distinct_median': statistics.median(inferred) if inferred else None,
            'inferred_distinct_range': cast(Json, [min(inferred), max(inferred)] if inferred else None),
            'gate2_variant_pass_both_sparks': cast(Json, res['gate2_variant_pass_both_sparks']),
            'status': 'ESTIMATED: KL1 analyze re-run (isolated rate at an inferred count, logical bytes)'}


def round_floors(windows: Mapping[str, Json]) -> list[dict[str, Json]]:
    """Per decode capture: logical bytes of its median-rows round at KL1's inferred distinct count and the streaming
    floor at 226.6 GB/s against the measured period (ESTIMATED non-streaming share of a verify round)."""
    out: list[dict[str, Json]] = []
    for wname, w in windows.items():
        for label, cap in record(record(w)['captures']).items():
            c = record(cap)
            if 'by_rows' not in c:
                continue
            hist = {int(k): int(number(v)) for k, v in record(c['rows_hist']).items()}
            rows = sorted(r for r, n in hist.items() for _ in range(n))
            median_rows = rows[len(rows) // 2]
            requests = 4 if '-b4' in label else 1
            distinct = min(KL1_INFERRED * requests, S.unique_uniform(median_rows))
            b = BM.step_bytes(distinct, median_rows)
            period = number(c['period_ms_mean'])
            fl = b['total'] / BM.KL1_ISOLATED_GBPS / 1e6
            out.append({'window': wname, 'capture': label, 'median_rows': median_rows, 'distinct': distinct,
                        'logical_gb': b['total'] / 1e9, 'floor_ms': fl, 'period_ms': period,
                        'non_streaming_share': (period - fl) / period,
                        'note': 'B4 distinct = 4 x 13 (independent requests, upper end)' if requests > 1 else ''})
    return out


def build(campaign: Path, sp1: Sequence[str], kl1w: str | None, archive: Path | None) -> dict[str, Json]:
    windows: dict[str, Json] = {}
    meta: dict[str, Json] = {}
    for name in sp1:
        root = campaign / name
        windows[name] = W.window(root)
        meta[name] = {'clock_caps_mhz': cast(Json, clock_caps(root)), 'platform': PLATFORM.get(name, 'unrecorded')}
    out: dict[str, Json] = {'type': 'cen0-census-table', 'rules_sha256': C.SHA256,
                            'waterfall': W.result(windows), 'window_meta': meta,
                            'round_floors': cast(Json, round_floors(windows)),
                            'floor_plan_one_mhc': BM.floor(1), 'floor_two_mhc': BM.floor(BM.HC_PRENORM_PER_LAYER)}
    if kl1w is not None:
        out['kl1w_rates'] = BM.kl1_window(campaign / kl1w)
        out['kl1_reproduction'] = kl1_reproduction(campaign, campaign / kl1w)
    if archive is not None:
        rows = CT.parse(archive.read_text().splitlines())
        out['resolution'] = {f'{metric}@{clock}': CT.resolution(rows, clock, metric)
                             for clock in (2150, 2300) for metric in ('decode_tps', 'ms_per_round')}
        out['archive_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
    return out


def _pct(x: float) -> str:
    return f'{100 * x:.1f} %'


def markdown(res: Mapping[str, Json]) -> str:
    lines = ['# CEN0 census table (re-analysed SP1 / KL1 evidence)', '',
             'Labels: MEASURED = profiled kineto durations or archive values; ESTIMATED = logical bytes, inferred '
             'counts, idle and broadcast-wait attribution. Profiled shares are attribution, never promotion timings.',
             '']
    meta = record(res['window_meta'])
    lines.append('| window | platform | clock cap (start log) |')
    lines.append('|---|---|---|')
    for name, m in meta.items():
        lines.append(f"| {name} | {record(m)['platform']} | {record(m)['clock_caps_mhz']} |")
    lines += ['', '## Waterfall per verify round (both-rank mean, ms per round and share of period; MEASURED)', '',
              W.markdown(record(res['waterfall']))]
    lines += ['## Verify-round logical bytes and streaming floor (ESTIMATED)', '',
              '| window | capture | median rows | distinct/layer | logical GB | floor @226.6 ms | period ms | '
              'non-streaming |', '|---|---|---|---|---|---|---|---|']
    for r in map(record, cast(list[Json], res['round_floors'])):
        lines.append(f"| {str(r['window'])[-8:]} | {str(r['capture']).removesuffix(':decode')} | {r['median_rows']} | "
                     f"{number(r['distinct']):.0f} | {number(r['logical_gb']):.2f} | {number(r['floor_ms']):.1f} | "
                     f"{number(r['period_ms']):.1f} | {_pct(number(r['non_streaming_share']))} |")
    lines += ['', '## Target-only token floor (F-CENSUS §2.2; ESTIMATED)', '',
              '| mHC weights per layer | GB per token | floor @226.6 / 230 / 273 ms | non-streaming @24.3 ms / 27.7 ms '
              '(226.6) | same (273) |', '|---|---|---|---|---|']
    for key in ('floor_plan_one_mhc', 'floor_two_mhc'):
        f = record(res[key])
        fl = record(f['floor_ms'])
        sh = record(f['non_streaming_share'])
        lines.append(f"| {f['mhc_weights_per_layer']} | {number(f['total_gb']):.2f} | "
                     f"{number(fl['kl1_isolated_226.6']):.1f} / {number(fl['source_230']):.1f} / "
                     f"{number(fl['spec_273']):.1f} | {_pct(number(sh['kl1_isolated_226.6@24.3ms']))} / "
                     f"{_pct(number(sh['kl1_isolated_226.6@27.7ms']))} | {_pct(number(sh['spec_273@24.3ms']))} / "
                     f"{_pct(number(sh['spec_273@27.7ms']))} |")
    if 'kl1w_rates' in res:
        lines += ['', '## KL1-W isolated microbenchmark rates as logical-byte rates (MEASURED time / ESTIMATED bytes)',
                  '', '| host | variant | B1 cases GB/s (min-max) | B4 cases GB/s (min-max) | dense mrow 4 / 16 / 20 rows '
                  '(ring, min32 9) |', '|---|---|---|---|---|']
        for host, h in record(record(res['kl1w_rates'])['hosts']).items():
            dense = record(record(h).get('dense') or {'rows': []})
            ring = {int(number(record(x)['rows'])): number(record(x)['logical_gbps'])
                    for x in cast(list[Json], dense['rows'])
                    if record(x)['variant'] == 'ring' and number(record(x)['min32']) == 9}
            for variant in ('prod-first', 'prod-last'):
                v = record(record(h).get(variant) or {'cases': []})
                b1 = [number(record(c)['logical_gbps']) for c in cast(list[Json], v['cases'])
                      if number(record(c)['requests']) == 1]
                b4 = [number(record(c)['logical_gbps']) for c in cast(list[Json], v['cases'])
                      if number(record(c)['requests']) > 1]
                if not b1:
                    continue
                lines.append(f"| {host} | {variant} | {min(b1):.0f}-{max(b1):.0f} | "
                             f"{(f'{min(b4):.0f}-{max(b4):.0f}' if b4 else 'n/a')} | "
                             + ' / '.join(f'{ring[r]:.0f}' for r in (4, 16, 20) if r in ring) + ' |')
        rep = record(res.get('kl1_reproduction') or {})
        if 'median_isolated_gbps_in_range' in rep:
            lines += ['', f"KL1 analyze re-run: median isolated smoe rate {number(rep['median_isolated_gbps_in_range']):.1f}"
                          f" GB/s over {rep['in_range_steps']} in-range steps; inferred distinct experts median "
                          f"{number(rep['inferred_distinct_median']):.1f}, range "
                          + '-'.join(f'{number(x):.1f}' for x in cast(list[Json], rep['inferred_distinct_range']))
                          + ' (ESTIMATED; logical bytes).']
    if 'resolution' in res:
        lines += ['', '## Control-archive resolution (control-to-control drift SD; MEASURED)', '',
                  '| metric @ clock | cell | pairs | SD | 2 x resolution | status |', '|---|---|---|---|---|---|']
        for key, rz in record(res['resolution']).items():
            cells = record(record(rz)['cells'])
            if not cells:
                lines.append(f'| {key} | (no controls at this clock) | 0 | | | INSUFFICIENT |')
            for cell, cv in cells.items():
                x = record(cv)
                sd = x['sd']
                lines.append(f"| {key} | {cell} | {x['pairs']} | {'n/a' if sd is None else _pct(number(sd))} | "
                             f"{'n/a' if sd is None else _pct(2 * number(sd))} | {x['status']} |")
    return '\n'.join(lines) + '\n'


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--campaign', type=Path, required=True)
    ap.add_argument('--sp1', nargs='+', required=True, help='SP1 window directory names under --campaign')
    ap.add_argument('--kl1w', help='KL1-W window directory name under --campaign')
    ap.add_argument('--archive', default='control-archive.jsonl', help='under --campaign; "" to skip')
    ap.add_argument('--output', required=True)
    ap.add_argument('--markdown', required=True)
    args = ap.parse_args(argv)
    out, md = args.campaign / args.output, args.campaign / args.markdown
    for p in (out, md):
        if p.exists():
            print(f'{p} exists; refusing to overwrite', file=sys.stderr)
            return 1
    archive = args.campaign / args.archive if args.archive else None
    res = build(args.campaign, args.sp1, args.kl1w, archive)
    out.write_text(json.dumps(res, indent=1, allow_nan=False) + '\n')
    md.write_text(markdown(res))
    print(md.read_text())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
