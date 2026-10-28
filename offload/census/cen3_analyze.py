"""CEN3 analysis after the window (CPU, Controller): one JSON of numbers from a CEN3-W directory.

  PYTHONPATH=. python3 -m offload.census.cen3_analyze --window <CEN3-W-dir> --output <json>

(b) waterfall: offload.census.waterfall over the window's decode captures (verify rounds at short-b1, 32k-b1, 32k-b4 and
the target-only capture; a capture that ended before its explicit stop is listed, not binned). (a) byte ratio: per Spark,
an offload.census.bytes_model manifest built from the driver's exact logical bytes (cen3_bench ncu-target outputs) and
the ncu CSV report of each group (--log-file, else stdout), then bytes_model.ratio_report; kernel times per case for
CEN4. (c) bandwidth: the per-Spark read / write / copy rates against the 273 GB/s spec (UPSTREAM CLAIM).
Every input is checked against the window's manifests (sha256). Traces and counts stay on Controller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import cast

from offload.census import bytes_model as BM
from offload.census import waterfall
from offload.kl1 import shapes as S
from offload.launch.windows import win2_history as wh
from offload.prefill_maxlen.evidence import Json, number, record

HOSTS = ('sparkhead', 'sparkworker-wired')                 # rank order (pf1_window.HOSTS)
NCU_GROUPS = ('smoe-b1', 'smoe-b4', 'dense', 'head')


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_bench(out: Path) -> dict[str, dict[str, Path]]:
    """{host: {name: path}} of bench outputs whose size and sha256 match bench-manifest.json."""
    rows = [record(x) for x in cast(list[Json], json.loads((out / 'bench-manifest.json').read_text()))]
    got: dict[str, dict[str, Path]] = {h: {} for h in HOSTS}
    for row in rows:
        path = (out / str(row['file'])).resolve()
        if out.resolve() not in path.parents:
            raise ValueError('bench manifest path leaves the window directory')
        if path.stat().st_size != int(number(row['bytes'])) or _sha(path) != row['sha256']:
            raise ValueError(f'{path.name}: differs from the bench manifest')
        got[str(row['host'])][path.name] = path
    return got


def report_csv(files: dict[str, Path], group: str) -> Path | None:
    """The file holding ncu's CSV report for a group: --log-file first, else the profiled process's stdout."""
    for name in (f'ncu-{group}.csv', f'ncu-{group}.app.txt'):
        path = files.get(name)
        if path is None:
            continue
        try:
            if BM.read_ncu_csv(path.read_text(errors='replace')):
                return path
        except (ValueError, IndexError):
            continue
    return None


def ncu_manifest(out: Path, host: str, files: dict[str, Path], *, clock_cap_mhz: Json, commit: Json) -> Path | None:
    """bench/<host>/ncu-manifest.json for bytes_model.ratio_report: one run a group, each family's exact bytes in
    launch order (bytes_by_index); None when part (a) left no metric choice on this Spark."""
    if 'ncu-metrics.json' not in files:
        return None
    picked = record(cast(Json, json.loads(files['ncu-metrics.json'].read_text())))
    runs: list[Json] = []
    for group in NCU_GROUPS:
        target, csv = files.get(f'ncu-{group}-target.json'), report_csv(files, group)
        if target is None or csv is None:
            continue
        doc = record(cast(Json, json.loads(target.read_text())))
        by_family: dict[str, list[Json]] = {}
        for case in map(record, cast(list[Json], doc['cases'])):
            for launch in map(record, cast(list[Json], case['launches'])):
                by_family.setdefault(str(launch['family']), []).append(number(launch['bytes']))
        runs.append({'csv': csv.name, 'sha256': _sha(csv), 'label': group,
                     'launches': [{'match': fam, 'logical': {'bytes_by_index': cast(Json, v)}}
                                  for fam, v in sorted(by_family.items())]})
    manifest: dict[str, Json] = {'type': 'cen3-ncu-manifest', 'host': host, 'clock_cap_mhz': clock_cap_mhz,
                                 'commit': commit, 'read_metrics': [picked['read']],
                                 'write_metrics': [picked['write']] if picked.get('write') else [], 'runs': runs}
    path = out / 'bench' / host / 'ncu-manifest.json'
    path.write_text(json.dumps(manifest, indent=1) + '\n')
    return path


def kernel_times(files: dict[str, Path]) -> dict[str, list[float]]:
    """gpu__time_duration (s) per profiled launch, by '<group>:<family>' in launch order (identical work on both
    Sparks: same driver, seeds and cases)."""
    out: dict[str, list[float]] = {}
    for group in NCU_GROUPS:
        csv = report_csv(files, group)
        if csv is None:
            continue
        for ln in BM.read_ncu_csv(csv.read_text(errors='replace')):
            if ln.family in ('smoe_gu', 'smoe_down', 'mrow', 'mrow_head') and BM.TIME in ln.metrics:
                out.setdefault(f'{group}:{ln.family}', []).append(ln.metrics[BM.TIME])
    return out


def bandwidth(files: dict[str, Path]) -> dict[str, Json] | None:
    if 'bw.json' not in files:
        return None
    doc = record(cast(Json, json.loads(files['bw.json'].read_text())))
    ops = {str(record(o)['op']): record(o) for o in cast(list[Json], doc['ops'])}
    return {op: {'gbps_median': row['gbps_median'], 'gbps_max': row['gbps_max'],
                 'median_over_spec': number(row['gbps_median']) / S.GB10_SPEC_GBPS} for op, row in ops.items()}


def analyze(out: Path) -> dict[str, Json]:
    try:
        clock: Json = wh.start_clock_cap(out, 'cen3')[0]          # the profiled start's own log
    except (ValueError, OSError):
        clock = None
    parts: dict[str, Json] = {}
    result: dict[str, Json] = {'type': 'cen3-analysis', 'window': out.name, 'clock_cap_mhz': clock,
                               'status': 'MEASURED profiled durations / ncu counters / CUDA-event rates; logical bytes '
                                         'and idle attribution ESTIMATED; never promotion timings',
                               'parts': parts}
    bench = verified_bench(out) if (out / 'bench-manifest.json').exists() else {h: {} for h in HOSTS}
    rates = {h: bandwidth(bench[h]) for h in HOSTS}
    parts['c'] = {'bandwidth': cast(Json, rates)} if all(rates.values()) else {'status': 'NO DATA'}
    ratios: dict[str, Json] = {}
    times: dict[str, Json] = {}
    for host in HOSTS:
        manifest = ncu_manifest(out, host, bench[host], clock_cap_mhz=clock, commit=None)
        if manifest is not None and json.loads(manifest.read_text())['runs']:
            ratios[host] = BM.ratio_report(manifest)
            times[host] = cast(Json, {k: {'launches': len(v), 'median_us': statistics.median(v) * 1e6}
                                      for k, v in kernel_times(bench[host]).items()})
    parts['a'] = {'ratio': ratios, 'kernel_time': times} if ratios else {'status': 'NO DATA'}
    if (out / 'traces-manifest.json').exists():
        parts['b'] = waterfall.window(out, strict=False)
    else:
        parts['b'] = {'status': 'NO DATA'}
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--window', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(argv)
    if args.output.exists():
        print(f'{args.output} exists; refusing to overwrite', file=sys.stderr)
        return 1
    args.output.write_text(json.dumps(analyze(args.window), indent=1, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
