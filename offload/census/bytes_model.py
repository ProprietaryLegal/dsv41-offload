"""CEN0 item 4: logical bytes per kernel family (offload/kl1/shapes.py) and the physical / logical byte ratio of CEN3's
Nsight Compute output; also re-derives KL1-W's isolated microbenchmark rates as logical-byte rates.

  PYTHONPATH=. python3 -m offload.census.bytes_model floor                       # F-CENSUS §2.2, both mHC counts
  PYTHONPATH=. python3 -m offload.census.bytes_model step --rows 6 --distinct 13   # one verify round's logical bytes
  PYTHONPATH=. python3 -m offload.census.bytes_model ratio --manifest <cen3-ncu-manifest.json> --output <json>
  PYTHONPATH=. python3 -m offload.census.bytes_model kl1w --window <KL1-W-dir> --output <json>

LOGICAL bytes are what a kernel must at least read: EXL3 trellis in*out*K/8 plus fp16 scales (shapes.exl3_bytes), at
legal-v2a's shapes, TP2, one rank; routed experts at the bank's mean K 2.75 unless a run gives exact bytes. Everything
from the model is ESTIMATED. PHYSICAL bytes are Nsight Compute counters (MEASURED when CEN3 runs them): DRAM read and
write by default, L2 sector counters as the fallback the manifest names (the report says which one was used). ncu's
default --cache-control all flushes caches between replays, so a ratio below 1 needs --cache-control none to mean L2
reuse. Pre-registered rule (F-CENSUS.plan.md §3): |physical/logical - 1| <= 10 % -> the §2.2 floor stands; > +10 % ->
a byte-efficiency lever inside that kernel family gets its own estimate; < -10 % -> L2 hits (l2pf, cycling) cut DRAM
traffic below the logical bytes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from offload.kl1 import shapes as S
from offload.kl1 import trace_kernels as T
from offload.prefill_maxlen.evidence import Json, number, record

# F-CENSUS.plan.md §2.2 inputs (labels as there)
KL1_ISOLATED_GBPS = 226.6           # KL1-W-20261026T194927Z analysis-rank0.json: isolated rate at inferred count
SOURCE_GBPS = 230.0                 # UPSTREAM CLAIM exl3_smoe.cu:28
TARGET_ONLY_MS = (24.3, 27.7)       # MEASURED SP1.report.md COMPARISON: config A c32 41.08 tok/s; D drill p50
TOLERANCE = 0.10                    # pre-registered physical ~ logical band
HC_PRENORM_PER_LAYER = 2            # one mHC mix weight before attention and one before the FFN (79-80 hc_prenorm
#                                     launches per B1 step MEASURED in SP1-W-20261026T191247Z, 24 x 20480 fp32 each)
UNITS = {'byte': 1.0, 'kbyte': 1e3, 'mbyte': 1e6, 'gbyte': 1e9, 'tbyte': 1e12, 'sector': 32.0,
         'nsecond': 1e-9, 'usecond': 1e-6, 'msecond': 1e-3, 'second': 1.0}
DEFAULT_READ = ('dram__bytes_read.sum',)
DEFAULT_WRITE = ('dram__bytes_write.sum',)
TIME = 'gpu__time_duration.sum'


def expert_mean(k: float | None = None) -> tuple[float, float]:
    """(gate|up, down) logical bytes of one routed expert on one rank at K (default: the bank's K mix)."""
    if k is not None:
        return S.expert_gu_bytes(k), S.expert_down_bytes(k)
    total = sum(S.K_MIX.values())
    return (sum(n * S.expert_gu_bytes(kk) for kk, n in S.K_MIX.items()) / total,
            sum(n * S.expert_down_bytes(kk) for kk, n in S.K_MIX.items()) / total)


def shared_parts() -> tuple[float, float]:
    return S.expert_gu_bytes(S.SHARED_K), S.expert_down_bytes(S.SHARED_K)


def layer_bytes(distinct: float, live_rows: int, mhc_per_layer: int = HC_PRENORM_PER_LAYER) -> dict[str, float]:
    """One decode layer on one rank: routed experts read once per run of <= 8 rows each (distinct of them), the shared
    expert once per run of 8 live rows, the four mrow projections, the bf16 router and the fp32 mHC mix weights."""
    gu, dn = expert_mean()
    sgu, sdn = shared_parts()
    runs = S.shared_runs(live_rows)
    out = {'routed_gu': distinct * gu, 'routed_down': distinct * dn, 'shared_gu': runs * sgu,
           'shared_down': runs * sdn, 'router': float(S.ROUTER_BYTES), 'mhc': mhc_per_layer * float(S.MHC_BYTES)}
    for lin in S.DECODE_DENSE:
        out[f'dense_{lin.name}'] = lin.bytes
    return out


def step_bytes(distinct: float, live_rows: int, mhc_per_layer: int = HC_PRENORM_PER_LAYER) -> dict[str, float]:
    """A decode step (round) on one rank: 40 layers plus the LM head, by family, and the total."""
    per = layer_bytes(distinct, live_rows, mhc_per_layer)
    fam = {'routed': per['routed_gu'] + per['routed_down'], 'shared': per['shared_gu'] + per['shared_down'],
           'dense': sum(v for k, v in per.items() if k.startswith('dense_')), 'router': per['router'],
           'mhc': per['mhc']}
    out = {k: v * S.LAYERS for k, v in fam.items()}
    out['lm_head'] = S.LM_HEAD.bytes
    out['total'] = sum(out.values())
    return out


def floor(mhc_per_layer: int) -> dict[str, Json]:
    """§2.2: one target-only token (1 live row, 6 routed experts), its streaming floor and non-streaming share."""
    per = layer_bytes(S.TOP_K, 1, mhc_per_layer)
    step = step_bytes(S.TOP_K, 1, mhc_per_layer)
    rates = {'kl1_isolated_226.6': KL1_ISOLATED_GBPS, 'source_230': SOURCE_GBPS, 'spec_273': S.GB10_SPEC_GBPS}
    floors = {k: step['total'] / g / 1e6 for k, g in rates.items()}
    return {'mhc_weights_per_layer': mhc_per_layer,
            'mb_per_layer': cast(Json, {'routed_6_experts': (per['routed_gu'] + per['routed_down']) / 1e6,
                                        'shared': (per['shared_gu'] + per['shared_down']) / 1e6,
                                        'dense': sum(v for k, v in per.items() if k.startswith('dense_')) / 1e6,
                                        'router': per['router'] / 1e6, 'mhc': per['mhc'] / 1e6}),
            'lm_head_mb': S.LM_HEAD.bytes / 1e6, 'total_gb': step['total'] / 1e9,
            'floor_ms': cast(Json, floors),
            'non_streaming_share': cast(Json, {f'{k}@{ms}ms': (ms - f) / ms for k, f in floors.items()
                                               for ms in TARGET_ONLY_MS}),
            'status': 'ESTIMATED logical bytes / rates; target-only times MEASURED (SP1.report.md)'}


def round_bytes(rows: int, distinct: float | None = None) -> dict[str, Json]:
    """One verify round of `rows` live rows: logical bytes at the uniform bound U(rows) and at a given distinct count,
    and the streaming floor at KL1's isolated rate."""
    out: dict[str, Json] = {'rows': rows, 'uniform_distinct': S.unique_uniform(rows)}
    for tag, u in (('uniform', min(S.unique_uniform(rows), float(S.E))), ('given', distinct)):
        if u is None:
            continue
        b = step_bytes(u, rows)
        out[tag] = {'distinct': u, 'gb': b['total'] / 1e9, 'by_family_mb': cast(Json, {k: v / 1e6 for k, v in b.items()}),
                    'floor_ms_at_226.6': b['total'] / KL1_ISOLATED_GBPS / 1e6}
    return out


# ------------------------------------------------------------------ Nsight Compute


@dataclass(frozen=True)
class Launch:
    id: int
    name: str
    family: str
    grid: tuple[int, int, int]
    metrics: dict[str, float]          # base units (bytes, seconds)


def _num(text: str) -> float:
    return float(text.replace(',', '').strip())


def _triple(text: str) -> tuple[int, int, int]:
    found = [int(x) for x in re.findall(r'\d+', text)]
    if not found:
        return (0, 0, 0)
    parts = found + [1, 1, 1]
    return (parts[0], parts[1], parts[2])


def _scale(unit: str) -> float:
    u = unit.strip().lower()
    if u in ('', '%', 'inst', 'cycle', 'count'):
        return 1.0
    if u not in UNITS:
        raise ValueError(f'unknown ncu unit {unit!r}')
    return UNITS[u]


def read_ncu_csv(text: str) -> list[Launch]:
    """`ncu --csv --page raw` (one row a launch, metric columns, an optional units row) or `--page details` (one row a
    metric: Metric Name / Metric Unit / Metric Value)."""
    rows = list(csv.reader(text.splitlines()))
    rows = [r for r in rows if r and not r[0].startswith('==')]          # drop ncu's ==PROF== log lines
    if not rows:
        raise ValueError('empty ncu csv')
    head = rows[0]
    col = {name: i for i, name in enumerate(head)}
    for need in ('ID', 'Kernel Name'):
        if need not in col:
            raise ValueError(f'ncu csv lacks column {need!r}')
    launches: dict[int, Launch] = {}
    if 'Metric Name' in col:
        for r in rows[1:]:
            i = int(_num(r[col['ID']]))
            if i not in launches:
                name = r[col['Kernel Name']]
                launches[i] = Launch(i, name, T.family(name), _triple(r[col.get('Grid Size', -1)])
                                     if 'Grid Size' in col else (0, 0, 0), {})
            value = r[col['Metric Value']]
            try:
                launches[i].metrics[r[col['Metric Name']]] = _num(value) * _scale(r[col['Metric Unit']])
            except ValueError:
                if value.strip() and not value.strip().lower() in ('n/a', '-'):
                    raise
        return [launches[i] for i in sorted(launches)]
    body = rows[1:]
    units: list[str] = [''] * len(head)
    if body and body[0][col['ID']].strip() == '':
        units, body = body[0], body[1:]
    fixed = {'ID', 'Process ID', 'Process Name', 'Host Name', 'Kernel Name', 'Context', 'Stream', 'Block Size',
             'Grid Size', 'Device', 'CC', 'Section Name'}
    out: list[Launch] = []
    for r in body:
        name = r[col['Kernel Name']]
        metrics: dict[str, float] = {}
        for j, h in enumerate(head):
            if h in fixed or j >= len(r) or not r[j].strip() or r[j].strip().lower() in ('n/a', '-'):
                continue
            try:
                metrics[h] = _num(r[j]) * _scale(units[j] if j < len(units) else '')
            except ValueError:
                continue                       # text-valued metric columns (launch__* names etc.)
        out.append(Launch(int(_num(r[col['ID']])), name, T.family(name),
                          _triple(r[col['Grid Size']]) if 'Grid Size' in col else (0, 0, 0), metrics))
    return out


def logical(spec: Mapping[str, Json], index: int) -> float:
    """Logical bytes of the index-th matched launch of a run from its manifest spec. `bytes_by_index` (CEN3's driver:
    exact bytes of every profiled launch of that family, in launch order) must cover every matched launch."""
    if 'bytes_by_index' in spec:
        exact = cast(list[Json], spec['bytes_by_index'])
        if index >= len(exact):
            raise ValueError(f'launch {index} of a family with {len(exact)} exact byte counts: the ncu launches do not '
                             'match the driver\'s profiled calls')
        return number(exact[index])
    if 'bytes' in spec:
        return number(spec['bytes'])
    model = str(spec.get('model'))
    k = None if spec.get('k') is None else number(spec['k'])
    if model in ('smoe_gu', 'smoe_down', 'smoe_layer'):
        gu, dn = expert_mean(k)
        sgu, sdn = shared_parts() if spec.get('shared', True) else (0.0, 0.0)
        u = number(spec['distinct'])
        runs = S.shared_runs(int(number(spec['rows'])))
        parts = {'smoe_gu': u * gu + runs * sgu, 'smoe_down': u * dn + runs * sdn}
        return parts[model] if model in parts else sum(parts.values())
    if model == 'dense':
        lin = next((x for x in S.DECODE_DENSE if x.name == spec.get('linear')), None)
        if lin is None:
            raise ValueError(f'unknown dense linear {spec.get("linear")!r}')
        return lin.bytes
    if model == 'dense_cycle':                 # four mrow launches a layer in forward order (bandwidth.dense_decode)
        return S.DECODE_DENSE[index % len(S.DECODE_DENSE)].bytes
    if model == 'lm_head':
        return S.LM_HEAD.bytes
    if model == 'router':
        return float(S.ROUTER_BYTES)
    if model == 'mhc':
        return float(S.MHC_BYTES)
    raise ValueError(f'unknown logical model {model!r}')


def _first(metrics: Mapping[str, float], names: Sequence[str]) -> tuple[str, float] | None:
    return next(((n, metrics[n]) for n in names if n in metrics), None)


def verdict(ratio: float) -> str:
    if abs(ratio - 1) <= TOLERANCE:
        return 'PHYSICAL ~ LOGICAL (floor stands)'
    return 'OVER-FETCH (byte-efficiency lever)' if ratio > 1 else 'BELOW LOGICAL (L2 reuse)'


def ratio_report(manifest_path: Path) -> dict[str, Json]:
    m = record(cast(Json, json.loads(manifest_path.read_text())))
    read_names = tuple(str(x) for x in cast(list[Json], m.get('read_metrics') or list(DEFAULT_READ)))
    write_names = tuple(str(x) for x in cast(list[Json], m.get('write_metrics') or list(DEFAULT_WRITE)))
    pooled: dict[str, list[tuple[float, float]]] = {}
    runs: list[Json] = []
    for run in map(record, cast(list[Json], m['runs'])):
        path = (manifest_path.parent / str(run['csv'])).resolve()
        data = path.read_bytes()
        if 'sha256' in run and hashlib.sha256(data).hexdigest() != run['sha256']:
            raise ValueError(f'{path.name}: sha256 differs from the manifest')
        launches = read_ncu_csv(data.decode())
        specs = {str(record(x)['match']): record(record(x)['logical']) for x in cast(list[Json], run['launches'])}
        seen: dict[str, int] = {}
        fams: dict[str, dict[str, list[float]]] = {}
        sources: set[str] = set()
        unassigned = 0
        for ln in launches:
            if ln.family not in specs:
                unassigned += 1
                continue
            idx = seen.get(ln.family, 0)
            seen[ln.family] = idx + 1
            rd = _first(ln.metrics, read_names)
            if rd is None:
                raise ValueError(f'{path.name} launch {ln.id}: none of {list(read_names)} present (probe metric names '
                                 'with ncu --query-metrics in the smoke start)')
            wr = _first(ln.metrics, write_names)
            sources.add(rd[0])
            lg = logical(specs[ln.family], idx)
            f = fams.setdefault(ln.family, {'logical': [], 'read': [], 'write': [], 'time': []})
            f['logical'].append(lg)
            f['read'].append(rd[1])
            f['write'].append(wr[1] if wr is not None else 0.0)
            if TIME in ln.metrics:
                f['time'].append(ln.metrics[TIME])
        for fam, spec in specs.items():
            if 'bytes_by_index' in spec and seen.get(fam, 0) != len(cast(list[Json], spec['bytes_by_index'])):
                raise ValueError(f'{path.name}: {seen.get(fam, 0)} {fam} launches, the driver profiled '
                                 f'{len(cast(list[Json], spec["bytes_by_index"]))}')
        rows: dict[str, Json] = {}
        for fam, f in sorted(fams.items()):
            lg_sum, rd_sum = sum(f['logical']), sum(f['read'])
            per = [r / g for r, g in zip(f['read'], f['logical']) if g > 0]
            row: dict[str, Json] = {'launches': len(f['logical']), 'logical_mb': lg_sum / 1e6, 'physical_read_mb': rd_sum / 1e6,
                                    'physical_write_mb': sum(f['write']) / 1e6, 'read_over_logical': rd_sum / lg_sum,
                                    'per_launch_ratio': {'min': min(per), 'median': statistics.median(per),
                                                         'max': max(per)},
                                    'verdict': verdict(rd_sum / lg_sum)}
            if len(f['time']) == len(f['logical']):
                t = sum(f['time'])
                row['logical_gbps'] = lg_sum / t / 1e9
                row['physical_gbps'] = rd_sum / t / 1e9
            rows[fam] = row
            pooled.setdefault(fam, []).append((lg_sum, rd_sum))
        runs.append({'label': run.get('label'), 'csv': path.name, 'sha256': hashlib.sha256(data).hexdigest(),
                     'read_metric': cast(Json, sorted(sources)), 'unassigned_launches': unassigned,
                     'families': rows})
    families = {fam: {'logical_mb': sum(a for a, _ in v) / 1e6, 'physical_read_mb': sum(b for _, b in v) / 1e6,
                      'read_over_logical': sum(b for _, b in v) / sum(a for a, _ in v),
                      'verdict': verdict(sum(b for _, b in v) / sum(a for a, _ in v))}
                for fam, v in sorted(pooled.items())}
    fallback = any(not name.startswith('dram__') for name in read_names[:1])
    return {'type': 'cen3-byte-ratio', 'clock_cap_mhz': m.get('clock_cap_mhz'), 'host': m.get('host'),
            'commit': m.get('commit'), 'read_metrics': cast(Json, list(read_names)),
            'physical_source': 'L2 fallback counters (not DRAM)' if fallback else 'DRAM counters',
            'status': 'MEASURED ncu counters / ESTIMATED logical bytes', 'rule': f'|ratio - 1| <= {TOLERANCE}',
            'families': cast(Json, families), 'runs': runs}


# ------------------------------------------------------------------ KL1-W isolated rates


def kl1_window(root: Path) -> dict[str, Json]:
    """KL1 stage-B microbenchmarks as logical-byte rates: smoe cases (the bench's exact bytes for its synthetic routing
    and per-layer K plan, beside the mean-K model) and dense mrow / mrow32 groups. Isolated, synthetic, profiler-free
    timings: MEASURED times, never serving bandwidth."""
    summary = record(cast(Json, json.loads((root / 'summary.json').read_text())))
    hashes = record(summary['artifact_sha256'])
    out: dict[str, Json] = {}
    for host_dir in sorted((root / 'results').iterdir()):
        host: dict[str, Json] = {}
        for p in sorted(host_dir.glob('*.json')):
            rel = f'results/{host_dir.name}/{p.name}'
            if hashlib.sha256(p.read_bytes()).hexdigest() != hashes.get(rel):
                raise ValueError(f'{rel}: sha256 differs from summary.json')
            raw = cast(Json, json.loads(p.read_text()))
            if isinstance(raw, list):                  # failures.json: the runner's failure list
                host[p.stem] = {'failures': len(raw)}
                continue
            doc = record(raw)
            if doc.get('type') == 'kl1-smoe-bench':
                cases: list[Json] = []
                for c in map(record, cast(list[Json], doc['results'])):
                    rows, u = int(number(c['rows'])), number(c['distinct'])
                    model = logical({'model': 'smoe_layer', 'rows': rows, 'distinct': u}, 0)
                    cases.append({'case': c['case'], 'rows': rows, 'requests': c['requests'], 'distinct': u,
                                  'bench_mb': number(c['bytes_per_layer']) / 1e6, 'model_mb': model / 1e6,
                                  'us': number(c['us_per_layer_median']),
                                  'logical_gbps': number(c['bytes_per_layer']) / number(c['us_per_layer_median']) / 1e3})
                gb = [number(record(x)['logical_gbps']) for x in cases]
                host[p.stem] = {'variant': doc.get('variant'), 'cases': cases,
                                'logical_gbps_median': statistics.median(gb), 'logical_gbps_max': max(gb)}
            elif doc.get('type') == 'kl1-dense-bench':
                dense = sum(lin.bytes for lin in S.DECODE_DENSE)
                host['dense'] = {'rows': [{'variant': record(r)['variant'], 'min32': record(r)['min32'],
                                           'rows': record(r)['rows'], 'us_per_layer': record(r)['us_per_layer'],
                                           'logical_gbps': dense / number(record(r)['us_per_layer']) / 1e3,
                                           'file_gbps': record(r)['gbps']}
                                          for r in cast(list[Json], doc['rows'])], 'layer_mb': dense / 1e6}
        out[host_dir.name] = host
    return {'type': 'cen0-kl1w-rates', 'window': root.name,
            'status': 'MEASURED isolated synthetic microbench times; ESTIMATED logical bytes (exact for the bench '
                      'routing); never serving bandwidth', 'hosts': out}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('floor')
    st = sub.add_parser('step')
    st.add_argument('--rows', type=int, required=True)
    st.add_argument('--distinct', type=float)
    ra = sub.add_parser('ratio')
    ra.add_argument('--manifest', type=Path, required=True)
    kw = sub.add_parser('kl1w')
    kw.add_argument('--window', type=Path, required=True)
    for p in (ra, kw):
        p.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == 'floor':
        print(json.dumps({'plan_one_mhc_weight': floor(1), 'two_mhc_weights': floor(HC_PRENORM_PER_LAYER)}, indent=1))
        return 0
    if args.cmd == 'step':
        print(json.dumps(round_bytes(args.rows, args.distinct), indent=1))
        return 0
    if args.output.exists():
        print(f'{args.output} exists; refusing to overwrite', file=sys.stderr)
        return 1
    res = ratio_report(args.manifest) if args.cmd == 'ratio' else kl1_window(args.window)
    args.output.write_text(json.dumps(res, indent=1, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
