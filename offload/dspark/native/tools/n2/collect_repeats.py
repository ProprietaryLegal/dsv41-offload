"""Aggregate three interleaved solo repetitions; no token/fixture input is read."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median
from typing import Any

from .profiling import category

ARMS = ('k2-tp2', 'native-hybrid-kx', 'native-hybrid-kx2')
UUID = 'GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records: dict[tuple[int, str], list[dict[str, Any]]] = {}
    source_hashes: dict[str, str] | None = None
    memory_rows: list[dict[str, Any]] = []
    clocks: dict[str, list[float]] = {key: [] for key in ('clocks.sm', 'clocks.mem', 'temperature.gpu', 'power.draw', 'power.limit')}
    for rank in (0, 1):
        for arm in ARMS:
            runs = [json.loads((args.root/f'repeat{rep}_{arm}_r{rank}.json').read_text()) for rep in (1, 2, 3)]
            for run in runs:
                if run['devices'] != [UUID] or run['mode'] != 'solo' or run['rank'] != rank or run['arm'] != arm:
                    raise ValueError('mismatched arm/rank/device in repeated results')
                if arm == 'native-hybrid-kx2' and run['cfg']['dense_profile'] != 'measured-small':
                    raise ValueError('initial profile cannot count as final-profile evidence')
                if run['iters'] != 200:
                    raise ValueError('repeated timing requires 200 event samples')
                if source_hashes is None:
                    source_hashes = run['source_sha256']
                elif source_hashes != run['source_sha256']:
                    raise ValueError('inference source changed during repeated timing')
                for case in run['draft'].values():
                    if case.get('settle_seconds') != .5 or 'kernels_us' not in case['breakdown']['device0']:
                        raise ValueError('missing settled timing/full kernel breakdown')
                    for where in ('gpu_before', 'gpu_after'):
                        for key in clocks:
                            clocks[key].append(float(case[where][UUID][key].split()[0]))
            records[rank, arm] = runs
            memory_rows.append({
                'rank': rank, 'arm': arm,
                'load_seconds_range': [min(run['load_seconds'] for run in runs), max(run['load_seconds'] for run in runs)],
                **{f'{phase}_max_{metric}_bytes_range': [
                    min(run[f'{phase}_peak']['cuda:0'][f'max_{metric}'] for run in runs),
                    max(run[f'{phase}_peak']['cuda:0'][f'max_{metric}'] for run in runs),
                ] for phase in ('load', 'serving') for metric in ('allocated', 'reserved')},
                'serving_nvidia_smi_mib_range': [
                    min(int(row.rsplit(',', 1)[1].split()[0]) for run in runs for row in run['serving_nvidia_smi']),
                    max(int(row.rsplit(',', 1)[1].split()[0]) for run in runs for row in run['serving_nvidia_smi']),
                ],
            })
    if set(clocks['power.limit']) != {325.}:
        raise ValueError('timing power limit differs from the owner 325W setting')
    rows = []
    gaps = []
    all_pass = True
    for rank in (0, 1):
        for batch in (1, 2, 3, 4):
            for count in (5, 4):
                case = f'B{batch}_k5_n{count}'
                values = {arm: [run['draft'][case]['p50_us'] for run in records[rank, arm]] for arm in ARMS}
                medians = {arm: median(v) for arm, v in values.items()}
                passed = medians['native-hybrid-kx2'] <= medians['k2-tp2']
                all_pass &= passed
                rows.append({'rank': rank, 'batch': batch, 'n_out': count, 'p50_runs_us': values,
                             'median_p50_us': medians,
                             'kx2_minus_k2_percent': (medians['native-hybrid-kx2']/medians['k2-tp2']-1)*100,
                             'target_pass': passed,
                             'worst_kx2_le_best_k2': max(values['native-hybrid-kx2']) <= min(values['k2-tp2'])})
                totals = {}
                kernel_medians = {}
                for arm in ARMS:
                    by_run = []
                    kernel_runs = []
                    for run in records[rank, arm]:
                        cats: Counter[str] = Counter()
                        kernels = run['draft'][case]['breakdown']['device0']['kernels_us']
                        kernel_runs.append(kernels)
                        for name, elapsed in kernels.items():
                            cats[category(name)] += elapsed
                        cats['dense'] = sum(cats.pop(key, 0) for key in ('native_bf16_dense', 'native_fp8_dense', 'dense_k5', 'wo_a'))
                        cats['experts'] += cats.pop('native_experts', 0)
                        cats['other'] += cats.pop('native_epilogues', 0)
                        by_run.append(cats)
                    totals[arm] = {key: median([v[key] for v in by_run]) for key in set().union(*(v.keys() for v in by_run))}
                    kernel_medians[arm] = {key: median([v.get(key, 0) for v in kernel_runs]) for key in sorted(set().union(*(v.keys() for v in kernel_runs)))}
                keys = totals['k2-tp2'].keys() | totals['native-hybrid-kx2'].keys()
                gaps.append({'rank': rank, 'batch': batch, 'n_out': count,
                             'profile_category_medians_us': totals,
                             'profile_kernel_medians_us': kernel_medians,
                             'kx2_minus_k2_us': dict(sorted(((key, totals['native-hybrid-kx2'].get(key, 0)-totals['k2-tp2'].get(key, 0)) for key in keys), key=lambda item: -item[1]))})
    result = {'method': 'Median of three run p50s (200 events each, 0.5s shape settling). Full run ranges are not confidence intervals.',
              'scope': 'Solo channel shards with local collective stand-ins; not pair/engine latency.',
              'rows': rows, 'kernel_gaps': gaps, 'all_targets_pass': all_pass,
              'source_sha256': source_hashes, 'memory_and_load_ranges': memory_rows,
              'telemetry_ranges': {key: [min(v), max(v)] for key, v in clocks.items()}}
    args.output.write_text(json.dumps(result, indent=1)+'\n')
    print('|rank|n_out|B|K2 us|kx us|kx2 us|kx2 vs K2|')
    print('|---:|---:|---:|---:|---:|---:|---:|')
    for row in rows:
        m = row['median_p50_us']
        print(f"|{row['rank']}|{row['n_out']}|{row['batch']}|{m[ARMS[0]]:.1f}|{m[ARMS[1]]:.1f}|{m[ARMS[2]]:.1f}|{row['kx2_minus_k2_percent']:+.2f}%|")
    print('ALL OBSERVED MEDIAN TARGETS PASS:', all_pass)


if __name__ == '__main__':
    main()
