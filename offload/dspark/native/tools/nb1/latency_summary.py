"""Reduce the ``latency_probe.py`` RESULT_JSON to evidence JSON and a Markdown table (aggregates only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DRAFT_ARMS = ('k2', 'native', 'native-invariant', 'native-microbatched')
INSERT_ARMS = ('k2', 'native', 'native-invariant')


def load(*logs: Path) -> dict[str, Any]:
    """One RESULT_JSON per log (one rank per process); cells, clocks and load times merged, settings must agree."""
    merged: dict[str, Any] = {}
    for log in logs:
        lines = [ln for ln in log.read_text().splitlines() if ln.startswith('RESULT_JSON ')]
        if len(lines) != 1:
            raise ValueError(f'{log}: expected one RESULT_JSON line, found {len(lines)}')
        result = json.loads(lines[0][len('RESULT_JSON '):])
        if not merged:
            merged = dict(result, cells=dict(result['cells']), clocks=list(result['clocks']),
                          loads_seconds=dict(result['loads_seconds']))
            continue
        for key in ('mode', 'reps', 'iters', 'warmup', 'torch_version'):
            if result[key] != merged[key]:
                raise ValueError(f'{log}: {key} differs from the first log')
        if merged['cells'].keys() & result['cells'].keys():
            raise ValueError(f'{log}: repeats cells of an earlier log')
        merged['cells'].update(result['cells'])
        merged['clocks'] += result['clocks']
        merged['loads_seconds'].update(result['loads_seconds'])
    return merged


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    cells = result['cells']
    ranks = sorted({key.split('/', 1)[0] for key in cells})

    def median(rank: str, arm: str, cell: str) -> float | None:
        entry = cells.get(f'{rank}/{arm}/{cell}')
        return None if entry is None else float(entry['median_us'])

    def spread(rank: str, arm: str, cell: str) -> float | None:
        entry = cells.get(f'{rank}/{arm}/{cell}')
        return None if entry is None else (entry['max_us'] - entry['min_us']) / entry['median_us']

    draft = []
    for b in range(1, 5):
        for n in (4, 5):
            row: dict[str, Any] = dict(B=b, n_out=n)
            for arm in DRAFT_ARMS:
                per_rank = {r: median(r, arm, f'B{b}/n{n}') for r in ranks}
                if arm == 'native-microbatched' and b == 1:
                    per_rank = {r: median(r, 'native', 'B1/n' + str(n)) for r in ranks}   # one B1 replay
                values = [v for v in per_rank.values() if v is not None]
                row[arm] = dict(per_rank_us=per_rank, slowest_rank_us=max(values) if values else None,
                                rep_spread=max((s for s in (spread(r, arm, f'B{b}/n{n}') for r in ranks)
                                                if s is not None), default=None))
            base = row['native']['slowest_rank_us']
            for arm in ('k2', 'native-invariant', 'native-microbatched'):
                value = row[arm]['slowest_rank_us']
                row[arm]['vs_native'] = None if value is None or not base else value / base
            draft.append(row)
    insert = []
    buckets = sorted({int(k.rsplit('insert', 1)[1]) for k in cells if '/insert' in k})
    for rows in buckets:
        irow: dict[str, Any] = dict(rows=rows)
        for arm in INSERT_ARMS:
            per_rank = {r: median(r, arm, f'insert{rows}') for r in ranks}
            values = [v for v in per_rank.values() if v is not None]
            irow[arm] = dict(per_rank_us=per_rank, slowest_rank_us=max(values) if values else None)
        insert.append(irow)
    return dict(label='MEASURED', mode=result['mode'], reps=result['reps'], iters=result['iters'],
                warmup=result['warmup'], ranks=ranks, draft=draft, insert=insert, clocks=result['clocks'],
                loads_seconds=result['loads_seconds'], torch_version=result['torch_version'])


def _us(value: float | None) -> str:
    return '-' if value is None else f'{value:.0f}'


def _ratio(value: float | None) -> str:
    return '-' if value is None else f'{value:.3f}'


def markdown(summary: dict[str, Any]) -> str:
    out = [f"Per-shard CUDA-graph replay, median of {summary['reps']} interleaved runs (each the median of "
           f"{summary['iters']} replays after {summary['warmup']} warm-up), slower of the two TP2 shards, µs. "
           'Ratios are against native default at the same B and n_out. MEASURED.', '',
           '|B|n_out|K2|native default|native invariant|microbatched (B × B1)|K2/native|invariant/native|'
           'microbatched/native|worst rep spread|', '|---|---|---|---|---|---|---|---|---|---|']
    for row in summary['draft']:
        spreads = [row[a]['rep_spread'] for a in DRAFT_ARMS if row[a]['rep_spread'] is not None]
        out.append('|' + '|'.join([str(row['B']), str(row['n_out'])] +
                                  [_us(row[a]['slowest_rank_us']) for a in DRAFT_ARMS] +
                                  [_ratio(row[a]['vs_native']) for a in DRAFT_ARMS if a != 'native'] +
                                  [f'{100 * max(spreads):.1f}%' if spreads else '-']) + '|')
    out += ['', '|insert rows|K2|native default|native invariant (8-row sub-batches)|invariant/native|',
            '|---|---|---|---|---|']
    for irow in summary['insert']:
        base, inv = irow['native']['slowest_rank_us'], irow['native-invariant']['slowest_rank_us']
        out.append(f"|{irow['rows']}|{_us(irow['k2']['slowest_rank_us'])}|{_us(base)}|{_us(inv)}|"
                   f"{_ratio(None if base is None or inv is None else inv / base)}|")
    return '\n'.join(out) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', type=Path, nargs='+', help='one latency_probe log per rank')
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(load(*args.logs))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'latency.json').write_text(json.dumps(summary, indent=1, sort_keys=True) + '\n')
    (args.out_dir / 'LATENCY.md').write_text(markdown(summary))
    print(markdown(summary))


if __name__ == '__main__':
    main()
