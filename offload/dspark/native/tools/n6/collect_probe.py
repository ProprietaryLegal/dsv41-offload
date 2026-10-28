"""Reduce the read-only GPU receipt to aggregate evidence with no tensors/IDs."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import TypedDict, cast


class Cell(TypedDict):
    fixture: str
    batch: int
    n_out: int
    row: int
    request_index: int
    agreement: list[bool]
    confidence_max_abs: float
    alone_depth: int
    packed_depth: int


class Permutation(TypedDict):
    fixture: str
    batch: int
    n_out: int
    ids_exact: bool
    confidence_max_abs: float


class Receipt(TypedDict):
    arm: str
    graph_checks: int
    graph_exact: int
    window_checks: int
    window_exact: int
    prefix_checks: int
    prefix_exact: int
    fixtures: list[str]
    cells: list[Cell]
    permutation_cells: list[Permutation]
    insert_cells: list[dict[str, object]]
    runtime_sources: dict[str, str]
    prebuilt_extensions_sha256: dict[str, str]
    read_only_runtime: bool
    torch_version: str
    seconds: float
    peak_allocated_bytes: int


def collect(log: Path) -> dict[str, object]:
    lines = [line.removeprefix('RESULT_JSON ') for line in log.read_text().splitlines()
             if line.startswith('RESULT_JSON ')]
    if len(lines) != 1:
        raise ValueError('expected one complete GPU result')
    receipt = cast(Receipt, json.loads(lines[0]))
    if not receipt['read_only_runtime']:
        raise ValueError('missing read-only runtime')
    for key in ('graph', 'window', 'prefix'):
        checks = cast(dict[str, int], receipt)
        if checks[key+'_checks'] <= 0 or checks[key+'_checks'] != checks[key+'_exact']:
            raise ValueError(key+' invariance failed')
    cells = receipt['cells']
    summaries = []
    for fixture in ('captured', 'synthetic'):
        for batch in (2, 3):
            for n_out in (4, 5):
                selected = [c for c in cells if c['fixture'].startswith('capture' if fixture == 'captured' else 'synthetic')
                            and c['batch'] == batch and c['n_out'] == n_out]
                if not selected or any(len(c['agreement']) != n_out for c in selected):
                    raise ValueError('incomplete comparison coverage')
                summaries.append(dict(fixture=fixture, batch=batch, n_out=n_out, comparisons=len(selected),
                                      per_position_equal=[sum(c['agreement'][i] for c in selected) for i in range(n_out)],
                                      block_exact=sum(all(c['agreement']) for c in selected),
                                      confidence_max_abs=max(c['confidence_max_abs'] for c in selected),
                                      depth_changes=sum(c['alone_depth'] != c['packed_depth'] for c in selected),
                                      depth_deeper=sum(c['alone_depth'] < c['packed_depth'] for c in selected),
                                      depth_shallower=sum(c['alone_depth'] > c['packed_depth'] for c in selected),
                                      depth_transitions=dict(Counter(f"{c['alone_depth']}->{c['packed_depth']}"
                                                                     for c in selected))))
    same_shape = []
    for batch in (2, 3):
        selected_p = [p for p in receipt['permutation_cells'] if p['batch'] == batch]
        same_shape.append(dict(batch=batch, comparisons=len(selected_p), ids_exact=sum(p['ids_exact'] for p in selected_p),
                               confidence_max_abs=max(p['confidence_max_abs'] for p in selected_p)))
    return dict(label='MEASURED', arm=receipt['arm'], log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                mode='both TP2 ranks on bc62df2a, unchanged 225 W, flock -n, no physical pair',
                graph_checks=receipt['graph_checks'], graph_exact=receipt['graph_exact'],
                prefix_checks=receipt['prefix_checks'], prefix_exact=receipt['prefix_exact'],
                window_checks=receipt['window_checks'], window_exact=receipt['window_exact'],
                fixtures=receipt['fixtures'], agreement=summaries, same_shape=same_shape,
                insert_cells=receipt['insert_cells'], runtime_sources_sha256=receipt['runtime_sources'],
                prebuilt_extensions_sha256=receipt['prebuilt_extensions_sha256'],
                seconds=receipt['seconds'], peak_allocated_bytes=receipt['peak_allocated_bytes'],
                torch_version=receipt['torch_version'], read_only_runtime=receipt['read_only_runtime'],
                depth_policy='ESTIMATED inherited policy decisions, not target acceptance: '
                             'conf, cap 5 for B2 / 4 for B3, ROUND_MS=28.79,31.86,34.44,37.76,41.56,43.68 '
                             '+ DRAFT_MS=4.4; alone confidence masked to the SAME packed-batch cap',
                limits='12 early states of one captured request, plus 9 synthetic stress states. '
                       'Repeated permutations are not independent task samples. Windows held fixed for draft comparison. '
                       'Separate insert check measures projection dispatch. No acceptance or physical-TP2 claim.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(collect(args.log), indent=2)+'\n')


if __name__ == '__main__':
    main()
