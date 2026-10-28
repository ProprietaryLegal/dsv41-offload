"""Export measured per-cell TP2 depth costs; refuse sparse/mixed/unqualified rows.

Input is V1 depth_rows output. For async, --overlap-audited asserts that the
coordinator populated overlap_ms from the paired trace, rather than accepting
V1's default zero for an absent field. DRAFT_MS is the observed non-overlapped
remote prep cost, never isolated worker compute. ROUND_MS is verify-only.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def export(rows: list[dict[str, Any]], *, build: str, mode: str, requests: int,
           context: int, overlap_audited: bool) -> dict[str, Any]:
    if mode not in ('two_card_tp','remote_async'):
        raise ValueError('explicit two_card_tp or remote_async mode required')
    if mode == 'remote_async' and not overlap_audited:
        raise ValueError('async overlap must be audited; absent overlap is not zero overlap')
    cells: dict[int,list[dict[str,Any]]] = {d:[] for d in range(6)}
    for r in rows:
        if r['schema'] != 'dsv41.depth-round.v1' or r['build_id'] != build or r['mode'] != mode:
            raise ValueError('mixed schema, build/identity or execution mode')
        if r['n_requests'] != requests or r['context_tokens_max'] != context or r['cache'] != 'steady':
            raise ValueError('mixed request count, occupied-context cell or cache state')
        depths = r['depth_vector']
        if len(depths) != requests or len(set(depths)) != 1 or depths[0] not in cells:
            raise ValueError('cost fit requires fixed depth 0..5, uniform across this batch')
        for key in ('verify_ms','prep_ms','overlap_ms','queue_ms','round_ms'):
            if not math.isfinite(r[key]) or r[key] < 0:
                raise ValueError(f'invalid {key}')
        if r['overlap_ms'] > r['prep_ms'] or r['verify_ms'] <= 0:
            raise ValueError('overlap exceeds prep or verify cost is nonpositive')
        if r['queue_ms'] != 0:
            raise ValueError('isolate quiet cost measurement from queueing')
        cells[depths[0]].append(r)
    for d, rs in cells.items():
        fit = {member for r in rs if r['partition']=='fit' for member in r['group_id'].split('+')}
        held = {member for r in rs if r['partition']=='heldout' for member in r['group_id'].split('+')}
        if len(fit)<3 or len(held)<2 or fit & held:
            raise ValueError(f'depth {d}: need >=3 fit and >=2 disjoint held-out groups')
    fit_cells = {d:[r for r in rs if r['partition']=='fit'] for d,rs in cells.items()}
    table = [statistics.median(r['verify_ms'] for r in fit_cells[d]) for d in range(6)]
    if any(b<a for a,b in zip(table,table[1:])):
        raise ValueError('measured verify table is nonmonotonic; investigate/repeat, do not clamp it')
    costs = [r['prep_ms']-r['overlap_ms'] for d in range(1,6) for r in fit_cells[d]]
    draft = statistics.median(costs)
    held_errors = [(table[d]+draft-r['round_ms'])/max(r['round_ms'],1e-12)
                   for d,rs in cells.items() for r in rs if r['partition']=='heldout']
    return {'build_id':build,'mode':mode,'n_requests':requests,'occupied_context_tokens':context,
            'fit_verify_only_ms':table,'fit_effective_remote_ms':draft,
            'heldout_round_relative_error_median':statistics.median(held_errors),
            'heldout_round_relative_error_abs_max':max(abs(x) for x in held_errors),
            'env_lines':[f'VLLM_DSPARK_REMOTE_DRAFT_MS={draft:.6g}',
                         'VLLM_DSPARK_REMOTE_ROUND_MS='+','.join(f'{x:.6g}' for x in table)],
            'promotion':'review required; demonstrate held-out useful throughput gain before adoption'}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('rows',type=Path)
    ap.add_argument('--build',required=True,help='arm + full drafter identity + runtime revision')
    ap.add_argument('--mode',required=True)
    ap.add_argument('--requests',required=True,type=int)
    ap.add_argument('--context',required=True,type=int)
    ap.add_argument('--overlap-audited',action='store_true')
    ap.add_argument('--out',required=True,type=Path)
    a = ap.parse_args()
    rows = [json.loads(x) for x in a.rows.read_text().splitlines() if x.strip()]
    rep = export(rows,build=a.build,mode=a.mode,requests=a.requests,context=a.context,
                 overlap_audited=a.overlap_audited)
    a.out.write_text(json.dumps(rep,indent=2)+'\n')
    print('\n'.join(rep['env_lines']))


if __name__ == '__main__':
    main()
