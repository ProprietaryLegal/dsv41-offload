#!/usr/bin/env python3
"""CPU-only grouped cost-table fitting. Reads paired engine rounds, never infers phases from SSE gaps."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Mapping, Sequence, cast

MODES = {'local', 'remote_sync', 'remote_async', 'two_card_tp', 'two_card_affine'}


def integer(obj: Mapping[str, object], key: str, lo: int, hi: int) -> int:
    value = obj.get(key)
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError(f'{key} must be integer {lo}..{hi}')
    return value


def number(obj: Mapping[str, object], key: str, *, positive: bool = False) -> float:
    value = obj.get(key)
    if type(value) not in (float, int):
        raise ValueError(f'{key} must be numeric')
    result = float(cast(float | int, value))
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f'{key} must be finite and {">0" if positive else ">=0"}')
    return result


def label(obj: Mapping[str, object], key: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{key} must be nonempty text')
    return value


def bucket(context: int) -> str:
    for limit, name in [(8192,'short'), (32768,'32k'), (150000,'150k'), (256000,'256k'), (300000,'300k')]:
        if context <= limit:
            return name
    return 'over300k'


@dataclass(frozen=True, order=True)
class Cell:
    mode: str
    n_requests: int
    planned_verify_rows: int
    graph_rows: int
    context_bucket: str
    cache: str
    depth_vector: tuple[int, ...]


@dataclass(frozen=True)
class Round:
    sample_id: str
    group_id: str  # stable hash of legal task/source family; shared across all mode/depth/repeat runs
    partition: str
    build_id: str
    cell: Cell
    verify_ms: float
    prep_ms: float
    overlap_ms: float
    queue_ms: float
    round_ms: float
    committed: int

    @classmethod
    def parse(cls, obj: Mapping[str, object]) -> Round:
        if obj.get('schema') != 'dsv41.depth-round.v1':
            raise ValueError('expected dsv41.depth-round.v1; bench2 request/run aggregates are NOT paired rounds')
        n=integer(obj,'n_requests',1,4)
        raw=obj.get('depth_vector')
        if not isinstance(raw,list) or len(raw)!=n or any(type(x) is not int or not 0 <= x <= 5 for x in raw):
            raise ValueError('depth_vector must contain one integer 0..5 per request')
        depths=tuple(cast(list[int],raw))
        rows=integer(obj,'planned_verify_rows',n,24)
        if rows != sum(1+d for d in depths):
            raise ValueError('planned_verify_rows must equal sum(1+depth)')
        graph_rows=integer(obj,'graph_rows',rows,4096)
        ctx=integer(obj,'context_tokens_max',0,10_000_000)
        mode=label(obj,'mode')
        if mode not in MODES:
            raise ValueError(f'unsupported mode {mode}')
        cache=label(obj,'cache')
        if cache not in {'cold','prefix_hit','steady'}:
            raise ValueError('cache must be cold, prefix_hit or steady')
        partition=label(obj,'partition')
        if partition not in {'fit','heldout'}:
            raise ValueError('partition must be fit or heldout')
        verify=number(obj,'verify_ms',positive=True)
        prep=number(obj,'prep_ms')
        overlap=number(obj,'overlap_ms')
        if overlap > min(verify,prep):
            raise ValueError('overlap_ms cannot exceed either paired interval')
        cost=number(obj,'round_ms',positive=True)
        # No sum assertion: phase intervals can overlap CPU scheduling and do not describe all work.
        return cls(label(obj,'sample_id'),label(obj,'group_id'),partition,label(obj,'build_id'),
                   Cell(mode,n,rows,graph_rows,bucket(ctx),cache,depths),verify,prep,overlap,
                   number(obj,'queue_ms'),cost,integer(obj,'committed_tokens',0,rows))


def quantile(values: Sequence[float], p: float) -> float:
    if not values or not 0 <= p <= 1:
        raise ValueError('invalid quantile input')
    ordered=sorted(values)
    return ordered[max(0,math.ceil(p*len(ordered))-1)]


def fit(rows: Sequence[Round], min_fit: int = 3, min_heldout: int = 2) -> dict[str, object]:
    if min_fit < 1 or min_heldout < 1 or not rows:
        raise ValueError('positive fit/heldout minima and at least one paired round required')
    group_split: dict[str,str]={}
    ids: set[str]=set()
    builds: dict[str,str]={}
    cells: dict[Cell,list[Round]]=defaultdict(list)
    for row in rows:
        if row.sample_id in ids:
            raise ValueError(f'duplicate sample_id {row.sample_id}')
        ids.add(row.sample_id)
        prior=group_split.setdefault(row.group_id,row.partition)
        if prior != row.partition:
            raise ValueError(f'group leakage: {row.group_id}')
        prior_build=builds.setdefault(row.cell.mode,row.build_id)
        if prior_build != row.build_id:
            raise ValueError(f'mixed builds within {row.cell.mode}; fit a new table')
        cells[row.cell].append(row)
    table: list[dict[str,object]]=[]
    errors: list[dict[str,object]]=[]
    unqualified: list[dict[str,object]]=[]
    for cell,population in sorted(cells.items()):
        train=[r for r in population if r.partition=='fit']
        hold=[r for r in population if r.partition=='heldout']
        # Count independent task groups, not correlated rounds from one prompt.
        ng=len({r.group_id for r in train}); nh=len({r.group_id for r in hold})
        if ng<min_fit or nh<min_heldout:
            unqualified.append({'cell':asdict(cell),'fit_groups':ng,'heldout_groups':nh})
        if not train:
            continue  # explicit unsupported cell; never nearest-neighbour/default cost substitution
        costs=[r.round_ms for r in train]
        prediction=statistics.fmean(costs)
        entry: dict[str,object]={'cell':asdict(cell),'fit_rounds':len(train),'fit_groups':ng,
                                'heldout_rounds':len(hold),'heldout_groups':nh,
                                'round_mean_ms':prediction,'round_p95_ms':quantile(costs,.95),
                                'committed_mean':statistics.fmean(r.committed for r in train),
                                'verify_mean_ms':statistics.fmean(r.verify_ms for r in train),
                                'prep_mean_ms':statistics.fmean(r.prep_ms for r in train),
                                'overlap_mean_ms':statistics.fmean(r.overlap_ms for r in train),
                                'queue_mean_ms':statistics.fmean(r.queue_ms for r in train)}
        table.append(entry)
        for row in hold:
            errors.append({'sample_id':row.sample_id,'group_id':row.group_id,'cell':asdict(cell),
                           'predicted_ms':prediction,'observed_ms':row.round_ms,
                           'error_ms':row.round_ms-prediction,'relative_error':(row.round_ms-prediction)/prediction,
                           'predicted_committed':entry['committed_mean'],'actual_committed':row.committed})
    return {'schema':'dsv41.depth-table.v1','qualified_coverage':not unqualified,'builds':builds,
            'cells':table,'heldout_errors':errors,'unqualified_cells':unqualified,
            'note':'Costs are exact-cell direct-round means. Phase means are diagnostics, never summed with round_ms.'}


def score(expected_committed: float, round_mean_ms: float, round_p95_ms: float, tail_weight: float) -> float:
    vals=(expected_committed,round_mean_ms,round_p95_ms,tail_weight)
    if any(not math.isfinite(x) or x<0 for x in vals) or round_mean_ms<=0:
        raise ValueError('invalid policy score inputs')
    return expected_committed/(round_mean_ms+tail_weight*max(round_p95_ms-round_mean_ms,0.0))


def read(path: Path) -> list[Round]:
    rows=[]
    for line_no,line in enumerate(path.read_text().splitlines(),1):
        if not line.strip():
            continue
        obj: object=json.loads(line)
        if not isinstance(obj,dict):
            raise ValueError(f'{path}:{line_no}: expected object')
        try:
            rows.append(Round.parse(cast(Mapping[str,object],obj)))
        except ValueError as exc:
            raise ValueError(f'{path}:{line_no}: {exc}') from exc
    return rows


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input',type=Path); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--min-fit',type=int,default=3); ap.add_argument('--min-heldout',type=int,default=2)
    args=ap.parse_args()
    result=fit(read(args.input),args.min_fit,args.min_heldout)
    result['source_sha256']=hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'qualified_coverage':result['qualified_coverage'],'cells':len(cast(list[object],result['cells'])),
                      'unqualified_cells':len(cast(list[object],result['unqualified_cells']))}))
    return 0 if result['qualified_coverage'] else 2


if __name__=='__main__':
    raise SystemExit(main())
