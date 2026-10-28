#!/usr/bin/env python3
"""Compare aligned teacher-forced full-vocabulary captures. Offline only; no free-running token alignment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, cast

import numpy as np
from numpy.typing import NDArray


def full_metrics(a: NDArray[np.float64], b: NDArray[np.float64], tolerance: float = 1e-5) -> dict[str, object]:
    if a.ndim != 1 or a.shape != b.shape or a.size < 2:
        raise ValueError('full distributions must have matching one-dimensional vocabulary')
    for name,x in [('A',a),('B',b)]:
        if np.isnan(x).any() or np.isposinf(x).any() or (x>tolerance).any():
            raise ValueError(f'{name}: invalid logprob')
        mass=float(np.exp(x).sum(dtype=np.float64))
        if abs(mass-1.0)>tolerance:
            raise ValueError(f'{name}: logprob mass {mass}; truncated/top-k or unnormalized captures refused')
    pa,pb=np.exp(a),np.exp(b)
    def kl(x: NDArray[np.float64], y: NDArray[np.float64], p: NDArray[np.float64]) -> float | str:
        positive=np.isfinite(x)
        if np.isneginf(y[positive]).any():
            return 'infinite'
        return max(0.0,float(np.sum(p[positive]*(x[positive]-y[positive]),dtype=np.float64)))
    finite=np.isfinite(a)&np.isfinite(b)
    max_delta=float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.0
    top_a=int(np.argmax(a)); top_b=int(np.argmax(b))
    return {'kl_a_b':kl(a,b,pa),'kl_b_a':kl(b,a,pb),'total_variation':float(np.abs(pa-pb).sum()/2),
            'top1_a':top_a,'top1_b':top_b,'top1_equal':top_a==top_b,
            'max_finite_dlogprob':max_delta,'support_mismatch':bool((np.isfinite(a)!=np.isfinite(b)).any())}


def load_manifest(path: Path) -> dict[str, object]:
    obj: object=json.loads(path.read_text())
    if not isinstance(obj,dict) or obj.get('schema')!='dsv41.fixed-prefix.v1':
        raise ValueError('expected dsv41.fixed-prefix.v1 manifest')
    result=cast(dict[str,object],obj)
    vocab=result.get('vocab_size')
    if type(vocab) is not int or vocab<2 or result.get('vocab_order')!='token_id':
        raise ValueError('full token-id vocabulary order required')
    rows=result.get('rows')
    if not isinstance(rows,list) or not rows:
        raise ValueError('capture must contain rows')
    for field in ['target_identity','tokenizer_identity','numerical_config','run_id','build_id']:
        if not isinstance(result.get(field),str) or not result[field]:
            raise ValueError(f'missing {field}')
    return result


def keyed(manifest: Mapping[str,object]) -> dict[tuple[str,int],dict[str,object]]:
    result: dict[tuple[str,int],dict[str,object]]={}
    for raw in cast(list[object],manifest['rows']):
        if not isinstance(raw,dict):
            raise ValueError('capture row must be object')
        row=cast(dict[str,object],raw)
        case=row.get('case_id'); pos=row.get('position')
        if not isinstance(case,str) or not case or type(pos) is not int or pos<0:
            raise ValueError('case_id/position invalid')
        for field in ['prefix_token_sha256','batch_token_sha256','schedule_sha256','array_sha256']:
            value=row.get(field)
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError(f'invalid {field}')
        token=row.get('forced_next_id')
        if type(token) is not int or not 0<=token<cast(int,manifest['vocab_size']):
            raise ValueError('invalid teacher-forced token id')
        if not isinstance(row.get('shape'),dict) or not row['shape']:
            raise ValueError('explicit shape metadata required')
        shape=cast(dict[str,object],row['shape'])
        required={'n_requests','query_lens','graph_mode','graph_rows','spec_depths','n_out','chunk_tokens'}
        if not required.issubset(shape):
            raise ValueError('shape missing required batch/graph/chunk fields')
        for field in ['n_requests','graph_rows','n_out','chunk_tokens']:
            if type(shape[field]) is not int or cast(int,shape[field]) < 0:
                raise ValueError(f'invalid shape {field}')
        if not 1 <= cast(int,shape['n_requests']) <= 4 or not isinstance(shape['graph_mode'],str):
            raise ValueError('invalid shape request count/graph mode')
        for field in ['query_lens','spec_depths']:
            vals=shape[field]
            if not isinstance(vals,list) or len(vals)!=shape['n_requests'] or any(type(x) is not int or x<0 for x in vals):
                raise ValueError(f'invalid shape {field}')
        key=(case,pos)
        if key in result:
            raise ValueError(f'duplicate aligned row {key}')
        result[key]=row
    return result


def distribution(root: Path, row: Mapping[str,object], vocab: int) -> NDArray[np.float64]:
    rel=row.get('logprobs_file')
    if not isinstance(rel,str) or not rel:
        raise ValueError('missing logprobs_file')
    path=(root/rel).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('capture array must stay inside its manifest directory')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=row['array_sha256']:
        raise ValueError('capture array sha256 mismatch')
    arr=np.load(path,allow_pickle=False,mmap_mode='r')
    if not isinstance(arr,np.ndarray) or arr.shape!=(vocab,) or arr.dtype.kind!='f':
        raise ValueError('array must contain one float logprob for EVERY token id')
    return np.asarray(arr,dtype=np.float64)


def compare(path_a: Path, path_b: Path) -> dict[str,object]:
    a,b=load_manifest(path_a),load_manifest(path_b)
    for key in ['vocab_size','vocab_order','target_identity','tokenizer_identity','numerical_config']:
        if a[key]!=b[key]:
            raise ValueError(f'mismatched {key}')
    aa,bb=keyed(a),keyed(b)
    if aa.keys()!=bb.keys():
        raise ValueError('missing/extra teacher-forced rows; partial alignment refused')
    rows: list[dict[str,object]]=[]
    for key,ra in sorted(aa.items()):
        rb=bb[key]
        for field in ['forced_next_id','prefix_token_sha256','batch_token_sha256','schedule_sha256','shape']:
            if ra[field]!=rb[field]:
                raise ValueError(f'{key}: mismatched {field}; free-running/shape-mismatched comparison refused')
        ma=distribution(path_a.parent,ra,cast(int,a['vocab_size']))
        mb=distribution(path_b.parent,rb,cast(int,b['vocab_size']))
        metrics=full_metrics(ma,mb)
        token=cast(int,ra['forced_next_id'])
        la,lb=float(ma[token]),float(mb[token])
        metrics['forced_logprob_a']=la if math.isfinite(la) else '-infinite'
        metrics['forced_logprob_b']=lb if math.isfinite(lb) else '-infinite'
        rows.append({'case_id':key[0],'position':key[1],**metrics})
    return {'schema':'dsv41.fixed-prefix-comparison.v1','run_a':a['run_id'],'run_b':b['run_id'],
            'build_a':a['build_id'],'build_b':b['build_id'],'rows':rows,
            'interpretation':'full distributions at identical token prefixes and declared shapes; no universal pass threshold'}


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('a',type=Path); ap.add_argument('b',type=Path); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    result=compare(args.a,args.b)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'rows':len(cast(list[object],result['rows'])),'out':str(args.out)}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
