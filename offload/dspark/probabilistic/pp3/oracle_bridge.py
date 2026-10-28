"""Exact PP1 law using q emitted by the real GPU producer (synthetic only)."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from fractions import Fraction
from pathlib import Path

import torch

from offload.dspark.probabilistic.reference import Categorical, acceptance, residual
from offload.dspark.worker.sampled import MASS, Top32
from offload.dspark.worker.tools.pp3_probe import UUID


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != UUID:
        raise RuntimeError('PP3 solo UUID required')
    import fcntl
    fcntl.flock(int(os.environ['PP3_LOCK_FD']),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-') != UUID.removeprefix('GPU-'):
        raise RuntimeError('wrong GPU')
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    dev=torch.device('cuda:0')
    sampler=Top32(1,64,dev)
    rows=[]
    for name, support, weights in (
        ('full',[40,41,42,43],[4.,1.,1.,2.]),
        ('sparse',[40,41],[1.,1.]),
        ('disjoint',[0,1,2,3],[4.,1.,1.,2.]),
        ('equal',[40,41,42,43],[1.,2.,2.,3.])):
        x=torch.full((1,64),-torch.inf,device=dev)
        x[0,support]=torch.tensor(weights,device=dev).log()*.7
        token,ids,q=sampler(x,torch.tensor([[4103,1,32768]],device=dev),torch.tensor([.7],device=dev))
        if sampler.error.any().item():
            raise AssertionError('GPU producer refused bridge input')
        counts=(q[0].double()*MASS).long().cpu().tolist()
        support_ids=ids[0].cpu().tolist()
        assert sum(counts)==MASS
        actual_q=Categorical(64,support_ids,counts)
        target=Categorical(64,[40,41,42,43],[1,2,2,3])
        accepted=[actual_q.prob(i)*acceptance(target,actual_q,i) if actual_q.prob(i) else Fraction()
                  for i in range(64)]
        reject=1-sum(accepted)
        correction=residual(target,actual_q) if reject else None
        law=[accepted[i]+(reject*correction.prob(i) if correction is not None else 0) for i in range(64)]
        assert law==[target.prob(i) for i in range(64)]
        assert actual_q.prob(int(token.item()))>0
        rows.append({'case':name,'support_ids':support_ids,'counts':counts,'exact_law':True})
    receipt={'schema':'pp3.gpu-q-pp1-oracle.v1','classification':'MEASURED actual GPU q, exact rational CPU law',
             'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':rows}
    for module_name in ('offload.dspark.worker.sampled', 'offload.dspark.probabilistic.reference'):
        module_path=importlib.import_module(module_name).__file__
        if module_path is None:
            raise RuntimeError('oracle/producer source path missing')
        receipt[module_name+'_sha256']=hashlib.sha256(Path(module_path).read_bytes()).hexdigest()
    a.out.write_text(json.dumps(receipt,indent=2)+'\n')
    print('PASS: actual GPU q recovers full target exactly in PP1 oracle (four synthetic cases)',flush=True)


if __name__ == '__main__':
    main()
