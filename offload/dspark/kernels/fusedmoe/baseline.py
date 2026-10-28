"""Same-session KX-regfed + N1 epilogue control at median captured reuse."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import torch
from offload.dspark.native import formats as F, ops as NO
from . import FusedMoE, Workspace
from .measure import ROUTES, elapsed, load, telemetry, timing_graph


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--stage',type=int,default=0);p.add_argument('--rank',type=int,default=0)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if os.environ.get('KX_FUSEDMOE_LOCKED')!='1':raise RuntimeError('exclusive GPU lock required')
    torch.set_num_threads(8)
    op,_,_=load(a.stage,a.rank)
    d=op.gate.codes.device
    records=torch.load(ROUTES,weights_only=False)
    meta=json.loads((Path.home()/'dsv41-offload/kx-grouped/routes-final.json').read_text())
    live={r['round'] for r in meta if r['category']=='live'}
    result=[]
    for rows in (5,10,20):
        cases=[r for r in records if r['round'] in live and len(r['experts'][a.stage])==rows]
        if cases:
            cases.sort(key=lambda r:len(set(e for row in r['experts'][a.stage] for e in row)))
            host_ids=torch.tensor(cases[len(cases)//2]['experts'][a.stage],dtype=torch.int32)
        else:
            host_ids=torch.arange(rows*3,dtype=torch.int32).reshape(rows,3)
        torch.manual_seed(321+a.stage)
        x=F.act_qdq((torch.randn(rows,5120)*.1).to(torch.bfloat16)).to(d)
        ids=host_ids.to(d);rw=torch.full((rows,3),.5,device=d)
        out=torch.empty(rows,3,5120,device=d)
        gu=[torch.empty(rows,3,1152,device=d) for _ in range(2)]
        status=torch.zeros(1,dtype=torch.int32,device=d)
        ws=Workspace.allocate(rows,1152,d)
        sched=torch.empty(rows*3,8,dtype=torch.int32,device=d)
        def baseline() -> None:
            op.gate.linear_into(x,ids,gu[0],status,sched,up=op.up,up_out=gu[1])
            NO.swiglu_qdq(*gu,rw,out=ws.hq)
            op.down.linear_into(ws.hq,ids,out,status,sched,per_slot=True)
        for variant in (0,16,64,256,512,1024,2048,4096,8192,16384,65536,131072):
            candidate=FusedMoE(op.gate,op.up,op.down,op.ext,variant or 4)
            call=baseline if variant==0 else lambda:candidate.forward_into(x,ids,rw,out,status,ws)
            graph=timing_graph(call)
            before=telemetry();times=elapsed(graph);after=telemetry()
            result.append({'stage':a.stage,'rank':a.rank,'rows':rows,'unique':host_ids.unique().numel(),
                'variant':variant,'us':statistics.median(times),'samples_us':times,'before':before,'after':after})
            del graph
        assert int(status)==0
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
