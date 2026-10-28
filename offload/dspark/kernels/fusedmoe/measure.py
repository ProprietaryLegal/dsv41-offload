"""Native fixture loading and reproducible lock-qualified CUDA graph timings.

Routing stays on X86host; only nonproprietary timing/count summaries leave it.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Callable

import torch
from offload.dspark.native.official import OfficialShards
from offload.dspark.native import formats as F, ops as NO
from offload.dspark.kernels.mxfp4.regfed import RegisterExperts, extension as reg_extension
from . import FusedMoE, Workspace

ROOT = '/opt/dsv41/x86host-data/native-1017'
ROUTES = Path.home()/'dsv41-offload/n1/results/parity_native-hybrid_emu_drafts.pt'


def telemetry() -> str:
    return subprocess.check_output(['nvidia-smi', '-i', os.environ['CUDA_VISIBLE_DEVICES'],
        '--query-gpu=timestamp,uuid,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu',
        '--format=csv,noheader'], text=True).strip()


def load(stage: int, rank: int, experts: int = 128) -> tuple[FusedMoE, NO.ExpertBank, OfficialShards]:
    """Stream one official expert at a time to raw oracle and prepacked final banks.

    All 16 E2M1 codes with scales 111..127 have been exhaustively qualified by
    KX-MXFP4. Check every scale against that admitted finite-normal BF16 domain.
    Verify every source tensor against N1's per-tensor manifest before use.
    """
    d = torch.device('cuda:0')
    sh = OfficialShards(ROOT)
    bank = NO.ExpertBank('fusedmoe-oracle', 'packed', experts, 5120, 1152, d)
    ext = reg_extension()
    regs = []
    for n, k in ((1152,5120),(1152,5120),(5120,1152)):
        regs.append(RegisterExperts(torch.empty(experts,n,k//2,dtype=torch.uint8,device=d),
            torch.empty(experts,n,k//32,dtype=torch.uint8,device=d),experts,n,k,ext))
    for e in range(experts):
        for i, proj in enumerate(('w1','w3','w2')):
            key = f'mtp.{stage}.ffn.experts.{e}.{proj}'
            for suffix in ('.weight','.scale'):
                sh.verify(key+suffix)
            w, s = sh.get(key+'.weight'), sh.get(key+'.scale')
            if i < 2:
                w=w[rank*1152:(rank+1)*1152].contiguous()
                s=s[rank*1152:(rank+1)*1152].contiguous()
                bank.w13[e,i].copy_(w);bank.s13[e,i].copy_(s)
            else:
                w=w[:,rank*576:(rank+1)*576].contiguous()
                s=s[:,rank*36:(rank+1)*36].contiguous()
                bank.w2[e].copy_(w);bank.s2[e].copy_(s)
            if int(s.min()) < 111 or int(s.max()) > 127:
                raise ValueError(f'{key}: outside qualified decoder scale domain')
            reg=regs[i]
            assert w.shape == (reg.outputs,reg.inputs//2) and s.shape == (reg.outputs,reg.inputs//32)
            wd,sd=w.to(d),s.to(d)
            ext.prepack(reg.codes[e].data_ptr(),reg.scales[e].data_ptr(),wd.data_ptr(),sd.data_ptr(),
                        reg.outputs,reg.inputs,torch.cuda.current_stream(d).cuda_stream)
        if e%32==0:
            print(f'load stage={stage} rank={rank} expert={e}',flush=True)
    return FusedMoE.from_register_banks(*regs), bank, sh


def timing_graph(call: Callable[[], None], iterations: int = 20) -> torch.cuda.CUDAGraph:
    for _ in range(3): call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iterations): call()
    graph.replay();torch.cuda.synchronize()
    return graph


def elapsed(graph: torch.cuda.CUDAGraph, iterations: int = 20) -> list[float]:
    samples=[]
    for _ in range(3):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();graph.replay();b.record();b.synchronize()
        samples.append(a.elapsed_time(b)*1000/iterations)
    return samples


def percentile(xs: list[float], q: float) -> float:
    return sorted(xs)[min(len(xs)-1,int((len(xs)-1)*q))]


def screen(stage: int, rank: int, output: Path, limit: int = 0, variants: tuple[int, ...] = (16,64)) -> None:
    op,_,_=load(stage,rank)
    d=op.gate.codes.device
    rounds=torch.load(ROUTES,weights_only=False)
    category_file=Path.home()/'dsv41-offload/kx-grouped/routes-final.json'
    categories={int(r['round']):r['category'] for r in json.loads(category_file.read_text())}
    assert sum(categories[r['round']]=='live' and len(r['experts'][0])==5 for r in rounds)==337
    assert sum(categories[r['round']]=='live' and len(r['experts'][0])==10 for r in rounds)==106
    result: dict[str, object]={'stage':stage,'rank':rank,'torch':torch.__version__,
        'source_sha256':hashlib.sha256(Path(__file__).with_name('fusedmoe.cu').read_bytes()).hexdigest(),
        'route_sha256':hashlib.sha256(ROUTES.read_bytes()).hexdigest(),
        'input':'synthetic seeded bf16 normal*0.1, official activation QDQ; rw=0.5',
        'conditions':'exclusive flock; graph20 iterations,3 warmups,3 event trials per fixture; no clocks changed'}
    records=[]
    for rows in (5,10,20):
        torch.manual_seed(321+stage)
        x=F.act_qdq((torch.randn(rows,5120)*0.1).to(torch.bfloat16)).to(d)
        ids=torch.zeros(rows,3,dtype=torch.int32,device=d)
        rw=torch.full((rows,3),0.5,device=d)
        store=torch.empty(rows,4,5120,device=d)
        out=store[:,:3]
        status=torch.zeros(1,dtype=torch.int32,device=d)
        ws=Workspace.allocate(rows,1152,d)
        fixtures=[(r['round'],torch.tensor(r['experts'][stage],dtype=torch.int32)) for r in rounds
                  if len(r['experts'][stage])==rows and categories[r['round']]=='live']
        if limit: fixtures=fixtures[:limit]
        fixtures.append((-1,torch.arange(rows*3,dtype=torch.int32).reshape(rows,3)))
        for warps in variants:
            variant=FusedMoE(op.gate,op.up,op.down,op.ext,warps)
            graph=timing_graph(lambda:variant.forward_into(x,ids,rw,out,status,ws))
            for rnd,host_ids in fixtures:
                ids.copy_(host_ids);graph.replay();torch.cuda.synchronize()
                before=telemetry();times=elapsed(graph);after=telemetry()
                records.append({'rows':rows,'round':rnd,'unique':host_ids.unique().numel(),'warps':warps,
                    'us':statistics.median(times),'samples_us':times,'before':before,'after':after})
            assert int(status)==0
            del graph
    result['records']=records
    summary=[]
    for rows in (5,10,20):
        for warps in variants:
            live=[float(r['us']) for r in records if r['rows']==rows and r['warps']==warps and r['round']!=-1]
            worst=[r['us'] for r in records if r['rows']==rows and r['warps']==warps and r['round']==-1][0]
            summary.append({'rows':rows,'warps':warps,'rounds':len(live),
                'median_us':statistics.median(live) if live else None,
                'p90_us':percentile(live,0.9) if live else None,'distinct_us':worst})
    result['summary']=summary
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--stage',type=int,default=0);p.add_argument('--rank',type=int,default=0)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--variants',type=int,nargs='+',default=[16,64]);args=p.parse_args()
    if os.environ.get('KX_FUSEDMOE_LOCKED')!='1':raise RuntimeError('run through run_locked.sh')
    torch.set_num_threads(8)
    screen(args.stage,args.rank,args.output,args.limit,tuple(args.variants))

if __name__=='__main__':main()
