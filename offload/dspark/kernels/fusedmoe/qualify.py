"""All-expert native numerical qualification, one stage/rank per bounded GPU window."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import NotRequired, TypedDict

import torch
from offload.dspark.native import formats as F, ops as NO
from . import Workspace
from .measure import load, telemetry
from .oracle import n1_projection_oracle


class Qualification(TypedDict):
    stage: int
    rank: int
    variant: int
    source_sha256: NotRequired[str]
    before: str
    input: str
    elements: int
    n1_bf16_mismatches: list[dict[str, int | float | str]]
    f64_bf16_mismatches: list[dict[str, int | float | str]]
    hq_self_mismatches: int
    hq_n1_mismatches: int
    slot_batches: list[dict[str, int | float]]
    experts_touched: list[int]
    n1_bf16_agreement: NotRequired[float]
    after: NotRequired[str]
    elapsed_s: NotRequired[float]
    gates: NotRequired[dict[str, bool]]


def qualify(stage: int, rank: int, output: Path, variant: int = 16) -> None:
    started=time.monotonic()
    op,bank,sh=load(stage,rank)
    op=type(op)(op.gate,op.up,op.down,op.ext,variant)
    d=op.gate.codes.device
    result: Qualification={'stage':stage,'rank':rank,'variant':variant,'before':telemetry(),
        'source_sha256':hashlib.sha256(Path(__file__).with_name('fusedmoe.cu').read_bytes()).hexdigest(),
        'input':'seed=20261023+stage; bf16 normal*0.1, official per32 QDQ; route weights uniform(0,1)',
        'elements':0,'n1_bf16_mismatches':[],'f64_bf16_mismatches':[],'hq_self_mismatches':0,
        'hq_n1_mismatches':0,'slot_batches':[],'experts_touched':[]}
    # Every one of 128 experts is actually selected. Tail repeats expert 127.
    seen=set()
    for base in (0,60,120):
        rows=20 if base<120 else 3
        torch.manual_seed(20261023+stage+base)
        xcpu=F.act_qdq((torch.randn(rows,5120)*0.1).to(torch.bfloat16))
        x=xcpu.to(d)
        ids_cpu=torch.arange(base,base+rows*3,dtype=torch.int32).clamp_max(127).reshape(rows,3)
        ids=ids_cpu.to(d)
        rw=torch.rand(rows,3,device=d)
        ws=Workspace.allocate(rows,1152,d)
        status=torch.zeros(1,dtype=torch.int32,device=d)
        storage=torch.full((rows,4,5120),12345.,device=d)
        out=storage[:,:3]
        gu=tuple(torch.empty(rows,3,1152,device=d) for _ in range(2))
        nu=tuple(torch.empty_like(v) for v in gu)
        op.forward_into(x,ids,rw,out,status,ws,projections=gu)
        self_h=NO.swiglu_qdq(*gu,rw)
        mismatch=int((self_h.view(torch.int16)!=ws.hq.view(torch.int16)).sum())
        result['hq_self_mismatches']+=mismatch
        assert mismatch==0,'fused hq differs from N1 swiglu_qdq of own projections'
        n1_projection_oracle[(rows*3,1152//64)](x,ids,rw,bank.w13,bank.s13,*nu,1152,5120,10.,
            PACKED=True,BN=64,BK=256,INTERP=False,num_warps=4,enable_fp_fusion=False)
        nh=torch.empty_like(ws.hq)
        no=torch.empty(rows,3,5120,device=d)
        bank.forward_slots(x,ids,rw,h=nh,out=no)
        # Instrumented projection reduction must reproduce unmodified N1 hq exactly.
        assert torch.equal(NO.swiglu_qdq(*nu,rw).view(torch.int16),nh.view(torch.int16))
        result['hq_n1_mismatches']+=int((ws.hq.view(torch.int16)!=nh.view(torch.int16)).sum())
        result['elements']+=2*gu[0].numel()
        ref_gu=[torch.empty_like(v,device='cpu') for v in gu]
        own_gu=[v.cpu() for v in gu];n1_gu=[v.cpu() for v in nu]
        ref_down=[]
        for p,e in enumerate(ids_cpu.flatten().tolist()):
            seen.add(e)
            for z,proj in enumerate(('w1','w3')):
                key=f'mtp.{stage}.ffn.experts.{e}.{proj}'
                w,s=sh.get(key+'.weight'),sh.get(key+'.scale')
                wd=F.decode_fp4(w[rank*1152:(rank+1)*1152],s[rank*1152:(rank+1)*1152])
                ref=(xcpu[p//3].double()@wd.double().t()).float()
                ref_gu[z][p//3,p%3]=ref
                own=own_gu[z][p//3,p%3];n1=n1_gu[z][p//3,p%3]
                for name,other in (('n1_bf16_mismatches',n1),('f64_bf16_mismatches',ref)):
                    diff=own.to(torch.bfloat16).view(torch.int16)!=other.to(torch.bfloat16).view(torch.int16)
                    for c in diff.nonzero().flatten().tolist():
                        result[name].append({'expert':e,'slot':p,'channel':c,'projection':proj,
                            'got_f32':float(own[c]),'n1_f32':float(n1[c]),'f64_sum_f32':float(ref[c]),
                            'got_bf16':float(own[c].to(torch.bfloat16)),
                            'oracle_bf16':float(other[c].to(torch.bfloat16))})
            if p%30==0:print(f'oracle stage={stage} rank={rank} expert={e}',flush=True)
        ref_h=NO.swiglu_qdq(*(v.to(d) for v in ref_gu),rw).cpu()
        own_h=ws.hq.cpu()
        for p,e in enumerate(ids_cpu.flatten().tolist()):
            key=f'mtp.{stage}.ffn.experts.{e}.w2'
            w,s=sh.get(key+'.weight'),sh.get(key+'.scale')
            wd=F.decode_fp4(w[:,rank*576:(rank+1)*576],s[:,rank*36:(rank+1)*36]).double()
            ref=ref_h[p].double()@wd.t()
            own_ref=own_h[p].double()@wd.t()
            ref_down.append((ref,own_ref))
        ref=torch.stack([v[0] for v in ref_down]).reshape(rows,3,5120)
        own_ref=torch.stack([v[1] for v in ref_down]).reshape(rows,3,5120)
        got=out.cpu().double()
        err=float(torch.linalg.vector_norm(got-ref)/torch.linalg.vector_norm(ref))
        down_err=float(torch.linalg.vector_norm(got-own_ref)/torch.linalg.vector_norm(own_ref))
        per_slot=torch.linalg.vector_norm(got-ref,dim=-1)/torch.linalg.vector_norm(ref,dim=-1)
        result['slot_batches'].append({'base':base,'full_f64_relative_frobenius':err,
            'down_from_own_hq_relative_frobenius':down_err,'max_slot_relative':float(per_slot.max()),
            'vs_n1_relative':float(torch.linalg.vector_norm(out-no)/torch.linalg.vector_norm(no))})
        assert int(status)==0 and bool((storage[:,3]==12345.).all())
        output.write_text(json.dumps(result,indent=2)+'\n') # resumable evidence
    result['experts_touched']=sorted(seen)
    result['n1_bf16_agreement']=1-len(result['n1_bf16_mismatches'])/result['elements']
    result['after']=telemetry();result['elapsed_s']=time.monotonic()-started
    result['gates']={'hq_self_exact':result['hq_self_mismatches']==0,
        'gu_n1_bf16_9999':result['n1_bf16_agreement']>=.9999,
        'all128':len(seen)==128,
        'slots_f64_1e6':max(b['full_f64_relative_frobenius'] for b in result['slot_batches'])<=1e-6}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('n1_bf16_mismatches','f64_bf16_mismatches')},indent=2),flush=True)


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--stage',type=int,required=True);p.add_argument('--rank',type=int,required=True)
    p.add_argument('--variant',type=int,default=16);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if os.environ.get('KX_FUSEDMOE_LOCKED')!='1':raise RuntimeError('exclusive run_locked.sh required')
    torch.set_num_threads(8)
    qualify(a.stage,a.rank,a.output,a.variant)

if __name__=='__main__':main()
