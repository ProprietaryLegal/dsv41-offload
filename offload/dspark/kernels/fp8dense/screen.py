"""Offline register-fed shape screen. Outputs measurements and exactness diagnostics."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from collections.abc import Callable
import torch
from .api import require_lock, DenseFP8
from ...native.formats import act_qdq, decode_fp8_block
from ...native.official import OfficialShards
from ...native.ops import NativeDense


def telemetry() -> str:
    return subprocess.check_output(['nvidia-smi', '-i', os.environ['CUDA_VISIBLE_DEVICES'],
        '--query-gpu=uuid,power.limit,power.draw,temperature.gpu,clocks.sm,clocks.mem,utilization.gpu,memory.used',
        '--format=csv,noheader'], text=True).strip()


def measured(fn: Callable[[], None], *, warm_seconds: float = 0.0) -> dict[str, object]:
    for _ in range(3): fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(20): fn()
    # Final qualification may settle dynamic clocks with bounded graph batches.
    # Synchronization here is outside the kernel call and measured interval.
    deadline=time.monotonic()+warm_seconds
    while time.monotonic()<deadline:
        for _ in range(8):graph.replay()
        torch.cuda.synchronize()
    samples: list[float] = []
    before = telemetry()
    for _ in range(5):
        a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();graph.replay();b.record();b.synchronize()
        samples.append(a.elapsed_time(b)*1000/20)
    return {'median_us': statistics.median(samples), 'samples_us': samples,
            'telemetry_before': before, 'telemetry_after': telemetry()}


SHAPES = {
    'wq_a': ('attn.wq_a', (1280,5120), 'rep', False),
    'wkv': ('attn.wkv', (512,5120), 'rep', False),
    'wq_b': ('attn.wq_b', (16384,1280), 'row', False),
    'wo_b': ('attn.wo_b', (5120,4096), 'col', True),
    'w1': ('ffn.shared_experts.w1', (1152,5120), 'row', False),
    'w3': ('ffn.shared_experts.w3', (1152,5120), 'row', False),
    'w2': ('ffn.shared_experts.w2', (5120,1152), 'col', True),
    'wo_a': ('attn.wo_a', (4096,4096), 'row', False),
    'main_proj': ('main_proj', (5120,15360), 'rep', False),
}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--shapes',nargs='+',default=list(SHAPES))
    parser.add_argument('--tune',action='store_true')
    parser.add_argument('--layout',choices=['register','simt'],default='register')
    args=parser.parse_args()
    if os.environ.get('KX_FP8DENSE_LOCKED')!='1' or os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc':
        raise RuntimeError('use run_locked.sh')
    require_lock()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.manual_seed(12354)
    sh=OfficialShards('/opt/dsv41/x86host-data/native-1017')
    results: list[dict[str,object]]=[]
    for shape in args.shapes:
        suffix,(n,k),cut,f32=SHAPES[shape]
        name='mtp.0.'+suffix
        w,s=sh.get(name+'.weight'),sh.get(name+'.scale')
        wc=w[:n,:k].contiguous()
        dense=DenseFP8.from_checkpoint(wc,s,torch.device('cuda:0'),groups=4 if shape=='wo_a' else 1,layout=args.layout)
        wd=decode_fp8_block(wc,s)
        oracle=NativeDense(name,'fp8',wc.cuda(), torch.ldexp(torch.ones_like(s,dtype=torch.float32),s.int()-127)[:n//32,:k//32].cuda())
        for m in ((8,32,128) if shape=='main_proj' else (5,20)):
            x=act_qdq(torch.randn(m,k*dense.groups,dtype=torch.bfloat16)*.1).cuda()
            y=torch.empty((m,n),device='cuda',dtype=torch.float32 if f32 else torch.bfloat16)
            if shape=='wo_a':
                ref=torch.cat([x[:,i*k:(i+1)*k].double()@wd[i*1024:(i+1)*1024].cuda().double().t() for i in range(4)],1).float()
                base=None
            else:
                ref=(x.double()@wd.cuda().double().t()).float()
                base=oracle(x,out_f32=f32)
            configs=[(1,1)] if args.layout=='simt' else [(r,sp) for r in (1,2,4,8,16) if r<=max(4,(m+7)//8) for sp in (1,2,4,8,16)] if args.tune else [(min(16,1<<((m-1)//8).bit_length()),1)]
            for rt,split in configs:
                scratch=torch.empty((split,m,dense.groups,dense.outputs),device='cuda') if split>1 else None
                call=lambda: dense.linear_into(x,y,scratch=scratch,splits=split,row_tiles=rt)
                call();torch.cuda.synchronize()
                err=y.float()-ref
                record:dict[str,object]={'shape':shape,'m':m,'row_tiles':rt,'splits':split,
                    'relative_l2':float(err.norm()/ref.norm()),'max_abs':float(err.abs().max()),
                    'bf16_vs_f64':float((y.bfloat16()==ref.bfloat16()).float().mean()),
                    'bf16_vs_n1':None if base is None else float((y.bfloat16()==base.bfloat16()).float().mean()),
                    **measured(call)}
                results.append(record)
                args.output.write_text(json.dumps(results,indent=2)+'\n')
            best=min(results[-len(configs):],key=lambda r:float(r['median_us']))
            print(json.dumps(best),flush=True)
        del dense,oracle,wd,w,s,wc
        torch.cuda.empty_cache()


if __name__=='__main__': main()
