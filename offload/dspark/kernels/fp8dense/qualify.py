"""All-stage/both-rank official tensor qualification, bounded one stage per run."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from collections.abc import Callable
import torch
from .api import require_lock, DenseFP8, domain_histogram
from .screen import SHAPES, measured, telemetry
from ...native import formats as F, reference as R
from ...native.official import OfficialShards
from ...native.ops import NativeDense


class Monitor:
    def __init__(self,path:Path):
        self.path=path
        self.stop=threading.Event()
        self.error:BaseException|None=None
        self.thread=threading.Thread(target=self.run,daemon=True)
    def run(self)->None:
        try:
            with self.path.open('w') as f:
                while not self.stop.is_set():
                    f.write(json.dumps({'utc_ns':time.time_ns(),'gpu':telemetry()})+'\n');f.flush()
                    self.stop.wait(.1)
        except BaseException as err: self.error=err
    def __enter__(self)->Monitor:
        self.thread.start();return self
    def __exit__(self,*args:object)->None:
        self.stop.set();self.thread.join(10)
        if self.thread.is_alive():raise RuntimeError('telemetry monitor failed to exit')
        if self.error is not None:raise RuntimeError('telemetry monitor failed') from self.error


def shard(sh:OfficialShards,name:str,n:int,k:int,cut:str,rank:int)->tuple[torch.Tensor,torch.Tensor,int,int]:
    sh.verify(name+'.weight');sh.verify(name+'.scale')
    w,s=sh.get(name+'.weight'),sh.get(name+'.scale')
    r0=rank*n if cut=='row' else 0
    c0=rank*k if cut=='col' else 0
    return w[r0:r0+n,c0:c0+k].contiguous(),s,r0,c0


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument('--stage',type=int,choices=(0,1,2),required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if os.environ.get('KX_FP8DENSE_LOCKED')!='1' or os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc':
        raise RuntimeError('use locked runner')
    require_lock()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.manual_seed(5879+args.stage)
    dev=torch.device('cuda:0')
    sh=OfficialShards('/opt/dsv41/x86host-data/native-1017')
    result:dict[str,object]={'stage':args.stage,'torch':torch.__version__,'cuda':torch.version.cuda,
        'boundary':'isolated dense kernels; original official weights; seeded synthetic BF16 QDQ inputs; high magnitude main inputs',
        'source_sha256':hashlib.sha256(Path(__file__).with_name('dense.cu').read_bytes()).hexdigest(),
        'timing':'CUDA graphs, warm, 0.2 s clock settling, 20 calls/sample, 5 samples; excludes QDQ/prepack/model/collectives',
        'records':[],'complete':False}
    rows:list[dict[str,object]]=[]
    result['records']=rows
    def save()->None:args.output.write_text(json.dumps(result,indent=2)+'\n')
    with Monitor(args.output.with_suffix('.telemetry.jsonl')):
        for shape,(suffix,(n,k),cut,f32) in SHAPES.items():
            if shape=='main_proj' and args.stage!=0:continue
            name=f'mtp.{args.stage}.'+suffix
            for rank in (0,1):
                w,s,r0,c0=shard(sh,name,n,k,cut,rank)
                groups=4 if shape=='wo_a' else 1
                dense=DenseFP8.from_checkpoint(w,s,dev,groups=groups,row_offset=r0,col_offset=c0)
                sc=s[r0//32:(r0+n)//32,c0//32:(c0+k)//32].contiguous()
                histogram=domain_histogram(w.view(groups,n//groups,k),sc.view(groups,n//groups//32,k//32))
                wd=F.decode_fp8_block(w,s,rows=(r0,r0+n),cols=(c0,c0+k))
                # Check independently decoded CPU weights are exactly BF16.
                assert torch.equal(wd,wd.bfloat16().float())
                scales=F.e8m0_to_f32(sc).to(dev)
                oracle=NativeDense(name,'fp8',w.to(dev),scales)
                expanded=NativeDense(name,'bf16',wd.bfloat16().to(dev),None,backend='triton')
                ms=(8,32,128) if shape=='main_proj' else (5,10,15,20)
                for m in ms:
                    xcpu=F.act_qdq(torch.randn(m,k*groups,dtype=torch.bfloat16)*.1)
                    # Include the large magnitude tap domain explicitly. The
                    # source-faithful BF16 kernel must not narrow it to FP16.
                    if shape=='main_proj':xcpu=F.act_qdq((xcpu.float()*2.0**20).bfloat16())
                    x=xcpu.to(dev)
                    out=torch.empty((m,n),device=dev,dtype=torch.float32 if f32 else torch.bfloat16)
                    ref=torch.cat([(xcpu[:,g*k:(g+1)*k].double()@wd[g*(n//groups):(g+1)*(n//groups)].double().t()).float() for g in range(groups)],1)
                    explicit_ref=torch.cat([R.linear_fp8(xcpu[:,g*k:(g+1)*k],w[g*(n//groups):(g+1)*(n//groups)],s,
                        rows=(r0+g*(n//groups),r0+(g+1)*(n//groups)),cols=(c0,c0+k)) for g in range(groups)],1)
                    assert torch.equal(explicit_ref,ref.bfloat16()),f'FP64/reference.linear_fp8 mismatch {name}'
                    base=torch.cat([oracle(x[:,g*k:(g+1)*k],f32).cpu() for g in range(groups)],1) if groups==1 else torch.cat([
                        NativeDense(name,'fp8',w[g*1024:(g+1)*1024].to(dev),scales[g*32:(g+1)*32])(x[:,g*k:(g+1)*k],f32).cpu() for g in range(groups)],1)
                    bfbase=expanded(x,f32).cpu() if groups==1 else None
                    splits,rt=dense.configuration(m)
                    scratch=torch.empty((splits,m,groups,n//groups),device=dev) if splits>1 else None
                    call:Callable[[],None]=lambda:dense.linear_into(x,out,scratch=scratch,splits=splits,row_tiles=rt)
                    call();got=out.cpu()
                    agree=float((got.bfloat16()==base.bfloat16()).double().mean())
                    if not f32:assert agree>=.999, f'{name} rank {rank} M{m}: N1 agreement {agree}'
                    error=got.float()-ref
                    rel=float(error.double().norm()/ref.double().norm())
                    if f32:assert rel<=1e-6,f'{name} rank{rank}: FP32 relative L2 {rel}'
                    # A second FP32 output separates accumulation error from the
                    # intentional BF16 rounding on BF16-output operations.
                    raw=torch.empty_like(out,dtype=torch.float32)
                    dense.linear_into(x,raw,scratch=scratch,splits=splits,row_tiles=rt)
                    rawcpu=raw.cpu()
                    rawerr=rawcpu-ref
                    assert float(rawerr.double().norm()/ref.double().norm())<=1e-6,f'{name}: raw FP32 accumulation error'
                    bad=(got.bfloat16()!=base.bfloat16()).nonzero()
                    ties=[]
                    for ij in bad.tolist():
                        i,j=ij
                        lo=min(float(got[i,j].bfloat16()),float(base[i,j].bfloat16()));hi=max(float(got[i,j].bfloat16()),float(base[i,j].bfloat16()))
                        ties.append({'index':[i,j],'candidate':float(got[i,j]),'n1':float(base[i,j]),
                            'f64_sum_f32':float(ref[i,j]),'candidate_sum_f32':float(rawcpu[i,j]),'bf16_midpoint':(lo+hi)/2})
                    row:dict[str,object]={'shape':shape,'name':name,'rank':rank,'m':m,'n':n,'k':k,'groups':groups,
                        'row_offset':r0,'col_offset':c0,'splits':splits,'row_tiles':rt,'bf16_vs_n1':agree,
                        'bf16_vs_f64':float((got.bfloat16()==ref.bfloat16()).double().mean()),
                        'bf16_vs_expanded':None if bfbase is None else float((got.bfloat16()==bfbase.bfloat16()).double().mean()),
                        'fp32_relative_l2':float(rawerr.double().norm()/ref.double().norm()),
                        'fp32_max_abs':float(rawerr.abs().max()),
                        'fp32_max_relative_element':float((rawerr.abs()/ref.abs().clamp_min(1e-30)).max()),
                        'fp32_relative_element_failures_1e6':int((rawerr.abs()>ref.abs()*1e-6).sum()),
                        'bf16_order_disagreements':ties,'input_max_abs':float(xcpu.float().abs().max()),
                        'weight_values':w.numel(),'present_domain_pairs':int((histogram!=0).sum()),
                        'weights_sha256':hashlib.sha256(w.numpy().tobytes()).hexdigest(),
                        'resident_weight_bytes':dense.codes.numel()+dense.scales.numel()}
                    if m in (5,20) or shape=='main_proj':
                        row['timing']=measured(call,warm_seconds=.2)
                    # Every shape/dtype/row count: replay reads current input.
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):call()
                    x.mul_(.5);graph.replay();replayed=out.clone();call()
                    assert torch.equal(replayed,out),f'graph/eager mismatch {name}'
                    row['graph_replay_current_input_exact']=True
                    rows.append(row);save()
                    print(json.dumps({k:row[k] for k in ('name','rank','m','bf16_vs_n1','fp32_relative_l2','splits','row_tiles')}),flush=True)
                del dense,oracle,expanded,w,wd,scales,out,x,raw,scratch
                torch.cuda.empty_cache()
    result['complete']=True
    result['peak_allocated']=torch.cuda.max_memory_allocated()
    result['peak_reserved']=torch.cuda.max_memory_reserved()
    save()


if __name__=='__main__':main()
