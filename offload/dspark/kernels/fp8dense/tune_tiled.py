"""Compare tiling configurations offline; no production autotuning."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import torch
from .api import require_lock
from .screen import SHAPES, measured
from .tiled import linear_into
from ...native.formats import act_qdq, decode_fp8_block
from ...native.official import OfficialShards


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shapes',nargs='+',default=['wq_b','wo_b','main_proj'])
    p.add_argument('--compact',action='store_true')
    args=p.parse_args()
    if os.environ.get('KX_FP8DENSE_LOCKED')!='1': raise RuntimeError('use run_locked.sh')
    require_lock()
    torch.manual_seed(12354)
    torch.set_num_threads(8)
    sh=OfficialShards('/opt/dsv41/x86host-data/native-1017')
    results:list[dict[str,object]]=[]
    for shape in args.shapes:
        suffix,(n,k),cut,f32=SHAPES[shape]
        name='mtp.0.'+suffix
        wc,s=sh.get(name+'.weight')[:n,:k].contiguous(),sh.get(name+'.scale')[:n//32,:k//32].contiguous()
        w,sc=wc.cuda(),s.cuda()
        fast_scales=bool(int(s.max())<=134)
        wd=decode_fp8_block(wc,s).cuda()
        for m in ((8,32,128) if shape=='main_proj' else (5,20)):
            x=act_qdq(torch.randn(m,k,dtype=torch.bfloat16)*.1).cuda()
            ref=(x.double()@wd.double().t()).float()
            out=torch.empty((m,n),device='cuda',dtype=torch.float32 if f32 else torch.bfloat16)
            row_results=[]
            for bm in ((16,32) if m<=20 else (32,64,128)):
                for bn,bk,wp in (((64,64,4),(128,64,4),(128,64,8),(128,128,8)) if args.compact else ((32,64,4),(32,128,4),(64,64,4),(64,128,4),(64,128,8),(128,64,4))):
                    for splits in ((1,2,4,8) if args.compact else (2,4,8,16)):
                        if splits>(k+bk-1)//bk: continue
                        scratch=torch.empty((splits,m,1,n),device='cuda')
                        stages=min(3,98304//((bm+bn)*bk*2))
                        if stages<1: raise ValueError('tile exceeds SM86 shared memory')
                        call=lambda:linear_into(x,w,sc,out,scratch,bm=bm,bn=bn,bk=bk,splits=splits,warps=wp,stages=stages,fast_scales=fast_scales)
                        call();torch.cuda.synchronize()
                        error=out.float()-ref
                        record={'shape':shape,'m':m,'bm':bm,'bn':bn,'bk':bk,'splits':splits,'warps':wp,'stages':stages,
                            'relative_l2':float(error.norm()/ref.norm()),'bf16_vs_f64':float((out.bfloat16()==ref.bfloat16()).float().mean()),
                            **measured(call)}
                        results.append(record);row_results.append(record)
                        args.output.write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps(min(row_results,key=lambda r:r['median_us'])),flush=True)
        del w,sc,wd,out,x,ref,scratch
        torch.cuda.empty_cache()


if __name__=='__main__':main()
