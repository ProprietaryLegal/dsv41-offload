"""W1b exactness/speed screens on an isolated current worker, no listeners.

Owned process and GPU window only. All replacements are local to this process.
Materialised weights stay fp16. Timings include transforms, not GEMM alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
import time
from typing import TYPE_CHECKING

from .profile_support import PROCESS_LIMIT_MIB, require_gpu_window
from .experiment_support import paired_graphs
from .qualify_donor import differences

if TYPE_CHECKING:
    import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("material", "wo", "reply", "mgemm", "misc", "experts"))
    p.add_argument("root", type=Path)
    p.add_argument("baseline", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--subset", choices=("small", "shared", "dense", "head"), default="small")
    p.add_argument("--donor-root", type=Path)
    p.add_argument("--build", type=Path)
    args = p.parse_args()
    require_gpu_window()
    import numpy as np
    import torch
    from .. import fused as FU
    from ..checkpoint import Checkpoint
    from ..exl3 import kernels as K
    from ..model import DSparkDrafter, Exl3Linear
    from ..server import GpuBackend
    from ..window import WindowRange
    torch.set_num_threads(2)
    dev = torch.device("cuda", 0)
    ck = Checkpoint(str(args.root))
    m = DSparkDrafter(ck, dev, slots=8)
    saved = torch.load(args.baseline/"synthetic.pt", weights_only=True)
    taps = saved["taps"].to(dev)
    for slot, last in enumerate((300,500,700,900)):
        pos = torch.arange(last-127,last+1,device=dev)
        m.insert(torch.full_like(pos,slot),pos,m.main_x(taps[pos]))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    output = args.out.open("x")

    def record(row: dict[str, object]) -> None:
        # Actual process/card footprint remains bounded, independent of allocator stats.
        import os
        import subprocess
        process = subprocess.run(["nvidia-smi", "--id="+os.environ["CUDA_VISIBLE_DEVICES"],
                                  "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                                 capture_output=True,text=True,check=True).stdout
        own = next(float(line.split(',')[1]) for line in process.splitlines() if int(line.split(',')[0])==os.getpid())
        if own >= PROCESS_LIMIT_MIB:
            raise RuntimeError(f"W1b footprint {own} MiB exceeds ceiling")
        row["process_mib"] = own
        row["peak_allocated_mib"] = torch.cuda.max_memory_allocated()/2**20
        output.write(json.dumps(row)+"\n")
        output.flush()
        print(json.dumps({k:v for k,v in row.items() if k!='timing'}),flush=True)

    class Material:
        def __init__(self, lin: Exl3Linear):
            self.lin = lin
            self.in_features,self.out_features=lin.in_features,lin.out_features
            self.w=torch.empty(self.in_features,self.out_features,dtype=torch.float16,device=dev)
            K.ext().reconstruct(self.w,lin.trellis,lin.k,False,True)
        def __call__(self,x: torch.Tensor,out: torch.Tensor|None=None) -> torch.Tensor:
            x=x.to(torch.float16).contiguous()
            xh=torch.empty_like(x)
            K.ext().had_r_128(x,xh,self.lin.suh,None,1.)
            y=torch.empty(x.shape[0],self.out_features,dtype=torch.float32,device=dev)
            K.ext().hgemm(xh,self.w,y)
            K.ext().had_r_128(y,y,None,self.lin.svh,1.)
            if out is not None:
                out.copy_(y)
                return out
            return y

    gen=torch.Generator(device=dev).manual_seed(41)
    if args.phase == "material":
        attrs={"small":("wq_a","wkv"), "shared":("wq_a","wkv","sh_w1","sh_w2","sh_w3"),
               "dense":("wq_a","wkv","wq_b","wo_b","sh_w1","sh_w2","sh_w3")}
        changes=[]
        if args.subset == "head":
            changes=[(m,"head",m.head,Material(m.head))]
        else:
            changes=[(st,a,getattr(st,a),Material(getattr(st,a))) for st in m.stages for a in attrs[args.subset]]
        extra=sum(new.w.numel()*2 for _,_,_,new in changes)
        record({"phase":"material_budget","subset":args.subset,"decoded_bytes":extra,
                "retains_original_packed":True,"margin_mib":512})
        for _,a,lin,new in changes:
            for rows in (1,2,3,4,5,10,15,20):
                require_gpu_window()
                x=(torch.randn(rows,lin.in_features,generator=gen,device=dev)*.3).half()
                got,ref=new(x),lin(x)
                record({"phase":"material_op","attribute":a,"rows":rows,
                        "vs_current":differences(got,ref),"timing":paired_graphs(lambda:lin(x),lambda:new(x))})
        def baseline():
            for owner,a,lin,new in changes: setattr(owner,a,lin)
            return m.draft(slots,anchor,last,5,n_out=n)
        def candidate():
            for owner,a,lin,new in changes: setattr(owner,a,new)
            return m.draft(slots,anchor,last,5,n_out=n)
        for b,n in ((1,5),(2,5),(3,4),(4,4)):
            require_gpu_window()
            slots=torch.arange(b,device=dev)
            last=torch.tensor((300,500,700,900)[:b],device=dev)
            anchor=torch.tensor([0]*b,device=dev)
            ri,rc=(v.clone() for v in baseline())
            ci,cc=candidate()
            record({"phase":"material_full","B":b,"n_out":n,"ids":differences(ci,ri),
                    "confidence":differences(cc,rc),"timing":paired_graphs(baseline,candidate)})
    elif args.phase == "wo":
        st=m.stages[0]
        for b in (1,2,3,4):
            rows=b*5
            x=torch.randn(rows,m.s.o_groups*4096,generator=gen,device=dev).bfloat16()
            ref=FU.wo_a_fp8(x,st.wo_a8,st.wo_a_sc,m.s.o_groups)
            for bn,bk,warps,stages in ((32,128,2,2),(32,128,4,2),(32,128,4,3),(16,128,2,3),(32,64,2,3),(32,256,2,2)):
                require_gpu_window()
                def base(): return FU.wo_a_fp8(x,st.wo_a8,st.wo_a_sc,m.s.o_groups)
                def candidate(): return FU.wo_a_fp8(x,st.wo_a8,st.wo_a_sc,m.s.o_groups,BN=bn,BK=bk,num_warps=warps,num_stages=stages)
                got=candidate()
                record({"phase":"wo","B":b,"rows":rows,"config":[bn,bk,warps,stages],
                        "vs_current":differences(got,ref),"timing":paired_graphs(base,candidate)})
    elif args.phase == "reply":
        require_gpu_window()
        m.capture(batch_sizes=(1,2,3,4),ks=(5,),insert_rows=(8,16,32,64,128),n_outs=(1,2,3,4,5),pack_reply=True)
        ordinary=GpuBackend(m,512,30720,graph_reply=False)
        captured=GpuBackend(m,512,30720,graph_reply=True)
        checked = 0
        for b in (1,2,3,4):
            for n in (1,2,3,4,5):
                for margin in (False,True):
                    for cache in ("hit","miss"):
                        require_gpu_window()
                        cand=torch.where((m.markov.slot_of_token>=0) if cache=="hit" else (m.markov.slot_of_token<0))[0][:b].cpu().tolist()
                        reqs=[(i,cand[i],lp+1,WindowRange(128,lp-127,lp),n) for i,lp in enumerate((300,500,700,900)[:b])]
                        ref=ordinary.draft(reqs,5,margin)
                        got=captured.draft(reqs,5,margin)
                        if not all(a is None and c is None or a is not None and c is not None and np.array_equal(a.view(np.uint8),c.view(np.uint8)) for a,c in zip(ref,got)):
                            raise RuntimeError(f"reply mismatch B{b} n{n} margin{margin} {cache}")
                        checked += 1
            # Three interleaved trials on canonical plain replies, same metadata/D2H.
            n=5 if b<3 else 4
            reqs=[(i,0,lp+1,WindowRange(128,lp-127,lp),n) for i,lp in enumerate((300,500,700,900)[:b])]
            for cadence in (0,.042):
                for trial in range(3):
                    pairs=[]
                    for i in range(45):
                        require_gpu_window()
                        values={}
                        for tag,backend in (("base",ordinary),("candidate",captured)) if i%2==0 else (("candidate",captured),("base",ordinary)):
                            if cadence: time.sleep(cadence)
                            t0=time.perf_counter_ns()
                            backend.draft(reqs,5,False)
                            values[tag]=(time.perf_counter_ns()-t0)/1000
                        if i>=5: pairs.append({**values,"saving_us":values['base']-values['candidate']})
                    record({"phase":"reply_timing","B":b,"trial":trial,"cadence_ms":cadence*1000,"preceding_reply_cases_bit_exact":checked,
                            "timing":{"pairs":pairs,"median":{k:median(r[k] for r in pairs) for k in pairs[0]}}})
    elif args.phase == "misc":
        import triton
        st=m.stages[0]
        for b in (1,2,3,4):
            require_gpu_window()
            rows=b*5
            slots=torch.arange(b,device=dev)
            last=torch.tensor((300,500,700,900)[:b],device=dev)
            lo=last-127
            q=torch.randn(b,5,m.s.n_heads,512,generator=gen,device=dev).bfloat16()
            kb=torch.randn(b,5,512,generator=gen,device=dev).bfloat16()
            def base(): return FU.block_attention_ring(q,m.ring_kv[:,0],m.ring_pos,slots,lo,last,kb,st.attn_sink,m.scale)
            ref=base()
            for bm,do,bd,warps in ((16,256,64,4),(32,256,64,4),(16,128,64,4),(32,128,64,4),(16,256,32,8)):
                def candidate(): return FU.block_attention_ring(q,m.ring_kv[:,0],m.ring_pos,slots,lo,last,kb,st.attn_sink,m.scale,BM=bm,DO=do,BD=bd,num_warps=warps)
                record({"phase":"attention","B":b,"config":[bm,do,bd,warps],"vs_current":differences(candidate(),ref),
                        "timing":paired_graphs(base,candidate)})
            x=torch.randn(rows,5120,generator=gen,device=dev).bfloat16()
            def base(): return FU.rms_norm(x,st.attn_norm,m.eps)
            ref=base()
            for warps in (1,2,8):
                def candidate():
                    out=torch.empty_like(x)
                    FU._rms_norm_kernel[(rows,)](x,st.attn_norm,out,5120,m.eps,BD=1024,num_warps=warps,enable_fp_fusion=False)
                    return out
                record({"phase":"norm","B":b,"warps":warps,"vs_current":differences(candidate(),ref),"timing":paired_graphs(base,candidate)})
            stream=torch.randn(rows,4,5120,generator=gen,device=dev).bfloat16()
            fn,scale,bias=st.hc_attn
            original=FU._HC_SPLITS
            def base():
                FU._HC_SPLITS=original
                return FU.hc_mixes(stream,fn,scale,bias,m.eps,m.s.hc_eps,m.s.hc_iters)
            ref=tuple(v.clone() for v in base())
            for ns in (10,20,80):
                def candidate():
                    FU._HC_SPLITS=ns
                    return FU.hc_mixes(stream,fn,scale,bias,m.eps,m.s.hc_eps,m.s.hc_iters)
                got=candidate()
                record({"phase":"mhc","B":b,"splits":ns,"vs_current":[differences(a,c) for a,c in zip(got,ref)],
                        "timing":paired_graphs(base,candidate)})
            FU._HC_SPLITS=original
    elif args.phase == "experts":
        if args.donor_root is None or args.build is None:
            p.error("experts requires --donor-root and --build")
        from .donor_adapter import CudaExl3Dense
        donor=CudaExl3Dense(args.donor_root,args.build)
        st=m.stages[0]
        matrices={}
        for e in range(60):
            matrices[e]={}
            for name in ("w1","w3","w2"):
                tr,su,sv,_=ck.exl3(f"mtp.0.ffn.experts.{e}.{name}")
                matrices[e][name]=(tr.to(dev),su.to(dev)[None],sv.to(dev))
        for rows in (1,2,3,4,5,10,15,20):
            for pattern in ("shared","distinct"):
                require_gpu_window()
                x=(torch.randn(rows,5120,generator=gen,device=dev)*.3).half()
                ids=(torch.arange(3).expand(rows,3) if pattern=="shared" else torch.arange(rows*3).view(rows,3)).to(dev)
                weights=torch.full((rows,3),m.s.route_scale/3,device=dev,dtype=torch.float16)
                sel=st.perm[ids].contiguous()
                counts=torch.bincount(ids.flatten(),minlength=128).cpu().tolist()
                populated=[(e,c) for e,c in enumerate(counts) if c]
                def base(): return K.run_decode_coop(x,sel,weights,st.groups,m.coop,m.s.swiglu_limit,False,True)[:rows]
                def candidate():
                    # Static synthetic counts; GPU sort, gather, transforms, clamps,
                    # scatter and fixed top-k-order combine are all timed.
                    order=torch.argsort(ids.flatten(),stable=True)
                    tokens=order//3
                    contrib=torch.zeros(rows*3,5120,device=dev,dtype=torch.float32)
                    start=0
                    for e,count in populated:
                        xx=x[tokens[start:start+count]]
                        def lin(name,xx):
                            tr,su,sv=matrices[e][name]
                            return donor.linear(xx,tr,su,sv,[sv.numel()])
                        gate=lin("w1",xx).float().clamp(max=m.s.swiglu_limit)
                        up=lin("w3",xx).float().clamp(-m.s.swiglu_limit,m.s.swiglu_limit)
                        down=lin("w2",(torch.nn.functional.silu(gate)*up).half()).float()
                        pick=order[start:start+count]
                        contrib[pick]=down*weights.flatten()[pick,None]
                        start+=count
                    out=torch.zeros(rows,5120,device=dev,dtype=torch.float32)
                    for pick in range(3): out.add_(contrib.view(rows,3,5120)[:,pick])
                    return out
                ref=base().clone()
                got=candidate()
                record({"phase":"experts","rows":rows,"pattern":pattern,"counts":counts,
                        "vs_current":differences(got,ref),"static_synthetic_group_counts":True,
                        "timing":paired_graphs(base,candidate)})
    else:
        # Same-width shared gate/up bundle using the installed pinned mgemm, with distinct suh per matrix.
        st=m.stages[0]
        for rows in (1,2,3,4,5,10,15,20):
            require_gpu_window()
            x=(torch.randn(rows,5120,generator=gen,device=dev)*.3).half()
            linears=(st.sh_w1,st.sh_w3)
            ptr=lambda ts:torch.tensor([t.data_ptr() for t in ts],dtype=torch.int64,device=dev)
            tr,su,sv=(ptr([getattr(l,a) for l in linears]) for a in ('trellis','suh','svh'))
            y=torch.empty(2,rows,2304,dtype=torch.float32,device=dev)
            xh=torch.empty(2,rows,5120,dtype=torch.float16,device=dev)
            def base(): return tuple(l(x) for l in linears)
            def candidate():
                K.ext().exl3_mgemm(x[None],tr,y,su,xh,sv,None,None,5,0,False,True,-1,-1,0)
                return y
            ref=base()
            got=candidate()
            record({"phase":"mgemm_pair","rows":rows,"vs_current":[differences(got[i],ref[i]) for i in (0,1)],
                    "timing":paired_graphs(base,candidate)})
    record({"phase":"complete","kind":args.phase})
    output.close()

if __name__ == "__main__": main()
