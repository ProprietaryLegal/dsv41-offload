"""CPU-only full dense tensor domain/provenance census; no CUDA context."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from .api import domain_histogram
from .screen import SHAPES
from ...native.official import OfficialShards


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU census requires empty CUDA visibility')
    torch.set_num_threads(8)
    sh=OfficialShards('/opt/dsv41/x86host-data/native-1017')
    hist=np.zeros(65536,dtype=np.int64)
    records=[]
    start=time.monotonic()
    for stage in range(3):
        for shape,(suffix,_,_,_) in SHAPES.items():
            if shape=='main_proj' and stage!=0:continue
            name=f'mtp.{stage}.{suffix}'
            sh.verify(name+'.weight');sh.verify(name+'.scale')
            w,s=sh.get(name+'.weight'),sh.get(name+'.scale')
            h=domain_histogram(w.unsqueeze(0),s.unsqueeze(0))
            hist+=h
            records.append({'name':name,'shape':list(w.shape),'scale_shape':list(s.shape),
                'weight_sha256':hashlib.sha256(w.numpy().tobytes()).hexdigest(),
                'scale_sha256':hashlib.sha256(s.numpy().tobytes()).hexdigest(),
                'values':w.numel(),'present_pairs':int((h!=0).sum()),
                'small_e4m3_codes':int(h.reshape(256,256)[:,np.r_[0:8,128:136]].sum())})
            print(name,flush=True)
    used=np.flatnonzero(hist)
    c=torch.from_numpy((used&255).astype(np.uint8))
    s=torch.from_numpy((used>>8).astype(np.uint8))
    values=torch.ldexp(c.view(torch.float8_e4m3fn).float(),s.int()-127)
    assert torch.equal(values,values.bfloat16().float())
    assert bool(torch.isfinite(values).all())
    result={'complete':True,'tensors':records,'elapsed_s':time.monotonic()-start,
        'total_values':int(hist.sum()),'present_pairs':int(used.size),
        'scale_min':int(s.min()),'scale_max':int(s.max()),
        'present_scale_code_counts':[[int(i>>8),int(i&255),int(hist[i])] for i in used],
        'source_revision':'1bae0e1088993b9b82de08f59de5be5a34e42170','all_values_exact_finite_bf16':True}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETE',len(records),int(hist.sum()),time.monotonic()-start,flush=True)


if __name__=='__main__':main()
