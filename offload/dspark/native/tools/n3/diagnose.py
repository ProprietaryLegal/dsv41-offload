"""Bounded CPU counterfactuals for failed V1 gates; changes no worker/reference file.

Compare the worker's single-pass BF16-probability attention arithmetic with the
reference's online 64-key arithmetic on identical inputs. Independently check
whether declared BF16 Markov/sum rounding reproduces draft self-check flags.
This is a diagnosis, never an alternate passing gate or V1 verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, cast

import torch


def single_pass(q: torch.Tensor, kv: torch.Tensor, sink: torch.Tensor,
                indices: torch.Tensor, scale: float) -> torch.Tensor:
    b = q.shape[0]
    idx = indices.long()
    valid = idx >= 0
    gather = kv[torch.arange(b).view(b,1,1),idx.clamp_min(0)].float()
    gather = torch.where(valid.unsqueeze(-1),gather,torch.zeros_like(gather))
    scores = torch.einsum('bmhd,bmtd->bmht',q.float(),gather)*scale
    scores = scores.masked_fill(~valid.unsqueeze(2),float('-inf'))
    sk = sink.float().view(1,1,-1,1)
    mx = torch.maximum(scores.amax(-1,keepdim=True),sk)
    p = torch.exp(scores-mx)
    den = p.sum(-1,keepdim=True)+torch.exp(sk-mx)
    return (torch.einsum('bmht,bmtd->bmhd',p.bfloat16().float(),gather)/den).to(q.dtype)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--verify-root',type=Path,required=True)
    ap.add_argument('--arm-dir',type=Path,required=True)
    ap.add_argument('--captures',required=True)
    ap.add_argument('--ckpt',required=True)
    ap.add_argument('--official',required=True)
    ap.add_argument('--keys',required=True)
    ap.add_argument('--out',type=Path,required=True)
    a = ap.parse_args()
    sys.path.insert(0,str(a.verify_root))
    from verify.analyze import load_dumps,markov_tables,final_logits,relerr
    from verify.captures import CaptureSet
    from verify.native_ref import NativeRef
    from verify.run_ref import WindowCaches,prepare_window
    torch.set_num_threads(12)
    arm = load_dumps(str(a.arm_dir))
    emb,head = markov_tables(a.ckpt)
    head_rows = []
    for key,A in sorted(arm.items()):
        prefix=[A['anchor']]+A['drafts'][:4]
        bias=emb[torch.tensor(prefix)]@head.t()
        official=final_logits(A['base_logits'],prefix,emb,head)
        engine=(A['base_logits'].bfloat16()+bias.bfloat16()).bfloat16()
        head_rows.append({'key':key,'official_fp32_disagrees':sum(x!=y for x,y in zip(official.argmax(-1).tolist(),A['drafts'])),
                          'engine_bf16_disagrees':sum(x!=y for x,y in zip(engine.argmax(-1).tolist(),A['drafts']))})
    cs=CaptureSet(a.captures)
    keys=set(a.keys.split(','))
    selected=[s for s in cs.samples if s.key in keys]
    if {s.key for s in selected} != keys:
        raise ValueError('missing requested keys')
    ref=NativeRef(a.ckpt,a.official,mode='mirror',max_seq_len=max(s.P for s in selected)+80)
    caches=cast(Callable[[], Any], WindowCaches)()
    module=cast(Any,ref.M)
    original=module.sparse_attn
    rows=[]
    for sample in selected:
        _,_,mx,_=prepare_window(ref,cs,sample,caches)
        r=ref.draft(sample.P,sample.anchor,mx,keep_full_logits=True)
        raw_errors=[]
        def alternate(q: torch.Tensor,kv: torch.Tensor,sink: torch.Tensor,idx: torch.Tensor,scale: float) -> torch.Tensor:
            online=original(q,kv,sink,idx,scale)
            single=single_pass(q,kv,sink,idx,scale)
            raw_errors.append(relerr(single,online))
            return single
        module.sparse_attn=alternate
        try:
            _,_,mx,_=prepare_window(ref,cs,sample,caches)
            c=ref.draft(sample.P,sample.anchor,mx,keep_full_logits=True)
        finally:
            module.sparse_attn=original
        A=arm[sample.key]
        rows.append({'key':sample.key,'raw_attention_relerr_single_vs_online':raw_errors,
                     's0_arm_vs_online':relerr(A['s0']['attn_out'],r['s0']['attn_out']),
                     's0_cpu_single_vs_online':relerr(c['s0']['attn_out'],r['s0']['attn_out']),
                     's0_arm_vs_cpu_single':relerr(A['s0']['attn_out'],c['s0']['attn_out'])})
    result: dict[str,Any]={'diagnosis_only':True,'head':head_rows,'attention_counterfactuals':rows,
        'worker_changed':False,'v1_classifier_changed':False,'gates_remain_required':True}
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
