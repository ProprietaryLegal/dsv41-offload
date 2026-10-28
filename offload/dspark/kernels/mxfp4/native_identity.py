"""Exact one-hot reconstruction of complete native shard matrices (no downloads)."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import torch
from .qualify import tensor, telemetry
from .reference import unpack
from .regfed import RegisterExperts


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != 'GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc'
            or os.environ.get('KX_MXFP4_LOCKED') != '1'):
        raise RuntimeError('requires run_locked.sh')
    dev = torch.device('cuda:0')
    checks: list[dict[str,int]] = []
    start = time.monotonic()
    initial = telemetry()
    for stage in range(3):
        for expert in (0,127):
            for proj in (1,2,3):
                pp,ss = tensor(stage,expert,proj)
                for rank in (0,1):
                    if proj==2:
                        p,s = pp[:,rank*576:(rank+1)*576],ss[:,rank*36:(rank+1)*36]
                    else:
                        p,s = pp[rank*1152:(rank+1)*1152],ss[rank*1152:(rank+1)*1152]
                    p,s = p[None].contiguous(),s[None].contiguous()
                    bank = RegisterExperts.from_checkpoint(p,s,dev)
                    # Bound routing work and identity storage to 64 active rows.
                    reference = unpack(p,s)[0].float().T
                    decoded_values = 0
                    for lo in range(0,bank.inputs,64):
                        count = min(64,bank.inputs-lo)
                        x = torch.zeros(count,bank.inputs,device=dev,dtype=torch.bfloat16)
                        x[torch.arange(count,device=dev),lo+torch.arange(count,device=dev)] = 1
                        ids = torch.zeros(count,1,device=dev,dtype=torch.int32)
                        schedule = torch.empty(count,8,device=dev,dtype=torch.int32)
                        out = torch.empty(count,1,bank.outputs,device=dev)
                        status = torch.zeros(1,device=dev,dtype=torch.int32)
                        bank.linear_into(x,ids,out,status,schedule)
                        actual = out[:,0].cpu()
                        expected = reference[lo:lo+count]
                        if not torch.equal(actual,expected) or status.item()!=0:
                            raise AssertionError(f'weight reconstruction failed: stage={stage},expert={expert},proj={proj},rank={rank},lo={lo},maxerr={(actual-expected).abs().max()}')
                        decoded_values += actual.numel()
                    check = {'stage':stage,'expert':expert,'projection':proj,'rank':rank,
                             'exact_values':decoded_values,'mismatches':0}
                    checks.append(check)
                    print(json.dumps(check),flush=True)
    result = {'boundary':'actual register prepack + MMA one-hot reconstruction; all columns and outputs of each selected native shard',
              'elapsed_s':time.monotonic()-start,'telemetry_start':initial,'telemetry_end':telemetry(),
              'checks':checks,'exact_values':sum(c['exact_values'] for c in checks),'mismatches':0}
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
