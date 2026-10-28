"""Test whether paired BF16 shared projections remove the remaining T20 cost.

M20 retains BF16 weights. No model/dispatch change is made by this screen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ....kernels.fp8dense.epilogue import shared_activation_into
from ....kernels.fp8dense.screen import measured
from ... import formats as F, ops as NO
from ...loader import LoadStats, load_dense
from ...official import OfficialShards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device('cuda:0')
    sh = OfficialShards('/opt/dsv41/x86host-data/native-1017')
    records = []
    for stage in range(3):
        for rank in (0, 1):
            prefix = f'mtp.{stage}.ffn.shared_experts.'
            weights = [load_dense(sh, prefix+name, dev, 'bf16', LoadStats(), (rank*1152, (rank+1)*1152))
                       for name in ('w1', 'w3')]
            pair = torch.cat([w.w for w in weights])
            torch.manual_seed(20261023+stage+rank)
            x = F.act_qdq(torch.randn(20, 5120, dtype=torch.bfloat16)*.1).to(dev)
            g, u = [torch.empty(20, 1152, dtype=torch.bfloat16, device=dev) for _ in range(2)]
            joined = torch.empty(20, 2304, dtype=torch.bfloat16, device=dev)
            ones = torch.ones(20, 1, device=dev)
            old_out, new_out = torch.empty_like(g), torch.empty_like(g)

            def old() -> None:
                torch.mm(x, weights[0].w.t(), out=g)
                torch.mm(x, weights[1].w.t(), out=u)
                shared_activation_into(g, u, ones, old_out)

            def new() -> None:
                torch.mm(x, pair.t(), out=joined)
                shared_activation_into(joined[:, :1152], joined[:, 1152:], ones, new_out)

            old(); new()
            expected = NO.act_qdq(NO.swiglu_bf16(joined[:, :1152], joined[:, 1152:]))
            assert torch.equal(new_out.view(torch.int16), expected.view(torch.int16))
            raw = torch.mm(x, pair.t(), out_dtype=torch.float32).cpu()
            ref = (x.cpu().double() @ pair.cpu().double().t()).float()
            error = float((raw-ref).double().norm()/ref.double().norm())
            agreement = float((joined == torch.cat((g,u), dim=1)).float().mean())
            assert agreement >= .999 and error <= 1e-6
            record = {'stage': stage, 'rank': rank, 'rows': 20, 'bf16_projection_agreement': agreement,
                      'raw_fp32_relative_l2': error, 'hq_self_exact': True,
                      'hq_vs_old_exact_fraction': float((old_out == new_out).float().mean()),
                      'separate': measured(old, warm_seconds=.2), 'paired': measured(new, warm_seconds=.2)}
            records.append(record)
            args.output.write_text(json.dumps({'records': records, 'complete': False}, indent=1)+'\n')
            print(json.dumps(record), flush=True)
            del weights, pair
            torch.cuda.empty_cache()
    args.output.write_text(json.dumps({'records': records, 'complete': True}, indent=1)+'\n')


if __name__ == '__main__':
    main()
