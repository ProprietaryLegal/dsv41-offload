"""Measure exact same-shard FP8 vs native BF16; no dispatch is changed here."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ....kernels.fp8dense import DenseFP8
from ....kernels.fp8dense.qualify import shard
from ....kernels.fp8dense.screen import SHAPES, measured, telemetry
from ... import formats as F
from ...loader import LoadStats, load_dense
from ...official import OfficialShards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, choices=(0, 1, 2), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dev = torch.device('cuda:0')
    sh = OfficialShards('/opt/dsv41/x86host-data/native-1017')
    records: list[dict[str, object]] = []
    for shape, (suffix, (n, k), cut, f32) in SHAPES.items():
        if shape == 'main_proj' and args.stage != 0:
            continue
        name = f'mtp.{args.stage}.' + suffix
        for rank in (0, 1):
            w, scales, r0, c0 = shard(sh, name, n, k, cut, rank)
            groups = 4 if shape == 'wo_a' else 1
            candidate = DenseFP8.from_checkpoint(w, scales, dev, groups=groups, row_offset=r0, col_offset=c0)
            base = load_dense(sh, name, dev, 'bf16', LoadStats(), (r0, r0+n), (c0, c0+k))
            for rows in ((8, 32, 128) if shape == 'main_proj' else (5, 10, 15, 20)):
                torch.manual_seed(5879 + args.stage + rows)
                x = F.act_qdq(torch.randn(rows, groups*k, dtype=torch.bfloat16)*.1).to(dev)
                if shape == 'main_proj':
                    x.mul_(2**20)
                out = torch.empty(rows, n, dtype=torch.float32 if f32 else torch.bfloat16, device=dev)
                baseline = torch.empty_like(out)
                splits, tiles = candidate.configuration(rows)
                scratch = torch.empty(splits, rows, groups, n//groups, device=dev) if splits > 1 else None

                def new() -> None:
                    candidate.linear_into(x, out, scratch=scratch, splits=splits, row_tiles=tiles)

                def old() -> None:
                    if groups == 1:
                        torch.mm(x, base.w.t(), out_dtype=torch.float32, out=baseline) if f32 else torch.mm(x, base.w.t(), out=baseline)
                    else:
                        torch.bmm(x.view(rows, groups, k).transpose(0, 1),
                                  base.w.view(groups, n//groups, k).transpose(1, 2), out=baseline.view(rows, groups, n//groups).transpose(0, 1))

                old(); new()
                agreement = float((out == baseline).float().mean())
                relative = float((out.float()-baseline.float()).norm()/baseline.float().norm())
                before = telemetry()
                old_us = measured(old, warm_seconds=.1)
                new_us = measured(new, warm_seconds=.1)
                record = {'stage': args.stage, 'rank': rank, 'shape': shape, 'rows': rows,
                          'bf16_agreement': agreement, 'relative_l2_vs_cublas': relative,
                          'bf16': old_us, 'fp8': new_us, 'before': before, 'after': telemetry()}
                records.append(record)
                args.output.write_text(json.dumps({'records': records, 'complete': False}, indent=1)+'\n')
                print(json.dumps(record), flush=True)
            del candidate, base
            torch.cuda.empty_cache()
    args.output.write_text(json.dumps({'records': records, 'complete': True}, indent=1)+'\n')


if __name__ == '__main__':
    main()
