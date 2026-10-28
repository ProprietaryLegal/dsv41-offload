"""Direct published N1 expert-projection gate inside complete emulated TP2.

Reuses KX's instrumented N1 reduction oracle; records every BF16 disagreement
with both FP32 sums and the decoded-value FP64 sum. No fixture is committed.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import MethodType
from typing import Any

import torch

from ....kernels.fusedmoe.oracle import n1_projection_oracle
from ....worker.checkpoint import Checkpoint
from ....worker.tools.parity_ref import make_taps
from ....worker.tools.tp2_bench import _EmbedOnly, fill_windows_tp, host_inp, setup_inputs
from ....worker.tp2 import check_visible
from ... import formats as F, ops as NO
from ...kx import KXFusedBank
from ...tp2_native import ARMS, build


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check_visible(pair=False)
    torch.set_num_threads(8)
    dev = torch.device('cuda:0')
    root = '/opt/dsv41/x86host-data/dspark'
    cfg = replace(ARMS['native-hybrid-kx2'], dense_profile='measured-small')
    model = build(cfg, [dev, dev], Checkpoint(root), slots=8)
    from ....kernels.fp8dense import epilogue as shared_module
    from ....kernels.fusedmoe import epilogue as combine_module
    shared_original, combine_original = shared_module.shared_activation_into, combine_module.combine_official_into
    shared_owners = {st.shared_hq.untyped_storage().data_ptr(): (rank, stage)
                     for rank in (0, 1) for stage, st in enumerate(model.ranks[rank].stages)}
    combine_owners = {out.untyped_storage().data_ptr(): (rank, stage)
                      for rank in (0, 1) for stage, st in enumerate(model.ranks[rank].stages)
                      for out in st.nbank.combined_outputs.values()}
    epilogues: list[dict[str, Any]] = []

    def shared_check(gate: torch.Tensor, up: torch.Tensor, ones: torch.Tensor, out: torch.Tensor) -> None:
        shared_original(gate, up, ones, out)
        expected = NO.act_qdq(NO.swiglu_bf16(gate, up))
        rank, stage = shared_owners[out.untyped_storage().data_ptr()]
        count = int((out.view(torch.int16) != expected.view(torch.int16)).sum())
        epilogues.append({'kind': 'shared', 'rank': rank, 'stage': stage, 'rows': out.shape[0],
                          'elements': out.numel(), 'bit_differences': count})
        assert count == 0, 'shared epilogue bits differ from old pipeline'

    def combine_check(parts: torch.Tensor, ids: torch.Tensor, out: torch.Tensor) -> None:
        combine_original(parts, ids, out)
        expected = NO.moe_combine_official(parts[:, :3], parts[:, 3], ids)
        rank, stage = combine_owners[out.untyped_storage().data_ptr()]
        count = int((out.view(torch.int16) != expected.view(torch.int16)).sum())
        epilogues.append({'kind': 'combine', 'rank': rank, 'stage': stage, 'rows': out.shape[0],
                          'elements': out.numel(), 'bit_differences': count})
        assert count == 0, 'post-TP epilogue bits differ from old pipeline'

    shared_module.shared_activation_into, combine_module.combine_official_into = shared_check, combine_check
    inputs: list[tuple[int, int, KXFusedBank, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for rank in (0, 1):
        for stage, st in enumerate(model.ranks[rank].stages):
            assert isinstance(st.nbank, KXFusedBank)

            def forward(self: KXFusedBank, x: torch.Tensor, ids: torch.Tensor, rw: torch.Tensor,
                        h: torch.Tensor | None = None, out: torch.Tensor | None = None,
                        rank: int = rank, stage: int = stage) -> torch.Tensor:
                result = KXFusedBank.forward_slots(self, x, ids, rw, h, out)
                inputs.append((stage, rank, self, x.clone(), ids.clone(), rw.clone(), result.clone()))
                return result

            st.nbank.forward_slots = MethodType(forward, st.nbank)
    tokens = setup_inputs(dev, root)
    taps = make_taps(_EmbedOnly(model.ranks[0].embed), tokens, torch.Generator(device=dev).manual_seed(5))
    model.insert_eager(torch.full((8,), 7), torch.arange(2000, 2008), taps[:8])
    fill_windows_tp(model, taps.cpu())
    for batch in (1, 2, 3, 4):
        model.draft(host_inp(batch, tokens.cpu()), 5)
    shared_module.shared_activation_into, combine_module.combine_official_into = shared_original, combine_original
    expected = {(stage, rank, rows) for stage in range(3) for rank in (0, 1) for rows in (5, 10, 15, 20)}
    if len(inputs) != 24 or {(v[0], v[1], v[3].shape[0]) for v in inputs} != expected:
        raise RuntimeError('incomplete full-model stage/rank/shape coverage')
    if len(epilogues) != 48:
        raise RuntimeError('incomplete full-model epilogue bit checks')
    records: list[dict[str, Any]] = []
    result = {'cfg': cfg.__dict__, 'execution_mode': 'emulated_tp2', 'records': records,
              'epilogue_bit_checks': epilogues, 'complete': False}
    previous = None
    w13 = scales = None
    for stage, rank, bank, x, ids, rw, old_slots in sorted(inputs, key=lambda v: (v[0], v[1], v[3].shape[0])):
        if previous != (stage, rank):
            # One raw gate/up bank at a time, never a second expanded expert bank.
            del w13, scales
            w13 = torch.empty(128, 2, 1152, 2560, dtype=torch.uint8, device=dev)
            scales = torch.empty(128, 2, 1152, 160, dtype=torch.uint8, device=dev)
            for expert in range(128):
                for index, projection in enumerate(('w1', 'w3')):
                    prefix = f'mtp.{stage}.ffn.experts.{expert}.{projection}'
                    model.sh.verify(prefix+'.weight'); model.sh.verify(prefix+'.scale')
                    w13[expert, index].copy_(model.sh.get(prefix+'.weight')[rank*1152:(rank+1)*1152])
                    scales[expert, index].copy_(model.sh.get(prefix+'.scale')[rank*1152:(rank+1)*1152])
            previous = stage, rank
        rows = x.shape[0]
        own = tuple(torch.empty(rows, 3, 1152, device=dev) for _ in range(2))
        reference = tuple(torch.empty_like(p) for p in own)
        slots = torch.empty_like(old_slots)
        pipeline, workspace = bank.pipelines[rows], bank.workspaces[rows]
        pipeline.forward_into(x, ids, rw, slots, bank.status, workspace, projections=own)
        diagnostic_equal = torch.equal(slots, old_slots)
        hq_exact = torch.equal(workspace.hq.view(torch.int16), NO.swiglu_qdq(*own, rw).view(torch.int16))
        assert diagnostic_equal, 'diagnostic stores changed expert SLOT output'
        assert hq_exact, 'routed activation bits differ from own-projection reference'
        n1_projection_oracle[(rows*3, 1152//64)](x, ids, rw, w13, scales, *reference, 1152, 5120, 10.,
            PACKED=True, BN=64, BK=256, INTERP=False, num_warps=4, enable_fp_fusion=False)
        differences = []
        xc, idc = x.cpu(), ids.cpu()
        for projection, got, expected in zip(('w1', 'w3'), own, reference):
            g, e = got.cpu(), expected.cpu()
            for row, slot, channel in (g.bfloat16().view(torch.int16) != e.bfloat16().view(torch.int16)).nonzero().tolist():
                expert = int(idc[row, slot])
                prefix = f'mtp.{stage}.ffn.experts.{expert}.{projection}'
                destination = rank*1152+channel
                w = model.sh.get(prefix+'.weight')[destination:destination+1]
                scale = model.sh.get(prefix+'.scale')[destination:destination+1]
                exact = float((xc[row].double() @ F.decode_fp4(w, scale)[0].double()).float())
                differences.append({'projection': projection, 'index': [row, slot, channel], 'expert': expert,
                                    'candidate_fp32': float(g[row, slot, channel]),
                                    'n1_fp32': float(e[row, slot, channel]), 'fp64_sum_fp32': exact,
                                    'candidate_bf16': float(g[row, slot, channel].bfloat16()),
                                    'n1_bf16': float(e[row, slot, channel].bfloat16())})
        elements = 2*own[0].numel()
        agreement = 1-len(differences)/elements
        records.append({'stage': stage, 'rank': rank, 'rows': rows, 'variant': pipeline.variant,
                        'elements': elements, 'bf16_agreement_vs_n1': agreement,
                        'differences': differences, 'gate_9999': agreement >= .9999,
                        'diagnostic_slot_equal': diagnostic_equal, 'hq_self_bit_exact': hq_exact})
        bank.check()
        args.output.write_text(json.dumps(result, indent=1)+'\n')
        print(stage, rank, rows, agreement, flush=True)
    result['complete'] = True
    result['all_gates_pass'] = (all(v['gate_9999'] and v['diagnostic_slot_equal'] and v['hq_self_bit_exact'] for v in records)
                                and all(v['bit_differences'] == 0 for v in epilogues))
    args.output.write_text(json.dumps(result, indent=1)+'\n')
    if not result['all_gates_pass']:
        raise SystemExit('published N1 expert BF16 agreement gate failed')


if __name__ == '__main__':
    main()
