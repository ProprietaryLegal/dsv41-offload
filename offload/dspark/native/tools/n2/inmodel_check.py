"""Audit actual eager model expert/dense calls against decoded-value FP64.

Outputs include element-level arithmetic differences and stay outside git.
Use only via N2's bounded lock runner. No audit hooks enter the serving model.
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace
import json
from pathlib import Path
from types import MethodType
from typing import Any

import torch

from ....worker.checkpoint import Checkpoint
from ....worker.tools.parity_ref import make_taps
from ....worker.tools.tp2_bench import _EmbedOnly, fill_windows_tp, host_inp, setup_inputs
from ....worker.tp2 import check_visible
from ... import formats as F, ops as NO
from ...kx import KXFusedBank, KXRegisterBank, KXSelectiveDense, KXSharedPair
from ...tp2_native import build

ROOT = '/opt/dsv41/x86host-data/dspark'
DETAIL_PATH: Path | None = None
DETAIL_INDEX = 0


def differences(got: torch.Tensor, ref: torch.Tensor) -> dict[str, Any]:
    global DETAIL_INDEX
    got, ref = got.cpu(), ref.cpu()
    delta = got.double() - ref.double()
    bad = (got != ref).nonzero()
    indices = tuple(bad.unbind(1))
    detail = [{'index': idx, 'got': value, 'reference': expected}
              for idx, value, expected in zip(bad.tolist(), got[indices].tolist(), ref[indices].tolist(), strict=True)]
    result: dict[str, Any] = {'elements': got.numel(), 'unequal': len(bad),
            'relative_l2': float(delta.norm() / ref.double().norm().clamp_min(1e-30)),
            'max_abs': float(delta.abs().max()),
            'pointwise_relative_failures_1e6': int((delta.abs() > ref.double().abs()*1e-6).sum())}
    if DETAIL_PATH is None:
        result['differences'] = detail
    else:
        with DETAIL_PATH.open('a') as stream:
            stream.write(json.dumps({'record': DETAIL_INDEX, 'differences': detail}) + '\n')
        result['differences_ref'] = {'file': DETAIL_PATH.name, 'record': DETAIL_INDEX}
        DETAIL_INDEX += 1
    return result


def main() -> None:
    global DETAIL_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--stage', type=int, choices=(0, 1, 2))
    parser.add_argument('--variant', type=int)
    parser.add_argument('--dense-profile', choices=('initial', 'measured-small'))
    parser.add_argument('--dense-only', action='store_true', help='reuse separate expert qualification; check all dense calls')
    parser.add_argument('--expert-report', type=Path, help='completed matching all-component report for dense-only reuse')
    parser.add_argument('--emulated', action='store_true', help='run both TP2 ranks and audit the selected rank')
    args = parser.parse_args()
    DETAIL_PATH = args.output.with_suffix('.differences.jsonl')
    DETAIL_PATH.write_text('')
    check_visible(pair=False)
    torch.set_num_threads(8)
    dev = torch.device('cuda:0')
    from ...tp2_native import ARMS
    cfg = ARMS['native-hybrid-kx2']
    if args.dense_profile is not None:
        cfg = replace(cfg, dense_profile=args.dense_profile)
    reused_experts: dict[str, str] | None = None
    if args.dense_only:
        prior_path = args.expert_report or args.output.with_name(f'final_check_r{args.rank}.json')
        prior_bytes = prior_path.read_bytes()
        prior = json.loads(prior_bytes)
        expected = {(stage, batch*5) for stage in range(3) for batch in args.batches}
        covered = {(v['stage'], v['rows']) for v in prior['records'] if v['kind'] == 'expert'}
        if (not prior.get('complete') or not prior.get('all_gates_pass') or not expected <= covered
                or prior.get('cfg') != json.loads(json.dumps(cfg.__dict__))
                or prior.get('execution_mode', 'solo_shard') != ('emulated_tp2' if args.emulated else 'solo_shard')):
            raise RuntimeError('dense-only reuse requires complete matching passed expert coverage')
        reused_experts = {'report': str(prior_path), 'sha256': hashlib.sha256(prior_bytes).hexdigest()}
    model = build(cfg, [dev, dev], Checkpoint(ROOT), slots=8,
                  **({} if args.emulated else {'solo_rank': args.rank}))
    rank = model.ranks[args.rank]
    owned_dense = {id(rank.main_proj)} | {id(getattr(st, name)) for st in rank.stages
                                         for name in ('wq_a', 'wkv', 'wq_b', 'wo_a_n', 'wo_b', 'sh_w1', 'sh_w3', 'sh_w2')}
    owned_pairs = {id(st.shared_pair) for st in rank.stages}
    owned_shared = {st.shared_hq.untyped_storage().data_ptr() for st in rank.stages if st.shared_hq is not None}
    owned_combined = {out.untyped_storage().data_ptr() for st in rank.stages for out in st.nbank.combined_outputs.values()}
    records: list[dict[str, Any]] = []
    dense_inputs: list[tuple[KXSelectiveDense, torch.Tensor, torch.Tensor]] = []
    pair_inputs: list[tuple[KXSharedPair, torch.Tensor, torch.Tensor]] = []
    original_pair = KXSharedPair.__call__

    def pair_call(self: KXSharedPair, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projections = original_pair(self, x)
        if x.shape[0] == 20 and id(self) in owned_pairs:
            pair_inputs.append((self, x.clone(), torch.cat(projections, dim=1)))
        return projections

    KXSharedPair.__call__ = pair_call
    shared_inputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    combine_inputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    from ....kernels.fusedmoe import epilogue as combine_epilogue
    original_combine = combine_epilogue.combine_official_into

    def combine_call(parts: torch.Tensor, ids: torch.Tensor, out: torch.Tensor) -> None:
        original_combine(parts, ids, out)
        if out.untyped_storage().data_ptr() in owned_combined:
            combine_inputs.append((parts.clone(), ids.clone(), out.clone()))

    combine_epilogue.combine_official_into = combine_call
    from ....kernels.fp8dense import epilogue
    original_shared = epilogue.shared_activation_into

    def shared_call(gate: torch.Tensor, up: torch.Tensor, ones: torch.Tensor, out: torch.Tensor) -> None:
        original_shared(gate, up, ones, out)
        if out.untyped_storage().data_ptr() in owned_shared:
            shared_inputs.append((gate.clone(), up.clone(), out.clone()))

    epilogue.shared_activation_into = shared_call
    original_call = KXSelectiveDense.__call__

    def dense_call(self: KXSelectiveDense, x: torch.Tensor, out_f32: bool = False,
                   out: torch.Tensor | None = None) -> torch.Tensor:
        result = original_call(self, x, out_f32, out)
        if x.shape[0] in self.configs and id(self) in owned_dense:
            dense_inputs.append((self, x.clone(), result.clone()))
        return result

    KXSelectiveDense.__call__ = dense_call
    # Collect from one-rank actual model activations, never substitute a random
    # projection input. Synthetic taps seed the standard N1 full-model bench.
    expert_inputs: list[tuple[int, KXFusedBank, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for stage, st in enumerate(rank.stages):
        bank = st.nbank
        assert isinstance(bank, KXFusedBank)
        if args.stage is not None and stage != args.stage:
            continue
        if args.variant is not None:
            bank.pipelines = {rows: replace(pipeline, variant=args.variant)
                              for rows, pipeline in bank.pipelines.items()}

        def forward(self: KXFusedBank, x: torch.Tensor, ids: torch.Tensor, rw: torch.Tensor,
                    h: torch.Tensor | None = None, out: torch.Tensor | None = None,
                    stage: int = stage) -> torch.Tensor:
            result = KXFusedBank.forward_slots(self, x, ids, rw, h, out)
            expert_inputs.append((stage, self, x.clone(), ids.clone(), rw.clone(), result.clone()))
            return result

        bank.forward_slots = MethodType(forward, bank)
    toks = setup_inputs(dev, ROOT)
    taps = make_taps(_EmbedOnly(rank.embed), toks, torch.Generator(device=dev).manual_seed(5))
    model.insert_eager(torch.full((8,), 7), torch.arange(2000, 2008), taps[:8])
    fill_windows_tp(model, taps.cpu())
    for batch in args.batches:
        model.draft(host_inp(batch, toks.cpu()), 5)
    KXSelectiveDense.__call__ = original_call
    KXSharedPair.__call__ = original_pair
    epilogue.shared_activation_into = original_shared
    combine_epilogue.combine_official_into = original_combine
    result: dict[str, Any] = {'arm': 'native-hybrid-kx2', 'rank': args.rank,
                              'cfg': cfg.__dict__,
                              'scope': 'dense_with_completed_expert_reuse' if args.dense_only else 'all_components',
                              'execution_mode': 'emulated_tp2' if args.emulated else 'solo_shard',
                              'reused_experts': reused_experts,
                              'inputs': 'N1 benchmark windows and actual model expert/dense activations; solo shard',
                              'records': records, 'complete': False}

    def save() -> None:
        args.output.write_text(json.dumps(result, indent=1) + '\n')

    for gate, up, output in shared_inputs:
        previous = NO.act_qdq(NO.swiglu_bf16(gate, up))
        comparison = differences(output, previous)
        records.append({'kind': 'shared_activation', 'rows': gate.shape[0], 'vs_previous': comparison,
                        'gate_bit_exact': comparison['unequal'] == 0})
        save()
    for parts, ids, output in combine_inputs:
        previous = NO.moe_combine_official(parts[:, :3], parts[:, 3], ids)
        comparison = differences(output, previous)
        records.append({'kind': 'post_tp_combine', 'rows': parts.shape[0], 'vs_previous': comparison,
                        'gate_bit_exact': comparison['unequal'] == 0})
        save()
    for pair, x, joined in pair_inputs:
        old = torch.cat((pair.gate.baseline(x), pair.up.baseline(x)), dim=1)
        ref = (x.cpu().double() @ pair.weight.cpu().double().t()).float()
        raw = torch.mm(x, pair.weight.t(), out_dtype=torch.float32)
        old_difference, ref_difference = differences(joined, old), differences(joined, ref.bfloat16())
        raw_difference = differences(raw, ref)
        half = joined.shape[1]//2
        own_hq = NO.act_qdq(NO.swiglu_bf16(joined[:, :half], joined[:, half:]))
        old_hq = NO.act_qdq(NO.swiglu_bf16(old[:, :half], old[:, half:]))
        records.append({'kind': 'shared_pair_bf16', 'rows': x.shape[0], 'vs_previous': old_difference,
                        'bf16_vs_f64': ref_difference, 'raw_fp32_vs_f64': raw_difference,
                        'hq_vs_previous': differences(own_hq, old_hq),
                        'gate_bf16_999': old_difference['unequal']/joined.numel() <= .001,
                        'gate_bf16_f64_999': ref_difference['unequal']/joined.numel() <= .001,
                        'gate_raw_l2_1e6': raw_difference['relative_l2'] <= 1e-6})
        save()
    for dense, x, got in dense_inputs:
        groups, inputs, outputs = dense.candidate.groups, dense.candidate.inputs, dense.candidate.outputs
        if groups == 1:
            base = dense.baseline(x, out_f32=got.dtype == torch.float32)
        else:
            base = torch.bmm(x.view(-1, groups, inputs).transpose(0, 1),
                             dense.w.view(groups, outputs, inputs).transpose(1, 2)).transpose(0, 1).reshape(x.shape[0], -1)
        xc, wc = x.cpu().double(), dense.w.cpu().double()
        ref = torch.cat([xc[:, g*inputs:(g+1)*inputs] @ wc[g*outputs:(g+1)*outputs].t()
                         for g in range(groups)], dim=1).float()
        raw = torch.empty_like(got, dtype=torch.float32)
        splits, tiles = dense.configs[x.shape[0]]
        dense.candidate.linear_into(x, raw, scratch=dense.scratch[x.shape[0]], splits=splits, row_tiles=tiles)
        versus_base = differences(got, base)
        raw_vs_f64 = differences(raw, ref)
        source_name, window = dense.name.rsplit('[', 1)
        row_window, col_window = window.rstrip(']').split(',')
        r0, r1 = map(int, row_window.split(':'))
        c0, c1 = map(int, col_window.split(':'))
        codes = model.sh.get(source_name + '.weight')[r0:r1, c0:c1]
        scale_grid = model.sh.get(source_name + '.scale')[r0//32:r1//32, c0//32:c1//32]
        n1_outputs = []
        for group in range(groups):
            oracle = NO.NativeDense(source_name, 'fp8', codes[group*outputs:(group+1)*outputs].contiguous().to(dev),
                                    F.e8m0_to_f32(scale_grid[group*outputs//32:(group+1)*outputs//32]).to(dev))
            n1_outputs.append(oracle(x[:, group*inputs:(group+1)*inputs], out_f32=got.dtype == torch.float32))
        versus_n1 = differences(got, torch.cat(n1_outputs, dim=1))
        records.append({'kind': 'dense', 'name': dense.name, 'rows': x.shape[0],
                        'output_dtype': str(got.dtype),
                        'vs_bf16_cublas': versus_base, 'fp32_vs_f64': raw_vs_f64,
                        'vs_n1_fp8': versus_n1,
                        'gate_bf16_999': got.dtype == torch.float32 or versus_base['unequal'] / got.numel() <= .001,
                        'gate_n1_bf16_999': got.dtype == torch.float32 or versus_n1['unequal'] / got.numel() <= .001,
                        'gate_raw_l2_1e6': raw_vs_f64['relative_l2'] <= 1e-6})
        save()
    for stage, bank, x, ids, rw, got in expert_inputs:
        if args.dense_only:
            continue
        rows = x.shape[0]
        pipeline, workspace = bank.pipelines[rows], bank.workspaces[rows]
        projections = tuple(torch.empty(rows, 3, bank.I, device=dev) for _ in range(2))
        again = torch.empty_like(got)
        pipeline.forward_into(x, ids, rw, again, bank.status, workspace, projections=projections)
        assert torch.equal(again, got)
        own_hq = workspace.hq.clone()
        assert torch.equal(NO.swiglu_qdq(*projections, rw).view(torch.int16), own_hq.view(torch.int16))
        old = KXRegisterBank.forward_slots(bank, x, ids, rw)
        from ....kernels.fp8dense.screen import measured

        timing = measured(lambda: pipeline.forward_into(x, ids, rw, again, bank.status, workspace), warm_seconds=.1)
        control_timing = None
        if args.variant is not None:
            control = replace(pipeline, variant=131072)
            control_timing = measured(lambda: control.forward_into(x, ids, rw, again, bank.status, workspace),
                                      warm_seconds=.1)
        ref_gu = [torch.empty_like(p, device='cpu') for p in projections]
        cpu_x, cpu_ids = x.cpu(), ids.cpu()
        for slot, expert in enumerate(cpu_ids.flatten().tolist()):
            for j, proj in enumerate(('w1', 'w3')):
                prefix = f'mtp.{stage}.ffn.experts.{expert}.{proj}'
                w, scale = model.sh.get(prefix + '.weight'), model.sh.get(prefix + '.scale')
                weight = F.decode_fp4(w[args.rank*1152:(args.rank+1)*1152],
                                      scale[args.rank*1152:(args.rank+1)*1152])
                ref_gu[j][slot//3, slot%3] = (cpu_x[slot//3].double() @ weight.double().t()).float()
        ref_hq = NO.swiglu_qdq(*(p.to(dev) for p in ref_gu), rw).cpu()
        ref_down = torch.empty_like(got, device='cpu')
        own_down = torch.empty_like(ref_down)
        own_hq_cpu = own_hq.cpu()
        for slot, expert in enumerate(cpu_ids.flatten().tolist()):
            prefix = f'mtp.{stage}.ffn.experts.{expert}.w2'
            w, scale = model.sh.get(prefix + '.weight'), model.sh.get(prefix + '.scale')
            weight = F.decode_fp4(w[:, args.rank*576:(args.rank+1)*576],
                                  scale[:, args.rank*36:(args.rank+1)*36]).double()
            ref_down[slot//3, slot%3] = (ref_hq[slot].double() @ weight.t()).float()
            own_down[slot//3, slot%3] = (own_hq_cpu[slot].double() @ weight.t()).float()
        gu_differences = [differences(p.bfloat16(), q.bfloat16()) for p, q in zip(projections, ref_gu)]
        full_error, down_error = differences(got, ref_down), differences(got, own_down)
        records.append({'kind': 'expert', 'stage': stage, 'rows': rows, 'variant': pipeline.variant,
                        'timing': timing,
                        'default_131072_timing': control_timing,
                        'gu_bf16_vs_f64': gu_differences, 'slot_vs_f64': full_error,
                        'slot_vs_f64_own_hq': down_error, 'slot_vs_kx': differences(got, old),
                        'hq_self_exact': True, 'hq_vs_f64': differences(own_hq, ref_hq),
                        'gate_gu_9999': all(d['unequal']/d['elements'] <= .0001 for d in gu_differences),
                        'gate_slot_1e6': full_error['relative_l2'] <= 1e-6})
        bank.check()
        save()
        print(f"stage={stage} rows={rows} slot_rel={full_error['relative_l2']}", flush=True)
    result['complete'] = True
    result['all_gates_pass'] = all(all(value for key, value in record.items() if key.startswith('gate_'))
                                   for record in records)
    save()
    if not result['all_gates_pass']:
        raise SystemExit('in-model numerical gate failed; inspect element-level evidence')


if __name__ == '__main__':
    main()
