"""Export aggregate qualification evidence without element values or token data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
from typing import Any


def bf16(value: float) -> float:
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    rounded = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack('<f', struct.pack('<I', rounded))[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    def read(name: str) -> dict[str, Any]:
        return json.loads((args.root/name).read_text())
    graph = read('final_graph.json')
    if not graph['all_ok'] or len(graph['cases']) != 8:
        raise ValueError('graph qualification incomplete')
    test_log = (args.root.parent/'logs/final-tests.log').read_text()
    match = re.search(r'(\d+) passed', test_log)
    if match is None or int(match[1]) != 171:
        raise ValueError('full focused suite does not show 171 passes')
    ranks = []
    for rank in (0, 1):
        report = read(f'final_emu_r{rank}.json')
        if not report['complete'] or not report['all_gates_pass'] or report['execution_mode'] != 'emulated_tp2':
            raise ValueError('full emulated model qualification incomplete')
        experts = [v for v in report['records'] if v['kind'] == 'expert']
        dense = [v for v in report['records'] if v['kind'] == 'dense']
        pairs = [v for v in report['records'] if v['kind'] == 'shared_pair_bf16']
        if len(experts) != 12 or len(dense) != 73 or len(pairs) != 3:
            raise ValueError('missing stage/shape coverage')
        bf_outputs = [v for v in dense if v['output_dtype'] == 'torch.bfloat16']
        partials = [v for v in dense if v['output_dtype'] == 'torch.float32']
        ranks.append({'rank': rank, 'cfg': report['cfg'],
            'expert_projection_elements': sum(d['elements'] for v in experts for d in v['gu_bf16_vs_f64']),
            'expert_gu_bf16_vs_f64_unequal': sum(d['unequal'] for v in experts for d in v['gu_bf16_vs_f64']),
            'expert_slot_elements': sum(v['slot_vs_f64']['elements'] for v in experts),
            'expert_slot_max_relative_f64': max(v['slot_vs_f64']['relative_l2'] for v in experts),
            'expert_slot_vs_kx_unequal': sum(v['slot_vs_kx']['unequal'] for v in experts),
            'dense_bf16_output_elements': sum(v['vs_n1_fp8']['elements'] for v in bf_outputs),
            'dense_bf16_vs_n1_unequal': sum(v['vs_n1_fp8']['unequal'] for v in bf_outputs),
            'dense_bf16_vs_previous_cublas_unequal': sum(v['vs_bf16_cublas']['unequal'] for v in bf_outputs),
            'dense_partial_vs_previous_cublas_unequal': sum(v['vs_bf16_cublas']['unequal'] for v in partials),
            'dense_partial_max_relative_previous_cublas': max(v['vs_bf16_cublas']['relative_l2'] for v in partials),
            'dense_min_bf16_agreement_n1': min(1-v['vs_n1_fp8']['unequal']/v['vs_n1_fp8']['elements'] for v in bf_outputs),
            'dense_raw_fp32_max_relative_f64': max(v['fp32_vs_f64']['relative_l2'] for v in dense),
            'dense_partial_fp32_max_relative_f64': max(v['fp32_vs_f64']['relative_l2'] for v in partials),
            'dense_partial_pointwise_failures_1e6': sum(v['fp32_vs_f64']['pointwise_relative_failures_1e6'] for v in partials),
            'paired_bf16_vs_previous_unequal': sum(v['vs_previous']['unequal'] for v in pairs),
            'paired_hq_vs_previous_unequal': sum(v['hq_vs_previous']['unequal'] for v in pairs),
            'shared_epilogue_exact_cases': sum(v['kind']=='shared_activation' and v['gate_bit_exact'] for v in report['records']),
            'post_tp_epilogue_exact_cases': sum(v['kind']=='post_tp_combine' and v['gate_bit_exact'] for v in report['records'])})
    projection = read('bit-oracle.json')
    if not projection['complete'] or not projection['all_gates_pass'] or len(projection['records']) != 24:
        raise ValueError('published expert projection qualification incomplete')
    diffs = [d for v in projection['records'] for d in v['differences']]
    if len(projection['epilogue_bit_checks']) != 48 or any(v['bit_differences'] for v in projection['epilogue_bit_checks']):
        raise ValueError('full emulated epilogue bit qualification incomplete')
    projection_summary = {
        'cases': 24, 'elements': sum(v['elements'] for v in projection['records']), 'bf16_differences': len(diffs),
        'exact_shared_and_post_tp_bit_cases': len(projection['epilogue_bit_checks']),
        'minimum_bf16_agreement_vs_n1': min(v['bf16_agreement_vs_n1'] for v in projection['records']),
        'candidate_matches_fp64_rounding': sum(struct.pack('<f', d['candidate_bf16']) == struct.pack('<f', bf16(d['fp64_sum_fp32'])) for d in diffs),
        'n1_matches_fp64_rounding': sum(struct.pack('<f', d['n1_bf16']) == struct.pack('<f', bf16(d['fp64_sum_fp32'])) for d in diffs)}
    source = Path(__file__).resolve().parents[3]
    digests = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
               (source/'kernels/fusedmoe/fusedmoe.cu', source/'kernels/fp8dense/dense.cu')}
    result = {'scope': 'Standard N1 benchmark windows; complete TP2 emulation, full real model activations, both shards. No captured-prefix state verdict.',
              'component_tests_passed': 171, 'graph_cases_passed': 8, 'ranks': ranks,
              'published_n1_projection_gate': projection_summary, 'cuda_source_sha256': digests,
              'numerical_policy': 'Recorded normwise 1e-6 FP32 gates and BF16 agreement gates; not universal pointwise relative 1e-6.'}
    args.output.write_text(json.dumps(result, indent=1)+'\n')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
