"""Run the independent V1 classifier, then enforce every N3 entry gate.

The K6 head is explicitly declared. V1's official FP32 Markov/self-logit
contract may reject its BF16 head/Markov semantics; that is a failing gate,
never relabelled as a pass by this wrapper. Use --expected-keys to bind the
scope, and retain the full classified report for V1/coordinator review.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def verdict(report: dict[str, Any], band_kl: float, expected_samples: int) -> dict[str, Any]:
    classes = report['classes']
    state = classes.get('state_provenance_defect', 0) + classes.get('agree_but_state_flag', 0)
    excess = report.get('median_excess_over_band_median', {}).get('s0.attn_out', float('inf'))
    kl = report['kl_ref_arm_final_logits']['median']
    gates = {'complete_requested_scope': report['samples'] == expected_samples,
             'zero_state_defects': state == 0, 'zero_unexplained': classes.get('unexplained', 0) == 0,
             's0_attn_excess_lt_3': math.isfinite(excess) and excess < 3,
             'kl_median_within_3x_summation_band': math.isfinite(kl) and kl <= 3 * max(band_kl, 1e-12)}
    return {'pass': all(gates.values()), 'gates': gates, 'samples': report['samples'],
            'state_defects': state, 'unexplained': classes.get('unexplained', 0),
            's0_attn_excess': excess, 'kl_median': kl, 'summation_band_kl_median': band_kl,
            'head': 'target-shared EXL3 K6; official V1 policy unchanged'}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--verify-root', type=Path, required=True)
    ap.add_argument('--arm-dir', type=Path, required=True)
    ap.add_argument('--arm-name', required=True)
    ap.add_argument('--dumps', type=Path, required=True)
    ap.add_argument('--captures', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--official', required=True)
    ap.add_argument('--expected-keys', required=True, help='comma-separated keys or @path to newline-separated keys')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    keys = set(Path(a.expected_keys[1:]).read_text().split()) if a.expected_keys.startswith('@') else set(a.expected_keys.split(','))
    if not keys or any('/' in k or not k.startswith('r') for k in keys):
        raise ValueError('nonempty explicit sample keys required')
    sys.path.insert(0, str(a.verify_root))
    from verify.analyze import load_dumps, final_logits, markov_tables, kl
    arm, ref, band = load_dumps(str(a.arm_dir)), load_dumps(str(a.dumps/'mirror')), load_dumps(str(a.dumps/'mirror64'))
    if set(arm) != keys or not keys <= set(ref) & set(band):
        raise RuntimeError('dump scope mismatch or missing mirror/mirror64 keys; classification refused')
    for k in keys:
        d = arm[k]
        if d.get('n_out') != 5 or any(n not in d for n in ('kv_window', 'main_x_P', 'base_logits', 'head_hidden')):
            raise ValueError(f'{k}: incomplete V1 record')
        if any(n not in d[f's{i}'] for i in range(3) for n in ('attn_out', 'ffn_in', 'ffn_out', 'h_out', 'route')):
            raise ValueError(f'{k}: missing layer trace')
    raw = a.out.with_suffix('.classification.json')
    env = {**os.environ, 'PYTHONPATH': str(a.verify_root)}
    subprocess.run([sys.executable, '-m', 'verify.classify', '--arm-dir', str(a.arm_dir),
        '--arm-name', a.arm_name, '--representation', 'native', '--dumps', str(a.dumps),
        '--ref-mode', 'mirror', '--band-mode', 'mirror64', '--captures', a.captures,
        '--ckpt', a.ckpt, '--official', a.official, '--out', str(raw)], check=True, env=env)
    emb, head = markov_tables(a.ckpt)
    values = []
    for k in keys:
        prefix = [ref[k]['anchor']] + arm[k]['drafts'][:4]
        values.extend(kl(final_logits(ref[k]['base_logits'], prefix, emb, head),
                         final_logits(band[k]['base_logits'], prefix, emb, head)).tolist())
    band_median = sorted(values)[len(values)//2]
    result = verdict(json.loads(raw.read_text()), band_median, len(keys))
    result.update({'arm': a.arm_name, 'requested_keys': sorted(keys), 'classifier_report': str(raw)})
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if not result['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
