"""NB1 item 2 gate over collected evidence (``collect.py`` JSONs).

1. OFF is inert: the NB1 runtime with the switch unset reproduces the frozen N4 runtime byte for byte (the digest
   of every eager/graph output and inserted window, B1 digests, and every aggregate).
2. ON is batch-invariant: IDs AND confidence bit-identical alone vs packed at every B / n_out / path / fixture.
3. ON keeps B1: every alone (B1) eager and graph output digest equals the frozen runtime's.

Level 2 (insertion in 8-row sub-batches) changes B1 window KV for inserts above 8 rows by design, so its B1
criterion is the brief's fallback, fixed before the frozen accuracy numbers were read: for every insert projection
and rows-per-request, mean |output - FP64| at level 2 <= 1.05 x the default runtime's (``--on2``). Insertion
invariance is required too: every packed insertion window equals the alone one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

VOLATILE = {'log_sha256', 'repo', 'seconds', 'peak_allocated_bytes', 'invariant_env', 'runtime_sources_sha256',
            'runtime_source_files', 'triton_compiled_keys'}


def aggregates(s: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in s.items() if k not in VOLATILE}


def gate(frozen: dict[str, Any], off: dict[str, Any], on: dict[str, Any]) -> dict[str, Any]:
    if not frozen.get('output_digest') or not off.get('output_digest'):
        raise ValueError('frozen and OFF evidence need output digests')
    off_inert = dict(output_digest_equal=frozen['output_digest'] == off['output_digest'],
                     b1_digests_equal=frozen['b1_digests'] == off['b1_digests'],
                     aggregates_equal=aggregates(frozen) == aggregates(off),
                     differing_aggregate_keys=sorted(k for k in aggregates(frozen)
                                                     if aggregates(frozen)[k] != aggregates(off).get(k)))
    cells = []
    for key, v in sorted(on['final'].items()):
        n = len(v['ids_exact'])
        cells.append(dict(cell=key, comparisons=v['comparisons'], ids_exact_all=all(x == v['comparisons']
                                                                                  for x in v['ids_exact']),
                          conf_bit_exact=v['conf_exact'], conf_max_abs=v['conf_max_abs'], positions=n))
    invariant = all(c['ids_exact_all'] and c['conf_bit_exact'] == c['comparisons'] for c in cells)
    b1 = dict(digests_equal_to_frozen=on['b1_digests'] == frozen['b1_digests'],
              requests=len(on['b1_digests']),
              differing=sorted(k for k in on['b1_digests'] if on['b1_digests'][k] != frozen['b1_digests'].get(k)))
    return dict(label='MEASURED', off_inert=off_inert, on_invariant=invariant, on_cells=cells, on_b1_unchanged=b1,
                on_first_divergent_e2e=on['first_divergent_e2e'],
                on_isolated_batch_dependent_ops=on['isolated_batch_dependent_ops'],
                on_insert_ring=on['insert_ring'], on_depth_transitions=on['depth_transitions'],
                passed=bool(off_inert['output_digest_equal'] and off_inert['b1_digests_equal'] and
                            off_inert['aggregates_equal'] and invariant and b1['digests_equal_to_frozen']))


NOT_WORSE = 1.05


def gate_level2(frozen: dict[str, Any], on2: dict[str, Any]) -> dict[str, Any]:
    finals = [v for v in on2['final'].values()]
    draft_invariant = all(all(x == v['comparisons'] for x in v['ids_exact']) and v['conf_exact'] == v['comparisons']
                          for v in finals)
    rings = on2['insert_ring']
    insert_invariant = bool(rings) and all(v['exact'] == v['comparisons'] for v in rings.values())

    def ops(s: dict[str, Any]) -> dict[str, dict[str, Any]]:
        # 'r0.insert.fixed_rows:<name>/n32/synthetic' and 'r0.insert.dense:<name>/n32/synthetic' -> '<name>/n32/...'
        return {k.split(':', 1)[1]: v for k, v in s['insert_alone_fp64'].items() if k.startswith('r0.')}
    base, new = ops(frozen), ops(on2)
    if not base or base.keys() != new.keys():
        raise ValueError('frozen and level-2 evidence need matching --fp64-alone-insert cells')
    accuracy = [dict(cell=k, default_mean_abs_err=base[k]['mean_abs_err'], level2_mean_abs_err=new[k]['mean_abs_err'],
                     ratio=new[k]['mean_abs_err'] / base[k]['mean_abs_err'],
                     default_max_abs_err=base[k]['max_abs_err'], level2_max_abs_err=new[k]['max_abs_err'])
                for k in sorted(base)]
    not_worse = all(c['ratio'] <= NOT_WORSE for c in accuracy)
    changed = sorted(k for k in on2['b1_digests'] if on2['b1_digests'][k] != frozen['b1_digests'].get(k))
    return dict(label='MEASURED', draft_invariant=draft_invariant, insert_invariant=insert_invariant,
                insert_ring=rings, b1_changed=changed, b1_requests=len(on2['b1_digests']),
                insert_accuracy=accuracy, accuracy_not_worse=not_worse, not_worse_bound=NOT_WORSE,
                passed=bool(draft_invariant and insert_invariant and not_worse))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--off', type=Path, required=True)
    parser.add_argument('--on', type=Path, required=True)
    parser.add_argument('--on2', type=Path, help='level-2 evidence (insertion sub-batches)')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = gate(*(json.loads(p.read_text()) for p in (args.frozen, args.off, args.on)))
    if args.on2 is not None:
        result['level2'] = gate_level2(json.loads(args.frozen.read_text()), json.loads(args.on2.read_text()))
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(passed=result['passed'], off_inert=result['off_inert'],
                          on_invariant=result['on_invariant'], on_b1=result['on_b1_unchanged'],
                          level2_passed=result.get('level2', {}).get('passed'))))


if __name__ == '__main__':
    main()
