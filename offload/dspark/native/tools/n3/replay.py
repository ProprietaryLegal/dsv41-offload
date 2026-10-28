"""V1 same-input capture replay through the serving backend (private dumps).

--verify-root names the dspark directory containing V1's verify package. This
lane does not copy or silently alter V1's independent oracle/classifier.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

import torch

from ....worker.server import build_core_tp2
from ....worker.window import WindowRange


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('arm')
    ap.add_argument('--verify-root', type=Path, required=True)
    ap.add_argument('--captures', default='/opt/dsv41/x86host-data/dspark/captures/A-capture-20261022/dspark-captures')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--keys', required=True, help='comma-separated V1 sample keys; no accidental partial-set pass')
    ap.add_argument('--root', default='/opt/dsv41/x86host-data/dspark')
    a = ap.parse_args()
    if os.environ.get('DSPARK_WORKER_TP_EMULATE') != '1':
        from .guards import require_pair_ownership
        require_pair_ownership()
    sys.path.insert(0, str(a.verify_root))
    from verify.captures import CaptureSet
    cs = CaptureSet(a.captures)
    if cs.plumbing:
        raise RuntimeError(cs.plumbing[:3])
    wanted = set(a.keys.split(','))
    samples = [s for s in cs.samples if s.key in wanted]
    if {s.key for s in samples} != wanted:
        raise ValueError('requested sample keys absent from capture')
    grouped = collections.defaultdict(list)
    for s in samples:
        grouped[s.round].append(s)
    os.environ['DSPARK_WORKER_ARM'] = a.arm
    os.environ['DSPARK_V1_DUMP_DIR'] = str(a.out)
    os.environ['DSPARK_V1_ROUNDS'] = ','.join(str(i+1) for i in range(len(grouped)))
    core, model = build_core_tp2(a.root, max_model_len=1048576)
    m = cast(Any, model)
    sink = core.backend.sink
    for rnd, batch in sorted(grouped.items()):
        m.reset_slots(list(range(4)))
        reqs = []
        records = []
        for slot, sample in enumerate(sorted(batch, key=lambda s: s.b)):
            inputs = cs.arm_inputs(sample)
            # Use the same source for EVERY position. Mixed prefill rows inject
            # engine_mx unchanged; tap rows pass the real production insert graph.
            groups = collections.defaultdict(list)
            for pos, source, row in inputs['window']:
                groups[source].append((pos, row))
            for source, rows in groups.items():
                for start in range(0, len(rows), 128):
                    chunk = rows[start:start+128]
                    positions = torch.tensor([p for p, _ in chunk]).pin_memory()
                    slots = torch.full_like(positions, slot).pin_memory()
                    values = torch.stack([row for _, row in chunk]).pin_memory()
                    if source == 'taps':
                        m.insert_taps_graphed(slots, positions, values)
                    else:
                        m.insert_mx_eager(slots, positions, values)
            rng = WindowRange(128)
            first = inputs['window'][0][0]
            rng.apply(first, len(inputs['window']), sample.P+1)
            reqs.append((slot, sample.anchor, sample.P+1, rng, 5))
            records.append(dict(sample.__dict__, key=sample.key))
        sink.samples = records
        sink.wire = {'capture_round': rnd}
        core.backend.draft(reqs, 5, False)
        print(f'captured {rnd}: {len(batch)} samples', flush=True)
    receipt = {'arm': a.arm, 'samples': len(sink.files), 'keys': sorted(wanted),
               'identity': core.expected.drafter_identity.hex(), 'head': 'target-shared EXL3 K6'}
    a.out.joinpath('capture-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
