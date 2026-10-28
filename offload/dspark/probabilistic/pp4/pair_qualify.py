"""Coordinator-only pair subprocess sequence. Default is a no-action preview."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from offload.launch.windows import pf1_window as pf
from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json

from .pair_contract import SCHEMA, checked_output, guard, validate

SOURCES = ('offload/dspark/probabilistic/pp4/pair_qualify.py',
           'offload/dspark/probabilistic/pp4/pair_probe.py', 'offload/dspark/probabilistic/pp4/pair_diagnostics.py',
           'offload/dspark/probabilistic/pp4/pair_contract.py',
           'offload/dspark/probabilistic/pp4/full_worker.py', 'offload/dspark/probabilistic/pp4/n4-greedy-reference.json',
           'offload/dspark/worker/sampled_fused.py', 'offload/dspark/worker/tp2.py', 'offload/dspark/worker/server.py')


def execute(out: Path, commit: str, run: str, deadline: float) -> dict[str, Json]:
    checked_output(out); guard()
    ck.require(sys.executable == '/home/operator/miniconda3/bin/python3', 'qualified interpreter required')
    ck.require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == commit,
               'pair source commit differs')
    out.mkdir(parents=True, exist_ok=False)
    result: dict[str, Json] = {'schema': SCHEMA, 'source_commit': commit, 'run': run,
        'complete': False, 'classification': 'MEASURED physical pair', 'gpus': pf.WORKER_GPUS,
        'source_sha256': {name: pf.digest(pf.ROOT/name) for name in SOURCES}}
    locks = tuple(int(os.environ[key]) for key in ('PP4_PAIR_LOCK_FD0', 'PP4_PAIR_LOCK_FD1'))
    started = time.monotonic()
    for mode, extra, limit in (
        ('correctness', ['-m', 'offload.dspark.probabilistic.pp4.pair_probe', '--mode', 'correctness'], 240),
        ('timeout0', ['-m', 'offload.dspark.probabilistic.pp4.pair_probe', '--mode', 'timeout0'], 60),
        ('timeout1', ['-m', 'offload.dspark.probabilistic.pp4.pair_probe', '--mode', 'timeout1'], 60),
        ('whole_worker', ['-m', 'offload.dspark.probabilistic.pp4.full_worker', '--physical-pair', '--qualification-only'], 600)):
        remaining = min(limit, deadline-time.time())
        ck.require(remaining > 5, 'pair qualification budget exhausted; restore without measurement')
        path = out/(mode+'.json')
        with (out/(mode+'.log')).open('x') as log:
            # Inherit this registered group/identity, plus exactly the GPU locks.
            # A timed-out child is killed and waited for by subprocess; the
            # outer registry still proves absence before pair restoration.
            subprocess.run([sys.executable, *extra, '--out', str(path)], stdout=log, stderr=subprocess.STDOUT,
                           timeout=remaining, check=True, pass_fds=locks)
        child = ck.read(path)
        if mode == 'correctness':
            result['cases'] = child['cases']
        else:
            if mode.startswith('timeout'):
                child['process_exited'] = True  # only after checked wait, never before launch
            result[mode] = child
        (out/'receipt.json').write_text(json.dumps(result, allow_nan=False)+'\n')
    result['seconds'] = time.monotonic()-started
    result['complete'] = True
    validate(result, commit, run)
    (out/'receipt.json').write_text(json.dumps(result, allow_nan=False)+'\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--commit', default='')
    parser.add_argument('--run', default='')
    parser.add_argument('--deadline', type=float, default=0)
    args = parser.parse_args()
    if not args.execute:
        print('DRY RUN: fenced HOLD only; physical pair B1-B4/depth4-5, changed-input graphs, mixed T0, '
              'fresh solo q/ID receipt, isolated peer-timeout/sticky-error processes, whole-K2 archived N4 identity; restore.')
        return
    ck.require(args.out is not None, 'pair output required')
    assert isinstance(args.out, Path)
    execute(args.out, args.commit, args.run, args.deadline)


if __name__ == '__main__':
    main()
