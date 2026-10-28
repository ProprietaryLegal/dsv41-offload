"""#61161 verifier-RNG gate for PP4 windows (Fable review 2026-10-27: "keep the #61161 RNG test as a gate").

The coordinator runs it before a PP4 window, Controller CPU only:
  python -m offload.launch.windows.pp4_rng_gate --execute --out <receipt.json> --basetemp <scratch dir>
It runs AUD1's image-kernel tests (production's one-hot path; block-mode negative and #61161 positive controls) and
R-PP4's PP4-path tests (the PP3-prepared kernel PP4 ships, sampled FP32 q, multi-round, T=1, a wrong-q negative control
and the block-verification refusal) under Triton's CPU interpreter, and binds the result to HEAD, every source involved
and the generated PP4 kernel's sha256. The PP4 runner refuses before any host contact without a PASS receipt at its own
commit whose PP4 kernel equals the admission receipt's prepared verifier; every start's speculative config must also be
standard verification (`standard_spec`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

from offload.dspark.probabilistic.pp3 import prepare_verifier as pp3
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, record

SCHEMA = 'pp4-rng-gate-v1'
IMAGE_SRC = Path('/opt/dsv41/data/src-image')
AUDIT = 'offload/audit/verifier_rng'
# Test module -> registered number of passing tests (no skips allowed).
EXPECTED = {'test_verifier_rng': 6, 'test_pp4_sampled_rng': 4}
TESTS = tuple(f'{AUDIT}/{name}.py' for name in EXPECTED)
SOURCES = (*TESTS, f'{AUDIT}/multiround.py', f'{AUDIT}/kernel_loader.py', f'{AUDIT}/pp4_sampled.py',
           'offload/dspark/probabilistic/pp3/prepare_verifier.py', 'offload/launch/windows/pp4_rng_gate.py')
ACCEPTANCE_DRAW = 'u = tl_rand32(seed, pos, includes_zero=False)'
BLOCK_REFUSAL = 'if use_block_verification or synthetic_conditional_rates is not None or watermark:'


def pp4_kernel(src_root: Path = IMAGE_SRC) -> str:
    data = (src_root / pp3.UTILS).read_bytes()
    ck.require(hashlib.sha256(data).hexdigest() == pp3.PINS[pp3.UTILS], 'image verifier differs from the AUD1/PP3 pin')
    return pp3.utilities(data.decode())


def static_checks(text: str) -> dict[str, Json]:
    """Standard verification's acceptance draw untouched; sampled q refuses block verification; no #61161 step keys."""
    result: dict[str, Json] = {'acceptance_draws': text.count(ACCEPTANCE_DRAW), 'block_refusal': BLOCK_REFUSAL in text,
                               'step_keyed': 'spec_step_seed' in text}
    result['pass'] = result['acceptance_draws'] == 1 and result['block_refusal'] is True and result['step_keyed'] is False
    return result


def standard_spec(spec: str) -> None:
    """A PP4 start's --speculative-config must use standard verification (#61161 only bites block verification)."""
    data = record(cast(Json, json.loads(spec)))
    ck.require(data.get('method') == 'dspark', 'PP4 speculative method differs')
    ck.require(data.get('rejection_sample_method', 'standard') == 'standard',
               'PP4 requires standard verification; block verification needs #61161 and its gate first')
    ck.require(data.get('enable_adaptive_verification', False) is False, 'adaptive verification is refused by remote DSpark')


def junit_counts(path: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for case in ET.parse(path).getroot().iter('testcase'):
        module = case.get('classname', '').rsplit('.', 1)[-1]
        row = counts.setdefault(module, {'passed': 0, 'failed': 0, 'error': 0, 'skipped': 0})
        outcome = ('failed' if case.find('failure') is not None else 'error' if case.find('error') is not None
                   else 'skipped' if case.find('skipped') is not None else 'passed')
        row[outcome] += 1
    return counts


def _sources() -> dict[str, Json]:
    return {name: pf.digest(pf.ROOT / name) for name in SOURCES}


def run(out: Path, basetemp: Path, *, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        src_root: Path = IMAGE_SRC) -> dict[str, Json]:
    ck.require(not out.exists(), 'fresh gate receipt path required')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=pf.ROOT, text=True).strip()
    ck.require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=pf.ROOT,
                                           text=True).strip(), 'tracked source dirty; the receipt binds HEAD')
    basetemp.mkdir(parents=True, exist_ok=True)
    junit = basetemp / 'rng-gate.junit.xml'
    env = {**os.environ, 'TRITON_INTERPRET': '1', 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
           'PYTHONPATH': f'{pf.ROOT}{os.pathsep}{pf.ROOT / "offload/dspark"}'}
    started = time.time()
    proc = runner([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', f'--basetemp={basetemp / "pytest"}',
                   f'--junitxml={junit}', *TESTS], cwd=pf.ROOT, env=env, capture_output=True, text=True, check=False)
    counts = junit_counts(junit) if junit.exists() else {}
    kernel = pp4_kernel(src_root)
    static = static_checks(kernel)
    tests_ok = proc.returncode == 0 and all(
        counts.get(name) == {'passed': n, 'failed': 0, 'error': 0, 'skipped': 0} for name, n in EXPECTED.items())
    receipt: dict[str, Json] = {
        'schema': SCHEMA, 'status': 'PASS' if tests_ok and static['pass'] is True else 'FAIL', 'commit': head,
        'returncode': proc.returncode, 'tests': cast(Json, counts), 'expected': cast(Json, dict(EXPECTED)),
        'seconds': time.time() - started, 'source_sha256': _sources(),
        'image_kernel_sha256': pp3.PINS[pp3.UTILS],
        'pp4_kernel_sha256': hashlib.sha256(kernel.encode()).hexdigest(), 'static': static,
        'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'scope': 'CPU interpreter, synthetic targets; the kernel logic and the multi-round law, not GPU rounding'}
    out.write_text(json.dumps(receipt, indent=1, allow_nan=False) + '\n')
    return receipt


def validate(path: Path, head: str, admission: Mapping[str, Json]) -> dict[str, Json]:
    data = ck.read(path)
    ck.require(data.get('schema') == SCHEMA and data.get('status') == 'PASS', '#61161 verifier-RNG gate did not PASS')
    ck.require(data.get('commit') == head, 'verifier-RNG gate receipt is for another commit')
    ck.require(data.get('source_sha256') == _sources(), 'verifier-RNG gate sources changed since the receipt')
    ck.require(data.get('tests') == {name: {'passed': n, 'failed': 0, 'error': 0, 'skipped': 0}
                                     for name, n in EXPECTED.items()}, 'verifier-RNG gate test counts differ')
    ck.require(record(data['static']).get('pass') is True, 'PP4 kernel static RNG checks failed')
    staged = record(admission['verifier_sha256']).get(pp3.UTILS)
    ck.require(data.get('pp4_kernel_sha256') == staged,
               'gated PP4 kernel differs from the admission-qualified prepared verifier')
    return {'status': 'PASS', 'receipt': str(path), 'receipt_sha256': pf.digest(path), 'commit': head,
            'pp4_kernel_sha256': staged}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--basetemp', type=Path, required=True)
    args = ap.parse_args(argv)
    if not args.execute:
        print('DRY RUN: would run ' + ' '.join(TESTS) + ' on the CPU interpreter and write a PP4 RNG gate receipt')
        return 0
    receipt = run(args.out, args.basetemp)
    print(json.dumps({k: receipt[k] for k in ('status', 'commit', 'tests', 'pp4_kernel_sha256', 'seconds')}))
    return 0 if receipt['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
