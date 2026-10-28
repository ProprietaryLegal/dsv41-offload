"""Render/apply/roll back a reviewed capture derivative of the exact K1 config.

CPU/files only; no ssh, GPU, HTTP, service or git mutation. Apply requires the
coordinator's SHA-bound independent review receipt. The K1 promoted hash is
updated with an explicit derivative record, preserving the base approval and
byte-exact rollback. Default CLI prints a reviewable diff and changes nothing.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
import re
import shutil
from typing import cast

BASE_COMMIT = '21bed313ac803d38327c4de1ff2e5235f329c716'
BASE_ENV = 'faa9d3e0019a8665f0b61dbf46a79134680731d43b9a9cb99ca2fa5af85373a8'
BASE_START = 'fa172986fcb3e9ee66fd62b6dc11435ec7a56abdc60247ad1e73d85fa561ad5d'
K1 = '559c5867ba42aa7daa7b4a7c8231d15ee1d35be40ae5b715b013f973193c58de'
BASE = '/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode'


# Standard library only: the apply/rollback runs on sparkhead's host python3, which has neither numpy nor torch
# (coordinator 2026-10-24); importing d2.decode_dataset for these helpers would refuse at import. Same semantics.
def obj(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('expected JSON metadata object: ' + str(path))
    return cast(dict[str, object], value)


def digest(path: Path) -> str:
    with path.open('rb') as fh:
        return hashlib.file_digest(fh, 'sha256').hexdigest()


def mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError('missing metadata object')
    return cast(Mapping[str, object], value)


def addition(source: str, run: str) -> str:
    if not re.fullmatch('[0-9a-f]{40}', source) or not re.fullmatch('[A-Za-z0-9_-]{1,80}', run):
        raise ValueError('full source SHA and safe fresh run ID required')
    folder = f'\\$HOME/dsv41-decode/{source}'
    mounts = [(folder + '/capture/speculator_replay.py', BASE + '/dspark/speculator.py', 'ro'),
              (folder + '/capture/decode_capture.py', BASE + '/dspark/decode_capture.py', 'ro'),
              (folder + '/spark/remote_dspark/speculator.py', BASE + '/remote_dspark/speculator.py', 'ro'),
              (f'\\$HOME/dsv41-offload-run/d2-decode-{run}', '/d2-decode', 'rw')]
    args = ' '.join(f'-v {src}:{dst}:{mode}' for src, dst, mode in mounts)
    return ('\n# D2 reviewed decode-capture derivative; original K1 profile unchanged.\n'
            'OFFLOAD_DOCKER_ENV="$OFFLOAD_DOCKER_ENV VLLM_DSPARK_DECODE_CAPTURE_DIR=/d2-decode"\n'
            f'OFFLOAD_DOCKER_ARGS="$OFFLOAD_DOCKER_ARGS {args}"\n')


def render(base: bytes, source: str, run: str) -> bytes:
    if hashlib.sha256(base).hexdigest() != BASE_ENV:
        raise ValueError('dsv41.env differs from reviewed sparkhead main/tag')
    return base + addition(source, run).encode()


def apply(root: Path, source: str, run: str, review: Path) -> None:
    before = (root / 'dsv41.env').read_bytes()
    after = render(before, source, run)
    record = obj(root / 'logs/k1-promotion/files.json')
    wanted = {'dsv41.env': BASE_ENV, 'start.sh': BASE_START}
    if (record.get('profile') != '327680' or record.get('promoted') != wanted
            or digest(root / 'start.sh') != BASE_START or digest(root / 'offload/kvpack/MANIFEST.json') != K1):
        raise ValueError('K1-327680 base approval/files/manifest mismatch')
    r = obj(review)
    target = hashlib.sha256(after).hexdigest()
    if (r.get('schema') != 'd2.decode-apply-review.v1' or r.get('approved') is not True
            or r.get('coordinator') != 'Deepseek/coordinator' or r.get('source') != source
            or r.get('base_commit') != BASE_COMMIT or r.get('base_env_sha256') != BASE_ENV
            or r.get('capture_env_sha256') != target):
        raise ValueError('missing SHA-bound coordinator review of the rendered diff')
    state = root / 'logs' / ('d2-decode-' + run)
    state.mkdir(mode=0o700, parents=True, exist_ok=False)
    shutil.copy2(root / 'dsv41.env', state / 'dsv41.env')
    shutil.copy2(root / 'start.sh', state / 'start.sh')
    shutil.copy2(root / 'logs/k1-promotion/files.json', state / 'k1-files.json')
    shutil.copy2(review, state / 'review.json')
    record['promoted'] = {'dsv41.env': target, 'start.sh': BASE_START}
    record['d2_decode_derivative'] = {'schema': 'd2.k1-derivative.v1', 'source': source,
        'base_commit': BASE_COMMIT, 'base_env_sha256': BASE_ENV, 'review_sha256': digest(review),
        'base_record_sha256': digest(state / 'k1-files.json')}
    # A partial apply fails K1's hash gate; rollback also handles either half written.
    (state / 'transaction.json').write_text(json.dumps({'source': source, 'run': run, 'env_sha256': target,
        'base_record_sha256': digest(state / 'k1-files.json'),
        'record_sha256': hashlib.sha256(json.dumps(record, indent=2).encode()).hexdigest()}))
    (root / 'dsv41.env').write_bytes(after)
    (root / 'logs/k1-promotion/files.json').write_text(json.dumps(record, indent=2))


def rollback(root: Path, run: str, *, after_k1_fallback: bool = False) -> None:
    if not re.fullmatch('[A-Za-z0-9_-]{1,80}', run):
        raise ValueError('unsafe run ID')
    state = root / 'logs' / ('d2-decode-' + run)
    txn = obj(state / 'transaction.json')
    config = root / 'dsv41.env'
    record = root / 'logs/k1-promotion/files.json'
    if (digest(state / 'k1-files.json') != txn['base_record_sha256']
            or digest(state / 'dsv41.env') != BASE_ENV or digest(state / 'start.sh') != BASE_START
            or obj(state / 'k1-files.json').get('promoted') != {'dsv41.env': BASE_ENV, 'start.sh': BASE_START}):
        raise ValueError('base rollback files corrupted')
    if after_k1_fallback:
        # The unchanged K1 ERR trap restores both pre-K1 files but leaves its
        # derivative receipt. Prove that exact state before overwriting anything.
        original = mapping(obj(state / 'k1-files.json').get('original'))
        if (digest(config) != original.get('dsv41.env')
                or digest(root / 'start.sh') != original.get('start.sh')
                or digest(record) != txn['record_sha256']):
            raise ValueError('not the exact K1 fallback state; refusing unrelated edits')
    elif digest(root / 'start.sh') != BASE_START:
        raise ValueError('base rollback files corrupted; start.sh differs')
    if not after_k1_fallback and digest(config) not in (BASE_ENV, txn['env_sha256']):
        raise ValueError('dsv41.env changed after apply; refusing to overwrite unrelated edits')
    if digest(record) not in (digest(state / 'k1-files.json'), txn['record_sha256']):
        raise ValueError('K1 receipt changed after apply; refusing rollback overwrite')
    shutil.copy2(state / 'start.sh', root / 'start.sh')
    shutil.copy2(state / 'dsv41.env', config)
    shutil.copy2(state / 'k1-files.json', record)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--source', default='')
    ap.add_argument('--run', required=True)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--apply', action='store_true')
    mode.add_argument('--rollback', action='store_true')
    mode.add_argument('--rollback-after-k1-fallback', action='store_true')
    ap.add_argument('--review', type=Path)
    a = ap.parse_args()
    if a.rollback or a.rollback_after_k1_fallback:
        rollback(a.root, a.run, after_k1_fallback=a.rollback_after_k1_fallback)
    elif a.apply:
        if a.review is None:
            ap.error('--apply requires --review')
        apply(a.root, a.source, a.run, a.review)
    else:
        before = (a.root / 'dsv41.env').read_bytes()
        after = render(before, a.source, a.run)
        print(''.join(difflib.unified_diff(before.decode().splitlines(True), after.decode().splitlines(True),
                                         fromfile='a/dsv41.env', tofile='b/dsv41.env')), end='')
        print('# capture_env_sha256=' + hashlib.sha256(after).hexdigest())


if __name__ == '__main__':
    main()
