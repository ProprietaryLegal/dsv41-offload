"""Offline export of frozen generation state; never makes inference/network requests.

The online validator plus frozen rows establish the completions-path top20 proof.
A real image-bound decode smoke still establishes the image/capture provenance;
its chat result alone cannot certify the different completions formatter.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from . import generation_top20 as T
from .assembly_paths import AssemblyPaths
from .replay import load_seq, read_plan, tokens_sha256

SCHEMA = 'd2.generation-top20-export.v1'


def digest(path: Path) -> str:
    with path.open('rb') as fh:
        return hashlib.file_digest(fh, 'sha256').hexdigest()


def obj(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError('expected metadata object: ' + str(path))
    return cast(dict[str, object], value)


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value))
    path.chmod(0o600)


def smoke(path: Path) -> None:
    check = obj(path)
    capture = Path(__file__).parents[1] / 'capture/decode_capture.py'
    if (check.get('schema') != 'd2.decode-smoke.v1' or check.get('synthetic') is not False
            or check.get('pass') is not True or check.get('option_ii') is not True
            or check.get('image') != T.IMAGE or check.get('capture_sha256') != digest(capture)):
        raise ValueError('generation top20 requires a successful real image-bound smoke')


def saved_arrays(state: Mapping[str, object]) -> dict[str, NDArray[np.int32] | NDArray[np.float32]]:
    if state.get('top20') != 'generation':
        raise ValueError('not a generation-top20 state')
    ids = cast(list[int], state['gen_ids'])
    rows = cast(list[T.Row], state['logprob_rows'])
    T.validate_saved(ids, rows)
    return {key: np.asarray([r[key] for r in rows], dtype=np.int32 if key in ('top_ids', 'tok_rank') else np.float32)
            for key in ('top_ids', 'top_lp', 'tok_lp', 'tok_rank', 'resid')}


def export(state: Path, out: Path, verification: Path) -> None:
    """Require the driver's exclusive lock, copy snapshots, then export. Publish receipt last."""
    if out.exists():
        raise ValueError('generation export requires a NEW output directory')
    smoke(verification)
    # r+ opens only an existing driver lock, and never modifies source state.
    with (state / 'LOCK').open('r+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('stop generator before frozen export: driver still holds LOCK') from exc
        out.mkdir(parents=True, mode=0o700)
        for sub in ('seqs', 'logprobs'):
            (out / sub).mkdir(mode=0o700)
        entries: list[dict[str, object]] = []
        receipts: dict[str, object] = {}
        for path in sorted((state / 'seqs').glob('*.json')):
            raw = obj(path)
            if raw.get('status') == 'running':
                raise ValueError('unreaped running state: restart and stop generator cleanly before export')
            seq = load_seq(str(path))
            if not seq:
                continue
            if len(seq['tokens']) - seq['gen_start'] < 3:
                continue
            sid = seq['seq_id']
            if not isinstance(sid, str) or not re.fullmatch('[A-Za-z0-9_-]+', sid):
                raise ValueError('unsafe sequence identity')
            arrays = saved_arrays(raw)
            # Bind the plan to immutable private copies, never a resumable live state.
            frozen = out / 'seqs' / (sid + '.json')
            write(frozen, raw)
            L, g = len(seq['tokens']), seq['gen_start']
            entry = {k: seq[k] for k in ('seq_id', 'split', 'family', 'kind', 'prompt_sha256', 'finish_reason', 'segments')}
            entry.update(src=str(frozen.resolve()), L=L, gen_start=g, **{'from': max(0, g - 128)},
                         tokens_sha256=tokens_sha256(seq['tokens']), request_id=f'd2g.top20.{sid}.{L}')
            zp = out / 'logprobs' / (sid + '.npz')
            np.savez(zp, top_ids=arrays['top_ids'], top_lp=arrays['top_lp'], tok_lp=arrays['tok_lp'],
                     tok_rank=arrays['tok_rank'], resid=arrays['resid'])
            zp.chmod(0o600)
            write(out / 'logprobs' / (sid + '.json'), entry | {
                'schema': T.SCHEMA, 'source': 'generation_verified', 'rank_semantics': 'top20_position_else_21',
                'rows': L - g, 'state_sha256': digest(frozen)})
            entries.append(entry)
            receipts[sid] = {'state_sha256': digest(frozen), 'npz_sha256': digest(zp), 'rows': L - g,
                             'metadata_sha256': digest(out / 'logprobs' / (sid + '.json'))}
        if not entries:
            raise ValueError('no exportable generation sequences')
        plan = out / 'plan.jsonl'
        plan.write_text(''.join(json.dumps(e) + '\n' for e in entries))
        plan.chmod(0o600)
        write(out / 'VERIFIED.json', {
            'schema': SCHEMA, 'pass': True, 'endpoint': '/v1/completions', 'image': T.IMAGE,
            'logprobs_mode': 'raw_logprobs', 'rank_semantics': 'top20_position_else_21',
            'validator_sha256': digest(Path(T.__file__)), 'plan_sha256': digest(plan),
            'smoke': str(verification.resolve()), 'smoke_sha256': digest(verification),
            'sequences': receipts, 'rows': sum(cast(int, cast(dict[str, object], r)['rows']) for r in receipts.values())})
    verify(out, out / 'VERIFIED.json', full=True)


def verify(root: Path, receipt: Path, *, full: bool = False,
           paths: AssemblyPaths | None = None) -> dict[str, object]:
    if paths:
        for path in (root, receipt, root / 'plan.jsonl'):
            paths.inside(path)
    check = obj(receipt)
    if (check.get('schema') != SCHEMA or check.get('pass') is not True
            or check.get('endpoint') != '/v1/completions' or check.get('image') != T.IMAGE
            or check.get('logprobs_mode') != 'raw_logprobs'
            or check.get('rank_semantics') != 'top20_position_else_21'
            or check.get('validator_sha256') != digest(Path(T.__file__))
            or check.get('plan_sha256') != digest(root / 'plan.jsonl')):
        raise ValueError('generation top20 requires a successful real image-bound smoke and completions validation receipt')
    smoke_path = paths.resolve(str(check['smoke'])) if paths else Path(str(check['smoke']))
    if digest(smoke_path) != check.get('smoke_sha256'):
        raise ValueError('generation image smoke changed')
    smoke(smoke_path)
    seqs = check.get('sequences')
    entries = read_plan(str(root))
    if not isinstance(seqs, dict) or not entries or set(seqs) != {e['seq_id'] for e in entries}:
        raise ValueError('generation receipt sequence coverage mismatch')
    if full:
        for e in entries:
            verify_sequence(root, e, check, paths=paths)
    return check


def verify_sequence(root: Path, entry: Mapping[str, object], check: Mapping[str, object], *,
                    paths: AssemblyPaths | None = None) -> None:
    sid = str(entry['seq_id'])
    if not re.fullmatch('[A-Za-z0-9_-]+', sid):
        raise ValueError('unsafe sequence identity')
    if paths:
        for path in (root / 'logprobs' / (sid + '.npz'), root / 'logprobs' / (sid + '.json')):
            paths.inside(path)
    seqs = cast(dict[str, object], check['sequences'])
    r = seqs[sid]
    if (not isinstance(r, dict) or r.get('rows') != cast(int, entry['L']) - cast(int, entry['gen_start'])
            or r.get('state_sha256') != digest(paths.resolve(str(entry['src'])) if paths else Path(str(entry['src'])))
            or r.get('npz_sha256') != digest(root / 'logprobs' / (sid + '.npz'))
            or r.get('metadata_sha256') != digest(root / 'logprobs' / (sid + '.json'))):
        raise ValueError('generation export changed after online-validation receipt')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--state', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--image-smoke', type=Path, required=True)
    a = ap.parse_args()
    export(a.state, a.out, a.image_smoke)


if __name__ == '__main__':
    main()
