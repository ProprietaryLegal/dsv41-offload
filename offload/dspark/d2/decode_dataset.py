"""CPU assembly of option-A generation taps; no network or inference calls.

Prepare a NEW decode plan from a top-20 replay tree and a frozen, durably flushed
copy of capture files. The replay taps/equivalence gate are deliberately unused.
Generation-request top20 can be supplied in the same normalized array layout,
but requires an image-bound smoke plus a frozen completions-validation receipt. No relabeling of
prefill taps is possible: every capture header/ledger/range/hash/id is checked.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
from typing import cast

import numpy as np
import torch

from . import shards as S
from .assembly_paths import AssemblyPaths, parse_mappings
from .replay import ROW_BYTES, load_seq, read_plan
from .sequence_audit import sequence_audit
from .window_checks import response_entry

SCHEMA = 'd2.decode-capture.v1'
PLAN = 'd2.decode-plan.v1'


def assembly_paths(meta: Mapping[str, object]) -> AssemblyPaths | None:
    relocation = meta.get('relocation')
    if relocation is None:
        return None
    r = mapping(relocation)
    maps = mapping(r.get('mappings'))
    return AssemblyPaths(Path(text(r.get('root'))), {k: text(v) for k, v in maps.items()})


def obj(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('expected JSON metadata object: ' + str(path))
    return cast(dict[str, object], value)


def digest(path: Path) -> str:
    with path.open('rb') as fh:
        return hashlib.file_digest(fh, 'sha256').hexdigest()


def number(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError('invalid nonnegative capture integer')
    return value


def text(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError('missing capture identity')
    return value


def mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError('missing metadata object')
    return cast(Mapping[str, object], value)


def named_error_policy(root: Path, enabled: bool) -> dict[str, object]:
    """Global errors/count drift are never eligible for quarantine."""
    path = root / 'ERRORS.jsonl'
    errors = [mapping(json.loads(line)) for line in path.read_text().splitlines()] if path.exists() else []
    ids: set[str] = set()
    for error in errors:
        seqs = error.get('seq_ids')
        if (error.get('schema') != SCHEMA or not isinstance(seqs, list) or not seqs
                or any(not isinstance(s, str) or re.fullmatch('[A-Za-z0-9_-]+', s) is None for s in seqs)):
            raise ValueError('unattributed capture error; quarantine refused')
        ids.update(cast(list[str], seqs))
    flushed, request = obj(root / 'FLUSHED.json'), obj(root / 'FLUSH.json')
    count = obj(root / 'ERROR_COUNT.json')
    if (flushed.get('schema') != SCHEMA or number(flushed.get('errors')) != len(errors)
            or not request.get('nonce') or request['nonce'] != flushed.get('nonce')
            or count.get('schema') != 'd2.decode-error-count.v1'
            or number(count.get('errors')) != len(errors) or count.get('global_refusal') is not False):
        raise ValueError('capture flush/error-count mismatch or global refusal')
    if ids and not enabled:
        raise ValueError('named capture errors require --quarantine-named-errors')
    return {'enabled': enabled, 'seq_ids': sorted(ids),
            'errors_sha256': digest(path) if path.exists() else None}


def capture_rows(root: Path, sid: str, r0: int, end: int,
                 tokens: Sequence[int]) -> tuple[torch.Tensor, dict[str, object]]:
    """Coverage r0..L_gen-2; extra aborted/checkpoint suffix rows are verified but excluded."""
    if not re.fullmatch('[A-Za-z0-9_-]+', sid):
        raise ValueError('unsafe sequence identity')
    errors_path = root / 'ERRORS.jsonl'
    errors = [mapping(json.loads(line)) for line in errors_path.read_text().splitlines()] if errors_path.exists() else []
    for error in errors:
        seqs = error.get('seq_ids')
        if (error.get('schema') != SCHEMA or not isinstance(seqs, list) or not seqs
                or any(not isinstance(s, str) or re.fullmatch('[A-Za-z0-9_-]+', s) is None for s in seqs)
                or sid in seqs):
            raise ValueError('capture writer recorded errors for this sequence or globally; data refused')
    flushed = obj(root / 'FLUSHED.json')
    request = obj(root / 'FLUSH.json')
    if (flushed.get('schema') != SCHEMA or number(flushed.get('errors')) != len(errors)
            or not request.get('nonce') or flushed.get('nonce') != request['nonce']):
        raise ValueError('missing/mismatched durable flush receipt')
    f = mapping(mapping(flushed.get('sequences')).get(sid))
    d = root / sid
    head = obj(d / 'CAPTURE.json')
    if (head.get('schema') != SCHEMA or head.get('taps_source') != 'decode_capture'
            or head.get('seq') != sid or head.get('from') != r0 or head.get('row_bytes') != ROW_BYTES
            or head.get('capture_sha256') != digest(Path(__file__).parents[1] / 'capture/decode_capture.py')):
        raise ValueError('capture mode/origin/sequence header mismatch')
    lp, tp, ip = (d / x for x in ('ledger.jsonl', 'taps.bin', 'ids.bin'))
    if (digest(lp) != f.get('ledger_sha256') or tp.stat().st_size != f.get('taps_bytes')
            or ip.stat().st_size != f.get('ids_bytes')):
        raise ValueError('capture changed after durable flush')
    covered: set[int] = set()
    # Replay ownership before inspecting disk: overwritten bytes can no longer
    # satisfy an older stage's aggregate hash. Every current row must satisfy
    # its newest actual writer's row hash; retained duplicates agree with that
    # writer, and only an explicitly named older segment may be superseded.
    writers: dict[int, tuple[int, str, str]] = {}
    dup = differing = zero = replacements = 0
    ledger = [cast(dict[str, object], json.loads(line)) for line in lp.read_text().splitlines()]
    with tp.open('rb') as taps, ip.open('rb') as ids:
        for row in ledger:
            first, n = number(row.get('first')), number(row.get('rows'))
            if (row.get('schema') != SCHEMA or row.get('seq') != sid or first < r0 or n < 1
                    or number(row.get('tag_r0')) < r0 or first < number(row.get('tag_r0'))):
                raise ValueError('invalid capture ledger identity/range')
            seg = number(row.get('seg'))
            prompt = number(row.get('prompt_length'))
            if max(0, prompt - 128) != number(row.get('tag_r0')):
                raise ValueError('capture ledger prompt length mismatch')
            number(row.get('round'))
            ht, hi = row.get('row_sha256_taps'), row.get('row_sha256_ids')
            if not isinstance(ht, list) or not isinstance(hi, list) or len(ht) != n or len(hi) != n:
                raise ValueError('missing capture per-row hashes')
            replaced = row.get('replaced_rows')
            if not isinstance(replaced, list):
                raise ValueError('missing capture replacement ledger')
            replace: dict[int, int] = {}
            for raw in cast(list[object], replaced):
                claim = mapping(raw)
                p, old_seg = number(claim.get('position')), number(claim.get('seg'))
                if p in replace or not first <= p < first + n:
                    raise ValueError('invalid capture replacement position')
                replace[p] = old_seg
            actual_dup = 0
            for j, p in enumerate(range(first, first + n)):
                th, ih = text(ht[j]), text(hi[j])
                if not re.fullmatch('[0-9a-f]{64}', th) or not re.fullmatch('[0-9a-f]{64}', ih):
                    raise ValueError('invalid capture per-row hash')
                previous = writers.get(p)
                if previous is not None:
                    actual_dup += 1
                    if p in replace:
                        if p < prompt or previous[0] != replace[p] or previous[0] >= seg:
                            raise ValueError('capture superseded writer is not an older segment suffix')
                    else:
                        if (th, ih) != previous[1:]:
                            raise ValueError('capture retained duplicate differs from its writer')
                        continue
                elif p in replace:
                    raise ValueError('capture replacement without an older writer')
                writers[p] = (seg, th, ih)
            if number(row.get('dup_rows')) != actual_dup:
                raise ValueError('capture ledger duplicate count mismatch')
            covered.update(range(first, first + n))
            dup += actual_dup
            nd = number(row.get('dup_rows_differing'))
            if nd > actual_dup:
                raise ValueError('invalid differing duplicate count')
            differing += nd
            zero += number(row.get('zero_rows'))
            replacements += len(replace)
        # Verify each final row against the newest writer, including any suffix
        # beyond the driver's final checkpoint (it is verified, then excluded).
        for p, (_, th, ih) in writers.items():
            taps.seek((p - r0) * ROW_BYTES)
            ids.seek((p - r0) * 4)
            tb, ib = taps.read(ROW_BYTES), ids.read(4)
            if len(tb) != ROW_BYTES or len(ib) != 4 or hashlib.sha256(tb).hexdigest() != th or hashlib.sha256(ib).hexdigest() != ih:
                raise ValueError('capture newest-writer bytes/hash mismatch')
        for row in ledger:
            first, n = number(row['first']), number(row['rows'])
            ht, hi = cast(list[str], row['row_sha256_taps']), cast(list[str], row['row_sha256_ids'])
            # Aggregate hashes remain mandatory wherever no later write changed
            # this stage's retained bytes. Superseded claims are checked above.
            if all(writers[p][1:] == (ht[j], hi[j]) for j, p in enumerate(range(first, first + n))):
                taps.seek((first - r0) * ROW_BYTES)
                ids.seek((first - r0) * 4)
                if (hashlib.sha256(taps.read(n * ROW_BYTES)).hexdigest() != row.get('sha256_taps')
                        or hashlib.sha256(ids.read(n * 4)).hexdigest() != row.get('sha256_ids')):
                    raise ValueError('capture ledger bytes/hash mismatch')
        if zero:
            raise ValueError('zero taps rows in capture ledger')
        if not set(range(r0, end)).issubset(covered):
            raise ValueError('decode capture coverage gap (including prompt tail)')
        ids.seek(0)
        actual_ids = np.frombuffer(ids.read((end - r0) * 4), dtype='<i4').tolist()
        if actual_ids != list(tokens[r0:end]):
            raise ValueError('decode ids.bin differs from final committed tokens')
        taps.seek(0)
        raw = bytearray(taps.read((end - r0) * ROW_BYTES))
    tensor = torch.frombuffer(raw, dtype=torch.bfloat16).view(end - r0, S.TAPW)
    if bool((tensor.view(-1, 3, 5120) == 0).all(-1).any()):
        raise ValueError('zero layer row in capture bytes')
    return tensor, {'schema': SCHEMA, 'from': r0, 'end': end, 'rows': end - r0,
                    'ledger_sha256': digest(lp), 'header_sha256': digest(d / 'CAPTURE.json'),
                    'flush_sha256': digest(root / 'FLUSHED.json'), 'capture_sha256': head.get('capture_sha256'),
                    'dup_rows': dup, 'dup_rows_differing': differing, 'zero_rows': zero, 'replaced_rows': replacements,
                    'extra_rows_excluded': len(covered - set(range(r0, end)))}


def decode_identity(root: Path, *, full_copy: bool = False) -> dict[str, object]:
    meta = obj(root / 'DECODE.json')
    if meta.get('schema') != PLAN or meta.get('taps_source') != 'decode_capture':
        raise ValueError('not a decode capture plan')
    paths = assembly_paths(meta)
    if paths:
        for path in (root / 'DECODE.json', root / 'plan.jsonl'):
            paths.inside(path)
        for entry in read_plan(str(root)):
            paths.inside(Path(text(entry.get('src'))))
        from .copy_binding import verify_copy
        receipt = paths.inside(paths.root / 'COPY_RECEIPT.json')
        if digest(receipt) != meta.get('copy_receipt_sha256'):
            raise ValueError('decode copy receipt changed')
        if full_copy and verify_copy(paths.root) != meta['copy_receipt_sha256']:
            raise ValueError('decode copy receipt changed during validation')
        for key in ('captures', 'top20_root', 'verification'):
            paths.inside(Path(text(meta[key])))
        if Path(text(meta['captures'])) != paths.root / 'capture':
            raise ValueError('captures are not the receipt-bound copy tree')
        paths.inside(root)
        if meta.get('quarantine') is not None:
            quarantine = mapping(meta['quarantine'])
            if named_error_policy(Path(text(meta['captures'])), quarantine.get('enabled') is True) != quarantine:
                raise ValueError('named capture error quarantine changed')
            if any(e['seq_id'] in cast(list[str], quarantine['seq_ids']) for e in read_plan(str(root))):
                raise ValueError('quarantined sequence present in decode plan')
        if digest(root / 'plan.jsonl') != meta.get('plan_sha256'):
            raise ValueError('relocated decode plan changed')
    source = meta.get('top20_source')
    if source == 'generation_verified':
        path = Path(text(meta.get('verification')))
        if digest(path) != meta.get('verification_sha256'):
            raise ValueError('generation top20 requires a successful real image-bound smoke and completions receipt')
        from .generation_export import verify
        verify(Path(text(meta['top20_root'])), path, paths=paths)
    elif source != 'prefill_replay_top20_only':
        raise ValueError('unverified top20 source')
    return meta


def build_record(entry: Mapping[str, object], root: str, pman: Mapping[str, object],
                 context: Mapping[str, object] | None) -> S.SeqRecord:
    meta = decode_identity(Path(root))
    sid, src = text(entry['seq_id']), text(entry['src'])
    if meta['top20_source'] == 'generation_verified':
        from .generation_export import verify_sequence
        verify_sequence(Path(text(meta['top20_root'])), entry, obj(Path(text(meta['verification']))),
                        paths=assembly_paths(meta))
    state = load_seq(src)
    toks = cast(list[int], state['tokens'])
    Lgen, g, r0 = number(entry['L']), number(entry['gen_start']), number(entry['from'])
    if (len(toks) != Lgen or S.tokens_sha256(toks) != entry['tokens_sha256']
            or state['gen_start'] != g or r0 != max(0, g - 128) or Lgen - g < 3):
        raise ValueError('decode source tokens/L/gen_start differ from plan or span too short')
    L = Lgen - 1
    taps, capture = capture_rows(Path(text(meta['captures'])), sid, r0, L, toks)
    lp_root = Path(text(meta['top20_root']))
    lm = obj(lp_root / 'logprobs' / (sid + '.json'))
    old_entry = dict(entry)
    old_entry['request_id'] = entry['logprob_request_id']
    if any(key not in lm for key in ('seq_id', 'request_id', 'L', 'gen_start', 'from', 'tokens_sha256')):
        raise ValueError('top20 response identity missing')
    response_entry(lm, old_entry)
    zp = lp_root / 'logprobs' / (sid + '.npz')
    if digest(zp) != entry['top20_sha256']:
        raise ValueError('top20 arrays changed since decode plan')
    arrays: dict[str, torch.Tensor] = {}
    with np.load(zp) as z:
        for key in ('top_ids', 'top_lp', 'tok_lp', 'tok_rank', 'resid'):
            if z[key].shape[0] != Lgen - g:
                raise ValueError('top20 span is incomplete')
            arrays[key] = torch.from_numpy(z[key][:-1].copy())
    pm = mapping(pman.get(text(entry['prompt_sha256']), {}))
    side: dict[str, object] = {key: entry[key] for key in ('seq_id', 'split', 'family', 'prompt_sha256',
                                                        'finish_reason', 'segments')}
    side.update(source=pm.get('source'), source_docs=pm.get('source_docs', []),
                generation_L=Lgen, generation_tokens_sha256=entry['tokens_sha256'],
                taps_source='decode_capture', decode_capture=capture, top20_source=meta['top20_source'],
                top20_sha256=entry['top20_sha256'], logprob_request_id=entry['logprob_request_id'])
    if context is not None:
        # Audit the FULL raw generation to preserve segment coverage, then record the shard truncation explicitly.
        audit = sequence_audit(toks, g, context, obj(Path(src)), text(entry['kind']))
        audit['shard_interval'] = [0, L]
        audit['dropped_final_sample'] = True
        side['sequence_audit'] = audit
    else:
        side['sequence_audit'] = {'schema': 'd2.sequence-audit.synthetic.v1', 'not_served_data': True}
    return S.SeqRecord(side, torch.tensor(toks[:L], dtype=torch.int32), g, taps,
                       arrays['top_ids'], arrays['top_lp'], arrays['tok_lp'], arrays['tok_rank'], arrays['resid'])


def prepare(top20: Path, captures: Path, out: Path, source: str, verification: Path | None = None, *,
            assembly_root: Path | None = None, path_maps: Mapping[str, str] | None = None,
            quarantine_named_errors: bool = False) -> None:
    if out.exists():
        raise ValueError('decode plan requires a NEW output directory')
    paths = AssemblyPaths(assembly_root, path_maps or {}) if assembly_root else None
    if path_maps and paths is None:
        raise ValueError('path maps require an explicit assembly root')
    if paths:
        from .copy_binding import verify_copy
        copy_sha = verify_copy(paths.root)
        top20, captures = paths.inside(top20), paths.inside(captures)
        if captures != paths.root / 'capture':
            raise ValueError('captures are not the receipt-bound copy tree')
        # out does not exist yet; check its existing parent instead.
        paths.inside(out.parent)
        if verification is not None:
            verification = paths.inside(verification)
    if source == 'generation_verified':
        from .generation_export import verify
        if verification is None:
            raise ValueError('generation top20 requires a completions validation receipt')
        verify(top20, verification, full=True, paths=paths)
    quarantine = (named_error_policy(captures, quarantine_named_errors)
                  if paths or quarantine_named_errors else None)
    entries = read_plan(str(top20))
    if not entries:
        raise ValueError('empty top20 plan')
    planned: list[dict[str, object]] = []
    for e in entries:
        sid, g = text(e['seq_id']), number(e['gen_start'])
        if not re.fullmatch('[A-Za-z0-9_-]+', sid):
            raise ValueError('unsafe sequence identity')
        if e['kind'] != 'gen':
            raise ValueError('original acap requires its existing decode exports; cannot relabel replay taps')
        if quarantine and sid in cast(list[str], quarantine['seq_ids']):
            continue
        head = obj(captures / sid / 'CAPTURE.json')
        if head.get('schema') != SCHEMA or head.get('taps_source') != 'decode_capture':
            raise ValueError('non-decode taps refused')
        row = dict(e)
        if paths:
            row['src'] = str(paths.resolve(text(e['src'])))
            copied_state = paths.inside(paths.root / 'generator/seqs' / (sid + '.json'))
            if obj(Path(text(row['src']))) != obj(copied_state):
                raise ValueError('export sequence differs from receipt-bound generator state')
        row.update(logprob_request_id=e['request_id'], request_id=f'd2c.{sid}.0.{max(0, g - 128)}.{g}',
                   top20_sha256=digest(top20 / 'logprobs' / (sid + '.npz')))
        planned.append(row)
    meta: dict[str, object] = {'schema': PLAN, 'taps_source': 'decode_capture', 'captures': str(captures.resolve()),
            'top20_root': str(top20.resolve()), 'top20_source': source,
            'verification': str(verification.resolve()) if verification else None,
            'verification_sha256': digest(verification) if verification else None}
    if not planned:
        raise ValueError('no sequences remain after named-error quarantine')
    if quarantine is not None:
        meta['quarantine'] = quarantine
    plan_text = ''.join(json.dumps(e) + '\n' for e in planned)
    if paths:
        meta['copy_receipt_sha256'] = copy_sha
        meta['relocation'] = {'root': str(paths.root), 'mappings': paths.mappings,
                              'top20_plan_sha256': digest(top20 / 'plan.jsonl'),
                              'policy': 'resolve only; original bytes and sha256 fields unchanged'}
        meta['plan_sha256'] = hashlib.sha256(plan_text.encode()).hexdigest()
    out.mkdir(mode=0o700, parents=True)
    (out / 'DECODE.json').write_text(json.dumps(meta))
    (out / 'plan.jsonl').write_text(plan_text)
    for path in (out / 'DECODE.json', out / 'plan.jsonl'):
        os.chmod(path, 0o600)
    decode_identity(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--top20', type=Path, required=True)
    ap.add_argument('--captures', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--top20-source', choices=('prefill_replay_top20_only', 'generation_verified'),
                    default='prefill_replay_top20_only')
    ap.add_argument('--verification', type=Path)
    ap.add_argument('--quarantine-named-errors', action='store_true')
    ap.add_argument('--assembly-root', type=Path)
    ap.add_argument('--path-map', action='append', default=[], metavar='OLD_ABSOLUTE=ROOT_RELATIVE')
    a = ap.parse_args()
    prepare(a.top20, a.captures, a.out, a.top20_source, a.verification,
            assembly_root=a.assembly_root, path_maps=parse_mappings(a.path_map), quarantine_named_errors=a.quarantine_named_errors)


if __name__ == '__main__':
    main()
