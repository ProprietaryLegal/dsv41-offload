"""Offline draft attestation from a pinned evidence map and reviewed resolutions.

Never discovers hosts/files, imports pinned generator/renderer code, or contacts a
service. JSON-pointer bindings verify values; semantic truth remains the named
coordinator's explicit review responsibility. Output is never deployment approval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import cast

from . import pass2_context as context
from .copy_binding import digest

COORDINATOR = 'Deepseek/coordinator'
RUN = 'd2pass2-r1'
TARGET = ('model_root', 'quantization', 'engine_revision',
          'weights_manifest_sha256', 'quantization_manifest_sha256')
FIELDS: dict[str, tuple[str, ...]] = {k: (k,) for k in ('schema', 'coordinator', 'run', 'evidence')}
for group, keys in (
    ('target', TARGET),
    ('decode_capture_target', (*TARGET, 'run', 'review_sha256', 'capture_env_sha256')),
    ('sampling', ('temperature', 'top_p', 'top_k', 'min_p', 'structured_outputs', 'stop', 'stop_token_ids', 'seed')),
    ('formatting', ('chat_template_kwargs', 'tools', 'tool_choice')),
    ('generator', ('revision', 'priority', 'top20')),
):
    for key in keys:
        FIELDS[group + '.' + key] = (group, key)
for filename in context.GEN_FILES:
    FIELDS['generator.source_sha256.' + filename] = ('generator', 'source_sha256', filename)
for label in ('tokenizer', 'encoder', 'chat_template'):
    for key in ('path', 'sha256'):
        FIELDS['identity.' + label + '.' + key] = ('identity', label, key)
OBLIGATIONS = ('verification.served_identity', 'verification.effective_defaults', 'verification.renderer_raw_ids')
# NDM 2026-10-25 ("Sign them"): these configuration fields may carry the named coordinator's signature from the
# configuration of record instead of a measurement. Integrity fields (target and derivative hashes, identity files,
# evidence, generator source, run identity) never can; every other proof rule still applies to a signed claim.
SIGNED = 'COORDINATOR_ATTESTED'
SIGNABLE = frozenset({'target.engine_revision', 'decode_capture_target.engine_revision',
                      'sampling.top_k', 'sampling.min_p', 'sampling.stop_token_ids', 'sampling.structured_outputs',
                      'formatting.chat_template_kwargs', 'formatting.tools', 'formatting.tool_choice', *OBLIGATIONS})


def obj(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError(label + ' must be an object')
    return cast(dict[str, object], value)


def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def parse(text: str) -> dict[str, object]:
    return obj(json.loads(text, object_pairs_hook=no_duplicates,
                          parse_constant=lambda v: (_ for _ in ()).throw(ValueError('nonfinite JSON: ' + v))), 'JSON')


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def bound(path: Path, sha: object) -> bytes:
    if not isinstance(sha, str) or re.fullmatch('[0-9a-f]{64}', sha) is None:
        raise ValueError('source SHA256 required')
    context.pinned(path, sha)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha:
        raise ValueError('source changed while reading')
    return data


def pointer(data: object, pointer_value: object) -> object:
    if not isinstance(pointer_value, str) or not pointer_value.startswith('/'):
        raise ValueError('source value requires a JSON pointer')
    for token in pointer_value[1:].split('/'):
        token = token.replace('~1', '/').replace('~0', '~')
        if isinstance(data, list):
            if not token.isdigit():
                raise ValueError('invalid array pointer')
            data = data[int(token)]
        else:
            data = obj(data, 'pointer object')[token]
    return data


def prove(name: str, claim: object) -> object:
    row = obj(claim, name)
    if row.get('status') == SIGNED and name not in SIGNABLE:
        raise ValueError(name + ': integrity field cannot be coordinator-attested')
    if row.get('status') not in ('MEASURED', SIGNED) or 'value' not in row:
        raise ValueError(name + ': missing or inferred-only evidence')
    if not isinstance(row.get('scope'), str) or not str(row['scope']).strip():
        raise ValueError(name + ': explicit semantic review scope required')
    sources = row.get('sources')
    if not isinstance(sources, list) or len(sources) < 2:
        raise ValueError(name + ': value binding and primary supporting source required')
    bindings = 0
    seen: set[str] = set()
    for item in sources:
        source = obj(item, 'source')
        path = Path(str(source.get('path', '')))
        if str(path) in seen:
            raise ValueError('duplicate source path')
        seen.add(str(path))
        data = bound(path, source.get('sha256'))
        if 'pointer' in source:
            value = pointer(parse(data.decode()), source['pointer'])
            if canonical(value) != canonical(row['value']):
                raise ValueError(name + ': contradictory source value')
            bindings += 1
    if not bindings:
        raise ValueError(name + ': no primary/reviewed JSON value binding')
    return row['value']


def put(tree: dict[str, object], keys: tuple[str, ...], value: object) -> None:
    for key in keys[:-1]:
        if key not in tree:
            tree[key] = {}
        tree = obj(tree[key], key)
    tree[keys[-1]] = value


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def generate(evidence_map: Path, map_sha: str, resolution: Path, resolution_sha: str,
             manifest: Path, out: Path) -> dict[str, str]:
    """Create a NEW private draft only after all claims and context checks pass."""
    raw_map = bound(evidence_map, map_sha).decode()
    blocks = re.findall(r'^```json\n(.*?)\n```$', raw_map, re.M | re.S)
    if len(blocks) != 1:
        raise ValueError('one machine evidence map JSON block required')
    inventory = parse(blocks[0])
    if inventory.get('schema') != 'd2.pass2-evidence-map.v1':
        raise ValueError('evidence map schema differs')
    mapped = obj(inventory.get('claims'), 'map claims')
    reviewed = parse(bound(resolution, resolution_sha).decode())
    if (reviewed.get('schema') != 'd2.pass2-resolution.v1' or reviewed.get('approved') is not True
            or reviewed.get('coordinator') != COORDINATOR or reviewed.get('run') != RUN):
        raise ValueError('explicit coordinator-reviewed resolution required')
    claims = obj(reviewed.get('claims'), 'resolved claims')
    required = set(FIELDS) | set(OBLIGATIONS)
    if set(mapped) != required or set(claims) != required:
        raise ValueError('missing/extra fields in evidence map or resolution')
    attestation: dict[str, object] = {}
    for name in sorted(required):
        value = prove(name, claims[name])
        prior = obj(mapped[name], name)
        if prior.get('status') == 'MEASURED' and canonical(prior.get('value')) != canonical(value):
            raise ValueError(name + ': contradiction with measured evidence map')
        if name in OBLIGATIONS:
            if value is not True:
                raise ValueError(name + ': unresolved semantic obligation')
        else:
            put(attestation, FIELDS[name], value)
    if (attestation.get('schema') != 'd2.pass2-attestation.v1'
            or attestation.get('coordinator') != COORDINATOR or attestation.get('run') != RUN):
        raise ValueError('attestation identity differs')
    target = obj(attestation['target'], 'target')
    derivative = obj(attestation['decode_capture_target'], 'decode target')
    if any(canonical(target[k]) != canonical(derivative[k]) for k in TARGET):
        raise ValueError('contradictory target and derivative')
    evidence = attestation['evidence']
    if not isinstance(evidence, list) or {obj(r, 'evidence').get('label') for r in evidence} != set((*context.W12_FILES, *context.GEN_FILES)):
        raise ValueError('exact W12 and generator evidence required; target receipt is generated')
    if (not out.is_absolute() or '..' in out.parts or out.exists()
            or out.parent.resolve() != out.parent or not out.parent.is_dir()
            or 'W12' in out.parts or out.is_relative_to('/opt/dsv41/data/d2')):
        raise ValueError('NEW offline output directory outside production/W12 required')
    # Parse the manifest before creating any output; primary byte hash retained.
    manifest_bytes = manifest.read_bytes()
    parse(manifest_bytes.decode())
    os.umask(0o077)
    out.mkdir(mode=0o700)
    try:
        receipt = out / 'target-receipt.json'
        write_json(receipt, target)
        evidence.append({'label': 'target-receipt.json', 'path': str(receipt), 'sha256': digest(receipt)})
        attestation['draft_status'] = 'REQUIRES_COORDINATOR_REVIEW'
        attestation['claim_review'] = {'map_path': str(evidence_map), 'map_sha256': map_sha,
                                       'resolution_path': str(resolution), 'resolution_sha256': resolution_sha,
                                       'claims': claims}
        draft = out / 'pass2-attestation.json'
        write_json(draft, attestation)
        # Reuse all existing checks unchanged, with only local files and CPU tokenizer parsing.
        validation = out / '.validation'
        (validation / 'inputs').mkdir(parents=True, mode=0o700)
        pm = validation / 'inputs/prompt-manifest.json'
        pm.write_bytes(manifest_bytes)
        context.stage(validation, draft, digest(draft), pm)
        shutil.rmtree(validation)
        review_receipt = {'status': 'DRAFT_REQUIRES_COORDINATOR_REVIEW', 'map_sha256': map_sha,
                          'resolution_sha256': resolution_sha, 'prompt_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
                          'attestation_sha256': digest(draft), 'target_receipt_sha256': digest(receipt),
                          'resolution_path': str(resolution), 'evidence_map_path': str(evidence_map)}
        write_json(out / 'DRAFT.json', review_receipt)
        return {'attestation': str(draft), 'sha256': digest(draft), 'target_receipt': str(receipt)}
    except BaseException:
        # Only this invocation's new private output exists here.
        shutil.rmtree(out)
        raise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evidence-map', type=Path, required=True)
    ap.add_argument('--map-sha256', required=True)
    ap.add_argument('--resolution', type=Path, required=True)
    ap.add_argument('--resolution-sha256', required=True)
    ap.add_argument('--prompt-manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(generate(args.evidence_map, args.map_sha256, args.resolution,
                              args.resolution_sha256, args.prompt_manifest, args.out), indent=2))


if __name__ == '__main__':
    main()
