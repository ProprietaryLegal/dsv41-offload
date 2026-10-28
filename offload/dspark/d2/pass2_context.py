"""Offline, fail-closed pass-2 context staging from coordinator-pinned receipts.

No inference, remote access, code execution or live generator state reads. The
coordinator supplies an attestation for facts absent from the W12 deployment
record; this tool never invents those facts or upgrades an incomplete scaffold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .copy_binding import digest
from .sequence_audit import MARKERS, object_map, validate_context

W12_FILES = ('w12.log', 'base-hashes.txt', 'staged-hashes.txt', 'mounted-hashes.txt',
             'rendered-d2pass2-r1.diff', 'review-d2pass2-r1.json', 'start.log')
REVISION = '1b8fc1c'
GEN_FILES = ('generate.py', 'generation_top20.py', 'prompts.py')


def pinned(path: Path, sha: str) -> Path:
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('evidence requires an absolute normalized path')
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError('evidence symlink refused')
    if digest(path) != sha:
        raise ValueError('evidence hash differs: ' + str(path))
    return path


def stage(root: Path, attestation: Path, expected_sha: str, manifest: Path) -> dict[str, object]:
    """Stage NEW inputs/context-inputs only after validating every supplied claim.

    Attestation schema is documented in ASSEMBLE.md. All evidence sources must
    be explicitly named and hash-pinned; directory discovery is never used.
    """
    from tokenizers import Tokenizer
    from .assembly_paths import AssemblyPaths
    paths = AssemblyPaths(root, {})
    root = paths.root
    paths.inside(root / 'inputs')
    manifest = paths.inside(manifest)
    a = json.loads(pinned(attestation, expected_sha).read_text())
    if a.get('schema') != 'd2.pass2-attestation.v1' or a.get('run') != 'd2pass2-r1':
        raise ValueError('coordinator input required: d2.pass2-attestation.v1 for d2pass2-r1')
    if a.get('coordinator') != 'Deepseek/coordinator':
        raise ValueError('coordinator attestation identity differs')
    required = ('evidence', 'generator', 'target', 'decode_capture_target', 'sampling', 'formatting', 'identity')
    missing = [k for k in required if k not in a]
    if missing:
        raise ValueError('coordinator inputs required: ' + ', '.join(missing))
    evidence: dict[str, Path] = {}
    for row in a['evidence']:
        label = row['label']
        if label in evidence or label not in (*W12_FILES, *GEN_FILES, 'target-receipt.json'):
            raise ValueError('unexpected/duplicate evidence label')
        evidence[label] = pinned(Path(row['path']), row['sha256'])
    if set(evidence) != set((*W12_FILES, *GEN_FILES, 'target-receipt.json')):
        raise ValueError('coordinator input required: complete W12, generator and target evidence')
    review = json.loads(evidence['review-d2pass2-r1.json'].read_text())
    if (review.get('schema') != 'd2.decode-apply-review.v1' or review.get('approved') is not True
            or review.get('coordinator') != a['coordinator'] or review.get('run') != a['run']
            or review.get('rendered_diff_sha256') != digest(evidence['rendered-d2pass2-r1.diff'])):
        raise ValueError('W12 derivative review binding differs')
    base = evidence['base-hashes.txt'].read_text()
    diff = evidence['rendered-d2pass2-r1.diff'].read_text()
    if (str(review['base_env_sha256']) + '  dsv41.env' not in base
            or 'capture_env_sha256=' + str(review['capture_env_sha256']) not in diff
            or '/d2-decode:rw' not in diff or 'd2pass2-r1' not in diff
            or str(review['source']) not in diff):
        raise ValueError('W12 base/derivative hashes differ')
    staged = evidence['staged-hashes.txt'].read_text().splitlines()
    mounted = evidence['mounted-hashes.txt'].read_text().splitlines()
    if len(staged) != 6 or len(mounted) != 6:
        raise ValueError('both ranks require three mounted/staged source hashes')
    mounts = {'capture/decode_capture.py': '/dspark/decode_capture.py',
              'capture/speculator_replay.py': '/dspark/speculator.py',
              'spark/remote_dspark/speculator.py': '/remote_dspark/speculator.py'}
    seen: set[tuple[str, str]] = set()
    for line in staged:
        host, sha, filename = line.split()
        if (host not in ('sparkhead', 'sparkworker-wired') or filename not in mounts
                or (host, filename) in seen):
            raise ValueError('unexpected/duplicate staged source')
        seen.add((host, filename))
        if not any(r.split()[0] == host and r.split()[1] == sha
                   and r.split()[2].endswith(mounts[filename]) for r in mounted):
            raise ValueError('mounted/staged capture derivative differs')
    log = evidence['w12.log'].read_text()
    if ('CLOSE: window closed; production K1-327680 + decode-capture derivative d2pass2-r1 serving' not in log
            or 'POST: derivative active on both ranks' not in log
            or 'K1 START VERIFIED' not in evidence['start.log'].read_text()):
        raise ValueError('W12 served derivative evidence missing')
    generator = object_map(a.get('generator'), 'generator')
    if (generator.get('revision') != REVISION or generator.get('priority') != 20
            or generator.get('top20') != 'generation'):
        raise ValueError('pass2 generator revision/settings differ')
    for label in GEN_FILES:
        sha = digest(evidence[label])
        sources = object_map(generator.get('source_sha256'), 'generator source hashes')
        if (('GEN: ' + sha + '  ' + str(evidence[label])) not in log
                or sources.get(label) != sha):
            raise ValueError('generator source not bound by W12 and attestation')
    target = object_map(a.get('target'), 'target')
    if json.loads(evidence['target-receipt.json'].read_text()) != dict(target):
        raise ValueError('production target identity differs from its receipt')
    derivative = object_map(a.get('decode_capture_target'), 'decode_capture_target')
    if (derivative.get('review_sha256') != digest(evidence['review-d2pass2-r1.json'])
            or derivative.get('capture_env_sha256') != review['capture_env_sha256']
            or derivative.get('run') != a['run']):
        raise ValueError('decode target differs from reviewed derivative')
    context: dict[str, object] = {'schema': 'd2.sequence-context.v1', 'status': 'COMPLETE',
        'mode': 'decode_capture', 'target': dict(target, receipt_sha256=digest(evidence['target-receipt.json'])),
        'decode_capture_target': dict(derivative, receipt_sha256=expected_sha),
        'sampling': a['sampling'], 'formatting': a['formatting'], 'generator': dict(generator)}
    identity = object_map(a.get('identity'), 'identity')
    identity_paths: dict[str, Path] = {}
    for label in ('tokenizer', 'encoder', 'chat_template'):
        row = object_map(identity.get(label), label)
        path = pinned(Path(str(row['path'])), str(row['sha256']))
        identity_paths[label] = path
        context[label + '_path'] = str(path)
        context[label + '_sha256'] = digest(path)
    tok = Tokenizer.from_file(str(identity_paths['tokenizer']))
    strings = ('<think>', '</think>', '<｜Assistant｜>', '<｜User｜>', '<｜end▁of▁sentence｜>', '｜DSML｜')
    context['markers'] = dict(zip(MARKERS, (tok.token_to_id(s) for s in strings)))
    validate_context(context, json.loads(manifest.read_text()), mode='decode_capture')
    if 'UNVERIFIED' in json.dumps(context) or 'INCOMPLETE' in json.dumps(context):
        raise ValueError('coordinator input required: unresolved attestation claims')
    out = root / 'inputs/context-inputs'
    if out.exists():
        raise ValueError('context staging requires NEW inputs/context-inputs')
    # Preserve source bytes and absolute paths; explicit maps provide relocation.
    copies = {**evidence, 'pass2-attestation.json': attestation}
    expected_copies = {row['label']: row['sha256'] for row in a['evidence']}
    expected_copies['pass2-attestation.json'] = expected_sha
    maps: dict[str, str] = {}
    receipts: list[dict[str, str]] = []
    out.mkdir(parents=True, mode=0o700)
    for label, path in copies.items():
        dest = out / 'evidence' / label
        dest.parent.mkdir(exist_ok=True, mode=0o700)
        shutil.copyfile(path, dest)
        if digest(dest) != expected_copies[label]:
            raise ValueError('evidence changed during copy')
        maps[str(path)] = str(dest.relative_to(root))
        receipts.append({'path': str(path), 'sha256': digest(dest)})
    for label, path in identity_paths.items():
        dest = out / 'identity' / label / path.name
        dest.parent.mkdir(parents=True, mode=0o700)
        shutil.copyfile(path, dest)
        if digest(dest) != context[label + '_sha256']:
            raise ValueError('identity changed during copy')
        maps[str(path)] = str(dest.relative_to(root))
    context['evidence_receipts'] = receipts
    (out / 'sequence-context.json').write_text(json.dumps(context, indent=2))
    result: dict[str, object] = {'sequence_context': str(out / 'sequence-context.json'), 'path_maps': maps,
        'files': {str(p.relative_to(root)): digest(p) for p in out.rglob('*') if p.is_file()}}
    (out / 'STAGED.json').write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--attestation', type=Path, required=True)
    ap.add_argument('--attestation-sha256', required=True)
    ap.add_argument('--prompt-manifest', type=Path, required=True)
    a = ap.parse_args()
    import os
    os.umask(0o077)
    print(json.dumps(stage(a.root, a.attestation, a.attestation_sha256, a.prompt_manifest), indent=2))


if __name__ == '__main__':
    main()
