"""CPU-only post-replay audits. Emits aggregates, never prompts, IDs or tensors."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .assembly_paths import AssemblyPaths


def sha(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def aggregate(root: Path) -> dict[str, object]:
    import torch
    from safetensors import safe_open
    manifest = json.loads((root/'MANIFEST.json').read_text())
    n = zeros = over_small = over_limit = 0
    minimum, error_max = math.inf, 0.0
    for split in manifest['splits'].values():
        for entry in split['files']:
            path = root/entry['file']
            if sha(path) != entry['sha256']:
                raise ValueError('shard hash mismatch before S25 aggregate')
            with safe_open(path, framework='pt', device='cpu') as f:
                lp, resid = f.get_tensor('top_lp').double(), f.get_tensor('resid').double()
            if not torch.isfinite(lp).all() or not torch.isfinite(resid).all():
                raise ValueError('nonfinite teacher data')
            if (resid < 0).any() or (lp > 0).any():
                raise ValueError('invalid teacher distribution')
            error = (lp.exp().sum(-1)+resid-1).abs()
            n += resid.numel()
            zeros += int((resid == 0).sum())
            minimum = min(minimum, float(lp.min()))
            error_max = max(error_max, float(error.max()))
            over_small += int((error > 1e-6).sum())
            over_limit += int((error > 1e-4).sum())
    if not n or over_limit:
        raise ValueError('empty population or probability mass error above schema tolerance')
    return {'schema': 'd2.s25-aggregate.v1', 'classification': 'MEASURED',
            'synthetic': bool(manifest.get('synthetic') or manifest.get('generation', {}).get('synthetic')
                              or manifest.get('checks', {}).get('synthetic') or manifest.get('taps_source') == 'synthetic'
                              or manifest.get('checks', {}).get('taps_source') == 'synthetic'),
            'manifest_sha256': sha(root/'MANIFEST.json'), 'span_positions': n,
            'resid_eq_zero': zeros, 'min_top_lp': minimum,
            'abs_sum_p_plus_resid_minus_1': {'max': error_max, 'count_gt_1e-6': over_small, 'count_gt_1e-4': over_limit}}


def k2_complete(root: Path, work: Path, split: str) -> None:
    manifest = json.loads((root/'MANIFEST.json').read_text())
    expected: dict[str, set[int]] = {}
    for entry in manifest['splits'][split]['files']:
        sidecar = json.loads((root/entry['sidecar']).read_text())
        for seq in sidecar['sequences']:
            expected[seq['seq_id']] = set(range(seq['gen_start']-1, seq['L']-2))
    rows = [json.loads(line) for line in (work/(split+'.k2_drafts.jsonl')).read_text().splitlines()]
    if len(rows) != len(expected) or {r['seq_id'] for r in rows} != expected.keys():
        raise ValueError('K2 deadline left incomplete sequences; resume g1 phase later under a new lease')
    for row in rows:
        if {int(p) for p in row['drafts']} != expected[row['seq_id']]:
            raise ValueError('incomplete K2 block coverage')
        if any(len(d) != 5 for d in row['drafts'].values()):
            raise ValueError('invalid K2 draft shape')
    print(json.dumps({'split': split, 'complete_sequences': len(expected)}))


def context_ready(context: Path, manifest: Path, plan: Path, *, paths: AssemblyPaths | None = None,
                  mode: str = 'prefill_replay') -> None:
    from .sequence_audit import validate_context
    data = json.loads(context.read_text())
    pm = json.loads(manifest.read_text())
    receipts = data.get('evidence_receipts', [])
    verified: set[str] = set()
    for receipt in receipts:
        path = paths.resolve(receipt['path']) if paths else Path(receipt['path'])
        digest = sha(path)
        if digest != receipt['sha256']:
            raise ValueError('context evidence receipt hash differs')
        verified.add(digest)
    entries = [json.loads(line) for line in plan.read_text().splitlines()]
    if not entries:
        raise ValueError('empty replay plan')
    for entry in entries:
        c = data.get('per_sequence', {}).get(entry['seq_id'], data)
        if c.get('status', '').startswith('INCOMPLETE') or 'UNVERIFIED' in json.dumps(c):
            raise ValueError('incomplete provenance scaffold is not a build receipt')
        if entry['kind'] == 'acap' and entry['seq_id'] not in data.get('per_sequence', {}):
            raise ValueError('original acap context missing')
        validate_context(c, pm, require_chat_match=entry['kind'] != 'acap', paths=paths, mode=mode)
        keys = ('target', 'decode_capture_target' if mode == 'decode_capture' else 'replay_target')
        if any(c[key]['receipt_sha256'] not in verified for key in keys):
            raise ValueError('target receipt lacks accessible hash-bound evidence_receipts')
        if mode == 'decode_capture':
            matches = [r for r in receipts if r['sha256'] == c['decode_capture_target']['receipt_sha256']]
            receipt_path = paths.resolve(matches[0]['path']) if paths else Path(matches[0]['path'])
            attestation = json.loads(receipt_path.read_text())
            claims = dict(c['decode_capture_target'])
            claims.pop('receipt_sha256')
            target_claims = dict(c['target'])
            target_claims.pop('receipt_sha256')
            if (attestation.get('target') != target_claims
                    or any(attestation.get('identity', {}).get(label, {}).get('sha256') != c[label + '_sha256']
                           for label in ('tokenizer', 'encoder', 'chat_template'))
                    or attestation.get('schema') != 'd2.pass2-attestation.v1'
                    or attestation.get('decode_capture_target') != claims
                    or any(attestation.get(k) != c.get(k) for k in ('sampling', 'formatting', 'generator'))):
                raise ValueError('decode-capture target receipt claims differ')
    print(json.dumps({'context_sha256': sha(context), 'sequences': len(entries), 'receipts_verified': len(verified)}))


def g1_gate(path: Path) -> None:
    gate = json.loads(path.read_text())
    if gate.get('split') != 'acap' or gate.get('g1', {}).get('pass') is not True or gate['g1'].get('binding') is not True:
        raise ValueError('binding acap G1 did not pass; pilot preparation refused')


def binding(root: Path, replay: Path, gate: Path, context: Path, prompts: Path) -> None:
    from .dataset import check_gate_file
    manifest = json.loads((root/'MANIFEST.json').read_text())
    g = json.loads(gate.read_text())
    if check_gate_file(g, str(replay)) or g['gate']['pass'] is not True:
        raise ValueError('foreign/failed equivalence gate')
    if (manifest['replay']['plan_sha256'] != sha(replay/'plan.jsonl')
            or manifest['prompt_manifest_sha256'] != sha(prompts)
            or manifest['generation']['sequence_context_sha256'] != sha(context)
            or manifest['checks']['equivalence_gate']['sha256'] != sha(gate)
            or manifest['checks']['skipped_sequences'] != 0
            or manifest['checks']['equivalence_gate'].get('override')
            or manifest['checks']['g1_plumbing'].get('override')
            or manifest.get('taps_source') != 'prefill_replay'
            or manifest['generation'].get('synthetic') is not False):
        raise ValueError('shards are not bound to these exact replay/context/gates or contain overrides/skips')
    print(json.dumps({'manifest_sha256': sha(root/'MANIFEST.json'), 'binding': 'verified'}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('check', choices=('aggregate', 'k2-complete', 'context', 'g1', 'binding'))
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--work', type=Path)
    ap.add_argument('--split')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--manifest', type=Path)
    ap.add_argument('--plan', type=Path)
    ap.add_argument('--replay', type=Path)
    ap.add_argument('--gate', type=Path)
    ap.add_argument('--context', type=Path)
    a = ap.parse_args()
    if a.check == 'aggregate':
        if a.out is None:
            raise ValueError('--out required')
        with a.out.open('x') as f:
            json.dump(aggregate(a.root), f, indent=2, allow_nan=False)
    elif a.check == 'k2-complete':
        if a.work is None or a.split is None:
            raise ValueError('--work and --split required')
        k2_complete(a.root, a.work, a.split)
    elif a.check == 'context':
        if a.manifest is None or a.plan is None:
            raise ValueError('--manifest and --plan required')
        context_ready(a.root, a.manifest, a.plan)
    elif a.check == 'binding':
        if None in (a.replay, a.gate, a.context, a.manifest):
            raise ValueError('--replay, --gate, --context, --manifest required')
        binding(a.root, a.replay, a.gate, a.context, a.manifest)
    else:
        g1_gate(a.root)


if __name__ == '__main__':
    main()
