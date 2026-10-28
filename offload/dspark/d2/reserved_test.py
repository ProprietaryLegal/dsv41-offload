"""Freeze public TEST prompts via the existing renderer/source/template pipeline.

No network, generation, replay or model evaluation. Nothing here selects a
candidate. Full prompts stay in the private campaign directory, never in git.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
from typing import cast

from . import prompts as P
from .split_audit import field, rows, source_units, validate_manifest

FAMILIES = ('test_record_consistency', 'test_alternative_disposition',
            'test_procedural_dependencies', 'test_evidence_foundation')
SEED = 20261024
TARGETS = (1500, 6000, 16000)


def validate_reserved(existing: list[dict[str, object]], reserved: list[dict[str, object]]) -> None:
    if len(reserved) != len(FAMILIES) * len(TARGETS):
        raise ValueError('reserved TEST population must be exactly 12 prompts')
    if Counter(field(p, 'family') for p in reserved) != Counter({f: 3 for f in FAMILIES}):
        raise ValueError('reserved TEST task families differ from frozen design')
    if any(p.get('split') != 'test' for p in reserved):
        raise ValueError('reserved TEST split must never be train/dev')
    old_families = {field(p, 'family') for p in existing}
    if old_families & set(FAMILIES):
        raise ValueError('reserved TEST family was previously used')
    P.check_disjoint(existing + reserved)
    # Treat every previously seen population as exclusion material, including
    # acap documents (excluded earlier) and development / historical heldout.
    P.check_near_duplicates([{**p, 'split': 'train'} for p in existing] + reserved)


def build(a: argparse.Namespace) -> dict[str, object]:
    out = Path(a.out)
    if out.exists():
        raise FileExistsError('refusing to overwrite a frozen TEST directory')
    parent = Path(a.existing_manifest)
    pm = json.loads(parent.read_text())
    validate_manifest(pm)
    pj = Path(a.existing_prompts)
    if P.sha256_file(str(pj)) != pm['prompts_jsonl_sha256']:
        raise ValueError('existing prompt JSONL differs from frozen manifest')
    existing = [dict(p) for p in rows([json.loads(line) for line in pj.read_text().splitlines()])]
    expected = {field(p, 'pid'): p for p in rows(pm['prompts'])}
    if len(existing) != len(expected) or any(
            p.get('prompt_sha256') != expected.get(field(p, 'pid'), {}).get('prompt_sha256') for p in existing):
        raise ValueError('existing prompt IDs/identities differ from manifest')
    used = {u for p in existing for u in source_units(p)}
    docs = P.load_legal(a.bench)  # verifies public corpus SHA256SUMS
    unused = [d for d in docs if d['text_sha256'] not in used and d['doc_id'] not in P.ACAP_DOCS]
    probes = [{**p, 'split': 'train'} for p in existing] + [
        {'pid': f'probe-{j}', 'split': 'test', 'messages': [{'role': 'user', 'content': d['text']}]}
        for j, d in enumerate(unused)]
    bad = {pid for pid, _, _ in P.near_duplicates(probes)}
    clean = [d for j, d in enumerate(unused) if f'probe-{j}' not in bad]
    if len(clean) < len(FAMILIES):
        raise ValueError('not enough unused public source families after overlap screening')
    rng = random.Random(SEED)
    rng.shuffle(clean)
    R = P.Renderer(a.tokenizer, a.encoder)
    reserved: list[dict[str, object]] = []
    for family, doc in zip(FAMILIES, clean):
        doc_tokens = R.count_text(doc['text'])
        for requested, fraction in zip(TARGETS, (0.25, 0.55, 0.95)):
            target = min(requested, max(P.MIN_TOK + 150, int(doc_tokens * fraction)))
            messages, selected, texts = P.build_legal_prompt(R, family, [doc], Counter(), rng, target,
                                                          {doc['text_sha256']: R.count_text(doc['text'])})
            n = R.count(messages)
            if not P.MIN_TOK <= n <= P.MAX_TOK:
                raise ValueError('TEST prompt outside existing pipeline length limits')
            reserved.append({'pid': f'd2test{len(reserved):04}', 'split': 'test', 'family': family,
                'source': doc['source'], 'source_docs': [d['text_sha256'] for d in selected],
                'doc_ids': [d['doc_id'] for d in selected], 'excerpt': [t != d['text'] for d, t in zip(selected, texts)],
                'target_tokens': target, 'est_tokens': n, 'messages': messages,
                'chat_template_kwargs': dict(P.CTK), 'prompt_sha256': P.prompt_sha256(messages, P.CTK)})
    validate_reserved(existing, reserved)
    out.mkdir(parents=True, mode=0o700)
    frozen = out / 'prompts.jsonl'
    frozen.write_text(''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in reserved))
    os.chmod(frozen, 0o400)
    manifest: dict[str, object] = {'schema': P.SCHEMA, 'reservation_schema': 'd2.reserved-test.v1',
        'seed': SEED, 'n_prompts': len(reserved), 'counts': {'test': dict(Counter(field(p, 'family') for p in reserved))},
        'selection_allowed': False, 'use': 'frozen_final_candidate_only', 'generation_order': ['test'],
        'parent_manifest_sha256': P.sha256_file(str(parent)), 'chat_template_kwargs': dict(P.CTK),
        'inputs': {**R.provenance, 'bench_files': {
            f: P.sha256_file(str(Path(a.bench) / f)) for f in ('opinions_cl.jsonl', 'filings_recap.jsonl')}},
        'prompts_jsonl_sha256': P.sha256_file(str(frozen)),
        'checks': {'unused_sources_before_screen': len(unused), 'rejected_for_overlap': len(bad),
                   'source_families': len(FAMILIES), 'disjoint': True,
                   'near_duplicate_rule': {'shingle_words': P.SHINGLE_K, 'max_fraction': P.NEAR_DUP_MAX}},
        'tokens': {'min': min(cast(int, p['est_tokens']) for p in reserved),
                   'max': max(cast(int, p['est_tokens']) for p in reserved)},
        'prompts': [{k: v for k, v in p.items() if k not in ('messages', 'chat_template_kwargs')} for p in reserved]}
    validate_manifest(manifest)
    mp = out / 'manifest.json'
    mp.write_text(json.dumps(manifest, indent=1) + '\n')
    os.chmod(mp, 0o400)
    sums = out / 'SHA256SUMS'
    sums.write_text(''.join(f'{P.sha256_file(str(p))}  {p.name}\n' for p in (frozen, mp)))
    os.chmod(sums, 0o400)
    if a.commit_manifest:
        Path(a.commit_manifest).write_text(mp.read_text())
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--existing-manifest', default='/opt/dsv41/data/d2/prompts/manifest.json')
    ap.add_argument('--existing-prompts', default='/opt/dsv41/data/d2/prompts/prompts.jsonl')
    ap.add_argument('--bench', default=P.BENCH)
    ap.add_argument('--tokenizer', default=P.TOKENIZER)
    ap.add_argument('--encoder', default=P.ENCODER)
    ap.add_argument('--commit-manifest', default='')
    a = ap.parse_args()
    m = build(a)
    print(json.dumps({k: v for k, v in m.items() if k not in ('prompts', 'inputs')}, indent=1))


if __name__ == '__main__':
    main()
