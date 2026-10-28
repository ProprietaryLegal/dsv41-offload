"""Fail-closed source-family/ID validation before any replay tensors are read.

Task labels may recur in train/dev; a source family is a connected component of
source documents used together. Such a component must belong to one split only.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import cast


def rows(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or not all(isinstance(x, dict) for x in value):
        raise ValueError('expected a list of metadata objects')
    return cast(list[Mapping[str, object]], value)


def field(row: Mapping[str, object], name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f'missing/non-string {name}')
    return value


def source_units(row: Mapping[str, object]) -> list[str]:
    value = row.get('source_docs')
    if not isinstance(value, list) or not value or not all(isinstance(x, str) and x for x in value):
        raise ValueError('source_docs must contain source-family identities')
    return cast(list[str], value)


def validate_manifest(manifest: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    prompts = rows(manifest.get('prompts'))
    if manifest.get('n_prompts') != len(prompts):
        raise ValueError('manifest n_prompts disagrees with exact prompt rows')
    by_id: dict[str, Mapping[str, object]] = {}
    hashes: set[str] = set()
    units: dict[str, str] = {}
    actual: dict[str, Counter[str]] = {}
    heldout = {'chronology', 'motion_outline'}
    for p in prompts:
        pid, split, family, sha = (field(p, k) for k in ('pid', 'split', 'family', 'prompt_sha256'))
        if pid in by_id or sha in hashes:
            raise ValueError('duplicate prompt ID or identity')
        by_id[pid] = p
        hashes.add(sha)
        if (family in heldout) != (split == 'heldout_legal'):
            raise ValueError('held-out task family crosses split boundary')
        for unit in source_units(p):
            if units.setdefault(unit, split) != split:
                raise ValueError('source family crosses split boundary')
        actual.setdefault(split, Counter())[family] += 1
    if manifest.get('counts') != {sp: dict(c) for sp, c in actual.items()}:
        raise ValueError('manifest counts disagree with exact split/family rows')
    excluded = manifest.get('excluded_pids', {})
    if not isinstance(excluded, dict) or set(by_id) & set(excluded):
        raise ValueError('excluded prompt IDs remain in the manifest')
    return by_id


def audit_plan(plan: Sequence[Mapping[str, object]], manifest: Mapping[str, object]) -> dict[str, object]:
    by_id = validate_manifest(manifest)
    seen: set[str] = set()
    identities: set[str] = set()
    used: set[str] = set()
    splits: Counter[str] = Counter()
    for e in plan:
        sid, split, family, sha = (field(e, k) for k in ('seq_id', 'split', 'family', 'prompt_sha256'))
        identity = field(e, 'tokens_sha256') if e.get('kind') == 'acap' else sha
        if sid in seen or identity in identities:
            raise ValueError('duplicate sequence ID or prompt identity in replay plan')
        seen.add(sid)
        identities.add(identity)
        if e.get('kind') == 'acap':
            if split != 'acap' or family != 'acap' or sid in by_id or sha in {
                    field(p, 'prompt_sha256') for p in by_id.values()}:
                raise ValueError('acap identity overlaps or has a foreign split')
        else:
            p = by_id.get(sid)
            if p is None or (split, family, sha) != tuple(field(p, k) for k in ('split', 'family', 'prompt_sha256')):
                raise ValueError('replay sequence ID/split/family/hash differs from prompt manifest')
            used.add(sid)
        splits[split] += 1
    # Partial generation is expected under the approved time cap. Retain exact
    # missing IDs in the private receipt; never invent rows to meet a headline.
    return {'schema': 'd2.split-audit.v1', 'manifest_prompts': len(by_id),
            'planned_sequences': len(plan), 'by_split': dict(splits),
            'included_prompt_ids': sorted(used), 'not_replayed_prompt_ids': sorted(set(by_id) - used),
            'source_families_disjoint': True, 'exact_ids_checked': True}
