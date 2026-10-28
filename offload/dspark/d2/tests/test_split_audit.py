from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import cast
import pytest

from d2.split_audit import audit_plan, validate_manifest


def fixture() -> tuple[dict[str, object], list[dict[str, object]]]:
    prompts = [{'pid': 'p1', 'split': 'train', 'family': 'holding_summary',
                'prompt_sha256': 'h1', 'source_docs': ['u1']},
               {'pid': 'p2', 'split': 'heldout_legal', 'family': 'chronology',
                'prompt_sha256': 'h2', 'source_docs': ['u2']}]
    manifest: dict[str, object] = {'prompts': prompts, 'n_prompts': 2,
        'counts': {'train': {'holding_summary': 1}, 'heldout_legal': {'chronology': 1}}}
    plan: list[dict[str, object]] = [{**p, 'seq_id': p['pid'], 'kind': 'gen'} for p in prompts]
    return manifest, plan


def test_exact_population_and_partial_coverage() -> None:
    m, plan = fixture()
    r = audit_plan(plan[:1], m)
    assert r['manifest_prompts'] == 2 and r['not_replayed_prompt_ids'] == ['p2']
    assert r['included_prompt_ids'] == ['p1']


@pytest.mark.parametrize('key,value', [('seq_id', 'unknown'), ('split', 'dev'),
                                     ('family', 'argument_section'), ('prompt_sha256', 'other')])
def test_refuses_foreign_exact_identity(key: str, value: str) -> None:
    m, plan = fixture()
    plan[0][key] = value
    with pytest.raises(ValueError, match='differs'):
        audit_plan(plan, m)


def test_refuses_duplicate_plan_and_acap_overlap() -> None:
    m, plan = fixture()
    with pytest.raises(ValueError, match='duplicate'):
        audit_plan(plan + plan[:1], m)
    plan[0].update(kind='acap', split='acap', family='acap', tokens_sha256='t1')
    with pytest.raises(ValueError, match='acap'):
        audit_plan(plan[:1], m)


def test_two_captures_of_one_acap_prompt(tmp_path: Path) -> None:
    m, _ = fixture()
    captures = tmp_path / 'acap'
    captures.mkdir()
    # Real export schema, synthetic IDs: the prefix is shared, sampled suffixes differ.
    for i in range(2):
        capture = {'schema': 'd2.acap.v1', 'seq_id': f'acap{i}', 'prompt_sha256': 'capture-prompt',
                   'gen_start': 20, 'tokens': list(range(20)) + list(range(30 + i, 40 + i))}
        (captures / f'acap{i}.json').write_text(json.dumps(capture))
    out = tmp_path / 'plan'
    subprocess.run([sys.executable, '-m', 'd2.replay', 'plan', '--seqs', '',
                    '--acap', str(captures), '--out', str(out)], check=True, capture_output=True)
    plan = cast(list[dict[str, object]], [json.loads(line) for line in (out / 'plan.jsonl').read_text().splitlines()])
    assert plan[0]['prompt_sha256'] == plan[1]['prompt_sha256']
    assert plan[0]['tokens_sha256'] != plan[1]['tokens_sha256']
    assert audit_plan(plan, m)['by_split'] == {'acap': 2}
    plan[1]['tokens_sha256'] = plan[0]['tokens_sha256']
    with pytest.raises(ValueError, match='duplicate'):
        audit_plan(plan, m)
    plan[1]['tokens_sha256'] = 'different-capture'
    plan[1]['prompt_sha256'] = 'h2'  # Even a manifest prompt absent from this plan is forbidden.
    with pytest.raises(ValueError, match='acap'):
        audit_plan(plan, m)


def test_refuses_source_family_leak_and_stale_counts() -> None:
    m, _ = fixture()
    prompts = m['prompts']
    assert isinstance(prompts, list)
    prompts[1]['source_docs'] = ['u1', 'u2']
    with pytest.raises(ValueError, match='source family'):
        validate_manifest(m)
    m, _ = fixture()
    m['n_prompts'] = 3
    with pytest.raises(ValueError, match='n_prompts'):
        validate_manifest(m)
    m, _ = fixture()
    m['counts'] = {}
    with pytest.raises(ValueError, match='counts'):
        validate_manifest(m)
