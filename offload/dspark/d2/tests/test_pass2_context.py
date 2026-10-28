"""CPU-only pass2 provenance staging and real assembly rehearsal, synthetic bytes."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from d2 import assemble as A, generation_export as E, pass2_context as P
from d2.post_replay import context_ready
from d2.s34.test_s34_probes import copied
from d2.tests.test_sequence_audit import context_fixture


def attested(tmp_path, root, prompts):
    original = tmp_path / 'attestation-sources'
    original.mkdir()
    c, pm = context_fixture(original)
    target = dict(c['target'])
    target.pop('receipt_sha256')
    evidence = {}
    def put(label, text):
        p = original / label
        p.write_text(text)
        evidence[label] = p
        return E.digest(p)
    put('target-receipt.json', json.dumps(target))
    for label in P.GEN_FILES:
        put(label, '# synthetic source ' + label)
    put('base-hashes.txt', 'a' * 64 + '  dsv41.env\n')
    put('rendered-d2pass2-r1.diff', 'capture_env_sha256=' + 'b' * 64 + '\nd2pass2-r1 /d2-decode:rw source-synthetic\n')
    review = {'schema': 'd2.decode-apply-review.v1', 'approved': True,
              'coordinator': A.COORDINATOR, 'run': 'd2pass2-r1', 'source': 'source-synthetic', 'base_env_sha256': 'a' * 64,
              'capture_env_sha256': 'b' * 64,
              'rendered_diff_sha256': E.digest(evidence['rendered-d2pass2-r1.diff'])}
    put('review-d2pass2-r1.json', json.dumps(review))
    files = ('capture/decode_capture.py', 'capture/speculator_replay.py', 'spark/remote_dspark/speculator.py')
    staged, mounted = [], []
    for host in ('sparkhead', 'sparkworker-wired'):
        for i, filename in enumerate(files):
            sha = str(i) * 64
            staged.append(f'{host} {sha} {filename}')
            dest = '/remote_dspark/speculator.py' if i == 2 else '/dspark/' + Path(filename).name
            mounted.append(f'{host} {sha} /synthetic' + dest)
    put('staged-hashes.txt', '\n'.join(staged))
    put('mounted-hashes.txt', '\n'.join(mounted).replace('/speculator_replay.py', '/speculator.py'))
    put('start.log', 'K1 START VERIFIED')
    put('w12.log', 'CLOSE: window closed; production K1-327680 + decode-capture derivative d2pass2-r1 serving\n'
        'POST: derivative active on both ranks\n' + '\n'.join('GEN: ' + E.digest(evidence[f]) + '  ' + str(evidence[f]) for f in P.GEN_FILES))
    derivative = dict(target, review_sha256=E.digest(evidence['review-d2pass2-r1.json']),
                      capture_env_sha256='b' * 64, run='d2pass2-r1')
    a = {'schema': 'd2.pass2-attestation.v1', 'coordinator': A.COORDINATOR, 'run': 'd2pass2-r1',
         'target': target, 'decode_capture_target': derivative, 'sampling': c['sampling'],
         'formatting': c['formatting'], 'generator': {'revision': P.REVISION, 'priority': 20, 'top20': 'generation',
            'source_sha256': {f: E.digest(evidence[f]) for f in P.GEN_FILES}},
         'identity': {label: {'path': c[label + '_path'], 'sha256': c[label + '_sha256']}
                      for label in ('tokenizer', 'encoder', 'chat_template')},
         'evidence': [{'label': k, 'path': str(v), 'sha256': E.digest(v)} for k, v in evidence.items()]}
    receipt = original / 'attestation.json'
    receipt.write_text(json.dumps(a))
    counts = {p['split']: {p['family']: 1} for p in prompts}
    manifest = dict(pm, schema='d2.prompts.v1', n_prompts=len(prompts), counts=counts, excluded_pids={},
                    prompts=[dict(p, source='courtlistener') for p in prompts])
    mp = root / 'inputs/prompt-manifest.json'
    mp.write_text(json.dumps(manifest))
    return receipt, mp


def test_context_tool_to_real_g1a(tmp_path, monkeypatch):
    root, prompts = copied(tmp_path, monkeypatch)
    receipt, pm = attested(tmp_path, root, prompts)
    staged = P.stage(root, receipt, E.digest(receipt), pm)
    ctx = Path(staged['sequence_context'])
    c = json.loads(ctx.read_text())
    assert c['status'] == 'COMPLETE' and 'replay_target' not in c
    smoke = root / 'inputs/R6-smoke.json'
    # Only external image smoke is stubbed; full provenance, copy, shards and G1 are real.
    with patch.object(E, 'smoke'):
        result = A.run(root, smoke, E.digest(smoke), pm, ctx, staged['path_maps'], expect_nonce='s34-synthetic')
    assert result['g1a_pass'] and not result['pilot_cleared']
    assert json.loads((root / 'shards/MANIFEST.json').read_text())['checks']['sequences'] == 4


@pytest.mark.parametrize('failure', ['missing', 'changed', 'replay'])
def test_context_refusals(tmp_path, monkeypatch, failure):
    root, prompts = copied(tmp_path, monkeypatch)
    receipt, pm = attested(tmp_path, root, prompts)
    sha = E.digest(receipt)
    a = json.loads(receipt.read_text())
    if failure in ('missing', 'changed'):
        path = Path(a['evidence'][0]['path'])
        if failure == 'missing':
            path.unlink()
        else:
            path.write_text('changed')
        with pytest.raises((ValueError, FileNotFoundError)):
            P.stage(root, receipt, sha, pm)
        assert not (root / 'inputs/context-inputs').exists()
    else:
        staged = P.stage(root, receipt, sha, pm)
        ctx = Path(staged['sequence_context'])
        c = json.loads(ctx.read_text())
        c['replay_target'] = c.pop('decode_capture_target')
        c.pop('mode')
        ctx.write_text(json.dumps(c))
        smoke = root / 'inputs/R6-smoke.json'
        with patch.object(E, 'smoke'), pytest.raises(SystemExit, match='explicit decode-capture context'):
            A.run(root, smoke, E.digest(smoke), pm, ctx, staged['path_maps'])


@pytest.mark.parametrize('failure', ['mounted', 'target', 'generator', 'derivative', 'missing_fields', 'attestation_hash'])
def test_inconsistent_attestation_refused(tmp_path, monkeypatch, failure):
    root, prompts = copied(tmp_path, monkeypatch)
    receipt, pm = attested(tmp_path, root, prompts)
    a = json.loads(receipt.read_text())
    if failure in ('mounted', 'target', 'generator'):
        label = {'mounted': 'mounted-hashes.txt', 'target': 'target-receipt.json', 'generator': 'generate.py'}[failure]
        row = next(r for r in a['evidence'] if r['label'] == label)
        path = Path(row['path'])
        if failure == 'mounted':
            path.write_text(path.read_text().replace('0' * 64, 'f' * 64, 1))
        elif failure == 'target':
            path.write_text('{}')
        else:
            path.write_text('# different generator')
        row['sha256'] = E.digest(path)
        if failure == 'generator':
            a['generator']['source_sha256'][label] = row['sha256']
    elif failure == 'derivative':
        a['decode_capture_target']['capture_env_sha256'] = 'f' * 64
    elif failure == 'missing_fields':
        del a['sampling']
    receipt.write_text(json.dumps(a))
    sha = 'f' * 64 if failure == 'attestation_hash' else E.digest(receipt)
    with pytest.raises(ValueError):
        P.stage(root, receipt, sha, pm)
    assert not (root / 'inputs/context-inputs').exists()
