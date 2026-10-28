"""CPU-only synthetic evidence; never reads production or runs the collector."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from d2 import pass2_attest as A, pass2_context as P
from d2.s34.test_s34_probes import copied
from d2.tests.test_pass2_context import attested


def fixture(tmp_path, monkeypatch):
    root, prompts = copied(tmp_path, monkeypatch)
    existing, manifest = attested(tmp_path, root, prompts)
    attestation = json.loads(existing.read_text())
    attestation['evidence'] = [r for r in attestation['evidence'] if r['label'] != 'target-receipt.json']
    values = {}
    for name, keys in A.FIELDS.items():
        value = attestation
        for key in keys:
            value = value[key]
        values[name] = value
    values.update({key: True for key in A.OBLIGATIONS})
    primary = tmp_path / 'synthetic-reviewed-values.json'
    primary.write_text(json.dumps({'values': values}))
    support = tmp_path / 'synthetic-primary-metadata.json'
    support.write_text(json.dumps({'synthetic': True, 'scope': 'no production identity claimed'}))
    claims = {name: {'value': value, 'status': 'MEASURED', 'scope': 'synthetic proof only', 'sources': [
        {'path': str(primary), 'sha256': P.digest(primary), 'pointer': '/values/' + name.replace('~', '~0').replace('/', '~1')},
        {'path': str(support), 'sha256': P.digest(support)}]} for name, value in values.items()}
    evidence_map = tmp_path / 'map.md'
    evidence_map.write_text('```json\n' + json.dumps({'schema': 'd2.pass2-evidence-map.v1', 'claims': claims}) + '\n```\n')
    resolution = tmp_path / 'resolution.json'
    resolution.write_text(json.dumps({'schema': 'd2.pass2-resolution.v1', 'approved': True,
                                     'coordinator': A.COORDINATOR, 'run': A.RUN, 'claims': claims}))
    return root, manifest, evidence_map, resolution, tmp_path / 'draft'


def run(manifest, evidence_map, resolution, out):
    return A.generate(evidence_map, P.digest(evidence_map), resolution, P.digest(resolution), manifest, out)


def test_draft_to_pass2_context(tmp_path, monkeypatch):
    root, manifest, evidence_map, resolution, out = fixture(tmp_path, monkeypatch)
    result = run(manifest, evidence_map, resolution, out)
    draft = Path(result['attestation'])
    a = json.loads(draft.read_text())
    assert a['draft_status'] == 'REQUIRES_COORDINATOR_REVIEW'
    assert json.loads(Path(result['target_receipt']).read_text()) == a['target']
    # Exercise the actual CLI on the newly generated synthetic draft, not just
    # the generator's internal stage validation. No git or host commands run.
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])
    process = subprocess.run([sys.executable, '-m', 'd2.pass2_context', '--root', str(root),
        '--attestation', str(draft), '--attestation-sha256', result['sha256'],
        '--prompt-manifest', str(manifest)], env=env, capture_output=True, text=True, check=True)
    staged = json.loads(process.stdout)
    ctx = json.loads(Path(staged['sequence_context']).read_text())
    assert ctx['status'] == 'COMPLETE' and ctx['mode'] == 'decode_capture'
    assert not (out / '.validation').exists()
    assert json.loads((out / 'DRAFT.json').read_text())['resolution_sha256'] == P.digest(resolution)


@pytest.mark.parametrize('failure', ['missing', 'inferred', 'hash', 'value', 'map', 'null', 'obligation',
                                     'unapproved', 'unbound', 'target', 'source_missing', 'identity', 'sampling'])
def test_refusals(tmp_path, monkeypatch, failure):
    _, manifest, evidence_map, resolution, out = fixture(tmp_path, monkeypatch)
    r = json.loads(resolution.read_text())
    claims = r['claims']
    key = 'target.weights_manifest_sha256'
    if failure == 'missing':
        del claims[key]
    elif failure == 'inferred':
        claims[key]['status'] = 'IMPLIED'
    elif failure == 'hash':
        claims[key]['sources'][0]['sha256'] = 'f' * 64
    elif failure == 'value':
        claims[key]['value'] = 'f' * 64
    elif failure == 'map':
        m = json.loads(evidence_map.read_text().split('```json\n')[1].split('\n```')[0])
        m['claims'][key]['value'] = 'e' * 64
        evidence_map.write_text('```json\n' + json.dumps(m) + '\n```\n')
    elif failure == 'null':
        del claims['sampling.seed']['value']
    elif failure == 'obligation':
        claims['verification.renderer_raw_ids']['status'] = 'MISSING'
    elif failure == 'unapproved':
        r['approved'] = False
    elif failure == 'unbound':
        del claims[key]['sources'][0]['pointer']
    elif failure == 'source_missing':
        Path(claims[key]['sources'][0]['path']).unlink()
    else:
        # Update primary JSON and the map, so the real context semantic checks
        # must find the contradiction rather than just source/hash binding.
        key = {'target': 'decode_capture_target.weights_manifest_sha256',
               'identity': 'identity.tokenizer.sha256', 'sampling': 'sampling.temperature'}[failure]
        value = 0.5 if failure == 'sampling' else 'e' * 64
        primary = Path(claims[key]['sources'][0]['path'])
        p = json.loads(primary.read_text())
        p['values'][key] = value
        primary.write_text(json.dumps(p))
        for c in claims.values():
            c['sources'][0]['sha256'] = P.digest(primary)
        claims[key]['value'] = value
        m = json.loads(evidence_map.read_text().split('```json\n')[1].split('\n```')[0])
        m['claims'][key]['status'] = 'IMPLIED'
        evidence_map.write_text('```json\n' + json.dumps(m) + '\n```\n')
    resolution.write_text(json.dumps(r))
    with pytest.raises((ValueError, FileNotFoundError)):
        run(manifest, evidence_map, resolution, out)
    assert not out.exists()


def test_coordinator_signed_configuration_fields(tmp_path, monkeypatch):
    _, manifest, evidence_map, resolution, out = fixture(tmp_path, monkeypatch)
    r = json.loads(resolution.read_text())
    for name in A.SIGNABLE:
        r['claims'][name]['status'] = A.SIGNED
    resolution.write_text(json.dumps(r))
    a = json.loads(Path(run(manifest, evidence_map, resolution, out)['attestation']).read_text())
    assert {n for n, c in a['claim_review']['claims'].items() if c['status'] == A.SIGNED} == set(A.SIGNABLE)


@pytest.mark.parametrize('failure', ['integrity', 'obligation_false', 'unbound', 'one_source'])
def test_coordinator_signature_limits(tmp_path, monkeypatch, failure):
    _, manifest, evidence_map, resolution, out = fixture(tmp_path, monkeypatch)
    r = json.loads(resolution.read_text())
    claims = r['claims']
    key = {'integrity': 'target.weights_manifest_sha256', 'obligation_false': 'verification.served_identity',
           'unbound': 'sampling.top_k', 'one_source': 'formatting.tools'}[failure]
    claims[key]['status'] = A.SIGNED
    if failure == 'obligation_false':
        primary = Path(claims[key]['sources'][0]['path'])
        p = json.loads(primary.read_text())
        p['values'][key] = False
        primary.write_text(json.dumps(p))
        for c in claims.values():
            c['sources'][0]['sha256'] = P.digest(primary)
        claims[key]['value'] = False
        m = json.loads(evidence_map.read_text().split('```json\n')[1].split('\n```')[0])
        m['claims'][key]['status'] = 'MISSING'
        evidence_map.write_text('```json\n' + json.dumps(m) + '\n```\n')
    elif failure == 'unbound':
        del claims[key]['sources'][0]['pointer']
    elif failure == 'one_source':
        del claims[key]['sources'][1]
    resolution.write_text(json.dumps(r))
    with pytest.raises(ValueError):
        run(manifest, evidence_map, resolution, out)
    assert not out.exists()


def test_refuse_existing_output_and_symlink(tmp_path, monkeypatch):
    _, manifest, evidence_map, resolution, out = fixture(tmp_path, monkeypatch)
    out.mkdir()
    sentinel = out / 'keep'
    sentinel.write_text('untouched')
    with pytest.raises(ValueError, match='NEW offline'):
        run(manifest, evidence_map, resolution, out)
    assert sentinel.read_text() == 'untouched'
    alias = tmp_path / 'alias'
    alias.symlink_to(resolution)
    with pytest.raises(ValueError, match='symlink'):
        run(manifest, evidence_map, alias, tmp_path / 'other-output')


def test_duplicate_json_keys_refused():
    with pytest.raises(ValueError, match='duplicate JSON key'):
        A.parse('{"approved": false, "approved": true}')
