"""S36 independent probes of merge ecd2a64 (43b1688, pass2_attest COORDINATOR_ATTESTED for 12 fields).

Synthetic probes use the existing CPU fixture. Real probes read the pass-2 artifacts under the campaign directory read
only (skipped when absent) and write only to pytest's tmp_path; no hosts, no inference, no W12 writes.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from d2 import pass2_attest as A, pass2_context as P
from d2.copy_binding import digest
from d2.tests.test_pass2_attest import fixture, run

CAMPAIGN = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD')
SIGN = CAMPAIGN/'PASS2-ATTEST-d2pass2-r1'
ATTESTATION = CAMPAIGN/'PASS2-ATTESTATION-d2pass2-r1'
COLLECT = CAMPAIGN/'PASS2-COLLECT-d2pass2-r1'
STAGE = CAMPAIGN/'PASS2-STAGE-d2pass2-r1/inputs'
MAP = CAMPAIGN/'PASS2-SIGN/checkout-ecd2a64/offload/dspark/d2/PASS2-ATTEST-EVIDENCE.md'
PROMPTS = Path('/opt/dsv41/data/d2/prompts')
GEN_STATE = Path('/opt/dsv41/data/d2/gen-decode-07fc7d6-d2pass2-r1')
REPO = Path(__file__).resolve().parents[4]
ATTESTATION_SHA = '1bf698d97035c621c5ed79b762756ef60024ff06346ba9f143fb7377c9ba661a'
needs_real = pytest.mark.skipif(not all(p.exists() for p in (SIGN, ATTESTATION, COLLECT, STAGE, MAP, PROMPTS, GEN_STATE)),
                                reason='real pass-2 artifacts absent')
# Everything that is a hash, an identity file, evidence, generator source or run identity.
INTEGRITY = {n for n in A.FIELDS if n.endswith(('sha256', '.path')) or n.startswith(('identity.', 'generator.'))
             or n in ('schema', 'coordinator', 'run', 'evidence', 'decode_capture_target.run')}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_signable_is_exactly_the_twelve_open_fields_and_never_integrity() -> None:
    assert len(A.SIGNABLE) == 12 and A.SIGNABLE <= set(A.FIELDS) | set(A.OBLIGATIONS)
    assert not A.SIGNABLE & INTEGRITY
    # model_root/quantization/temperature/top_p/stop/seed stay measured-only too.
    assert {n for n in A.SIGNABLE if not n.startswith('verification.')} == {
        'target.engine_revision', 'decode_capture_target.engine_revision', 'sampling.top_k', 'sampling.min_p',
        'sampling.stop_token_ids', 'sampling.structured_outputs', 'formatting.chat_template_kwargs',
        'formatting.tools', 'formatting.tool_choice'}


@pytest.mark.parametrize('name', sorted(set(A.FIELDS) - A.SIGNABLE))
def test_every_non_signable_field_refuses_the_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    _, manifest, evidence_map, resolution, out = fixture(tmp_path, monkeypatch)
    r = json.loads(resolution.read_text())
    r['claims'][name]['status'] = A.SIGNED
    resolution.write_text(json.dumps(r))
    with pytest.raises(ValueError, match='integrity field cannot be coordinator-attested'):
        run(manifest, evidence_map, resolution, out)
    assert not out.exists()


@pytest.mark.parametrize('failure', ['map_measured', 'obligation_string', 'scope', 'hash', 'status_variant'])
def test_other_proof_rules_bind_signed_claims(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    _, manifest, evidence_map, resolution, out = fixture(tmp_path, monkeypatch)
    r = json.loads(resolution.read_text())
    claims = r['claims']
    key = 'verification.effective_defaults' if failure == 'obligation_string' else 'sampling.top_k'
    claims[key]['status'] = A.SIGNED
    if failure == 'map_measured':
        # The map's MEASURED value stays an immutable constraint on a signed claim.
        primary = Path(claims[key]['sources'][0]['path'])
        p = json.loads(primary.read_text())
        p['values'][key] = 40
        primary.write_text(json.dumps(p))
        for c in claims.values():
            c['sources'][0]['sha256'] = digest(primary)
        claims[key]['value'] = 40
    elif failure == 'obligation_string':
        primary = Path(claims[key]['sources'][0]['path'])
        p = json.loads(primary.read_text())
        p['values'][key] = 'true'
        primary.write_text(json.dumps(p))
        for c in claims.values():
            c['sources'][0]['sha256'] = digest(primary)
        claims[key]['value'] = 'true'
        m = json.loads(evidence_map.read_text().split('```json\n')[1].split('\n```')[0])
        m['claims'][key]['status'] = 'MISSING'
        evidence_map.write_text('```json\n' + json.dumps(m) + '\n```\n')
    elif failure == 'scope':
        claims[key]['scope'] = ' '
    elif failure == 'hash':
        claims[key]['sources'][1]['sha256'] = 'f' * 64
    else:
        claims[key]['status'] = A.SIGNED.lower()
    resolution.write_text(json.dumps(r))
    with pytest.raises(ValueError):
        run(manifest, evidence_map, resolution, out)
    assert not out.exists()


@needs_real
def test_real_resolution_signs_exactly_signable_and_binds_signed_values_file() -> None:
    r = json.loads((SIGN/'resolution.json').read_text())
    assert sha(SIGN/'resolution.json') == '6ea54daca9c5015a9c72e7ab444f706bff14c95d44b698d59ba096a5ad7ebfb6'
    statuses = collections.Counter(c['status'] for c in r['claims'].values())
    assert statuses == {'MEASURED': 31, A.SIGNED: 12}
    assert {n for n, c in r['claims'].items() if c['status'] == A.SIGNED} == set(A.SIGNABLE)
    signed = json.loads((SIGN/'coordinator-signed-values.json').read_text())
    for name in A.SIGNABLE:
        pointers = [s for s in r['claims'][name]['sources'] if 'pointer' in s]
        # Observation: the only value binding of a signed claim is the coordinator's own signed-values file.
        assert [Path(s['path']).name for s in pointers] == ['coordinator-signed-values.json']
        assert signed['values'][name] == r['claims'][name]['value']
        for s in r['claims'][name]['sources']:
            assert sha(Path(s['path'])) == s['sha256']


@needs_real
def test_real_attestation_regenerates_byte_identical(tmp_path: Path) -> None:
    out = tmp_path/'regen'
    result = A.generate(MAP, sha(MAP), SIGN/'resolution.json', sha(SIGN/'resolution.json'),
                        PROMPTS/'manifest.json', out)
    text = Path(result['attestation']).read_text().replace(str(out) + '/', str(ATTESTATION) + '/')
    assert hashlib.sha256(text.encode()).hexdigest() == ATTESTATION_SHA == sha(ATTESTATION/'pass2-attestation.json')
    assert sha(out/'target-receipt.json') == sha(ATTESTATION/'target-receipt.json')
    # Observation (pre-existing, 4035fb0): claim_review aliases the evidence list, so it records 11 entries
    # (incl. the generated target receipt) where the reviewed resolution has 10. No consumer reads claim_review.
    a = json.loads((ATTESTATION/'pass2-attestation.json').read_text())
    r = json.loads((SIGN/'resolution.json').read_text())
    assert len(a['claim_review']['claims']['evidence']['value']) == 11 == len(r['claims']['evidence']['value']) + 1


@needs_real
def test_real_staged_context_regenerates_identical(tmp_path: Path) -> None:
    root = tmp_path/'root'
    (root/'inputs').mkdir(parents=True)
    (root/'inputs/prompt-manifest.json').write_bytes((STAGE/'prompt-manifest.json').read_bytes())
    P.stage(root, ATTESTATION/'pass2-attestation.json', ATTESTATION_SHA, root/'inputs/prompt-manifest.json')
    mine = root/'inputs/context-inputs'
    real = STAGE/'context-inputs'
    assert (mine/'sequence-context.json').read_bytes() == (real/'sequence-context.json').read_bytes()
    assert json.loads((mine/'STAGED.json').read_text())['files'] == json.loads((real/'STAGED.json').read_text())['files']
    assert sha(real/'sequence-context.json') == '9ec34fe1d23ea54e20889373a9eac063e8b23efda26973db25574be2ee4bd100'
    assembled = json.loads((CAMPAIGN/'S36-inputs/ASSEMBLED.json').read_text())
    assert assembled['inputs']['inputs/context-inputs/sequence-context.json'] == sha(real/'sequence-context.json')
    assert assembled['inputs']['inputs/prompt-manifest.json'] == sha(PROMPTS/'manifest.json')


@needs_real
def test_real_coordinator_review_facts_rederive() -> None:
    review = json.loads((SIGN/'coordinator-review.json').read_text())
    # Prompts: manifest binds the jsonl; repo manifest is the pass manifest; 538 x thinking/high; no tools.
    manifest = json.loads((PROMPTS/'manifest.json').read_text())
    assert manifest['prompts_jsonl_sha256'] == sha(PROMPTS/'prompts.jsonl') == review['prompts']['prompts_jsonl']['sha256']
    assert sha(PROMPTS/'manifest.json') == sha(REPO/'offload/dspark/d2/results/prompt_manifest.json')
    rows = [json.loads(line) for line in (PROMPTS/'prompts.jsonl').read_text().splitlines()]
    assert len(rows) == 538 == manifest['n_prompts']
    assert collections.Counter(json.dumps(r['chat_template_kwargs'], sort_keys=True) for r in rows) == {
        '{"reasoning_effort": "high", "thinking": true}': 538}
    assert not any('tools' in r or 'tool_choice' in r for r in rows)
    # Renderer: the counts come from the generator state (the cited D2-GEN2.out.log does not contain them).
    seqs = [json.loads(p.read_text()) for p in sorted((GEN_STATE/'seqs').glob('*.json'))]
    assert len(seqs) == 131 and all(s['local_ids_match'] is True for s in seqs)
    assert collections.Counter(s['status'] for s in seqs) == {'done': 128, 'partial': 3}
    assert sum(len(s['gen_ids']) for s in seqs) == 974303
    log = Path(review['renderer']['generator_log']['path']).read_text()
    assert 'local_ids_match' not in log and '"tokenize_local_mismatch": 0' in log
    launch = (CAMPAIGN/'W12/decode_gen.sh').read_text()
    assert '--prompts /opt/dsv41/data/d2/prompts/prompts.jsonl' in launch and '--no-local-check' not in launch
    # Serving: config of record equals sparkhead main 21bed31; no generation-config/chat-template flags.
    for name, git_path in (('dsv41_tp2_offload.sh', 'offload/launch/dsv41_tp2_offload.sh'), ('serve.sh', 'serve.sh')):
        blob = subprocess.run(['git', '-C', str(REPO), 'show', '21bed31:' + git_path], capture_output=True, check=True).stdout
        assert blob == (SIGN/f'config-of-record-21bed31-{name}').read_bytes()
        assert b'generation-config' not in blob and b'--chat-template' not in blob
    files = {h: json.loads((COLLECT/f'{h}-files.json').read_text())['files'] for h in ('sparkhead', 'sparkworker-wired')}
    assert files['sparkhead'] == files['sparkworker-wired']
    by_path = {f['path']: f for f in files['sparkhead']}
    assert by_path['/srv/dsv41/legal-v2a/generation_config.json']['status'] == 'ABSENT'
    # vLLM falls back to GenerationConfig.from_model_config(config.json) when the file is absent: no sampling keys there.
    config = by_path['/srv/dsv41/legal-v2a/config.json']['text']
    assert not {'temperature', 'top_k', 'top_p', 'min_p', 'repetition_penalty', 'presence_penalty',
                'frequency_penalty', 'max_new_tokens'} & set(json.dumps(json.loads(config)).replace('"', ' ').split())
    version = by_path['/usr/local/lib/python3.12/dist-packages/vllm/_version.py']
    assert hashlib.sha256(version['text'].encode()).hexdigest() == version['sha256']
    assert "__version__ = version = '0.31.1.dev0+g6e517b15.dsv41gb10'" in version['text']
    inspect = {h: json.loads((COLLECT/f'{h}-inspect.json').read_text())[0] for h in ('sparkhead', 'sparkworker-wired')}
    assert {i['Image'] for i in inspect.values()} == {review['serving']['container_image']['sparkhead']}
    assert all(i['Config']['Cmd'] == ['/serve.sh'] for i in inspect.values())
