"""Bounded idle-gated retries never qualify contamination, drift, or partial runs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from d2 import greedy_reference as G
from d2 import smoke_prepare as P
from d2.decode_smoke_dry import DrySmoke

REPO = Path(__file__).resolve().parents[4]


def fixture(tmp_path, monkeypatch, failure='none'):
    monkeypatch.setattr(G, 'PRIVATE', tmp_path)
    monkeypatch.setattr(P, 'PRIVATE', tmp_path)
    io = DrySmoke(REPO, tmp_path)
    spec = io.inputs()
    capture = Path(spec['greedy_reference']['path'])
    monkeypatch.setattr(G, 'K1_CAPTURE', capture)
    proof_path = Path(spec['greedy_reference']['request_receipt'])
    proof = json.loads(proof_path.read_text())
    proof['request']['priority'] = 0
    proof_path.write_text(json.dumps(proof))
    spec['requests'][2]['body']['priority'] = 0
    spec['greedy_reference']['request_receipt_sha256'] = G.digest(proof_path)
    monkeypatch.setattr(G, 'historical_request', lambda *args: proof['request'])
    inputs = tmp_path/'input.json'
    inputs.write_text(json.dumps(spec))

    class Fake:
        count = 0
        metric_calls = 0
        calls = []
        pauses = 0
        def metrics(self):
            self.metric_calls += 1
            gauges = 'vllm:num_requests_running{} 0\nvllm:num_requests_waiting{} 0\n'
            if failure == 'busy' or (failure == 'busy_once' and self.metric_calls == 1):
                gauges = 'vllm:num_requests_running{} 1\nvllm:num_requests_waiting{} 0\n'
            return io.latency_metrics(self.count)+gauges
        def pause(self): self.pauses += 1
        def post(self, body, attempt):
            self.calls.append(dict(body))
            self.count += 1
            if failure in ('contaminated', 'contaminated_once') and (failure == 'contaminated' or len(self.calls) == 1):
                self.count += 1
            if failure == 'foreign_live' and len(self.calls) == 1:
                raise G.Contamination('foreign work; owned child cancelled')
            response = {'choices':[{'finish_reason':'stop','message':{'content':
                'drift' if failure == 'drift' else 'synthetic-greedy'}}],
                'usage':{'prompt_tokens_details':{'cached_tokens':0}}}
            (attempt/'response.json').write_text(json.dumps(response))
            return response
    return Fake(), inputs


@pytest.mark.parametrize('failure', ['none','busy_once','contaminated_once','foreign_live'])
def test_reference_collects_five_warm_repeats_with_bounded_retries(tmp_path, monkeypatch, failure):
    io, inputs = fixture(tmp_path, monkeypatch, failure)
    result = G.capture(io, inputs, {}, tmp_path/'r2', lambda: {})
    assert result['qualified'] is True and result['repeats'] == 5 and result['warmups'] == 1
    assert result['requests_sent'] == (7 if failure in ('contaminated_once','foreign_live') else 6)
    assert all(body['priority'] == 0 for body in io.calls)
    assert result['attempts'] == (6 if failure == 'none' else 7)
    assert io.pauses == (0 if failure == 'none' else 1)
    qualified = json.loads((tmp_path/'smoke-input-r2-qualified.json').read_text())
    G.checked_spec(qualified)
    count = len(io.calls)
    with pytest.raises(ValueError, match='overwrite'):
        G.capture(io, inputs, {}, tmp_path/'other', lambda: {})
    assert len(io.calls) == count


@pytest.mark.parametrize('failure', ['busy','contaminated','drift','runtime'])
def test_reference_never_qualifies_bad_or_exhausted_collection(tmp_path, monkeypatch, failure):
    io, inputs = fixture(tmp_path, monkeypatch, failure)
    with pytest.raises(ValueError):
        G.capture(io, inputs, {}, tmp_path/'r2', lambda: {'changed':True} if failure == 'runtime' else {})
    assert not (tmp_path/'smoke-input-r2-qualified.json').exists()
    assert not (tmp_path/'r2/k1-request-receipt.json').exists()
    assert len(io.calls) == (0 if failure == 'busy' else 1 if failure == 'drift' else 6 if failure == 'runtime' else 10)
    if failure in ('busy','contaminated'):
        assert (tmp_path/'r2/checkpoint-10.json').exists()


def test_offline_qualification_rechecks_saved_bytes_without_sending(tmp_path, monkeypatch):
    io, inputs = fixture(tmp_path, monkeypatch)
    G.capture(io, inputs, {}, tmp_path/'r2', lambda: {})
    (tmp_path/'smoke-input-r2-qualified.json').unlink()
    (tmp_path/'r2/k1-request-receipt.json').unlink()
    (tmp_path/'r2/summary.json').unlink()
    def forbidden(*args, **kwargs): raise AssertionError('offline recovery sent network/process')
    monkeypatch.setattr(G.subprocess, 'Popen', forbidden)
    assert G.qualify_saved(inputs, tmp_path/'r2', {}, 'synthetic recovery')['qualified']
    assert len(io.calls) == 6
    (tmp_path/'r2/attempt-02/response.json').write_text('{}')
    with pytest.raises(ValueError, match='response changed'):
        G.qualify_saved(inputs, tmp_path/'r2', {})


def test_docker_mount_order_does_not_change_runtime_identity(tmp_path, monkeypatch):
    profiles = REPO/'offload/operations/profiles.v1.json'
    profile = next(p for p in json.loads(profiles.read_text())['profiles'] if p['name'] == 'K1-327680')
    env = [k+'='+v for k,v in profile['match']['env'].items() if v is not None]
    mounts = [{'Destination':'/z'},{'Destination':'/a'}]
    def output(argv, **kwargs):
        if argv[-1] == 'docker exec dsv41-serve cat /serve.sh':
            return '/srv/dsv41/legal-v2a --max-model-len 327680 --port 8000'
        mounts.reverse()
        return json.dumps([{'Image':G.IMAGE,'Id':'a'*64,'State':{'Running':True},
                            'Config':{'Env':env},'Mounts':mounts}])
    monkeypatch.setattr(G.subprocess,'check_output',output)
    assert G.runtime_identity(REPO) == G.runtime_identity(REPO)
