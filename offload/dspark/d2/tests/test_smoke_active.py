"""Exercise the actual concurrent CPU HTTP helper with synthetic in-process IO."""
from __future__ import annotations

from concurrent.futures import Future
import json
import threading
import time

import pytest

from d2 import smoke_active as A


def test_real_group_starts_three_controls_during_two_growing_captures(monkeypatch):
    ready = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    started = []
    info_bodies = []
    probes = []
    tagged_bodies = []
    class Response:
        def __init__(self, body): self.body = body
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            with lock:
                started.append(self.body)
                if len(started) == 2: ready.set()
            assert release.wait(3), 'informational group did not start'
            return json.dumps({'choices':[{'finish_reason':'length','token_ids':[20,21,22]}]}).encode()
        def __iter__(self):
            # One token-ID chunk is an output delivery; role-only/usage frames
            # are excluded. Keep tags alive until all three controls start.
            with lock:
                info_bodies.append(self.body)
                if len(info_bodies) == 3: release.set()
            for chunk in [
                {'prompt_token_ids':[10], 'choices':[{'delta':{'role':'assistant'}}]},
                {'choices':[{'token_ids':[20,21], 'delta':{'content':'syn'}}]},
                {'choices':[{'token_ids':[22], 'delta':{'content':'thetic'},'finish_reason':'stop'}]},
                {'choices':[], 'usage':{'prompt_tokens_details':{'cached_tokens':0}}},
            ]:
                time.sleep(.001)
                yield b'data: '+json.dumps(chunk).encode()+b'\n'
            yield b'data: [DONE]\n'
    def open_request(req, **kwargs):
        if isinstance(req,str):
            assert req.endswith('/metrics')
            class Metrics(Response):
                def read(self):
                    return b'vllm:num_requests_running{} 2\nvllm:num_requests_waiting{} 0\n'
            return Metrics({})
        body = json.loads(req.data)
        if not body.get('stream'): tagged_bodies.append(body)
        return Response(body)
    def probe(argv, **kwargs):
        assert ready.wait(3)
        assert argv[0] == 'ssh' and 'rank0user@10.202.28.2' in argv
        assert 'stat().st_size' in argv[-1]
        probes.append(argv)
        return json.dumps({'smoke0':len(probes)*100,'smoke1':len(probes)*100})
    monkeypatch.setattr(A.urllib.request,'urlopen',open_request)
    monkeypatch.setattr(A.subprocess,'check_output',probe)
    def forbidden(*args,**kwargs): raise AssertionError('real subprocess in synthetic test')
    monkeypatch.setattr(A.subprocess,'Popen',forbidden)
    requests = [{'endpoint':'/v1/chat/completions', 'body':{'request_id':'tag-'+str(i),
                 'temperature':0 if i==2 else 1, 'stream':False}} for i in range(3)]
    result = A.group('http://synthetic.invalid', requests, '/synthetic/capture')
    assert len(result['tagged']) == 2 and len(result['informational']) == 3
    assert result['gated'] is False
    assert len(probes) == 4
    assert tagged_bodies == [r['body'] for r in requests[:2]]
    assert all(b['stream'] is True and b['temperature'] == 0 for b in info_bodies)
    for row in result['informational']:
        assert row['both_tagged_inflight_at_start'] is True
        assert row['response']['choices'][0]['message']['content'] == 'synthetic'
        assert row['response']['choices'][0]['token_ids'] == [20,21,22]
        assert row['response']['prompt_token_ids'] == [10]
        assert row['latency']['delivery_chunks'] == 2
        assert row['latency']['ttft_s'] > 0 and row['latency']['delivery_chunk_mean_s'] > 0
    assert all(r['body']['stream'] is False for r in requests)


def test_stream_refuses_control_after_tagged_decoding_ended(monkeypatch):
    done = Future()
    done.set_result({})
    class Metrics:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self): return b'vllm:num_requests_running{} 2\nvllm:num_requests_waiting{} 0\n'
    def only_metrics(req, **kwargs):
        assert isinstance(req,str) and req.endswith('/metrics'), 'late control was sent'
        return Metrics()
    monkeypatch.setattr(A.urllib.request,'urlopen',only_metrics)
    with pytest.raises(ValueError,match='decoding ended'):
        A.streamed('http://synthetic.invalid', {'body':{},'endpoint':'/unused'}, [done])


def test_group_records_early_tag_completion_without_sending_controls(monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b'{"http_status":400,"error":"unsupported"}'
    def open_request(req, **kwargs):
        assert not json.loads(req.data).get('stream'), 'control sent after failed tag'
        return Response()
    monkeypatch.setattr(A.urllib.request,'urlopen',open_request)
    monkeypatch.setattr(A.subprocess,'check_output',lambda *args,**kwargs: '{"smoke0":0,"smoke1":0}')
    reqs=[{'body':{},'endpoint':'/unused'}]*3
    result = A.group('http://synthetic.invalid',reqs,'/synthetic')
    assert result['informational'] == [] and result['gated'] is False
    assert result['informational_status'].startswith('incomplete: tagged decoding ended')
    assert result['informational_medians'] == {'ttft_s': None, 'delivery_chunk_mean_s': None}


@pytest.mark.parametrize('running,waiting', [(1,0),(3,0),(2,1),(float('nan'),0)])
def test_active_control_refuses_wrong_or_foreign_scheduler_state(monkeypatch, running, waiting):
    class Metrics:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self):
            return f'vllm:num_requests_running{{}} {running}\nvllm:num_requests_waiting{{}} {waiting}\n'.encode()
    def only_metrics(req, **kwargs):
        assert isinstance(req,str) and req.endswith('/metrics'), 'control was sent'
        return Metrics()
    monkeypatch.setattr(A.urllib.request,'urlopen',only_metrics)
    pending = Future()
    with pytest.raises(ValueError):
        A.streamed('http://synthetic.invalid', {'body':{},'endpoint':'/unused'}, [pending,pending])
