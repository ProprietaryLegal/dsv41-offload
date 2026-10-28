"""PP4-90f 0132: an HTTP 400 from the served API is never opaque again (CPU only, no host)."""
from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path

import pytest

from offload.gb10_budget import window_cells as wc
from offload.launch.windows import pp4_admission_requests as api
from offload.prefill_maxlen.evidence import Json

# The image's own 400 body for the 0132 request (sampling_params.py:1110-1115 via error_response.py:41-44).
IMAGE_400 = json.dumps({'error': {'message': 'The min_p and logit_bias sampling parameters are not yet supported '
    'with speculative decoding.', 'type': 'BadRequestError', 'param': None, 'code': 400}})


def refuse(request: urllib.request.Request, timeout: float = 0) -> object:
    raise urllib.error.HTTPError(request.full_url, 400, 'Bad Request', Message(), io.BytesIO(IMAGE_400.encode()))


def test_post_logs_and_carries_the_server_error_body(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(urllib.request, 'urlopen', refuse)
    body: dict[str, Json] = {'model': api.MODEL, 'prompt': 'public', 'max_tokens': 8, 'logit_bias': {'5': 100}}
    with pytest.raises(api.HttpRefused) as caught:
        api.Http('http://fixture.invalid').post('/v1/completions', body)
    assert caught.value.status == 400 and caught.value.endpoint == '/v1/completions'
    assert 'logit_bias' in caught.value.body and 'speculative decoding' in str(caught.value)
    assert isinstance(caught.value, ValueError)  # a STOP like every other failed assertion
    logged = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert logged['http_error'] == 400 and logged['body'] == IMAGE_400
    assert logged['request_settings'] == {'model': api.MODEL, 'max_tokens': 8, 'logit_bias': {'5': 100}}
    assert 'prompt' not in logged['request_settings']


def test_unreadable_error_body_is_named_never_dropped(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    class Broken(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes: raise ConnectionResetError('peer reset')
    def opener(request: urllib.request.Request, timeout: float = 0) -> object:
        raise urllib.error.HTTPError(request.full_url, 400, 'Bad Request', Message(), Broken())
    monkeypatch.setattr(urllib.request, 'urlopen', opener)
    with pytest.raises(api.HttpRefused, match='error body unreadable: ConnectionResetError'):
        api.Http('http://fixture.invalid').post('/tokenize', {'model': api.MODEL})
    assert 'unreadable' in capsys.readouterr().err


def test_cancel_stream_refusal_also_logs_the_body(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(urllib.request, 'urlopen', refuse)
    class Idle:
        def post(self, path: str, data: dict[str, Json]) -> dict[str, Json]: raise AssertionError('no post')
        def metrics(self) -> str: return 'vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n'
    with pytest.raises(api.HttpRefused, match='HTTP 400 Bad Request on POST /v1/chat/completions'):
        api.cancel('http://fixture.invalid', Idle(), time.time()+60)
    assert json.loads(capsys.readouterr().err.strip())['body'] == IMAGE_400


def test_main_uses_the_logging_client() -> None:
    assert issubclass(api.Http, wc.Http) and api.Http.post is not wc.Http.post
    source = Path(api.__file__).read_text()
    assert 'wc.Http(wc.URL)' not in source and 'api = Http(wc.URL)' in source
