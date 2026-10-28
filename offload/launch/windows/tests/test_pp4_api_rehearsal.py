"""Pre-window API rehearsal (CPU only): an image-like fake server, never a host."""
from __future__ import annotations

import ast
import io
import json
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path
from typing import cast

import pytest

from offload.launch.windows import pp4_admission_requests as api
from offload.launch.windows import pp4_api_rehearsal as r
from offload.prefill_maxlen.evidence import Json, record

EOS, STOP = 1, 7
SPEC_400 = 'The min_p and logit_bias sampling parameters are not yet supported with speculative decoding.'
IMAGE = Path('/opt/dsv41/data/src-image/vllm/sampling_params.py')
# The R16 (90f0278) stop request exactly as PP4-90f cmd/0132 sent it.
R16_STOP: dict[str, Json] = {'model': api.MODEL, 'prompt': 'Public arithmetic: 1 + 1 =', 'temperature': 1, 'top_p': 1,
    'max_tokens': 8, 'seed': 4104, 'logit_bias': {str(STOP): 100}, 'ignore_eos': False, 'stop': ['!']}


class Reply(io.BytesIO):
    status = 200


class Image:
    """The image's request validation with speculative decoding on (sampling_params.py:1103-1116)."""
    def __init__(self, model: str = api.MODEL, down: bool = False, spec: bool = True) -> None:
        self.model, self.down, self.spec = model, down, spec
        self.bodies: list[tuple[str, dict[str, Json]]] = []

    def __call__(self, request: urllib.request.Request, timeout: float = 0) -> Reply:
        assert timeout == r.TIMEOUT_S
        if self.down:
            raise urllib.error.URLError('connection refused')
        endpoint = request.full_url.removeprefix('http://production.fixture')
        body = record(cast(Json, json.loads(cast(bytes, request.data)))) if request.data else {}
        self.bodies.append((endpoint, body))
        if self.spec and (body.get('logit_bias') or float(str(body.get('min_p', 0))) > 0):
            error = json.dumps({'error': {'message': SPEC_400, 'type': 'BadRequestError', 'param': None, 'code': 400}})
            raise urllib.error.HTTPError(request.full_url, 400, 'Bad Request', Message(), io.BytesIO(error.encode()))
        if endpoint == '/v1/models':
            return Reply(json.dumps({'data': [{'id': self.model}]}).encode())
        if endpoint == '/tokenize':
            return Reply(json.dumps({'tokens': [STOP]}).encode())
        if endpoint == '/v1/completions':
            forced = cast(list[int], body.get('allowed_token_ids') or [int(k) for k in record(body['logit_bias'])])
            assert forced in ([STOP], [EOS]) and int(str(body['max_tokens'])) <= r.MAX_TOKENS
            reason: Json = '!' if forced == [STOP] else None
            return Reply(json.dumps({'choices': [{'text': '', 'finish_reason': 'stop', 'stop_reason': reason}],
                                     'usage': {'completion_tokens': 1}}).encode())
        assert int(str(body['max_tokens'])) <= r.MAX_TOKENS and body.get('priority') == 10
        if body.get('stream'):
            return Reply(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\ndata: {"choices":[{"delta":{"content":"1"}}]}\n\n')
        message: dict[str, Json] = {'content': '{"answer": "PP4"}'}
        finish = 'stop'
        if 'tools' in body:
            message = {'content': None, 'tool_calls': [{'function': {'name': 'report', 'arguments': '{"answer": "PP4"}'}}]}
        elif record(body['chat_template_kwargs']).get('thinking') is True:
            message, finish = {'content': None, 'reasoning': 'Seventeen'}, 'length'
        return Reply(json.dumps({'choices': [{'message': message, 'finish_reason': finish}],
                                 'usage': {'completion_tokens': 7}}).encode())


def execute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, image: Image) -> tuple[int, dict[str, Json]]:
    monkeypatch.setattr(urllib.request, 'urlopen', image)
    monkeypatch.setenv('WINDOW_COORDINATOR', 'Deepseek/coordinator')
    monkeypatch.setattr(r, 'source', lambda: {'commit': 'f'*40, 'dirty': False, 'sha256': {}})
    code = r.main(['--execute', '--url', 'http://production.fixture', '--eos', str(EOS), '--out', str(tmp_path/'rehearsal')])
    return code, record(cast(Json, json.loads((tmp_path/'rehearsal'/'rehearsal.json').read_text())))


def test_dry_run_never_sends_and_every_probe_is_tiny(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def forbidden(*args: object, **kwargs: object) -> object: raise AssertionError('request in dry run')
    monkeypatch.setattr(urllib.request, 'urlopen', forbidden)
    assert r.main(['--eos', str(EOS), '--out', '/nonexistent/never-written']) == 0
    plan = record(cast(Json, json.loads(capsys.readouterr().out)))
    probes = [record(p) for p in cast(list[Json], plan['probes'])]
    assert [p['name'] for p in probes] == ['grammar', 'penalties', 'tokenize', 'stop', 'eos', r.CONTROL,
                                           'json', 'tool', 'reasoning', 'cancel']
    assert all(int(str(record(p['settings']).get('max_tokens', 0))) <= r.MAX_TOKENS for p in probes)
    assert plan['url'] == r.PRODUCTION == 'http://HEAD_IP:8000'


def test_rehearsal_sends_the_admissions_own_bodies() -> None:
    for name, endpoint, mode, body in r.plan(EOS, STOP):
        if name in ('stop', 'eos'):
            assert mode == 'exact' and body == dict(api.terminator_bodies(STOP, EOS))[name]
        elif name == r.CONTROL:
            assert mode == 'control' and api.sha(cast(Json, body)) == api.sha(cast(Json, R16_STOP))
        elif name == 'tokenize':
            assert body == api.TOKENIZE
        elif name == 'cancel':
            assert body == {**api.cancel_body(), 'max_tokens': r.MAX_TOKENS}
        else:
            assert mode == 'capped' and body == {**api.chat_body(*api.CHATS[name]), 'max_tokens': r.MAX_TOKENS}


def test_needs_the_coordinator_and_a_fresh_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('WINDOW_COORDINATOR', raising=False)
    with pytest.raises(ValueError, match='coordinator required'):
        r.main(['--execute', '--eos', '1', '--out', str(tmp_path/'x')])
    monkeypatch.setenv('WINDOW_COORDINATOR', 'Deepseek/coordinator')
    with pytest.raises(ValueError, match='fresh rehearsal directory'):
        r.main(['--execute', '--eos', '1', '--out', str(tmp_path)])


def test_production_down_is_no_answer_never_a_probe_verdict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code, result = execute(tmp_path, monkeypatch, Image(down=True))
    assert code == 1 and result['refused'] == [] and 'models' in cast(list[Json], result['no_answer'])
    assert any('not a probe verdict' in str(line) for line in cast(list[Json], result['reading']))


def test_wrong_served_model_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code, result = execute(tmp_path, monkeypatch, Image(model='other'))
    assert code == 1 and 'models' in cast(list[Json], result['semantic_failures'])


def test_responses_are_held_privately(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    execute(tmp_path, monkeypatch, Image())
    held = tmp_path/'rehearsal'/'responses.jsonl'
    assert held.stat().st_mode & 0o777 == 0o600
    assert [record(cast(Json, json.loads(line)))['name'] for line in held.read_text().splitlines()][:3] == ['grammar', 'penalties', 'tokenize']


def test_fixed_probes_pass_and_the_r16_control_confirms_0132(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code, result = execute(tmp_path, monkeypatch, Image())
    assert code == 0 and result['verdict'] == 'PASS' and result['cause_confirmed'] is True and result['refused'] == []
    rows = {str(record(p)['name']): record(p) for p in cast(list[Json], result['probes'])}
    assert all(rows[n]['check'] == 'pass' for n in ('models', 'grammar', 'penalties', 'tokenize', 'stop', 'eos', 'json', 'tool', 'cancel'))
    assert rows['stop']['stop_reason'] == '!' and rows['eos']['stop_reason'] is None
    assert rows[r.CONTROL]['status'] == 400 and SPEC_400 in str(rows[r.CONTROL]['error_body'])
    assert str(rows['reasoning']['check']).startswith('not-run: capped')


def test_control_accepted_says_cause_not_reproduced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code, result = execute(tmp_path, monkeypatch, Image(spec=False))
    assert code == 0 and result['cause_confirmed'] is False
    assert any('CAUSE NOT REPRODUCED' in str(line) for line in cast(list[Json], result['reading']))


def test_a_refused_probe_reads_probe_wrong_for_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, 'terminator_bodies', lambda stop, eos: (('stop', R16_STOP), ('eos', {**R16_STOP, 'stop': []})))
    code, result = execute(tmp_path, monkeypatch, Image())
    assert code == 1 and result['refused'] == ['stop', 'eos']
    assert str(cast(list[Json], result['reading'])[0]).startswith('PROBE WRONG FOR THIS IMAGE')


def test_no_admission_probe_carries_a_field_the_image_refuses_under_spec_decode() -> None:
    bodies = [api.chat_body(*api.CHATS[n]) for n in api.CHATS] + [b for _, b in api.terminator_bodies(STOP, EOS)]
    for body in [*bodies, api.cancel_body(), api.TOKENIZE]:
        assert not body.get('logit_bias') and float(str(body.get('min_p', 0))) == 0
    assert [b['allowed_token_ids'] for _, b in api.terminator_bodies(STOP, EOS)] == [[STOP], [EOS]]


@pytest.mark.skipif(not IMAGE.exists(), reason='image source copy only on Controller')
def test_refused_fields_match_the_image_spec_decode_check() -> None:
    tree = ast.parse(IMAGE.read_text())
    check = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_validate_spec_decode')
    branch = next(n for n in check.body if isinstance(n, ast.If) and isinstance(n.test, ast.BoolOp))
    fields = {n.attr for n in ast.walk(branch.test) if isinstance(n, ast.Attribute)}
    assert fields == set(api.SPEC_DECODE_REFUSED)
    assert any(isinstance(n, ast.Raise) for n in ast.walk(branch))
