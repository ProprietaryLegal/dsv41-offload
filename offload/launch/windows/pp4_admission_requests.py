"""Public API qualification traffic; execute only through the owned admission HOLD.

Responses live only in memory. Receipts retain settings, hashes and explicit
assertions. A refused API feature, incomplete generation or missing EOS proof
fails the gate; no synthetic success, retries or prompt text in receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol, cast

from offload.gb10_budget import window_cells as wc
from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, number, record

MODEL = 'DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2'
SCHEMA: dict[str, Json] = {'type': 'object', 'properties': {'answer': {'type': 'string', 'enum': ['PP4']}},
                           'required': ['answer'], 'additionalProperties': False}
BODY_LIMIT = 16384  # bytes of a server error body kept in the log and the exception


class Api(Protocol):
    def post(self, path: str, data: dict[str, Json]) -> dict[str, Json]: ...
    def metrics(self) -> str: ...


class HttpRefused(ValueError):
    """The served API answered with an HTTP error; carries the server's own body."""
    def __init__(self, status: int, reason: str, endpoint: str, body: str) -> None:
        super().__init__(f'STOP: HTTP {status} {reason} on POST {endpoint}: {body}')
        self.status, self.reason, self.endpoint, self.body = status, reason, endpoint, body


def refused(error: urllib.error.HTTPError, endpoint: str, request: dict[str, Json]) -> HttpRefused:
    """Log the server's error body (never opaque) and return the exception to raise."""
    try:
        body = error.read(BODY_LIMIT).decode('utf-8', 'replace')
    except OSError as exc:  # the body itself could not be read: say so, never drop it silently
        body = f'<error body unreadable: {type(exc).__name__}: {exc}>'
    settings = {k: v for k, v in request.items() if k not in ('prompt', 'messages')}
    print(json.dumps({'http_error': error.code, 'reason': str(error.reason), 'endpoint': endpoint, 'body': body,
        'request_sha256': hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
        'request_settings': settings}, sort_keys=True), file=sys.stderr, flush=True)
    return HttpRefused(error.code, str(error.reason), endpoint, body)


class Http(wc.Http):
    """window_cells.Http whose HTTP errors always log and carry the server's body."""
    def post(self, endpoint: str, body: dict[str, Json]) -> dict[str, Json]:
        try:
            return super().post(endpoint, body)
        except urllib.error.HTTPError as error:
            raise refused(error, endpoint, body) from error


def idle(api: Api, deadline: float) -> None:
    while wc.busy(api.metrics()) != 0:
        ck.require(time.time() < deadline, 'cancel/recovery did not return to idle')
        time.sleep(.5)


GRAMMAR: dict[str, Json] = {'response_format': {'type': 'json_schema', 'json_schema': {'name': 'pp4', 'strict': True, 'schema': SCHEMA}}}
# The admission's chat probes, name -> (public prompt, request extras). semantics(), tools()
# and probe() send exactly these bodies; pp4_api_rehearsal rehearses them against production.
CHATS: dict[str, tuple[str, dict[str, Json]]] = {
    'grammar': ('Return a JSON object with answer set to PP4.', GRAMMAR),
    'penalties': ('Return a JSON object with answer set to PP4.',
                  {**GRAMMAR, 'presence_penalty': .7, 'frequency_penalty': .5, 'repetition_penalty': 1.1}),
    'json': ('Return JSON with answer set to PP4.', {'response_format': {'type': 'json_object'}}),
    'tool': ('Call report with answer PP4.', {
        'tools': [{'type': 'function', 'function': {'name': 'report', 'description': 'Report a public probe result.', 'parameters': SCHEMA}}],
        'tool_choice': {'type': 'function', 'function': {'name': 'report'}}}),
    'reasoning': ('What is 17 times 19? Think briefly, then give the number.',
                  {'max_tokens': 512, 'chat_template_kwargs': {'thinking': True, 'reasoning_effort': 'low'}}),
}
TOKENIZE: dict[str, Json] = {'model': MODEL, 'prompt': '!', 'add_special_tokens': False}
# Request fields the image refuses with HTTP 400 whenever speculative decoding is on,
# production and every PP4 arm alike (image vllm/sampling_params.py:1103-1116
# _validate_spec_decode, called from SamplingParams.verify :850 via
# v1/engine/input_processor.py:128; mapped to 400 by
# entrypoints/serve/exception_handling/error_response.py:41-44). PP4-90f cmd/0132.
SPEC_DECODE_REFUSED = ('logit_bias', 'min_p')


def sha(value: Json) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def chat_body(prompt: str, extra: dict[str, Json] | None = None) -> dict[str, Json]:
    return {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 1, 'top_p': 1, 'seed': 4104, 'max_tokens': 256, 'priority': 10,
        'chat_template_kwargs': {'thinking': False}, **(extra or {})}


def chat_evidence(body: dict[str, Json], response: dict[str, Json]) -> tuple[dict[str, Json], dict[str, Json]]:
    choices = ck.rows(response['choices'])
    ck.require(len(choices) == 1 and choices[0].get('finish_reason') in ('stop', 'tool_calls'),
               'API qualification incomplete/truncated')
    message = record(choices[0]['message'])
    evidence: dict[str, Json] = {'request_sha256': sha(cast(Json, body)), 'response_sha256': sha(cast(Json, response)),
        'finish_reason': choices[0]['finish_reason'], 'usage': response['usage']}
    return message, evidence


def chat(api: Api, prompt: str, extra: dict[str, Json] | None = None) -> tuple[dict[str, Json], dict[str, Json]]:
    body = chat_body(prompt, extra)
    return chat_evidence(body, api.post('/v1/chat/completions', body))


def check_chat(name: str, message: dict[str, Json]) -> None:
    """The exact public answer each chat probe must return."""
    if name in ('grammar', 'penalties'):
        ck.require(json.loads(str(message['content'])) == {'answer': 'PP4'}, name+' output violates exact grammar')
    elif name == 'json':
        ck.require(json.loads(str(message['content'])) == {'answer': 'PP4'}, 'JSON mode violated')
    elif name == 'tool':
        calls = ck.rows(message['tool_calls'])
        ck.require(len(calls) == 1, 'tool parser did not return one call')
        function = record(calls[0]['function'])
        ck.require(function.get('name') == 'report' and json.loads(str(function['arguments'])) == {'answer': 'PP4'}, 'tool parser content differs')
    elif name == 'reasoning':
        reasoning = message.get('reasoning_content', message.get('reasoning'))
        content = message.get('content')
        ck.require(isinstance(reasoning, str) and bool(reasoning.strip()) and isinstance(content, str)
                   and '323' in content and '<think>' not in content and '</think>' not in content, 'reasoning parser separation failed')
    else:
        ck.require(False, 'unknown chat probe '+name)


def stop_token(tokenized: dict[str, Json]) -> int:
    tokens = cast(list[Json], tokenized['tokens'])
    ck.require(len(tokens) == 1 and isinstance(tokens[0], int), 'single-token stop probe unavailable')
    return int(str(tokens[0]))


def terminator_bodies(stop: int, eos: int) -> tuple[tuple[str, dict[str, Json]], ...]:
    """Raw completion: an explicit stop string forced on the first token, then a
    separate actual model EOS token. Never confuse API length with EOS.

    The token is forced with allowed_token_ids (every other logit -inf in the V2
    sampler's LogitBiasState, applied to the target logits the speculative verifier
    uses: image v1/worker/gpu/sample/logit_bias.py:62-75, spec_decode/rejection_sampler.py:258),
    never logit_bias, which the image refuses under speculative decoding."""
    bodies: list[tuple[str, dict[str, Json]]] = []
    for name, token, settings in (('stop', stop, {'stop': ['!']}), ('eos', eos, {})):
        bodies.append((name, {'model': MODEL, 'prompt': 'Public arithmetic: 1 + 1 =', 'temperature': 1,
            'top_p': 1, 'max_tokens': 8, 'seed': 4104, 'allowed_token_ids': [token],
            'ignore_eos': False, **cast(dict[str, Json], settings)}))
    return tuple(bodies)


def check_terminator(name: str, eos: int, response: dict[str, Json]) -> dict[str, Json]:
    choice = ck.rows(response['choices'])[0]
    ck.require(choice.get('finish_reason') == 'stop', name+' failed to stop')
    ck.require(choice.get('text') == '' and number(record(response['usage'])['completion_tokens']) <= 1,
               name+' emitted content beyond forced terminator')
    if name == 'eos':
        ck.require(choice.get('stop_reason') in (None, eos), 'wrong EOS stop_reason')
    else:
        ck.require(choice.get('stop_reason') == '!', 'missing exact stop-string evidence')
    return choice


def semantics(api: Api, eos: int) -> dict[str, Json]:
    results: dict[str, Json] = {}
    for name in ('grammar', 'penalties'):
        prompt, extras = CHATS[name]
        message, evidence = chat(api, prompt, extras)
        check_chat(name, message)
        results[name] = {**evidence, 'exact': True, 'settings': cast(Json, extras)}
    for name, body in terminator_bodies(stop_token(api.post('/tokenize', TOKENIZE)), eos):
        response = api.post('/v1/completions', body)
        choice = check_terminator(name, eos, response)
        results[name] = {'exact': True, 'stop_reason': choice.get('stop_reason'),
            'request_sha256': sha(cast(Json, body)), 'response_sha256': sha(cast(Json, response))}
    return results


def tools(api: Api) -> dict[str, Json]:
    results: dict[str, Json] = {}
    for name in ('json', 'tool', 'reasoning'):
        message, evidence = chat(api, *CHATS[name])
        check_chat(name, message)
        results[name] = {**evidence, 'exact': True}
    return results


def probe(api: Api) -> dict[str, Json]:
    message, evidence = chat(api, *CHATS['grammar'])
    ck.require(json.loads(str(message['content'])) == {'answer': 'PP4'}, 'post-fault request invalid')
    return {**evidence, 'exact': True}


def cancel_body() -> dict[str, Json]:
    return {'model': MODEL, 'messages': [{'role': 'user', 'content': 'Count upwards from one, one number per line.'}],
        'temperature': 1, 'top_p': 1, 'max_tokens': 4096, 'ignore_eos': True, 'stream': True,
        'chat_template_kwargs': {'thinking': False}, 'priority': 10}


def cancel(url: str, api: Api, deadline: float) -> dict[str, Json]:
    body = cancel_body()
    request = urllib.request.Request(url+'/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    chunks = 0
    try:
        with urllib.request.urlopen(request, timeout=min(120, deadline-time.time())) as response:
            for line in response:
                ck.require(time.time() < deadline, 'cancel probe deadline')
                if line.startswith(b'data: {'):
                    chunk = record(cast(Json, json.loads(line[6:])))
                    choices = ck.rows(chunk.get('choices', []))
                    if choices and record(choices[0].get('delta', {})).get('content'):
                        chunks += 1
                        break  # context close is an actual client disconnect mid-generation
    except urllib.error.HTTPError as error:
        raise refused(error, '/v1/chat/completions', body) from error
    ck.require(chunks == 1, 'no actual streamed token before client cancel')
    idle(api, min(deadline, time.time()+60))
    return {'closed_after_content_chunk': True, 'idle_after_cancel': True, 'fresh_request': probe(api)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--case', choices=('semantics', 'tools', 'probe', 'cancel'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--eos', type=int, default=-1)
    args = parser.parse_args()
    if not args.execute:
        print('DRY: owned admission API qualification only; no request/file')
        return
    ck.require(os.environ.get('WINDOW_COORDINATOR') == 'Deepseek/coordinator', 'coordinator required')
    ck.require(not args.out.exists() and time.time()+120 < args.deadline, 'fresh evidence/deadline required')
    api = Http(wc.URL)
    idle(api, args.deadline)
    if args.case == 'semantics':
        ck.require(args.eos >= 0, 'verified model EOS token required')
        result = semantics(api, args.eos)
    elif args.case == 'tools': result = tools(api)
    elif args.case == 'cancel': result = cancel(wc.URL, api, args.deadline)
    else: result = probe(api)
    idle(api, args.deadline)
    args.out.write_text(json.dumps({'case': args.case, 'complete': True, 'checks': result}, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
