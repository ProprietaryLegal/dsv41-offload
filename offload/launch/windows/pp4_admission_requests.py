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
import time
import urllib.request
from pathlib import Path
from typing import Protocol, cast

from offload.gb10_budget import window_cells as wc
from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, number, record

MODEL = 'DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2'
SCHEMA: dict[str, Json] = {'type': 'object', 'properties': {'answer': {'type': 'string', 'enum': ['PP4']}},
                           'required': ['answer'], 'additionalProperties': False}


class Api(Protocol):
    def post(self, path: str, data: dict[str, Json]) -> dict[str, Json]: ...
    def metrics(self) -> str: ...


def idle(api: Api, deadline: float) -> None:
    while wc.busy(api.metrics()) != 0:
        ck.require(time.time() < deadline, 'cancel/recovery did not return to idle')
        time.sleep(.5)


def chat(api: Api, prompt: str, extra: dict[str, Json] | None = None) -> tuple[dict[str, Json], dict[str, Json]]:
    body: dict[str, Json] = {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 1, 'top_p': 1, 'seed': 4104, 'max_tokens': 256, 'priority': 10,
        'chat_template_kwargs': {'thinking': False}, **(extra or {})}
    response = api.post('/v1/chat/completions', body)
    choices = ck.rows(response['choices'])
    ck.require(len(choices) == 1 and choices[0].get('finish_reason') in ('stop', 'tool_calls'),
               'API qualification incomplete/truncated')
    message = record(choices[0]['message'])
    evidence: dict[str, Json] = {'request_sha256': hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
        'response_sha256': hashlib.sha256(json.dumps(response, sort_keys=True).encode()).hexdigest(),
        'finish_reason': choices[0]['finish_reason'], 'usage': response['usage']}
    return message, evidence


def semantics(api: Api, eos: int) -> dict[str, Json]:
    results: dict[str, Json] = {}
    for name, extras in (
        ('grammar', {'response_format': {'type': 'json_schema', 'json_schema': {'name': 'pp4', 'strict': True, 'schema': SCHEMA}}}),
        ('penalties', {'response_format': {'type': 'json_schema', 'json_schema': {'name': 'pp4', 'strict': True, 'schema': SCHEMA}},
                       'presence_penalty': .7, 'frequency_penalty': .5, 'repetition_penalty': 1.1})):
        message, evidence = chat(api, 'Return a JSON object with answer set to PP4.', cast(dict[str, Json], extras))
        ck.require(json.loads(str(message['content'])) == {'answer': 'PP4'}, name+' output violates exact grammar')
        results[name] = {**evidence, 'exact': True, 'settings': cast(Json, extras)}
    # Raw completion: an explicit stop string forced on the first token, then a
    # separate actual model EOS token. Never confuse API length with EOS.
    tokenized = api.post('/tokenize', {'model': MODEL, 'prompt': '!', 'add_special_tokens': False})
    tokens = cast(list[Json], tokenized['tokens'])
    ck.require(len(tokens) == 1 and isinstance(tokens[0], int), 'single-token stop probe unavailable')
    terminators: tuple[tuple[str, int, dict[str, Json]], ...] = (('stop', int(str(tokens[0])), {'stop': ['!']}), ('eos', eos, {}))
    for name, token, terminator_settings in terminators:
        body: dict[str, Json] = {'model': MODEL, 'prompt': 'Public arithmetic: 1 + 1 =', 'temperature': 1,
            'top_p': 1, 'max_tokens': 8, 'seed': 4104, 'logit_bias': {str(token): 100},
            'ignore_eos': False, **terminator_settings}
        response = api.post('/v1/completions', body)
        choice = ck.rows(response['choices'])[0]
        ck.require(choice.get('finish_reason') == 'stop', name+' failed to stop')
        ck.require(choice.get('text') == '' and number(record(response['usage'])['completion_tokens']) <= 1,
                   name+' emitted content beyond forced terminator')
        if name == 'eos':
            ck.require(choice.get('stop_reason') in (None, eos), 'wrong EOS stop_reason')
        else:
            ck.require(choice.get('stop_reason') == '!', 'missing exact stop-string evidence')
        results[name] = {'exact': True, 'stop_reason': choice.get('stop_reason'),
            'request_sha256': hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
            'response_sha256': hashlib.sha256(json.dumps(response, sort_keys=True).encode()).hexdigest()}
    return results


def tools(api: Api) -> dict[str, Json]:
    results: dict[str, Json] = {}
    message, evidence = chat(api, 'Return JSON with answer set to PP4.', {'response_format': {'type': 'json_object'}})
    ck.require(json.loads(str(message['content'])) == {'answer': 'PP4'}, 'JSON mode violated')
    results['json'] = {**evidence, 'exact': True}
    message, evidence = chat(api, 'Call report with answer PP4.', {
        'tools': [{'type': 'function', 'function': {'name': 'report', 'description': 'Report a public probe result.', 'parameters': SCHEMA}}],
        'tool_choice': {'type': 'function', 'function': {'name': 'report'}}})
    calls = ck.rows(message['tool_calls'])
    ck.require(len(calls) == 1, 'tool parser did not return one call')
    function = record(calls[0]['function'])
    ck.require(function.get('name') == 'report' and json.loads(str(function['arguments'])) == {'answer': 'PP4'}, 'tool parser content differs')
    results['tool'] = {**evidence, 'exact': True}
    message, evidence = chat(api, 'What is 17 times 19? Think briefly, then give the number.',
        {'max_tokens': 512, 'chat_template_kwargs': {'thinking': True, 'reasoning_effort': 'low'}})
    reasoning = message.get('reasoning_content', message.get('reasoning'))
    content = message.get('content')
    ck.require(isinstance(reasoning, str) and bool(reasoning.strip()) and isinstance(content, str)
               and '323' in content and '<think>' not in content and '</think>' not in content, 'reasoning parser separation failed')
    results['reasoning'] = {**evidence, 'exact': True}
    return results


def probe(api: Api) -> dict[str, Json]:
    message, evidence = chat(api, 'Return a JSON object with answer set to PP4.', {
        'response_format': {'type': 'json_schema', 'json_schema': {'name': 'pp4', 'strict': True, 'schema': SCHEMA}}})
    ck.require(json.loads(str(message['content'])) == {'answer': 'PP4'}, 'post-fault request invalid')
    return {**evidence, 'exact': True}


def cancel(url: str, api: Api, deadline: float) -> dict[str, Json]:
    body: dict[str, Json] = {'model': MODEL, 'messages': [{'role': 'user', 'content': 'Count upwards from one, one number per line.'}],
        'temperature': 1, 'top_p': 1, 'max_tokens': 4096, 'ignore_eos': True, 'stream': True,
        'chat_template_kwargs': {'thinking': False}, 'priority': 10}
    request = urllib.request.Request(url+'/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    chunks = 0
    with urllib.request.urlopen(request, timeout=min(120, deadline-time.time())) as response:
        for line in response:
            ck.require(time.time() < deadline, 'cancel probe deadline')
            if line.startswith(b'data: {'):
                chunk = record(cast(Json, json.loads(line[6:])))
                choices = ck.rows(chunk.get('choices', []))
                if choices and record(choices[0].get('delta', {})).get('content'):
                    chunks += 1
                    break  # context close is an actual client disconnect mid-generation
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
    api = wc.Http(wc.URL)
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
