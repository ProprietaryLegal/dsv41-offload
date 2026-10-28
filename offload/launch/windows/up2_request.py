"""One bounded UP2 public-corpus probe; private output contains hashes/numbers only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

from offload.prefill_maxlen.evidence import Json, number, record
from offload.prefill_trace.window_requests import corpus, exact_prompt, idle, stream_summary
from offload.launch.windows.up2_checks import CAP, require, screen_record

URL = 'http://127.0.0.1:18001'


class NeedleBuilder(Protocol):
    def build(self, ctx: int, n: int, seed: int) -> tuple[str, dict[str, str]]: ...


def needle(target: int, salt: str, output: Path, ledger: Path, deadline: float, data_dir: Path, *, screen: bool = False) -> None:
    def event(row: dict[str, Json]) -> None:
        write(ledger,screen_record(row,screen))
    from offload.kvpack.needle import Builder, score
    make = cast(Callable[[str | None, str | None, Path], NeedleBuilder], Builder)
    grade = cast(Callable[[str, dict[str,str], set[str]], dict[str,Json]], score)
    idle(URL)
    with urllib.request.urlopen(URL+'/v1/models',timeout=10) as response:
        models = cast(list[Json],record(cast(Json,json.load(response)))['data'])
    model = record(models[0])
    require(model.get('max_model_len') == CAP, 'needle cap mismatch')
    prompt,codes = make(URL,str(model['id']),data_dir).build(target,6,42+target)
    request_id = uuid.uuid4().hex
    body: dict[str,Json] = {'model':model['id'],'request_id':request_id,
        'messages':[{'role':'user','content':prompt}], 'max_tokens':300, 'temperature':0,'top_p':1,
        'seed':42,'priority':10,'cache_salt':salt,'chat_template_kwargs':{'thinking':False}}
    require(time.time() < deadline,f'needle submission deadline {deadline:.6f} reached')
    event({'event':'send_intent','id':request_id,'target':target,'kind':'needle','time':time.time()})
    try:
        req = urllib.request.Request(URL+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=900) as response:
            event({'event':'http_opened','id':request_id,'time':time.time()})
            result = record(cast(Json,json.load(response)))
        choice = record(cast(list[Json],result['choices'])[0])
        text = str(record(choice['message']).get('content') or '')
        foreign = set(re.findall(r'\b\d{6}\b',text))-set(codes.values())
        graded = grade(text,codes,foreign)
        require(graded.get('recall') == 6 and not graded.get('foreign_codes'),'needle6/6/foreign-code failure')
        usage = record(result['usage'])
        require(number(usage['prompt_tokens'])+300 <= CAP and
                record(usage['prompt_tokens_details']).get('cached_tokens') == 0,'needle length/cache failure')
        require(choice.get('finish_reason') == 'stop','needle did not complete normally')
        with output.open('x') as handle:
            json.dump(screen_record({'context':target,'recall':6,'of':6,'foreign_codes':0,'usage':usage},screen),handle)
            handle.write('\n')
        event({'event':'completed','id':request_id,'time':time.time()})
    except Exception as exc:
        event({'event':'failed','id':request_id,'error_class':type(exc).__name__,'time':time.time()})
        raise


def write(path: Path, row: dict[str, Json]) -> None:
    with path.open('a') as output:
        output.write(json.dumps(row, allow_nan=False) + '\n')
        output.flush()


def execute(target: int, budget: int, salt: str, output: Path, ledger: Path,
            deadline: float, data_dir: Path, *, numeric: bool = False, force_length: bool = False,
            rep: int = 0, kind: str = 'diagnostic', screen: bool = False) -> dict[str, Json]:
    def event(row: dict[str, Json]) -> None:
        write(ledger,screen_record(row,screen))
    require(0 < target and target + budget <= CAP, 'prompt/output exceeds low cap')
    require(not output.exists(), 'refusing existing request result')
    template: dict[str, Json] = {'thinking':False} if numeric else {'thinking':True, 'reasoning_effort':'high'}
    def post(endpoint: str, body: dict[str, Json]) -> dict[str, Json]:
        require(time.time() < deadline, f'submission deadline {deadline:.6f} reached')
        req = urllib.request.Request(URL+endpoint, json.dumps(body).encode(), {'Content-Type':'application/json'})
        with urllib.request.urlopen(req, timeout=min(900, max(1, deadline-time.time()))) as response:
            return record(cast(Json, json.load(response)))
    idle(URL)
    with urllib.request.urlopen(URL+'/v1/models', timeout=10) as response:
        model = record(cast(list[Json], record(cast(Json,json.load(response)))['data'])[0])
    require(model.get('max_model_len') == CAP and model.get('root') == '/srv/dsv41/legal-v2a', 'served target/cap mismatch')
    materials, hashes = corpus(data_dir)
    def count(text: str) -> int:
        return int(number(post('/tokenize', {'model':model['id'], 'messages':[{'role':'user','content':text}],
            'add_generation_prompt':True, 'chat_template_kwargs':template})['count']))
    prompt = exact_prompt(materials, target, count)
    request_id = uuid.uuid4().hex
    body: dict[str, Json] = {'model':model['id'], 'request_id':request_id,
        'messages':[{'role':'user','content':prompt}], 'max_tokens':budget,
        'temperature':0 if numeric else 1, 'top_p':1, 'seed':42, 'priority':10,
        'cache_salt':salt, 'chat_template_kwargs':template, 'ignore_eos':force_length,
        'stream':not numeric}
    if numeric:
        body.update({'logprobs':True, 'top_logprobs':20})
    else:
        body['stream_options'] = {'include_usage':True}
    require(time.time() < deadline, f'submission deadline {deadline:.6f} reached')
    req = urllib.request.Request(URL+'/v1/chat/completions', json.dumps(body).encode(), {'Content-Type':'application/json'})
    started = time.monotonic()
    event( {'event':'send_intent', 'id':request_id, 'target':target, 'kind':kind, 'time':time.time()})
    try:
        with urllib.request.urlopen(req, timeout=900) as response:
            event( {'event':'http_opened', 'id':request_id, 'time':time.time()})
            if numeric:
                result = record(cast(Json,json.load(response)))
                choice = record(cast(list[Json],result['choices'])[0])
                tokens = cast(list[Json],record(choice['logprobs'])['content'])
                require(bool(tokens), 'no numerical logprobs')
                top = cast(list[Json],record(tokens[0])['top_logprobs'])
                probabilities: dict[str, Json] = {}
                # vLLM may put the sampled token first or return21 entries. Order is immaterial.
                for item in top:
                    row = record(item)
                    key = hashlib.sha256(str(row['token']).encode()).hexdigest()
                    probabilities[key] = math.exp(number(row['logprob']))
                msg = record(choice['message'])
                summary: dict[str, Json] = {'usage':result['usage'], 'finish':choice['finish_reason'],
                    'probabilities':probabilities, 'text_sha256':hashlib.sha256(str(msg.get('content') or '').encode()).hexdigest(),
                    'total_s':time.monotonic()-started}
            else:
                summary = stream_summary(response, started)
        usage = record(summary['usage'])
        cached = record(usage.get('prompt_tokens_details'))['cached_tokens']
        require(usage.get('prompt_tokens') == target and cached == 0, 'exact-size/cold usage failure')
        require(number(usage['completion_tokens']) > 0 and summary.get('finish') in ('stop','length'), 'unfinished probe')
        if force_length:
            require(usage.get('completion_tokens') == budget and summary.get('finish') == 'length', 'near-cap/warmup length finish failure')
        row = {'window_id':str(output.parent.resolve()), 'type':'request', 'kind':kind, 'rep':rep, 'target_ctx':target, 'cap':CAP,
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
            'cache_salt_sha256':hashlib.sha256(salt.encode()).hexdigest(), 'corpus_sha256':hashes,
            'cached_tokens':cached, 'uncached_prompt_tokens':target,
            'sampling':{'temperature':0 if numeric else 1, 'top_p':1, 'seed':42, 'priority':10, 'max_tokens':budget},
            **summary}
        row = screen_record(row,screen)
        with output.open('x') as handle:
            json.dump(row, handle, allow_nan=False)
            handle.write('\n')
        event( {'event':'completed', 'id':request_id, 'time':time.time()})
        return row
    except Exception as exc:
        event( {'event':'failed', 'id':request_id, 'error_class':type(exc).__name__, 'time':time.time()})
        raise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--target', type=int, required=True)
    ap.add_argument('--budget', type=int, required=True)
    ap.add_argument('--salt', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--ledger', type=Path, required=True)
    ap.add_argument('--deadline', type=float, required=True)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--screen', action='store_true')
    ap.add_argument('--numeric', action='store_true')
    ap.add_argument('--needle', action='store_true')
    ap.add_argument('--force-length', action='store_true')
    ap.add_argument('--rep', type=int, default=0)
    ap.add_argument('--kind', default='diagnostic')
    args = ap.parse_args()
    if args.needle:
        needle(args.target,args.salt,args.output,args.ledger,args.deadline,args.data_dir,screen=args.screen)
    else:
        execute(args.target, args.budget, args.salt, args.output, args.ledger, args.deadline, args.data_dir,
                numeric=args.numeric, force_length=args.force_length, rep=args.rep, kind=args.kind, screen=args.screen)


if __name__ == '__main__':
    main()
