"""PF1-CAUSE2 matched-window output checks (coordinator only; dry plan by default). Hashes and numbers only, never text.

After an arm's timed requests: the six W5 greedy prompts (W5/greedy_check.py body: temperature 0, thinking off, 300
tokens) -> per prompt the SHA-256 of the answer and token counts; then an exact 8,192-token public-corpus prompt, cold
(fresh cache salt), one greedy token with the top-20 log-probabilities -> the probabilities keyed by token hash. The
off arm sends the logit prompt twice under two salts (the A-A floor), the fix arm once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import urllib.request
import uuid
from pathlib import Path
from typing import cast

from offload.prefill_maxlen.evidence import Json, number, record
from offload.prefill_trace import window_requests as wr

URL = 'http://127.0.0.1:18001'
# Byte-identical to /opt/dsv41/campaigns/DSV41-OFFLOAD/W5/greedy_check.py PROMPTS (test_match.py checks it).
# Public release: the six greedy reference prompts are not published (data boundary). Provide them as a JSON list of
# strings in the file named by DSV41_GREEDY_PROMPTS_FILE.
def _load_greedy_prompts() -> list[str]:
    import json as _json, os as _os
    path = _os.environ.get("DSV41_GREEDY_PROMPTS_FILE")
    if not path:
        return []
    with open(path, encoding="utf-8") as fh:
        return [str(x) for x in _json.load(fh)]


PROMPTS = _load_greedy_prompts()
GREEDY_TOKENS = 300
LOGIT_TARGET = 8192
TEMPLATE: dict[str, Json] = {'thinking': False}
SALTS = {'off': ('a', 'aa'), 'fix': ('b',)}


def greedy_body(model: Json, prompt: str) -> dict[str, Json]:
    return {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0,
            'max_tokens': GREEDY_TOKENS, 'chat_template_kwargs': TEMPLATE}


def logit_body(model: Json, prompt: str, salt: str) -> dict[str, Json]:
    return {'model': model, 'request_id': uuid.uuid4().hex, 'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 1, 'temperature': 0, 'top_p': 1, 'seed': 42, 'priority': 10, 'cache_salt': salt,
            'chat_template_kwargs': TEMPLATE, 'logprobs': True, 'top_logprobs': 20}


def greedy_row(i: int, result: dict[str, Json]) -> dict[str, Json]:
    choice = record(cast(list[Json], result['choices'])[0])
    text = str(record(choice['message']).get('content') or '')
    usage = record(result['usage'])
    return {'type': 'greedy', 'i': i, 'text_sha256': hashlib.sha256(text.encode()).hexdigest(), 'chars': len(text),
            'prompt_tokens': usage.get('prompt_tokens'), 'completion_tokens': usage.get('completion_tokens'),
            'finish': choice.get('finish_reason')}


def logit_row(role: str, prompt: str, salt: str, result: dict[str, Json]) -> dict[str, Json]:
    choice = record(cast(list[Json], result['choices'])[0])
    tokens = cast(list[Json], record(choice['logprobs'])['content'])
    if not tokens:
        raise ValueError('no numerical logprobs')
    probabilities: dict[str, Json] = {}
    for item in cast(list[Json], record(tokens[0])['top_logprobs']):
        row = record(item)
        probabilities[hashlib.sha256(str(row['token']).encode()).hexdigest()] = math.exp(number(row['logprob']))
    usage = record(result['usage'])
    cached = record(usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    if usage.get('prompt_tokens') != LOGIT_TARGET or cached != 0:
        raise ValueError('logit probe is not an exact cold 8,192-token prompt')
    return {'type': 'logit', 'role': role, 'salt_suffix': salt.rsplit(':', 1)[1], 'target_ctx': LOGIT_TARGET,
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'cache_salt_sha256': hashlib.sha256(salt.encode()).hexdigest(), 'cached_tokens': cached,
            'usage': cast(Json, usage), 'finish': choice.get('finish_reason'), 'probabilities': cast(Json, probabilities)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--url', default=URL)
    ap.add_argument('--cap', type=int, choices=(524288,), required=True)
    ap.add_argument('--role', choices=tuple(SALTS), required=True)
    ap.add_argument('--cache-salt-prefix', required=True)
    ap.add_argument('--deadline-utc', type=float, default=0)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--ledger', type=Path)
    ap.add_argument('--data-dir', type=Path, default=Path('/opt/dsv41/data/bench-data'))
    args = ap.parse_args()
    if not args.execute:
        print(json.dumps({'executed': False, 'role': args.role, 'greedy_prompts': len(PROMPTS),
                          'logit_requests': len(SALTS[args.role]), 'logit_target': LOGIT_TARGET}))
        return
    if args.deadline_utc <= time.time():
        ap.error('--execute requires a future --deadline-utc')
    if args.url != URL:
        ap.error('coordinator must use the reserved experiment tunnel at 127.0.0.1:18001')
    if args.output.exists():
        ap.error('refusing an existing output')
    wr.idle(args.url)
    with urllib.request.urlopen(args.url + '/v1/models', timeout=10) as response:
        models = cast(list[Json], record(cast(Json, json.load(response)))['data'])
    if len(models) != 1 or record(models[0]).get('max_model_len') != args.cap:
        raise ValueError('served target/cap differs from the arm')
    model = record(models[0])['id']

    def event(row: dict[str, Json]) -> None:
        if args.ledger is not None:
            with args.ledger.open('a') as handle:
                handle.write(json.dumps({**row, 'time': time.time()}) + '\n')

    def send(kind: str, body: dict[str, Json]) -> dict[str, Json]:
        if time.time() >= args.deadline_utc:
            raise RuntimeError('window submission deadline reached; checks incomplete, restore production')
        wr.idle(args.url)
        ident = str(body.get('request_id') or uuid.uuid4().hex)
        event({'event': 'send_intent', 'id': ident, 'kind': kind})
        try:
            result = wr.post(args.url, '/v1/chat/completions', body)
        except BaseException as exc:
            event({'event': 'failed', 'id': ident, 'error_class': type(exc).__name__})
            raise
        event({'event': 'completed', 'id': ident})
        return result

    with args.output.open('x') as output:
        output.write(json.dumps({'type': 'identity', 'role': args.role, 'cap': args.cap, 'model': model}) + '\n')
        for i, prompt in enumerate(PROMPTS):
            output.write(json.dumps(greedy_row(i, send('greedy', greedy_body(model, prompt))), allow_nan=False) + '\n')
            output.flush()
        materials, hashes = wr.corpus(args.data_dir)

        def count(text: str) -> int:
            result = wr.post(args.url, '/tokenize', {'model': model, 'messages': [{'role': 'user', 'content': text}],
                                                     'add_generation_prompt': True, 'chat_template_kwargs': TEMPLATE})
            return int(number(result['count']))
        prompt = wr.exact_prompt(materials, LOGIT_TARGET, count)
        for suffix in SALTS[args.role]:
            salt = f'{args.cache_salt_prefix}:logit:{suffix}'
            row = logit_row(args.role, prompt, salt, send('logit', logit_body(model, prompt, salt)))
            output.write(json.dumps({**row, 'corpus_sha256': cast(Json, hashes)}, allow_nan=False) + '\n')
            output.flush()


if __name__ == '__main__':
    main()
