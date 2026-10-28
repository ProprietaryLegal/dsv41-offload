"""Coordinator-only public prompt matrix. Dry plan by default; never save text."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.request
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import cast

from offload.prefill_maxlen.evidence import Json, number, record

TEMPLATE: dict[str, Json] = {'thinking': True, 'reasoning_effort': 'high'}
WARMUP, WARMUP_TOKENS = 32000, 64
HEADER = ('PUBLIC MATERIALS BENCHMARK. Summarize these public documents, compare recurring themes, '
          'and identify inconsistencies using only the materials. Do not invent sources.\n\n')


def post(url: str, endpoint: str, body: dict[str, Json]) -> dict[str, Json]:
    req = urllib.request.Request(url + endpoint, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as response:
        return record(cast(Json, json.load(response)))


def corpus(root: Path) -> tuple[str, dict[str, Json]]:
    docs: list[str] = []
    hashes: dict[str, Json] = {}
    for name in ('opinions_cl.jsonl', 'filings_recap.jsonl'):
        path = root / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            row = record(cast(Json, json.loads(line)))
            text = row.get('text')
            if not isinstance(text, str):
                raise ValueError(f'{name}: public record missing text')
            docs.append(text)
    random.Random(42).shuffle(docs)
    return HEADER + '\n\n'.join(docs), hashes


def exact_cut(materials: str, target: int, count: Callable[[str], int]) -> tuple[int, int]:
    """(characters of the shared family, appended ' x' pads) whose complete chat is exactly target tokens."""
    # All lengths draw from the same family/order; only the truncation boundary differs.
    lo, hi = len(HEADER), len(materials)
    best = 0
    if count(materials) < target:
        raise ValueError(f'public corpus too short for {target}; do not repeat filler')
    while lo <= hi:
        mid = (lo + hi) // 2
        n = count(materials[:mid])
        if n == target:
            return mid, 0
        if n < target:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    # Token boundaries can skip an integer under character bisection. Bound correction.
    for pads in range(32):
        n = count(render(materials, best, pads))
        if n == target:
            return best, pads
        if n > target or not best:
            break
    raise ValueError(f'cannot construct exact {target}-token chat; no approximate 500K claim')


def render(materials: str, chars: int, pads: int) -> str:
    return materials[:chars] + ' x' * pads


def exact_prompt(materials: str, target: int, count: Callable[[str], int]) -> str:
    return render(materials, *exact_cut(materials, target, count))


def reused_cuts(path: Path) -> dict[int, tuple[int, int, str]]:
    """Numeric cuts from an earlier arm of the SAME window; each is re-hashed and recounted before use."""
    cuts: dict[int, tuple[int, int, str]] = {}
    for line in path.read_text().splitlines():
        row = record(cast(Json, json.loads(line)))
        if row.get('type') != 'request':
            continue
        cut = (int(number(row['prompt_chars'])), int(number(row['prompt_pad_x'])), str(row['prompt_sha256']))
        if cuts.setdefault(int(number(row['target_ctx'])), cut) != cut:
            raise ValueError('earlier arm used two different prompts for one target')
    return cuts


def cache_salt(prefix: str | None, target: int, rep: int, mode: str) -> str:
    """Deterministic coordinator pairing; default retains fresh random salts."""
    if prefix is None:
        return uuid.uuid4().hex
    if not prefix or mode not in ('main', 'partial', 'warmup') or target < 1 or rep < 0:
        raise ValueError('invalid fixed cache salt identity')
    return f'{prefix}:{target}:{rep}:{mode}'


def stream_summary(lines: Iterable[bytes], started: float) -> dict[str, Json]:
    first: dict[str, Json] = {}
    last: dict[str, Json] = {}
    events: list[Json] = []
    usage: dict[str, Json] | None = None
    finish: Json = None
    done = False
    response_id: Json = None
    for raw in lines:
        if time.monotonic() - started > 900:
            raise RuntimeError('request exceeded 900-second wall budget; coordinator cancellation boundary required')
        if not raw.startswith(b'data:'):
            continue
        payload = raw[5:].strip()
        if payload == b'[DONE]':
            done = True
            break
        obj = record(cast(Json, json.loads(payload)))
        if obj.get('error'):
            raise RuntimeError('server returned stream error; inspect coordinator server logs')
        response_id = obj.get('id', response_id)
        if obj.get('usage'):
            usage = record(obj['usage'])
        pieces = False
        choices = obj.get('choices', [])
        if not isinstance(choices, list):
            raise ValueError('choices must be a list')
        for choice in choices:
            row = record(choice)
            delta = record(row.get('delta') or {})
            for kind, value in (('reasoning', delta.get('reasoning_content') or delta.get('reasoning')),
                                ('answer', delta.get('content'))):
                if value:
                    elapsed = time.monotonic() - started
                    first.setdefault(kind, elapsed)
                    last[kind] = elapsed
                    pieces = True
            finish = row.get('finish_reason') or finish
        if pieces:
            events.append((time.monotonic() - started) * 1000)
    total = time.monotonic() - started
    if not done or usage is None or finish is None:
        raise RuntimeError('incomplete stream: missing DONE, usage or finish')
    details = record(usage.get('completion_tokens_details') or {})
    reason = details.get('reasoning_tokens')
    answer = number(usage['completion_tokens']) - number(reason) if reason is not None else None
    return {'response_id': response_id, 'usage': usage, 'finish': finish,
            'ttft_s': min(number(v) for v in first.values()) if first else None,
            'total_s': total, 'first_by_kind_s': first, 'last_by_kind_s': last,
            'reasoning_tokens': reason, 'answer_and_control_tokens': answer,
            'sse_times_ms': events}


def idle(url: str) -> None:
    with urllib.request.urlopen(url + '/metrics', timeout=10) as response:
        lines = response.read().decode().splitlines()
    for name in ('vllm:num_requests_running', 'vllm:num_requests_waiting'):
        values = [float(line.split()[-1]) for line in lines if line.startswith(name + '{') or line.startswith(name + ' ')]
        if not values or sum(values) != 0:
            raise RuntimeError(f'serve not verified idle: {name}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--url', default='http://127.0.0.1:18001')
    ap.add_argument('--cap', type=int, choices=(327680, 524288), required=True)
    ap.add_argument('--contexts', type=int, nargs='+', default=[150000, 300000, 400000, 450000, 500000])
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--modes', nargs='+', choices=('cold', 'full', 'partial'), default=['cold', 'full', 'partial'])
    ap.add_argument('--data-dir', type=Path, default=Path('/opt/dsv41/data/bench-data'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--trace-run', required=True)
    ap.add_argument('--cache-salt-prefix', help='same fixed value across restarted paired arms; cold usage must be zero')
    ap.add_argument('--deadline-utc', type=float, default=0, help='shared absolute window submission deadline')
    ap.add_argument('--warmup', action='store_true', help='first one discarded exact-32000 cold request, 64 tokens, EOS ignored')
    ap.add_argument('--reuse-cuts', type=Path, help="earlier arm's output in this window: numeric cuts, re-hashed and recounted")
    ap.add_argument('--ledger', type=Path, help='append send/open/complete/fail events: ids, kinds, sizes and times only')
    args = ap.parse_args()
    if args.reps < 1 or any(n < 100001 or n > 500000 or n + 1500 > args.cap for n in args.contexts):
        ap.error('choose in-scope points with 1500 output reserve; partial suffix is 100000 tokens')
    if not args.execute:
        print(json.dumps({'executed': False, 'cap': args.cap, 'points': args.contexts,
                          'reps': args.reps, 'modes': args.modes, 'output_budget': 1500, 'warmup': args.warmup,
                          'fixed_cache_salts': args.cache_salt_prefix is not None}))
        return
    if args.deadline_utc <= time.time():
        ap.error('--execute requires a future --deadline-utc; stop submissions at the window boundary')
    if args.url != 'http://127.0.0.1:18001':
        ap.error('coordinator must use the reserved experiment tunnel at 127.0.0.1:18001')
    if args.output.exists():
        ap.error('refusing an existing output; use a fresh window directory')
    idle(args.url)
    with urllib.request.urlopen(args.url + '/v1/models', timeout=10) as response:
        info = record(cast(Json, json.load(response)))
    models = info['data']
    if not isinstance(models, list) or len(models) != 1:
        raise ValueError('expected one served target')
    model = record(models[0])
    if model.get('max_model_len') != args.cap:
        raise ValueError('served cap differs from requested arm')
    model_id = model['id']
    materials, hashes = corpus(args.data_dir)
    def count(text: str) -> int:
        result = post(args.url, '/tokenize', {'model': model_id, 'messages': [{'role': 'user', 'content': text}],
            'add_generation_prompt': True, 'chat_template_kwargs': TEMPLATE})
        return int(number(result['count']))
    cuts = reused_cuts(args.reuse_cuts) if args.reuse_cuts is not None else {}
    def prompt_for(target: int) -> tuple[str, int, int]:
        if target in cuts:
            chars, pads, digest = cuts[target]
            prompt = render(materials, chars, pads)
            # One /tokenize instead of a bisection; any drift refuses rather than re-bisecting silently.
            if hashlib.sha256(prompt.encode()).hexdigest() != digest or count(prompt) != target:
                raise ValueError('reused prompt cut no longer reproduces the earlier prompt/token count')
            return prompt, chars, pads
        chars, pads = exact_cut(materials, target, count)
        return render(materials, chars, pads), chars, pads
    def event(row: dict[str, Json]) -> None:
        if args.ledger is not None:
            with args.ledger.open('a') as handle:
                handle.write(json.dumps({**row, 'time': time.time()}) + '\n')
    with args.output.open('x') as output:
        output.write(json.dumps({'type': 'identity', 'corpus_sha256': hashes, 'seed': 42,
                                 'cap': args.cap, 'trace_run': args.trace_run, 'model': model_id}) + '\n')
        output.flush()
        def request(built: tuple[str, int, int], target: int, salt: str, kind: str, rep: int, budget: int = 1500,
                    force_length: bool = False) -> None:
            prompt, chars, pads = built
            if time.time() >= args.deadline_utc:
                raise RuntimeError('window submission deadline reached; matrix incomplete, restore production')
            idle(args.url)
            request_id = uuid.uuid4().hex
            body: dict[str, Json] = {'model': model_id, 'request_id': request_id,
                'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': budget,
                'temperature': 1.0, 'top_p': 1.0, 'seed': 42, 'priority': 10, 'cache_salt': salt,
                'chat_template_kwargs': TEMPLATE, 'stream': True, 'stream_options': {'include_usage': True}}
            if force_length:
                body['ignore_eos'] = True
            req = urllib.request.Request(args.url + '/v1/chat/completions', json.dumps(body).encode(),
                                         {'Content-Type': 'application/json'})
            event({'event': 'send_intent', 'id': request_id, 'kind': kind, 'target': target})
            try:
                started = time.monotonic()
                with urllib.request.urlopen(req, timeout=900) as response:
                    event({'event': 'http_opened', 'id': request_id})
                    summary = stream_summary(response, started)
                usage = record(summary['usage'])
                cached = record(usage.get('prompt_tokens_details') or {}).get('cached_tokens')
                if usage.get('prompt_tokens') != target or cached is None:
                    raise ValueError('usage disagrees with exact occupied prompt or lacks cache count')
                cached_n = int(number(cached))
                if kind in ('cold', 'warmup', 'prime-full', 'prime-partial') and cached_n != 0:
                    raise ValueError('fresh cache salt produced a hit')
                if force_length and (usage.get('completion_tokens') != budget or summary.get('finish') != 'length'):
                    raise ValueError('warm-up did not generate its full forced length')
                if kind == 'full' and not target - 2048 <= cached_n < target:
                    raise ValueError('full prefix hit outside expected alignment/replay bound')
                if kind == 'partial' and not target - 102048 <= cached_n <= target - 100000:
                    raise ValueError('partial prefix hit outside controlled 100K suffix/replay bound')
                response_id = summary['response_id']
                if not isinstance(response_id, str):
                    raise ValueError('missing response id for two-rank correlation')
                trace_request = hashlib.sha256((args.trace_run + ':' + response_id).encode()).hexdigest()[:24]
                output.write(json.dumps({'type': 'request', 'kind': kind, 'rep': rep, 'target_ctx': target,
                    'cap': args.cap, 'trace_request': trace_request,
                    'cache_salt_sha256': hashlib.sha256(salt.encode()).hexdigest(), 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                    'prompt_chars': chars, 'prompt_pad_x': pads,
                    'cached_tokens': cached_n, 'uncached_prompt_tokens': target - cached_n,
                    'sampling': {'temperature': 1, 'top_p': 1, 'seed': 42, 'priority': 10, 'max_tokens': budget},
                    **summary}, allow_nan=False) + '\n')
                output.flush()
            except BaseException as exc:
                event({'event': 'failed', 'id': request_id, 'error_class': type(exc).__name__})
                raise
            event({'event': 'completed', 'id': request_id})
        if args.warmup:
            request(prompt_for(WARMUP), WARMUP, cache_salt(args.cache_salt_prefix, WARMUP, 0, 'warmup'), 'warmup', 0,
                    WARMUP_TOKENS, force_length=True)
        for target in args.contexts:
            prompt = prompt_for(target)
            partial = prompt_for(target - 100000) if 'partial' in args.modes else ('', 0, 0)
            for rep in range(args.reps):
                salt = cache_salt(args.cache_salt_prefix, target, rep, 'main')
                if 'cold' in args.modes:
                    request(prompt, target, salt, 'cold', rep)
                elif 'full' in args.modes:
                    request(prompt, target, salt, 'prime-full', rep, 64)
                if 'full' in args.modes:
                    request(prompt, target, salt, 'full', rep)
                if 'partial' in args.modes:
                    partial_salt = cache_salt(args.cache_salt_prefix, target, rep, 'partial')
                    request(partial, target - 100000, partial_salt, 'prime-partial', rep, 64)
                    request(prompt, target, partial_salt, 'partial', rep)

if __name__ == '__main__':
    main()
