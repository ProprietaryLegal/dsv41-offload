"""SP1-WINDOW cell driver: build public prompts once, then run one control or profiled cell per invocation.

Coordinator-only: offload/launch/windows/sp1_window.py calls it inside the HOLD through the owned tunnel 127.0.0.1:18001.
Without --execute it prints the cell plan and touches nothing. It records numbers and hashes only: no prompt or generated
text is written. Profiling uses vLLM's own /start_profile and /stop_profile (both TP ranks, collective RPC); a capture
starts only if the latest 1 s telemetry of both Sparks shows MemAvailable >= the capture floor, otherwise it is recorded
SKIPPED_MEMORY and the request still runs. See offload/gb10_budget/PROFILE-WINDOW-DESIGN.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from offload.gb10_budget.capture import control, load_bench
from offload.prefill_maxlen.evidence import Json, number, record
from offload.prefill_trace.window_requests import HEADER, TEMPLATE, corpus, stream_summary

URL = 'http://127.0.0.1:18001'
CAP = 327680
TOLERANCE = .02             # built prompt: target*(1 - TOLERANCE) <= tokens <= target
MAX_COUNT_CALLS = 4         # /tokenize calls per prompt (PF1's bisection used 20-23; its dips reached 4.17 GiB)
FIRST_RATIO = 3.3           # characters per token, first guess (bench2 build_real)
SETTLE_S = 2.0              # decode capture starts this long after every stream emitted
DECODE_S = 3.0              # explicit stop after this; the profiler stops itself after 16 iterations first
PREFILL_S = 5.0             # 2-3 chunks of 4,092 tokens at ~1.85 s each
LATE_DELAY_S = 50.0         # 150K TTFT is 68.9-71.2 s (MEASURED PF1 W1/W3 off-327)
TTFT_TIMEOUT_S = 600.0
TELEMETRY_AGE_S = 3.0


@dataclass(frozen=True)
class Prompt:
    key: str
    target: int
    offset: int             # characters into the corpus body (after HEADER)


# Distinct, far-apart corpus offsets (11.8 M characters): no two cells send the same text.
PROMPTS = {p.key: p for p in (
    Prompt('warm-32k', 32000, 9_000_000), Prompt('short', 1024, 0),
    Prompt('32k-0', 32000, 1_000_000), Prompt('32k-1', 32000, 1_500_000), Prompt('32k-2', 32000, 2_000_000),
    Prompt('32k-3', 32000, 2_500_000), Prompt('150k', 150000, 4_000_000))}


@dataclass(frozen=True)
class Cell:
    name: str
    prompts: tuple[str, ...]
    captures: tuple[str, ...] = ()
    max_tokens: int = 1500
    warmup: bool = False
    estimate_s: float = 60
    late: int = 0           # WIN2: the last `late` prompts arrive only after every earlier stream is decoding

    def __post_init__(self) -> None:
        if not 0 <= self.late < len(self.prompts) or (self.late and (self.captures or self.warmup)):
            raise ValueError(f'{self.name}: late arrivals need >= 1 early stream and no captures/warm-up')

    @property
    def batch(self) -> int:
        return len(self.prompts)


B4 = ('32k-0', '32k-1', '32k-2', '32k-3')
CELLS = {c.name: c for c in (
    Cell('warmup', ('warm-32k',), max_tokens=64, warmup=True, estimate_s=60),
    Cell('ctl-short-b1', ('short',), estimate_s=60),
    Cell('ctl-32k-b1', ('32k-0',), estimate_s=90),
    Cell('ctl-32k-b4', B4, estimate_s=180),
    Cell('ctl-150k-b1', ('150k',), estimate_s=180),
    Cell('prof-short-b1', ('short',), ('decode',), estimate_s=120),
    Cell('prof-32k-b1', ('32k-0',), ('prefill-early', 'decode'), estimate_s=150),
    Cell('prof-32k-b4', B4, ('decode',), estimate_s=240),
    Cell('prof-150k-b1', ('150k',), ('prefill-late', 'decode'), estimate_s=240),
    # WIN2 (Fable review): multi-user and long-answer controls; not in ORDER, so existing windows are unchanged.
    # mixed: three 32K decoders start (MEASURED 32K B1 TTFT ~14.5-15.4 s each; B3 contends), the 150K document arrives
    # only after all three are decoding (MEASURED 150K TTFT 69 s alone, 117 s beside two decoders in PC1): ~300 s.
    Cell('ctl-mixed150k-4user', ('32k-1', '32k-2', '32k-3', '150k'), late=1, estimate_s=300),
    # long answer: 5,000 tokens at MEASURED 51-57 tok/s B1 decode ~ 90-100 s plus a short prefill: 180 s bound.
    Cell('ctl-long-5k', ('short',), max_tokens=5000, estimate_s=180))}
WIN2_CELLS = ('ctl-mixed150k-4user', 'ctl-long-5k')   # opt-in by name only
ORDER = tuple(n for n in CELLS if n not in WIN2_CELLS)
CAPTURE_KINDS = ('decode', 'prefill-early', 'prefill-late')


def body(materials: str) -> str:
    if not materials.startswith(HEADER):
        raise ValueError('corpus no longer starts with the benchmark header')
    return materials[len(HEADER):]


def render(materials: str, offset: int, chars: int) -> str:
    text = body(materials)
    if offset < 0 or chars < 1 or offset + chars > len(text):
        raise ValueError('prompt cut outside the public corpus; never repeat filler')
    return HEADER + text[offset:offset + chars]


def cut(materials: str, prompt: Prompt, count: Callable[[str], int]) -> tuple[int, int]:
    """(characters, tokens) with target*(1-TOLERANCE) <= tokens <= target in at most MAX_COUNT_CALLS /tokenize calls."""
    chars = int(prompt.target * FIRST_RATIO)
    low = prompt.target * (1 - TOLERANCE)
    for _ in range(MAX_COUNT_CALLS):
        n = count(render(materials, prompt.offset, chars))
        if n < 1:
            raise ValueError('tokenizer returned no tokens')
        if low <= n <= prompt.target:
            return chars, n
        chars = max(1, int(chars * prompt.target / n * (1 - TOLERANCE / 2)))   # aim at the middle of the band
    raise ValueError(f'{prompt.key}: no {prompt.target}-token cut within {MAX_COUNT_CALLS} /tokenize calls; window stops')


class Counters(Protocol):
    """bench2's /metrics parser and speculative-counter delta (reused, not copied)."""
    def parse_metrics(self, text: str) -> dict[str, float]: ...
    def spec_delta(self, before: dict[str, float], after: dict[str, float]) -> dict[str, Json]: ...


class Api(Protocol):
    def tokens(self, text: str) -> int: ...
    def metrics(self) -> str: ...
    def model(self) -> dict[str, Json]: ...
    def stream(self, body: dict[str, Json], first: threading.Event) -> dict[str, Json]: ...
    def profile(self, verb: str) -> None: ...


def first_signal(lines: Iterable[bytes], first: threading.Event) -> Iterator[bytes]:
    """Pass SSE lines through unchanged; set `first` at the first reasoning/answer piece."""
    for raw in lines:
        if not first.is_set() and raw.startswith(b'data:') and raw[5:].strip() != b'[DONE]':
            obj = record(cast(Json, json.loads(raw[5:].strip())))
            choices = obj.get('choices') or []
            for choice in choices if isinstance(choices, list) else []:
                delta = record(record(choice).get('delta') or {})
                if delta.get('reasoning_content') or delta.get('reasoning') or delta.get('content'):
                    first.set()
        yield raw


class Http:
    def __init__(self, url: str) -> None:
        self.url = url
        self.served: dict[str, Json] | None = None

    def post(self, endpoint: str, body: dict[str, Json]) -> dict[str, Json]:
        req = urllib.request.Request(self.url + endpoint, json.dumps(body).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=900) as response:
            return record(cast(Json, json.load(response)))

    def model(self) -> dict[str, Json]:
        if self.served is None:
            with urllib.request.urlopen(self.url + '/v1/models', timeout=10) as response:
                models = record(cast(Json, json.load(response)))['data']
            if not isinstance(models, list) or len(models) != 1:
                raise ValueError('expected one served target')
            self.served = record(models[0])
        return self.served

    def tokens(self, text: str) -> int:
        result = self.post('/tokenize', {'model': self.model()['id'], 'messages': [{'role': 'user', 'content': text}],
                                         'add_generation_prompt': True, 'chat_template_kwargs': TEMPLATE})
        return int(number(result['count']))

    def metrics(self) -> str:
        with urllib.request.urlopen(self.url + '/metrics', timeout=10) as response:
            return str(response.read().decode())

    def stream(self, body: dict[str, Json], first: threading.Event) -> dict[str, Json]:
        req = urllib.request.Request(self.url + '/v1/chat/completions', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
        started = time.monotonic()
        with urllib.request.urlopen(req, timeout=900) as response:
            return stream_summary(first_signal(response, first), started)

    def profile(self, verb: str) -> None:
        control(self.url, verb)


def busy(text: str) -> float:
    total, seen = 0.0, set()
    for line in text.splitlines():
        for name in ('vllm:num_requests_running', 'vllm:num_requests_waiting'):
            if line.startswith((name + '{', name + ' ')):
                total += float(line.split()[-1])
                seen.add(name)
    if len(seen) != 2:
        raise ValueError('metrics lack the request gauges')
    return total


def running(text: str) -> float:
    """vllm:num_requests_running only (admitted requests), for late-arrival overlap proof."""
    values = [float(line.split()[-1]) for line in text.splitlines()
              if line.startswith(('vllm:num_requests_running{', 'vllm:num_requests_running '))]
    if not values:
        raise ValueError('metrics lack vllm:num_requests_running')
    return sum(values)


def latest_memory(paths: list[Path], now: Callable[[], float] = time.time) -> dict[str, int]:
    """MemAvailable (kB) from the runner's 1 s telemetry files; refuses a stale or missing sample."""
    found: dict[str, int] = {}
    for path in paths:
        if now() - path.stat().st_mtime > TELEMETRY_AGE_S:
            raise RuntimeError(f'{path.name}: telemetry older than {TELEMETRY_AGE_S:g} s; capture floor cannot be checked')
        rows = [record(cast(Json, json.loads(x))) for x in path.read_text().splitlines()[-8:] if x.startswith('{')]
        memory = [r for r in rows if r.get('kind') == 'memory']
        if not memory:
            raise RuntimeError(f'{path.name}: no recent memory sample')
        found[path.name] = int(number(record(memory[-1]['kb'])['MemAvailable']))
    return found


@dataclass
class Plan:
    deadline: float
    floor_kb: int
    telemetry: list[Path]
    salt_prefix: str
    ledger: Path | None = None


def salt(prefix: str, cell: str, stream: int) -> str:
    if not prefix or not cell:
        raise ValueError('invalid cache salt identity')
    return f'{prefix}:{cell}:{stream}'


def run_cell(api: Api, cell: Cell, texts: dict[str, tuple[str, int]], plan: Plan, *,
             sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
             wall: Callable[[], float] = time.time, cap: int = CAP, late_delay_s: float = LATE_DELAY_S) -> dict[str, Json]:
    """One cell. Captures always stop in `finally`; a stream failure propagates after the stop. `cap` and `late_delay_s`
    are PF2-LOWKV's (524,288; mid-prompt at 35 s); SP1 uses the defaults."""
    if wall() + cell.estimate_s > plan.deadline:
        raise RuntimeError(f'window submission deadline reached: {cell.name} does not fit')
    model = api.model()
    if model.get('max_model_len') != cap:
        raise ValueError('served cap differs from the profiled arm')
    if busy(api.metrics()):
        raise RuntimeError('serve not verified idle before the cell')
    bench = cast(Counters, load_bench())
    before = bench.parse_metrics(api.metrics())
    origin = clock()
    captures: list[dict[str, Json]] = []
    firsts = [threading.Event() for _ in cell.prompts]
    submitted: list[float] = [0.0] * cell.batch
    early = cell.batch - cell.late
    barrier = threading.Barrier(early)
    late: dict[str, Json] = {}

    def event(row: dict[str, Json]) -> None:
        if plan.ledger is not None:
            with plan.ledger.open('a') as handle:
                handle.write(json.dumps({**row, 'cell': cell.name, 'time': wall()}) + '\n')

    def request(i: int) -> dict[str, Json]:
        key = cell.prompts[i]
        text, tokens = texts[key]
        cache = salt(plan.salt_prefix, cell.name, i)
        body: dict[str, Json] = {'model': model['id'], 'messages': [{'role': 'user', 'content': text}],
                                 'max_tokens': cell.max_tokens, 'temperature': 1.0, 'top_p': 1.0, 'seed': 42,
                                 'priority': 10, 'cache_salt': cache, 'chat_template_kwargs': TEMPLATE,
                                 'stream': True, 'stream_options': {'include_usage': True}}
        if cell.warmup:
            body['ignore_eos'] = True
        if i < early:
            barrier.wait(timeout=30)
        submitted[i] = clock() - origin
        event({'event': 'send', 'stream': i})
        try:
            summary = api.stream(body, firsts[i])
        except BaseException as exc:
            event({'event': 'failed', 'stream': i, 'error_class': type(exc).__name__})
            raise
        event({'event': 'completed', 'stream': i})
        usage = record(summary['usage'])
        cached = record(usage.get('prompt_tokens_details') or {}).get('cached_tokens')
        if usage.get('prompt_tokens') != tokens or cached is None or number(cached) != 0:
            raise ValueError('usage differs from the built prompt or a fresh salt hit the cache')
        if cell.warmup and (usage.get('completion_tokens') != cell.max_tokens or summary.get('finish') != 'length'):
            raise ValueError('warm-up did not generate its forced length')
        response = summary.pop('response_id')
        return {'stream': i, 'prompt': key, 'prompt_tokens': tokens, 'submitted_rel_s': submitted[i],
                'prompt_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'cache_salt_sha256': hashlib.sha256(cache.encode()).hexdigest(),
                'response_id_sha256': hashlib.sha256(str(response).encode()).hexdigest(),
                'sampling': {'temperature': 1, 'top_p': 1, 'seed': 42, 'priority': 10, 'max_tokens': cell.max_tokens},
                **summary}

    futures: list[Future[dict[str, Json]]] = []

    def submit() -> None:
        futures.extend(pool.submit(request, i) for i in range(early))
        if cell.late:
            arrive_late()

    def arrive_late() -> None:
        """Late streams start only after every early stream is decoding, and must be admitted beside them."""
        end = clock() + TTFT_TIMEOUT_S
        while not all(firsts[i].is_set() for i in range(early)):
            check_failed()
            if ended():
                raise RuntimeError('an early stream ended before every early stream reached decode')
            if clock() >= end:
                raise TimeoutError('early streams did not all reach decode before the TTFT bound')
            sleep(.02)
        settle = clock() + SETTLE_S
        while clock() < settle:
            check_failed()
            sleep(.05)
        if ended():
            raise RuntimeError('an early stream finished before the late arrival')
        late['submitted_rel_s'] = clock() - origin
        futures.extend(pool.submit(request, i) for i in range(early, cell.batch))
        end = clock() + TTFT_TIMEOUT_S
        while True:
            check_failed()
            if any(f.done() for f in futures[:early]):
                raise RuntimeError('a decoder finished before the late arrival was admitted beside it')
            if running(api.metrics()) >= cell.batch:
                late['admitted_rel_s'] = clock() - origin
                late['overlap'] = True
                return
            if clock() >= end:
                raise TimeoutError('late arrival was never admitted beside the live decoders')
            sleep(.05)

    def ended() -> bool:
        return any(f.done() for f in futures)

    def check_failed() -> None:
        for f in futures:
            if f.done():
                f.result()        # propagate a stream failure instead of waiting for a bound

    def capture(kind: str, then: Callable[[], None] | None = None) -> None:
        """Memory floor, start, optional action (submission), bounded wait, ALWAYS stop once started."""
        row: dict[str, Json] = {'kind': kind}
        captures.append(row)
        memory = latest_memory(plan.telemetry)
        row['mem_available_kb'] = cast(Json, memory)
        if min(memory.values()) < plan.floor_kb:
            row['status'] = 'SKIPPED_MEMORY'
            if then is not None:
                then()
            return
        row['start_call_rel_s'] = clock() - origin
        api.profile('start_profile')
        row['start_return_rel_s'] = clock() - origin
        row['start_wall'] = wall()
        try:
            if then is not None:
                then()
            end = clock() + (DECODE_S if kind == 'decode' else PREFILL_S)
            while clock() < end and not ended():
                sleep(.05)
            row['status'] = ('MIXED' if kind != 'decode' and any(f.is_set() for f in firsts) else   # never "prefill"
                             'ENDED' if ended() else 'TAKEN')
        finally:
            row['stop_call_rel_s'] = clock() - origin
            api.profile('stop_profile')
            row['stop_return_rel_s'] = clock() - origin
            row['stop_wall'] = wall()

    with ThreadPoolExecutor(max_workers=cell.batch) as pool:
        try:
            if 'prefill-early' in cell.captures:
                capture('prefill-early', submit)      # the first chunks are the first iterations recorded
            else:
                submit()
            if 'prefill-late' in cell.captures:
                end = clock() + late_delay_s
                while clock() < end:
                    check_failed()
                    if ended():
                        raise RuntimeError('stream ended before the late prefill capture')
                    sleep(.05)
                if any(f.is_set() for f in firsts):
                    raise RuntimeError('decode already started; a late prefill capture would be mislabelled')
                capture('prefill-late')
            if 'decode' in cell.captures:
                end = clock() + TTFT_TIMEOUT_S
                while not all(f.is_set() for f in firsts):
                    check_failed()
                    if ended():
                        raise RuntimeError('a stream ended before every stream reached decode')
                    if clock() >= end:
                        raise TimeoutError('not every stream reached decode before the TTFT bound')
                    sleep(.02)
                settle = clock() + SETTLE_S
                while clock() < settle:
                    check_failed()
                    sleep(.05)
                capture('decode')
        finally:
            while not all(f.done() for f in futures):
                sleep(.05)            # each stream keeps its own 900-second wall bound (stream_summary)
            streams = [f.result() for f in futures]
    after = bench.parse_metrics(api.metrics())
    return {'type': 'cell', 'cell': cell.name, 'batch': cell.batch, 'profiled': bool(cell.captures),
            'warmup': cell.warmup, 'origin_wall': wall() - (clock() - origin), 'late': cell.late,
            'late_arrival': cast(Json, late or None),
            'streams': cast(Json, streams), 'captures': cast(Json, captures),
            'spec': cast(Json, bench.spec_delta(before, after)),
            'status': 'MEASURED client observations; profiled cells are profile-affected, never promotion timings'}


def build(api: Api, materials: str, hashes: dict[str, Json]) -> dict[str, Json]:
    if busy(api.metrics()):
        raise RuntimeError('serve not verified idle before prompt construction')
    prompts: dict[str, Json] = {}
    for prompt in PROMPTS.values():
        chars, tokens = cut(materials, prompt, api.tokens)
        text = render(materials, prompt.offset, chars)
        prompts[prompt.key] = {'target': prompt.target, 'offset': prompt.offset, 'chars': chars, 'tokens': tokens,
                               'sha256': hashlib.sha256(text.encode()).hexdigest()}
    return {'type': 'cuts', 'corpus_sha256': hashes, 'header_sha256': hashlib.sha256(HEADER.encode()).hexdigest(),
            'tolerance': TOLERANCE, 'prompts': prompts}


def texts_from(cuts: dict[str, Json], materials: str, hashes: dict[str, Json]) -> dict[str, tuple[str, int]]:
    """Re-render every prompt from its numeric cut; refuses any drift (no /tokenize call)."""
    if cuts.get('corpus_sha256') != hashes:
        raise ValueError('public corpus changed since the prompts were built')
    out: dict[str, tuple[str, int]] = {}
    for key, value in record(cuts['prompts']).items():
        row = record(value)
        prompt = PROMPTS[key]
        if row.get('offset') != prompt.offset or row.get('target') != prompt.target:
            raise ValueError('prompt plan changed since the prompts were built')
        text = render(materials, prompt.offset, int(number(row['chars'])))
        if hashlib.sha256(text.encode()).hexdigest() != row.get('sha256'):
            raise ValueError(f'{key}: rebuilt prompt differs from its recorded hash')
        out[key] = (text, int(number(row['tokens'])))
    if set(out) != set(PROMPTS):
        raise ValueError('incomplete prompt cuts')
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--url', default=URL)
    ap.add_argument('--data-dir', type=Path, default=Path('/opt/dsv41/data/bench-data'))
    sub = ap.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build', help='construct every prompt once; write numeric cuts and hashes only')
    b.add_argument('--output', type=Path, required=True)
    r = sub.add_parser('run', help='one cell')
    r.add_argument('--cell', choices=ORDER, required=True)
    r.add_argument('--cuts', type=Path, required=True)
    r.add_argument('--output', type=Path, required=True)
    r.add_argument('--deadline-utc', type=float, required=True)
    r.add_argument('--telemetry', type=Path, nargs=2, required=True)
    r.add_argument('--capture-floor-kb', type=int, required=True)
    r.add_argument('--cache-salt-prefix', required=True)
    r.add_argument('--ledger', type=Path)
    args = ap.parse_args(argv)
    if not args.execute:
        print(json.dumps({'executed': False, 'cells': {n: {'prompts': list(c.prompts), 'captures': list(c.captures),
                                                           'max_tokens': c.max_tokens} for n, c in CELLS.items()},
                          'prompts': {k: {'target': p.target, 'offset': p.offset} for k, p in PROMPTS.items()}}))
        return 0
    if args.url != URL:
        ap.error('coordinator must use the reserved experiment tunnel at 127.0.0.1:18001')
    if args.output.exists():
        ap.error('refusing an existing output; use a fresh window directory')
    materials, hashes = corpus(args.data_dir)
    api = Http(args.url)
    if args.command == 'build':
        result = build(api, materials, hashes)
    else:
        if args.deadline_utc <= time.time():
            ap.error('--deadline-utc is in the past')
        if not 4 * 1024**2 <= args.capture_floor_kb <= 8 * 1024**2:
            ap.error('capture floor must lie in 4..8 GiB')
        texts = texts_from(record(cast(Json, json.loads(args.cuts.read_text()))), materials, hashes)
        plan = Plan(args.deadline_utc, args.capture_floor_kb, list(args.telemetry), args.cache_salt_prefix, args.ledger)
        result = run_cell(api, CELLS[args.cell], texts, plan)
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, allow_nan=False) + '\n')
    print(f'{args.command}: wrote {args.output.name} (numbers and hashes only)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
