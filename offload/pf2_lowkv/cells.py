"""PF2-LOWKV cell driver: build the public prompts once, then run one cell per invocation at 524,288.

Coordinator-only: offload/launch/windows/pf2_window.py calls it inside the HOLD through the owned tunnel 127.0.0.1:18001.
Without --execute it prints the cell plan and touches nothing. Numbers and hashes only: no prompt or generated text is
written. The 150K cells reuse SP1's reviewed driver (offload/gb10_budget/window_cells.run_cell: cold-cache proof, capture
floor, ALWAYS /stop_profile once started) with cap 524,288 and the mid-prompt capture at 35 s. The needle cell reuses
offload/kvpack/needle.py's builder and scorer, keeps only recall / foreign-code counts and the answer's sha256, and bounds
/tokenize calls. See offload/pf2_lowkv/DESIGN.md.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

from offload.gb10_budget import window_cells as wc
from offload.prefill_maxlen.evidence import Json, number, record
from offload.prefill_trace.window_requests import HEADER, corpus

# offload/kvpack/needle.py is untyped K1 tooling: used as-is (builder, scorer, request), never edited here.
needle: Any = importlib.import_module('offload.kvpack.needle')

URL = wc.URL
CAP = 524_288
MID_DELAY_S = 35.0           # 150K at 524,288: TTFT 69-71 s fast / 91-95 s slow (PF1 MEASURED) -> chunk ~14-19 of 37
NEEDLE_CTX, NEEDLES, NEEDLE_SEED, NEEDLE_MAX_TOKENS = 480_000, 6, 1022, 300
MAX_NEEDLE_COUNTS = 6        # /tokenize calls for the 480K haystack (PF1's 20-23 calls drove sparkhead's 4.17 GiB dips)
PROMPTS = {key: wc.PROMPTS[key] for key in ('warm-32k', '150k')}
CELLS = {c.name: c for c in (
    wc.Cell('warmup', ('warm-32k',), max_tokens=64, warmup=True, estimate_s=60),
    wc.Cell('c150-a', ('150k',), estimate_s=150),
    wc.Cell('c150-b', ('150k',), estimate_s=150),
    wc.Cell('p150', ('150k',), ('prefill-late',), estimate_s=180))}
ORDER = tuple(CELLS)
NEEDLE_ESTIMATE_S = 360.0


def build(api: wc.Api, materials: str, hashes: dict[str, Json]) -> dict[str, Json]:
    """SP1's build restricted to PF2's two prompts (same corpus offsets and +-2 % band, <= 4 /tokenize calls each)."""
    if wc.busy(api.metrics()):
        raise RuntimeError('serve not verified idle before prompt construction')
    prompts: dict[str, Json] = {}
    for prompt in PROMPTS.values():
        chars, tokens = wc.cut(materials, prompt, api.tokens)
        text = wc.render(materials, prompt.offset, chars)
        prompts[prompt.key] = {'target': prompt.target, 'offset': prompt.offset, 'chars': chars, 'tokens': tokens,
                               'sha256': hashlib.sha256(text.encode()).hexdigest()}
    return {'type': 'cuts', 'corpus_sha256': hashes, 'header_sha256': hashlib.sha256(HEADER.encode()).hexdigest(),
            'tolerance': wc.TOLERANCE, 'prompts': prompts}


def texts_from(cuts: dict[str, Json], materials: str, hashes: dict[str, Json]) -> dict[str, tuple[str, int]]:
    if cuts.get('corpus_sha256') != hashes:
        raise ValueError('public corpus changed since the prompts were built')
    out: dict[str, tuple[str, int]] = {}
    for key, value in record(cuts['prompts']).items():
        row = record(value)
        prompt = PROMPTS[key]
        if row.get('offset') != prompt.offset or row.get('target') != prompt.target:
            raise ValueError('prompt plan changed since the prompts were built')
        text = wc.render(materials, prompt.offset, int(number(row['chars'])))
        if hashlib.sha256(text.encode()).hexdigest() != row.get('sha256'):
            raise ValueError(f'{key}: rebuilt prompt differs from its recorded hash')
        out[key] = (text, int(number(row['tokens'])))
    if set(out) != set(PROMPTS):
        raise ValueError('incomplete prompt cuts')
    return out


class NeedleApi(Protocol):
    def count(self, text: str) -> int: ...
    def ask(self, prompt: str, salt: str) -> dict[str, Json]: ...
    def metrics(self) -> str: ...


class Bounded:
    """needle.Builder's build() with at most MAX_NEEDLE_COUNTS /tokenize calls; the corpus is passed in, never re-read."""

    def __init__(self, docs: list[dict[str, str]], count: Callable[[str], int]) -> None:
        self.builder: Any = needle.Builder.__new__(needle.Builder)
        self.builder.url, self.builder.model, self.builder.docs = None, None, docs
        self.builder.count = self.count       # instance attribute: build() calls self.count(...)
        self._count, self.calls = count, 0

    def count(self, text: str) -> int:
        self.calls += 1
        if self.calls > MAX_NEEDLE_COUNTS:
            raise RuntimeError(f'needle haystack needs more than {MAX_NEEDLE_COUNTS} /tokenize calls; window stops the cell')
        return self._count(text)

    def build(self, ctx: int, n: int, seed: int) -> tuple[str, dict[str, str]]:
        prompt, codes = self.builder.build(ctx, n, seed)
        return str(prompt), {str(k): str(v) for k, v in dict(codes).items()}


def run_needle(api: NeedleApi, docs: list[dict[str, str]], deadline: float, salt_prefix: str, *,
               wall: Callable[[], float] = time.time, clock: Callable[[], float] = time.monotonic,
               ledger: Path | None = None) -> dict[str, Json]:
    """One 480K needle request on the L arm: recall/foreign counts and hashes only."""
    if wall() + NEEDLE_ESTIMATE_S > deadline:
        return {'type': 'needle', 'status': 'SKIPPED_TIME'}
    if wc.busy(api.metrics()):
        raise RuntimeError('serve not verified idle before the needle')
    builder = Bounded(docs, api.count)
    started = clock()
    prompt, codes = builder.build(NEEDLE_CTX, NEEDLES, NEEDLE_SEED)
    built_s = clock() - started
    if wall() + NEEDLE_ESTIMATE_S > deadline:
        return {'type': 'needle', 'status': 'SKIPPED_TIME', 'build_s': built_s, 'count_calls': builder.calls}
    salt = wc.salt(salt_prefix, 'needle-480k', 0)

    def event(row: dict[str, Json]) -> None:
        if ledger is not None:
            with ledger.open('a') as handle:
                handle.write(json.dumps({**row, 'cell': 'needle-480k', 'time': wall()}) + '\n')
    event({'event': 'send', 'stream': 0})
    try:
        answer = api.ask(prompt, salt)
    except BaseException as exc:
        event({'event': 'failed', 'stream': 0, 'error_class': type(exc).__name__})
        raise
    event({'event': 'completed', 'stream': 0})
    text = str(answer.pop('text'))
    scored: dict[str, Any] = needle.score(text, codes, set())
    usage = record(answer.get('usage') or {})
    cached = record(usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    return {'type': 'needle', 'status': 'RAN', 'ctx_target': NEEDLE_CTX, 'needles': NEEDLES, 'seed': NEEDLE_SEED,
            'prompt_tokens': usage.get('prompt_tokens'), 'cached_tokens': cached,
            'completion_tokens': usage.get('completion_tokens'), 'elapsed_s': answer.get('elapsed_s'),
            'build_s': built_s, 'count_calls': builder.calls, 'recall': scored['recall'], 'of': scored['of'],
            'recalled_all': scored['recall'] == scored['of'],
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'cache_salt_sha256': hashlib.sha256(salt.encode()).hexdigest(),
            'answer_sha256': hashlib.sha256(text.encode()).hexdigest()}


class NeedleHttp:
    def __init__(self, url: str) -> None:
        self.http = wc.Http(url)
        self.url = url

    def count(self, text: str) -> int:
        result = self.http.post('/tokenize', {'model': self.http.model()['id'], 'prompt': text})
        return int(number(result['count']))

    def ask(self, prompt: str, salt: str) -> dict[str, Json]:
        model = self.http.model()
        if model.get('max_model_len') != CAP:
            raise ValueError('served cap differs from the PF2 arm')
        return cast(dict[str, Json], needle.ask(self.url, str(model['id']), prompt, NEEDLE_MAX_TOKENS, salt))

    def metrics(self) -> str:
        return self.http.metrics()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--url', default=URL)
    ap.add_argument('--data-dir', type=Path, default=Path('/opt/dsv41/data/bench-data'))
    sub = ap.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build', help='construct the warm-up and 150K prompts once; numeric cuts and hashes only')
    b.add_argument('--output', type=Path, required=True)
    for name in ('run', 'needle'):
        r = sub.add_parser(name)
        if name == 'run':
            r.add_argument('--cell', choices=ORDER, required=True)
            r.add_argument('--cuts', type=Path, required=True)
            r.add_argument('--telemetry', type=Path, nargs=2, required=True)
            r.add_argument('--capture-floor-kb', type=int, required=True)
        r.add_argument('--output', type=Path, required=True)
        r.add_argument('--deadline-utc', type=float, required=True)
        r.add_argument('--cache-salt-prefix', required=True)
        r.add_argument('--ledger', type=Path)
    args = ap.parse_args(argv)
    if not args.execute:
        print(json.dumps({'executed': False, 'cap': CAP, 'mid_delay_s': MID_DELAY_S,
                          'cells': {n: {'prompts': list(c.prompts), 'captures': list(c.captures), 'max_tokens': c.max_tokens}
                                    for n, c in CELLS.items()},
                          'needle': {'ctx': NEEDLE_CTX, 'needles': NEEDLES, 'max_tokens': NEEDLE_MAX_TOKENS,
                                     'max_count_calls': MAX_NEEDLE_COUNTS}}))
        return 0
    if args.url != URL:
        ap.error('coordinator must use the reserved experiment tunnel at 127.0.0.1:18001')
    if args.output.exists():
        ap.error('refusing an existing output; use a fresh window directory')
    if args.command != 'build' and args.deadline_utc <= time.time():
        ap.error('--deadline-utc is in the past')
    result: dict[str, Json]
    if args.command == 'needle':
        result = run_needle(NeedleHttp(args.url), list(needle.load_docs(args.data_dir)), args.deadline_utc, args.cache_salt_prefix,
                            ledger=args.ledger)
    else:
        materials, hashes = corpus(args.data_dir)
        api = wc.Http(args.url)
        if args.command == 'build':
            result = build(api, materials, hashes)
        else:
            if not 4 * 1024**2 <= args.capture_floor_kb <= 8 * 1024**2:
                ap.error('capture floor must lie in 4..8 GiB')
            texts = texts_from(record(cast(Json, json.loads(args.cuts.read_text()))), materials, hashes)
            plan = wc.Plan(args.deadline_utc, args.capture_floor_kb, list(args.telemetry), args.cache_salt_prefix, args.ledger)
            result = wc.run_cell(api, CELLS[args.cell], texts, plan, cap=CAP, late_delay_s=MID_DELAY_S)
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, allow_nan=False) + '\n')
    print(f'{args.command}: wrote {args.output.name} (numbers and hashes only)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
