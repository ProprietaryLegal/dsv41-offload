"""UPX-0085B request driver: one subprocess per step of battery_window.py, over the owned tunnel 127.0.0.1:18001.
Dry by default. Coordinator only. Numbers and hashes only: no prompt or generated text is written.

  --mode build    cut the public 150K prompt once per window (window_cells.cut), write its pins
  --mode greedy   the six W5/K1 greedy checks at NDM's 0.9 nats (nbw_cells.greedy, record_distribution)
  --mode kl       greedy_kl.compare_recorded over the previous candidate start's divergences
  --mode battery  A (cold 150K, 16 tokens), IDLE_WAIT_S without a request, B (cold 150K), POST_B_IDLE_S
                  (MEASURE-0085B.md); writes the wall-clock times measure_0085b.battery_result needs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

from offload.gb10_budget import window_cells as wc
from offload.launch.windows import greedy_kl as gk
from offload.launch.windows import nbw_cells as cells
from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, number, record
from offload.prefill_trace.window_requests import TEMPLATE, corpus
from offload.upx import measure_0085b as m
from offload.upx import windows as w

PROMPT = wc.PROMPTS['150k']
MAX_TOKENS = 16             # a battery measures prefill memory; the answer only proves the request completed
NEAR_TIE_NATS = 0.9 + 1e-3  # NDM 2026-10-27 ".9"
KL_BOUND = 0.005            # greedy_kl registration: 0085b changes no arithmetic, so UP2's floor bound, never wider
BATTERY_S = w.GATE_S['battery']


class Api(Protocol):
    def model(self) -> dict[str, Json]: ...
    def metrics(self) -> str: ...
    def stream(self, body: dict[str, Json], first: threading.Event) -> dict[str, Json]: ...


def build(api: wc.Http, materials: str, hashes: dict[str, Json]) -> dict[str, Json]:
    ck.require(wc.busy(api.metrics()) == 0, 'serve busy before public prompt construction')
    chars, tokens = wc.cut(materials, PROMPT, api.tokens)
    text = wc.render(materials, PROMPT.offset, chars)
    ck.require(tokens >= m.THRESHOLD, f'150K prompt cut to {tokens} tokens, below the {m.THRESHOLD} threshold')
    return {'corpus_sha256': hashes, 'prompt': {'key': PROMPT.key, 'offset': PROMPT.offset, 'target': PROMPT.target,
                                                 'chars': chars, 'tokens': tokens,
                                                 'sha256': hashlib.sha256(text.encode()).hexdigest()}}


def text_from(cuts: dict[str, Json], materials: str, hashes: dict[str, Json]) -> tuple[str, int]:
    ck.require(cuts.get('corpus_sha256') == hashes, 'public corpus changed')
    row = record(cuts['prompt'])
    ck.require(row.get('key') == PROMPT.key and row.get('offset') == PROMPT.offset, 'prompt plan changed')
    text = wc.render(materials, PROMPT.offset, int(number(row['chars'])))
    ck.require(hashlib.sha256(text.encode()).hexdigest() == row.get('sha256'), 'prompt bytes changed')
    return text, int(number(row['tokens']))


def request(api: Api, model: str, text: str, tokens: int, salt: str, wall: Callable[[], float]) -> dict[str, Json]:
    body: dict[str, Json] = {'model': model, 'messages': [{'role': 'user', 'content': text}], 'max_tokens': MAX_TOKENS,
                             'temperature': 1.0, 'top_p': 1.0, 'seed': 42, 'priority': 10, 'cache_salt': salt,
                             'chat_template_kwargs': TEMPLATE, 'ignore_eos': True, 'stream': True,
                             'stream_options': {'include_usage': True}}
    sent = wall()
    summary = api.stream(body, threading.Event())
    done = wall()
    usage = record(summary['usage'])
    cached = record(usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    ck.require(usage.get('prompt_tokens') == tokens and cached is not None and number(cached) == 0,
               'battery request usage differs from the built prompt or a fresh salt hit the cache')
    ck.require(usage.get('completion_tokens') == MAX_TOKENS and summary.get('finish') == 'length',
               'battery request did not generate its fixed length')
    ttft = number(summary['ttft_s'])
    ck.require(0 < ttft <= done - sent, 'battery request TTFT outside its own span')
    return {'sent': sent, 'first': sent + ttft, 'done': done, 'ttft_s': ttft, 'prompt_tokens': tokens,
            'salt_sha256': hashlib.sha256(salt.encode()).hexdigest()}


def wait_until(target: float, wall: Callable[[], float], sleep: Callable[[float], None]) -> None:
    while (left := target - wall()) > 0:
        sleep(min(left, 1.0))


def battery(api: Api, text: str, tokens: int, salt: str, deadline: float, *, wall: Callable[[], float] = time.time,
            sleep: Callable[[float], None] = time.sleep) -> dict[str, Json]:
    """A, the idle period, B, the period after B. Nothing else may run on the server meanwhile (gauges checked)."""
    ck.require(wall() + BATTERY_S <= deadline, 'window submission deadline reached: the battery does not fit')
    served = api.model()
    ck.require(served.get('max_model_len') == wc.CAP, 'served cap differs from production 327,680')
    ck.require(wc.busy(api.metrics()) == 0, 'serve not idle before the battery')
    a = request(api, str(served['id']), text, tokens, f'{salt}:A', wall)
    wait_until(number(a['done']) + m.IDLE_WAIT_S, wall, sleep)
    ck.require(wc.busy(api.metrics()) == 0, 'a request was in flight at the end of the idle period')
    b = request(api, str(served['id']), text, tokens, f'{salt}:B', wall)
    wait_until(number(b['done']) + m.POST_B_IDLE_S, wall, sleep)
    ck.require(wc.busy(api.metrics()) == 0, 'a request was in flight after B')
    return {'a_sent': a['sent'], 'a_done': a['done'], 'b_sent': b['sent'], 'b_first': b['first'], 'b_done': b['done'],
            'a_prompt_tokens': tokens, 'b_prompt_tokens': tokens, 'a_ttft_s': a['ttft_s'], 'b_ttft_s': b['ttft_s'],
            'salt_sha256': [a['salt_sha256'], b['salt_sha256']], 'max_tokens': MAX_TOKENS,
            'status': 'MEASURED wall-clock request times (runner host clock)'}


def times(result: dict[str, Json], mode: str) -> m.Battery:
    return m.Battery(mode, number(result['a_sent']), number(result['a_done']), number(result['b_sent']),
                     number(result['b_first']), number(result['b_done']), int(number(result['a_prompt_tokens'])),
                     int(number(result['b_prompt_tokens'])))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--mode', choices=('build', 'greedy', 'kl', 'battery'), required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, default=Path('/opt/dsv41/data/bench-data'))
    ap.add_argument('--deadline-utc', type=float, default=0)
    ap.add_argument('--cuts', type=Path)
    ap.add_argument('--salt', default='')
    ap.add_argument('--candidate', type=Path)
    args = ap.parse_args(argv)
    if not args.execute:
        print(f'DRY RUN: UPX-0085B {args.mode}; no network or file')
        return 0
    ck.require(os.environ.get('WINDOW_COORDINATOR') == 'Deepseek/coordinator', 'coordinator required')
    ck.require(not args.output.exists(), 'fresh step output required')
    ck.require(time.time() < args.deadline_utc, 'submission deadline reached')
    api = cells.Http(wc.URL)
    result: dict[str, Json]
    if args.mode == 'greedy':
        ck.require(bool(args.salt), 'probe salt prefix required')
        prompts, reference = cells.greedy_inputs()
        result = cells.greedy(api, args.deadline_utc, prompts, reference, args.output, NEAR_TIE_NATS,
                              record_distribution=True, probe_salt_prefix=args.salt)
    elif args.mode == 'kl':
        ck.require(args.candidate is not None and bool(args.salt), 'candidate greedy result and salt required')
        rows = ck.rows(ck.read(cast(Path, args.candidate))['checks'])
        checked = gk.compare_recorded(api, rows, args.salt, KL_BOUND)
        result = {'rows': cast(Json, checked), 'bound': KL_BOUND,
                  'pass': all(record(r)['pass'] is True for r in checked)}
    else:
        materials, hashes = corpus(args.data_dir)
        if args.mode == 'build':
            result = build(api, materials, hashes)
        else:
            ck.require(args.cuts is not None and bool(args.salt), 'cuts and salt required')
            text, tokens = text_from(ck.read(cast(Path, args.cuts)), materials, hashes)
            result = battery(api, text, tokens, args.salt, args.deadline_utc)
    args.output.write_text(json.dumps(result, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
