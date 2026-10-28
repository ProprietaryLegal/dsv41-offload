"""Short public cells, including real staggered 150K prefill beside two decoders. Dry by default."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import threading
import time
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

from offload.gb10_budget import window_cells as wc
from offload.gb10_budget.capture import load_bench
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import nbw_cells as greedy_driver
from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, number, record
from offload.prefill_trace.window_requests import corpus, stream_summary, TEMPLATE

CELLS = ('short-b1', '32k-b1', '32k-b3', '32k-b4', 'mixed150k-2decode')
KEYS = ('short', '32k-0', '32k-1', '32k-2', '32k-3', '150k')
PROMPTS = {key: wc.PROMPTS[key] for key in KEYS}
BUDGET = 512
REPEATS = dict(zip(CELLS, (4, 4, 2, 1, 2)))
# Registered greedy near-tie limits (nats): NBW's default, NDM's general 0.9 and DEP1 fixed2's 1.1 (NDM 2026-10-27).
NEAR_TIE_REGISTRATIONS = {greedy_driver.NEAR_TIE_NATS: 'nbw-0p501', 0.9 + 1e-3: 'win2-0p9', 1.1 + 1e-3: 'dep1-fixed2-1p1'}
MAX_REPEATS = 12
# R-UPF decision cells (offload/upfix/windows.py DECISION_CELLS): the shared catalogue's prompts and bounds, run here at
# UP3's 512 tokens/request (the catalogue default is 1,500; WINDOW.md "pass 512 explicitly"). Repeats are always explicit.
WC_CELLS = ('ctl-150k-b1', 'ctl-32k-b4')


def batch_for(name: str) -> int:
    return 3 if name == CELLS[-1] else int(name[-1])
ESTIMATES = dict(zip(CELLS, (20, 35, 80, 105, 145)))
ESTIMATES.update({name: int(wc.CELLS[name].estimate_s) for name in WC_CELLS})


def keys_for(name: str) -> tuple[str, ...]:
    if name in WC_CELLS:
        return wc.CELLS[name].prompts
    return ('32k-0', '32k-1', '150k') if name == CELLS[-1] else (
        ('short',) if name == 'short-b1' else tuple('32k-' + str(i) for i in range(int(name[-1]))))


def p99(values: list[float]) -> float:
    ck.require(bool(values) and all(math.isfinite(x) and x >= 0 for x in values), 'invalid delivery gaps')
    return sorted(values)[math.ceil(.99 * len(values)) - 1]


class Http(wc.Http):
    def stream(self, body: dict[str, Json], first: threading.Event) -> dict[str, Json]:
        # Continuous usage supplies actual delivered token counts per chunk, including speculative bursts.
        body = {**body, 'ignore_eos': True, 'stream_options': {'include_usage': True, 'continuous_usage_stats': True}}
        req = urllib.request.Request(self.url + '/v1/chat/completions', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
        started = time.monotonic()
        arrivals: list[float] = []
        counts: list[int] = []
        def observe(response: Iterable[bytes]) -> Iterator[bytes]:
            for line in wc.first_signal(response, first):
                if line.startswith(b'data:') and line[5:].strip() != b'[DONE]':
                    row = record(cast(Json, json.loads(line[5:].strip())))
                    usage = row.get('usage')
                    if usage:
                        count = int(number(record(usage).get('completion_tokens', 0)))
                        old = counts[-1] if counts else 0
                        ck.require(count >= old, 'continuous usage went backwards')
                        if count > old:
                            counts.append(count)
                            arrivals.extend([time.monotonic() - started] * (count - old))
                yield line
        with urllib.request.urlopen(req, timeout=600) as response:
            summary = stream_summary(observe(response), started)
        ck.require(len(counts) > 1 and len(arrivals) == number(record(summary['usage'])['completion_tokens']),
                   'per-token delivery timing unavailable: continuous usage not supported')
        gaps = [1000 * (right - left) for left, right in zip(arrivals, arrivals[1:])]
        summary.update({'p99_inter_token_delivery_gap_ms': p99(gaps), 'delivery_gap_samples': len(gaps),
            'timing_scope': 'client delivery; tokens in one SSE/speculative burst share arrival time, within-burst gaps zero'})
        return summary


def cuts(api: wc.Api, materials: str, hashes: dict[str, Json]) -> dict[str, Json]:
    ck.require(wc.busy(api.metrics()) == 0, 'serve busy before token cuts')
    rows: dict[str, Json] = {}
    for key, prompt in PROMPTS.items():
        chars, n = wc.cut(materials, prompt, api.tokens)
        text = wc.render(materials, prompt.offset, chars)
        rows[key] = {'chars': chars, 'tokens': n, 'sha256': hashlib.sha256(text.encode()).hexdigest(),
                     'offset': prompt.offset, 'target': prompt.target}
    return {'corpus_sha256': hashes, 'prompts': rows}


def texts(cut: dict[str, Json], materials: str, hashes: dict[str, Json]) -> dict[str, tuple[str, int]]:
    ck.require(cut['corpus_sha256'] == hashes and set(record(cut['prompts'])) == set(KEYS), 'public corpus/cuts changed')
    output: dict[str, tuple[str, int]] = {}
    for key, prompt in PROMPTS.items():
        row = record(record(cut['prompts'])[key])
        text = wc.render(materials, prompt.offset, int(number(row['chars'])))
        n = int(number(row['tokens']))
        ck.require(row.get('offset') == prompt.offset and row.get('target') == prompt.target
            and .98 * prompt.target <= n <= prompt.target
            and hashlib.sha256(text.encode()).hexdigest() == row['sha256'], 'public prompt identity changed')
        output[key] = text, n
    return output


def counter(before: dict[str, float], after: dict[str, float], name: str) -> float:
    keys = {k for k in before.keys() | after.keys() if k.split('{')[0] == name}
    ck.require(bool(keys) and all(k in after and math.isfinite(after[k]) and math.isfinite(before.get(k, 0))
        and after[k] >= before.get(k, 0) for k in keys), 'missing/reset counter ' + name)
    return sum(after[k] - before.get(k, 0) for k in keys)


def run(api: wc.Api, name: str, prompts: dict[str, tuple[str, int]], *, deadline: float,
        salt: str, ledger: Path, seed: int = 42,
        sleep: Callable[[float], None] = time.sleep) -> dict[str, Json]:
    ck.require(name in (*CELLS, *WC_CELLS) and time.time() + ESTIMATES[name] < deadline, 'cell cannot fit submission deadline')
    ck.require(wc.busy(api.metrics()) == 0 and api.model().get('max_model_len') == 327680, 'busy/wrong target before cell')
    keys = keys_for(name)
    started = time.monotonic()
    first = [threading.Event() for _ in keys]
    budgets = [BUDGET] * len(keys)
    barrier = threading.Barrier(2 if name == CELLS[-1] else len(keys))
    lock = threading.Lock()
    def event(row: dict[str, Json]) -> None:
        with lock, ledger.open('a') as handle:
            handle.write(json.dumps({'cell': name, 'time': time.time(), **row}) + '\n')
    def request(i: int) -> dict[str, Json]:
        text, n = prompts[keys[i]]
        if name != CELLS[-1] or i < 2:
            barrier.wait(timeout=30)
        sent = time.monotonic() - started
        ck.require(time.time() < deadline, 'request submission deadline exhausted')
        event({'event': 'send', 'stream': i})
        try:
            row = api.stream({'model': api.model()['id'], 'messages': [{'role': 'user', 'content': text}],
                'temperature': 1, 'top_p': 1, 'seed': seed, 'priority': 10, 'max_tokens': budgets[i],
                'cache_salt': f'{salt}:{name}:{i}', 'chat_template_kwargs': TEMPLATE, 'stream': True}, first[i])
            usage = record(row['usage'])
            ck.require(usage.get('completion_tokens') == budgets[i] and row.get('finish') == 'length'
                and usage.get('prompt_tokens') == n
                and record(usage.get('prompt_tokens_details') or {}).get('cached_tokens') == 0,
                'incomplete cell/prompt usage/cache mismatch')
            span = max(number(v) for v in record(row['last_by_kind_s']).values()) - min(
                number(v) for v in record(row['first_by_kind_s']).values())
            ck.require(span > 0 and number(row['p99_inter_token_delivery_gap_ms']) >= 0, 'invalid timing')
            event({'event': 'complete', 'stream': i})
            return {**row, 'prompt_sha256': hashlib.sha256(text.encode()).hexdigest(), 'prompt_tokens': n,
                'budget': budgets[i], 'sampling': {'temperature': 1, 'top_p': 1, 'seed': seed,
                    'priority': 10, 'max_tokens': budgets[i]},
                'cache_salt_sha256': hashlib.sha256(f'{salt}:{name}:{i}'.encode()).hexdigest(),
                'decode_tps': (budgets[i] - 1) / span, 'sent_s': sent}
        except BaseException:
            event({'event': 'failed', 'stream': i})
            raise
    bench = cast(wc.Counters, load_bench())
    before = bench.parse_metrics(api.metrics())
    overlap = False
    with ThreadPoolExecutor(max_workers=len(keys)) as pool:
        futures = [pool.submit(request, i) for i in range(2 if name == CELLS[-1] else len(keys))]
        if name == CELLS[-1]:
            until = time.monotonic() + 120
            while not all(e.is_set() for e in first[:2]):
                ck.require(time.monotonic() < until and not any(f.done() for f in futures), 'decoders failed/finished before mixed arrival')
                time.sleep(.05)
            ck.require(not any(f.done() for f in futures), 'mixed cell lacks two live decoders')
            futures.append(pool.submit(request, 2))
            # Both512-token decoders must overlap the admitted long prefill.
            # Require server admission of all three while BOTH decoders are alive.
            while not overlap:
                values = re.findall(r'^vllm:num_requests_running(?:\{[^\n]*\})?\s+([^\s]+)\s*$', api.metrics(), re.M)
                ck.require(bool(values) and all(math.isfinite(float(v)) and float(v) >= 0 for v in values),
                    'mixed active request gauges missing/invalid')
                overlap = sum(float(v) for v in values) == 3 and not any(f.done() for f in futures[:2])
                ck.require(overlap or (time.monotonic() < until and not any(f.done() for f in futures[:2])),
                    '150K prefill was never admitted beside two live decoders')
                if not overlap:
                    time.sleep(.01)
        rows = [f.result() for f in futures]
    if name == CELLS[-1]:
        long_first = number(rows[2]['sent_s'])
        ck.require(all(number(r['sent_s']) + max(number(v) for v in record(r['last_by_kind_s']).values()) >= long_first
            for r in rows[:2]), 'two decoders did not overlap the admitted long prefill')
    sleep(2)
    after = bench.parse_metrics(api.metrics())
    drafts = counter(before, after, 'vllm:spec_decode_num_drafts_total')
    accepted = counter(before, after, 'vllm:spec_decode_num_accepted_tokens_total')
    seconds = counter(before, after, 'vllm:request_decode_time_seconds_sum')
    completed = counter(before, after, 'vllm:request_decode_time_seconds_count')
    ck.require(drafts > 0 and seconds > 0 and completed == len(keys),
        'no speculative rounds/decode time or completed server request count mismatch')
    return {'cell': name, 'streams': cast(Json, rows), 'batch': len(rows),
        'median_decode_tps': statistics.median(number(r['decode_tps']) for r in rows),
        'median_ttft_s': statistics.median(number(r['ttft_s']) for r in rows),
        'acceptance_per_round': accepted / drafts, 'spec_request_rounds': drafts, 'accepted_tokens': accepted,
        'server_decode_ms': 1000 * seconds, 'completed_requests': completed,
        'mixed_three_active_overlap': overlap if name == CELLS[-1] else None, 'metrics_before': cast(Json, before), 'metrics_after': cast(Json, after)}


def repeated(api: wc.Api, name: str, prompts: dict[str, tuple[str, int]], *, deadline: float,
             salt: str, ledger: Path, output: Path, repeats: int | None = None,
             sleep: Callable[[float], None] = time.sleep) -> dict[str, Json]:
    rows: list[Json] = []
    ck.require(repeats is not None or name in REPEATS, f'{name}: explicit repeats required')
    count = REPEATS[name] if repeats is None else repeats
    ck.require(1 <= count <= MAX_REPEATS, 'repeat count outside 1..12')
    ck.require(time.time() + count * ESTIMATES[name] < deadline, 'complete repeated cell cannot fit deadline')
    for repeat in range(count):
        row = run(api, name, prompts, deadline=deadline, salt=f'{salt}-repeat{repeat}', ledger=ledger,
                  seed=42 + repeat, sleep=sleep)
        row['repeat'] = repeat
        rows.append(row)
        output.write_text(json.dumps({'cell': name, 'batch': batch_for(name), 'repetitions': rows}, allow_nan=False)+'\n')
    return {'cell': name, 'batch': batch_for(name), 'repetitions': rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--mode', choices=('cuts', 'cell', 'greedy'), required=True)
    parser.add_argument('--cell', choices=(*CELLS, *WC_CELLS))
    parser.add_argument('--cuts', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--salt', default='')
    parser.add_argument('--ledger', type=Path)
    parser.add_argument('--repeats', type=int, help='cell only: repeats of this cell (WIN2 groups); default REPEATS')
    parser.add_argument('--near-tie-nats', type=float, default=greedy_driver.NEAR_TIE_NATS,
                        help='greedy only: a registered near-tie limit (NBW 0.501, WIN2 0.9+1e-3, DEP1 fixed2 1.1+1e-3)')
    args = parser.parse_args()
    ck.require(args.near_tie_nats in NEAR_TIE_REGISTRATIONS
               and (args.mode == 'greedy' or args.near_tie_nats == greedy_driver.NEAR_TIE_NATS),
               'unregistered near-tie threshold')
    ck.require(args.repeats is None or (args.mode == 'cell' and 1 <= args.repeats <= MAX_REPEATS), 'repeats: cell mode, 1..12')
    if not args.execute:
        print('DRY: short/32K B1/B3/B4 and staggered150K+2decoders; 512 tokens/request; >=4 matched requests/cell/start; no network/file')
        return 0
    ck.require(os.environ.get('WINDOW_COORDINATOR') == 'Deepseek/coordinator', 'coordinator required')
    ck.require(not args.output.exists() and time.time() < args.deadline, 'fresh output and live deadline required')
    api = Http(wc.URL)
    if args.mode == 'greedy':
        # DEP1 delegates unchanged NBW first-divergence top20/0.501nat checks.
        result = greedy_driver.greedy(api, args.deadline, *greedy_driver.greedy_inputs(), args.output,
                                      near_tie_nats=args.near_tie_nats)
        if args.near_tie_nats != greedy_driver.NEAR_TIE_NATS:
            result.update(near_tie_nats=args.near_tie_nats, greedy_registration=NEAR_TIE_REGISTRATIONS[args.near_tie_nats])
    else:
        materials, hashes = corpus(args.data_dir)
        if args.mode == 'cuts':
            result = cuts(api, materials, hashes)
        else:
            ck.require(args.cuts is not None and args.ledger is not None and bool(args.salt), 'cuts/salt/ledger required')
            result = repeated(api, str(args.cell), texts(ck.read(cast(Path, args.cuts)), materials, hashes),
                         deadline=args.deadline, salt=args.salt, ledger=cast(Path, args.ledger), output=args.output,
                         repeats=args.repeats)
    args.output.write_text(json.dumps(result, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
