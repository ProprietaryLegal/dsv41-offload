"""NBW public cells and the exact six W5/K1 greedy checks. Dry by default; numbers/hashes only."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import statistics
import threading
import time
from pathlib import Path
from typing import cast

from offload.gb10_budget import window_cells as wc
from offload.gb10_budget.capture import load_bench
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, number, record
from offload.prefill_trace.window_requests import corpus

GREEDY_SCRIPT = pf.CAMPAIGN / 'W5/greedy_check.py'
GREEDY_SHA = 'e40c52c0e9ab13d46df9b9cf49020ad87034d8651e4585e741b5eb14ef4f0845'
REFERENCE = pf.CAMPAIGN / 'K1-production/greedy-post.jsonl'
CELLS = {c.name: c for c in (
    wc.Cell('short-b1', ('short',), estimate_s=60),
    wc.Cell('32k-b1', ('32k-0',), estimate_s=90),
    wc.Cell('32k-b2', ('32k-0', '32k-1'), estimate_s=120),
    wc.Cell('32k-b3', ('32k-0', '32k-1', '32k-2'), estimate_s=150))}
PROMPTS = {key: wc.PROMPTS[key] for key in ('short', '32k-0', '32k-1', '32k-2')}


class Http(wc.Http):
    def stream(self, body: dict[str, Json], first: threading.Event) -> dict[str, Json]:
        # Fixed work, including a natural early EOS; every arm uses the same setting.
        return super().stream({**body, 'ignore_eos': True}, first)


def build(api: wc.Api, materials: str, hashes: dict[str, Json]) -> dict[str, Json]:
    ck.require(wc.busy(api.metrics()) == 0, 'serve busy before public prompt construction')
    prompts: dict[str, Json] = {}
    for prompt in PROMPTS.values():
        chars, tokens = wc.cut(materials, prompt, api.tokens)
        text = wc.render(materials, prompt.offset, chars)
        prompts[prompt.key] = {'offset': prompt.offset, 'target': prompt.target, 'chars': chars,
                               'tokens': tokens, 'sha256': hashlib.sha256(text.encode()).hexdigest()}
    return {'corpus_sha256': hashes, 'prompts': prompts}


def texts_from(cuts: dict[str, Json], materials: str, hashes: dict[str, Json]) -> dict[str, tuple[str, int]]:
    ck.require(cuts.get('corpus_sha256') == hashes, 'public corpus changed')
    rows = record(cuts['prompts'])
    ck.require(set(rows) == set(PROMPTS), 'incomplete prompt cuts')
    out: dict[str, tuple[str, int]] = {}
    for key, prompt in PROMPTS.items():
        row = record(rows[key])
        ck.require(row.get('offset') == prompt.offset and row.get('target') == prompt.target, 'prompt plan changed')
        text = wc.render(materials, prompt.offset, int(number(row['chars'])))
        ck.require(hashlib.sha256(text.encode()).hexdigest() == row.get('sha256'), 'prompt bytes changed')
        tokens = int(number(row['tokens']))
        ck.require(prompt.target * .98 <= tokens <= prompt.target, 'prompt tokens outside registered band')
        out[key] = text, tokens
    return out


def metric_delta(before: dict[str, float], after: dict[str, float], name: str) -> float:
    keys = {k for k in before.keys() | after.keys() if k.split('{')[0] == name}
    ck.require(bool(keys) and all(k in after and after[k] >= before.get(k, 0) for k in keys),
               f'missing/reset server counter {name}')
    return sum(after[k] - before.get(k, 0) for k in keys)


def summarize(result: dict[str, Json], before: dict[str, float], after: dict[str, float]) -> dict[str, Json]:
    streams = ck.rows(result['streams'])
    ck.require(len(streams) == result['batch'], 'cell stream count mismatch')
    rates: list[float] = []
    ttfts: list[float] = []
    for stream in streams:
        usage = record(stream['usage'])
        ck.require(usage.get('completion_tokens') == 1500 and stream.get('finish') == 'length', 'cell output budget incomplete')
        first, last = record(stream['first_by_kind_s']), record(stream['last_by_kind_s'])
        span = max(number(v) for v in last.values()) - min(number(v) for v in first.values())
        ck.require(span > 0, 'no measured decode span')
        rate = 1499 / span
        stream['decode_tps'] = rate
        rates.append(rate)
        ttfts.append(number(stream['ttft_s']))
    rounds = metric_delta(before, after, 'vllm:spec_decode_num_drafts_total')
    accepted = metric_delta(before, after, 'vllm:spec_decode_num_accepted_tokens_total')
    seconds = metric_delta(before, after, 'vllm:request_decode_time_seconds_sum')
    ck.require(rounds > 0 and seconds > 0, 'no spec rounds/server decode time')
    return {**result, 'median_decode_tps': statistics.median(rates), 'median_ttft_s': statistics.median(ttfts),
            'acceptance_per_round': accepted / rounds, 'spec_request_rounds': rounds,
            'server_decode_ms_per_spec_request_round': 1000 * seconds / rounds,
            'round_time_scope': 'server request-decode duration sum / spec request-round count; includes terminal effects; not engine step wall time',
            'metrics_before': cast(Json, before), 'metrics_after': cast(Json, after),
            'status': 'MEASURED unprofiled client timing and server counters'}


def greedy_inputs() -> tuple[list[str], list[dict[str, Json]]]:
    ck.require(pf.digest(GREEDY_SCRIPT) == GREEDY_SHA, 'W5 greedy source changed')
    assignments = [n for n in ast.parse(GREEDY_SCRIPT.read_text()).body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'PROMPTS' for t in n.targets)]
    ck.require(len(assignments) == 1, 'greedy PROMPTS missing')
    prompts = ast.literal_eval(assignments[0].value)
    reference = [record(cast(Json, json.loads(line))) for line in REFERENCE.read_text().splitlines()]
    ck.require(isinstance(prompts, list) and len(prompts) == 6 and all(isinstance(p, str) for p in prompts), 'six greedy prompts required')
    ck.require([r.get('i') for r in reference] == list(range(6)) and all(isinstance(r.get('text'), str) for r in reference),
               'six ordered K1 reference outputs required')
    return cast(list[str], prompts), reference


# W8 (2026-10-24, LANES.md "W8 native A/B DONE"; W8/neartie_check.py): the native drafter changes the target's
# verify-row shapes, so greedy text can flip at near-ties (C1 = C2 6/6, C1 vs X 4/6 divergent; every divergence a near-tie
# on a 0.5-nat logit grid or an exact tie). A native divergence therefore passes only when, at its first differing token,
# both tokens are in the target's top-20 and within NEAR_TIE_NATS of each other; anything else is a STOP.
NEAR_TIE_NATS = 0.5 + 1e-3
MODEL = 'DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2'


def token_ids(value: Json) -> list[int]:
    ck.require(isinstance(value, list) and all(isinstance(t, int) and not isinstance(t, bool) for t in value), 'token ids expected')
    return cast(list[int], value)


def near_tie(api: wc.Http, prompt: str, reference: str, text: str,
             threshold: float = NEAR_TIE_NATS, *, record_distribution: bool = False,
             probe_salt: str | None = None) -> dict[str, Json]:
    """WIN2 opt-ins (defaults unchanged): record_distribution keeps the probe's top-20 and prefix token ids in the row
    (window output only, never git) so a later control arm can measure the distribution difference at the same prefix
    (greedy_kl); probe_salt sends a cache_salt so the probe does not reuse the generation's prefix cache."""
    p = token_ids(api.post('/tokenize', {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
                                         'add_generation_prompt': True, 'chat_template_kwargs': {'thinking': False}})['tokens'])
    ta = token_ids(api.post('/tokenize', {'model': MODEL, 'prompt': reference, 'add_special_tokens': False})['tokens'])
    tb = token_ids(api.post('/tokenize', {'model': MODEL, 'prompt': text, 'add_special_tokens': False})['tokens'])
    j = next((k for k, (u, v) in enumerate(zip(ta, tb)) if u != v), None)
    ck.require(j is not None, 'greedy divergence is a length/prefix difference, not a token near-tie')
    index = cast(int, j)
    prefix = [*p, *ta[:index]]
    body: dict[str, Json] = {'model': MODEL, 'prompt': cast(Json, prefix), 'max_tokens': 1, 'temperature': 0,
                             'logprobs': 20, 'return_tokens_as_token_ids': True, 'priority': 10}
    if probe_salt is not None:
        ck.require(bool(probe_salt.strip()), 'empty probe cache salt')
        body['cache_salt'] = probe_salt
    r = api.post('/v1/completions', body)
    logprobs = record(record(ck.rows(r['choices'])[0])['logprobs'])
    tops = logprobs['top_logprobs']
    ck.require(isinstance(tops, list) and len(tops) == 1, 'one top-logprob position expected')
    top = record(cast(list[Json], tops)[0])
    lps = {int(str(k).split(':')[1]): float(number(v)) for k, v in top.items()}
    a, b = ta[index], tb[index]
    best = max(lps.values())
    ok = a in lps and b in lps and abs(lps[a] - lps[b]) <= threshold and max(lps[a], lps[b]) >= best - threshold
    result: dict[str, Json] = {'token_index': index, 'near_tie': ok,
            'margin_nats': None if a not in lps or b not in lps else round(abs(lps[a] - lps[b]), 4),
            'gap_to_top_nats': None if a not in lps or b not in lps else round(best - max(lps[a], lps[b]), 4)}
    if record_distribution:
        result.update(top20_logprobs=cast(Json, dict(top)), probe_prefix_token_ids=cast(Json, prefix),
                      reference_token_id=a, experiment_token_id=b)
    return result


def greedy(api: wc.Http, deadline: float, prompts: list[str], reference: list[dict[str, Json]],
           output: Path, near_tie_nats: float = NEAR_TIE_NATS, *, record_distribution: bool = False,
           probe_salt_prefix: str | None = None) -> dict[str, Json]:
    # near_tie_nats defaults to NBW's registered 0.501; only a caller's own registration may pass another
    # (TV1 r10: NDM 2026-10-27 "You can loosen the nats to .9").
    rows: list[Json] = []
    for i, prompt in enumerate(prompts):
        ck.require(time.time() + 30 < deadline, 'greedy submission deadline reached')
        ck.require(wc.busy(api.metrics()) == 0, 'serve busy before greedy check')
        # Exactly W5 greedy_check.run settings; text lives in memory only.
        response = api.post('/v1/chat/completions', {
            'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0, 'max_tokens': 300, 'chat_template_kwargs': {'thinking': False}})
        choices = ck.rows(response['choices'])
        ck.require(len(choices) == 1, 'greedy choices mismatch')
        text = record(choices[0]['message'])['content']
        ck.require(isinstance(text, str), 'greedy text missing')
        same = text == reference[i]['text']
        row: dict[str, Json] = {'i': i, 'identical': same, 'sha256': hashlib.sha256(str(text).encode()).hexdigest(),
                                'reference_sha256': hashlib.sha256(str(reference[i]['text']).encode()).hexdigest()}
        if not same:
            row.update(near_tie(api, prompt, str(reference[i]['text']), str(text), near_tie_nats,
                                record_distribution=record_distribution,
                                probe_salt=None if probe_salt_prefix is None else f'{probe_salt_prefix}:probe:{i}'))
        rows.append(row)
        output.write_text(json.dumps({'checks': rows}, allow_nan=False) + '\n')
        ck.require(same or row.get('near_tie') is True, f'greedy divergence on K1 reference prompt {i} is not a near-tie')
    ck.require(len(rows) == 6, 'six greedy checks required')
    identical = sum(1 for r in rows if record(r)['identical'] is True)
    return {'checks': rows, 'identical': identical, 'near_tie': 6 - identical}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--data-dir', type=Path, default=Path('/opt/dsv41/data/bench-data'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cuts', type=Path)
    ap.add_argument('--cell', choices=tuple(CELLS))
    ap.add_argument('--mode', choices=('build', 'cell', 'greedy'), required=True)
    ap.add_argument('--deadline-utc', type=float, default=0)
    ap.add_argument('--salt', default='')
    ap.add_argument('--ledger', type=Path)
    args = ap.parse_args(argv)
    if not args.execute:
        print('DRY RUN: NBW four public cells, T=1/top_p=1, 1500 tokens; six W5/K1 greedy checks; no network or file')
        return 0
    ck.require(os.environ.get('WINDOW_COORDINATOR') == 'Deepseek/coordinator', 'coordinator required')
    ck.require(not args.output.exists(), 'fresh cell output required')
    ck.require(time.time() < args.deadline_utc, 'submission deadline reached')
    api = Http(wc.URL)
    if args.mode == 'greedy':
        result = greedy(api, args.deadline_utc, *greedy_inputs(), args.output)
    else:
        materials, hashes = corpus(args.data_dir)
        if args.mode == 'build':
            result = build(api, materials, hashes)
        else:
            ck.require(args.cuts is not None and args.cell is not None and bool(args.salt), 'cell/cuts/salt required')
            texts = texts_from(ck.read(cast(Path, args.cuts)), materials, hashes)
            bench = cast(wc.Counters, load_bench())
            before = bench.parse_metrics(api.metrics())
            plan = wc.Plan(args.deadline_utc, pf.MEM_FLOOR_KB, [], args.salt, args.ledger)
            result = wc.run_cell(api, CELLS[args.cell], texts, plan)
            time.sleep(2)  # counter publication, excluded from measured request timing
            after = bench.parse_metrics(api.metrics())
            result['spec'] = bench.spec_delta(before, after)
            result = summarize(result, before, after)
    args.output.write_text(json.dumps(result, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
