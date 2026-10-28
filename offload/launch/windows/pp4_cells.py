"""PP4 public cells and the exact six W5/K1 greedy checks. Dry by default; numbers/hashes only."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import statistics
import threading
import time
import urllib.request
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import cast

from offload.gb10_budget import window_cells as wc
from offload.gb10_budget.capture import load_bench
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import pp4_metrics as metric
from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, number, record
from offload.prefill_trace.window_requests import HEADER

GREEDY_SCRIPT = pf.CAMPAIGN / 'W5/greedy_check.py'
GREEDY_SHA = 'e40c52c0e9ab13d46df9b9cf49020ad87034d8651e4585e741b5eb14ef4f0845'
REFERENCE = pf.CAMPAIGN / 'K1-production/greedy-post.jsonl'
FAMILIES = metric.FAMILIES
BASE_CELLS = {c.name: c for c in (
    wc.Cell('short-b1', ('short',), max_tokens=512, estimate_s=4*20),
    wc.Cell('32k-b1', ('32k-0',), max_tokens=512, estimate_s=4*35),
    wc.Cell('32k-b4', ('32k-0', '32k-1', '32k-2', '32k-3'), max_tokens=512, estimate_s=2*105))}
CELLS = BASE_CELLS
PROMPTS = {'short': 1024, **{f'short-{i}': 1024 for i in range(1, 4)},
           **{f'32k-{i}': 32000 for i in range(20)}}
CORPUS_FILES = {'opinions': 'opinions_cl.jsonl', 'filings': 'filings_recap.jsonl'}
CORPUS_SHA256 = {
    'opinions': '80461203f88a6b837eed8fdf9b7e77a40660acb36dfa9887e00a0ae37483b6a1',
    'filings': 'b1f2744634cddb0716074c451df5f46e2da3d851324ab794f0caff05ab7202a9'}
BODY_CHARS = {'opinions': 10_195_729, 'filings': 1_592_473}
BODY_SHA256 = {
    'opinions': 'ce1a8a50d0e5e490bcc72e8f444da874e0c10e9d25aa76c18e012b27a5517812',
    'filings': 'f92ce1677d0f523fb037ef858dbb69b1e2a6ca2b868b4e097c4fb82bd762170a'}
# Offline tokenization of ALL nonoverlapping 32,000-character chunks (319/50)
# measured maxima 4.65658/4.55776 chars/token. Reserve >1.7x that maximum.
# This is a conservative planning bound, enforced on actual cuts; never pad or
# repeat corpus text to satisfy it. See PP4-COORDINATOR.md for tokenizer provenance.
CHARS_PER_TOKEN_BOUND = 8
MIN_START_SPACING = 4096
CUT_REGISTRATION = 'pp4-r9-per-family-public-cuts-v1'
GREEDY_REGISTRATION = 'pp4-r14-near-tie-1p1'
NEAR_TIE_NATS = 1.1 + 1e-3  # NDM's 1.1-nat ceiling for the drafting fix, with NBW's existing rounding allowance.
MODEL = 'DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2'


def wave_prompts(name: str, wave: int) -> tuple[str, ...]:
    ck.require(name in CELLS and 0 <= wave < metric.WAVES, 'unknown frozen cell/wave')
    if name == 'short-b1':
        return ('short' if wave == 0 else f'short-{wave}',)
    if name == '32k-b1':
        return (f'32k-{wave}',)
    ck.require(wave < 2, 'B4 requires exactly two whole groups')
    return tuple(f'32k-{4+wave*4+i}' for i in range(4))


def family_corpus(root: Path, family: str) -> tuple[str, dict[str, Json]]:
    """Disjoint public source clusters; no PP2 fit/held-out capture is read."""
    ck.require(family in FAMILIES, 'unknown source family')
    path = root / CORPUS_FILES[family]
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    ck.require(digest == CORPUS_SHA256[family], family+': frozen public corpus changed')
    docs: list[str] = []
    for line in raw.decode().splitlines():
        row = record(cast(Json, json.loads(line)))
        ck.require(isinstance(row.get('text'), str), 'public source missing text')
        docs.append(str(row['text']))
    random.Random(42).shuffle(docs)
    return HEADER + '\n\n'.join(docs), {path.name: digest}


def family_prompts(materials: str, hashes: dict[str, Json]) -> tuple[dict[str, wc.Prompt], dict[str, Json]]:
    """One frozen layout for admission and measurement, using only this family."""
    families = [f for f in FAMILIES if hashes == {CORPUS_FILES[f]: CORPUS_SHA256[f]}]
    ck.require(len(families) == 1, 'unknown frozen public family')
    family = families[0]
    text = wc.body(materials)
    ck.require(len(text) == BODY_CHARS[family]
               and hashlib.sha256(text.encode()).hexdigest() == BODY_SHA256[family],
               family+': public family body changed')
    widths = [target * CHARS_PER_TOKEN_BOUND for target in PROMPTS.values()]
    disjoint = sum(widths) <= len(text)
    # If disjoint reservations cannot fit, spread all starts across the family.
    # Reserve the largest possible cut at the last start, including every retry.
    step = (len(text) - max(widths)) // (len(widths)-1)
    ck.require(disjoint or step >= MIN_START_SPACING, family+': insufficient public text for spaced cuts')
    prompts: dict[str, wc.Prompt] = {}
    offset = 0
    for i, (key, target) in enumerate(PROMPTS.items()):
        if not disjoint:
            offset = i*step
        ck.require(offset + widths[i] <= len(text), family+': reserved prompt cut outside public corpus')
        prompts[key] = wc.Prompt(key, target, offset)
        if disjoint:
            offset += widths[i]
    plan: dict[str, Json] = {'registration': CUT_REGISTRATION, 'family': family,
        'body_chars': len(text), 'body_sha256': BODY_SHA256[family],
        'chars_per_token_bound': CHARS_PER_TOKEN_BOUND, 'minimum_start_spacing_chars': MIN_START_SPACING,
        'overlap_required': not disjoint}
    return prompts, plan


def require_cache_salt(body: dict[str, Json]) -> None:
    cache = body.get('cache_salt')
    ck.require(isinstance(cache, str) and bool(cache.strip()), 'cold per-stream cache_salt required')


class Http(wc.Http):
    def post(self, endpoint: str, body: dict[str, Json]) -> dict[str, Json]:
        if endpoint in ('/v1/chat/completions', '/v1/completions'):
            require_cache_salt(body)
        return super().post(endpoint, body)

    def stream(self, body: dict[str, Json], first: threading.Event) -> dict[str, Json]:
        require_cache_salt(body)
        # Fixed public work. A decode that exceeds120s is censored, never silently discarded.
        request = urllib.request.Request(self.url + '/v1/chat/completions',
            json.dumps({**body, 'ignore_eos': True}).encode(), {'Content-Type': 'application/json'})
        started = time.monotonic()
        from offload.prefill_trace.window_requests import stream_summary
        with urllib.request.urlopen(request, timeout=600) as response:
            return stream_summary(decode_bound(wc.first_signal(response, first), first), started)


def decode_bound(lines: Iterable[bytes], first: threading.Event) -> Iterator[bytes]:
    began: float | None = None
    for line in lines:
        now = time.monotonic()
        if first.is_set():
            began = now if began is None else began
        if began is not None and now - began > 120:
            raise TimeoutError('CENSORED: 120-second decode bound; preserve partial ledger, no candidate')
        yield line


def build(api: wc.Api, materials: str, hashes: dict[str, Json]) -> dict[str, Json]:
    ck.require(wc.busy(api.metrics()) == 0, 'serve busy before public prompt construction')
    layout, plan = family_prompts(materials, hashes)
    prompts: dict[str, Json] = {}
    for prompt in layout.values():
        # Give the unchanged token-count search only its own reserved span.
        # A tokenizer needing >8 chars/token refuses before crossing that span.
        bounded = materials[:len(HEADER) + prompt.offset + prompt.target*CHARS_PER_TOKEN_BOUND]
        chars, tokens = wc.cut(bounded, prompt, api.tokens)
        text = wc.render(materials, prompt.offset, chars)
        prompts[prompt.key] = {'offset': prompt.offset, 'target': prompt.target, 'chars': chars,
                               'tokens': tokens, 'sha256': hashlib.sha256(text.encode()).hexdigest()}
    return {'corpus_sha256': hashes, 'plan': plan, 'prompts': prompts}


def texts_from(cuts: dict[str, Json], materials: str, hashes: dict[str, Json]) -> dict[str, tuple[str, int]]:
    ck.require(cuts.get('corpus_sha256') == hashes, 'public corpus changed')
    layout, plan = family_prompts(materials, hashes)
    ck.require(cuts.get('plan') == plan, 'public family cut registration changed')
    rows = record(cuts['prompts'])
    ck.require(set(rows) == set(PROMPTS), 'incomplete prompt cuts')
    out: dict[str, tuple[str, int]] = {}
    for key, prompt in layout.items():
        row = record(rows[key])
        ck.require(row.get('offset') == prompt.offset and row.get('target') == prompt.target, 'prompt plan changed')
        chars = number(row['chars'])
        ck.require(chars == int(chars) and 0 < chars <= prompt.target*CHARS_PER_TOKEN_BOUND,
                   'public cut exceeds conservative character reserve')
        text = wc.render(materials, prompt.offset, int(chars))
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
        ck.require(usage.get('completion_tokens') == 512 and stream.get('finish') == 'length', 'cell output budget incomplete')
        first, last = record(stream['first_by_kind_s']), record(stream['last_by_kind_s'])
        span = max(number(v) for v in last.values()) - min(number(v) for v in first.values())
        ck.require(span > 0, 'no measured decode span')
        rate = 511 / span
        stream['decode_tps'] = rate
        rates.append(rate)
        ttfts.append(number(stream['ttft_s']))
    rounds = metric_delta(before, after, 'vllm:spec_decode_num_drafts_total')
    accepted = metric_delta(before, after, 'vllm:spec_decode_num_accepted_tokens_total')
    seconds = metric_delta(before, after, 'vllm:request_decode_time_seconds_sum')
    ck.require(rounds > 0 and seconds > 0, 'no spec rounds/server decode time')
    return {**result, 'median_decode_tps': statistics.median(rates), 'median_ttft_s': statistics.median(ttfts),
            'acceptance_per_round': accepted / rounds, 'spec_request_rounds': rounds,
            'accepted_tokens': accepted, 'server_decode_seconds': seconds,
            'server_decode_ms_per_spec_request_round': 1000 * seconds / rounds,
            'round_time_scope': 'server request-decode duration sum / spec request-round count; includes terminal effects; not engine step wall time',
            'metrics_before': cast(Json, before), 'metrics_after': cast(Json, after),
            'status': 'MEASURED unprofiled client timing and server counters'}


class SeededApi:
    """Match each wave seed across starts; retain explicit stream receipt settings."""
    def __init__(self, api: wc.Api, seed: int) -> None:
        self.api, self.seed = api, seed
    def tokens(self, text: str) -> int: return self.api.tokens(text)
    def model(self) -> dict[str, Json]: return self.api.model()
    def metrics(self) -> str: return self.api.metrics()
    def profile(self, verb: str) -> None: self.api.profile(verb)
    def stream(self, body: dict[str, Json], first: threading.Event) -> dict[str, Json]:
        require_cache_salt(body)
        return self.api.stream({**body, 'seed': self.seed, 'ignore_eos': True}, first)


def repeated_cell(api: wc.Api, name: str, cuts: dict[str, Json], data_dir: Path, plan: wc.Plan,
                  output: Path, selected: tuple[int, ...] | None = None) -> dict[str, Json]:
    """Four independently salted requests/waves; save each sufficient-statistic row immediately."""
    bench = cast(wc.Counters, load_bench())
    rows: list[Json] = []
    selected = tuple(range(metric.group_count(name))) if selected is None else selected
    for wave in selected:
        family = FAMILIES[wave % len(FAMILIES)]
        materials, hashes = family_corpus(data_dir, family)
        texts = texts_from(record(cuts[family]), materials, hashes)
        prompts = wave_prompts(name, wave)
        cell = wc.Cell(name+'-wave'+str(wave), prompts, max_tokens=512)
        before = bench.parse_metrics(api.metrics())
        seed = 42+wave
        subplan = wc.Plan(plan.deadline, plan.floor_kb, plan.telemetry, plan.salt_prefix+'-wave'+str(wave), plan.ledger)
        result = wc.run_cell(SeededApi(api, seed), cell, texts, subplan)
        for stream in ck.rows(result['streams']):
            record(stream['sampling']).update({'seed': seed, 'ignore_eos': True})
        time.sleep(2)  # metrics publication, excluded from request timings
        after = bench.parse_metrics(api.metrics())
        result['spec'] = bench.spec_delta(before, after)
        completed = metric_delta(before, after, 'vllm:request_decode_time_seconds_count')
        ck.require(completed == cell.batch, 'server completed-request count mismatch')
        rows.append({**summarize(result, before, after), 'completed_requests': completed, 'wave': wave, 'family': family})
        output.write_text(json.dumps({'status': 'PARTIAL', 'metric_registration': metric.REGISTRATION,
                                     'cell': name, 'waves': rows}, allow_nan=False)+'\n')
    observations = ck.rows(cast(Json, rows))
    streams = [stream for row in observations for stream in ck.rows(row['streams'])]
    return {'cell': name, 'batch': CELLS[name].batch, 'waves': rows, 'metric_registration': metric.REGISTRATION,
            **(metric.start_interval(observations) if selected == tuple(range(metric.group_count(name))) else metric.pool(observations)), 'independent_requests': len(streams),
            'median_decode_tps': statistics.median(number(s['decode_tps']) for s in streams),
            'median_ttft_s': statistics.median(number(s['ttft_s']) for s in streams),
            'streams': cast(Json, streams), 'status': 'MEASURED independent whole request groups'}


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


def near_tie_values(top: dict[str, Json], reference_id: int, experiment_id: int) -> dict[str, Json]:
    """NBW's two-token/top gap semantics with PP4's registered 1.1-nat ceiling."""
    result: dict[str, Json] = {'top20_logprobs': top, 'reference_token_id': reference_id,
        'experiment_token_id': experiment_id, 'near_tie': False}
    try:
        ck.require(len(top) == 20, 'exactly twenty distinct top-logprob tokens required')
        ck.require(all(k.startswith('token_id:') and k[9:].isdigit() for k in top), 'top20 token IDs malformed')
        lps = {int(k[9:]): number(v) for k, v in top.items()}
        ck.require(len(lps) == 20, 'duplicate top20 token IDs')
        ck.require(all(-9999 < v <= 0 for v in lps.values()), 'clamped or invalid top20 logprobs')
        ck.require(reference_id in lps and experiment_id in lps, 'both differing tokens must be in top20')
        a, b = lps[reference_id], lps[experiment_id]
        best = max(lps.values())
        result.update(margin_nats=abs(a-b), gap_to_top_nats=best-max(a, b),
                      experiment_gap_to_top_nats=best-b)
        ck.require(abs(a-b) <= NEAR_TIE_NATS, 'divergence exceeds 1.1 nats plus 1e-3 rounding allowance')
        ck.require(best-max(a, b) <= NEAR_TIE_NATS, 'better token exceeds 1.1 nats from top plus 1e-3 rounding allowance')
        result['near_tie'] = True
    except ValueError as exc:
        result['diagnostic_error'] = str(exc)
    return result


def token_ids(value: Json) -> list[int]:
    ck.require(isinstance(value, list) and all(isinstance(t, int) and not isinstance(t, bool) and t >= 0
                                             for t in value), 'nonnegative token ids expected')
    return cast(list[int], value)


def near_tie(api: wc.Http, prompt: str, reference: str, text: str, salt: str) -> dict[str, Json]:
    p = token_ids(api.post('/tokenize', {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
        'add_generation_prompt': True, 'chat_template_kwargs': {'thinking': False}})['tokens'])
    ta = token_ids(api.post('/tokenize', {'model': MODEL, 'prompt': reference, 'add_special_tokens': False})['tokens'])
    tb = token_ids(api.post('/tokenize', {'model': MODEL, 'prompt': text, 'add_special_tokens': False})['tokens'])
    result: dict[str, Json] = {'near_tie': False, 'reference_tokens': len(ta), 'experiment_tokens': len(tb),
                               'probe_cache_salt': salt}
    # Round 15: a real flip changes everything after it, so the two continuations usually end at different lengths.
    # Search the common prefix for the first differing token; only a pure prefix (no differing token) is refused.
    index = next((k for k, (u, v) in enumerate(zip(ta, tb)) if u != v), None)
    ck.require(index is not None, 'greedy divergence has no differing token: a pure prefix difference')
    index = cast(int, index)
    response = api.post('/v1/completions', {'model': MODEL, 'prompt': cast(Json, [*p, *ta[:index]]),
        'max_tokens': 1, 'temperature': 0, 'logprobs': 20, 'return_tokens_as_token_ids': True,
        'priority': 10, 'cache_salt': salt})
    choices = ck.rows(response['choices'])
    ck.require(len(choices) == 1, 'one top-logprob choice expected')
    tops = record(choices[0]['logprobs'])['top_logprobs']
    ck.require(isinstance(tops, list) and len(tops) == 1, 'one top-logprob position expected')
    top = record(cast(list[Json], tops)[0])
    return {**result, 'token_index': index, **near_tie_values(top, ta[index], tb[index])}


def greedy(api: wc.Http, deadline: float, prompts: list[str], reference: list[dict[str, Json]],
           output: Path) -> dict[str, Json]:
    """PP4-only registration; strict W8 identity and shared NBW remain unchanged."""
    ck.require(len(prompts) == len(reference) == 6 and [r.get('i') for r in reference] == list(range(6))
               and all(isinstance(r.get('text'), str) for r in reference), 'six ordered greedy references required')
    # The output path binds the run/arm; a fresh nonce also isolates retries.
    salt_root = 'pp4-r13-'+hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:16]+'-'+uuid.uuid4().hex
    rows: list[Json] = []
    divergent = 0
    for i, prompt in enumerate(prompts):
        ck.require(time.time()+30 < deadline, 'greedy submission deadline reached')
        ck.require(wc.busy(api.metrics()) == 0, 'serve busy before greedy check')
        generation_salt = f'{salt_root}-{i}-generation'
        response = api.post('/v1/chat/completions', {'model': MODEL,
            'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0, 'max_tokens': 300,
            'chat_template_kwargs': {'thinking': False}, 'cache_salt': generation_salt})
        choices = ck.rows(response['choices'])
        ck.require(len(choices) == 1, 'greedy choices mismatch')
        text = record(choices[0]['message'])['content']
        ck.require(isinstance(text, str), 'greedy text missing')
        same = text == reference[i]['text']
        row: dict[str, Json] = {'i': i, 'identical': same, 'cache_salt': generation_salt,
            'sha256': hashlib.sha256(str(text).encode()).hexdigest(),
            'reference_sha256': hashlib.sha256(str(reference[i]['text']).encode()).hexdigest()}
        rows.append(row)
        try:
            if not same:
                divergent += 1
                row.update(near_tie(api, prompt, str(reference[i]['text']), str(text), f'{salt_root}-{i}-probe'))
            # Persist full diagnostics before applying the registered near-tie gate.
            output.write_text(json.dumps({'greedy_registration': GREEDY_REGISTRATION, 'checks': rows}, allow_nan=False)+'\n')
        except Exception as exc:
            row.update(near_tie=False, diagnostic_error=f'{type(exc).__name__}: {exc}')
            output.write_text(json.dumps({'greedy_registration': GREEDY_REGISTRATION, 'checks': rows}, allow_nan=False)+'\n')
            raise
        ck.require(same or row.get('near_tie') is True,
                   f'greedy divergence on K1 reference prompt {i} fails {GREEDY_REGISTRATION}: {row.get("diagnostic_error")}')
    return {'greedy_registration': GREEDY_REGISTRATION, 'checks': rows, 'identical': 6-divergent, 'near_tie': divergent}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--data-dir', type=Path, default=Path('/opt/dsv41/data/bench-data'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cuts', type=Path)
    ap.add_argument('--cell', choices=tuple(CELLS))
    ap.add_argument('--mode', choices=('build', 'cell', 'greedy', 'warm'), required=True)
    ap.add_argument('--deadline-utc', type=float, default=0)
    ap.add_argument('--salt', default='')
    ap.add_argument('--ledger', type=Path)
    ap.add_argument('--profile-wave', type=int, choices=(0, 1), help='separate profiled companion only; never headline')
    args = ap.parse_args(argv)
    if not args.execute:
        print('DRY RUN: PP4 three public cells, four B1 groups/two B4 groups across two source families, T=1/top_p=1, 512 tokens; six W5/K1 greedy checks; no network or file')
        return 0
    ck.require(os.environ.get('WINDOW_COORDINATOR') == 'Deepseek/coordinator', 'coordinator required')
    ck.require(not args.output.exists(), 'fresh cell output required')
    ck.require(time.time() < args.deadline_utc, 'submission deadline reached')
    api = Http(wc.URL)
    if args.mode == 'greedy':
        result = greedy(api, args.deadline_utc, *greedy_inputs(), args.output)
    else:
        if args.mode == 'build':
            result = {f: build(api, *family_corpus(args.data_dir, f)) for f in FAMILIES}
        elif args.mode == 'warm':
            ck.require(args.cuts is not None, 'warmup requires frozen cuts')
            materials, hashes = family_corpus(args.data_dir, FAMILIES[0])
            texts = texts_from(record(ck.read(cast(Path, args.cuts))[FAMILIES[0]]), materials, hashes)
            plan = wc.Plan(args.deadline_utc, pf.MEM_FLOOR_KB, [], args.salt, args.ledger)
            result = wc.run_cell(api, wc.Cell('warmup', ('32k-0',), max_tokens=64, warmup=True), texts, plan)
        else:
            ck.require(args.cuts is not None and args.cell is not None and bool(args.salt), 'cell/cuts/salt required')
            plan = wc.Plan(args.deadline_utc, pf.MEM_FLOOR_KB, [], args.salt, args.ledger)
            result = repeated_cell(api, str(args.cell), ck.read(cast(Path, args.cuts)), args.data_dir, plan, args.output,
                None if args.profile_wave is None else (args.profile_wave,))
            if args.profile_wave is not None:
                result['status'] = 'PROFILED companion; excluded from headline'
                result['metric_registration'] = 'pp4-profile-only'
    args.output.write_text(json.dumps(result, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
