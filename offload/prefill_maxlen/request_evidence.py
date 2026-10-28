"""Extract numeric-only request evidence from existing files; never contacts a server."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

from offload.prefill_maxlen.evidence import Json, number, record


def read_rows(path: Path) -> list[dict[str, Json]]:
    return [record(cast(Json, json.loads(line))) for line in path.read_text().splitlines()
            if line.startswith('{')]


def bench(row: dict[str, Json]) -> dict[str, Json]:
    # Whitelist: raw prompts, generated text and salts never enter the artifact.
    keys = ('run_id', 'request_index', 'config_label', 'rep', 'target_ctx', 'concurrency',
            'mixed', 'cache_state', 'ts', 'occupied_prompt_tokens', 'prompt_tokens',
            'completion_tokens', 'ttft_s', 'first_visible_s', 'total_s', 'start_offset_s',
            'first_by_kind_s', 'last_by_kind_s', 'reasoning_tokens', 'answer_tokens',
            'reasoning_token_source', 'answer_token_source', 'step_gap_ms', 'step_events',
            'step_times_ms', 'finish')
    out = {key: row[key] for key in keys if key in row}
    usage = record(row.get('usage', {}))
    prompt = usage.get('prompt_tokens')
    details = record(usage.get('prompt_tokens_details') or {})
    cached = details.get('cached_tokens')
    out['cached_tokens'] = cached
    out['uncached_prompt_tokens'] = (number(prompt) - number(cached)
                                     if prompt is not None and cached is not None else None)
    out['status'] = 'failed' if row.get('error') else 'completed'
    out['after_first_token_s'] = (number(row['total_s']) - number(row['ttft_s'])
                                  if row.get('ttft_s') is not None else None)
    for kind in ('reasoning', 'answer'):
        first = record(row.get('first_by_kind_s') or {}).get(kind)
        last = record(row.get('last_by_kind_s') or {}).get(kind)
        out[f'{kind}_delivery_span_s'] = (number(last) - number(first)
                                         if first is not None and last is not None else None)
    out['prefill_gpu_s'] = None
    out['timing_boundary'] = 'TTFT includes queue/host/first generation; SSE gaps are delivery, not engine steps'
    return out


def needle(row: dict[str, Json]) -> dict[str, Json]:
    usage = record(row['usage'])
    details = record(usage.get('prompt_tokens_details') or {})
    comp = record(usage.get('completion_tokens_details') or {})
    prompt, cached = usage.get('prompt_tokens'), details.get('cached_tokens')
    reasoning, completion = comp.get('reasoning_tokens'), usage.get('completion_tokens')
    return {'kind': 'needle', 'target_ctx': row['ctx'], 'rep': row['rep'], 'slot': row['slot'],
            'concurrency': row['concurrent'], 'prompt_tokens': prompt, 'cached_tokens': cached,
            'uncached_prompt_tokens': number(prompt) - number(cached)
            if prompt is not None and cached is not None else None,
            'completion_tokens': completion, 'reasoning_tokens': reasoning,
            'answer_and_control_tokens': number(completion) - number(reasoning)
            if completion is not None and reasoning is not None else None,
            'total_s': row['elapsed_s'], 'status': 'completed' if row['ok'] else 'failed',
            'ttft_s': None, 'prefill_gpu_s': None,
            'timing_boundary': 'non-streaming full response; no segment timestamps'}


def prefix(row: dict[str, Json]) -> list[dict[str, Json]]:
    if row['kind'] == 'partial':
        # This file omits total prompt length, completion usage and all timing.
        return [{'kind': 'partial', 'target_ctx': row['n'], 'cached_tokens': row['cached'],
                 'shared_tokens': row['common_tokens'], 'prompt_tokens': None,
                 'uncached_prompt_tokens': None, 'ttft_s': None, 'total_s': None,
                 'timing_boundary': 'counts only; cannot infer total from nominal n'}]
    times = row.get('elapsed')
    if not isinstance(times, list) or len(times) != 3:
        raise ValueError('prefix identity requires cold/hit/control elapsed triple')
    prompt = row['prompt_tokens']
    return [{'kind': 'prefix-' + kind, 'target_ctx': row['n'], 'prompt_tokens': prompt,
             'cached_tokens': row[key], 'uncached_prompt_tokens': number(prompt) - number(row[key]),
             'ttft_s': None, 'prefill_gpu_s': None, 'total_s': elapsed,
             'reasoning_tokens': None, 'answer_tokens': None,
             'timing_boundary': 'non-streaming response; completion usage not recorded'}
            for kind, key, elapsed in zip(('cold', 'hit', 'control'),
                                         ('cold_cached', 'cached', 'control_cached'), times)]


def memory_timing(rows: list[dict[str, Json]]) -> list[Json]:
    """Extract actual host timestamps and disk-reader times, never infer GPU time."""
    out: list[Json] = []
    previous_step: dict[str, Json] | None = None
    for row in rows:
        if row.get('ev') not in ('phase', 'step'):
            raise ValueError('unknown memprobe event; review new timing schema')
        item: dict[str, Json] = {key: row[key] for key in ('ev', 't', 'n', 'kind', 'layer', 'tag', 'toks', 'reqs') if key in row}
        number(row['t'])
        if row['ev'] == 'step':
            item['host_end_to_end_s'] = (number(row['t']) - number(previous_step['t'])
                if previous_step is not None and number(row['n']) == number(previous_step['n']) + 1 else None)
            item['previous_logged_step_n'] = previous_step['n'] if previous_step else None
            eng = row.get('eng', [])
            if not isinstance(eng, list):
                raise ValueError('Engram memory probe counters must be a list')
            timings: list[Json] = []
            for i, values in enumerate(eng):
                if not isinstance(values, list) or len(values) != 4:
                    raise ValueError('Engram memory probe needs served/sum_ms/running_max_ms/errors')
                for value in values:
                    number(value)
                timings.append({'table': i, 'reads_since_previous_record': values[0],
                                'reader_service_sum_ms': values[1],
                                'reader_service_running_max_ms': values[2], 'errors': values[3]})
            item['engram_reader_timing'] = timings
            previous_step = row
        item['gpu_prefill_s'] = None
        out.append(item)
    return out


def collect(results: Path, production: Path) -> dict[str, Json]:
    hashes: dict[str, Json] = {}
    requests: list[Json] = []
    suites: list[Json] = []
    roots = [('window6', results, ('R0', 'R1', 'R1p', 'R2p', 'R5')),
             ('production', production, ('K1-prod',))]
    for label, root, arms in roots:
        files = sorted({p for arm in arms for p in root.glob(f'{arm}-*.jsonl')})
        for path in files:
            rows = read_rows(path)
            if not any(r.get('type') == 'request' for r in rows):
                continue
            name = f'{label}/{path.name}'
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            for row in rows:
                if row.get('type') == 'request':
                    requests.append({'source': name, **bench(row)})
                elif row.get('type') == 'run':
                    suites.append({'source': name, **{k: row[k] for k in
                        ('run_id', 'elapsed_s', 'client_elapsed_s', 'request_count', 'delivered_tokens') if k in row}})
    for name, parser in (('needle-R5-500k.jsonl', needle), ('prefix-R5.jsonl', prefix)):
        path = results / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in read_rows(path):
            parsed = parser(row)
            for out in parsed if isinstance(parsed, list) else [parsed]:
                requests.append({'source': name, **out})
    logs: list[Json] = []
    for path in sorted((results / 'extra').rglob('*')):
        if not path.is_file():
            continue
        name = str(path.relative_to(results))
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = read_rows(path)
        if not rows and any(not any(marker in line for marker in
                ('Model loading took', 'GPU KV cache size:', 'Unknown vLLM environment variable'))
                for line in path.read_text().splitlines() if line.strip()):
            raise ValueError(f'{name}: new serve-log schema; review before claiming no chunk timing')
        logs.append({'source': name, 'lines': len(path.read_text().splitlines()),
                     'memory_events': len(rows),
                     'has_request_chunk_timing': False,
                     'scope': 'memory snapshots/host timestamps/reader counters' if rows else 'startup load/KV/env lines',
                     'timing_records': memory_timing(rows) if rows else []})
    # Suite .log repeats run summaries, not streaming request timing. Hash for provenance.
    for path in sorted(results.glob('*suite.log')):
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'label': 'MEASURED from saved files; missing fields are null, never zero',
            'source_sha256': hashes, 'requests': requests, 'suite_windows': suites, 'serve_logs': logs,
            'attribution': 'No saved per-prefill-chunk CUDA or wall durations; GPU prefill attribution unavailable'}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--production', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.write_text(json.dumps(collect(args.results, args.production), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
