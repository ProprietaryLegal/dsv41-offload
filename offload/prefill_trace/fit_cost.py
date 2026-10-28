"""Fit full cold chunks on BOTH ranks; reject missing/cross-rank/missing-cap evidence."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import cast

import numpy as np

from offload.prefill_maxlen.evidence import Json, number, record


def read(paths: list[Path]) -> list[dict[str, Json]]:
    return [record(cast(Json, json.loads(line))) for path in paths for line in path.read_text().splitlines()]


def match_by_size(traces: list[dict[str, Json]], requests: list[dict[str, Json]]) -> list[dict[str, Json]]:
    """BY_SIZE_ONLY windows (engine req_id hash differs from the SSE id): map each cold request to the one traced
    request of the same cap and prompt size. Refuses unless that request is unique and identical on both ranks."""
    extent: dict[tuple[int, int, str], int] = defaultdict(int)
    for row in traces:
        if row.get('phase') != 'chunk.begin':
            continue
        for member in (record(m) for m in cast(list[Json], row['members'])):
            key = (int(number(row['rank'])), int(number(row['max_model_len'])), str(member['request']))
            extent[key] = max(extent[key], int(number(member['prefix_tokens']) + number(member['new_tokens'])))
    traced: dict[tuple[int, int, int], set[str]] = defaultdict(set)
    for (rank, cap, request), size in extent.items():
        traced[rank, cap, size].add(request)
    out: list[dict[str, Json]] = []
    seen: set[tuple[int, int]] = set()
    for row in requests:
        if row.get('kind') != 'cold':
            out.append(row)
            continue
        cap, size = int(number(row['cap'])), int(number(row['target_ctx']))
        if (cap, size) in seen:
            raise ValueError('by-size match needs one cold request per cap and size')
        seen.add((cap, size))
        found = [traced.get((rank, cap, size), set()) for rank in (0, 1)]
        if len(found[0]) != 1 or found[0] != found[1]:
            raise ValueError(f'by-size match needs exactly one traced request of cap {cap} and size {size} on both ranks')
        out.append({**row, 'trace_request': next(iter(found[0])), 'matched_by': 'size'})
    return out


def spans(traces: list[dict[str, Json]], chunks: dict[tuple[int, str], dict[str, Json]],
          common: dict[int, set[int]]) -> dict[str, Json]:
    """Per rank and span label: per-chunk CUDA ms summed over the chunk's rows (layers), matched-prefix medians at both
    caps and their delta as a share of the chunk.total delta. Spans nest/overlap: never add labels together."""
    total: dict[tuple[int, str, str], float] = defaultdict(float)
    plans: dict[tuple[int, int, str], int] = defaultdict(int)
    for row in traces:
        phase = str(row.get('phase'))
        members = row.get('members')
        if phase.startswith('chunk.') or not isinstance(members, list) or len(members) != 1:
            continue
        key = (int(number(row['rank'])), str(record(members[0])['chunk']))
        if key not in chunks:
            continue
        if phase == 'indexer.plan':
            for flag in ('workspace_exceeded', 'workspace_group_split', 'query_subchunked'):
                plans[key[0], int(number(row['max_model_len'])), flag] += bool(row.get(flag))
            plans[key[0], int(number(row['max_model_len'])), 'rows'] += 1
        elif row.get('cuda_ms') is not None:
            total[key[0], key[1], phase] += number(row['cuda_ms'])
    out: dict[str, Json] = {}
    for rank in (0, 1):
        by: dict[tuple[str, int, int], float] = {}
        for (r, chunk, phase), ms in total.items():
            if r == rank:
                row = chunks[rank, chunk]
                by[phase, int(number(row['max_model_len'])), int(number(record(row['member'])['prefix_tokens']))] = ms
        whole = {(int(number(row['max_model_len'])), int(number(record(row['member'])['prefix_tokens']))): number(row['cuda_ms'])
                 for (r, _), row in chunks.items() if r == rank}
        base = statistics.median(whole[524288, p] - whole[327680, p] for p in common[rank])
        labels: dict[str, Json] = {}
        for phase in sorted({k[0] for k in by}):
            shared = [p for p in sorted(common[rank]) if (phase, 327680, p) in by and (phase, 524288, p) in by]
            if not shared:
                labels[phase] = {'matched_prefixes': 0, 'only_at_cap': cast(Json, sorted({c for f, c, _ in by if f == phase}))}
                continue
            delta = statistics.median(by[phase, 524288, p] - by[phase, 327680, p] for p in shared)
            labels[phase] = {'matched_prefixes': len(shared),
                             'median_low_ms': statistics.median(by[phase, 327680, p] for p in shared),
                             'median_high_ms': statistics.median(by[phase, 524288, p] for p in shared),
                             'median_delta_ms': delta, 'share_of_chunk_total_delta': delta / base if base else None}
        out[f'rank{rank}'] = {'chunk_total_median_delta_ms': base, 'labels': labels,
                              'indexer_plan': {f'{cap}:{flag}': n for (r, cap, flag), n in sorted(plans.items()) if r == rank}}
    return out


def fit(traces: list[dict[str, Json]], requests: list[dict[str, Json]]) -> dict[str, Json]:
    allowed = {str(row['trace_request']) for row in requests if row.get('kind') == 'cold'}
    if not allowed:
        raise ValueError('no cold requests selected from coordinator request artifacts')
    # Full scheduler chunks have the largest new_tokens of the selected requests (the last chunk is the remainder).
    # Measured, not assumed: real GB10 full chunks are 4,092 tokens (PF1 window 1), not the nominal 4,096.
    full = max((int(number(record(m[0])['new_tokens'])) for row in traces
                if row.get('phase') == 'chunk.begin' and isinstance(m := row.get('members'), list) and len(m) == 1
                and str(record(m[0])['request']) in allowed), default=0)
    if full <= 0:
        raise ValueError('no single-request scheduler chunks for the selected cold requests')
    # Per rank, request and scheduler chunk identity. Do not mix indexer subchunks with scheduler chunks.
    data: dict[tuple[int, str], dict[str, Json]] = {}
    began: set[tuple[int, str]] = set()
    identities: dict[str, set[int]] = defaultdict(set)
    for row in traces:
        if row.get('phase') not in ('chunk.begin', 'chunk.total'):
            continue
        members = row.get('members')
        if not isinstance(members, list) or len(members) != 1:
            continue  # B1 attribution only; mixed intervals cannot be divided among members.
        member = record(members[0])
        if str(member['request']) not in allowed or member['new_tokens'] != full:
            continue
        rank = int(number(row['rank']))
        if rank not in (0, 1):
            raise ValueError('expected TP2 global rank 0/1')
        key = (rank, str(member['chunk']))
        if row['phase'] == 'chunk.begin':
            if key in began:
                raise ValueError('duplicate chunk begin')
            began.add(key)
        else:
            if key in data:
                raise ValueError('duplicate chunk.total')
            if row.get('failed') or row.get('cuda_ms') is None:
                raise ValueError('failed/uncompleted CUDA chunk event')
            if number(row['cuda_ms']) <= 0 or number(row['wall_ms']) < 0:
                raise ValueError('invalid chunk duration')
            data[key] = {**row, 'member': member}
            identities[key[1]].add(rank)
    if began != set(data):
        raise ValueError('missing begin or completed total events; drain/collect both ranks')
    if not data or any(ranks != {0, 1} for ranks in identities.values()):
        raise ValueError('both rank timings required for every selected chunk')
    # Collapse repeated requests at each cap/prefix per rank to prevent longer
    # requests from weighting shared prefix buckets as independent experiments.
    groups: dict[tuple[int, int, int], list[dict[str, Json]]] = defaultdict(list)
    for (rank, _), row in data.items():
        cap = int(number(row['max_model_len']))
        prefix = int(number(record(row['member'])['prefix_tokens']))
        if cap not in (327680, 524288):
            raise ValueError('unexpected cap')
        groups[rank, cap, prefix].append(row)
    outputs: list[Json] = []
    commons: dict[int, set[int]] = {}
    for rank in (0, 1):
        bins = [(cap, prefix, vals) for (r, cap, prefix), vals in sorted(groups.items()) if r == rank]
        if {cap for cap, _, _ in bins} != {327680, 524288}:
            raise ValueError('fit needs both caps')
        # Common prefix support identifies a cap effect. Long-only points extend
        # the curve but cannot distinguish a cap effect without this overlap.
        supports = [{prefix for cap, prefix, _ in bins if cap == wanted} for wanted in (327680, 524288)]
        common = commons[rank] = supports[0] & supports[1]
        if len(common) < 3:
            raise ValueError('need >=3 matching full-chunk prefixes at both caps')
        common_bins = [(c, p, vals) for c, p, vals in bins if p in common]
        design = np.asarray([[1.0, p / 100000, float(c == 524288), float(c == 524288) * p / 100000]
                             for c, p, _ in common_bins], dtype=np.float64)
        models: dict[str, Json] = {}
        for metric in ('cuda_ms', 'wall_ms'):
            target = np.asarray([statistics.median(number(v[metric]) for v in vals)
                                 for _, _, vals in common_bins], dtype=np.float64)
            beta, _, rank_n, _ = np.linalg.lstsq(design, target, rcond=None)
            if rank_n != 4:
                raise ValueError('cost fit is rank deficient')
            residual = target - design @ beta
            models[metric] = {'intercept_ms': float(beta[0]), 'prefix_ms_per_100K': float(beta[1]),
                             'high_cap_extra_ms': float(beta[2]), 'high_cap_prefix_extra_ms_per_100K': float(beta[3]),
                             'rms_residual_ms': float(np.sqrt(np.mean(residual**2)))}
        pairs: list[Json] = []
        for prefix in sorted(common):
            low = statistics.median(number(v['cuda_ms']) for v in groups[rank, 327680, prefix])
            high = statistics.median(number(v['cuda_ms']) for v in groups[rank, 524288, prefix])
            pairs.append({'prefix_tokens': prefix, 'low_cuda_ms': low, 'high_cuda_ms': high,
                          'high_minus_low_ms': high - low})
        bins_json: list[Json] = [{'cap': cap, 'prefix_tokens': prefix, 'samples': len(vals),
            'median_cuda_ms': statistics.median(number(v['cuda_ms']) for v in vals),
            'median_wall_ms': statistics.median(number(v['wall_ms']) for v in vals)} for cap, prefix, vals in bins]
        outputs.append({'rank': rank, 'models_common_support': models, 'matched_prefix_pairs': pairs,
                        'all_prefix_bins': bins_json})
    return {'label': 'MEASURED window regression, descriptive; chunks are correlated within request/family',
            'selected_requests': len(allowed), 'full_chunk_rank_observations': len(data), 'ranks': outputs,
            'spans': spans(traces, data, commons),
            'model': f'ms = a + b*(prefix/100K) + c*high_cap + d*high_cap*(prefix/100K); new={full}', 'full_chunk_tokens': full,
            'limits': 'Common-support fit only. Inspect residuals and dispatch/telemetry transitions; no causal proof or IID chunk CI.'}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--traces', nargs='+', type=Path, required=True)
    ap.add_argument('--requests', nargs='+', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--match-by-size', action='store_true',
                    help='only when the window reported correlation BY_SIZE_ONLY: select traced requests by cap and size')
    args = ap.parse_args()
    traces, requests = read(args.traces), read(args.requests)
    if args.match_by_size:
        requests = match_by_size(traces, requests)
    result = {**fit(traces, requests), 'correlation': 'BY_SIZE (unique cap/size, both ranks)' if args.match_by_size else 'HASH'}
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
