"""Locate the 524,288 per-chunk cost inside the layer stack from existing PF1 traces (CPU only, numbers only).

Every GroupCoordinator all-reduce of a full chunk is a two-rank meeting point. On each rank, the CUDA span of one
all-reduce is (transfer + the time this rank waited for the other one). Between two consecutive all-reduces both ranks
run their own local work, so a growth of the wait on rank A at all-reduce i means rank B's local work just before i grew.
The all-reduces of a full chunk are classified by what precedes them: an attention output all-reduce follows that
layer's sparse attention dispatch; an FFN output (MoE) all-reduce follows no traced span. Output: per rank and per
all-reduce sequence index, matched-prefix medians at both caps, grouped by layer half. MEASURED from trace files only.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import cast

from offload.prefill_maxlen.evidence import Json, number, record

LOW, HIGH = 327680, 524288
ATTENTION = ('sparse.dispatch.flashinfer', 'sparse.dispatch.triton')
# PF1-CAUSE layer/MoE spans (offload/prefill_cause/build_overlay.py): reported per label, ignored when classifying
CAUSE = ('attn.', 'mhc.', 'moe.', 'tf.')


def read(paths: list[Path]) -> list[dict[str, Json]]:
    return [record(cast(Json, json.loads(line))) for path in paths for line in path.read_text().splitlines()]


def chunk_rows(traces: list[dict[str, Json]], request_size: int | None = None
               ) -> dict[tuple[int, int, int], list[dict[str, Json]]]:
    """(rank, cap, prefix) -> event-ordered rows of one single-request full chunk of the largest traced request."""
    sizes: dict[tuple[int, int, str], int] = defaultdict(int)
    full = 0
    for row in traces:
        if row.get('phase') != 'chunk.begin':
            continue
        members = cast(list[Json], row['members'])
        if len(members) != 1:
            continue
        m = record(members[0])
        key = (int(number(row['rank'])), int(number(row['max_model_len'])), str(m['request']))
        sizes[key] = max(sizes[key], int(number(m['prefix_tokens']) + number(m['new_tokens'])))
        full = max(full, int(number(m['new_tokens'])))
    chosen: dict[tuple[int, int], str] = {}
    for (rank, cap, request), size in sizes.items():
        best = chosen.get((rank, cap))
        if request_size is not None and size != request_size:
            continue
        if best is None or size > sizes[rank, cap, best]:
            chosen[rank, cap] = request
    if {(r, c) for r in (0, 1) for c in (LOW, HIGH)} - set(chosen):
        raise ValueError('need one selected request per rank and cap')
    out: dict[tuple[int, int, int], list[dict[str, Json]]] = defaultdict(list)
    for row in traces:
        found = row.get('members')
        if not isinstance(found, list) or len(found) != 1:
            continue
        m = record(found[0])
        rank, cap = int(number(row['rank'])), int(number(row['max_model_len']))
        if chosen.get((rank, cap)) != m['request'] or int(number(m['new_tokens'])) != full:
            continue
        at = (rank, cap, int(number(m['prefix_tokens'])))
        if row.get('phase') == 'chunk.begin' and out.get(at):
            raise ValueError(f'repeated chunk {at}: rollback occurrences are not supported here')
        out[at].append(row)
    for rows in out.values():
        rows.sort(key=lambda r: int(number(r['event'])) if 'event' in r else -1)
    return dict(out)


def classify(rows: list[dict[str, Json]]) -> list[tuple[str, int, dict[str, Json]]]:
    """[(kind, rows_reduced, row)] for each all-reduce of one chunk in event order.
    kind: 'embed' (before any attention), 'attn' (after a sparse dispatch), 'ffn' (after an attention all-reduce
    with no traced span in between)."""
    out: list[tuple[str, int, dict[str, Json]]] = []
    seen_attention = False
    since_last: list[str] = []
    for row in rows:
        phase = str(row.get('phase'))
        if phase != 'collective.all_reduce':
            if phase not in ('chunk.begin', 'chunk.end', 'chunk.total') and not phase.startswith(CAUSE):
                since_last.append(phase)
            continue
        if row.get('cuda_ms') is None or row.get('failed'):
            raise ValueError('incomplete all-reduce event; collect after drain')
        reduced = int(str(row.get('input__shape', '0x0')).split('x')[0])
        if any(p in ATTENTION for p in since_last):
            kind, seen_attention = 'attn', True
        elif not seen_attention:
            kind = 'embed'
        elif out and out[-1][0] == 'attn' and not since_last:
            kind = 'ffn'
        else:
            kind = 'other'
        out.append((kind, reduced, row))
        since_last = []
    return out


def localize(traces: list[dict[str, Json]], request_size: int | None = None) -> dict[str, Json]:
    data = chunk_rows(traces, request_size)
    result: dict[str, Json] = {'label': 'MEASURED from PF1 trace files; medians over matched full-chunk prefixes',
                               'caps': [LOW, HIGH]}
    for rank in (0, 1):
        common = sorted({p for r, c, p in data if r == rank and c == LOW} & {p for r, c, p in data if r == rank and c == HIGH})
        if len(common) < 3:
            raise ValueError('need >=3 matched full-chunk prefixes per rank')
        seqs = {(cap, p): classify(data[rank, cap, p]) for cap in (LOW, HIGH) for p in common}
        shape = [(k, n) for k, n, _ in seqs[LOW, common[0]]]
        for key, seq in seqs.items():
            if [(k, n) for k, n, _ in seq] != shape:
                raise ValueError(f'all-reduce sequence differs at {key}; not a matched layer stack')
        index: list[Json] = []
        groups: dict[str, list[float]] = defaultdict(list)
        layer = -1
        for i, (kind, reduced) in enumerate(shape):
            if kind == 'attn':
                layer += 1
            low = [number(seqs[LOW, p][i][2]['cuda_ms']) for p in common]
            high = [number(seqs[HIGH, p][i][2]['cuda_ms']) for p in common]
            delta = [h - lo for h, lo in zip(high, low)]
            q = statistics.quantiles(delta, n=4) if len(delta) >= 4 else [min(delta), statistics.median(delta), max(delta)]
            index.append({'index': i, 'kind': kind, 'rows': reduced, 'layer_seq': layer,
                          'median_low_ms': statistics.median(low), 'median_high_ms': statistics.median(high),
                          'median_delta_ms': statistics.median(delta), 'q1_delta_ms': q[0], 'q3_delta_ms': q[-1]})
            groups[f'{kind}:{reduced}'].append(statistics.median(delta))
        labels = cause_labels({(cap, p): data[rank, cap, p] for cap in (LOW, HIGH) for p in common}, seqs, common)
        totals = [number(next(r for r in data[rank, cap, p] if r.get('phase') == 'chunk.total')['cuda_ms'])
                  for cap in (LOW, HIGH) for p in common]
        n = len(common)
        result[f'rank{rank}'] = {
            'matched_prefixes': n,
            'chunk_total_median_delta_ms': statistics.median(totals[n + i] - totals[i] for i in range(n)),
            'groups': {k: {'all_reduces': len(v), 'sum_of_median_delta_ms': sum(v)} for k, v in sorted(groups.items())},
            'all_reduces': index, 'cause_labels': labels}
    return result


def cause_labels(chunks: dict[tuple[int, int], list[dict[str, Json]]],
                 seqs: dict[tuple[int, int], list[tuple[str, int, dict[str, Json]]]], common: list[int]) -> dict[str, Json]:
    """Per PF1-CAUSE label and row count: CUDA ms summed over the chunk's occurrences, matched-prefix medians.
    'moe.local' = each layer's moe.ffn span minus the FFN all-reduce nested in it: this rank's own MoE-half time."""
    sums: dict[tuple[int, int], dict[str, float]] = {}
    for key, rows in chunks.items():
        acc: dict[str, float] = defaultdict(float)
        moe = [r for r in rows if r.get('phase') == 'moe.ffn']
        for r in rows:
            phase = str(r.get('phase'))
            if phase.startswith(CAUSE):
                if r.get('cuda_ms') is None or r.get('failed'):
                    raise ValueError(f'incomplete {phase} event; collect after drain')
                acc[f"{phase}:{int(number(r.get('rows', 0)))}"] += number(r['cuda_ms'])
        ffn = [(n, row) for kind, n, row in seqs[key] if kind == 'ffn']
        if moe:
            if len(moe) != len(ffn):
                raise ValueError(f'{len(moe)} moe.ffn spans but {len(ffn)} FFN all-reduces at {key}')
            for span, (n, ar) in zip(moe, ffn):
                if int(number(span['rows'])) != n:
                    raise ValueError(f'moe.ffn rows {span["rows"]} do not match its all-reduce rows {n} at {key}')
                acc[f'moe.local:{n}'] += number(span['cuda_ms']) - number(ar['cuda_ms'])
        sums[key] = acc
    out: dict[str, Json] = {}
    for name in sorted({n for acc in sums.values() for n in acc}):
        if not all(name in sums[cap, p] for cap in (LOW, HIGH) for p in common):
            raise ValueError(f'{name} missing in some matched chunks')
        low = [sums[LOW, p][name] for p in common]
        high = [sums[HIGH, p][name] for p in common]
        out[name] = {'median_low_ms': statistics.median(low), 'median_high_ms': statistics.median(high),
                     'median_delta_ms': statistics.median(h - lo for h, lo in zip(high, low))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--traces', nargs='+', type=Path, required=True, help='both ranks, both caps')
    ap.add_argument('--request-size', type=int, help='select the traced request of this prefix+new size (default largest)')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    out = localize(read(args.traces), args.request_size)
    with args.output.open('x') as stream:
        stream.write(json.dumps(out, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
