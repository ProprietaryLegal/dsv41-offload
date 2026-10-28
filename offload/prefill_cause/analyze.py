"""CPU-only cause-window reading; measured inputs, per-layer deltas and pre-registered T0 decision. No network."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import cast

from offload.launch.windows.pf1_window import cells, paired, trace_check, CAUSE_ARMS, CAUSE_T0
from offload.prefill_cause.localize import LOW, HIGH, chunk_rows, classify, localize, read
from offload.prefill_maxlen.evidence import Json, number, record
from offload.prefill_trace.fit_cost import match_by_size


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def ms(row: dict[str, Json]) -> float:
    value = number(row['cuda_ms'])
    require(not row.get('failed') and math.isfinite(value) and value >= 0, 'invalid/incomplete cause duration')
    return value


def measured_summary(directory: Path) -> dict[str, Json]:
    summary = record(cast(Json, json.loads((directory / 'summary.json').read_text())))
    restore = record(summary['restore'])
    require(all(restore.get(s) == 'PASS' for s in ('start', 'check', 'close')), 'window restore not verified')
    pins = record(summary['artifact_sha256'])
    require(bool(pins), 'missing window artifact pins')
    for name, sha in pins.items():
        require(Path(name).name == name, 'artifact pin must name a direct child')
        require(hashlib.sha256((directory / name).read_bytes()).hexdigest() == sha, f'evidence changed: {name}')
    return summary


def layers(data: dict[tuple[int, int, int], list[dict[str, Json]]]) -> dict[str, Json]:
    """Subtract each FFN's nested all-reduce; preserve layer identity and row count, including the replay tail."""
    out: dict[str, Json] = {}
    for rank in (0, 1):
        common = sorted({p for r, c, p in data if r == rank and c == LOW} &
                        {p for r, c, p in data if r == rank and c == HIGH})
        require(len(common) >= 3, 'need >=3 matched prefixes for per-layer deltas')
        values: dict[tuple[int, int], dict[tuple[int, int, str], float]] = {}
        for cap in (LOW, HIGH):
            for prefix in common:
                rows = data[rank, cap, prefix]
                require(sum(r.get('phase') == 'chunk.total' for r in rows) == 1, 'need exactly one chunk total')
                moe = [r for r in rows if r.get('phase') == 'moe.ffn']
                ffn = [(n, r) for kind, n, r in classify(rows) if kind == 'ffn']
                require(len(moe) == len(ffn) == 40, 'need 40 cause MoE spans and nested FFN all-reduces per chunk')
                acc: dict[tuple[int, int, str], float] = {}
                for row in rows:
                    phase = str(row.get('phase'))
                    if phase not in ('moe.ffn', 'mhc.ffn_pre', 'attn.total'):
                        continue
                    key = (int(number(row['layer'])), int(number(row['rows'])), phase)
                    require(key not in acc, f'duplicate layer span: {key}')
                    acc[key] = ms(row)
                require([int(number(row['layer'])) for row in moe] == list(range(40)), 'layer identity/order incomplete')
                for span, (n, ar) in zip(moe, ffn):
                    layer = int(number(span['layer']))
                    require(int(number(span['rows'])) == n, 'FFN rows do not match MoE span')
                    acc[layer, n, 'moe.local'] = ms(span) - ms(ar)
                values[cap, prefix] = acc
        keys = set(values[LOW, common[0]])
        require(all(set(v) == keys for v in values.values()), 'per-layer cause span coverage differs')
        require(len(keys) == 160, 'need attention, mHC, MoE and local MoE for every layer')
        result: list[Json] = []
        for layer, rows_n, phase in sorted(keys):
            key = (layer, rows_n, phase)
            low = [values[LOW, p][key] for p in common]
            high = [values[HIGH, p][key] for p in common]
            result.append({'layer': layer, 'rows': rows_n, 'phase': phase, 'matched_prefixes': len(common),
                           'median_low_ms': statistics.median(low), 'median_high_ms': statistics.median(high),
                           'median_delta_ms': statistics.median(h - lo for h, lo in zip(high, low))})
        out[f'rank{rank}'] = result
    return out


def t0_reading(ttft: float, off327: float, off524: float, within_chunk_range: bool) -> str:
    require(all(math.isfinite(x) and x > 0 for x in (ttft, off327, off524)), 'invalid TTFT')
    near_low = .95 * off327 <= ttft <= 1.05 * off327
    near_high = .95 * off524 <= ttft <= 1.05 * off524
    if near_low and within_chunk_range and not near_high:
        return 'H1 CONFIRMED'
    if near_high and not near_low:
        return 'H1 RULED OUT'
    return 'AMBIGUOUS'


def analyze(directory: Path, off_window: Path) -> dict[str, Json]:
    summary, prior = measured_summary(directory), measured_summary(off_window)
    require(summary.get('phase') == 'cause' and summary.get('status') == 'COMPLETE' and 'stop' not in summary,
            'need a complete cause window')
    require(record(prior['gate']).get('status') == 'REPRODUCED', 'OFF window did not reproduce the gap')
    arm_records = record(summary['arms'])
    require(set(arm_records) in ({'cause-327', 'cause-524'}, {'cause-327', 'cause-524', 'cause-524-t0'}), 'unexpected cause arms')
    traces: dict[str, list[dict[str, Json]]] = {}
    cold: dict[str, list[dict[str, Json]]] = {}
    hashes: dict[str, Json] = {}
    for arm in (*CAUSE_ARMS, CAUSE_T0):
        if arm.name not in arm_records:
            continue
        request_file = directory / f'{arm.name}-requests.jsonl'
        require(request_file.name in record(summary['artifact_sha256']), 'request artifact not pinned')
        requests = read([request_file])
        run = str(requests[0]['trace_run'])
        require(Path(run).name == run and run not in ('.', '..'), 'unsafe trace run')
        cold[arm.name] = cells(requests, arm, run, (150000,), 1)
        files = {rank: sorted((directory / 'traces' / run / host).glob(f'prefill-r{rank}-p*.jsonl'))
                 for rank, host in enumerate(('sparkhead', 'sparkworker-wired'))}
        trace_check(files, requests)
        paths = [p for paths in files.values() for p in paths]
        rows = read(paths)
        require(all(row.get('run') == run and row.get('max_model_len') == arm.cap for row in rows), 'trace run/cap drift')
        matched = match_by_size(rows, requests)
        selected = str(next(r['trace_request'] for r in matched if r.get('kind') == 'cold'))
        traces[arm.name] = [row for row in rows if isinstance(m := row.get('members'), list) and len(m) == 1
                            and record(m[0])['request'] == selected]
        for rank in (0, 1):
            begins = [row for row in traces[arm.name] if row['rank'] == rank and row.get('phase') == 'chunk.begin']
            segments = sorted((int(number(record(cast(list[Json], row['members'])[0])['prefix_tokens'])),
                               int(number(record(cast(list[Json], row['members'])[0])['new_tokens']))) for row in begins)
            end = 0
            for prefix, count in segments:
                require(prefix == end and count > 0, 'missing/duplicate cold chunk prefix')
                end += count
            require(end == 150000, 'cold trace does not cover the complete 150K request')
        for path in (request_file, *paths):
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    paired(cold)
    combined = traces['cause-327'] + traces['cause-524']
    data = chunk_rows(combined, 150000)
    out: dict[str, Json] = {'classification': 'MEASURED from pinned window artifacts and trace files',
                            'off_window': str(off_window), 'cause_window': str(directory),
                            'localization': localize(combined, 150000), 'per_layer': layers(data)}
    # OFF arms were uninstrumented: there are no OFF CUDA chunk totals to infer or fabricate.
    out['off_chunk_totals'] = 'UNAVAILABLE: window-1 OFF arms were uninstrumented; compare T0 chunks to cause-327'
    off_arms = record(prior['arms'])
    off_ttft: dict[str, float] = {}
    for name, cap in (('off-327', LOW), ('off-524', HIGH)):
        rows = [record(r) for r in cast(list[Json], record(off_arms[name])['cold']) if record(r)['target_ctx'] == 150000]
        require(len(rows) == 2 and all(r.get('cap') == cap and r.get('cached_tokens') == 0 for r in rows), 'OFF cells missing/cached')
        require(all(r['prompt_sha256'] == cold['cause-327'][0]['prompt_sha256'] for r in rows), 'OFF/cause prompt hashes differ')
        off_ttft[name] = statistics.median(number(r['ttft_s']) for r in rows)
    out['off_150K_median_ttft_s'] = cast(Json, off_ttft)
    out['cause_150K_ttft_s'] = {name: rows[0]['ttft_s'] for name, rows in cold.items()}
    if 'cause-524-t0' in traces:
        # Use T0 as the high cap in a separate pair, never pool two high-cap arms in chunk_rows.
        tdata = chunk_rows(traces['cause-327'] + traces['cause-524-t0'], 150000)
        comparisons: dict[str, Json] = {}
        inside = True
        for rank in (0, 1):
            prefixes = sorted({p for r, c, p in data if r == rank and c == LOW})
            require(prefixes == sorted(p for r, c, p in tdata if r == rank and c == HIGH), 'T0 prefix coverage differs')
            low = [ms(next(r for r in data[rank, LOW, p] if r.get('phase') == 'chunk.total')) for p in prefixes]
            t0 = [ms(next(r for r in tdata[rank, HIGH, p] if r.get('phase') == 'chunk.total')) for p in prefixes]
            require(len(low) >= 3, 'need >=3 full-chunk totals per rank')
            high = [ms(next(r for r in data[rank, HIGH, p] if r.get('phase') == 'chunk.total')) for p in prefixes]
            med = statistics.median(t0)
            within = min(low) <= med <= max(low)
            inside = inside and within
            comparisons[f'rank{rank}'] = {'matched_prefixes': len(prefixes), 'cause327_range_ms': [min(low), max(low)],
                'cause327_median_ms': statistics.median(low), 'cause524_median_ms': statistics.median(high),
                't0_median_ms': med, 't0_minus_cause524_median_ms': med - statistics.median(high),
                't0_range_ms': [min(t0), max(t0)], 't0_median_within_cause327_range': within,
                'median_paired_delta_ms': statistics.median(t - lo for t, lo in zip(t0, low))}
        ttft = number(cold['cause-524-t0'][0]['ttft_s'])
        out['t0'] = {'reading': t0_reading(ttft, off_ttft['off-327'], off_ttft['off-524'], inside),
                     'ttft_s': ttft, 'relative_to_off327': ttft / off_ttft['off-327'] - 1,
                     'relative_to_off524': ttft / off_ttft['off-524'] - 1, 'per_chunk': comparisons,
                     'localization_vs_cause327': localize(traces['cause-327'] + traces['cause-524-t0'], 150000),
                     'per_layer_vs_cause327': layers(tdata)}
    else:
        out['t0'] = {'reading': 'NOT RUN'}
    for path in (directory / 'summary.json', off_window / 'summary.json'):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    out['input_sha256'] = hashes
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cause-window', type=Path, required=True)
    ap.add_argument('--off-window', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    result = analyze(args.cause_window, args.off_window)
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
