"""SP1-WINDOW trace budget: per-step decode and per-chunk prefill GPU budgets from both TP ranks' torch traces.

CPU only; reads files the runner already collected and verified (offload/launch/windows/sp1_window.py), never a service.

  python3 -m offload.gb10_budget.trace_budget --window /opt/dsv41/campaigns/DSV41-OFFLOAD/SP1-W-<tag>

Method (offload/gb10_budget/PROFILE-WINDOW-DESIGN.md):
  * iterations are vLLM's `execute_*` record_function annotations (gpu_worker.annotate_profile, detailed form);
  * a GPU activity belongs to iteration i when its launch call (same correlation id; a CUDA graph's kernels carry the
    cudaGraphLaunch's id) was issued on the annotation's thread inside the annotation;
  * the GPU step period of iteration i runs from its first activity to the first activity of iteration i+1, so the
    budget includes work launched outside execute_model (sample_tokens, the drafter hand-off) and every idle gap;
    the last iteration of a capture has no successor and is reported as a tail (first to last own activity);
  * busy time is split equally among the buckets active at each instant (exclusive budget, sums to busy time);
    inclusive per-bucket union time is reported beside it;
  * idle time is classified by the activity that ends it (buckets.json wait_rules), else device_wait / host_gap /
    unrecorded_wait;
  * graph visibility: cudaGraphLaunch calls with and without correlated kernels. 'invisible' means the decode budget
    cannot be read from this trace (contingency HOLD E, eager).
Kernel-name buckets and the hypothesis rules are pre-registered in buckets.json; hypotheses map to experiments.json.
Output holds numbers, kernel names and hashes only.
"""
from __future__ import annotations

import argparse
import bisect
import fnmatch
import gzip
import hashlib
import json
import re
import statistics
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, cast

from offload.prefill_maxlen.evidence import Json, number, record

HERE = Path(__file__).resolve().parent
BUCKETS = HERE / 'buckets.json'
EXPERIMENTS = HERE / 'experiments.json'
MAX_TRACE_BYTES = 128 * 1024**2          # sp1_window.FILE_CAP: the runner refuses larger files
MAX_TEXT_BYTES = 4 * 1024**3             # decompressed bound per trace
CHUNK = 1 << 20
CHUNK_TOKENS = 4092                      # production prefill chunk (PROFILE-WINDOW-RECONCILE.md)
GPU_KINDS = {'kernel': 'kernel', 'gpu_memcpy': 'memcpy', 'gpu_memset': 'memset'}
CALL_KINDS = {'cuda_runtime', 'cuda_driver'}
ANNOTATION = re.compile(
    r'^execute_(?:(?P<total>\d+)_)?context_(?P<cr>\d+)\((?:sq)?(?P<ct>\d+)(?:sk(?P<csk>\d+)sqsq\d+sqsk\d+)?\)'
    r'_generation_(?P<gr>\d+)\((?:sq)?(?P<gt>\d+)(?:sk(?P<gsk>\d+)sqsq\d+sqsk\d+)?\)$')
IDLE = ('host_gap', 'device_wait', 'unrecorded_wait')
OUTSIDE_PAD_S = 1.0                      # SSE gaps this close to a capture are neither inside nor outside


@dataclass(frozen=True)
class Activity:
    name: str
    kind: str
    start: float          # microseconds, kineto's common CPU/GPU timebase
    end: float
    stream: int
    corr: int | None


@dataclass(frozen=True)
class Call:
    name: str
    start: float
    end: float
    tid: str
    corr: int | None


@dataclass(frozen=True)
class Annotation:
    name: str
    start: float
    end: float
    tid: str


@dataclass(frozen=True)
class Trace:
    activities: list[Activity]
    calls: list[Call]
    annotations: list[Annotation]
    events: int


@dataclass(frozen=True)
class WaitRule:
    bucket: str
    next: re.Pattern[str]
    launch: re.Pattern[str] | None
    unrecorded: bool


@dataclass(frozen=True)
class Rules:
    kernels: tuple[tuple[str, re.Pattern[str]], ...]
    waits: tuple[WaitRule, ...]
    graph_launch: re.Pattern[str]
    report: tuple[str, ...]
    limits: dict[str, float]
    hypotheses: tuple[dict[str, Json], ...]
    sha256: str


def rules(path: Path = BUCKETS) -> Rules:
    raw = path.read_bytes()
    data = record(cast(Json, json.loads(raw)))
    kernels = tuple((str(b), re.compile(str(p))) for b, p in cast(list[list[str]], data['kernel_rules']))
    waits = tuple(WaitRule(str(w['bucket']), re.compile(str(w['next'])),
                           re.compile(str(w['launch'])) if w.get('launch') is not None else None, w.get('unrecorded') is True)
                  for w in map(record, cast(list[Json], data['wait_rules'])))
    report = tuple(str(b) for b in cast(list[Json], data['report_buckets']))
    named = {b for b, _ in kernels} | {w.bucket for w in waits} | set(IDLE) | {'unmatched'}
    if set(report) != named:
        raise ValueError(f'report_buckets {sorted(set(report) ^ named)} differ from the rules')
    limits = {k: number(v) for k, v in record(data['limits']).items()}
    return Rules(kernels, waits, re.compile(str(data['graph_launch'])), report, limits,
                 tuple(map(record, cast(list[Json], data['hypotheses']))), hashlib.sha256(raw).hexdigest())


_BUCKET_CACHE: dict[tuple[str, str], str] = {}


def bucket(name: str, r: Rules) -> str:
    key = (r.sha256, name)
    if key not in _BUCKET_CACHE:
        _BUCKET_CACHE[key] = next((b for b, pattern in r.kernels if pattern.search(name)), 'unmatched')
    return _BUCKET_CACHE[key]


# ------------------------------------------------------------------ reading


def events(handle: IO[str], limit: int = MAX_TEXT_BYTES) -> Iterator[dict[str, Json]]:
    """The objects of the top-level traceEvents array, one at a time (kineto writes one event per few lines)."""
    decoder = json.JSONDecoder()
    buf = ''
    total = 0

    def more() -> bool:
        nonlocal buf, total
        piece = handle.read(CHUNK)
        if not piece:
            return False
        total += len(piece)
        if total > limit:
            raise ValueError('trace text exceeds the decompressed bound')
        buf += piece
        return True

    while (match := re.search(r'"traceEvents"\s*:\s*\[', buf)) is None:
        if not more():
            raise ValueError('no traceEvents array')
    pos = match.end()
    while True:
        while True:
            while pos < len(buf) and buf[pos] in ' \t\r\n,':
                pos += 1
            if pos < len(buf) or not more():
                break
        if pos >= len(buf):
            raise ValueError('trace ends inside traceEvents')
        if buf[pos] == ']':
            return
        try:
            obj, end = decoder.raw_decode(buf, pos)
        except json.JSONDecodeError:
            if more():
                continue
            raise ValueError('truncated or malformed trace event') from None
        if not isinstance(obj, dict):
            raise ValueError('trace event is not an object')
        yield cast(dict[str, Json], obj)
        pos = end
        if pos > CHUNK:
            buf, pos = buf[pos:], 0


def _correlation(args: dict[str, Json]) -> int | None:
    value = args.get('correlation')
    return None if value is None else int(number(value))


def parse(handle: IO[str]) -> Trace:
    activities: list[Activity] = []
    calls: list[Call] = []
    annotations: list[Annotation] = []
    count = 0
    for event in events(handle):
        count += 1
        cat = event.get('cat')
        if event.get('ph') != 'X' or not isinstance(cat, str):
            continue
        if cat not in GPU_KINDS and cat not in CALL_KINDS and cat != 'user_annotation':
            continue
        name = str(event.get('name', ''))
        start = number(event['ts'])
        end = start + number(event.get('dur', 0))
        args = record(event.get('args') or {})
        tid = str(event.get('tid'))
        if cat in GPU_KINDS:
            stream = args.get('stream')
            activities.append(Activity(name, GPU_KINDS[cat], start, end, -1 if stream is None else int(number(stream)),
                                       _correlation(args)))
        elif cat in CALL_KINDS:
            calls.append(Call(name, start, end, tid, _correlation(args)))
        elif name.startswith('execute_'):
            annotations.append(Annotation(name, start, end, tid))
    activities.sort(key=lambda a: (a.start, a.end))
    calls.sort(key=lambda c: c.start)
    annotations.sort(key=lambda a: a.start)
    return Trace(activities, calls, annotations, count)


def load(path: Path) -> Trace:
    if path.stat().st_size > MAX_TRACE_BYTES:
        raise ValueError(f'{path.name}: larger than the per-file cap')
    with (gzip.open(path, 'rt', encoding='utf-8') if path.name.endswith('.gz') else path.open(encoding='utf-8')) as handle:
        trace = parse(handle)
        while handle.read(CHUNK):       # read to the end: gzip checks its CRC and length there
            pass
    return trace


# ------------------------------------------------------------------ one rank


def details(name: str) -> dict[str, Json]:
    match = ANNOTATION.match(name)
    if match is None:
        raise ValueError(f'unrecognised iteration annotation {name[:80]!r}')
    g = {k: (int(v) if v is not None else None) for k, v in match.groupdict().items()}
    phase = ('mixed' if g['cr'] and g['gr'] else 'prefill' if g['cr'] else 'decode' if g['gr'] else 'empty')
    return {'phase': phase, 'context_requests': g['cr'], 'context_tokens': g['ct'], 'context_seq_sum': g['csk'],
            'generation_requests': g['gr'], 'generation_tokens': g['gt'], 'generation_seq_sum': g['gsk']}


def sweep(spans: Sequence[tuple[float, float, str]], lo: float, hi: float
          ) -> tuple[dict[str, float], dict[str, float], list[tuple[float, float]]]:
    """Exclusive (equal split among active buckets) and inclusive time per bucket over [lo, hi], and the idle gaps."""
    marks: list[tuple[float, int, str]] = []
    for start, end, b in spans:
        s, e = max(start, lo), min(end, hi)
        if e > s:
            marks += [(s, 1, b), (e, -1, b)]
    marks.sort(key=lambda m: (m[0], m[1]))
    exclusive: dict[str, float] = {}
    inclusive: dict[str, float] = {}
    gaps: list[tuple[float, float]] = []
    active: dict[str, int] = {}
    t = lo
    for when, step, b in [*marks, (hi, 0, '')]:
        if when > t:
            live = [k for k, n in active.items() if n > 0]
            if live:
                for k in live:
                    exclusive[k] = exclusive.get(k, 0.0) + (when - t) / len(live)
                    inclusive[k] = inclusive.get(k, 0.0) + (when - t)
            elif gaps and gaps[-1][1] == t:
                gaps[-1] = (gaps[-1][0], when)
            else:
                gaps.append((t, when))
            t = when
        if step:
            active[b] = active.get(b, 0) + step
    return exclusive, inclusive, gaps


def classify_gap(start: float, nxt: Activity, launch: Call | None, r: Rules) -> str:
    for w in r.waits:
        if not w.next.search(nxt.name):
            continue
        if launch is None and w.unrecorded:
            return w.bucket
        if launch is not None and w.launch is not None and w.launch.search(launch.name) and launch.start <= start:
            return w.bucket
    if launch is None:
        return 'unrecorded_wait'
    return 'device_wait' if launch.start <= start else 'host_gap'


def steps(trace: Trace, r: Rules) -> list[dict[str, Json]]:
    by_corr: dict[int, list[Activity]] = {}
    for a in trace.activities:
        if a.corr is not None:
            by_corr.setdefault(a.corr, []).append(a)
    launch_of = {c.corr: c for c in trace.calls if c.corr is not None and c.corr in by_corr}
    call_starts = [c.start for c in trace.calls]
    starts = [a.start for a in trace.activities]
    owned: list[set[int]] = []
    graphs: list[tuple[int, int]] = []
    for ann in trace.annotations:
        ids: set[int] = set()
        launched = visible = 0
        for c in trace.calls[bisect.bisect_left(call_starts, ann.start):bisect.bisect_right(call_starts, ann.end)]:
            if c.tid != ann.tid or c.corr is None:
                continue
            ids.add(c.corr)
            if r.graph_launch.search(c.name):
                launched += 1
                visible += c.corr in by_corr
        owned.append(ids)
        graphs.append((launched, visible))
    annotated = set().union(*owned) if owned else set()
    firsts = [min((a.start for i in ids for a in by_corr.get(i, ())), default=None) for ids in owned]
    rows: list[dict[str, Json]] = []
    for i, ann in enumerate(trace.annotations):
        mine = [a for k in owned[i] for a in by_corr.get(k, ())]
        row: dict[str, Json] = {'index': i, 'annotation': ann.name, **details(ann.name),
                                'cpu_ms': (ann.end - ann.start) / 1e3,
                                'cpu_period_ms': (trace.annotations[i + 1].start - ann.start) / 1e3
                                if i + 1 < len(trace.annotations) else None,
                                'launches': len(owned[i]), 'activities': len(mine),
                                'graph_launches': graphs[i][0], 'graph_launches_with_activity': graphs[i][1]}
        lo = firsts[i]
        nxt_first = firsts[i + 1] if i + 1 < len(firsts) else None
        if lo is None:
            rows.append({**row, 'complete': False, 'gpu_period_ms': None})
            continue
        complete = nxt_first is not None and nxt_first > lo
        hi = nxt_first if complete and nxt_first is not None else max(a.end for a in mine)
        window = trace.activities[:bisect.bisect_left(starts, hi)]
        inside = [a for a in window if a.end > lo]
        exclusive, inclusive, gaps = sweep([(a.start, a.end, bucket(a.name, r)) for a in inside], lo, hi)
        idle: dict[str, float] = {}
        for g0, g1 in gaps:
            j = bisect.bisect_left(starts, g1)
            if j >= len(trace.activities):
                continue        # cannot happen inside [lo, hi]: hi is an activity start or end
            nxt = trace.activities[j]
            kind = classify_gap(g0, nxt, launch_of.get(nxt.corr) if nxt.corr is not None else None, r)
            idle[kind] = idle.get(kind, 0.0) + (g1 - g0)
        budget = dict(exclusive)
        for k, v in idle.items():
            budget[k] = budget.get(k, 0.0) + v
        unannotated = sum(min(a.end, hi) - max(a.start, lo) for a in inside if a.corr not in annotated)
        rows.append({**row, 'complete': complete, 'gpu_period_ms': (hi - lo) / 1e3,
                     'busy_ms': sum(exclusive.values()) / 1e3,
                     'budget_ms': {k: v / 1e3 for k, v in sorted(budget.items())},
                     'inclusive_ms': {k: v / 1e3 for k, v in sorted(inclusive.items())},
                     'idle_ms': {k: v / 1e3 for k, v in sorted(idle.items())},
                     'gaps_over_100us': sum(1 for g0, g1 in gaps if g1 - g0 >= 100),
                     'unannotated_busy_ms': unannotated / 1e3})
    return rows


def graph_flag(rows: Sequence[dict[str, Json]]) -> dict[str, Json]:
    launched = sum(int(number(s['graph_launches'])) for s in rows)
    seen = sum(int(number(s['graph_launches_with_activity'])) for s in rows)
    flag = 'no_graphs' if not launched else 'visible' if seen == launched else 'invisible' if not seen else 'partial'
    return {'graph_visibility': flag, 'graph_launches': launched, 'graph_launches_with_activity': seen}


def aggregate(rows: Sequence[dict[str, Json]], phase: str, r: Rules) -> dict[str, Json]:
    chosen = [s for s in rows if s['phase'] == phase and s['complete']]
    if not chosen:
        return {'phase': phase, 'complete_steps': 0}
    periods = [number(s['gpu_period_ms']) for s in chosen]
    total = sum(periods)
    sums = {b: sum(number(record(s['budget_ms']).get(b, 0.0)) for s in chosen) for b in r.report}
    busy = sum(number(s['busy_ms']) for s in chosen)
    tokens = [number(s['generation_tokens' if phase == 'decode' else 'context_tokens']) for s in chosen]
    idle = sum(number(v) for s in chosen for v in record(s['idle_ms']).values())
    return {'phase': phase, 'complete_steps': len(chosen),
            'period_ms_median': statistics.median(periods), 'period_ms_max': max(periods),
            'tokens_per_step_median': statistics.median(tokens),
            'ms': {b: sums[b] / len(chosen) for b in r.report},
            'share': {b: sums[b] / total for b in r.report},
            'idle_share': idle / total,
            'coverage': (busy - sums['unmatched']) / busy if busy else None}


def unmatched(trace: Trace, r: Rules, top: int = 12) -> list[dict[str, Json]]:
    totals: dict[str, float] = {}
    for a in trace.activities:
        if bucket(a.name, r) == 'unmatched':
            totals[a.name] = totals.get(a.name, 0.0) + (a.end - a.start)
    ranked = sorted(totals.items(), key=lambda kv: -kv[1])[:top]
    return [{'name': name[:160], 'ms': ms / 1e3} for name, ms in ranked]


# ------------------------------------------------------------------ one capture, both ranks


def phase_of(kind: str) -> str:
    return 'decode' if kind == 'decode' else 'prefill'


def capture_budget(traces: dict[int, Trace], kind: str, r: Rules) -> dict[str, Json]:
    phase = phase_of(kind)
    ranks: dict[str, Json] = {}
    sequences: dict[int, list[str]] = {}
    aggregates: dict[int, dict[str, Json]] = {}
    for rank, trace in sorted(traces.items()):
        rows = steps(trace, r)
        sequences[rank] = [a.name for a in trace.annotations]
        flag = graph_flag(rows)
        # Graphed decode whose kernels the profiler did not record has no split to report (S42 FIX 1): eager prep
        # kernels inside the step would otherwise make a step look complete and invent shares.
        if phase == 'decode' and flag['graph_visibility'] not in ('visible', 'no_graphs'):
            aggregates[rank] = {'phase': phase, 'complete_steps': 0, 'unbudgeted': f"graph kernels {flag['graph_visibility']}"}
        else:
            aggregates[rank] = aggregate(rows, phase, r)
        chunks = [{k: s.get(k) for k in ('index', 'complete', 'context_tokens', 'context_seq_sum', 'gpu_period_ms',
                                          'cpu_ms', 'busy_ms', 'budget_ms', 'idle_ms')}
                  for s in rows if s['phase'] == 'prefill']
        ranks[str(rank)] = {'events': trace.events, 'iterations': len(rows), **flag,
                            'phases': {p: sum(1 for s in rows if s['phase'] == p) for p in ('decode', 'prefill', 'mixed', 'empty')},
                            'aggregate': cast(Json, aggregates[rank]), 'chunks': cast(Json, chunks) if phase == 'prefill' else [],
                            'unmatched_top': cast(Json, unmatched(trace, r)), 'steps': cast(Json, rows)}
    names = list(sequences.values())
    common = min((len(s) for s in names), default=0)
    aligned = len(names) == 2 and all(s[:common] == names[0][:common] for s in names)
    result: dict[str, Json] = {'kind': kind, 'phase': phase, 'ranks_present': cast(Json, sorted(traces)), 'aligned': aligned,
                               'iterations_common': common, 'ranks': ranks}
    measured = [a for a in aggregates.values() if a.get('complete_steps')]
    if len(measured) == 2:
        result['share'] = {b: statistics.fmean(number(record(a['share'])[b]) for a in measured) for b in r.report}
        result['ms'] = {b: statistics.fmean(number(record(a['ms'])[b]) for a in measured) for b in r.report}
        result['idle_share'] = statistics.fmean(number(a['idle_share']) for a in measured)
        result['period_ms_median'] = statistics.fmean(number(a['period_ms_median']) for a in measured)
        result['collective_imbalance_ms'] = abs(number(record(measured[0]['ms'])['collectives']) -
                                                number(record(measured[1]['ms'])['collectives']))
        result['coverage_min'] = min(number(a['coverage']) for a in measured if a['coverage'] is not None)
    return result


def metrics(captures: dict[str, dict[str, Json]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for label, c in captures.items():
        if 'share' not in c:
            continue
        for b, v in record(c['share']).items():
            out[f'{label}.share.{b}'] = number(v)
        for b, v in record(c['ms']).items():
            out[f'{label}.ms.{b}'] = number(v)
        for key in ('idle_share', 'collective_imbalance_ms', 'period_ms_median'):
            out[f'{label}.{key}'] = number(c[key])
    return out


# ------------------------------------------------------------------ profiler overhead from the cell artifacts


def _gaps(stream: dict[str, Json]) -> list[tuple[float, float]]:
    base = number(stream['submitted_rel_s'])
    times = [base + number(t) / 1e3 for t in cast(list[Json], stream.get('sse_times_ms') or [])]
    return list(zip(times, times[1:]))


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _ratio(a: float | None, b: float | None) -> float | None:
    return a / b if a is not None and b else None


def overhead(cells: dict[str, dict[str, Json]], chunks: dict[str, int]) -> dict[str, Json]:
    """(a) SSE gaps inside vs outside each decode capture, (b) profiled cell outside its captures vs its control,
    (c) TTFT difference per captured prefill chunk against the control's per-chunk time. MEASURED client numbers."""
    rows: dict[str, Json] = {}
    decode_cost: list[float] = []
    prefill_cost: list[float] = []
    for name, cell in sorted(cells.items()):
        caps = [c for c in map(record, cast(list[Json], cell.get('captures') or [])) if c.get('status') not in (None, 'SKIPPED_MEMORY')]
        if not caps:
            continue
        control = cells.get('ctl-' + name.removeprefix('prof-'))
        streams = [record(s) for s in cast(list[Json], cell['streams'])]
        windows = [(number(c['start_call_rel_s']), number(c['stop_return_rel_s']), str(c['kind'])) for c in caps]
        row: dict[str, Json] = {}
        inside: list[float] = []
        outside: list[float] = []
        for s in streams:
            for t0, t1 in _gaps(s):
                if any(w0 <= t0 and t1 <= w1 for w0, w1, k in windows if k == 'decode'):
                    inside.append(t1 - t0)
                elif all(t1 < w0 - OUTSIDE_PAD_S or t0 > w1 + OUTSIDE_PAD_S for w0, w1, _ in windows):
                    outside.append(t1 - t0)
        row['sse_gap_ms_inside_decode_capture'] = None if not inside else 1e3 * statistics.median(inside)
        row['sse_gap_ms_outside_captures'] = None if not outside else 1e3 * statistics.median(outside)
        ratio = _ratio(_median(inside), _median(outside))
        row['inside_over_outside'] = ratio
        if ratio is not None:
            decode_cost.append(ratio - 1)
        if control is not None:
            reference = [t1 - t0 for s in cast(list[Json], control['streams']) for t0, t1 in _gaps(record(s))]
            row['outside_over_control'] = _ratio(_median(outside), _median(reference))
            for kind in ('prefill-early', 'prefill-late'):
                if not any(k == kind for _, _, k in windows):
                    continue
                prof = [number(s['ttft_s']) for s in streams if s.get('ttft_s') is not None]
                ctl = [number(record(s)['ttft_s']) for s in cast(list[Json], control['streams'])
                       if record(s).get('ttft_s') is not None]
                tokens = [number(s['prompt_tokens']) for s in streams]
                count = chunks.get(f'{name}:{kind}')
                if prof and ctl and tokens and count:
                    per_chunk = statistics.median(ctl) / -(-max(tokens) // CHUNK_TOKENS)
                    extra = (statistics.median(prof) - statistics.median(ctl)) / count
                    row[f'{kind}_ttft_extra_s_per_chunk'] = extra
                    row[f'{kind}_relative'] = extra / per_chunk
                    prefill_cost.append(extra / per_chunk)
        rows[name] = row
    return {'status': 'MEASURED client-side (SSE event times, TTFT); n is small, read as a bound, not a precise cost',
            'cells': rows, 'decode_overhead': max(decode_cost) if decode_cost else None,
            'prefill_overhead': max(prefill_cost) if prefill_cost else None}


# ------------------------------------------------------------------ hypotheses


def rank_next(values: dict[str, float], r: Rules, experiments: dict[str, Json], limited: list[str]) -> dict[str, Json]:
    order = {str(record(e)['id']): int(number(record(e)['rank'])) for e in cast(list[Json], experiments['experiments'])}
    rows: list[dict[str, Json]] = []
    for h in r.hypotheses:
        hid = str(h['id'])
        if hid not in order:
            raise ValueError(f'hypothesis {hid} is not in experiments.json')
        row: dict[str, Json] = {'id': hid, 'experiments_rank': order[hid], 'then': h.get('then')}
        if h.get('trace') is False:
            rows.append({**row, 'status': 'NOT_FROM_TRACE'})
            continue
        seen: list[Json] = []
        hits: list[Json] = []
        for pattern, minimum in cast(list[list[Json]], h['any']):
            for key, value in sorted(values.items()):
                if fnmatch.fnmatchcase(key, str(pattern)):
                    seen.append(key)
                    if value >= number(minimum):
                        hits.append({'metric': key, 'value': value, 'min': minimum})
        addressable = [v for k, v in values.items() if fnmatch.fnmatchcase(k, str(h['addressable']))]
        rows.append({**row, 'status': 'SUPPORTED' if hits else 'NOT_SUPPORTED' if seen else 'NOT_MEASURED',
                     'evidence': hits, 'addressable_share': max(addressable) if addressable else None})
    supported = sorted((x for x in rows if x['status'] == 'SUPPORTED'),
                       key=lambda x: (-number(x['addressable_share'] or 0.0), int(number(x['experiments_rank']))))
    return {'ranking': 'LIMITED' if limited else 'OK', 'limited_by': cast(Json, limited),
            'next': [str(x['id']) for x in supported], 'hypotheses': cast(Json, rows),
            'rule': 'pre-registered in buckets.json; profiled shares are attribution, never promotion timings'}


# ------------------------------------------------------------------ window


def _verified(root: Path, row: dict[str, Json]) -> Path:
    path = (root / str(row['file'])).resolve()
    if root.resolve() not in path.parents:
        raise ValueError('manifest path leaves the window directory')
    if path.stat().st_size != int(number(row['bytes'])):
        raise ValueError(f'{path.name}: size differs from the manifest')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    if digest.hexdigest() != row['sha256']:
        raise ValueError(f'{path.name}: sha256 differs from the manifest')
    return path


def analyze_window(root: Path, r: Rules | None = None) -> dict[str, Json]:
    r = r or rules()
    manifest = [record(x) for x in cast(list[Json], json.loads((root / 'traces-manifest.json').read_text()))]
    groups: dict[tuple[str, str], dict[int, Path]] = {}
    statuses: dict[str, str] = {}
    for row in manifest:
        key = (str(row['cell']), str(row['capture']))
        rank = int(number(row['rank']))
        if rank in groups.setdefault(key, {}):
            raise ValueError(f'{key}: two traces for rank {rank}')
        groups[key][rank] = _verified(root, row)
        statuses[f'{key[0]}:{key[1]}'] = str(row['status'])
    captures: dict[str, dict[str, Json]] = {}
    chunk_counts: dict[str, int] = {}
    for (cell, kind), paths in sorted(groups.items()):
        label = f'{cell}:{kind}'
        budget = capture_budget({rank: load(path) for rank, path in paths.items()}, kind, r)
        budget['capture_status'] = statuses[label]
        captures[label] = budget
        if kind != 'decode' and '0' in record(budget['ranks']):
            chunk_counts[label] = len(cast(list[Json], record(record(budget['ranks'])['0'])['chunks']))
    cells = {p.stem: record(cast(Json, json.loads(p.read_text()))) for p in sorted((root / 'cells').glob('*.json'))}
    cost = overhead(cells, chunk_counts)
    values = metrics(captures)
    limited: list[str] = []
    for label, c in captures.items():
        if len(cast(list[Json], c['ranks_present'])) != 2:
            limited.append(f'{label}: one rank missing')
        elif not c['aligned']:
            limited.append(f'{label}: rank iteration sequences differ')
        for rank_name, info in record(c['ranks']).items():
            if c['phase'] == 'decode' and record(info)['graph_visibility'] in ('invisible', 'partial'):
                limited.append(f'{label} rank {rank_name}: graph kernels {record(info)["graph_visibility"]}')
        if 'coverage_min' in c and number(c['coverage_min']) < r.limits['coverage_min']:
            limited.append(f'{label}: kernel-name coverage {number(c["coverage_min"]):.2f}')
    for which in ('decode_overhead', 'prefill_overhead'):
        value = cost[which]
        if value is not None and number(value) > r.limits['profiler_overhead_max']:
            limited.append(f'{which} {number(value):.2f} > {r.limits["profiler_overhead_max"]}')
    experiments = record(cast(Json, json.loads(EXPERIMENTS.read_text())))
    return {'type': 'sp1-trace-budget', 'window': root.name, 'buckets_sha256': r.sha256,
            'status': 'MEASURED from profiled traces (profile-affected): attribution of GPU time, not promotion timings',
            'captures': cast(Json, captures), 'metrics': cast(Json, values), 'overhead': cost,
            'next_work': rank_next(values, r, experiments, limited)}


def summary(result: dict[str, Json], r: Rules) -> str:
    lines = [f"{result['window']}  buckets {str(result['buckets_sha256'])[:12]}  {result['status']}"]
    for label, c in record(result['captures']).items():
        c = record(c)
        flags = ','.join(f"r{k}:{record(v)['graph_visibility']}" for k, v in record(c['ranks']).items())
        if 'share' not in c:
            lines.append(f'{label:28} no complete {c["phase"]} steps on both ranks  [{flags}]')
            continue
        share = record(c['share'])
        top = ' '.join(f'{b}={100 * number(share[b]):.1f}%' for b in r.report if number(share[b]) >= .005)
        lines.append(f"{label:28} period {number(c['period_ms_median']):9.2f} ms  {top}  [{flags}]"
                     f" coverage {number(c['coverage_min']):.2f}")
    cost = record(result['overhead'])
    lines.append(f"profiler overhead: decode {cost['decode_overhead']}  prefill {cost['prefill_overhead']}")
    nxt = record(result['next_work'])
    lines.append(f"next work ({nxt['ranking']}): {', '.join(cast(list[str], nxt['next'])) or 'none supported'}")
    for reason in cast(list[str], nxt['limited_by']):
        lines.append(f'  limited: {reason}')
    return '\n'.join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--window', type=Path, help='runner output directory (traces-manifest.json, cells/, traces/)')
    ap.add_argument('--trace', type=Path, nargs='+', help='ad hoc: one trace per rank, rank order')
    ap.add_argument('--kind', choices=('decode', 'prefill-early', 'prefill-late'), default='decode')
    ap.add_argument('--output', type=Path, help='JSON result (default <window>/trace-budget.json); never overwritten')
    args = ap.parse_args(argv)
    r = rules()
    if args.window is not None:
        result = analyze_window(args.window, r)
        output = args.output or args.window / 'trace-budget.json'
        if output.exists():
            print(f'{output} exists; refusing to overwrite', file=sys.stderr)
            return 1
        output.write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
        print(summary(result, r))
        return 0
    if args.trace:
        budget = capture_budget({i: load(p) for i, p in enumerate(args.trace)}, args.kind, r)
        text = json.dumps(budget, indent=1, allow_nan=False) + '\n'
        if args.output is not None:
            if args.output.exists():
                print(f'{args.output} exists; refusing to overwrite', file=sys.stderr)
                return 1
            args.output.write_text(text)
        else:
            sys.stdout.write(text)
        return 0
    ap.error('give --window or --trace')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
