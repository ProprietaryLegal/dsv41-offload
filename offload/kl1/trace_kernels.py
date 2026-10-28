"""Kernel-level view of the SP1 kineto traces: per-step kernel families, per-layer MoE spans and collective pairs.

Read only. Step periods follow offload/gb10_budget/trace_budget.py exactly (an iteration's GPU period runs from its first
own activity to the next iteration's first own activity; the last iteration has no successor and is dropped), so
per-step numbers here line up with SP1's trace-budget.json. All durations are kineto's (MEASURED, profile-affected:
SP1 measured decode overhead 1.8-4.5 %, prefill 24.9 %).
"""

from __future__ import annotations

import bisect
import gzip
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, cast

from offload.gb10_budget.trace_budget import (CALL_KINDS, GPU_KINDS, MAX_TRACE_BYTES, Rules, bucket, details, events,
                                              rules)
from offload.prefill_maxlen.evidence import Json, number, record

# Ordered: first match wins. Finer than buckets.json; the SP1 bucket of every kernel is kept beside its family.
FAMILIES: tuple[tuple[str, re.Pattern[str]], ...] = tuple((f, re.compile(p)) for f, p in (
    ('l2pf', r'\bl2pf_kernel\b'),
    ('smoe_rot', r'\bsmoe_rot\b'),
    ('smoe_gu', r'\bsmoe_gu\b'),
    ('smoe_down', r'\bsmoe_down\b'),
    ('smoe_fused', r'\bsmoe_fused\b'),
    ('router', r'\brouter_kernel\b'),
    ('route_topk', r'\b_dsv4_topk_kernel\b'),
    ('tf_gu', r'\bgrouped_mma\w*kernel\b'),
    ('tf_down', r'\bgrouped_rows\w*kernel\b'),
    ('tf_aux', r'\brot_in_kernel\b|\bgateup_epilogue_kernel\b|\bdown_combine_kernel\b|\bdecode_prep_kernel\b|'
               r'\bwork_list_kernel\b|\bgroup_(count|place)?_?kernel\b|\bcombine(_y)?_kernel\b'),
    ('mrow_head', r'\bmrow(32)?_kernel<6\b'),
    ('mrow', r'\bmrow(32)?_kernel<'),
    ('exl3_gemm', r'\bexl3_gemm_kernel<'),
    ('reconstruct', r'\breconstruct_had_kernel<'),
    ('marlin', r'[Mm]arlin'),
    ('hc_prenorm', r'sm120_tf32_hc_prenorm_gemm'),
    ('mqa_logits', r'paged_mqa_logits'),
    ('gemm_other', r'cutlass|nvjet|cublas|gemm|Gemm|GEMM|xmma'),
    ('coll_stage', r'\bstage_kernel<'),
    ('coll_sum', r'\bsum_kernel\('),
    ('coll_gather', r'::gather_kernel<'),
    ('nccl', r'nccl'),
    ('sparse_mla', r'sparse_mla'),
))
COLLECTIVE = ('coll_stage', 'coll_sum', 'coll_gather', 'nccl')
TEMPLATE_K = re.compile(r'(?:mrow(?:32)?_kernel|exl3_gemm_kernel|reconstruct_had_kernel)<(\d+)')


def family(name: str) -> str:
    return next((f for f, p in FAMILIES if p.search(name)), 'rest')


def template_k(name: str) -> int | None:
    m = TEMPLATE_K.search(name)
    return int(m.group(1)) if m else None


@dataclass(frozen=True)
class Kernel:
    name: str
    family: str
    bucket: str                     # SP1's buckets.json bucket (l2pf is 'routed_moe' there)
    start: float                    # us, kineto timebase of this rank
    end: float
    stream: int
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    blocks_per_sm: float | None     # kineto's theoretical active blocks per SM (occupancy record)
    corr: int | None

    @property
    def dur(self) -> float:
        return self.end - self.start


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


@dataclass
class RankTrace:
    kernels: list[Kernel]           # GPU kernels, memcpys and memsets, by start
    calls: list[Call]
    annotations: list[Annotation]


@dataclass
class Step:
    index: int
    annotation: str
    phase: str
    rows: int                       # scheduled tokens of the phase (generation tokens for decode)
    requests: int
    lo: float
    hi: float
    kernels: list[Kernel] = field(default_factory=list)

    @property
    def period(self) -> float:
        return self.hi - self.lo


def _triple(value: Json) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        return (0, 0, 0)
    a, b, c = (int(number(v)) for v in value)
    return (a, b, c)


def parse(handle: IO[str], r: Rules) -> RankTrace:
    kernels: list[Kernel] = []
    calls: list[Call] = []
    annotations: list[Annotation] = []
    for event in events(handle):
        cat = event.get('cat')
        if event.get('ph') != 'X' or not isinstance(cat, str):
            continue
        name = str(event.get('name', ''))
        start = number(event['ts'])
        end = start + number(event.get('dur', 0))
        args = record(event.get('args') or {})
        corr = None if args.get('correlation') is None else int(number(args['correlation']))
        if cat in GPU_KINDS:
            occ = args.get('occupancy')
            bps = None
            if isinstance(occ, dict) and occ.get('activeBlocksPerMultiprocessor') is not None:
                bps = number(occ['activeBlocksPerMultiprocessor'])
            stream = args.get('stream')
            kernels.append(Kernel(name, family(name), bucket(name, r), start, end,
                                  -1 if stream is None else int(number(stream)), _triple(args.get('grid')),
                                  _triple(args.get('block')), bps, corr))
        elif cat in CALL_KINDS:
            calls.append(Call(name, start, end, str(event.get('tid')), corr))
        elif cat == 'user_annotation' and name.startswith('execute_'):
            annotations.append(Annotation(name, start, end, str(event.get('tid'))))
    kernels.sort(key=lambda k: (k.start, k.end))
    calls.sort(key=lambda c: c.start)
    annotations.sort(key=lambda a: a.start)
    return RankTrace(kernels, calls, annotations)


def load(path: Path, r: Rules | None = None) -> RankTrace:
    if path.stat().st_size > MAX_TRACE_BYTES:
        raise ValueError(f'{path.name}: larger than the per-file cap')
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        trace = parse(cast(IO[str], handle), r or rules())
        while handle.read(1 << 20):
            pass
    return trace


def steps(trace: RankTrace) -> list[Step]:
    """Complete iterations with their GPU period [first own activity, next iteration's first own activity)."""
    by_corr: dict[int, float] = {}
    for k in trace.kernels:
        if k.corr is not None:
            by_corr[k.corr] = min(by_corr.get(k.corr, k.start), k.start)
    call_starts = [c.start for c in trace.calls]
    firsts: list[float | None] = []
    for ann in trace.annotations:
        lo: float | None = None
        for c in trace.calls[bisect.bisect_left(call_starts, ann.start):bisect.bisect_right(call_starts, ann.end)]:
            if c.tid == ann.tid and c.corr is not None and c.corr in by_corr:
                t = by_corr[c.corr]
                lo = t if lo is None else min(lo, t)
        firsts.append(lo)
    starts = [k.start for k in trace.kernels]
    out: list[Step] = []
    for i, ann in enumerate(trace.annotations[:-1]):
        lo, hi = firsts[i], firsts[i + 1]
        if lo is None or hi is None or hi <= lo:
            continue
        d = details(ann.name)
        phase = str(d['phase'])
        rows = d['generation_tokens'] if phase == 'decode' else d['context_tokens']
        reqs = d['generation_requests'] if phase == 'decode' else d['context_requests']
        mine = trace.kernels[bisect.bisect_left(starts, lo):bisect.bisect_left(starts, hi)]
        out.append(Step(i, ann.name, phase, int(number(rows or 0)), int(number(reqs or 0)), lo, hi, list(mine)))
    return out


def union(spans: Iterable[tuple[float, float]], lo: float = float('-inf'), hi: float = float('inf')) -> float:
    total = 0.0
    cur_s = cur_e = None
    for s, e in sorted((max(s, lo), min(e, hi)) for s, e in spans):
        if e <= s:
            continue
        if cur_e is None or s > cur_e:
            if cur_e is not None and cur_s is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None and cur_s is not None:
        total += cur_e - cur_s
    return total


@dataclass(frozen=True)
class MoeLayer:
    """One layer's routed-expert launches in a step (decode: smoe rot/gu/down; prefill: TF grouped gate|up / down)."""
    router: Kernel | None
    topk: Kernel | None
    rot: Kernel | None
    gu: Kernel
    down: Kernel

    @property
    def start(self) -> float:
        return self.rot.start if self.rot is not None else self.gu.start

    @property
    def span(self) -> float:
        return max(self.down.end, self.gu.end) - self.start

    @property
    def down_after_gu(self) -> float:
        return max(0.0, self.down.end - self.gu.end)


def moe_layers(step: Step) -> list[MoeLayer]:
    ks = step.kernels
    out: list[MoeLayer] = []
    router: Kernel | None = None
    topk: Kernel | None = None
    rot: Kernel | None = None
    gu: Kernel | None = None
    for k in ks:
        if k.family == 'router':
            router = k
        elif k.family == 'route_topk':
            topk = k
        elif k.family in ('smoe_rot',):
            rot = k
        elif k.family in ('smoe_gu', 'tf_gu'):
            gu = k
        elif k.family in ('smoe_down', 'tf_down') and gu is not None:
            out.append(MoeLayer(router, topk, rot, gu, k))
            router = topk = rot = gu = None
    return out


def family_totals(step: Step) -> dict[str, dict[str, float]]:
    """Per family: launches, summed kernel time and union (wall) time inside the step period."""
    groups: dict[str, list[Kernel]] = {}
    for k in step.kernels:
        groups.setdefault(k.family, []).append(k)
    return {f: {'n': float(len(v)), 'sum_us': sum(min(k.end, step.hi) - k.start for k in v),
                'union_us': union(((k.start, k.end) for k in v), step.lo, step.hi)}
            for f, v in sorted(groups.items())}


def collective_calls(step: Step) -> list[Kernel]:
    return [k for k in step.kernels if k.family in COLLECTIVE]


def compute_segments(step: Step) -> list[tuple[str, float]]:
    """Main-stream work between collectives: (kind, us) from the end of one all-reduce's sum to the next stage. Kind is
    'moe' when the segment holds smoe/TF expert kernels, 'attn' otherwise (the attention half of a layer)."""
    marks = [k for k in step.kernels if k.family in ('coll_stage', 'coll_sum')]
    segs: list[tuple[str, float]] = []
    prev_end: float | None = None
    for k in marks:
        if k.family == 'coll_stage' and prev_end is not None:
            inside = [x for x in step.kernels if prev_end <= x.start < k.start]
            kind = 'moe' if any(x.family in ('smoe_gu', 'tf_gu') for x in inside) else 'attn'
            segs.append((kind, k.start - prev_end))
            prev_end = None
        elif k.family == 'coll_sum':
            prev_end = k.end
    return segs


def pair_steps(a: Sequence[Step], b: Sequence[Step]) -> list[tuple[Step, Step]]:
    """Steps of the two ranks with the same iteration index and annotation (the ranks run the same schedule)."""
    bi = {s.index: s for s in b}
    return [(s, bi[s.index]) for s in a if s.index in bi and bi[s.index].annotation == s.annotation]
