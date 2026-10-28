"""CEN0 item 1: the per-round decode waterfall of F-CENSUS.plan.md §3 from both TP ranks' traces.

  PYTHONPATH=. python3 -m offload.census.waterfall --window <SP1-W-dir> [--window ...] --output <json> [--markdown <md>]
  PYTHONPATH=. python3 -m offload.census.waterfall --trace <rank0 trace> <rank1 trace> --output <json>
  PYTHONPATH=. python3 -m offload.census.waterfall --nsys <rank0.sqlite> <rank1.sqlite> --output <json>

Kineto traces (.pt.trace.json[.gz], the SP1 profiler config) or Nsight Systems SQLite exports (`nsys export --type
sqlite`, offload/census/nsys.py; step NVTX ranges and node-level graph tracing required). CPU only, read only.

Method (inherits SP1's trace_budget.py so the two agree; tested against trace-budget-v3.json):
  * steps are vLLM's `execute_*` iterations; a step's GPU period runs from its first own activity to the next step's
    first own activity (offload/kl1/trace_kernels.steps); the two ranks' steps are paired by index and annotation;
  * busy time is split equally among the buckets active at each instant (trace_budget.sweep), so a step's buckets sum
    to its period; idle time is classified by the activity that ends it (buckets.json wait rules: drafter / Engram
    wait, else device wait / host gap / unrecorded);
  * buckets are offload/census/rules.py: SP1's, with mHC, Engram, the indexer and its top-k, the MoE router and the
    LM head split out;
  * cross-rank waits (KL1 item 3): the ranks' collectives are paired by order inside a step; for each all-reduce
    sum_kernel and NCCL broadcast the early rank's extra duration over the peer's (|d0 - d1|, the early rank polling)
    becomes `rank_wait` / `bcast_wait` at the start of its kernel, the rest stays intrinsic. Clocks are never
    compared across ranks;
  * the SP1 buckets are re-binned beside the census buckets on the same steps; with a window's trace-budget json
    present the difference per bucket is reported (it should be ~0: same method, same traces).
Every duration is MEASURED by the profiler (profile-affected; SP1 measured decode overhead 1.8-4.5 %). Attribution of
idle time and of the broadcast wait to the drafter is ESTIMATED from trace structure. Output: numbers, kernel-family
names and file hashes only.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, cast

from offload.census import rules as C
from offload.gb10_budget.trace_budget import MAX_TRACE_BYTES, Rules, WaitRule, sweep
from offload.gb10_budget.trace_budget import rules as sp1_rules
from offload.kl1 import collectives as KC
from offload.kl1 import trace_kernels as T
from offload.prefill_maxlen.evidence import Json, number, record

CHUNK = 1 << 20
WAIT_OF = {'coll_sum': 'rank_wait', 'nccl': 'bcast_wait'}
KL1_KEYS = ('sum_intrinsic_ms', 'sum_wait_ms', 'sum_wait_r0_ms', 'sum_wait_r1_ms', 'stage_mean_ms', 'gather_min_ms',
            'nccl_r0_ms', 'nccl_r1_ms', 'idle_r0_ms', 'idle_r1_ms', 'attn_imbalance_ms', 'moe_imbalance_ms',
            'attn_r1_slower_frac', 'moe_r1_slower_frac')
BUDGET_FILES = ('trace-budget-v3.json', 'trace-budget-v2.json', 'trace-budget.json')


@dataclass(frozen=True)
class Rank:
    """One rank's trace with the lookups the binner needs."""
    trace: T.RankTrace
    starts: list[float]
    max_dur: float
    launch_of: dict[int, T.Call]
    source: str
    sha256: str


def prepare(trace: T.RankTrace, source: str, sha256: str) -> Rank:
    corrs = {k.corr for k in trace.kernels if k.corr is not None}
    # trace_budget.steps keeps the last call per correlation id (dict comprehension over calls in start order)
    launch_of = {c.corr: c for c in trace.calls if c.corr is not None and c.corr in corrs}
    return Rank(trace, [k.start for k in trace.kernels], max((k.dur for k in trace.kernels), default=0.0),
                launch_of, source, sha256)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def load_kineto(path: Path, r: Rules | None = None) -> Rank:
    if path.stat().st_size > MAX_TRACE_BYTES:
        raise ValueError(f'{path.name}: larger than the per-file cap')
    if path.name.endswith('.gz'):
        trace = T.load(path, r)
    else:
        with path.open(encoding='utf-8') as handle:
            trace = T.parse(cast(IO[str], handle), r or sp1_rules())
    return prepare(trace, 'kineto', _sha256(path))


def overlapping(rank: Rank, lo: float, hi: float) -> list[T.Kernel]:
    """Every GPU activity that overlaps [lo, hi] (trace_budget.steps: started before hi, ends after lo)."""
    i0 = bisect.bisect_left(rank.starts, lo - rank.max_dur)
    i1 = bisect.bisect_left(rank.starts, hi)
    return [k for k in rank.trace.kernels[i0:i1] if k.end > lo]


def classify(gap_start: float, nxt: T.Kernel, launch: T.Call | None, waits: Sequence[WaitRule],
             names: Mapping[str, str]) -> str:
    """trace_budget.classify_gap on this module's types; `names` renames the wait-rule buckets."""
    for w in waits:
        if not w.next.search(nxt.name):
            continue
        if launch is None and w.unrecorded:
            return names.get(w.bucket, w.bucket)
        if launch is not None and w.launch is not None and w.launch.search(launch.name) and launch.start <= gap_start:
            return names.get(w.bucket, w.bucket)
    if launch is None:
        return 'unrecorded_wait'
    return 'device_wait' if launch.start <= gap_start else 'host_gap'


Labeler = Callable[[T.Kernel], list[tuple[float, float, str]]]


def budget(rank: Rank, step: T.Step, label: Labeler, waits: Sequence[WaitRule], names: Mapping[str, str]
           ) -> tuple[dict[str, float], dict[str, float]]:
    """(exclusive budget incl. classified idle, inclusive union) in microseconds over the step's GPU period."""
    spans = [span for k in overlapping(rank, step.lo, step.hi) for span in label(k)]
    exclusive, inclusive, gaps = sweep(spans, step.lo, step.hi)
    out = dict(exclusive)
    for g0, g1 in gaps:
        j = bisect.bisect_left(rank.starts, g1)
        if j >= len(rank.trace.kernels):
            continue                      # cannot happen inside [lo, hi]: hi is an activity start or end
        nxt = rank.trace.kernels[j]
        launch = rank.launch_of.get(nxt.corr) if nxt.corr is not None else None
        kind = classify(g0, nxt, launch, waits, names)
        out[kind] = out.get(kind, 0.0) + (g1 - g0)
    return out, inclusive


def sp1_label(k: T.Kernel) -> list[tuple[float, float, str]]:
    return [(k.start, k.end, k.bucket)]


def census_label(wait: Mapping[T.Kernel, float]) -> Labeler:
    def label(k: T.Kernel) -> list[tuple[float, float, str]]:
        b = C.bucket(k.name)
        w = min(wait.get(k, 0.0), k.dur)
        if w <= 0:
            return [(k.start, k.end, b)]
        return [(k.start, k.start + w, WAIT_OF[k.family]), (k.start + w, k.end, b)]
    return label


def collective_waits(a: T.Step, b: T.Step) -> tuple[dict[T.Kernel, float], dict[T.Kernel, float]]:
    """Per-kernel cross-rank wait of each rank: collectives paired by order inside the step (both ranks replay the same
    graph); for sum_kernel and NCCL pairs the longer kernel's excess over the peer's is that rank's wait."""
    ca, cb = T.collective_calls(a), T.collective_calls(b)
    if [k.family for k in ca] != [k.family for k in cb]:
        raise ValueError(f'step {a.index}: collective sequences differ between ranks')
    wa: dict[T.Kernel, float] = {}
    wb: dict[T.Kernel, float] = {}
    for x, y in zip(ca, cb):
        if x.family in WAIT_OF:
            if x.dur > y.dur:
                wa[x] = x.dur - y.dur
            elif y.dur > x.dur:
                wb[y] = y.dur - x.dur
    return wa, wb


def graph_visibility(rank: Rank, steps: Sequence[T.Step], r: Rules) -> dict[str, Json]:
    """Graph launches inside the steps' annotations with and without recorded kernels (trace_budget.graph_flag)."""
    corrs = {k.corr for k in rank.trace.kernels if k.corr is not None}
    call_starts = [c.start for c in rank.trace.calls]
    launched = visible = 0
    for s in steps:
        ann = rank.trace.annotations[s.index]
        for c in rank.trace.calls[bisect.bisect_left(call_starts, ann.start):bisect.bisect_right(call_starts, ann.end)]:
            if c.tid == ann.tid and c.corr is not None and r.graph_launch.search(c.name):
                launched += 1
                visible += c.corr in corrs
    flag = 'no_graphs' if not launched else 'visible' if visible == launched else 'invisible' if not visible \
        else 'partial'
    return {'graph_visibility': flag, 'graph_launches': launched, 'graph_launches_with_activity': visible}


def _ms(us: Mapping[str, float], keys: Sequence[str]) -> dict[str, float]:
    return {k: us.get(k, 0.0) / 1e3 for k in keys}


def _fmean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError('mean of nothing')
    return statistics.fmean(values)


def sp1_rebin(rank: Rank, steps: Sequence[T.Step], r: Rules) -> dict[str, Json]:
    """SP1's buckets on every complete decode step of one rank (trace_budget.aggregate's 'ms' and period median)."""
    names = {w.bucket: w.bucket for w in r.waits}
    rows = [_ms(budget(rank, s, sp1_label, r.waits, names)[0], r.report) for s in steps]
    return {'steps': len(rows), 'period_ms_median': statistics.median(s.period / 1e3 for s in steps),
            'ms': {b: _fmean([row[b] for row in rows]) for b in r.report}}


def capture(ranks: Mapping[int, Rank], r: Rules | None = None, reference: Mapping[str, Json] | None = None
            ) -> dict[str, Json]:
    """Census waterfall of one decode capture from both ranks."""
    r = r or sp1_rules()
    if sorted(ranks) != [0, 1]:
        raise ValueError('need both ranks (0 and 1)')
    decode = {k: [s for s in T.steps(v.trace) if s.phase == 'decode'] for k, v in ranks.items()}
    vis = {k: graph_visibility(ranks[k], decode[k], r) for k in (0, 1)}
    hidden = [k for k in (0, 1) if vis[k]['graph_visibility'] not in ('visible', 'no_graphs')]
    out: dict[str, Json] = {'ranks': {str(k): {'source': ranks[k].source, 'sha256': ranks[k].sha256,
                                               'decode_steps': len(decode[k]), **vis[k]} for k in (0, 1)}}
    if hidden:
        out['status'] = f'UNBUDGETED: graph kernels {[vis[k]["graph_visibility"] for k in hidden]} (eager capture needed)'
        return out
    pairs = T.pair_steps(decode[0], decode[1])
    if not pairs:
        raise ValueError('no paired decode steps')
    rows: list[dict[str, Json]] = []
    for a, b in pairs:
        wa, wb = collective_waits(a, b)
        names = dict(C.WAIT_RULE_NAMES)
        per: dict[str, dict[str, float]] = {}
        incl: dict[str, dict[str, float]] = {}
        for tag, rank, step, wait in (('0', ranks[0], a, wa), ('1', ranks[1], b, wb)):
            ex, inc = budget(rank, step, census_label(wait), r.waits, names)
            unknown = set(ex) - set(C.REPORT)
            if unknown:
                raise ValueError(f'buckets outside the census report: {sorted(unknown)}')
            per[tag] = {'period_ms': step.period / 1e3, **_ms(ex, C.REPORT)}
            incl[tag] = _ms(inc, ('rank_wait', 'bcast_wait', 'mhc', 'prefetch', 'smoe', 'allreduce'))
        kl1 = KC.step_pair(a, b)
        rows.append({'index': a.index, 'rows': a.rows, 'requests': a.requests,
                     'ranks': cast(Json, per), 'inclusive': cast(Json, incl),
                     'kl1': cast(Json, {k: kl1[k] for k in KL1_KEYS})})
    out.update(aggregate(rows))
    out['by_rows'] = cast(Json, {str(n): aggregate([x for x in rows if x['rows'] == n])
                                 for n in sorted({int(number(x['rows'])) for x in rows})})
    sp1 = {str(k): sp1_rebin(ranks[k], decode[k], r) for k in (0, 1)}
    both = {b: _fmean([number(record(sp1[k]['ms'])[b]) for k in ('0', '1')]) for b in r.report}
    check: dict[str, Json] = {'ms': cast(Json, both), 'period_ms_median':
                              _fmean([number(sp1[k]['period_ms_median']) for k in ('0', '1')]),
                              'steps': cast(Json, {k: sp1[k]['steps'] for k in sp1})}
    if reference is not None and 'ms' in reference:
        ref = record(reference['ms'])
        check['reference_max_abs_diff_ms'] = max(abs(both[b] - number(ref[b])) for b in r.report)
        check['reference_period_diff_ms'] = abs(number(check['period_ms_median']) -
                                                number(reference['period_ms_median']))
    out['sp1_rebin'] = check
    out['status'] = 'MEASURED kineto/nsys durations (profile-affected); idle and broadcast-wait attribution ESTIMATED'
    return out


def aggregate(rows: Sequence[dict[str, Json]]) -> dict[str, Json]:
    """Means over paired steps: per rank, both-rank mean (the waterfall), shares of the mean period, KL1 fields."""
    if not rows:
        return {'pairs': 0}
    keys = ('period_ms', *C.REPORT)
    rank_mean = {tag: {k: _fmean([number(record(record(x['ranks'])[tag])[k]) for x in rows]) for k in keys}
                 for tag in ('0', '1')}
    both = {k: (rank_mean['0'][k] + rank_mean['1'][k]) / 2 for k in keys}
    period = both['period_ms']
    groups = {name: sum(both[b] for b in bs) for name, bs in C.GROUPS}
    kl1 = {k: _fmean([number(record(x['kl1'])[k]) for x in rows]) for k in KL1_KEYS}
    kl1['allreduce_intrinsic_ms'] = kl1['sum_intrinsic_ms'] + kl1['stage_mean_ms'] + kl1['gather_min_ms']
    inclusive = {tag: {k: _fmean([number(record(record(x['inclusive'])[tag])[k]) for x in rows])
                       for k in record(record(rows[0]['inclusive'])[tag])} for tag in ('0', '1')}
    return {'pairs': len(rows), 'rows_hist': cast(Json, _hist([int(number(x['rows'])) for x in rows])),
            'period_ms_mean': period,
            'period_ms_median': statistics.median(
                (number(record(record(x['ranks'])['0'])['period_ms']) +
                 number(record(record(x['ranks'])['1'])['period_ms'])) / 2 for x in rows),
            'ms': cast(Json, {k: both[k] for k in C.REPORT}),
            'share': cast(Json, {k: both[k] / period for k in C.REPORT}),
            'groups_ms': cast(Json, groups),
            'rank_ms': cast(Json, {tag: {k: rank_mean[tag][k] for k in keys} for tag in ('0', '1')}),
            'inclusive_ms': cast(Json, inclusive),
            # rank 0 idles before the drafts' H2D while rank 1 waits in the step-start broadcast for rank 0: the same
            # wall interval seen from each rank (ESTIMATED attribution)
            'drafter_wait_incl_bcast_ms': both['drafter_wait'] + both['bcast_wait'],
            'cross_rank_wait_ms': both['rank_wait'] + both['bcast_wait'],
            'kl1': cast(Json, kl1)}


def _hist(xs: Sequence[int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in sorted(xs):
        out[str(x)] = out.get(str(x), 0) + 1
    return out


# ------------------------------------------------------------------ windows


def _verified(root: Path, row: Mapping[str, Json]) -> Path:
    path = (root / str(row['file'])).resolve()
    if root.resolve() not in path.parents:
        raise ValueError('manifest path leaves the window directory')
    if path.stat().st_size != int(number(row['bytes'])):
        raise ValueError(f'{path.name}: size differs from the manifest')
    if _sha256(path) != row['sha256']:
        raise ValueError(f'{path.name}: sha256 differs from the manifest')
    return path


def reference_budget(root: Path) -> tuple[str | None, dict[str, Json]]:
    """The newest trace-budget json the window holds (v3 > v2 > v1), by capture label."""
    for name in BUDGET_FILES:
        p = root / name
        if p.exists():
            return name, record(record(cast(Json, json.loads(p.read_text())))['captures'])
    return None, {}


def window(root: Path, r: Rules | None = None, *, strict: bool = True) -> dict[str, Json]:
    """Every decode capture of a window's traces-manifest.json. strict (SP1): a decode capture that is not TAKEN
    refuses; otherwise (CEN3, whose cells may end before the explicit stop) it is listed under skipped_captures."""
    r = r or sp1_rules()
    manifest = [record(x) for x in cast(list[Json], json.loads((root / 'traces-manifest.json').read_text()))]
    groups: dict[str, dict[int, Path]] = {}
    skipped: list[str] = []
    for row in manifest:
        label = f"{row['cell']}:{row['capture']}"
        if row['capture'] != 'decode':
            skipped.append(label)
            continue
        if row.get('status') != 'TAKEN':
            if not strict:
                skipped.append(f'{label} ({row.get("status")})')
                continue
            raise ValueError(f'{label}: capture status {row.get("status")!r}')
        rank = int(number(row['rank']))
        if rank in groups.setdefault(label, {}):
            raise ValueError(f'{label}: two traces for rank {rank}')
        groups[label][rank] = _verified(root, row)
    ref_name, refs = reference_budget(root)
    captures: dict[str, Json] = {}
    for label, paths in sorted(groups.items()):
        ref = record(refs[label]) if label in refs else None
        captures[label] = capture({k: load_kineto(p, r) for k, p in paths.items()}, r, ref)
    return {'window': root.name, 'reference_budget': ref_name, 'captures': cast(Json, captures),
            'skipped_captures': cast(Json, sorted(set(skipped)))}


def result(windows: Mapping[str, Json]) -> dict[str, Json]:
    return {'type': 'cen0-waterfall', 'rules_sha256': C.SHA256, 'sp1_buckets_sha256': sp1_rules().sha256,
            'status': 'MEASURED profiled durations (profile-affected), ESTIMATED idle/broadcast-wait attribution; '
                      'attribution of GPU time, never promotion timings',
            'windows': cast(Json, dict(windows))}


# ------------------------------------------------------------------ report


def _union_mean(cap: Mapping[str, Json], b: str) -> float:
    """Both-rank mean of a bucket's union (wall) time per round, overlap with other streams included."""
    inc = record(cap['inclusive_ms'])
    return (number(record(inc['0'])[b]) + number(record(inc['1'])[b])) / 2


def markdown(res: Mapping[str, Json]) -> str:
    """One waterfall table per window: census lines x decode captures, ms per round (share of the period)."""
    out: list[str] = []
    for wname, w in record(res['windows']).items():
        caps = {k: record(v) for k, v in record(record(w)['captures']).items() if 'ms' in record(v)}
        if not caps:
            continue
        labels = list(caps)
        out.append(f'### {wname}\n')
        out.append('| ms per round (share) | ' + ' | '.join(lab.removesuffix(':decode') for lab in labels) + ' |')
        out.append('|---|' + '---|' * len(labels))
        out.append('| period (mean of ranks) | ' + ' | '.join(f"{number(caps[x]['period_ms_mean']):.1f}"
                                                          for x in labels) + ' |')
        out.append('| paired steps; rows | ' + ' | '.join(
            f"{caps[x]['pairs']}; " + ','.join(f'{k}x{v}' for k, v in record(caps[x]['rows_hist']).items())
            for x in labels) + ' |')
        for name, _ in C.GROUPS:
            cells = []
            for x in labels:
                ms = number(record(caps[x]['groups_ms'])[name])
                cells.append(f'{ms:.2f} ({100 * ms / number(caps[x]["period_ms_mean"]):.1f} %)')
            out.append(f'| {name} | ' + ' | '.join(cells) + ' |')
        for b in ('smoe', 'mhc', 'prefetch', 'allreduce', 'rank_wait', 'bcast_wait'):
            out.append(f'| wall (union, incl. overlap): {b} | '
                       + ' | '.join(f'{_union_mean(caps[x], b):.2f}' for x in labels) + ' |')
        out.append('| drafter wait incl. rank-1 broadcast wait | ' + ' | '.join(
            f"{number(caps[x]['drafter_wait_incl_bcast_ms']):.2f}" for x in labels) + ' |')
        out.append('| KL1: rank waits both ranks / all-reduce intrinsic | ' + ' | '.join(
            f"{number(record(caps[x]['kl1'])['sum_wait_ms']):.2f} / "
            f"{number(record(caps[x]['kl1'])['allreduce_intrinsic_ms']):.2f}" for x in labels) + ' |')
        out.append('| KL1: rank 1 slower in MoE segments | ' + ' | '.join(
            f"{100 * number(record(caps[x]['kl1'])['moe_r1_slower_frac']):.0f} %" for x in labels) + ' |')
        out.append('| SP1 re-bin vs ' + str(record(w).get('reference_budget')) + ': max abs diff ms | ' + ' | '.join(
            (f"{number(record(caps[x]['sp1_rebin'])['reference_max_abs_diff_ms']):.4f}"
             if 'reference_max_abs_diff_ms' in record(caps[x]['sp1_rebin']) else 'n/a') for x in labels) + ' |')
        out.append('')
    return '\n'.join(out)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--window', type=Path, action='append', help='SP1-style window dir (traces-manifest.json)')
    src.add_argument('--trace', type=Path, nargs=2, metavar=('RANK0', 'RANK1'), help='ad hoc Kineto pair')
    src.add_argument('--nsys', type=Path, nargs=2, metavar=('RANK0', 'RANK1'), help='Nsight Systems SQLite pair')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--markdown', type=Path)
    args = ap.parse_args(argv)
    for p in (args.output, args.markdown):
        if p is not None and p.exists():
            print(f'{p} exists; refusing to overwrite', file=sys.stderr)
            return 1
    r = sp1_rules()
    if args.window:
        res = result({root.name: window(root, r) for root in args.window})
    else:
        if args.trace:
            ranks = {i: load_kineto(p, r) for i, p in enumerate(args.trace)}
        else:
            import offload.census.nsys as nsys
            ranks = {i: nsys.load(p, r) for i, p in enumerate(args.nsys)}
        res = result({'ad-hoc': {'window': 'ad-hoc', 'reference_budget': None,
                                 'captures': {'ad-hoc:decode': capture(ranks, r)}}})
    args.output.write_text(json.dumps(res, indent=1, allow_nan=False) + '\n')
    if args.markdown is not None:
        args.markdown.write_text(markdown(res) + '\n')
    print(markdown(res))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
