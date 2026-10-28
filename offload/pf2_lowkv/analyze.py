#!/usr/bin/env python3
"""PF2-LOWKV analysis (CPU only, read-only on a finished window directory; never run by the lane on real data).

    python3 -m offload.pf2_lowkv.analyze --window <PF2 window dir> [--reference <SP1 rerun dir> | --reference-ttft S]

1. Classifies every unprofiled cold 150K (c150-a/b) fast / slow / ambiguous against the same-day 327K reference with the
   thresholds pre-registered in DESIGN.md section 5, then each start, then the window (the pre-registered reading).
2. Memory pressure and compaction around each 150K prefill, per Spark: deltas of the /proc/vmstat counters and PSI totals
   from 10 s before submission to the first token, MemAvailable min/median, swap change, and per-process PSS/RSS change
   (in-container sampler).
3. Kernel dispatch in each profiled p150 capture: SP1 trace_budget's per-chunk bucket budget, plus a signature table
   (kernel name, grid, block, registers, shared memory) with launches and per-launch time; a diff between one fast and one
   slow capture per rank (H vs L, labelled 'no state contrast', when the window has no fast/slow pair).
Numbers only; the output repeats no prompt or generated text (the window never recorded any).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from offload.gb10_budget import trace_budget as tb
from offload.prefill_maxlen.evidence import Json, number, record

FAST_MAX, SLOW_MIN = 1.12, 1.20       # TTFT / reference (DESIGN.md section 5, pre-registered)
DEFAULT_REF_S = 70.0                  # PF1/SP1 MEASURED 327K 150K cold TTFT 68.8-71.2 s
PRE_S = 10.0                          # memory window starts this long before submission
CHANGE = .10                          # per-launch time or launch-share change worth listing in the dispatch diff
ROLLUP_FLAG_MS = 50.0                 # smaps_rollup read time that may perturb the engine (DESIGN.md section 4)
DECODE_FLAG = .02                     # L decode slower than H by more than this is flagged for promotion
COUNTERS = ('compact_stall', 'compact_fail', 'compact_success', 'compact_migrate_scanned', 'compact_free_scanned',
            'pgscan_kswapd', 'pgscan_direct', 'pgsteal_kswapd', 'pgsteal_direct', 'thp_fault_alloc', 'thp_fault_fallback',
            'pgmajfault', 'pswpin', 'pswpout')
HOSTS = ('sparkhead', 'sparkworker-wired')
CLASSIFIED_CELLS = ('c150-a', 'c150-b')


# ------------------------------------------------------------------ 1. classification and reading

def classify(ttft: float, ref: float) -> str:
    ratio = ttft / ref
    return 'fast' if ratio <= FAST_MAX else 'slow' if ratio >= SLOW_MIN else 'ambiguous'


def start_class(classes: Sequence[str]) -> str:
    if len(classes) != len(CLASSIFIED_CELLS):
        return 'incomplete'
    if all(c == 'fast' for c in classes):
        return 'fast'
    if all(c == 'slow' for c in classes):
        return 'slow'
    return 'ambiguous' if 'ambiguous' in classes else 'mixed'


def reading(starts: dict[str, tuple[str, str]]) -> dict[str, Json]:
    """starts: name -> (role H|L, start class). Rules in DESIGN.md section 5 order (slow L first); the first that matches wins."""
    by = {r: [c for role, c in starts.values() if role == r and c != 'incomplete'] for r in ('H', 'L')}
    if 'slow' in by['L']:              # one clearly slow L start settles it (brief's pre-registered rule; S44 note 1)
        verdict = 'NOT_HEADROOM'
    elif min(len(by['H']), len(by['L'])) < 2:
        verdict = 'INCOMPLETE'
    elif any(c in ('ambiguous', 'mixed') for c in by['H'] + by['L']):
        verdict = 'AMBIGUOUS'
    elif 'slow' in by['H']:
        verdict = 'HEADROOM_IMPLICATED'
    else:
        verdict = 'NO_CAUSE_SHOWN'
    meaning = {'INCOMPLETE': 'fewer than 2 classified starts per arm: no reading',
               'AMBIGUOUS': 'an ambiguous request or a mixed start: no reading',
               'NOT_HEADROOM': 'L was slow in a start: the smaller reservation did not prevent the slow state',
               'HEADROOM_IMPLICATED': 'every L start fast while H was slow in this window: headroom implicated',
               'NO_CAUSE_SHOWN': 'every start fast: the slow state did not appear; L stays a candidate, no cause shown'}
    return {'verdict': verdict, 'meaning': meaning[verdict], 'starts_by_role': cast(Json, by),
            'thresholds': {'fast_max_ratio': FAST_MAX, 'slow_min_ratio': SLOW_MIN}}


# ------------------------------------------------------------------ 2. memory around a prefill

@dataclass(frozen=True)
class Sample:
    utc: float
    mem: dict[str, int]
    vm: dict[str, int]
    psi: dict[str, float]


def telemetry(path: Path) -> list[Sample]:
    rows: list[Sample] = []
    for line in path.read_text().splitlines():
        row = record(cast(Json, json.loads(line)))
        if row.get('kind') != 'memory':
            continue
        psi: dict[str, float] = {}
        raw = row.get('psi')
        if isinstance(raw, dict):
            for kind, values in raw.items():
                for key, value in record(values).items():
                    psi[f'{kind}_{key}'] = number(value)
        rows.append(Sample(number(row['utc']), {k: int(number(v)) for k, v in record(row['kb']).items()},
                           {k: int(number(v)) for k, v in record(row.get('vmstat') or {}).items()}, psi))
    return rows


def around(samples: list[Sample], lo: float, hi: float) -> dict[str, Json]:
    inside = [x for x in samples if lo <= x.utc <= hi]
    if len(inside) < 2:
        return {'samples': len(inside), 'status': 'NO_DATA'}
    first, last = inside[0], inside[-1]
    stalls = {k: last.vm[k] - first.vm[k] for k in (*COUNTERS, *sorted(k for k in last.vm if k.startswith('allocstall_')))
              if k in first.vm and k in last.vm}
    avail = [x.mem['MemAvailable'] for x in inside if 'MemAvailable' in x.mem]
    out: dict[str, Json] = {'samples': len(inside), 'seconds': last.utc - first.utc, 'vmstat_delta': cast(Json, stalls),
                            'mem_available_min_kb': min(avail), 'mem_available_median_kb': statistics.median(avail)}
    if 'SwapFree' in first.mem and 'SwapFree' in last.mem:
        out['swap_used_delta_kb'] = first.mem['SwapFree'] - last.mem['SwapFree']
    for key in ('some_total', 'full_total'):
        if key in first.psi and key in last.psi:
            out[f'psi_{key}_delta_us'] = last.psi[key] - first.psi[key]
    for key in ('some_avg10', 'full_avg10'):
        values = [x.psi[key] for x in inside if key in x.psi]
        if values:
            out[f'psi_{key}_max'] = max(values)
    return out


def procmem(path: Path) -> list[tuple[float, dict[str, dict[str, int]]]]:
    """(utc, role -> summed fields) per sample; worker rows of several processes are summed."""
    rows: list[tuple[float, dict[str, dict[str, int]]]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.startswith('{'):
            continue
        row = record(cast(Json, json.loads(line)))
        roles: dict[str, dict[str, int]] = {}
        for proc in map(record, cast(list[Json], row.get('procs') or [])):
            role = str(proc.get('role'))
            mine = roles.setdefault(role, {})
            for key, value in proc.items():
                if key in ('pid', 'role', 'error') or not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                if key == 'rollup_ms':
                    mine['rollup_ms_max'] = max(mine.get('rollup_ms_max', 0), int(value))
                else:
                    mine[key] = mine.get(key, 0) + int(value)
        rows.append((number(row['utc']), roles))
    return rows


def procmem_delta(rows: list[tuple[float, dict[str, dict[str, int]]]], lo: float, hi: float) -> dict[str, Json]:
    inside = [r for r in rows if lo <= r[0] <= hi]
    if len(inside) < 2:
        return {'status': 'NO_DATA'}
    first, last = inside[0][1], inside[-1][1]
    out: dict[str, Json] = {}
    for role in sorted(set(first) & set(last)):
        out[role] = {f'{k}_delta_kb': last[role][k] - first[role][k] for k in ('Pss', 'Rss', 'Pss_Anon', 'Swap', 'VmRSS')
                     if k in first[role] and k in last[role]}
        record(out[role])['rollup_ms_max'] = max(r[1].get(role, {}).get('rollup_ms_max', 0) for r in inside)
    return out


# ------------------------------------------------------------------ 3. kernel dispatch

def signatures(path: Path) -> dict[str, dict[str, float]]:
    """(name, grid, block, registers, shared memory) -> launches, total/mean/median us, bucket; numbers only."""
    if path.stat().st_size > tb.MAX_TRACE_BYTES:
        raise ValueError(f'{path.name}: larger than the per-file cap')
    durations: dict[str, list[float]] = {}
    r = tb.rules()
    opener = gzip.open(path, 'rt', encoding='utf-8') if path.name.endswith('.gz') else path.open(encoding='utf-8')
    with opener as handle:
        for event in tb.events(handle):
            if event.get('ph') != 'X' or event.get('cat') != 'kernel':
                continue
            args = record(event.get('args') or {})
            key = json.dumps([str(event.get('name', '')), args.get('grid'), args.get('block'), args.get('registers per thread'),
                              args.get('shared memory'), args.get('dynamic shared memory')])
            durations.setdefault(key, []).append(number(event.get('dur', 0)))
    out: dict[str, dict[str, float]] = {}
    for key, ds in durations.items():
        out[key] = {'launches': len(ds), 'total_us': sum(ds), 'mean_us': statistics.fmean(ds), 'median_us': statistics.median(ds)}
    buckets = {key: tb.bucket(str(json.loads(key)[0]), r) for key in out}
    return {key: {**v, 'bucket_is_routed_moe': float(buckets[key] == 'routed_moe')} for key, v in out.items()}


def diff(a: dict[str, dict[str, float]], b: dict[str, dict[str, float]], top: int = 25) -> dict[str, Json]:
    """a = fast (or H), b = slow (or L). Launch shares are per capture (captures differ in length)."""
    na, nb = sum(v['launches'] for v in a.values()) or 1, sum(v['launches'] for v in b.values()) or 1
    ta, tb_ = sum(v['total_us'] for v in a.values()) or 1, sum(v['total_us'] for v in b.values()) or 1
    only_a = sorted(set(a) - set(b), key=lambda k: -a[k]['total_us'])
    only_b = sorted(set(b) - set(a), key=lambda k: -b[k]['total_us'])
    changed: list[dict[str, Json]] = []
    for key in set(a) & set(b):
        share = (b[key]['launches'] / nb) / (a[key]['launches'] / na)
        per = b[key]['mean_us'] / a[key]['mean_us'] if a[key]['mean_us'] else None
        if abs(share - 1) > CHANGE or per is None or abs(per - 1) > CHANGE:
            changed.append({'signature': cast(Json, json.loads(key)), 'routed_moe': bool(a[key]['bucket_is_routed_moe']),
                            'launch_share_ratio': share, 'mean_us_ratio': per,
                            'mean_us': [a[key]['mean_us'], b[key]['mean_us']],
                            'kernel_time_share': [a[key]['total_us'] / ta, b[key]['total_us'] / tb_],
                            'weight': abs(b[key]['total_us'] / tb_ - a[key]['total_us'] / ta)})
    changed.sort(key=lambda x: -number(x['weight']))
    return {'only_first': [cast(Json, json.loads(k)) for k in only_a[:top]],
            'only_second': [cast(Json, json.loads(k)) for k in only_b[:top]],
            'only_first_count': len(only_a), 'only_second_count': len(only_b), 'changed': cast(Json, changed[:top]),
            'changed_count': len(changed), 'signatures': [len(a), len(b)]}


def _verified(root: Path, row: dict[str, Json]) -> Path:
    path = root / str(row['file'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != row.get('sha256'):
        raise ValueError(f'{row["file"]}: sha256 differs from the window manifest')
    return path


# ------------------------------------------------------------------ the window

def reference(args_ref: Path | None, args_ttft: float | None) -> tuple[float, str]:
    if args_ttft is not None:
        return args_ttft, 'given'
    if args_ref is None:
        return DEFAULT_REF_S, 'default (PF1/SP1 MEASURED 68.8-71.2 s, pre-FW1)'
    cell = record(cast(Json, json.loads((args_ref / 'cells' / 'ctl-150k-b1.json').read_text())))
    streams = [record(x) for x in cast(list[Json], cell['streams'])]
    if cell.get('cell') != 'ctl-150k-b1' or len(streams) != 1:
        raise ValueError('reference is not an SP1 unprofiled 150K control cell')
    return number(streams[0]['ttft_s']), f'{args_ref.name}/cells/ctl-150k-b1.json'


def analyze(root: Path, ref: float, ref_source: str) -> dict[str, Json]:
    summary = record(cast(Json, json.loads((root / 'summary.json').read_text())))
    starts = record(summary['starts'])
    tele = {h: telemetry(root / f'telemetry-{h}.jsonl') for h in HOSTS}
    rows: dict[str, Json] = {}
    classes: dict[str, tuple[str, str]] = {}
    decode: dict[str, list[float]] = {'H': [], 'L': []}
    for name, raw in starts.items():
        start = record(raw)
        role = str(start['role'])
        if start.get('status') != 'COMPLETE':
            rows[name] = {'role': role, 'status': start.get('status')}
            classes[name] = (role, 'incomplete')
            continue
        procs = {h: procmem(root / f'procmem-{name}-{h}.jsonl') for h in HOSTS}
        cells: dict[str, Json] = {}
        found: list[str] = []
        for cell in (*CLASSIFIED_CELLS, 'p150'):
            data = record(cast(Json, json.loads((root / 'cells' / f'{name}-{cell}.json').read_text())))
            stream = record(cast(list[Json], data['streams'])[0])
            ttft, total = number(stream['ttft_s']), number(stream['total_s'])
            tokens = number(record(stream['usage'])['completion_tokens'])
            submitted = number(data['origin_wall']) + number(stream['submitted_rel_s'])
            row: dict[str, Json] = {'ttft_s': ttft, 'ratio': ttft / ref,
                                    'decode_tok_s': (tokens - 1) / (total - ttft) if total > ttft and tokens > 1 else None,
                                    'memory': {h: around(tele[h], submitted - PRE_S, submitted + ttft) for h in HOSTS},
                                    'procmem': {h: procmem_delta(procs[h], submitted - PRE_S, submitted + ttft) for h in HOSTS}}
            if cell in CLASSIFIED_CELLS:
                row['class'] = classify(ttft, ref)
                found.append(str(row['class']))
                if row['decode_tok_s'] is not None:
                    decode[role].append(number(row['decode_tok_s']))
            else:
                row['class'] = 'profiled (never classified)'
                row['captures'] = [{k: c.get(k) for k in ('kind', 'status')} for c in map(record, cast(list[Json], data['captures']))]
            cells[cell] = row
        flags = [f'{h}:{role_name}' for h in HOSTS for t, roles in procs[h] for role_name, v in roles.items()
                 if v.get('rollup_ms_max', 0) > ROLLUP_FLAG_MS]
        classes[name] = (role, start_class(found))
        rows[name] = {'role': role, 'position': start.get('position'), 'kv_bytes': start.get('kv_bytes'),
                      'class': classes[name][1], 'cells': cells, 'rollup_over_50ms': cast(Json, sorted(set(flags))),
                      'needle': record(start.get('cells') or {}).get('needle-480k')}
    result: dict[str, Json] = {'reference_ttft_s': ref, 'reference_source': ref_source, 'starts': rows,
                               'reading': reading(classes)}
    h, low = decode['H'], decode['L']
    if h and low:
        change = statistics.median(low) / statistics.median(h) - 1
        result['decode'] = {'median_tok_s': {'H': statistics.median(h), 'L': statistics.median(low)}, 'l_vs_h': change,
                            'flag': change < -DECODE_FLAG}
    result['dispatch'] = dispatch(root, rows)
    return result


def dispatch(root: Path, rows: dict[str, Json]) -> dict[str, Json]:
    manifest_path = root / 'traces-manifest.json'
    if not manifest_path.exists():
        return {'status': 'NO_TRACES'}
    manifest = [record(x) for x in cast(list[Json], json.loads(manifest_path.read_text()))]
    r = tb.rules()
    captures: dict[str, dict[int, Path]] = {}
    for row in manifest:
        captures.setdefault(str(row['cell']).rsplit('-p150', 1)[0], {})[int(number(row['rank']))] = _verified(root, row)
    budgets: dict[str, Json] = {}
    tables: dict[tuple[str, int], dict[str, dict[str, float]]] = {}
    for start, files in sorted(captures.items()):
        budget = tb.capture_budget({rank: tb.load(path) for rank, path in files.items()}, 'prefill-late', r)
        budgets[start] = {k: budget.get(k) for k in ('share', 'ms', 'period_ms_median', 'idle_share', 'coverage_min',
                                                     'iterations_common', 'aligned')}
        for rank, path in files.items():
            tables[(start, rank)] = signatures(path)
    state = {name: str(record(v).get('class')) for name, v in rows.items()}
    roles = {name: str(record(v).get('role')) for name, v in rows.items()}
    fast = [s for s in captures if state.get(s) == 'fast']
    slow = [s for s in captures if state.get(s) == 'slow']
    if fast and slow:
        pair, label = (fast[0], slow[0]), 'fast vs slow'
    else:
        hs = [s for s in captures if roles.get(s) == 'H']
        ls = [s for s in captures if roles.get(s) == 'L']
        if not (hs and ls):
            return {'budgets': budgets, 'diff': {'status': 'NO_PAIR'}}
        pair, label = (hs[0], ls[0]), 'H vs L (no state contrast)'
    diffs = {f'rank{rank}': diff(tables[(pair[0], rank)], tables[(pair[1], rank)])
             for rank in (0, 1) if (pair[0], rank) in tables and (pair[1], rank) in tables}
    return {'budgets': budgets, 'diff': {'label': label, 'first': pair[0], 'second': pair[1], 'ranks': cast(Json, diffs)}}


def text(result: dict[str, Json]) -> str:
    read = record(result['reading'])
    lines = [f"PF2-LOWKV reading: {read['verdict']} - {read['meaning']}",
             f"reference 150K TTFT {number(result['reference_ttft_s']):.2f} s ({result['reference_source']}); "
             f"fast <= {FAST_MAX:.2f}x, slow >= {SLOW_MIN:.2f}x"]
    for name, raw in record(result['starts']).items():
        row = record(raw)
        cells = record(row.get('cells') or {})
        ttfts = ', '.join(f"{c} {number(record(v)['ttft_s']):.1f} s ({record(v)['class']})" for c, v in cells.items())
        lines.append(f"  {name} pos {row.get('position')} {row.get('role')}: {row.get('class')}  {ttfts}")
    if 'decode' in result:
        d = record(result['decode'])
        lines.append(f"decode L vs H {number(d['l_vs_h']):+.1%}{' FLAG (> 2 % slower)' if d['flag'] else ''}")
    lines.append(f"dispatch: {record(record(result['dispatch']).get('diff') or {}).get('label', record(result['dispatch']).get('status'))}")
    return '\n'.join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--window', type=Path, required=True)
    group = ap.add_mutually_exclusive_group()
    group.add_argument('--reference', type=Path, help='the post-FW1 SP1 rerun window (unprofiled ctl-150k-b1)')
    group.add_argument('--reference-ttft', type=float)
    ap.add_argument('--output', type=Path, help='default <window>/pf2-analysis.json (never overwritten)')
    args = ap.parse_args(argv)
    ref, source = reference(args.reference, args.reference_ttft)
    result = analyze(args.window, ref, source)
    output = args.output or args.window / 'pf2-analysis.json'
    with output.open('x') as handle:
        handle.write(json.dumps(result, indent=1, allow_nan=False) + '\n')
    print(text(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
