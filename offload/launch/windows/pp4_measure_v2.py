"""PP4 measurement on WIN2 (R-PP4; NDM 2026-10-27 "Fairer win rule is essential. ... Do better deeper testing and measuring").

Two registered modes, each one OFF/ON/OFF HOLD on the QUALIFIED PP4 runtime (worker tree, verifier overlay, remote_dspark):
- controlled: s1-A / s2-B / s3-A. Fixed5 greedy drafts against fixed5 sampled drafts (Round5/6's comparison, re-decided).
- production-policy: s1-P / s2-B / s3-P. Production's own policy (VLLM_DSV41_ADAPTIVE_K=1 with the AK_* defaults and the
  per-batch schedule [[1,2,5],[3,4,4]]) with greedy drafts, against the PP4 promotion configuration (fixed5 sampled).
  P runs on the PP4 worker and verifier overlay: production POLICY on the PP4 runtime, not the production binary.

Measurement: decision cells short-b1 and 32k-b1 (32k-b4 is dropped so three repeats fit the 85-minute HOLD). Each start
runs six request waves per cell: three repeats of two waves, one per public source family (seed 42+wave, matched across
starts; fresh cache salt per stream). A repeat pools its two waves' server counters exactly as pp4_metrics.pool does.

Decision (decision_v2 rule win2-v1, paired by repeat): control = mean of the two bracketing starts, candidate = s2-B.
Series per decision cell, primary first: pooled decode rate (rounds + accepted) / decode seconds (tokens per server decode
second, = (1 + accepted per round) / ms per round), then the WIN2 guards accepted/round and client decode tok/s (LOSS
only). WIN2.md prefers ms/round as the primary for decode cells because it is the steadiest across days; PP4 is the one
change that deliberately SPENDS ms per round (+0.31 ms MEASURED) to buy accepted tokens (+6.9 % MEASURED offline), and
the production-policy mode compares adaptive depths 2..5 with fixed 5, so ms/round alone can never show PP4's effect and
would read its cost as a loss. ms/round therefore decides through the rate (it is its denominator) and is reported with
its own win2-v1 verdict; `checklist_literal` records what ms/round-primary would have said. TTFT is reported only.
Status INCONCLUSIVE overrides any verdict when the GPU clock cap differs between starts or nodes, or the thermal
watchdog dialled the clock back during the HOLD.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import json
import math
import os
import re
import statistics
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

from offload.gb10_budget import window_cells as wc
from offload.gb10_budget.capture import load_bench
from offload.launch.windows import decision_v2 as d
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import pp4_budget as budget
from offload.launch.windows import pp4_cells as cells
from offload.launch.windows import pp4_metrics as metric
from offload.launch.windows import pp4_prerequisites as prereq
from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, number, record

REGISTRATION = 'pp4-win2-r3x2-v1'
REPEATS, WAVES_PER_REPEAT = 3, 2
WAVES = REPEATS * WAVES_PER_REPEAT
CELLS = ('short-b1', '32k-b1')
GROUPS: dict[str, tuple[str, ...]] = {'single-user': CELLS}
# Six distinct frozen cuts per family and cell. Wave w uses family FAMILIES[w % 2]; short-1 and short appear twice, once per
# family, so every request text differs. 32k-12/13 are frozen cuts no other PP4 cell uses.
ORDERS = {'short-b1': ('short', 'short-1', 'short-2', 'short-3', 'short-1', 'short'),
          '32k-b1': ('32k-0', '32k-1', '32k-2', '32k-3', '32k-12', '32k-13')}
MODES: dict[str, tuple[tuple[str, str], ...]] = {
    'controlled': (('s1-A', 'A'), ('s2-B', 'B'), ('s3-A', 'A')),
    'production-policy': (('s1-P', 'P'), ('s2-B', 'B'), ('s3-P', 'P'))}
# Starts that run the six K1-reference greedy checks (registration pp4-r14-near-tie-1p1) before their measurement baseline.
GREEDY: dict[str, tuple[str, ...]] = {'controlled': ('s2-B',), 'production-policy': ('s1-P', 's2-B')}
COMPARISON = {
    'controlled': 'CONTROLLED: fixed5 greedy drafts (A) vs fixed5 sampled drafts (B), both on the PP4 runtime',
    'production-policy': 'PRODUCTION POLICY: adaptive [[1,2,5],[3,4,4]] greedy drafts (P) vs the PP4 promotion configuration '
                         'fixed5 sampled (B), both on the PP4 runtime (PP4 worker + verifier overlay); not the production binary'}
ARCHIVE_CELL = {'controlled': 'pp4rt-fixed5-greedy', 'production-policy': 'pp4rt-production-policy'}
PRIMARY = 'pooled_rate_tps'
GUARDS: tuple[tuple[str, bool], ...] = (('accepted_per_round', True), ('decode_tps', True))
REPORTED: tuple[tuple[str, bool], ...] = (('ms_per_round', False), ('ttft_s', False))
MAX_TOKENS = 512
DEFAULT_CAP_MHZ = 2150  # dsv41.env: GPU_CLOCK_CAP=${GPU_CLOCK_CAP-2150} (production and experiment checkouts alike)
ARCHIVE = pf.CAMPAIGN / 'control-archive.jsonl'
THERMAL_LOG = pf.CAMPAIGN / 'scratch/THERMAL.log'
CLOCK_LINE = re.compile(r'^(\S+) clock cap ([0-9]+) MHz$', re.MULTILINE)
DIAL_BACK = re.compile(r'^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) DIAL BACK to ([0-9]+)', re.MULTILINE)
# Carried qualification: a measurement at HEAD may reuse the admission receipt of the qualified commit only when every
# file outside this measurement layer is byte-identical (worker, verifier preparation, remote_dspark, admission, pair,
# scoped ownership, HOLD, restore, launch scripts and PP4-sampled.env all included).
CARRIED_FILES = frozenset({
    'offload/launch/windows/pp4_window.py', 'offload/launch/windows/pp4_measure_v2.py',
    'offload/launch/windows/pp4_rng_gate.py', 'offload/launch/windows/PP4-production-policy.env',
    'offload/launch/windows/PP4-WIN2.md', 'offload/launch/windows/PP4-PROMOTE-GATE-V2.md',
    'offload/launch/windows/pp4_v2_stage.sh',
    # WIN2's shared measurement layer (offload/win2, reviewed there; S92): rule, chain, archive history, opt-in greedy KL
    # and cells. PP4 runs none of it on a Spark or x86host; window_cells' change is the opt-in late-arrival field only and
    # nbw_cells gained defaulted parameters (NBW's own defaults unchanged).
    'offload/launch/windows/decision_v2.py', 'offload/launch/windows/chain.py', 'offload/launch/windows/chain_prev_ok.sh',
    'offload/launch/windows/WIN2.md', 'offload/launch/windows/greedy_kl.py', 'offload/launch/windows/win2_history.py',
    'offload/launch/windows/nbw_cells.py', 'offload/gb10_budget/window_cells.py',
    'offload/gb10_budget/test_window_cells_win2.py', 'offload/launch/windows/chain_abort.sh',
    # offload-x86host: the test-suite git-variable guard, and OPS1's production maintenance script (no PP4 path runs it).
    'offload/conftest.py', 'offload/launch/maintenance/ops1_apply.sh'})
CARRIED_PREFIXES = ('documents/', 'offload/launch/windows/tests/', 'offload/audit/verifier_rng/')
# pp4_window.py may change only in its measurement/chain layer: every unit that builds or checks a remote mutation or the
# qualified runtime must be AST-identical to the qualified commit's.
RUNNER = 'offload/launch/windows/pp4_window.py'
CARRIED_UNITS = ('Worker', 'host_check', 'worker_script', 'native_probe_script', 'native_worker', 'round_summary',
                 'Live.local', 'Live.recovery', 'Live.owned', 'Live.fence_phase', 'Live.pair_close', 'Live.restore',
                 'Live.verify_overlay', 'Live.probe_worker', 'Live.begin_measurement', 'Live.measured_stats', 'Live.finish')


def roles(mode: str) -> tuple[tuple[str, str], ...]:
    ck.require(mode in MODES, 'unknown PP4 v2 measurement mode')
    return MODES[mode]


def wave_prompts(name: str, wave: int) -> tuple[str, ...]:
    ck.require(name in ORDERS and 0 <= wave < WAVES, 'unknown v2 cell/wave')
    return (ORDERS[name][wave],)


def wave_seconds(name: str) -> float:
    """Registered per-wave allowance (pp4 round6 budget), validated by pp4_budget.estimate."""
    budget.estimate()
    return number(record(record(ck.read(budget.PATH)['cells'])[name])['allowance_per_group_s'])


def repeated_cell(api: wc.Api, name: str, cuts: dict[str, Json], data_dir: Path, plan: wc.Plan,
                  output: Path) -> dict[str, Json]:
    """Six independently salted B1 waves; every sufficient-statistic row is saved as soon as it exists."""
    ck.require(name in CELLS, 'v2 measures short-b1 and 32k-b1 only')
    bench = cast(wc.Counters, load_bench())
    rows: list[Json] = []
    for wave in range(WAVES):
        family = metric.FAMILIES[wave % len(metric.FAMILIES)]
        materials, hashes = cells.family_corpus(data_dir, family)
        texts = cells.texts_from(record(cuts[family]), materials, hashes)
        cell = wc.Cell(f'{name}-v2wave{wave}', wave_prompts(name, wave), max_tokens=MAX_TOKENS)
        before = bench.parse_metrics(api.metrics())
        seed = 42 + wave
        subplan = wc.Plan(plan.deadline, plan.floor_kb, plan.telemetry, f'{plan.salt_prefix}-v2wave{wave}', plan.ledger)
        result = wc.run_cell(cells.SeededApi(api, seed), cell, texts, subplan)
        for stream in ck.rows(result['streams']):
            record(stream['sampling']).update({'seed': seed, 'ignore_eos': True})
        time.sleep(2)  # metrics publication, excluded from request timings
        after = bench.parse_metrics(api.metrics())
        result['spec'] = bench.spec_delta(before, after)
        completed = cells.metric_delta(before, after, 'vllm:request_decode_time_seconds_count')
        ck.require(completed == cell.batch, 'server completed-request count mismatch')
        rows.append({**cells.summarize(result, before, after), 'completed_requests': completed, 'wave': wave,
                     'repeat': wave // WAVES_PER_REPEAT, 'family': family})
        output.write_text(json.dumps({'status': 'PARTIAL', 'metric_registration': REGISTRATION,
                                      'cell': name, 'waves': rows}, allow_nan=False) + '\n')
    observations = ck.rows(cast(Json, rows))
    streams = [stream for row in observations for stream in ck.rows(row['streams'])]
    return {'cell': name, 'batch': 1, 'waves': rows, 'metric_registration': REGISTRATION,
            'repeats': cast(Json, repeat_values(observations)), **metric.pool(observations),
            'independent_requests': len(streams),
            'median_decode_tps': statistics.median(number(s['decode_tps']) for s in streams),
            'median_ttft_s': statistics.median(number(s['ttft_s']) for s in streams),
            'streams': cast(Json, streams), 'status': 'MEASURED three repeats of two independent B1 waves'}


def _positive(value: Json, what: str) -> float:
    result = number(value)
    ck.require(math.isfinite(result) and result > 0, f'invalid {what}')
    return result


def waves(cell: Mapping[str, Json]) -> list[dict[str, Json]]:
    """The registered six-wave evidence of one cell in one start; refuses anything else."""
    ck.require(cell.get('metric_registration') == REGISTRATION and cell.get('batch') == 1 and cell.get('cell') in CELLS,
               'v2 metric registration/batch/cell differs')
    rows = ck.rows(cell['waves'])
    ck.require([r.get('wave') for r in rows] == list(range(WAVES)), 'six registered waves required in every cell per start')
    hashes: list[str] = []
    salts: list[str] = []
    for i, row in enumerate(rows):
        ck.require(row.get('family') == metric.FAMILIES[i % 2] and row.get('batch') == 1
                   and row.get('repeat') == i // WAVES_PER_REPEAT, 'frozen family/repeat order differs')
        streams = ck.rows(row['streams'])
        ck.require(len(streams) == 1 and row.get('completed_requests') == 1, 'completed request count differs')
        for stream in streams:
            ck.require(stream.get('sampling') == {'temperature': 1, 'top_p': 1, 'seed': 42 + i, 'priority': 10,
                                                  'max_tokens': MAX_TOKENS, 'ignore_eos': True}, 'registered512 sampling differs')
            ck.require(record(stream['usage']).get('completion_tokens') == 512 and stream.get('finish') == 'length',
                       'exact512 output/length finish required')
            salts.append(str(stream['cache_salt_sha256']))
            hashes.append(str(stream['prompt_sha256']))
        metric.pool([row])
        _positive(row['median_ttft_s'], 'wave TTFT')
        _positive(row['median_decode_tps'], 'wave client decode rate')
    ck.require(len(set(hashes)) == len(hashes), 'every wave needs a distinct frozen prompt')
    ck.require(len(set(salts)) == len(salts), 'duplicate stream salt within a cell')
    return rows


def repeat_values(rows: Sequence[dict[str, Json]]) -> list[dict[str, Json]]:
    """Per repeat: pooled counters of its two waves (never resampled rounds) plus mean TTFT / client decode rate."""
    ck.require(len(rows) == WAVES, 'six waves required')
    out: list[dict[str, Json]] = []
    for r in range(REPEATS):
        part = list(rows[r * WAVES_PER_REPEAT:(r + 1) * WAVES_PER_REPEAT])
        ck.require([row.get('wave') for row in part] == [r * WAVES_PER_REPEAT + k for k in range(WAVES_PER_REPEAT)],
                   'repeat waves out of order')
        pooled = metric.pool(part)
        out.append({'repeat': r, 'waves': [row['wave'] for row in part],
                    'families': [row['family'] for row in part],
                    PRIMARY: float(metric.rate(part)),
                    'accepted_per_round': pooled['acceptance_per_round'],
                    'ms_per_round': pooled['server_decode_ms_per_spec_request_round'],
                    'ttft_s': statistics.fmean(number(row['median_ttft_s']) for row in part),
                    'decode_tps': statistics.fmean(number(row['median_decode_tps']) for row in part),
                    'spec_request_rounds': pooled['spec_request_rounds'], 'accepted_tokens': pooled['accepted_tokens'],
                    'server_decode_seconds': pooled['server_decode_seconds']})
    return out


def caps_from_start(text: str) -> dict[str, Json]:
    """start_exp.sh prints '<node> clock cap N MHz' for each node when GPU_CLOCK_CAP is set; none means stock clocks."""
    caps: dict[str, Json] = {}
    for host, mhz in CLOCK_LINE.findall(text):
        ck.require(caps.get(host, int(mhz)) == int(mhz), 'one start applied two clock caps to a node')
        caps[host] = int(mhz)
    return caps


def dial_backs(log: Path, start: float, end: float) -> list[Json]:
    """THERMAL-WATCH dial-backs (it applies -lgc live with --now, even inside a HOLD) between start and end."""
    if not log.exists():
        return []
    found: list[Json] = []
    for stamp, mhz in DIAL_BACK.findall(log.read_text(errors='replace')):
        moment = datetime.datetime.strptime(stamp, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()
        if start <= moment <= end:
            found.append({'utc': stamp, 'mhz': int(mhz)})
    return found


def production_cap(env_lines: Sequence[str]) -> int | None:
    """Production's persistent cap from each node's /etc/environment GPU_CLOCK_CAP lines (CLOCK-PERSIST.sh): absent ->
    dsv41.env's default; empty -> stock clocks (None). Both nodes must agree."""
    values: set[int | None] = set()
    for text in env_lines:
        found = re.findall(r'^GPU_CLOCK_CAP=([0-9]*)$', text, re.MULTILINE)
        ck.require(len(found) <= 1, 'duplicate GPU_CLOCK_CAP lines in /etc/environment')
        values.add(DEFAULT_CAP_MHZ if not found else int(found[0]) if found[0] else None)
    ck.require(len(values) == 1, f'Spark nodes disagree on the persistent clock cap: {sorted(map(str, values))}')
    return values.pop()


def clock_guard(caps: Mapping[str, Json], names: Sequence[str], dialled: Sequence[Json],
                production: Mapping[str, Json] | None = None) -> dict[str, Json]:
    """Every start at one cap on both nodes, no thermal dial-back, and (when given) production's own configured cap, so
    the control is a same-clock control (WIN2-R2: comparisons and archive spreads never mix clocks)."""
    per_arm: dict[str, Json] = {name: dict(record(caps[name])) if name in caps else None for name in names}
    recorded = [record(v) for v in per_arm.values() if v is not None]
    same = len(recorded) == len(names) and all(v == recorded[0] for v in recorded)
    applied = {int(number(v)) for v in recorded[0].values()} if same else set()
    single = same and len(applied) <= 1
    expected = None if production is None else production.get('mhz')
    at_production = production is None or (applied == ({int(number(expected))} if expected is not None else set()))
    consistent = same and single and at_production and not dialled
    reason = (None if consistent else 'thermal watchdog dialled the clock back during the HOLD' if dialled else
              'GPU clock cap differs between starts or nodes, or was not recorded' if not single else
              f'starts ran at {sorted(applied)} MHz, not production\'s configured {expected} MHz')
    return {'consistent': consistent, 'per_start': per_arm, 'thermal_dial_backs': list(dialled),
            'production_cap_mhz': expected, 'reason': reason}


def start_cap(clock: Mapping[str, Json], start: str) -> int:
    """WIN2-R2 archive `clock_cap_mhz` of one start: the cap both Sparks printed in that start's own start log."""
    caps = record(record(clock.get('per_start') or {}).get(start) or {})
    values = {int(number(v)) for v in caps.values()}
    ck.require(len(caps) >= 2 and len(values) == 1, f'{start}: both Sparks must show one clock cap in the start log, got {caps}')
    return values.pop()


def caps_list(clock: Mapping[str, Json]) -> list[Json]:
    values = sorted({int(number(v)) for row in record(clock.get('per_start') or {}).values() if row is not None
                     for v in record(row).values()})
    return [cast(Json, v) for v in values]


def _check_greedy(block: Mapping[str, Json], name: str) -> None:
    greedy = record(block['greedy'])
    checks = ck.rows(greedy['checks'])
    ck.require(greedy.get('greedy_registration', cells.GREEDY_REGISTRATION) == cells.GREEDY_REGISTRATION
               and [r.get('i') for r in checks] == list(range(6))
               and all(r.get('identical') is True or r.get('near_tie') is True for r in checks),
               f'{name}: six greedy identical/qualified near-tie checks required')


def _spread(a: float, b: float) -> float:
    return abs(a - b) / ((a + b) / 2)


def _pair(reps: Sequence[Sequence[dict[str, Json]]], key: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    first, candidate, last = reps
    control = tuple((number(a[key]) + number(c[key])) / 2 for a, c in zip(first, last))
    return control, tuple(number(b[key]) for b in candidate)


def decide(starts: Mapping[str, Json], mode: str, *, clock: Mapping[str, Json], rng: Mapping[str, Json] | None,
           archive: Path | None = ARCHIVE, production_commit: str | None = None) -> dict[str, Json]:
    """win2-v1 over three paired repeats; pure arithmetic over recorded evidence (the archive is read, never written)."""
    names = [name for name, _ in roles(mode)]
    base: dict[str, Json] = {'rule': d.RULE, 'metric_registration': REGISTRATION, 'mode': mode,
                             'comparison': COMPARISON[mode], 'decision_cells': list(CELLS), 'primary_metric': PRIMARY,
                             'guards': [k for k, _ in GUARDS], 'max_tokens': MAX_TOKENS,
                             'pairing': 'repeat r: control = mean(first start, last start), candidate = s2-B',
                             'promotion': 'none; separate independent review required'}
    if list(starts) != names:
        return {**base, 'status': 'INCONCLUSIVE', 'reason': 'one complete OFF/ON/OFF HOLD required'}
    blocks = [record(starts[name]) for name in names]
    for (name, role), block in zip(roles(mode), blocks):
        ck.require(block.get('role') == role and set(record(block['cells'])) == set(CELLS), 'registered arm/cell set differs')
        if block.get('status') != 'COMPLETE':
            return {**base, 'status': 'INCONCLUSIVE', 'reason': 'incomplete start'}
        if name in GREEDY[mode]:
            _check_greedy(block, name)
    if rng is None or rng.get('status') != 'PASS':
        raise ValueError('#61161 verifier-RNG gate receipt required')
    series: list[d.Series] = []
    literal: list[d.Series] = []
    all_reps: dict[str, list[list[dict[str, Json]]]] = {}
    reported: dict[str, Json] = {}
    starts_pooled: dict[str, Json] = {}
    off_off: dict[str, Json] = {}
    spreads: dict[str, Json] = {}
    for cell in CELLS:
        observations = [waves(record(record(block['cells'])[cell])) for block in blocks]
        signatures = [[(r['family'], [s['prompt_sha256'] for s in ck.rows(r['streams'])]) for r in rows]
                      for rows in observations]
        ck.require(all(sig == signatures[0] for sig in signatures), 'OFF/ON/OFF matched inputs/families differ')
        salts = [str(s['cache_salt_sha256']) for rows in observations for r in rows for s in ck.rows(r['streams'])]
        ck.require(len(set(salts)) == len(salts), 'fresh cold salts required across every start/wave/stream')
        reps = [repeat_values(rows) for rows in observations]
        all_reps[cell] = reps
        series.append(d.Series(cell, PRIMARY, True, *_pair(reps, PRIMARY)))
        series.extend(d.Series(cell, key, higher, *_pair(reps, key)) for key, higher in GUARDS)
        literal.append(d.Series(cell, 'ms_per_round', False, *_pair(reps, 'ms_per_round')))
        literal.extend(d.Series(cell, key, higher, *_pair(reps, key)) for key, higher in GUARDS)
        reported[cell] = {key: {**d.decide_cell(d.Series(cell, key, higher, *_pair(reps, key))),
                                'role': 'reported only; never a guard'} for key, higher in REPORTED}
        pooled = {name: {**metric.pool(rows), PRIMARY: float(metric.rate(rows))} for name, rows in zip(names, observations)}
        starts_pooled[cell] = cast(Json, pooled)
        off_off[cell] = {'pooled_off_off_spread': _spread(number(pooled[names[0]][PRIMARY]), number(pooled[names[2]][PRIMARY])),
                         'per_repeat': [_spread(number(a[PRIMARY]), number(c[PRIMARY])) for a, c in zip(reps[0], reps[2])]}
        # Same definition only: max_tokens, production commit and clock caps (WIN2-R2 refuses mixed commits/clocks).
        # WIN2-R2: spreads never pool different max_tokens, production commits or clock caps.
        caps = caps_list(clock)
        where: dict[str, Json] = {'max_tokens': MAX_TOKENS, **({'clock_cap_mhz': caps[0]} if len(caps) == 1 else {}),
                                  **({'commit': production_commit} if production_commit else {})}
        day = (d.archive_spread(archive, f'{ARCHIVE_CELL[mode]}-{cell}', PRIMARY, where=where)
               if archive is not None and len(caps) == 1 else None)
        spreads[cell] = {'archive_day_to_day_spread': day,
                         'note': None if day is None or day <= 0.02 else
                         f'archive day-to-day spread {day:.4f} exceeds the 2 % required effect; the effect is NOT raised'}
    v2 = d.decide_window(series, CELLS, groups=GROUPS)
    checklist = d.decide_window(literal, CELLS, groups=GROUPS)
    # decide_window's own fields stay top-level so a consumer (pp4_promote measurement_gate) recomputes them exactly
    # from decision_series; a clock-guard override therefore shows as a status that does not recompute.
    result: dict[str, Json] = {**base, **v2, 'reason': None, 'reported_metrics': reported,
        'checklist_literal': {'status': checklist['status'], 'primary_metric': 'ms_per_round',
                              'cells': checklist['cells'], 'role': 'record only; does not decide (see module doc)'},
        'pooled_by_start': starts_pooled, 'off_off_spread': off_off, 'archive': spreads, 'clock': dict(clock),
        'rng_gate': dict(rng), 'decision_series': [_row(s) for s in series],
        'reported_series': [_row(d.Series(c, k, h, *_pair(all_reps[c], k))) for c in CELLS for k, h in REPORTED]}
    if clock.get('consistent') is not True:
        result.update({'win2_status': v2['status'], 'status': 'INCONCLUSIVE', 'reason': f"clock guard: {clock.get('reason')}"})
    return result


def _row(s: d.Series) -> dict[str, Json]:
    return {'cell': s.cell, 'metric': s.metric, 'higher_is_better': s.higher_is_better, 'control': list(s.control),
            'candidate': list(s.candidate), 'paired': s.paired}


def budget_v2(mode: str, *, pair: dict[str, Json] | None = None) -> dict[str, Json]:
    """ESTIMATED HOLD from the registered round6 inputs: six waves per decision cell per start, 32k-b4 omitted."""
    names = [name for name, _ in roles(mode)]
    registered = budget.estimate(pair=pair)  # validates the registered file, its prices and the pair reserve
    data = ck.read(budget.PATH)
    per_start = sum(wave_seconds(cell) * WAVES for cell in CELLS)
    common = (number(data['worker_startup_s']) + max(number(x) for x in cast(list[Json], data['startup_observations_s']))
              + number(data['warm_per_start_s']) + number(data['transition_per_start_s']))
    greedy = number(data['greedy_on_s'])
    blocks: dict[str, Json] = {}
    for i, name in enumerate(names):
        blocks[name] = (common + per_start + (number(data['cuts_s']) if i == 0 else 0)
                        + (greedy if name in GREEDY[mode] else 0))
    pair_s = number(record(registered['pair'])['seconds'])
    measurement_end = number(data['setup_s']) + pair_s + sum(number(v) for v in blocks.values())
    total = measurement_end + number(data['restore_s'])
    return {'classification': 'ESTIMATED from the registered round6 inputs; not measured v2 blocks',
            'registration': REGISTRATION, 'mode': mode, 'source_sha256': pf.digest(budget.PATH),
            'pair': registered['pair'], 'block_budget_s': blocks, 'cells': list(CELLS), 'omitted_cells': ['32k-b4'],
            'waves_per_cell_per_start': WAVES, 'headline_per_start_s': per_start, 'measurement_end_s': measurement_end,
            'measurement_end_minutes': measurement_end / 60, 'total_s': total, 'total_minutes': total / 60,
            'fits': total <= budget.MAX_HOLD_S and measurement_end <= budget.ABORT_S,
            'restore_s': data['restore_s'], 'abort_s': budget.ABORT_S, 'restore_deadline_s': budget.MAX_HOLD_S}


def _git(args: list[str]) -> str:
    return subprocess.check_output(['git', *args], cwd=pf.ROOT, text=True)


def units(text: str) -> dict[str, str]:
    """AST dump of every module-level function/class and every Live method (formatting-insensitive)."""
    found: dict[str, str] = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            found[node.name] = ast.dump(node)
            if isinstance(node, ast.ClassDef) and node.name == 'Live':
                found.update({f'Live.{n.name}': ast.dump(n) for n in node.body if isinstance(n, ast.FunctionDef)})
    return found


def carried(path: Path, qualified: str, head: str, mode: str, *,
            git: Callable[[list[str]], str] = _git) -> dict[str, Json]:
    """The qualified commit's QUALIFIED admission receipt, carried to HEAD when only the measurement layer changed."""
    ck.require(all(re.fullmatch('[0-9a-f]{40}', c) is not None for c in (qualified, head)), 'full commits required')
    data = prereq.qualification(path, qualified)  # QUALIFIED at that commit; every primary evidence hash rechecked
    ck.require(git(['merge-base', qualified, head]).strip() == qualified, 'qualified commit is not an ancestor of HEAD')
    changed = sorted(p for p in git(['diff', '--name-only', qualified, head]).split() if p)
    outside = [p for p in changed if p not in CARRIED_FILES and not p.startswith(CARRIED_PREFIXES)]
    ck.require(not outside, f'qualified sources changed since {qualified[:12]}: {outside}; a new admission is required')
    if RUNNER in changed:
        old, new = units(git(['show', f'{qualified}:{RUNNER}'])), units(git(['show', f'{head}:{RUNNER}']))
        drift = [u for u in CARRIED_UNITS if u not in old or old.get(u) != new.get(u)]
        ck.require(not drift, f'runner units that touch remote mutations or the qualified runtime changed: {drift}')
    registered = budget_v2(mode, pair=budget.pair_reserve(path, qualified))
    ck.require(registered['fits'] is True,
               f"v2 ESTIMATED {number(registered['total_minutes']):.2f} min / measurement end "
               f"{number(registered['measurement_end_minutes']):.2f} min exceeds 85 / 73; refuses before host contact")
    data['registered_budget'] = registered
    data['block_budget_s'] = registered['block_budget_s']
    data['carried_qualification'] = {'qualified_commit': qualified, 'head': head, 'changed_files': cast(Json, changed),
                                     'rule': 'every file outside the measurement layer byte-identical to the qualified commit'}
    return data


def archive_rows(starts: Mapping[str, Json], mode: str, *, window: str, production_commit: str,
                 experiment_commit: str, clock: Mapping[str, Json]) -> list[dict[str, Json]]:
    """One row per finished control start, repeat and metric (also after a later stop; the candidate is never archived)."""
    names = [name for name, _ in roles(mode)]
    caps = caps_list(clock)
    date = window.rsplit('-W-', 1)[-1][:8] if '-W-' in window else time.strftime('%Y%m%d', time.gmtime())
    rows: list[dict[str, Json]] = []
    for name in (names[0], names[2]):
        if name not in starts or record(starts[name]).get('status') != 'COMPLETE':
            continue
        block = record(starts[name])
        cap = start_cap(clock, name)
        for cell in CELLS:
            for rep in repeat_values(waves(record(record(block['cells'])[cell]))):
                for key in (PRIMARY, *(k for k, _ in (*GUARDS, *REPORTED))):
                    rows.append({'window': window, 'cell': f'{ARCHIVE_CELL[mode]}-{cell}', 'metric': key,
                                 'value': rep[key], 'max_tokens': MAX_TOKENS, 'commit': production_commit, 'experiment_commit': experiment_commit,
                                 'clock_cap_mhz': cap, 'clock_source': 'start-log',
                                 'clock_caps_mhz': cast(Json, caps), 'date': date, 'source': 'pp4_measure_v2',
                                 'registration': REGISTRATION, 'start': name, 'repeat': rep['repeat']})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--mode', choices=('cell',), default='cell')
    ap.add_argument('--data-dir', type=Path, default=Path('/opt/dsv41/data/bench-data'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cuts', type=Path)
    ap.add_argument('--cell', choices=CELLS)
    ap.add_argument('--deadline-utc', type=float, default=0)
    ap.add_argument('--salt', default='')
    ap.add_argument('--ledger', type=Path)
    args = ap.parse_args(argv)
    if not args.execute:
        print('DRY RUN: PP4 v2 cell, six B1 waves (three repeats x two families), T=1/top_p=1, 512 tokens; no network or file')
        return 0
    ck.require(os.environ.get('WINDOW_COORDINATOR') == 'Deepseek/coordinator', 'coordinator required')
    ck.require(not args.output.exists(), 'fresh cell output required')
    ck.require(time.time() < args.deadline_utc, 'submission deadline reached')
    ck.require(args.cuts is not None and args.cell is not None and bool(args.salt), 'cell/cuts/salt required')
    plan = wc.Plan(args.deadline_utc, pf.MEM_FLOOR_KB, [], args.salt, args.ledger)
    result = repeated_cell(cells.Http(wc.URL), str(args.cell), ck.read(cast(Path, args.cuts)), args.data_dir, plan,
                           args.output)
    args.output.write_text(json.dumps(result, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
