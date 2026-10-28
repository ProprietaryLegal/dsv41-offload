"""UP3 on WIN2 (NDM 2026-10-27 "go window after window"; coordinator R-UP3 scope + 2026-10-27 additions).

Three chained windows, each its candidates bracketed by two controls in ONE HOLD:
- 'ring-depth2': C / dual ring / DEP1 fixed depth 2 / C;
- 'defer-shares': C / 0051 indexer defer alone / dual ring + 0051 / C (separates 0051's share from the ring's, which upstream
  measured only jointly, -1.20 %/round);
- 'k5-batch': C / K5 / C in B3 and B4 only, where [[1,4,5]] raises k from 4 to 5 (at one request both schedules give
  k=5: UP3-k5-W-20261027T063155Z's B1 cells were an A/A test, not a K5 test).
Each candidate start is compared, unpaired (S92 / coordinator 2026-10-27: paired=False for C/X/C), with the repeats of
BOTH bracketing control starts (same public prompts, seeds 42+r, fresh cold salts) and decided by WIN2 rule win2-v1
(decision_v2) in its window's two decision cells. Every request
generates exactly 512 tokens; 32k-b3/32k-b4 here are never comparable with NBW's 1,500-token cells of the same name.
Greedy near-tie limits (NDM 2026-10-27): 0.9+1e-3 nats everywhere except DEP1 fixed2, 1.1+1e-3 ("Depth 2 to 1.1 nat is
fine"); NDM set 1.1 only for PP4 and DEP1 depth 2, so the bit-identical 0051 arms keep 0.9.

Primary metric per arm (WIN2.md: pooled server ms/round for decode cells; accepted/round only as a guard):
- dual ring, 0051 and ring+0051 change only how fast a round runs, so pooled ms/round decides and accepted/round guards;
- fixed2 and K5 change how many draft tokens a round verifies, so ms/round and accepted/round move together BY DESIGN;
  the only fair single number is pooled committed tokens per server decode second, (rounds + accepted) / decode time,
  the (R+A)/D UP3 and DEP1 registered before WIN2. Their ms/round and accepted/round are reported, never decide.
"""
from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from offload.launch.windows import decision_v2 as dv2
from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, number, record
from offload.up3 import cells


@dataclass(frozen=True)
class Plan:
    roles: tuple[str, ...]
    cells: tuple[str, ...]
    decision_cells: tuple[str, ...]
    repeats: Mapping[str, int]
    segments: Mapping[str, tuple[str, ...]]


DECODE_CELLS = ('short-b1', '32k-b3', 'mixed150k-2decode')
DECODE_REPEATS = {'short-b1': 6, '32k-b3': 4, 'mixed150k-2decode': 1}
DECODE_SEGMENTS = {'single-user': ('short-b1',), 'multi-user': ('32k-b3',)}
GROUPS: dict[str, Plan] = {
    'ring-depth2': Plan(('control', 'dual-ring', 'fixed2', 'control'), DECODE_CELLS, ('short-b1', '32k-b3'),
                        DECODE_REPEATS, DECODE_SEGMENTS),
    'defer-shares': Plan(('control', 'defer', 'ring-defer', 'control'), DECODE_CELLS, ('short-b1', '32k-b3'),
                  DECODE_REPEATS, DECODE_SEGMENTS),
    'k5-batch': Plan(('control', 'k5', 'control'), ('32k-b3', '32k-b4'), ('32k-b3', '32k-b4'),
                     {'32k-b3': 5, '32k-b4': 5}, {'multi-user': ('32k-b3', '32k-b4')}),
}
MAX_TOKENS = cells.BUDGET
NEAR_TIE_NATS = {'control': 0.9 + 1e-3, 'dual-ring': 0.9 + 1e-3, 'defer': 0.9 + 1e-3, 'ring-defer': 0.9 + 1e-3,
                 'k5': 0.9 + 1e-3, 'fixed2': 1.1 + 1e-3}
# (metric, higher_is_better); in a decision cell the FIRST decides and the rest are guards (LOSS only).
ROUND_SPEED: tuple[tuple[str, bool], ...] = (('ms_per_round', False), ('accepted_per_round', True))
COMMITTED: tuple[tuple[str, bool], ...] = (('committed_per_s', True),)
DECIDE: dict[str, tuple[tuple[str, bool], ...]] = {
    'dual-ring': ROUND_SPEED, 'defer': ROUND_SPEED, 'ring-defer': ROUND_SPEED, 'fixed2': COMMITTED, 'k5': COMMITTED}
REPORTED: dict[str, tuple[tuple[str, bool], ...]] = {
    **{role: (('committed_per_s', True), ('ttft_s', False)) for role in ('dual-ring', 'defer', 'ring-defer')},
    **{role: (('ms_per_round', False), ('accepted_per_round', True), ('ttft_s', False)) for role in ('fixed2', 'k5')}}
MIXED_METRICS = (('ms_per_round', False), ('accepted_per_round', True), ('committed_per_s', True),
                 ('ttft_s', False), ('p99_inter_token_delivery_gap_ms', False))
ARCHIVE_METRICS = ('ms_per_round', 'accepted_per_round', 'committed_per_s', 'ttft_s')

# MEASURED (UP3-dual-ring-W-20261027T165101Z, production 596ba1c, three starts, checkpoints.jsonl): STARTED -> gates
# PASS 420.2 / 430.6 / 439.5 s (start, greedy, OPS1 warm-up, host evidence); per repeat including the cell's own
# stats/driver overhead, the slowest start: short-b1 50.8 s / 4, 32k-b1 108.9 s / 4, 32k-b3 145.2 s / 2, 32k-b4 89.9 s / 1,
# mixed150k-2decode 342.6 s / 2; finish 2.8 s. ESTIMATED: +30 s for fixed2's near-tie diagnostics (one tokenize pair +
# one top-20 probe per divergent greedy check; DEP1-W-20261027T020407Z diverged on its first three of six checks).
PREPARE_S = 440.0
NEAR_TIE_DIAGNOSTICS_S = 30.0
PER_REPEAT_S = {'short-b1': 50.8 / 4, '32k-b1': 108.9 / 4, '32k-b3': 145.2 / 2, '32k-b4': 89.9,
                'mixed150k-2decode': 342.6 / 2}
FINISH_S = 2.8
MEASURED_SOURCE = 'UP3-dual-ring-W-20261027T165101Z/checkpoints.jsonl'
# Calibration (coordinator 2026-10-27, from UP3-k5-W-20261027T063155Z): at one request K5 and production both run k=5,
# so its 32K-B1 cell was an A/A comparison, yet its paired interval excluded zero: -4.4 % [-8.5, -0.4].
AA_CALIBRATION = {'window': 'UP3-k5-W-20261027T063155Z', 'cell': '32k-b1', 'point': -0.044, 'interval95': [-0.085, -0.004]}


def plan_for(group: str) -> Plan:
    ck.require(group in GROUPS, f'{group}: not a WIN2 UP3 group')
    return GROUPS[group]


def roles(group: str) -> tuple[str, ...]:
    return plan_for(group).roles


def block_s(group: str, role: str) -> int:
    """One start's MEASURED-basis block: prepare + every cell's repeats + finish, rounded up to whole seconds."""
    plan = plan_for(group)
    prepare = PREPARE_S + (NEAR_TIE_DIAGNOSTICS_S if role == 'fixed2' else 0)
    return math.ceil(prepare + sum(PER_REPEAT_S[name] * plan.repeats[name] for name in plan.cells) + FINISH_S)


def repeat_values(cell: Mapping[str, Json], metric: str) -> list[float]:
    """One value per repeat, pooled over that repeat's simultaneous requests, from fields cells.run records:
    Prometheus request decode time (ms), speculative rounds and accepted tokens; client TTFT and delivery-gap p99."""
    rows = [record(r) for r in ck.rows(cell['repetitions'])]
    ck.require(bool(rows), 'cell has no measured repeat')
    if metric in ('ttft_s', 'p99_inter_token_delivery_gap_ms'):
        return [statistics.fmean(number(s[metric]) for s in ck.rows(r['streams'])) for r in rows]
    out: list[float] = []
    for r in rows:
        rounds, accepted, decode_ms = (number(r[k]) for k in ('spec_request_rounds', 'accepted_tokens', 'server_decode_ms'))
        ck.require(all(math.isfinite(v) for v in (rounds, accepted, decode_ms)) and rounds > 0 and decode_ms > 0
                   and accepted >= 0, 'no speculative rounds or decode time in a measured repeat')
        if metric == 'ms_per_round':
            out.append(decode_ms / rounds)
        elif metric == 'accepted_per_round':
            out.append(accepted / rounds)
        else:
            ck.require(metric == 'committed_per_s', f'unknown UP3 WIN2 metric {metric}')
            out.append(1000 * (rounds + accepted) / decode_ms)
    return out


def _arm_cells(starts: Mapping[str, Json], name: str) -> dict[str, Json]:
    return record(record(record(starts[name])['cells']))


def _same_requests(measured: Sequence[Mapping[str, Json]], cell: str, repeats: int) -> None:
    """Pairing is only fair on identical public prompts, budgets, seeds and repeat counts."""
    def identity(c: Mapping[str, Json]) -> list[tuple[str, str, str, str]]:
        return [(str(s['prompt_sha256']), str(s['prompt_tokens']), str(s['budget']), str(record(s['sampling'])['seed']))
                for r in ck.rows(c['repetitions']) for s in ck.rows(record(r)['streams'])]
    first = identity(measured[0])
    ck.require(all(identity(c) == first for c in measured) and len(ck.rows(measured[0]['repetitions'])) == repeats
               and all(int(t[2]) == MAX_TOKENS for t in first), f'{cell}: public prompt, budget, seed or repeats differ')


def clock_caps(starts: Mapping[str, Json], name: str) -> list[int]:
    """Clock ceiling(s) the start reported applying (Live.finish records start_exp.sh's 'clock cap N MHz' lines)."""
    value = record(record(starts[name]).get('checks', {})).get('clock_caps_mhz', [])
    ck.require(isinstance(value, list), f'{name}: clock caps must be a list')
    return [int(number(v)) for v in cast(list[Json], value)]


def candidate_role(name: str) -> str:
    return name.split('-', 1)[1]


def series_for(plan: Plan, starts: Mapping[str, Json], names: Sequence[str],
               candidate: str) -> tuple[list[dv2.Series], dict[str, Json]]:
    """Unpaired series for one candidate start against both bracketing controls' repeats, plus reported components."""
    role = candidate_role(candidate)
    series: list[dv2.Series] = []
    components: dict[str, Json] = {}
    for cell in plan.cells:
        measured = [record(_arm_cells(starts, n)[cell]) for n in (names[0], candidate, names[-1])]
        _same_requests(measured, cell, plan.repeats[cell])
        for metric, higher in DECIDE[role] if cell in plan.decision_cells else MIXED_METRICS:
            before, arm, after = (repeat_values(c, metric) for c in measured)
            series.append(dv2.Series(cell, metric, higher, tuple(before + after), tuple(arm), paired=False))
        if cell in plan.decision_cells:
            rows: dict[str, Json] = {}
            for metric, higher in REPORTED[role]:
                before, arm, after = (repeat_values(c, metric) for c in measured)
                k = statistics.fmean(before + after)
                gains = [((a - k) if higher else (k - a)) / k for a in arm]
                rows[metric] = {'verdict': 'REPORTED', 'point_gain': statistics.fmean(gains),
                                'per_repeat_gain': cast(Json, gains), 'higher_is_better': higher}
            components[cell] = rows
    return series, components


def decide(group: str, starts: Mapping[str, Json], archive: Path, production: str) -> dict[str, Json]:
    """win2-v1 per candidate start, each against the shared bracketing controls of its window."""
    plan = plan_for(group)
    names = [f's{i}-{role}' for i, role in enumerate(plan.roles, 1)]
    if list(starts) != names or any(record(v).get('status') != 'COMPLETE' for v in starts.values()):
        return {'status': 'NOT A CANDIDATE', 'reason': 'all mandatory complete starts required', 'rule': dv2.RULE}
    for name in names:
        row = record(starts[name])
        ck.require(row.get('role') == candidate_role(name), 'start role mismatch')
        ck.require(record(row['greedy']).get('near_tie_nats') == NEAR_TIE_NATS[candidate_role(name)],
                   f'{name}: greedy ran without its registered near-tie limit')
    caps = {name: clock_caps(starts, name) for name in names}
    if any(len(c) != 1 for c in caps.values()) or len({json.dumps(c) for c in caps.values()}) != 1:
        # A THERMAL-WATCH dial-back (2300 -> 2150) between starts would bias every comparison in the window.
        return {'status': 'NOT A CANDIDATE', 'reason': 'clock cap unproved or changed between starts',
                'clock_caps_mhz': cast(Json, caps), 'rule': dv2.RULE}
    window_caps = caps[names[0]]
    arms: dict[str, Json] = {}
    for name in names[1:-1]:
        series, components = series_for(plan, starts, names, name)
        result = dv2.decide_window(series, plan.decision_cells, groups=plan.segments)
        result['components_reported'] = components
        arms[candidate_role(name)] = result
    spreads: dict[str, Json] = {}
    for cell in plan.decision_cells:
        for metric in ARCHIVE_METRICS:
            try:
                # Same cell definition, production commit and clock ceiling only (2150 MHz rows predate NDM's 2300 MHz).
                spreads[f'{cell}/{metric}'] = dv2.archive_spread(archive, cell, metric, where={
                    'max_tokens': MAX_TOKENS, 'clock_cap_mhz': window_caps[0], 'commit': production})
            except ValueError as exc:
                spreads[f'{cell}/{metric}'] = f'UNREADABLE: {exc}'
    statuses = [str(record(v)['status']) for v in arms.values()]
    overall = ('CANDIDATE' if 'CANDIDATE' in statuses else 'INCONCLUSIVE' if 'INCONCLUSIVE' in statuses
               else 'NOT A CANDIDATE')
    return {'rule': dv2.RULE, 'status': overall, 'arms': arms, 'archive_spread_512': spreads,
            'clock_caps_mhz': cast(Json, window_caps),
            'decision_cells': cast(Json, list(plan.decision_cells)), 'max_tokens': MAX_TOKENS,
            'design': f'{group}: {"/".join(plan.roles)}; each candidate start against both bracketing control starts, '
                      'unpaired (Welch), same prompts and seeds 42+r, fresh salts',
            'metrics': {role: ('pooled committed tokens per decode second (R+A)/D decides; ms/round and accepted/round '
                               'reported') if DECIDE[role] == COMMITTED else 'pooled ms/round decides; accepted/round guards'
                        for role in (candidate_role(n) for n in names[1:-1])},
            'aa_calibration': cast(Json, dict(AA_CALIBRATION)),
            'promotion': 'none; a CANDIDATE goes to independent review and a promotion window'}


def archive_rows(group: str, window: str, starts: Mapping[str, Json], *, production: str, tag: str,
                 date: str, cap_of: Callable[[str], tuple[int, str]]) -> list[dict[str, Json]]:
    """Every completed control start's decision-cell repeats, per metric, for the WIN2 control archive. Each row carries
    the production commit, its OWN start's single clock ceiling and start/repeat (S92 FIX 2); a control start whose cap
    is missing or ambiguous is refused, never archived as unknown."""
    plan = plan_for(group)
    rows: list[dict[str, Json]] = []
    for name, value in starts.items():
        row = record(value)
        if row.get('role') != 'control' or row.get('status') != 'COMPLETE':
            continue
        cap, source = cap_of(name)   # WIN2-R2: the start's own start log (win2_history.start_clock_cap)
        caps = clock_caps(starts, name)
        ck.require(caps == [cap], f'{name}: start log cap {cap} differs from the start output {caps!r}')
        for cell in plan.decision_cells:
            for metric in ARCHIVE_METRICS:
                for i, v in enumerate(repeat_values(record(_arm_cells(starts, name)[cell]), metric)):
                    rows.append({'window': window, 'start': name, 'arm': name, 'cell': cell, 'metric': metric,
                                 'repeat': i, 'value': v, 'max_tokens': MAX_TOKENS, 'commit': production,
                                 'production_tag': tag, 'runner': 'up3', 'clock_cap_mhz': cap, 'clock_source': source,
                                 'clock_caps_mhz': cast(Json, caps), 'date': date})
    return rows
