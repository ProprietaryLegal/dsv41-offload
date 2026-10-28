"""R-UPX window groups in the UP3 runner (coordinator 2026-10-27): offload/upx windows.py's designs, executed here.

UP3 group  -> R-UPX design (offload/upx/windows.py, WINDOWS.md), OFF / ON / OFF:
  upx-0064 -> UPX-0064  C / split / C    0064 (R-UP3 arm 'split'), band cells + band probes + band greedy + engagement
  upx-pin  -> UPX-PIN   C / cpu-pin / C  CPU pin (R-UP3 arm 'cpu-pin' with the Round 8 receipt census), band cells
  upx-0067 -> UPX-0067  C / vor / C      0067 (R-UP3 arm 'vor'), long greedy + 150K needle, B1 cells
split and vor were parked before HOLD (UP3 R4: the resident-worker bitwise harness was never built); they are admitted
ONLY inside these groups, under registration upx-0p9: the six-prompt greedy at 0.9+1e-3 nats with recorded
distributions, the group's band / long gates, and greedy_kl's re-probe in the following control at max(0.005,
2 x the band probes' A/A floor). Any failed gate is a STOP (the strict W8 restore runs). Decisions: windows.series
(C1 + C2 pooled against X, unpaired), decision_v2.decide_window and windows.arm_verdict. Nothing here contacts a host.

0067's decision cells are UP3's own short-b1 / 32k-b1 at 512 tokens (window_cells' ctl- cells generate 1,500 tokens;
WINDOWS.md decides 0067 at 512), passed to windows.series under the design's names ctl-short-b1 / ctl-32k-b1.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from offload.launch.windows import decision_v2 as dv2
from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, number, record
from offload.upx import band
from offload.upx import windows as upx
from offload.up3 import win2_plan

REGISTRATION = 'upx-0p9'
GATE_S = upx.GATE_S                 # the design's ESTIMATED gate seconds (re-exported for the runner)
B1_NAMES = {'short-b1': 'ctl-short-b1', '32k-b1': 'ctl-32k-b1'}   # UP3 cell -> R-UPX design name
B1_MAX_TOKENS = 512


@dataclass(frozen=True)
class Group:
    design: str                         # offload/upx windows.WINDOWS / GROUPS key
    cells: tuple[str, ...]              # UP3 runner cell names, in run order

    @property
    def window(self) -> upx.Window:
        return upx.WINDOWS[self.design]

    @property
    def spec(self) -> upx.Group:
        return upx.GROUPS[self.design]

    @property
    def candidate(self) -> str:
        return self.spec.candidate


GROUPS: dict[str, Group] = {
    'upx-0064': Group('UPX-0064', tuple(band.BAND_CELLS)),
    'upx-pin': Group('UPX-PIN', tuple(band.BAND_CELLS)),
    'upx-0067': Group('UPX-0067', tuple(B1_NAMES)),
}
# Gates the runner itself performs elsewhere: greedy (Live.greedy), census (cpu-pin host gate in Live.qualify).
RUNNER_GATES = ('greedy', 'census')
UPX_ROLES = ('split', 'vor')            # admitted only here (host_gates parks them everywhere else)


def group_for(name: str) -> Group:
    ck.require(name in GROUPS, f'{name}: not an R-UPX group')
    return GROUPS[name]


def roles(name: str) -> tuple[str, ...]:
    return ('control', group_for(name).candidate, 'control')


def design_cell(cell: str) -> str:
    return B1_NAMES.get(cell, cell)


def steps(name: str, position: int) -> tuple[upx.Step, ...]:
    """The R-UPX design's steps for start `position` (0 = C1, 1 = X, 2 = C2)."""
    return group_for(name).window.starts[position].steps


def gates(name: str, position: int) -> tuple[str, ...]:
    """Gate kinds of that start that this module's driver runs (regions, band-probes, band-greedy, long-greedy, kl,
    engagement, needle-150k), in the design's order. long-greedy's prompts come from band.REGIONS['long'], but
    UPX-0067's design lists no 'regions' step (R-UPX gap, reported): the runner adds it first wherever a start uses
    token regions, so every start still reproduces the first start's pins."""
    kinds = tuple(s.kind for s in steps(name, position)
                  if s.kind not in ('warmup', 'cell', 'band') and s.kind not in RUNNER_GATES)
    uses = {'band-probes', 'band-greedy', 'long-greedy'} & set(kinds) or any(c in band.BAND_CELLS for c in group_for(name).cells)
    return kinds if 'regions' in kinds or not uses else ('regions', *kinds)


def repeats(name: str, cell: str, position: int) -> int:
    found = [s.repeats for s in steps(name, position) if s.cell == design_cell(cell) and s.kind in ('cell', 'band')]
    ck.require(len(found) == 1, f'{name}: {cell} not scheduled once in start {position}')
    return found[0]


def gate_s(name: str, position: int) -> float:
    return sum(upx.GATE_S[g] for g in gates(name, position))


def cell_s(cell: str, count: int) -> float:
    if cell in band.BAND_CELLS:
        return band.BAND_CELLS[cell].estimate_s * count
    return win2_plan.PER_REPEAT_S[cell] * count


def block_s(name: str, position: int) -> int:
    """MEASURED-basis prepare (start, six greedy, OPS1 warm-up, host evidence: win2_plan.PREPARE_S) + the design's
    ESTIMATED gates and cells + finish."""
    group = group_for(name)
    cells_total = sum(cell_s(c, repeats(name, c, position)) for c in group.cells)
    return math.ceil(win2_plan.PREPARE_S + gate_s(name, position) + cells_total + win2_plan.FINISH_S)


def repeat_values(cell: Mapping[str, Json], metric: str) -> list[float]:
    """Band cells carry per-repeat ms_per_round / accepted_per_round / short_latency_median_s (band.run_band_cell);
    B1 cells are pooled per repeat from UP3's own counters (win2_plan.repeat_values)."""
    rows = [record(r) for r in ck.rows(cell['repetitions'])]
    if str(cell.get('cell')) in band.BAND_CELLS:
        ck.require(all(r.get('band_ok') is True for r in rows), 'band cell repeat below the band registration')
        return [number(r[metric]) for r in rows]
    return win2_plan.repeat_values(cell, metric)


def kl_bound(screen: Mapping[str, Json] | None) -> float:
    """Registered greedy-divergence KL bound: max(0.005, 2 x the band probes' A/A floor); 0.005 without band probes."""
    if screen is None:
        return band.KL_FLOOR
    floors = [number(record(r)['aa_kl']) for r in record(screen['rows']).values()]
    return max(band.KL_FLOOR, 2 * max(floors))


def decide(name: str, starts: Mapping[str, Json], archive: Path, production: str) -> dict[str, Json]:
    group = group_for(name)
    names = [f's{i}-{role}' for i, role in enumerate(roles(name), 1)]
    if list(starts) != names or any(record(v).get('status') != 'COMPLETE' for v in starts.values()):
        return {'status': 'NOT A CANDIDATE', 'reason': 'all three mandatory complete starts required', 'rule': dv2.RULE}
    caps = {n: win2_plan.clock_caps(starts, n) for n in names}
    if any(len(c) != 1 for c in caps.values()) or len({c[0] for c in caps.values()}) != 1:
        return {'status': 'NOT A CANDIDATE', 'reason': 'clock cap unproved or changed between starts',
                'clock_caps_mhz': cast(Json, caps), 'rule': dv2.RULE}
    series: list[dv2.Series] = []
    for cell in group.cells:
        measured = [record(record(record(starts[n])['cells'])[cell]) for n in names]
        spec = dict(group.spec.metrics[design_cell(cell)])
        for metric in spec:
            control = repeat_values(measured[0], metric) + repeat_values(measured[2], metric)
            series.append(upx.series(group.spec, design_cell(cell), metric, control, repeat_values(measured[1], metric)))
    decision = dv2.decide_window(series, group.spec.decision_cells, groups=group.spec.groups)
    passed = {g: True for g in (*group.spec.required_gates, *group.spec.follow_gates)}   # a failed gate STOPs earlier
    verdict = upx.arm_verdict(group.design, decision, passed)
    spreads: dict[str, Json] = {}
    for cell in group.cells:
        tokens = band.BAND_CELLS[cell].archive_max_tokens if cell in band.BAND_CELLS else B1_MAX_TOKENS
        for metric in dict(group.spec.metrics[design_cell(cell)]):
            try:
                spreads[f'{cell}/{metric}'] = dv2.archive_spread(archive, cell, metric, where={
                    'max_tokens': tokens, 'clock_cap_mhz': caps[names[0]][0], 'commit': production})
            except ValueError as exc:
                spreads[f'{cell}/{metric}'] = f'UNREADABLE: {exc}'
    gates_seen = {n: record(record(starts[n]).get('gates', {})).get('upx_gates', {}) for n in names}
    return {**decision, 'verdict': verdict, 'registration': REGISTRATION, 'design': group.design,
            'clock_caps_mhz': caps[names[0]][0], 'archive_spread': spreads, 'gates': cast(Json, gates_seen),
            'upstream': group.spec.upstream, 'estimated_effect': group.spec.estimated_effect,
            'promotion': 'none; a CANDIDATE FOR PROMOTION goes to independent review and a promotion window'}


def archive_rows(name: str, window: str, starts: Mapping[str, Json], *, production: str, tag: str,
                 date: str, cap_of: Callable[[str], tuple[int, str]]) -> list[dict[str, Json]]:
    """Every completed control start's decision-cell repeats with WIN2-R2 provenance (band cells under their own names
    at the largest decoder's max_tokens; B1 cells at 512)."""
    group = group_for(name)
    rows: list[dict[str, Json]] = []
    for start, value in starts.items():
        row = record(value)
        if row.get('role') != 'control' or row.get('status') != 'COMPLETE':
            continue
        cap, source = cap_of(start)   # WIN2-R2: the start's own start log (win2_history.start_clock_cap)
        caps = win2_plan.clock_caps(starts, start)
        ck.require(caps == [cap], f'{start}: start log cap {cap} differs from the start output {caps!r}')
        for cell in group.cells:
            measured = record(record(row['cells'])[cell])
            tokens = band.BAND_CELLS[cell].archive_max_tokens if cell in band.BAND_CELLS else B1_MAX_TOKENS
            for metric in dict(group.spec.metrics[design_cell(cell)]):
                for i, v in enumerate(repeat_values(measured, metric)):
                    rows.append({'window': window, 'start': start, 'arm': start, 'cell': cell, 'metric': metric,
                                 'repeat': i, 'value': v, 'max_tokens': tokens, 'commit': production,
                                 'production_tag': tag, 'runner': 'up3-upx', 'clock_cap_mhz': cap, 'clock_source': source,
                                 'clock_caps_mhz': cast(Json, caps), 'date': date})
    return rows
