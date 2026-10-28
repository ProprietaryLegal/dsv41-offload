"""R-UPX window groups (rule win2-v1, offload/launch/windows/WIN2.md) as data for a WIN2 chain runner.

One group per item, OFF / ON / OFF, priority order (UPAUDIT ranked fixes 4, 5, 9 and row C2):
  UPX-0064   C1 / split / C2       0064 alone: R-UP3's arm 'split' (production-exact base, offload/up3 @ac23d4f)
  UPX-PIN    C1 / cpu-pin / C2     EXL3_CPU_PIN=5-9,15-19 alone: R-UP3's arm 'cpu-pin' with the Round 8 receipt census
  UPX-0085B  C1 / release / C2     0085b memory: file-toggled ON/OFF battery pairs inside the candidate start
  UPX-0067   C1 / vor / C2         0067 alone: R-UP3's arm 'vor' (lowest priority; likely INCONCLUSIVE, see WINDOWS.md)
UPX-W1 (C1 / split / C2 / cpu-pin / C3, shared middle control) is listed for comparison only: one 2.6-hour HOLD.

Correctness gate (NDM's, replacing the never-built resident-worker bitwise harness that parked 0064 and 0067 before
HOLD: S59 FIX 1, UP3 R4): the six W5/K1 greedy prompts at 0.9 nats, plus for 0064 the band probes and band greedy of
band.py (the six prompts never reach a 33-row step), plus greedy_kl's KL re-probe in the following control.

Estimates: START_S and W8_S are MEASURED (arm starts 6.5-6.8 min, strict W8 restore 7.4 min); cell bounds come from the
shared catalogue (window_cells.CELLS) or band.BAND_CELLS (ESTIMATED from MEASURED NBW-W3 spans); every gate figure is
ESTIMATED and labelled. Nothing here contacts a host: a runner executes the steps and feeds results to arm_verdict.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from offload.gb10_budget import window_cells as wc
from offload.launch.windows import decision_v2 as dv
from offload.prefill_maxlen.evidence import Json
from offload.upx import band
from offload.upx import configs

UP3_COMMIT = 'ac23d4f'      # R-UP3 Round 8 (S84 cleared the CPU-pin receipt census); arms read, never edited
UP3_CONFIG = 'offload/up3/window.env'
START_S = 6.8 * 60          # MEASURED upper end of an arm start
W8_S = 7.4 * 60             # MEASURED strict W8 restore (last window of a chain, or any stop)
WATCH_S = 15 * 60           # post-close watch after a W8 restore
FENCES_S = 5 * 60           # ESTIMATED: HOLD open/close fences, quiet and absence proofs
MAX_TOKENS_B1 = 512         # ctl-short-b1 / ctl-32k-b1 decision cells: the control archive's majority definition
GATE_S = {                  # ESTIMATED
    'greedy': 60,           # six W5 prompts x 300 tokens at MEASURED 51-57 tok/s B1, plus near-tie probes
    'regions': 30,          # four /tokenize calls over 772K public characters (band.REGIONS), pinned by sha256
    'band-probes': 60,      # eight one-token probes; the two long ones prefill 8,232 and 32,808 tokens (~3 + ~15 s)
    'band-greedy': 60,      # six 33-63-token prompts x 300 tokens at ~51 tok/s, plus near-tie probes
    'long-greedy': 120,     # 32K + 150K token prompts x 300 tokens: MEASURED 15 s + 69 s prefill, ~12 s decode
    'needle-150k': 120,     # MEASURED 150K TTFT 69 s plus the answer
    'kl': 60,               # following control re-probes every recorded divergence (zero if none)
    'engagement': 15,       # docker logs of both ranks: 0064's once-per-process line present (X) / absent (C)
    'census': 30,           # R-UP3 Round 8 receipt + direct thread census (R-UP3's 30 s host-evidence reserve)
    'battery': 200,         # MEASURE-0085B.md: two cold 150K B1 prefills (MEASURED 69 s each) + 20 s and 15 s idle
                            # + threshold file write on both ranks + release census
    'release-census': 15,   # docker logs of both ranks: release lines per idle period
}


@dataclass(frozen=True)
class Arm:
    config: str
    env: Mapping[str, str]
    source: str


ARMS: dict[str, Arm] = {
    'C-prod': Arm(UP3_CONFIG, {'UP3_ARM': 'control'}, f'R-UP3 {UP3_COMMIT}: production bytes, port 8001'),
    'split': Arm(UP3_CONFIG, {'UP3_ARM': 'split'}, f'R-UP3 {UP3_COMMIT}: 0064 forward.py/smoe.py + SPLIT_MAX_TOKENS=63'),
    'cpu-pin': Arm(UP3_CONFIG, {'UP3_ARM': 'cpu-pin'}, f'R-UP3 {UP3_COMMIT}: EXL3_CPU_PIN=5-9,15-19 + Round 8 receipt'),
    'vor': Arm(UP3_CONFIG, {'UP3_ARM': 'vor'}, f'R-UP3 {UP3_COMMIT}: 0067 three files + VLLM_DSV41_VOR_FUSED=1'),
    'C-k1': Arm(configs.CONFIG, configs.ARMS['C-k1'], 'R-UPX: K1 production-equivalent base'),
    'release-file': Arm(configs.CONFIG, configs.ARMS['release-file'], 'R-UPX: 0085b, VLLM_DSV41_RELEASE_FILE'),
}
CONTROLS = ('C-prod', 'C-k1')


@dataclass(frozen=True)
class Group:
    name: str
    candidate: str
    control: str
    decision_cells: tuple[str, ...]
    groups: Mapping[str, tuple[str, ...]]
    metrics: Mapping[str, tuple[tuple[str, bool], ...]]   # cell -> (metric, higher_is_better), primary first
    required_gates: tuple[str, ...]                       # candidate start
    follow_gates: tuple[str, ...]                         # the control start after the candidate
    paired: bool                                          # False: C1 + C2 pooled against X (Welch)
    band_required: bool                                   # decision cells must contain 33-63-row steps
    upstream: str
    estimated_effect: str


BAND_METRICS: tuple[tuple[str, bool], ...] = (('ms_per_round', False), ('accepted_per_round', True),
                                              ('short_latency_median_s', False))
B1_METRICS: tuple[tuple[str, bool], ...] = (('ms_per_round', False), ('accepted_per_round', True))
MEM_METRICS: tuple[tuple[str, bool], ...] = (('next_low_kb', True), ('idle_kb', True))
BAND_CELLS = tuple(band.BAND_CELLS)

GROUPS: dict[str, Group] = {g.name: g for g in (
    Group('UPX-0064', 'split', 'C-prod', BAND_CELLS,
          {'32k-context': ('band-32k-4user',), '150k-context': ('band-150k-4user',)},
          {c: BAND_METRICS for c in BAND_CELLS},
          ('greedy', 'regions', 'band-probes', 'band-greedy', 'engagement'), ('kl', 'band-probes', 'band-greedy'),
          paired=False, band_required=True,
          upstream='UPSTREAM sustained 118 -> 129 tok/s (+9 %) together with CPU pin at 8 streams; cost/round -0.24 %',
          estimated_effect='unknown per band step on our 4-slot stack; production band share UNMEASURED'),
    Group('UPX-PIN', 'cpu-pin', 'C-prod', BAND_CELLS,
          {'32k-context': ('band-32k-4user',), '150k-context': ('band-150k-4user',)},
          {c: BAND_METRICS for c in BAND_CELLS},
          ('greedy', 'regions', 'census'), ('kl', 'census'),
          paired=False, band_required=True,
          upstream='UPSTREAM measured only together with 0064 (shares not separated)',
          estimated_effect='RS1 ESTIMATED 0.5-1.5 % on host-bound multi-user steps; may not clear 2 %'),
    Group('UPX-0067', 'vor', 'C-prod', ('ctl-short-b1', 'ctl-32k-b1'),
          {'short': ('ctl-short-b1',), '32k': ('ctl-32k-b1',)},
          {'ctl-short-b1': B1_METRICS, 'ctl-32k-b1': B1_METRICS},
          ('greedy', 'long-greedy', 'needle-150k'), ('kl', 'long-greedy'),
          paired=False, band_required=False,
          upstream='UPSTREAM -0.66 % per round together with 0066 (not isolated)',
          estimated_effect='RS1 ESTIMATED 0-0.7 % per round: below the 2 % WIN bar, so INCONCLUSIVE is the likely result'),
    Group('UPX-0085B', 'release-file', 'C-k1', ('mem-r0', 'mem-r1'),
          {'head': ('mem-r0',), 'worker': ('mem-r1',)},
          {'mem-r0': (*MEM_METRICS, ('next_ttft_s', False)), 'mem-r1': MEM_METRICS},
          ('greedy', 'release-census'), ('kl',),
          paired=True, band_required=False,
          upstream='UPSTREAM idle free +1.8 GiB per node; next 300K prefill low point 3,804 vs 3,269 MiB (+535 MiB)',
          estimated_effect='+0.5-1.8 GiB idle, +~0.5 GiB at the next 150K low point (from UPSTREAM; not yet MEASURED)'),
)}


@dataclass(frozen=True)
class Step:
    kind: str                   # gate kinds (GATE_S), 'cell', 'band', 'battery-pair', 'battery-image', 'warmup'
    repeats: int = 1
    cell: str = ''
    max_tokens: int | None = None

    def seconds(self) -> float:
        if self.kind in ('cell', 'warmup'):
            return wc.CELLS[self.cell].estimate_s * self.repeats
        if self.kind == 'band':
            return band.BAND_CELLS[self.cell].estimate_s * self.repeats
        if self.kind == 'battery-pair':
            return 2 * GATE_S['battery'] * self.repeats
        if self.kind == 'battery-image':
            return GATE_S['battery'] * self.repeats
        return GATE_S[self.kind] * self.repeats


@dataclass(frozen=True)
class Start:
    label: str
    arm: str
    steps: tuple[Step, ...]


@dataclass(frozen=True)
class Window:
    name: str
    group: str
    starts: tuple[Start, ...]

    def hold_s(self, *, last: bool = False) -> float:
        total = FENCES_S + sum(START_S + sum(step.seconds() for step in s.steps) for s in self.starts)
        return total + (W8_S + WATCH_S if last else 0.0)


WARMUP = Step('warmup', 1, 'warmup', 64)


def band_cells(repeats: int) -> tuple[Step, ...]:
    return tuple(Step('band', repeats, c, band.BAND_CELLS[c].archive_max_tokens) for c in BAND_CELLS)


def b1_cells(repeats: int) -> tuple[Step, ...]:
    return tuple(Step('cell', repeats, c, MAX_TOKENS_B1) for c in ('ctl-short-b1', 'ctl-32k-b1'))


def control(label: str, arm: str, *, gates: tuple[str, ...], cells: tuple[Step, ...]) -> Start:
    return Start(label, arm, (WARMUP, Step('greedy'), *(Step(g) for g in gates), *cells))


# 0064: C1 records the band probe / band greedy references; X is screened against them; C2 gives the A/A floor and
# re-probes X's divergences (kl). Every start runs the regions first (C1 pins them, later starts must match).
C1_0064 = control('C1', 'C-prod', gates=('regions', 'band-probes', 'band-greedy', 'engagement'), cells=band_cells(2))
X_0064 = Start('X-split', 'split', (WARMUP, Step('greedy'), Step('regions'), Step('band-probes'), Step('band-greedy'),
                                    Step('engagement'), *band_cells(3)))
C2_0064 = control('C2', 'C-prod', gates=('regions', 'band-probes', 'band-greedy', 'kl', 'engagement'),
                  cells=band_cells(2))
C1_PIN = control('C1', 'C-prod', gates=('regions', 'census'), cells=band_cells(2))
X_PIN = Start('X-pin', 'cpu-pin', (WARMUP, Step('greedy'), Step('regions'), Step('census'), *band_cells(3)))
C2_PIN = control('C2', 'C-prod', gates=('regions', 'kl', 'census'), cells=band_cells(2))
C1_0067 = control('C1', 'C-prod', gates=('long-greedy',), cells=b1_cells(2))
X_0067 = Start('X-vor', 'vor', (WARMUP, Step('greedy'), Step('long-greedy'), Step('needle-150k'), *b1_cells(3)))
C2_0067 = control('C2', 'C-prod', gates=('long-greedy', 'kl'), cells=b1_cells(2))
C1_0085B = control('C1', 'C-k1', gates=(), cells=(Step('battery-image', 2, 'battery-150k', 16),))
X_0085B = Start('X-release', 'release-file', (WARMUP, Step('greedy'), Step('battery-pair', 3, 'battery-150k', 16),
                                              Step('release-census')))
C2_0085B = control('C2', 'C-k1', gates=('kl',), cells=(Step('battery-image', 2, 'battery-150k', 16),))

WINDOWS = {
    'UPX-0064': Window('UPX-0064', 'UPX-0064', (C1_0064, X_0064, C2_0064)),
    'UPX-PIN': Window('UPX-PIN', 'UPX-PIN', (C1_PIN, X_PIN, C2_PIN)),
    'UPX-0085B': Window('UPX-0085B', 'UPX-0085B', (C1_0085B, X_0085B, C2_0085B)),
    'UPX-0067': Window('UPX-0067', 'UPX-0067', (C1_0067, X_0067, C2_0067)),
}
# Comparison only (not recommended): both band arms in one HOLD with a shared middle control.
SHARED = Window('UPX-W1', 'UPX-0064', (
    control('C1', 'C-prod', gates=('regions', 'band-probes', 'band-greedy', 'engagement', 'census'), cells=band_cells(2)),
    X_0064,
    control('C2', 'C-prod', gates=('regions', 'band-probes', 'band-greedy', 'kl', 'engagement', 'census'),
            cells=band_cells(2)),
    X_PIN,
    control('C3', 'C-prod', gates=('regions', 'kl', 'census'), cells=band_cells(2))))
CHAIN = ('UPX-0064', 'UPX-PIN', 'UPX-0085B', 'UPX-0067')     # recommended order; the last one adds W8 + watch


def decision_repeats(window: Window, group: Group) -> dict[str, dict[str, int]]:
    """Repeats per decision cell: control starts vs candidate starts (0085b: the ON/OFF halves of the battery pairs)."""
    out: dict[str, dict[str, int]] = {c: {'control': 0, 'candidate': 0} for c in group.decision_cells}
    for start in window.starts:
        for step in start.steps:
            if step.kind == 'battery-pair':
                for c in group.decision_cells:
                    out[c]['control'] += step.repeats
                    out[c]['candidate'] += step.repeats
            elif step.cell in group.decision_cells:
                out[step.cell]['control' if start.arm in CONTROLS else 'candidate'] += step.repeats
    return out


def validate(window: Window) -> None:
    """WIN2 constraints and the brief's: <= 2 decision cells, >= 3 repeats per arm in each, max_tokens on every cell,
    controls bracket the candidate, every start begins with the warm-up, the candidate's and the following control's
    gates are present, and 0064 / CPU-pin decide only on cells that contain 33-63-row steps."""
    group = GROUPS[window.group]
    if not 1 <= len(group.decision_cells) <= dv.MAX_DECISION_CELLS:
        raise ValueError(f'{window.name}: {len(group.decision_cells)} decision cells')
    if group.band_required and not all(c in band.BAND_CELLS for c in group.decision_cells):
        raise ValueError(f'{window.name}: decision cells without guaranteed 33-63-row steps cannot decide {group.candidate}')
    first, last = window.starts[0], window.starts[-1]
    if not (first.arm == last.arm and first.arm in CONTROLS):
        raise ValueError(f'{window.name}: controls must bracket the candidates')
    for i, start in enumerate(window.starts):
        if start.arm not in ARMS:
            raise ValueError(f'{window.name}/{start.label}: unknown arm {start.arm}')
        if start.steps[0] != WARMUP:
            raise ValueError(f'{window.name}/{start.label}: every start begins with the warm-up cell')
        for step in start.steps:
            if step.kind in ('cell', 'band', 'battery-pair', 'battery-image') and step.max_tokens is None:
                raise ValueError(f'{window.name}/{start.label}: {step.cell} without max_tokens')
        kinds = {s.kind for s in start.steps}
        if start.arm not in CONTROLS:
            if start.arm != group.candidate and window is not SHARED:
                raise ValueError(f'{window.name}: {start.arm} is not {group.name}\'s candidate')
            own = GROUPS[next(g for g in GROUPS if GROUPS[g].candidate == start.arm)]
            missing = [g for g in own.required_gates if g not in kinds]
            follow = {s.kind for s in window.starts[i + 1].steps} if i + 1 < len(window.starts) else set()
            missing += [f'next control: {g}' for g in own.follow_gates if g not in follow]
            if missing:
                raise ValueError(f'{window.name}/{start.label}: gates {missing} missing')
    for cell, counts in decision_repeats(window, group).items():
        if min(counts.values()) < dv.MIN_REPEATS:
            raise ValueError(f'{window.name}: {cell} repeats {counts} < {dv.MIN_REPEATS} per arm')


def series(group: Group, cell: str, metric: str, control: Sequence[float], candidate: Sequence[float]) -> dv.Series:
    """0064 / CPU pin / 0067: C1 + C2 pooled (4) against X (3), unpaired (Welch). 0085b: ON/OFF pairs inside X, paired."""
    higher = dict(group.metrics[cell])[metric]
    return dv.Series(cell, metric, higher, tuple(control), tuple(candidate), paired=group.paired)


def arm_verdict(group_name: str, decision: Mapping[str, Json], gates: Mapping[str, bool]) -> str:
    """CANDIDATE FOR PROMOTION only with every gate passing and a win2-v1 CANDIDATE (>= 1 WIN, no LOSS); these are speed
    or memory claims, so INCONCLUSIVE keeps the item off."""
    group = GROUPS[group_name]
    required = set(group.required_gates) | set(group.follow_gates)
    if set(gates) != required:
        raise ValueError(f'{group_name}: gates {sorted(gates)} != required {sorted(required)}')
    if decision.get('rule') != dv.RULE:
        raise ValueError('decision is not a win2-v1 result')
    failed = sorted(g for g, ok in gates.items() if not ok)
    if failed:
        return 'STOP: ' + ', '.join(failed)
    status = decision['status']
    if status == 'NOT A CANDIDATE':
        return 'NOT A CANDIDATE'
    if status == 'INCONCLUSIVE':
        return 'NO GAIN SHOWN (stays off)'
    if status != 'CANDIDATE':
        raise ValueError(f'unknown decide_window status {status!r}')
    return 'CANDIDATE FOR PROMOTION'


def hold_minutes(window: Window, *, last: bool = False) -> int:
    return math.ceil(window.hold_s(last=last) / 60)


def plan() -> dict[str, Json]:
    out: dict[str, Json] = {}
    for name in (*CHAIN, SHARED.name):
        window = WINDOWS.get(name, SHARED)
        group = GROUPS[window.group]
        out[name] = {
            'group': group.name, 'candidate': group.candidate,
            'arm': {**dict(ARMS[group.candidate].env), 'config': ARMS[group.candidate].config},
            'decision_cells': list(group.decision_cells), 'groups': {k: list(v) for k, v in group.groups.items()},
            'metrics': {c: [m for m, _ in ms] for c, ms in group.metrics.items()},
            'starts': [{'label': s.label, 'arm': s.arm,
                        'steps': [f'{st.kind}{":" + st.cell if st.cell else ""}x{st.repeats}' for st in s.steps],
                        'estimated_s': round(START_S + sum(st.seconds() for st in s.steps))} for s in window.starts],
            'hold_min_chained': hold_minutes(window), 'hold_min_last': hold_minutes(window, last=True),
            'upstream': group.upstream, 'estimated_effect': group.estimated_effect}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('plan', 'validate'))
    args = parser.parse_args()
    for window in (*WINDOWS.values(), SHARED):
        validate(window)
    if args.action == 'plan':
        print(json.dumps(plan(), indent=1))
    else:
        print('R-UPX windows: valid under win2-v1')


if __name__ == '__main__':
    main()
