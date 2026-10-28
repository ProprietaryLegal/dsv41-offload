"""R-UPF v2 window designs (rule win2-v1, offload/launch/windows/WIN2.md) as data for a config-only chain runner.

Both items are correctness backports, so a WIN is not required. An arm is PROMOTABLE when every identity gate passes and
decide_window finds no decision-cell LOSS (status CANDIDATE or INCONCLUSIVE; the LOSS semantics of decide_cell, as R-PP4's
promotion gate uses). A decision-cell LOSS makes it NOT A CANDIDATE; a failed gate stops the arm. Nothing here contacts a
host: a runner executes the steps (WINDOW.md) and feeds the results to arm_verdict / soak_ok.

Estimates: START_S, W8_S are MEASURED (arm starts 6.5-6.8 min, W8 restore 7.4 min, earlier windows); cell bounds come from
the shared catalogue (gb10_budget/window_cells.py); every other figure is ESTIMATED and labelled.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from offload.gb10_budget import window_cells as wc
from offload.launch.windows import decision_v2 as dv
from offload.launch.windows import win2_history as wh
from offload.prefill_maxlen.evidence import Json
from offload.upfix import configs

MAX_TOKENS = 512            # every decision cell and archive row (the archive's majority definition; WIN2.md)
GREEDY_NEAR_TIE_NATS = configs.GREEDY_NEAR_TIE_NATS
KL_BOUND_FLOOR = 0.005      # greedy_kl bound: max(0.005, 2 x the window's C1/C2 A/A floor), UP2's registration
START_S = 6.8 * 60          # MEASURED upper end of an arm start
W8_S = 7.4 * 60             # MEASURED strict W8 restore (last window of a chain, or any stop)
WATCH_S = 15 * 60           # post-close watch after a W8 restore
FENCES_S = 5 * 60           # ESTIMATED: HOLD open/close fences, quiet and absence proofs
SOAK_S = 600                # ESTIMATED choice: 10 min of sustained 4-stream load per soak
SOAK_FLOOR_MIB = 16.0       # soak gate: ON growth per process <= max(16 MiB, 2 x the same process's OFF growth)
GATE_S = {                  # ESTIMATED
    'greedy': 60,           # six prompts x 300 tokens at MEASURED 51-57 tok/s B1 decode, plus tokenization
    'kl': 60,               # following control re-probes each recorded divergence (zero if none)
    'needle-150k': 120,     # MEASURED 150K TTFT 69 s plus the answer
    'needle-300k': 240,     # ESTIMATED from the 150K TTFT, prefill roughly linear to the 327,680 cap
    'stop_idle': 60,        # ./stop.sh + start_exp.sh's GPU-idle check on both nodes (S93 FIX 3)
    'jit_build': 600,       # nvcc for SM121 inside the image, CPU only; MEASURED 104 s for sm_86 on Controller (WINDOW.md)
    'bitcompare': 180,      # container start, CUDA init, 149 cases (harness MEASURED ~12 s on Controller CPU)
}
T2_TIMEOUT_S = 600          # hard bound on T2's docker run (WINDOW.md); a hung kernel costs at most this, then X-topk skips
DECISION_CELLS = ('ctl-150k-b1', 'ctl-32k-b4')
GROUPS = {'single-user': ('ctl-150k-b1',), 'multi-user': ('ctl-32k-b4',)}
# Primary metric first (it decides); the rest are guards that can only make a cell LOSS (decide_window).
METRICS: dict[str, tuple[tuple[str, bool], ...]] = {
    'ctl-150k-b1': (('ms_per_round', False), ('accepted_per_round', True), ('ttft_s', False)),
    'ctl-32k-b4': (('ms_per_round', False), ('accepted_per_round', True)),
}
# Control-archive history of the same cell definitions predates the ctl- names (S93): report archive_spread for both.
HISTORY_NAMES = {'ctl-150k-b1': '150k-b1', 'ctl-32k-b4': '32k-b4'}
RULE_REVISION = 2           # WIN2-R2 (3bd73a7): guard LOSS needs its interval below zero; spread 1.69 x s/c4(n)
REQUIRED_GATES = {
    'upf-0091': ('greedy', 'needle-150k', 'soak'),
    'topk-fix': ('bitcompare', 'greedy', 'needle-150k', 'needle-300k'),
}


@dataclass(frozen=True)
class Step:
    kind: str                   # cell | warmup | greedy | kl | needle-* | soak | stop_idle | jit_build | bitcompare
    repeats: int = 1
    max_tokens: int | None = None

    def seconds(self) -> float:
        if self.kind in ('cell', 'warmup'):
            raise ValueError('cell steps take their bound from the catalogue')
        return SOAK_S if self.kind == 'soak' else GATE_S[self.kind]


@dataclass(frozen=True)
class CellStep(Step):
    cell: str = ''

    def seconds(self) -> float:
        return wc.CELLS[self.cell].estimate_s * self.repeats


@dataclass(frozen=True)
class Start:
    label: str
    arm: str
    steps: tuple[Step, ...]


@dataclass(frozen=True)
class Window:
    name: str
    pre: tuple[Step, ...]       # inside the HOLD, before the first start; begins with stop_idle (no server, GPUs idle)
    starts: tuple[Start, ...]

    def candidates(self) -> list[Start]:
        return [s for s in self.starts if not s.arm.startswith('C-')]

    def hold_s(self, *, last: bool = False) -> float:
        total = FENCES_S + sum(s.seconds() for s in self.pre)
        total += sum(START_S + sum(step.seconds() for step in s.steps) for s in self.starts)
        return total + (W8_S + WATCH_S if last else 0.0)


def cells(repeats: int) -> tuple[Step, ...]:
    return tuple(CellStep('cell', repeats, MAX_TOKENS, cell) for cell in DECISION_CELLS)


WARMUP = CellStep('warmup', 1, 64, 'warmup')
C1 = Start('C1', 'C-k1', (WARMUP, Step('greedy'), *cells(2), Step('soak')))
X_0091 = Start('X-0091', 'upf-0091', (WARMUP, Step('greedy'), Step('needle-150k'), *cells(3), Step('soak')))
X_TOPK = Start('X-topk', 'topk-fix', (WARMUP, Step('greedy'), Step('needle-150k'), Step('needle-300k'), *cells(3)))
C2 = Start('C2', 'C-k1', (WARMUP, Step('kl'), *cells(2)))
# T2 uses the GB10 GPU: in either preflight mode ('production' or 'chained') it runs only after ./stop.sh and the
# start_exp.sh GPU-idle check on both nodes (S93 FIX 3); T1 follows the stop too, so nvcc never competes with serving.
TOPK_PRE = (Step('stop_idle'), Step('jit_build'), Step('bitcompare'))

WINDOWS = {
    # Recommended: one chained window for both items, one control pair shared by both candidates.
    'UPF-W1': Window('UPF-W1', TOPK_PRE, (C1, X_0091, X_TOPK, C2)),
    # Fallbacks when a shorter HOLD is wanted: each item alone, C/X/C.
    'UPF-0091': Window('UPF-0091', (), (C1, X_0091, C2)),
    'UPF-TOPK': Window('UPF-TOPK', TOPK_PRE, (Start('C1', 'C-k1', (WARMUP, Step('greedy'), *cells(2))), X_TOPK, C2)),
}


def validate(window: Window) -> None:
    """WIN2 constraints: <= 2 decision cells, >= 3 repeats per arm in each, max_tokens on every cell, arms exist."""
    if len(DECISION_CELLS) > dv.MAX_DECISION_CELLS:
        raise ValueError('too many decision cells')
    pre = [s.kind for s in window.pre]
    if pre and pre[0] != 'stop_idle':
        raise ValueError(f'{window.name}: pre-steps must start with stop_idle (./stop.sh + GPU-idle check), got {pre}')
    control: dict[str, int] = {c: 0 for c in DECISION_CELLS}
    for start in window.starts:
        if start.arm not in configs.ARMS:
            raise ValueError(f'{window.name}/{start.label}: unknown arm {start.arm}')
        if start.steps[0] != WARMUP:
            raise ValueError(f'{window.name}/{start.label}: every start begins with the warm-up cell')
        seen: dict[str, int] = {}
        for step in start.steps:
            if isinstance(step, CellStep):
                if step.cell not in wc.CELLS or step.max_tokens is None:
                    raise ValueError(f'{window.name}/{start.label}: {step.cell} unknown or without max_tokens')
                seen[step.cell] = seen.get(step.cell, 0) + step.repeats
        if start.arm.startswith('C-'):
            for cell in DECISION_CELLS:
                control[cell] += seen.get(cell, 0)
        else:
            kinds = {s.kind for s in start.steps} | {s.kind for s in window.pre}
            missing = [g for g in REQUIRED_GATES[start.arm] if g not in kinds]
            if missing or any(seen.get(c, 0) < dv.MIN_REPEATS for c in DECISION_CELLS):
                raise ValueError(f'{window.name}/{start.label}: gates {missing} missing or < 3 repeats')
    if any(n < dv.MIN_REPEATS for n in control.values()):
        raise ValueError(f'{window.name}: control repeats {control} below {dv.MIN_REPEATS}')
    if window.starts[0].arm != window.starts[-1].arm or not window.starts[0].arm.startswith('C-'):
        raise ValueError(f'{window.name}: controls must bracket the candidates')


def series(cell: str, metric: str, control: Sequence[float], candidate: Sequence[float]) -> dv.Series:
    """Controls pool C1 + C2 (4 repeats) against one candidate start (3): unpaired, Welch interval."""
    higher = dict(METRICS[cell])[metric]
    return dv.Series(cell, metric, higher, tuple(control), tuple(candidate), paired=False)


def soak_ok(off_mib: Mapping[str, float], on_mib: Mapping[str, float]) -> bool:
    """Per-process RSS growth over the soak, keyed by role (node:process). Same roles both ways, or the soak is invalid."""
    if set(off_mib) != set(on_mib) or not on_mib:
        raise ValueError(f'soak roles differ: {sorted(off_mib)} vs {sorted(on_mib)}')
    return all(on_mib[r] <= max(SOAK_FLOOR_MIB, 2 * off_mib[r]) for r in on_mib)


def arm_verdict(arm: str, decision: Mapping[str, Json], gates: Mapping[str, bool]) -> str:
    """Identity / no-loss verdict for one candidate arm."""
    if set(gates) != set(REQUIRED_GATES[arm]):
        raise ValueError(f'{arm}: gates {sorted(gates)} != required {sorted(REQUIRED_GATES[arm])}')
    if decision.get('rule') != dv.RULE or decision.get('rule_revision') != RULE_REVISION:
        raise ValueError(f'decision is not a win2-v1 revision {RULE_REVISION} result')
    failed = sorted(g for g, ok in gates.items() if not ok)
    if failed:
        return 'STOP: ' + ', '.join(failed)
    status = decision['status']
    if status == 'NOT A CANDIDATE':
        return 'NOT A CANDIDATE'
    if status not in ('CANDIDATE', 'INCONCLUSIVE'):
        raise ValueError(f'unknown decide_window status {status!r}')
    return 'PROMOTABLE (no loss)'


def archive_rows(out: Path, start: str, values: Mapping[tuple[str, str], Sequence[float]]) -> list[dict[str, Json]]:
    """Control-archive v3 rows (WIN2-R2 FIX 2) for one CONTROL start (C1 or C2): every repeat of each decision-cell metric,
    with the window's 40-hex production commit, that start's proven clock cap and its source, date YYYYMMDD, the start
    and the repeat. Refuses anything that is not a decision-cell metric. Record dv.archive_head before and after
    dv.archive_append."""
    if start not in {s.label for w in WINDOWS.values() for s in w.starts if s.arm.startswith('C-')}:
        raise ValueError(f'{start}: only control starts are archived')
    commit = wh.production_commit(out)
    cap, source = wh.start_clock_cap(out, start)
    date = wh.window_date(out)
    rows: list[dict[str, Json]] = []
    for (cell, metric), repeats in sorted(values.items()):
        if cell not in DECISION_CELLS or metric not in dict(METRICS[cell]):
            raise ValueError(f'{cell}/{metric} is not a decision-cell metric of this design')
        for repeat, value in enumerate(repeats):
            rows.append({'window': out.name, 'cell': cell, 'metric': metric, 'value': float(value),
                         'max_tokens': MAX_TOKENS, 'commit': commit, 'clock_cap_mhz': cap, 'clock_source': source,
                         'date': date, 'start': start, 'repeat': repeat})
    return rows


def spread_where(out: Path, start: str) -> dict[str, Json]:
    """The `where=` that archive_spread needs for this window's cell definition (WIN2-R2: one max_tokens, one clock cap,
    one production commit)."""
    return {'max_tokens': MAX_TOKENS, 'clock_cap_mhz': wh.start_clock_cap(out, start)[0],
            'commit': wh.production_commit(out)}


def hold_minutes(window: Window, *, last: bool = False) -> int:
    return math.ceil(window.hold_s(last=last) / 60)
