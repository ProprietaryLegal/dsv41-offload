"""R-SP prefill window (WIN2 rule win2-v1, chain mode): the registered plan, budget and the decision, archive and close
wiring. Pure functions: no host, file or network access except decision_v2's archive helpers called by the runner.

Design (offload/sp_prefill/WINDOW.md): C/X/C, three starts of the production launch, the X start adding the R-SP overlay
and VLLM_DSV41_PREFILL_SP=1 (offload/launch/configs/RSP-window.env, RSP_SP=1). Decision cells: cold 32K and 150K B1
first-token time, three repeats in every start; candidate repeat r is paired with the mean of the two bracketing
controls' repeat r. Reported only: short B1 decode and the mixed 150K + 2 decoders cell, once per start. Cell
definitions, prompts, budgets and per-cell metrics are PC1's chunk-window cells (offload/pc1/cells.py CHUNK_PLAN, same
public prompts, 512 tokens), so the archive pools R-SP's control rows with PC1's chunk windows'. Greedy: the six W5
prompts on the X start, registered near-tie 0.9 nats (NDM 2026-10-27); expected IDENTICAL by construction (CPU proof),
any divergence is reported with its first differing token. Memory: the shared 4 GiB MemAvailable floor (production chunk
size; NDM's 3 GiB floor is for PC1's larger chunks only).
"""
from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from offload.launch.windows import chain
from offload.launch.windows import decision_v2 as d2
from offload.prefill_maxlen.evidence import Json

CONFIG = 'offload/launch/configs/RSP-window.env'
RULE_NOTE = ('win2-v1 via offload.launch.windows.decision_v2: WIN = point gain >2% and 95% interval excludes 0; '
             'LOSS = worse than own control spread in 2 of 3 repeats; at most two decision cells')
GREEDY_POLICY = 'rsp-near-tie-0p9'
GREEDY_NEAR_TIE_NATS = 0.9 + 1e-3  # NDM 2026-10-27 "You can loosen the nats to .9" (same registration as PC1 / TV1 r10)
MEM_FLOOR_KB = 4 * 1024**2  # shared floor; production chunk size
MAX_TOKENS = 512  # PC1 cells' budget; part of every archive row's cell definition

Role = Literal['control', 'candidate']


@dataclass(frozen=True)
class Arm:
    name: str
    role: Role

    @property
    def env(self) -> dict[str, str]:
        return {'RSP_SP': '1' if self.role == 'candidate' else '0'}


ARMS = (Arm('c1', 'control'), Arm('x1', 'candidate'), Arm('c2', 'control'))
DECISION = ('32k-b1', '150k-b1')
REPORTED = ('short-b1', 'mixed150k-2decode')
GROUPS: Mapping[str, tuple[str, ...]] = {'single-user': DECISION}
DECISION_REPEATS, REPORTED_REPEATS = 3, 1
# Per-cell seconds (PC1 ESTIMATES: MEASURED-derived cold TTFT + 512-token decode), start-up and fixed costs (PC1 window).
CELL_S = {'short-b1': 20, '32k-b1': 35, '150k-b1': 100, 'mixed150k-2decode': 145}
STARTUP_S, WARMUP_S, TRANSITION_S, GREEDY_S = 404, 90, 60, 60
SETUP_S, RECOVERY_S, QUIET_PROOF_S, POST_CLOSE_WATCH_S = 120, 720, 120, 15 * 60
MICROBENCH_S = 300  # rs_bench.py on both Sparks with the Sparks quiet, before c1 (ESTIMATED)
# Primary metric first (only it can WIN); cells with a suffix are reported, never decide.
METRICS: Mapping[str, tuple[tuple[str, str, bool], ...]] = {
    '32k-b1': (('32k-b1', 'ttft_s', False), ('32k-b1:server', 'committed_tps', True)),
    '150k-b1': (('150k-b1', 'ttft_s', False), ('150k-b1:server', 'committed_tps', True)),
    'short-b1': (('short-b1', 'committed_tps', True), ('short-b1', 'ms_per_round', False)),
    'mixed150k-2decode': (('mixed150k-2decode', 'mean_completion_s', False),
                          ('mixed150k-2decode:long-doc', 'ttft_s', False)),
}


def repeats(cell: str) -> int:
    if cell in DECISION:
        return DECISION_REPEATS
    if cell in REPORTED:
        return REPORTED_REPEATS
    raise ValueError(f'R-SP: cell outside the plan: {cell}')


def start_seconds() -> int:
    return STARTUP_S + WARMUP_S + TRANSITION_S + GREEDY_S + sum(CELL_S[c] * repeats(c) for c in (*DECISION, *REPORTED))


def hold_seconds(chained_success: bool) -> int:
    """ESTIMATED HOLD: setup, the in-window microbenchmark, three starts, then either the chained close (quiet proof,
    production left down) or the strict W8 restore and the post-close watch."""
    body = SETUP_S + MICROBENCH_S + len(ARMS) * start_seconds()
    return body + (QUIET_PROOF_S if chained_success else RECOVERY_S + POST_CLOSE_WATCH_S)


Measurements = Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]]  # arm -> cell -> metric -> per repeat


def series(measured: Measurements) -> tuple[list[d2.Series], dict[str, Json]]:
    """decision_v2 Series (candidate repeat r paired with the mean of c1 and c2 repeat r) and the bracket spreads."""
    if set(measured) != {a.name for a in ARMS}:
        raise ValueError(f'R-SP: measurements for arms {sorted(measured)}, need {[a.name for a in ARMS]}')
    out: list[d2.Series] = []
    brackets: dict[str, Json] = {}
    for cell in (*DECISION, *REPORTED):
        n = repeats(cell)
        for series_cell, metric, higher in METRICS[cell]:
            first, cand, last = (list(measured[arm][cell][metric]) for arm in ('c1', 'x1', 'c2'))
            if not (len(first) == len(cand) == len(last) == n):
                raise ValueError(f'R-SP: {cell}/{metric} needs {n} repeats in every start')
            control = tuple((a + c) / 2 for a, c in zip(first, last))
            out.append(d2.Series(cell=series_cell, metric=metric, higher_is_better=higher, control=control,
                                 candidate=tuple(cand), paired=True))
            brackets[f'{series_cell}/{metric}'] = {
                'first_control': cast(Json, first), 'candidate': cast(Json, cand), 'last_control': cast(Json, last),
                'bracket_spread_per_repeat': cast(Json, [abs(a - c) / ((a + c) / 2) for a, c in zip(first, last)])}
    return out, brackets


def decide(measured: Measurements, greedy: Mapping[str, Json]) -> dict[str, Json]:
    """The window's summary `decision`. A greedy result other than IDENTICAL / NEAR-TIE makes it NOT A CANDIDATE."""
    rows, brackets = series(measured)
    result = d2.decide_window(rows, DECISION, groups=dict(GROUPS))
    verdict = str(greedy.get('verdict'))
    if verdict not in ('IDENTICAL', 'NEAR-TIE'):
        result['status'] = 'NOT A CANDIDATE'
        result['greedy_stop'] = f'greedy {verdict} under {GREEDY_POLICY}'
    result.update(brackets=brackets, note=RULE_NOTE, greedy_policy=GREEDY_POLICY, greedy=cast(Json, dict(greedy)),
                  promotion='CANDIDATE is a measured speed verdict for review; promotion is a separate coordinator step')
    return result


def archive_rows(window: str, measured: Measurements, *, commit: str, clock_cap_mhz: Mapping[str, int],
                 date: str) -> list[dict[str, Json]]:
    """Every control repeat (c1 and c2, every cell and metric) for decision_v2.archive_append. Provenance per S92 FIX 2
    (WIN2-R2): the 40-hex PRODUCTION commit (never the experiment's), the date, and the clock cap of the row's own start
    (both Sparks' "clock cap N MHz" lines of that start's log agree), plus max_tokens, start and repeat."""
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError(f'R-SP archive: production commit must be 40 hex, got {commit!r}')
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date):
        raise ValueError(f'R-SP archive: date must be YYYY-MM-DD, got {date!r}')
    rows: list[dict[str, Json]] = []
    for arm in ('c1', 'c2'):
        cap = clock_cap_mhz.get(arm)
        if not (isinstance(cap, int) and not isinstance(cap, bool) and cap > 0):
            raise ValueError(f'R-SP archive: start {arm} needs its own clock cap, got {cap!r}')
        for cell, metrics in measured.get(arm, {}).items():
            for metric, values in metrics.items():
                for repeat, value in enumerate(values):
                    rows.append({'window': window, 'cell': cell, 'metric': metric, 'value': float(value),
                                 'max_tokens': MAX_TOKENS, 'start': arm, 'repeat': repeat, 'commit': commit,
                                 'clock_cap_mhz': cap, 'date': date, 'runner': 'R-SP', 'cells': 'pc1-chunk'})
    return rows


def spreads(archive: Path, *, commit: str, clock_cap_mhz: int, cells: Sequence[str] = DECISION) -> dict[str, Json]:
    """Day-to-day control spread of each decision cell's primary metric for this window's cell definition (max_tokens,
    production commit, clock cap); None until the archive holds two such windows."""
    where: dict[str, Json] = {'max_tokens': MAX_TOKENS, 'commit': commit, 'clock_cap_mhz': clock_cap_mhz}
    return {cell: d2.archive_spread(archive, cell, METRICS[cell][0][1], where=where) for cell in cells}


def memory_ok(min_available_kb: Mapping[str, int]) -> bool:
    """Every Spark kept MemAvailable >= the 4 GiB floor through the arm (1 s telemetry minima)."""
    return bool(min_available_kb) and all(kb >= MEM_FLOOR_KB for kb in min_available_kb.values())


def close(stopped: bool) -> dict[str, Json]:
    """The runner's end-of-window branch (WIN2 chain rule). 'CHAINED': stop the experiment, prove the Sparks quiet,
    chain.finish_chained(out, proof), close the HOLD, no W8 restore and no post-close watch. 'RESTORE': strict W8 restore,
    post-close watch, then chain.clear()."""
    if chain.skip_restore(stopped):
        return {'chain': 'CHAINED', 'restore': {'status': 'CHAINED: production left down'}, 'post_close_watch': False}
    return {'chain': 'LAST' if chain.enabled() else 'OFF', 'restore': {'status': 'W8 strict restore'},
            'post_close_watch': True, 'clear_marker': True}


def preflight(production_serving: bool, hold_exists: bool) -> dict[str, Json]:
    """'chained': production is down from the previous chained window; c1 is the baseline (no identity snapshot of a
    running production server). 'production': the usual production identity snapshot first."""
    mode = chain.preflight(production_serving, hold_exists)
    return {'mode': mode, 'baseline': 'c1 control start' if mode == 'chained' else 'production identity snapshot'}


def plan() -> dict[str, Json]:
    return {
        'config': CONFIG, 'arms': [{'name': a.name, 'role': a.role, 'env': cast(Json, a.env)} for a in ARMS],
        'decision_cells': list(DECISION), 'reported_cells': list(REPORTED), 'groups': {k: list(v) for k, v in GROUPS.items()},
        'repeats': {c: repeats(c) for c in (*DECISION, *REPORTED)}, 'max_tokens': MAX_TOKENS,
        'greedy': {'policy': GREEDY_POLICY, 'near_tie_nats': GREEDY_NEAR_TIE_NATS, 'arm': 'x1'},
        'memory_floor_kb': MEM_FLOOR_KB, 'microbenchmark': 'offload/sp_prefill/rs_bench.py before c1, Sparks quiet',
        'estimated_s': {'start': start_seconds(), 'hold_chained': hold_seconds(True), 'hold_last': hold_seconds(False)},
        'rule': d2.RULE,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true', required=True, help='print the registered plan (no side effects)')
    parser.parse_args(argv)
    print(json.dumps(plan(), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
