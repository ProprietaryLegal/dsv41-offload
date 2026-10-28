"""WIN2 fairer decision rule (NDM 2026-10-27 "Fairer win rule is essential"; Fable review e8e3e3a).

Pure arithmetic over per-repeat values the runners already record; no host, file or network access except the
explicit archive helpers. One value per repeat per arm. Gains are oriented so that positive always means better.

Rule `win2-v1`, revision 2 (WIN2-R2), per decision cell (its FIRST series is the primary metric; any further series are
guards):
- WIN: point gain > min_gain (default 2 %) AND the 95 % interval of the gain excludes 0 (lower bound > 0).
- Loss-repeat test: the candidate is worse than the loss threshold in at least ceil(2n/3) of n repeats (2 of 3). The
  control spread is 1.69 x the unbiased control SD (s / c4(n)) / mean (S92 FIX 3 item 5): 1.69 sigma is the expected
  range of three values, so it matches the 3 v 3 rule for every number of control repeats. Revision 1 used the range
  (max - min) of all control repeats, which widens as repeats are added: a true 5 % loss was caught 74 % of the time
  at 3 v 3 but 28 % at 8 v 4. The c4 correction makes the expectation exactly 1.69 sigma (plain 1.69 s is 1.50 sigma
  at n = 3) and keeps the no-effect false loss at 4.8 % paired 3 v 3 and 5.8 % unpaired 4 v 3 (1.69 s: 6.2 %, 6.9 %).
  The threshold is that spread for unpaired arms and sqrt(2) x it for paired arms: a paired difference of two single
  values has sqrt(2) x the SD of one value (coordinator decision on S92: paired 8.2 % vs unpaired 3.9 % per-cell false
  loss with the single-value spread).
- Primary LOSS: the loss-repeat test fires.
- Guard LOSS (S92 FIX 3): the guard's loss-repeat test fires AND its whole 95 % interval lies below zero. A guard can make
  a cell LOSS, never WIN. Revision 1 let a guard lose on the loss-repeat test alone, which rejected a no-effect candidate
  in 16-33 % of windows and accepted it in under 2 % ("easy to lose and nearly impossible to win").
- otherwise INCONCLUSIVE.
Window: any decision cell LOSS -> NOT A CANDIDATE; >= 1 WIN and no LOSS -> CANDIDATE; else INCONCLUSIVE.
At most two decision cells per window; every other cell is reported only.
Pair only true adjacent pairs (K/I/K/I alternation, one control and one candidate repeat per pair); C/X/C brackets and any
other layout are unpaired (Series(..., paired=False)).

Interval: paired relative differences use a Student t interval (mean +- t(0.975, n-1) * sd / sqrt(n)); unpaired arms use
Welch's t interval on the difference of means divided by the control mean (floored degrees of freedom, which widens
the interval). A 2-4 sample bootstrap cannot reach 95 % coverage and a sign-flip permutation bound cannot exclude zero
below n = 6, which is why neither is used.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from offload.prefill_maxlen.evidence import Json, number, record

RULE = 'win2-v1'
# Revision 2 (WIN2-R2, S92 FIX 3): guard LOSS also needs the guard's whole 95 % interval below zero; the paired loss
# threshold is sqrt(2) x the control spread. The name stays 'win2-v1' so runner code keeps working; every decision
# records 'rule_revision'.
RULE_REVISION = 2
# Control spread = SPREAD_SD_FACTOR x (s / c4(n)) / mean: d2(3) = 1.693 is the expected range of three normal values in
# units of sigma, and s / c4(n) is the unbiased estimate of sigma, so the spread no longer grows (or shrinks) with the
# number of control repeats (S92 FIX 3 item 5).
SPREAD_SD_FACTOR = 1.69


def c4(n: int) -> float:
    """E[s] / sigma for n normal values: sqrt(2 / (n - 1)) * Gamma(n / 2) / Gamma((n - 1) / 2)."""
    if n < 2:
        raise ValueError('c4 needs at least two values')
    return math.sqrt(2 / (n - 1)) * math.exp(math.lgamma(n / 2) - math.lgamma((n - 1) / 2))
MIN_REPEATS = 3
MAX_DECISION_CELLS = 2
GENESIS = 'genesis'
# v3 (WIN2-R2, S92 FIX 2 items 1-8): every row carries its production commit, the clock cap of its own start, date, start
# and repeat. An older row breaks the chain, so a runner on older code refuses instead of appending rows without them.
ARCHIVE_SCHEMA = 'dsv41-control-archive-v3'
# max_tokens is part of a cell's definition: NBW ran short-b1/32k-b1/32k-b3 at 1,500 tokens, every other runner at 512,
# and at B3 that alone moves ms/round ~72 -> ~81 (shorter answers overlap the other streams' 32K prefills more).
ARCHIVE_REQUIRED = ('window', 'cell', 'metric', 'value', 'max_tokens', 'commit', 'clock_cap_mhz', 'date', 'start', 'repeat')
# clock_cap_mhz: the ONE cap that applied to the row's own start (win2_history.start_clock_cap: both Sparks' "<host> clock
# cap N MHz" lines in that start's log, which start_exp.sh prints only after `nvidia-smi -lgc 0,N` succeeded; without a
# start log, scratch/CLOCK.log by the start's time, clock_source 'clock-log'). Never inferred from the date alone.
CLOCK_SOURCES = ('start-log', 'clock-log')
# clock_caps_mhz (optional) is a window-level audit field only: a list of MHz or 'unknown'.
UNKNOWN_CLOCK = 'unknown'
# Rows of one cell that differ in any of these never pool in a spread; `where=` must select one value.
ARCHIVE_POOL_KEYS = ('max_tokens', 'commit', 'clock_cap_mhz')

# Two-sided 95 % Student t quantiles t(0.975, df) for df 1..30; df > 30 uses the df = 30 value (conservative).
_T975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
         2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)


@dataclass(frozen=True)
class Series:
    cell: str
    metric: str
    higher_is_better: bool
    control: tuple[float, ...]
    candidate: tuple[float, ...]
    paired: bool = True


def t975(df: float) -> float:
    if not math.isfinite(df) or df < 1:
        raise ValueError(f'degrees of freedom must be >= 1, got {df!r}')
    return _T975[min(int(math.floor(df)), len(_T975)) - 1]


def _values(name: str, values: Sequence[float]) -> list[float]:
    result = [number(cast(Json, v)) for v in values]
    if len(result) < MIN_REPEATS:
        raise ValueError(f'{name}: {len(result)} repeats; win2-v1 needs at least {MIN_REPEATS} per arm')
    if any(v <= 0 for v in result):
        raise ValueError(f'{name}: metric values must be positive')
    return result


def _oriented(candidate: float, control: float, higher_is_better: bool) -> float:
    """Relative gain of one candidate value over one control value; positive means better."""
    return (candidate - control) / control if higher_is_better else (control - candidate) / control


def decide_cell(s: Series, *, min_gain: float = 0.02, guard: bool = False) -> dict[str, Json]:
    """One series of one decision cell. guard=True applies the guard rule: LOSS only when the loss-repeat test fires
    AND the whole 95 % interval is below zero; never WIN."""
    control = _values(f'{s.cell}/{s.metric} control', s.control)
    candidate = _values(f'{s.cell}/{s.metric} candidate', s.candidate)
    if not (0 <= min_gain < 1):
        raise ValueError('min_gain must be a fraction in [0, 1)')
    mean_control = statistics.fmean(control)
    spread = SPREAD_SD_FACTOR * statistics.stdev(control) / c4(len(control)) / mean_control
    if s.paired:
        if len(control) != len(candidate):
            raise ValueError(f'{s.cell}/{s.metric}: paired series need equal repeats')
        gains = [_oriented(c, k, s.higher_is_better) for c, k in zip(candidate, control)]
        point = statistics.fmean(gains)
        half = t975(len(gains) - 1) * statistics.stdev(gains) / math.sqrt(len(gains))
        method = f'paired relative-difference Student t, n={len(gains)}'
        per_repeat = gains
        threshold = math.sqrt(2) * spread
    else:
        mean_candidate = statistics.fmean(candidate)
        point = _oriented(mean_candidate, mean_control, s.higher_is_better)
        vc, vk = statistics.variance(candidate) / len(candidate), statistics.variance(control) / len(control)
        se = math.sqrt(vc + vk)
        df = (vc + vk) ** 2 / ((vc ** 2) / (len(candidate) - 1) + (vk ** 2) / (len(control) - 1)) if se > 0 else 1e9
        half = t975(df) * se / mean_control
        method = f'Welch t on mean difference / control mean, n={len(candidate)}+{len(control)}, df={df:.2f} floored'
        per_repeat = [_oriented(c, mean_control, s.higher_is_better) for c in candidate]
        threshold = spread
    interval = (point - half, point + half)
    losses = sum(1 for g in per_repeat if g < -threshold)
    needed = math.ceil(2 * len(per_repeat) / 3)
    loss_test = losses >= needed
    if guard:
        verdict = 'LOSS' if loss_test and interval[1] < 0 else 'INCONCLUSIVE'
    elif loss_test:
        verdict = 'LOSS'
    elif point > min_gain and interval[0] > 0:
        verdict = 'WIN'
    else:
        verdict = 'INCONCLUSIVE'
    return {'cell': s.cell, 'metric': s.metric, 'role': 'guard' if guard else 'primary', 'verdict': verdict,
            'point_gain': point, 'interval95': [interval[0], interval[1]], 'interval_method': method,
            'loss_repeats': losses, 'loss_repeats_needed': needed, 'loss_repeat_test': loss_test,
            'control_spread': spread, 'spread_method': f'{SPREAD_SD_FACTOR} x control SD / c4(n) / control mean',
            'control_range': (max(control) - min(control)) / mean_control, 'loss_threshold': threshold,
            'repeats': len(per_repeat),
            'per_repeat_gain': cast(Json, per_repeat), 'min_gain': min_gain,
            'higher_is_better': s.higher_is_better, 'paired': s.paired}


def decide_decision_cell(series: Sequence[Series], *, min_gain: float = 0.02) -> dict[str, Json]:
    """One decision cell: series[0] is the primary metric, the rest are guards (`decide_window` and
    `win2_history.reanalyze` both use this, so the two never disagree on the same data)."""
    if not series or len({s.cell for s in series}) != 1:
        raise ValueError('a decision cell needs its primary series first, all of one cell')
    if len({s.metric for s in series}) != len(series):
        raise ValueError(f'{series[0].cell}: duplicate metric series')
    results = [decide_cell(s, min_gain=min_gain, guard=i > 0) for i, s in enumerate(series)]
    return {'decides': True, 'verdict': _cell_verdict(results), 'primary_metric': series[0].metric,
            'metrics': {str(r['metric']): r for r in results}}


def _report_only(s: Series) -> dict[str, Json]:
    """Non-decision cells: descriptive gain only, never a verdict (fewer than three repeats allowed)."""
    control = [number(cast(Json, v)) for v in s.control]
    candidate = [number(cast(Json, v)) for v in s.candidate]
    if not control or not candidate:
        raise ValueError(f'{s.cell}/{s.metric}: empty reported series')
    point = _oriented(statistics.fmean(candidate), statistics.fmean(control), s.higher_is_better)
    return {'cell': s.cell, 'metric': s.metric, 'verdict': 'REPORTED', 'point_gain': point,
            'repeats': min(len(control), len(candidate)), 'higher_is_better': s.higher_is_better}


def _cell_verdict(results: Sequence[Mapping[str, Json]]) -> str:
    # Guard results already carry the guard rule (decide_cell guard=True): LOSS only with the interval below zero.
    verdicts = [str(r['verdict']) for r in results]
    if 'LOSS' in verdicts:
        return 'LOSS'
    return 'WIN' if verdicts[0] == 'WIN' else 'INCONCLUSIVE'


def _status(verdicts: Sequence[str]) -> str:
    if 'LOSS' in verdicts:
        return 'NOT A CANDIDATE'
    return 'CANDIDATE' if 'WIN' in verdicts else 'INCONCLUSIVE'


def decide_window(series: Sequence[Series], decision_cells: Sequence[str], *, min_gain: float = 0.02,
                  groups: Mapping[str, Sequence[str]] | None = None) -> dict[str, Json]:
    decision = list(decision_cells)
    if not decision or len(decision) > MAX_DECISION_CELLS or len(set(decision)) != len(decision):
        raise ValueError(f'win2-v1 needs 1..{MAX_DECISION_CELLS} distinct decision cells, got {decision!r}')
    by_cell: dict[str, list[Series]] = {}
    for s in series:
        by_cell.setdefault(s.cell, []).append(s)
    missing = [c for c in decision if c not in by_cell]
    if missing:
        raise ValueError(f'decision cells without measurements: {missing!r}')
    cells: dict[str, Json] = {}
    verdict_of: dict[str, str] = {}
    for cell, rows in by_cell.items():
        if cell in decision:
            decided = decide_decision_cell(rows, min_gain=min_gain)
            verdict_of[cell] = str(decided['verdict'])
            cells[cell] = decided
        else:
            cells[cell] = {'decides': False, 'verdict': 'REPORTED',
                           'metrics': {s.metric: _report_only(s) for s in rows}}
    group_status: dict[str, Json] = {}
    for name, members in (groups or {}).items():
        unknown = [m for m in members if m not in decision]
        if unknown or not members:
            raise ValueError(f'group {name!r} must list decision cells only, got {list(members)!r}')
        group_status[name] = _status([verdict_of[m] for m in members])
    return {'rule': RULE, 'rule_revision': RULE_REVISION, 'status': _status([verdict_of[c] for c in decision]),
            'min_gain': min_gain, 'decision_cells': cast(Json, decision), 'cells': cells, 'groups': group_status}


def _line_sha(line: str) -> str:
    return hashlib.sha256(line.encode()).hexdigest()


def _read_chain(path: Path) -> tuple[list[dict[str, Json]], str]:
    """Return the archive rows and the sha of its last line, refusing a broken chain."""
    rows: list[dict[str, Json]] = []
    prev = GENESIS
    if not path.exists():
        return rows, prev
    for n, line in enumerate(path.read_text().splitlines(), 1):
        row = record(cast(Json, json.loads(line)))
        if row.get('schema') != ARCHIVE_SCHEMA:
            raise ValueError(f'{path}:{n}: control archive chain broken (schema {row.get("schema")!r}, '
                             f'expected {ARCHIVE_SCHEMA!r})')
        if row.get('prev_sha256') != prev:
            raise ValueError(f'{path}:{n}: control archive chain broken')
        rows.append(row)
        prev = _line_sha(line)
    return rows, prev


def archive_head(path: Path) -> dict[str, Json]:
    """{'rows', 'head_sha256'} after verifying the chain. Runners record it before and after appending: the chain alone
    detects an edited line but not a truncated tail (S92 note)."""
    rows, head = _read_chain(path)
    return {'rows': len(rows), 'head_sha256': head}


def _positive_int(value: Json) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def check_row(row: Mapping[str, Json]) -> None:
    """Refuse an archive row without full provenance (S92 FIX 2)."""
    absent = [k for k in ARCHIVE_REQUIRED if k not in row]
    if absent:
        raise ValueError(f'control archive row lacks {absent!r}')
    number(row['value'])
    if not _positive_int(row['max_tokens']):
        raise ValueError(f'control archive max_tokens must be a positive integer, not {row["max_tokens"]!r}')
    commit = row['commit']
    if not (isinstance(commit, str) and re.fullmatch('[0-9a-f]{40}', commit)):
        raise ValueError(f'control archive commit must be the 40-hex PRODUCTION commit, not {commit!r}')
    if not _positive_int(row['clock_cap_mhz']):
        raise ValueError(f'control archive clock_cap_mhz must be the positive integer cap of the row\'s own start, '
                         f'not {row["clock_cap_mhz"]!r}')
    if 'clock_source' in row and row['clock_source'] not in CLOCK_SOURCES:
        raise ValueError(f'control archive clock_source must be one of {CLOCK_SOURCES}, not {row["clock_source"]!r}')
    audit = row.get('clock_caps_mhz', UNKNOWN_CLOCK)
    if audit != UNKNOWN_CLOCK and not (isinstance(audit, list) and all(_positive_int(c) for c in audit)):
        raise ValueError(f'control archive clock_caps_mhz (window audit) must be a list of MHz or {UNKNOWN_CLOCK!r}, '
                         f'not {audit!r}')
    date = row['date']
    if not (isinstance(date, str) and re.fullmatch('20[0-9]{6}', date)):
        raise ValueError(f'control archive date must be YYYYMMDD, not {date!r}')
    start, repeat = row['start'], row['repeat']
    if not (isinstance(start, str) and start.strip()):
        raise ValueError(f'control archive start must name the control start, not {start!r}')
    if not (isinstance(repeat, int) and not isinstance(repeat, bool) and repeat >= 0):
        raise ValueError(f'control archive repeat must be an integer >= 0, not {repeat!r}')


def archive_rows(path: Path) -> list[dict[str, Json]]:
    """Every archived row after verifying the whole chain (empty when the archive does not exist yet)."""
    return _read_chain(path)[0]


def archive_append(path: Path, rows: Sequence[Mapping[str, Json]]) -> None:
    """Append control-cell rows to the append-only, sha-chained archive (exclusive lock; refuses a broken chain)."""
    for row in rows:
        check_row(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        _, prev = _read_chain(path)
        for row in rows:
            line = json.dumps({**row, 'schema': ARCHIVE_SCHEMA, 'prev_sha256': prev}, sort_keys=True, allow_nan=False)
            handle.write(line + '\n')
            prev = _line_sha(line)
        handle.flush()
        os.fsync(handle.fileno())


def archive_spread(path: Path, cell: str, metric: str, *, where: Mapping[str, Json] | None = None) -> float | None:
    """Day-to-day relative spread of one control cell: (max - min) / mean over per-window means; None below two windows.

    `where` keeps only rows whose fields equal every given value (e.g. {'max_tokens': 512, 'clock_cap_mhz': 2150}).
    Rows of one cell name that differ in max_tokens, production commit or clock cap are different cells: pooling them
    refuses unless `where` picks one of each (2150 and 2300 MHz pooled would mostly measure the clock step).
    """
    rows, _ = _read_chain(path)
    per_window: dict[str, list[float]] = {}
    seen: dict[str, set[str]] = {k: set() for k in ARCHIVE_POOL_KEYS}
    for row in rows:
        if row.get('cell') != cell or row.get('metric') != metric:
            continue
        if where is not None and any(row.get(k) != v for k, v in where.items()):
            continue
        for key in ARCHIVE_POOL_KEYS:
            seen[key].add(json.dumps(row.get(key)))
        per_window.setdefault(str(row['window']), []).append(number(row['value']))
    mixed = {k: sorted(v) for k, v in seen.items() if len(v) > 1}
    if mixed:
        raise ValueError(f'control archive {cell}/{metric} mixes {mixed}; pass where= with one value of each '
                         "(e.g. {'max_tokens': 512, 'commit': '<40-hex>', 'clock_cap_mhz': 2300})")
    if len(per_window) < 2:
        return None
    means = [statistics.fmean(v) for v in per_window.values()]
    return (max(means) - min(means)) / statistics.fmean(means)


def archive_spreads(path: Path, cell: str, metric: str) -> dict[str, float | None]:
    """The spread of one cell/metric per (max_tokens, clock cap, commit) definition present in the archive (reporting)."""
    groups = sorted({(int(number(r['max_tokens'])), int(number(r['clock_cap_mhz'])), str(r['commit']))
                     for r in _read_chain(path)[0] if r.get('cell') == cell and r.get('metric') == metric})
    return {f'max_tokens={t} clock_cap_mhz={c} commit={k[:12]}':
            archive_spread(path, cell, metric, where={'max_tokens': t, 'clock_cap_mhz': c, 'commit': k})
            for t, c, k in groups}
