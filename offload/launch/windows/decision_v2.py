"""WIN2 fairer decision rule (NDM 2026-10-27 "Fairer win rule is essential"; Fable review e8e3e3a).

Pure arithmetic over per-repeat values the runners already record; no host, file or network access except the
explicit archive helpers. One value per repeat per arm. Gains are oriented so that positive always means better.

Rule `win2-v1`, per decision cell (its FIRST series is the primary metric; any further series are guards):
- WIN: point gain > min_gain (default 2 %) AND the 95 % interval of the gain excludes 0 (lower bound > 0).
- LOSS: the candidate is worse than the cell's own control spread in at least ceil(2n/3) of n repeats (2 of 3).
- otherwise INCONCLUSIVE. A guard metric can make a cell LOSS, never WIN.
Window: any decision cell LOSS -> NOT A CANDIDATE; >= 1 WIN and no LOSS -> CANDIDATE; else INCONCLUSIVE.
At most two decision cells per window; every other cell is reported only.

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
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from offload.prefill_maxlen.evidence import Json, number, record

RULE = 'win2-v1'
MIN_REPEATS = 3
MAX_DECISION_CELLS = 2
GENESIS = 'genesis'
ARCHIVE_SCHEMA = 'dsv41-control-archive-v1'
# max_tokens is part of a cell's definition: NBW ran short-b1/32k-b1/32k-b3 at 1,500 tokens, every other runner at 512,
# and at B3 that alone moves ms/round ~72 -> ~81 (shorter answers overlap the other streams' 32K prefills more).
ARCHIVE_REQUIRED = ('window', 'cell', 'metric', 'value', 'max_tokens')

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


def decide_cell(s: Series, *, min_gain: float = 0.02) -> dict[str, Json]:
    control = _values(f'{s.cell}/{s.metric} control', s.control)
    candidate = _values(f'{s.cell}/{s.metric} candidate', s.candidate)
    if not (0 <= min_gain < 1):
        raise ValueError('min_gain must be a fraction in [0, 1)')
    mean_control = statistics.fmean(control)
    spread = (max(control) - min(control)) / mean_control
    if s.paired:
        if len(control) != len(candidate):
            raise ValueError(f'{s.cell}/{s.metric}: paired series need equal repeats')
        gains = [_oriented(c, k, s.higher_is_better) for c, k in zip(candidate, control)]
        point = statistics.fmean(gains)
        half = t975(len(gains) - 1) * statistics.stdev(gains) / math.sqrt(len(gains))
        method = f'paired relative-difference Student t, n={len(gains)}'
        per_repeat = gains
    else:
        mean_candidate = statistics.fmean(candidate)
        point = _oriented(mean_candidate, mean_control, s.higher_is_better)
        vc, vk = statistics.variance(candidate) / len(candidate), statistics.variance(control) / len(control)
        se = math.sqrt(vc + vk)
        df = (vc + vk) ** 2 / ((vc ** 2) / (len(candidate) - 1) + (vk ** 2) / (len(control) - 1)) if se > 0 else 1e9
        half = t975(df) * se / mean_control
        method = f'Welch t on mean difference / control mean, n={len(candidate)}+{len(control)}, df={df:.2f} floored'
        per_repeat = [_oriented(c, mean_control, s.higher_is_better) for c in candidate]
    interval = (point - half, point + half)
    losses = sum(1 for g in per_repeat if g < -spread)
    needed = math.ceil(2 * len(per_repeat) / 3)
    if losses >= needed:
        verdict = 'LOSS'
    elif point > min_gain and interval[0] > 0:
        verdict = 'WIN'
    else:
        verdict = 'INCONCLUSIVE'
    return {'cell': s.cell, 'metric': s.metric, 'verdict': verdict, 'point_gain': point,
            'interval95': [interval[0], interval[1]], 'interval_method': method,
            'loss_repeats': losses, 'loss_repeats_needed': needed, 'control_spread': spread,
            'repeats': len(per_repeat), 'per_repeat_gain': cast(Json, per_repeat), 'min_gain': min_gain,
            'higher_is_better': s.higher_is_better, 'paired': s.paired}


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
            results = [decide_cell(s, min_gain=min_gain) for s in rows]
            verdict_of[cell] = _cell_verdict(results)
            cells[cell] = {'decides': True, 'verdict': verdict_of[cell], 'primary_metric': rows[0].metric,
                           'metrics': {str(r['metric']): r for r in results}}
        else:
            cells[cell] = {'decides': False, 'verdict': 'REPORTED',
                           'metrics': {s.metric: _report_only(s) for s in rows}}
    group_status: dict[str, Json] = {}
    for name, members in (groups or {}).items():
        unknown = [m for m in members if m not in decision]
        if unknown or not members:
            raise ValueError(f'group {name!r} must list decision cells only, got {list(members)!r}')
        group_status[name] = _status([verdict_of[m] for m in members])
    return {'rule': RULE, 'status': _status([verdict_of[c] for c in decision]), 'min_gain': min_gain,
            'decision_cells': cast(Json, decision), 'cells': cells, 'groups': group_status}


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
        if row.get('schema') != ARCHIVE_SCHEMA or row.get('prev_sha256') != prev:
            raise ValueError(f'{path}:{n}: control archive chain broken')
        rows.append(row)
        prev = _line_sha(line)
    return rows, prev


def archive_rows(path: Path) -> list[dict[str, Json]]:
    """Every archived row after verifying the whole chain (empty when the archive does not exist yet)."""
    return _read_chain(path)[0]


def archive_append(path: Path, rows: Sequence[Mapping[str, Json]]) -> None:
    """Append control-cell rows to the append-only, sha-chained archive (exclusive lock; refuses a broken chain)."""
    for row in rows:
        absent = [k for k in ARCHIVE_REQUIRED if k not in row]
        if absent:
            raise ValueError(f'control archive row lacks {absent!r}')
        number(row['value'])
        tokens = row['max_tokens']
        if not (isinstance(tokens, int) and not isinstance(tokens, bool) and tokens > 0):
            raise ValueError(f'control archive max_tokens must be a positive integer, not {tokens!r}')
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

    `where` keeps only rows whose fields equal every given value (e.g. {'max_tokens': 512}). Rows of one cell name with
    different max_tokens are different cells: pooling them refuses unless `where` picks one.
    """
    rows, _ = _read_chain(path)
    per_window: dict[str, list[float]] = {}
    tokens: set[str] = set()
    for row in rows:
        if row.get('cell') != cell or row.get('metric') != metric:
            continue
        if where is not None and any(row.get(k) != v for k, v in where.items()):
            continue
        tokens.add(json.dumps(row.get('max_tokens')))
        per_window.setdefault(str(row['window']), []).append(number(row['value']))
    if len(tokens) > 1:
        raise ValueError(f'control archive {cell}/{metric} mixes max_tokens {sorted(tokens)}; '
                         "pass where={'max_tokens': N}")
    if len(per_window) < 2:
        return None
    means = [statistics.fmean(v) for v in per_window.values()]
    return (max(means) - min(means)) / statistics.fmean(means)
