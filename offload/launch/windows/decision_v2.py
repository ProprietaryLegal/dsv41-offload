"""WIN2 fairer decision rule (NDM 2026-10-27 "Fairer win rule is essential"; Fable review e8e3e3a).

Pure arithmetic over per-repeat values the runners already record; no host, file or network access except the
explicit archive helpers. One value per repeat per arm. Gains are oriented so that positive always means better.

Rule `win2-v1`, revision 3 (WIN2-R3 item 0; revision 2 was WIN2-R2), per decision cell (its FIRST series is the primary
metric; any further series are guards):
- WIN: point gain > min_gain (default 2 %) AND the 95 % interval of the gain excludes 0 (lower bound > 0).
- Loss-repeat test: the candidate is worse than the loss threshold in at least ceil(2n/3) of n repeats (2 of 3). The
  control spread is 1.69 x the unbiased control SD (s / c4(n)) / mean (S92 FIX 3 item 5): 1.69 sigma is the expected
  range of three values, so it matches the 3 v 3 rule for every number of control repeats. Revision 1 used the range
  (max - min) of all control repeats, which widens as repeats are added: a true 5 % loss was caught 74 % of the time
  at 3 v 3 but 28 % at 8 v 4. The c4 correction makes the expectation exactly 1.69 sigma (plain 1.69 s is 1.50 sigma
  at n = 3); the loss-repeat test alone fires on a no-effect cell 4.8 % paired 3 v 3 and 5.8 % unpaired 4 v 3
  (1.69 s: 6.2 %, 6.9 %).
  The threshold is that spread for unpaired arms and sqrt(2) x it for paired arms: a paired difference of two single
  values has sqrt(2) x the SD of one value (coordinator decision on S92: paired 8.2 % vs unpaired 3.9 % per-cell false
  loss with the single-value spread).
- Primary LOSS (revision 3, coordinator decision on WIN2-R2's report): the loss-repeat test fires AND the whole 95 %
  interval lies below zero, exactly as for the guards. Revision 2's primary lost on the loss-repeat test alone, so a
  no-effect candidate was rejected in 11-15 % of windows and accepted in under 2 %; revision 3 makes it 4.0 % / 1.9 %
  (paired 3 v 3) and 6.2 % / 2.0 % (unpaired 4 v 3), with guards (WIN2.md has the power table).
- Guard LOSS (S92 FIX 3): the guard's loss-repeat test fires AND its whole 95 % interval lies below zero. A guard can make
  a cell LOSS, never WIN. Revision 1 let a guard lose on the loss-repeat test alone, which rejected a no-effect candidate
  in 16-33 % of windows and accepted it in under 2 % ("easy to lose and nearly impossible to win").
- otherwise INCONCLUSIVE.
Window: any decision cell LOSS -> NOT A CANDIDATE; >= 1 WIN and no LOSS -> CANDIDATE; else INCONCLUSIVE.
At most two decision cells per window; every other cell is reported only.
Pair only true adjacent pairs (K/I/K/I alternation, one control and one candidate repeat per pair); C/X/C brackets and any
other layout are unpaired (Series(..., paired=False)).

Prompt identity (WIN2-R3, DP-2 98aa385): acceptance is dominated by the prompt (same-day prompts differ by up to 0.59
accepted per round; each prompt holds steady over three days), so values of different prompts are never pooled or
paired. Every decision series carries the prompt identity of each value (`prompt_identity` of the request's stream
hashes); a decision cell refuses values without it or of more than one identity, and every archive row carries it.

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
# threshold is sqrt(2) x the control spread. Revision 3 (WIN2-R3 item 0): the primary LOSS needs it too. The name stays
# 'win2-v1' so runner code keeps working; every decision records 'rule_revision'.
RULE_REVISION = 3
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
# and repeat. v4 (WIN2-R3, DP-2): and the prompt identity of its request (`prompt_sha256`; `prompt_members` lists the
# stream hashes at B >= 2). An older row breaks the chain, so a runner on older code refuses instead of appending rows
# without them.
ARCHIVE_SCHEMA = 'dsv41-control-archive-v4'
# max_tokens is part of a cell's definition: NBW ran short-b1/32k-b1/32k-b3 at 1,500 tokens, every other runner at 512,
# and at B3 that alone moves ms/round ~72 -> ~81 (shorter answers overlap the other streams' 32K prefills more).
ARCHIVE_REQUIRED = ('window', 'cell', 'metric', 'value', 'max_tokens', 'commit', 'clock_cap_mhz', 'date', 'start', 'repeat',
                    'prompt_sha256')
# clock_cap_mhz: the ONE cap that applied to the row's own start (win2_history.start_clock_cap: both Sparks' "<host> clock
# cap N MHz" lines in that start's log, which start_exp.sh prints only after `nvidia-smi -lgc 0,N` succeeded; without a
# start log, scratch/CLOCK.log by the start's time, clock_source 'clock-log'). Never inferred from the date alone.
CLOCK_SOURCES = ('start-log', 'clock-log')
# clock_caps_mhz (optional) is a window-level audit field only: a list of MHz or 'unknown'.
UNKNOWN_CLOCK = 'unknown'
# Rows of one cell that differ in any of these never pool in a spread; `where=` must select one value.
ARCHIVE_POOL_KEYS = ('max_tokens', 'commit', 'clock_cap_mhz', 'prompt_sha256')
SHA256 = re.compile('[0-9a-f]{64}')

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
    # The prompt identity of every value (one per value, in order). Required in decision cells: values of different
    # prompts are never pooled or paired (DP-2).
    control_prompts: tuple[str, ...] = ()
    candidate_prompts: tuple[str, ...] = ()


def prompt_identity(stream_sha256s: Sequence[str]) -> str:
    """One identity for one request (B1: its prompt sha256) or one concurrent group (B >= 2: sha256 of the sorted member
    hashes joined by '|'). Refuses anything that is not a 64-hex sha256."""
    members = list(stream_sha256s)
    if not members or any(not (isinstance(m, str) and SHA256.fullmatch(m)) for m in members):
        raise ValueError(f'prompt identity needs 64-hex prompt sha256s, got {members!r}')
    return members[0] if len(members) == 1 else hashlib.sha256('|'.join(sorted(members)).encode()).hexdigest()


def same_prompt(name: str, *groups: Sequence[str], counts: Sequence[int] | None = None,
                required: bool = True) -> str | None:
    """The ONE prompt identity shared by every value of every group; refuses values of different prompts (never pooled or
    paired), malformed identities, a length that differs from its value count, and (required) missing identities."""
    if all(not g for g in groups):
        if required:
            raise ValueError(f'{name}: no prompt sha256 per value; win2 never pools or pairs values without proving they '
                             'came from one prompt (DP-2: acceptance is dominated by prompt identity)')
        return None
    if counts is not None and [len(g) for g in groups] != list(counts):
        raise ValueError(f'{name}: one prompt sha256 per value required, got {[len(g) for g in groups]} for {list(counts)}')
    found = {p for g in groups for p in g}
    if any(not (isinstance(p, str) and SHA256.fullmatch(p)) for p in found):
        raise ValueError(f'{name}: prompt identities must be 64-hex sha256, got {sorted(map(str, found))}')
    if len(found) != 1:
        raise ValueError(f'{name}: values come from {len(found)} different prompts ({", ".join(sorted(x[:12] for x in found))}); '
                         'win2 never pools or pairs across prompt hashes')
    return found.pop()


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
    """One series of one decision cell. LOSS (primary and guard alike, revision 3) only when the loss-repeat test fires
    AND the whole 95 % interval is below zero. guard=True never gives WIN."""
    control = _values(f'{s.cell}/{s.metric} control', s.control)
    candidate = _values(f'{s.cell}/{s.metric} candidate', s.candidate)
    if not (0 <= min_gain < 1):
        raise ValueError('min_gain must be a fraction in [0, 1)')
    prompt = same_prompt(f'{s.cell}/{s.metric}', s.control_prompts, s.candidate_prompts,
                         counts=(len(control), len(candidate)), required=False)
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
    if loss_test and interval[1] < 0:
        verdict = 'LOSS'
    elif guard:
        verdict = 'INCONCLUSIVE'
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
            'higher_is_better': s.higher_is_better, 'paired': s.paired, 'prompt_sha256': prompt}


def decide_decision_cell(series: Sequence[Series], *, min_gain: float = 0.02) -> dict[str, Json]:
    """One decision cell: series[0] is the primary metric, the rest are guards (`decide_window` and
    `win2_history.reanalyze` both use this, so the two never disagree on the same data)."""
    if not series or len({s.cell for s in series}) != 1:
        raise ValueError('a decision cell needs its primary series first, all of one cell')
    if len({s.metric for s in series}) != len(series):
        raise ValueError(f'{series[0].cell}: duplicate metric series')
    # DP-2: one prompt identity for every value of every series of the cell, proved (never assumed).
    prompt = same_prompt(series[0].cell, *(g for s in series for g in (s.control_prompts, s.candidate_prompts)),
                         counts=[n for s in series for n in (len(s.control), len(s.candidate))])
    results = [decide_cell(s, min_gain=min_gain, guard=i > 0) for i, s in enumerate(series)]
    return {'decides': True, 'verdict': _cell_verdict(results), 'primary_metric': series[0].metric,
            'prompt_sha256': prompt, 'metrics': {str(r['metric']): r for r in results}}


def _report_only(s: Series) -> dict[str, Json]:
    """Non-decision cells: descriptive gain only, never a verdict (fewer than three repeats allowed)."""
    control = [number(cast(Json, v)) for v in s.control]
    candidate = [number(cast(Json, v)) for v in s.candidate]
    if not control or not candidate:
        raise ValueError(f'{s.cell}/{s.metric}: empty reported series')
    prompt = same_prompt(f'{s.cell}/{s.metric}', s.control_prompts, s.candidate_prompts,
                         counts=(len(control), len(candidate)), required=False)
    point = _oriented(statistics.fmean(candidate), statistics.fmean(control), s.higher_is_better)
    return {'cell': s.cell, 'metric': s.metric, 'verdict': 'REPORTED', 'point_gain': point,
            'repeats': min(len(control), len(candidate)), 'higher_is_better': s.higher_is_better, 'prompt_sha256': prompt}


def _cell_verdict(results: Sequence[Mapping[str, Json]]) -> str:
    # Every result already carries the LOSS rule (revision 3: loss-repeat test AND the interval below zero).
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
    prompt = row['prompt_sha256']
    if not (isinstance(prompt, str) and SHA256.fullmatch(prompt)):
        raise ValueError(f'control archive prompt_sha256 must be the 64-hex prompt identity, not {prompt!r}')
    if 'prompt_members' in row:
        members = row['prompt_members']
        if not isinstance(members, list) or prompt_identity([str(m) for m in members]) != prompt:
            raise ValueError(f'control archive prompt_members {members!r} do not give prompt_sha256 {prompt}')


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
    Rows of one cell name that differ in max_tokens, production commit, clock cap or prompt identity are different cells:
    pooling them refuses unless `where` picks one of each (2150 and 2300 MHz pooled would mostly measure the clock step;
    two prompts pooled would mostly measure the prompts, DP-2).
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
                         "(e.g. {'max_tokens': 512, 'commit': '<40-hex>', 'clock_cap_mhz': 2300, "
                         "'prompt_sha256': '<64-hex>'})")
    if len(per_window) < 2:
        return None
    means = [statistics.fmean(v) for v in per_window.values()]
    return (max(means) - min(means)) / statistics.fmean(means)


def archive_spreads(path: Path, cell: str, metric: str) -> dict[str, float | None]:
    """The spread of one cell/metric per (max_tokens, clock cap, commit, prompt) definition in the archive (reporting)."""
    groups = sorted({(int(number(r['max_tokens'])), int(number(r['clock_cap_mhz'])), str(r['commit']),
                      str(r['prompt_sha256']))
                     for r in _read_chain(path)[0] if r.get('cell') == cell and r.get('metric') == metric})
    return {f'max_tokens={t} clock_cap_mhz={c} commit={k[:12]} prompt={q[:12]}':
            archive_spread(path, cell, metric, where={'max_tokens': t, 'clock_cap_mhz': c, 'commit': k, 'prompt_sha256': q})
            for t, c, k, q in groups}


# ---------------------------------------------------------------- WIN2-R3: seeded paired cells (F-METHOD §3.1)
# Rule `win2-paired-v1`, per decision cell of seeded pairs (window_cells.run_seeded_cell; the pair unit is one seeded
# request at B1 and one group of B concurrent seeded requests at B >= 2):
# - n >= 6 pairs; d_i = oriented relative difference of candidate pair i against control pair i (with two bracketing
#   control starts, control pair i is the mean of C1_i and C2_i).
# - Paired Student t 95 % interval of mean(d) AND the exact two-sided sign-flip p of mean(d) (6 pairs: smallest p is
#   2/64 = 0.031, so every pair must agree; 3 pairs cannot reach 0.05).
# - Primary WIN: point > min_gain, lower bound > 0 and p < 0.05. LOSS (primary or guard): upper bound < 0 and p < 0.05.
#   Otherwise INCONCLUSIVE. Guards never WIN. The rule is symmetric apart from the 2 % effect a WIN must also clear.
# - Built-in A/A: C1 vs C2 of the same seeded cell. Drift (A/A upper < 0 or lower > 0 with A/A p < 0.05) makes the cell
#   INCONCLUSIVE; for a target-side lever (expect_identical) the two control starts must also produce identical token
#   streams and round counts (window_cells.streams_identical), else INCONCLUSIVE (start nondeterminism).
# - Every decision cell carries a profile ('single' = B1, 'multi' = B >= 2); the window reports one verdict per profile,
#   with at most two decision cells per profile. Promotions state which profile they rely on.
PAIRED_RULE = 'win2-paired-v1'
MIN_PAIRS = 6
MAX_SIGN_FLIP_PAIRS = 20
ALPHA = 0.05
PROFILES = ('single', 'multi')


@dataclass(frozen=True)
class PairedSeries:
    cell: str
    metric: str
    higher_is_better: bool
    control: tuple[float, ...]
    candidate: tuple[float, ...]
    profile: str
    aa: tuple[tuple[float, ...], tuple[float, ...]] | None = None
    # The ONE prompt identity of every pair in every arm (seeded_series proves it from the seeded results; DP-2).
    prompt_sha256: str | None = None


def sign_flip_p(diffs: Sequence[float]) -> float:
    """Exact two-sided sign-flip (paired permutation) p of the mean: the share of the 2^n sign assignments whose
    |sum| is at least the observed |sum| (ties count, so the test is never anti-conservative)."""
    values = [float(v) for v in diffs]
    if not values or len(values) > MAX_SIGN_FLIP_PAIRS or not all(math.isfinite(v) for v in values):
        raise ValueError(f'sign-flip test needs 1..{MAX_SIGN_FLIP_PAIRS} finite differences')
    observed = abs(sum(values))
    sums = [0.0]
    for v in values:
        sums = [x + v for x in sums] + [x - v for x in sums]
    tolerance = 1e-12 * max(1.0, sum(abs(v) for v in values))
    return sum(1 for x in sums if abs(x) >= observed - tolerance) / len(sums)


def _paired_stats(control: Sequence[float], candidate: Sequence[float], higher: bool) -> tuple[list[float], float, float, float, float]:
    diffs = [_oriented(x, c, higher) for c, x in zip(control, candidate)]
    point = statistics.fmean(diffs)
    half = t975(len(diffs) - 1) * statistics.stdev(diffs) / math.sqrt(len(diffs))
    return diffs, point, point - half, point + half, sign_flip_p(diffs)


def decide_paired_cell(s: PairedSeries, *, min_gain: float = 0.02, guard: bool = False,
                       expect_identical: bool = False, streams_identical: bool | None = None,
                       start_sd: float = 0.0) -> dict[str, Json]:
    """`start_sd`: registered between-start relative SD of the metric (MEASURED by the seeded A/A pilot). Pairs share one
    candidate start, so a start offset is indistinguishable from an effect; a non-zero start_sd widens the interval by
    start_sd^2 x (1 + 1/k) (k = control starts). The sign-flip test is within-start and is not widened."""
    if s.profile not in PROFILES:
        raise ValueError(f'{s.cell}: profile must be one of {PROFILES}, got {s.profile!r}')
    if s.prompt_sha256 is not None and not SHA256.fullmatch(s.prompt_sha256):
        raise ValueError(f'{s.cell}/{s.metric}: prompt_sha256 must be a 64-hex identity, not {s.prompt_sha256!r}')
    control = _values(f'{s.cell}/{s.metric} control', s.control)
    candidate = _values(f'{s.cell}/{s.metric} candidate', s.candidate)
    if len(control) != len(candidate) or len(control) < MIN_PAIRS:
        raise ValueError(f'{s.cell}/{s.metric}: {PAIRED_RULE} needs at least {MIN_PAIRS} complete pairs per arm, '
                         f'got {len(control)} control and {len(candidate)} candidate')
    if not (0 <= min_gain < 1):
        raise ValueError('min_gain must be a fraction in [0, 1)')
    if not (math.isfinite(start_sd) and 0 <= start_sd < 1):
        raise ValueError('start_sd must be a registered fraction in [0, 1)')
    diffs, point, low, high, p = _paired_stats(control, candidate, s.higher_is_better)
    method = f'paired relative-difference Student t, n={len(diffs)} pairs'
    if start_sd > 0:
        within = (high - point) / t975(len(diffs) - 1)
        half = t975(len(diffs) - 1) * math.sqrt(within ** 2 + start_sd ** 2 * (1 + 1 / (2 if s.aa is not None else 1)))
        low, high = point - half, point + half
        method += f' + registered between-start SD {start_sd:.2%}'
    result: dict[str, Json] = {'cell': s.cell, 'metric': s.metric, 'role': 'guard' if guard else 'primary',
                               'profile': s.profile, 'pairs': len(diffs), 'point_gain': point, 'interval95': [low, high],
                               'interval_method': method, 'start_sd': start_sd,
                               'sign_flip_p': p, 'diff_sd': statistics.stdev(diffs), 'per_pair_gain': cast(Json, diffs),
                               'min_gain': min_gain, 'higher_is_better': s.higher_is_better,
                               'prompt_sha256': s.prompt_sha256}
    reasons: list[str] = []
    if s.aa is not None:
        c1, c2 = _values(f'{s.cell}/{s.metric} A/A C1', s.aa[0]), _values(f'{s.cell}/{s.metric} A/A C2', s.aa[1])
        if len(c1) != len(diffs) or len(c2) != len(diffs):
            raise ValueError(f'{s.cell}/{s.metric}: A/A needs one C1 and one C2 value per pair')
        _, aa_point, aa_low, aa_high, aa_p = _paired_stats(c1, c2, s.higher_is_better)
        result['aa'] = {'point_gain': aa_point, 'interval95': [aa_low, aa_high], 'sign_flip_p': aa_p}
        if aa_p < ALPHA and (aa_low > 0 or aa_high < 0):
            reasons.append(f'A/A drift between the control starts ({aa_point:+.2%}, p={aa_p:.3f})')
    if expect_identical:
        if streams_identical is None:
            raise ValueError(f'{s.cell}: a target-side lever must pass the A/A stream identity (streams_identical=)')
        result['aa_streams_identical'] = streams_identical
        if not streams_identical:
            reasons.append('A/A token streams or round counts differ between the control starts (nondeterminism)')
    significant = p < ALPHA
    if reasons:
        verdict = 'INCONCLUSIVE'
        result['inconclusive_reasons'] = cast(Json, reasons)
    elif significant and high < 0:
        verdict = 'LOSS'
    elif not guard and significant and low > 0 and point > min_gain:
        verdict = 'WIN'
    else:
        verdict = 'INCONCLUSIVE'
    result['verdict'] = verdict
    return result


def decide_paired_window(series: Sequence[PairedSeries], decision_cells: Sequence[str], *, min_gain: float = 0.02,
                         expect_identical: Mapping[str, bool] | None = None,
                         streams_identical: Mapping[str, bool] | None = None,
                         start_sd: Mapping[str, float] | None = None) -> dict[str, Json]:
    """One verdict per profile. Per decision cell: series[0] of that cell is the primary, the rest are guards. A cell in
    `expect_identical` (target-side lever) needs its `streams_identical` A/A result. `start_sd` maps '<cell>/<metric>' to
    the registered between-start SD (default 0 until the seeded A/A pilot measures it)."""
    decision = list(decision_cells)
    if not decision or len(set(decision)) != len(decision):
        raise ValueError(f'{PAIRED_RULE} needs distinct decision cells, got {decision!r}')
    by_cell: dict[str, list[PairedSeries]] = {}
    for s in series:
        by_cell.setdefault(s.cell, []).append(s)
    missing = [c for c in decision if c not in by_cell]
    if missing:
        raise ValueError(f'decision cells without measurements: {missing!r}')
    profile_of: dict[str, str] = {}
    for cell, rows in by_cell.items():
        tags = {r.profile for r in rows}
        if len(tags) != 1:
            raise ValueError(f'{cell}: every series of a cell carries the same profile')
        profile_of[cell] = tags.pop()
    per_profile: dict[str, list[str]] = {}
    for cell in decision:
        per_profile.setdefault(profile_of[cell], []).append(cell)
    crowded = {p: c for p, c in per_profile.items() if len(c) > MAX_DECISION_CELLS}
    if crowded:
        raise ValueError(f'at most {MAX_DECISION_CELLS} decision cells per profile, got {crowded}')
    cells: dict[str, Json] = {}
    verdict_of: dict[str, str] = {}
    for cell, rows in by_cell.items():
        if cell in decision:
            if len({r.metric for r in rows}) != len(rows):
                raise ValueError(f'{cell}: duplicate metric series')
            prompts = {r.prompt_sha256 for r in rows}
            if None in prompts or len(prompts) != 1:
                raise ValueError(f'{cell}: every series of a decision cell needs the one proved prompt identity '
                                 f'(seeded_series), got {sorted(map(str, prompts))}; never paired across prompt hashes')
            expected = bool((expect_identical or {}).get(cell, False))
            identical = (streams_identical or {}).get(cell)
            results = [decide_paired_cell(r, min_gain=min_gain, guard=i > 0, expect_identical=expected,
                                          streams_identical=identical, start_sd=(start_sd or {}).get(f'{cell}/{r.metric}', 0.0))
                       for i, r in enumerate(rows)]
            verdict_of[cell] = _cell_verdict(results)
            cells[cell] = {'decides': True, 'profile': profile_of[cell], 'verdict': verdict_of[cell],
                           'primary_metric': rows[0].metric, 'prompt_sha256': rows[0].prompt_sha256,
                           'metrics': {str(r['metric']): r for r in results}}
        else:
            reported: dict[str, Json] = {}
            for r in rows:
                diffs = [_oriented(x, c, r.higher_is_better) for c, x in zip(r.control, r.candidate)]
                reported[r.metric] = {'verdict': 'REPORTED', 'pairs': len(diffs),
                                      'point_gain': statistics.fmean(diffs) if diffs else None}
            cells[cell] = {'decides': False, 'profile': profile_of[cell], 'verdict': 'REPORTED', 'metrics': reported}
    profiles: dict[str, Json] = {p: _status([verdict_of[c] for c in cs]) for p, cs in sorted(per_profile.items())}
    status = (str(next(iter(profiles.values()))) if len(profiles) == 1
              else 'PER PROFILE: ' + '; '.join(f'{p}={v}' for p, v in profiles.items()))
    return {'rule': PAIRED_RULE, 'status': status, 'profiles': profiles, 'min_gain': min_gain, 'alpha': ALPHA,
            'decision_cells': cast(Json, decision), 'cells': cells}


def seeded_series(metric: str, higher_is_better: bool, c1: Mapping[str, Json], x: Mapping[str, Json],
                  c2: Mapping[str, Json] | None = None) -> PairedSeries:
    """PairedSeries from run_seeded_cell results of C1, X (and C2): control pair i = C1_i, or mean(C1_i, C2_i) with the
    built-in A/A (C1, C2). Refuses arms that ran different cells or seeds, and any pair whose prompt identity differs from
    the others' in any arm (DP-2: never paired across prompt hashes; results without prompt hashes refuse).
    A report-only result (the A/A pilot's bridging cell) never decides."""
    arms = [c1, x] + ([c2] if c2 is not None else [])
    cells = {str(a.get('cell')) for a in arms}
    profiles = {str(a.get('profile')) for a in arms}
    if len(cells) != 1 or len(profiles) != 1:
        raise ValueError(f'seeded arms ran different cells or profiles: {sorted(cells)} {sorted(profiles)}')
    if any(a.get('report_only') for a in arms):
        raise ValueError(f'{sorted(cells)}: a report-only seeded cell never decides')
    rows = [[record(p) for p in cast(list[Json], a['pairs'])] for a in arms]
    if len({len(r) for r in rows}) != 1 or any(rows[0][i]['seeds'] != r[i]['seeds'] for r in rows for i in range(len(r))):
        raise ValueError(f'{cells.pop()}: seeded arms differ in pairs or registered seeds')
    prompt = same_prompt(next(iter(cells)), *([str(p.get('prompt_sha256'))] for r in rows for p in r))
    values = [tuple(number(p[metric]) for p in r) for r in rows]
    if c2 is None:
        control, aa = values[0], None
    else:
        control, aa = tuple((a + b) / 2 for a, b in zip(values[0], values[2])), (values[0], values[2])
    return PairedSeries(next(iter(cells)), metric, higher_is_better, control, values[1], next(iter(profiles)), aa, prompt)
