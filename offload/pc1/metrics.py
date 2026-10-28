"""Preregistered TTFT screen; exact paired two-group resampling, numeric evidence only."""
from __future__ import annotations

import itertools
import math
import re
import statistics
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, cast

from offload.launch.windows import up2_checks as ck
from offload.pc1 import cells
from offload.prefill_maxlen.evidence import Json, number, record


def positive(value: Json, *, zero: bool = False) -> float:
    n = number(value)
    ck.require(math.isfinite(n) and (n >= 0 if zero else n > 0), 'invalid timing/counter')
    return n


def compare(values: list[list[float]], *, lower_is_better: bool, allow_zero: bool = False,
            weights: list[list[float]] | None = None) -> dict[str, Json]:
    """Exhaustive paired bootstrap: four equiprobable draws for n=2.

    Nearest-rank 2.5/97.5 percentiles are min/max here. This is a conditional
    screen, not restart or task-family inference. The win threshold is fixed
    from observed bracketing controls before resampling.
    """
    ck.require(len(values) == 3 and all(len(v) == 2 for v in values), 'three matched starts/two reps required')
    rows = [[Fraction(str(positive(v, zero=allow_zero))) for v in row] for row in values]
    mass = [[Fraction(1), Fraction(1)] for _ in rows] if weights is None else [
        [Fraction(str(positive(v))) for v in row] for row in weights]
    ck.require(len(mass) == 3 and all(len(row) == 2 for row in mass), 'matched group weights required')
    def mean(start: int, indices: tuple[int, ...]) -> Fraction:
        return sum((rows[start][i] * mass[start][i] for i in indices), Fraction()) / sum(
            (mass[start][i] for i in indices), Fraction())
    points = [mean(start, (0, 1)) for start in range(3)]
    reference = (points[0] + points[2]) / 2
    spread = abs(points[0] - points[2])
    sign = 1 if lower_is_better else -1
    delta = sign * (reference - points[1])
    required = max(reference / 50, spread)
    draws = []
    for indices in itertools.product(range(2), repeat=2):
        means = [mean(start, indices) for start in range(3)]
        draws.append(sign * ((means[0] + means[2]) / 2 - means[1]))
    per_repeat = [sign * ((rows[0][i] + rows[2][i]) / 2 - rows[1][i]) >= -abs(rows[0][i] - rows[2][i])
                  for i in range(2)]
    return {'start_means': cast(Json, [float(v) for v in points]), 'control_spread': float(spread),
        'required_gain': float(required), 'point_gain': float(delta),
        'paired_gain_interval95': [float(min(draws)), float(max(draws))],
        'win': delta > required and min(draws) > required,
        'no_loss_beyond_spread': delta >= -spread and min(draws) >= -spread and all(per_repeat),
        'method': 'exact paired whole-group bootstrap n=2;4 equiprobable draws;nearest-rank95%;fixed control threshold',
        'scope': 'conditional on fixed public prompts and three starts; no restart/task-family uncertainty'}


def decision(starts: list[dict[str, Json]]) -> dict[str, Json]:
    ck.require(len(starts) == 3 and all(set(s) == set(cells.CELLS) for s in starts), 'complete registered cells required')
    salts: set[str] = set()
    reports: dict[str, Json] = {}
    winners: list[Json] = []
    passes = True
    for name in cells.CELLS:
        batches = cells.batch_for(name)
        rows = [ck.rows(record(s[name])['repetitions']) for s in starts]
        ck.require(all(len(r) == cells.REPEATS[name] for r in rows), 'two repetitions required')
        fingerprints: list[Json] | None = None
        for start_index, reps in enumerate(rows):
            cell = record(starts[start_index][name])
            ck.require(cell.get('cell') == name and cell.get('batch') == batches, 'cell identity mismatch')
            for i, row in enumerate(reps):
                ck.require(row.get('repeat') == i and row.get('batch') == batches
                    and row.get('completed_requests') == batches, 'request group identity/completion mismatch')
                if name == cells.CELLS[-1]:
                    ck.require(row.get('mixed_three_active_overlap') is True, 'mixed admission/overlap unproved')
                rounds = positive(row['spec_request_rounds'])
                accepted = positive(row['accepted_tokens'], zero=True)
                ck.require(rounds.is_integer() and accepted.is_integer() and accepted <= 5 * rounds,
                           'invalid speculative counters')
                positive(row['server_decode_ms'])
                streams = ck.rows(row['streams'])
                ck.require(len(streams) == batches, 'stream count mismatch')
                hashes: list[Json] = [[s['prompt_sha256'], s['prompt_tokens'], s['budget']] for s in streams]
                ck.require(fingerprints is None or fingerprints == hashes, 'public prompt mismatch')
                fingerprints = hashes
                for stream_index, stream in enumerate(streams):
                    usage = record(stream['usage'])
                    target = 1024 if name == 'short-b1' else 150000 if (
                        name == '150k-b1' or name == cells.CELLS[-1] and stream_index == 2) else 32000
                    occupied = positive(stream['prompt_tokens'])
                    ck.require(occupied.is_integer() and .98 * target <= occupied <= target,
                               'occupied prompt outside registered context')
                    ck.require(stream.get('budget') == usage.get('completion_tokens') == 512
                        and stream.get('finish') == 'length' and usage.get('prompt_tokens') == stream.get('prompt_tokens')
                        and record(usage.get('prompt_tokens_details') or {}).get('cached_tokens') == 0
                        and stream.get('delivery_gap_samples') == 511,
                        'incomplete/cached request')
                    ck.require(record(stream['sampling']) == {'temperature': 1, 'top_p': 1, 'seed': 42+i,
                        'priority': 10, 'max_tokens': 512}, 'sampling mismatch')
                    salt = str(stream['cache_salt_sha256'])
                    ck.require(re.fullmatch('[0-9a-f]{64}', salt) is not None and salt not in salts,
                               'invalid/reused cold salt')
                    ck.require(re.fullmatch('[0-9a-f]{64}', str(stream['prompt_sha256'])) is not None,
                               'invalid prompt hash')
                    salts.add(salt)
        gates: dict[str, Json] = {}
        for stream_index in range(batches):
            for metric, lower in (('ttft_s', True), ('decode_tps', False)):
                values = [[positive(ck.rows(r['streams'])[stream_index][metric]) for r in reps] for reps in rows]
                point = compare(values, lower_is_better=lower)
                gates[f'{stream_index}:{metric}'] = point
                passes &= point['no_loss_beyond_spread'] is True
                if name in ('32k-b1', '150k-b1') and metric == 'ttft_s' and point['win'] is True:
                    winners.append(name)
            if name == cells.CELLS[-1] and stream_index < 2:
                point = compare([[positive(ck.rows(r['streams'])[stream_index]['p95_inter_token_delivery_gap_ms'], zero=True)
                                  for r in reps] for reps in rows], lower_is_better=True, allow_zero=True)
                gates[f'{stream_index}:p95_delivery_gap_ms'] = point
                passes &= point['no_loss_beyond_spread'] is True
        # Server pooled decode is an additional no-regression guard, never the objective.
        point = compare([[1000 * (positive(r['spec_request_rounds']) + positive(r['accepted_tokens'], zero=True)) /
                          positive(r['server_decode_ms']) for r in reps] for reps in rows], lower_is_better=False,
                        weights=[[positive(r['server_decode_ms']) for r in reps] for reps in rows])
        gates['server_committed_tps_proxy'] = point
        passes &= point['no_loss_beyond_spread'] is True
        reports[name] = gates
    return {'status': 'CANDIDATE FOR REVIEW' if passes and winners else 'NOT A CANDIDATE', 'cells': reports,
        'passing_target_cells': winners, 'rule': 'cold32K or150K B1 TTFT win: point and paired lower bound >max(2%,C spread); '
        'every cell no loss beyond spread; mixed decoder p95 and short decode no regression',
        'promotion': 'none; each arm requires a separate HOLD and review'}


# ---------------------------------------------------------------------------------------------
# WIN2 (NDM 2026-10-27: "Fairer win rule is essential"; Fable review, roadmap e8e3e3a).
# Decision cells are repeated three times in every start; repeat r of the candidate is paired
# with the mean of the two bracketing controls' repeat r (same public prompts and seed 42+r).
# Only the plan's decision cells decide; every other number is reported.
WIN2_RULE_NOTE = ('win2-v1 via offload.launch.windows.decision_v2: primary ms/round (decode cells) or TTFT (prefill cells); '
                  'candidate x3 vs both controls x3, unpaired Welch t; WIN = point gain >2% and 95% interval excludes 0; '
                  'LOSS = primary or guard worse than the control spread in 2 of 3 repeats; at most two decision cells')


def validate_plan(starts: list[dict[str, Json]], plan: cells.Plan) -> None:
    """Request identity/completion integrity for every cell the plan ran (unchanged per-stream rules)."""
    ck.require(len(starts) == 3 and all(set(s) == set(plan.cells) for s in starts), 'complete planned cells required')
    salts: set[str] = set()
    for name in plan.cells:
        batches = cells.batch_for(name)
        rows = [ck.rows(record(s[name])['repetitions']) for s in starts]
        ck.require(all(len(r) == plan.repeats(name) for r in rows), 'planned repetitions required')
        fingerprints: list[Json] | None = None
        for start_index, reps in enumerate(rows):
            cell = record(starts[start_index][name])
            ck.require(cell.get('cell') == name and cell.get('batch') == batches, 'cell identity mismatch')
            for i, row in enumerate(reps):
                ck.require(row.get('repeat') == i and row.get('batch') == batches
                    and row.get('completed_requests') == batches, 'request group identity/completion mismatch')
                if name == cells.CELLS[-1]:
                    ck.require(row.get('mixed_three_active_overlap') is True, 'mixed admission/overlap unproved')
                rounds = positive(row['spec_request_rounds'])
                accepted = positive(row['accepted_tokens'], zero=True)
                ck.require(rounds.is_integer() and accepted.is_integer() and accepted <= 5 * rounds,
                           'invalid speculative counters')
                positive(row['server_decode_ms'])
                streams = ck.rows(row['streams'])
                ck.require(len(streams) == batches, 'stream count mismatch')
                hashes: list[Json] = [[s['prompt_sha256'], s['prompt_tokens'], s['budget']] for s in streams]
                ck.require(fingerprints is None or fingerprints == hashes, 'public prompt mismatch')
                fingerprints = hashes
                for stream_index, stream in enumerate(streams):
                    usage = record(stream['usage'])
                    target = 1024 if name == 'short-b1' else 150000 if (
                        name == '150k-b1' or name == cells.CELLS[-1] and stream_index == 2) else 32000
                    occupied = positive(stream['prompt_tokens'])
                    ck.require(occupied.is_integer() and .98 * target <= occupied <= target,
                               'occupied prompt outside registered context')
                    ck.require(stream.get('budget') == usage.get('completion_tokens') == 512
                        and stream.get('finish') == 'length' and usage.get('prompt_tokens') == stream.get('prompt_tokens')
                        and record(usage.get('prompt_tokens_details') or {}).get('cached_tokens') == 0
                        and stream.get('delivery_gap_samples') == 511, 'incomplete/cached request')
                    ck.require(record(stream['sampling']) == {'temperature': 1, 'top_p': 1, 'seed': 42 + i,
                        'priority': 10, 'max_tokens': 512}, 'sampling mismatch')
                    salt = str(stream['cache_salt_sha256'])
                    ck.require(re.fullmatch('[0-9a-f]{64}', salt) is not None and salt not in salts,
                               'invalid/reused cold salt')
                    ck.require(re.fullmatch('[0-9a-f]{64}', str(stream['prompt_sha256'])) is not None,
                               'invalid prompt hash')
                    salts.add(salt)


def committed_tps(row: dict[str, Json]) -> float:
    """Pooled server committed tokens per decode second: (rounds + accepted) / server decode time."""
    return 1000 * (positive(row['spec_request_rounds']) + positive(row['accepted_tokens'], zero=True)) / positive(
        row['server_decode_ms'])


def ms_per_round(row: dict[str, Json]) -> float:
    """WIN2 primary for decode cells: pooled server decode ms per speculative round (archive name ms_per_round)."""
    return positive(row['server_decode_ms']) / positive(row['spec_request_rounds'])


def accepted_per_round(row: dict[str, Json]) -> float:
    return positive(row['accepted_tokens'], zero=True) / positive(row['spec_request_rounds'])


def median_stream(metric: str) -> Callable[[dict[str, Json]], float]:
    """Median over the repeat's streams: the cells' recorded median_decode_tps / median_ttft_s (win2_history names)."""
    def value(row: dict[str, Json]) -> float:
        return statistics.median(positive(s[metric]) for s in ck.rows(row['streams']))
    return value


def mean_completion_s(row: dict[str, Json]) -> float:
    """Multi-user cell: mean per-request completion time from each request's own send (lower is better)."""
    return sum(positive(s['total_s']) for s in ck.rows(row['streams'])) / len(ck.rows(row['streams']))


def stream_metric(index: int, metric: str) -> Callable[[dict[str, Json]], float]:
    def value(row: dict[str, Json]) -> float:
        return positive(ck.rows(row['streams'])[index][metric], zero=metric.startswith('p95'))
    return value


def decoders_metric(metric: str) -> Callable[[dict[str, Json]], float]:
    def value(row: dict[str, Json]) -> float:
        streams = ck.rows(row['streams'])[:2]
        return sum(positive(s[metric], zero=metric.startswith('p95')) for s in streams) / 2
    return value


@dataclass(frozen=True)
class Metric:
    name: str               # archive metric name; unique within its cell
    higher_is_better: bool
    extract: Callable[[dict[str, Json]], float]
    decides: bool           # in a decision cell: primary (first) or guard; False = reported only


# WIN2.md: pooled server ms/round is the primary of decode cells and TTFT of prefill cells; accepted/round and client
# tok/s guard decode cells (a guard can only make a LOSS). The mixed cell also guards the 150K document's first token,
# the multi-user cost a threshold arm can impose (threshold512, 2026-10-27: +32 %). Same-named metrics match
# win2_history's definitions, so archived rows pool with the backfilled history at max_tokens 512.
METRICS: dict[str, tuple[Metric, ...]] = {
    'short-b1': (Metric('ms_per_round', False, ms_per_round, True),
                 Metric('accepted_per_round', True, accepted_per_round, True),
                 Metric('decode_tps', True, median_stream('decode_tps'), True),
                 Metric('ttft_s', False, median_stream('ttft_s'), False),
                 Metric('committed_tps', True, committed_tps, False)),
    '32k-b1': (Metric('ttft_s', False, median_stream('ttft_s'), True),
               Metric('ms_per_round', False, ms_per_round, False),
               Metric('accepted_per_round', True, accepted_per_round, False),
               Metric('decode_tps', True, median_stream('decode_tps'), False)),
    '150k-b1': (Metric('ttft_s', False, median_stream('ttft_s'), True),
                Metric('ms_per_round', False, ms_per_round, False),
                Metric('accepted_per_round', True, accepted_per_round, False),
                Metric('decode_tps', True, median_stream('decode_tps'), False)),
    'mixed150k-2decode': (Metric('ms_per_round', False, ms_per_round, True),
                          Metric('accepted_per_round', True, accepted_per_round, True),
                          Metric('decode_tps', True, median_stream('decode_tps'), True),
                          Metric('long_doc_ttft_s', False, stream_metric(2, 'ttft_s'), True),
                          Metric('ttft_s', False, median_stream('ttft_s'), False),
                          Metric('mean_completion_s', False, mean_completion_s, False),
                          Metric('decoders_decode_tps', True, decoders_metric('decode_tps'), False),
                          Metric('decoders_p95_gap_ms', False, decoders_metric('p95_inter_token_delivery_gap_ms'), False),
                          Metric('committed_tps', True, committed_tps, False)),
}


def fault_attribution(starts: Mapping[str, tuple[str, int]]) -> dict[str, Json]:
    """Drafter faults by start (name -> (role, attributed fault count)): control vs arm, the same counters per repeat.

    The three starts run the same plan, so under chance each fault lands in the arm start with probability 1/3.
    """
    control = sum(n for role, n in starts.values() if role == 'control')
    arm = sum(n for role, n in starts.values() if role != 'control')
    if control == arm == 0:
        verdict, note = 'NONE', 'no drafter fault in any start'
    elif arm == 0:
        verdict, note = 'CONTROL-ONLY', 'drafter faults only in control starts: transport or chance, not the arm'
    elif control > 0:
        verdict, note = 'BOTH', 'drafter faults in control and arm starts: transport, not attributable to the arm'
    elif arm == 1:
        verdict, note = 'ARM-ONLY-1', 'one arm-only drafter fault: chance cannot be excluded (p = 1/3, equal exposure)'
    else:
        verdict, note = 'ARM-SUSPECT', (f'{arm} arm-only drafter faults with clean controls (p = {(1 / 3) ** arm:.3f} '
                                        'by chance, equal exposure): reliability veto')
    return {'verdict': verdict, 'note': note, 'control_faults': control, 'arm_faults': arm,
            'by_start': {name: n for name, (_role, n) in starts.items()}}


def series_cell(name: str, metric: Metric, plan: cells.Plan) -> str:
    """decision_v2 groups series by cell: a decision cell's reported-only metrics go to '<cell>:reported'."""
    return name + ':reported' if name in plan.decision and not metric.decides else name


def series_v2(starts: list[dict[str, Json]], plan: cells.Plan) -> tuple[list[Any], dict[str, Json]]:
    """Unpaired Series (both control starts' repeats vs the candidate's) plus each metric's brackets (reported).

    S92 (WIN2 review): pair only true adjacent pairs; C/X/C is unpaired. Same-seed repeats are not pairs in practice:
    the two controls of PC1-threshold512-W-20261027T181314Z ran short B1 repeat 0 at 1.51 and 1.18 accepted/round.
    """
    from offload.launch.windows import decision_v2 as d2
    out: list[Any] = []
    brackets: dict[str, Json] = {}
    for name in plan.cells:
        rows = [ck.rows(record(s[name])['repetitions']) for s in starts]
        for metric in METRICS[name]:
            first, candidate, last = ([metric.extract(r) for r in reps] for reps in rows)
            out.append(d2.Series(cell=series_cell(name, metric, plan), metric=metric.name,
                                 higher_is_better=metric.higher_is_better, control=tuple(first + last),
                                 candidate=tuple(candidate), paired=False))
            brackets[f'{name}/{metric.name}'] = {
                'first_control': cast(Json, first), 'candidate': cast(Json, candidate), 'last_control': cast(Json, last),
                'bracket_spread_per_repeat': cast(Json, [abs(a - c) / ((a + c) / 2) for a, c in zip(first, last)])}
    return out, brackets


def decision_v2(group: str, starts: list[dict[str, Json]]) -> dict[str, Json]:
    from offload.launch.windows import decision_v2 as d2
    plan = cells.plan_for(group)
    validate_plan(starts, plan)
    series, brackets = series_v2(starts, plan)
    result = d2.decide_window(series, plan.decision, groups=dict(plan.groups))
    result.update(group=group, brackets=brackets, note=WIN2_RULE_NOTE,
                  promotion='CANDIDATE is a measured speed verdict for review; promotion is a separate coordinator step')
    return result
