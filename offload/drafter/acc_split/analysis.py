"""ACC-SPLIT tables over `sources.Obs`: acceptance by prompt, answer length, segment, batch and date, plus the metric
reconciliation (server counters vs SSE-derived) and the 32K / short shift decomposition.

Every value is MEASURED from the files `sources` reads, except where a row says ESTIMATED (SSE-derived segment
acceptance, the late-tokens estimate). Intervals are 95 % t-intervals over independent runs; they describe run-to-run
spread at T = 1, not a population claim about unseen prompts.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from offload.drafter.acc_split.sources import Obs
from offload.prefill_maxlen.evidence import Json

# two-sided 95 % t quantiles by degrees of freedom (1..30); 1.96 beyond
_T95 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
        2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)
POSITIONS = 5
# window start roles that run production as is (WIN2 CONTROL_ROLES); every other role is a candidate arm whose change may
# move acceptance (DEP1 fixed depth, PP4 sampling, native drafter) and is excluded from prompt/date/length tables.
CONTROL_ROLES = ('control', 'K', 'off', 'H')


def _jlist(values: Iterable[Json]) -> list[Json]:
    return list(values)


def t95(df: int) -> float:
    if df < 1:
        raise ValueError('need at least two values for an interval')
    return _T95[df - 1] if df <= len(_T95) else 1.96


@dataclass(frozen=True)
class Summary:
    n: int
    mean: float
    sd: float
    lo: float
    hi: float

    def json(self) -> dict[str, Json]:
        return {'n': self.n, 'mean': round(self.mean, 4), 'sd': round(self.sd, 4), 'ci95': [round(self.lo, 4), round(self.hi, 4)]}


def summarize(values: Sequence[float]) -> Summary:
    if not values:
        raise ValueError('nothing to summarize')
    mean = statistics.fmean(values)
    if len(values) == 1:
        return Summary(1, mean, 0.0, mean, mean)
    sd = statistics.stdev(values)
    half = t95(len(values) - 1) * sd / math.sqrt(len(values))
    return Summary(len(values), mean, sd, mean - half, mean + half)


def prompt_names(obs: Iterable[Obs]) -> dict[str, str]:
    """A readable family name per prompt sha: bench2 '<workload>-<ctx>' or the window cells that sent it."""
    names: dict[str, set[str]] = defaultdict(set)
    for o in obs:
        if o.batch == 1:
            base = o.cell if o.source == 'bench2' else o.cell.removeprefix('ctl-').removeprefix('prof-')
            names[o.prompt].add(('bench2:' if o.source == 'bench2' else 'corpus:') + base)
    return {sha: '+'.join(sorted(v)) for sha, v in names.items()}


def _b1(obs: Iterable[Obs], keep: Callable[[Obs], bool] | None = None) -> list[Obs]:
    """Single-stream, reasoning-only-or-mixed runs; bench2 cold only (warm repeats and drills are other questions)."""
    out = []
    for o in obs:
        if o.batch != 1 or o.rounds <= 0:
            continue
        if o.source == 'bench2' and (o.cache != 'cold' or 'drill' in o.label):
            continue
        if o.source == 'cell' and o.role not in CONTROL_ROLES:
            continue
        if keep is None or keep(o):
            out.append(o)
    return out


def position_rates(group: Sequence[Obs]) -> list[float]:
    """Pooled unconditional accepted-at-position-k per round (positions 0..4)."""
    rounds = sum(o.rounds for o in group)
    return [round(sum(o.per_position[k] if k < len(o.per_position) else 0.0 for o in group) / rounds, 4)
            for k in range(POSITIONS)]


def by_prompt(obs: Iterable[Obs]) -> list[dict[str, Json]]:
    """Acceptance per (prompt, answer budget) at B1, with dates, depth, per-position rates and round time."""
    rows = _b1(obs)
    names = prompt_names(rows)
    groups: dict[tuple[str, int], list[Obs]] = defaultdict(list)
    for o in rows:
        groups[(o.prompt, o.max_tokens)].append(o)
    out: list[dict[str, Json]] = []
    for (sha, budget), group in sorted(groups.items(), key=lambda kv: (names[kv[0][0]], kv[0][1])):
        ms = [o.ms_per_round for o in group if o.ms_per_round is not None]
        out.append({
            'prompt': names[sha], 'prompt_sha256_12': sha[:12], 'prompt_tokens': group[0].prompt_tokens[0],
            'max_tokens': budget, 'dates': _jlist(sorted({o.date for o in group})),
            'configs': len({o.label for o in group}),
            'answer_tokens_mean': round(statistics.fmean(o.answer_tokens for o in group), 1),
            'accepted_per_round': summarize([o.accepted_per_round for o in group]).json(),
            'pooled_accepted_per_round': round(sum(o.accepted for o in group) / sum(o.rounds for o in group), 4),
            'depth': round(sum(o.draft_tokens for o in group) / sum(o.rounds for o in group), 3),
            'per_position': _jlist(position_rates(group)),
            'ms_per_round': round(statistics.fmean(ms), 2) if ms else None})
    return out


def by_prompt_date(obs: Iterable[Obs]) -> list[dict[str, Json]]:
    """Per prompt and budget, the mean per date: the stability test for H1 (same prompt, different days and configs)."""
    rows = _b1(obs)
    names = prompt_names(rows)
    groups: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for o in rows:
        groups[(o.prompt, o.max_tokens, o.date)].append(o.accepted_per_round)
    return [{'prompt': names[sha], 'max_tokens': budget, 'date': date, **summarize(values).json()}
            for (sha, budget, date), values in sorted(groups.items(), key=lambda kv: (names[kv[0][0]], kv[0][1], kv[0][2]))]


def by_window(obs: Iterable[Obs]) -> list[dict[str, Json]]:
    """Per window (or bench2 config label) and prompt/budget at B1: drift within and across days."""
    rows = _b1(obs)
    names = prompt_names(rows)
    groups: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
    for o in rows:
        groups[(o.prompt, o.max_tokens, o.date, o.label)].append(o.accepted_per_round)
    return [{'prompt': names[sha], 'max_tokens': budget, 'date': date, 'window': label, **summarize(v).json()}
            for (sha, budget, date, label), v in sorted(groups.items(), key=lambda kv: (names[kv[0][0]], kv[0][1],
                                                                                        kv[0][2], kv[0][3]))]


def metric_reconciliation(obs: Iterable[Obs]) -> list[dict[str, Json]]:
    """H2: server accepted/round vs what a client derives from tok/s and the SSE step gap (bench2, B1)."""
    rows = [o for o in _b1(obs) if o.source == 'bench2' and o.decode_tps and o.step_gap_p50_ms and o.step_gap_mean_ms
            and o.step_events]
    names = prompt_names(rows)
    groups: dict[tuple[str, int], list[Obs]] = defaultdict(list)
    for o in rows:
        groups[(o.prompt, o.max_tokens)].append(o)
    out: list[dict[str, Json]] = []
    for (sha, budget), group in sorted(groups.items(), key=lambda kv: (names[kv[0][0]], kv[0][1])):
        if len(group) < 2:
            continue

        def mean(f: Callable[[Obs], float]) -> float:
            return round(statistics.fmean(f(o) for o in group), 4)
        out.append({
            'prompt': names[sha], 'max_tokens': budget, 'n': len(group),
            'server': mean(lambda o: o.accepted_per_round),
            'tps_x_p50_gap': mean(lambda o: (o.decode_tps or 0.0) * (o.step_gap_p50_ms or 0.0) / 1000 - 1),
            'tps_x_mean_gap': mean(lambda o: (o.decode_tps or 0.0) * (o.step_gap_mean_ms or 0.0) / 1000 - 1),
            'tokens_per_sse_event': mean(lambda o: (o.reasoning_tokens + o.answer_tokens) / (o.step_events or 1) - 1),
            'rounds_per_sse_event': mean(lambda o: o.rounds / (o.step_events or 1))})
    return out


def length_split(obs: Iterable[Obs]) -> list[dict[str, Json]]:
    """Same prompt at two answer budgets: acceptance of the first budget, and ESTIMATED acceptance of the tokens after
    it ((accepted_L - accepted_S) / (rounds_L - rounds_S) over group means)."""
    rows = _b1(obs)
    names = prompt_names(rows)
    groups: dict[str, dict[int, list[Obs]]] = defaultdict(lambda: defaultdict(list))
    for o in rows:
        groups[o.prompt][o.max_tokens].append(o)
    out: list[dict[str, Json]] = []
    for sha, budgets in sorted(groups.items(), key=lambda kv: names[kv[0]]):
        keys = sorted(b for b, g in budgets.items() if len(g) >= 2)
        for short, long in zip(keys, keys[1:]):
            s, lg = budgets[short], budgets[long]
            acc_s, r_s = statistics.fmean(o.accepted for o in s), statistics.fmean(o.rounds for o in s)
            acc_l, r_l = statistics.fmean(o.accepted for o in lg), statistics.fmean(o.rounds for o in lg)
            late = (acc_l - acc_s) / (r_l - r_s) if r_l > r_s else None
            out.append({'prompt': names[sha], 'short_budget': short, 'long_budget': long,
                        'short': summarize([o.accepted_per_round for o in s]).json(),
                        'long': summarize([o.accepted_per_round for o in lg]).json(),
                        'late_tokens_estimated': round(late, 4) if late is not None else None,
                        'short_dates': _jlist(sorted({o.date for o in s})),
                        'long_dates': _jlist(sorted({o.date for o in lg}))})
    return out


def segment_split(obs: Iterable[Obs]) -> dict[str, Json]:
    """ESTIMATED acceptance per output kind (reasoning vs answer), B1 bench2 runs.

    Two estimators: (a) SSE events per kind, tokens per event - 1, validated on reasoning-only runs against the server
    counter; (b) for runs without SSE step times, per-kind client decode tok/s with one assumption, equal round time C in
    both segments: C = (r_tok / tps_r + a_tok / tps_a) / server_rounds, A_kind = tps_kind * C - 1. Both reproduce the
    server's whole-run figure by construction ((b) exactly, (a) within its validated bias)."""
    rows = [o for o in _b1(obs) if o.source == 'bench2']
    only = [o for o in rows if set(o.segments) == {'reasoning'}]
    validation = [o.segments['reasoning'][0] / o.segments['reasoning'][1] - 1 - o.accepted_per_round for o in only]
    names = prompt_names(rows)
    mixed: list[dict[str, Json]] = []
    for o in rows:
        if o.answer_tokens <= 0 or o.reasoning_tokens <= 0:
            continue
        row: dict[str, Json] = {'prompt': names[o.prompt], 'run': o.where[:12], 'label': o.label, 'date': o.date,
                                'max_tokens': o.max_tokens, 'reasoning_tokens': o.reasoning_tokens,
                                'answer_tokens': o.answer_tokens, 'server_all': round(o.accepted_per_round, 4)}
        if {'reasoning', 'answer'} <= set(o.segments):
            (rt, rev), (at, aev) = o.segments['reasoning'], o.segments['answer']
            row.update(method='sse_events', reasoning_estimated=round(rt / rev - 1, 4),
                       answer_estimated=round(at / aev - 1, 4))
        elif {'reasoning', 'answer'} <= set(o.segment_tps):
            tr, ta = o.segment_tps['reasoning'], o.segment_tps['answer']
            c = (o.reasoning_tokens / tr + o.answer_tokens / ta) / o.rounds
            row.update(method='tps_equal_round_time', round_s_estimated=round(c, 5),
                       reasoning_estimated=round(tr * c - 1, 4), answer_estimated=round(ta * c - 1, 4))
        else:
            continue
        mixed.append(row)
    return {'validation_runs': len(only),
            'validation_sse_minus_server': summarize(validation).json() if len(validation) >= 2 else None,
            'mixed_runs': _jlist(mixed)}


def batch_split(obs: Iterable[Obs]) -> list[dict[str, Json]]:
    """Acceptance by batch for window cells at the same budget (multi-user strata)."""
    groups: dict[tuple[str, int, int], list[Obs]] = defaultdict(list)
    for o in obs:
        if o.source == 'cell' and o.rounds > 0 and o.role in CONTROL_ROLES:
            groups[(o.cell.removeprefix('ctl-'), o.max_tokens, o.batch)].append(o)
    return [{'cell': cell, 'max_tokens': budget, 'batch': batch, 'dates': _jlist(sorted({o.date for o in g})),
             'accepted_per_round': summarize([o.accepted_per_round for o in g]).json(),
             'depth': round(sum(o.draft_tokens for o in g) / sum(o.rounds for o in g), 3),
             'per_position': _jlist(position_rates(g))}
            for (cell, budget, batch), g in sorted(groups.items())]


def decompose(obs: Iterable[Obs], old: str, new: str, old_budget: int, new_budget: int) -> dict[str, Json]:
    """Walk one shift from (old prompt, old budget) to (new prompt, new budget) through the shared budget: prompt step at
    the old budget, then length step on the new prompt. `old`/`new` are prompt-name substrings."""
    rows = _b1(obs)
    names = prompt_names(rows)

    def pick(name: str, budget: int) -> list[float]:
        return [o.accepted_per_round for o in rows if name in names[o.prompt].split('+') and o.max_tokens == budget]

    a, b, c = pick(old, old_budget), pick(new, old_budget), pick(new, new_budget)
    if not a or not b or not c:
        raise ValueError(f'decompose {old}->{new}: missing cell(s): n={len(a)},{len(b)},{len(c)}')
    sa, sb, sc = summarize(a), summarize(b), summarize(c)
    return {'from': {'prompt': old, 'max_tokens': old_budget, **sa.json()},
            'via': {'prompt': new, 'max_tokens': old_budget, **sb.json()},
            'to': {'prompt': new, 'max_tokens': new_budget, **sc.json()},
            'prompt_step': round(sb.mean - sa.mean, 4), 'length_step': round(sc.mean - sb.mean, 4),
            'total': round(sc.mean - sa.mean, 4)}


def report(obs: Sequence[Obs], shifts: Mapping[str, tuple[str, str, int, int]]) -> dict[str, Json]:
    return {
        'schema': 'dsv41-acc-split-v1',
        'counts': {'bench2_runs': sum(o.source == 'bench2' for o in obs), 'window_cells': sum(o.source == 'cell' for o in obs),
                   'b1_used': len(_b1(obs))},
        'by_prompt': _jlist(by_prompt(obs)), 'by_prompt_date': _jlist(by_prompt_date(obs)), 'by_window': _jlist(by_window(obs)),
        'metric_reconciliation': _jlist(metric_reconciliation(obs)), 'length_split': _jlist(length_split(obs)),
        'segment_split': segment_split(obs), 'batch_split': _jlist(batch_split(obs)),
        'shifts': {name: decompose(obs, *spec) for name, spec in shifts.items()}}
