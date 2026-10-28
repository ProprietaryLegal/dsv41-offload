"""Frozen PP4 Round5 paired whole-group estimates; never resample speculative rounds."""
from __future__ import annotations

import math
import random
from collections.abc import Sequence
from fractions import Fraction
from typing import cast

from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, number, record

REGISTRATION = 'pp4-r6-512-aba-no-measurement-profiles-v1'
WAVES = 4
FAMILIES = ('opinions', 'filings')
RESAMPLES, SEED = 2000, 4103


def group_count(name: str) -> int:
    ck.require(name in ('short-b1', '32k-b1', '32k-b4'), 'unknown Round5 cell')
    return 2 if name == '32k-b4' else 4


def pool(rows: Sequence[dict[str, Json]]) -> dict[str, Json]:
    ck.require(bool(rows), 'no independent request-wave observations')
    rounds = accepted = seconds = 0.0
    for row in rows:
        r, a, t = (number(row[k]) for k in ('spec_request_rounds', 'accepted_tokens', 'server_decode_seconds'))
        ck.require(all(math.isfinite(v) for v in (r, a, t)) and r > 0 and t > 0
                   and r.is_integer() and a.is_integer() and 0 <= a <= 5*r, 'invalid measured round/acceptance/time totals')
        rounds += r
        accepted += a
        seconds += t
    ms = 1000*seconds/rounds
    return {'spec_request_rounds': rounds, 'accepted_tokens': accepted, 'server_decode_seconds': seconds,
            'acceptance_per_round': accepted/rounds, 'server_decode_ms_per_spec_request_round': ms,
            'expected_tps': 1000*(1+accepted/rounds)/ms}


def waves(cell: dict[str, Json], batch: int) -> list[dict[str, Json]]:
    ck.require(cell.get('metric_registration') == REGISTRATION and cell.get('batch') == batch,
               'Round5 metric registration/batch differs')
    rows = ck.rows(cell['waves'])
    ck.require(len(rows) == group_count(str(cell['cell'])) and [r.get('wave') for r in rows] == list(range(group_count(str(cell['cell'])))),
               'registered independent whole groups required in every cell per start')
    hashes: list[str] = []
    for i, row in enumerate(rows):
        ck.require(row.get('family') == FAMILIES[i % 2] and row.get('batch') == batch,
                   'frozen source-family/wave order differs')
        streams = ck.rows(row['streams'])
        ck.require(len(streams) == batch and row.get('completed_requests') == batch, 'completed request count differs')
        salts: list[str] = []
        for stream in streams:
            ck.require(stream.get('sampling') == {'temperature': 1, 'top_p': 1, 'seed': 42+i,
                'priority': 10, 'max_tokens': 512, 'ignore_eos': True}, 'registered512 sampling differs')
            ck.require(record(stream['usage']).get('completion_tokens') == 512 and stream.get('finish') == 'length',
                       'exact512 output/length finish required')
            salts.append(str(stream['cache_salt_sha256']))
            hashes.append(str(stream['prompt_sha256']))
        ck.require(len(set(salts)) == len(salts), 'duplicate stream salt within group')
        pool([row])
    ck.require(len(set(hashes)) == len(hashes), 'independent requests require distinct frozen prompt hashes')
    return rows


def interval(values: Sequence[float]) -> list[Json]:
    ordered = sorted(values)
    return [ordered[math.ceil(.025*len(ordered))-1], ordered[math.ceil(.975*len(ordered))-1]]


def start_interval(rows: Sequence[dict[str, Json]]) -> dict[str, Json]:
    ck.require(len(rows) in (2, 4), 'two B4 or four B1 whole groups required')
    point = pool(rows)
    rng = random.Random(SEED)
    draws: dict[str, list[float]] = {key: [] for key in (
        'acceptance_per_round', 'server_decode_ms_per_spec_request_round', 'expected_tps')}
    for _ in range(RESAMPLES):
        values = pool(rng.choices(rows, k=len(rows)))
        for key, draw in draws.items():
            draw.append(number(values[key]))
    return {**point, 'pooled_95pct': {key: interval(values) for key, values in draws.items()},
            'cluster': 'whole independent request groups; B4 streams intact; one start only',
            'bootstrap_seed': SEED, 'resamples': RESAMPLES}


def rate(rows: Sequence[dict[str, Json]]) -> Fraction:
    """Exact decimal counter arithmetic preserves strict-win/inclusive-loss boundaries."""
    return (sum((Fraction(str(r['spec_request_rounds']))+Fraction(str(r['accepted_tokens'])) for r in rows), Fraction())
            /sum((Fraction(str(r['server_decode_seconds'])) for r in rows), Fraction()))


def compare(starts: list[list[dict[str, Json]]]) -> dict[str, Json]:
    ck.require(len(starts) == 3 and len({len(rows) for rows in starts}) == 1 and len(starts[0]) in (2, 4),
               'three matched OFF/ON/OFF starts required')
    rates = [rate(rows) for rows in starts]
    reference = (rates[0]+rates[2])/2
    spread = abs(rates[0]-rates[2])
    need = max(reference/50, spread)
    delta = rates[1]-reference
    rng = random.Random(SEED)
    cache: dict[tuple[int, ...], tuple[list[dict[str, Json]], Fraction, Fraction]] = {}
    differences: list[Fraction] = []
    gains: list[float] = []
    samples: list[list[dict[str, Json]]] = [[], [], []]
    count = len(starts[0])
    for _ in range(RESAMPLES):
        indices = tuple(rng.randrange(count) for _ in range(count))
        if indices not in cache:
            selected = [[rows[i] for i in indices] for rows in starts]
            draw_rates = [rate(rows) for rows in selected]
            ref = (draw_rates[0]+draw_rates[2])/2
            cache[indices] = ([pool(rows) for rows in selected], draw_rates[1]-ref, draw_rates[1]/ref-1)
        points, difference, gain = cache[indices]
        differences.append(difference)
        gains.append(float(gain))
        for sample, point in zip(samples, points):
            sample.append(point)
    lower = sorted(differences)[math.ceil(.025*RESAMPLES)-1]
    return {'point_rates_tps': cast(Json, [float(r) for r in rates]), 'reference_tps': float(reference),
        'off_spread_tps': float(spread), 'point_delta_tps': float(delta), 'required_delta_tps': float(need),
        'point_gain': float(delta/reference), 'paired_gain_95pct': interval(gains),
        'paired_delta_95pct_tps': interval([float(d) for d in differences]),
        'win': delta > need and lower > need, 'no_loss_beyond_spread': delta >= -spread and lower >= -spread,
        'starts': [{**pool(rows), 'pooled_95pct': {key: interval([number(p[key]) for p in sample]) for key in (
            'acceptance_per_round', 'server_decode_ms_per_spec_request_round', 'expected_tps')}}
            for rows, sample in zip(starts, samples)],
        'bootstrap_seed': SEED, 'resamples': RESAMPLES, 'independent_groups': count,
        'interval_scope': 'paired whole-group percentile95%; same indices across starts; '
            'conditional on fixed cuts and starts; one ON start has no restart replication',
        'metric_registration': REGISTRATION}
