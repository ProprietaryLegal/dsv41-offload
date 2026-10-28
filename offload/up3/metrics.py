"""Request-group pooled UP3 decision (S57-cleared DEP1 arithmetic); no token-event resampling or client-rate ranking."""
from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import cast

from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, number, record

RESAMPLES = 2000
SEED = 4101
SCOPE = ('server request-decode duration sum / speculative request-round count; '
         'terminal effects and omitted zero-proposal rounds remain; not engine-step wall time')


@dataclass(frozen=True)
class Group:
    rounds: float
    accepted: float
    decode_ms: float


def positive(value: Json, label: str, *, zero: bool = False) -> float:
    result = number(value)
    ck.require(math.isfinite(result) and (result >= 0 if zero else result > 0), f'invalid {label}')
    return result


def groups(cell: dict[str, Json], batch: int, repeats: int) -> list[Group]:
    rows = ck.rows(cell['repetitions'])
    ck.require(len(rows) == repeats and repeats * batch >= 4, 'at least four requests per cell per start required')
    result: list[Group] = []
    fingerprints: list[Json] | None = None
    salts: set[str] = set()
    for i, row in enumerate(rows):
        if cell.get('cell') == 'mixed150k-2decode':
            ck.require(row.get('mixed_three_active_overlap') is True, 'mixed three-request admission/overlap unproved')
        ck.require(row.get('repeat') == i and row.get('batch') == batch, 'repetition identity mismatch')
        streams = ck.rows(row['streams'])
        ck.require(len(streams) == batch, 'repetition stream count mismatch')
        ck.require(all(s.get('budget') == 512 and s.get('finish') == 'length'
                   and record(s['usage']).get('completion_tokens') == 512
                   and record(s['usage']).get('prompt_tokens') == s.get('prompt_tokens')
                   and record(record(s['usage']).get('prompt_tokens_details') or {}).get('cached_tokens') == 0
                   for s in streams), 'incomplete output/prompt/cache evidence')
        hashes: list[Json] = [[s['prompt_sha256'], s['prompt_tokens'], s['budget']] for s in streams]
        ck.require(fingerprints is None or fingerprints == hashes, 'repeated prompt mismatch')
        fingerprints = hashes
        for stream in streams:
            sampling = record(stream['sampling'])
            ck.require(sampling == {'temperature': 1, 'top_p': 1, 'seed': 42 + i,
                                    'priority': 10, 'max_tokens': 512}, 'registered sampling mismatch')
            salt = str(stream['cache_salt_sha256'])
            ck.require(salt not in salts, 'reused cold cache salt')
            salts.add(salt)
        rounds = positive(row['spec_request_rounds'], 'round count')
        accepted = positive(row['accepted_tokens'], 'accepted count', zero=True)
        ck.require(rounds.is_integer() and accepted.is_integer() and accepted <= 5 * rounds,
                   'invalid speculative counter totals')
        ck.require(row.get('completed_requests') == batch, 'server completed-request count mismatch')
        result.append(Group(rounds, accepted, positive(row['server_decode_ms'], 'server decode time')))
    return result


def pooled(rows: Sequence[Group]) -> tuple[float, float, float]:
    ck.require(bool(rows), 'empty pool')
    rounds = sum(r.rounds for r in rows)
    accepted = sum(r.accepted for r in rows) / rounds
    ms = sum(r.decode_ms for r in rows) / rounds
    return accepted, ms, (1 + accepted) / ms


def rate(rows: Sequence[Group]) -> Fraction:
    """Exact arithmetic over recorded decimal counters preserves strict boundaries."""
    return (sum((Fraction(str(r.rounds)) + Fraction(str(r.accepted)) for r in rows), Fraction())
            / sum((Fraction(str(r.decode_ms)) for r in rows), Fraction()))


def interval(values: list[float]) -> list[Json]:
    values = sorted(values)
    # Predeclared nearest-rank percentile interval, not a normal/token-event CI.
    return [values[math.ceil(.025 * len(values)) - 1], values[math.ceil(.975 * len(values)) - 1]]


def compare(cells: list[dict[str, Json]], batch: int, repeats: int) -> dict[str, Json]:
    ck.require(len(cells) in (3, 5), 'three or five matched starts required')
    candidates = tuple(range(1, len(cells), 2))
    rows = [groups(c, batch, repeats) for c in cells]
    hashes = [[(s['prompt_sha256'], s['prompt_tokens'], s['budget'])
               for r in ck.rows(c['repetitions']) for s in ck.rows(r['streams'])] for c in cells]
    ck.require(all(h == hashes[0] for h in hashes), 'public prompt mismatch')
    points = [pooled(r) for r in rows]
    rates = [rate(r) for r in rows]
    spread = max(rates[::2]) - min(rates[::2])
    neighbors = [(rates[i-1] + rates[i+1])/2 for i in candidates]
    delta = [rates[i] - ref for i, ref in zip(candidates, neighbors)]
    required = [max(ref / 50, spread) for ref in neighbors]
    rng = random.Random(SEED)
    samples: list[list[tuple[float, float, float]]] = [[] for _ in rows]
    differences: list[list[Fraction]] = [[] for _ in candidates]
    gains: list[list[float]] = [[] for _ in candidates]
    cache: dict[tuple[int, ...], tuple[list[tuple[float, float, float]], list[Fraction]]] = {}
    for _ in range(RESAMPLES):
        # Match repeat slots across all starts; B3's simultaneous requests stay
        # together as ONE group, never three independent samples.
        indices = tuple(rng.randrange(repeats) for _ in range(repeats))
        if indices not in cache:
            selected = [[r[i] for i in indices] for r in rows]
            cache[indices] = ([pooled(r) for r in selected], [rate(r) for r in selected])
        draws, draw_rates = cache[indices]
        for dest, draw in zip(samples, draws):
            dest.append(draw)
        for j, index in enumerate(candidates):
            ref = (draw_rates[index-1] + draw_rates[index+1])/2
            differences[j].append(draw_rates[index] - ref)
            gains[j].append(float(100 * (draw_rates[index]/ref - 1)))
    deltas_ci = [interval([float(value) for value in d]) for d in differences]
    lower = [sorted(d)[math.ceil(.025 * len(d)) - 1] for d in differences]
    # Require point AND interval evidence. Four repeated requests on one cut
    # cannot establish task-family generalisation; this interval is conditional.
    win = all(d > need and low > need for d, need, low in zip(delta, required, lower))
    no_loss = all(d >= -spread and low >= -spread for d, low in zip(delta, lower))
    return {'rate_unit': 'estimated committed tokens/ms/request', 'scope': SCOPE,
            'point_rates': cast(Json, [float(r) for r in rates]), 'C_spread_tokens_per_ms': float(spread),
            'paired_gain_percent': cast(Json, [float(100 * d/ref) for d, ref in zip(delta, neighbors)]),
            'paired_gain_percent_interval95': cast(Json, [interval(g) for g in gains]),
            'paired_delta_interval95': cast(Json, deltas_ci),
            'required_delta': cast(Json, [float(r) for r in required]), 'win': win, 'no_loss_beyond_spread': no_loss,
            'start_estimates': [{'accepted_per_round': point[0], 'ms_per_round': point[1],
                'committed_per_ms': point[2], 'requests': batch * repeats, 'independent_groups': repeats,
                'accepted_per_round_interval95': interval([s[0] for s in sample]),
                'ms_per_round_interval95': interval([s[1] for s in sample]),
                'committed_per_ms_interval95': interval([s[2] for s in sample])}
                for point, sample in zip(points, samples)],
            'interval_method': f'paired whole request-group percentile bootstrap; n={repeats}; '
                               f'{RESAMPLES} draws, seed {SEED}; conditional on fixed prompts and starts; '
                               'not task-family or restart uncertainty'}
