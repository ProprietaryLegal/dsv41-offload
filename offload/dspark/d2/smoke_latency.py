"""Repeated smoke latency qualification, with no network or service operations."""
from __future__ import annotations

import math
from statistics import median
from typing import cast

from .decode_dataset import mapping, number, text
from .replay_window import gauges, require

REPEATS = 5
MAX_ATTEMPTS = 10
MAX_LATENCY_REGRESSION = .05
KEYS = ('ttft_s', 'step_mean_s')


def medians(samples: list[dict[str, float]], count: int = REPEATS) -> dict[str, float]:
    require(len(samples) == count, f'exactly {count} latency samples required')
    require(all(math.isfinite(s[k]) and s[k] > 0 for s in samples for k in KEYS),
            'invalid latency sample')
    return {k: median(s[k] for s in samples) for k in KEYS}


def baseline_samples(proof: dict[str, object]) -> list[dict[str, float]]:
    # Import late: decode_smoke also uses this module for its gate.
    from .decode_smoke import latency_delta
    require(proof.get('schema') == 'd2.k1-greedy-reference.v2'
            and proof.get('timing_qualified') is True, 'five-repeat K1 receipt required')
    warm = mapping(proof.get('warmup'))
    require(warm.get('output_identical') is True and warm.get('accepted') is True,
            'verified K1 warm-up required')
    rows = proof.get('samples')
    require(isinstance(rows, list) and len(rows) == REPEATS, 'five warm K1 repeats required')
    result: list[dict[str, float]] = []
    previous_attempt = number(warm.get('attempt'))
    require(1 <= previous_attempt <= MAX_ATTEMPTS, 'invalid K1 warm-up attempt')
    for raw in cast(list[object], rows):
        row = mapping(raw)
        attempt = number(row.get('attempt'))
        require(previous_attempt < attempt <= MAX_ATTEMPTS, 'K1 repeats not distinct or exceed attempt limit')
        previous_attempt = attempt
        require(row.get('output_identical') is True and row.get('accepted') is True,
                'unqualified K1 repeat')
        require(row.get('cached_prompt_tokens') == proof.get('cached_prompt_tokens'),
                'mixed K1 cache conditions')
        require(gauges(text(row.get('before'))) and gauges(text(row.get('after'))),
                'busy K1 repeat boundaries')
        result.append(latency_delta(text(row['before']), text(row['after'])))
    medians(result)
    return result


def latency_gate(baseline: list[dict[str, float]], measured: list[dict[str, float]]) -> dict[str, object]:
    reference, candidate = medians(baseline), medians(measured)
    ratios = {k: candidate[k] / reference[k] for k in KEYS}
    require(all(r <= 1 + MAX_LATENCY_REGRESSION + 1e-12 for r in ratios.values()),
            'untagged median TTFT/step latency regressed above 5% against K1')
    return {'baseline': reference, 'measured': candidate, 'ratios': ratios,
            'baseline_samples': baseline, 'measured_samples': measured,
            'statistic': 'median', 'repeats': REPEATS,
            'max_regression_fraction': MAX_LATENCY_REGRESSION, 'pass': True}
