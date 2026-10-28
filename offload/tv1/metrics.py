"""Counter-based TV1 decision metric; request-rounds are not engine iterations."""
from __future__ import annotations

import math
from typing import cast

from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, number, record

SCOPE = ('server request-decode duration sum / speculative request-round count; '
         'includes request terminal effects; not engine iteration wall time')
NAMES = {'rounds': 'vllm:spec_decode_num_drafts_total',
         'accepted_tokens': 'vllm:spec_decode_num_accepted_tokens_total',
         'proposed_tokens': 'vllm:spec_decode_num_draft_tokens_total',
         'decode_seconds': 'vllm:request_decode_time_seconds_sum',
         'decode_requests': 'vllm:request_decode_time_seconds_count',
         'completed_requests': 'vllm:request_success_total'}


def delta(before: dict[str, float], after: dict[str, float], name: str) -> float:
    keys = {key for key in before.keys() | after.keys() if key.split('{')[0] == name}
    ck.require(bool(keys) and all(key in after and math.isfinite(after[key])
               and math.isfinite(before.get(key, 0)) and after[key] >= before.get(key, 0) >= 0 for key in keys),
               'TV1 missing/reset/nonfinite server counter: '+name)
    return sum(after[key]-before.get(key, 0) for key in keys)


def counters(before: dict[str, float], after: dict[str, float], batch: int) -> dict[str, Json]:
    values = {key: delta(before, after, name) for key, name in NAMES.items()}
    ck.require(values['decode_requests'] == values['completed_requests'] == batch,
               'TV1 counter interval contains missing/foreign/unfinished requests')
    ck.require(values['rounds'] > 0 and values['decode_seconds'] > 0
               and 0 <= values['accepted_tokens'] <= values['proposed_tokens'] and values['proposed_tokens'] > 0,
               'TV1 invalid speculative/decode counter delta')
    return {**values, 'metrics_before': cast(Json, before), 'metrics_after': cast(Json, after)}


def pool(trials: list[dict[str, Json]], *, minimum_requests: int = 4) -> dict[str, Json]:
    ck.require(bool(trials), 'TV1 no request trials')
    totals = {key: 0. for key in NAMES}
    prior: dict[str, float] | None = None
    for trial in trials:
        raw_before = record(record(trial['server_counters'])['metrics_before'])
        raw_after = record(record(trial['server_counters'])['metrics_after'])
        before = {key: number(value) for key, value in raw_before.items()}
        after = {key: number(value) for key, value in raw_after.items()}
        if prior is not None:
            ck.require(all(key in before and before[key] >= value for key,value in prior.items()
                           if key.split('{')[0] in NAMES.values()),
                       'TV1 repeated counter intervals overlap/reset; requests cannot be counted twice')
        prior = after
        batch = int(number(trial['batch']))
        ck.require(number(trial['batch']) == batch and batch in (1, 4) and len(ck.rows(trial['streams'])) == batch, 'TV1 trial batch mismatch')
        checked = counters(before, after, batch)  # recompute; saved summaries cannot waive raw evidence
        for key in totals:
            ck.require(number(record(trial['server_counters'])[key]) == number(checked[key]),
                       'TV1 saved counter delta differs from raw metrics')
            totals[key] += number(checked[key])
    requests = totals['completed_requests']
    ck.require(requests >= minimum_requests, f'TV1 acceptance needs >= {minimum_requests} completed requests per cell')
    return {**totals, 'requests': int(requests),
            'server_decode_ms_per_spec_request_round': 1000*totals['decode_seconds']/totals['rounds'],
            'acceptance_fraction': totals['accepted_tokens']/totals['proposed_tokens'],
            'mean_accepted_per_round': totals['accepted_tokens']/totals['rounds'],
            'round_time_scope': SCOPE}
