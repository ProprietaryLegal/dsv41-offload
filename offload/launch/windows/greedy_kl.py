"""WIN2 greedy refinement (Fable review): distribution check at the first greedy divergence.

A near-tie only says the two competing tokens were close. This adds the question "did the whole next-token
distribution change?": in a control/candidate/control window the candidate arm's greedy check records the top-20 and
the prefix at each divergence (nbw_cells.greedy(record_distribution=True)); the following control arm re-probes the
same prefixes (fresh cache salt) and top20_kl measures KL(control || candidate) over the shared top-20 tokens plus one
OTHER bucket, the same coarse estimator UP2's screen uses (up2_checks.numerical). The bound is a registration choice
of each runner; there is no default.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import cast

from offload.gb10_budget import window_cells as wc
from offload.launch.windows import nbw_cells as nbw
from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, number, record

MIN_COMMON_MASS = .99


def _probabilities(top: Mapping[str, Json]) -> dict[int, float]:
    ck.require(len(top) == 20, 'exactly twenty top-logprob tokens required')
    out: dict[int, float] = {}
    for key, value in top.items():
        ck.require(key.startswith('token_id:') and key[9:].isdigit(), 'malformed top-20 token id')
        lp = number(value)
        ck.require(-9999 < lp <= 0, 'clamped or invalid logprob')
        out[int(key[9:])] = math.exp(lp)
    ck.require(sum(out.values()) <= 1 + 1e-6, 'top-20 probabilities exceed one')
    return out


def top20_kl(reference: Mapping[str, Json], experiment: Mapping[str, Json]) -> dict[str, float]:
    """KL(reference || experiment) over shared top-20 tokens + OTHER; common_mass = smaller shared probability mass."""
    p, q = _probabilities(reference), _probabilities(experiment)
    keys = sorted(p.keys() & q.keys())
    px, qx = [p[k] for k in keys], [q[k] for k in keys]
    common = min(sum(px), sum(qx))
    px.append(max(0.0, 1 - sum(px)))
    qx.append(max(0.0, 1 - sum(qx)))
    kl = sum(v * math.log(v / w) if w > 0 else math.inf for v, w in zip(px, qx) if v > 0)
    ck.require(math.isfinite(kl), 'nonfinite coarse KL at the divergence')
    return {'kl_nats': kl, 'common_mass': common, 'shared_tokens': float(len(keys))}


def passes(result: Mapping[str, float], bound: float) -> bool:
    if not (bound > 0 and math.isfinite(bound)):
        raise ValueError('a positive registered KL bound is required')
    return result['common_mass'] >= MIN_COMMON_MASS and result['kl_nats'] <= bound


def probe(api: wc.Http, prefix: Sequence[int], salt: str) -> dict[str, Json]:
    """Top-20 at an exact prefix from the server under test, with a fresh cache salt."""
    ck.require(bool(salt.strip()), 'probe cache salt required')
    r = api.post('/v1/completions', {'model': nbw.MODEL, 'prompt': cast(Json, list(prefix)), 'max_tokens': 1,
                                     'temperature': 0, 'logprobs': 20, 'return_tokens_as_token_ids': True,
                                     'priority': 10, 'cache_salt': salt})
    tops = record(record(ck.rows(r['choices'])[0])['logprobs'])['top_logprobs']
    ck.require(isinstance(tops, list) and len(tops) == 1, 'one top-logprob position expected')
    return record(cast(list[Json], tops)[0])


def compare_recorded(api: wc.Http, candidate_rows: Sequence[Mapping[str, Json]], salt_prefix: str,
                     bound: float) -> list[dict[str, Json]]:
    """Re-probe every recorded candidate divergence on the CURRENT (control) server and measure the KL."""
    out: list[dict[str, Json]] = []
    for row in candidate_rows:
        if row.get('identical') is True:
            continue
        ck.require('top20_logprobs' in row and 'probe_prefix_token_ids' in row,
                   'candidate divergence lacks a recorded distribution (record_distribution=True)')
        prefix = nbw.token_ids(row['probe_prefix_token_ids'])
        control = probe(api, prefix, f'{salt_prefix}:kl:{row.get("i")}')
        result = top20_kl(control, record(row['top20_logprobs']))
        out.append({'i': row.get('i'), **{k: v for k, v in result.items()}, 'bound': bound,
                    'pass': passes(result, bound)})
    return out
