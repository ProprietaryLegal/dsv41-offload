"""CPU validation for the pinned image's completions logprob wire format."""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict

SCHEMA = 'd2.generation-top20.v1'
IMAGE = 'sha256:435a5236ce3a7db9d45f5a2c483817d60ea629c4ea0a0b3af33da2ef8122334b'


class Row(TypedDict):
    token: int
    tok_lp: float
    tok_rank: int
    top_ids: list[int]
    top_lp: list[float]
    resid: float


def latch(pause: Path, reason: str) -> None:
    pause.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pause.open('x') as fh:
            fh.write(reason + '\n')
    except FileExistsError:
        pass  # The coordinator's existing PAUSE reason is preserved.


def token_id(value: object) -> int:
    if not isinstance(value, str) or not re.fullmatch(r'token_id:\d+', value):
        raise ValueError('missing token_id placeholder')
    number = int(value.split(':')[1])
    if not 0 <= number < 129280:
        raise ValueError('token ID outside target vocabulary')
    return number


def lp(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError('non-numeric LP')
    result = float(value)
    # The formatter clamps -inf to -9999: never certify a clamped value as exact.
    if not math.isfinite(result) or not -9999 < result <= 0:
        raise ValueError('non-finite, positive or clamped LP')
    return result


def row(token: int, label: object, own: object, raw: object) -> Row:
    if (not isinstance(token, int) or isinstance(token, bool)
            or token_id(label) != token or not isinstance(raw, dict)):
        raise ValueError('committed token mismatch or missing top20 map')
    pairs = [(token_id(k), lp(v)) for k, v in raw.items()]
    if len(pairs) not in (20, 21) or len({k for k, _ in pairs}) != len(pairs):
        raise ValueError('expected 20 or 21 distinct IDs')
    own_lp = lp(own)
    mine = [v for k, v in pairs if k == token]
    if mine != [own_lp]:
        raise ValueError('committed LP missing or not exact')
    # For 21 entries the extra entry is the sample (possibly tied at the edge).
    # Preserve the server's chosen top20 membership even for boundary ties.
    top = [p for p in pairs if p[0] != token] if len(pairs) == 21 else pairs
    top.sort(key=lambda p: (-p[1], p[0]))
    if len(top) != 20 or (len(pairs) == 21 and own_lp > top[-1][1]):
        raise ValueError('sample plus top20 ordering invalid')
    mass = math.fsum(math.exp(v) for _, v in top)
    # Real rows exceed 1 by ~1e-7 (fp32 logsumexp; S32): same 1e-4 tolerance as replay, shards and the trainer.
    if mass > 1 + 1e-4:
        raise ValueError('top20 probability mass exceeds one')
    ids = [k for k, _ in top]
    return {'token': token, 'tok_lp': own_lp, 'tok_rank': ids.index(token) + 1 if token in ids else 21,
            'top_ids': ids, 'top_lp': [v for _, v in top], 'resid': max(0.0, 1 - mass)}


def rows(ids: Sequence[int], raw: object) -> list[Row]:
    if not isinstance(raw, Mapping):
        raise TypeError('missing generation logprobs')
    fields = [raw.get(k) for k in ('tokens', 'token_logprobs', 'top_logprobs')]
    if any(not isinstance(v, list) or len(v) != len(ids) for v in fields):
        raise ValueError('generation logprob arrays do not match committed IDs')
    labels, own, top = fields
    assert isinstance(labels, list) and isinstance(own, list) and isinstance(top, list)
    return [row(t, labels[j], own[j], top[j]) for j, t in enumerate(ids)]


def validate_saved(ids: Sequence[int], saved: Sequence[Row]) -> None:
    if len(ids) != len(saved):
        raise ValueError('saved generation rows and IDs are not 1:1')
    for token, value in zip(ids, saved):
        raw = {f'token_id:{k}': v for k, v in zip(value['top_ids'], value['top_lp'], strict=True)}
        raw[f'token_id:{token}'] = value['tok_lp']
        expected = row(token, f'token_id:{token}', value['tok_lp'], raw)
        if value != expected:
            raise ValueError('saved generation row failed revalidation')
