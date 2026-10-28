"""Synthetic completions rows: exact top20 membership, censored rank and fail-closed checks."""
import json
import math

import pytest

from d2 import generate as G
from d2 import generation_top20 as T


def wire(token=100, outside=False, tie=False):
    ids = list(range(100, 120))
    vals = [-math.log(100) - j * 0.1 for j in range(20)]
    if tie:
        vals[-1] = vals[-2]
    top = dict(zip(ids, vals))
    if outside:
        top[token] = vals[-1] if tie else vals[-1] - 1
    pairs = {f'token_id:{token}': top[token]}
    pairs.update({f'token_id:{k}': v for k, v in zip(ids, vals)})
    return {'tokens': [f'token_id:{token}'], 'token_logprobs': [top[token]], 'top_logprobs': [pairs],
            'text_offset': [0]}


@pytest.mark.parametrize('token,outside,tie', [(100, False, False), (999, True, False), (999, True, True)])
def test_rows_and_roundtrip(token, outside, tie):
    r = T.rows([token], wire(token, outside, tie))[0]
    assert r['token'] == token
    assert len(r['top_ids']) == 20
    assert r['tok_rank'] == (21 if outside else 1)
    assert r['resid'] == max(0.0, 1 - math.fsum(math.exp(v) for v in r['top_lp']))
    T.validate_saved([token], [r])


@pytest.mark.parametrize('bad', ['missing', 'short', 'own', 'nan', 'mass', 'id', 'clamp'])
def test_invalid_refuses(bad):
    w = wire()
    if bad == 'missing':
        w = None
    elif bad == 'short':
        w['top_logprobs'][0].pop('token_id:119')
    elif bad == 'own':
        w['token_logprobs'][0] -= 1
    elif bad == 'nan':
        w['top_logprobs'][0]['token_id:119'] = float('nan')
    elif bad == 'mass':
        w['top_logprobs'][0] = {f'token_id:{i}': 0.0 for i in range(100, 120)}
        w['token_logprobs'][0] = 0.0
    elif bad == 'id':
        w['top_logprobs'][0]['token_id:999999'] = w['top_logprobs'][0].pop('token_id:119')
    elif bad == 'clamp':
        w['top_logprobs'][0]['token_id:119'] = -9999
    with pytest.raises((ValueError, TypeError)):
        T.rows([100], w)


def test_default_request_byte_identical_and_opt_in():
    server = G.Server('http://127.0.0.1:1', 'm')
    off = G.Stream(server, 'p', 0, [1, 2], 3)
    expected = {'model': 'm', 'prompt': [1, 2], 'max_tokens': 3, 'temperature': 1.0,
                'top_p': 1.0, 'stream': True, 'stream_options': {'include_usage': True},
                'return_token_ids': True, 'priority': 20, 'request_id': off.request_id}
    assert json.dumps(off.body).encode() == json.dumps(expected).encode()
    on = G.Stream(server, 'p', 0, [1, 2], 3, top20=True)
    assert on.body == expected | {'logprobs': 20, 'return_tokens_as_token_ids': True}


def test_latch_preserves_coordinator_reason(tmp_path):
    pause = tmp_path / 'PAUSE'
    T.latch(pause, 'invalid row')
    T.latch(pause, 'another row')
    assert pause.read_text() == 'invalid row\n'


@pytest.mark.parametrize('excess,ok', [(1e-7, True), (5e-5, True), (2e-4, False)])
def test_mass_tolerance_matches_downstream_and_resid_clips(excess, ok):
    # S32 FIX-1: real rows reach mass 1 + ~1e-7; replay, shards and the trainer accept up to 1 + 1e-4 with resid >= 0.
    # Top-1 LP exactly 0.0 (fp32 logsumexp rounding, as in R6) plus 19 others carrying the excess; every LP <= 0.
    other = math.log(excess / 19)
    pairs = {'token_id:7': 0.0} | {f'token_id:{k}': other for k in range(1000, 1019)}
    w = {'tokens': ['token_id:7'], 'token_logprobs': [0.0], 'top_logprobs': [pairs], 'text_offset': [0]}
    if ok:
        r = T.rows([7], w)[0]
        assert r['resid'] == 0.0
        T.validate_saved([7], [r])
    else:
        with pytest.raises(ValueError, match='mass exceeds one'):
            T.rows([7], w)
