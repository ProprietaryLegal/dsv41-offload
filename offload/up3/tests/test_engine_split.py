from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import torch

from offload.prefill_maxlen.evidence import Json, record
from offload.up3 import engine_split as s
from offload.up3.engine_common import invoke, replaced


class FakeSmoe:
    top_k = 6

    def max_rows(self) -> int:
        return 42

    def __call__(self, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, perm: torch.Tensor,
                 limit: float, pad: torch.Tensor | None = None) -> torch.Tensor:
        assert len(x) <= 42
        out = (x.float() + (ids.float()*weights).sum(dim=-1, keepdim=True)).to(x.dtype)
        if pad is not None:
            out[pad] = 0
        return out


class FakeForward:
    _SPLIT_MAX = 63

    def __init__(self) -> None:
        self.rows: list[int] = []
        self.corrupt = False

    def smoe_split_ok(self, m: int, x: torch.Tensor, decode: str, coop: object, smoe: object) -> bool:
        return 32 < m <= self._SPLIT_MAX

    def routed_moe_forward(self, x: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor, perm: torch.Tensor,
                           *args: object, **kwargs: object) -> torch.Tensor:
        self.rows.append(len(x))
        smoe = kwargs['smoe']
        pad = kwargs.get('pad')
        mask = pad if isinstance(pad, torch.Tensor) else None
        result = torch.cat([cast(torch.Tensor, invoke(smoe, x[a:a+42], ids[a:a+42], weights[a:a+42], perm, 10.,
            None if mask is None else mask[a:a+42])) for a in range(0, len(x), 42)])
        if self.corrupt and len(x) == 43:
            result[42, 0] += 1
        return result


def layer() -> SimpleNamespace:
    return SimpleNamespace(bank=SimpleNamespace(hidden=8, num_experts=384), smoe=FakeSmoe(),
        perm=torch.arange(384), act_limit=10., groups=[], coop=SimpleNamespace(rows_max=32),
        fused=None, mcg=False, mul1=True, settings=SimpleNamespace(decode_kernel='coop'), par=None)


def test_split_calls_real_dispatch_all_rows_and_oracle_capacity() -> None:
    f = FakeForward()
    with patch.object(torch.random, 'fork_rng', lambda **kw: nullcontext()), patch.object(torch.cuda, 'synchronize'):
        rows = s.check_layer(layer(), f)
    assert f.rows == [33, 33, 42, 42, 43, 43, 63, 63, 32, 32, 64, 64]
    assert set(rows) == {'m33', 'm42', 'm43', 'm63', 'dispatch32', 'dispatch64'}
    assert record(rows['m43'])['oracle'] == 'per-row-decode'
    assert record(rows['m63'])['bytes_compared'] == 63*8*2*2
    assert f._SPLIT_MAX == 63


def test_split_single_byte_failure_stops_and_restores_switch() -> None:
    f = FakeForward()
    f.corrupt = True
    with patch.object(torch.random, 'fork_rng', lambda **kw: nullcontext()), patch.object(torch.cuda, 'synchronize'):
        with pytest.raises(ValueError, match='m43: output bytes differ'):
            s.check_layer(layer(), f)
    assert f._SPLIT_MAX == 63
    with pytest.raises(RuntimeError):
        with replaced(f, '_SPLIT_MAX', 0):
            raise RuntimeError('signal')
    assert f._SPLIT_MAX == 63


def test_oracle_must_be_unchanged_stock_body() -> None:
    stock = (Path(s.__file__).parent/'stock_smoe.py.txt').read_bytes()
    s.unchanged_call(stock, stock)
    with pytest.raises(ValueError, match='oracle'):
        s.unchanged_call(stock.replace(b'm = x.shape[0]', b'm = 1'), stock)


def test_rounding_full_vocab_and_bound_review() -> None:
    review: dict[str, Json] = {'run': 'test', 'task_review': 'pass', 'same_prefix_and_batch_shape': True,
        **{k: 'a'*64 for k in ('prefix_sha256', 'batch_sha256', 'probe_sha256')}}
    x = torch.zeros((3, 128))
    review['probe_sha256'] = s.logits_sha256((x, x, x))
    assert record(s.rounding(x, x, x, review, run='test'))['kl'] == 0
    for bad in (x + float('nan'), torch.eye(3, 128)*100):
        current = {**review, 'probe_sha256': s.logits_sha256((x, x, bad))}
        with pytest.raises(ValueError):
            s.rounding(x, x, bad, current, run='test')
    with pytest.raises(ValueError, match='review'):
        s.rounding(x, x, x, review, run='other')
    with pytest.raises(ValueError, match='exact C/C/A'):
        s.rounding(x, x, x+1, review, run='test')
