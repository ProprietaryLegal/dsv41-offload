"""R-SP on a fake TP2 group: the real K1 and overlay decoder code, bit-for-bit.

Each case runs K1's DeepseekV4Model/DecoderLayer, the overlay with SP off (threshold 0) and the overlay with SP on, two
ranks each, on the same toy weights and batch, and requires identical final hidden states, aux hidden states, KV writes
and MTP buffer rows. The SP-on run must really split (the row-wise stand-ins see L rows, the collectives change).
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest
import torch

from offload.sp_prefill import build_overlay as bo
from offload.sp_prefill.tests import toy
from offload.sp_prefill.tests.fake_tp import FakeTP

K1_SOURCE = (bo.K1 / bo.MODEL).read_text()
ON_SOURCE = bo.overlay_model(K1_SOURCE)
SP = toy.load_helper()


@pytest.fixture(autouse=True)
def _helper_on_cpu(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)  # no device here
    saved = (SP.engram_post, SP.forward_context)
    SP.engram_post = toy.engram_post_ref
    toy.REC.clear()
    yield
    SP.engram_post, SP.forward_context = saved
    SP.use_comm(None)
    toy.REC.clear()


Result = dict[str, Any]


def run(source: str, cfg: toy.Config, lengths: tuple[int, ...], *, replay_batch: bool = True,
        image_rows: tuple[int, ...] = ()) -> tuple[list[Result], FakeTP, dict[int, toy.Rec]]:
    tp = FakeTP(2)
    helper = SP if source is ON_SOURCE else None
    built = toy.Built(source, tp, helper)
    if helper is not None:
        SP.use_comm(tp)
        SP.forward_context = built._override
    toy.REC.clear()
    ids, positions = toy.batch(lengths)
    for r in image_rows:
        ids[r] = toy.IMAGE_TOKEN + r % 5
    lookback = torch.full((len(lengths), 2), -1)

    def rank(_: int) -> Result:
        model = built.model(cfg)
        if model.decoder_replay_layers is not None and replay_batch:
            model.decoder_replay_layers.rows = toy.replay_rows(lengths)
            model.decoder_replay_layers.forward_context = ('replay-ctx', lengths)
        out = model.forward(ids.clone(), positions.clone(), None, lookback_token_ids=lookback)
        hidden, aux = (out if isinstance(out, tuple) else (out, []))
        mtp = None if model._mtp_hidden_buffer is None else model._mtp_hidden_buffer[: ids.shape[0]].clone()
        return {'hidden': hidden, 'aux': list(aux), 'mtp': mtp}

    results = tp.run(rank)
    for r, res in enumerate(results):
        res['kv'] = built.kv_logs[r]
    return results, tp, dict(toy.REC)


def same(a: Result, b: Result) -> None:
    assert a['hidden'].dtype == b['hidden'].dtype and torch.equal(a['hidden'], b['hidden'])
    assert len(a['aux']) == len(b['aux'])
    for x, y in zip(a['aux'], b['aux']):
        assert x.shape == y.shape and torch.equal(x, y)
    assert (a['mtp'] is None) == (b['mtp'] is None)
    if a['mtp'] is not None:
        assert torch.equal(a['mtp'], b['mtp'])
    assert [(k, layer) for k, layer, _ in a['kv']] == [(k, layer) for k, layer, _ in b['kv']]
    for (_, _, x), (_, _, y) in zip(a['kv'], b['kv']):
        assert torch.equal(x, y)


CASES = {
    'replay-trim-odd': (toy.Config(), (37, 30)),            # T = 67: rank 1 carries one zero pad row
    'replay-trim-even': (toy.Config(), (40, 24)),
    'no-replay': (toy.Config(replay_start=None), (33, 21)),
    'mtp-buffer': (toy.Config(mtp_buffer=True), (29, 26)),
    'engram-late': (toy.Config(engram_layers=(1, 2)), (19, 36)),
    'one-request': (toy.Config(), (51,)),
}


@pytest.mark.parametrize('name', sorted(CASES))
def test_sp_is_bit_identical_to_k1(name: str) -> None:
    cfg, lengths = CASES[name]
    k1, k1_tp, _ = run(K1_SOURCE, cfg, lengths, image_rows=(3, 40 % sum(lengths)))
    off, off_tp, _ = run(ON_SOURCE, replace(cfg, min_rows=0), lengths, image_rows=(3, 40 % sum(lengths)))
    on, on_tp, rec = run(ON_SOURCE, replace(cfg, min_rows=8), lengths, image_rows=(3, 40 % sum(lengths)))
    for rank in range(2):
        same(k1[0], k1[rank])
        same(k1[rank], off[rank])
        same(k1[rank], on[rank])
    # OFF runs K1's collectives; ON swaps the per-layer all-reduces for gathers + reduce-scatters
    assert off_tp.calls == k1_tp.calls
    assert on_tp.calls.get('reduce_scatter', 0) > 0 and on_tp.calls['all_reduce'] < k1_tp.calls['all_reduce']
    T = sum(lengths)
    L = -(-T // 2)
    split_layers = cfg.replay_start if cfg.replay_start is not None else cfg.layers
    for r in range(2):
        pre_rows = rec[r].rows['mhc_pre']
        assert pre_rows[: 2 * split_layers] == [L] * (2 * split_layers), 'the full-row layers ran split'


def test_replay_without_a_batch_stays_split() -> None:
    """A forward with replay layers but no replay batch keeps SP through every layer (DecoderReplayLayers' rows None)."""
    cfg, lengths = toy.Config(), (45, 20)
    k1, _, _ = run(K1_SOURCE, cfg, lengths, replay_batch=False)
    on, _, rec = run(ON_SOURCE, replace(cfg, min_rows=8), lengths, replay_batch=False)
    for r in range(2):
        same(k1[r], on[r])
        assert set(rec[r].rows['mhc_pre']) == {33}


def test_below_threshold_is_the_k1_path() -> None:
    cfg, lengths = toy.Config(), (12, 9)
    k1, k1_tp, _ = run(K1_SOURCE, cfg, lengths)
    on, on_tp, rec = run(ON_SOURCE, replace(cfg, min_rows=22), lengths)
    for r in range(2):
        same(k1[r], on[r])
        assert set(rec[r].rows['mhc_pre']) <= {21, 8}  # 21 rows; the replay batch is 2 x 4
    assert on_tp.calls == k1_tp.calls


def test_engram_gate_runs_on_local_real_rows_only() -> None:
    """Pad rows pass through untouched; real rows match Engram.forward's rows."""
    tp = FakeTP(2)
    gen = torch.Generator().manual_seed(5)
    engram = toy.Engram(gen, 0)
    T = 9
    hidden = torch.randn(T, toy.HC, toy.HIDDEN, generator=gen).to(toy.DTYPE)
    hashes = torch.randint(0, 1000, (T, toy.N_HASH_COLS), generator=gen)
    mask = torch.tensor([True, False, True, True, True, False, True, True, True])
    full = engram(hidden, hashes, mask)

    def rank(r: int) -> torch.Tensor:
        rows = SP.SPRows(T, 2, r)
        local = rows.local_slice(hidden)
        if r == 1:
            local[-1] = 7.0  # a pad row with junk: it must pass through
        out: torch.Tensor = SP.engram_local(engram, rows, local, hashes, mask)
        if r == 1:
            assert torch.equal(out[-1], local[-1])
        return out[: rows.real_rows]

    SP.use_comm(tp)
    parts = tp.run(rank)
    assert torch.equal(torch.cat(parts), full)


def test_moe_restores_the_runner_and_refuses_reduced_configs() -> None:
    """One MoE per rank (a rank is its own process in vLLM; the toggle is per runner)."""
    tp = FakeTP(2)
    moes = [toy.MoE(tp, torch.Generator().manual_seed(3)) for _ in range(2)]
    x = torch.randn(10, toy.HIDDEN, generator=torch.Generator().manual_seed(4)).to(toy.DTYPE)
    ids = torch.arange(10)
    SP.use_comm(tp)

    def rank(r: int) -> torch.Tensor:
        rows = SP.SPRows(10, 2, r)
        out: torch.Tensor = SP.moe(moes[r], rows, rows.local_slice(x), ids)
        assert moes[r].experts.moe_config.skip_final_all_reduce is False
        return out

    full = tp.run(lambda r: moes[r](x, ids))[0]
    assert torch.equal(torch.cat(tp.run(rank)), full)
    moes[0].experts.moe_config.skip_final_all_reduce = True
    with pytest.raises(RuntimeError, match='own all-reduce'):
        SP.moe(moes[0], SP.SPRows(10, 2, 0), x[:5], ids)
    moes[0].experts.moe_config.skip_final_all_reduce = False
    with pytest.raises(ValueError, match='input_ids'):
        SP.moe(moes[0], SP.SPRows(10, 2, 0), x[:5], None)


def test_unreduced_restores_on_error() -> None:
    linear = toy.RowLinear(FakeTP(2), torch.zeros(1))
    with pytest.raises(KeyError):
        with SP.unreduced(linear):
            assert linear.reduce_results is False
            raise KeyError('boom')
    assert linear.reduce_results is True
    linear.reduce_results = False
    with pytest.raises(RuntimeError, match='already unreduced'):
        with SP.unreduced(linear):
            pass


def _engram_from_row_zero(engram: Any, rows: Any, hidden: torch.Tensor, hashes: torch.Tensor,
                          mask: torch.Tensor | None) -> torch.Tensor:
    kv = engram.wkv(engram.embed(hashes).flatten(-2))
    out: torch.Tensor = SP.engram_post(engram, hidden, kv[: rows.real_rows],
                                       None if mask is None else mask[: rows.real_rows], rows.real_rows)
    return out


def _moe_reduced_twice(ffn: Any, rows: Any, x: torch.Tensor, ids: torch.Tensor | None) -> torch.Tensor:
    full = rows.gather(x)
    out: torch.Tensor = rows.local_slice(ffn.experts(hidden_states=full, router_logits=full, input_ids=ids))
    return out * 2


MUTATIONS: dict[str, tuple[Any, str, Any]] = {
    'engram-offset': (SP, 'engram_local', _engram_from_row_zero),
    'replay-select': (SP.SPRows, 'select', lambda self, x, idx: x.index_select(0, idx.long() % self.local_rows)),
    'moe-double': (SP, 'moe', _moe_reduced_twice),
    'rs-missing': (SP.SPRows, 'reduce_scatter', lambda self, p: self.local_slice(p)),
}


@pytest.mark.parametrize('name', sorted(MUTATIONS))
def test_a_wrong_dataflow_is_caught(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The comparison is sharp: each deliberate dataflow bug breaks bit-identity."""
    owner, attr, bad = MUTATIONS[name]
    monkeypatch.setattr(owner, attr, bad)
    with pytest.raises(AssertionError):
        test_sp_is_bit_identical_to_k1('replay-trim-odd')
