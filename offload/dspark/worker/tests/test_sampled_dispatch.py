"""CPU checks of the actual K2 branch boundary; GPU numerics have a solo probe."""
from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from offload.dspark.worker import fused as fused_ops
from offload.dspark.worker import sampled_fused, tp2


def test_sampled_replaces_original_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("duplicate original selector")

    def select(ranks: Any, logits: Any, prev: Any, outputs: Any, position: int) -> None:
        calls.append(position)
        for out in outputs.values():
            out[:, position] = 9 + position

    def gemm(rows: torch.Tensor, out: torch.Tensor) -> None:
        out.zero_()

    rank = SimpleNamespace(r=0, dev=torch.device("cpu"), v0=0, v1=64,
                           head=SimpleNamespace(gemm=gemm), norm=torch.ones(2),
                           markov=SimpleNamespace(embed=lambda ids: torch.zeros(*ids.shape, 1)),
                           conf_w=torch.zeros(3))
    model = SimpleNamespace(s=SimpleNamespace(hidden=2), eps=1e-6, HEAD_CHUNK=16,
                            ranks=[rank], comm=SimpleNamespace(real=False),
                            _pp4_context=SimpleNamespace(select=select))
    model._each = lambda ranks, fn: {r: fn(rank) for r in ranks}
    monkeypatch.setattr(tp2, "markov_local_argmax", forbidden)
    monkeypatch.setattr(tp2, "markov_partials", forbidden)
    monkeypatch.setattr(fused_ops, "rms_norm", lambda hh, norm, eps: hh)
    monkeypatch.setattr(torch.cuda, "device", lambda dev: nullcontext())
    result, conf, margins = tp2.TPDrafter.heads(cast(tp2.TPDrafter, model), (0,), {0: torch.ones(5, 2)},
                                               {0: (None, None, None, torch.zeros(1, dtype=torch.int64))}, 1, 5)
    assert calls == list(range(5))
    assert result[0].tolist() == [[9, 10, 11, 12, 13]]
    assert torch.equal(conf, torch.full((1, 5), .5))
    assert margins is None


def test_all_t0_reuses_original_graph_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DSPARK_SAMPLED_TOP32", "1")
    ids = torch.tensor([[7, 8, 9, 10, 0]])
    conf = torch.tensor([[.1, .2, .3, .4, 0.]])
    called: list[int] = []

    def graph(inp: torch.Tensor, k: int, n_out: int) -> tuple[torch.Tensor, torch.Tensor, None]:
        called.append(n_out)
        return ids, conf, None

    model = SimpleNamespace(dev=torch.device("cpu"), has_graph=lambda *args: True, draft_graphed=graph)
    actual, actual_conf, support, q = tp2.TPDrafter.draft_sampled(
        cast(tp2.TPDrafter, model), torch.zeros(4, 1, dtype=torch.int64), 5, torch.zeros(1, 3, dtype=torch.int64),
        torch.zeros(1, dtype=torch.bool), n_out=4)
    assert actual is ids and actual_conf is conf
    assert called == [4]
    assert support.shape == q.shape == (1, 5, 32)
    assert torch.count_nonzero(q).item() == 0


def test_missing_sampled_graph_refuses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DSPARK_SAMPLED_TOP32", "1")
    model = SimpleNamespace(dev=torch.device("cpu"), graphs={(1, 5, 5): {}})
    with pytest.raises(RuntimeError, match="sampled graph missing"):
        tp2.TPDrafter.draft_sampled(cast(tp2.TPDrafter, model), torch.zeros(4, 1, dtype=torch.int64), 5,
                                  torch.zeros(1, 3, dtype=torch.int64), torch.ones(1, dtype=torch.bool))


def test_off_refuses_sampled_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DSPARK_SAMPLED_TOP32", raising=False)
    with pytest.raises(RuntimeError, match="explicit switch"):
        tp2.TPDrafter.draft_sampled(cast(tp2.TPDrafter, SimpleNamespace()), torch.zeros(4, 1, dtype=torch.int64), 5,
                                  torch.zeros(1, 3, dtype=torch.int64), torch.ones(1, dtype=torch.bool))


def paired_fixture() -> Any:
    ranks = []
    for r in (0, 1):
        markov = SimpleNamespace(slot_of_token=torch.zeros(128, dtype=torch.int64),
                                 cache=torch.zeros(1, 64, dtype=torch.bfloat16),
                                 w1=torch.zeros(128, 256, dtype=torch.bfloat16),
                                 w2=torch.zeros(64, 256, dtype=torch.bfloat16))
        ranks.append(SimpleNamespace(v0=r * 64, v1=(r + 1) * 64, markov=markov, dev=torch.device("cpu")))
    return SimpleNamespace(ALL=(0, 1), devs=[torch.device("cpu")] * 2, ranks=ranks, emulated=False)


def test_pair_primes_both_before_any_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class Compiler:
        def warmup(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
            events.append("compile")
            return SimpleNamespace(run=lambda *a, **kw: None)

        def __getitem__(self, grid: Any) -> Any:
            def launch(*args: Any, **kwargs: Any) -> None:
                events.append("launch")
            return launch

    monkeypatch.setattr(sampled_fused, "_kernel", Compiler())
    monkeypatch.setattr(torch.cuda, "device", lambda dev: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    model = paired_fixture()
    ctx = sampled_fused.FusedContext(model, 1, 5)
    logits = {r: torch.zeros(1, 5, 64, dtype=torch.bfloat16) for r in (0, 1)}
    previous = {r: torch.zeros(1, dtype=torch.int64) for r in (0, 1)}
    output = {r: torch.zeros(1, 5, dtype=torch.int64) for r in (0, 1)}
    ctx.select((0, 1), logits, previous, output, 0)
    assert events == ["compile", "compile", "launch", "launch"]
    events.clear()
    ctx.select((0, 1), logits, previous, output, 0)
    assert events == ["launch", "launch"]
    assert not torch.count_nonzero(ctx.state[0]["ids"][:, 4]).item()
    assert not torch.count_nonzero(ctx.state[0]["q"][:, 4]).item()


def test_unprimed_pair_capture_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "device", lambda dev: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    ctx = sampled_fused.FusedContext(paired_fixture(), 1, 5)
    with pytest.raises(RuntimeError, match="primed together"):
        ctx.select((0,), {0: torch.zeros(1, 5, 64, dtype=torch.bfloat16)},
                   {0: torch.zeros(1, dtype=torch.int64)}, {0: torch.zeros(1, 5, dtype=torch.int64)}, 0)
