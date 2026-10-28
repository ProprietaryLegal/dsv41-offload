"""CPU stream model: two distinct CUDA devices, never solo-emulation evidence."""
from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from offload.dspark.probabilistic.pp4 import pair_probe
from offload.dspark.worker import sampled_fused, tp2
from offload.dspark.worker.tests.test_sampled_dispatch import paired_fixture


class Streams:
    def __init__(self) -> None:
        self.device = torch.device('cuda:0')
        self.captured: set[torch.device] = set()
        self.launches: list[tuple[torch.device, bool]] = []
        self.streams: list[torch.device] = []

    @contextmanager
    def on_device(self, dev: torch.device) -> Iterator[None]:
        before, self.device = self.device, dev
        try:
            yield
        finally:
            self.device = before

    def stream(self, *, device: torch.device) -> torch.device:
        self.streams.append(device)
        return device

    @contextmanager
    def graph(self, graph: object, *, stream: torch.device | None = None) -> Iterator[None]:
        # Reproduce PyTorch's shared default: it was first created on rank0.
        captured = torch.device('cuda:0') if stream is None else stream
        self.captured.add(captured)
        try:
            yield
        finally:
            self.captured.remove(captured)

    def __getitem__(self, grid: object) -> Any:
        def launch(*args: Any, **kwargs: Any) -> None:
            self.launches.append((self.device, self.device in self.captured))
        return launch


def context(monkeypatch: pytest.MonkeyPatch) -> tuple[Streams, sampled_fused.FusedContext, dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    streams = Streams()
    ctx = sampled_fused.FusedContext(paired_fixture(), 1, 4)
    ctx.model.devs = [torch.device('cuda:0'), torch.device('cuda:1')]
    logits = {r: torch.zeros(1, 4, 64, dtype=torch.bfloat16) for r in (0, 1)}
    previous = {r: torch.zeros(1, dtype=torch.int64) for r in (0, 1)}
    outputs = {r: torch.zeros(1, 4, dtype=torch.int64) for r in (0, 1)}
    ctx.primed = {(r, pos, 256, 1) for r in (0, 1) for pos in range(4)}
    # Previous-token column views after position0 have stride4.
    ctx.primed.update((r, pos, 256, 4) for r in (0, 1) for pos in range(4))
    monkeypatch.setattr(torch.cuda, 'device', streams.on_device)
    monkeypatch.setattr(torch.cuda, 'Stream', streams.stream)
    monkeypatch.setattr(torch.cuda, 'CUDAGraph', object)
    monkeypatch.setattr(torch.cuda, 'graph', streams.graph)
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: streams.device in streams.captured)
    monkeypatch.setattr(sampled_fused, '_kernel', streams)
    return streams, ctx, logits, previous, outputs


def test_actual_pair_capture_uses_distinct_device_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    streams, ctx, logits, previous, outputs = context(monkeypatch)
    graphs = pair_probe.capture_pair(ctx, logits, previous, outputs, 4)
    assert set(graphs) == {0, 1}
    assert streams.streams == [torch.device('cuda:0'), torch.device('cuda:1')]
    assert streams.launches == [(torch.device(f'cuda:{r}'), True) for r in (0, 1) for _ in range(4)]
    assert not ctx.capture_expected


def test_shared_default_stream_refuses_before_eager_peer_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    streams, ctx, logits, previous, outputs = context(monkeypatch)
    with pytest.raises(RuntimeError, match='rank 1: expected capture'), ctx.capturing((1,)), streams.graph(object()):
        ctx.select((1,), logits, previous, outputs, 0)
    assert streams.launches == []
    assert not ctx.capture_expected
    # A failed capture cannot leave intent armed during ordinary eager work.
    ctx.select((0, 1), logits, previous, outputs, 0)
    assert streams.launches == [(torch.device('cuda:0'), False), (torch.device('cuda:1'), False)]


def test_capture_rank_ownership_cannot_change(monkeypatch: pytest.MonkeyPatch) -> None:
    streams, ctx, logits, previous, outputs = context(monkeypatch)
    with ctx.capturing((0,)), pytest.raises(RuntimeError, match='rank ownership'):
        ctx.select((1,), logits, previous, outputs, 0)
    assert not streams.launches


@pytest.mark.parametrize('wrong_rank1_stream', [False, True])
def test_real_worker_capture_path(monkeypatch: pytest.MonkeyPatch, wrong_rank1_stream: bool) -> None:
    streams, ctx, logits, previous, outputs = context(monkeypatch)

    class Stream:
        def __init__(self, device: torch.device) -> None:
            self.device = device

        def wait_stream(self, other: Stream) -> None:
            pass

    def graph(g: object, *, pool: object, stream: Stream) -> Any:
        return streams.graph(g, stream=stream.device)

    monkeypatch.setattr(torch.cuda, 'Stream', Stream)
    monkeypatch.setattr(torch.cuda, 'stream', lambda st: streams.on_device(st.device))
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda dev: Stream(dev))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda dev: None)
    monkeypatch.setattr(torch.cuda, 'graph', graph)
    ranks = [SimpleNamespace(r=r, dev=dev, pool=None,
             cap_stream=Stream(ctx.model.devs[0] if wrong_rank1_stream else dev)) for r, dev in enumerate(ctx.model.devs)]
    model = SimpleNamespace(ALL=(0, 1), devs=ctx.model.devs, emulated=False, ranks=ranks, _pp4_context=ctx)

    def fn(active: tuple[int, ...]) -> dict[str, object]:
        ctx.select(active, logits, previous, outputs, 0)
        return {}

    if wrong_rank1_stream:
        with pytest.raises(RuntimeError, match='rank 1: expected capture'):
            tp2.TPDrafter._capture_one(cast(tp2.TPDrafter, model), fn)
        assert not ctx.capture_expected
        assert streams.launches[-1] == (torch.device('cuda:0'), True)
    else:
        result = tp2.TPDrafter._capture_one(cast(tp2.TPDrafter, model), fn)
        assert set(result['graphs']) == {0, 1}
        assert streams.launches[-2:] == [(torch.device('cuda:0'), True), (torch.device('cuda:1'), True)]
    assert streams.launches[:4] == [(torch.device(f'cuda:{r}'), False) for _ in range(2) for r in (0, 1)]


def test_all_paired_capture_sites_have_explicit_stream_and_intent() -> None:
    for fn in (pair_probe.capture_pair, tp2.TPDrafter._capture_one):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        graphs = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'graph']
        assert graphs
        assert all(any(k.arg == 'stream' and not isinstance(k.value, ast.Constant) for k in n.keywords) for n in graphs)
        assert any(isinstance(n.func, ast.Attribute) and n.func.attr == 'capturing' for n in calls)
    source = Path(inspect.getfile(tp2)).read_text()
    assert 'self.cap_stream = torch.cuda.Stream(dev)' in source
