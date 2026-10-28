"""CPU model of Triton's compile-only warmup and lazy CUDA launch handles.

Proves host ordering, never the cause or repair of the physical timeout.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import torch

from offload.dspark.worker import sampled_fused as f
from offload.dspark.worker.tests.test_sampled_dispatch import paired_fixture


class LazyCompiler:
    def __init__(self, fail_rank: int | None) -> None:
        self.rank = 0
        self.fail_rank = fail_rank
        self.loaded: set[int] = set()
        self.events: list[tuple[str, int]] = []

    @contextmanager
    def device(self, dev: torch.device) -> Iterator[None]:
        assert dev.index is not None
        before, self.rank = self.rank, dev.index
        try:
            yield
        finally:
            self.rank = before

    def warmup(self, *args: Any, **kwargs: Any) -> Compiled:
        self.events.append(('compile', self.rank))
        return Compiled(self, self.rank)

    def __getitem__(self, grid: object) -> Callable[..., None]:
        def launch(*args: Any, **kwargs: Any) -> None:
            assert self.loaded == {0, 1}, 'peer launch before both CUDA modules loaded'
            self.events.append(('launch', self.rank))
        return launch


class Compiled:
    def __init__(self, compiler: LazyCompiler, rank: int) -> None:
        self.compiler, self.rank = compiler, rank

    @property
    def run(self) -> Callable[..., None]:
        assert self.compiler.rank == self.rank, 'loading CUDA module on wrong device'
        self.compiler.events.append(('load', self.rank))
        if self.rank == self.compiler.fail_rank:
            raise RuntimeError(f'CPU fixture rank{self.rank} module load failed')
        self.compiler.loaded.add(self.rank)

        def forbidden(*args: Any, **kwargs: Any) -> None:
            raise AssertionError('priming executed a peer kernel')
        return forbidden


@pytest.mark.parametrize('fail_rank', [None, 0, 1])
def test_actual_select_loads_both_devices_before_any_peer_launch(monkeypatch: pytest.MonkeyPatch,
                                                               fail_rank: int | None) -> None:
    compiler = LazyCompiler(fail_rank)
    model = paired_fixture()
    ctx = f.FusedContext(model, 1, 4)
    model.devs = [torch.device('cuda:0'), torch.device('cuda:1')]
    monkeypatch.setattr(torch.cuda, 'device', compiler.device)
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    monkeypatch.setattr(f, '_kernel', compiler)
    logits = {r: torch.zeros(1, 4, 64, dtype=torch.bfloat16) for r in (0, 1)}
    previous = {r: torch.zeros(1, dtype=torch.int64) for r in (0, 1)}
    outputs = {r: torch.zeros(1, 4, dtype=torch.int64) for r in (0, 1)}
    if fail_rank is not None:
        with pytest.raises(RuntimeError, match=f'rank{fail_rank} module load failed'):
            ctx.select((0, 1), logits, previous, outputs, 0)
        assert not any(event == 'launch' for event, _ in compiler.events)
        assert (fail_rank, 0, 256, 1) not in ctx.primed
    else:
        ctx.select((0, 1), logits, previous, outputs, 0)
        assert compiler.events == [('compile', 0), ('load', 0), ('compile', 1), ('load', 1),
                                   ('launch', 0), ('launch', 1)]
        compiler.events.clear()
        ctx.select((0, 1), logits, previous, outputs, 0)
        assert compiler.events == [('launch', 0), ('launch', 1)]
