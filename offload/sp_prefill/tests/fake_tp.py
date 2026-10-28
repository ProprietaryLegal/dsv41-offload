"""A fake TP group for CPU tests: one thread per rank, collectives meet at a barrier.

Reductions add the ranks' tensors in fp32 in rank order and round once to the input dtype, which is what NCCL's
two-rank sum computes for bf16 (one addition; a + b is commutative, so all-reduce and reduce-scatter agree bit for bit).
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TypeVar

import torch

T = TypeVar('T')


class FakeTP:
    def __init__(self, world_size: int, timeout_s: float = 60.0) -> None:
        self.world_size = world_size
        self._barrier = threading.Barrier(world_size, timeout=timeout_s)
        self._slots: list[torch.Tensor | None] = [None] * world_size
        self._local = threading.local()
        self.calls: dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def rank(self) -> int:
        rank: int = self._local.rank
        return rank

    def _count(self, name: str) -> None:
        if self.rank == 0:
            with self._lock:
                self.calls[name] = self.calls.get(name, 0) + 1

    def _exchange(self, x: torch.Tensor) -> list[torch.Tensor]:
        self._slots[self.rank] = x
        self._barrier.wait()
        tensors = [t for t in self._slots if t is not None]
        assert len(tensors) == self.world_size
        shapes = {tuple(t.shape) for t in tensors}
        assert len(shapes) == 1, f'ranks disagree on the collective shape: {shapes}'
        self._barrier.wait()
        return tensors

    @staticmethod
    def _sum(tensors: list[torch.Tensor]) -> torch.Tensor:
        acc = tensors[0].float()
        for t in tensors[1:]:
            acc = acc + t.float()
        return acc.to(tensors[0].dtype)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self._count('all_reduce')
        return self._sum(self._exchange(x))

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        self._count('all_gather')
        return torch.cat(self._exchange(x), dim=0)

    def reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        self._count('reduce_scatter')
        assert x.shape[0] % self.world_size == 0
        chunk = x.shape[0] // self.world_size
        total = self._sum(self._exchange(x))
        return total[self.rank * chunk : (self.rank + 1) * chunk].clone()

    def run(self, fn: Callable[[int], T]) -> list[T]:
        """fn(rank) on every rank concurrently; re-raises the first failure."""
        results: list[T | None] = [None] * self.world_size
        errors: list[BaseException] = []

        def target(rank: int) -> None:
            self._local.rank = rank
            try:
                results[rank] = fn(rank)
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                errors.append(exc)
                self._barrier.abort()

        threads = [threading.Thread(target=target, args=(r,)) for r in range(self.world_size)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        return [r for r in results if r is not None]
