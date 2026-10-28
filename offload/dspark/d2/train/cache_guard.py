"""Reject stale feature/Markov state after any adapted dependency changes.

This is an evaluation guard, checked outside graph capture/replay. Adapters are
immutable for a worker's lifetime; a changed dependency requires a fresh worker.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping

from torch import Tensor


class DependencyGuard:
    def __init__(self, dependencies: Mapping[str, Tensor], invalidate: Callable[[], None],
                 current: Callable[[], Mapping[str, Tensor]] | None = None):
        if not dependencies:
            raise ValueError("cache guard requires declared adapted dependencies")
        self.dependencies = dict(dependencies)
        self.current = current
        self.versions = {name: (t.data_ptr(), t._version) for name, t in dependencies.items()}
        self.invalidate = invalidate
        self.invalidated = False

    def check(self) -> None:
        actual = self.dependencies if self.current is None else self.current()
        if self.invalidated or actual.keys() != self.versions.keys() or any((t.data_ptr(), t._version) != self.versions[name]
                                   for name, t in actual.items()):
            if not self.invalidated:
                self.invalidate()
            self.invalidated = True
            raise RuntimeError("adapted dependency changed: Markov/feature caches and graphs invalidated; recreate worker")
