"""Pure scheduling decision; no torch, network, resource allocation or model calls."""
from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Options:
    enabled: bool = False
    period: int = 24
    quantum: int = 32

    def __post_init__(self) -> None:
        if not 2 <= self.period <= 1024:
            raise ValueError('SO2 period must be in 2..1024')
        if not 1 <= self.quantum <= 1024:
            raise ValueError('SO2 quantum must be in 1..1024')

    @classmethod
    def from_env(cls) -> Options:
        flag = os.environ.get('VLLM_SO2_BG_PREFILL', '0')
        if flag not in ('0', '1'):
            raise ValueError('VLLM_SO2_BG_PREFILL must be 0 or 1')
        return cls(flag == '1', int(os.environ.get('VLLM_SO2_PERIOD', '24')),
                   int(os.environ.get('VLLM_SO2_QUANTUM', '32')))


@dataclass(frozen=True)
class Work:
    request_id: str
    priority: int
    demand: int
    prefill: bool
    ready: bool


@dataclass(frozen=True)
class Decision:
    mode: str
    due: bool
    quantum: int


def decide(options: Options, step: int, running: list[Work], *,
           waiting_foreground: bool, has_waiters: bool, dependency: bool,
           input_budget: int, token_budget: int, draft_slots: int,
           max_sequences: int) -> Decision:
    """Only defer in an explicitly bounded, resource-ready decode envelope.

    Allocation can still fail: native preemption/readiness always decides whether
    a request is runnable. On non-service rounds actual foreground progress is
    required before any background request is skipped.
    """
    due = step % options.period == 0
    fallback = Decision('native', due, options.quantum)
    if (not options.enabled or dependency or input_budget != 4096 or token_budget != 4096
            or max_sequences != 4 or not 0 <= draft_slots <= 4 or len(running) > 4):
        return fallback
    foreground = [r for r in running if r.priority < 20]
    background = [r for r in running if r.priority >= 20]
    if (waiting_foreground and not foreground and 0 < len(background) <= 3
            and all(r.prefill and r.ready for r in background)):
        return Decision('admission', due, options.quantum)
    if (has_waiters or not foreground or not background
            or not all(not r.prefill and r.ready and 0 < r.demand <= 16 for r in foreground)
            or not all(r.ready and r.demand > 0 and (r.prefill or r.demand <= 16)
                       for r in background)):
        return fallback
    # Even custom quanta must leave room for ALL pending foreground rows.
    need = (sum(r.demand for r in foreground)
            + sum(min(r.demand, options.quantum) if r.prefill else r.demand for r in background)
            + len(running) * draft_slots)
    return Decision('mixed', due, options.quantum) if need <= 4096 else fallback
