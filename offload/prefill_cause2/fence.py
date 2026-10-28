"""PF1-CAUSE2 fix candidate (mounted as vllm/v1/worker/gpu/dsv41_moe_fence.py): CUDA event records around the MoE half's calls.

VLLM_DSV41_MOE_FENCE=1 (default 0: every call goes straight through). With it, each wrapped call of an eager pass of
at least MIN_ROWS rows records one timing-enabled CUDA event on the current stream before the call and one after it
(also on exception), exactly where, and of the kind, the PF1-CAUSE spans record theirs (window CW: no 524,288 gap).
Nothing reads the events: no query, no elapsed time, no synchronization, no file, no device read, no allocation (CUDA
events take no caching-allocator memory). Arguments and return values pass through unchanged; the pairs are kept
referenced in a bounded ring so an event is released long after it completed, as the span runtime releases its own at
the step boundary. Outside eager passes (CUDA-graph capture) and below MIN_ROWS (decode) nothing is recorded.
"""
from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable
from typing import ParamSpec, Protocol, TypeVar, cast

P = ParamSpec('P')
T = TypeVar('T')
ENABLED = os.environ.get('VLLM_DSV41_MOE_FENCE', '0') == '1'
# The TensorFold prompt path's own threshold (tf_experts.EXACT_ROWS): prompt chunks and the 128-row replay tail are
# fenced, decode windows (CUDA-graph replays, or eager verify rows) never are.
MIN_ROWS = 64
KEEP = 4096                        # pairs referenced at once: ~8.5 full chunks at 480 pairs per chunk


class Event(Protocol):
    def record(self) -> None: ...


class Backend(Protocol):
    def event(self) -> Event: ...
    def capturing(self) -> bool: ...


class CudaBackend:
    def event(self) -> Event:
        import torch
        factory = cast(Callable[..., Event], torch.cuda.Event)
        return factory(enable_timing=True)

    def capturing(self) -> bool:
        import torch
        return torch.cuda.is_current_stream_capturing()


backend: Backend = CudaBackend()
_kept: deque[tuple[Event, Event]] = deque(maxlen=KEEP)


def call(rows: int, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
    if not ENABLED or rows < MIN_ROWS or backend.capturing():
        return fn(*args, **kwargs)
    start, end = backend.event(), backend.event()
    start.record()
    try:
        return fn(*args, **kwargs)
    finally:
        end.record()
        _kept.append((start, end))
