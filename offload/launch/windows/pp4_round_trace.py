"""Experiment-only complete model-runner round timing; disabled without a numeric marker.

Installed as a read-only engine helper. No torch import/CUDA work while disabled. Profiled cells deliberately
synchronize completed rounds; their request rates NEVER enter the unprofiled headline throughput result.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Concatenate, ParamSpec, Protocol, TypeVar, cast

P = ParamSpec('P')
R = TypeVar('R')


class Event(Protocol):
    def record(self) -> None: ...
    def synchronize(self) -> None: ...
    def elapsed_time(self, other: Event) -> float: ...


class Scheduler(Protocol):
    @property
    def num_scheduled_tokens(self) -> Mapping[str, int]: ...

    @property
    def scheduled_spec_decode_tokens(self) -> Mapping[str, list[int]]: ...


@dataclass
class Span:
    marker: str
    root: Path
    rank: int
    batch: int
    sequence: int
    request_hash: str
    start: Event
    wall: float


_active: dict[int, Span] = {}
_sequences: dict[tuple[int, str], int] = {}


def begin(runner: object, scheduler: object, dummy: bool) -> None:
    if dummy or not os.environ.get('VLLM_PP4_ROUND_DIR'):
        return
    root = Path(os.environ['VLLM_PP4_ROUND_DIR'])
    marker_path = root / 'ACTIVE'
    if not marker_path.exists():
        return
    marker = marker_path.read_text().strip()
    if re.fullmatch('[A-Za-z0-9:_-]+', marker) is None:
        raise ValueError('PP4 invalid profile cell marker')
    scheduled_state = cast(Scheduler, scheduler)
    scheduled = scheduled_state.num_scheduled_tokens
    drafts = scheduled_state.scheduled_spec_decode_tokens
    # Complete pure decode rounds only: no prefill or mixed batch can contaminate the p95.
    if not scheduled or any(not drafts.get(k) or n != len(drafts[k]) + 1 for k, n in scheduled.items()):
        return
    key = id(runner)
    if key in _active:
        raise ValueError('PP4 prior execute_model round has no sample_tokens completion')
    seq_key = key, marker
    sequence = _sequences.get(seq_key, 0)
    _sequences[seq_key] = sequence + 1
    torch = importlib.import_module('torch')
    rank = cast(Callable[[], int], torch.distributed.get_rank)()
    if rank not in (0, 1):
        raise ValueError('PP4 complete-round profile requires the reviewed two-rank target')
    event_factory = cast(Callable[..., Event], torch.cuda.Event)
    start = event_factory(enable_timing=True)
    start.record()
    _active[key] = Span(marker, root, rank, len(scheduled), sequence,
        hashlib.sha256('\n'.join(sorted(scheduled)).encode()).hexdigest(), start, time.perf_counter())


def finish(runner: object, failed: bool = False) -> None:
    span = _active.pop(id(runner), None)
    if span is None:
        return
    torch = importlib.import_module('torch')
    factory = cast(Callable[..., Event], torch.cuda.Event)
    end = factory(enable_timing=True)
    end.record()
    end.synchronize()  # PROFILED cells only; complete all target/verification/proposal stream work.
    wall_ms = 1000 * (time.perf_counter() - span.wall)
    cuda_ms = span.start.elapsed_time(end)
    row = {'schema': 'pp4-complete-round-v1', 'cell': span.marker, 'rank': span.rank,
           'round': span.sequence, 'batch': span.batch, 'request_hash': span.request_hash,
           'complete_wall_ms': wall_ms, 'cuda_elapsed_ms': cuda_ms, 'failed': failed,
           'scope': 'execute_model entry through sample_tokens completion plus end-event synchronization'}
    with (span.root / f'rank{span.rank}.jsonl').open('a') as out:
        out.write(json.dumps(row, allow_nan=False) + '\n')


def execution(fn: Callable[Concatenate[object, P], R]) -> Callable[Concatenate[object, P], R]:
    @wraps(fn)
    def call(runner: object, /, *args: P.args, **kwargs: P.kwargs) -> R:
        scheduler = args[0] if args else kwargs.get('scheduler_output')
        dummy = bool(kwargs.get('dummy_run', False)) or (len(args) > 2 and bool(args[2]))
        begin(runner, scheduler, dummy)
        try:
            result = fn(runner, *args, **kwargs)
        except BaseException:
            finish(runner, True)
            raise
        if getattr(runner, 'execute_model_state', None) is None:
            finish(runner, True)
        return result
    return call


def sampling(fn: Callable[Concatenate[object, P], R]) -> Callable[Concatenate[object, P], R]:
    @wraps(fn)
    def call(runner: object, /, *args: P.args, **kwargs: P.kwargs) -> R:
        try:
            result = fn(runner, *args, **kwargs)
        except BaseException:
            finish(runner, True)
            raise
        finish(runner)
        return result
    return call
