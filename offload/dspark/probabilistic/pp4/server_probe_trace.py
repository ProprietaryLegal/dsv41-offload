"""Diagnostic host wrappers only. Never add synchronization between peer launches."""
from __future__ import annotations

import functools
import importlib
import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

OUT = Path(os.environ.get('PP4_SERVER_OUT', '.'))
STAGE = 'startup'
EVENTS: list[dict[str, Any]] = []
LOCK = threading.Lock()
EPOCHS: dict[int, list[int]] = {}
DEFER_EVENTS = False
PENDING: list[dict[str, Any]] = []


def emit(event: str, **fields: object) -> None:
    row = {'event': event, 'stage': STAGE, 'host_ns': time.perf_counter_ns(),
           'pid': os.getpid(), 'python_thread': threading.get_ident(), 'native_thread': threading.get_native_id(), **fields}
    EVENTS.append(row)
    if DEFER_EVENTS:
        # No lock, file operation or print while a paired launch can be in
        # flight. The serialized worker owns this interval; append is under GIL.
        PENDING.append(row)
        return
    with LOCK:
        rows = [*PENDING, row]
        PENDING.clear()
        with (OUT/'phases.jsonl').open('a') as handle:
            for item in rows:
                handle.write(json.dumps(item, allow_nan=False)+'\n')
        print('\n'.join('PP4 SERVER '+json.dumps(item, allow_nan=False) for item in rows), flush=True)


@contextmanager
def phase(name: str) -> Iterator[None]:
    global DEFER_EVENTS
    emit('BEGIN', phase=name)
    start = time.perf_counter_ns()
    try:
        yield
    except BaseException as exc:
        if name in ('model._replay', 'selector.select'):
            DEFER_EVENTS = False
        emit('FAIL', phase=name, error=type(exc).__name__, elapsed_ns=time.perf_counter_ns()-start)
        raise
    else:
        if name in ('model._replay', 'selector.select'):
            DEFER_EVENTS = False
        emit('END', phase=name, elapsed_ns=time.perf_counter_ns()-start)


def patch(target: Any, name: str, label: str | None = None) -> None:
    original = getattr(target, name)
    @functools.wraps(original)
    def call(*args: Any, **kwargs: Any) -> Any:
        with phase(label or name):
            return original(*args, **kwargs)
    setattr(target, name, call)


def snapshot(ctx: Any) -> None:
    """Called ONLY after original result(), hence both submissions and drain.

    Do not infer device overlap from these host records. Exact advancing peer
    state proves completed exchange, not a CUDA activity trace.
    """
    import torch
    states = {r: {k: ctx.state[r][k].detach().cpu() for k in ('sticky', 'error', 'done', 'epoch', 'ids', 'q')}
              for r in (0, 1)}
    healthy = all(not bool(states[r][k].any().item()) for r in (0, 1) for k in ('sticky', 'error', 'done'))
    exact = all(torch.equal(states[0][k], states[1][k]) for k in ('epoch', 'ids', 'q'))
    outputs = getattr(ctx, '_pp4_probe_outputs', {})
    proposed_exact = (set(outputs) == {0, 1} and torch.equal(outputs[0].detach().cpu(), outputs[1].detach().cpu()))
    epochs = [int(x) for x in states[0]['epoch'].tolist()]
    previous = EPOCHS.get(id(ctx))
    advancing = previous is None or all(a > b for a, b in zip(epochs, previous, strict=True))
    # Keep the context alive through the model graph dictionary. Eager contexts
    # can reuse object IDs, so monotonicity is enforced only for graph records.
    if ctx in [row['context'] for row in getattr(ctx.model, 'sampled_graphs', {}).values()]:
        EPOCHS[id(ctx)] = epochs
    else:
        advancing = True
    emit('peer-state-after-drain', healthy=healthy, both_rank_exact=exact, proposed_ids_exact=proposed_exact, advancing=advancing,
         epochs=epochs, batch=ctx.batch, depth=ctx.depth)
    if not (healthy and exact and proposed_exact and advancing):
        raise RuntimeError('peer sticky/error/done/epoch or exact q/ID failure after drain')


def install(namespace: dict[str, Any]) -> None:
    """Called at entry to REAL worker.server.main, after definitions, before build."""
    import torch
    sampled_fused: Any = importlib.import_module('worker.sampled_fused')
    tp2 = importlib.import_module('worker.tp2')
    for name in ('capture', '_capture_one', 'draft', '_draft', 'heads', 'draft_sampled', '_replay', 'insert_eager',
                 'insert_taps_graphed', 'insert_mx_eager'):
        patch(tp2.TPDrafter, name, 'model.'+name)
    for name in ('reset', 'stage', 'flush', 'discard', 'draft_sampled'):
        patch(tp2.TP2Backend, name, 'backend.'+name)
    patch(namespace['DraftCore'], 'handle', 'DraftCore.handle')
    patch(namespace['DraftCore'], '_step', 'DraftCore._step')
    patch(namespace['P'], 'encode_step_reply', 'reply.serialization')
    patch(namespace['P'], 'reply', 'frame.serialization')
    original_select = sampled_fused.FusedContext.select
    def select(ctx: Any, ranks: Any, logits: Any, previous: Any, outputs: Any, position: int) -> None:
        # Retain existing tensor references only; inspect AFTER both submissions
        # and result drain, including the physical rank1 proposal IDs.
        if not hasattr(ctx, '_pp4_probe_outputs'):
            ctx._pp4_probe_outputs = {}
        ctx._pp4_probe_outputs.update(outputs)
        with phase('selector.select'):
            original_select(ctx, ranks, logits, previous, outputs, position)
    sampled_fused.FusedContext.select = select
    original_result = sampled_fused.FusedContext.result
    def result(ctx: Any) -> Any:
        with phase('selector.result'):
            value = original_result(ctx)
        snapshot(ctx)
        return value
    sampled_fused.FusedContext.result = result

    original_replay = torch.cuda.CUDAGraph.replay
    def replay(graph: Any) -> None:
        global DEFER_EVENTS
        DEFER_EVENTS = True
        # Only host device/stream queries; no event, item, D2H or synchronize.
        with phase('graph.rank-submission'):
            emit('graph-stream', device=torch.cuda.current_device(),
                 stream=int(torch.cuda.current_stream().cuda_stream))
            original_replay(graph)
    graph_type: Any = torch.cuda.CUDAGraph  # diagnostic monkeypatch of the untyped CUDA extension boundary
    graph_type.replay = replay

    # This interception sees actual lazy handle initialization anywhere in the
    # model; no implementation or kernel source is copied into the probe.
    compiler = importlib.import_module('triton.compiler.compiler')
    patch(compiler.CompiledKernel, '_init_handles', 'triton.handle-load')
    jit = importlib.import_module('triton.runtime.jit')
    patch(jit.JITFunction, 'warmup', 'triton.warmup')
    original_kernel = sampled_fused._kernel
    class Compiled:
        def __init__(self, value: Any):
            self.value = value
        @property
        def run(self) -> Callable[..., Any]:
            with phase('selector.handle-resolution'):
                value: Callable[..., Any] = self.value.run
                return value
    class Kernel:
        def warmup(self, *args: Any, **kwargs: Any) -> Compiled:
            with phase('selector.warmup'):
                return Compiled(original_kernel.warmup(*args, **kwargs))
        def __getitem__(self, grid: Any) -> Callable[..., Any]:
            def launch(*args: Any, **kwargs: Any) -> Any:
                global DEFER_EVENTS
                DEFER_EVENTS = True
                with phase('selector.rank-submission'):
                    emit('selector-stream', device=torch.cuda.current_device(),
                         stream=int(torch.cuda.current_stream().cuda_stream), position=int(args[43]))
                    return original_kernel[grid](*args, **kwargs)
            return launch
    sampled_fused._kernel = Kernel()
    ctypes_module = namespace['ctypes']
    class CtypesProxy:
        def __getattr__(self, name: str) -> Any:
            if name != 'memmove':
                return getattr(ctypes_module, name)
            def memmove(*args: Any) -> Any:
                with phase('native.reply-copy'):
                    return ctypes_module.memmove(*args)
            return memmove
    namespace['ctypes'] = CtypesProxy()
