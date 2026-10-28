"""Window-only numeric telemetry. No tensor contents, device reads or synchronization."""
from __future__ import annotations

import functools
import hashlib
import json
import os
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sized
from contextlib import contextmanager
from typing import ContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import ParamSpec, Protocol, TypeVar, cast

Scalar = str | int | float | bool | None
P = ParamSpec('P')
T = TypeVar('T')
ENABLED = os.environ.get('VLLM_DSV41_PREFILL_TRACE', '0') == '1'


class Event(Protocol):
    def record(self) -> None: ...
    def query(self) -> bool: ...
    def elapsed_time(self, end: Event) -> float: ...


class Backend(Protocol):
    def event(self) -> Event: ...
    def rank(self) -> int: ...
    def capturing(self) -> bool: ...


class CudaBackend:
    def event(self) -> Event:
        import torch
        factory = cast(Callable[..., Event], torch.cuda.Event)
        return factory(enable_timing=True)

    def rank(self) -> int:
        import torch.distributed as dist
        return dist.get_rank()

    def capturing(self) -> bool:
        import torch
        return torch.cuda.is_current_stream_capturing()


@dataclass
class Pending:
    row: dict[str, object]
    start: Event
    end: Event


class Recorder:
    def __init__(self, path: Path, backend: Backend, limit: int = 8192) -> None:
        if limit < 1:
            raise ValueError('trace pending limit must be positive')
        self.backend = backend
        self.limit = limit
        self.pending: deque[Pending] = deque()
        self.path = path
        self.context: dict[str, object] | None = None
        self.sequence = 0
        self.whole: ContextManager[None] | None = None
        self.occurrences: dict[str, int] = {}
        # Exclusive file creation: never append into a previous window's evidence.
        self.stream = path.open('x', buffering=1)

    def write(self, row: dict[str, object]) -> None:
        self.stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')

    def drain(self) -> None:
        # query() is nonblocking; never synchronize, even at final batch.
        remaining: deque[Pending] = deque()
        while self.pending:
            item = self.pending.popleft()
            if item.end.query():
                item.row['cuda_ms'] = item.start.elapsed_time(item.end)
                self.write(item.row)
            else:
                remaining.append(item)
        self.pending = remaining

    def call(self, label: str, metadata: dict[str, Scalar], fn: Callable[P, T],
             *args: P.args, **kwargs: P.kwargs) -> T:
        if self.context is None:
            return fn(*args, **kwargs)
        if self.backend.capturing():
            raise RuntimeError('PF1 trace requires eager prefill; graph capture is unsupported')
        if len(self.pending) >= self.limit:
            self.drain()
        if len(self.pending) >= self.limit:
            raise RuntimeError('PF1 trace pending event bound exceeded; stop window, do not drop samples')
        start, end = self.backend.event(), self.backend.event()
        row: dict[str, object] = {**self.context, **metadata, 'phase': label, 'event': self.sequence,
                                  'host_start_ns': time.monotonic_ns()}
        self.sequence += 1
        start.record()
        begin = time.perf_counter_ns()
        try:
            return fn(*args, **kwargs)
        except BaseException:
            row['failed'] = True
            raise
        finally:
            row['wall_ms'] = (time.perf_counter_ns() - begin) / 1e6
            end.record()
            self.pending.append(Pending(row, start, end))


_recorder: Recorder | None = None


def recorder() -> Recorder:
    global _recorder
    if _recorder is None:
        root = Path(os.environ['VLLM_DSV41_PREFILL_TRACE_DIR'])
        if not root.is_dir():
            raise RuntimeError(f'PF1 trace directory must already exist: {root}')
        backend = CudaBackend()
        _recorder = Recorder(root / f'prefill-r{backend.rank()}-p{os.getpid()}.jsonl', backend)
    return _recorder


def attr(value: object, name: str) -> object:
    return getattr(value, name, None)


def meta(scope: Mapping[str, object]) -> dict[str, Scalar]:
    """Read only scalar Python attributes and tensor SHAPES, never GPU values."""
    out: dict[str, Scalar] = {}
    for key in ('row_start', 'row_end', 'topk_tokens', 'num_rows', 'num_tokens',
                'num_prefills', 'num_prefill_tokens', 'num_chunks', 'chunk_idx',
                'chunk_start', 'chunk_end', 'total_seq_lens', 'head_dim', 'use_fp4_cache',
                'candidate_write', 'candidate_block_size', 'topk_backend', 'layer_name'):
        value = scope.get(key)
        if isinstance(value, (int, float, bool, str)):
            out[key] = value
    owner = scope.get('self')
    for key in ('compress_ratio', 'max_prefill_buffer_size', 'unique_name', 'prefix'):
        value = attr(owner, key)
        if isinstance(value, (int, float, bool, str)):
            out[key] = value
    chunk = scope.get('chunk')
    for key in ('token_start', 'token_end', 'total_seq_lens', 'local_total_seq_lens',
                'max_local_total_seq_lens', 'num_reqs', 'skip_kv_gather'):
        value = attr(chunk, key)
        if isinstance(value, (int, bool)):
            out['indexer_chunk_' + key] = value
    for key in ('logits', 'k_quant', 'k_scale', 'q_chunk', 'q_slice', 'swa_indices_chunk',
                'extra_sparse_indices_chunk', 'topk_indices', 'block_table',
                'indexer_block_table', 'input_'):
        value = scope.get(key)
        shape = attr(value, 'shape')
        if shape is not None:
            # torch.Size is host-only, including for device tensors.
            dims = tuple(cast(tuple[int, ...], shape))
            out[key + '_shape'] = 'x'.join(str(int(dim)) for dim in dims)
            size = 1
            for dim in dims:
                size *= int(dim)
            out[key + '_elements'] = size
            width = cast(Callable[[], int], getattr(value, 'element_size'))()
            out[key + '_bytes'] = size * width
    cm = scope.get('common_attn_metadata')
    table = attr(cm, 'block_table_tensor')
    shape = attr(table, 'shape')
    if shape is not None:
        dims = tuple(cast(tuple[int, ...], shape))
        out['metadata_block_table_rows'] = int(dims[0])
        out['metadata_block_table_columns'] = int(dims[1])
        out['metadata_block_table_elements_bound'] = int(dims[0]) * int(dims[1])
    return out


class HostArray(Protocol):
    def __getitem__(self, key: int) -> object: ...


class Batch(Protocol):
    req_ids: list[str]
    num_reqs: int
    has_prefill: bool
    is_prefilling_np: HostArray
    query_start_loc_np: HostArray
    num_computed_prefill_tokens_np: HostArray


def begin(batch: Batch, dummy: bool, mode: object, cap: int = 0) -> None:
    if not ENABLED:
        return
    rec = recorder()
    rec.drain()
    rec.context = None
    if dummy or not batch.has_prefill:
        return
    if str(mode).split('.')[-1] != 'NONE':
        raise RuntimeError('PF1 trace requires CUDAGraphMode.NONE for prefill')
    members: list[dict[str, object]] = []
    salt = os.environ['VLLM_DSV41_PREFILL_TRACE_RUN']
    for i in range(batch.num_reqs):
        # *_np arrays are host-known; no .item() on a CUDA tensor.
        if not bool(batch.is_prefilling_np[i]):
            continue
        key = hashlib.sha256((salt + ':' + batch.req_ids[i]).encode()).hexdigest()[:24]
        prefix = int(cast(int, batch.num_computed_prefill_tokens_np[i]))
        q = int(cast(int, batch.query_start_loc_np[i + 1])) - int(cast(int, batch.query_start_loc_np[i]))
        identity = f'{key}:{prefix}:{q}'
        occurrence = rec.occurrences.get(identity, 0)
        rec.occurrences[identity] = occurrence + 1
        if len(rec.occurrences) > 32768:
            raise RuntimeError('PF1 trace chunk identity bound exceeded; rotate window')
        members.append({'request': key, 'chunk': identity + f':{occurrence}',
                        'new_tokens': q, 'prefix_tokens': prefix,
                        'batch_row_start': int(cast(int, batch.query_start_loc_np[i])),
                        'batch_row_end': int(cast(int, batch.query_start_loc_np[i + 1]))})
    rec.context = {'rank': rec.backend.rank(), 'run': salt, 'max_model_len': cap, 'members': members}
    rec.write({**rec.context, 'phase': 'chunk.begin', 'host_start_ns': time.monotonic_ns()})
    rec.whole = span('chunk.total', {})
    rec.whole.__enter__()


def call(label: str, metadata: dict[str, Scalar], fn: Callable[P, T],
         *args: P.args, **kwargs: P.kwargs) -> T:
    if not ENABLED or _recorder is None:
        return fn(*args, **kwargs)
    return _recorder.call(label, metadata, fn, *args, **kwargs)


def execution(fn: Callable[P, T]) -> Callable[P, T]:
    @functools.wraps(fn)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return fn(*args, **kwargs)
        finally:
            if ENABLED and _recorder is not None:
                if _recorder.whole is not None:
                    _recorder.whole.__exit__(*sys.exc_info())
                    _recorder.whole = None
                if _recorder.context is not None:
                    _recorder.write({**_recorder.context, 'phase': 'chunk.end',
                                     'host_end_ns': time.monotonic_ns(),
                                     'pending_events': len(_recorder.pending)})
                _recorder.context = None
                _recorder.drain()
    return wrapped


def plan(chunks: list[tuple[slice, slice]], seq: HostArray, query: HostArray,
         workspace: int, logits_bytes: int, offset: int) -> list[tuple[slice, slice]]:
    if ENABLED and _recorder is not None and _recorder.context is not None:
        rec = _recorder
        assert rec.context is not None
        requested_rows = sum(int(cast(int, seq[j])) for j in range(len(cast(Sized, seq))))
        for i, (req, rows) in enumerate(chunks):
            start, stop = cast(int, req.start) - offset, cast(int, req.stop) - offset
            key_rows = sum(int(cast(int, seq[j])) for j in range(start, stop))
            new_rows = sum(int(cast(int, query[j])) for j in range(start, stop))
            rec.write({**rec.context, 'phase': 'indexer.plan', 'plan_chunk': i,
                       'req_start': start + offset, 'req_stop': stop + offset,
                       'query_start': rows.start, 'query_stop': rows.stop,
                       'compressed_key_rows': key_rows, 'query_rows_before_split': new_rows,
                       'workspace_rows': workspace, 'logits_budget_bytes': logits_bytes,
                       'workspace_exceeded': key_rows > workspace,
                       'all_requested_key_rows': requested_rows,
                       'workspace_group_split': requested_rows > workspace and key_rows <= workspace,
                       'query_subchunked': rows.start != 0 or rows.stop != new_rows,
                       'fallback': 'single_request_over_workspace' if key_rows > workspace else 'none'})
    return chunks

# A context manager is also needed around existing collective bodies (including returns).
@contextmanager
def span(label: str, metadata: dict[str, Scalar]) -> Iterator[None]:
    if not ENABLED or _recorder is None or _recorder.context is None:
        yield
        return
    rec = _recorder
    assert rec.context is not None
    if rec.backend.capturing():
        raise RuntimeError('PF1 trace cannot insert events into a captured graph')
    if len(rec.pending) >= rec.limit:
        rec.drain()
    if len(rec.pending) >= rec.limit:
        raise RuntimeError('PF1 trace pending event bound exceeded')
    start, end = rec.backend.event(), rec.backend.event()
    row: dict[str, object] = {**rec.context, **metadata, 'phase': label, 'event': rec.sequence,
                              'host_start_ns': time.monotonic_ns()}
    rec.sequence += 1
    start.record()
    begin_ns = time.perf_counter_ns()
    try:
        yield
    except BaseException:
        row['failed'] = True
        raise
    finally:
        row['wall_ms'] = (time.perf_counter_ns() - begin_ns) / 1e6
        end.record()
        rec.pending.append(Pending(row, start, end))
