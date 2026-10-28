# SPDX-License-Identifier: Apache-2.0
"""MC1 streamed landing: load a long persistent-tier hit window by window.

Without this, a hit from a persistent (secondary) tier is counted only once every chunk is promoted into the CPU
primary tier, so the whole prefix must be resident there at once; 0081 caps a request at 0.2 of the tier (29,568
tokens at 256 MiB). Here the connector counts a persistent-tier chunk as a hit without promoting it, allocates the
GPU blocks for the whole hit, and then lands it through the registered CPU region in windows of at most
``window_chunks``:

    promote window k (tier -> CPU slots)  ->  pin (prepare_load)  ->  load job (0077 gloo copy + CPU -> GPU)
    complete_load unpins window k; window k+1 was already promoting (one window of prefetch).

Peak CPU residency per request is two windows (<= the per-request budget), never the whole prefix. Every window
is an ordinary load job on the donor-proven data path; only its GPU spec carries ``dsv41_stream`` = ``window``
(more to come: the worker does not release the request) or ``final``. If a window cannot be landed (a chunk
vanished, failed its CRC, or never became resident before the deadline) or the request is aborted, the
controller emits one ``fail`` job: the workers report the request finished *and failed*, so the scheduler frees
its blocks and recomputes from scratch (multi-group layouts cannot report block-level errors). A landed prefix is
never reported partially valid. Once the final window is issued no fail job follows (the workers already
release the request with it), so a request is never reported finished twice.

Planning (``GroupLoad``, ``split_windows``) is pure; ``StreamController`` drives a TieringOffloadingManager.

Tier-agnostic. Nothing here or in the connector patches names a tier type: the matter tag travels in the
request's ``kv_transfer_params`` (``ReqContext``) to every secondary tier, and a tier takes part through this
contract alone, so Tier 1 (Spark NVMe, ``Dsv41MatterTier``) and a later Tier 2 (e.g. x86host RAM/NVMe) plug in the
same way (one more ``secondary_tiers`` entry, loaded by ``module_path``):

* class attribute ``dsv41_matter_gated = True`` and a MatterGate (``tiering/fs/dsv41_matter.py``, standard library
  only, no file-tier dependency) deciding every PUT and GET: default deny, excluded namespaces, fail closed, audit;
* ``allows_get(req_context) -> bool``: True only when the gate would serve this request (streaming eligibility);
* the SecondaryTierManager API with its usual meaning: a refused request gets ``lookup`` MISS and a failed
  ``JobResult`` with 0 bytes for any store or load; a load that fails is a failed ``JobResult`` (the stream then
  fails the request and it recomputes).

In multinode (production) mode the scheduler refuses to start with a secondary tier that does not declare the
contract (``ungated_tiers``), so no tier can persist an untagged request.
"""
from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
import time
from typing import Any

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import GPULoadStoreSpec, LoadStoreSpec, LookupResult, OffloadKey, ReqContext

logger = init_logger(__name__)

WINDOW, FINAL, FAIL = 'window', 'final', 'fail'


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def ungated_tiers(tiers: Sequence[Any]) -> list[str]:
    """Tier types (or class names) that do not declare the matter-gated persistent-tier contract."""
    return [str(getattr(t, 'tier_type', None) or type(t).__name__) for t in tiers
            if getattr(type(t), 'dsv41_matter_gated', False) is not True
            or not callable(getattr(t, 'allows_get', None))]


@dataclass(frozen=True)
class GroupLoad:
    """One KV cache group's pending load, exactly as update_state_after_alloc computed it."""

    keys: tuple[OffloadKey, ...]       # one offload key per CPU chunk, in chunk order
    dst_block_ids: tuple[int, ...]     # GPU blocks to fill, in order
    start_block: int                   # request-relative index of dst_block_ids[0] (block_indices[g])
    sliding: bool = False              # sliding-window group: landed whole, in the last window


@dataclass(frozen=True)
class Window:
    keys: tuple[OffloadKey, ...]
    dst_block_ids: tuple[int, ...]
    group_sizes: tuple[int, ...]       # one entry per KV cache group (0 = not in this window)
    block_indices: tuple[int, ...]


def _check_group(g: GroupLoad, bpc: int) -> None:
    if g.keys and cdiv(len(g.dst_block_ids) + g.start_block % bpc, bpc) != len(g.keys):
        raise ValueError('MC1 stream: a group\'s keys do not cover its GPU blocks')
    if bool(g.keys) != bool(g.dst_block_ids):
        raise ValueError('MC1 stream: keys without blocks (or blocks without keys)')


def _window(groups: Sequence[GroupLoad], parts: Sequence[tuple[int, int, int]], bpc: int) -> Window:
    sizes = [0] * len(groups)
    indices = [0] * len(groups)
    keys: list[OffloadKey] = []
    dst: list[int] = []
    for gi, c0, c1 in sorted(parts):
        g = groups[gi]
        j0 = g.start_block // bpc
        b0 = max(g.start_block, (j0 + c0) * bpc)
        b1 = min(g.start_block + len(g.dst_block_ids), (j0 + c1) * bpc)
        if b1 <= b0 or cdiv(b1 - b0 + b0 % bpc, bpc) != c1 - c0:
            raise ValueError('MC1 stream: window chunk range does not map onto whole GPU blocks')
        sizes[gi] = b1 - b0
        indices[gi] = b0
        keys.extend(g.keys[c0:c1])
        dst.extend(g.dst_block_ids[b0 - g.start_block:b1 - g.start_block])
    return Window(tuple(keys), tuple(dst), tuple(sizes), tuple(indices))


def split_windows(groups: Sequence[GroupLoad], blocks_per_chunk: int, window_chunks: int) -> list[Window]:
    """Pack the load into windows of <= window_chunks CPU chunks, in group then chunk order.

    The union of the windows is exactly the single load the upstream connector would issue: same keys, same GPU
    blocks, same first-chunk skip (block_indices). Sliding-window groups are never split and ride in the last
    window, which may therefore exceed window_chunks by at most their (window-bounded) size."""
    if window_chunks < 1 or blocks_per_chunk < 1:
        raise ValueError('MC1 stream: window and blocks_per_chunk must be >= 1')
    for g in groups:
        _check_group(g, blocks_per_chunk)
    packs: list[list[tuple[int, int, int]]] = [[]]
    room = window_chunks
    for gi, g in enumerate(groups):
        if g.sliding or not g.keys:
            continue
        c = 0
        while c < len(g.keys):
            if room == 0:
                packs.append([])
                room = window_chunks
            take = min(room, len(g.keys) - c)
            packs[-1].append((gi, c, c + take))
            c += take
            room -= take
    packs[-1].extend((gi, 0, len(g.keys)) for gi, g in enumerate(groups) if g.sliding and g.keys)
    return [_window(groups, p, blocks_per_chunk) for p in packs if p]


@dataclass
class _Stream:
    req_id: str
    req_context: ReqContext
    windows: list[Window]
    next: int = 0                      # next window to issue
    inflight: int | None = None        # job id of the issued window, until done
    fail_reason: str | None = None     # set: emit one fail job, then the stream ends
    fail_emitted: bool = False
    since: float = field(default_factory=time.monotonic)   # current window's wait start
    misses: int = 0
    prefetched: int = -1               # last window whose promotion was started ahead


class StreamController:
    """Owns the per-request streams of one OffloadingConnectorScheduler (scheduler thread only).

    ``emit(req_id, src_spec, dst_spec, keys) -> job_id`` registers one load job with the connector's job tables;
    the controller never touches them itself."""

    def __init__(self, manager: Any, window_chunks: int, max_wait_s: float = 60.0, max_miss_steps: int = 32) -> None:
        for attr in ('primary_tier', 'secondary_tiers', 'lookup', 'prepare_load'):
            if not hasattr(manager, attr):
                raise ValueError('MC1 stream: needs the TieringOffloadingManager (primary + secondary tiers)')
        if window_chunks < 1:
            raise ValueError('MC1 stream: window_chunks must be >= 1')
        self.manager = manager
        self.window_chunks = window_chunks
        self.max_wait_s = max_wait_s
        self.max_miss_steps = max_miss_steps
        self.streams: dict[str, _Stream] = {}
        self.stats = {'streams': 0, 'windows': 0, 'failed': 0, 'aborted': 0, 'completed': 0}

    # ----------------------------------------------------------- lookup side
    def eligible(self, req_context: ReqContext) -> bool:
        """Only requests a persistent tier will actually serve (matter gate ALLOW) stream."""
        for tier in self.manager.secondary_tiers:
            allows: Callable[[ReqContext], bool] | None = getattr(tier, 'allows_get', None)
            if allows is not None and allows(req_context):
                return True
        return False

    def exists(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        """TieringOffloadingManager.lookup without promotion: a persistent-tier chunk is a HIT where it lies."""
        primary = self.manager.primary_tier.lookup(key, req_context)
        if primary in (LookupResult.HIT, LookupResult.HIT_PENDING):
            return primary
        retry = False
        for tier in self.manager.secondary_tiers:
            if not req_context.load_tier_filter.allows(tier.medium, tier.locality):
                continue
            result = tier.lookup(key, req_context)
            if result is LookupResult.HIT:
                return LookupResult.HIT
            retry |= result is LookupResult.RETRY
        return LookupResult.RETRY if retry else LookupResult.MISS

    # ------------------------------------------------------------- lifecycle
    def start(self, req_id: str, req_context: ReqContext, groups: Sequence[GroupLoad], blocks_per_chunk: int) -> int:
        if req_id in self.streams:
            raise RuntimeError('MC1 stream: request already streaming')
        windows = split_windows(groups, blocks_per_chunk, self.window_chunks)
        if not windows:
            raise ValueError('MC1 stream: empty load')
        self.streams[req_id] = _Stream(req_id, req_context, windows)
        self.stats['streams'] += 1
        return len(windows)

    def active(self) -> bool:
        return bool(self.streams)

    def is_streaming(self, req_id: str) -> bool:
        return req_id in self.streams

    @staticmethod
    def _final_issued(s: _Stream) -> bool:
        return s.next == len(s.windows)

    def abort(self, req_id: str) -> None:
        """The request finished (e.g. aborted) mid-stream: stop issuing windows and free it with a fail job,
        unless the final window is already out (it releases the request by itself)."""
        s = self.streams.get(req_id)
        if s is not None and s.fail_reason is None and not self._final_issued(s):
            s.fail_reason = 'aborted'
            self.stats['aborted'] += 1

    def reset(self) -> list[str]:
        """reset_cache discarded every scheduler-side job. Streams whose final window is out end now (the workers
        still release the request with it); every other stream ends with a fail job. Returns the dropped ids."""
        dropped = [r for r, s in self.streams.items() if self._final_issued(s) and not s.fail_emitted]
        for r in dropped:
            del self.streams[r]
        for s in self.streams.values():
            if s.fail_emitted:
                continue  # its fail job is already with the workers
            s.inflight = None
            if s.fail_reason is None:
                s.fail_reason = 'reset'
        return dropped

    def remaining_keys(self, req_id: str) -> list[OffloadKey]:
        """Keys of windows not yet issued (issued windows release their own keys when they complete)."""
        s = self.streams.get(req_id)
        return [] if s is None else [k for w in s.windows[s.next:] for k in w.keys]

    def on_job_done(self, req_id: str, job_id: int) -> tuple[OffloadKey, ...] | None:
        """A stream job completed on every worker. None while the stream goes on (or for a non-stream job);
        when it is over (final or fail job), the keys of windows it never issued, which the caller releases."""
        s = self.streams.get(req_id)
        if s is None:
            return None
        if s.inflight != job_id:
            raise RuntimeError('MC1 stream: completion of a job this stream did not issue')
        if s.fail_emitted or self._final_issued(s):
            leftover = tuple(self.remaining_keys(req_id))
            del self.streams[req_id]
            if not s.fail_emitted:
                self.stats['completed'] += 1
            return leftover
        s.inflight = None
        s.since = time.monotonic()
        s.misses = 0
        return None

    # ------------------------------------------------------------- per step
    def advance(self, emit: Callable[[str, LoadStoreSpec, GPULoadStoreSpec, Collection[OffloadKey]], int],
                num_groups: int) -> None:
        """Issue every stream's next window whose chunks are resident; start promotions otherwise."""
        now = time.monotonic()
        for s in list(self.streams.values()):
            if s.inflight is not None:
                continue
            if s.fail_reason is not None:
                if not s.fail_emitted:
                    dst = GPULoadStoreSpec([], [0] * num_groups, [0] * num_groups)
                    setattr(dst, 'dsv41_stream', FAIL)  # read by the worker (getattr), pickles with the spec
                    s.fail_emitted = True
                    s.inflight = emit(s.req_id, LoadStoreSpec(), dst, ())
                    self.stats['failed'] += 1
                    logger.warning('MC1 stream: request load abandoned (%s) after %d/%d windows; recomputing',
                                   s.fail_reason, s.next, len(s.windows))
                continue
            w = s.windows[s.next]
            results = [self.manager.lookup(k, s.req_context) for k in w.keys]
            if all(r is LookupResult.HIT for r in results):
                src = self.manager.prepare_load(w.keys, s.req_context)
                dst = GPULoadStoreSpec(list(w.dst_block_ids), list(w.group_sizes), list(w.block_indices))
                setattr(dst, 'dsv41_stream', FINAL if s.next == len(s.windows) - 1 else WINDOW)
                s.inflight = emit(s.req_id, src, dst, w.keys)
                s.next += 1
                self.stats['windows'] += 1
                # Prefetch: start promoting the following window while this one lands (<= 2 windows resident).
                if s.next < len(s.windows) and s.prefetched < s.next:
                    s.prefetched = s.next
                    for k in s.windows[s.next].keys:
                        self.manager.lookup(k, s.req_context)
                continue
            if any(r is LookupResult.MISS for r in results):
                # Not resident and not promotable now: vanished from its tier (eviction, purge, CRC) or the
                # CPU tier is momentarily full. Bounded retry, then recompute.
                s.misses += 1
                if s.misses > self.max_miss_steps:
                    s.fail_reason = 'chunk unavailable'
                    continue
            if now - s.since > self.max_wait_s:
                s.fail_reason = 'window wait deadline'
