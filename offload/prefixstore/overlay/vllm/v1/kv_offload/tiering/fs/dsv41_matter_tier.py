# SPDX-License-Identifier: Apache-2.0
"""MC1 Tier 1: matter-scoped persistent KV file tier with the MatterGate.

Loaded as an out-of-tree secondary tier (SecondaryTierFactory ``module_path``)::

    {"type": "Dsv41MatterTier", "module_path": "vllm.v1.kv_offload.tiering.fs.dsv41_matter_tier",
     "root_dir": "/root/.cache/kv_matter_mc1", "excluded_ns_file": ".../matter-cache-excluded-ns.txt",
     "excluded_ns_sha256": "<pinned>", "audit_path": ".../audit.jsonl", "n_read_threads": 2, "n_write_threads": 1}

Layout: ``<root>/ns/<matter_ns>/<FileMapper namespace>_r<rank>/<hhh>/<hh>_g<group>/<hash>.bin`` (+ ``.crc``,
patch 0056). Every namespace has its own directory, so a purge is one subtree and two matters never share a file
even when their prompts hash alike. The lookup cache is keyed by (namespace, key) for the same reason.

Same I/O as the upstream FileSystemTierManager (0056 CRC sidecars, 0081 I/O threads, O_DIRECT only in multinode
mode) behind the gate in ``dsv41_matter``: default deny, excluded namespaces refused, fail closed, audit, purge.

Bounds (item 6): ``max_bytes`` for the tier and ``max_ns_bytes`` per namespace are hard admission limits; above
``high`` x a bound the janitor evicts down to ``low`` x it (oldest files first inside a namespace, least recently
used namespaces first for the tier). Files unused for ``ttl_days`` expire. A load refreshes its files' mtime, so
mtime is last use. Excluded namespaces found on disk are purged at start. Every deletion is audited.
"""
from __future__ import annotations

from collections.abc import Collection, Iterable
import contextlib
from dataclasses import dataclass, field
import fcntl
import functools
import json
import os
import queue
import threading
import time
from typing import TYPE_CHECKING, Any, ClassVar

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadKey,
    ReqContext,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobResult,
    RequestOffloadingContext,
    ScheduleEndContext,
    SecondaryTierManager,
    TransferJob,
)
from vllm.v1.kv_offload.tiering.fs import dsv41_matter as matter
from vllm.v1.kv_offload.tiering.fs.io import batch_load_block, batch_store_block, probe_o_direct
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec
    from vllm.v1.kv_offload.tiering.backpressure import BackpressureDetector

logger = init_logger(__name__)

_NS_BYTES = 32  # composite lookup key = bytes.fromhex(matter_ns) + OffloadKey


@dataclass
class _ReqState:
    tag: matter.RequestTag
    refusal_audited: bool = False
    cap_audited: bool = False
    missed: set[bytes] = field(default_factory=set)


@dataclass
class _Job:
    kind: str                     # 'put' or 'get'
    ns: str
    keys: list[OffloadKey]
    composite: list[bytes] = field(default_factory=list)
    tokens: int | None = None


class _MatterLookup(AsyncLookupManager):  # type: ignore[misc]
    """Existence checks on the worker thread; keys carry their namespace."""

    def __init__(self, tier: Dsv41MatterTier, tier_type: str) -> None:
        super().__init__(tier_type=tier_type)
        self._tier = tier

    def batch_lookup(self, keys: list[bytes], req_context: ReqContext) -> Iterable[bool]:
        return [os.path.exists(self._tier.path_of(k[:_NS_BYTES].hex(), OffloadKey(k[_NS_BYTES:]))) for k in keys]


class Dsv41MatterTier(SecondaryTierManager):  # type: ignore[misc]
    medium: ClassVar[Medium] = Medium.STORAGE
    # The upstream fs tier refuses to run unscoped in multinode mode (patch 0101); this tier is the scoped one.
    dsv41_matter_gated: ClassVar[bool] = True

    def __init__(
        self,
        offloading_spec: OffloadingSpec,
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        excluded_ns_file: str,
        excluded_ns_sha256: str,
        audit_path: str,
        max_bytes: int,
        max_ns_bytes: int,
        ttl_days: float = 30.0,
        n_read_threads: int = 2,
        n_write_threads: int = 1,
        tombstone_s: float = 600.0,
        janitor_interval_s: float = 30.0,
        gate_recheck_s: float = 30.0,
        ttl_scan_s: float = 6 * 3600.0,
        high_water: float = 0.9,
        low_water: float = 0.8,
        enable_kv_events: bool = False,
        locality: str | None = None,
        backpressure_detector: BackpressureDetector | None = None,
    ) -> None:
        super().__init__(offloading_spec, primary_kv_view, tier_type, backpressure_detector)
        if enable_kv_events:
            # Events would publish block hashes of case documents to an event bus.
            raise ValueError('MC1: the matter tier never emits KV events')
        if ttl_days <= 0:
            raise ValueError('MC1: ttl_days must be positive')
        self._usage = matter.StoreUsage(int(max_bytes), int(max_ns_bytes), high_water, low_water)
        self.locality = Locality(locality) if locality is not None else None
        assert primary_kv_view.strides is not None
        self._block_size: int = primary_kv_view.strides[0]
        self._spec = offloading_spec
        self._root = os.path.abspath(root_dir)
        matter.ensure_private_dir(self._root)
        matter.ensure_private_dir(os.path.join(self._root, matter.NS_DIR))
        matter.ensure_private_dir(os.path.join(self._root, matter.PURGE_DIR))
        # One engine per root: LRU/purge bookkeeping assumes a single owner. Held for the process lifetime.
        self._lock_fd = os.open(os.path.join(self._root, matter.LOCK_FILE), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(self._lock_fd)
            raise ValueError(f'MC1: matter tier root {self._root} is in use by another engine') from exc

        self._use_o_direct = probe_o_direct(self._root)
        if not self._use_o_direct and os.environ.get('VLLM_DSV41_OFFLOAD_MULTINODE', '0') == '1':
            raise ValueError('RP1: file tier requires O_DIRECT; buffered fallback violates Spark memory budget')

        self._rank = int(offloading_spec.config.parallel.rank)
        audit: matter.AuditLog | None = None
        audit_error: str | None = None
        try:
            audit = matter.AuditLog(audit_path, self._rank)
        except (OSError, ValueError) as exc:
            audit_error = 'audit-unavailable'
            logger.error('MC1 matter tier: audit log unusable (%s); persistence OFF for every request', type(exc).__name__)
        self._gate = matter.MatterGate(matter.ExclusionSet(excluded_ns_file, excluded_ns_sha256, gate_recheck_s),
                                       audit, audit_error)
        self._tokens_per_chunk = {int(g.group_id): int(g.tokens_per_block) * int(offloading_spec.blocks_per_chunk)
                                  for g in offloading_spec.config.groups}
        self._mappers: dict[str, FileMapper] = {}
        self._mapper_lock = threading.Lock()
        self._jobs: dict[JobId, _Job] = {}
        self._immediate: list[JobResult] = []
        self._load_progress: dict[JobId, int] = {}
        self._tombstone_s = tombstone_s
        self._ttl_s = ttl_days * 86400.0
        self._ttl_scan_s = ttl_scan_s
        self._caps_queued = False
        self._pool = DualQueueThreadPool(n_read_threads, n_write_threads, thread_name_prefix='vllm_kv_matter_fs')
        self._lookup = _MatterLookup(self, tier_type)
        self._janitor_interval_s = janitor_interval_s
        self._janitor_queue: queue.Queue[tuple[str, str] | None] = queue.Queue()
        reason = self._gate.closed_reason
        if reason is None:
            logger.info('MC1 matter tier OPEN at %s: %d excluded namespaces (hashes), O_DIRECT %s, bounds %d B tier / '
                        '%d B namespace, TTL %.1f days', self._root, len(self._gate.exclusions.excluded),
                        self._use_o_direct, self._usage.max_bytes, self._usage.max_ns_bytes, ttl_days)
        else:
            logger.error('MC1 matter tier CLOSED (%s): no persistent PUT/GET for any request', reason)
        self._audit('gate', None, state='open' if reason is None else 'closed', reason=reason)
        # Started last: its first job is the startup scan (usage, purge of excluded namespaces, TTL, bounds).
        self._janitor = threading.Thread(target=self._janitor_loop, name='vllm_kv_matter_janitor', daemon=True)
        self._janitor.start()

    # ------------------------------------------------------------------ helpers
    def _audit(self, op: str, ns: str | None, **fields: Any) -> None:
        audit = self._gate.audit
        if audit is None:
            return
        try:
            audit.write(op, ns, **fields)
        except OSError:
            # An audit that cannot be written closes persistence; never continue silently.
            self._gate.audit = None
            self._gate.audit_error = 'audit-write-failed'
            logger.error('MC1 matter tier: audit write failed; persistence OFF for every request')

    def mapper(self, ns: str) -> FileMapper:
        with self._mapper_lock:
            m = self._mappers.get(ns)
            if m is None:
                m = FileMapper.from_offloading_spec(root_dir=matter.ns_root(self._root, ns),
                                                    offloading_spec=self._spec,
                                                    blocks_per_file=self._spec.blocks_per_chunk,
                                                    parallel_agnostic=True)
                self._mappers[ns] = m
            return m

    def path_of(self, ns: str, key: OffloadKey) -> str:
        return str(self.mapper(ns).get_file_name(key))

    def _tokens(self, keys: Collection[OffloadKey]) -> int | None:
        total = 0
        for key in keys:
            n = self._tokens_per_chunk.get(get_offload_group_idx(key))
            if n is None:
                return None
            total += n
        return total

    def _state(self, ctx: ReqContext) -> _ReqState:
        state: _ReqState | None = ctx.get_state(_ReqState)
        if state is None:
            state = _ReqState(self._gate.classify(ctx.kv_transfer_params))
            ctx.set_state(state)
            if state.tag.verdict is matter.Verdict.UNTAGGED:
                self._gate.untagged_requests += 1
        return state

    def _decide(self, ctx: ReqContext) -> tuple[matter.Verdict, str | None]:
        state = self._state(ctx)
        verdict = self._gate.decide(state.tag)
        if verdict in matter.AUDITED_REFUSALS and not state.refusal_audited:
            state.refusal_audited = True
            self._audit('refuse', state.tag.ns, reason=verdict.value)
        return verdict, state.tag.ns

    def _refuse_job(self, job: TransferJob) -> None:
        self._immediate.append(JobResult(job_id=job.job_id, success=False, transfer_bytes=0))

    def allows_get(self, req_context: ReqContext) -> bool:
        """MC1 streamed landing: only requests this tier would serve stream (no audit side effect beyond _decide)."""
        return self._decide(req_context)[0] is matter.Verdict.ALLOW

    # ------------------------------------------------------------- tier API
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        self._decide(req_context)
        return RequestOffloadingContext()

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        verdict, ns = self._decide(req_context)
        if verdict is not matter.Verdict.ALLOW:
            return LookupResult.MISS  # no disk access for a refused request
        assert ns is not None
        result = self._lookup.lookup(bytes.fromhex(ns) + key, req_context)
        if result is None:
            return LookupResult.RETRY
        if not result:
            self._state(req_context).missed.add(key)
        return LookupResult.HIT if result else LookupResult.MISS

    def submit_store(self, job_metadata: TransferJob) -> None:
        verdict, ns = self._decide(job_metadata.req_context)
        if verdict is not matter.Verdict.ALLOW:
            self._refuse_job(job_metadata)
            return
        assert ns is not None
        keys = list(job_metadata.keys)
        refused = self._usage.admit(ns, len(keys) * (self._block_size + 16))
        if refused is not None:
            state = self._state(job_metadata.req_context)
            if not state.cap_audited:
                state.cap_audited = True
                self._audit('refuse', ns, reason=refused)
            self._queue_caps()
            self._refuse_job(job_metadata)
            return
        mapper = self.mapper(ns)
        paths = [str(mapper.get_file_name(k)) for k in keys]
        offsets = [int(cid) * self._block_size for cid in job_metadata.chunk_ids]
        task = functools.partial(self._store_task, mapper, paths, offsets)
        self._jobs[job_metadata.job_id] = _Job('put', ns, keys, tokens=self._tokens(keys))
        self._pool.enqueue_store(job_metadata.job_id, 1, [task])

    def _store_task(self, mapper: FileMapper, paths: list[str], offsets: list[int]) -> None:
        config_path = str(mapper.get_config_file_path())
        if not os.path.exists(config_path):
            os.makedirs(os.path.dirname(config_path), mode=0o700, exist_ok=True)
            tmp = config_path + f'.{threading.get_ident()}.tmp'
            with open(tmp, 'w') as f:
                json.dump(mapper.get_run_config(), f, indent=2, sort_keys=True)
            os.replace(tmp, config_path)
        batch_store_block(paths, self._primary_kv_view, offsets, self._block_size, self._use_o_direct)

    def submit_load(self, job_metadata: TransferJob) -> None:
        verdict, ns = self._decide(job_metadata.req_context)
        if verdict is not matter.Verdict.ALLOW:
            self._refuse_job(job_metadata)
            return
        assert ns is not None
        job_id = job_metadata.job_id
        keys = list(job_metadata.keys)
        prefix = bytes.fromhex(ns)
        self._jobs[job_id] = _Job('get', ns, keys, [prefix + k for k in keys], self._tokens(keys))
        paths = [self.path_of(ns, k) for k in keys]
        offsets = [int(cid) * self._block_size for cid in job_metadata.chunk_ids]

        def load_task() -> None:
            try:
                batch_load_block(paths, self._primary_kv_view, offsets, self._block_size, self._use_o_direct)
            except OSError as exc:
                # Written before task_done publishes the job (see FileSystemTierManager.submit_load).
                self._load_progress[job_id] = getattr(exc, 'num_succeeded', 0)
                raise
            for path in paths:  # mtime = last use: LRU inside the namespace and the TTL clock
                with contextlib.suppress(FileNotFoundError):
                    os.utime(path)

        self._pool.enqueue_load(job_id, 1, [load_task])

    def get_finished_jobs(self) -> Iterable[JobResult]:
        results = self._immediate
        self._immediate = []
        for job_id, success, transfer_time in self._pool.get_finished():
            job = self._jobs.pop(job_id)
            nbytes = len(job.keys) * self._block_size
            if job.kind == 'put':
                if success:
                    self._audit('put', job.ns, chunks=len(job.keys), tokens=job.tokens, bytes=nbytes)
                    self._usage.add(job.ns, len(job.keys) * (self._block_size + 16), time.time())
                    if self._usage.over_high():
                        self._queue_caps()
                results.append(JobResult(job_id=job_id, success=success, transfer_time=transfer_time,
                                         transfer_bytes=nbytes if success else 0))
                continue
            done = len(job.keys) if success else self._load_progress.pop(job_id, 0)
            if not success:
                self._lookup.mark_miss(job.composite[done:])
            if done:
                self._usage.touch(job.ns, time.time())
                self._audit('get-hit', job.ns, chunks=done, tokens=self._tokens(job.keys[:done]),
                            bytes=done * self._block_size)
            if not success:
                self._audit('get-miss', job.ns, chunks=len(job.keys) - done, reason='load-failed')
            results.append(JobResult(job_id=job_id, success=success,
                                     successful_keys=tuple(job.keys[:done]) if not success and done else None,
                                     transfer_time=transfer_time, transfer_bytes=done * self._block_size))
        return results

    def take_events(self) -> Iterable[OffloadingEvent]:
        return ()

    def drain_jobs(self) -> None:
        self._pool.wait_idle()

    def on_request_finished(self, req_context: ReqContext) -> None:
        state: _ReqState | None = req_context.get_state(_ReqState)
        if state is not None and state.missed and state.tag.ns is not None:
            self._audit('get-miss', state.tag.ns, chunks=len(state.missed), reason='absent')
        self._lookup.cleanup(req_context.req_id)

    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._lookup.flush()
        self._revalidate_gate()

    def _revalidate_gate(self) -> None:
        if self._gate.exclusions.revalidate():
            logger.error('MC1 matter tier CLOSED (%s): excluded-ns file changed or vanished', self._gate.closed_reason)
            self._audit('gate', None, state='closed', reason=self._gate.closed_reason)

    def _queue_caps(self) -> None:
        if not self._caps_queued:
            self._caps_queued = True
            self._janitor_queue.put(('caps', ''))

    @property
    def usage(self) -> matter.StoreUsage:
        return self._usage

    def wait_ready(self, timeout: float) -> bool:
        """True once the startup scan finished (stores are refused as 'scan-pending' until then)."""
        deadline = time.monotonic() + timeout
        while not self._usage.ready and time.monotonic() < deadline:
            time.sleep(0.01)
        return bool(self._usage.ready)

    # ----------------------------------------------------------------- purge
    def request_purge(self, target: str) -> None:
        """Purge one namespace (or ALL): refuse GET/PUT for it now, delete in the janitor, sweep again later."""
        until = time.monotonic() + self._tombstone_s
        self._gate.tombstone(target, until)
        self._janitor_queue.put(('purge', target))

    def _janitor_loop(self) -> None:
        """Owns every deletion: purges, TTL expiry and bound enforcement, plus purge-request polling, tombstone
        sweeps and gate re-checks, so an idle engine still honours them."""
        try:
            self._startup_scan()
        except (OSError, ValueError) as exc:
            logger.error('MC1 matter tier: startup scan failed (%s); stores stay refused', type(exc).__name__)
        last_ttl = time.monotonic()
        while True:
            try:
                item = self._janitor_queue.get(timeout=self._janitor_interval_s)
            except queue.Empty:
                item = ('tick', '')
            if item is None:
                return
            action, target = item
            try:
                if action in ('purge', 'sweep'):
                    self._purge(action, target)
                elif action == 'caps':
                    self._caps_queued = False
                    self._enforce_bounds()
                else:
                    for request in matter.take_purge_requests(self._root):
                        self.request_purge(request)
                    for ns in self._gate.expire_tombstones(time.monotonic()):
                        self._janitor_queue.put(('sweep', ns))
                    self._revalidate_gate()
                    if time.monotonic() - last_ttl >= self._ttl_scan_s:
                        last_ttl = time.monotonic()
                        self._expire()
                    self._enforce_bounds()
            except (OSError, ValueError) as exc:
                logger.error('MC1 matter tier: janitor %s failed (%s)', action, type(exc).__name__)
                if action in ('purge', 'sweep'):
                    logger.error('MC1 matter tier: namespace stays refused until restart')
                    self._gate.tombstone(target, float('inf'))

    def _purge(self, action: str, target: str) -> None:
        targets = matter.namespaces(self._root) if target == matter.PURGE_ALL else [target]
        files = size = 0
        for ns in targets:
            f, s = matter.purge_namespace(self._root, ns)
            files += f
            size += s
            self._usage.set_ns(ns, 0, 0.0)
            with self._mapper_lock:
                self._mappers.pop(ns, None)
        self._audit('purge' if action == 'purge' else 'purge-sweep',
                    None if target == matter.PURGE_ALL else target,
                    scope='all' if target == matter.PURGE_ALL else 'namespace', files=files, bytes=size)

    def _rescan(self, ns: str) -> list[matter.BlockFile]:
        """Exact usage of one namespace; a namespace left without blocks is removed (config files only)."""
        blocks, other = matter.scan_namespace(self._root, ns)
        if not blocks:
            matter.purge_namespace(self._root, ns)
            self._usage.set_ns(ns, 0, 0.0)
            with self._mapper_lock:
                self._mappers.pop(ns, None)
            return blocks
        last = max(b.mtime for b in blocks)
        self._usage.set_ns(ns, sum(b.nbytes for b in blocks) + other, last)
        return blocks

    def _startup_scan(self) -> None:
        excluded = self._gate.exclusions.excluded if self._gate.closed_reason is None else frozenset()
        for ns in matter.namespaces(self._root):
            if ns in excluded:
                # Persisted before it was excluded: case data that must not stay at rest.
                files, size = matter.purge_namespace(self._root, ns)
                self._audit('purge', ns, scope='namespace', reason='excluded', files=files, bytes=size)
                continue
            self._rescan(ns)
        self._expire()
        self._usage.ready = True
        self._enforce_bounds()

    def _expire(self) -> None:
        cutoff = time.time() - self._ttl_s
        for ns in matter.namespaces(self._root):
            if self._gate.is_tombstoned(ns):
                continue
            blocks, _ = matter.scan_namespace(self._root, ns)
            old = [b for b in blocks if b.mtime < cutoff]
            freed = sum(matter.remove_block(b) for b in old)
            freed += matter.remove_stale_temps(self._root, ns, time.time() - 3600.0)
            if old or freed:
                matter.prune_empty_dirs(matter.ns_root(self._root, ns))
                self._audit('expire', ns, files=len(old), bytes=freed)
            self._rescan(ns)

    def _evict(self, ns: str, target_bytes: float, reason: str) -> int:
        """Oldest files of one namespace until it holds <= target_bytes (or is empty). Returns bytes freed."""
        blocks = self._rescan(ns)
        have = self._usage.ns_bytes(ns)
        freed = files = 0
        for block in blocks:
            if have - freed <= target_bytes:
                break
            freed += matter.remove_block(block)
            files += 1
        if files:
            matter.prune_empty_dirs(matter.ns_root(self._root, ns))
            self._audit('evict', ns, files=files, bytes=freed, reason=reason)
            self._rescan(ns)
        return freed

    def _enforce_bounds(self) -> None:
        u = self._usage
        for ns in u.namespaces():
            if u.ns_bytes(ns) > u.high * u.max_ns_bytes:
                self._evict(ns, u.low * u.max_ns_bytes, 'ns-cap')
        if u.total > u.high * u.max_bytes:
            for ns in u.lru_namespaces():
                excess = u.total - u.low * u.max_bytes
                if excess <= 0:
                    break
                self._evict(ns, max(u.ns_bytes(ns) - excess, 0.0), 'tier-cap')

    def shutdown(self) -> None:
        self._lookup.shutdown()
        self._pool.shutdown(wait=True)
        self._janitor_queue.put(None)
        self._janitor.join()
        if self._gate.audit is not None:
            self._gate.audit.close()
        os.close(self._lock_fd)

    # ------------------------------------------------------------ diagnostics
    @property
    def gate(self) -> matter.MatterGate:
        return self._gate
