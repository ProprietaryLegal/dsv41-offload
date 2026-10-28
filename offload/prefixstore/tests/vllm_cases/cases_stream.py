"""MC1 item 4: streamed landing. Planning, equivalence with the real CPU->GPU handler's copy descriptors, the
controller state machine, and the real connector scheduler/worker around a real tiering manager + matter tier
landing a prefix far larger than the CPU tier (CPU only; the GPU copy is simulated with the handler's arithmetic)."""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import os
from pathlib import Path
import random
import sys
import time
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc1_connector as c
import mc1_helpers as h

import vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler as sched_mod
from vllm.distributed.kv_transfer.kv_connector.v1.offloading import dsv41_stream as ds
from vllm.v1.kv_offload.base import CanonicalKVCacheRef, GPULoadStoreSpec, LookupResult, make_offload_key
from vllm.v1.kv_offload.cpu import gpu_worker
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.tiering.base import TransferJob
from vllm.v1.request import RequestStatus


# ------------------------------------------------------------------ planning
def groups(seed: int, n_groups: int = 2, bpc: int = 2) -> list[ds.GroupLoad]:
    rng = random.Random(seed)
    out = []
    next_block = 0
    for g in range(n_groups):
        start = rng.randrange(0, 9)            # request-relative first pending block (may be mid-chunk)
        n_blocks = rng.randrange(0, 60)
        n_keys = -(-(n_blocks + start % bpc) // bpc) if n_blocks else 0
        out.append(ds.GroupLoad(tuple(h.key(i, g) for i in range(n_keys)),
                                tuple(range(next_block, next_block + n_blocks)), start))
        next_block += 1000
    return out


@pytest.mark.parametrize('seed', range(40))
def test_windows_partition_the_single_load(seed: int) -> None:
    gs = groups(seed)
    window = random.Random(seed).randrange(1, 12)
    ws = ds.split_windows(gs, 2, window)
    assert all(len(w.keys) <= window for w in ws)
    assert [k for w in ws for k in w.keys] == [k for g in gs for k in g.keys]
    for gi, g in enumerate(gs):
        blocks = [b for w in ws for b in w.dst_block_ids[sum(w.group_sizes[:gi]):sum(w.group_sizes[:gi + 1])]]
        assert blocks == list(g.dst_block_ids)
        firsts = [w.block_indices[gi] for w in ws if w.group_sizes[gi]]
        if firsts:
            assert firsts[0] == g.start_block                       # first window keeps the first-chunk skip
            assert all(b % 2 == 0 for b in firsts[1:])              # later windows start chunk-aligned


def test_sliding_group_rides_whole_in_last_window() -> None:
    full = ds.GroupLoad(tuple(h.key(i) for i in range(10)), tuple(range(20)), 0)
    swa = ds.GroupLoad(tuple(h.key(i, 1) for i in range(3)), tuple(range(100, 106)), 40, sliding=True)
    ws = ds.split_windows([full, swa], 2, 4)
    assert [len(w.keys) for w in ws] == [4, 4, 2 + 3]
    assert ws[-1].group_sizes[1] == 6 and all(w.group_sizes[1] == 0 for w in ws[:-1])


def test_planning_refuses_inconsistent_loads() -> None:
    with pytest.raises(ValueError):
        ds.split_windows([ds.GroupLoad((h.key(0),), (1, 2, 3), 0)], 2, 4)   # 3 blocks need 2 chunks
    with pytest.raises(ValueError):
        ds.split_windows([ds.GroupLoad((), (1,), 0)], 2, 4)
    with pytest.raises(ValueError):
        ds.split_windows([], 2, 0)


# ------------------------------------------- equivalence with the real handler
def handler_ops() -> Any:
    """The descriptor-building part of SingleDirectionOffloadingHandler.transfer_async, executed as is."""
    src = Path(gpu_worker.__file__).read_text()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SingleDirectionOffloadingHandler')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'transfer_async')
    stop = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'stream')
    fn.body = fn.body[:stop] + ast.parse('return src.numpy().copy(), dst.numpy().copy(), sizes.numpy().copy()').body
    fn.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    scope = dict(vars(gpu_worker))
    # Descriptor buffers are pinned on GPU hosts; on CPU they are ordinary tensors (allocation only, no logic).
    scope['_new_descriptor_buffers'] = lambda n: tuple(torch.empty(n, dtype=torch.int64) for _ in range(3))
    exec(compile(module, gpu_worker.__file__, 'exec'), scope)
    return scope['transfer_async']


def fake_handler(bpc: int, pages: list[int]) -> Any:
    obj = gpu_worker.SingleDirectionOffloadingHandler.__new__(gpu_worker.SingleDirectionOffloadingHandler)
    obj.src_tensors = [torch.zeros((64, p * bpc), dtype=torch.int8) for p in pages]     # CPU chunks
    obj.dst_tensors = [torch.zeros((4096, p), dtype=torch.int8) for p in pages]         # "GPU" blocks
    obj.gpu_to_cpu = False
    obj.src_blocks_per_chunk, obj.dst_blocks_per_chunk = bpc, 1
    obj.layer_refs_per_group = [[CanonicalKVCacheRef(i, p)] for i, p in enumerate(pages)]
    obj._canonical_copy_plans = None
    obj._buffer_pool = []
    obj._fill_group_ops = MethodType(gpu_worker.SingleDirectionOffloadingHandler._fill_direct_ops, obj)
    return obj


@pytest.mark.parametrize('seed', range(25))
def test_window_copies_equal_the_single_load_copies(seed: int) -> None:
    gs = groups(seed)
    if not any(g.keys for g in gs):
        return
    ops = handler_ops()
    obj = fake_handler(2, [512, 768])
    chunk_of = {k: i % 64 for i, k in enumerate(k for g in gs for k in g.keys)}

    def run(keys: tuple[Any, ...], dst: tuple[int, ...], sizes: tuple[int, ...], indices: tuple[int, ...]) -> Counter[tuple[int, int, int]]:
        src_spec = CPULoadStoreSpec([chunk_of[k] for k in keys])
        s, d, n = ops(obj, 0, src_spec, GPULoadStoreSpec(list(dst), list(sizes), list(indices)))
        return Counter(zip(s.tolist(), d.tolist(), n.tolist()))

    full = run(tuple(k for g in gs for k in g.keys), tuple(b for g in gs for b in g.dst_block_ids),
               tuple(len(g.dst_block_ids) for g in gs), tuple(g.start_block for g in gs))
    window = random.Random(seed).randrange(1, 9)
    windowed: Counter[tuple[int, int, int]] = Counter()
    for w in ds.split_windows(gs, 2, window):
        windowed += run(w.keys, w.dst_block_ids, w.group_sizes, w.block_indices)
    assert windowed == full and sum(full.values()) == sum(len(g.dst_block_ids) for g in gs)


# ------------------------------------------------------ controller (fake manager)
class FakeManager:
    """Primary tier state per key: 'disk' (file tier only) -> 'promoting' -> 'resident' -> pinned."""

    def __init__(self, keys: list[Any], promote_steps: int = 1) -> None:
        self.state = {k: 'disk' for k in keys}
        self.age: dict[Any, int] = {}
        self.pinned: Counter[Any] = Counter()
        self.promote_steps = promote_steps
        self.primary_tier = SimpleNamespace(lookup=self._primary)
        self.secondary_tiers = [SimpleNamespace(medium=None, locality=None, allows_get=lambda ctx: True,
                                                lookup=lambda k, ctx: LookupResult.HIT if self.state.get(k) else LookupResult.MISS)]
        self.peak_resident = 0

    def _primary(self, k: Any, ctx: Any) -> Any:
        st = self.state.get(k)
        if st == 'resident':
            return LookupResult.HIT
        return LookupResult.HIT_PENDING if st == 'promoting' else LookupResult.MISS

    def lookup(self, k: Any, ctx: Any) -> Any:
        st = self.state.get(k)
        if st == 'disk':
            self.state[k] = 'promoting'
            self.age[k] = 0
            return LookupResult.HIT_PENDING
        return self._primary(k, ctx)

    def prepare_load(self, keys: Any, ctx: Any) -> Any:
        for k in keys:
            assert self.state[k] == 'resident'
            self.pinned[k] += 1
        return CPULoadStoreSpec(list(range(len(keys))))

    def complete_load(self, keys: Any) -> None:
        for k in keys:
            self.pinned[k] -= 1
            self.state[k] = 'evicted-after-use'

    def tick(self) -> None:
        for k, st in self.state.items():
            if st == 'promoting':
                self.age[k] += 1
                if self.age[k] >= self.promote_steps:
                    self.state[k] = 'resident'
        resident = sum(1 for st in self.state.values() if st in ('promoting', 'resident'))
        self.peak_resident = max(self.peak_resident, resident)


def drive(ctl: ds.StreamController, mgr: FakeManager, steps: int, on_emit: Any = None) -> list[tuple[str, int]]:
    log: list[tuple[str, int]] = []
    jobs: dict[int, tuple[str, Any]] = {}

    def emit(req_id: str, src: Any, dst: Any, keys: Any) -> int:
        job = len(log) + 1
        log.append((getattr(dst, 'dsv41_stream'), len(keys)))
        jobs[job] = (req_id, tuple(keys))
        if on_emit:
            on_emit(len(log))
        return job

    for _ in range(steps):
        ctl.advance(emit, 2)
        mgr.tick()
        for job, (req_id, keys) in list(jobs.items()):
            mgr.complete_load(keys)
            ctl.on_job_done(req_id, job)
            del jobs[job]
        if not ctl.active():
            break
    return log


def test_controller_orders_windows_prefetches_and_bounds_residency() -> None:
    gs = [ds.GroupLoad(tuple(h.key(i) for i in range(20)), tuple(range(40)), 0),
          ds.GroupLoad(tuple(h.key(i, 1) for i in range(13)), tuple(range(100, 126)), 0)]
    mgr = FakeManager([k for g in gs for k in g.keys], promote_steps=2)
    ctl = ds.StreamController(mgr, window_chunks=5)
    assert ctl.start('r', h.ctx('r', None), gs, 2) == 7
    log = drive(ctl, mgr, 200)
    assert [kind for kind, _ in log] == ['window'] * 6 + ['final']
    assert [n for _, n in log] == [5, 5, 5, 5, 5, 5, 3]
    assert mgr.peak_resident <= 2 * 5            # the window landing + one prefetching, never the prefix
    assert not ctl.active() and ctl.stats['completed'] == 1


def test_controller_fails_once_on_vanished_chunk_and_after_abort() -> None:
    gs = [ds.GroupLoad(tuple(h.key(i) for i in range(12)), tuple(range(24)), 0)]
    mgr = FakeManager(list(gs[0].keys))
    del mgr.state[gs[0].keys[7]]                  # evicted / purged / CRC-removed from the file tier
    ctl = ds.StreamController(mgr, window_chunks=4, max_miss_steps=3)
    ctl.start('r', h.ctx('r', None), gs, 2)
    log = drive(ctl, mgr, 100)
    assert [kind for kind, _ in log] == ['window', 'fail']
    assert ctl.stats['failed'] == 1 and not ctl.active()

    mgr = FakeManager(list(gs[0].keys))
    ctl = ds.StreamController(mgr, window_chunks=4)
    ctl.start('r', h.ctx('r', None), gs, 2)
    log = drive(ctl, mgr, 100, on_emit=lambda n: ctl.abort('r') if n == 1 else None)
    assert [kind for kind, _ in log] == ['window', 'fail']

    mgr = FakeManager(list(gs[0].keys))
    ctl = ds.StreamController(mgr, window_chunks=12)
    ctl.start('r', h.ctx('r', None), gs, 2)
    log = drive(ctl, mgr, 100, on_emit=lambda n: ctl.abort('r'))
    assert [kind for kind, _ in log] == ['final']   # final already out: never a second release


def test_controller_reset_and_deadline() -> None:
    gs = [ds.GroupLoad(tuple(h.key(i) for i in range(8)), tuple(range(16)), 0)]
    mgr = FakeManager(list(gs[0].keys), promote_steps=10**9)   # never becomes resident
    ctl = ds.StreamController(mgr, window_chunks=4, max_wait_s=0.0)
    ctl.start('r', h.ctx('r', None), gs, 2)
    log = drive(ctl, mgr, 5)
    assert [kind for kind, _ in log] == ['fail']
    mgr = FakeManager(list(gs[0].keys))
    ctl = ds.StreamController(mgr, window_chunks=8)
    ctl.start('a', h.ctx('a', None), gs, 2)
    ctl.advance(lambda *a: 1, 1)
    mgr.tick()
    ctl.advance(lambda *a: 1, 1)               # final issued for 'a'
    ctl.start('b', h.ctx('b', None), gs, 2)
    assert ctl.reset() == ['a']                # 'a' is released by its final job; 'b' gets a fail job
    kinds: list[str] = []

    def record(req: str, src: Any, dst: Any, keys: Any) -> int:
        kinds.append(getattr(dst, 'dsv41_stream'))
        return 9

    ctl.advance(record, 1)
    assert kinds == ['fail']
    with pytest.raises(RuntimeError):
        ctl.on_job_done('b', 10)


# --------------------------------------- real connector around real tiers (CPU)
PROMPT = 6000


def stored_bytes(k: Any) -> np.ndarray:
    return np.frombuffer(hashlib.shake_256(bytes(k)).digest(h.BLOCK), dtype=np.uint8)


@pytest.fixture()
def store(tmp_path: Path) -> Any:
    return make_store(tmp_path)


def make_store(tmp_path: Path) -> Any:
    """A previous engine run persisted the prompt's chunks for one synthetic matter; then it stopped."""
    ns = h.ns()
    excluded = h.excluded_file(tmp_path, [h.ns()])
    root = tmp_path / 'store'
    tokens = [random.Random(7).randrange(1000, 50000) for _ in range(PROMPT)]
    req = c.request('writer', tokens, h.tag(ns))
    keys = [make_offload_key(req.block_hashes[(j + 1) * hpc - 1], g)
            for g, hpc in ((0, 2), (1, 3)) for j in range(len(req.block_hashes) // hpc)]
    mgr, region, tier = c.manager(len(keys), h.tier_config(root, excluded, tmp_path / 'audit0.jsonl'))
    for i, k in enumerate(keys):
        region.array[i] = stored_bytes(k)
    ctx = h.ctx('writer', h.tag(ns))
    tier.on_new_request(ctx)
    job = TransferJob(job_id=1, keys=keys, chunk_ids=np.arange(len(keys), dtype=np.int32), is_promotion=False,
                      req_context=ctx)
    tier.submit_store(job)
    assert h.finish(tier, 1).success
    mgr.shutdown()
    return SimpleNamespace(ns=ns, excluded=excluded, root=root, tokens=tokens, keys=keys, tmp=tmp_path)


def restart(store: Any, monkeypatch: pytest.MonkeyPatch, stream: bool, cpu_chunks: int = 32) -> Any:
    monkeypatch.setattr(sched_mod, '_DSV41_STREAM', stream)
    monkeypatch.setattr(sched_mod, '_DSV41_STREAM_MB', 64.0)
    monkeypatch.setattr(sched_mod, '_DSV41_LOAD_FRACTION', 0.2)
    mgr, region, tier = c.manager(cpu_chunks, h.tier_config(store.root, store.excluded, store.tmp / 'audit1.jsonl'))
    pool = c.GpuPool(512)
    return SimpleNamespace(mgr=mgr, region=region, tier=tier, sched=c.scheduler(mgr, cpu_chunks), pool=pool,
                           workers=[c.worker(region, pool), c.worker(region, c.GpuPool(512))])


def admit(env: Any, req: Any) -> int:
    env.sched.on_new_request(req)
    for _ in range(500):
        n, is_async = env.sched.get_num_new_matched_tokens(req, 0)
        if n is not None:
            break
        c.step(env.sched, env.workers)
    else:
        raise AssertionError('lookup never resolved')
    if n:
        assert is_async
        per_group = [list(range(0, -(-n // 64))), list(range(256, 256 + -(-n // 96)))]
        env.sched.update_state_after_alloc(req, c.Blocks(per_group), n)
        req.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    return int(n)


def run_until_released(env: Any, req_id: str, limit: int = 2000) -> tuple[set[str], set[str], int]:
    for i in range(limit):
        finished, failed = c.step(env.sched, env.workers)
        if req_id in finished:
            return finished, failed, i + 1
    raise AssertionError('request never released')


def test_restart_streams_whole_prefix_through_a_small_cpu_tier(store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    env = restart(store, monkeypatch, stream=True, cpu_chunks=32)
    try:
        req = c.request('reader', store.tokens, h.tag(store.ns))
        n = admit(env, req)
        assert n >= 5760 and n > env.sched._dsv41_load_cap * 10      # not capped by 0081 (cap would be ~384)
        ctl = env.sched._dsv41_stream
        assert ctl is not None and ctl.window_chunks == 3            # 0.2 x 32 chunks -> 6, two windows of 3
        finished, failed, steps = run_until_released(env, 'reader')
        assert 'reader' not in failed
        assert ctl.stats['windows'] >= len(store.keys) // 3 and ctl.stats['completed'] == 1
        # every GPU block of the hit holds exactly the persisted bytes (both groups, sub-block arithmetic)
        for g, (tpb, hpc) in enumerate(((64, 2), (96, 3))):
            for b in range(-(-n // tpb)):
                j, sub = divmod(b, 2)
                key = make_offload_key(req.block_hashes[(j + 1) * hpc - 1], g)
                expect = stored_bytes(key).reshape(2, c.GPU_PAGE)[sub]
                block_id = b if g == 0 else 256 + b
                assert np.array_equal(env.pool.rows[g][block_id], expect), (g, b)
        assert not env.sched._chunks_being_loaded and not ctl.active()
        assert len(store.keys) > 32                                   # the prefix never fit the CPU tier
    finally:
        env.mgr.shutdown()


def test_without_streaming_the_same_restart_is_capped(store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    env = restart(store, monkeypatch, stream=False, cpu_chunks=32)
    try:
        n = admit(env, c.request('reader', store.tokens, h.tag(store.ns)))
        assert 0 < n <= env.sched._dsv41_load_cap == 384
    finally:
        env.mgr.shutdown()


def test_untagged_or_excluded_request_does_not_stream(store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    env = restart(store, monkeypatch, stream=True)
    try:
        assert admit(env, c.request('anon', store.tokens, None)) == 0
        assert not env.sched._dsv41_stream.active()
        excluded_ns = Path(store.excluded[0]).read_text().split()[0]
        assert admit(env, c.request('excl', store.tokens, h.tag(excluded_ns))) == 0
    finally:
        env.mgr.shutdown()


def test_corrupt_chunk_mid_stream_fails_whole_request(store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    env = restart(store, monkeypatch, stream=True)
    try:
        victim = Path(env.tier.path_of(store.ns, store.keys[20]))
        raw = bytearray(victim.read_bytes())
        raw[1] ^= 0x55
        victim.write_bytes(raw)
        env.sched._dsv41_stream.max_miss_steps = 4
        req = c.request('reader', store.tokens, h.tag(store.ns))
        admit(env, req)
        finished, failed, _ = run_until_released(env, 'reader')
        assert 'reader' in failed                                     # miss + recompute, never a partial prefix
        assert env.sched._dsv41_stream.stats['failed'] == 1
        assert not env.sched._chunks_being_loaded and not victim.exists()
    finally:
        env.mgr.shutdown()


def test_abort_mid_stream_releases_once(store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    env = restart(store, monkeypatch, stream=True)
    try:
        req = c.request('reader', store.tokens, h.tag(store.ns))
        admit(env, req)
        for _ in range(6):
            c.step(env.sched, env.workers)
        req.status = RequestStatus.FINISHED_ABORTED
        env.sched.request_finished(req)
        releases = 0
        for i in range(200):
            finished, failed = c.step(env.sched, env.workers, finished_req_ids={'reader'} if i == 0 else None)
            releases += 'reader' in finished
        assert releases == 1 and 'reader' not in env.sched._req_status
        assert not env.sched._dsv41_stream.active() and not env.sched._chunks_being_loaded
    finally:
        env.mgr.shutdown()


def test_connector_reports_failed_receives_from_its_worker() -> None:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import OffloadingConnectorMetadata
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import OffloadingConnector
    conn = OffloadingConnector.__new__(OffloadingConnector)
    want = KVConnectorTransferResults(finished_recving={'r'}, failed_recving={'r'})
    conn.connector_worker = SimpleNamespace(get_transfer_results=lambda ids: want)
    conn._connector_metadata = OffloadingConnectorMetadata(load_jobs={}, store_jobs={})
    assert conn.get_transfer_results(set()) is want


def test_gate_closing_between_lookup_and_allocation_fails_safely(store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The janitor can close the gate between a request's lookup and its allocation: the counted file-tier chunks
    must still go through the stream (which can no longer read them, fails, and recomputes), never a direct load."""
    env = restart(store, monkeypatch, stream=True)
    try:
        req = c.request('reader', store.tokens, h.tag(store.ns))
        env.sched.on_new_request(req)
        for _ in range(500):
            n, _ = env.sched.get_num_new_matched_tokens(req, 0)
            if n is not None:
                break
            c.step(env.sched, env.workers)
        assert n and n > 384
        env.tier.gate.audit = None                      # e.g. the audit log failed: persistence is now OFF
        env.sched._dsv41_stream.max_miss_steps = 3
        per_group = [list(range(0, -(-n // 64))), list(range(256, 256 + -(-n // 96)))]
        env.sched.update_state_after_alloc(req, c.Blocks(per_group), n)
        req.status = RequestStatus.WAITING_FOR_REMOTE_KVS
        finished, failed, _ = run_until_released(env, 'reader')
        assert 'reader' in failed and env.sched._dsv41_stream.stats['failed'] == 1
    finally:
        env.mgr.shutdown()
