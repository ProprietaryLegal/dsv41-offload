"""Real OffloadingConnectorScheduler / OffloadingConnectorWorker on CPU around a real TieringOffloadingManager,
CPU primary tier (page-aligned fake shared region) and the matter tier. Only the GPU copy is simulated: a load
job "lands" by copying the pinned CPU rows into a fake GPU block pool with the same chunk/block arithmetic the
CPU->GPU handler uses (one GPU block = 1/blocks_per_chunk of a CPU chunk)."""
from __future__ import annotations

import mmap
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc1_helpers as h

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import OffloadingWorkerMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import OffloadingConnectorScheduler
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import OffloadingConnectorWorker
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.kv_offload.tiering.manager import CPUPrimaryTierOffloadingManager, TieringOffloadingManager
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

import torch

TOKENS_PER_BLOCK = (64, 96)   # two full-attention groups -> 128 / 192 tokens per chunk at blocks_per_chunk 2
BPC = 2
HASH = 64
GPU_PAGE = h.BLOCK // BPC
init_none_hash(sha256)


class Region:
    """Stand-in for SharedOffloadRegion: page-aligned memory, viewed as (chunks, chunk bytes)."""

    def __init__(self, chunks: int) -> None:
        self.mm = mmap.mmap(-1, chunks * h.BLOCK)
        self.array = np.frombuffer(self.mm, dtype=np.uint8).reshape(chunks, h.BLOCK)

    def create_kv_memoryview(self) -> memoryview:
        return memoryview(self.array)

    def cleanup(self) -> None:
        """The real region unmaps /dev/shm; this one is anonymous memory, freed with the object."""


def manager(cpu_chunks: int, tier_config: dict[str, Any]) -> tuple[TieringOffloadingManager, Region, Any]:
    mgr, region, tiers = manager_tiers(cpu_chunks, [tier_config])
    return mgr, region, tiers[0]


def manager_tiers(cpu_chunks: int, tier_configs: list[dict[str, Any]]
                  ) -> tuple[TieringOffloadingManager, Region, list[Any]]:
    """A real tiering manager whose secondary tiers come from config dicts via SecondaryTierFactory, in order
    (exactly how the connector builds them from kv_connector_extra_config.secondary_tiers)."""
    region = Region(cpu_chunks)
    primary = CPUPrimaryTierOffloadingManager(cpu_chunks, region)
    from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
    tiers = [SecondaryTierFactory.create_secondary_tier(cfg, primary.get_kv_memoryview(), h.spec())
             for cfg in tier_configs]
    for tier in tiers:
        wait_ready = getattr(tier, 'wait_ready', None)
        assert wait_ready is None or wait_ready(10.0), 'startup scan did not finish'
    return TieringOffloadingManager(primary, tiers), region, tiers


def kv_cache_config(num_blocks: int) -> KVCacheConfig:
    groups = [KVCacheGroupSpec([f'layer{i}'], FullAttentionSpec(block_size=tpb, num_kv_heads=1, head_size=64,
                                                                   dtype=torch.int8))
              for i, tpb in enumerate(TOKENS_PER_BLOCK)]
    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups)


def scheduler(mgr: TieringOffloadingManager, cpu_chunks: int, num_blocks: int = 4096) -> OffloadingConnectorScheduler:
    spec = SimpleNamespace(
        config=h.spec().config, blocks_per_chunk=BPC, tokens_per_hash=HASH, tokens_per_block=TOKENS_PER_BLOCK,
        offload_prompt_only=True, num_chunks=cpu_chunks, get_manager=lambda: mgr,
        kv_events_config=SimpleNamespace(enable_kv_cache_events=False, self_describing_kv_events=False))
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(prefix_cache_retention_interval=None, enable_prefix_caching=True),
        speculative_config=None, parallel_config=SimpleNamespace(world_size=2, decode_context_parallel_size=1))
    return OffloadingConnectorScheduler(spec, vllm_config, kv_cache_config(num_blocks))


def request(req_id: str, tokens: list[int], params: dict[str, Any] | None) -> Request:
    sp = SamplingParams(max_tokens=1, extra_args={'kv_transfer_params': params} if params else None)
    return Request(req_id, tokens, sp, None, block_hasher=get_request_block_hasher(HASH, sha256))


class Block:
    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.is_null = False
        self.block_hash = None


class Blocks:
    """KVCacheBlocks stand-in: per group, the request's blocks in order (fresh, unhashed)."""

    def __init__(self, per_group: list[list[int]]) -> None:
        self.blocks = tuple([Block(b) for b in ids] for ids in per_group)


class GpuPool:
    """Fake GPU KV pool per group: rows of GPU_PAGE bytes, filled by landing load jobs."""

    def __init__(self, num_blocks: int) -> None:
        self.rows = [np.zeros((num_blocks, GPU_PAGE), dtype=np.uint8) for _ in TOKENS_PER_BLOCK]

    def land(self, region: Region, src_chunks: np.ndarray, dst: Any) -> None:
        """Same arithmetic as the CPU->GPU handler: group g's GPU blocks map onto consecutive CPU chunk
        sub-blocks, the first chunk skipping block_indices[g] % blocks_per_chunk sub-blocks."""
        ids = list(dst.block_ids)
        src_off = dst_off = 0
        for g, (size, start) in enumerate(zip(dst.group_sizes, dst.block_indices)):
            if size == 0:
                continue
            skip = start % BPC
            n_chunks = -(-(size + skip) // BPC)
            chunks = src_chunks[src_off:src_off + n_chunks]
            for i in range(size):
                sub = skip + i
                row = region.array[chunks[sub // BPC]].reshape(BPC, GPU_PAGE)[sub % BPC]
                self.rows[g][ids[dst_off + i]] = row
            src_off += n_chunks
            dst_off += size
        assert src_off == len(src_chunks) and dst_off == len(ids)


class FakeHandlerWorker:
    """OffloadingWorker stand-in: submit_load lands synchronously, get_finished reports it."""

    def __init__(self, region: Region, pool: GpuPool) -> None:
        self.region, self.pool = region, pool
        self.done: list[Any] = []

    def submit_load(self, job_id: int, src: Any, dst: Any) -> bool:
        self.pool.land(self.region, np.asarray(src.block_ids), dst)
        self.done.append(SimpleNamespace(job_id=job_id, success=True, transfer_time=0.001,
                                         transfer_size=len(src.block_ids) * h.BLOCK))
        return True

    def submit_store(self, *args: Any) -> bool:
        raise AssertionError('no stores in these cases')

    def get_finished(self) -> list[Any]:
        out, self.done = self.done, []
        return out


def worker(region: Region, pool: GpuPool) -> OffloadingConnectorWorker:
    w = OffloadingConnectorWorker.__new__(OffloadingConnectorWorker)
    w.worker = FakeHandlerWorker(region, pool)
    w._mn_group = None
    w._load_jobs = {}
    w._unsubmitted_store_jobs = []
    w._connector_worker_meta = OffloadingWorkerMetadata()
    w._dsv41_window_jobs = set()
    w._dsv41_failed = set()
    w._dsv41_failed_out = set()
    w._dsv41_kv = None
    w._dsv41_dump = None
    w._dsv41_dump_loads = {}
    w._is_store_writer = True
    return w


def step(sched: OffloadingConnectorScheduler, workers: list[OffloadingConnectorWorker],
         finished_req_ids: set[str] | None = None) -> tuple[set[str], set[str]]:
    """One engine step: build metadata, run it on every worker, feed the aggregated output back."""
    out = SimpleNamespace(scheduled_new_reqs=[], preempted_req_ids=set(), finished_req_ids=finished_req_ids or set(),
                          scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[], resumed_req_ids=set()),
                          num_scheduled_tokens={}, kv_cache_block_copies=None, kv_connector_block_state=None)
    meta = sched.build_connector_meta(out)
    finished: set[str] = set()
    failed: set[str] = set()
    merged = OffloadingWorkerMetadata()
    for w in workers:
        w.start_kv_transfers(meta)
        results = w.get_transfer_results(set())
        finished |= results.finished_recving
        failed |= results.failed_recving
        m = w.build_connector_worker_meta()
        if m is not None:
            merged = merged.aggregate(m)
    sched.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=merged))
    time.sleep(0.002)
    return finished, failed
