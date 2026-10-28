"""Shared fixtures for the real-vLLM CPU cases (synthetic namespaces, keys and bytes only)."""
from __future__ import annotations

import hashlib
import mmap
import os
from pathlib import Path
import secrets
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from vllm.v1.kv_offload.base import LookupResult, OffloadKey, ReqContext, make_offload_key
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.tiering.base import ScheduleEndContext, TransferJob

BLOCK = 8192          # bytes per CPU chunk in the fake primary tier (page-aligned for O_DIRECT)
CHUNKS = 64
TIER_MODULE = 'vllm.v1.kv_offload.tiering.fs.dsv41_matter_tier'


def ns() -> str:
    return secrets.token_hex(32)


def key(i: int, group: int = 0, salt: bytes = b'') -> OffloadKey:
    return make_offload_key(hashlib.sha256(salt + i.to_bytes(4, 'big')).digest(), group)


def spec(rank: int = 0) -> Any:
    config = OffloadingConfig(
        groups=(OffloadingGroupConfig(64, ('layer0',), 0), OffloadingGroupConfig(96, ('layer1',), 1)),
        worker_kv_bytes_per_block=BLOCK // 2, enable_kv_cache_events=False, extra_config={}, engine_id='mc1-test',
        model=OffloadingModelConfig('/synthetic/model@rp1-' + 'a' * 64 + '-' + 'b' * 64, 'fp8'),
        cache=OffloadingCacheConfig(tokens_per_hash=64, blocks_per_chunk=2),
        parallel=OffloadingParallelConfig(rank=rank, world_size=2, tp_size=2, pp_size=1, pcp_size=1, dcp_size=1,
                                          data_parallel_index=0, data_parallel_size=1, data_parallel_rank_local=None,
                                          is_parallelism_agnostic=False),
        replicated_layout=True)
    return SimpleNamespace(config=config, blocks_per_chunk=2,
                           kv_events_config=SimpleNamespace(enable_kv_cache_events=False))


class Primary:
    """Page-aligned fake primary tier memory, like the shared CPU region."""

    def __init__(self, chunks: int = CHUNKS) -> None:
        self.mm = mmap.mmap(-1, chunks * BLOCK)
        self.array = np.frombuffer(self.mm, dtype=np.uint8).reshape(chunks, BLOCK)
        self.view = memoryview(self.array)

    def fill(self, chunk_ids: list[int], seed: int) -> None:
        rng = np.random.default_rng(seed)
        for cid in chunk_ids:
            self.array[cid] = rng.integers(0, 256, BLOCK, dtype=np.uint8)


def excluded_file(directory: Path, names: list[str]) -> tuple[str, str]:
    path = directory / 'excluded.txt'
    data = ''.join(f'{n}\n' for n in sorted(set(names))).encode()
    path.write_bytes(data)
    os.chmod(path, 0o600)
    return str(path), hashlib.sha256(data).hexdigest()


def tier_config(root: Path, excluded: tuple[str, str], audit: Path, **extra: Any) -> dict[str, Any]:
    config = {'type': 'Dsv41MatterTier', 'module_path': TIER_MODULE, 'root_dir': str(root),
              'excluded_ns_file': excluded[0], 'excluded_ns_sha256': excluded[1], 'audit_path': str(audit),
              'max_bytes': 1 << 30, 'max_ns_bytes': 1 << 29, 'n_read_threads': 2, 'n_write_threads': 1}
    config.update(extra)
    return config


def make_tier(primary: Primary, config: dict[str, Any]) -> Any:
    from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
    tier = SecondaryTierFactory.create_secondary_tier(config, primary.view, spec())
    assert tier.wait_ready(10.0), 'startup scan did not finish'
    return tier


def ctx(req_id: str, params: dict[str, Any] | None) -> ReqContext:
    return ReqContext(req_id=req_id, kv_transfer_params=params)


def tag(namespace: str, persist: Any = True) -> dict[str, Any]:
    return {'matter_ns': namespace, 'persist': persist}


_next_job = [1000]


def job(keys: list[OffloadKey], chunk_ids: list[int], req: ReqContext, promotion: bool) -> TransferJob:
    _next_job[0] += 1
    return TransferJob(job_id=_next_job[0], keys=keys, chunk_ids=np.array(chunk_ids, dtype=np.int32),
                       is_promotion=promotion, req_context=req)


_finished: dict[int, Any] = {}


def finish(tier: Any, job_id: int, timeout: float = 10.0) -> Any:
    """Wait for one job's JobResult; results of other jobs are kept for their own finish() call."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for result in tier.get_finished_jobs():
            _finished[result.job_id] = result
        if job_id in _finished:
            return _finished.pop(job_id)
        time.sleep(0.005)
    raise AssertionError(f'job {job_id} did not finish')


def resolve(tier: Any, k: OffloadKey, req: ReqContext, timeout: float = 10.0) -> LookupResult:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = tier.lookup(k, req)
        if result is not LookupResult.RETRY:
            return result
        tier.on_schedule_end(ScheduleEndContext(new_req_ids=(), preempted_req_ids=()))
        time.sleep(0.005)
    raise AssertionError('lookup never resolved')


def files_under(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob('*') if p.is_file())


def data_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in files_under(path) if p.suffix == '.bin')
