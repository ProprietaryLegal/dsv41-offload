"""MC1 item 5: per-block payload hash dump. Layout arithmetic, the real worker's store/load dump points, key
labelling by the real scheduler over a real primary tier, and the restore-vs-in-memory and rank gates end to end
(CPU only; synthetic bytes)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases_stream as cs
import mc1_connector as c
import mc1_helpers as h

import vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler as sched_mod
import vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker as worker_mod
from vllm.distributed.kv_transfer.kv_connector.v1.offloading import dsv41_blockhash as bh
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    TransferJob,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    make_offload_key,
)

COMPARE = Path(__file__).resolve().parents[2] / 'blockhash_compare.py'


def test_layout_matches_handler_arithmetic() -> None:
    keys = ['k0', 'k1', 'k2', 'm0', 'm1']
    got = list(bh.block_layout([5, 3], [3, 0], keys, 2))
    # group 0 starts mid-chunk (block 3 = sub-block 1 of chunk 1): 5 blocks over chunks k0..k2
    assert [(g, lb, k, s) for _, g, lb, k, s in got] == [
        (0, 3, 'k0', 1), (0, 4, 'k1', 0), (0, 5, 'k1', 1), (0, 6, 'k2', 0), (0, 7, 'k2', 1),
        (1, 0, 'm0', 0), (1, 1, 'm0', 1), (1, 2, 'm1', 0)]
    with pytest.raises(ValueError):
        list(bh.block_layout([5, 3], [3, 0], keys[:4], 2))
    with pytest.raises(ValueError):
        list(bh.block_layout([5, 3], [3, 0], keys + ['x'], 2))


def caches(pools: list[np.ndarray]) -> CanonicalKVCaches:
    tensors = [CanonicalKVCacheTensor(torch.from_numpy(p).view(torch.int8), p.shape[1]) for p in pools]
    return CanonicalKVCaches(tensors, [[CanonicalKVCacheRef(i, p.shape[1])] for i, p in enumerate(pools)])


def dump_worker(pool: c.GpuPool, region: c.Region, prefix: str, rank: int) -> Any:
    w = c.worker(region, pool)
    w._dsv41_kv = caches(pool.rows)
    w._dsv41_dump = bh.BlockHashDump(prefix, rank, c.BPC, 1024)
    w._dsv41_dump_loads = {}
    return w


def prefix_store_spec(store: Any, n_tokens: int) -> GPULoadStoreSpec:
    """The store job the in-memory run made for the prefix: blocks 0.. of group 0 and 256.. of group 1."""
    n0, n1 = -(-n_tokens // 64), -(-n_tokens // 96)
    ids = list(range(n0)) + list(range(256, 256 + n1))
    spec = GPULoadStoreSpec(ids, [n0, n1], [0, 0])
    keys0 = [k for k in store.keys if k[-4:] == (0).to_bytes(4, 'big')][: -(-n0 // 2)]
    keys1 = [k for k in store.keys if k[-4:] == (1).to_bytes(4, 'big')][: -(-n1 // 2)]
    setattr(spec, 'dsv41_keys', [k.hex() for k in keys0 + keys1])
    return spec


def fill_in_memory_pool(pool: c.GpuPool, spec: GPULoadStoreSpec) -> None:
    keys = getattr(spec, 'dsv41_keys')
    ids = list(spec.block_ids)
    for pos, g, lb, key, sub in bh.block_layout(spec.group_sizes, spec.block_indices, keys, c.BPC):
        pool.rows[g][ids[pos]] = cs.stored_bytes(bytes.fromhex(key)).reshape(c.BPC, c.GPU_PAGE)[sub]


def run_compare(*args: str) -> dict[str, Any]:
    r = subprocess.run([sys.executable, str(COMPARE), *args], text=True, capture_output=True)
    out: dict[str, Any] = json.loads(r.stdout)
    assert (r.returncode == 0) == out['pass'], r.stderr
    return out


def test_restore_equals_in_memory_on_both_ranks(cs_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = cs_store
    # In-memory run: the prefix's KV sits in each rank's GPU pool; the store job dump hashes it on both ranks.
    region = c.Region(8)
    mem = [c.GpuPool(512), c.GpuPool(512)]
    spec = prefix_store_spec(store, 5888)
    for pool in mem:
        fill_in_memory_pool(pool, spec)
    for rank, pool in enumerate(mem):
        w = dump_worker(pool, region, str(tmp_path / 'A'), rank)
        meta = OffloadingConnectorMetadata(load_jobs={}, store_jobs={7: TransferJob('writer', spec, object())})
        w._is_store_writer = rank == 0
        w.prepare_store_kv(meta)
        w._dsv41_dump.close()
    # Restart + streamed restore: each rank's worker dumps its GPU blocks as every window lands.
    env = cs.restart(store, monkeypatch, stream=True)
    try:
        env.workers = [dump_worker(env.pool, env.region, str(tmp_path / 'B'), 0),
                       dump_worker(c.GpuPool(512), env.region, str(tmp_path / 'B'), 1)]
        monkeypatch.setattr(sched_mod, '_DSV41_HASH_DUMP', str(tmp_path / 'B'))
        req = c.request('reader', store.tokens, h.tag(store.ns))
        assert cs.admit(env, req) == 5888
        cs.run_until_released(env, 'reader')
    finally:
        env.mgr.shutdown()
        for w in env.workers:
            w._dsv41_dump.close()
    for rank in (0, 1):
        assert os.stat(tmp_path / f'B_r{rank}.jsonl').st_mode & 0o777 == 0o600
        out = run_compare('restore', '--in-memory', str(tmp_path / f'A_r{rank}.jsonl'),
                          '--restored', str(tmp_path / f'B_r{rank}.jsonl'))
        assert out['pass'] and out['compared'] == sum(spec.group_sizes) and out['unlabeled'] == 0
    out = run_compare('ranks', '--rank0', str(tmp_path / 'A_r0.jsonl'), '--rank1', str(tmp_path / 'A_r1.jsonl'))
    assert out['pass']
    # A one-byte difference on rank 1 is caught, named by segment.
    mem[1].rows[1][256 + 3][2500] ^= 1
    w = dump_worker(mem[1], region, str(tmp_path / 'C'), 1)
    w.prepare_store_kv(OffloadingConnectorMetadata(load_jobs={}, store_jobs={7: TransferJob('w', spec, object())}))
    w._dsv41_dump.close()
    out = run_compare('ranks', '--rank0', str(tmp_path / 'A_r0.jsonl'), '--rank1', str(tmp_path / 'C_r1.jsonl'))
    assert not out['pass'] and out['differ'] == 1 and out['differing_segments'] == {'2': 1}
    out = run_compare('restore', '--in-memory', str(tmp_path / 'C_r1.jsonl'),
                      '--restored', str(tmp_path / 'B_r1.jsonl'))
    assert not out['pass'] and out['differ'] == 1


@pytest.fixture()
def cs_store(tmp_path: Path) -> Any:
    return cs.make_store(tmp_path)


def test_scheduler_labels_jobs_in_cpu_chunk_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    excluded = h.excluded_file(tmp_path, [h.ns()])
    mgr, region, tier = c.manager(16, h.tier_config(tmp_path / 's', excluded, tmp_path / 'a.jsonl'))
    try:
        sched = c.scheduler(mgr, 16)
        ns = h.ns()
        ctx = h.ctx('r', h.tag(ns))
        mgr.on_new_request(ctx)
        keys = [h.key(i, g) for g in (0, 1) for i in range(3)]
        out = mgr.prepare_store(keys, ctx)
        gpu = GPULoadStoreSpec(list(range(12)), [6, 6], [0, 0])
        meta = OffloadingConnectorMetadata(load_jobs={}, store_jobs={5: TransferJob('r', gpu, out.store_spec)})
        sched._jobs[5] = sched_mod.TransferJobStatus(req_id='r', pending_count=2, keys=set(keys), is_store=True)
        sched._dsv41_label_jobs(meta)
        assert getattr(gpu, 'dsv41_keys') == [k.hex() for k in out.keys_to_store] == [k.hex() for k in keys]
        # A key the primary tier does not hold cannot be labelled: the job stays unlabelled, the engine runs on.
        stray = GPULoadStoreSpec([1, 2], [2, 0], [0, 0])
        meta2 = OffloadingConnectorMetadata(load_jobs={}, store_jobs={6: TransferJob('r', stray, out.store_spec)})
        sched._jobs[6] = sched_mod.TransferJobStatus(req_id='r', pending_count=2, keys={h.key(99)}, is_store=True)
        sched._dsv41_label_jobs(meta2)
        assert getattr(stray, 'dsv41_keys', None) is None
    finally:
        mgr.shutdown()


def test_dump_is_off_by_default() -> None:
    assert worker_mod._DSV41_HASH_DUMP == '' and sched_mod._DSV41_HASH_DUMP == ''
