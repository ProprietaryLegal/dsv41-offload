"""MC1: requests that do not opt in (untagged, ``persist`` false, or an excluded namespace) never reach the disk but
keep the normal in-memory reuse: their chunks land in the CPU primary tier and a later request with the same prefix
loads them from there through the real connector scheduler/worker (CPU only; synthetic tokens, namespaces, bytes)."""
from __future__ import annotations

import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc1_connector as c
import mc1_helpers as h
from cases_stream import admit, run_until_released, stored_bytes

import vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler as sched_mod
from vllm.v1.kv_offload.base import make_offload_key

PROMPT = 1920          # 15 + 10 chunks: fits the 64-chunk CPU tier whole
HIT = PROMPT // 384 * 384          # hits are 384-token aligned (lcm of the two groups' chunk sizes)
CPU_CHUNKS = 64


def prefix_keys(req: Any) -> list[Any]:
    return [make_offload_key(req.block_hashes[(j + 1) * hpc - 1], g)
            for g, hpc in ((0, 2), (1, 3)) for j in range(len(req.block_hashes) // hpc)]


def engine(tmp_path: Path, excluded: tuple[str, str], monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(sched_mod, '_DSV41_STREAM', True)
    monkeypatch.setattr(sched_mod, '_DSV41_STREAM_MB', 64.0)
    monkeypatch.setattr(sched_mod, '_DSV41_LOAD_FRACTION', 0.2)
    mgr, region, tier = c.manager(CPU_CHUNKS, h.tier_config(tmp_path / 'store', excluded, tmp_path / 'audit.jsonl'))
    pool = c.GpuPool(512)
    return SimpleNamespace(mgr=mgr, region=region, tier=tier, sched=c.scheduler(mgr, CPU_CHUNKS), pool=pool,
                           workers=[c.worker(region, pool), c.worker(region, c.GpuPool(512))])


def offload(env: Any, req_id: str, params: dict[str, Any] | None, keys: list[Any]) -> None:
    """What the connector does after a prefill: GPU -> CPU primary, then the cascade to every secondary tier."""
    ctx = h.ctx(req_id, params)
    env.mgr.on_new_request(ctx)
    out = env.mgr.prepare_store(keys, ctx)
    assert out is not None and list(out.keys_to_store) == keys
    for k, cid in zip(out.keys_to_store, out.store_spec.chunk_ids):
        env.region.array[cid] = stored_bytes(k)
    env.mgr.complete_store(out.keys_to_store, ctx)
    env.tier.drain_jobs()
    env.mgr.on_request_finished(ctx)


@pytest.mark.parametrize('kind', ['untagged', 'not-persist', 'excluded'])
def test_non_persisting_request_keeps_cpu_reuse_and_writes_nothing(tmp_path: Path, kind: str,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    excluded_ns = h.ns()
    excluded = h.excluded_file(tmp_path, [excluded_ns])
    params = {'untagged': None, 'not-persist': h.tag(h.ns(), persist=False), 'excluded': h.tag(excluded_ns)}[kind]
    tokens = [random.Random(11).randrange(1000, 50000) for _ in range(PROMPT)]
    env = engine(tmp_path, excluded, monkeypatch)
    try:
        keys = prefix_keys(c.request('writer', tokens, params))
        offload(env, 'writer', params, keys)
        req = c.request('reader', tokens, params)
        n = admit(env, req)
        cap = env.sched._dsv41_load_cap
        assert n == min(cap, HIT) > 0          # served from the CPU tier, 0081 cap as before MC1
        assert not env.sched._dsv41_stream.active()                # never streamed: nothing persistent to serve
        finished, failed, _ = run_until_released(env, 'reader')
        assert 'reader' not in failed
        for g, (tpb, hpc) in enumerate(((64, 2), (96, 3))):
            for b in range(-(-n // tpb)):
                j, sub = divmod(b, 2)
                key = make_offload_key(req.block_hashes[(j + 1) * hpc - 1], g)
                expect = stored_bytes(key).reshape(2, c.GPU_PAGE)[sub]
                assert np.array_equal(env.pool.rows[g][b if g == 0 else 256 + b], expect), (g, b)
        assert h.data_bytes(tmp_path / 'store') == 0               # not one byte reached the disk
        ops = [json.loads(line)['op'] for line in (tmp_path / 'audit.jsonl').read_text().splitlines()]
        assert 'put' not in ops and 'get-hit' not in ops
        # one refusal line per excluded request (writer, reader); untagged / not-persist traffic is only counted
        assert ops.count('refuse') == (2 if kind == 'excluded' else 0)
    finally:
        env.mgr.shutdown()
    # Restart: the CPU tier is gone and nothing was persisted, so the same request recomputes.
    env = engine(tmp_path, excluded, monkeypatch)
    try:
        assert admit(env, c.request('again', tokens, params)) == 0
    finally:
        env.mgr.shutdown()


def test_tagged_request_persists_and_restores_after_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The contrast case: the same offload with ``persist: true`` survives the restart."""
    excluded = h.excluded_file(tmp_path, [h.ns()])
    params = h.tag(h.ns())
    tokens = [random.Random(12).randrange(1000, 50000) for _ in range(PROMPT)]
    env = engine(tmp_path, excluded, monkeypatch)
    try:
        offload(env, 'writer', params, prefix_keys(c.request('writer', tokens, params)))
        assert h.data_bytes(tmp_path / 'store') > 0
    finally:
        env.mgr.shutdown()
    env = engine(tmp_path, excluded, monkeypatch)
    try:
        n = admit(env, c.request('reader', tokens, params))
        assert n == HIT > env.sched._dsv41_load_cap                  # whole stored prefix, streamed past the cap
        finished, failed, _ = run_until_released(env, 'reader')
        assert 'reader' not in failed and env.sched._dsv41_stream.stats['completed'] == 1
    finally:
        env.mgr.shutdown()
