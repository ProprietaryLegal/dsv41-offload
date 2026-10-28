"""MC1: the connector is tier-agnostic. A second persistent tier of a different kind (``mc1_ram_tier``: RAM, no
files, standing in for a later Tier 2 such as x86host) plugs in as one more ``secondary_tiers`` entry and the
unchanged connector scheduler/worker store, stream and refuse through it, alone or next to Tier 1. In multinode
mode a tier that does not declare the matter-gated contract is refused at start (patch 0105). CPU only;
synthetic tokens, namespaces and bytes."""
from __future__ import annotations

import json
from pathlib import Path
import random
import secrets
import sys
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc1_connector as c
import mc1_helpers as h
import mc1_ram_tier
from cases_stream import admit, run_until_released, stored_bytes

import vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler as sched_mod
from vllm.distributed.kv_transfer.kv_connector.v1.offloading import dsv41_stream as ds
from vllm.v1.kv_offload.base import make_offload_key
from vllm.v1.kv_offload.tiering.fs import dsv41_matter as matter

PROMPT = 6000          # 77 chunks: larger than the 32-chunk CPU tier of the restarted engine


def ram_config(store: str, excluded: tuple[str, str], audit: Path) -> dict[str, Any]:
    return {'type': 'RamMatterTier', 'module_path': 'mc1_ram_tier', 'store': store,
            'excluded_ns_file': excluded[0], 'excluded_ns_sha256': excluded[1], 'audit_path': str(audit)}


def prefix_keys(req: Any) -> list[Any]:
    return [make_offload_key(req.block_hashes[(j + 1) * hpc - 1], g)
            for g, hpc in ((0, 2), (1, 3)) for j in range(len(req.block_hashes) // hpc)]


def engine(configs: list[dict[str, Any]], cpu_chunks: int, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(sched_mod, '_DSV41_STREAM', True)
    monkeypatch.setattr(sched_mod, '_DSV41_STREAM_MB', 64.0)
    monkeypatch.setattr(sched_mod, '_DSV41_LOAD_FRACTION', 0.2)
    mgr, region, tiers = c.manager_tiers(cpu_chunks, configs)
    pool = c.GpuPool(512)
    return SimpleNamespace(mgr=mgr, region=region, tiers=tiers, sched=c.scheduler(mgr, cpu_chunks), pool=pool,
                           workers=[c.worker(region, pool), c.worker(region, c.GpuPool(512))])


def offload(env: Any, req_id: str, params: dict[str, Any] | None, keys: list[Any]) -> None:
    """GPU -> CPU primary, then the manager's cascade to every secondary tier (as after a prefill)."""
    ctx = h.ctx(req_id, params)
    env.mgr.on_new_request(ctx)
    out = env.mgr.prepare_store(keys, ctx)
    assert out is not None and list(out.keys_to_store) == keys
    for k, cid in zip(out.keys_to_store, out.store_spec.chunk_ids):
        env.region.array[cid] = stored_bytes(k)
    env.mgr.complete_store(out.keys_to_store, ctx)
    for tier in env.tiers:
        tier.drain_jobs()
    env.mgr.on_request_finished(ctx)


def assert_landed(env: Any, req: Any, n: int) -> None:
    for g, (tpb, hpc) in enumerate(((64, 2), (96, 3))):
        for b in range(-(-n // tpb)):
            j, sub = divmod(b, 2)
            key = make_offload_key(req.block_hashes[(j + 1) * hpc - 1], g)
            expect = stored_bytes(key).reshape(2, c.GPU_PAGE)[sub]
            assert np.array_equal(env.pool.rows[g][b if g == 0 else 256 + b], expect), (g, b)


def ops(path: Path) -> list[tuple[str, str | None]]:
    return [(r['op'], r['ns']) for r in map(json.loads, path.read_text().splitlines())]


def test_tier2_alone_stores_streams_and_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = secrets.token_hex(8)
    excluded_ns, ns = h.ns(), h.ns()
    excluded = h.excluded_file(tmp_path, [excluded_ns])
    config = ram_config(store, excluded, tmp_path / 'tier2.jsonl')
    prompts = {name: [random.Random(seed).randrange(1000, 50000) for _ in range(PROMPT)]
               for name, seed in (('w', 21), ('anon', 23), ('x', 24))}
    tokens = prompts['w']
    env = engine([config], 256, monkeypatch)
    try:
        for req_id, params in (('w', h.tag(ns)), ('anon', None), ('x', h.tag(excluded_ns))):
            offload(env, req_id, params, prefix_keys(c.request(req_id, prompts[req_id], params)))
        ram = env.tiers[0]
        assert ram.entries(ns) == len(prefix_keys(c.request('w', tokens, None))) and ram.entries(excluded_ns) == 0
        assert len(mc1_ram_tier.STORES[store]) == ram.entries(ns)    # untagged: nothing stored
    finally:
        env.mgr.shutdown()
    env = engine([config], 32, monkeypatch)                        # restart: the CPU tier is empty and small
    try:
        req = c.request('reader', tokens, h.tag(ns))
        n = admit(env, req)
        assert n >= 5760 and n > env.sched._dsv41_load_cap          # the whole prefix, past the 0081 cap
        finished, failed, _ = run_until_released(env, 'reader')
        assert 'reader' not in failed and env.sched._dsv41_stream.stats['completed'] == 1
        assert_landed(env, req, n)
        assert admit(env, c.request('anon2', prompts['anon'], None)) == 0    # untagged: nothing persisted
        assert admit(env, c.request('x2', prompts['x'], h.tag(excluded_ns))) == 0
    finally:
        env.mgr.shutdown()
    lines = ops(tmp_path / 'tier2.jsonl')
    assert ('put', ns) in lines and ('get-hit', ns) in lines
    assert ('refuse', excluded_ns) in lines and ('put', excluded_ns) not in lines


def test_tier2_next_to_tier1_serves_what_tier1_lost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = secrets.token_hex(8)
    ns = h.ns()
    excluded = h.excluded_file(tmp_path, [h.ns()])
    root = tmp_path / 'tier1'
    configs = [h.tier_config(root, excluded, tmp_path / 'tier1.jsonl'),
               ram_config(store, excluded, tmp_path / 'tier2.jsonl')]
    tokens = [random.Random(22).randrange(1000, 50000) for _ in range(PROMPT)]
    env = engine(configs, 128, monkeypatch)
    try:
        offload(env, 'w', h.tag(ns), prefix_keys(c.request('w', tokens, None)))
        assert h.data_bytes(root) > 0 and env.tiers[1].entries(ns) > 0    # the cascade wrote both tiers
    finally:
        env.mgr.shutdown()
    matter.purge_namespace(str(root), ns)                             # Tier 1 lost it (e.g. evicted, purged)
    env = engine(configs, 32, monkeypatch)
    try:
        req = c.request('reader', tokens, h.tag(ns))
        n = admit(env, req)
        assert n >= 5760
        finished, failed, _ = run_until_released(env, 'reader')
        assert 'reader' not in failed
        assert_landed(env, req, n)
    finally:
        env.mgr.shutdown()
    assert ('get-hit', ns) in ops(tmp_path / 'tier2.jsonl')
    assert ('get-hit', ns) not in ops(tmp_path / 'tier1.jsonl')


class _UngatedRamTier(mc1_ram_tier.RamMatterTier):
    dsv41_matter_gated: ClassVar[bool] = False


def test_multinode_refuses_a_tier_without_the_gate_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    excluded = h.excluded_file(tmp_path, [h.ns()])
    gated = ram_config(secrets.token_hex(8), excluded, tmp_path / 'a.jsonl')
    mgr, _, tiers = c.manager_tiers(32, [gated, {**gated, 'type': '_UngatedRamTier', 'module_path': __name__}])
    try:
        assert ds.ungated_tiers(tiers) == ['_UngatedRamTier']
        assert ds.ungated_tiers([SimpleNamespace(tier_type='no-allows-get')]) == ['no-allows-get']
        monkeypatch.setenv('VLLM_DSV41_OFFLOAD_MULTINODE', '1')
        with pytest.raises(ValueError, match='not matter-gated'):
            c.scheduler(mgr, 32)
        monkeypatch.setenv('VLLM_DSV41_OFFLOAD_MULTINODE', '0')
        c.scheduler(mgr, 32)                                       # outside production the check is not applied
    finally:
        mgr.shutdown()
    mgr, _, tiers = c.manager_tiers(32, [gated])
    try:
        monkeypatch.setenv('VLLM_DSV41_OFFLOAD_MULTINODE', '1')
        assert ds.ungated_tiers(tiers) == []
        c.scheduler(mgr, 32)
    finally:
        mgr.shutdown()
