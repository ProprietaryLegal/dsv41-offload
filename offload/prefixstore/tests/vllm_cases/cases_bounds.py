"""MC1 item 6: LRU byte bounds per namespace and per tier, TTL expiry, purge of excluded namespaces at start, and
load-refreshed recency, through the real tier (CPU only; synthetic namespaces)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc1_helpers as h

from vllm.v1.kv_offload.base import LookupResult

PER_BLOCK = h.BLOCK + 16   # admission charge per chunk (data + sidecar upper bound)


def audit(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def wait_for(cond: Any, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError('condition not reached')
        time.sleep(0.02)


def store(tier: Any, primary: h.Primary, ns: str, keys: list[Any], first_chunk: int = 0) -> Any:
    req = h.ctx(f'r{first_chunk}{ns[:6]}', h.tag(ns))
    tier.on_new_request(req)
    chunks = list(range(first_chunk, first_chunk + len(keys)))
    primary.fill(chunks, seed=first_chunk)
    j = h.job(keys, chunks, req, promotion=False)
    tier.submit_store(j)
    return h.finish(tier, j.job_id)


def age(tier: Any, ns: str, keys: list[Any], seconds_ago: float) -> None:
    t = time.time() - seconds_ago
    for k in keys:
        os.utime(tier.path_of(ns, k), (t, t))


def blocks_on_disk(root: Path, ns: str) -> int:
    return len([p for p in (root / 'ns' / ns).rglob('*.bin')]) if (root / 'ns' / ns).exists() else 0


def test_namespace_bound_evicts_oldest_first(tmp_path: Path) -> None:
    excluded = h.excluded_file(tmp_path, [h.ns()])
    primary = h.Primary()
    tier = h.make_tier(primary, h.tier_config(tmp_path / 's', excluded, tmp_path / 'a.jsonl',
                                              max_bytes=100 * PER_BLOCK, max_ns_bytes=10 * PER_BLOCK,
                                              janitor_interval_s=0.02))
    try:
        ns = h.ns()
        old = [h.key(i, 0, b'old') for i in range(6)]
        assert store(tier, primary, ns, old).success
        for i, k in enumerate(old):                           # least recently used, old[0] the oldest
            age(tier, ns, [k], 3600 - i)
        new = [h.key(i, 0, b'new') for i in range(4)]
        assert store(tier, primary, ns, new, 6).success       # 10 blocks > 0.9 x 10: janitor evicts to <= 8
        wait_for(lambda: any(r['op'] == 'evict' for r in audit(tmp_path / 'a.jsonl')))
        ev = [r for r in audit(tmp_path / 'a.jsonl') if r['op'] == 'evict']
        assert ev[0]['reason'] == 'ns-cap' and ev[0]['ns'] == ns
        assert all(os.path.exists(tier.path_of(ns, k)) for k in new)
        gone = [k for k in old if not os.path.exists(tier.path_of(ns, k))]
        assert gone == old[:len(gone)] and len(gone) == ev[0]['files'] >= 2   # strictly oldest first
        assert tier.usage.ns_bytes(ns) <= 0.8 * 10 * PER_BLOCK
        too_many = [h.key(i, 0, b'big') for i in range(11)]   # a single store over the hard bound is refused
        assert not store(tier, primary, ns, too_many, 20).success
        assert any(r['op'] == 'refuse' and r['reason'] == 'ns-cap' for r in audit(tmp_path / 'a.jsonl'))
    finally:
        tier.shutdown()


def test_tier_bound_evicts_least_recently_used_namespace(tmp_path: Path) -> None:
    excluded = h.excluded_file(tmp_path, [h.ns()])
    primary = h.Primary()
    tier = h.make_tier(primary, h.tier_config(tmp_path / 's', excluded, tmp_path / 'a.jsonl',
                                              max_bytes=12 * PER_BLOCK, max_ns_bytes=12 * PER_BLOCK,
                                              janitor_interval_s=0.02))
    try:
        a, b = h.ns(), h.ns()
        ka = [h.key(i, 0, b'a') for i in range(5)]
        kb = [h.key(i, 0, b'b') for i in range(5)]
        assert store(tier, primary, a, ka).success
        assert store(tier, primary, b, kb, 5).success
        # a hit makes namespace a the most recently used, although b was written later
        rq = h.ctx('hit', h.tag(a))
        tier.on_new_request(rq)
        assert h.resolve(tier, ka[0], rq) is LookupResult.HIT
        j = h.job(ka[:1], [40], rq, promotion=True)
        tier.submit_load(j)
        assert h.finish(tier, j.job_id).success
        kc = [h.key(i, 0, b'c') for i in range(1)]
        assert store(tier, primary, a, kc, 20).success          # 11 blocks > 0.9 x 12: evict from the LRU (b)
        wait_for(lambda: any(r['op'] == 'evict' for r in audit(tmp_path / 'a.jsonl')))
        ev = [r for r in audit(tmp_path / 'a.jsonl') if r['op'] == 'evict']
        assert {r['ns'] for r in ev} == {b} and ev[0]['reason'] == 'tier-cap'
        assert blocks_on_disk(tmp_path / 's', a) == 6 and blocks_on_disk(tmp_path / 's', b) < 5
        assert tier.usage.total <= 0.8 * 12 * PER_BLOCK + PER_BLOCK
    finally:
        tier.shutdown()


def test_ttl_expiry_and_excluded_purge_at_start(tmp_path: Path) -> None:
    excluded_ns, kept, stale_ns = h.ns(), h.ns(), h.ns()
    excluded = h.excluded_file(tmp_path, [excluded_ns])
    primary = h.Primary()
    # A previous run (before the exclusion existed) persisted three namespaces.
    (tmp_path / 'x').mkdir()
    unrestricted = h.excluded_file(tmp_path / 'x', [h.ns()])
    first = h.make_tier(primary, h.tier_config(tmp_path / 's', unrestricted, tmp_path / 'a0.jsonl'))
    try:
        for i, ns in enumerate((excluded_ns, kept, stale_ns)):
            assert store(first, primary, ns, [h.key(j, 0, ns.encode()) for j in range(2)], 4 * i).success
        age(first, stale_ns, [h.key(j, 0, stale_ns.encode()) for j in range(2)], 40 * 86400)
        (tmp_path / 's' / 'ns' / kept / 'leftover.tmp').write_bytes(b'x')
        os.utime(tmp_path / 's' / 'ns' / kept / 'leftover.tmp', (time.time() - 7200, time.time() - 7200))
    finally:
        first.shutdown()
    tier = h.make_tier(primary, h.tier_config(tmp_path / 's', excluded, tmp_path / 'a.jsonl', ttl_days=30))
    try:
        lines = audit(tmp_path / 'a.jsonl')
        purged = [r for r in lines if r['op'] == 'purge']
        assert [(r['ns'], r['reason']) for r in purged] == [(excluded_ns, 'excluded')]
        assert not (tmp_path / 's' / 'ns' / excluded_ns).exists()
        expired = [r for r in lines if r['op'] == 'expire']
        assert {r['ns'] for r in expired if r['files']} == {stale_ns}
        assert blocks_on_disk(tmp_path / 's', stale_ns) == 0 and blocks_on_disk(tmp_path / 's', kept) == 2
        assert not (tmp_path / 's' / 'ns' / kept / 'leftover.tmp').exists()
        assert tier.usage.namespaces() == [kept]
    finally:
        tier.shutdown()


def test_bounds_are_required(tmp_path: Path) -> None:
    excluded = h.excluded_file(tmp_path, [h.ns()])
    config = h.tier_config(tmp_path / 's', excluded, tmp_path / 'a.jsonl')
    for bad in ({'max_bytes': 0}, {'max_ns_bytes': -1}, {'ttl_days': 0}):
        with pytest.raises(ValueError):
            h.make_tier(h.Primary(), {**config, **bad, 'root_dir': str(tmp_path / f's{len(str(bad))}')})
