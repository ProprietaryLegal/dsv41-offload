"""MC1 item 2: the matter tier through the real tier factory, FileMapper, async lookup, thread pool and the
0056 CRC I/O path (CPU only, synthetic namespaces). Every refusal case asserts that zero bytes reach disk."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc1_helpers as h

from vllm.v1.kv_offload.base import LookupResult
from vllm.v1.kv_offload.tiering.base import ScheduleEndContext


@pytest.fixture()
def env(tmp_path: Path) -> Any:
    excluded_ns = h.ns()
    excluded = h.excluded_file(tmp_path, [excluded_ns, h.ns()])
    root = tmp_path / 'store'
    audit = tmp_path / 'audit' / 'audit.jsonl'
    primary = h.Primary()
    tier = h.make_tier(primary, h.tier_config(root, excluded, audit, janitor_interval_s=0.02, tombstone_s=0.2))
    yield type('Env', (), {'tier': tier, 'root': root, 'audit': audit, 'primary': primary,
                           'excluded_ns': excluded_ns, 'excluded': excluded, 'tmp': tmp_path})
    tier.shutdown()


def audit_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def store(env: Any, req: Any, keys: list[Any], chunks: list[int]) -> Any:
    env.primary.fill(chunks, seed=len(keys) * 7 + chunks[0])
    j = h.job(keys, chunks, req, promotion=False)
    env.tier.submit_store(j)
    return h.finish(env.tier, j.job_id)


def test_tagged_roundtrip_with_crc_and_audit(env: Any) -> None:
    ns = h.ns()
    req = h.ctx('r1', h.tag(ns))
    env.tier.on_new_request(req)
    keys = [h.key(i, g) for i in range(3) for g in (0, 1)]
    chunks = list(range(6))
    result = store(env, req, keys, chunks)
    assert result.success and result.transfer_bytes == 6 * h.BLOCK
    expected = env.primary.array[:6].copy()
    files = h.files_under(env.root / 'ns' / ns)
    assert len([f for f in files if f.suffix == '.bin']) == 6
    assert len([f for f in files if f.name.endswith('.bin.crc')]) == 6
    assert all(f.stat().st_size == h.BLOCK for f in files if f.suffix == '.bin')
    for k in keys:
        assert h.resolve(env.tier, k, req) is LookupResult.HIT
    env.primary.array[:] = 0
    j = h.job(keys, list(range(10, 16)), h.ctx('r2', h.tag(ns)), promotion=True)
    env.tier.submit_load(j)
    loaded = h.finish(env.tier, j.job_id)
    assert loaded.success and loaded.transfer_bytes == 6 * h.BLOCK
    assert np.array_equal(env.primary.array[10:16], expected)
    env.tier.on_request_finished(req)
    ops = [(r['op'], r.get('chunks'), r.get('tokens')) for r in audit_lines(env.audit)]
    assert ('put', 6, 3 * (128 + 192)) in ops and ('get-hit', 6, 3 * (128 + 192)) in ops
    assert ops[0][0] == 'gate'
    text = env.audit.read_text()
    assert 'r1' not in text and 'r2' not in text  # no request ids, only namespaces/counts
    assert os.stat(env.root).st_mode & 0o777 == 0o700


@pytest.mark.parametrize('params', [None, {}, {'matter_ns': 'NOT-HEX', 'persist': True},
                                    'excluded', 'not-persist', 'persist-string'])
def test_refusals_write_zero_bytes(env: Any, params: Any) -> None:
    ns = h.ns()
    if params == 'excluded':
        params = h.tag(env.excluded_ns)
    elif params == 'not-persist':
        params = h.tag(ns, persist=False)
    elif params == 'persist-string':
        params = h.tag(ns, persist='true')
    req = h.ctx('rx', params)
    env.tier.on_new_request(req)
    keys = [h.key(i, 0, b'refuse') for i in range(4)]
    result = store(env, req, keys, list(range(4)))
    assert not result.success and result.transfer_bytes == 0
    assert h.data_bytes(env.root) == 0 and not h.files_under(env.root / 'ns')
    assert env.tier.lookup(keys[0], req) is LookupResult.MISS  # refused before any disk access
    j = h.job(keys, list(range(4)), req, promotion=True)
    env.tier.submit_load(j)
    assert not h.finish(env.tier, j.job_id).success
    refusals = [r for r in audit_lines(env.audit) if r['op'] == 'refuse']
    if isinstance(params, dict) and params.get('matter_ns') == env.excluded_ns:
        assert [r['reason'] for r in refusals] == ['excluded'] and refusals[0]['ns'] == env.excluded_ns
    elif isinstance(params, dict) and params.get('persist') == 'true':
        assert [r['reason'] for r in refusals] == ['bad-tag']
    elif isinstance(params, dict) and params.get('matter_ns') == 'NOT-HEX':
        assert [r['reason'] for r in refusals] == ['bad-tag'] and refusals[0]['ns'] is None
    else:
        assert refusals == []  # untagged / not-persist traffic is never persisted and not audited per request


@pytest.mark.parametrize('breakage', ['missing', 'mode', 'sha', 'malformed', 'audit'])
def test_fail_closed_persists_nothing(tmp_path: Path, breakage: str) -> None:
    excluded = h.excluded_file(tmp_path, [h.ns()])
    path, pin = excluded
    audit = tmp_path / 'audit' / 'audit.jsonl'
    if breakage == 'missing':
        os.unlink(path)
    elif breakage == 'mode':
        os.chmod(path, 0o644)
    elif breakage == 'sha':
        pin = '0' * 64
    elif breakage == 'malformed':
        Path(path).write_bytes(b'zz\n')
        os.chmod(path, 0o600)
        pin = __import__('hashlib').sha256(b'zz\n').hexdigest()
    elif breakage == 'audit':
        audit.parent.mkdir()
        audit.mkdir()  # a directory where the log file must be
    primary = h.Primary()
    tier = h.make_tier(primary, h.tier_config(tmp_path / 'store', (path, pin), audit))
    try:
        assert tier.gate.closed_reason is not None
        ns = h.ns()
        req = h.ctx('r', h.tag(ns))
        tier.on_new_request(req)
        primary.fill([0, 1], 3)
        j = h.job([h.key(0), h.key(1)], [0, 1], req, promotion=False)
        tier.submit_store(j)
        assert not h.finish(tier, j.job_id).success
        assert tier.lookup(h.key(0), req) is LookupResult.MISS
        assert h.data_bytes(tmp_path / 'store') == 0
    finally:
        tier.shutdown()
    if breakage != 'audit':
        lines = audit_lines(audit)
        assert lines[0]['op'] == 'gate' and lines[0]['state'] == 'closed'
        assert any(r['op'] == 'refuse' and r['reason'] == 'gate-closed' for r in lines)


def test_exclusion_file_change_closes_during_run(tmp_path: Path) -> None:
    excluded = h.excluded_file(tmp_path, [h.ns()])
    primary = h.Primary()
    tier = h.make_tier(primary, h.tier_config(tmp_path / 'store', excluded, tmp_path / 'a.jsonl', gate_recheck_s=0.0))
    try:
        ns = h.ns()
        req = h.ctx('r', h.tag(ns))
        tier.on_new_request(req)
        assert tier.gate.closed_reason is None
        Path(excluded[0]).write_bytes(Path(excluded[0]).read_bytes())  # rewritten: identity changes
        os.utime(excluded[0], ns=(5, 5))
        tier.on_schedule_end(ScheduleEndContext(new_req_ids=(), preempted_req_ids=()))
        assert tier.gate.closed_reason == 'changed'
        primary.fill([0], 1)
        j = h.job([h.key(0)], [0], req, promotion=False)
        tier.submit_store(j)
        assert not h.finish(tier, j.job_id).success
        assert h.data_bytes(tmp_path / 'store') == 0
    finally:
        tier.shutdown()


def test_namespaces_are_isolated(env: Any) -> None:
    a, b = h.ns(), h.ns()
    ra, rb = h.ctx('ra', h.tag(a)), h.ctx('rb', h.tag(b))
    for r in (ra, rb):
        env.tier.on_new_request(r)
    shared_key = h.key(42)  # identical content hash in two matters (e.g. no cache salt)
    store(env, ra, [shared_key], [0])
    assert h.resolve(env.tier, shared_key, ra) is LookupResult.HIT
    assert h.resolve(env.tier, shared_key, rb) is LookupResult.MISS  # b never sees a's file
    assert len(h.files_under(env.root / 'ns' / a)) >= 2 and not (env.root / 'ns' / b).exists()


def test_crc_corruption_is_a_miss_and_partial_keep(env: Any) -> None:
    ns = h.ns()
    req = h.ctx('r', h.tag(ns))
    env.tier.on_new_request(req)
    keys = [h.key(i, 0, b'crc') for i in range(3)]
    store(env, req, keys, [0, 1, 2])
    victim = Path(env.tier.path_of(ns, keys[1]))
    raw = bytearray(victim.read_bytes())
    raw[100] ^= 0xFF
    victim.write_bytes(raw)
    j = h.job(keys, [20, 21, 22], h.ctx('r2', h.tag(ns)), promotion=True)
    env.tier.submit_load(j)
    result = h.finish(env.tier, j.job_id)
    assert not result.success and list(result.successful_keys) == keys[:1]
    assert not victim.exists()  # CRC failure removes block + sidecar
    lines = audit_lines(env.audit)
    assert any(r['op'] == 'get-miss' and r['reason'] == 'load-failed' and r['chunks'] == 2 for r in lines)


def test_purge_by_namespace_and_request_file(env: Any) -> None:
    a, b = h.ns(), h.ns()
    ra, rb = h.ctx('ra', h.tag(a)), h.ctx('rb', h.tag(b))
    for r in (ra, rb):
        env.tier.on_new_request(r)
    store(env, ra, [h.key(1, 0, b'a')], [0])
    store(env, rb, [h.key(1, 0, b'b')], [1])
    env.tier.request_purge(a)
    assert env.tier.lookup(h.key(1, 0, b'a'), ra) is LookupResult.MISS  # tombstoned immediately
    deadline = time.monotonic() + 5
    while (env.root / 'ns' / a).exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not (env.root / 'ns' / a).exists() and (env.root / 'ns' / b).exists()
    # online purge request file, as the purge CLI writes it while an engine holds the root lock; the janitor
    # polls for it even while the engine is idle (no scheduler steps)
    (env.root / 'purge-requests' / f'{b}.req').write_text('')
    deadline = time.monotonic() + 5
    while (env.root / 'ns' / b).exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not (env.root / 'ns' / b).exists()
    deadline = time.monotonic() + 5   # tombstone_s=0.2 -> sweep, then the namespace is usable again
    while 'purge-sweep' not in [r['op'] for r in audit_lines(env.audit)] and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.2)
    ops = [r['op'] for r in audit_lines(env.audit)]
    assert ops.count('purge') == 2 and 'purge-sweep' in ops
    rc = h.ctx('rc', h.tag(a))
    env.tier.on_new_request(rc)
    assert store(env, rc, [h.key(9, 0, b'a')], [2]).success


def test_root_is_single_owner_and_events_refused(env: Any) -> None:
    with pytest.raises(ValueError, match='in use'):
        h.make_tier(h.Primary(), h.tier_config(env.root, env.excluded, env.tmp / 'other.jsonl'))
    with pytest.raises(ValueError, match='never emits KV events'):
        h.make_tier(h.Primary(), h.tier_config(env.tmp / 'r2', env.excluded, env.tmp / 'o.jsonl',
                                               enable_kv_events=True))


def test_unscoped_fs_tier_refused_in_multinode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('VLLM_DSV41_OFFLOAD_MULTINODE', '1')
    with pytest.raises(ValueError, match='unscoped fs tier'):
        h.make_tier(h.Primary(), {'type': 'fs', 'root_dir': str(tmp_path / 'fs')})
    assert not (tmp_path / 'fs').exists()
