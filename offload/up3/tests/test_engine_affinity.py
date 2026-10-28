from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from offload.prefill_maxlen.evidence import record
from offload.up3 import engine_affinity as a


def row(tid: int, cpus: frozenset[int] = a.CPUS) -> a.Thread:
    return a.Thread(100, tid, 'stamp'+str(tid), cpus)


def fixture() -> tuple[dict[int, a.Thread], dict[int, a.Thread], a.Births]:
    engine = {i: row(i) for i in range(1, 5)}
    engine[2] = row(2, frozenset({14}))
    proxy = {10: row(10), 11: row(11)}
    births = a.Births()
    births.roles = {'rdma-poller': {2: engine[2]}, 'engram-reconnect': {3: engine[3]}, 'engram-spin': {4: engine[4]}}
    return engine, proxy, births


def test_every_actual_tid_roles_and_rank0_only_proxy() -> None:
    engine, proxy, births = fixture()
    result = a.check_threads(engine, proxy, births)
    assert record(result['all-thread-affinities'])['thread_count'] == 6
    assert record(result['poller14'])['tids'] == [2]
    assert record(result['engram'])['tids'] == [3, 4]
    rank1 = a.check_threads(engine, {}, births, rank=1)
    assert record(rank1['async-proxy'])['present'] is False
    with pytest.raises(ValueError, match='presence'):
        a.check_threads(engine, proxy, births, rank=1)


@pytest.mark.parametrize('kind', ['worker', 'proxy', 'poller', 'reuse', 'births'])
def test_affinity_and_role_failures_are_not_env_approvals(kind: str) -> None:
    engine, proxy, births = fixture()
    if kind == 'worker':
        engine[1] = row(1, frozenset({5}))
    elif kind == 'proxy':
        proxy[10] = row(10, frozenset({14}))
    elif kind == 'poller':
        engine[2] = row(2)
    elif kind == 'reuse':
        engine[3] = a.Thread(100, 3, 'new', a.CPUS)
    else:
        births.roles.pop('engram-spin')
    with pytest.raises(ValueError):
        a.check_threads(engine, proxy, births)


def test_birth_tracking_exact_native_call_and_cleanup() -> None:
    snapshots = iter([{1: row(1)}, {1: row(1), 2: row(2)}])
    births = a.Births(snapshot=lambda: next(snapshots))
    ext = SimpleNamespace(start=lambda: 'started')
    original = ext.start
    try:
        births.watch(ext, 'start', 'rdma-poller', exactly_one=True)
        assert ext.start() == 'started'
        assert set(births.roles['rdma-poller']) == {2}
    finally:
        births.close()
    assert ext.start is original
    ambiguous = iter([{1: row(1)}, {1: row(1), 2: row(2), 3: row(3)}])
    bad = a.Births(snapshot=lambda: next(ambiguous))
    try:
        bad.watch(ext, 'start', 'rdma-poller', exactly_one=True)
        with pytest.raises(ValueError, match='exactly one'):
            ext.start()
    finally:
        bad.close()


def test_read_only_real_local_census() -> None:
    actual = a.census(os.getpid())
    assert os.getpid() in actual
    assert all(t.pid == os.getpid() and t.cpus and t.start.isdecimal() for t in actual.values())
