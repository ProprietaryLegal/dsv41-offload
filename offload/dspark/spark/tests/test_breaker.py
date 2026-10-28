# SPDX-License-Identifier: Apache-2.0
"""S5 #5: the round-count circuit breaker of CONTRACT 10.

After VLLM_DSPARK_REMOTE_TRIP consecutive failed attempts (a STEP that failed, or whose
requests all failed), the next VLLM_DSPARK_REMOTE_COOLDOWN rounds do not use the network
(they fail fast exactly like a round without a connection: zero drafts in capacity, the
local drafter in resilience); then one probe round. A failed probe opens it again; a
successful one closes it, and the probe resyncs every slot it carries.
"""

import time

import numpy as np
import pytest
import torch

from remote_dspark import wire
from remote_dspark.config import RemoteConfig
from remote_dspark.mock_worker import TcpStubServer, WorkerCore
from remote_dspark.round import STATUS_OK, STATUS_ROUND_FAILED, RemoteRound

import sim
from test_round import cfg_for, hello, run_step


# ------------------------------------------------------------------- state machine


def test_breaker_state_machine():
    from remote_dspark.breaker import CircuitBreaker

    b = CircuitBreaker(trip=3, cooldown=4)
    assert b.allow() and b.state == "closed"
    assert b.record(False) is None and b.record(False) is None
    assert b.record(True) is None and b.failures == 0      # a success resets the count
    assert b.record(False) is None and b.record(False) is None
    assert b.record(False) == "tripped" and b.state == "open" and b.trips == 1
    assert [b.allow() for _ in range(4)] == [False] * 4     # exactly `cooldown` rounds
    assert b.state == "half_open" and b.allow()             # the probe
    assert b.record(False) == "tripped" and b.trips == 2    # one failed probe re-opens
    assert [b.allow() for _ in range(5)] == [False] * 4 + [True]
    assert b.record(True) == "recovered" and b.state == "closed" and b.failures == 0
    assert b.allow()


def test_breaker_disabled_and_config():
    from remote_dspark.breaker import CircuitBreaker

    b = CircuitBreaker(trip=0, cooldown=256)
    for _ in range(100):
        assert b.allow()
        assert b.record(False) is None
    base = {"VLLM_DSPARK_REMOTE": "tcp://10.202.28.1:18620",
            "VLLM_DSPARK_REMOTE_IDENTITY": "ab" * 32}
    c = RemoteConfig.from_env(base)
    assert (c.trip, c.cooldown) == (3, 256)
    c = RemoteConfig.from_env({**base, "VLLM_DSPARK_REMOTE_TRIP": "0",
                               "VLLM_DSPARK_REMOTE_COOLDOWN": "1"})
    assert (c.trip, c.cooldown) == (0, 1)
    for kw in ({"VLLM_DSPARK_REMOTE_COOLDOWN": "0"}, {"VLLM_DSPARK_REMOTE_TRIP": "-1"},
               {"VLLM_DSPARK_REMOTE_TRIP": "x"}):
        with pytest.raises(ValueError):
            RemoteConfig.from_env({**base, **kw})


# ------------------------------------------------------------------- rounds


@pytest.fixture
def server():
    srv = TcpStubServer(WorkerCore(hello()), "127.0.0.1", 0).start()
    # these tests drive the reference worker's persistent fault injection; another
    # worker core (e.g. lane O2's run_o3_conformance) has the protocol, not these hooks
    if not hasattr(srv.core.inject, "frame_status_always"):
        srv.stop()
        pytest.skip("needs the reference worker's persistent fault injection")
    yield srv
    srv.stop()


def make(port, **kw):
    rnd = RemoteRound(cfg_for(port, **kw), hello(), device=torch.device("cpu"), tp_rank=0,
                      broadcast_fn=None, background=True)
    wait_ready(rnd)
    return rnd


def wait_ready(rnd, timeout=3.0):
    t = time.monotonic() + timeout
    while not rnd.client.ready and time.monotonic() < t:
        time.sleep(0.002)
    assert rnd.client.ready


def frames(srv):
    return srv.core.counters.values.get("step_frames", 0)


def warm(rnd, eng, rounds=12):
    for _ in range(rounds):
        res, _ = run_step(rnd, eng, eng.step(schedule=[0, 1]))
        assert res.read_status() == STATUS_OK


def engine(seed):
    eng = sim.Engine(seed=seed)
    eng.new_request(0, prompt_len=60)
    eng.new_request(1, prompt_len=90)
    return eng


@pytest.mark.parametrize("fault", ["frame_internal", "every_request_internal"])
def test_persistent_worker_failure_trips_then_probes(server, fault):
    rnd = make(server.port, VLLM_DSPARK_REMOTE_COOLDOWN="10")
    eng = engine(31)
    warm(rnd, eng)
    if fault == "frame_internal":
        server.core.inject.frame_status_always = wire.ST_INTERNAL
    else:
        server.core.inject.req_status_always = wire.RS_INTERNAL
    f0 = frames(server)
    t0 = time.perf_counter()
    for _ in range(25):
        res, _ = run_step(rnd, eng, eng.step(schedule=[0, 1]))
        assert res.read_status() != STATUS_OK and not res.ok_mask.any()
    # 3 failures trip it; 10 rounds without network; a failed probe; 10 more; a probe
    assert frames(server) - f0 == 5
    c = rnd.stats.snapshot()["counters"]
    assert c["breaker_trips"] == 3 and c["fallback_cooldown"] == 20
    assert rnd.stats.snapshot()["breaker"]["state"] == "open"
    # the worker heals: the next probe closes the breaker, its round resyncs, all exact
    server.core.inject.frame_status_always = None
    server.core.inject.req_status_always = None
    statuses = []
    exact = 0
    for _ in range(30):
        res, checked = run_step(rnd, eng, eng.step(schedule=[0, 1]))
        statuses.append(res.read_status())
        exact += checked
    first_ok = statuses.index(STATUS_OK)
    assert first_ok <= 10 and statuses[first_ok:] == [STATUS_OK] * (30 - first_ok)
    assert exact >= 2 * (30 - first_ok)
    snap = rnd.stats.snapshot()
    assert snap["breaker"]["state"] == "closed" and snap["counters"]["resync_rounds"] >= 1
    assert snap["counters"]["breaker_recoveries"] == 1
    rnd.client.close()


def test_persistent_timeouts_trip(server):
    """A worker that accepts HELLO but misses every STEP deadline: without the breaker
    each reconnect costs one full deadline stall; with it, one probe per cooldown."""
    rnd = make(server.port, VLLM_DSPARK_REMOTE_TIMEOUT_MS="30", VLLM_DSPARK_REMOTE_COOLDOWN="12")
    eng = engine(33)
    warm(rnd, eng)
    server.core.inject.delay_ms = 80
    stalls = 0
    for _ in range(3):                       # three timed-out attempts
        wait_ready(rnd)
        t = time.perf_counter()
        res, _ = run_step(rnd, eng, eng.step(schedule=[0, 1]))
        assert res.read_status() == STATUS_ROUND_FAILED
        stalls += time.perf_counter() - t >= 0.025
    assert stalls == 3
    c = rnd.stats.snapshot()["counters"]
    assert c["fallback_timeout"] == 3 and c["breaker_trips"] == 1
    for _ in range(12):                      # open: no deadline is paid, even when connected
        wait_ready(rnd)
        t = time.perf_counter()
        res, _ = run_step(rnd, eng, eng.step(schedule=[0, 1]))
        assert res.read_status() == STATUS_ROUND_FAILED
        assert time.perf_counter() - t < 0.02
    c = rnd.stats.snapshot()["counters"]
    assert c["fallback_timeout"] == 3 and c["fallback_cooldown"] == 12
    server.core.inject.delay_ms = 0
    time.sleep(0.2)                          # the late replies of the dead connections land
    wait_ready(rnd)
    res, checked = run_step(rnd, eng, eng.step(schedule=[0, 1]))
    assert res.read_status() == STATUS_OK and checked == 2
    rnd.client.close()


def test_some_failed_rounds_do_not_trip(server):
    rnd = make(server.port, VLLM_DSPARK_REMOTE_COOLDOWN="10")
    eng = engine(35)
    warm(rnd, eng)
    f0 = frames(server)
    for _ in range(20):
        server.core.inject.req_status_once[0] = wire.RS_INTERNAL   # slot 0 fails every round
        res, _ = run_step(rnd, eng, eng.step(schedule=[0, 1]))
        assert int(res.ok_mask[0]) == 0 and int(res.ok_mask[1]) == 1
    assert frames(server) - f0 == 20
    assert rnd.stats.snapshot()["counters"].get("breaker_trips", 0) == 0
    rnd.client.close()


def test_trip_zero_keeps_retrying_every_round(server):
    rnd = make(server.port, VLLM_DSPARK_REMOTE_TRIP="0")
    eng = engine(37)
    warm(rnd, eng)
    server.core.inject.frame_status_always = wire.ST_INTERNAL
    f0 = frames(server)
    for _ in range(15):
        res, _ = run_step(rnd, eng, eng.step(schedule=[0, 1]))
        assert res.read_status() == STATUS_ROUND_FAILED
    assert frames(server) - f0 == 15
    assert rnd.stats.snapshot()["counters"].get("breaker_trips", 0) == 0
    rnd.client.close()


def test_rounds_without_connection_are_neutral_and_count_down():
    """Rounds while disconnected (no STEP sent) neither trip nor close the breaker, and
    an open breaker's cooldown keeps counting rounds."""
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    rnd = RemoteRound(cfg_for(port, VLLM_DSPARK_REMOTE_COOLDOWN="5"), hello(),  # no worker
                      device=torch.device("cpu"), tp_rank=0, broadcast_fn=None, background=True)
    eng = engine(39)
    for _ in range(20):
        res, _ = run_step(rnd, eng, eng.step(schedule=[0, 1]))
        assert res.read_status() == STATUS_ROUND_FAILED
    c = rnd.stats.snapshot()["counters"]
    assert c["fallback_not_ready"] == 20 and c.get("breaker_trips", 0) == 0
    rnd.breaker.record(False), rnd.breaker.record(False), rnd.breaker.record(False)
    assert rnd.breaker.state == "open"
    for _ in range(5):
        run_step(rnd, eng, eng.step(schedule=[0, 1]))
    assert rnd.breaker.state == "half_open"
    run_step(rnd, eng, eng.step(schedule=[0, 1]))   # probe round, but not connected: neutral
    assert rnd.breaker.state == "half_open"
    rnd.client.close()
