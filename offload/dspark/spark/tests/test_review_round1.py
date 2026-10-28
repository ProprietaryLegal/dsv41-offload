# SPDX-License-Identifier: Apache-2.0
"""Findings of the independent review of the O3-FIX round (all low severity), as
regressions: the epoch/ready snapshot of a round, new_session in one critical section, a
HELLO abandoned for a newer session retried at once, append-only rounds neutral for the
breaker, and fixed:<d> needing no cost inputs."""

import time

import numpy as np
import pytest
import torch

from remote_dspark import wire
from remote_dspark.client import RemoteDraftClient
from remote_dspark.config import RemoteConfig, apply_cost_overrides
from remote_dspark.mock_worker import TcpStubServer, WorkerCore
from remote_dspark.round import STATUS_OK, STATUS_ROUND_FAILED, RemoteRound

import sim
from test_round import cfg_for, hello, run_step


@pytest.fixture
def server():
    srv = TcpStubServer(WorkerCore(hello()), "127.0.0.1", 0).start()
    if not hasattr(srv.core.inject, "req_status_always"):
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


def test_round_never_sends_on_a_connection_newer_than_it_checked(server):
    """The round reads (ready, epoch) once, atomically, and STEP refuses a connection from a
    later epoch: a reconnect between the epoch check and the send can never carry entries
    that were not resynced for the new connection."""
    rnd = make(server.port)
    eng = sim.Engine(seed=41)
    eng.new_request(0, prompt_len=80)
    eng.new_request(1, prompt_len=80)
    for _ in range(10):
        res, _ = run_step(rnd, eng, eng.step(schedule=[0, 1]))
        assert res.read_status() == STATUS_OK
    rnd.client.new_session("test: the worker forgets every window")
    wait_ready(rnd)
    stale = rnd._last_epoch                  # what a round that checked before the HELLO saw
    real = rnd.client.connection_epoch
    rnd.client.connection_epoch = lambda: stale
    f0 = frames(server)
    res, _ = run_step(rnd, eng, eng.step(schedule=[0, 1]))
    rnd.client.connection_epoch = real
    assert res.read_status() == STATUS_ROUND_FAILED and frames(server) == f0
    c = rnd.stats.snapshot()["counters"]
    assert c["fallback_reconnected"] == 1 and c.get("breaker_trips", 0) == 0
    exact = 0
    for _ in range(10):
        res, checked = run_step(rnd, eng, eng.step(schedule=[0, 1]))
        assert res.read_status() == STATUS_OK
        exact += checked
    c = rnd.stats.snapshot()["counters"]
    assert exact == 20 and c.get("req_status_partial_window", 0) == 0
    rnd.client.close()


class CountingLock:
    def __init__(self, lock):
        self.lock, self.entered = lock, 0

    def __enter__(self):
        self.lock.acquire()
        self.entered += 1
        return self

    def __exit__(self, *exc):
        self.lock.release()

    def locked(self):
        return self.lock.locked()


def test_new_session_drops_connection_in_the_same_critical_section(server):
    """No reader can see the new session id with the old connection: the id changes and
    the connection is dropped under one acquisition of the client lock."""
    cli = RemoteDraftClient(cfg_for(server.port), hello(), background=False)
    assert cli.connect_now()
    s0 = cli.session
    lock = cli._lock = CountingLock(cli._lock)
    t = time.monotonic()
    cli.new_session("test")
    assert lock.entered == 1
    assert cli.session not in (s0, 0) and not cli.ready and cli._client is None
    assert t <= cli._next_attempt <= time.monotonic()        # reconnect at once, no backoff
    assert cli.connect_now() and cli.ready and cli.epoch == 2
    cli.close()


def test_hello_for_an_old_session_is_retried_at_once():
    """A session rotation during connect + HELLO: the HELLO is abandoned (never installed)
    and redone under the new session at once, with no reconnect backoff."""
    class Worker:
        calls = 0

        def connect(self, timeout):
            pass

        def request(self, msg, buf, timeout_s):
            h, _ = wire.split_frame(msg)
            Worker.calls += 1
            if Worker.calls == 1:
                cli.session = (cli.session + 1) & wire.U32_MASK or 1   # rotated meanwhile
            rep = wire.HelloReply(drafter_identity=cli.hello.drafter_identity,
                                  markov_identity=cli.hello.markov_identity)
            raw = wire.frame(wire.OP_HELLO, h.seq, h.session, rep.pack())
            buf[:len(raw)] = raw
            return len(raw)

        def close(self):
            pass

    cli = RemoteDraftClient(cfg_for(1), hello(), background=False,
                            transport_factory=lambda ep: Worker())
    assert cli.connect_now()
    assert Worker.calls == 2 and cli.ready and cli.epoch == 1
    assert cli.stats.snapshot()["counters"].get("connect_failures", 0) == 0


def test_append_only_rounds_neither_reset_nor_close_the_breaker(server):
    """Rounds where no request needs drafts (prefill chunks) do not exercise drafting: their
    success is neutral. Decode failures interleaved with them still trip the breaker, and a
    half-open breaker is closed only by a round that drafted."""
    rnd = make(server.port, VLLM_DSPARK_REMOTE_COOLDOWN="4")
    eng = sim.Engine(seed=43, chunk_sizes=(64,))
    eng.new_request(0, prompt_len=60)
    for _ in range(4):
        run_step(rnd, eng, eng.step(schedule=[0]))
    eng.new_request(1, prompt_len=4000)          # ~62 append-only prefill rounds
    server.core.inject.req_status_always = wire.RS_INTERNAL
    for k in range(3):                           # decode fails, prefill appends "succeed"
        res, _ = run_step(rnd, eng, eng.step(schedule=[0]))
        assert res.read_status() != STATUS_OK
        if k < 2:
            sd = eng.step(schedule=[1])
            assert not any(sd.need_draft)
            run_step(rnd, eng, sd)
    c = rnd.stats.snapshot()["counters"]
    assert c["breaker_trips"] == 1 and rnd.breaker.state == "open"
    server.core.inject.req_status_always = None
    for _ in range(4):
        run_step(rnd, eng, eng.step(schedule=[1]))
    assert rnd.breaker.state == "half_open"
    sd = eng.step(schedule=[1])                  # append-only probe: worker fine, no drafting
    assert not any(sd.need_draft)
    run_step(rnd, eng, sd)
    assert rnd.breaker.state == "half_open"
    res, checked = run_step(rnd, eng, eng.step(schedule=[0]))   # a drafting probe closes it
    assert res.read_status() == STATUS_OK and checked == 1
    assert rnd.breaker.state == "closed"
    snap = rnd.stats.snapshot()
    assert snap["counters"]["breaker_recoveries"] == 1
    rnd.write_stats()
    assert rnd.stats.snapshot()["breaker"]["state"] == "closed"
    rnd.client.close()


def test_breaker_stats_are_current_when_written(server, tmp_path):
    f = tmp_path / "stats.json"
    rnd = make(server.port, VLLM_DSPARK_REMOTE_COOLDOWN="50",
               VLLM_DSPARK_REMOTE_STATS_FILE=str(f))
    eng = sim.Engine(seed=45)
    eng.new_request(0, prompt_len=40)
    for _ in range(4):
        run_step(rnd, eng, eng.step(schedule=[0]))
    server.core.inject.frame_status_always = wire.ST_INTERNAL
    for _ in range(3 + 7):
        run_step(rnd, eng, eng.step(schedule=[0]))
    rnd.write_stats()
    import json
    b = json.loads(f.read_text())["breaker"]
    assert b["state"] == "open" and b["cooldown_left"] == 50 - 7
    rnd.client.close()


def test_fixed_depth_needs_no_cost_inputs_unless_a_depth_file_can_switch_it():
    e = {"VLLM_DSPARK_REMOTE": "tcp://10.202.28.1:18620", "VLLM_DSPARK_REMOTE_IDENTITY": "ab" * 32,
         "VLLM_DSPARK_REMOTE_POLICY": "capacity"}
    c = RemoteConfig.from_env(e)
    assert apply_cost_overrides(c, dict(e), depth_policy_on=True, depth_mode="fixed") == {}
    with pytest.raises(ValueError, match="REMOTE_DRAFT_MS"):
        apply_cost_overrides(c, dict(e), depth_policy_on=True, depth_mode="fixed",
                             depth_file="/tmp/depth")
    for mode in ("conf", "confb", None):
        with pytest.raises(ValueError, match="REMOTE_DRAFT_MS"):
            apply_cost_overrides(c, dict(e), depth_policy_on=True, depth_mode=mode)
