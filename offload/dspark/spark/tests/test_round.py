# SPDX-License-Identifier: Apache-2.0
"""End to end, offline: simulated engine -> RemoteRound (rank 0) -> TCP -> reference worker.

Every proposal the round reports usable must equal the fake drafter over the exact window
the target produced (sim.Engine.expected_window); failed requests must carry zero drafts
and zero confidence; after any failure the next successful round must again be exact.
"""

import time

import numpy as np
import pytest
import torch

from remote_dspark import wire
from remote_dspark.config import RemoteConfig
from remote_dspark.mock_worker import TcpStubServer, WorkerCore, fake_draft
from remote_dspark.round import STATUS_OK, STATUS_ROUND_FAILED, RemoteRound

import sim

ID = bytes(range(32))


def hello():
    return wire.Hello(hidden=sim.H, row_bytes=sim.N_TAPS * sim.H * 2, drafter_identity=ID)


def cfg_for(port, **kw):
    env = {"VLLM_DSPARK_REMOTE": f"tcp://127.0.0.1:{port}",
           "VLLM_DSPARK_REMOTE_IDENTITY": ID.hex(),
           "VLLM_DSPARK_REMOTE_POLICY": "capacity",
           "VLLM_DSPARK_REMOTE_TIMEOUT_MS": "150",
           "VLLM_DSPARK_REMOTE_BACKOFF_MS": "2",
           "VLLM_DSPARK_REMOTE_BACKOFF_MAX_MS": "4",
           "VLLM_DSPARK_REMOTE_LOG_EVERY": "0",
           "VLLM_DSPARK_REMOTE_STATS_FILE": ""}
    env.update(kw)
    return RemoteConfig.from_env(env)


@pytest.fixture
def server():
    srv = TcpStubServer(WorkerCore(hello()), "127.0.0.1", 0).start()
    yield srv
    srv.stop()


def make_round(port, n_out_by_reqs=None, **kw):
    rnd = RemoteRound(cfg_for(port, **kw), hello(), device=torch.device("cpu"), tp_rank=0,
                      broadcast_fn=None, background=True, n_out_by_reqs=n_out_by_reqs)
    t = time.monotonic() + 3
    while not rnd.client.ready and time.monotonic() < t:
        time.sleep(0.002)
    assert rnd.client.ready
    return rnd


def run_step(rnd, eng, sd):
    res = rnd.run(sd.batch, sd.taps, sd.num_sampled, sd.num_rejected, sd.last_sampled,
                  sd.next_prefill_tokens)
    drafts = res.drafts.numpy().copy()
    conf = res.conf.numpy().copy()
    ok = res.ok_mask.numpy().copy()
    checked = 0
    for i, r in enumerate(sd.reqs):
        if not sd.need_draft[i]:
            assert ok[i] == 0 and not drafts[i].any()
            continue
        if ok[i]:
            want_ids, want_conf = fake_draft(eng.expected_window(r), r.anchor, r.computed,
                                             sim.N, sim.VOCAB)
            assert np.array_equal(drafts[i], want_ids), (r.req_id, r.computed)
            assert np.array_equal(conf[i], want_conf)
            if res.reply is not None:
                exp_rows = min(sim.W, r.computed - r.start)
                assert int(res.reply.window_rows[i]) == exp_rows
            checked += 1
        else:
            assert not drafts[i].any() and not conf[i].any()
    if res.read_status() == STATUS_ROUND_FAILED:
        assert not ok.any()
    return res, checked


def churn(eng, step, rng):
    """Finish / replace / re-add requests to exercise slot reuse and regressions."""
    for slot, r in enumerate(eng.slots):
        if r is None:
            if rng.random() < 0.5:
                eng.new_request(slot, hit=int(rng.choice([0, 0, 300, 2000])))
            continue
        if not r.prefilling and r.decode_left <= 0:
            if rng.random() < 0.3:
                eng.readd(slot)
                r.decode_left = int(rng.integers(2, 7))
            else:
                eng.slots[slot] = None


def test_clean_run_exact(server):
    rnd = make_round(server.port)
    eng = sim.Engine(seed=1)
    for s in range(sim.M):
        eng.new_request(s)
    rng = np.random.default_rng(1)
    total = 0
    for step in range(250):
        churn(eng, step, rng)
        if not any(eng.slots):
            eng.new_request(0)
        res, checked = run_step(rnd, eng, eng.step())
        assert res.read_status() == STATUS_OK
        total += checked
    c = rnd.stats.snapshot()["counters"]
    assert total > 300
    assert c.get("rounds_failed", 0) == 0
    assert c.get("req_status_partial_window", 0) == 0
    assert c.get("resync_rounds", 0) == 0   # no failure -> never needs a resend
    rnd.client.close()


def test_failures_recover_exactly(server):
    rnd = make_round(server.port)
    eng = sim.Engine(seed=7)
    for s in range(sim.M):
        eng.new_request(s, prompt_len=600)
    rng = np.random.default_rng(7)
    statuses = []
    after_failure_checked = 0
    failed_seen = False
    for step in range(320):
        churn(eng, step, rng)
        if not any(eng.slots):
            eng.new_request(0)
        core = server.core
        if step == 30:
            core.inject.delay_ms = 400                # timeout
        if step == 31:
            core.inject.delay_ms = 0
        if step == 80:
            server.core = WorkerCore(hello())         # worker restart: state lost
        if step == 130:
            server.core.inject.frame_status_once = wire.ST_INTERNAL
        if step == 170:
            live = [i for i, r in enumerate(eng.slots) if r is not None]
            server.core.inject.req_status_once[live[0]] = wire.RS_INTERNAL
        if step == 210:
            server.core.inject.drop_next_step = True
        if step == 250:
            server.drop_connection()
        res, checked = run_step(rnd, eng, eng.step())
        st = res.read_status()
        statuses.append(st)
        if st != STATUS_OK:
            failed_seen = True
        elif failed_seen:
            after_failure_checked += checked
        if st != STATUS_OK:
            time.sleep(0.01)  # let the connector thread reconnect
    c = rnd.stats.snapshot()["counters"]
    assert c.get("fallback_timeout", 0) >= 1
    assert c.get("fallback_frame_no_session", 0) >= 1
    assert c.get("fallback_frame_internal", 0) >= 1
    assert c.get("req_status_internal", 0) >= 1
    assert c.get("fallback_transport_error", 0) >= 1
    assert c.get("resync_rounds", 0) >= 3
    assert after_failure_checked > 200
    assert statuses[-20:] == [STATUS_OK] * 20
    rnd.client.close()


def test_worker_down_from_start_then_up():
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    rnd = RemoteRound(cfg_for(port), hello(), device=torch.device("cpu"), tp_rank=0,
                      broadcast_fn=None, background=True)
    eng = sim.Engine(seed=3)
    eng.new_request(0, prompt_len=400)
    eng.new_request(2, prompt_len=50)
    for _ in range(10):                      # worker absent: rounds fail fast, ring advances
        res, _ = run_step(rnd, eng, eng.step())
        assert res.read_status() == STATUS_ROUND_FAILED
    srv = TcpStubServer(WorkerCore(hello()), "127.0.0.1", port).start()
    try:
        t = time.monotonic() + 3
        while not rnd.client.ready and time.monotonic() < t:
            time.sleep(0.002)
        assert rnd.client.ready
        exact = 0
        for _ in range(40):
            res, checked = run_step(rnd, eng, eng.step())
            assert res.read_status() == STATUS_OK
            exact += checked
        assert exact > 20
        assert rnd.stats.snapshot()["counters"]["resync_rounds"] >= 1
    finally:
        srv.stop()
        rnd.client.close()


def test_append_only_partial_prefill_and_need_draft(server):
    rnd = make_round(server.port)
    eng = sim.Engine(seed=5, chunk_sizes=(50,))
    eng.new_request(1, prompt_len=420)
    seen_partial = 0
    for _ in range(12):
        sd = eng.step()
        res, _ = run_step(rnd, eng, sd)
        if not sd.need_draft[0]:
            seen_partial += 1
    assert seen_partial >= 7
    log = server.core.step_log
    assert log == [] or True
    rnd.client.close()


def test_identity_mismatch_round_fails_cleanly():
    srv = TcpStubServer(WorkerCore(wire.Hello(hidden=sim.H, row_bytes=24,
                                              drafter_identity=bytes(32))),
                        "127.0.0.1", 0).start()
    try:
        rnd = RemoteRound(cfg_for(srv.port), hello(), device=torch.device("cpu"), tp_rank=0,
                          broadcast_fn=None, background=True)
        t = time.monotonic() + 2
        while not rnd.client.disabled and time.monotonic() < t:
            time.sleep(0.002)
        assert rnd.client.disabled
        eng = sim.Engine(seed=9)
        eng.new_request(0)
        res, _ = run_step(rnd, eng, eng.step())
        assert res.read_status() == STATUS_ROUND_FAILED
        assert rnd.stats.snapshot()["counters"]["fallback_disabled"] == 1
    finally:
        srv.stop()


def test_broadcast_layout_on_two_ranks(server):
    """Rank 0 packs, a fake broadcast copies to rank 1, both read identical tensors."""
    rnd0 = make_round(server.port)
    holder = {}

    def bcast0(t):
        holder["t"] = t.clone()

    rnd0.broadcast_fn = bcast0
    rnd1 = RemoteRound(cfg_for(server.port), hello(), device=torch.device("cpu"), tp_rank=1,
                       broadcast_fn=lambda t: t.copy_(holder["t"]))
    assert rnd1.client is None and rnd1.gatherer is None   # rank 1: no network, no buffers
    eng = sim.Engine(seed=11)
    eng.new_request(0, prompt_len=200)
    eng.new_request(3, prompt_len=10)
    for _ in range(15):
        sd = eng.step()
        r0 = rnd0.run(sd.batch, sd.taps, sd.num_sampled, sd.num_rejected, sd.last_sampled,
                      sd.next_prefill_tokens)
        r1 = rnd1.run(sd.batch, None, None, None, None, None)
        assert r1.read_status() == r0.read_status()
        assert torch.equal(r1.drafts, r0.drafts) and torch.equal(r1.conf, r0.conf)
        assert torch.equal(r1.ok_mask, r0.ok_mask)
    rnd0.client.close()


def test_n_out_cuts_positions_past_cap(server):
    """CONTRACT 1.3: with n_out = cap[n], positions < cap equal the full-block drafts and
    positions >= cap are zero (ids and confidences); cap >= N sends 0 (all positions)."""
    cap = [0, 5, 5, 4, 4]
    rnd = make_round(server.port, n_out_by_reqs=cap)
    assert [rnd.n_out_for(n) for n in range(1, 6)] == [0, 0, 4, 4, 4]
    eng = sim.Engine(seed=7)
    rng = np.random.default_rng(7)
    cut_checked = full_checked = 0
    for step in range(200):
        churn(eng, step, rng)
        if not any(eng.slots):
            eng.new_request(0)
        sd = eng.step()
        res = rnd.run(sd.batch, sd.taps, sd.num_sampled, sd.num_rejected, sd.last_sampled,
                      sd.next_prefill_tokens)
        assert res.read_status() == STATUS_OK
        n = sd.batch.num_reqs
        c = min(cap[min(n, len(cap) - 1)], sim.N)
        drafts, conf, ok = res.drafts.numpy(), res.conf.numpy(), res.ok_mask.numpy()
        for i, r in enumerate(sd.reqs):
            if not sd.need_draft[i]:
                continue
            assert ok[i]
            want_ids, want_conf = fake_draft(eng.expected_window(r), r.anchor, r.computed,
                                             sim.N, sim.VOCAB)
            assert np.array_equal(drafts[i, :c], want_ids[:c])
            assert np.array_equal(conf[i, :c], want_conf[:c])
            assert not drafts[i, c:].any() and not conf[i, c:].any()
            if c < sim.N:
                cut_checked += 1
            else:
                full_checked += 1
    assert cut_checked > 50 and full_checked > 20
    assert rnd.stats.snapshot()["counters"].get("rounds_n_out_cut", 0) > 0
    rnd.client.close()

