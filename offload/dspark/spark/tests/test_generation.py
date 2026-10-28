# SPDX-License-Identifier: Apache-2.0
"""S5 #2: generations (CONTRACT 7.1, 7.2 step 1).

* The reference worker rejects an entry whose generation is older than the slot's, with
  or without NEW_GEN; NEW_GEN with the slot's own generation is still a resync.
* The Spark never sends an older generation for a slot (with NEW_GEN or not): its slot
  generation only grows, also across the u32 wrap, through timeouts, worker restarts,
  frame / request errors and dropped connections.
* If the worker ever reports STALE_GENERATION anyway (its state no longer matches rank
  0's), rank 0 starts a new session, so the worker drops every window, and resyncs.
"""

import time

import numpy as np
import pytest
import torch

from remote_dspark import wire
from remote_dspark.mock_worker import TcpStubServer, WorkerCore
from remote_dspark.round import STATUS_OK, RemoteRound
from remote_dspark.state import SlotTable

import sim
from test_round import cfg_for, churn, hello, run_step

H = wire.Hello(hidden=4, row_bytes=24)


def call(c, op, seq, body=b""):
    return wire.split_frame(c.handle(wire.frame(op, seq, 10, body)))


def step(c, seq, entry):
    rows = np.zeros((entry.n_rows, 24), dtype=np.uint8)
    h, b = call(c, wire.OP_STEP, seq, wire.encode_step(seq, [entry], rows, 24))
    assert h.status == wire.ST_OK
    return wire.StepReply.unpack(b)


# ------------------------------------------------------------------- reference worker


def test_s5_stale_generation_rejected_even_with_new_gen():
    """S5 test_stale_generation_rejected_even_with_new_gen[O3], verbatim."""
    c = WorkerCore(H)
    call(c, wire.OP_HELLO, 1, H.pack())
    step(c, 2, wire.ReqEntry(0, 5, 0, 3, 7, 3, flags=wire.RF_NEED_DRAFT))
    r = step(c, 3, wire.ReqEntry(0, 4, 0, 2, 7, 2, flags=wire.RF_NEW_GEN | wire.RF_NEED_DRAFT))
    assert r.status[0] == wire.RS_STALE_GENERATION
    assert c.slots[0].generation == 5


def test_generation_rules_with_new_gen_and_wrap():
    c = WorkerCore(H)
    call(c, wire.OP_HELLO, 1, H.pack())
    nd, ng = wire.RF_NEED_DRAFT, wire.RF_NEW_GEN | wire.RF_NEED_DRAFT
    r = step(c, 2, wire.ReqEntry(1, 9, 0, 3, 7, 3, flags=nd))
    assert r.status[0] == wire.RS_OK
    # older, with and without NEW_GEN: ignored, window kept
    for seq, f in ((3, ng), (4, nd), (5, ng | wire.RF_RESYNC)):
        r = step(c, seq, wire.ReqEntry(1, 8, 0, 1, 7, 1, flags=f))
        assert r.status[0] == wire.RS_STALE_GENERATION and not r.draft_ids.any()
        assert c.slots[1].generation == 9 and (c.slots[1].rng.lo, c.slots[1].rng.hi) == (0, 2)
    # same generation with NEW_GEN = resync: the window restarts at the new rows
    r = step(c, 6, wire.ReqEntry(1, 9, 100, 4, 7, 104, flags=ng | wire.RF_RESYNC))
    assert r.status[0] == wire.RS_PARTIAL_WINDOW and r.window_rows[0] == 4
    assert (c.slots[1].rng.lo, c.slots[1].rng.hi) == (100, 103)
    # u32 wrap: 0xFFFFFFFF -> 0 is newer, 0 -> 0xFFFFFFFF is older (also with NEW_GEN)
    r = step(c, 7, wire.ReqEntry(2, 0xFFFFFFFF, 0, 2, 7, 2, flags=ng))
    assert r.status[0] == wire.RS_OK
    r = step(c, 8, wire.ReqEntry(2, 0, 0, 2, 7, 2, flags=nd))
    assert r.status[0] == wire.RS_OK and c.slots[2].generation == 0
    r = step(c, 9, wire.ReqEntry(2, 0xFFFFFFFF, 0, 2, 7, 2, flags=ng))
    assert r.status[0] == wire.RS_STALE_GENERATION and c.slots[2].generation == 0
    # RESET forgets the generation: any generation is accepted again
    h, _ = call(c, wire.OP_RESET, 10, wire.RESET_BODY.pack(1 << 2))
    assert h.status == wire.ST_OK
    r = step(c, 11, wire.ReqEntry(2, 0xFFFFFFFF, 0, 2, 7, 2, flags=ng))
    assert r.status[0] == wire.RS_OK


# ------------------------------------------------------------------- Spark side


def test_slot_generation_only_grows_across_wrap():
    t = SlotTable(2, 128)
    t[0].generation = 0xFFFFFFFD
    seen = []
    for k in range(6):
        t.begin_request(0, f"r{k}")
        t.begin_request(0, f"r{k}")          # same owner: unchanged
        t.mark_all_dirty()                   # failures never touch the generation
        seen.append(t[0].generation)
    assert seen == [0xFFFFFFFE, 0xFFFFFFFF, 0, 1, 2, 3]
    assert all(wire.gen_newer(b, a) for a, b in zip(seen, seen[1:]))


@pytest.fixture
def server():
    srv = TcpStubServer(WorkerCore(hello()), "127.0.0.1", 0).start()
    srv.core.keep_step_log = True
    yield srv
    srv.stop()


def make_round(port):
    rnd = RemoteRound(cfg_for(port), hello(), device=torch.device("cpu"), tp_rank=0,
                      broadcast_fn=None, background=True)
    t = time.monotonic() + 3
    while not rnd.client.ready and time.monotonic() < t:
        time.sleep(0.002)
    assert rnd.client.ready
    return rnd


def test_spark_never_sends_an_older_generation(server):
    """Churn (finish, slot reuse, re-add) plus every failure kind, with the slot
    generations starting just below the u32 wrap: the worker never sees a generation
    older than one it already saw for that slot, so it never answers STALE_GENERATION."""
    rnd = make_round(server.port)
    for s in range(sim.M):
        rnd.slots[s].generation = 0xFFFFFFFF - s
    cores = [server.core]
    eng = sim.Engine(seed=13)
    for s in range(sim.M):
        eng.new_request(s, prompt_len=300)
    rng = np.random.default_rng(13)
    new_gen_entries = 0
    for st in range(320):
        churn(eng, st, rng)
        if not any(eng.slots):
            eng.new_request(0)
        if st == 40:
            server.core.inject.delay_ms = 400              # timeout: a late frame lands
        if st == 41:
            server.core.inject.delay_ms = 0
        if st == 90:
            server.core = WorkerCore(hello())            # worker restart
            server.core.keep_step_log = True
            cores.append(server.core)
        if st == 140:
            server.core.inject.frame_status_once = wire.ST_INTERNAL
        if st == 180:
            live = [i for i, r in enumerate(eng.slots) if r is not None]
            server.core.inject.req_status_once[live[0]] = wire.RS_BAD_ENTRY
        if st == 220:
            server.core.inject.drop_next_step = True
        if st == 260:
            server.drop_connection()
        res, _ = run_step(rnd, eng, eng.step())
        if res.read_status() != STATUS_OK:
            time.sleep(0.01)
    wrapped = False
    for core in cores:
        last: dict[int, int] = {}
        for rec in core.step_log:
            s, g = rec["slot"], rec["gen"]
            if s in last:
                assert not wire.gen_newer(last[s], g), (s, last[s], g, rec)
                wrapped |= g < last[s]
            last[s] = g
            new_gen_entries += bool(rec["flags"] & wire.RF_NEW_GEN)
        assert core.counters.values.get("stale_generation", 0) == 0
    c = rnd.stats.snapshot()["counters"]
    assert c.get("req_status_stale_generation", 0) == 0
    assert c.get("stale_generation_new_sessions", 0) == 0
    assert wrapped and new_gen_entries > 20 and c["resync_rounds"] >= 3
    rnd.client.close()


def test_stale_generation_starts_new_session_and_resyncs(server):
    """The worker's slot generation is ahead of rank 0's (state divergence). The worker
    answers STALE_GENERATION; rank 0 starts a new session (the worker drops every window)
    and resyncs; drafts are exact again right after."""
    rnd = make_round(server.port)
    eng = sim.Engine(seed=17)
    for s in range(sim.M):
        eng.new_request(s, prompt_len=200)
    for _ in range(30):
        res, _ = run_step(rnd, eng, eng.step(schedule=list(range(sim.M))))
        assert res.read_status() == STATUS_OK
    session0 = rnd.client.session
    s = 2
    server.core.slots[s].generation = (rnd.slots[s].generation + 7) & wire.U32_MASK
    res, _ = run_step(rnd, eng, eng.step(schedule=list(range(sim.M))))
    assert int(res.ok_mask[s]) == 0 and int(res.ok_mask[0]) == 1   # only slot 2 failed
    t = time.monotonic() + 3
    while not rnd.client.ready and time.monotonic() < t:
        time.sleep(0.002)
    assert rnd.client.ready and rnd.client.session != session0
    exact = 0
    for _ in range(40):
        res, checked = run_step(rnd, eng, eng.step(schedule=list(range(sim.M))))
        assert res.read_status() == STATUS_OK
        exact += checked
    c = rnd.stats.snapshot()["counters"]
    assert c["req_status_stale_generation"] == 1
    assert c["stale_generation_new_sessions"] == 1
    assert c["resync_rounds"] >= 1 and exact >= 150
    assert server.core.counters.values["new_sessions"] == 2
    assert server.core.counters.values.get("stale_generation", 0) == 1
    rnd.client.close()
