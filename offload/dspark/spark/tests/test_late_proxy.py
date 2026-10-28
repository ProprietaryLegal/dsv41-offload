# SPDX-License-Identifier: Apache-2.0
"""S11 P-1: a proxy preempted by the watchdog in the middle of a round must never write into the shared STEP staging,
which the engine stream refills with the next round's rows as soon as the watchdog released the round.

The probes are S11's (S11-evidence/test_s11_probes.py, Q1 and its two controls) with the expectation reversed, plus
two stall points S11 did not cover: inside client.step (after the round's last taken() check, where the STEP prefix is
written) and inside the resync-row build of a resync round. Each stall lasts 150 ms against a 100 ms watchdog, so the
round is completed by the watchdog while the proxy is inside it; the next round is one fresh request whose gathered
rows ARE its whole window (NEW_GEN without RESYNC), so a late write into those rows would become the worker's window
for that request and make every later draft of it wrong while the status says OK. Both proxies share RoundCore and are
covered: the process proxy (stall injected into the child interpreter's boot) and the thread proxy (instance patch)."""

import os
import time

import numpy as np
import pytest
import torch

from remote_dspark import proxy_proc, wire
from remote_dspark.mock_worker import TcpStubServer, WorkerCore, fake_draft
from remote_dspark.round import STATUS_ROUND_FAILED, RemoteRound
from remote_dspark.streamsync import CpuHostFlagSync

import sim
from test_async_process import _wait, make_proc_round, proxy_stats
from test_round import cfg_for, hello

S = 6                 # the round the proxy is preempted in
STALL_S = 0.15        # > the 100 ms watchdog
WATCHDOG_MS = "100"

# the proxy's interpreter with one injected stall, chosen by R3_STALL_AT (after_gather: entry of the round, after its
# taken() check; step: inside client.step, before the STEP prefix is written; rows_from_staged: inside the resync-row
# build) at round R3_STALL_ROUND. Models a proxy process descheduled mid-round for longer than the deadline.
STALL_BOOT = proxy_proc.BOOT.replace(
    "from remote_dspark.proxy_proc import main",
    "import os, time\n"
    "import remote_dspark.roundcore as _rc\n"
    "import remote_dspark.client as _cl\n"
    "_W = os.environ.get('R3_STALL_AT', '')\n"
    "_S, _D = int(os.environ.get('R3_STALL_ROUND', '0')), float(os.environ.get('R3_STALL_S', '0'))\n"
    "_cur = [0]\n"
    "_ag = _rc.RoundCore._after_gather\n"
    "def _after_gather(self, bv, g, res, step_id):\n"
    "    _cur[0] = step_id\n"
    "    if _W == 'after_gather' and step_id == _S:\n"
    "        time.sleep(_D)\n"
    "    return _ag(self, bv, g, res, step_id)\n"
    "_rc.RoundCore._after_gather = _after_gather\n"
    "_st = _cl.RemoteDraftClient.step\n"
    "def _step(self, *a, **k):\n"
    "    if _W == 'step' and k.get('step_id') == _S:\n"
    "        time.sleep(_D)\n"
    "    return _st(self, *a, **k)\n"
    "_cl.RemoteDraftClient.step = _step\n"
    "_rs = _rc.RoundCore._rows_from_staged\n"
    "def _rows(self, *a, **k):\n"
    "    if _W == 'rows_from_staged' and _cur[0] == _S:\n"
    "        time.sleep(_D)\n"
    "    return _rs(self, *a, **k)\n"
    "_rc.RoundCore._rows_from_staged = _rows\n"
    "from remote_dspark.proxy_proc import main")
assert STALL_BOOT != proxy_proc.BOOT


@pytest.fixture
def server():
    srv = TcpStubServer(WorkerCore(hello()), "127.0.0.1", 0).start()
    yield srv
    srv.stop()


def _run(rnd, sd):
    return rnd.run(sd.batch, sd.taps, sd.num_sampled, sd.num_rejected, sd.last_sampled, sd.next_prefill_tokens)


def _check(eng, sd, res):
    """(usable drafts that are exact, usable drafts that are WRONG) for one round."""
    drafts, conf, ok = res.drafts.numpy(), res.conf.numpy(), res.ok_mask.numpy()
    exact = wrong = 0
    for i, r in enumerate(sd.reqs):
        if not sd.need_draft[i] or not ok[i]:
            continue
        want_ids, want_conf = fake_draft(eng.expected_window(r), r.anchor, r.computed, sim.N, sim.VOCAB)
        if np.array_equal(drafts[i], want_ids) and np.array_equal(conf[i], want_conf):
            exact += 1
        else:
            wrong += 1
    return exact, wrong


def _proc_round(server, tmp_path, monkeypatch, at: str, stall_round: int = S):
    monkeypatch.setattr(proxy_proc, "BOOT", STALL_BOOT)
    monkeypatch.setenv("R3_STALL_AT", at)
    monkeypatch.setenv("R3_STALL_ROUND", str(stall_round))
    monkeypatch.setenv("R3_STALL_S", str(STALL_S))
    return make_proc_round(server.port, tmp_path, timeout_s=5.0, VLLM_DSPARK_REMOTE_ASYNC_WATCHDOG_MS=WATCHDOG_MS)


def _thread_round(server, monkeypatch, at: str, stall_round: int = S):
    rnd = RemoteRound(cfg_for(server.port, VLLM_DSPARK_REMOTE_ASYNC="1",
                              VLLM_DSPARK_REMOTE_ASYNC_WATCHDOG_MS=WATCHDOG_MS),
                      hello(), device=torch.device("cpu"), tp_rank=0, broadcast_fn=None, background=True,
                      stream_sync=CpuHostFlagSync(timeout_s=5.0))
    assert _wait(lambda: rnd.client.ready)
    cur = [0]
    ag, st, rs = rnd._after_gather, rnd.client.step, rnd._rows_from_staged

    def after_gather(bv, g, res, step_id):
        cur[0] = step_id
        if at == "after_gather" and step_id == stall_round:
            time.sleep(STALL_S)
        return ag(bv, g, res, step_id)

    def step(*a, **k):
        if at == "step" and k.get("step_id") == stall_round:
            time.sleep(STALL_S)
        return st(*a, **k)

    def rows(*a, **k):
        if at == "rows_from_staged" and cur[0] == stall_round:
            time.sleep(STALL_S)
        return rs(*a, **k)

    monkeypatch.setattr(rnd, "_after_gather", after_gather)
    monkeypatch.setattr(rnd.client, "step", step)
    monkeypatch.setattr(rnd, "_rows_from_staged", rows)
    return rnd


def _counters(rnd) -> dict:
    if rnd.proxy is not None:
        return proxy_stats(rnd)["counters"]
    return rnd.stats.snapshot()["counters"]


def _three_then_fresh(rnd, *, resync: bool, server):
    """Rounds 1..S-1 with requests in slots 0-2 (all exact); round S with the same three, preempted mid-round (the
    watchdog fails it); then one fresh request alone for 12 rounds, then all four together for 12 rounds. Returns
    (round S's status, wall time of round S, (exact, wrong) per fresh-only round, (exact, wrong) per joint round).
    resync: round S is a resync round (slot 0 fails on the worker at S-2, round S-1 asks for the windows)."""
    eng = sim.Engine(seed=3)
    for s in range(3):
        eng.new_request(s, prompt_len=5)
        eng.slots[s].decode_left = 10_000
    for t in range(1, S):
        if resync and t == S - 2:
            server.core.inject.req_status_once[0] = wire.RS_INTERNAL
        sd = eng.step(schedule=[0, 1, 2])
        e, w = _check(eng, sd, _run(rnd, sd))
        assert w == 0, t
    sd = eng.step(schedule=[0, 1, 2])
    t0 = time.monotonic()
    res = _run(rnd, sd)
    status, wall = res.read_status(), time.monotonic() - t0
    r3 = eng.new_request(3, prompt_len=5)        # round S+1: a fresh request alone
    r3.decode_left = 10_000
    fresh = []
    for _ in range(12):
        sd = eng.step(schedule=[3])
        fresh.append(_check(eng, sd, _run(rnd, sd)))
    joint = []
    for _ in range(12):                          # slots 0-2 were marked dirty: whole windows are resent
        sd = eng.step(schedule=[0, 1, 2, 3])
        joint.append(_check(eng, sd, _run(rnd, sd)))
    return status, wall, fresh, joint


def _assert_exact_after_preemption(rnd, status, wall, fresh, joint, *, resync: bool, at: str):
    c = _counters(rnd)
    msg = f"fresh {fresh} joint {joint} counters {c}"
    assert status == STATUS_ROUND_FAILED and 0.08 < wall < 0.14, (status, wall)
    assert c.get("watchdog_preempted", 0) >= 1, msg
    # stalled before the round's last taken() check: the late round is not sent at all; stalled inside the send:
    # it is sent (torn rows at worst), its reply is discarded and every slot resends its window
    assert c.get("fallback_preempted", 0) == (0 if at == "step" else 1), msg
    if resync:
        assert c.get("resync_rounds", 0) >= 1 or c.get("fallback_preempted", 0) >= 1, msg
        assert c.get("fallback_resync_pending", 0) >= 1, msg
    # no usable draft computed from a corrupted window, and every fresh-only round drafts exactly
    assert sum(w for _, w in fresh) == 0 and sum(e for e, _ in fresh) == 12, msg
    assert sum(w for _, w in joint) == 0, msg
    assert sum(e for e, _ in joint[-6:]) == 24, msg       # the dirty slots resynced: all four exact again


# ------------------------------------------------------------------------------------------- process proxy

@pytest.mark.parametrize("at", ["after_gather", "step"])
def test_process_late_proxy_never_writes_a_later_rounds_rows(server, tmp_path, monkeypatch, at):
    """S11 Q1 reversed (at=after_gather, S11's stall point) and the same preemption after the round's last taken()
    check (at=step: the old code wrote the STEP prefix into the shared buffer there)."""
    rnd = _proc_round(server, tmp_path, monkeypatch, at)
    try:
        out = _three_then_fresh(rnd, resync=False, server=server)
        _assert_exact_after_preemption(rnd, *out, resync=False, at=at)
    finally:
        rnd.close()


def test_process_late_resync_round_never_writes_a_later_rounds_rows(server, tmp_path, monkeypatch):
    """The preempted round is a resync round: its rows are rebuilt from the staged windows (the old code built them
    in the shared STEP buffer, over the next round's rows)."""
    rnd = _proc_round(server, tmp_path, monkeypatch, "rows_from_staged")
    try:
        out = _three_then_fresh(rnd, resync=True, server=server)
        _assert_exact_after_preemption(rnd, *out, resync=True, at="rows_from_staged")
    finally:
        rnd.close()


def test_process_control_without_the_overlap(server, tmp_path, monkeypatch):
    """S11 Q1 control: the same stall, but the next round has the same n (no prefix overlap): no wrong drafts."""
    rnd = _proc_round(server, tmp_path, monkeypatch, "after_gather")
    try:
        eng = sim.Engine(seed=3)
        for s in range(3):
            eng.new_request(s, prompt_len=5)
            eng.slots[s].decode_left = 10_000
        for _ in range(S):
            _run(rnd, eng.step(schedule=[0, 1, 2]))
        eng.new_request(3, prompt_len=5).decode_left = 10_000
        wrong = 0
        for _ in range(12):
            sd = eng.step(schedule=[1, 2, 3])
            wrong += _check(eng, sd, _run(rnd, sd))[1]
        assert wrong == 0
    finally:
        rnd.close()


def test_process_control_without_the_stall(server, tmp_path, monkeypatch):
    """S11 Q1 control: the same n=3 -> n=1 schedule with no preemption: every draft of the fresh request is exact."""
    rnd = _proc_round(server, tmp_path, monkeypatch, "after_gather", stall_round=0)
    try:
        eng = sim.Engine(seed=3)
        for s in range(3):
            eng.new_request(s, prompt_len=5)
            eng.slots[s].decode_left = 10_000
        for _ in range(S):
            _run(rnd, eng.step(schedule=[0, 1, 2]))
        eng.new_request(3, prompt_len=5).decode_left = 10_000
        after = []
        for _ in range(12):
            sd = eng.step(schedule=[3])
            after.append(_check(eng, sd, _run(rnd, sd)))
        assert sum(w for _, w in after) == 0 and sum(e for e, _ in after) == 12
        assert proxy_stats(rnd)["counters"].get("watchdog_preempted", 0) == 0
    finally:
        rnd.close()


# ------------------------------------------------------------------------------------------- thread proxy

@pytest.mark.parametrize("at", ["after_gather", "step"])
def test_thread_late_proxy_never_writes_a_later_rounds_rows(server, monkeypatch, at):
    """The thread proxy runs the same RoundCore over the engine's pinned staging: same hazard, same fix."""
    rnd = _thread_round(server, monkeypatch, at)
    try:
        out = _three_then_fresh(rnd, resync=False, server=server)
        _assert_exact_after_preemption(rnd, *out, resync=False, at=at)
    finally:
        rnd.close()


def test_thread_late_resync_round_never_writes_a_later_rounds_rows(server, monkeypatch):
    rnd = _thread_round(server, monkeypatch, "rows_from_staged")
    try:
        out = _three_then_fresh(rnd, resync=True, server=server)
        _assert_exact_after_preemption(rnd, *out, resync=True, at="rows_from_staged")
    finally:
        rnd.close()
