# SPDX-License-Identifier: Apache-2.0
"""The engine's view of the process proxy's life: stall detection (S11 P-3), respawn (S11 P-4), bounded kill and
shutdown (S11 P-6). CPU: RemoteRound with the CPU stand-in of the hand-off on a real shared region, the real proxy
child process, the reference worker over TCP."""

import json
import os
import signal
import time

import numpy as np
import pytest

import sim
from remote_dspark.mock_worker import TcpStubServer, WorkerCore, fake_draft
from remote_dspark.round import STATUS_ROUND_FAILED

from test_async_process import make_proc_round, proxy_stats
from test_round import hello


@pytest.fixture
def server():
    srv = TcpStubServer(WorkerCore(hello()), "127.0.0.1", 0).start()
    yield srv
    srv.stop()


def _run(rnd, sd):
    return rnd.run(sd.batch, sd.taps, sd.num_sampled, sd.num_rejected, sd.last_sampled, sd.next_prefill_tokens)


def _check(eng, sd, res):
    """(usable drafts that are exact, usable drafts that are wrong) for one round."""
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


def _one_request(seed=5):
    eng = sim.Engine(seed=seed)
    eng.new_request(0, prompt_len=5).decode_left = 10_000
    return eng


def _sigcont(pid):
    try:
        os.kill(pid, signal.SIGCONT)
    except ProcessLookupError:
        pass


# ------------------------------------------------------------------------------------------- P-3 stall detection

def test_Q3_stalled_proxy_is_lost_within_a_few_deadlines(server, tmp_path):
    """S11 Q3 bounded: a SIGSTOPped proxy used to be declared lost only when the 64-slot descriptor ring would overflow,
    every round until then parked for the full deadline (64 x 20 ms here; 64 x 200 ms = 12.8 s in production). Now
    after stall_fires() = 3 watchdog fires with no round finished."""
    rnd = make_proc_round(server.port, tmp_path, timeout_s=3.0, VLLM_DSPARK_REMOTE_ASYNC_WATCHDOG_MS="20",
                          VLLM_DSPARK_REMOTE_TIMEOUT_MS="20")
    eng = _one_request()
    for _ in range(3):
        _run(rnd, eng.step(schedule=[0]))
    pid = rnd.proxy.pid
    os.kill(pid, signal.SIGSTOP)
    try:
        parked, t0 = 0, time.monotonic()
        while rnd._dead is None and parked < 200:
            a = time.monotonic()
            _run(rnd, eng.step(schedule=[0]))
            if rnd._dead is None and time.monotonic() - a >= 0.015:
                parked += 1
        total = time.monotonic() - t0
    finally:
        _sigcont(pid)
    print(f"\nrounds parked for the deadline before 'lost': {parked}, {total:.2f} s; dead: {rnd._dead}")
    assert rnd._dead is not None and "watchdog fires with no round finished" in rnd._dead, rnd._dead
    assert parked <= 3 and total < 0.5, (parked, total)
    assert rnd.stats.snapshot()["counters"]["proxy_lost_fires"] == 1
    rnd.close()


def test_stopped_proxy_is_lost_by_its_heartbeat(server, tmp_path):
    """A proxy stopped while the engine is idle: the first round after the idle stretch is not judged by the heartbeat
    (the engine itself did not run), it parks for one deadline; the next submit finds the heartbeat stale (limit
    max(2 x 100, 150 + 250) = 400 ms) and declares the proxy lost after one fire, not stall_fires() = 4."""
    rnd = make_proc_round(server.port, tmp_path, timeout_s=3.0, VLLM_DSPARK_REMOTE_ASYNC_WATCHDOG_MS="100")
    eng = _one_request()
    for _ in range(3):
        _run(rnd, eng.step(schedule=[0]))
    pid = rnd.proxy.pid
    os.kill(pid, signal.SIGSTOP)
    try:
        time.sleep(0.6)
        assert _run(rnd, eng.step(schedule=[0])).read_status() == STATUS_ROUND_FAILED
        assert rnd._dead is None                             # one parked round, judged at the next submit
        _run(rnd, eng.step(schedule=[0]))
    finally:
        _sigcont(pid)
    assert rnd._dead is not None and "heartbeat" in rnd._dead and "limit 400 ms" in rnd._dead, rnd._dead
    c = rnd.stats.snapshot()["counters"]
    assert c["proxy_lost_heartbeat"] == 1 and "proxy_lost_fires" not in c
    rnd.close()


def test_slow_but_live_proxy_is_not_declared_lost(server, tmp_path):
    """Every STEP takes 60 ms against a 30 ms watchdog: rounds fire, but the proxy finishes rounds (late) and keeps its
    heartbeat, so it is never declared lost (stall_fires() = max(3, ceil(150 / 30) + 2) = 7)."""
    rnd = make_proc_round(server.port, tmp_path, timeout_s=3.0, VLLM_DSPARK_REMOTE_ASYNC_WATCHDOG_MS="30")
    eng = _one_request()
    for _ in range(3):
        _run(rnd, eng.step(schedule=[0]))
    server.core.inject.delay_ms = 60
    for _ in range(30):
        _run(rnd, eng.step(schedule=[0]))
    server.core.inject.delay_ms = 0
    snap = proxy_stats(rnd)
    assert rnd._dead is None and snap["async_watchdog"]["fires"] >= 10, snap["async_watchdog"]
    assert snap["async_stall_detection"] == {"heartbeat_limit_ms": 400, "stall_fires": 7}
    exact = 0
    for _ in range(10):                                      # and it drafts exactly again
        sd = eng.step(schedule=[0])
        e, w = _check(eng, sd, _run(rnd, sd))
        assert w == 0
        exact += e
    assert exact >= 8
    rnd.close()


# ------------------------------------------------------------------------------------------- P-4 respawn

def _expected_partial(eng, r, p0):
    """The window the worker holds after a respawn: the rows gathered since (from p0), within the last W positions."""
    a = r.computed
    return [(p, r.truth[p]) for p in range(max(a - sim.W, p0, r.start), a)]


def _check_since(eng, sd, res, p0):
    """(exact, wrong) for one round, each request's window starting at p0[slot] (a respawn) or its start."""
    drafts, conf, ok = res.drafts.numpy(), res.conf.numpy(), res.ok_mask.numpy()
    exact = wrong = 0
    for i, r in enumerate(sd.reqs):
        if not sd.need_draft[i] or not ok[i]:
            continue
        want_ids, want_conf = fake_draft(_expected_partial(eng, r, p0.get(r.slot, 0)), r.anchor, r.computed,
                                         sim.N, sim.VOCAB)
        if np.array_equal(drafts[i], want_ids) and np.array_equal(conf[i], want_conf):
            exact += 1
        else:
            wrong += 1
    return exact, wrong


def _until_live(rnd, eng, schedule, p0, old_pid, limit_s=8.0):
    """Rounds until a proxy other than old_pid serves (the first submit sees the loss, then target-only rounds);
    p0 = each scheduled slot's first position in the first live round (its window restarts there). Returns the
    number of target-only rounds."""
    t_end = time.monotonic() + limit_s
    dead_rounds = 0
    while rnd._dead is not None or rnd.proxy.pid == old_pid:
        assert time.monotonic() < t_end, f"no respawn within {limit_s} s: {rnd.stats.snapshot().get('async_respawn')}"
        before = {s: eng.slots[s].computed for s in schedule}
        res = _run(rnd, eng.step(schedule=schedule))
        if rnd._dead is None:                  # this round went to the new proxy
            p0.update(before)
        else:
            assert res.read_status() == STATUS_ROUND_FAILED and not res.drafts.any()
            dead_rounds += 1
        time.sleep(0.02)
    return dead_rounds


@pytest.fixture
def quick_respawn(monkeypatch):
    import remote_dspark.round as R
    monkeypatch.setattr(R, "RESPAWN_DELAY_S", 0.1)


def test_killed_proxy_is_respawned_and_drafts_resume_exact(server, tmp_path, quick_respawn):
    """SIGKILL in the middle of a round: the engine sees the exit at the next submit, fails everything far ahead,
    runs target-only rounds, then starts a new proxy (new session: the worker drops its windows, every slot NEW_GEN)
    that continues the round numbering. Every usable draft afterwards is exact: the requests in flight over the
    window gathered since the respawn, a request started afterwards over its whole window."""
    import threading

    rnd = make_proc_round(server.port, tmp_path, timeout_s=3.0, VLLM_DSPARK_REMOTE_ASYNC_WATCHDOG_MS="100")
    eng = sim.Engine(seed=17)
    for s in (0, 1):
        eng.new_request(s, prompt_len=5).decode_left = 10_000
    for _ in range(5):
        sd = eng.step(schedule=[0, 1])
        assert _check(eng, sd, _run(rnd, sd))[1] == 0
    old_pid, rounds_before = rnd.proxy.pid, rnd.rounds
    server.core.inject.delay_ms = 40                          # the proxy is inside the round when it is killed
    threading.Timer(0.015, os.kill, (old_pid, signal.SIGKILL)).start()
    assert _run(rnd, eng.step(schedule=[0, 1])).read_status() == STATUS_ROUND_FAILED     # the watchdog's
    server.core.inject.delay_ms = 0
    p0: dict[int, int] = {}
    dead_rounds = _until_live(rnd, eng, [0, 1], p0, old_pid)
    assert dead_rounds >= 1 and rnd.proxy.pid != old_pid
    assert rnd.rounds == rounds_before + 2                     # target-only rounds are not handed-out rounds
    eng.new_request(2, prompt_len=5).decode_left = 10_000       # starts after the respawn: full window
    exact = {0: 0, 1: 0, 2: 0}
    wrong = 0
    for _ in range(40):
        sd = eng.step(schedule=[0, 1, 2])
        res = _run(rnd, sd)
        e, w = _check_since(eng, sd, res, p0)
        wrong += w
        ok = res.ok_mask.numpy()
        for i, r in enumerate(sd.reqs):
            exact[r.slot] += int(bool(sd.need_draft[i] and ok[i]))
    assert wrong == 0 and min(exact.values()) >= 30, exact
    snap = proxy_stats(rnd)
    c = snap["counters"]
    assert snap["async_dead"] is None and snap["async_respawns"] == 1 and snap["async_respawn"]["left"] == 2
    assert snap["async_proxy"]["pid"] == rnd.proxy.pid and snap["async_proxy"]["generation"] == 1
    assert snap["async_proxy_process"]["start_round"] == rounds_before + 2
    assert [x["kind"] for x in snap["async_proxy_losses"]] == ["exit"] and "-9" in snap["async_proxy_losses"][0]["reason"]
    assert c["proxy_lost_exit"] == 1 and c["fallback_proxy_dead"] == dead_rounds
    assert c["hellos"] >= 2 and c["rounds_ok"] >= 5 + 30      # the dead proxy's counts are carried on
    rnd.close()
    assert rnd.proxy.exit_code() == 0


def test_respawn_limit_reached_then_target_only_with_a_loud_log(server, tmp_path, quick_respawn, caplog):
    rnd = make_proc_round(server.port, tmp_path, timeout_s=3.0, VLLM_DSPARK_REMOTE_ASYNC_RESPAWNS="1")
    eng = _one_request(seed=23)
    for _ in range(3):
        _run(rnd, eng.step(schedule=[0]))
    first = rnd.proxy.pid
    os.kill(first, signal.SIGKILL)
    _until_live(rnd, eng, [0], {}, first)                     # the one respawn
    for _ in range(3):
        _run(rnd, eng.step(schedule=[0]))
    second = rnd.proxy.pid
    os.kill(second, signal.SIGKILL)
    t_end = time.monotonic() + 3
    while rnd.proxy.exit_code() is None and time.monotonic() < t_end:
        time.sleep(0.01)
    with caplog.at_level("ERROR", logger="vllm.remote_dspark"):
        for _ in range(30):                                   # 0.6 s, six respawn delays: none comes
            t0 = time.monotonic()
            res = _run(rnd, eng.step(schedule=[0]))
            assert res.read_status() == STATUS_ROUND_FAILED and time.monotonic() - t0 < 0.2
            time.sleep(0.02)
    assert rnd._dead is not None and rnd._respawn is None and rnd.proxy.pid == second
    assert any("respawn budget is spent" in m for m in caplog.messages), caplog.messages
    st = rnd.stats.snapshot()
    assert st["async_respawn"]["state"] == "exhausted" and st["async_respawns"] == 1
    rnd.write_stats()
    snap = json.load(open(rnd.cfg.stats_file))
    assert snap["async_dead"] == rnd._dead and snap["counters"]["proxy_lost_exit"] == 2
    # the file says so too, not the dead proxy's last copy ("live", the first proxy's pid): S12 F3 / window 5
    assert snap["async_respawn"]["state"] == "exhausted" and snap["async_respawns"] == 1, snap["async_respawn"]
    assert snap["async_proxy"]["pid"] == second and snap["async_proxy"]["generation"] == 1, snap["async_proxy"]
    rnd.close()


def test_no_respawn_before_the_lost_proxy_is_reaped_and_the_stream_drained(server, tmp_path, quick_respawn):
    """The words are reset only when the lost process can no longer write the region and no stream operation of its
    rounds is pending."""
    rnd = make_proc_round(server.port, tmp_path, timeout_s=3.0)
    eng = _one_request(seed=29)
    for _ in range(3):
        _run(rnd, eng.step(schedule=[0]))
    first = rnd.proxy.pid
    os.kill(first, signal.SIGKILL)
    t_end = time.monotonic() + 3
    while rnd.proxy.proc.poll() is None and time.monotonic() < t_end:
        time.sleep(0.005)
    _run(rnd, eng.step(schedule=[0]))
    assert rnd._dead is not None and rnd._respawn is not None
    old = rnd._respawn.old
    real_exit, real_drained = old.exit_code, rnd.sync.drained
    old.exit_code = lambda: None                              # not reaped yet (e.g. uninterruptible sleep)
    for _ in range(10):
        time.sleep(0.03)
        _run(rnd, eng.step(schedule=[0]))
    assert rnd._respawn is not None and rnd._respawn.new is None
    old.exit_code = real_exit
    rnd.sync.drained = lambda marker: False                   # a stream operation of the lost rounds still pending
    for _ in range(10):
        time.sleep(0.03)
        _run(rnd, eng.step(schedule=[0]))
    assert rnd._respawn is not None and rnd._respawn.new is None
    rnd.sync.drained = real_drained
    _until_live(rnd, eng, [0], {}, first)
    assert rnd._dead is None
    rnd.close()


def test_failed_respawn_start_is_retried_within_the_budget(server, tmp_path, quick_respawn, monkeypatch):
    from remote_dspark import proxy_proc

    rnd = make_proc_round(server.port, tmp_path, timeout_s=3.0, VLLM_DSPARK_REMOTE_ASYNC_RESPAWNS="2")
    eng = _one_request(seed=31)
    for _ in range(3):
        _run(rnd, eng.step(schedule=[0]))
    real_boot = proxy_proc.BOOT
    monkeypatch.setattr(proxy_proc, "BOOT", "import sys\nsys.exit(7)\n")    # the first respawn cannot start
    first = rnd.proxy.pid
    os.kill(first, signal.SIGKILL)
    t_end = time.monotonic() + 5
    while rnd._respawn_failures == 0 and time.monotonic() < t_end:
        _run(rnd, eng.step(schedule=[0]))
        time.sleep(0.02)
    assert rnd._respawn_failures == 1 and rnd._dead is not None
    monkeypatch.setattr(proxy_proc, "BOOT", real_boot)
    _until_live(rnd, eng, [0], {}, first)
    st = rnd.stats.snapshot()
    assert st["async_respawns"] == 1 and st["async_respawn"] == {"state": "live", "limit": 2, "left": 0,
                                                                 "failed_starts": 1, "reset_s": 3600}
    rnd.close()


# ------------------------------------------------------------------------------------------- P-6

def test_losing_a_proxy_never_blocks_the_engine_thread(server, tmp_path, quick_respawn):
    """S11 P-6: the engine declared a proxy lost and then waited up to 5 s for its exit on the engine thread (a child
    in uninterruptible sleep keeps that long). Modelled by a reap that takes 3 s: no submit may wait for it. (A 4-slot
    descriptor ring, so that the pre-R3 detection also fires within the test.)"""
    rnd = make_proc_round(server.port, tmp_path, ring=4, timeout_s=3.0, VLLM_DSPARK_REMOTE_ASYNC_WATCHDOG_MS="20",
                          VLLM_DSPARK_REMOTE_TIMEOUT_MS="20")
    eng = _one_request(seed=37)
    for _ in range(3):
        _run(rnd, eng.step(schedule=[0]))
    proc = rnd.proxy.proc
    real_wait = proc.wait

    def slow_wait(timeout=None):
        time.sleep(3.0)
        return real_wait(timeout)

    proc.wait = slow_wait
    pid = rnd.proxy.pid
    os.kill(pid, signal.SIGSTOP)
    try:
        worst = 0.0
        for _ in range(12):
            t0 = time.monotonic()
            _run(rnd, eng.step(schedule=[0]))
            worst = max(worst, time.monotonic() - t0)
    finally:
        _sigcont(pid)
        proc.wait = real_wait
    assert worst < 0.3, worst                  # a parked round (20 ms deadline), never the 3 s reap
    assert rnd._dead is not None or rnd.proxy.pid != pid, rnd.stats.snapshot().get("async_respawn")
    assert rnd.stats.snapshot()["counters"]["proxy_lost_fires"] == 1
    rnd.close()


def test_speculator_closes_its_round_when_dropped(monkeypatch, server):
    """S11 P-6: the model runner drops the speculator at shutdown (no close call); the speculator's finalizer closes
    the round: proxy stopped, every pending round failed far ahead, worker connection closed."""
    import gc

    import vllm_stubs

    vllm_stubs.install()
    import torch

    from remote_dspark import speculator as S
    from test_speculator import ID

    for k, v in {"VLLM_DSPARK_REMOTE": f"tcp://127.0.0.1:{server.port}", "VLLM_DSPARK_REMOTE_IDENTITY": ID.hex(),
                 "VLLM_DSPARK_REMOTE_POLICY": "capacity", "VLLM_DSPARK_REMOTE_LOG_EVERY": "0",
                 "VLLM_DSPARK_REMOTE_STATS_FILE": "", "VLLM_DSPARK_REMOTE_STRICT_START": "1"}.items():
        monkeypatch.setenv(k, v)
    spec = S.init_remote_dspark_speculator(vllm_stubs.vllm_config(), torch.device("cpu"))
    rnd = spec.remote
    closed = []
    real = type(rnd).close
    monkeypatch.setattr(type(rnd), "close", lambda self: (closed.append(self), real(self)))
    assert rnd.client.ready
    del spec
    gc.collect()
    assert closed == [rnd] and rnd.client._stop and not rnd.client.ready


def test_respawns_under_churn_keep_every_draft_exact(server, tmp_path, quick_respawn):
    """The GPU soak's schedule on CPU: requests come, go and are re-added (replayed from below their window), the proxy
    is SIGKILLed twice; every usable draft equals the fake drafter over the window the new proxy has: the rows of the
    request gathered under it, from the lowest first position of its rounds since the respawn."""
    rnd = make_proc_round(server.port, tmp_path, timeout_s=3.0)
    eng = sim.Engine(seed=7)
    for s in range(sim.M):
        eng.new_request(s, prompt_len=400)
    rng = np.random.default_rng(7)
    since: dict[str, int] = {}
    seen: dict[str, int] = {}
    checked = wrong = kills = 0
    for step in range(1000):
        for slot, r in enumerate(eng.slots):                  # the harness's churn, re-adds included
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
        if not any(eng.slots):
            eng.new_request(0)
        if step in (250, 650) and rnd._dead is None:
            os.kill(rnd.proxy.pid, signal.SIGKILL)
            kills += 1
        before = {r.req_id: r.computed for r in eng.slots if r is not None}
        sd = eng.step()
        res = _run(rnd, sd)
        if rnd._dead is None and rnd._generation:
            for r in sd.reqs:
                if seen.get(r.req_id) != rnd._generation:
                    seen[r.req_id], since[r.req_id] = rnd._generation, before[r.req_id]
                else:
                    since[r.req_id] = min(since[r.req_id], before[r.req_id])
        ids, conf, ok = res.drafts.numpy(), res.conf.numpy(), res.ok_mask.numpy()
        for i, r in enumerate(sd.reqs):
            if not sd.need_draft[i] or not ok[i]:
                continue
            win = [(p, b) for p, b in eng.expected_window(r) if p >= since.get(r.req_id, 0)]
            want_ids, want_conf = fake_draft(win, r.anchor, r.computed, sim.N, sim.VOCAB)
            checked += 1
            wrong += not (np.array_equal(ids[i], want_ids) and np.array_equal(conf[i], want_conf))
        if rnd._dead is not None:
            time.sleep(0.002)                                  # target-only rounds take no time on CPU
    assert kills == 2 and rnd._respawn_count == 2 and rnd._dead is None, rnd.stats.snapshot().get("async_respawn")
    assert wrong == 0 and checked > 1500, (wrong, checked)
    rnd.close()


# ------------------------------------------------------------------------------------------- the stats file (S12 F3)

def _read_stats(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def test_stats_file_is_current_only_after_the_new_proxys_mark(server, tmp_path, quick_respawn):
    """S12 F3, the rule window 5 reads the stats by (promotion README): the proxy writes the file only at each LOG_EVERY
    mark and the engine only at a loss, so there is no file before the first mark, and right after a respawn the file
    still shows the loss (the old pid, state waiting) until the new proxy's next mark; after that mark
    async_proxy_process.pid is the new pid and rounds > async_proxy_losses[-1].round. LOG_EVERY 64 for speed (the
    production 1,024 runs the same rule)."""
    every = 64
    rnd = make_proc_round(server.port, tmp_path, timeout_s=3.0, VLLM_DSPARK_REMOTE_LOG_EVERY=str(every))
    path = rnd.cfg.stats_file
    os.remove(path)                                           # make_proc_round's readiness check forced a write
    eng = _one_request(seed=31)
    for _ in range(every - 8):
        _run(rnd, eng.step(schedule=[0]))
    time.sleep(0.3)
    assert _read_stats(path) is None, "no stats file may exist before the first LOG_EVERY mark"
    for _ in range(16):
        _run(rnd, eng.step(schedule=[0]))
    t_end = time.monotonic() + 3
    while _read_stats(path) is None and time.monotonic() < t_end:
        time.sleep(0.02)
    pid1 = rnd.proxy.pid
    s1 = _read_stats(path)
    assert s1 is not None and s1["async_proxy_process"]["pid"] == pid1 and s1["rounds"] >= every, s1
    os.kill(pid1, signal.SIGKILL)
    _until_live(rnd, eng, [0], {}, pid1)
    pid2 = rnd.proxy.pid
    s2 = _read_stats(path)                                    # written by the engine at the loss: stale by design
    loss = s2["async_proxy_losses"][-1]
    assert loss["pid"] == pid1 and s2["async_proxy_process"]["pid"] == pid1, s2
    assert s2["async_respawn"]["state"] == "waiting" and s2["rounds"] == loss["round"], s2
    target = (rnd.rounds // every + 1) * every                # the new proxy's next mark
    while rnd.rounds < target + 2:
        _run(rnd, eng.step(schedule=[0]))
    t_end = time.monotonic() + 3
    while (_read_stats(path) or {}).get("async_proxy_process", {}).get("pid") != pid2 and time.monotonic() < t_end:
        time.sleep(0.02)
    s3 = _read_stats(path)
    print(f"\nloss at round {loss['round']} (pid {pid1}); stale file rounds {s2['rounds']} state "
          f"{s2['async_respawn']['state']}; current after round {target}: pid {s3['async_proxy_process']['pid']}, "
          f"rounds {s3['rounds']}, state {s3['async_respawn']['state']}")
    assert s3["async_proxy_process"]["pid"] == pid2 and s3["async_proxy"]["pid"] == pid2, s3
    assert s3["rounds"] > s3["async_proxy_losses"][-1]["round"] and s3["async_respawn"]["state"] == "live", s3
    assert s3["async_respawns"] == 1 and s3["counters"]["proxy_lost_exit"] == 1, s3
    rnd.close()
