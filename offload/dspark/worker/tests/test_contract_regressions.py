"""CPU regressions for the contract state machine (DraftCore + a recording fake backend), from the independent check S5
(/opt/dsv41/repos/.worktrees/dsv41-s5/s5-evidence/test_s5_regressions.py) and the whole-branch review."""

import numpy as np
import pytest

from worker import protocol as P
from worker.server import DIRECT_LINK_IPS, DraftCore, FakeBackend, OflServer, TcpStubServer, check_bind

HIDDEN = 4
ROW_BYTES = 3 * HIDDEN * 2
SESSION = 0x5E55


def fake_draft(win, tok, apos, N, vocab):
    return np.array([(tok + j + 1) % vocab for j in range(N)], np.int32), np.full(N, 0.5, np.float32)


def expected() -> P.Hello:
    from worker.server import expected_hello
    e = expected_hello(bytes(32), bytes(32))
    e.hidden, e.row_bytes = HIDDEN, ROW_BYTES
    return e


class RecordingBackend(FakeBackend):
    def __init__(self, exp: P.Hello):
        super().__init__(exp.window, exp.row_bytes, exp.max_slots, fake_draft, exp.vocab_size)
        self.positions: list[int] = []

    def stage(self, slot, first_pos, rows):
        self.positions.append(first_pos)
        super().stage(slot, first_pos, rows)


class Conn:
    def __init__(self):
        self.exp = expected()
        self.core = DraftCore(self.exp, RecordingBackend(self.exp))
        self.seq = 0

    def call(self, op, body=b""):
        self.seq += 1
        rep = self.core.handle(P.AppHeader(P.MAGIC, P.VERSION, op, self.seq, SESSION, len(body), 0, 0).encode() + body)
        return P.AppHeader.decode(rep), rep[P.APP_HDR.size:]

    def hello(self, **over):
        h, _ = self.call(P.OP_HELLO, P.Hello(**{**self.exp.__dict__, **over}).encode())
        return h

    def step(self, *entries):
        rows = b"".join(bytes(e.n_rows * ROW_BYTES) for e in entries)
        h, b = self.call(P.OP_STEP, P.encode_step(self.seq + 1, list(entries), rows, ROW_BYTES))
        assert h.status == P.OK, h.status
        return P.decode_step_reply(b)


def entry(slot, gen, first, n, anchor_pos, flags=P.REQ_NEED_DRAFT, tok=7):
    return P.ReqEntry(slot, flags, 0, gen, first, n, tok, anchor_pos)


def test_stale_generation_rejected_even_with_new_gen():
    c = Conn()
    assert c.hello().status == P.OK
    c.step(entry(0, 5, 0, 3, 3))
    r = c.step(entry(0, 4, 0, 2, 2, P.REQ_NEW_GEN | P.REQ_NEED_DRAFT))
    assert r.statuses[0][0] == P.R_STALE_GENERATION
    assert c.core.slots[0].generation == 5


def test_same_generation_new_gen_resync_and_serial_wrap_still_accepted():
    c = Conn()
    c.hello()
    c.step(entry(0, 5, 0, 3, 3))
    r = c.step(entry(0, 5, 0, 4, 4, P.REQ_NEW_GEN | P.REQ_NEED_DRAFT))         # resync, same generation
    assert r.statuses[0][0] == P.R_OK and c.core.slots[0].rng.lo == 0 and c.core.slots[0].rng.hi == 3
    c.step(entry(1, 0xFFFFFFFF, 0, 2, 2))
    r = c.step(entry(1, 0, 0, 2, 2, P.REQ_NEW_GEN | P.REQ_NEED_DRAFT))        # 0 is newer than 2^32 - 1
    assert r.statuses[0][0] == P.R_OK and c.core.slots[1].generation == 0


def test_position_beyond_model_range_rejected_before_staging():
    c = Conn()
    c.hello()
    first = c.exp.max_model_len + 10000
    r = c.step(entry(0, 1, first, 1, first + 1))
    assert r.statuses[0][0] == P.R_BAD_ENTRY
    assert c.core.backend.positions == []
    assert c.core.slots[0].generation is None                   # slot state untouched


def test_positions_checked_against_the_negotiated_max_model_len():
    c = Conn()
    assert c.hello(max_model_len=4096).status == P.OK
    r = c.step(entry(0, 1, 4090, 6, 4096))                       # context rows 4090..4095, anchor at the limit
    assert r.statuses[0][0] in (P.R_OK, P.R_PARTIAL_WINDOW)
    r = c.step(entry(1, 1, 4095, 2, 4097))
    assert r.statuses[0][0] == P.R_BAD_ENTRY and c.core.backend.positions == [4090]


@pytest.mark.parametrize("field,value", [("max_model_len", 0), ("max_slots", 0), ("max_rows_per_step", 0)])
def test_hello_lower_bounds_refused_without_state_change(field, value):
    c = Conn()
    assert c.hello().status == P.OK
    c.step(entry(0, 3, 0, 2, 2))
    c.seq += 1
    h, _ = c.call(P.OP_HELLO, P.Hello(**{**c.exp.__dict__, field: value}).encode())
    assert h.status == P.CONFIG_MISMATCH
    assert c.core.slots[0].generation == 3 and not c.core.slots[0].rng.empty


def test_failed_step_leaves_no_staged_rows():
    c = Conn()
    c.hello()

    def boom(*a, **k):
        raise RuntimeError("backend failure")
    c.core.backend.draft = boom
    r = c.step(entry(0, 1, 0, 4, 4))
    assert r.statuses[0][0] == P.R_INTERNAL
    assert c.core.backend.pending == [] if hasattr(c.core.backend, "pending") else True


@pytest.mark.parametrize("cls", [OflServer, TcpStubServer])
@pytest.mark.parametrize("ip", ["0.0.0.0", "LAN_IP", "::"])
def test_no_bind_outside_the_direct_links(cls, ip, monkeypatch):
    monkeypatch.setenv("DSPARK_WORKER_ALLOW_LOOPBACK", "1")
    with pytest.raises(ValueError):
        cls(Conn().core, ip, 18699)


def test_direct_link_list():
    assert DIRECT_LINK_IPS == ("10.202.28.1", "10.202.29.1")


def test_loopback_only_with_the_test_override(monkeypatch):
    monkeypatch.delenv("DSPARK_WORKER_ALLOW_LOOPBACK", raising=False)
    with pytest.raises(ValueError):
        check_bind("127.0.0.1")
    monkeypatch.setenv("DSPARK_WORKER_ALLOW_LOOPBACK", "1")
    check_bind("127.0.0.1")
    check_bind("10.202.29.1")
    for ip in ("0.0.0.0", "::", "::1", "localhost", "HEAD_IP", ""):
        with pytest.raises(ValueError):
            check_bind(ip)


def test_dead_backend_sets_fatal_and_stops_drafting():
    c = Conn()
    c.hello()

    def boom(*a, **k):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")
    c.core.backend.draft = boom
    c.core.backend.healthy = lambda: False
    r = c.step(entry(0, 1, 0, 4, 4))
    assert r.statuses[0][0] == P.R_INTERNAL and c.core.fatal.is_set()
    staged = len(c.core.backend.positions)
    h, _ = c.call(P.OP_STEP, P.encode_step(9, [entry(0, 1, 4, 1, 5)], bytes(ROW_BYTES), ROW_BYTES))
    assert h.status == P.INTERNAL and len(c.core.backend.positions) == staged


def test_live_backend_failure_is_not_fatal():
    c = Conn()
    c.hello()

    def boom(*a, **k):
        raise RuntimeError("transient")
    real = c.core.backend.draft
    c.core.backend.draft = boom
    assert c.step(entry(0, 1, 0, 4, 4)).statuses[0][0] == P.R_INTERNAL and not c.core.fatal.is_set()
    c.core.backend.draft = real
    assert c.step(entry(0, 2, 0, 4, 4, P.REQ_NEW_GEN | P.REQ_NEED_DRAFT)).statuses[0][0] == P.R_OK


# S5R fix 2: a dead CUDA context can first surface in any backend call, not only in flush/draft.
def _dead(c, attr):
    def boom(*a, **k):
        raise RuntimeError("CUDA error: an illegal memory access was encountered (scratch backend)")
    setattr(c.core.backend, attr, boom)
    c.core.backend.healthy = lambda: False


@pytest.mark.parametrize("operation", ["step_reset", "hello_reset", "reset_op", "hello_clamp"])
def test_dead_backend_at_every_entry_point_is_fatal(operation):
    c = Conn()
    assert c.hello().status == P.OK
    _dead(c, "set_max_model_len" if operation == "hello_clamp" else "reset")
    if operation == "step_reset":
        h, _ = c.call(P.OP_STEP, P.encode_step(c.seq + 1, [entry(0, 1, 0, 1, 1, P.REQ_NEW_GEN | P.REQ_NEED_DRAFT)],
                                                bytes(ROW_BYTES), ROW_BYTES))
    elif operation == "hello_reset":
        c.seq += 1
        body = P.Hello(**c.exp.__dict__).encode()
        rep = c.core.handle(P.AppHeader(P.MAGIC, P.VERSION, P.OP_HELLO, c.seq, SESSION + 1, len(body), 0, 0).encode()
                            + body)
        h = P.AppHeader.decode(rep)
    elif operation == "reset_op":
        h, _ = c.call(P.OP_RESET, (1).to_bytes(4, "little"))
    else:
        h = c.hello()
    assert h.status == P.INTERNAL
    assert c.core.fatal.is_set(), f"{operation}: dead backend answered INTERNAL but never requested process exit"


def test_health_probe_that_raises_counts_as_dead():
    c = Conn()
    c.hello()
    _dead(c, "reset")

    def probe_raises():
        raise RuntimeError("CUDA error: unspecified launch failure")
    c.core.backend.healthy = probe_raises
    h, _ = c.call(P.OP_RESET, (1).to_bytes(4, "little"))
    assert h.status == P.INTERNAL and c.core.fatal.is_set()


def test_cleanup_failure_cannot_prevent_fatal():
    """The draft fails on a dead context and the recovery's own discard raises too: fatal is still set."""
    c = Conn()
    c.hello()
    _dead(c, "draft")
    _dead(c, "discard")
    h, _ = c.call(P.OP_STEP, P.encode_step(c.seq + 1, [entry(0, 1, 0, 4, 4)], bytes(4 * ROW_BYTES), ROW_BYTES))
    assert h.status == P.INTERNAL and c.core.fatal.is_set()


def test_transient_reset_failure_on_a_healthy_backend_is_not_fatal():
    c = Conn()
    c.hello()
    real = c.core.backend.reset
    calls = {"n": 0}

    def flaky(i):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        real(i)
    c.core.backend.reset = flaky
    h, _ = c.call(P.OP_RESET, (1).to_bytes(4, "little"))
    assert h.status == P.INTERNAL and not c.core.fatal.is_set()
    assert c.step(entry(0, 2, 0, 4, 4, P.REQ_NEW_GEN | P.REQ_NEED_DRAFT)).statuses[0][0] == P.R_OK


def test_after_fatal_hello_and_reset_do_not_touch_the_backend():
    c = Conn()
    c.hello()
    c.core.fatal.set()
    touched = []
    c.core.backend.reset = lambda *a: touched.append("reset")
    c.core.backend.set_max_model_len = lambda *a: touched.append("clamp")
    assert c.hello().status == P.INTERNAL
    assert c.call(P.OP_RESET, (1).to_bytes(4, "little"))[0].status == P.INTERNAL
    assert touched == []


def test_one_million_worker_accepts_shorter_and_full_sessions_and_refuses_beyond():
    """--max-model-len 1048576 (1M engine cap): today's 327,680-token engine and a 1M engine both negotiate; a session
    above the worker's cap is refused; positions past 327,680 are served only when the session announced them."""
    from worker.server import expected_hello
    for session_len, ok in ((327680, True), (1048576, True), (1048577, False)):
        c = Conn()
        c.exp = expected_hello(bytes(32), bytes(32), max_model_len=1048576)
        c.exp.hidden, c.exp.row_bytes = HIDDEN, ROW_BYTES
        c.core = DraftCore(c.exp, RecordingBackend(c.exp))
        status = c.hello(max_model_len=session_len).status
        assert (status == P.OK) == ok, (session_len, status)
        if session_len == 1048576:
            r = c.step(entry(0, 1, 900000, 4, 900004))
            assert r.statuses[0][0] in (P.R_OK, P.R_PARTIAL_WINDOW)
        if session_len == 327680:
            r = c.step(entry(0, 1, 400000, 1, 400001))
            assert r.statuses[0][0] == P.R_BAD_ENTRY


def test_cli_rejects_max_model_len_outside_the_model_range(monkeypatch):
    import sys
    from worker import server
    for bad in ("0", "1048577"):
        monkeypatch.setattr(sys, "argv", ["server", "--ofl", "10.202.28.1:18620", "--max-model-len", bad])
        with pytest.raises(SystemExit) as e:
            server.main()
        assert "max-model-len" in str(e.value)


def test_cli_passes_max_model_len_to_the_tp2_worker(monkeypatch):
    """--tp 2 --max-model-len N reaches the two-card build (lane E's port of the one-card flag)."""
    import sys
    from worker import server
    seen = []

    def fake_tp2(root, graphs, max_model_len):
        seen.append(max_model_len)
        raise SystemExit("built")
    monkeypatch.setattr(server, "build_core_tp2", fake_tp2)
    for argv, want in ((["--tp", "2", "--max-model-len", "1048576"], 1048576), (["--tp", "2"], 327680)):
        monkeypatch.setattr(sys, "argv", ["server", "--ofl", "10.202.28.1:18620"] + argv)
        with pytest.raises(SystemExit) as e:
            server.main()
        assert str(e.value) == "built"
        assert seen[-1] == want
