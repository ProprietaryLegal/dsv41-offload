# SPDX-License-Identifier: Apache-2.0
"""RemoteDraftClient against the reference worker over the TCP stub (127.0.0.1)."""

import time

import numpy as np
import pytest

from remote_dspark import wire
from remote_dspark.client import RemoteDraftClient
from remote_dspark.config import RemoteConfig
from remote_dspark.mock_worker import TcpStubServer, WorkerCore

ID = bytes(range(32))
H = 4
RB = 3 * H * 2


def hello(**kw):
    base = dict(hidden=H, row_bytes=RB, drafter_identity=ID)
    base.update(kw)
    return wire.Hello(**base)


def cfg_for(port, **kw):
    env = {"VLLM_DSPARK_REMOTE": f"tcp://127.0.0.1:{port}",
           "VLLM_DSPARK_REMOTE_IDENTITY": ID.hex(),
           "VLLM_DSPARK_REMOTE_TIMEOUT_MS": "200",
           "VLLM_DSPARK_REMOTE_BACKOFF_MS": "5",
           "VLLM_DSPARK_REMOTE_BACKOFF_MAX_MS": "20",
           "VLLM_DSPARK_REMOTE_CONNECT_TIMEOUT_MS": "500"}
    env.update(kw)
    return RemoteConfig.from_env(env)


@pytest.fixture
def server():
    core = WorkerCore(hello())
    srv = TcpStubServer(core, "127.0.0.1", 0).start()
    yield srv
    srv.stop()


def msg_with_rows(n_req, rows):
    buf = bytearray(wire.APP_HEADER_SIZE + wire.rows_offset_for(n_req) + rows.nbytes)
    off = wire.APP_HEADER_SIZE + wire.rows_offset_for(n_req)
    buf[off:off + rows.nbytes] = rows.tobytes()
    return buf


def wait_ready(cli, timeout=3.0):
    t = time.monotonic() + timeout
    while not cli.ready and time.monotonic() < t:
        time.sleep(0.005)
    return cli.ready


def test_connect_hello_step(server):
    cli = RemoteDraftClient(cfg_for(server.port), hello(), background=False)
    assert cli.connect_now() and cli.ready and cli.epoch == 1
    rows = np.ones((2, RB), np.uint8)
    e = [wire.ReqEntry(0, 1, 0, 2, 3, 2, flags=wire.RF_NEED_DRAFT | wire.RF_NEW_GEN)]
    out = cli.step(msg_with_rows(1, rows), e, 2, step_id=1)
    assert out.ok and out.reply.status[0] == wire.RS_OK and out.rtt_us > 0
    st, body = cli.simple_request(wire.OP_PING)
    assert st == wire.ST_OK
    cli.close()


def test_identity_mismatch_disables(server):
    cli = RemoteDraftClient(cfg_for(server.port), hello(drafter_identity=bytes(32)),
                            background=False)
    assert not cli.connect_now()
    assert cli.disabled and "IDENTITY_MISMATCH" in cli.disabled_reason
    out = cli.step(bytearray(4096), [wire.ReqEntry(0, 1, 0, 1, 1, 1)], 1, step_id=1)
    assert not out.ok and out.reason == "disabled"


def test_timeout_then_background_reconnect(server):
    cli = RemoteDraftClient(cfg_for(server.port), hello(), background=True)
    assert wait_ready(cli)
    server.core.inject.delay_ms = 400               # > 200 ms deadline
    e = [wire.ReqEntry(0, 1, 0, 1, 3, 1, flags=wire.RF_NEED_DRAFT)]
    out = cli.step(msg_with_rows(1, np.zeros((1, RB), np.uint8)), e, 1, step_id=1)
    assert not out.ok and out.reason == "timeout" and out.unknown_apply
    assert not cli.ready
    server.core.inject.delay_ms = 0
    assert wait_ready(cli)
    assert cli.epoch == 2
    out = cli.step(msg_with_rows(1, np.zeros((1, RB), np.uint8)), e, 1, step_id=2)
    assert out.ok
    cli.close()


def test_worker_restart_no_session_recovers(server):
    cli = RemoteDraftClient(cfg_for(server.port), hello(), background=True)
    assert wait_ready(cli)
    server.core = WorkerCore(hello())               # worker lost its state
    e = [wire.ReqEntry(0, 1, 0, 1, 3, 1, flags=wire.RF_NEED_DRAFT)]
    out = cli.step(msg_with_rows(1, np.zeros((1, RB), np.uint8)), e, 1, step_id=1)
    assert not out.ok and out.reason == "frame_no_session" and not out.unknown_apply
    assert wait_ready(cli) and cli.epoch == 2
    out = cli.step(msg_with_rows(1, np.zeros((1, RB), np.uint8)), e, 1, step_id=2)
    assert out.ok
    cli.close()


def test_frame_error_keeps_connection(server):
    cli = RemoteDraftClient(cfg_for(server.port), hello(), background=False)
    assert cli.connect_now()
    server.core.inject.frame_status_once = wire.ST_INTERNAL
    e = [wire.ReqEntry(0, 1, 0, 1, 3, 1, flags=wire.RF_NEED_DRAFT)]
    out = cli.step(msg_with_rows(1, np.zeros((1, RB), np.uint8)), e, 1, step_id=1)
    assert not out.ok and out.reason == "frame_internal" and cli.ready
    out = cli.step(msg_with_rows(1, np.zeros((1, RB), np.uint8)), e, 1, step_id=2)
    assert out.ok
    cli.close()


def test_dropped_connection_is_transport_error(server):
    cli = RemoteDraftClient(cfg_for(server.port), hello(), background=False)
    assert cli.connect_now()
    server.core.inject.drop_next_step = True
    e = [wire.ReqEntry(0, 1, 0, 1, 3, 1, flags=wire.RF_NEED_DRAFT)]
    out = cli.step(msg_with_rows(1, np.zeros((1, RB), np.uint8)), e, 1, step_id=1)
    assert not out.ok and out.reason == "transport_error" and out.unknown_apply
    assert not cli.ready


def test_unreachable_worker_never_blocks_engine():
    cli = RemoteDraftClient(cfg_for(1), hello(), background=True)   # port 1: refused
    t0 = time.perf_counter()
    out = cli.step(bytearray(4096), [wire.ReqEntry(0, 1, 0, 1, 1, 1)], 1, step_id=1)
    assert not out.ok and out.reason == "not_ready"
    assert time.perf_counter() - t0 < 0.05
    time.sleep(0.05)
    assert cli.stats.snapshot()["counters"].get("connect_failures", 0) >= 1
    cli.close()


def test_unreachable_worker_logs_first_failure_then_once_per_interval(caplog):
    """A worker that stays down must not flood the serve log: the retry loop runs every backoff interval (5 s in
    production), so the first failure is logged and then one line per CONNECT_FAIL_LOG_S."""
    import socket
    from remote_dspark import client as client_mod
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()                                            # nothing listens: every attempt is refused
    caplog.set_level("INFO", logger="vllm.remote_dspark")
    cli = RemoteDraftClient(cfg_for(port), hello(), background=True)
    deadline = time.monotonic() + 3.0
    while cli.stats.snapshot()["counters"].get("connect_failures", 0) < 10 and time.monotonic() < deadline:
        time.sleep(0.01)
    fails = cli.stats.snapshot()["counters"].get("connect_failures", 0)
    assert fails >= 10
    warned = [r for r in caplog.records if "connect/HELLO" in r.getMessage()]
    assert len(warned) == 1 and "further failures logged once per" in warned[0].getMessage()
    # the interval elapses: the next failure logs one "still failing" line with the running count
    cli._fail_logged -= client_mod.CONNECT_FAIL_LOG_S
    deadline = time.monotonic() + 3.0
    while len([r for r in caplog.records if "still failing" in r.getMessage()]) < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    still = [r for r in caplog.records if "still failing" in r.getMessage()]
    assert len(still) == 1
    # the worker comes up on that port: the ready line names the failed attempts and the run resets
    srv = TcpStubServer(WorkerCore(hello()), "127.0.0.1", port).start()
    try:
        assert wait_ready(cli)
        ready = [r for r in caplog.records if "ready at" in r.getMessage()]
        assert len(ready) == 1 and "failed connect attempts" in ready[0].getMessage()
        assert cli._fail_run == 0
    finally:
        cli.close()
        srv.stop()


# ------------------------------------------------------------------ two-part STEP (S11 P-1)

def test_step_with_private_headers_and_separate_rows(server):
    """client.step(rows=): the headers go into the caller's private buffer, the rows are sent from their own memory,
    which is never written; the worker sees the same STEP as the in-place form."""
    cli = RemoteDraftClient(cfg_for(server.port), hello(), background=False)
    assert cli.connect_now()
    rows = (np.arange(3 * RB, dtype=np.int64) % 251).astype(np.uint8).reshape(3, RB)
    rows.flags.writeable = False                    # read-only: nothing may be written into the staging
    prefix = bytearray(wire.APP_HEADER_SIZE + wire.rows_offset_for(4))
    e = [wire.ReqEntry(0, 1, 0, 3, 3, 3, flags=wire.RF_NEED_DRAFT | wire.RF_NEW_GEN)]
    out = cli.step(prefix, e, 3, step_id=1, rows=rows)
    assert out.ok and out.reply.status[0] == wire.RS_OK
    ref = cli.step(msg_with_rows(1, rows), [wire.ReqEntry(1, 1, 0, 3, 3, 3, flags=wire.RF_NEED_DRAFT
                                                          | wire.RF_NEW_GEN)], 3, step_id=2)
    assert ref.ok and np.array_equal(ref.reply.draft_ids[0], out.reply.draft_ids[0])   # same window, same drafts
    with pytest.raises(ValueError, match="rows hold"):          # 3 rows given, the entries carry 2
        cli.step(prefix, [wire.ReqEntry(2, 1, 0, 2, 3, 2, flags=wire.RF_NEW_GEN)], 2, step_id=3, rows=rows)
    cli.close()


def test_tcp_request_parts_survives_partial_writes():
    """TcpClient.request_parts sends the length prefix and every part as one frame through partial sendmsg writes
    (a small send buffer and a slow reader), never joining them."""
    import socket
    import threading

    from remote_dspark.transport import TcpClient, recv_frame_blocking, send_frame_blocking

    lst = socket.socket()
    lst.bind(("127.0.0.1", 0))
    lst.listen(1)
    got = {}

    def serve():
        conn, _ = lst.accept()
        buf = bytearray(16)
        time.sleep(0.2)                              # the client's sends fill the small buffers first
        n = recv_frame_blocking(conn, buf)
        got["frame"] = bytes(buf[:n])
        send_frame_blocking(conn, b"ok")
        conn.close()

    th = threading.Thread(target=serve)
    th.start()
    cli = TcpClient("127.0.0.1", lst.getsockname()[1])
    cli.connect(2.0)
    cli._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    calls = []
    real = cli._sock.sendmsg

    class Sock:                                       # counts the sendmsg calls of one request
        def __getattr__(self, k):
            return getattr(cli_sock, k)

        def sendmsg(self, bufs):
            calls.append(len(bufs))
            return real(bufs)

    cli_sock = cli._sock
    cli._sock = Sock()
    a = bytearray(b"head" * 10)
    b = np.arange(3 << 20, dtype=np.int64).astype(np.uint8)            # 3 MiB: many partial writes
    resp = bytearray(8)
    n = cli.request_parts((a, b, bytearray()), resp, 10.0)
    cli._sock = cli_sock
    th.join()
    lst.close()
    cli.close()
    assert bytes(resp[:n]) == b"ok"
    assert got["frame"] == bytes(a) + b.tobytes()
    assert len(calls) > 1 and calls[0] == 3        # partial writes; first call: length prefix + the two parts
