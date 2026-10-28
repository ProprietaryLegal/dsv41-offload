# SPDX-License-Identifier: Apache-2.0
"""Reference implementation of the worker side of CONTRACT.md, with a fake drafter.

Two uses:
  * offline tests of the Spark proxy (lane O3): every state rule of CONTRACT 7 is
    implemented here exactly; the fake drafter's output is a hash of the window rows the
    anchor may read, so any window error changes the drafts and the tests see it;
  * a reference for lane O2's server.py: ``WorkerCore`` is the whole protocol state
    machine with the drafter behind one function (``draft_fn``); O2 can replace
    ``fake_draft`` by the real model call and keep the rest, or port it.

Transports: the contract's TCP stub (length-prefixed frames, newest connection wins) and
lane O1's ofl server (``--ofl-dir``; TCP + RDMA backends, service "dspark-v1").

    python -m remote_dspark.mock_worker --listen 127.0.0.1:18620 --identity <64hex>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import wire
from .state import WindowRange
from .transport import TransportError, recv_frame_blocking, send_frame_blocking

log = logging.getLogger("dsv41.mock_worker")

_LCG_A = 6364136223846793005
_LCG_C = 1442695040888963407
_M64 = (1 << 64) - 1


def fake_draft(window: list[tuple[int, bytes]], anchor_token: int, anchor_pos: int,
               num_spec: int, vocab: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic stand-in for the drafter: ids/confidences from a hash of the window
    rows (position + bytes, ascending) and the anchor."""
    h = hashlib.blake2b(digest_size=8)
    for pos, row in window:
        h.update(int(pos).to_bytes(8, "little", signed=True))
        h.update(row)
    h.update(int(anchor_token).to_bytes(8, "little", signed=True))
    h.update(int(anchor_pos).to_bytes(8, "little", signed=True))
    x = int.from_bytes(h.digest(), "little")
    ids = np.empty(num_spec, dtype=np.int32)
    conf = np.empty(num_spec, dtype=np.float32)
    for j in range(num_spec):
        x = (x * _LCG_A + _LCG_C) & _M64
        ids[j] = (x >> 17) % vocab
        conf[j] = ((x >> 40) & 0xFFFF) / 65535.0
    return ids, conf


@dataclass
class SlotWindow:
    window: int
    row_bytes: int
    generation: int | None = None
    rng: WindowRange = field(init=False)
    rows: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.rng = WindowRange(self.window)
        self.rows = np.zeros((self.window, self.row_bytes), dtype=np.uint8)

    def reset(self, generation: int | None) -> None:
        self.generation = generation
        self.rng.clear()

    def window_rows(self, anchor_pos: int) -> list[tuple[int, bytes]]:
        return [(p, self.rows[p % self.window].tobytes())
                for p in self.rng.positions(anchor_pos)]


@dataclass
class Inject:
    """One-shot / persistent fault injection for tests."""
    delay_ms: float = 0.0                     # persistent: sleep before replying to STEP
    frame_status_once: int | None = None      # next STEP gets this frame status
    frame_status_always: int | None = None    # persistent: every STEP gets this frame status
    req_status_once: dict[int, int] = field(default_factory=dict)  # slot -> status
    req_status_always: int | None = None      # persistent: every entry gets this status
    drop_next_step: bool = False              # close the connection instead of replying
    # called with (StepReply, StepRequest) before the reply is packed; may edit the reply
    # (e.g. corrupt values); it stays armed until it returns True
    reply_hook_once: Callable[..., bool] | None = None
    reply_hook_fired: bool = False


class WorkerCore:
    """Transport-agnostic contract state machine: handle(frame) -> reply frame."""

    def __init__(self, expected: wire.Hello, *, worker_name: str = "mock-worker",
                 draft_fn: Callable[..., tuple[np.ndarray, np.ndarray]] = fake_draft,
                 capabilities: int = wire.CAP_TAPS_BF16 | wire.CAP_HAS_MARGIN):
        self.expected = expected
        self.worker_name = worker_name
        self.draft_fn = draft_fn
        self.capabilities = capabilities
        self.session: int | None = None
        self.last_seq = 0
        self.slots = [SlotWindow(expected.window, expected.row_bytes)
                      for _ in range(expected.max_slots)]
        self.counters = wire.Counters()
        self.inject = Inject()
        self.lock = threading.Lock()
        self.started = time.time()
        self.step_log: list[dict] = []    # tests: decoded entries per STEP
        self.keep_step_log = False

    # -------------------------------------------------------------- entry point

    def handle(self, buf) -> bytes | None:
        """Returns the reply frame, or None to drop the connection (fault injection)."""
        with self.lock:
            return self._handle(buf)

    def _handle(self, buf) -> bytes | None:
        try:
            h, body = wire.split_frame(buf)
        except wire.WireError as e:
            self.counters.inc("malformed_frames")
            log.warning("malformed frame: %s", e)
            return wire.frame(0, 0, 0, b"", wire.ST_MALFORMED)
        if h.op == wire.OP_PING:
            return wire.frame(h.op, h.seq, h.session)
        if h.op == wire.OP_HELLO:
            return self._hello(h, body)
        if h.op not in (wire.OP_STEP, wire.OP_RESET, wire.OP_STATS):
            return wire.frame(h.op, h.seq, h.session, b"", wire.ST_UNSUPPORTED_OP)
        if self.session is None or h.session != self.session:
            self.counters.inc("no_session")
            return wire.frame(h.op, h.seq, h.session, b"", wire.ST_NO_SESSION)
        if h.seq <= self.last_seq:
            self.counters.inc("stale_seq")
            return wire.frame(h.op, h.seq, h.session, b"", wire.ST_STALE_SEQ)
        self.last_seq = h.seq
        if h.op == wire.OP_STEP:
            return self._step(h, body)
        if h.op == wire.OP_RESET:
            if len(body) != wire.RESET_BODY.size:
                return wire.frame(h.op, h.seq, h.session, b"", wire.ST_MALFORMED)
            (mask,) = wire.RESET_BODY.unpack(bytes(body))
            for i, s in enumerate(self.slots):
                if mask >> i & 1:
                    s.reset(None)
            self.counters.inc("resets")
            return wire.frame(h.op, h.seq, h.session)
        stats = json.dumps(self.stats()).encode()
        return wire.frame(h.op, h.seq, h.session, stats)

    def stats(self) -> dict:
        return {"worker": self.worker_name, "session": self.session,
                "uptime_s": round(time.time() - self.started, 1),
                **self.counters.values}

    # -------------------------------------------------------------- HELLO

    def _hello(self, h: wire.AppHeader, body) -> bytes:
        try:
            req = wire.Hello.unpack(body)
        except wire.WireError:
            return wire.frame(h.op, h.seq, h.session, b"", wire.ST_MALFORMED)
        rep = wire.HelloReply(
            proto=1, max_slots=self.expected.max_slots,
            max_rows_per_step=self.expected.max_rows_per_step,
            capabilities=self.capabilities, window=self.expected.window,
            num_spec=self.expected.num_spec,
            drafter_identity=self.expected.drafter_identity,
            markov_identity=self.expected.markov_identity, worker_name=self.worker_name)
        if not req.proto_min <= 1 <= req.proto_max:
            return wire.frame(h.op, h.seq, h.session, rep.pack(), wire.ST_BAD_VERSION)
        if (req.drafter_identity != self.expected.drafter_identity
                or req.markov_identity != self.expected.markov_identity):
            self.counters.inc("identity_mismatch")
            return wire.frame(h.op, h.seq, h.session, rep.pack(), wire.ST_IDENTITY_MISMATCH)
        if (req.config_tuple() != self.expected.config_tuple()
                or req.max_slots > self.expected.max_slots
                or req.max_rows_per_step > self.expected.max_rows_per_step):
            self.counters.inc("config_mismatch")
            return wire.frame(h.op, h.seq, h.session, rep.pack(), wire.ST_CONFIG_MISMATCH)
        if h.session == 0:
            return wire.frame(h.op, h.seq, h.session, b"", wire.ST_MALFORMED)
        if h.session != self.session:
            for s in self.slots:
                s.reset(None)
            self.session = h.session
            self.last_seq = h.seq
            self.counters.inc("new_sessions")
        else:
            if h.seq <= self.last_seq:
                return wire.frame(h.op, h.seq, h.session, b"", wire.ST_STALE_SEQ)
            self.last_seq = h.seq
            self.counters.inc("rehellos")
        return wire.frame(h.op, h.seq, h.session, rep.pack())

    # -------------------------------------------------------------- STEP

    def _step(self, h: wire.AppHeader, body) -> bytes | None:
        t0 = time.perf_counter()
        inj = self.inject
        self.counters.inc("step_frames")
        if inj.frame_status_once is not None:
            st, inj.frame_status_once = inj.frame_status_once, None
            return wire.frame(h.op, h.seq, h.session, b"", st)
        if inj.frame_status_always is not None:
            return wire.frame(h.op, h.seq, h.session, b"", inj.frame_status_always)
        exp = self.expected
        try:
            req = wire.decode_step(body, max_slots=exp.max_slots,
                                   max_rows=exp.max_rows_per_step,
                                   expected_row_bytes=exp.row_bytes, window=exp.window)
        except wire.WireError as e:
            self.counters.inc("malformed_steps")
            log.warning("malformed STEP: %s", e)
            return wire.frame(h.op, h.seq, h.session, b"", wire.ST_MALFORMED)
        n, N = req.n_req, exp.num_spec
        status = np.zeros(n, dtype=np.uint16)
        wrows = np.zeros(n, dtype=np.uint16)
        gens = np.zeros(n, dtype=np.uint32)
        ids = np.zeros((n, N), dtype=np.int32)
        conf = np.zeros((n, N), dtype=np.float32)
        want_margin = bool(req.step_flags & wire.SF_WANT_MARGIN) and bool(
            self.capabilities & wire.CAP_HAS_MARGIN)
        margin = np.zeros((n, N), dtype=np.float32) if want_margin else None
        row = 0
        for i, e in enumerate(req.entries):
            gens[i] = e.generation & wire.U32_MASK
            rows = req.rows[row:row + e.n_rows]
            row += e.n_rows
            if self.keep_step_log:
                self.step_log.append({"step_id": req.step_id, "slot": e.slot,
                                      "flags": e.flags, "gen": e.generation,
                                      "first_pos": e.first_pos, "n_rows": e.n_rows,
                                      "anchor_pos": e.anchor_pos,
                                      "anchor_token": e.anchor_token})
            s = self.slots[e.slot]
            # CONTRACT 7.2 step 1: an older generation is stale whatever its flags; NEW_GEN
            # resets the window only for the slot's own or a newer generation (resync)
            if s.generation is not None and wire.gen_newer(s.generation, e.generation):
                status[i] = wire.RS_STALE_GENERATION
                self.counters.inc("stale_generation")
                continue
            if s.generation is None or e.flags & wire.RF_NEW_GEN or e.generation != s.generation:
                s.reset(e.generation)
            try:
                s.rng.apply(e.first_pos, e.n_rows, e.anchor_pos)
            except ValueError as err:
                status[i] = wire.RS_BAD_ENTRY
                self.counters.inc("bad_entries")
                log.warning("slot %d: bad entry: %s", e.slot, err)
                s.rng.clear()
                continue
            for k in range(e.n_rows):
                s.rows[(e.first_pos + k) % exp.window] = rows[k]
            self.counters.inc("rows", e.n_rows)
            have = s.rng.rows_for_anchor(e.anchor_pos)
            wrows[i] = have
            forced = inj.req_status_once.pop(e.slot, None)
            if forced is None:
                forced = inj.req_status_always
            if forced is not None:
                status[i] = forced
                if forced not in wire.USABLE_REQ_STATUS:
                    continue
            if e.flags & wire.RF_NEED_DRAFT:
                d, c = self.draft_fn(s.window_rows(e.anchor_pos), e.anchor_token,
                                     e.anchor_pos, N, exp.vocab_size)
                # CONTRACT 1.3: positions >= n_out are not consumed and come back as zeros;
                # the others are exactly the full-block draft
                k = N if e.n_out == 0 else min(int(e.n_out), N)
                ids[i, :k], conf[i, :k] = d[:k], c[:k]
                if margin is not None:
                    margin[i, :k] = c[:k]
                self.counters.inc(f"n_out_{k}")
                self.counters.inc("drafted_requests")
                if forced is None:
                    if have < s.rng.expected_rows(e.anchor_pos):
                        status[i] = wire.RS_PARTIAL_WINDOW
                        self.counters.inc("partial_windows")
                    else:
                        status[i] = wire.RS_OK
        self.counters.inc("steps")
        if inj.delay_ms > 0:
            time.sleep(inj.delay_ms / 1000.0)
        if inj.drop_next_step:
            inj.drop_next_step = False
            return None
        worker_us = int((time.perf_counter() - t0) * 1e6)
        rep = wire.StepReply(step_id=req.step_id, num_spec=N, status=status,
                             window_rows=wrows, generation=gens, draft_ids=ids,
                             confidence=conf, margin=margin, worker_us=worker_us,
                             recv_to_send_us=worker_us)
        if inj.reply_hook_once is not None and inj.reply_hook_once(rep, req):
            inj.reply_hook_once = None
            inj.reply_hook_fired = True
        return wire.frame(h.op, h.seq, h.session, rep.pack())


# ------------------------------------------------------------------ TCP stub server


class TcpStubServer:
    """Length-prefixed frames; the newest connection supersedes the previous one."""

    def __init__(self, core: WorkerCore, host: str, port: int):
        if host in ("0.0.0.0", "", "::"):
            raise ValueError("bind a specific (direct-link) address, never 0.0.0.0")
        self.core = core
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        self.host = host
        self._conn: socket.socket | None = None
        self._stop = False
        self._lock = threading.Lock()
        self.connections = 0
        self._thread = threading.Thread(target=self._accept_loop, daemon=True,
                                        name="mock-worker-accept")

    def start(self) -> "TcpStubServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop = True
        try:
            self.sock.close()
        except OSError:
            pass
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except OSError:
                    pass

    def drop_connection(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def _accept_loop(self) -> None:
        while not self._stop:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._lock:
                old, self._conn = self._conn, conn
                self.connections += 1
            if old is not None:
                try:
                    old.shutdown(socket.SHUT_RDWR)
                    old.close()
                except OSError:
                    pass
            threading.Thread(target=self._serve, args=(conn,), daemon=True,
                             name="mock-worker-conn").start()

    def _serve(self, conn: socket.socket) -> None:
        buf = bytearray(1 << 20)
        try:
            while not self._stop:
                n = recv_frame_blocking(conn, buf)
                if n is None:
                    return
                reply = self.core.handle(memoryview(buf)[:n])
                if reply is None:
                    conn.close()
                    return
                send_frame_blocking(conn, reply)
        except (OSError, TransportError):
            return
        finally:
            try:
                conn.close()
            except OSError:
                pass


def serve_ofl(core: WorkerCore, bind_ip: str, port: int, ofl_dir: str,
              rdma: bool = True):
    """Serve the contract on lane O1's ofl server (service dspark-v1)."""
    import os

    os.environ["VLLM_DSPARK_REMOTE_OFL_DIR"] = ofl_dir
    from .o1_adapter import DEFAULT_MAX_REQ, DEFAULT_MAX_RESP, SERVICE, load_ofl

    ofl = load_ofl()

    def handler(req) -> int:
        reply = core.handle(req.payload)
        if reply is None:
            return -1
        req.respond(reply)
        return 0

    backends = (ofl.BACKEND_TCP, ofl.BACKEND_RDMA) if rdma else (ofl.BACKEND_TCP,)
    srv = ofl.Server(bind_ip, port, service=SERVICE, handler=handler, backends=backends,
                     max_req=DEFAULT_MAX_REQ, max_resp=DEFAULT_MAX_RESP, inflight=1)
    srv.start()
    return srv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", required=True, help="host:port (direct-link address)")
    ap.add_argument("--identity", required=True, help="64-hex drafter identity")
    ap.add_argument("--markov-identity", default="0" * 64)
    ap.add_argument("--ofl-dir", default="", help="serve over lane O1's ofl instead of TCP stub")
    ap.add_argument("--no-rdma", action="store_true")
    ap.add_argument("--delay-ms", type=float, default=0.0, help="simulated draft time")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    host, port = args.listen.rsplit(":", 1)
    expected = wire.Hello(drafter_identity=wire.parse_hex32(args.identity, "--identity"),
                          markov_identity=wire.parse_hex32(args.markov_identity,
                                                           "--markov-identity"))
    core = WorkerCore(expected)
    core.inject.delay_ms = args.delay_ms
    if args.ofl_dir:
        srv = serve_ofl(core, host, int(port), args.ofl_dir, rdma=not args.no_rdma)
        log.info("mock worker on ofl %s", srv.describe())
    else:
        srv = TcpStubServer(core, host, int(port)).start()
        log.info("mock worker on tcp %s:%d", host, srv.port)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        srv.stop()


if __name__ == "__main__":
    main()
