# SPDX-License-Identifier: Apache-2.0
"""Rank-0 client of the remote DSpark worker: session, sequence numbers, HELLO,
background reconnect with backoff, and the STEP exchange (CONTRACT 6, 7.5, 9).

torch-free. The engine thread only calls ``step()`` (never blocks on connects); a
daemon thread owns connect + HELLO and hands a ready connection over.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import wire
from .config import RemoteConfig
from .stats import Stats
from .transport import RpcClient, TransportError, TransportTimeout, make_client

log = logging.getLogger("vllm.remote_dspark")

# While the worker stays unreachable the background thread retries every backoff interval (5 s in production), so only the
# first failure of a run and then one line per CONNECT_FAIL_LOG_S are logged; the counter connect_failures counts them all.
CONNECT_FAIL_LOG_S = 60.0


def check_reply_values(reply: wire.StepReply, entries: list[wire.ReqEntry], *,
                       vocab_size: int, window: int) -> None:
    """Refuse a STEP reply whose values the target must not see (CONTRACT 6.3, 7.2, 9.2).

    Shape and the step / generation echoes are checked before this. Draft ids become the
    target's input ids on both ranks and confidences feed the depth policy, so per entry:
      * status is a known per-request status;
      * window_rows <= min(W, anchor_pos) (no more rows than the anchor can read); for a
        drafted entry OK means exactly that many and PARTIAL_WINDOW fewer (7.2 step 5);
      * consumed positions (NEED_DRAFT, usable status, below n_out): draft ids in
        [0, vocab_size), confidences finite in [0, 1], margins finite and >= 0;
      * every other position (past n_out, entries without NEED_DRAFT, failed entries):
        draft 0, confidence 0.0, margin 0.0.
    Raises wire.WireError naming the first violation. Hot path (every round): scalar checks
    per entry, then one masked pass over the [n, N] arrays; the per-position search for the
    message runs only when that pass fails."""
    N = reply.num_spec
    status, wrows = reply.status.tolist(), reply.window_rows.tolist()
    ks = []
    for i, e in enumerate(entries):
        rs, wr = status[i], wrows[i]
        if rs not in wire.REQ_STATUS_NAMES:
            raise wire.WireError(f"slot {e.slot}: unknown request status {rs}")
        expected = min(window, max(e.anchor_pos, 0))
        if wr > expected:
            raise wire.WireError(f"slot {e.slot}: window_rows {wr} > {expected} readable rows "
                                 f"(anchor_pos {e.anchor_pos})")
        consumed = bool(e.flags & wire.RF_NEED_DRAFT) and rs in wire.USABLE_REQ_STATUS
        if consumed:
            if rs == wire.RS_OK and wr != expected:
                raise wire.WireError(f"slot {e.slot}: status OK with window_rows {wr} "
                                     f"!= {expected}")
            if rs == wire.RS_PARTIAL_WINDOW and wr >= expected:
                raise wire.WireError(f"slot {e.slot}: PARTIAL_WINDOW with window_rows {wr} "
                                     f">= {expected}")
        ks.append((N if e.n_out == 0 else min(e.n_out, N)) if consumed else 0)
    live = np.arange(N)[None, :] < np.array(ks)[:, None]          # [n, N] consumed positions
    ids, conf, m = reply.draft_ids, reply.confidence, reply.margin
    # NaN fails every comparison below, so a NaN anywhere fails the pass
    ok = (np.where(live, (ids >= 0) & (ids < vocab_size), ids == 0).all()
          and np.where(live, (conf >= 0.0) & (conf <= 1.0), conf == 0.0).all()
          and (m is None or np.where(live, np.isfinite(m) & (m >= 0.0), m == 0.0).all()))
    if not ok:
        _raise_first_value_violation(reply, entries, ks, vocab_size)


def _raise_first_value_violation(reply: wire.StepReply, entries: list[wire.ReqEntry],
                                 ks: list[int], vocab_size: int) -> None:
    for i, (e, k) in enumerate(zip(entries, ks)):
        ids, conf = reply.draft_ids[i], reply.confidence[i]
        bad = np.flatnonzero((ids[:k] < 0) | (ids[:k] >= vocab_size))
        if bad.size:
            j = int(bad[0])
            raise wire.WireError(f"slot {e.slot} position {j}: draft id {int(ids[j])} "
                                 f"outside [0, {vocab_size})")
        bad = np.flatnonzero(~((conf[:k] >= 0.0) & (conf[:k] <= 1.0)))
        if bad.size:
            j = int(bad[0])
            raise wire.WireError(f"slot {e.slot} position {j}: confidence {float(conf[j])!r} "
                                 "is not a probability")
        tails = [("draft id", ids[k:]), ("confidence", conf[k:])]
        if reply.margin is not None:
            m = reply.margin[i]
            bad = np.flatnonzero(~(np.isfinite(m[:k]) & (m[:k] >= 0.0)))
            if bad.size:
                j = int(bad[0])
                raise wire.WireError(f"slot {e.slot} position {j}: margin {float(m[j])!r} "
                                     "is not a finite top1 - top2")
            tails.append(("margin", m[k:]))
        for what, tail in tails:
            bad = np.flatnonzero(tail != 0)
            if bad.size:
                j = k + int(bad[0])
                raise wire.WireError(
                    f"slot {e.slot} position {j}: {what} {tail[j - k].item()!r} where the "
                    f"contract requires 0 (n_out {e.n_out}, status {int(reply.status[i])}, "
                    f"need_draft {bool(e.flags & wire.RF_NEED_DRAFT)})")
    raise wire.WireError("reply values violate CONTRACT 6.3")   # unreachable: pass and search agree


@dataclass
class StepOutcome:
    ok: bool
    reply: wire.StepReply | None = None
    reason: str | None = None        # short counter key when not ok
    rtt_us: float = 0.0
    # True when the frame may have been applied by the worker but its result is unknown
    # (timeout / transport error after sending): every slot in the STEP must resync.
    unknown_apply: bool = False


class RemoteDraftClient:
    def __init__(
        self,
        cfg: RemoteConfig,
        hello: wire.Hello,
        *,
        stats: Stats | None = None,
        transport_factory: Callable[..., RpcClient] | None = None,
        background: bool = True,
        session: int | None = None,
    ):
        """session: the session id of every HELLO and STEP (default: random). A respawned process proxy gets one its
        predecessor never used, so its first HELLO makes the worker drop every window (CONTRACT 6.2)."""
        self.cfg = cfg
        self.hello = hello
        self.stats = stats or Stats()
        self._factory = transport_factory or make_client
        if session is not None and not 0 < session <= 0xFFFFFFFF:
            raise ValueError(f"session {session} outside 1..2**32-1")
        self.session = session or secrets.randbits(32) or 1
        self._lock = threading.Lock()
        self._seq = 0
        self._client: RpcClient | None = None
        self._ready = False
        self.epoch = 0                       # successful HELLOs so far
        self.disabled_reason: str | None = None
        self.worker_info: wire.HelloReply | None = None
        self.last_error: str | None = None
        self._backoff_ms = cfg.backoff_ms
        self._next_attempt = 0.0
        self._fail_run = 0                   # connect/HELLO failures since the last successful HELLO
        self._fail_logged = 0.0              # monotonic time of the last connect-failure log line
        self._wake = threading.Event()
        self._stop = False
        self._resp = bytearray(1 << 16)
        self._thread: threading.Thread | None = None
        if background:
            self.start_background()

    def start_background(self) -> None:
        """Start the daemon thread that (re)connects whenever the connection is down."""
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="dsv41-remote-dspark-connect", daemon=True)
            self._thread.start()

    # ------------------------------------------------------------------ state

    @property
    def ready(self) -> bool:
        return self._ready and self.disabled_reason is None

    @property
    def disabled(self) -> bool:
        return self.disabled_reason is not None

    def _next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def close(self) -> None:
        self._stop = True
        self._wake.set()
        with self._lock:
            if self._client is not None:
                self._client.close()
            self._client = None
            self._ready = False

    def _disable(self, reason: str) -> None:
        self.disabled_reason = reason
        self.stats.inc("remote_disabled")
        self.stats.set("disabled_reason", reason)
        log.error("remote DSpark disabled for this engine run: %s", reason)
        with self._lock:
            if self._client is not None:
                self._client.close()
            self._client = None
            self._ready = False
        self._wake.set()

    def new_session(self, reason: str) -> None:
        """Start a new session and reconnect at once. The HELLO with the new session id
        makes the worker drop every window (CONTRACT 6.2); the epoch change then resyncs
        every live slot from rank 0's ring. For a worker state that no longer matches rank
        0's (STALE_GENERATION: rank 0 never sends an older generation, CONTRACT 7.1)."""
        with self._lock:     # one critical section: no reader sees the new id on the old connection
            old = self.session
            new = old
            while new == old or new == 0:
                new = secrets.randbits(32)
            self.session = new
            self._drop_locked(backoff=False)
        self.last_error = f"new session: {reason}"
        self.stats.inc("new_sessions")
        self.stats.inc("disconnects")
        log.error("remote DSpark: new session %#010x (was %#010x): %s", new, old, reason)
        self._wake.set()

    def connection_epoch(self) -> int | None:
        """The epoch of the ready connection, or None when there is none (one read under the
        lock, so ready and epoch belong to the same connection)."""
        with self._lock:
            return self.epoch if self._ready and self.disabled_reason is None else None

    def mark_failed(self, reason: str, *, backoff: bool = True) -> None:
        """Drop the connection; the connector thread reconnects (after the backoff)."""
        self.last_error = reason
        with self._lock:
            self._drop_locked(backoff=backoff)
        self.stats.inc("disconnects")
        self._wake.set()

    def _drop_locked(self, *, backoff: bool) -> None:
        if self._client is not None:
            self._client.close()
        self._client = None
        self._ready = False
        now = time.monotonic()
        if backoff:
            self._next_attempt = now + self._backoff_ms / 1000.0
            self._backoff_ms = min(self._backoff_ms * 2, self.cfg.backoff_max_ms)
        else:
            self._next_attempt = now

    # ------------------------------------------------------------------ connect

    def connect_now(self) -> bool:
        """Connect + HELLO synchronously. True when a ready connection is installed."""
        if self.disabled:
            return False
        timeout = self.cfg.connect_timeout_ms / 1000.0
        cli = None
        session = self.session
        try:
            cli = self._factory(self.cfg.endpoint)
            cli.connect(timeout)
            seq = self._next_seq()
            msg = bytearray(wire.frame(wire.OP_HELLO, seq, session, self.hello.pack()))
            buf = bytearray(1 << 16)  # >= the O1 transport's registered reply size
            n = cli.request(msg, buf, timeout)
            h, body = wire.split_frame(memoryview(buf)[:n])
            if h.op != wire.OP_HELLO or h.seq != seq or h.session != session:
                raise wire.WireError(
                    f"HELLO reply echoes op={h.op} seq={h.seq} session={h.session:#x}")
            if h.status != wire.ST_OK:
                name = wire.FRAME_STATUS_NAMES.get(h.status, str(h.status))
                if h.status in wire.FATAL_FRAME_STATUS:
                    detail = ""
                    if len(body) == wire.HELLO_REP.size:
                        rep = wire.HelloReply.unpack(body)
                        detail = (f" (worker identity {rep.drafter_identity.hex()}, "
                                  f"markov {rep.markov_identity.hex()}; ours "
                                  f"{self.hello.drafter_identity.hex()}, "
                                  f"{self.hello.markov_identity.hex()})")
                    cli.close()
                    self._disable(f"HELLO refused: {name}{detail}")
                    return False
                raise wire.WireError(f"HELLO refused: {name}")
            rep = wire.HelloReply.unpack(body)
            problems = []
            if rep.drafter_identity != self.hello.drafter_identity:
                problems.append("drafter identity")
            if rep.markov_identity != self.hello.markov_identity:
                problems.append("markov identity")
            if rep.window != self.hello.window or rep.num_spec != self.hello.num_spec:
                problems.append(f"window/num_spec {rep.window}/{rep.num_spec}")
            if self.hello.sample_mode == wire.SAMPLE_TOP32 and not rep.capabilities & wire.CAP_TOP32:
                problems.append("missing dyadic24 top32 capability")
            if not rep.capabilities & wire.CAP_TAPS_BF16:
                problems.append("no TAPS_BF16 capability")
            if rep.max_slots < self.hello.max_slots:
                problems.append(f"max_slots {rep.max_slots} < {self.hello.max_slots}")
            if rep.max_rows_per_step < self.hello.max_rows_per_step:
                problems.append(f"max_rows_per_step {rep.max_rows_per_step}")
            if problems:
                cli.close()
                self._disable("HELLO reply mismatch: " + ", ".join(problems))
                return False
        except (TransportError, wire.WireError, OSError) as e:
            if cli is not None:
                cli.close()
            self.last_error = f"connect: {e}"
            self.stats.inc("connect_failures")
            self._fail_run += 1
            now = time.monotonic()
            if self._fail_run == 1:
                self._fail_logged = now
                log.warning("remote DSpark connect/HELLO to %s failed: %s (target-only rounds; retrying, "
                            "further failures logged once per %.0f s)", self.cfg.endpoint, e, CONNECT_FAIL_LOG_S)
            elif now - self._fail_logged >= CONNECT_FAIL_LOG_S:
                self._fail_logged = now
                log.warning("remote DSpark connect/HELLO to %s still failing after %d attempts: %s",
                            self.cfg.endpoint, self._fail_run, e)
            return False
        with self._lock:
            stale = session != self.session     # new_session() ran meanwhile
            if stale:
                cli.close()
        if stale:                                # HELLO again at once under the new id
            return self.connect_now()
        with self._lock:
            if self._client is not None:
                self._client.close()
            self._client = cli
            self._ready = True
            self.epoch += 1
            self._backoff_ms = self.cfg.backoff_ms
        self.worker_info = rep
        self.stats.inc("hellos")
        self.stats.set("worker", {"name": rep.worker_name, "max_slots": rep.max_slots,
                                  "max_rows_per_step": rep.max_rows_per_step,
                                  "capabilities": rep.capabilities})
        failed, self._fail_run = self._fail_run, 0
        log.info("remote DSpark worker %r ready at %s (session %#010x, epoch %d)%s",
                 rep.worker_name, self.cfg.endpoint, session, self.epoch,
                 f" after {failed} failed connect attempts" if failed else "")
        return True

    def _run(self) -> None:
        while not self._stop:
            if self.disabled:
                return
            if self._ready:
                self._wake.wait()
                self._wake.clear()
                continue
            delay = self._next_attempt - time.monotonic()
            if delay > 0:
                self._wake.wait(delay)
                self._wake.clear()
                continue
            if not self.connect_now() and not self.disabled:
                with self._lock:
                    self._next_attempt = time.monotonic() + self._backoff_ms / 1000.0
                    self._backoff_ms = min(self._backoff_ms * 2, self.cfg.backoff_max_ms)

    # ------------------------------------------------------------------ STEP

    def step(
        self,
        buf,
        entries: list[wire.ReqEntry],
        n_rows: int,
        step_id: int,
        step_flags: int = 0,
        epoch: int | None = None,
        rows=None,
        sample_meta=None,
    ) -> StepOutcome:
        """Send one STEP. Without ``rows``: ``buf`` is a writable buffer whose rows already
        sit at ``wire.APP_HEADER_SIZE + wire.rows_offset_for(len(entries))``; the headers are
        written in front of them here, so the message goes out without a copy. With ``rows``
        (n_rows x row_bytes, contiguous; RoundCore always): ``buf`` is the caller's private
        buffer and takes only the headers and entries, and the message goes out as
        [headers, rows] (RpcClient.request_parts), so nothing is written into the memory that
        holds the rows: the gather staging the engine stream may already be refilling for a
        later round (S11 P-1). ``epoch``: the connection epoch the entries were built for; a
        newer connection is refused (reason "reconnected", nothing sent), since its slots
        were never resynced."""
        if self.disabled:
            return StepOutcome(False, reason="disabled")
        with self._lock:
            cli = self._client if self._ready else None
            session = self.session
            moved = epoch is not None and self.epoch != epoch
        if cli is None:
            return StepOutcome(False, reason="not_ready")
        if moved:
            return StepOutcome(False, reason="reconnected")
        row_bytes = self.hello.row_bytes
        seq = self._next_seq()
        ro = wire.pack_step_prefix(buf, wire.APP_HEADER_SIZE, step_id, entries, n_rows,
                                   row_bytes, step_flags, sample_meta=sample_meta)
        body_len = ro + n_rows * row_bytes
        wire.AppHeader(op=wire.OP_STEP, seq=seq, session=session,
                       body_len=body_len).pack_into(buf, 0)
        if rows is None:
            parts = (memoryview(buf).cast("B")[:wire.APP_HEADER_SIZE + body_len],)
        else:
            if rows.nbytes != n_rows * row_bytes:
                raise ValueError(f"rows hold {rows.nbytes} bytes, {n_rows} rows need {n_rows * row_bytes}")
            parts = (memoryview(buf).cast("B")[:wire.APP_HEADER_SIZE + ro], rows)
        t0 = time.perf_counter()
        try:
            if len(parts) == 1:
                n = cli.request(parts[0], self._resp, self.cfg.timeout_ms / 1000.0)
            else:
                n = cli.request_parts(parts, self._resp, self.cfg.timeout_ms / 1000.0)
        except TransportTimeout:
            self.mark_failed("timeout")
            return StepOutcome(False, reason="timeout", unknown_apply=True)
        except TransportError as e:
            self.mark_failed(f"transport: {e}")
            return StepOutcome(False, reason="transport_error", unknown_apply=True)
        rtt_us = (time.perf_counter() - t0) * 1e6
        self.stats.inc("bytes_sent", wire.APP_HEADER_SIZE + body_len)
        self.stats.inc("bytes_received", n)
        try:
            h, body = wire.split_frame(memoryview(self._resp)[:n])
            if h.op != wire.OP_STEP or h.seq != seq or h.session != session:
                raise wire.WireError(
                    f"reply echoes op={h.op} seq={h.seq} (want {seq}) "
                    f"session={h.session:#x}")
        except wire.WireError as e:
            self.mark_failed(f"bad reply: {e}")
            return StepOutcome(False, reason="bad_reply", rtt_us=rtt_us, unknown_apply=True)
        if h.status != wire.ST_OK:
            name = wire.FRAME_STATUS_NAMES.get(h.status, str(h.status))
            if h.status in wire.FATAL_FRAME_STATUS:
                self._disable(f"STEP refused: {name}")
            elif h.status in (wire.ST_NO_SESSION, wire.ST_STALE_SEQ):
                # The worker did not apply the frame; a fresh HELLO fixes the session.
                self.mark_failed(f"STEP refused: {name}", backoff=False)
            return StepOutcome(False, reason=f"frame_{name.lower()}", rtt_us=rtt_us,
                               unknown_apply=False)
        try:
            reply = wire.StepReply.unpack(body, expect_n_req=len(entries),
                                          expect_num_spec=self.hello.num_spec)
            if reply.step_id != step_id:
                raise wire.WireError(f"reply step_id {reply.step_id} != {step_id}")
            for e, g in zip(entries, reply.generation.tolist()):
                if g != (e.generation & wire.U32_MASK):
                    raise wire.WireError(f"slot {e.slot}: generation echo {g} != {e.generation}")
            if self.hello.sample_mode == wire.SAMPLE_TOP32:
                from .sampled_wire import validate
                if reply.support_ids is None or reply.proposal_q is None or reply.sampled_mask is None:
                    raise wire.WireError("missing sampled q; refusing sampled IDs")
                try:
                    needed = np.array([bool(e.flags & wire.RF_NEED_DRAFT) and
                                       int(reply.status[i]) in wire.USABLE_REQ_STATUS
                                       for i, e in enumerate(entries)])
                    validate(reply.sampled_mask, reply.support_ids, reply.proposal_q, reply.draft_ids,
                             needed, self.hello.vocab_size,
                             np.array([int(m[1]) for m in sample_meta], dtype=np.int32))
                except ValueError as exc:
                    raise wire.WireError(str(exc)) from exc
            elif reply.support_ids is not None:
                raise wire.WireError("unexpected q on greedy session")
            check_reply_values(reply, entries, vocab_size=self.hello.vocab_size,
                               window=self.hello.window)
        except wire.WireError as e:
            self.stats.set("last_bad_reply", str(e))
            log.warning("remote DSpark: STEP reply refused (round fails, slots resync): %s", e)
            self.mark_failed(f"bad reply: {e}")
            return StepOutcome(False, reason="bad_reply", rtt_us=rtt_us, unknown_apply=True)
        return StepOutcome(True, reply=reply, rtt_us=rtt_us)

    # ------------------------------------------------------------------ tools

    def simple_request(self, op: int, body: bytes = b"") -> tuple[int, bytes]:
        """PING / STATS / RESET on the engine's connection (not on the hot path)."""
        with self._lock:
            cli = self._client if self._ready else None
        if cli is None:
            raise TransportError("not connected")
        seq = self._next_seq()
        msg = bytearray(wire.frame(op, seq, self.session, body))
        n = cli.request(msg, self._resp, self.cfg.timeout_ms / 1000.0)
        h, rb = wire.split_frame(memoryview(self._resp)[:n])
        if h.op != op or h.seq != seq:
            raise wire.WireError("reply does not echo the request")
        return h.status, bytes(rb)
