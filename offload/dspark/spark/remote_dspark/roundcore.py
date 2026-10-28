# SPDX-License-Identifier: Apache-2.0
"""The rank-0 round after the gather, torch-free: ring bookkeeping, circuit breaker, STEP, reply validation and the
packed reply, plus the async proxy's per-round driver (wait for the gather, run the round, publish).

Moved out of round.py without behaviour change so that one implementation runs in all three places:
  * the engine thread (synchronous rounds; round.RemoteRound),
  * the in-process proxy thread (async, VLLM_DSPARK_REMOTE_ASYNC_PROXY=thread; round.RemoteRound),
  * the out-of-process proxy (async, =process; proxy_proc.ProcessRound), which has its own interpreter and GIL, so
    the success path (publish) never waits for the engine's GIL.
RoundCore holds no torch object and makes no torch or CUDA call; a subclass sets the attributes listed in its
docstring and may override the hooks (_bcast_writable, _request_resync, _clear_resync).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from . import wire
from .client import StepOutcome
from .hostflag import FAR_AHEAD

log = logging.getLogger("vllm.remote_dspark")

STATUS_OK = 0            # every request that needed drafts got usable remote drafts
STATUS_SOME_FAILED = 1   # some requests did not (see ok_mask)
STATUS_ROUND_FAILED = 2  # no remote drafts this round
STATUS_NAMES = {STATUS_OK: "ok", STATUS_SOME_FAILED: "some_failed",
                STATUS_ROUND_FAILED: "round_failed"}
# async: the gather metadata of round t lives in slot t % META_SLOTS, so a proxy that reaches round t after the
# watchdog completed it (and the stream went on) still reads round t's own metadata, never a later round's
META_SLOTS = 64
WATCHDOG_EVENTS_KEPT = 64


class ReplyLayout:
    """Offsets of the int32 reply: [status, n, ok_mask[M], drafts[M*N], conf_bits[M*N]]."""

    def __init__(self, max_reqs: int, num_spec: int, sampled: bool = False):
        self.M, self.N = max_reqs, num_spec
        self.o_ok = 2
        self.o_ids = 2 + max_reqs
        self.o_conf = self.o_ids + max_reqs * num_spec
        self.o_sample_mask = self.o_conf + max_reqs * num_spec
        self.o_qids = self.o_sample_mask + max_reqs
        self.o_q = self.o_qids + max_reqs * num_spec * 32
        self.size = self.o_q + max_reqs * num_spec * 32 if sampled else self.o_sample_mask
        self.sampled = sampled


@dataclass
class BatchView:
    """CPU-side facts of a batch, from InputBatch (batch order)."""
    req_ids: list[str]
    slots: np.ndarray       # [n] request-state index (idx_mapping_np)
    qsl: np.ndarray         # [n+1] query_start_loc_np
    need_draft: np.ndarray  # [n] bool
    k_window: np.ndarray    # [n] drafts verified this step (0 for prefill)
    max_q: int | None = None  # longest query span; None = from qsl (the process proxy gets it in the descriptor)

    @property
    def n(self) -> int:
        return len(self.req_ids)

    @property
    def longest_query(self) -> int:
        if self.max_q is not None:
            return int(self.max_q)
        return int(np.diff(self.qsl).max()) if len(self.qsl) > 1 else 0


@dataclass
class GatherResult:
    valid: np.ndarray       # [n] valid rows per request (rows sent)
    first_pos: np.ndarray   # [n] position of the first candidate (= first valid) row
    anchor: np.ndarray      # [n] anchor token
    sampled: np.ndarray     # [n] bool: num_sampled > 0
    n_candidates: int
    sample_meta: list[tuple[int, int]] | None = None

    @property
    def n_valid(self) -> int:
        return int(self.valid.sum())


def collect_meta(meta_np: np.ndarray, n: int, n_candidates: int) -> GatherResult:
    """The gather's metadata rows [valid, first_pos, anchor, sampled] x max_slots (int64, host) of one enqueue (its
    metadata slot), read after its copy completed."""
    mh = meta_np
    return GatherResult(valid=mh[0, :n].copy(), first_pos=mh[1, :n].copy(),
                        anchor=mh[2, :n].copy(), sampled=mh[3, :n].astype(bool),
                        n_candidates=n_candidates,
                        sample_meta=[(int(mh[5, i]), int(mh[4, i])) for i in range(n)]
                        if mh.shape[0] == 6 else None)


class RoundReply:
    """What an async round keeps of its result: the accepted reply (nothing reads it after the round)."""
    reply: wire.StepReply | None = None


class RoundCore:
    """Rank-0 round logic shared by RemoteRound and the process proxy. A subclass sets:

      cfg, stats, N, W, M, layout (ReplyLayout), bcast_np (int32 reply being built), n_out_by_reqs, rounds,
      _last_epoch, _last_log, _err_logged, client, slots, breaker, gatherer (msg_np, max_rows, row_bytes,
      collect(n, R); gather_from_ring for synchronous resyncs), is_sender, async_mode;
      async only: sync (HostFlagCore: wait_gathered, taken, publish, fail_all, values, watchdog_stats), _stop
      (is_set()), _meta_slots (metadata slots of the gatherer: round t uses t % _meta_slots),
      _staged, _dead, resync_np, and either _resync_lock + _resync_req or its own _request_resync / _clear_resync.

    The round only READS the gather staging (gatherer.msg_np rows, resync_np windows): the STEP headers and entries
    go into a private buffer (_step_prefix) and resync rows into another (_resync_rows), and the message is sent as
    [headers, rows] (client.step rows=). In async mode the watchdog can complete a round while the proxy is inside
    it, and the engine stream then refills the same staging with the next round's rows; a proxy that wrote there
    would corrupt that later round's rows, silently and persistently (S11 P-1). A late round reads torn rows at
    worst: its reply is discarded (lost claim) and every slot resends its window.
    """

    # ------------------------------------------------------------------ hooks

    def _bcast_writable(self) -> None:
        """Wait until bcast_np may be rewritten (RemoteRound: the last pinned copy out of it ran)."""

    def _request_resync(self, slots: set[int]) -> None:
        """Async: ask the engine side to stage these slots' ring windows with a later round's gather."""
        with self._resync_lock:
            self._resync_req |= slots

    def _clear_resync(self, staged: frozenset[int]) -> None:
        """Async: these slots' windows were staged with the round being run."""
        with self._resync_lock:
            self._resync_req -= staged

    # ------------------------------------------------------------------ round

    def n_out_for(self, n: int) -> int:
        """ReqEntry.n_out for a round of n requests (0 = all N positions)."""
        t = self.n_out_by_reqs
        if not t:
            return 0
        c = int(t[min(n, len(t) - 1)])
        return 0 if c >= self.N else max(1, c)

    def _pack_failed(self, n: int) -> None:
        self._bcast_writable()
        b = self.bcast_np
        b[:] = 0
        b[0] = STATUS_ROUND_FAILED
        b[1] = n

    def _after_gather(self, bv: BatchView, g, res: RoundReply, step_id: int) -> int:
        assert self.gatherer is not None and self.slots is not None and self.client is not None
        n = bv.n
        # rank-0 bookkeeping of the ring (CONTRACT 7.1, 7.3). ready + epoch are read once, together:
        # every entry below is built for this connection, and step() refuses any newer one
        epoch = self.client.connection_epoch()
        if epoch is not None and epoch != self._last_epoch:
            self._last_epoch = epoch
            self.slots.mark_all_dirty()
        self.stats.inc(f"rounds_{self._phase(bv)}")
        slots = self.slots
        anchors_pos = self._apply_ring(bv, g)
        assert self.breaker is not None
        if not self.breaker.allow():
            # open breaker: no network this round (CONTRACT 10); the ring above stays exact
            if self.breaker.state == "half_open":
                self.stats.set("breaker", self.breaker.snapshot())
            return self._fail_round(bv, "cooldown")
        if epoch is None:
            reason = "disabled" if self.client.disabled else "not_ready"
            return self._fail_round(bv, reason)
        entries: list[wire.ReqEntry] = []
        spans: list[tuple[int, int, int]] = []
        resync = False
        n_out = self.n_out_for(n)
        for i in range(n):
            s = int(bv.slots[i])
            st = slots[s]
            v = int(g.valid[i])
            anchor_pos = int(anchors_pos[i])
            flags = wire.RF_NEED_DRAFT if bv.need_draft[i] else 0
            first, k = int(g.first_pos[i]), v
            if st.dirty or not st.announced:
                # the worker's window is unknown or new: (re)send the whole ring window,
                # which already holds this round's rows (CONTRACT 7.4)
                flags |= wire.RF_NEW_GEN
                if not st.rng.empty:
                    pos = st.rng.positions(anchor_pos)
                    if (pos.start, len(pos)) != (first, k):
                        first, k = pos.start, len(pos)
                        flags |= wire.RF_RESYNC
                        resync = True
            if k == 0:
                # nothing to send for this request (cannot happen for a scheduled
                # request); an empty NEW_GEN entry would only fail BAD_ENTRY
                return self._fail_round(bv, "no_rows")
            entries.append(wire.ReqEntry(slot=s, generation=st.generation, first_pos=first,
                                         n_rows=k, anchor_token=int(g.anchor[i]),
                                         anchor_pos=anchor_pos, flags=flags,
                                         k_window=int(min(bv.k_window[i], 255)),
                                         n_out=n_out))
            spans.append((s, first, k))
        n_rows = sum(e.n_rows for e in entries)
        if resync:
            if self.async_mode:
                # the whole message is rebuilt from the ring: every slot's window must have been
                # staged by the engine stream this round; otherwise ask for it and fail this round
                need = {sl for sl, _, _ in spans}
                if not need <= self._staged:
                    self._request_resync(need)
                    return self._fail_round(bv, "resync_pending")
                rows = self._rows_from_staged(spans)
                got = rows.shape[0]
            else:
                got = self.gatherer.gather_from_ring(n, spans)
                rows = self._staged_rows(n, got)
            if got != n_rows:
                raise RuntimeError(f"resync gathered {got} rows, expected {n_rows}")
            self.stats.inc("resync_rounds")
            self.stats.inc("resync_rows", n_rows)
        elif n_rows != g.n_valid:
            raise RuntimeError(f"{n_rows} rows to send but {g.n_valid} valid rows staged")
        else:
            rows = self._staged_rows(n, n_rows)
        if self.async_mode and self.sync.taken(step_id):
            # completed by the watchdog (or a fail-all) while this proxy built it: the stream may already be
            # refilling the staging for a later round, so the rows are not sent (the publish below loses too)
            return self._fail_round(bv, "preempted")
        step_flags = wire.SF_WANT_MARGIN if self.cfg.want_margin else 0
        if self.cfg.sampled_top32:
            if g.sample_meta is None:
                raise RuntimeError("sampled mode needs same-round seed and target-temperature metadata")
            step_flags |= wire.SF_SAMPLED
        out: StepOutcome = self.client.step(self._step_prefix(), entries, n_rows, step_id=step_id,
                                            step_flags=step_flags, epoch=epoch, rows=rows,
                                            sample_meta=g.sample_meta if self.cfg.sampled_top32 else None)
        if not out.ok:
            if out.reason not in ("not_ready", "disabled", "reconnected"):   # sent, and failed
                self._breaker_record(False, out.reason or "unknown")
            return self._fail_round(bv, out.reason or "unknown")
        self.stats.observe("rtt_us", out.rtt_us)
        self.stats.observe(f"rtt_us_{self._phase(bv)}", out.rtt_us)
        self.stats.observe("worker_us", out.reply.worker_us)
        self.stats.inc("rows_sent", n_rows)
        self.stats.inc(f"rows_sent_{self._phase(bv)}", n_rows)
        if n_out:
            self.stats.inc("rounds_n_out_cut")
        status = self._accept_reply(bv, entries, out.reply, res)
        needed = bv.need_draft[:n].astype(bool)
        if needed.any():
            # failed for the breaker when every request that needed drafts failed on the
            # worker. A round with nothing to draft (prefill appends only) never exercises
            # drafting: neutral, so it neither resets the count nor closes a probe
            usable = np.isin(out.reply.status, list(wire.USABLE_REQ_STATUS))
            self._breaker_record(bool(usable[needed].any()), "every request failed")
        return status

    def _apply_ring(self, bv: BatchView, g) -> np.ndarray:
        """The round's rows in rank 0's model of the device ring (owner, generation, [lo, hi]); returns the anchor
        positions."""
        slots = self.slots
        anchors_pos = g.first_pos + g.valid
        for i in range(bv.n):
            s = int(bv.slots[i])
            st = slots[s]
            slots.begin_request(s, bv.req_ids[i])
            v = int(g.valid[i])
            if v <= 0:
                st.rng.clear()
                st.dirty = True
                continue
            try:
                st.rng.apply(int(g.first_pos[i]), v, int(anchors_pos[i]))
            except ValueError as e:
                log.warning("slot %d ring bookkeeping: %s", s, e)
                st.rng.clear()
                st.dirty = True
        return anchors_pos

    def _phase(self, bv: BatchView | None) -> str:
        """decode: every request's query is a verify span (<= N + 1 tokens); prefill: the step carried a longer chunk."""
        if bv is None:
            return "unknown"
        return "prefill" if bv.longest_query > self.N + 1 else "decode"

    def _breaker_record(self, ok: bool, why: str) -> None:
        assert self.breaker is not None
        change = self.breaker.record(ok)
        if change is None:
            return
        self.stats.set("breaker", self.breaker.snapshot())
        if change == "tripped":
            self.stats.inc("breaker_trips")
            log.warning("remote DSpark circuit breaker open (%s; trip %d): no network for the "
                        "next %d rounds, then one probe", why, self.breaker.trips,
                        self.breaker.cooldown)
        else:
            self.stats.inc("breaker_recoveries")
            log.info("remote DSpark circuit breaker closed: the probe round succeeded")

    def _fail_round(self, bv: BatchView, reason: str) -> int:
        assert self.slots is not None
        for s in bv.slots:
            self.slots[int(s)].dirty = True
        self.stats.inc(f"fallback_{reason}")
        self.stats.inc("rounds_failed")
        self._pack_failed(bv.n)
        return STATUS_ROUND_FAILED

    def _accept_reply(self, bv: BatchView, entries: list[wire.ReqEntry],
                      reply: wire.StepReply, res: RoundReply) -> int:
        assert self.slots is not None
        n, N, L = bv.n, self.N, self.layout
        self._bcast_writable()
        b = self.bcast_np
        b[:] = 0
        b[1] = n
        okm = b[L.o_ok:L.o_ok + n]
        ids = b[L.o_ids:L.o_ids + n * N].reshape(n, N)
        conf = b[L.o_conf:L.o_conf + n * N].view(np.float32).reshape(n, N)
        all_ok = True
        for i, e in enumerate(entries):
            rs = int(reply.status[i])
            st = self.slots[e.slot]
            self.stats.inc(f"req_status_{wire.REQ_STATUS_NAMES.get(rs, rs)}".lower())
            usable = rs in wire.USABLE_REQ_STATUS
            if usable:
                st.announced = True
                # a partial window is resent only if the ring holds more than the worker
                # has; otherwise it is genuine (rows never existed) and only counted
                st.dirty = (rs == wire.RS_PARTIAL_WINDOW
                            and int(reply.window_rows[i]) < st.rng.rows_for_anchor(
                                e.anchor_pos))
            else:
                st.dirty = True
            if not bv.need_draft[i]:
                continue
            if usable:
                okm[i] = 1
                ids[i] = reply.draft_ids[i]
                conf[i] = reply.confidence[i]
                if L.sampled:
                    if reply.support_ids is None or reply.proposal_q is None or reply.sampled_mask is None:
                        raise RuntimeError("sampled q missing before publish")
                    b[L.o_sample_mask+i] = reply.sampled_mask[i]
                    b[L.o_qids:L.o_qids+n*N*32].reshape(n,N,32)[i] = reply.support_ids[i]
                    b[L.o_q:L.o_q+n*N*32].view(np.float32).reshape(n,N,32)[i] = reply.proposal_q[i]
            else:
                all_ok = False
        res.reply = reply
        stale = [e.slot for i, e in enumerate(entries)
                 if int(reply.status[i]) == wire.RS_STALE_GENERATION]
        if stale:
            # rank 0's generations only grow (CONTRACT 7.1), so the worker's slot state does
            # not match ours; a resync under the same session would stay stale
            self.stats.inc("stale_generation_new_sessions")
            self.client.new_session(f"worker reported STALE_GENERATION for slots {stale}")
        self.stats.inc("rounds_ok")
        status = STATUS_OK if all_ok else STATUS_SOME_FAILED
        if not all_ok:
            self.stats.inc("rounds_some_failed")
        b[0] = status
        return status

    # ------------------------------------------------------------------ async proxy (thread or process)

    def _wait_gathered(self, t: int) -> bool:
        """Proxy thread: True once the engine stream finished round t's gather; False when
        stopping. A prefill chunk's forward can take seconds, so there is no give-up."""
        assert self.sync is not None
        t0 = time.monotonic()
        next_log = 30.0
        while not self._stop.is_set():
            if self.sync.wait_gathered(t, 0.1):
                return True
            waited = time.monotonic() - t0
            if waited >= next_log:
                next_log += 30.0
                log.warning("remote DSpark async: the engine stream has not finished round %d's "
                            "gather after %.0f s (words: %s)", t, waited, self.sync.values())
        return False

    def _async_round(self, item) -> bool:
        """One round on the proxy thread. False when close() stopped it before round t was gathered."""
        assert self.sync is not None
        t, bv, R, staged = item
        self._clear_resync(staged)     # a request made below persists to the next round
        t0 = time.perf_counter()
        # outside the per-round handler: a failing wait means the hand-off itself is broken (dies)
        if not self._wait_gathered(t):
            return False
        # from here the reply buffer may be written: the stream finished round t-1's reply copy
        waited_us = (time.perf_counter() - t0) * 1e6
        self.stats.observe("gather_us", waited_us)
        if self.sync.taken(t):
            self._skip_taken_round(t, bv, R, waited_us)
            self.stats.observe("rank0_round_us", (time.perf_counter() - t0) * 1e6)
            self._maybe_log()
            return True
        try:
            if bv is None:
                raise RuntimeError("enqueue failed on the engine thread")
            g = self.gatherer.collect(bv.n, R, t % self._meta_slots)
            res = RoundReply()
            self._staged = staged
            self._after_gather(bv, g, res, step_id=t)
        except Exception as e:  # the round fails closed; the thread goes on
            n = bv.n if bv is not None else 0
            self._pack_failed(n)
            if bv is not None:
                self.stats.inc("fallback_internal_error")
            if self.slots is not None:
                self.slots.mark_all_dirty()
            now = time.monotonic()
            if now - self._err_logged > 10.0:
                self._err_logged = now
                log.exception("remote DSpark async round %d failed: %s", t, e)
        finally:
            self._staged = frozenset()
        if not self.sync.reply_consumed(t - 1):
            # stream order makes gathered >= t imply that round t-1's reply was copied; a missing acknowledgment
            # means the hand-off's ordering is broken, and writing the page now could tear that copy
            raise RuntimeError(f"round {t}: the engine stream has not acknowledged round {t - 1}'s reply "
                               f"(words {self.sync.values()})")
        # claim t, copy the reply into the page, release t: one GIL-free C call (S9 F5)
        if not self.sync.publish(t, self.bcast_np):
            # the watchdog failed round t first: the worker may have taken rows we now treat as
            # unsent, so every slot re-announces its window
            if self.slots is not None:
                self.slots.mark_all_dirty()
            self.stats.inc("watchdog_preempted")
            self.stats.inc(f"watchdog_preempted_{self._phase(bv)}")
            self._note_watchdog_event(t, bv, R, "late_publish", waited_us)
        done_us = (time.perf_counter() - t0) * 1e6
        self.stats.observe("rank0_round_us", done_us)
        # gathered seen -> published: how long the engine stream stays parked on this round, by phase
        self.stats.observe(f"proxy_parked_us_{self._phase(bv)}", done_us - waited_us)
        self._maybe_log()
        return True

    def _skip_taken_round(self, t: int, bv: BatchView | None, R: int, waited_us: float) -> None:
        """Round t was completed (failed) by the watchdog or a fail-all before the proxy reached it. The stream has
        gone on and may already be writing a later round's rows into the staging, so no row is read and nothing is
        sent. The ring bookkeeping still runs, from round t's own metadata slot, and the round's slots resend their
        windows (NEW_GEN, resync from the ring) in a later round. A proxy more than META_SLOTS rounds behind cannot
        trust that slot either: those slots start their windows afresh."""
        self.stats.inc("watchdog_preempted")
        self.stats.inc(f"watchdog_preempted_{self._phase(bv)}")
        self.stats.inc("watchdog_skipped_rounds")
        self._note_watchdog_event(t, bv, R, "skipped", waited_us)
        if bv is None or self.slots is None:
            return
        if self.rounds - t >= self._meta_slots:
            for i in range(bv.n):
                s = int(bv.slots[i])
                self.slots.begin_request(s, bv.req_ids[i])
                self.slots[s].rng.clear()
                self.slots[s].dirty = True
            self.stats.inc("watchdog_skipped_unknown_ring")
            return
        self._apply_ring(bv, self.gatherer.collect(bv.n, R, t % self._meta_slots))
        for s in bv.slots:
            self.slots[int(s)].dirty = True

    def _note_watchdog_event(self, t: int, bv: BatchView | None, R: int, how: str, waited_us: float) -> None:
        """The last WATCHDOG_EVENTS_KEPT rounds the watchdog completed, with wall time and batch shape, so a run can
        attribute them to its stages (stats key watchdog_events)."""
        ev = getattr(self, "_wd_events", None)
        if ev is None:
            ev = self._wd_events = []
        ev.append({"round": t, "time": round(time.time(), 3), "how": how, "phase": self._phase(bv),
                   "n": bv.n if bv is not None else 0, "rows": int(R),
                   "longest_query": bv.longest_query if bv is not None else 0,
                   "proxy_wait_ms": round(waited_us / 1e3, 3)})
        del ev[:-WATCHDOG_EVENTS_KEPT]
        self.stats.set("watchdog_events", list(ev))

    def _die(self, reason: str) -> None:
        """The proxy thread is ending abnormally: fail every pending and later round first (one
        C call: failed reply, monotonic far-ahead release), then record it."""
        assert self.sync is not None
        self.sync.fail_all(self.rounds + FAR_AHEAD)
        if self._dead is None:
            self._dead = reason
            try:
                self.stats.set("async_dead", reason)
                log.error("remote DSpark async proxy thread stopped (%s): every later round is "
                          "target-only (zero drafts)", reason)
            except Exception:  # noqa: BLE001 -- logging must not undo the release above
                pass

    def _rows_from_staged(self, spans: list[tuple[int, int, int]]) -> np.ndarray:
        """Async resync: the STEP rows for spans [(slot, first_pos, n_rows)] from the windows the engine stream staged
        (= gatherer.gather_from_ring on the CPU, same ring indexing), built in the round's private buffer: never in
        the shared STEP staging, which a later round may already own (S11 P-1). Returns [total, row_bytes] uint8."""
        assert self.gatherer is not None
        g, W = self.gatherer, self.W
        total = sum(k for _, _, k in spans)
        if total > g.max_rows:
            raise ValueError(f"{total} resync rows > max_rows {g.max_rows}")
        buf = getattr(self, "_resync_rows", None)
        if buf is None or buf.shape[0] < total:
            buf = self._resync_rows = np.empty((total, g.row_bytes), dtype=np.uint8)
        rows = buf[:total]
        o = 0
        for sl, p, k in spans:
            rows[o:o + k] = self.resync_np[sl][(p + np.arange(k, dtype=np.int64)) % W]
            o += k
        return rows

    def _staged_rows(self, n_req: int, n_rows: int) -> np.ndarray:
        """The first n_rows rows the gather staged for a round of n_req requests: a read-only use of the staging
        (uint8 view, no copy)."""
        g = self.gatherer
        off = wire.APP_HEADER_SIZE + wire.rows_offset_for(n_req, self.cfg.sampled_top32)
        return g.msg_np[off:off + n_rows * g.row_bytes]

    def _step_prefix(self) -> bytearray:
        """The round's private buffer for the STEP's AppHeader, StepHeader and entries (client.step rows=)."""
        b = getattr(self, "_prefix", None)
        if b is None:
            b = self._prefix = bytearray(wire.APP_HEADER_SIZE + wire.rows_offset_for(self.M, self.cfg.sampled_top32))
        return b

    # ------------------------------------------------------------------ logging

    def _maybe_log(self) -> None:
        every = self.cfg.log_every
        if every <= 0 or self.rounds % every:
            return
        if self.breaker is not None:
            self.stats.set("breaker", self.breaker.snapshot())
        if self.async_mode:
            self.stats.set("async_watchdog", self.sync.watchdog_stats())
        snap = self.stats.snapshot()
        snap["rounds"] = self.rounds
        c = snap["counters"]
        lat = snap["latency"]
        log.info(
            "remote DSpark %s: %d rounds, ok %d, failed %d, resync %d; rtt p50/p99 %s/%s us, "
            "worker p50 %s us, gather p50 %s us, rank-0 round p50 %s us",
            self.cfg.policy, self.rounds, c.get("rounds_ok", 0), c.get("rounds_failed", 0),
            c.get("resync_rounds", 0), lat.get("rtt_us", {}).get("p50_us"),
            lat.get("rtt_us", {}).get("p99_us"), lat.get("worker_us", {}).get("p50_us"),
            lat.get("gather_us", {}).get("p50_us"), lat.get("rank0_round_us", {}).get("p50_us"))
        self.write_stats()

    def write_stats(self) -> None:
        if not self.is_sender or not self.cfg.stats_file:
            return
        self.stats.set("rounds", self.rounds)
        if self.breaker is not None:
            self.stats.set("breaker", self.breaker.snapshot())
        if self.async_mode:
            self.stats.set("async_watchdog", self.sync.watchdog_stats())
        try:
            self.stats.write_json(self.cfg.stats_file)
        except OSError as e:
            log.warning("remote DSpark stats file %s: %s", self.cfg.stats_file, e)
