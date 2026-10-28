"""The DSpark worker: CONTRACT.md v1.3 (lane O3) over lane O1's ofl transport, the drafter on one RTX 3090.

State machine (DraftCore.handle(frame) -> reply frame) follows the contract and lane O3's reference worker
(remote_dspark/mock_worker.py WorkerCore) rule for rule: HELLO identity / config checks, sessions, strictly increasing
seq, per-slot generations (u32 serial arithmetic; older generations are stale even with NEW_GEN), the 7.2 window
rule (window.py), positions bounded by the session's max_model_len, PARTIAL_WINDOW, RESET / STATS / PING,
WANT_MARGIN, n_out (1.3). It exposes the same test surface (inject, slots[i].rng, step_log) so lane O3's tests can drive it
(tools/run_o3_conformance.py). The drafter sits behind a backend:
  * GpuBackend: rows -> main_x -> window KV on the GPU (incremental: only the rows a STEP carries), batched drafts
    through CUDA graphs per (batch size, k, n_out), with and without margins (WANT_MARGIN). If the CUDA context
    dies, the worker answers INTERNAL and exits with status 3 so systemd restarts it.
  * FakeBackend: lane O3's fake_draft over the raw rows (conformance tests only).

Transports: ofl (service dspark-v1, TCP + RDMA backends, zero-copy payload) on the direct-link addresses only, and the
contract's length-prefixed TCP stub (optional). One lock serializes all frames; within a connection ofl answers in
order; frames of a superseded connection are refused by seq.

Run (x86host):  CUDA_VISIBLE_DEVICES=GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93 python -m worker.server \\
                  --root /opt/dsv41/x86host-data/dspark --ofl 10.202.28.1:18620
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import signal
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from . import protocol as P
from .window import WindowRange

log = logging.getLogger("dspark_worker")

GPU_UUID = "GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93"
SERVICE = "dspark-v1"
MAX_REQ = 16 << 20
MAX_RESP = 64 << 10
DIRECT_LINK_IPS = ("10.202.28.1", "10.202.29.1")


def check_bind(ip: str) -> None:
    """COMMON rule 9: listeners only on the direct-link addresses. DSPARK_WORKER_ALLOW_LOOPBACK=1 additionally allows
    127.0.0.0/8 (tests on one host); nothing else, never a wildcard."""
    if ip in DIRECT_LINK_IPS:
        return
    if os.environ.get("DSPARK_WORKER_ALLOW_LOOPBACK") == "1":
        import ipaddress
        try:
            if ipaddress.ip_address(ip).version == 4 and ipaddress.ip_address(ip).is_loopback:
                return
        except ValueError:
            pass
    raise ValueError(f"bind {ip}: listeners only on the direct links {DIRECT_LINK_IPS}")


# ------------------------------------------------------------------------------------------------ state


@dataclass
class Inject:
    """Fault injection (tests): the same knobs as lane O3's mock worker."""
    delay_ms: float = 0.0
    frame_status_once: int | None = None
    req_status_once: dict[int, int] = field(default_factory=dict)
    drop_next_step: bool = False


@dataclass
class SlotState:
    window: int
    generation: int | None = None
    rng: WindowRange = field(init=False)

    def __post_init__(self) -> None:
        self.rng = WindowRange(self.window)


class Counters:
    def __init__(self) -> None:
        self.values: dict[str, int] = {}

    def inc(self, key: str, n: int = 1) -> None:
        self.values[key] = self.values.get(key, 0) + n


def expected_hello(drafter_identity: bytes, markov_identity: bytes, **kw) -> P.Hello:
    """The live configuration (CONTRACT 6.2); kw overrides for tests."""
    base = dict(proto_min=1, proto_max=1, num_spec=5, window=128, hidden=5120, n_taps=3, tap_layer_ids=(37, 38, 39, 0),
                payload_kind=P.PAYLOAD_TAPS_BF16, row_bytes=3 * 5120 * 2, noise_token_id=128799, vocab_size=129280,
                max_model_len=327680, max_slots=4, max_rows_per_step=512, sample_mode=P.SAMPLE_GREEDY, tp_size=2,
                drafter_identity=drafter_identity, markov_identity=markov_identity, client_name="")
    base.update(kw)
    return P.Hello(**base)


def _config_tuple(h: P.Hello) -> tuple:
    return (h.num_spec, h.window, h.hidden, h.n_taps, tuple(h.tap_layer_ids[:h.n_taps]), h.payload_kind,
            h.row_bytes, h.noise_token_id, h.vocab_size, h.sample_mode)


class DraftCore:
    """Transport-independent contract state machine."""

    def __init__(self, expected: P.Hello, backend, *, worker_name: str = "o2-dspark-3090",
                 capabilities: int = P.CAP_TAPS_BF16 | P.CAP_HAS_MARGIN):
        self.expected = expected
        self.backend = backend
        self.worker_name = worker_name
        self.capabilities = capabilities
        self.session: int | None = None
        self.last_seq = 0
        self.max_model_len = expected.max_model_len      # the session's (HELLO), never above the worker's
        self.slots = [SlotState(expected.window) for _ in range(expected.max_slots)]
        self.counters = Counters()
        self.inject = Inject()
        self.lock = threading.Lock()
        self.fatal = threading.Event()          # the backend is unusable (dead CUDA context): main() exits non-zero
        self.started = time.time()
        self.step_log: list[dict] = []
        self.keep_step_log = False
        self.worker_us: list[int] = []

    # ------------------------------------------------------------------ entry
    def handle(self, buf) -> bytes | None:
        t_in = time.perf_counter()
        with self.lock:
            try:
                return self._handle(buf, t_in)
            except Exception:       # noqa: BLE001  -- fail loudly in the log, answer INTERNAL, keep serving
                log.exception("frame failed")
                self.counters.inc("internal_errors")
                # S5R fix 2: a dead CUDA context can first surface in any backend call (STEP's slot reset, HELLO's
                # reset and clamp, RESET), not only in flush/draft: classify the backend before anything else, so
                # no later cleanup can keep main() from exiting 3 for systemd to restart the worker
                self._classify_backend("frame")
                try:
                    h = P.AppHeader.decode(buf)
                    return P.reply(h, P.INTERNAL)
                except P.ProtocolError:
                    return P.AppHeader(P.MAGIC, P.VERSION, 0, 0, 0, 0, P.MALFORMED, 0).encode()

    def _handle(self, buf, t_in: float) -> bytes | None:
        mv = memoryview(buf)
        try:
            h = P.AppHeader.decode(mv)
            if h.magic != P.MAGIC or h.version != P.VERSION or h.body_len != len(mv) - P.APP_HDR.size:
                raise P.ProtocolError(f"magic {h.magic:#x} version {h.version} body_len {h.body_len} "
                                      f"for {len(mv)} B")
        except P.ProtocolError as e:
            self.counters.inc("malformed_frames")
            log.warning("malformed frame: %s", e)
            return P.AppHeader(P.MAGIC, P.VERSION, 0, 0, 0, 0, P.MALFORMED, 0).encode()
        body = mv[P.APP_HDR.size:]
        if h.op == P.OP_PING:
            return P.reply(h, P.OK)
        if h.op == P.OP_HELLO:
            return self._hello(h, body)
        if h.op not in (P.OP_STEP, P.OP_RESET, P.OP_STATS):
            return P.reply(h, P.UNSUPPORTED_OP)
        if self.session is None or h.session != self.session:
            self.counters.inc("no_session")
            return P.reply(h, P.NO_SESSION)
        if h.seq <= self.last_seq:
            self.counters.inc("stale_seq")
            return P.reply(h, P.STALE_SEQ)
        self.last_seq = h.seq
        if h.op == P.OP_STEP:
            return self._step(h, body, t_in)
        if h.op == P.OP_RESET:
            if self.fatal.is_set():
                return P.reply(h, P.INTERNAL)
            if len(body) != 4:
                return P.reply(h, P.MALFORMED)
            (mask,) = struct.unpack("<I", bytes(body))
            for i, s in enumerate(self.slots):
                if mask >> i & 1:
                    self._reset_slot(i, None)
            self.counters.inc("resets")
            return P.reply(h, P.OK)
        return P.reply(h, P.OK, json.dumps(self.stats()).encode())

    def _classify_backend(self, where: str) -> None:
        """After a failure: if the backend is unusable (dead CUDA context), set fatal; a healthy backend keeps
        serving (transient failure). A health probe that itself raises counts as unusable."""
        if self.fatal.is_set():
            return
        try:
            ok = bool(self.backend.healthy())
        except Exception:           # noqa: BLE001
            log.exception("backend health probe raised after a %s failure", where)
            ok = False
        if not ok:
            log.critical("the GPU backend is unusable after a %s failure: stopping the worker (non-zero exit; "
                         "systemd restarts it)", where)
            self.fatal.set()

    def _reset_slot(self, i: int, generation: int | None) -> None:
        s = self.slots[i]
        s.generation = generation
        s.rng.clear()
        self.backend.reset(i)

    def stats(self) -> dict:
        us = sorted(self.worker_us[-4096:])
        out = {"worker": self.worker_name, "session": self.session, "uptime_s": round(time.time() - self.started, 1),
               **self.counters.values}
        if us:
            out["worker_us_p50"] = us[len(us) // 2]
            out["worker_us_p99"] = us[min(len(us) - 1, int(len(us) * 0.99))]
        out.update(self.backend.stats())
        return out

    # ------------------------------------------------------------------ HELLO
    def _hello(self, h: P.AppHeader, body) -> bytes:
        if self.fatal.is_set():
            return P.reply(h, P.INTERNAL)
        exp = self.expected
        rep = P.hello_reply(1, exp.max_slots, exp.max_rows_per_step, self.capabilities, exp.window, exp.num_spec,
                            exp.drafter_identity, exp.markov_identity, self.worker_name)
        try:
            req = P.Hello.decode(body)
        except P.ProtocolError:
            return P.reply(h, P.MALFORMED)
        if not req.proto_min <= 1 <= req.proto_max:
            return P.reply(h, P.BAD_VERSION, rep)
        if req.drafter_identity != exp.drafter_identity or req.markov_identity != exp.markov_identity:
            self.counters.inc("identity_mismatch")
            log.error("HELLO identity mismatch: drafter %s (mine %s), markov %s (mine %s)",
                      req.drafter_identity.hex(), exp.drafter_identity.hex(), req.markov_identity.hex(),
                      exp.markov_identity.hex())
            return P.reply(h, P.IDENTITY_MISMATCH, rep)
        if (_config_tuple(req) != _config_tuple(exp) or not 1 <= req.max_slots <= exp.max_slots
                or not 1 <= req.max_rows_per_step <= exp.max_rows_per_step
                or not 1 <= req.max_model_len <= exp.max_model_len):
            self.counters.inc("config_mismatch")
            log.error("HELLO config mismatch: %s vs mine %s", req, exp)
            return P.reply(h, P.CONFIG_MISMATCH, rep)
        if h.session == 0:
            return P.reply(h, P.MALFORMED)
        if h.session != self.session:
            for i in range(len(self.slots)):
                self._reset_slot(i, None)
            self.session = h.session
            self.last_seq = h.seq
            self.counters.inc("new_sessions")
            log.info("HELLO new session %#x from %s (seq %d)", h.session, req.client_name, h.seq)
        else:
            if h.seq <= self.last_seq:
                return P.reply(h, P.STALE_SEQ)
            self.last_seq = h.seq
            self.counters.inc("rehellos")
        self.backend.set_max_model_len(req.max_model_len)
        self.max_model_len = req.max_model_len
        return P.reply(h, P.OK, rep)

    # ------------------------------------------------------------------ STEP
    def _entry_problem(self, e: P.ReqEntry) -> str | None:
        """CONTRACT 7.2 step 2 checks that need no slot state, plus the model range; None if fine."""
        exp = self.expected
        if not 0 <= e.n_rows <= exp.window:
            return f"n_rows {e.n_rows} outside 0..{exp.window}"
        if not 0 <= e.anchor_pos <= self.max_model_len:
            return f"anchor_pos {e.anchor_pos} outside 0..max_model_len {self.max_model_len}"
        if e.n_rows and (e.first_pos < 0 or e.anchor_pos != e.first_pos + e.n_rows):
            return f"rows [{e.first_pos}, +{e.n_rows}) do not end at anchor_pos {e.anchor_pos}"
        if e.req_flags & P.REQ_NEED_DRAFT and not 0 <= e.anchor_token < exp.vocab_size:
            return f"anchor_token {e.anchor_token} outside the vocabulary"
        return None

    def _step(self, h: P.AppHeader, body, t_in: float) -> bytes | None:
        t0 = time.perf_counter()
        if self.fatal.is_set():
            return P.reply(h, P.INTERNAL)
        self.backend.discard()                  # rows a failed earlier frame may have left staged
        inj = self.inject
        if inj.frame_status_once is not None:
            st, inj.frame_status_once = inj.frame_status_once, None
            return P.reply(h, st)
        exp = self.expected
        try:
            req = P.decode_step(body, exp.row_bytes, exp.max_slots, exp.max_rows_per_step)
            for e in req.entries:
                if e.slot >= exp.max_slots:
                    raise P.ProtocolError(f"slot {e.slot} >= max_slots {exp.max_slots}")
        except P.CapacityError as e:
            self.counters.inc("capacity_steps")
            log.warning("STEP over capacity: %s", e)
            return P.reply(h, P.CAPACITY)
        except P.ProtocolError as e:
            self.counters.inc("malformed_steps")
            log.warning("malformed STEP: %s", e)
            return P.reply(h, P.MALFORMED)
        n, N = req.n_req, exp.num_spec
        statuses = [[P.R_OK, 0, e.generation & 0xFFFFFFFF] for e in req.entries]
        ids = np.zeros((n, N), dtype=np.int32)
        conf = np.zeros((n, N), dtype=np.float32)
        want_margin = bool(req.step_flags & P.STEP_WANT_MARGIN) and bool(self.capabilities & P.CAP_HAS_MARGIN)
        margin = np.zeros((n, N), dtype=np.float32) if want_margin else None
        # (entry index, slot, anchor token, anchor_pos, window range, positions consumed)
        drafts: list[tuple[int, int, int, int, WindowRange, int]] = []
        forced: dict[int, int | None] = {}
        for i, e in enumerate(req.entries):
            if self.keep_step_log:
                self.step_log.append({"step_id": req.step_id, "slot": e.slot, "flags": e.req_flags,
                                      "gen": e.generation, "first_pos": e.first_pos, "n_rows": e.n_rows,
                                      "anchor_pos": e.anchor_pos, "anchor_token": e.anchor_token})
            s = self.slots[e.slot]
            # CONTRACT 7.2 step 1: an older generation is stale even with NEW_GEN (same-generation NEW_GEN = resync)
            if s.generation is not None and P.serial_newer(s.generation, e.generation):
                statuses[i][0] = P.R_STALE_GENERATION
                self.counters.inc("stale_generation")
                continue
            # the entry's own fields, before any slot state changes (S5 P1): positions inside the session's model
            # range (context rows <= max_model_len - 1 index the RoPE table; queries are clamped to it)
            problem = self._entry_problem(e)
            if problem is not None:
                statuses[i][0] = P.R_BAD_ENTRY
                self.counters.inc("bad_entries")
                log.warning("slot %d: bad entry: %s", e.slot, problem)
                continue
            if s.generation is None or e.req_flags & P.REQ_NEW_GEN or e.generation != s.generation:
                self._reset_slot(e.slot, e.generation)
            try:
                s.rng.apply(e.first_pos, e.n_rows, e.anchor_pos)
            except ValueError as err:
                statuses[i][0] = P.R_BAD_ENTRY
                self.counters.inc("bad_entries")
                log.warning("slot %d: bad entry: %s", e.slot, err)
                self._reset_slot(e.slot, s.generation)
                continue
            if e.n_rows:
                self.backend.stage(e.slot, e.first_pos, req.rows[e.row_start: e.row_start + e.n_rows])
                self.counters.inc("rows", e.n_rows)
            have = s.rng.rows_for_anchor(e.anchor_pos)
            statuses[i][1] = have
            f = inj.req_status_once.pop(e.slot, None)
            forced[i] = f
            if f is not None:
                statuses[i][0] = f
                if f not in (P.R_OK, P.R_PARTIAL_WINDOW):
                    continue
            if e.req_flags & P.REQ_NEED_DRAFT:
                # CONTRACT 1.3: positions >= n_out are not consumed (0 = all N; larger values clamp to N)
                drafts.append((i, e.slot, e.anchor_token, e.anchor_pos, s.rng, N if e.n_out == 0 else min(e.n_out, N)))
                if f is None and have < s.rng.expected_rows(e.anchor_pos):
                    statuses[i][0] = P.R_PARTIAL_WINDOW
                    self.counters.inc("partial_windows")
        try:
            self.backend.flush()
            if drafts:
                d_ids, d_conf, d_margin = self.backend.draft(
                    [(slot, tok, apos, rng, k) for _, slot, tok, apos, rng, k in drafts], N, want_margin)
                for j, (i, *_rest, k) in enumerate(drafts):
                    ids[i, :k], conf[i, :k] = d_ids[j, :k], d_conf[j, :k]
                    if margin is not None:
                        margin[i, :k] = d_margin[j, :k]
                    self.counters.inc(f"n_out_{k}")
                self.counters.inc("drafted_requests", len(drafts))
        except Exception:           # noqa: BLE001
            log.exception("STEP %d: backend failed", req.step_id)
            self.counters.inc("internal_errors")
            self._classify_backend(f"STEP {req.step_id}")      # before any cleanup that touches the backend
            self.backend.discard()
            for i, e in enumerate(req.entries):        # nothing of this step can be trusted: windows resync
                if statuses[i][0] in (P.R_OK, P.R_PARTIAL_WINDOW):
                    statuses[i][0] = P.R_INTERNAL
                    self._reset_slot(e.slot, self.slots[e.slot].generation)
            ids[:] = 0
            conf[:] = 0.0
        self.counters.inc("steps")
        if inj.delay_ms > 0:
            time.sleep(inj.delay_ms / 1000.0)
        if inj.drop_next_step:
            inj.drop_next_step = False
            return None
        worker_us = int((time.perf_counter() - t0) * 1e6)
        self.worker_us.append(worker_us)
        if len(self.worker_us) > 65536:
            del self.worker_us[:32768]
        rts = int((time.perf_counter() - t_in) * 1e6)
        body_out = P.encode_step_reply(req.step_id, N, [tuple(x) for x in statuses], ids, conf, margin,
                                       worker_us, rts)
        return P.reply(h, P.OK, body_out)


# ------------------------------------------------------------------------------------------------ backends


class GpuBackend:
    """Rows -> window KV and drafts on the 3090 through model.DSparkDrafter."""

    def __init__(self, model, max_rows: int, row_bytes: int, graphs: bool = True, graph_reply: bool = False):
        import torch

        self.torch = torch
        self.m = model
        self.dev = model.dev
        self.row_bytes = row_bytes
        self.hidden3 = row_bytes // 2
        self.stage_buf = torch.empty((max_rows, row_bytes), dtype=torch.uint8, pin_memory=True)
        self.pending: list[tuple[int, int, np.ndarray]] = []
        self.copy_done = torch.cuda.Event()      # the last H2D out of stage_buf (it is reused by the next flush)
        self.copy_done.record()
        self.graphs = graphs
        if graph_reply and not graphs:
            raise ValueError("graph_reply requires CUDA graphs")
        self.graph_reply = graph_reply
        self.n_flush_rows = 0
        self.max_model_len = model.max_model_len

    def set_max_model_len(self, n: int) -> None:
        self.m.set_query_clamp(n)

    def reset(self, slot: int) -> None:
        self.pending = [p for p in self.pending if p[0] != slot]
        self.m.reset_slots(self.torch.tensor([slot], device=self.dev))

    def stage(self, slot: int, first_pos: int, rows: np.ndarray) -> None:
        self.pending.append((slot, first_pos, rows))

    def discard(self) -> None:
        self.pending = []

    def healthy(self) -> bool:
        """A trivial kernel and a sync: False once the CUDA context is dead (illegal address, Xid, ...)."""
        torch = self.torch
        try:
            torch.cuda.synchronize(self.dev)
            return float((torch.ones(1, device=self.dev) + 1).item()) == 2.0
        except Exception:           # noqa: BLE001
            log.exception("GPU health probe failed")
            return False

    def flush(self) -> None:
        torch = self.torch
        if not self.pending:
            return
        n = sum(r.shape[0] for _, _, r in self.pending)
        self.copy_done.synchronize()
        sb = self.stage_buf[:n].numpy()
        slot_rows = np.empty(n, dtype=np.int64)
        pos_rows = np.empty(n, dtype=np.int64)
        k = 0
        for slot, p0, rows in self.pending:
            c = rows.shape[0]
            sb[k:k + c] = rows
            slot_rows[k:k + c] = slot
            pos_rows[k:k + c] = np.arange(p0, p0 + c)
            k += c
        self.pending = []
        taps = self.stage_buf[:n].to(self.dev, non_blocking=True).view(torch.bfloat16).view(n, self.hidden3)
        self.copy_done.record()
        sr = torch.from_numpy(slot_rows).to(self.dev, non_blocking=True)
        pr = torch.from_numpy(pos_rows).to(self.dev, non_blocking=True)
        if self.graphs and n <= max(self.m.insert_graphs or {0: 0}):
            self.m.insert_taps_graphed(sr, pr, taps)
        else:
            for a in range(0, n, 128):
                b = min(n, a + 128)
                self.m.insert(sr[a:b], pr[a:b], self.m.main_x(taps[a:b]))
        self.n_flush_rows += n

    def draft(self, reqs, N: int, want_margin: bool):
        """reqs: (slot, anchor token, anchor pos, window range, k = positions consumed). The head runs for the batch's
        largest k (the caller zeroes each request's positions >= its own k)."""
        torch = self.torch
        B = len(reqs)
        n = max(k for *_r, k in reqs)
        host = np.array([[slot, tok, apos - 1, rng.read_lo(apos)] for slot, tok, apos, rng, _k in reqs],
                        dtype=np.int64)
        t = torch.from_numpy(host).to(self.dev, non_blocking=True)
        slots, anchors, last, lo = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
        mg = None
        packed = None
        if self.graphs and self.m.has_graph(B, N, n, margin=want_margin):
            out = self.m.draft_graphed(slots, anchors, last, N, ctx_lo=lo, n_out=n, margin=want_margin,
                                      packed_reply=self.graph_reply)
            if self.graph_reply:
                packed = out[0]
            else:
                ids, conf = out[0][:B], out[1][:B]
                mg = out[2][:B] if want_margin else None
        elif want_margin:
            ids, conf, mg = self.m.draft_with_margin(slots, anchors, last, N, ctx_lo=lo, n_out=n)
        else:
            ids, conf = self.m.draft(slots, anchors, last, N, ctx_lo=lo, n_out=n)
        if packed is None:
            parts = [ids.to(torch.float64), conf.to(torch.float64)] + ([mg.to(torch.float64)] if want_margin else [])
            packed = torch.cat(parts, 1)
        packed = packed.cpu().numpy()                                             # one D2H + sync
        margin = packed[:, 2 * N:].astype(np.float32) if want_margin else None
        return packed[:, :N].astype(np.int32), packed[:, N:2 * N].astype(np.float32), margin

    def stats(self) -> dict:
        torch = self.torch
        return {"gpu_mem_bytes": torch.cuda.memory_reserved(self.dev), "rows_inserted": self.n_flush_rows}


class FakeBackend:
    """Raw rows per slot + a draft function of the window rows (lane O3's fake_draft); conformance tests only."""

    def __init__(self, window: int, row_bytes: int, max_slots: int, draft_fn, vocab: int):
        self.window, self.row_bytes, self.draft_fn, self.vocab = window, row_bytes, draft_fn, vocab
        self.rows = [np.zeros((window, row_bytes), np.uint8) for _ in range(max_slots)]

    def set_max_model_len(self, n: int) -> None:
        pass

    def reset(self, slot: int) -> None:
        pass

    def stage(self, slot: int, first_pos: int, rows: np.ndarray) -> None:
        for k in range(rows.shape[0]):
            self.rows[slot][(first_pos + k) % self.window] = rows[k]

    def discard(self) -> None:
        pass

    def healthy(self) -> bool:
        return True

    def flush(self) -> None:
        pass

    def draft(self, reqs, N: int, want_margin: bool):
        ids = np.zeros((len(reqs), N), np.int32)
        conf = np.zeros((len(reqs), N), np.float32)
        for j, (slot, tok, apos, rng, _k) in enumerate(reqs):
            win = [(p, self.rows[slot][p % self.window].tobytes()) for p in rng.positions(apos)]
            ids[j], conf[j] = self.draft_fn(win, tok, apos, N, self.vocab)
        return ids, conf, (conf.copy() if want_margin else None)

    def stats(self) -> dict:
        return {}


# ------------------------------------------------------------------------------------------------ transports


def _load_ofl():
    """Lane O1's ofl.py: $OFL_PY_DIR, else offload/transport next to offload/dspark (merged branch), else the copy
    pinned by run/pin_ofl.sh, else O1's x86host tree."""
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.environ.get("OFL_PY_DIR"), os.path.normpath(os.path.join(here, "..", "..", "transport")),
              os.path.expanduser("~/dsv41-offload/o2/lib"),
              os.path.expanduser("~/dsv41-offload/o1/src/offload/transport")):
        if d and os.path.exists(os.path.join(d, "ofl.py")):
            if d not in sys.path:
                sys.path.insert(0, d)
            import ofl  # type: ignore[import-not-found]
            return ofl
    raise ImportError("ofl.py not found (set OFL_PY_DIR)")


class OflServer:
    """core.handle on an ofl server; the request payload is read in place (no copy) and the reply memmoved into the
    registered response buffer."""

    def __init__(self, core: DraftCore, bind_ip: str, port: int, rdma: bool = True):
        check_bind(bind_ip)
        ofl = _load_ofl()
        L = ofl.lib()
        self.core = core

        def _cb(user, req):
            try:
                n = ctypes.c_uint32()
                p = L.ofl_req_payload(req, ctypes.byref(n))
                buf = (ctypes.c_char * n.value).from_address(p) if n.value else bytearray()
                rep = core.handle(memoryview(buf))
                if rep is None:
                    return -1
                cap = ctypes.c_uint32()
                out = L.ofl_req_resp_buf(req, ctypes.byref(cap))
                if len(rep) > cap.value:
                    log.error("reply %d B > response capacity %d B", len(rep), cap.value)
                    return -2
                ctypes.memmove(out, rep, len(rep))
                L.ofl_req_set_len(req, len(rep))
                return 0
            except Exception:       # noqa: BLE001  -- never let an exception cross into C
                log.exception("ofl handler")
                return -3

        self._cb = ofl.HANDLER(_cb)
        mask = (1 << ofl.BACKEND_TCP) | ((1 << ofl.BACKEND_RDMA) if rdma else 0)
        cfg = ofl._ServerConfig(bind_ip.encode(), port, mask, None, 1, -1, SERVICE.encode(), MAX_REQ, MAX_RESP,
                                1, 4, 2_000_000)
        err = ctypes.create_string_buffer(512)
        self.h = L.ofl_server_create(ctypes.byref(cfg), self._cb, None, err, 512)
        if not self.h:
            raise RuntimeError(f"ofl_server_create {bind_ip}:{port}: {err.value.decode()}")
        self.L = L
        L.ofl_server_start(self.h)
        self.desc = L.ofl_server_describe(self.h).decode()

    def stop(self) -> None:
        if self.h:
            self.L.ofl_server_destroy(self.h)
            self.h = None


class TcpStubServer:
    """The contract's TCP stub (CONTRACT 5): u32 length + frame; the newest connection supersedes the previous one."""

    def __init__(self, core: DraftCore, host: str, port: int):
        if host in ("0.0.0.0", "", "::"):
            raise ValueError("bind a specific (direct-link) address, never 0.0.0.0")
        check_bind(host)
        self.core = core
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        self._conn: socket.socket | None = None
        self._stop = False
        self._lock = threading.Lock()
        self.connections = 0
        threading.Thread(target=self._accept, daemon=True, name="dspark-tcp-accept").start()

    def _accept(self) -> None:
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
            threading.Thread(target=self._serve, args=(conn,), daemon=True, name="dspark-tcp-conn").start()

    @staticmethod
    def _recv_exact(conn: socket.socket, view: memoryview) -> bool:
        got = 0
        while got < len(view):
            k = conn.recv_into(view[got:])
            if k == 0:
                return False
            got += k
        return True

    def _serve(self, conn: socket.socket) -> None:
        buf = bytearray(MAX_REQ + 64)
        mv = memoryview(buf)
        try:
            while not self._stop:
                if not self._recv_exact(conn, mv[:4]):
                    return
                (n,) = struct.unpack("<I", mv[:4])
                if n > MAX_REQ + 64 or not self._recv_exact(conn, mv[:n]):
                    return
                rep = self.core.handle(mv[:n])
                if rep is None:
                    return
                conn.sendall(struct.pack("<I", len(rep)) + rep)
        except OSError:
            return
        finally:
            try:
                conn.close()
            except OSError:
                pass

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


# ------------------------------------------------------------------------------------------------ main


def build_core(root: str, graphs: bool = True, tp: int = 1, max_model_len: int = 327680) -> tuple[DraftCore, object]:
    if tp == 2:
        return build_core_tp2(root, graphs, max_model_len=max_model_len)
    if tp != 1:
        raise SystemExit(f"--tp {tp}: 1 (one 3090) or 2 (the NVLink pair)")
    import torch

    from .checkpoint import Checkpoint
    from .identity import compute as compute_identity
    from .model import DSparkDrafter

    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if vis != GPU_UUID:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={vis!r}: this worker runs on {GPU_UUID} only (COMMON rule 3)")
    t0 = time.time()
    ck = Checkpoint(root)
    ident = compute_identity(root, ck.manifest)
    dev = torch.device("cuda", 0)
    # max_model_len sizes the RoPE table (max_model_len + 2048 rows) and caps what a session may announce in HELLO; an
    # engine with a smaller max_model_len is accepted and clamps to its own (set_max_model_len).
    model = DSparkDrafter(ck, dev, slots=8, max_model_len=max_model_len)
    exp = expected_hello(bytes.fromhex(ident["drafter_identity"]), bytes.fromhex(ident["markov_identity"]),
                         max_model_len=max_model_len)
    graph_reply = os.environ.get("DSPARK_GRAPH_REPLY", "0") == "1"
    if graphs:
        model.capture(batch_sizes=tuple(range(1, exp.max_slots + 1)), ks=(exp.num_spec,),
                      insert_rows=(8, 16, 32, 64, 128), n_outs=tuple(range(1, exp.num_spec + 1)),
                      pack_reply=graph_reply)
    backend = GpuBackend(model, exp.max_rows_per_step, exp.row_bytes, graphs=graphs, graph_reply=graph_reply)
    core = DraftCore(exp, backend)
    log.info("worker ready in %.1f s: drafter_identity %s, markov_identity %s, %.2f GiB reserved", time.time() - t0,
             ident["drafter_identity"], ident["markov_identity"], torch.cuda.memory_reserved(dev) / 2**30)
    return core, model


def build_core_tp2(root: str, graphs: bool = True, max_model_len: int = 327680) -> tuple[DraftCore, object]:
    """The TP = 2 worker (tp2.py, lane E) on the NVLink pair: CUDA_VISIBLE_DEVICES = c4e7a371,7744b44b (in that order;
    rank 0 = c4e7a371, PHB to the ConnectX-5). DSPARK_WORKER_TP_EMULATE=1 runs both ranks on GPU-7744b44b alone
    (tests only: one card's speed, the same contract and drafts). max_model_len as in build_core (each rank holds the
    replicated RoPE table: (max_model_len + 2048) x 64 fp32, 269 MB at 1,048,576)."""
    import torch

    from .checkpoint import Checkpoint
    from .identity import compute as compute_identity
    from .tp2 import TP2Backend, TPDrafter, check_visible

    emulate = os.environ.get("DSPARK_WORKER_TP_EMULATE") == "1"
    check_visible(pair=not emulate)
    t0 = time.time()
    ck = Checkpoint(root)
    ident = compute_identity(root, ck.manifest)
    devs = [torch.device("cuda", 0), torch.device("cuda", 0 if emulate else 1)]
    model = TPDrafter(ck, devs, slots=8, max_model_len=max_model_len)
    exp = expected_hello(bytes.fromhex(ident["drafter_identity"]), bytes.fromhex(ident["markov_identity"]),
                         max_model_len=max_model_len)
    if graphs:
        model.capture(batch_sizes=tuple(range(1, exp.max_slots + 1)), ks=(exp.num_spec,),
                      insert_rows=(8, 16, 32, 64, 128), n_outs=tuple(range(1, exp.num_spec + 1)))
    backend = TP2Backend(model, exp.max_rows_per_step, exp.row_bytes, graphs=graphs, max_slots=exp.max_slots)
    core = DraftCore(exp, backend, worker_name="e-dspark-3090x2" + ("-emulated" if emulate else ""))
    log.info("TP=2 worker ready in %.1f s: drafter_identity %s, markov_identity %s, %s GiB reserved per device",
             time.time() - t0, ident["drafter_identity"], ident["markov_identity"],
             [round(torch.cuda.memory_reserved(d) / 2**30, 2) for d in dict.fromkeys(devs)])
    return core, model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/opt/dsv41/x86host-data/dspark")
    ap.add_argument("--ofl", action="append", default=[], help="ip:port for an ofl listener (TCP + RDMA)")
    ap.add_argument("--tcp-stub", action="append", default=[], help="ip:port for the contract's TCP stub")
    ap.add_argument("--no-rdma", action="store_true")
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--tp", type=int, default=int(os.environ.get("DSPARK_WORKER_TP", "1")),
                    help="1: one 3090 (default); 2: the NVLink pair (tp2.py; or DSPARK_WORKER_TP=2)")
    ap.add_argument("--max-model-len", type=int, default=327680,
                    help="largest max_model_len a session may announce (engine MAX_LEN); sizes the RoPE table")
    ap.add_argument("--status-file", default=os.path.expanduser("~/dsv41-offload/o2/run/worker.status.json"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not args.ofl and not args.tcp_stub:
        raise SystemExit("no listener: give --ofl and/or --tcp-stub")
    if not 1 <= args.max_model_len <= 1048576:
        raise SystemExit(f"--max-model-len {args.max_model_len} outside 1..1048576 (the model's max_position_embeddings)")
    core, model = build_core(args.root, graphs=not args.no_graphs, tp=args.tp, max_model_len=args.max_model_len)
    servers = []
    for spec in args.ofl:
        ip, port = spec.rsplit(":", 1)
        s = OflServer(core, ip, int(port), rdma=not args.no_rdma)
        log.info("ofl listener %s", s.desc)
        servers.append(s)
    for spec in args.tcp_stub:
        ip, port = spec.rsplit(":", 1)
        s = TcpStubServer(core, ip, int(port))
        log.info("TCP stub listener %s:%d", ip, s.port)
        servers.append(s)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    os.makedirs(os.path.dirname(args.status_file), exist_ok=True)
    tick = 0
    while not stop.is_set() and not core.fatal.is_set():
        if tick % 10:                             # status snapshot at once, then every 5 s
            tick += 1
            stop.wait(0.5)
            continue
        tick += 1
        try:
            with core.lock:
                st = core.stats()
            st["listeners"] = args.ofl + args.tcp_stub
            st["pid"] = os.getpid()
            st["time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            tmp = args.status_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(st, f)
            os.replace(tmp, args.status_file)
        except Exception:           # noqa: BLE001
            log.exception("status file")
    log.info("stopping")
    for s in servers:
        s.stop()
    if core.fatal.is_set():
        log.critical("exit 3: GPU backend unusable")
        sys.exit(3)


if __name__ == "__main__":
    main()
