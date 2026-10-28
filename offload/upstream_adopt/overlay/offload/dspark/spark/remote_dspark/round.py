# SPDX-License-Identifier: Apache-2.0
"""One remote draft round, vLLM-agnostic (the TP broadcast is injected).

Rank 0: gather rows (gather.py) -> slot bookkeeping (state.py) -> STEP (client.py) ->
pack [status, n, ok_mask, drafts, confidences] -> broadcast. Other ranks: broadcast only.
Both ranks then read the same device tensor, so every draft token and every fallback
decision is identical across TP ranks (CONTRACT 4.1).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import secrets
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from . import wire
from .breaker import CircuitBreaker
from .client import RemoteDraftClient
from .config import RemoteConfig
from .gather import BatchView, RowGatherer, batch_view_from_input_batch
from .hostflag import _V, set_timer_slack
from .roundcore import (  # noqa: F401 -- STATUS_* are re-exported (speculator.py, tests)
    META_SLOTS, STATUS_NAMES, STATUS_OK, STATUS_ROUND_FAILED, STATUS_SOME_FAILED, ReplyLayout, RoundCore)
from .proxy_proc import (START_TIMEOUT_S, ProxyProcess, ProxyStartError, encode_init, merge_snapshots,
                         monotonic_ms, unmerge_snapshot, write_json_atomic)
from .shmregion import (C_DESC_HEAD, C_DESC_TAIL, C_DONE, C_ENGINE_SEQ, C_FLUSH_ACK, C_FLUSH_REQ, C_HEARTBEAT, C_PID,
                        C_START, C_STATE, C_STOP, C_WD, F_ENQUEUE_FAILED, R_PENDING, S_STARTING, RegionLayout)
from .suffix_provenance import Snapshot, SuffixAudit, SuffixRow, agree
from .state import SlotTable
from .stats import Stats
from .streamsync import FAR_AHEAD, release_hand_off

log = logging.getLogger("vllm.remote_dspark")

# process proxy stall detection (S11 P-3): the proxy is declared lost after this many watchdog fires in a row during
# which it finished no round (C_DONE did not move), or when its heartbeat is older than heartbeat_limit_ms()
STALL_FIRES = 3
_M32 = 0xFFFFFFFF
# process proxy respawn (S11 P-4): no new proxy sooner than this after a loss or a failed start
RESPAWN_DELAY_S = 1.0
LOSSES_KEPT = 16


def stall_fires(cfg: RemoteConfig) -> int:
    """Watchdog fires in a row, with no round finished, that mean a stalled proxy: STALL_FIRES, and more when an explicit
    deadline is shorter than the network timeout (one slow but live round may then span several deadlines; the auto
    deadline is >= 4 x the timeout)."""
    return max(STALL_FIRES, -(-cfg.timeout_ms // cfg.watchdog_ms) + 2)


def heartbeat_limit_ms(cfg: RemoteConfig) -> int:
    """Age at which the process proxy's heartbeat means it is stalled: 2 x the watchdog deadline, and never less than
    one network round plus two housekeeping polls (the proxy's main thread beats between rounds and every 100 ms while
    it waits; inside a round it is blocked for at most the client timeout)."""
    return max(2 * cfg.watchdog_ms, cfg.timeout_ms + 250)


def _new_session(old: int | None) -> int:
    """A worker session id (u32, not 0) other than old."""
    while True:
        s = secrets.randbits(32)
        if s and s != old:
            return s


@dataclass
class _Respawn:
    """A lost process proxy being replaced (S11 P-4). Engine thread only."""
    old: ProxyProcess             # the lost (or failed) process, SIGKILLed; the words are reset only once it is reaped
    drain: object                 # sync.drain_marker() at the loss: every stream operation of the lost proxy's rounds
    lost_at: float
    due: float                    # no new start before this
    new: ProxyProcess | None = None
    started_at: float = 0.0
    warned: bool = False


def _region_tensor(region, offset: int, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
    """A torch view of the process proxy's shared region (shares its memory; the region's
    owner keeps it mapped and registered)."""
    n = 1
    for d in shape:
        n *= d
    return torch.frombuffer(region.mm, dtype=dtype, count=n, offset=offset).view(shape)


class BroadcastLayout(ReplyLayout):
    """int32 tensor: [status, n, ok_mask[M], drafts[M*N], conf_bits[M*N]]."""

    def drafts(self, t: torch.Tensor, n: int) -> torch.Tensor:
        return t[self.o_ids:self.o_ids + n * self.N].view(n, self.N)

    def conf(self, t: torch.Tensor, n: int) -> torch.Tensor:
        return t[self.o_conf:self.o_conf + n * self.N].view(torch.float32).view(n, self.N)

    def ok_mask(self, t: torch.Tensor, n: int) -> torch.Tensor:
        return t[self.o_ok:self.o_ok + n]


def make_remote_round(cfg: RemoteConfig, hello: wire.Hello, *, device: torch.device, tp_rank: int,
                      broadcast_fn: Callable[[torch.Tensor], None] | None, stats: Stats, background: bool,
                      stream_sync=None, n_out_by_reqs: list[int] | None = None) -> "RemoteRound":
    """The engine's RemoteRound (speculator._make_round). With the process proxy, a proxy that does not start falls
    back to synchronous rounds, as a refused hand-off self-test does (S11 P-2): the watchdog stops, the shared region is
    unregistered and unmapped (release_hand_off), stats async_fallback_reason says why. A strict start (background
    False: VLLM_DSPARK_REMOTE_STRICT_START) raises ProxyStartError instead, after the same release."""
    kw = dict(device=device, tp_rank=tp_rank, broadcast_fn=broadcast_fn, stats=stats, background=background,
              n_out_by_reqs=n_out_by_reqs)
    try:
        return RemoteRound(cfg, hello, stream_sync=stream_sync, **kw)
    except ProxyStartError as e:
        why = str(e)
    # the exception and the failed constructor's frames (and their views of the region) are gone here
    leak = release_hand_off(stream_sync)
    if leak:
        log.error("remote DSpark: after the failed proxy start the %s", leak)
    if not background:
        raise ProxyStartError(why)
    log.warning("remote DSpark async proxy process did not start (%s); synchronous rounds", why)
    stats.set("async_fallback_reason", f"proxy process: {why}")
    return RemoteRound(cfg, hello, stream_sync=None, **kw)


@dataclass
class RoundResult:
    n: int
    bcast: torch.Tensor                 # device, layout above
    layout: BroadcastLayout
    status_host: int | None = None      # rank 0 only (others: read_status())
    reply: wire.StepReply | None = None  # rank 0 only
    need_draft: np.ndarray | None = None  # rank 0 only

    @property
    def drafts(self) -> torch.Tensor:
        return self.layout.drafts(self.bcast, self.n)

    @property
    def conf(self) -> torch.Tensor:
        return self.layout.conf(self.bcast, self.n)

    @property
    def ok_mask(self) -> torch.Tensor:
        return self.layout.ok_mask(self.bcast, self.n)

    def read_status(self) -> int:
        """Host value of the round status on any rank (syncs on non-zero ranks)."""
        if self.status_host is None:
            self.status_host = int(self.bcast[0].item())
        return self.status_host


class RemoteRound(RoundCore):
    def __init__(
        self,
        cfg: RemoteConfig,
        hello: wire.Hello,
        *,
        device: torch.device,
        tp_rank: int,
        broadcast_fn: Callable[[torch.Tensor], None] | None,
        stats: Stats | None = None,
        client: RemoteDraftClient | None = None,
        transport_factory=None,
        background: bool = True,
        stream_sync=None,
        n_out_by_reqs: list[int] | None = None,
    ):
        self._suffix_audit: SuffixAudit | None = None
        self._suffix_agree: Callable[[Snapshot], None] | None = None
        if os.environ.get("VLLM_DSV41_REPLAY_AUDIT", "0") == "1":
            self._suffix_audit = SuffixAudit(hello.window, hello.max_slots)
        self.cfg = cfg
        self.hello = hello
        self.device = torch.device(device)
        self.tp_rank = tp_rank
        self.is_sender = tp_rank == 0
        self.broadcast_fn = broadcast_fn
        self.stats = stats or Stats()
        self.N = hello.num_spec
        self.W = hello.window
        self.M = hello.max_slots
        self.layout = BroadcastLayout(self.M, self.N)
        self.bcast = torch.zeros(self.layout.size, dtype=torch.int32, device=self.device)
        # CONTRACT 6.3 n_out by request count (index n, clamped); None = always N
        self.n_out_by_reqs = list(n_out_by_reqs) if n_out_by_reqs else None
        self.rounds = 0
        self._last_epoch = 0
        self._last_log = time.monotonic()
        self._err_logged = 0.0
        self.client: RemoteDraftClient | None = None
        self._meta_slots = 1
        self._bcast_ev = None
        self.gatherer: RowGatherer | None = None
        self.slots: SlotTable | None = None
        self.breaker: CircuitBreaker | None = None
        # process proxy: the hand-off lives in a shared region (shmregion.py) the round's proxy
        # process maps too; that process owns the client, slots and breaker (proxy_proc.py)
        self.region = getattr(stream_sync, "shared", None) if self.is_sender else None
        self.proxy: ProxyProcess | None = None
        if self.is_sender:
            pin = self.device.type == "cuda"
            if stream_sync is None:
                self.bcast_host = torch.zeros(self.layout.size, dtype=torch.int32, pin_memory=pin)
                self.bcast_np = self.bcast_host.numpy()
                # recorded after every copy out of bcast_host; waited on before the host rewrites
                # it (normally already complete: each round synchronizes on its gather first)
                self._bcast_ev = torch.cuda.Event() if pin else None
            else:
                # async: the proxy builds the reply in this private buffer and publishes it into
                # the hand-off's registered page in one GIL-free call (streamsync.HostFlagCore);
                # the engine stream copies it from there after the wait. No event, no CUDA here
                self.bcast_host = None
                self.bcast_np = np.zeros(self.layout.size, dtype=np.int32)
            # async: round t's gather metadata goes to slot t % _meta_slots (roundcore.META_SLOTS)
            self._meta_slots = META_SLOTS if stream_sync is not None else 1
            host = {}
            if self.region is not None:
                L = self.region.layout
                if L != RegionLayout.for_hello(hello, L.ring):
                    raise ValueError(f"shared region layout {L} does not match the HELLO")
                self._meta_slots = L.ring
                host = dict(host_msg=_region_tensor(self.region, L.off_msg, torch.uint8, (L.msg_capacity,)),
                            host_meta=_region_tensor(self.region, L.off_meta, torch.int64,
                                                     (L.ring, 4, self.M)),
                            d2h=stream_sync.d2h)
            self.gatherer = RowGatherer(max_slots=self.M, window=self.W, hidden=hello.hidden,
                                        n_taps=hello.n_taps, max_rows=hello.max_rows_per_step,
                                        device=self.device, fused=cfg.fused_gather,
                                        meta_slots=self._meta_slots, **host)
            if self.gatherer.row_bytes != hello.row_bytes:
                raise ValueError(f"row bytes {self.gatherer.row_bytes} != {hello.row_bytes}")
            if self.region is None:
                self.slots = SlotTable(self.M, self.W)
                self.breaker = CircuitBreaker(cfg.trip, cfg.cooldown)
                self.stats.set("breaker", self.breaker.snapshot())
                self.client = client or RemoteDraftClient(
                    cfg, hello, stats=self.stats, transport_factory=transport_factory,
                    background=background)
                self.stats.set("contract_version", wire.CONTRACT_VERSION)
                self.stats.set("policy", cfg.policy)
                self.stats.set("endpoint", str(cfg.endpoint))
            self.stats.set("rank0_buffers", {"device_bytes": self.gatherer.device_bytes,
                                             "pinned_host_bytes": self.gatherer.host_bytes})
            self.stats.set("fused_gather", {"on": self.gatherer.fused,
                                            "check": self.gatherer.fused_check})
            self.stats.set("n_out_by_reqs", self.n_out_by_reqs)
        # async mode (rank 0, capacity policy): the engine stream and the proxy thread hand each
        # round over through two host words (streamsync.HostFlagSync); the proxy thread makes no
        # CUDA call, so a module load on the engine thread can never wait on it
        self.sync = stream_sync if self.is_sender else None
        self.async_mode = self.sync is not None
        self._q: queue.Queue | None = None
        self._staged: frozenset[int] = frozenset()
        if self.async_mode:
            if not getattr(self.sync, "cuda_free_proxy", False):
                raise ValueError("async rounds need a CUDA-free hand-off (streamsync.HostFlagSync)")
            assert self.gatherer is not None
            self._stop = threading.Event()
            self._dead: str | None = None
            self._reply_bytes = self.layout.size * 4
            if self.region is not None:
                self._start_process_proxy(background)
                return
            # resync windows: the engine stream copies a requested slot's whole ring window here
            # after the round's gather; the proxy builds resync rows from it on the CPU
            self.resync_host = torch.zeros(self.M, self.W, self.gatherer.row_elems,
                                           dtype=self.gatherer.dtype,
                                           pin_memory=self.device.type == "cuda")
            self.resync_np = self.resync_host.view(torch.uint8).numpy()
            self._resync_lock = threading.Lock()
            self._resync_req: set[int] = set()
            # the native watchdog fails and releases a round nobody released within the deadline:
            # a proxy that cannot run (GIL held by a waiting extension, a blocked GC finalizer,
            # death, interpreter exit) costs one fail-closed round, never a parked GPU (S9 F2/F5)
            self.sync.attach(self._reply_bytes, STATUS_ROUND_FAILED, cfg.watchdog_ms)
            self._q = queue.Queue()
            self._thread = threading.Thread(target=self._async_main, daemon=True,
                                            name="dsv41-remote-dspark-round")
            self._thread.start()
            self.stats.set("async", True)
            self.stats.set("async_handoff", self.sync.describe())

    def _start_process_proxy(self, background: bool) -> None:
        """Async, process proxy: region-backed resync windows, the watchdog (this process), then
        the proxy process; returns once it reports its start (strict: connected + HELLO)."""
        assert self.region is not None and self.sync is not None and self.gatherer is not None
        L = self.region.layout
        g = self.gatherer
        self.resync_host = _region_tensor(self.region, L.off_resync, g.dtype,
                                          (self.M, self.W, g.row_elems))
        self._session = _new_session(None)
        self._generation = 0
        self._respawn: _Respawn | None = None
        self._respawns_left = self.cfg.async_respawns
        self._respawn_count = self._respawn_failures = 0
        self._losses: list[dict] = []
        try:
            # every failure from here on is a failed start of the async path (make_remote_round falls back)
            self.sync.attach(self._reply_bytes, STATUS_ROUND_FAILED, self.cfg.watchdog_ms)
            init = self._proxy_init(start_round=1, strict=not background)
            self.proxy = ProxyProcess(self.region, self.sync.lib, init)
            self.proxy.wait_started(START_TIMEOUT_S + self.cfg.connect_timeout_ms / 1000)
        except BaseException as e:
            self.sync.stop_watchdog()
            if self.proxy is not None:
                self.proxy.kill()
            # nothing of this half-built round may keep a view of the region (make_remote_round unmaps it)
            self.gatherer = self.resync_host = None
            if isinstance(e, Exception) and not isinstance(e, ProxyStartError):
                raise ProxyStartError(f"remote DSpark proxy process could not be started: "
                                      f"{type(e).__name__}: {e}") from e
            raise
        self._live_since = time.monotonic()
        self.stats.set("async", True)
        self._proxy_info = {"mode": "process", "pid": self.proxy.pid, "generation": 0,
                            "descriptor_ring": L.ring, "region_bytes": L.size,
                            "region_mlock": "ok" if not self.region.mlock_errno else f"errno {self.region.mlock_errno}"}
        self.stats.set("async_proxy", dict(self._proxy_info))
        self.stats.set("async_dead", None)
        self._respawn_stats("live")
        self.stats.set("async_handoff", self.sync.describe())
        self._wd_fires = self.region.u64(C_WD, 1)          # watchdog mirror: fires (stall detection)
        self._hb_limit_ms = heartbeat_limit_ms(self.cfg)
        self._stall_limit = stall_fires(self.cfg)
        self.stats.set("async_stall_detection", {"heartbeat_limit_ms": self._hb_limit_ms,
                                                 "stall_fires": self._stall_limit})
        self._health_reset()
        self.proxy.publish_engine_stats(self.stats.snapshot())

    def _proxy_init(self, *, start_round: int, strict: bool, base_counters: dict | None = None) -> bytes:
        return encode_init(self.cfg, self.hello, self.region.layout, n_out_by_reqs=self.n_out_by_reqs,
                           lib_path=self.sync.lib._name, spin_us=self.sync.spin_ns // 1000,
                           nap_us=self.sync.nap_ns // 1000, watchdog_ms=self.cfg.watchdog_ms,
                           strict=strict, parent_pid=os.getpid(),
                           pdeathsig=threading.current_thread() is threading.main_thread(),
                           start_round=start_round, session=self._session, base_counters=base_counters,
                           generation=self._generation)

    def _health_reset(self) -> None:
        """Stall detection baselines for the current proxy (start, respawn)."""
        self.sync.watchdog_mirror(self.region.word(C_WD))
        self._hc_fires = int(self._wd_fires[0])
        self._hc_done = self.proxy.word(C_DONE)
        self._stall_fires = 0
        self._last_submit = time.monotonic()

    def _proxy_health(self, t: int) -> tuple[str, str] | None:
        """Process proxy, live: (kind, reason) when it must be declared lost, else None. Called once per submit.
        exit: the process is gone. ring: descriptor t would overwrite one the proxy never took. fires: stall_fires()
        watchdog fires with no round finished in between (a proxy that is alive but stuck, or one that runs but has
        lost track of the rounds). heartbeat: its housekeeping has not run for heartbeat_limit_ms (a stopped or
        descheduled process; judged only when this engine thread itself ran in that time, so a pause of the whole
        container is not taken for a proxy stall). Each stall costs at most a few watchdog deadlines, not a whole
        descriptor ring of them (S11 P-3: 64 x 200 ms before)."""
        p = self.proxy
        rc = p.exit_code()
        if rc is not None:
            return "exit", f"proxy process exited with {rc}"
        ring = self.region.layout.ring
        if ((t - p.tail()) & _M32) > ring:
            return "ring", f"proxy process stalled ({ring} rounds not taken at round {t})"
        now = time.monotonic()
        ran = now - self._last_submit < self._hb_limit_ms / 1000
        self._last_submit = now
        fires, done = int(self._wd_fires[0]), p.word(C_DONE)
        if done != self._hc_done:
            self._hc_done, self._stall_fires = done, 0
        else:
            self._stall_fires += fires - self._hc_fires
        self._hc_fires = fires
        if self._stall_fires >= self._stall_limit:
            return "fires", (f"proxy process stalled: {self._stall_fires} watchdog fires with no round finished "
                             f"(last finished round {done}, at round {t})")
        if ran:
            age = p.heartbeat_age_ms()
            if age > self._hb_limit_ms:
                return "heartbeat", f"proxy process heartbeat {age} ms old (limit {self._hb_limit_ms} ms, round {t})"
        return None

    # ------------------------------------------------------------------ run

    def run(self, input_batch, aux_hidden_states, num_sampled: torch.Tensor,
            num_rejected: torch.Tensor, last_sampled: torch.Tensor,
            next_prefill_tokens: torch.Tensor) -> RoundResult:
        n = input_batch.num_reqs
        if self._suffix_audit is not None:
            self._audit_suffix(input_batch, num_sampled, num_rejected, last_sampled, next_prefill_tokens)
        res = RoundResult(n=n, bcast=self.bcast, layout=self.layout)
        if self.proxy is not None:
            self._submit_process(input_batch, aux_hidden_states, num_sampled, num_rejected,
                                 last_sampled, next_prefill_tokens)
        elif self.async_mode:
            self._submit_async(input_batch, aux_hidden_states, num_sampled, num_rejected,
                               last_sampled, next_prefill_tokens)
        elif self.is_sender:
            t0 = time.perf_counter()
            bv = None
            try:
                bv = batch_view_from_input_batch(input_batch)
                status = self._sender_round(bv, input_batch, aux_hidden_states, num_sampled,
                                            num_rejected, last_sampled, next_prefill_tokens,
                                            res)
            except Exception as e:  # never skip the broadcast: rank 1 waits for it
                status = STATUS_ROUND_FAILED
                self._pack_failed(n)
                self.stats.inc("fallback_internal_error")
                if self.slots is not None:
                    self.slots.mark_all_dirty()
                now = time.monotonic()
                if now - self._err_logged > 10.0:
                    self._err_logged = now
                    log.exception("remote DSpark round failed on rank 0: %s", e)
            res.status_host = status
            self.bcast.copy_(self.bcast_host, non_blocking=True)
            self._bcast_copied()
            self.stats.observe("rank0_round_us", (time.perf_counter() - t0) * 1e6)
            if bv is not None:
                self.stats.observe(f"rank0_round_us_{self._phase(bv)}",
                                   (time.perf_counter() - t0) * 1e6)
            self.rounds += 1
            self._maybe_log()
        if self.broadcast_fn is not None:
            self.broadcast_fn(self.bcast)
        return res

    def _audit_suffix(self, input_batch: Any, num_sampled: torch.Tensor, num_rejected: torch.Tensor,
                      last_sampled: torch.Tensor, next_prefill_tokens: torch.Tensor) -> None:
        # QUALIFICATION ONLY: synchronizes device metadata on both ranks, then a
        # CPU-group agreement. Disabled in headline timings; no tensor/text log.
        assert self._suffix_audit is not None
        error: str | None = None
        rows: tuple[SuffixRow, ...] = ()
        try:
            bv = batch_view_from_input_batch(input_batch)
            rows = self._suffix_audit.observe(
                bv.req_ids, bv.slots.tolist(), bv.qsl.tolist(),
                None if bv.tap_lens is None else bv.tap_lens.tolist(),
                input_batch.positions[:int(bv.qsl[-1])].cpu().tolist(),
                num_rejected[:bv.n].cpu().tolist(), num_sampled[:bv.n].cpu().tolist(),
                last_sampled.reshape(-1).cpu().tolist(), next_prefill_tokens[0].cpu().tolist(),
                bv.need_draft.tolist())
        except (ValueError, IndexError) as exc:
            error = type(exc).__name__  # exchange failures; don't strand a peer collective
        snapshot = (error, rows)
        if self._suffix_agree is None:
            if self.tp_rank != 0 or self.broadcast_fn is not None:
                raise RuntimeError("0075 rank provenance agreement callback missing")
            agree([snapshot])
        else:
            self._suffix_agree(snapshot)
        self.stats.inc("replay_provenance_audited_rounds")

    def _bcast_copied(self) -> None:
        if self._bcast_ev is not None:
            self._bcast_ev.record()

    def _bcast_writable(self) -> None:
        """Wait until the last copy out of bcast_host has run (pinned memory: the host must
        not rewrite it under a pending non_blocking copy). Rank 0's own streams only."""
        if self._bcast_ev is not None:
            self._bcast_ev.synchronize()

    def _sender_round(self, bv: BatchView, input_batch, aux, num_sampled, num_rejected,
                      last_sampled, next_prefill_tokens, res: RoundResult) -> int:
        assert self.gatherer is not None
        res.need_draft = bv.need_draft
        tg = time.perf_counter()
        g = self.gatherer.gather(bv, list(aux), input_batch.positions, num_rejected,
                                 num_sampled, input_batch.idx_mapping, last_sampled,
                                 next_prefill_tokens)
        self.stats.observe("gather_us", (time.perf_counter() - tg) * 1e6)
        return self._after_gather(bv, g, res, step_id=self.rounds + 1)

    # ------------------------------------------------------------------ async mode

    def _submit_async(self, input_batch, aux, num_sampled, num_rejected, last_sampled,
                      next_prefill_tokens) -> None:
        """Rank 0, async: enqueue the gather (plus any requested resync windows), the gathered
        word, the wait for the release and the reply copy; hand the round to the proxy thread;
        return without blocking the CPU (CONTRACT 4.2, async variant; streamsync.py)."""
        assert self.gatherer is not None and self.sync is not None and self._q is not None
        t = self.rounds + 1
        tq = time.perf_counter()
        if self._dead is not None:
            # the proxy thread is gone: every round is target-only. No hand-off, no wait: the
            # page holds the failed reply for good (fail_all claimed far ahead; S9 F2)
            self.rounds = t
            self.sync.copy_reply(self.bcast)
            self.stats.inc("fallback_proxy_dead")
            return
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            # a captured wait would replay with a fixed value and pass silently (S9 F6)
            raise RuntimeError("remote DSpark async round inside CUDA graph capture")
        staged: frozenset[int] = frozenset()
        try:
            bv = batch_view_from_input_batch(input_batch)
            R = self.gatherer.enqueue(bv, list(aux), input_batch.positions, num_rejected,
                                      num_sampled, input_batch.idx_mapping, last_sampled,
                                      next_prefill_tokens, meta_slot=t % self._meta_slots)
            with self._resync_lock:
                want = sorted(self._resync_req)
            if want:
                W, ring = self.W, self.gatherer.ring
                for sl in want:
                    self.resync_host[sl].copy_(ring[sl * W:(sl + 1) * W], non_blocking=True)
                staged = frozenset(want)
                self.stats.inc("resync_windows_staged", len(want))
            item = (t, bv, R, staged)
        except Exception as e:
            # nothing reaches the thread for this round: it releases t with a failed reply
            self.stats.inc("fallback_internal_error")
            log.exception("remote DSpark async enqueue failed: %s", e)
            item = (t, None, 0, frozenset())
        self.rounds = t
        # stream order: the gather's copies (and staged windows), then the word the proxy waits for
        self.sync.gathered(t)
        self._q.put(item)
        self.sync.wait_release(t)
        self.sync.copy_reply(self.bcast)
        self.sync.consumed(t)              # explicit consumer acknowledgment of round t's reply
        self.stats.observe("rank0_submit_us", (time.perf_counter() - tq) * 1e6)
        if item[1] is not None:
            self.stats.observe(f"rank0_submit_us_{self._phase(item[1])}", (time.perf_counter() - tq) * 1e6)

    def _submit_process(self, input_batch, aux, num_sampled, num_rejected, last_sampled,
                        next_prefill_tokens) -> None:
        """Rank 0, async with the process proxy: as _submit_async, but the round goes to the
        proxy process through the region's descriptor ring (shmregion.py), and the gather's
        rows / metadata / staged windows land in the region. Never blocks the CPU."""
        assert self.gatherer is not None and self.sync is not None and self.proxy is not None
        assert self.region is not None
        t = self.rounds + 1
        tq = time.perf_counter()
        if self._dead is None:
            lost = self._proxy_health(t)
            if lost is not None:
                self._proxy_lost(lost[1], lost[0])
        if self._dead is not None and self._respawn is not None:
            self._respawn_step()                 # never blocks; may bring a new proxy live for this round
        if self._dead is not None:
            # no proxy: target-only rounds (zero drafts). The page holds the failed reply (fail_all
            # claimed far ahead); no wait, no gather. The round number does not advance: it numbers the
            # rounds handed to a proxy, and a respawned proxy continues from the last of them
            self.sync.copy_reply(self.bcast)
            self.stats.inc("fallback_proxy_dead")
            return
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("remote DSpark async round inside CUDA graph capture")
        bv = None
        try:
            bv = batch_view_from_input_batch(input_batch)
            R = self.gatherer.enqueue(bv, list(aux), input_batch.positions, num_rejected,
                                      num_sampled, input_batch.idx_mapping, last_sampled,
                                      next_prefill_tokens, meta_slot=t % self._meta_slots)
            staged: frozenset[int] = frozenset()
            want = self.proxy.resync_requests()
            if want:
                W, ring = self.W, self.gatherer.ring
                for sl in want:
                    self.sync.d2h(self.resync_host[sl], ring[sl * W:(sl + 1) * W])
                staged = frozenset(want)
                self.stats.inc("resync_windows_staged", len(want))
            self.region.write_descriptor(t, bv.n, R, 0, bv.slots, bv.k_window, bv.need_draft,
                                         staged, bv.req_ids, max_q=bv.longest_query)
        except Exception as e:
            # the proxy fails round t (descriptor flag) and releases it with a failed reply
            self.stats.inc("fallback_internal_error")
            log.exception("remote DSpark async enqueue failed: %s", e)
            self.region.write_descriptor(t, 0, 0, F_ENQUEUE_FAILED, (), (), (), frozenset(), [])
        self.rounds = t
        self.proxy.publish_head(t)
        # stream order: the gather's copies (and staged windows), then the word the proxy waits for
        self.sync.gathered(t)
        self.sync.wait_release(t)
        self.sync.copy_reply(self.bcast)
        self.sync.consumed(t)              # explicit consumer acknowledgment of round t's reply
        self.sync.watchdog_mirror(self.region.word(C_WD))
        self.stats.observe("rank0_submit_us", (time.perf_counter() - tq) * 1e6)
        if bv is not None:
            self.stats.observe(f"rank0_submit_us_{self._phase(bv)}", (time.perf_counter() - tq) * 1e6)
        every = self.cfg.log_every
        if t == 1 or (every > 0 and t % every == 0):
            self.proxy.publish_engine_stats(self.stats.snapshot())

    def _proxy_lost(self, reason: str, kind: str) -> None:
        """The proxy process is gone or stalled: fail every pending and later round (the same far-ahead fail-all a
        dying proxy thread does), SIGKILL it without waiting for it (S11 P-6), record it, and plan its replacement
        (S11 P-4) while the respawn budget lasts; then target-only rounds until a new proxy runs."""
        assert self.proxy is not None
        self.stats.inc(f"proxy_lost_{kind}")
        self.sync.fail_all(self.rounds + FAR_AHEAD)
        self._dead = reason
        self.proxy.signal_kill()
        now = time.monotonic()
        self._losses.append({"round": self.rounds, "time": round(time.time(), 3), "kind": kind, "reason": reason,
                             "pid": self.proxy.pid, "generation": self._generation})
        del self._losses[:-LOSSES_KEPT]
        self.stats.set("async_dead", reason)
        self.stats.set("async_proxy_losses", list(self._losses))
        reset_s = self.cfg.async_respawn_reset_s
        if reset_s and now - self._live_since >= reset_s:
            self._respawns_left = self.cfg.async_respawns      # it served long enough: the full budget again
        if self._respawns_left > 0:
            self._respawn = _Respawn(old=self.proxy, drain=self.sync.drain_marker(), lost_at=now,
                                     due=now + RESPAWN_DELAY_S)
            log.error("remote DSpark async proxy process %d lost (%s): target-only rounds (zero drafts) until a new "
                      "proxy process runs (%d respawn(s) left)", self.proxy.pid, reason, self._respawns_left)
            self._respawn_stats("waiting")
        else:
            self._respawn = None
            self._respawn_exhausted(reason)
        self._write_engine_stats_file()

    def _respawn_exhausted(self, why: str) -> None:
        log.error("remote DSpark async proxy process lost (%s) and its respawn budget is spent (%d per %s): "
                  "target-only rounds (zero drafts) for the rest of this engine run", why, self.cfg.async_respawns,
                  f"{self.cfg.async_respawn_reset_s} s of service" if self.cfg.async_respawn_reset_s else "run")
        self._respawn_stats("exhausted")

    def _respawn_stats(self, state: str) -> None:
        self.stats.set("async_respawns", self._respawn_count)
        self.stats.set("async_respawn", {"state": state, "limit": self.cfg.async_respawns,
                                         "left": self._respawns_left, "failed_starts": self._respawn_failures,
                                         "reset_s": self.cfg.async_respawn_reset_s})

    def _respawn_step(self) -> None:
        """No proxy (S11 P-4): advance its replacement; called once per submit, never blocks. A new proxy is started
        only when (1) the lost one has been reaped (a killed process runs no more user code, but until it is reaped
        it may be finishing a system call), (2) the engine stream has passed every operation of the lost proxy's
        rounds (an event query, never a stream sync), and (3) RESPAWN_DELAY_S has passed. Then every hand-off word is
        reset to the last round handed out and the new proxy is started with a new worker session: its HELLO makes
        the worker drop every window, and its empty slot table announces every slot anew (NEW_GEN). It goes live at
        the first submit that finds its start reported. Requests in flight then draft from the rows gathered since
        (target-only rounds gather nothing), a partial window that refills within W positions."""
        r = self._respawn
        now = time.monotonic()
        if r.new is not None:
            state, why = r.new.start_status()
            if state == "ok":
                self._respawn_live(r.new, now)
                return
            limit = START_TIMEOUT_S + self.cfg.connect_timeout_ms / 1000
            if state == "pending":
                if now - r.started_at < limit:
                    return
                why = f"remote DSpark proxy process did not start within {limit:.0f} s"
            r.new.signal_kill()
            self._respawn_failures += 1
            log.error("remote DSpark async proxy respawn %d failed: %s", self._generation, why)
            r.old, r.new, r.due = r.new, None, now + RESPAWN_DELAY_S
            if self._respawns_left <= 0:
                self._respawn = None
                self._respawn_exhausted(why)
            else:
                self._respawn_stats("waiting")
            return
        if now < r.due:
            return
        if r.old.exit_code() is None:
            if not r.warned and now - r.lost_at > 10:
                r.warned = True
                log.error("remote DSpark: proxy process %d has not exited %.0f s after SIGKILL (uninterruptible "
                          "sleep?); no new proxy until it has", r.old.pid, now - r.lost_at)
            return
        if not self.sync.drained(r.drain):
            return
        self._respawns_left -= 1
        self._generation += 1
        self._session = _new_session(self._session)
        base = self.rounds
        self._reset_hand_off(base)
        try:
            r.new = ProxyProcess(self.region, self.sync.lib,
                                 self._proxy_init(start_round=base + 1, strict=False,
                                                  base_counters=self._proxy_counters()))
        except OSError as e:
            self._respawn_failures += 1
            log.error("remote DSpark async proxy respawn %d: cannot start a process: %s", self._generation, e)
            r.due = now + RESPAWN_DELAY_S
            if self._respawns_left <= 0:
                self._respawn = None
                self._respawn_exhausted(str(e))
            else:
                self._respawn_stats("waiting")
            return
        r.started_at = now
        self._respawn_stats("starting")
        log.warning("remote DSpark: starting proxy process %d (respawn %d, session %#010x, rounds from %d)",
                    r.new.pid, self._generation, self._session, base + 1)

    def _respawn_live(self, new: ProxyProcess, now: float) -> None:
        self.proxy = new
        self._respawn = None
        self._dead = None
        self._respawn_count += 1
        self._live_since = now
        self._proxy_info.update(pid=new.pid, generation=self._generation)
        self.stats.set("async_proxy", dict(self._proxy_info))
        self.stats.set("async_dead", None)
        self._respawn_stats("live")
        self._health_reset()
        self.proxy.publish_engine_stats(self.stats.snapshot())
        log.warning("remote DSpark async proxy process %d running (respawn %d, session %#010x): drafting resumes at "
                    "round %d", new.pid, self._generation, self._session, self.rounds + 1)

    def _reset_hand_off(self, base: int) -> None:
        """Respawn: every hand-off word to base, the last round handed to a proxy, so the new proxy serves base + 1
        on. Only with no proxy process alive and the stream past every operation of earlier rounds. The watchdog keeps
        running: gathered goes first and released last but one, so it never sees a gathered-unreleased round (claimed
        and released come down from the fail-all's far-ahead value; nothing waits on them now)."""
        lib, sync, reg = self.sync.lib, self.sync, self.region
        b = base & _M32
        for addr in (sync.gathered_host, sync.claimed_host, sync.released_host, sync.consumed_host):
            lib.dsv41_hf_set(_V(addr), b)
        for off, v in ((C_DESC_HEAD, b), (C_DESC_TAIL, b), (C_DONE, b), (C_STOP, 0), (C_START, R_PENDING),
                       (C_STATE, S_STARTING), (C_PID, 0), (C_FLUSH_REQ, 0), (C_FLUSH_ACK, 0), (C_ENGINE_SEQ, 0),
                       (C_HEARTBEAT, monotonic_ms())):
            lib.dsv41_hf_set(_V(reg.word(off)), v)
        reg.resync_flags[:] = 0
        reg.set_start_reason("")

    def _proxy_counters(self) -> dict:
        """The lost proxies' counters from the stats file (as last written), carried on by the next proxy."""
        if not self.cfg.stats_file:
            return {}
        try:
            with open(self.cfg.stats_file) as f:
                return dict(unmerge_snapshot(json.load(f)).get("counters") or {})
        except (OSError, ValueError) as e:
            log.warning("remote DSpark stats file %s unreadable at respawn (%s): the new proxy's counters start at 0",
                        self.cfg.stats_file, e)
            return {}

    def _write_engine_stats_file(self) -> None:
        """Without a live proxy the engine writes the stats file: the proxy's last one merged
        with this side's counters."""
        if not self.cfg.stats_file:
            return
        self.stats.set("rounds", self.rounds)
        self.stats.set("async_watchdog", self.sync.watchdog_stats())
        proxy_snap: dict = {}
        try:
            with open(self.cfg.stats_file) as f:
                proxy_snap = unmerge_snapshot(json.load(f))
        except (OSError, ValueError):
            pass
        engine_snap = self.stats.snapshot()
        # this side's values are current: the dead proxy's file carries the engine keys (async_respawn, async_proxy,
        # rounds, ...) only as published at its last mark, and merge_snapshots() lets the proxy's copy win (S12 F3)
        for k in engine_snap:
            if k not in ("counters", "latency", "uptime_s"):
                proxy_snap.pop(k, None)
        try:
            write_json_atomic(self.cfg.stats_file, merge_snapshots(proxy_snap, engine_snap))
        except OSError as e:
            log.warning("remote DSpark stats file %s: %s", self.cfg.stats_file, e)

    def _async_main(self) -> None:
        """Proxy thread. CPU only: no CUDA or torch call (streamsync.py). Any exit other than the
        sentinel or close() fails every pending and later round (S9 F2)."""
        assert self._q is not None and self.sync is not None
        set_timer_slack()                   # short naps (ASYNC_NAP_US > 0) wake on time
        try:
            while True:
                item = self._q.get()
                if item is None or not self._async_round(item):
                    return                       # sentinel, or stopping inside a round (close)
        except BaseException as e:  # noqa: BLE001 -- whatever ends the thread must not park the GPU
            self._die(f"{type(e).__name__}: {e}")      # the thread ends here, failed closed

    def write_stats(self) -> None:
        if self.proxy is None:
            return RoundCore.write_stats(self)
        if self._dead is not None:
            return self._write_engine_stats_file()
        self.proxy.publish_engine_stats(self.stats.snapshot())
        if not self.proxy.flush():
            log.warning("remote DSpark proxy process did not write the stats file within 2 s")

    def close(self) -> None:
        if self.proxy is not None:
            if self._dead is None:
                self.proxy.publish_engine_stats(self.stats.snapshot())
            procs = [self.proxy]
            if self._respawn is not None and self._respawn.new is not None:
                procs.append(self._respawn.new)          # a respawned proxy still starting
            for p in procs:
                if p.stop(5.0) is None:
                    log.error("remote DSpark proxy process %d did not stop within 5 s (killed)", p.pid)
            # nothing may stay parked on a round the stopped proxy never released
            self.sync.fail_all(self.rounds + FAR_AHEAD)
            if self._dead is not None:
                self._write_engine_stats_file()
            return
        if self._q is not None:
            self._stop.set()
            self._q.put(None)
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                log.error("remote DSpark async proxy thread did not stop within 5 s (a round in "
                          "flight); failing every pending round")
            if self.sync is not None:
                # nothing may stay parked on a round the stopped thread never released. Monotonic
                # and claimed far ahead: a late proxy can neither publish nor move the word back
                self.sync.fail_all(self.rounds + FAR_AHEAD)
        if self.client is not None:
            self.client.close()

