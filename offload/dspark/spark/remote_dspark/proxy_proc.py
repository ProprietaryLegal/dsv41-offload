# SPDX-License-Identifier: Apache-2.0
"""Out-of-process async proxy (VLLM_DSPARK_REMOTE_ASYNC_PROXY=process). torch-free: the proxy process imports this
package as a top-level package and never imports vllm or torch.

Why a process (directive 2026-10-23 §3): the thread proxy's success path - wait for the gather, network round,
publish - is Python, so it needs the engine's GIL; during prefill the engine thread holds the GIL for long stretches
and the round's publication waits for it (GB10: +4-8 % TTFT, 31 watchdog-preempted rounds of 5,113 in the async
compact set). Here the same reviewed round code (roundcore.RoundCore, unchanged) runs in a child interpreter with its
own GIL; it shares nothing with the engine but the memory region (shmregion.py) and the C hand-off words (hostflag.c).
Its success path is: acquire-wait on the descriptor and gathered words, the round, one GIL-free C publish into the
shared page. No step needs the engine's GIL, a CUDA call or an engine-held lock.

    engine thread:  gather -> D2H into the region -> descriptor t -> desc_head = t -> [stream: gathered = t,
                    wait released >= t, H2D reply] -> return (never blocks)
    proxy process:  wait desc_head >= t -> copy descriptor, desc_tail = t -> wait gathered >= t -> RoundCore round
                    -> publish(t) (claim, copy, release: hostflag.c)
    engine watchdog (C thread in the engine process): a round gathered but unreleased past the deadline is claimed,
                    failed and released, exactly as with the thread proxy.

Ownership of round t's completion is the claim word (one winner: proxy publish, watchdog, or a fail-all). Failure of
the proxy process is seen by the engine at the next round (Popen.poll): fail-all far ahead, then target-only rounds,
the same end state as a dead proxy thread (S9 F2). The proxy watches its parent (getppid, and PR_SET_PDEATHSIG when
started from the main thread) and exits when the engine is gone.
"""

from __future__ import annotations

import ctypes
import dataclasses
import json
import logging
import os
import signal
import subprocess
import sys
import time

import numpy as np

from . import wire
from .breaker import CircuitBreaker
from .client import RemoteDraftClient
from .config import RemoteConfig
from .hostflag import _M32, _V, HostFlagCore, set_timer_slack
from .roundcore import STATUS_ROUND_FAILED, BatchView, ReplyLayout, RoundCore, collect_meta
from .shmregion import (C_DESC_HEAD, C_DESC_TAIL, C_DONE, C_ENGINE_SEQ, C_FLUSH_ACK, C_FLUSH_REQ, C_HEARTBEAT, C_PID,
                        C_START, C_STATE, C_STOP, C_WD, F_ENQUEUE_FAILED, R_FAILED, R_OK, R_PENDING, S_FAILED,
                        S_RUNNING, S_STOPPED, RegionLayout, SharedRegion)
from .state import SlotTable
from .stats import Stats
from .transport import Endpoint

log = logging.getLogger("vllm.remote_dspark")

PR_SET_PDEATHSIG = 1
DESC_SPIN_NS = 200_000      # descriptor wait: spin 200 us, then 100 us naps (the gather leaves that much slack)
DESC_NAP_NS = 100_000
POLL_S = 0.1                # every wait wakes this often to check stop / parent / flush requests
START_TIMEOUT_S = 60.0      # interpreter start + imports (+ connect and HELLO when strict)

# ---------------------------------------------------------------------------------------------------- init blob


def _enc(v):
    if isinstance(v, bytes):
        return {"__hex__": v.hex()}
    if isinstance(v, dict):
        return {k: _enc(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_enc(x) for x in v]
    return v


def _dec(v):
    if isinstance(v, dict):
        if set(v) == {"__hex__"}:
            return bytes.fromhex(v["__hex__"])
        return {k: _dec(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_dec(x) for x in v]
    return v


def encode_init(cfg: RemoteConfig, hello: wire.Hello, layout: RegionLayout, *, n_out_by_reqs, lib_path: str,
                spin_us: int, nap_us: int, watchdog_ms: int, strict: bool, parent_pid: int, pdeathsig: bool,
                start_round: int = 1, session: int | None = None, base_counters: dict | None = None,
                generation: int = 0) -> bytes:
    """start_round: the first round this proxy serves (a respawned proxy continues the engine's numbering);
    session: its worker session id; base_counters: its predecessors' counters, carried on in its stats file;
    generation: 0 for the first proxy, k for the k-th respawn."""
    return json.dumps(_enc({
        "cfg": dataclasses.asdict(cfg), "hello": dataclasses.asdict(hello), "layout": layout.to_dict(),
        "n_out_by_reqs": list(n_out_by_reqs) if n_out_by_reqs else None, "lib_path": lib_path,
        "spin_us": int(spin_us), "nap_us": int(nap_us), "watchdog_ms": int(watchdog_ms), "strict": bool(strict),
        "parent_pid": int(parent_pid), "pdeathsig": bool(pdeathsig), "start_round": int(start_round),
        "session": session, "base_counters": dict(base_counters or {}), "generation": int(generation)})).encode()


def decode_init(data: bytes) -> dict:
    d = _dec(json.loads(data.decode()))
    c = d["cfg"]
    c["endpoint"] = Endpoint(**c["endpoint"])
    d["cfg"] = RemoteConfig(**c)
    h = d["hello"]
    h["tap_layer_ids"] = tuple(h["tap_layer_ids"])
    d["hello"] = wire.Hello(**h)
    d["layout"] = RegionLayout(**d["layout"])
    return d


ENGINE_OWNED = ("async_dead",)    # the engine decides whether a proxy serves (a dying proxy also records its reason)


def merge_snapshots(proxy: dict, engine: dict | None) -> dict:
    """One stats file from the proxy's snapshot and the engine side's (counters add up; histograms and extras are
    disjoint by construction, the proxy's win on a clash except ENGINE_OWNED)."""
    if not engine:
        return proxy
    out = {k: v for k, v in engine.items() if k not in ("counters", "latency", "uptime_s")}
    out.update({k: v for k, v in proxy.items() if k not in ("counters", "latency")})
    for k in ENGINE_OWNED:
        if k in engine:
            out[k] = engine[k]
    cnt = dict(engine.get("counters") or {})
    for k, v in (proxy.get("counters") or {}).items():
        cnt[k] = cnt.get(k, 0) + v
    out["counters"] = dict(sorted(cnt.items()))
    lat = dict(engine.get("latency") or {})
    lat.update(proxy.get("latency") or {})
    out["latency"] = dict(sorted(lat.items()))
    # what the engine side added, so that the engine can rewrite the file without counting it twice
    out["engine_share"] = {"counters": dict(engine.get("counters") or {}),
                           "latency": sorted((engine.get("latency") or {}).keys())}
    return out


def unmerge_snapshot(merged: dict) -> dict:
    """The proxy's part of a file merge_snapshots() wrote (the inverse of the engine share)."""
    share = merged.get("engine_share") or {}
    out = {k: v for k, v in merged.items() if k != "engine_share"}
    eng_c = share.get("counters") or {}
    out["counters"] = {k: v - eng_c.get(k, 0) for k, v in (merged.get("counters") or {}).items()
                       if v - eng_c.get(k, 0)}
    eng_l = set(share.get("latency") or ())
    out["latency"] = {k: v for k, v in (merged.get("latency") or {}).items() if k not in eng_l}
    return out


def write_json_atomic(path: str, obj: dict) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------------- proxy side


class _ProxyFlags(HostFlagCore):
    """The proxy process's view of the engine's hand-off page: wait, publish, fail-all. The watchdog runs in the
    engine process; its stats are mirrored into the control block every round."""

    def __init__(self, region: SharedRegion, lib_path: str, spin_us: int, nap_us: int, reply_bytes: int,
                 watchdog_ms: int):
        super().__init__(lib_path, spin_us, nap_us, page=region.page_addr)
        self.configure_reply(reply_bytes, STATUS_ROUND_FAILED)
        self.region = region
        self.deadline_ms = int(watchdog_ms)

    def attach(self, *a, **k) -> None:
        raise RuntimeError("the watchdog belongs to the engine process")

    def watchdog_stats(self) -> dict:
        fires, last, late = (int(x) for x in self.region.u64(C_WD, 3))
        return {"running": True, "deadline_ms": self.deadline_ms, "fires": fires, "last_failed_round": last,
                "max_late_ms": round(late / 1e6, 3), "mirrored": True}

    def describe(self) -> dict:
        d = super().describe()
        d["mechanism"] = "host_flags_shared"
        return d


class _RegionRows:
    """What RoundCore reads of the gatherer, from the region: the STEP message buffer and the gather metadata."""

    def __init__(self, region: SharedRegion):
        L = region.layout
        self.region = region
        self.msg_np = region.msg_np
        self.max_rows = L.max_rows
        self.row_bytes = L.row_bytes

    def collect(self, n: int, n_candidates: int, meta_slot: int = 0):
        return collect_meta(self.region.meta_np[meta_slot], n, n_candidates)

    def gather_from_ring(self, n_req, spans):
        raise RuntimeError("synchronous resync in the process proxy")


def monotonic_ms() -> int:
    """CLOCK_MONOTONIC in ms as a u32 (the heartbeat's clock; system-wide, so the engine compares it with its own)."""
    return int(time.monotonic() * 1000) & _M32


class _Housekeeping:
    """RoundCore._stop for the proxy process: polled at least every POLL_S while it waits and once per round. Writes
    the heartbeat (C_HEARTBEAT, S11 P-3), serves stats flush requests; reports stop when the engine asked for it or
    the engine process is gone."""

    def __init__(self, rnd: "ProcessRound"):
        self.r = rnd
        self.lib = rnd.sync.lib
        self._stop_w = _V(rnd.region.word(C_STOP))
        self._flush_w = _V(rnd.region.word(C_FLUSH_REQ))
        self._hb_w = _V(rnd.region.word(C_HEARTBEAT))
        self._flushed = 0

    def beat(self) -> None:
        self.lib.dsv41_hf_set(self._hb_w, monotonic_ms())

    def is_set(self) -> bool:
        self.beat()
        req = int(self.lib.dsv41_hf_load(self._flush_w))
        if req != self._flushed:
            self._flushed = req
            try:
                self.r.write_stats()
            finally:
                self.lib.dsv41_hf_store(_V(self.r.region.word(C_FLUSH_ACK)), req)
        if self.lib.dsv41_hf_load(self._stop_w):
            return True
        return os.getppid() != self.r.parent_pid


class ProcessRound(RoundCore):
    """The proxy process's round: RoundCore fed from the descriptor ring instead of the thread proxy's queue."""

    def __init__(self, region: SharedRegion, cfg: RemoteConfig, hello: wire.Hello, *, n_out_by_reqs, lib_path: str,
                 spin_us: int, nap_us: int, watchdog_ms: int, parent_pid: int, background: bool,
                 transport_factory=None, start_round: int = 1, session: int | None = None,
                 base_counters: dict | None = None, generation: int = 0):
        self.cfg = cfg
        self.hello = hello
        self.region = region
        self.parent_pid = parent_pid
        self.start_round = int(start_round)
        self.stats = Stats()
        for k, v in (base_counters or {}).items():         # a respawned proxy carries its predecessors' counts
            self.stats.inc(k, int(v))
        self.N, self.W, self.M = hello.num_spec, hello.window, hello.max_slots
        self.layout = ReplyLayout(self.M, self.N, cfg.sampled_top32)
        self.bcast_np = np.zeros(self.layout.size, dtype=np.int32)
        self.n_out_by_reqs = list(n_out_by_reqs) if n_out_by_reqs else None
        self.rounds = self.start_round - 1
        self._last_epoch = 0
        self._last_log = time.monotonic()
        self._err_logged = 0.0
        self.is_sender = True
        self.async_mode = True
        L = region.layout
        if (L.max_slots, L.window, L.row_bytes, L.max_rows) != (self.M, self.W, hello.row_bytes,
                                                                 hello.max_rows_per_step):
            raise ValueError(f"region layout {L} does not match the HELLO")
        self.gatherer = _RegionRows(region)
        self._meta_slots = L.ring
        self.resync_np = region.resync_np
        self.slots = SlotTable(self.M, self.W)
        self.breaker = CircuitBreaker(cfg.trip, cfg.cooldown)
        self.stats.set("breaker", self.breaker.snapshot())
        self.sync = _ProxyFlags(region, lib_path, spin_us, nap_us, self.layout.size * 4, watchdog_ms)
        self.lib = self.sync.lib
        self._staged: frozenset[int] = frozenset()
        self._dead: str | None = None
        self.client = RemoteDraftClient(cfg, hello, stats=self.stats, transport_factory=transport_factory,
                                        background=background, session=session)
        self.stats.set("contract_version", wire.CONTRACT_VERSION)
        self.stats.set("policy", cfg.policy)
        self.stats.set("endpoint", str(cfg.endpoint))
        self.stats.set("n_out_by_reqs", self.n_out_by_reqs)
        # the engine's "async_proxy" (region, mlock, current pid) stays authoritative in the merged stats file
        self.stats.set("async_proxy_process", {"pid": os.getpid(), "generation": int(generation),
                                               "start_round": self.start_round})
        self._stop = _Housekeeping(self)
        self._engine_seq = 0
        self._engine_snap: dict | None = None

    # resync requests: one byte per slot in the control block (engine reads them at submit)
    def _request_resync(self, slots: set[int]) -> None:
        f = self.region.resync_flags
        for sl in slots:
            f[sl] = 1

    def _clear_resync(self, staged: frozenset[int]) -> None:
        f = self.region.resync_flags
        for sl in staged:
            f[sl] = 0

    def _wait_descriptor(self, t: int) -> bool:
        head = _V(self.region.word(C_DESC_HEAD))
        while not self._stop.is_set():
            if self.lib.dsv41_hf_wait_geq(head, t & _M32, int(POLL_S * 1e9), DESC_SPIN_NS, DESC_NAP_NS) == 0:
                return True
        return False

    def _take(self, t: int):
        d = self.region.read_descriptor(t)
        if d.flags & F_ENQUEUE_FAILED:
            return (t, None, 0, frozenset())
        bv = BatchView(req_ids=d.owners, slots=d.slots, qsl=np.zeros(0, dtype=np.int64),
                       need_draft=d.need_draft, k_window=d.k_window, max_q=d.max_q)
        return (t, bv, d.R, d.staged)

    def serve(self) -> None:
        """Rounds start_round, start_round + 1, ... until stopped. Raises on a broken hand-off (the caller fails
        everything closed)."""
        t = self.start_round
        tail = _V(self.region.word(C_DESC_TAIL))
        done = _V(self.region.word(C_DONE))
        while True:
            if not self._wait_descriptor(t):
                return
            item = self._take(t)
            self.rounds = t
            if not self._async_round(item):
                return
            # slot t % K (descriptor, metadata) may be reused only now (shmregion.py)
            self.lib.dsv41_hf_store(tail, t & _M32)
            self.lib.dsv41_hf_store(done, t & _M32)
            t += 1

    def _engine_snapshot(self) -> dict | None:
        seq = int(self.lib.dsv41_hf_load(_V(self.region.word(C_ENGINE_SEQ))))
        if seq and seq != self._engine_seq:
            try:
                self._engine_snap = json.loads(self.region.read_engine_blob(seq).decode())
                self._engine_seq = seq
            except (ValueError, UnicodeDecodeError):
                pass    # overwritten while read (stats only): the previous snapshot stands
        return self._engine_snap

    def write_stats(self) -> None:
        if not self.cfg.stats_file:
            return
        self.stats.set("rounds", self.rounds)
        self.stats.set("breaker", self.breaker.snapshot())
        self.stats.set("async_watchdog", self.sync.watchdog_stats())
        self.stats.set("proxy_torch_loaded", "torch" in sys.modules)     # evidence: the proxy never imports it
        try:
            write_json_atomic(self.cfg.stats_file, merge_snapshots(self.stats.snapshot(), self._engine_snapshot()))
        except OSError as e:
            log.warning("remote DSpark stats file %s: %s", self.cfg.stats_file, e)


def _pdeathsig() -> None:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    except (OSError, AttributeError):
        pass


def _store(region: SharedRegion, lib, off: int, value: int) -> None:
    lib.dsv41_hf_set(_V(region.word(off)), value & _M32)


def main(fd: int, init: bytes | None = None, transport_factory=None) -> int:
    """Proxy process entry (BOOT below): the init blob comes on stdin."""
    d = decode_init(init if init is not None else sys.stdin.buffer.read())
    if d["pdeathsig"]:
        _pdeathsig()
    set_timer_slack()                       # short naps (ASYNC_NAP_US > 0) wake on time
    if os.getppid() != d["parent_pid"]:
        return 5                                     # the engine is already gone
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="(remote-dspark proxy pid=%(process)d) %(levelname)s %(asctime)s %(message)s")
    region = SharedRegion.attach(d["layout"], fd)
    region.u32(C_PID)[0] = os.getpid()
    cfg: RemoteConfig = d["cfg"]
    rnd = None
    lib = None
    try:
        rnd = ProcessRound(region, cfg, d["hello"], n_out_by_reqs=d["n_out_by_reqs"], lib_path=d["lib_path"],
                           spin_us=d["spin_us"], nap_us=d["nap_us"], watchdog_ms=d["watchdog_ms"],
                           parent_pid=d["parent_pid"], background=not d["strict"],
                           transport_factory=transport_factory, start_round=d.get("start_round", 1),
                           session=d.get("session"), base_counters=d.get("base_counters"),
                           generation=d.get("generation", 0))
        lib = rnd.lib
        if d["strict"]:
            if not rnd.client.connect_now():
                raise RuntimeError(f"connect/HELLO to {cfg.endpoint} failed at startup "
                                   f"({rnd.client.disabled_reason or rnd.client.last_error})")
            rnd.client.start_background()
    except BaseException as e:  # noqa: BLE001 -- the engine reads the reason and raises
        region.set_start_reason(f"{type(e).__name__}: {e}")
        if lib is not None:
            _store(region, lib, C_START, R_FAILED)
            _store(region, lib, C_STATE, S_FAILED)
        else:
            region.u32(C_START)[0] = R_FAILED
            region.u32(C_STATE)[0] = S_FAILED
        if rnd is not None:
            rnd.client.close()
        return 2
    rnd._stop.beat()
    _store(region, lib, C_START, R_OK)
    _store(region, lib, C_STATE, S_RUNNING)
    log.info("remote DSpark proxy process running (engine pid %d, %s)", d["parent_pid"], rnd.sync.describe())
    code = 0
    try:
        rnd.serve()
    except BaseException as e:  # noqa: BLE001 -- whatever ends the proxy must not park the GPU
        rnd._die(f"proxy process: {type(e).__name__}: {e}")
        code = 4
    finally:
        try:
            rnd.write_stats()
        finally:
            rnd.client.close()
            _store(region, lib, C_STATE, S_FAILED if code else S_STOPPED)
    return code


# ---------------------------------------------------------------------------------------------------- engine side


class ProxyStartError(RuntimeError):
    """The proxy process did not start (spawn, interpreter, imports, start timeout; strict: connect + HELLO). The engine
    then runs synchronous rounds unless the start is strict (round.make_remote_round, S11 P-2)."""

# the proxy's interpreter imports this package from its directory as the top-level package "remote_dspark" (never
# through vllm: no torch, no CUDA in the proxy)
BOOT = """import importlib.util, sys
pkg, fd = sys.argv[1], int(sys.argv[2])
spec = importlib.util.spec_from_file_location("remote_dspark", pkg + "/__init__.py", submodule_search_locations=[pkg])
mod = importlib.util.module_from_spec(spec)
sys.modules["remote_dspark"] = mod
spec.loader.exec_module(mod)
from remote_dspark.proxy_proc import main
sys.exit(main(fd))
"""


class ProxyProcess:
    """Engine side: start the proxy process on a region, watch it, stop it. CPU only."""

    def __init__(self, region: SharedRegion, lib, init: bytes, *, python: str | None = None):
        self.region = region
        self.lib = lib
        pkg = os.path.dirname(os.path.abspath(__file__))
        # -I: isolated (no PYTHONPATH / user site / working directory on sys.path): the proxy imports only this
        # package (by path) and the interpreter's own site packages
        self.proc = subprocess.Popen([python or sys.executable, "-I", "-c", BOOT, pkg, str(region.fd)],
                                     stdin=subprocess.PIPE, pass_fds=(region.fd,), close_fds=True)
        try:
            self.proc.stdin.write(init)
            self.proc.stdin.close()
        except OSError:
            pass                                     # died at once: wait_started() reports it
        self._flush = 0
        self._engine_seq = 0
        self._blob_warned = False

    @property
    def pid(self) -> int:
        return self.proc.pid

    def word(self, off: int) -> int:
        return int(self.lib.dsv41_hf_load(_V(self.region.word(off))))

    def start_status(self) -> tuple[str, str]:
        """Without waiting: ("ok", ""), ("pending", "") or ("failed", why) (it reported a failure or exited)."""
        r = self.word(C_START)
        if r == R_OK:
            return "ok", ""
        if r == R_FAILED:
            return "failed", f"remote DSpark proxy process failed to start: {self.region.start_reason()}"
        rc = self.proc.poll()
        if rc is not None:
            return "failed", f"remote DSpark proxy process exited with {rc} before it started"
        return "pending", ""

    def wait_started(self, timeout_s: float = START_TIMEOUT_S) -> None:
        """Until the proxy reports its start (client built; when strict: connected + HELLO). Raises ProxyStartError
        when it failed, exited or did not report in time; the process is then gone or killed."""
        t_end = time.monotonic() + timeout_s
        while True:
            state, why = self.start_status()
            if state == "ok":
                return
            if state == "failed":
                self.kill(2.0)
                raise ProxyStartError(why)
            if time.monotonic() > t_end:
                self.kill()
                raise ProxyStartError(f"remote DSpark proxy process did not start within {timeout_s:.0f} s")
            time.sleep(0.005)

    def exit_code(self) -> int | None:
        return self.proc.poll()

    def tail(self) -> int:
        return self.word(C_DESC_TAIL)

    def heartbeat_age_ms(self) -> int:
        """Milliseconds since the proxy's last housekeeping poll (0 if its clock read is a hair ahead of ours)."""
        d = (monotonic_ms() - self.word(C_HEARTBEAT)) & _M32
        return d if d < (1 << 31) else 0

    def publish_head(self, t: int) -> None:
        self.lib.dsv41_hf_store(_V(self.region.word(C_DESC_HEAD)), t & _M32)

    def resync_requests(self) -> list[int]:
        return [int(i) for i in np.flatnonzero(self.region.resync_flags)]

    def publish_engine_stats(self, snapshot: dict) -> None:
        data = json.dumps(snapshot, default=str).encode()
        seq = self._engine_seq + 1
        if self.region.write_engine_blob(seq, data):
            self._engine_seq = seq
            self.lib.dsv41_hf_store(_V(self.region.word(C_ENGINE_SEQ)), seq & _M32)
        elif not self._blob_warned:
            self._blob_warned = True
            log.warning("remote DSpark engine-side stats (%d B) do not fit the shared region's blob; the stats file "
                        "keeps the last engine snapshot that did", len(data))

    def flush(self, timeout_s: float = 2.0) -> bool:
        """Ask the proxy to write the stats file now; True once it did."""
        if self.proc.poll() is not None:
            return False
        self._flush += 1
        self.lib.dsv41_hf_store(_V(self.region.word(C_FLUSH_REQ)), self._flush & _M32)
        r = self.lib.dsv41_hf_wait_geq(_V(self.region.word(C_FLUSH_ACK)), self._flush & _M32, int(timeout_s * 1e9),
                                       200_000, 1_000_000)
        return r == 0

    def stop(self, timeout_s: float = 5.0) -> int | None:
        self.lib.dsv41_hf_store(_V(self.region.word(C_STOP)), 1)
        try:
            return self.proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.kill()
            return None

    def signal_kill(self) -> None:
        """SIGKILL, without waiting: the engine thread must not block on a child that may sit in uninterruptible
        sleep (S11 P-6). exit_code() reaps it later. A killed process runs no more user code, so it can no longer
        write the region (it may still be finishing a system call)."""
        if self.proc.poll() is None:
            self.proc.kill()

    def kill(self, wait_s: float = 5.0) -> None:
        """SIGKILL and wait for the exit at most wait_s (startup and shutdown only; the round path uses signal_kill)."""
        self.signal_kill()
        if self.proc.returncode is None:
            try:
                self.proc.wait(timeout=wait_s)
            except subprocess.TimeoutExpired:
                log.error("remote DSpark proxy process %d did not exit within %.0f s of SIGKILL", self.pid, wait_s)


__all__ = ["ProcessRound", "ProxyProcess", "ProxyStartError", "encode_init", "decode_init", "merge_snapshots", "main",
           "R_PENDING"]
