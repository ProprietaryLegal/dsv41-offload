# SPDX-License-Identifier: Apache-2.0
"""Host side of the async hand-off (torch-free): libdsv41_hostflag.so (hostflag.c), the page with the gathered /
released / claimed words and the reply area, the proxy's wait and publish, fail-all and the native watchdog.

Split out of streamsync.py unchanged so that the out-of-process proxy (proxy_proc.py) can use it without importing
torch; streamsync.py re-exports every name. The page is either allocated here (one process: the thread proxy) or lies
in a shared region another object owns (shmregion.py: the process proxy), see HostFlagCore(page=...).
"""

from __future__ import annotations

import atexit
import ctypes
import hashlib
import os
import shutil
import stat
import subprocess
import tempfile

import numpy as np

CU_STREAM_WAIT_VALUE_GEQ = 0x0          # (int32)(*addr - value) >= 0: wrap-safe
CU_STREAM_WRITE_VALUE_DEFAULT = 0x0     # with a memory fence before the write
CU_MEMHOSTREGISTER_PORTABLE = 0x01
CU_MEMHOSTREGISTER_DEVICEMAP = 0x02

HOSTFLAG_LIB = "libdsv41_hostflag.so"
HOSTFLAG_VERSION = 2
PAGE_BYTES = 65536
OFF_GATHERED = 0        # written by the engine stream
OFF_RELEASED = 128      # written by publish / fail-release (CPU), each word on its own cache line
OFF_CLAIMED = 256       # claim word: the last round the proxy or the watchdog took
OFF_CONSUMED = 384      # consumer acknowledgment: the last round whose reply the engine stream has copied (R2)
OFF_REPLY = 4096        # the reply the engine stream copies after the wait
REPLY_CAPACITY = PAGE_BYTES - OFF_REPLY
FAR_AHEAD = 1 << 30     # fail-all value: cyclically after every round within 2**30
_M32 = 0xFFFFFFFF
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hostflag.c")


class StreamSyncError(RuntimeError):
    pass


class StreamStuckError(RuntimeError):
    """The engine stream stayed parked after the self-test's rescue: this device does not observe host stores. Not
    recoverable (the stream cannot be unparked), so the caller must fail loudly, not fall back (S9 F4)."""


# ---------------------------------------------------------------------------------------------------- the C helper

_V = ctypes.c_void_p


def _bind(lib: ctypes.CDLL) -> ctypes.CDLL:
    try:
        sig = {
            "dsv41_hf_version": (ctypes.c_int, []),
            "dsv41_hf_alloc": (_V, [ctypes.c_size_t]),
            "dsv41_hf_free": (None, [_V]),
            "dsv41_hf_lock": (ctypes.c_int, [_V, ctypes.c_size_t]),
            "dsv41_hf_load": (ctypes.c_uint32, [_V]),
            "dsv41_hf_release_max": (None, [_V, ctypes.c_uint32]),
            "dsv41_hf_store": (None, [_V, ctypes.c_uint32]),
            "dsv41_hf_set": (None, [_V, ctypes.c_uint32]),
            "dsv41_hf_claim": (ctypes.c_int, [_V, ctypes.c_uint32]),
            "dsv41_hf_publish": (ctypes.c_int, [_V, _V, ctypes.c_uint32, _V, _V, ctypes.c_size_t]),
            "dsv41_hf_fail_release": (ctypes.c_int, [_V, _V, ctypes.c_uint32, _V, ctypes.c_size_t, ctypes.c_int32]),
            "dsv41_hf_wait_geq": (ctypes.c_int, [_V, ctypes.c_uint32, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]),
            "dsv41_hf_watchdog_start": (_V, [_V, _V, _V, _V, ctypes.c_size_t, ctypes.c_int32, ctypes.c_int64,
                                             ctypes.c_int64]),
            "dsv41_hf_watchdog_stats": (None, [_V, ctypes.POINTER(ctypes.c_uint64)]),
            "dsv41_hf_watchdog_stop": (ctypes.c_int, [_V]),
        }
        for name, (res, args) in sig.items():
            f = getattr(lib, name)
            f.restype = res
            f.argtypes = args
    except AttributeError as e:   # an older library without a v2 symbol: a clean refusal (S9 F7)
        raise StreamSyncError(f"{lib._name}: {e}") from e
    v = lib.dsv41_hf_version()
    if v != HOSTFLAG_VERSION:
        raise StreamSyncError(f"{lib._name}: hostflag version {v}, expected {HOSTFLAG_VERSION}")
    return lib


def _private_dir(path: str) -> None:
    """Create (or accept) a build directory only this user can write."""
    os.makedirs(path, mode=0o700, exist_ok=True)
    st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o022:
        raise StreamSyncError(f"hostflag build directory {path} is not a private directory of uid {os.getuid()}")


def _compile(cc: str, out_dir: str) -> str:
    with open(_SRC, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:16]
    d = os.path.join(out_dir, f"dsv41-hostflag-{os.getuid()}-{digest}")
    _private_dir(d)
    so = os.path.join(d, HOSTFLAG_LIB)
    if not os.path.exists(so):
        tmp = so + f".tmp{os.getpid()}"
        try:
            r = subprocess.run([cc, "-O2", "-Wall", "-shared", "-fPIC", "-o", tmp, _SRC, "-lpthread"],
                               capture_output=True, text=True, timeout=120)
        except (subprocess.TimeoutExpired, OSError) as e:   # S9 F7: a clean refusal, not a crash
            raise StreamSyncError(f"compiling {_SRC} with {cc}: {e}") from e
        if r.returncode != 0:
            raise StreamSyncError(f"compiling {_SRC} with {cc} failed ({r.returncode}): {r.stderr.strip()[:400]}")
        os.replace(tmp, so)
    return so


def load_hostflag(lib_path: str = "") -> tuple[ctypes.CDLL, str]:
    """libdsv41_hostflag.so: an explicit path, else VLLM_DSPARK_REMOTE_OFL_DIR's copy, else hostflag.c compiled with
    the system C compiler into a private directory under the temp dir. Returns (library, where it came from)."""
    if lib_path:
        if not os.path.exists(lib_path):
            raise StreamSyncError(f"VLLM_DSPARK_REMOTE_HOSTFLAG_LIB={lib_path} does not exist")
        return _bind(ctypes.CDLL(lib_path)), lib_path
    ofl_dir = (os.environ.get("VLLM_DSPARK_REMOTE_OFL_DIR") or "").strip()
    if ofl_dir and os.path.exists(os.path.join(ofl_dir, HOSTFLAG_LIB)):
        p = os.path.join(ofl_dir, HOSTFLAG_LIB)
        return _bind(ctypes.CDLL(p)), p
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        raise StreamSyncError(f"no {HOSTFLAG_LIB} (set VLLM_DSPARK_REMOTE_HOSTFLAG_LIB or put it in "
                              "VLLM_DSPARK_REMOTE_OFL_DIR) and no C compiler to build hostflag.c")
    p = _compile(cc, tempfile.gettempdir())
    return _bind(ctypes.CDLL(p)), f"{p} (compiled at startup)"


PR_SET_TIMERSLACK = 29


def set_timer_slack(ns: int = 1000) -> bool:
    """Linux timer slack of the calling thread: with VLLM_DSPARK_REMOTE_ASYNC_NAP_US > 0 the proxy's waits nap in
    short nanosleeps, which the default 50 us slack would stretch. False where prctl is unavailable."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        return libc.prctl(PR_SET_TIMERSLACK, ctypes.c_ulong(int(ns)), ctypes.c_ulong(0), ctypes.c_ulong(0),
                          ctypes.c_ulong(0)) == 0
    except (OSError, AttributeError):
        return False


# ---------------------------------------------------------------------------------------------------- host side

class HostFlagCore:
    """The host side of the hand-off (no CUDA): the C-owned page with the three words and the reply area, the proxy's
    wait and publish, fail-all, and the native watchdog. The page is never freed while the process may still copy from
    it (close() only after the round stopped and the stream drained)."""

    cuda_free_proxy = True

    def __init__(self, lib_path: str = "", spin_us: int = 1000, nap_us: int = 0, page: int = 0):
        """page: 0 = allocate the page here (freed by close()); else the address of a PAGE_BYTES page, 4096-aligned,
        in memory the caller owns and keeps mapped (a shared region: shmregion.py). An external page is never freed
        here, and its words are not reset: the region's creator zeroed them."""
        self.lib, self.lib_origin = load_hostflag(lib_path)
        self.spin_ns = int(spin_us) * 1000
        self.nap_ns = int(nap_us) * 1000
        self.owns_page = not page
        if page:
            if page % 4096:
                raise StreamSyncError(f"external hostflag page {page:#x} is not 4096-aligned")
        else:
            page = self.lib.dsv41_hf_alloc(PAGE_BYTES)
            if not page:
                raise StreamSyncError("hostflag page allocation failed")
        self.page = int(page)
        self.mlock_errno = int(self.lib.dsv41_hf_lock(_V(self.page), PAGE_BYTES))
        self.gathered_host = self.page + OFF_GATHERED
        self.released_host = self.page + OFF_RELEASED
        self.claimed_host = self.page + OFF_CLAIMED
        self.consumed_host = self.page + OFF_CONSUMED
        self.reply_host = self.page + OFF_REPLY
        self.reply_bytes = 0
        self.fail_status = 0
        self.deadline_ms = 0
        self._wd: int | None = None
        self._atexit_registered = False

    # -------------------------------------------------- proxy thread (CPU only, no torch)

    def wait_gathered(self, value: int, timeout_s: float) -> bool:
        """True once gathered >= value; False after timeout_s. Releases the GIL while it waits."""
        r = self.lib.dsv41_hf_wait_geq(_V(self.gathered_host), value & _M32, int(timeout_s * 1e9),
                                       self.spin_ns, self.nap_ns)
        return r == 0

    def publish(self, value: int, src: np.ndarray) -> bool:
        """Claim round value, copy src into the reply area, release value: one GIL-free C call. False when the round
        had already been claimed (the watchdog failed it first); nothing is written then."""
        if not src.flags.c_contiguous or src.nbytes > REPLY_CAPACITY:
            raise ValueError(f"reply of {src.nbytes} bytes (contiguous {src.flags.c_contiguous}) does not fit")
        return bool(self.lib.dsv41_hf_publish(_V(self.claimed_host), _V(self.released_host), value & _M32,
                                              _V(self.reply_host), _V(src.ctypes.data), src.nbytes))

    def taken(self, value: int) -> bool:
        """True once round value (or a later one) has been claimed: by this proxy's publish, the watchdog or a
        fail-all. A proxy that finds its round taken before it starts must not read that round's staging."""
        c = int(self.lib.dsv41_hf_load(_V(self.claimed_host)))
        return ((c - value) & _M32) < (1 << 31)

    def reply_consumed(self, value: int) -> bool:
        """The consumer acknowledgment: True once the engine stream has copied round value's reply out of the page
        (it writes the word right after the copy). Round t's reply may be written only when round t-1's is consumed;
        gathered >= t implies it by stream order, and the proxy checks the word before every publish."""
        c = int(self.lib.dsv41_hf_load(_V(self.consumed_host)))
        return ((c - value) & _M32) < (1 << 31)

    def consumed(self, value: int, stream=None) -> None:
        """Engine side, after copy_reply(): acknowledge round value's reply. CPU stand-ins copy synchronously, so the
        store follows the copy; HostFlagSync enqueues it on the stream instead."""
        self.lib.dsv41_hf_release_max(_V(self.consumed_host), value & _M32)

    def release(self, value: int) -> None:
        """Monotonic release (cyclic max) without a reply: the self-test's rescue."""
        self.lib.dsv41_hf_release_max(_V(self.released_host), value & _M32)

    def drain_marker(self):
        """Engine side: a marker after everything enqueued on the engine stream so far (process proxy respawn, S11 P-4).
        CPU stand-ins run their "stream" synchronously, so nothing is ever pending: None."""
        return None

    def drained(self, marker) -> bool:
        """True once everything before the marker ran (never waits)."""
        return True

    # -------------------------------------------------- any thread

    def fail_all(self, value: int) -> bool:
        """Claim value, write the failed reply, release value. With value FAR_AHEAD of every enqueued round this fails
        every pending and later round at once (proxy death, close, exit). False if value was already claimed."""
        n = self.reply_bytes or REPLY_CAPACITY
        return bool(self.lib.dsv41_hf_fail_release(_V(self.claimed_host), _V(self.released_host), value & _M32,
                                                   _V(self.reply_host), n, self.fail_status))

    def fail_ahead(self) -> bool:
        """fail_all at FAR_AHEAD past the furthest of the three words."""
        g, r, c = self.values()
        base = g
        for w in (r, c):
            if ((w - base) & _M32) < (1 << 31):    # w cyclically at or after base
                base = w
        return self.fail_all(base + FAR_AHEAD)

    def values(self) -> tuple[int, int, int]:
        ld = self.lib.dsv41_hf_load
        return (int(ld(_V(self.gathered_host))), int(ld(_V(self.released_host))), int(ld(_V(self.claimed_host))))

    # -------------------------------------------------- setup (engine thread)

    def attach(self, reply_bytes: int, fail_status: int, deadline_ms: int, poll_us: int = 1000) -> None:
        """Size the reply, write a failed reply into it (what the stream copies before any round was published) and
        start the watchdog."""
        if self._wd is not None:
            raise StreamSyncError("hostflag watchdog already attached")
        if not 4 <= reply_bytes <= REPLY_CAPACITY:
            raise StreamSyncError(f"reply of {reply_bytes} bytes does not fit the {REPLY_CAPACITY}-byte area")
        if deadline_ms <= 0:
            raise StreamSyncError(f"watchdog deadline {deadline_ms} ms")
        self.configure_reply(reply_bytes, fail_status)
        self.deadline_ms = int(deadline_ms)
        failed = np.zeros(self.reply_bytes // 4, dtype=np.int32)
        failed[0] = self.fail_status
        ctypes.memmove(self.reply_host, failed.ctypes.data, self.reply_bytes)
        poll_ns = max(50_000, min(int(poll_us) * 1000, self.deadline_ms * 100_000))
        wd = self.lib.dsv41_hf_watchdog_start(_V(self.gathered_host), _V(self.released_host),
                                              _V(self.claimed_host), _V(self.reply_host), self.reply_bytes,
                                              self.fail_status, self.deadline_ms * 1_000_000, poll_ns)
        if not wd:
            raise StreamSyncError(f"hostflag watchdog did not start (errno {ctypes.get_errno()})")
        self._wd = int(wd)
        if not self._atexit_registered:
            atexit.register(self._at_exit)
            self._atexit_registered = True

    def configure_reply(self, reply_bytes: int, fail_status: int) -> None:
        """Reply size and failed status, without touching the page (attach() and a process proxy's view of a page
        the engine process attached)."""
        if not 4 <= reply_bytes <= REPLY_CAPACITY:
            raise StreamSyncError(f"reply of {reply_bytes} bytes does not fit the {REPLY_CAPACITY}-byte area")
        self.reply_bytes = int(reply_bytes)
        self.fail_status = int(fail_status)

    def watchdog_stats(self) -> dict:
        if self._wd is None:
            return {"running": False}
        out = (ctypes.c_uint64 * 3)()
        self.lib.dsv41_hf_watchdog_stats(_V(self._wd), out)
        return {"running": True, "deadline_ms": self.deadline_ms, "fires": int(out[0]),
                "last_failed_round": int(out[1]), "max_late_ms": round(out[2] / 1e6, 3)}

    def watchdog_mirror(self, dst: int) -> None:
        """Copy the watchdog's raw stats (fires, last failed round, max lateness in ns: 3 x u64) to address dst (the
        process proxy's shared control block). No-op without a watchdog."""
        if self._wd is not None:
            self.lib.dsv41_hf_watchdog_stats(_V(self._wd), ctypes.cast(dst, ctypes.POINTER(ctypes.c_uint64)))

    def stop_watchdog(self) -> None:
        if self._wd is not None:
            wd, self._wd = self._wd, None
            self.lib.dsv41_hf_watchdog_stop(_V(wd))

    def reset(self) -> None:
        """All words to 0. Only with no round in flight and no watchdog (setup, self-test)."""
        if self._wd is not None:
            raise StreamSyncError("reset with the watchdog running")
        for a in (self.gathered_host, self.released_host, self.claimed_host, self.consumed_host):
            self.lib.dsv41_hf_set(_V(a), 0)

    def describe(self) -> dict:
        return {"mechanism": "host_flags", "lib": self.lib_origin, "spin_us": self.spin_ns // 1000,
                "nap_us": self.nap_ns // 1000, "mlock": "ok" if self.mlock_errno == 0 else f"errno {self.mlock_errno}",
                "watchdog_deadline_ms": self.deadline_ms or None}

    def _at_exit(self) -> None:
        # interpreter shutdown: nothing may stay parked on a round no thread will release (no free: the stream may
        # still copy the reply during teardown)
        if self.page:
            self.fail_ahead()
            self.stop_watchdog()

    def close(self) -> None:
        """Only after the round stopped and the stream drained (nothing may copy from the page afterwards)."""
        self.stop_watchdog()
        if self.page:
            if self.owns_page:
                self.lib.dsv41_hf_free(_V(self.page))
            self.page = 0

