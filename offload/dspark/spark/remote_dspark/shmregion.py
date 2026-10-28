# SPDX-License-Identifier: Apache-2.0
"""Shared memory of the out-of-process async proxy (VLLM_DSPARK_REMOTE_ASYNC_PROXY=process). torch-free.

One memfd mapping, created by the engine process and inherited by the proxy process (proxy_proc.py). The engine
registers the whole mapping with CUDA once (streamsync.RegisteredRegion: Portable | DeviceMap, malloc-style memory as
the hand-off page always was), so the engine stream writes the gather's rows and metadata straight into it and copies
the reply out of it; the proxy process reads and writes it with plain CPU loads and stores and makes no CUDA call.

    offset            bytes                  writer -> reader
    0                 PAGE_BYTES             hostflag page (hostflag.py): gathered (engine stream) -> proxy;
                                             released / claimed / reply (proxy publish, engine watchdog) -> stream
    PAGE_BYTES        CTRL_BYTES             control block (below)
    off_desc          K x desc_bytes         round descriptors (engine thread -> proxy), slot t % K
    off_meta          K x 4 x M int64        gather metadata [valid, first_pos, anchor, sampled] of round t in slot
                                             t % K (stream -> proxy)
    off_msg           msg_capacity           STEP message: header + entries (proxy), rows (stream) (gather.py)
    off_resync        M x W x row_bytes      staged resync windows (stream -> proxy)

Round descriptors (engine thread -> proxy). The engine writes descriptor t into slot t % K, then advances desc_head to
t (release); the proxy waits desc_head >= t (acquire), copies the descriptor out, runs the round and then advances
desc_tail to t (release). The engine writes slot t % K (descriptor and metadata) only when desc_tail >= t - K (the
proxy finished round t - K): a ring that would overflow is a stalled proxy (proxy_proc.py fails it closed). A descriptor carries everything the round needs
from the batch: n, R, the slot of every request, a 16-byte digest of its request id (the slot owner key: equality is
all state.SlotTable uses), need_draft, k_window, and which slots' windows were staged with this round.

Reuse of the gather staging (meta, msg rows, resync windows) and of the reply area needs no extra protocol: the engine
stream writes round t+1's staging only after it passed round t's release and copied round t's reply (stream order),
and the proxy writes round t+1's reply only after it saw gathered >= t+1. The mapping is never unmapped or
unregistered while the stream may still touch it (the process keeps it until exit, like the thread proxy's page).
"""

from __future__ import annotations

import hashlib
import mmap
import os
from dataclasses import dataclass

import numpy as np

from . import wire
from .hostflag import PAGE_BYTES

MAGIC = 0x31565344          # "DSV1"
VERSION = 1
CTRL_BYTES = 65536
RING = 64                   # descriptor slots; > every lead the engine can build before a stalled round is failed
DIGEST_BYTES = 16

# control block: byte offsets of u32 words (hot words each on their own cache line)
C_MAGIC = 0
C_VERSION = 4
C_DESC_HEAD = 64            # engine: last round whose descriptor is written (release)
C_DESC_TAIL = 128           # proxy: last round whose descriptor it took (release)
C_STATE = 192               # proxy: S_* below
C_STOP = 256                # engine: 1 = stop (close)
C_PID = 320                 # proxy: its pid
C_DONE = 384                # proxy: last round it ran to the end (published or lost to the watchdog)
C_FLUSH_REQ = 448           # engine: stats flush requests (counter)
C_FLUSH_ACK = 512           # proxy: last flush request served
C_START = 576               # proxy: startup result R_* below (strict start: connect + HELLO)
C_ENGINE_SEQ = 640          # engine: seq of the engine-stats blob (odd: buffer 1, even: buffer 0)
C_WD = 704                  # engine: watchdog mirror, 3 x u64: fires, last_failed_round, max_late_ns
C_PROBE = 768               # engine stream: a D2H probe target (self-test of the region)
C_HEARTBEAT = 832           # proxy: CLOCK_MONOTONIC in ms (u32, wraps) at each housekeeping poll, <= 100 ms apart
                            # while it waits, once per round while it works (stall detection, S11 P-3)
C_START_REASON = 1024       # proxy: startup failure text (utf-8, NUL-padded), 1 KiB
START_REASON_BYTES = 1024
C_RESYNC = 4096             # proxy: one byte per slot, 1 = stage this slot's window with a later round
C_ENGINE_BLOB = 16384       # engine: 2 x ENGINE_BLOB_BYTES, JSON of the engine-side Stats snapshot
ENGINE_BLOB_BYTES = 24576
MAX_SLOTS = C_ENGINE_BLOB - C_RESYNC

S_STARTING, S_RUNNING, S_STOPPED, S_FAILED = 0, 1, 2, 3
R_PENDING, R_OK, R_FAILED = 0, 1, 2

F_ENQUEUE_FAILED = 1        # descriptor flag: the engine's enqueue raised; the proxy fails the round


def _mlock(addr: int, size: int) -> int:
    """mlock the mapping (resident for the copy engine and the proxy); 0 or the errno (e.g. a container memlock limit)."""
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    return 0 if libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(size)) == 0 else ctypes.get_errno()


def _align(x: int, a: int = 4096) -> int:
    return (x + a - 1) // a * a


def req_digest(req_id: str) -> bytes:
    return hashlib.blake2b(req_id.encode("utf-8", "surrogatepass"), digest_size=DIGEST_BYTES).digest()


@dataclass(frozen=True)
class RegionLayout:
    max_slots: int
    window: int
    row_bytes: int
    max_rows: int
    ring: int = RING
    sampled: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.max_slots <= MAX_SLOTS:
            raise ValueError(f"max_slots {self.max_slots} outside 1..{MAX_SLOTS}")
        if self.ring < 2:
            raise ValueError(f"descriptor ring {self.ring} < 2")

    @classmethod
    def for_hello(cls, hello: wire.Hello, ring: int = RING) -> "RegionLayout":
        return cls(max_slots=hello.max_slots, window=hello.window, row_bytes=hello.row_bytes,
                   max_rows=hello.max_rows_per_step, ring=ring, sampled=hello.sample_mode == wire.SAMPLE_TOP32)

    # descriptor: [t, n, R, flags, max_q, 0, 0, 0] u32, slots int64[M], k_window int32[M], need_draft u8[M],
    # staged u8[M], digests [M x 16]
    @property
    def d_slots(self) -> int:
        return 32

    @property
    def d_kwin(self) -> int:
        return self.d_slots + 8 * self.max_slots

    @property
    def d_need(self) -> int:
        return self.d_kwin + 4 * self.max_slots

    @property
    def d_staged(self) -> int:
        return self.d_need + self.max_slots

    @property
    def d_digest(self) -> int:
        return _align(self.d_staged + self.max_slots, 16)

    @property
    def desc_bytes(self) -> int:
        return _align(self.d_digest + DIGEST_BYTES * self.max_slots, 64)

    @property
    def msg_capacity(self) -> int:
        return wire.APP_HEADER_SIZE + wire.rows_offset_for(self.max_slots, self.sampled) + self.max_rows * self.row_bytes

    @property
    def off_ctrl(self) -> int:
        return PAGE_BYTES

    @property
    def off_desc(self) -> int:
        return self.off_ctrl + CTRL_BYTES

    @property
    def off_meta(self) -> int:
        return _align(self.off_desc + self.ring * self.desc_bytes)

    @property
    def meta_bytes(self) -> int:
        return self.ring * (6 if self.sampled else 4) * self.max_slots * 8

    @property
    def off_msg(self) -> int:
        return _align(self.off_meta + self.meta_bytes)

    @property
    def off_resync(self) -> int:
        return _align(self.off_msg + self.msg_capacity)

    @property
    def size(self) -> int:
        return _align(self.off_resync + self.max_slots * self.window * self.row_bytes)

    def to_dict(self) -> dict:
        return {"max_slots": self.max_slots, "window": self.window, "row_bytes": self.row_bytes,
                "max_rows": self.max_rows, "ring": self.ring, "sampled": self.sampled}


@dataclass
class Descriptor:
    t: int
    n: int
    R: int
    flags: int
    max_q: int              # longest query span of the step (phase attribution)
    slots: np.ndarray       # int64 [n]
    k_window: np.ndarray    # int64 [n]
    need_draft: np.ndarray  # bool [n]
    staged: frozenset[int]
    owners: list[str]       # hex digests of the request ids [n]


class SharedRegion:
    """The mapping and its numpy views. create() in the engine process; attach(fd) in the proxy process."""

    def __init__(self, layout: RegionLayout, fd: int, mm: mmap.mmap, created: bool):
        self.layout = L = layout
        self.fd = fd
        self.mm = mm
        self.created = created
        self.buf = np.frombuffer(mm, dtype=np.uint8)
        self.base = int(self.buf.ctypes.data)
        self.page_addr = self.base
        self.ctrl_addr = self.base + L.off_ctrl
        self.ctrl = self.buf[L.off_ctrl:L.off_ctrl + CTRL_BYTES]
        self.resync_flags = self.ctrl[C_RESYNC:C_RESYNC + L.max_slots]
        self.meta_np = self.buf[L.off_meta:L.off_meta + L.meta_bytes].view(np.int64).reshape(L.ring, 6 if L.sampled else 4, L.max_slots)
        self.msg_np = self.buf[L.off_msg:L.off_msg + L.msg_capacity]
        self.resync_np = self.buf[L.off_resync:L.off_resync + L.max_slots * L.window * L.row_bytes].reshape(
            L.max_slots, L.window, L.row_bytes)
        d = self.buf[L.off_desc:L.off_desc + L.ring * L.desc_bytes].reshape(L.ring, L.desc_bytes)
        M = L.max_slots
        self._d_hdr = d[:, 0:32].view(np.uint32)                                   # [K, 8]
        self._d_slots = d[:, L.d_slots:L.d_slots + 8 * M].view(np.int64)            # [K, M]
        self._d_kwin = d[:, L.d_kwin:L.d_kwin + 4 * M].view(np.int32)               # [K, M]
        self._d_need = d[:, L.d_need:L.d_need + M]                                  # [K, M]
        self._d_staged = d[:, L.d_staged:L.d_staged + M]                            # [K, M]
        self._d_digest = d[:, L.d_digest:L.d_digest + DIGEST_BYTES * M].reshape(L.ring, M, DIGEST_BYTES)
        if created:
            self.ctrl[C_MAGIC:C_MAGIC + 4].view(np.uint32)[0] = MAGIC
            self.ctrl[C_VERSION:C_VERSION + 4].view(np.uint32)[0] = VERSION
        else:
            magic = int(self.ctrl[C_MAGIC:C_MAGIC + 4].view(np.uint32)[0])
            ver = int(self.ctrl[C_VERSION:C_VERSION + 4].view(np.uint32)[0])
            if (magic, ver) != (MAGIC, VERSION):
                raise ValueError(f"shared region: magic {magic:#x} version {ver}, expected {MAGIC:#x} {VERSION}")

    @classmethod
    def create(cls, layout: RegionLayout) -> "SharedRegion":
        fd = os.memfd_create("dsv41-remote-dspark", os.MFD_CLOEXEC)
        try:
            os.ftruncate(fd, layout.size)
            mm = mmap.mmap(fd, layout.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        except BaseException:
            os.close(fd)
            raise
        reg = cls(layout, fd, mm, created=True)
        reg.mlock_errno = _mlock(reg.base, layout.size)
        return reg

    mlock_errno: int | None = None      # best effort, as the hand-off page (S9 F7: GB10 registration does not pin)

    @classmethod
    def attach(cls, layout: RegionLayout, fd: int) -> "SharedRegion":
        size = os.fstat(fd).st_size
        if size != layout.size:
            raise ValueError(f"shared region fd {fd}: {size} bytes, layout expects {layout.size}")
        mm = mmap.mmap(fd, layout.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        return cls(layout, fd, mm, created=False)

    _VIEWS = ("buf", "ctrl", "resync_flags", "meta_np", "msg_np", "resync_np", "_d_hdr", "_d_slots", "_d_kwin",
              "_d_need", "_d_staged", "_d_digest")

    @property
    def closed(self) -> bool:
        return self.mm is None

    def close(self) -> None:
        """Unmap the region and close its memfd: both of the process's fds (this one and mmap's own dup) and the
        mapping, mlocked pages included. Only on a refusal path, when nothing uses the region any more: no CUDA
        registration (unregister first), no stream operation, no proxy, no other view. Drops this object's views and
        closes the fd first; raises BufferError while another view of the mapping is alive (call again once it is
        gone)."""
        if self.mm is None:
            return
        for name in self._VIEWS:
            setattr(self, name, None)
        if self.fd >= 0:
            fd, self.fd = self.fd, -1
            os.close(fd)
        self.mm.close()
        self.mm = None

    # -------------------------------------------------- control words (addresses for hostflag's atomics)

    def word(self, off: int) -> int:
        return self.ctrl_addr + off

    def u32(self, off: int) -> np.ndarray:
        return self.ctrl[off:off + 4].view(np.uint32)

    def u64(self, off: int, n: int = 1) -> np.ndarray:
        return self.ctrl[off:off + 8 * n].view(np.uint64)

    def set_start_reason(self, text: str) -> None:
        b = text.encode("utf-8", "replace")[:START_REASON_BYTES - 1]
        area = self.ctrl[C_START_REASON:C_START_REASON + START_REASON_BYTES]
        area[:] = 0
        area[:len(b)] = np.frombuffer(b, dtype=np.uint8)

    def start_reason(self) -> str:
        area = bytes(self.ctrl[C_START_REASON:C_START_REASON + START_REASON_BYTES])
        return area.split(b"\0", 1)[0].decode("utf-8", "replace")

    # -------------------------------------------------- engine-stats blob (double-buffered; stats only)

    def write_engine_blob(self, seq: int, data: bytes) -> bool:
        """Into buffer seq % 2; the caller then release-stores seq at C_ENGINE_SEQ. False if it does not fit."""
        if len(data) > ENGINE_BLOB_BYTES - 4:
            return False
        o = C_ENGINE_BLOB + (seq % 2) * ENGINE_BLOB_BYTES
        self.ctrl[o:o + 4].view(np.uint32)[0] = len(data)
        self.ctrl[o + 4:o + 4 + len(data)] = np.frombuffer(data, dtype=np.uint8)
        return True

    def read_engine_blob(self, seq: int) -> bytes:
        o = C_ENGINE_BLOB + (seq % 2) * ENGINE_BLOB_BYTES
        n = int(self.ctrl[o:o + 4].view(np.uint32)[0])
        return bytes(self.ctrl[o + 4:o + 4 + min(n, ENGINE_BLOB_BYTES - 4)])

    # -------------------------------------------------- descriptors

    def write_descriptor(self, t: int, n: int, R: int, flags: int, slots, k_window, need_draft,
                         staged: frozenset[int], req_ids: list[str], max_q: int = 0) -> None:
        """Engine thread, before desc_head advances to t."""
        L = self.layout
        if n > L.max_slots:
            raise ValueError(f"{n} requests > max_slots {L.max_slots}")
        k = t % L.ring
        self._d_hdr[k, 1:5] = (n, R, flags, min(int(max_q), 0xFFFFFFFF))
        if n:
            self._d_slots[k, :n] = slots[:n]
            self._d_kwin[k, :n] = np.minimum(np.asarray(k_window[:n], dtype=np.int64), 0x7FFFFFFF)
            self._d_need[k, :n] = np.asarray(need_draft[:n], dtype=bool)
            for i in range(n):
                self._d_digest[k, i] = np.frombuffer(req_digest(req_ids[i]), dtype=np.uint8)
        st = self._d_staged[k]
        st[:] = 0
        for sl in staged:
            st[sl] = 1
        self._d_hdr[k, 0] = t & 0xFFFFFFFF

    def read_descriptor(self, t: int) -> Descriptor:
        """Proxy process, after desc_head >= t (acquire): a private copy of descriptor t."""
        L = self.layout
        k = t % L.ring
        seq, n, R, flags, max_q = (int(x) for x in self._d_hdr[k, :5])
        if seq != t & 0xFFFFFFFF:
            raise RuntimeError(f"descriptor slot {k} holds round {seq}, expected {t & 0xFFFFFFFF}")
        if n > L.max_slots:
            raise RuntimeError(f"descriptor {t}: {n} requests > max_slots {L.max_slots}")
        return Descriptor(
            t=t, n=n, R=R, flags=flags, max_q=max_q,
            slots=self._d_slots[k, :n].copy(),
            k_window=self._d_kwin[k, :n].astype(np.int64),
            need_draft=self._d_need[k, :n].astype(bool),
            staged=frozenset(int(i) for i in np.flatnonzero(self._d_staged[k])),
            owners=[bytes(self._d_digest[k, i]).hex() for i in range(n)])
