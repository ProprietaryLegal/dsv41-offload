# SPDX-License-Identifier: Apache-2.0
"""Hand-off between the engine stream and the round's proxy thread (async mode).

The engine stream must not run past a remote round before its drafts are in device memory, and the CPU must not block
either. Words and the reply live in one C-owned page of registered, device-mapped host memory:

    engine thread (enqueues, never blocks):
        gather(t) -> D2H rows/meta -> [WRITE gathered = t] -> [WAIT released >= t] -> H2D reply -> TP broadcast ...
    proxy thread (no CUDA call):
        wait gathered >= t (acquire) -> network round -> build the reply in a private buffer ->
        publish(t): claim t, copy the reply into the page, release t (one GIL-free C call)
    watchdog (C thread, no Python, no CUDA):
        gathered > released for longer than the deadline -> claim that round, write a failed reply, release it

Why no CUDA on the proxy: the first async design (a side stream on the proxy thread wrote a device flag after an H2D
copy) deadlocked on GB10: a CUDA call on the proxy waited behind a module load on the engine thread, and the module load
waited for the device, whose stream was parked on the flag (offload/results/C/lazy-repro/RESULT.md). A host word is
released by a CPU store, which needs no CUDA lock. The engine's Engram spin reader uses the same handshake on GB10
(engram_disk.cpp enable_spin / poll): malloc memory registered Mapped|Portable (not cudaHostAlloc, which GB10's CPU
reaches over an uncached path), cuStreamWriteValue32 / cuStreamWaitValue32 on its device pointer, CPU acquire loads /
release stores.

Why the watchdog (review S9, F2/F5): the proxy is a Python thread. If it cannot run - the engine thread waits for the
parked stream inside an extension that holds the GIL (P1), a GC finalizer on the proxy makes a CUDA call that blocks
behind a module load, the proxy died (P2), the interpreter is shutting down - nothing would release the stream, and
cuStreamWaitValue32 has no timeout. The watchdog bounds every such case to one fail-closed round (zero drafts).
CUDA_MODULE_LOADING=EAGER stays in the async configs (S9 F1): it removes the first-launch module loads that make an
extension wait for the parked stream in the first place.

Ordering:
  * gathered: CU_STREAM_WRITE_VALUE_DEFAULT puts a stream-scoped system fence before the write, so the D2H rows and
    metadata enqueued before it are visible to the CPU once the word is; the proxy's acquire load orders its reads after.
  * released: publish / fail-release write the reply, then release-store the word (monotonic); the engine stream's H2D
    copy of the reply is enqueued after the wait, so the copy engine reads the page only after the wait passed.
  * reply / staging reuse: the proxy or the watchdog writes round t's reply only after gathered >= t, i.e. after the
    stream ran everything before round t's gather, including round t-1's reply copy.
  * claim word: exactly one of the proxy's publish and the watchdog's failure writes and releases a given round.
  * every release is monotonic (cyclic max): nothing can move the word backwards (S9 F3).
"""

from __future__ import annotations

import ctypes
import gc
import time

import numpy as np
import torch

from .hostflag import (  # noqa: F401 -- re-exported: the host side moved to hostflag.py (torch-free)
    _M32, _SRC, _V, CU_MEMHOSTREGISTER_DEVICEMAP, CU_MEMHOSTREGISTER_PORTABLE, CU_STREAM_WAIT_VALUE_GEQ,
    CU_STREAM_WRITE_VALUE_DEFAULT, FAR_AHEAD, HOSTFLAG_LIB, HOSTFLAG_VERSION, OFF_CLAIMED, OFF_CONSUMED, OFF_GATHERED,
    OFF_RELEASED,
    OFF_REPLY, PAGE_BYTES, REPLY_CAPACITY, HostFlagCore, StreamStuckError, StreamSyncError, _bind, _compile,
    _private_dir, load_hostflag)

# ---------------------------------------------------------------------------------------------------- CUDA side

class _Driver:
    """The few driver entry points, from the already loaded libcuda (no link-time dependency)."""

    def __init__(self):
        try:
            lib = ctypes.CDLL("libcuda.so.1")
        except OSError as e:
            raise StreamSyncError(f"libcuda.so.1: {e}") from e

        def sym(*names):
            for n in names:
                f = getattr(lib, n, None)
                if f is not None:
                    return f
            raise StreamSyncError(f"libcuda has none of {names}")

        self.wait = sym("cuStreamWaitValue32_v2", "cuStreamWaitValue32")
        self.write = sym("cuStreamWriteValue32_v2", "cuStreamWriteValue32")
        for f in (self.wait, self.write):
            f.argtypes = [_V, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint]
            f.restype = ctypes.c_int
        self.register = sym("cuMemHostRegister_v2", "cuMemHostRegister")
        self.register.argtypes = [_V, ctypes.c_size_t, ctypes.c_uint]
        self.register.restype = ctypes.c_int
        self.unregister = sym("cuMemHostUnregister")
        self.unregister.argtypes = [_V]
        self.unregister.restype = ctypes.c_int
        self.devptr = sym("cuMemHostGetDevicePointer_v2", "cuMemHostGetDevicePointer")
        self.devptr.argtypes = [ctypes.POINTER(ctypes.c_uint64), _V, ctypes.c_uint]
        self.devptr.restype = ctypes.c_int
        self.h2d = sym("cuMemcpyHtoDAsync_v2", "cuMemcpyHtoDAsync")
        self.h2d.argtypes = [ctypes.c_uint64, _V, ctypes.c_size_t, _V]
        self.h2d.restype = ctypes.c_int
        self.d2h = sym("cuMemcpyDtoHAsync_v2", "cuMemcpyDtoHAsync")
        self.d2h.argtypes = [_V, ctypes.c_uint64, ctypes.c_size_t, _V]
        self.d2h.restype = ctypes.c_int


def _check(r: int, what: str) -> None:
    if r != 0:
        raise StreamSyncError(f"{what} failed: CUresult {r}")


class RegisteredRegion:
    """A process proxy's shared region (shmregion.SharedRegion) registered with CUDA once, Portable | DeviceMap (memfd
    pages, ordinary cached memory: the property the hand-off page always had on GB10). The engine stream writes the
    gather staging into it, reads the reply from it and runs its stream memory operations on its hostflag page.
    Unregistered only by unregister(), after the stream drained; normally never (the process exit releases it)."""

    def __init__(self, region, device: torch.device):
        self.region = region
        self.device = torch.device(device)
        self.base = region.base
        self.size = region.layout.size
        self.page_addr = region.page_addr
        self.drv = _Driver()
        torch.cuda.set_device(self.device)
        torch.cuda.synchronize(self.device)
        _check(self.drv.register(_V(self.base), self.size, CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP),
               "cuMemHostRegister(region)")
        d = ctypes.c_uint64(0)
        r = self.drv.devptr(ctypes.byref(d), _V(self.base), 0)
        if r != 0:
            self.drv.unregister(_V(self.base))
            _check(r, "cuMemHostGetDevicePointer(region)")
        self.dev_base = int(d.value)
        self.registered = True

    def dev_addr(self, host_addr: int) -> int:
        off = host_addr - self.base
        if not 0 <= off < self.size:
            raise StreamSyncError(f"address {host_addr:#x} outside the registered region")
        return self.dev_base + off

    def d2h(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        """Enqueue src (device, contiguous) -> dst (a contiguous view of this region) on the current stream with the
        driver's async copy: registered memory, so it never degrades to a synchronous pageable copy."""
        n = src.numel() * src.element_size()
        if (src.device.type != "cuda" or not src.is_contiguous() or not dst.is_contiguous()
                or dst.numel() * dst.element_size() != n):
            raise StreamSyncError(f"region copy {tuple(src.shape)} {src.dtype} on {src.device} -> "
                                  f"{tuple(dst.shape)} {dst.dtype} on {dst.device}")
        if n == 0:
            return
        self.dev_addr(dst.data_ptr() + n - 1)        # the whole destination lies in the region
        s = torch.cuda.current_stream(self.device)
        _check(self.drv.d2h(_V(dst.data_ptr()), src.data_ptr(), n, _V(s.cuda_stream)), "cuMemcpyDtoHAsync(region)")

    def probe_tensor(self) -> torch.Tensor:
        """int32 [1] at the control block's probe word (self-test target)."""
        from .shmregion import C_PROBE

        L = self.region.layout
        return torch.frombuffer(self.region.mm, dtype=torch.int32, count=1, offset=L.off_ctrl + C_PROBE)

    def unregister(self) -> None:
        """Only after the stream drained and the proxy stopped."""
        if self.registered:
            torch.cuda.synchronize(self.device)
            self.drv.unregister(_V(self.base))
            self.registered = False


class HostFlagSync(HostFlagCore):
    """Rank 0's async hand-off on a CUDA device. Engine-thread calls (gathered, wait_release, copy_reply, self_test,
    attach, close) may use CUDA; proxy-thread calls (wait_gathered, publish) and the watchdog never do."""

    def __init__(self, device: torch.device, lib_path: str = "", spin_us: int = 1000, nap_us: int = 0,
                 region=None):
        """region: None = a page of its own, registered here; else a registered shared region (RegisteredRegion,
        process proxy) whose hostflag page this uses and whose owner unregisters it."""
        device = torch.device(device)
        if device.type != "cuda":
            raise StreamSyncError("HostFlagSync needs a CUDA device")
        super().__init__(lib_path, spin_us, nap_us, page=region.page_addr if region is not None else 0)
        self.device = device
        self.region = region
        self.shared = region.region if region is not None else None     # shmregion.SharedRegion (process proxy)
        if region is not None:
            self.drv = region.drv
            self.page_dev = region.dev_addr(self.page)
            self.gathered_dev = self.page_dev + OFF_GATHERED
            self.released_dev = self.page_dev + OFF_RELEASED
            self.consumed_dev = self.page_dev + OFF_CONSUMED
            return
        try:
            self.drv = _Driver()
            # the driver calls below need the device's primary context current on this thread
            torch.cuda.set_device(self.device)
            torch.cuda.synchronize(self.device)
            _check(self.drv.register(_V(self.page), PAGE_BYTES,
                                     CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP),
                   "cuMemHostRegister")
        except BaseException:
            HostFlagCore.close(self)
            raise
        d = ctypes.c_uint64(0)
        r = self.drv.devptr(ctypes.byref(d), _V(self.page), 0)
        if r != 0:     # S9 F7: do not leak the registration
            self.drv.unregister(_V(self.page))
            HostFlagCore.close(self)
            _check(r, "cuMemHostGetDevicePointer")
        self.page_dev = int(d.value)
        self.gathered_dev = self.page_dev + OFF_GATHERED
        self.released_dev = self.page_dev + OFF_RELEASED
        self.consumed_dev = self.page_dev + OFF_CONSUMED

    # -------------------------------------------------- engine thread (stream-ordered, never blocks the CPU)

    def gathered(self, value: int, stream: torch.cuda.Stream | None = None) -> None:
        """Enqueue "gathered = value" after everything already on the stream (fenced)."""
        s = stream or torch.cuda.current_stream(self.device)
        _check(self.drv.write(_V(s.cuda_stream), self.gathered_dev, value & _M32, CU_STREAM_WRITE_VALUE_DEFAULT),
               "cuStreamWriteValue32(gathered)")

    def wait_release(self, value: int, stream: torch.cuda.Stream | None = None) -> None:
        """Enqueue a wait until released >= value (wrap-safe)."""
        s = stream or torch.cuda.current_stream(self.device)
        _check(self.drv.wait(_V(s.cuda_stream), self.released_dev, value & _M32, CU_STREAM_WAIT_VALUE_GEQ),
               "cuStreamWaitValue32(released)")

    def consumed(self, value: int, stream: torch.cuda.Stream | None = None) -> None:
        """Enqueue "consumed = value" after the reply copy (fenced): the explicit consumer acknowledgment."""
        s = stream or torch.cuda.current_stream(self.device)
        _check(self.drv.write(_V(s.cuda_stream), self.consumed_dev, value & _M32, CU_STREAM_WRITE_VALUE_DEFAULT),
               "cuStreamWriteValue32(consumed)")

    def drain_marker(self, stream: torch.cuda.Stream | None = None) -> torch.cuda.Event:
        """An event recorded after everything enqueued on the stream so far (the respawn waits for it: no stream
        operation of the lost proxy's rounds may still reference the words when they are reset)."""
        ev = torch.cuda.Event()
        ev.record(stream or torch.cuda.current_stream(self.device))
        return ev

    def drained(self, marker) -> bool:
        """cudaEventQuery only: never a synchronization on a stream that may be parked."""
        return marker is None or bool(marker.query())

    def d2h(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        """Process proxy: a stream-ordered copy into the registered shared region."""
        if self.region is None:
            raise StreamSyncError("d2h needs a registered shared region")
        self.region.d2h(dst, src)

    def copy_reply(self, dst: torch.Tensor, nbytes: int | None = None,
                   stream: torch.cuda.Stream | None = None) -> None:
        """Enqueue the H2D copy of the reply area into dst (async: the page is registered memory)."""
        n = self.reply_bytes if nbytes is None else int(nbytes)
        if dst.device.type != "cuda" or not dst.is_contiguous() or dst.numel() * dst.element_size() < n:
            raise StreamSyncError(f"reply destination {tuple(dst.shape)} {dst.dtype} on {dst.device} cannot take "
                                  f"{n} bytes")
        s = stream or torch.cuda.current_stream(self.device)
        _check(self.drv.h2d(dst.data_ptr(), _V(self.reply_host), n, _V(s.cuda_stream)), "cuMemcpyHtoDAsync(reply)")

    # -------------------------------------------------- setup

    def describe(self) -> dict:
        d = super().describe()
        d["device"] = str(self.device)
        return d

    def self_test(self, timeout_s: float = 2.0) -> tuple[bool, str]:
        """On this device: (1) the CPU sees data the stream copied before "gathered" once it sees the word; (2) the
        stream does not pass the wait before the CPU's publish, and the reply copy after the wait reads the bytes the
        publish wrote (the production path: claim, copy into the registered page, release, driver H2D). The rescue is
        a CPU store; if the stream still does not drain, this device does not observe host stores and the stream
        cannot be recovered: StreamStuckError (S9 F4)."""
        if self._wd is not None:
            raise StreamSyncError("self-test after attach")
        dev = self.device
        main = torch.cuda.current_stream(dev)
        self.reset()
        y = torch.zeros(1, dtype=torch.int32, device=dev)
        x = torch.zeros(4, dtype=torch.int32, device=dev)
        h_out = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        # every kernel / copy this test launches runs once before anything parks the stream (lazy module loading)
        torch.cuda._sleep(1)
        y.fill_(5)
        h_out.copy_(y, non_blocking=True)
        self.copy_reply(x, 16, main)
        torch.cuda.synchronize(dev)
        y.zero_()
        x.zero_()
        torch.cuda.synchronize(dev)
        try:
            # (1) gathered: a ~10 ms kernel, then the value the CPU must see with the word
            torch.cuda._sleep(20_000_000)
            y.fill_(7)
            h_out.copy_(y, non_blocking=True)
            probe = None
            if self.region is not None:
                # the process proxy's staging path: a driver D2H into the shared region, read by another process
                probe = self.region.probe_tensor()
                probe.zero_()
                self.region.d2h(probe, y)
            self.gathered(1, main)
            if not self.wait_gathered(1, timeout_s):
                return False, "gathered word not written within the timeout"
            if int(h_out.numpy()[0]) != 7:
                return False, f"CPU saw gathered before the copied data ({int(h_out.numpy()[0])})"
            if probe is not None and int(probe.numpy()[0]) != 7:
                return False, f"CPU saw gathered before the region copy ({int(probe.numpy()[0])})"
            # (2) released: the stream parks; the CPU publishes the reply
            self.wait_release(1, main)
            self.copy_reply(x, 16, main)
            self.consumed(1, main)
            done = torch.cuda.Event()
            done.record(main)
            time.sleep(0.02)
            if done.query():
                return False, "stream passed the wait before the release"
            if not self.publish(1, np.array([9, 8, 7, 6], dtype=np.int32)):
                return False, "publish did not claim round 1"
            t0 = time.monotonic()
            while not done.query():
                if time.monotonic() - t0 > timeout_s:
                    return False, "stream did not pass the wait within the timeout after the release"
                time.sleep(0.001)
            got = x.tolist()
            if got != [9, 8, 7, 6]:
                return False, f"stream copied {got} after the release, expected [9, 8, 7, 6]"
            if not self.reply_consumed(1):
                return False, "stream did not acknowledge the reply copy (consumed word)"
            return True, "ok"
        finally:
            self.release(0x7FFFFFFF)        # rescue: never leave the stream parked (a CPU store)
            drained = torch.cuda.Event()
            drained.record(main)
            t0 = time.monotonic()
            while not drained.query():
                if time.monotonic() - t0 > max(5.0, 2 * timeout_s):
                    raise StreamStuckError("the engine stream stayed parked after the host-word rescue: this device "
                                           "does not observe host stores; the stream cannot be recovered")
                time.sleep(0.001)
            self.reset()

    def close(self) -> None:
        """Only after the round stopped and the stream drained."""
        self.stop_watchdog()
        if self.page and self.region is None:
            torch.cuda.synchronize(self.device)
            self.drv.unregister(_V(self.page))
        HostFlagCore.close(self)


def release_hand_off(sync=None, registered=None, shared=None) -> str | None:
    """Undo a refused or abandoned async hand-off at startup, before any stream operation references it (S11 P-2):
    stop the watchdog, unregister the process proxy's shared region from CUDA and unmap it (both fds and the mapping:
    about 30 MiB, mlocked). Each argument is optional; the sync's own region and shared region are found from it.
    After this the sync object must not be used again (its page address is cleared, so the atexit hook is a no-op).
    Returns None, or why the mapping is still there (another view of it is alive; the caller logs it)."""
    if sync is not None:
        if registered is None and isinstance(getattr(sync, "region", None), RegisteredRegion):
            registered = sync.region
        if shared is None:
            shared = getattr(sync, "shared", None)
        sync.close()
    if registered is not None:
        if shared is None:
            shared = registered.region
        registered.unregister()
    if shared is None or shared.closed:
        return None
    try:
        shared.close()
    except BufferError:
        gc.collect()            # views held by a failed constructor's frames, freed with their exception
        try:
            shared.close()
        except BufferError as e:
            return f"shared region still mapped ({e})"
    return None


# ---------------------------------------------------------------------------------------------------- CPU stand-in

class CpuHostFlagSync(HostFlagCore):
    """CPU tests: the real C words, publish, fail-all and watchdog; the "stream" is the calling thread. gathered() stores
    the word at once (CPU copies are synchronous), wait_release() blocks the engine thread in the C wait (GIL released)
    until released, copy_reply() copies the reply area into a CPU tensor."""

    def __init__(self, lib_path: str = "", timeout_s: float = 10.0, region=None):
        super().__init__(lib_path, spin_us=200, nap_us=20, page=region.page_addr if region is not None else 0)
        self.timeout_s = timeout_s
        self.region = region
        self.shared = region            # shmregion.SharedRegion (process proxy)

    def d2h(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        dst.copy_(src)

    def gathered(self, value: int, stream=None) -> None:
        self.lib.dsv41_hf_release_max(_V(self.gathered_host), value & _M32)

    def wait_release(self, value: int, stream=None) -> None:
        r = self.lib.dsv41_hf_wait_geq(_V(self.released_host), value & _M32, int(self.timeout_s * 1e9),
                                       self.spin_ns, self.nap_ns)
        if r != 0:
            raise StreamSyncError(f"wait for release {value} timed out (words {self.values()})")

    def copy_reply(self, dst: torch.Tensor, nbytes: int | None = None, stream=None) -> None:
        n = self.reply_bytes if nbytes is None else int(nbytes)
        if dst.device.type != "cpu" or not dst.is_contiguous() or dst.numel() * dst.element_size() < n:
            raise StreamSyncError(f"reply destination cannot take {n} bytes")
        ctypes.memmove(dst.data_ptr(), self.reply_host, n)

    def describe(self) -> dict:
        d = super().describe()
        d["mechanism"] = "host_flags_cpu"
        return d

    def self_test(self, timeout_s: float = 2.0) -> tuple[bool, str]:
        return True, "cpu"


# name of the v1 CPU stand-in (tests, reviewers' probes)
FakeHostFlagSync = CpuHostFlagSync
