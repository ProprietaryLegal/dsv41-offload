"""NVLink peer-to-peer collectives for the TP = 2 worker (p2p.cu): build / load and the per-rank state.

The extension is compiled once with torch.utils.cpp_extension (nvcc from CUDA 12.6, sm_86 only) into
$DSPARK_P2P_BUILD (default ~/dsv41-offload/e/build/p2p); later loads reuse the build while the source is unchanged.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Iterable
from types import ModuleType

import torch

_EXT: ModuleType | None = None
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p2p.cu")
DEFAULT_TIMEOUT_MS = 2000.0


def ext() -> ModuleType:
    global _EXT
    if _EXT is None and os.environ.get("DSPARK_P2P_PREBUILT", "").strip():
        # production: the pinned build of exactly this p2p.cu, never a compile at start (prebuilt.py)
        from .prebuilt import load_prebuilt
        _EXT = load_prebuilt(os.environ["DSPARK_P2P_PREBUILT"].strip(), SRC)
    if _EXT is None:
        from torch.utils.cpp_extension import load
        cuda = os.environ.get("DSPARK_P2P_CUDA_HOME", "/usr/local/cuda-12.6")
        os.environ["CUDA_HOME"] = cuda
        if os.path.join(cuda, "bin") not in os.environ.get("PATH", "").split(":"):
            os.environ["PATH"] = os.path.join(cuda, "bin") + ":" + os.environ.get("PATH", "")
        build = os.path.expanduser(os.environ.get("DSPARK_P2P_BUILD", "~/dsv41-offload/e/build/p2p"))
        os.makedirs(build, exist_ok=True)
        _EXT = load(name="dspark_p2p", sources=[SRC], build_directory=build, verbose=False,
                    extra_cflags=["-O2"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_86,code=sm_86"])
    return _EXT


def preload(dev: torch.device) -> None:
    """Launch every collective kernel once on `dev` with itself as the peer (each passes its own handshake at once).
    CUDA lazy loading may synchronize the context on a kernel's first launch; a first launch while a collective
    kernel spins on the same device, waiting for work the blocked host has not issued yet, would deadlock until the
    timeout (CUDA programming guide, lazy loading: concurrent execution). Run before any collective is in flight."""
    e = ext()
    with torch.cuda.device(dev):
        st = RankState(dev, 4 * 5120)
        x = torch.ones(5120, dtype=torch.float32, device=dev)
        e.ar_sum(x, torch.empty_like(x), st.buf, st.ptr, st.max_words, 1, int(1e9))
        e.ar_sum(x, torch.empty(5120, dtype=torch.bfloat16, device=dev), st.buf, st.ptr, st.max_words, 1, int(1e9))
        v = torch.zeros(1, dtype=torch.float32, device=dev)
        i = torch.zeros(1, dtype=torch.int64, device=dev)
        out = torch.zeros(1, 1, dtype=torch.int64, device=dev)
        e.xchg_argmax(v, i, 0, 1, out, 0, True, st.buf, st.ptr, st.max_words, int(1e9))
        e.xchg_margin(torch.zeros(1, 2, dtype=torch.float32, device=dev), i, 0, 1, out,
                      torch.zeros(1, 1, dtype=torch.float32, device=dev), 0, True, st.buf, st.ptr, st.max_words,
                      int(1e9))
        e.xchg_markov(i, torch.zeros(1, dtype=torch.int32, device=dev), torch.zeros(1, 1, device=dev),
                      torch.zeros(1, 1, dtype=torch.int64, device=dev), torch.zeros(1, 1, device=dev),
                      torch.zeros(1, 1, dtype=torch.int64, device=dev), 0, 1, out, 0, True, st.buf, st.ptr,
                      st.max_words, int(1e9))
        torch.cuda.synchronize(dev)
        h = [int(a) for a in st.hdr[:5].cpu()]
        if h[2] != 0 or h[4] != 5:
            raise RuntimeError(f"p2p preload on {dev}: header {h}")
        e.cancel(st.buf)
        e.reset(st.buf)
        torch.cuda.synchronize(dev)


class RankState:
    """One rank's state block (header, flags, double receive buffer) in its own device memory."""

    def __init__(self, dev: torch.device, max_words: int):
        self.dev = dev
        self.max_words = max_words
        self.buf = ext().allocate_state(max_words, dev.index)
        self.hdr = self.buf[:64].view(torch.int32)          # epoch, done, error, err_epoch, count, ...

    @property
    def ptr(self) -> int:
        return self.buf.data_ptr()


def check_allocator() -> None:
    """Async Torch allocation remains unsupported. Peer blocks use native cudaMalloc independently of Torch's
    expandable-segment configuration, including runtime changes made by exllamav3's Python import."""
    for var in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        v = os.environ.get(var, "").replace(" ", "").lower()
        if "backend:cudamallocasync" in v:
            raise RuntimeError(f"{var}={os.environ[var]!r}: the p2p worker has not qualified cudaMallocAsync")


def wait_idle(devs: Iterable[torch.device], timeout_s: float,
              streams: Iterable[torch.cuda.Stream] = ()) -> None:
    """Wait until every device's current stream is idle, at most timeout_s (raises TimeoutError): never an unbounded
    synchronize on a path that runs after a failure, where a wedged GPU would block the worker forever."""
    evs = []
    all_streams = dict.fromkeys([*streams, *(torch.cuda.current_stream(d) for d in dict.fromkeys(devs))])
    for stream in all_streams:
        d = stream.device
        with torch.cuda.device(d):
            ev = torch.cuda.Event()
            ev.record(stream)
            evs.append((d, ev))
    deadline = time.monotonic() + timeout_s
    for d, ev in evs:
        while not ev.query():
            if time.monotonic() > deadline:
                raise TimeoutError(f"device {d} did not go idle within {timeout_s:.1f} s")
            time.sleep(0.0005)


class P2PPair:
    """The two ranks' states with peer access enabled both ways. Collective calls launch on the calling rank's
    current stream; both ranks must issue the same sequence of collectives (TPDrafter does). vocab bounds the ids an
    exchange accepts from the peer."""

    def __init__(self, devs: list[torch.device], max_words: int, timeout_ms: float | None = None,
                 vocab: int = 1 << 62):
        if len(devs) != 2 or devs[0] == devs[1] or any(d.type != "cuda" or d.index is None for d in devs):
            raise ValueError(f"P2PPair needs two distinct devices, got {devs}")
        if max_words <= 0 or max_words % 4 or vocab <= 0:
            raise ValueError("P2PPair requires positive 4-aligned max_words and positive vocab")
        if timeout_ms is None:
            timeout_ms = float(os.environ.get("DSPARK_TP_TIMEOUT_MS", DEFAULT_TIMEOUT_MS))
        if not math.isfinite(timeout_ms) or timeout_ms <= 0:
            raise ValueError(f"P2PPair requires finite positive timeout_ms, got {timeout_ms}")
        check_allocator()
        self.poisoned: str | None = None
        self.launch_streams: set[torch.cuda.Stream] = set()
        self.vocab = int(vocab)
        e = ext()
        e.enable_peer(devs[0].index, devs[1].index)
        e.enable_peer(devs[1].index, devs[0].index)
        for d in devs:
            preload(d)
        self.devs = devs
        self.max_words = max_words
        self.timeout_ns = int(timeout_ms * 1e6)
        self.idle_timeout_s = max(10.0, 20 * timeout_ms / 1000.0)    # host waits on a failure path are bounded too
        self.st = [RankState(d, max_words) for d in devs]
        self.cancel_streams = [torch.cuda.Stream(d) for d in devs]
        for d in devs:
            torch.cuda.synchronize(d)

    def peer(self, r: int) -> int:
        return self.st[1 - r].ptr

    def poison(self, reason: str) -> None:
        """Host-only latch. A CUDA exception requires process restart, never header clearing or buffer reuse."""
        if self.poisoned is None:
            self.poisoned = reason

    def _usable(self) -> None:
        if self.poisoned is not None:
            raise RuntimeError(f"TP CUDA context poisoned; restart worker: {self.poisoned}")

    def _launch(self, r: int, name: str, *args: object) -> None:
        self._usable()
        try:
            # Retain every eager stream: checking only the caller's current stream can miss a parked collective.
            stream = torch.cuda.current_stream(self.devs[r])
            self.launch_streams.add(stream)
            getattr(ext(), name)(*args)
        except RuntimeError as error:
            self.poison(str(error))
            raise

    def ar_sum(self, r: int, x: torch.Tensor, out: torch.Tensor, nblocks: int) -> torch.Tensor:
        self._launch(r, "ar_sum", x, out, self.st[r].buf, self.peer(r), self.max_words, nblocks, self.timeout_ns)
        return out

    def xchg_argmax(self, r: int, val: torch.Tensor, idx: torch.Tensor, v_off: int, out: torch.Tensor,
                    col: int) -> None:
        self._launch(r, "xchg_argmax", val, idx, v_off, self.vocab, out, col, r == 0, self.st[r].buf, self.peer(r),
                          self.max_words, self.timeout_ns)

    def xchg_margin(self, r: int, top2: torch.Tensor, idx: torch.Tensor, v_off: int, out: torch.Tensor,
                    mg: torch.Tensor, col: int) -> None:
        self._launch(r, "xchg_margin", top2, idx, v_off, self.vocab, out, mg, col, r == 0, self.st[r].buf, self.peer(r),
                          self.max_words, self.timeout_ns)

    def xchg_markov(self, r: int, prev: torch.Tensor, slot_of_token: torch.Tensor, parts: tuple, v_off: int,
                    out: torch.Tensor, col: int) -> None:
        vals_c, idx_c, vals_u, idx_u = parts
        self._launch(r, "xchg_markov", prev, slot_of_token, vals_c, idx_c, vals_u, idx_u, v_off, self.vocab, out, col, r == 0,
                          self.st[r].buf, self.peer(r), self.max_words, self.timeout_ns)

    def headers(self) -> list[list[int]]:
        """Host read of both headers once both devices are idle (bounded wait: TimeoutError)."""
        self._usable()
        try:
            wait_idle(self.devs, self.idle_timeout_s, self.launch_streams)
            out = []
            for s in self.st:
                with torch.cuda.device(s.dev):
                    out.append([int(v) for v in s.hdr[:5].cpu()])
            return out
        except (RuntimeError, TimeoutError) as error:
            self.poison(str(error))
            raise

    def cancel(self) -> None:
        """Abort both ranks on independent, pre-created streams. Buffers remain owned until idle/reset completes."""
        self._usable()
        for r, stream in enumerate(self.cancel_streams):
            with torch.cuda.device(self.devs[r]), torch.cuda.stream(stream):
                self._launch(r, "cancel", self.st[r].buf)

    def reset(self) -> None:
        """Zero both headers and flag sets. Only with both devices idle (bounded waits before and after)."""
        self._usable()
        try:
            wait_idle(self.devs, self.idle_timeout_s, self.launch_streams)
            for s in self.st:
                with torch.cuda.device(s.dev):
                    ext().reset(s.buf)
            wait_idle(self.devs, self.idle_timeout_s)
            self.launch_streams.clear()
        except (RuntimeError, TimeoutError) as error:
            self.poison(str(error))
            raise
