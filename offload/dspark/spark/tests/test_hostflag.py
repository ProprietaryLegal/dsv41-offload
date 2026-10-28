# SPDX-License-Identifier: Apache-2.0
"""hostflag.c through streamsync.load_hostflag (built with the system C compiler when no prebuilt library is given):
the proxy thread's CPU side of the async hand-off. Acquire waits, release stores, wrap-safe GEQ, timeouts, and that
a waiting call releases the GIL."""

import ctypes
import shutil
import threading
import time

import pytest

from remote_dspark import streamsync

pytestmark = pytest.mark.skipif(shutil.which("cc") is None and shutil.which("gcc") is None,
                                reason="no C compiler")


@pytest.fixture(scope="module")
def lib():
    lib, origin = streamsync.load_hostflag("")
    assert "compiled" in origin or origin.endswith(streamsync.HOSTFLAG_LIB)
    return lib


@pytest.fixture
def word(lib):
    p = lib.dsv41_hf_alloc(4096)
    assert p
    yield int(p)
    lib.dsv41_hf_free(ctypes.c_void_p(p))


def test_alloc_is_zeroed_and_aligned(lib, word):
    assert word % 4096 == 0
    assert all(lib.dsv41_hf_load(ctypes.c_void_p(word + 4 * i)) == 0 for i in range(64))


def test_wait_returns_on_store_from_another_thread(lib, word):
    def later():
        time.sleep(0.05)
        lib.dsv41_hf_store(ctypes.c_void_p(word), 5)

    th = threading.Thread(target=later)
    t0 = time.perf_counter()
    th.start()
    r = lib.dsv41_hf_wait_geq(ctypes.c_void_p(word), 5, int(2e9), 0, 0)
    dt = time.perf_counter() - t0
    th.join()
    assert r == 0 and 0.04 < dt < 1.0


def test_wait_times_out(lib, word):
    t0 = time.perf_counter()
    assert lib.dsv41_hf_wait_geq(ctypes.c_void_p(word), 1, int(30e6), 0, 0) == 1
    assert 0.025 < time.perf_counter() - t0 < 0.5
    # napping mode times out too
    assert lib.dsv41_hf_wait_geq(ctypes.c_void_p(word), 1, int(30e6), int(1e6), int(200e3)) == 1


def test_geq_is_wrap_safe(lib, word):
    lib.dsv41_hf_store(ctypes.c_void_p(word), 5)
    # 5 is "after" 0xFFFFFFF0 in wrapped order (as CU_STREAM_WAIT_VALUE_GEQ compares)
    assert lib.dsv41_hf_wait_geq(ctypes.c_void_p(word), 0xFFFFFFF0, 0, 0, 0) == 0
    assert lib.dsv41_hf_wait_geq(ctypes.c_void_p(word), 6, 0, 0, 0) == 1
    assert lib.dsv41_hf_wait_geq(ctypes.c_void_p(word), 5, 0, 0, 0) == 0


def test_wait_releases_the_gil(lib, word):
    """A thread spinning in the wait must not stop Python threads from running."""
    counter = {"n": 0}
    stop = threading.Event()

    def spin_python():
        while not stop.is_set():
            counter["n"] += 1

    th = threading.Thread(target=spin_python)
    th.start()
    try:
        assert lib.dsv41_hf_wait_geq(ctypes.c_void_p(word), 1, int(100e6), 0, 0) == 1
        n_during = counter["n"]
    finally:
        stop.set()
        th.join()
    assert n_during > 10000      # the Python thread ran the whole 100 ms


def test_explicit_missing_library_is_refused():
    with pytest.raises(streamsync.StreamSyncError, match="does not exist"):
        streamsync.load_hostflag("/nonexistent/libdsv41_hostflag.so")
