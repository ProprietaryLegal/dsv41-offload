// SPDX-License-Identifier: Apache-2.0
// Host-memory hand-off for the async remote round (streamsync.HostFlagCore / HostFlagSync).
//
// The engine stream and the round's proxy thread hand a round back and forth through words in registered,
// device-mapped host memory: the stream writes "gathered = t" (cuStreamWriteValue32) and waits for "released >= t"
// (cuStreamWaitValue32) before copying the reply in. The engine's Engram spin reader uses the same handshake on GB10
// (engram_disk.cpp, enable_spin / poll).
//
// Version 2 (lane A2 after review S9): the release no longer depends on Python.
//   * Releases are monotonic (cyclic max): no caller can move the word backwards (S9 F3).
//   * publish(t): claim round t, copy the reply into the registered reply area, release t - one call, no GIL needed
//     once inside (S9 F5).
//   * A watchdog thread (pure C, no Python, no CUDA) fails and releases a round that was gathered but not released
//     within a deadline, so a proxy that cannot run (GIL held by a waiting extension, GC finalizer blocked on CUDA,
//     proxy death, interpreter shutdown) costs one fail-closed round instead of a parked GPU (S9 F2/F5).
//   * The claim word makes the proxy's publish and the watchdog's failure mutually exclusive per round.
// All words compare cyclically, as CU_STREAM_WAIT_VALUE_GEQ does: a is at or after b iff (int32)(a - b) >= 0.
//
// Called through ctypes.CDLL, which releases the GIL for the whole call. Build (any host; libc + pthread only):
//   cc -O2 -Wall -shared -fPIC -o libdsv41_hostflag.so hostflag.c -lpthread
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#define DSV41_HF_VERSION 2

static inline int64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static inline void cpu_relax(void) {
#if defined(__aarch64__)
  __asm__ __volatile__("isb" ::: "memory");  // as engram_disk.cpp's spin loop
#elif defined(__x86_64__)
  __builtin_ia32_pause();
#endif
}

static inline void nap(int64_t ns) {
  struct timespec ts = {(time_t)(ns / 1000000000LL), (long)(ns % 1000000000LL)};
  nanosleep(&ts, NULL);
}

// a is strictly after b, cyclically
static inline int after(uint32_t a, uint32_t b) { return (int32_t)(a - b) > 0; }

int dsv41_hf_version(void) { return DSV41_HF_VERSION; }

// One zeroed, page-aligned block for the words and the reply (registered by the caller; malloc memory, not
// cudaHostAlloc: GB10 reaches cudaHostAlloc memory over an uncached path, engram_disk.cpp). NULL on failure.
void* dsv41_hf_alloc(size_t bytes) {
  void* p = NULL;
  if (bytes == 0 || posix_memalign(&p, 4096, bytes) != 0) return NULL;
  memset(p, 0, bytes);
  return p;
}

void dsv41_hf_free(void* p) { free(p); }

// Best-effort mlock of the block (registration populates but does not pin pageable memory on GB10; S9 F7). 0 or errno.
int dsv41_hf_lock(void* p, size_t bytes) { return mlock(p, bytes) == 0 ? 0 : errno; }

uint32_t dsv41_hf_load(const uint32_t* addr) { return __atomic_load_n(addr, __ATOMIC_ACQUIRE); }

// Monotonic release store: advances the word to value only if value is cyclically after it; never moves it backwards.
// Every store this thread made before the call (the reply bytes) is visible before the new value.
void dsv41_hf_release_max(uint32_t* addr, uint32_t value) {
  uint32_t old = __atomic_load_n(addr, __ATOMIC_RELAXED);
  while (after(value, old)) {
    if (__atomic_compare_exchange_n(addr, &old, value, 0, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) return;
  }
}

// The release (and gathered) store: monotonic (S9 F3). Kept under its v1 name for callers of the plain store.
void dsv41_hf_store(uint32_t* addr, uint32_t value) { dsv41_hf_release_max(addr, value); }

// Plain release store that may go backwards: only for resetting the words while no round is in flight (setup,
// self-test), never on a release path.
void dsv41_hf_set(uint32_t* addr, uint32_t value) { __atomic_store_n(addr, value, __ATOMIC_RELEASE); }

// Claim round t on the claim word: 1 if this caller now owns t (the word advanced to t), 0 if t or a later round was
// already claimed (by the proxy, the watchdog or a fail-all).
int dsv41_hf_claim(uint32_t* claimed, uint32_t t) {
  uint32_t old = __atomic_load_n(claimed, __ATOMIC_ACQUIRE);
  while (after(t, old)) {
    if (__atomic_compare_exchange_n(claimed, &old, t, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) return 1;
  }
  return 0;
}

// Publish round t's reply: claim t, copy n bytes from src into the reply area, release t. 1 if published, 0 if the
// round had already been claimed (the watchdog failed it first): nothing is written then.
int dsv41_hf_publish(uint32_t* claimed, uint32_t* released, uint32_t t, void* reply, const void* src, size_t n) {
  if (!dsv41_hf_claim(claimed, t)) return 0;
  memcpy(reply, src, n);
  dsv41_hf_release_max(released, t);
  return 1;
}

// Fail round t: claim t, write the failed reply (zeros, word 0 = status) and release t. 1 if done, 0 if already claimed.
// With t far ahead of every enqueued round (cyclically), it fails every pending and later round at once.
int dsv41_hf_fail_release(uint32_t* claimed, uint32_t* released, uint32_t t, void* reply, size_t n, int32_t status) {
  if (!dsv41_hf_claim(claimed, t)) return 0;
  memset(reply, 0, n);
  if (n >= sizeof(int32_t)) memcpy(reply, &status, sizeof(int32_t));
  dsv41_hf_release_max(released, t);
  return 1;
}

// Wait until (int32)(*addr - value) >= 0. Acquire loads, so the caller's later reads (the gathered rows) are ordered
// after the word. Spins for spin_ns, then sleeps nap_ns between loads (nap_ns <= 0: spin until the deadline). Returns 0
// when reached, 1 when timeout_ns passed first.
int dsv41_hf_wait_geq(const uint32_t* addr, uint32_t value, int64_t timeout_ns, int64_t spin_ns, int64_t nap_ns) {
  const int64_t t0 = now_ns();
  int64_t now = t0;
  for (uint32_t i = 1;; i++) {
    if ((int32_t)(__atomic_load_n(addr, __ATOMIC_ACQUIRE) - value) >= 0) return 0;
    const int napping = nap_ns > 0 && now - t0 >= spin_ns;
    if (napping || (i & 63u) == 0) {
      now = now_ns();
      if (now - t0 >= timeout_ns) return 1;
    }
    if (napping) {
      nap(nap_ns);
    } else {
      cpu_relax();
    }
  }
}

// ------------------------------------------------------------------------------------------------- watchdog

typedef struct dsv41_hf_wd {
  uint32_t* gathered;
  uint32_t* released;
  uint32_t* claimed;
  void* reply;
  size_t reply_bytes;
  int32_t fail_status;
  int64_t deadline_ns;
  int64_t poll_ns;
  int stop;           // atomic
  uint64_t fires;     // atomic: rounds the watchdog failed
  uint32_t last;      // atomic: the last round it failed
  int64_t max_late_ns;  // atomic: longest gathered-unreleased time it saw (diagnostic)
  pthread_t th;
} dsv41_hf_wd;

static void* wd_main(void* arg) {
  dsv41_hf_wd* w = (dsv41_hf_wd*)arg;
  uint32_t pend = 0;
  int have = 0;
  int64_t since = 0;
  while (!__atomic_load_n(&w->stop, __ATOMIC_ACQUIRE)) {
    const uint32_t g = __atomic_load_n(w->gathered, __ATOMIC_ACQUIRE);
    const uint32_t r = __atomic_load_n(w->released, __ATOMIC_ACQUIRE);
    const int64_t now = now_ns();
    if (after(g, r)) {  // the stream gathered round g and is (or soon will be) parked on it
      if (!have || pend != g) {
        pend = g;
        since = now;
        have = 1;
      } else {
        const int64_t late = now - since;
        if (late > __atomic_load_n(&w->max_late_ns, __ATOMIC_RELAXED))
          __atomic_store_n(&w->max_late_ns, late, __ATOMIC_RELAXED);
        if (late >= w->deadline_ns) {
          if (dsv41_hf_fail_release(w->claimed, w->released, g, w->reply, w->reply_bytes, w->fail_status)) {
            __atomic_add_fetch(&w->fires, 1, __ATOMIC_RELAXED);
            __atomic_store_n(&w->last, g, __ATOMIC_RELAXED);
          }
          have = 0;
        }
      }
    } else {
      have = 0;
    }
    nap(w->poll_ns);
  }
  return NULL;
}

// Start the watchdog over the three words and the reply area. NULL on failure (errno set).
dsv41_hf_wd* dsv41_hf_watchdog_start(uint32_t* gathered, uint32_t* released, uint32_t* claimed, void* reply,
                                     size_t reply_bytes, int32_t fail_status, int64_t deadline_ns, int64_t poll_ns) {
  if (deadline_ns <= 0 || poll_ns <= 0) {
    errno = EINVAL;
    return NULL;
  }
  dsv41_hf_wd* w = (dsv41_hf_wd*)calloc(1, sizeof(*w));
  if (!w) return NULL;
  w->gathered = gathered;
  w->released = released;
  w->claimed = claimed;
  w->reply = reply;
  w->reply_bytes = reply_bytes;
  w->fail_status = fail_status;
  w->deadline_ns = deadline_ns;
  w->poll_ns = poll_ns;
  int e = pthread_create(&w->th, NULL, wd_main, w);
  if (e != 0) {
    free(w);
    errno = e;
    return NULL;
  }
  return w;
}

// fires, last failed round, longest gathered-unreleased interval seen (ns)
void dsv41_hf_watchdog_stats(const dsv41_hf_wd* w, uint64_t out[3]) {
  out[0] = __atomic_load_n(&w->fires, __ATOMIC_RELAXED);
  out[1] = __atomic_load_n(&w->last, __ATOMIC_RELAXED);
  out[2] = (uint64_t)__atomic_load_n(&w->max_late_ns, __ATOMIC_RELAXED);
}

// Stop and join the watchdog, then free it. 0 or the pthread_join error.
int dsv41_hf_watchdog_stop(dsv41_hf_wd* w) {
  if (!w) return 0;
  __atomic_store_n(&w->stop, 1, __ATOMIC_RELEASE);
  int e = pthread_join(w->th, NULL);
  free(w);
  return e;
}
