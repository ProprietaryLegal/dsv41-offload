// engram_server: the x86host Engram table server (lane O1, DSV41-OFFLOAD).
//
// Loads the Engram tables of the checkpoint's shards (model-00047/48-of-00048.safetensors: layer 1 and layer 14) into
// ONE anonymous resident copy each: the whole shard file is read with O_DIRECT (no page cache) straight into an
// anonymous mapping advised for transparent huge pages, while a second thread hashes it in order; the shard must match
// its sha256 in the repo's manifest.json or the server refuses to start. Nothing is pinned or registered except the
// transport's bounded per-connection pools. The few hundred MB of non-Engram tensors at the end of each shard are hashed
// and then released.
//
// Serves "engram-v1" (engram_proto.h) over ofl (TCP + RDMA) with one listener per direct link (10.202.28.1 for Spark A /
// rank 0, 10.202.29.1 for Spark B / rank 1). LOOKUP gathers rows in request order into the registered response buffer
// (weights block, then scales block = the engine's staging layout), in parallel chunks that go out on the wire as they
// finish (ofl_req_ready). Row id kZeroRow reads as zeros, exactly like the engine's disk reader for its -1 padding.
//
//   engram_server --data /opt/dsv41/x86host-data/engram --manifest ~/dsv41-offload/o1/manifest.json
//     --listen 10.202.28.1 --listen 10.202.29.1 [--port 18610] [--backends tcp,rdma] [--threads 8]
//     [--budget-gib 215] [--wait-marker ENGRAM_COPY_DONE] [--max-rows 98304] [--stats FILE] [--no-verify]
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <dirent.h>
#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../../transport/ofl.h"
#include "../engram_proto.h"

using namespace engram;

namespace {

volatile sig_atomic_t g_stop = 0;
void on_sig(int) { g_stop = 1; }

int64_t now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

void logf(const char* f, ...) __attribute__((format(printf, 1, 2)));
void logf(const char* f, ...) {
  char tb[32];
  time_t t = time(nullptr);
  strftime(tb, sizeof(tb), "%Y-%m-%dT%H:%M:%SZ", gmtime(&t));
  va_list ap;
  va_start(ap, f);
  fprintf(stderr, "%s engram_server: ", tb);
  vfprintf(stderr, f, ap);
  fprintf(stderr, "\n");
  va_end(ap);
}

[[noreturn]] void die(const char* f, ...) __attribute__((format(printf, 1, 2)));
void die(const char* f, ...) {
  va_list ap;
  va_start(ap, f);
  fprintf(stderr, "engram_server: FATAL: ");
  vfprintf(stderr, f, ap);
  fprintf(stderr, "\n");
  va_end(ap);
  exit(2);
}

uint64_t meminfo_kb(const char* key) {
  FILE* f = fopen("/proc/meminfo", "r");
  if (!f) return 0;
  char line[256];
  uint64_t v = 0;
  const size_t kl = strlen(key);
  while (fgets(line, sizeof(line), f))
    if (!strncmp(line, key, kl) && line[kl] == ':') { v = strtoull(line + kl + 1, nullptr, 10); break; }
  fclose(f);
  return v;
}

uint64_t self_kb(const char* key) {  // /proc/self/status (VmRSS, VmHWM, ...)
  FILE* f = fopen("/proc/self/status", "r");
  if (!f) return 0;
  char line[256];
  uint64_t v = 0;
  const size_t kl = strlen(key);
  while (fgets(line, sizeof(line), f))
    if (!strncmp(line, key, kl) && line[kl] == ':') { v = strtoull(line + kl + 1, nullptr, 10); break; }
  fclose(f);
  return v;
}

// ---- whole-shard hashing: OpenSSL libcrypto when present (SHA-NI), else the portable implementation -------------
struct Hasher {
  using NewFn = void* (*)();
  using FreeFn = void (*)(void*);
  using MdFn = const void* (*)();
  using InitFn = int (*)(void*, const void*, void*);
  using UpdFn = int (*)(void*, const void*, size_t);
  using FinFn = int (*)(void*, unsigned char*, unsigned int*);
  NewFn mk = nullptr;
  FreeFn fr = nullptr;
  MdFn md = nullptr;
  InitFn init = nullptr;
  UpdFn upd = nullptr;
  FinFn fin = nullptr;
  bool ok = false;
  Hasher() {
    void* h = dlopen("libcrypto.so.3", RTLD_NOW | RTLD_LOCAL);
    if (!h) return;
    mk = (NewFn)dlsym(h, "EVP_MD_CTX_new");
    fr = (FreeFn)dlsym(h, "EVP_MD_CTX_free");
    md = (MdFn)dlsym(h, "EVP_sha256");
    init = (InitFn)dlsym(h, "EVP_DigestInit_ex");
    upd = (UpdFn)dlsym(h, "EVP_DigestUpdate");
    fin = (FinFn)dlsym(h, "EVP_DigestFinal_ex");
    ok = mk && fr && md && init && upd && fin;
    if (ok) {  // self-test against the portable implementation
      uint8_t a[32], b[32];
      const char* t = "engram_server sha256 self-test";
      Sha256 s;
      s.update(t, strlen(t));
      s.final(a);
      void* c = mk();
      unsigned n = 0;
      ok = c && init(c, md(), nullptr) == 1 && upd(c, t, strlen(t)) == 1 && fin(c, b, &n) == 1 && n == 32 && !memcmp(a, b, 32);
      if (c) fr(c);
    }
  }
};

struct Stream {  // one running whole-file hash
  const Hasher& H;
  void* ctx = nullptr;
  Sha256 portable;
  explicit Stream(const Hasher& h) : H(h) {
    if (H.ok) { ctx = H.mk(); H.init(ctx, H.md(), nullptr); }
  }
  ~Stream() { if (ctx) H.fr(ctx); }
  void update(const void* p, size_t n) {
    if (ctx) H.upd(ctx, p, n);
    else portable.update(p, n);
  }
  void final(uint8_t out[32]) {
    if (ctx) { unsigned n = 0; H.fin(ctx, out, &n); }
    else portable.final(out);
  }
};

// ---- a loaded table ---------------------------------------------------------------------------------------------
struct Table {
  TableDesc d;
  uint8_t identity[32];
  std::string path;
  uint8_t* base = nullptr;  // anonymous mapping holding file bytes [0, file size)
  size_t map_bytes = 0;
  const uint8_t* w = nullptr;  // row 0 weights
  const uint8_t* s = nullptr;  // row 0 scales
  double load_s = 0, hash_s = 0;
  std::string sha_actual;
};

constexpr size_t kChunk = 64ull << 20;  // O_DIRECT read unit
constexpr size_t kHuge = 2ull << 20;

void load_table(Table& t, const Hasher& H, bool verify, int readers) {
  const int64_t t0 = now_ns();
  const uint64_t size = t.d.shard_size;
  t.map_bytes = ((size + kHuge - 1) / kHuge) * kHuge + kHuge;
  void* m = mmap(nullptr, t.map_bytes + kHuge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (m == MAP_FAILED) die("mmap %zu bytes for %s: %s", t.map_bytes, t.d.shard.c_str(), strerror(errno));
  uint8_t* aligned = reinterpret_cast<uint8_t*>(((uintptr_t)m + kHuge - 1) & ~(uintptr_t)(kHuge - 1));
  t.base = aligned;
  if (madvise(t.base, t.map_bytes, MADV_HUGEPAGE) != 0) logf("madvise(MADV_HUGEPAGE) failed: %s (small pages)", strerror(errno));
  const int fd = ::open(t.path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
  if (fd < 0) die("open %s O_DIRECT: %s", t.path.c_str(), strerror(errno));
  const size_t nchunks = (size + kChunk - 1) / kChunk;
  std::unique_ptr<std::atomic<uint8_t>[]> ready(new std::atomic<uint8_t>[nchunks]);
  for (size_t i = 0; i < nchunks; i++) ready[i].store(0);
  std::atomic<size_t> next{0};
  std::atomic<int> err{0};
  std::vector<std::thread> th;
  for (int r = 0; r < readers; r++)
    th.emplace_back([&] {
      for (size_t c; (c = next.fetch_add(1)) < nchunks && !err.load() && !g_stop;) {
        const uint64_t off = (uint64_t)c * kChunk;
        size_t want = (size_t)std::min<uint64_t>(kChunk, size - off);
        want = (want + 4095) & ~size_t(4095);  // O_DIRECT length; the mapping has room past EOF
        size_t got = 0;
        while (got < want) {
          const ssize_t k = pread(fd, t.base + off + got, want - got, (off_t)(off + got));
          if (k < 0 && errno == EINTR) continue;
          if (k < 0) { logf("pread %s at %llu: %s", t.path.c_str(), (unsigned long long)(off + got), strerror(errno)); err = 1; break; }
          if (k == 0) break;  // EOF (last chunk)
          got += (size_t)k;
        }
        if (!err && off + got < std::min<uint64_t>(off + kChunk, size)) {
          logf("short read of %s at chunk %zu", t.path.c_str(), c);
          err = 1;
        }
        ready[c].store(1, std::memory_order_release);
      }
    });
  uint8_t digest[32] = {0};
  int64_t hash_ns = 0;
  if (verify) {
    Stream st(H);
    for (size_t c = 0; c < nchunks && !err.load() && !g_stop; c++) {
      while (!ready[c].load(std::memory_order_acquire) && !err.load() && !g_stop) usleep(200);
      if (err.load() || g_stop) break;
      const int64_t h0 = now_ns();
      const uint64_t off = (uint64_t)c * kChunk;
      st.update(t.base + off, (size_t)std::min<uint64_t>(kChunk, size - off));
      hash_ns += now_ns() - h0;
    }
    st.final(digest);
  }
  for (auto& x : th) x.join();
  ::close(fd);
  if (g_stop) die("stopped while loading %s", t.d.shard.c_str());
  if (err.load()) die("reading %s failed", t.path.c_str());
  t.hash_s = hash_ns / 1e9;
  if (verify) {
    t.sha_actual = hex(digest, 32);
    if (t.sha_actual != t.d.shard_sha256)
      die("%s: sha256 %s, the manifest says %s: refusing to serve it", t.path.c_str(), t.sha_actual.c_str(),
          t.d.shard_sha256.c_str());
  }
  // keep the Engram tensors, release the rest of the shard (header pages stay: they share a page with the weights)
  const uint64_t w_end = t.d.weight.abs_offset + t.d.weight.nbytes, s_end = t.d.scale.abs_offset + t.d.scale.nbytes;
  if (w_end > size || s_end > size) die("%s: Engram tensors run past the end of the file", t.path.c_str());
  const uint64_t keep_end = std::max(w_end, s_end);
  const uintptr_t rel = ((uintptr_t)(t.base + keep_end) + 4095) & ~(uintptr_t)4095;
  const uintptr_t end = (uintptr_t)(t.base + t.map_bytes);
  if (end > rel) madvise(reinterpret_cast<void*>(rel), end - rel, MADV_DONTNEED);
  t.w = t.base + t.d.weight.abs_offset;
  t.s = t.base + t.d.scale.abs_offset;
  t.load_s = (now_ns() - t0) / 1e9;
  // the copy we read is resident now; drop the shard's (clean) page cache so the bytes are not held twice
  const int pfd = ::open(t.path.c_str(), O_RDONLY | O_CLOEXEC);
  if (pfd >= 0) { posix_fadvise(pfd, 0, 0, POSIX_FADV_DONTNEED); ::close(pfd); }
}

// ---- gather pool ------------------------------------------------------------------------------------------------
struct Job {
  const uint8_t* w = nullptr;  // range row 0
  const uint8_t* s = nullptr;
  const uint32_t* ids = nullptr;
  uint32_t n = 0;
  uint8_t* out = nullptr;      // weights [n*256] then scales [n*8]
  ofl_req* req = nullptr;
  uint32_t chunk = 2048;
  uint32_t nchunks = 0;
  std::atomic<uint32_t> next{0}, done{0};
  std::atomic<uint32_t> active{0};  // pool workers holding a pointer to this (stack) job; run() waits for 0
  std::atomic<uint64_t> zero_rows{0};
};

inline void gather_rows(Job& j, uint32_t i0, uint32_t i1) {
  uint8_t* ow = j.out;
  uint8_t* os = j.out + (size_t)j.n * kRowBytes;
  constexpr uint32_t PF = 12;  // rows prefetched ahead
  uint64_t zeros = 0;
  for (uint32_t i = i0; i < std::min(i1, i0 + PF); i++) {
    const uint32_t id = j.ids[i];
    if (id != kZeroRow) {
      const uint8_t* p = j.w + (size_t)id * kRowBytes;
      __builtin_prefetch(p);
      __builtin_prefetch(p + 64);
      __builtin_prefetch(p + 128);
      __builtin_prefetch(p + 192);
      __builtin_prefetch(j.s + (size_t)id * kScaleBytes);
    }
  }
  for (uint32_t i = i0; i < i1; i++) {
    if (i + PF < i1) {
      const uint32_t nid = j.ids[i + PF];
      if (nid != kZeroRow) {
        const uint8_t* p = j.w + (size_t)nid * kRowBytes;
        __builtin_prefetch(p);
        __builtin_prefetch(p + 64);
        __builtin_prefetch(p + 128);
        __builtin_prefetch(p + 192);
        __builtin_prefetch(j.s + (size_t)nid * kScaleBytes);
      }
    }
    const uint32_t id = j.ids[i];
    if (id == kZeroRow) {
      memset(ow + (size_t)i * kRowBytes, 0, kRowBytes);
      memset(os + (size_t)i * kScaleBytes, 0, kScaleBytes);
      zeros++;
    } else {
      memcpy(ow + (size_t)i * kRowBytes, j.w + (size_t)id * kRowBytes, kRowBytes);
      memcpy(os + (size_t)i * kScaleBytes, j.s + (size_t)id * kScaleBytes, kScaleBytes);
    }
  }
  if (zeros) j.zero_rows.fetch_add(zeros, std::memory_order_relaxed);
}

class Pool {
 public:
  explicit Pool(int n, int64_t spin_ns) : spin_ns_(spin_ns) {
    for (int i = 0; i < n; i++) th_.emplace_back([this] { loop(); });
  }
  ~Pool() {
    {
      std::lock_guard<std::mutex> g(mu_);
      stop_.store(true);
    }
    cv_.notify_all();
    for (auto& t : th_) t.join();
  }
  // Run j with the pool's help; the caller works too. Returns when every chunk is gathered and marked ready and no
  // worker holds a pointer to j any more (j lives on the caller's stack).
  void run(Job& j) {
    {
      std::lock_guard<std::mutex> g(mu_);
      jobs_.push_back(&j);
      gen_.fetch_add(1, std::memory_order_release);
    }
    cv_.notify_all();
    work(j);
    while (j.done.load(std::memory_order_acquire) < j.nchunks) cpu_relax();
    {
      std::lock_guard<std::mutex> g(mu_);  // after this no worker can pick j ...
      jobs_.erase(std::remove(jobs_.begin(), jobs_.end(), &j), jobs_.end());
    }
    while (j.active.load(std::memory_order_acquire) != 0) cpu_relax();  // ... and every one that did has let go
  }
  static void work(Job& j) {
    for (uint32_t c; (c = j.next.fetch_add(1, std::memory_order_relaxed)) < j.nchunks;) {
      const uint32_t i0 = c * j.chunk, i1 = std::min(j.n, i0 + j.chunk);
      gather_rows(j, i0, i1);
      if (j.req) {
        ofl_req_ready(j.req, i0 * kRowBytes, (i1 - i0) * kRowBytes);
        ofl_req_ready(j.req, j.n * kRowBytes + i0 * kScaleBytes, (i1 - i0) * kScaleBytes);
      }
      j.done.fetch_add(1, std::memory_order_release);
    }
  }

 private:
  static void cpu_relax() {
#if defined(__x86_64__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("isb" ::: "memory");
#endif
  }
  // A job with unclaimed chunks, pinned (active + 1, under mu_, so run() cannot have erased it yet); release() after.
  Job* pick() {
    std::lock_guard<std::mutex> g(mu_);
    for (Job* j : jobs_)
      if (j->next.load(std::memory_order_relaxed) < j->nchunks) {
        j->active.fetch_add(1, std::memory_order_acq_rel);
        return j;
      }
    return nullptr;
  }
  static void release(Job* j) { j->active.fetch_sub(1, std::memory_order_release); }  // the worker's last touch of j
  // Workers spin (lock-free, on gen_) for spin_ns_ after the last job they saw, then sleep on the condition variable
  // until a job is pushed. Only a new job restarts the spin window (a timed-out wait does not).
  void loop() {
    int64_t last = now_ns();
    uint64_t seen = gen_.load(std::memory_order_acquire);
    for (;;) {
      if (stop_.load(std::memory_order_relaxed)) return;
      const uint64_t g = gen_.load(std::memory_order_acquire);
      if (g != seen) {  // jobs were pushed: help with every one that still has chunks
        seen = g;
        while (Job* j = pick()) {
          work(*j);
          release(j);
        }
        last = now_ns();
        continue;
      }
      if (now_ns() - last < spin_ns_) {
        for (int k = 0; k < 64; k++) cpu_relax();
        continue;
      }
      std::unique_lock<std::mutex> l(mu_);
      cv_.wait_for(l, std::chrono::milliseconds(100),
                   [&] { return stop_.load() || gen_.load(std::memory_order_acquire) != seen; });
    }
  }
  std::vector<std::thread> th_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::vector<Job*> jobs_;
  std::atomic<uint64_t> gen_{0};
  std::atomic<bool> stop_{false};
  int64_t spin_ns_;
};

// ---- service state ----------------------------------------------------------------------------------------------
struct Handle {
  int table;
  uint64_t row_start, num_rows;
};

struct Hist {  // log2 histogram of handler time, ns
  std::atomic<uint64_t> b[32];
  Hist() { for (auto& x : b) x.store(0); }
  void add(int64_t ns) {
    int k = 0;
    while (k < 31 && (1LL << (k + 1)) <= ns) k++;
    b[k].fetch_add(1, std::memory_order_relaxed);
  }
};

struct Service {
  std::vector<std::unique_ptr<Table>> tables;
  std::mutex hmu;
  std::vector<Handle> handles;
  std::unique_ptr<Pool> pool;
  uint32_t inline_rows = 1024;
  uint32_t chunk_rows = 2048;
  std::atomic<bool> ready{false};
  std::atomic<uint64_t> opens{0}, lookups{0}, rows{0}, zero_rows{0}, bytes{0}, refused{0};
  std::atomic<uint64_t> lookup_ns{0};
  Hist hist;
  std::string started;
};

Service g_svc;

int32_t handle_open(ofl_req* r) {
  uint32_t n = 0;
  const void* p = ofl_req_payload(r, &n);
  if (n != sizeof(OpenReq)) return kErrBadRequest;
  OpenReq q;
  memcpy(&q, p, sizeof(q));
  q.client[sizeof(q.client) - 1] = 0;
  if (q.version != kProtoVersion) return kErrBadRequest;
  int ti = -1;
  for (size_t i = 0; i < g_svc.tables.size(); i++)
    if (!memcmp(g_svc.tables[i]->identity, q.identity, 32)) ti = (int)i;
  if (ti < 0) {
    g_svc.refused++;
    logf("OPEN refused for '%s' (conn %llu): identity %s matches no loaded table", q.client,
         (unsigned long long)ofl_req_conn_id(r), hex(q.identity, 32).c_str());
    return kErrIdentity;
  }
  const Table& t = *g_svc.tables[ti];
  if ((int)q.layer_id != t.d.layer_id || q.row_bytes != kRowBytes || q.scale_bytes != kScaleBytes ||
      q.num_rows == 0 || q.row_start > t.d.rows() || q.num_rows > t.d.rows() - q.row_start || q.num_rows >= kZeroRow) {
    g_svc.refused++;
    logf("OPEN refused for '%s': layer %u rows [%llu, +%llu) row %u/%u B vs table layer %d of %llu rows", q.client,
         q.layer_id, (unsigned long long)q.row_start, (unsigned long long)q.num_rows, q.row_bytes, q.scale_bytes,
         t.d.layer_id, (unsigned long long)t.d.rows());
    return kErrRange;
  }
  uint32_t h;
  {
    std::lock_guard<std::mutex> g(g_svc.hmu);
    h = (uint32_t)g_svc.handles.size();
    for (uint32_t i = 0; i < g_svc.handles.size(); i++)
      if (g_svc.handles[i].table == ti && g_svc.handles[i].row_start == q.row_start && g_svc.handles[i].num_rows == q.num_rows) h = i;
    if (h == g_svc.handles.size()) g_svc.handles.push_back(Handle{ti, q.row_start, q.num_rows});
  }
  uint32_t cap = 0;
  void* out = ofl_req_resp_buf(r, &cap);
  if (cap < sizeof(OpenResp)) return kErrTooMany;
  OpenResp a{};
  a.version = kProtoVersion;
  a.handle = h;
  a.table_rows = t.d.rows();
  memcpy(a.identity, t.identity, 32);
  snprintf(a.desc, sizeof(a.desc), "%s layer %d rows [%llu, %llu) of %llu", t.d.shard.c_str(), t.d.layer_id,
           (unsigned long long)q.row_start, (unsigned long long)(q.row_start + q.num_rows), (unsigned long long)t.d.rows());
  memcpy(out, &a, sizeof(a));
  ofl_req_set_len(r, sizeof(a));
  g_svc.opens++;
  logf("OPEN '%s' (conn %llu, %s): handle %u = %s", q.client, (unsigned long long)ofl_req_conn_id(r),
       ofl_req_backend(r) == OFL_BACKEND_RDMA ? "rdma" : "tcp", h, a.desc);
  return 0;
}

int32_t handle_lookup(ofl_req* r) {
  const int64_t t0 = now_ns();
  uint32_t n = 0;
  const uint8_t* p = static_cast<const uint8_t*>(ofl_req_payload(r, &n));
  if (n < sizeof(LookupHdr)) return kErrBadRequest;
  LookupHdr q;
  memcpy(&q, p, sizeof(q));
  if ((uint64_t)n != sizeof(LookupHdr) + 4ull * q.n) return kErrBadRequest;
  Handle h;
  {
    std::lock_guard<std::mutex> g(g_svc.hmu);
    if (q.handle >= g_svc.handles.size()) return kErrHandle;
    h = g_svc.handles[q.handle];
  }
  const Table& t = *g_svc.tables[h.table];
  if (memcmp(t.identity, q.identity, 32)) { g_svc.refused++; return kErrIdentity; }
  uint32_t cap = 0;
  uint8_t* out = static_cast<uint8_t*>(ofl_req_resp_buf(r, &cap));
  const uint64_t need = (uint64_t)q.n * (kRowBytes + kScaleBytes);
  if (need > cap) return kErrTooMany;
  const uint32_t* ids = reinterpret_cast<const uint32_t*>(p + sizeof(LookupHdr));  // 4-byte aligned: 64+16k+40
  for (uint32_t i = 0; i < q.n; i++)
    if (ids[i] != kZeroRow && ids[i] >= h.num_rows) { g_svc.refused++; return kErrRowId; }
  Job j;
  j.w = t.w + h.row_start * kRowBytes;
  j.s = t.s + h.row_start * kScaleBytes;
  j.ids = ids;
  j.n = q.n;
  j.out = out;
  if (q.n <= g_svc.inline_rows || !g_svc.pool) {
    gather_rows(j, 0, q.n);  // decode-size: one pass in this thread; the transport sends it after we return
  } else {
    j.req = r;
    j.chunk = g_svc.chunk_rows;
    j.nchunks = (q.n + j.chunk - 1) / j.chunk;
    g_svc.pool->run(j);
  }
  ofl_req_set_len(r, (uint32_t)need);
  const int64_t dt = now_ns() - t0;
  g_svc.lookups++;
  g_svc.rows += q.n;
  g_svc.zero_rows += j.zero_rows.load();
  g_svc.bytes += need;
  g_svc.lookup_ns += (uint64_t)dt;
  g_svc.hist.add(dt);
  return 0;
}

std::string stats_json() {
  std::string s = "{\"service\":\"engram-v1\",\"started\":\"" + g_svc.started + "\",\"ready\":" +
                  (g_svc.ready.load() ? "true" : "false") + ",\"tables\":[";
  for (size_t i = 0; i < g_svc.tables.size(); i++) {
    const Table& t = *g_svc.tables[i];
    char b[640];
    snprintf(b, sizeof(b),
             "%s{\"shard\":\"%s\",\"layer\":%d,\"rows\":%llu,\"identity\":\"%s\",\"sha256\":\"%s\",\"verified\":%s,"
             "\"load_s\":%.1f,\"hash_s\":%.1f}",
             i ? "," : "", t.d.shard.c_str(), t.d.layer_id, (unsigned long long)t.d.rows(), hex(t.identity, 32).c_str(),
             t.d.shard_sha256.c_str(), t.sha_actual.empty() ? "false" : "true", t.load_s, t.hash_s);
    s += b;
  }
  char b[1024];
  snprintf(b, sizeof(b),
           "],\"opens\":%llu,\"lookups\":%llu,\"rows\":%llu,\"zero_rows\":%llu,\"bytes\":%llu,\"refused\":%llu,"
           "\"lookup_us_mean\":%.1f,\"rss_gib\":%.2f,\"hwm_gib\":%.2f,\"anon_huge_gib\":%.2f,\"mem_available_gib\":%.1f,"
           "\"hist_log2_ns\":[",
           (unsigned long long)g_svc.opens.load(), (unsigned long long)g_svc.lookups.load(),
           (unsigned long long)g_svc.rows.load(), (unsigned long long)g_svc.zero_rows.load(),
           (unsigned long long)g_svc.bytes.load(), (unsigned long long)g_svc.refused.load(),
           g_svc.lookups.load() ? g_svc.lookup_ns.load() / 1e3 / g_svc.lookups.load() : 0.0,
           self_kb("VmRSS") / 1048576.0, self_kb("VmHWM") / 1048576.0, meminfo_kb("AnonHugePages") / 1048576.0,
           meminfo_kb("MemAvailable") / 1048576.0);
  s += b;
  for (int k = 0; k < 32; k++) s += (k ? "," : "") + std::to_string(g_svc.hist.b[k].load());
  return s + "]}";
}

int32_t handler(void*, ofl_req* r) {
  switch (ofl_req_op(r)) {
    case kOpOpen:
      return g_svc.ready.load() ? handle_open(r) : kErrNotReady;
    case kOpLookup:
      return g_svc.ready.load() ? handle_lookup(r) : kErrNotReady;
    case kOpStats: {
      const std::string s = stats_json();
      uint32_t cap = 0;
      void* out = ofl_req_resp_buf(r, &cap);
      const uint32_t k = (uint32_t)std::min<size_t>(s.size(), cap);
      memcpy(out, s.data(), k);
      ofl_req_set_len(r, k);
      return 0;
    }
    default:
      return kErrBadRequest;
  }
}

std::vector<std::string> args_all(int argc, char** argv, const char* name) {
  std::vector<std::string> v;
  for (int i = 1; i + 1 < argc; i++)
    if (!strcmp(argv[i], name)) v.push_back(argv[i + 1]);
  return v;
}
const char* arg(int argc, char** argv, const char* name, const char* def) {
  for (int i = 1; i + 1 < argc; i++)
    if (!strcmp(argv[i], name)) return argv[i + 1];
  return def;
}
bool flag(int argc, char** argv, const char* name) {
  for (int i = 1; i < argc; i++)
    if (!strcmp(argv[i], name)) return true;
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string data = arg(argc, argv, "--data", "/opt/dsv41/x86host-data/engram");
  const std::string manifest = arg(argc, argv, "--manifest", "");
  const std::vector<std::string> listen = args_all(argc, argv, "--listen");
  const uint16_t port = (uint16_t)atoi(arg(argc, argv, "--port", "18610"));
  const std::string be = arg(argc, argv, "--backends", "tcp,rdma");
  const int threads = atoi(arg(argc, argv, "--threads", "8"));
  const double budget_gib = atof(arg(argc, argv, "--budget-gib", "215"));
  const double reserve_gib = atof(arg(argc, argv, "--reserve-gib", "48"));  // MemAvailable that must remain
  const std::string marker = arg(argc, argv, "--wait-marker", "");
  const uint32_t max_rows = (uint32_t)atol(arg(argc, argv, "--max-rows", "98304"));
  const std::string stats_path = arg(argc, argv, "--stats", "");
  const bool verify = !flag(argc, argv, "--no-verify");
  const std::vector<std::string> only = args_all(argc, argv, "--shard");  // default: every *.safetensors in --data
  const int readers = atoi(arg(argc, argv, "--readers", "4"));
  g_svc.inline_rows = (uint32_t)atol(arg(argc, argv, "--inline-rows", "1024"));
  g_svc.chunk_rows = (uint32_t)atol(arg(argc, argv, "--chunk-rows", "2048"));
  if (manifest.empty() || listen.empty()) {
    fprintf(stderr, "usage: engram_server --manifest manifest.json --listen IP [--listen IP] [--data DIR] [--port N] "
                    "[--backends tcp,rdma] [--threads N] [--budget-gib G] [--wait-marker NAME] [--stats FILE]\n");
    return 64;
  }
  signal(SIGINT, on_sig);
  signal(SIGTERM, on_sig);
  signal(SIGPIPE, SIG_IGN);
  {
    char tb[32];
    time_t t = time(nullptr);
    strftime(tb, sizeof(tb), "%Y-%m-%dT%H:%M:%SZ", gmtime(&t));
    g_svc.started = tb;
  }
  if (!marker.empty()) {
    const std::string mp = data + "/" + marker;
    struct stat st {};
    if (stat(mp.c_str(), &st) != 0) logf("waiting for %s (the copy of the shards)", mp.c_str());
    while (stat(mp.c_str(), &st) != 0 && !g_stop) sleep(10);
    if (g_stop) return 1;
  }
  std::vector<std::string> files = only;
  if (files.empty()) {
    DIR* d = opendir(data.c_str());
    if (!d) die("cannot open %s: %s", data.c_str(), strerror(errno));
    while (dirent* e = readdir(d)) {
      const std::string n = e->d_name;
      if (n.size() > 12 && n.compare(n.size() - 12, 12, ".safetensors") == 0) files.push_back(n);
    }
    closedir(d);
    std::sort(files.begin(), files.end());
  }
  if (files.empty()) die("no .safetensors shards in %s", data.c_str());
  uint64_t need = 0;
  for (const auto& f : files) {
    auto t = std::make_unique<Table>();
    t->path = f.find('/') == std::string::npos ? data + "/" + f : f;
    std::string msg;
    if (!describe_table(t->path, -1, -1, manifest, t->d, msg)) die("%s", msg.c_str());
    identity_digest(t->d, t->identity);
    need += t->d.shard_size;
    logf("table %s: layer %d, %llu rows, weights @%llu, scales @%llu, identity %s", t->d.shard.c_str(), t->d.layer_id,
         (unsigned long long)t->d.rows(), (unsigned long long)t->d.weight.abs_offset,
         (unsigned long long)t->d.scale.abs_offset, hex(t->identity, 32).c_str());
    g_svc.tables.push_back(std::move(t));
  }
  // memory budget: the resident copies + the transport pools (per connection: 2 x max response + receive slots)
  const uint64_t pools = (uint64_t)listen.size() * 8 * 2 * ((uint64_t)max_rows * (kRowBytes + kScaleBytes) + (2u << 20));
  const double need_gib = (need + pools) / 1073741824.0;
  const double avail_gib = meminfo_kb("MemAvailable") / 1048576.0;
  logf("memory: tables %.1f GiB + pools <= %.1f GiB = %.1f GiB; budget %.0f GiB; MemAvailable %.1f GiB (must keep %.0f)",
       need / 1073741824.0, pools / 1073741824.0, need_gib, budget_gib, avail_gib, reserve_gib);
  if (need_gib > budget_gib) die("%.1f GiB exceeds the budget of %.0f GiB", need_gib, budget_gib);
  if (avail_gib - need_gib < reserve_gib)
    die("loading would leave %.1f GiB available (< %.0f GiB reserve): refusing (never swap)", avail_gib - need_gib, reserve_gib);
  Hasher H;
  logf("sha256: %s; verify %s", H.ok ? "OpenSSL libcrypto" : "portable implementation", verify ? "on" : "OFF");
  {
    std::vector<std::thread> lt;
    for (auto& t : g_svc.tables) lt.emplace_back([&H, &t, verify, readers] { load_table(*t, H, verify, readers); });
    for (auto& x : lt) x.join();
  }
  for (auto& t : g_svc.tables)
    logf("loaded %s in %.1f s (hash %.1f s)%s", t->d.shard.c_str(), t->load_s, t->hash_s,
         verify ? (", sha256 " + t->sha_actual + " = manifest").c_str() : ", NOT verified");
  logf("resident: RSS %.1f GiB, AnonHugePages %.1f GiB, MemAvailable %.1f GiB", self_kb("VmRSS") / 1048576.0,
       meminfo_kb("AnonHugePages") / 1048576.0, meminfo_kb("MemAvailable") / 1048576.0);
  const int64_t pool_spin_us = atol(arg(argc, argv, "--pool-spin-us", "20000"));
  g_svc.pool.reset(threads > 0 ? new Pool(threads, pool_spin_us * 1000LL) : nullptr);
  std::vector<ofl_server*> servers;
  for (const auto& ip : listen) {
    ofl_server_config c{};
    c.bind_ip = ip.c_str();
    c.port = port;
    c.backends = (be.find("tcp") != std::string::npos ? 1 << OFL_BACKEND_TCP : 0) |
                 (be.find("rdma") != std::string::npos ? 1 << OFL_BACKEND_RDMA : 0);
    c.gid_index = -1;
    c.service = kService;
    c.max_req_bytes = (uint32_t)(sizeof(LookupHdr) + 4ull * max_rows);
    c.max_resp_bytes = (uint32_t)((uint64_t)max_rows * (kRowBytes + kScaleBytes));
    c.max_inflight = 2;
    c.max_conns = 8;
    c.spin_us = atol(arg(argc, argv, "--spin-us", "20000"));  // then sleep on the completion channel
    char err[512];
    ofl_server* s = ofl_server_create(&c, handler, nullptr, err, sizeof(err));
    if (!s) die("listener %s: %s", ip.c_str(), err);
    ofl_server_start(s);
    logf("listening: %s", ofl_server_describe(s));
    servers.push_back(s);
  }
  g_svc.ready.store(true);
  logf("ready (max %u rows a lookup, %d gather threads, inline up to %u rows, chunks of %u)", max_rows, threads,
       g_svc.inline_rows, g_svc.chunk_rows);
  int64_t last_stats = 0;
  while (!g_stop) {
    usleep(200000);
    if (!stats_path.empty() && now_ns() - last_stats > 5000000000LL) {
      last_stats = now_ns();
      const std::string tmp = stats_path + ".tmp";
      FILE* f = fopen(tmp.c_str(), "w");
      if (f) {
        fputs(stats_json().c_str(), f);
        fputc('\n', f);
        fclose(f);
        rename(tmp.c_str(), stats_path.c_str());
      }
    }
  }
  logf("stopping: %s", stats_json().c_str());
  for (auto* s : servers) ofl_server_destroy(s);
  g_svc.pool.reset();
  return 0;
}
