// engram_check: standalone exactness + latency client for the x86host Engram table server (no torch, no CUDA).
//
// Reads the same rows two ways and compares them byte for byte:
//   remote - the "engram-v1" service over ofl (TCP or RDMA), into two staging buffers like the engine's;
//   local  - the shard on this host, O_DIRECT native AIO (queue depth 256, the access pattern of the image's spin-mode
//            disk reader; no page cache), with the disk reader's rule that a negative / out-of-range id reads zeros.
//
//   engram_check --shard /srv/dsv41/legal-v2a/model-00047-of-00048.safetensors --manifest manifest.json
//     --server 10.202.28.1 --local 10.202.28.2 --backend rdma --row-start 0 --num-rows 192001740
//     [--mode exact|bench|negative|all] [--sizes 288,4096,16384,49152] [--iters 20] [--seed 1] [--label S]
//     [--no-local] [--clients N]
// Prints JSON lines (one per test / size). Exit 0 only if every comparison matched and every negative case was refused.
#include <errno.h>
#include <fcntl.h>
#include <linux/aio_abi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "engram_client.h"

using namespace engram;

namespace {

int64_t now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000000000LL + ts.tv_nsec;
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

void* aligned(size_t n) {
  void* p = nullptr;
  if (posix_memalign(&p, 4096, std::max<size_t>(n, 4096)) != 0) {
    fprintf(stderr, "out of memory\n");
    exit(2);
  }
  memset(p, 0, std::max<size_t>(n, 4096));
  return p;
}

// O_DIRECT native-AIO row reader (the spin-mode disk reader's I/O: one aligned read a weight row, one a scale row,
// up to qd in flight).
class LocalReader {
 public:
  LocalReader(const std::string& path, const TableDesc& d, uint64_t row_start, uint64_t num_rows, int qd)
      : d_(d), row_start_(row_start), num_rows_(num_rows), qd_(qd) {
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    if (fd_ < 0) { fprintf(stderr, "open %s O_DIRECT: %s\n", path.c_str(), strerror(errno)); exit(2); }
    if (syscall(SYS_io_setup, qd_, &ctx_) != 0) { fprintf(stderr, "io_setup: %s\n", strerror(errno)); exit(2); }
    bounce_ = static_cast<uint8_t*>(aligned((size_t)qd_ * kSlot));
    cbs_.resize(qd_);
    cbp_.resize(qd_);
  }
  ~LocalReader() {
    if (ctx_) syscall(SYS_io_destroy, ctx_);
    if (fd_ >= 0) ::close(fd_);
    free(bounce_);
  }
  // rows of ids (engine local ids) -> w [n*256], s [n*8]; false on an I/O error
  bool read(const int64_t* ids, int64_t n, uint8_t* w, uint8_t* s) {
    struct Piece { uint64_t off; uint32_t len; uint8_t* dst; };
    std::vector<Piece> ps;
    ps.reserve(2 * n);
    for (int64_t i = 0; i < n; i++) {
      const int64_t v = ids[i];
      if (v < 0 || v >= (int64_t)num_rows_) {
        memset(w + i * kRowBytes, 0, kRowBytes);
        memset(s + i * kScaleBytes, 0, kScaleBytes);
        continue;
      }
      const uint64_t row = row_start_ + (uint64_t)v;
      ps.push_back({d_.weight.abs_offset + row * kRowBytes, kRowBytes, w + i * kRowBytes});
      ps.push_back({d_.scale.abs_offset + row * kScaleBytes, kScaleBytes, s + i * kScaleBytes});
    }
    std::vector<int> freelist;
    for (int k = qd_ - 1; k >= 0; k--) freelist.push_back(k);
    std::vector<size_t> slot_piece(qd_);
    size_t next = 0, done = 0;
    io_event ev[64];
    while (done < ps.size()) {
      int k = 0;
      while (next < ps.size() && !freelist.empty()) {
        const Piece& p = ps[next];
        const int slot = freelist.back();
        freelist.pop_back();
        const uint64_t a = p.off & ~4095ull, e = (p.off + p.len + 4095) & ~4095ull;
        iocb& cb = cbs_[slot];
        memset(&cb, 0, sizeof(cb));
        cb.aio_lio_opcode = IOCB_CMD_PREAD;
        cb.aio_fildes = (uint32_t)fd_;
        cb.aio_buf = (uint64_t)(uintptr_t)(bounce_ + (size_t)slot * kSlot);
        cb.aio_nbytes = e - a;
        cb.aio_offset = (int64_t)a;
        cb.aio_data = (uint64_t)slot;
        slot_piece[slot] = next++;
        cbp_[k++] = &cb;
      }
      if (k) {
        const long r = syscall(SYS_io_submit, ctx_, k, cbp_.data());
        if (r != k) { fprintf(stderr, "io_submit: %ld of %d (%s)\n", r, k, strerror(errno)); return false; }
      }
      const long got = syscall(SYS_io_getevents, ctx_, 1L, 64L, ev, nullptr);
      if (got < 0) { if (errno == EINTR) continue; fprintf(stderr, "io_getevents: %s\n", strerror(errno)); return false; }
      for (long q = 0; q < got; q++) {
        const int slot = (int)ev[q].data;
        const Piece& p = ps[slot_piece[slot]];
        const uint64_t delta = p.off & 4095ull;
        if (ev[q].res < (long long)(delta + p.len)) { fprintf(stderr, "short AIO read (%lld)\n", (long long)ev[q].res); return false; }
        memcpy(p.dst, bounce_ + (size_t)slot * kSlot + delta, p.len);
        freelist.push_back(slot);
        done++;
      }
    }
    return true;
  }

 private:
  static constexpr size_t kSlot = 8192;  // a 256-byte row can straddle a 4 KiB block
  TableDesc d_;
  uint64_t row_start_, num_rows_;
  int qd_;
  int fd_ = -1;
  aio_context_t ctx_ = 0;
  uint8_t* bounce_ = nullptr;
  std::vector<iocb> cbs_;
  std::vector<iocb*> cbp_;
};

struct Ctx {
  std::string shard, manifest, label;
  TableDesc d;
  ClientConfig cc;
  uint64_t row_start = 0, num_rows = 0;
  uint32_t max_rows = 49152;
  bool local = true;
  int fails = 0;
};

// A batch with the cases that matter: uniform rows, in-batch duplicates, the range's first/last rows, the engine's
// -1 padding and ids past the range (both must read as zeros).
std::vector<int64_t> make_ids(std::mt19937_64& g, int64_t n, uint64_t num_rows, bool mixed) {
  std::vector<int64_t> v(n);
  std::uniform_int_distribution<uint64_t> U(0, num_rows - 1);
  for (int64_t i = 0; i < n; i++) {
    const unsigned c = mixed ? (unsigned)(g() % 100) : 0;
    if (c < 70 || i == 0) v[i] = (int64_t)U(g);
    else if (c < 85) v[i] = v[g() % i];                          // duplicate of an earlier id
    else if (c < 90) v[i] = (int64_t)((g() & 1) ? (g() % 2) : num_rows - 1 - (g() % 2));  // boundaries
    else if (c < 95) v[i] = -1;                                  // padding
    else v[i] = (int64_t)(num_rows + (g() % 7));                 // out of range
  }
  return v;
}

std::string pct_json(std::vector<int64_t> v) {
  if (v.empty()) return "null";
  std::sort(v.begin(), v.end());
  auto p = [&](double q) { return v[std::min(v.size() - 1, (size_t)(q * v.size()))] / 1000.0; };
  double m = 0;
  for (auto x : v) m += x;
  char b[256];
  snprintf(b, sizeof(b), "{\"n\":%zu,\"mean\":%.1f,\"p50\":%.1f,\"p90\":%.1f,\"p99\":%.1f,\"min\":%.1f,\"max\":%.1f}", v.size(),
           m / v.size() / 1000.0, p(0.5), p(0.9), p(0.99), v.front() / 1000.0, v.back() / 1000.0);
  return b;
}

const char* be_name(int b) { return b == OFL_BACKEND_RDMA ? "rdma" : "tcp"; }

// exactness: remote vs local for every size; returns mismatching rows
int run_exact(Ctx& x, const std::vector<int64_t>& sizes, int iters, uint64_t seed, int client_id) {
  EngramClient cl;
  std::string msg;
  ClientConfig cc = x.cc;
  cc.name += " client " + std::to_string(client_id);
  if (!cl.open(cc, x.d, x.row_start, x.num_rows, msg)) { fprintf(stderr, "open: %s\n", msg.c_str()); return 1 << 30; }
  uint8_t* w = static_cast<uint8_t*>(aligned((size_t)x.max_rows * kRowBytes));
  uint8_t* s = static_cast<uint8_t*>(aligned((size_t)x.max_rows * kScaleBytes));
  if (!cl.register_staging(w, (size_t)x.max_rows * kRowBytes, s, (size_t)x.max_rows * kScaleBytes, msg)) {
    fprintf(stderr, "register: %s\n", msg.c_str());
    return 1 << 30;
  }
  std::unique_ptr<LocalReader> lr(x.local ? new LocalReader(x.shard, x.d, x.row_start, x.num_rows, 256) : nullptr);
  uint8_t* lw = static_cast<uint8_t*>(aligned((size_t)x.max_rows * kRowBytes));
  uint8_t* ls = static_cast<uint8_t*>(aligned((size_t)x.max_rows * kScaleBytes));
  std::mt19937_64 g(seed * 7919 + client_id);
  int bad_total = 0;
  for (int64_t n : sizes) {
    uint64_t rows = 0, bad = 0, zeros = 0, dups = 0;
    for (int it = 0; it < iters; it++) {
      auto ids = make_ids(g, n, x.num_rows, true);
      memset(w, 0xA5, (size_t)n * kRowBytes);  // poison: a row the server skipped would show
      memset(s, 0xA5, (size_t)n * kScaleBytes);
      ofl_result r{};
      const int rc = cl.lookup(ids.data(), n, 2000000000LL, &r, msg);
      if (rc != OFL_OK) { fprintf(stderr, "lookup %lld rows: %d %s\n", (long long)n, rc, msg.c_str()); return bad_total + (1 << 20); }
      if (lr) {
        if (!lr->read(ids.data(), n, lw, ls)) return bad_total + (1 << 20);
        for (int64_t i = 0; i < n; i++)
          if (memcmp(w + i * kRowBytes, lw + i * kRowBytes, kRowBytes) || memcmp(s + i * kScaleBytes, ls + i * kScaleBytes, kScaleBytes))
            bad++;
      } else {  // no local copy: at least the zero-row rule
        for (int64_t i = 0; i < n; i++)
          if (ids[i] < 0 || ids[i] >= (int64_t)x.num_rows) {
            for (uint32_t k = 0; k < kRowBytes; k++) bad += w[i * kRowBytes + k] != 0;
          }
      }
      for (int64_t i = 0; i < n; i++) zeros += ids[i] < 0 || ids[i] >= (int64_t)x.num_rows;
      std::vector<int64_t> sorted(ids);
      std::sort(sorted.begin(), sorted.end());
      dups += (uint64_t)(sorted.end() - std::unique(sorted.begin(), sorted.end()));  // duplicate entries
      rows += (uint64_t)n;
    }
    printf("{\"test\":\"exact\",\"label\":\"%s\",\"backend\":\"%s\",\"client\":%d,\"rows_per_batch\":%lld,\"batches\":%d,"
           "\"rows\":%llu,\"zero_rows\":%llu,\"dup_entries\":%llu,\"compared_local\":%s,\"bad_rows\":%llu,\"conn\":\"%s\"}\n",
           x.label.c_str(), be_name(x.cc.backend), client_id, (long long)n, iters, (unsigned long long)rows,
           (unsigned long long)zeros, (unsigned long long)dups, lr ? "true" : "false", (unsigned long long)bad,
           cl.describe().c_str());
    fflush(stdout);
    bad_total += (int)bad;
  }
  free(w); free(s); free(lw); free(ls);
  return bad_total;
}

// latency: remote (fresh random rows each batch = cold caches on x86host; the same rows again = warm) and local AIO
int g_gap_ms = 0;  // bench: pause between lookups (decode-like pacing: the server's threads may go to sleep)

void run_bench(Ctx& x, const std::vector<int64_t>& sizes, int iters, uint64_t seed) {
  EngramClient cl;
  std::string msg;
  if (!cl.open(x.cc, x.d, x.row_start, x.num_rows, msg)) { fprintf(stderr, "open: %s\n", msg.c_str()); x.fails++; return; }
  uint8_t* w = static_cast<uint8_t*>(aligned((size_t)x.max_rows * kRowBytes));
  uint8_t* s = static_cast<uint8_t*>(aligned((size_t)x.max_rows * kScaleBytes));
  if (!cl.register_staging(w, (size_t)x.max_rows * kRowBytes, s, (size_t)x.max_rows * kScaleBytes, msg)) { x.fails++; return; }
  std::unique_ptr<LocalReader> lr(x.local ? new LocalReader(x.shard, x.d, x.row_start, x.num_rows, 256) : nullptr);
  std::mt19937_64 g(seed);
  for (int64_t n : sizes) {
    std::vector<int64_t> cold, warm, srv, lcold, lwarm;
    for (int it = 0; it < iters + 2; it++) {
      auto ids = make_ids(g, n, x.num_rows, false);
      ofl_result r{};
      if (g_gap_ms > 0) usleep((useconds_t)g_gap_ms * 1000);
      int64_t t0 = now_ns();
      int rc = cl.lookup(ids.data(), n, 2000000000LL, &r, msg);
      int64_t t1 = now_ns();
      if (rc != OFL_OK) { fprintf(stderr, "lookup: %s\n", msg.c_str()); x.fails++; return; }
      rc = cl.lookup(ids.data(), n, 2000000000LL, &r, msg);
      const int64_t t2 = now_ns();
      if (rc != OFL_OK) { fprintf(stderr, "lookup: %s\n", msg.c_str()); x.fails++; return; }
      if (it >= 2) { cold.push_back(t1 - t0); warm.push_back(t2 - t1); srv.push_back(r.server_ns); }
      if (lr && it >= 2) {
        auto ids2 = make_ids(g, n, x.num_rows, false);
        std::vector<uint8_t> tw((size_t)n * kRowBytes), ts((size_t)n * kScaleBytes);
        t0 = now_ns();
        if (!lr->read(ids2.data(), n, tw.data(), ts.data())) { x.fails++; return; }
        t1 = now_ns();
        if (!lr->read(ids2.data(), n, tw.data(), ts.data())) { x.fails++; return; }
        lcold.push_back(t1 - t0);
        lwarm.push_back(now_ns() - t1);
      }
    }
    printf("{\"test\":\"bench\",\"label\":\"%s\",\"backend\":\"%s\",\"gap_ms\":%d,\"rows_per_batch\":%lld,\"bytes\":%lld,"
           "\"remote_cold_us\":%s,\"remote_warm_us\":%s,\"server_us\":%s,\"local_odirect_aio_cold_us\":%s,"
           "\"local_odirect_aio_reread_us\":%s,\"conn\":\"%s\"}\n",
           x.label.c_str(), be_name(x.cc.backend), g_gap_ms, (long long)n, (long long)n * (kRowBytes + kScaleBytes),
           pct_json(cold).c_str(), pct_json(warm).c_str(), pct_json(srv).c_str(), pct_json(lcold).c_str(),
           pct_json(lwarm).c_str(), cl.describe().c_str());
    fflush(stdout);
  }
  free(w);
  free(s);
}

void expect(Ctx& x, const char* what, bool ok, const std::string& detail) {
  printf("{\"test\":\"negative\",\"label\":\"%s\",\"backend\":\"%s\",\"case\":\"%s\",\"pass\":%s,\"detail\":\"%s\"}\n",
         x.label.c_str(), be_name(x.cc.backend), what, ok ? "true" : "false", detail.c_str());
  fflush(stdout);
  if (!ok) x.fails++;
}

void run_negative(Ctx& x) {
  EngramClient cl;
  std::string msg;
  if (!cl.open(x.cc, x.d, x.row_start, x.num_rows, msg)) { expect(x, "open", false, msg); return; }
  uint8_t* w = static_cast<uint8_t*>(aligned((size_t)x.max_rows * kRowBytes));
  uint8_t* s = static_cast<uint8_t*>(aligned((size_t)x.max_rows * kScaleBytes));
  cl.register_staging(w, (size_t)x.max_rows * kRowBytes, s, (size_t)x.max_rows * kScaleBytes, msg);
  int64_t ids[4] = {0, 1, 2, 3};
  ofl_result r{};
  uint8_t bad_id[32];
  memcpy(bad_id, cl.identity(), 32);
  bad_id[7] ^= 0x40;
  int rc = cl.lookup_with_identity(bad_id, ids, 4, 1000000000LL, &r, msg);
  expect(x, "lookup with another identity refused", rc == OFL_EREMOTE && r.status == kErrIdentity,
         "rc " + std::to_string(rc) + " status " + std::to_string(r.status));
  const uint32_t raw[2] = {0, (uint32_t)x.num_rows};
  rc = cl.lookup_raw32(raw, 2, 1000000000LL, &r, msg);
  expect(x, "row id == num_rows (not the zero id) refused", rc == OFL_EREMOTE && r.status == kErrRowId,
         "rc " + std::to_string(rc) + " status " + std::to_string(r.status));
  rc = cl.lookup(ids, 4, 1000000000LL, &r, msg);
  expect(x, "connection usable after refusals", rc == OFL_OK, msg);
  rc = cl.lookup(ids, (int64_t)x.max_rows + 1, 1000000000LL, &r, msg);
  expect(x, "more rows than negotiated refused locally", rc == OFL_EINVAL, msg);
  {
    EngramClient c2;
    TableDesc other = x.d;
    other.shard_sha256[0] = other.shard_sha256[0] == '0' ? '1' : '0';  // a different checkpoint
    const bool ok = c2.open(x.cc, other, x.row_start, x.num_rows, msg);
    expect(x, "OPEN of another checkpoint's table refused", !ok && msg.find(std::to_string(kErrIdentity)) != std::string::npos, msg);
  }
  {
    EngramClient c3;
    const uint64_t start = x.d.rows() - 10;
    // a range running past the table: the client itself refuses; ask the server with a range of the right length
    // but the wrong layer instead
    TableDesc wrong = x.d;
    wrong.layer_id += 1;
    const bool ok = c3.open(x.cc, wrong, start, 10, msg);
    expect(x, "OPEN with a mismatched layer refused", !ok, msg);
  }
  free(w);
  free(s);
}

std::vector<int64_t> parse_sizes(const char* s) {
  std::vector<int64_t> v;
  for (const char* p = s; *p;) {
    v.push_back(strtoll(p, const_cast<char**>(&p), 10));
    if (*p == ',') p++;
  }
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  Ctx x;
  x.shard = arg(argc, argv, "--shard", "");
  x.manifest = arg(argc, argv, "--manifest", "");
  x.label = arg(argc, argv, "--label", "");
  x.cc.server_ip = arg(argc, argv, "--server", "");
  x.cc.local_ip = arg(argc, argv, "--local", "");
  x.cc.port = (uint16_t)atoi(arg(argc, argv, "--port", "18610"));
  x.cc.backend = std::string(arg(argc, argv, "--backend", "rdma")) == "tcp" ? OFL_BACKEND_TCP : OFL_BACKEND_RDMA;
  x.max_rows = (uint32_t)atol(arg(argc, argv, "--max-rows", "49152"));
  x.cc.max_rows = x.max_rows;
  char host[64] = {0};
  gethostname(host, sizeof(host) - 1);
  x.cc.name = std::string(host) + " engram_check";
  x.row_start = strtoull(arg(argc, argv, "--row-start", "0"), nullptr, 10);
  x.num_rows = strtoull(arg(argc, argv, "--num-rows", "0"), nullptr, 10);
  x.local = !flag(argc, argv, "--no-local");
  const std::string mode = arg(argc, argv, "--mode", "all");
  const auto sizes = parse_sizes(arg(argc, argv, "--sizes", "288,4096,16384,49152"));
  const int iters = atoi(arg(argc, argv, "--iters", "20"));
  const uint64_t seed = strtoull(arg(argc, argv, "--seed", "1"), nullptr, 10);
  const int clients = atoi(arg(argc, argv, "--clients", "1"));
  g_gap_ms = atoi(arg(argc, argv, "--gap-ms", "0"));
  if (x.shard.empty() || x.manifest.empty() || x.cc.server_ip.empty() || x.cc.local_ip.empty()) {
    fprintf(stderr, "usage: engram_check --shard F --manifest M --server IP --local IP [--row-start N --num-rows M] ...\n");
    return 64;
  }
  std::string msg;
  if (!describe_table(x.shard, -1, -1, x.manifest, x.d, msg)) { fprintf(stderr, "%s\n", msg.c_str()); return 2; }
  if (!x.num_rows) x.num_rows = x.d.rows() - x.row_start;
  uint8_t id[32];
  identity_digest(x.d, id);
  fprintf(stderr, "table %s layer %d rows [%llu, +%llu) of %llu, identity %s\n", x.d.shard.c_str(), x.d.layer_id,
          (unsigned long long)x.row_start, (unsigned long long)x.num_rows, (unsigned long long)x.d.rows(), hex(id, 32).c_str());
  if (mode == "ident") { printf("%s%s\n", identity_string(x.d).c_str(), hex(id, 32).c_str()); return 0; }
  int bad = 0;
  if (mode == "exact" || mode == "all") {
    if (clients <= 1) {
      bad += run_exact(x, sizes, iters, seed, 0);
    } else {  // concurrent clients, each its own connection, each checking its own rows
      std::vector<std::thread> th;
      std::atomic<int> b{0};
      for (int c = 0; c < clients; c++) th.emplace_back([&, c] { b += run_exact(x, sizes, iters, seed, c); });
      for (auto& t : th) t.join();
      bad += b.load();
    }
  }
  if (mode == "negative" || mode == "all") run_negative(x);
  if (mode == "bench" || mode == "all") run_bench(x, sizes, iters, seed);
  const bool ok = bad == 0 && x.fails == 0;
  fprintf(stderr, "engram_check %s: bad rows %d, failed checks %d\n", ok ? "PASS" : "FAIL", bad, x.fails);
  return ok ? 0 : 1;
}
