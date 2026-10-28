// engram_remote: Engram table reads served by the x86host table server ("engram-v1" over ofl, RDMA or TCP), as a drop-in
// for the image's EngramDiskTable (common/engram_disk.cpp), JIT-built in the serve container the same way.
//
// The Python-facing interface is EngramDiskTable's (constructor, read_rows, enqueue_read, enable_spin, serve_rows,
// stats, spin_stats, mode) plus register_staging (called once with the engine's fixed pinned staging buffers, which are
// registered with the NIC as the response destination: the server writes the rows straight into them) and
// remote_stats.
//
// Lookup path in spin mode (the serve's mode): enqueue_read puts the image's three stream memory operations on the
// stream (write the row count to a request word in registered host memory, wait until a done word reads 1, reset
// it), exactly as EngramDiskTable does; one proxy thread per process spins on the request words, sends the row ids
// to x86host and busy-polls for the response (which lands in the staging buffers), then sets done. Without spin mode
// enqueue_read launches a host node that does the same synchronously.
//
// Exactness and fallback: the image's own EngramDiskTable is compiled into this module (engram_disk.cpp is included
// below with its module definition turned off) and constructed with the same arguments; it is the exact fallback. A
// remote lookup that fails (timeout, transport error, refusal) is answered from the local shard by that reader
// (counted and logged, never zeros or stale rows); after a transport failure the connection's queue pair is already
// in the error state (no late write can land), the proxy waits a short quiesce, and the local reader overwrites every
// row of the request. A background thread reconnects (re-OPEN with the table identity) and remote reads resume.
//
// Table identity (engram_proto.h) comes from this shard's own safetensors header (read with O_DIRECT) plus its
// size/sha256 in the repo's manifest.json; the server refuses a mismatch at OPEN and at every LOOKUP.
#include <torch/extension.h>
#include <pybind11/stl.h>

// The image's disk reader, as the exact fallback: same translation unit, its PYBIND11_MODULE turned into an unused
// function (the include path has the image's common/ directory first).
#pragma push_macro("PYBIND11_MODULE")
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, variable) [[maybe_unused]] static void engram_disk_image_module(pybind11::module_& variable)
#include "engram_disk.cpp"
#undef PYBIND11_MODULE
#pragma pop_macro("PYBIND11_MODULE")

#include <stdarg.h>
#include <stdio.h>

#include "engram_client.h"

namespace {

void rlog(const char* f, ...) __attribute__((format(printf, 1, 2)));
void rlog(const char* f, ...) {
  char tb[32];
  time_t t = time(nullptr);
  strftime(tb, sizeof(tb), "%m-%d %H:%M:%S", localtime(&t));
  va_list ap;
  va_start(ap, f);
  fprintf(stderr, "INFO %s [engram_remote] ", tb);
  vfprintf(stderr, f, ap);
  fprintf(stderr, "\n");
  fflush(stderr);
  va_end(ap);
}

// "rdma://10.202.28.1:18610@10.202.28.2" or "tcp://..." (server ip:port @ local ip of the direct link)
bool parse_endpoint(const std::string& e, engram::ClientConfig& c, std::string& msg) {
  const size_t s = e.find("://");
  if (s == std::string::npos) { msg = "endpoint '" + e + "' lacks scheme://"; return false; }
  const std::string scheme = e.substr(0, s);
  if (scheme == "rdma") c.backend = OFL_BACKEND_RDMA;
  else if (scheme == "tcp") c.backend = OFL_BACKEND_TCP;
  else { msg = "endpoint scheme '" + scheme + "' is not rdma or tcp"; return false; }
  const std::string rest = e.substr(s + 3);
  const size_t at = rest.find('@');
  if (at == std::string::npos) { msg = "endpoint '" + e + "' lacks @local_ip"; return false; }
  const std::string hp = rest.substr(0, at);
  c.local_ip = rest.substr(at + 1);
  const size_t colon = hp.rfind(':');
  c.server_ip = colon == std::string::npos ? hp : hp.substr(0, colon);
  c.port = colon == std::string::npos ? engram::kDefaultPort : (uint16_t)atoi(hp.c_str() + colon + 1);
  if (c.server_ip.empty() || c.local_ip.empty() || !c.port) { msg = "bad endpoint '" + e + "'"; return false; }
  return true;
}

}  // namespace

class EngramRemoteTable;

// One proxy thread per process for every remote table in spin mode (mirrors the image's Proxy, minus the NVMe
// keep-alive: the reads go to the network).
class RemoteProxy {
 public:
  static RemoteProxy& get() {
    static RemoteProxy* p = new RemoteProxy();
    return *p;
  }
  void add(EngramRemoteTable* t, int64_t spin_ns) {
    std::lock_guard<std::mutex> g(mu_);
    tables_.push_back(t);
    if (spin_ns > spin_ns_.load()) spin_ns_.store(spin_ns);
    version_.fetch_add(1, std::memory_order_acq_rel);
    if (!started_) {
      started_ = true;
      std::thread([this] { loop(); }).detach();
    }
    cv_.notify_all();
  }
  void remove(EngramRemoteTable* t) {
    uint64_t v;
    {
      std::lock_guard<std::mutex> g(mu_);
      tables_.erase(std::remove(tables_.begin(), tables_.end(), t), tables_.end());
      v = version_.fetch_add(1, std::memory_order_acq_rel) + 1;
      cv_.notify_all();
    }
    while (acked_.load(std::memory_order_acquire) < v) std::this_thread::sleep_for(std::chrono::microseconds(20));
  }

 private:
  void loop();
  std::mutex mu_;
  std::condition_variable cv_;
  std::vector<EngramRemoteTable*> tables_;
  std::atomic<uint64_t> version_{0}, acked_{0};
  std::atomic<int64_t> spin_ns_{0};
  bool started_ = false;
};

class EngramRemoteTable {
 public:
  EngramRemoteTable(const std::string& weight_file, int64_t off_w, const std::string& scale_file, int64_t off_s,
                    int64_t row_start, int64_t num_rows, int64_t row_bytes, int64_t scale_bytes, int64_t cache_rows,
                    int threads, const std::string& endpoint, const std::string& manifest, int64_t timeout_ms,
                    int64_t retry_ms, const std::string& name, bool required)
      : local_(weight_file, off_w, scale_file, off_s, row_start, num_rows, row_bytes, scale_bytes, cache_rows, threads),
        row_start_(row_start), num_rows_(num_rows), name_(name), required_(required) {
    if (row_bytes != engram::kRowBytes || scale_bytes != engram::kScaleBytes)
      throw std::runtime_error("engram_remote: rows must be 256 + 8 bytes");
    if (weight_file != scale_file) throw std::runtime_error("engram_remote: weights and scales must share one shard");
    std::string msg;
    if (!engram::describe_table(weight_file, off_w, off_s, manifest, desc_, msg))
      throw std::runtime_error("engram_remote: " + msg);
    if (row_start < 0 || num_rows <= 0 || (uint64_t)(row_start + num_rows) > desc_.rows())
      throw std::runtime_error("engram_remote: row range outside the table");
    if (!parse_endpoint(endpoint, cfg_, msg)) throw std::runtime_error("engram_remote: " + msg);
    cfg_.name = name;
    timeout_ns_ = std::max<int64_t>(1, timeout_ms) * 1000000LL;
    retry_ns_ = std::max<int64_t>(100, retry_ms) * 1000000LL;
    uint8_t id[32];
    engram::identity_digest(desc_, id);
    identity_hex_ = engram::hex(id, 32);
    endpoint_ = endpoint;
  }

  ~EngramRemoteTable() {
    stop_.store(true);
    if (reconnector_.joinable()) reconnector_.join();
    {
      std::lock_guard<std::mutex> g(client_mu_);
      client_.close();  // deregisters the destination memory before the bounce buffers go away
    }
    if (bounce_) {
      std::free(dst_w_);
      std::free(dst_s_);
    }
    if (spin_) {
      RemoteProxy::get().remove(this);
      if (cudaHostUnregister(ctl_) != cudaSuccess) cudaGetLastError();
      std::free(ctl_);
    }
  }

  // The engine's fixed staging buffers (weights [max_rows, 256], scales [max_rows, 8]): connect, OPEN, register.
  // Throws if the server cannot be reached and `required`; otherwise starts in fallback mode and keeps retrying.
  std::string register_staging(int64_t w_ptr, int64_t s_ptr, int64_t max_rows) {
    std::lock_guard<std::mutex> g(client_mu_);
    if (st_w_) throw std::runtime_error("engram_remote: staging already registered");
    st_w_ = reinterpret_cast<uint8_t*>(w_ptr);
    st_s_ = reinterpret_cast<uint8_t*>(s_ptr);
    max_rows_ = max_rows;
    cfg_.max_rows = (uint32_t)max_rows;
    // VLLM_ENGRAM_REMOTE_BOUNCE=1: the NIC writes into a separate registered buffer and the proxy copies the rows into
    // the staging buffers with the CPU, as the disk reader does (a switch in case direct NIC placement into
    // GPU-read staging ever shows a coherence problem on a platform)
    const char* b = getenv("VLLM_ENGRAM_REMOTE_BOUNCE");
    bounce_ = b && std::string(b) == "1";
    dst_w_ = st_w_;
    dst_s_ = st_s_;
    if (bounce_) {
      void* pw = nullptr;
      void* ps = nullptr;
      if (posix_memalign(&pw, 4096, (size_t)max_rows * engram::kRowBytes) != 0 ||
          posix_memalign(&ps, 4096, (size_t)max_rows * engram::kScaleBytes) != 0)
        throw std::runtime_error("engram_remote: no memory for the bounce buffers");
      dst_w_ = static_cast<uint8_t*>(pw);
      dst_s_ = static_cast<uint8_t*>(ps);
    }
    std::string msg;
    const bool ok = client_.open(cfg_, desc_, (uint64_t)row_start_, (uint64_t)num_rows_, msg) &&
                    client_.register_staging(dst_w_, (size_t)max_rows * engram::kRowBytes, dst_s_,
                                             (size_t)max_rows * engram::kScaleBytes, msg);
    if (!ok) {
      client_.close();
      if (required_)
        throw std::runtime_error("engram_remote: " + name_ + ": cannot use " + endpoint_ + ": " + msg +
                                 " (VLLM_ENGRAM_REMOTE_REQUIRED=0 starts on the local reader instead)");
      rlog("%s: %s unreachable (%s); serving from the local shard and retrying every %lld ms", name_.c_str(),
           endpoint_.c_str(), msg.c_str(), (long long)(retry_ns_ / 1000000));
      down_.store(true);
    } else {
      down_.store(false);
      rlog("%s: remote rows from %s (%s), table identity %s", name_.c_str(), endpoint_.c_str(), client_.describe().c_str(),
           identity_hex_.c_str());
    }
    reconnector_ = std::thread([this] { reconnect_loop(); });
    return describe_locked();
  }

  void read_rows(int64_t ids_ptr, int64_t w_ptr, int64_t s_ptr, int64_t n) {
    serve(reinterpret_cast<const int64_t*>(ids_ptr), reinterpret_cast<uint8_t*>(w_ptr), reinterpret_cast<uint8_t*>(s_ptr), n);
  }

  void enqueue_read(int64_t stream, int64_t ids_ptr, int64_t w_ptr, int64_t s_ptr, int64_t n) {
    if (spin_) {
      if (n <= 0) return;
      if (n > INT32_MAX) throw std::runtime_error("engram_remote: too many rows for one spin-mode request");
      {
        std::lock_guard<std::mutex> g(jobs_mu_);
        if (!sp_ids_.load()) {
          sp_ids_.store(reinterpret_cast<const int64_t*>(ids_ptr));
          sp_w_.store(reinterpret_cast<uint8_t*>(w_ptr));
          sp_s_.store(reinterpret_cast<uint8_t*>(s_ptr));
        } else if (sp_ids_.load() != reinterpret_cast<const int64_t*>(ids_ptr) ||
                   sp_w_.load() != reinterpret_cast<uint8_t*>(w_ptr) || sp_s_.load() != reinterpret_cast<uint8_t*>(s_ptr)) {
          throw std::runtime_error("engram_remote: spin mode serves one set of staging buffers per table");
        }
        if (n > sp_max_.load()) sp_max_.store(n);
      }
      CUstream st = reinterpret_cast<CUstream>(stream);
      const Memops& m = memops();
      CUresult r = m.write(st, req_dev_, static_cast<cuuint32_t>(n), CU_STREAM_WRITE_VALUE_DEFAULT);
      if (r == CUDA_SUCCESS) r = m.wait(st, done_dev_, 1, CU_STREAM_WAIT_VALUE_EQ);
      if (r == CUDA_SUCCESS) r = m.write(st, done_dev_, 0, CU_STREAM_WRITE_VALUE_DEFAULT);
      if (r != CUDA_SUCCESS) throw std::runtime_error("engram_remote: stream memory operation failed: " + std::to_string(r));
      return;
    }
    Job* job;
    {
      std::lock_guard<std::mutex> g(jobs_mu_);
      auto& slot = jobs_[std::make_tuple(ids_ptr, w_ptr, s_ptr, n)];
      if (!slot) slot.reset(new Job{this, ids_ptr, w_ptr, s_ptr, n});
      job = slot.get();
    }
    cudaError_t e = cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream), &EngramRemoteTable::host_fn, job);
    if (e != cudaSuccess) throw std::runtime_error(std::string("engram_remote: cudaLaunchHostFunc: ") + cudaGetErrorString(e));
  }

  // Same signature as EngramDiskTable::enable_spin; queue_depth / keepalive_us / pool_rows are the local reader's
  // and do not apply to remote reads (the fallback uses the local reader's pread pool).
  std::string enable_spin(int64_t spin_ms, int64_t, int64_t, int64_t) {
    if (spin_) return mode();
    if (!memops().write || !memops().wait) throw std::runtime_error("engram_remote: libcuda has no stream memory operations");
    void* p = nullptr;
    if (posix_memalign(&p, 4096, 4096) != 0) throw std::runtime_error("engram_remote: posix_memalign");
    std::memset(p, 0, 4096);
    check_cuda(cudaHostRegister(p, 4096, cudaHostRegisterMapped | cudaHostRegisterPortable), "cudaHostRegister");
    ctl_ = static_cast<int32_t*>(p);
    void* d = nullptr;
    check_cuda(cudaHostGetDevicePointer(&d, p, 0), "cudaHostGetDevicePointer");
    req_dev_ = reinterpret_cast<CUdeviceptr>(d);
    done_dev_ = req_dev_ + kDone * sizeof(int32_t);
    spin_ = true;
    RemoteProxy::get().add(this, spin_ms * 1000000LL);
    return mode();
  }

  void serve_rows(int64_t ids_ptr, int64_t w_ptr, int64_t s_ptr, int64_t n) {
    serve(reinterpret_cast<const int64_t*>(ids_ptr), reinterpret_cast<uint8_t*>(w_ptr), reinterpret_cast<uint8_t*>(s_ptr), n);
  }

  // Called by the proxy thread: serve a pending request; false if there was none.
  bool poll() {
    const int32_t n = __atomic_load_n(&ctl_[0], __ATOMIC_ACQUIRE);
    if (n == 0) return false;
    __atomic_store_n(&ctl_[0], 0, __ATOMIC_RELAXED);
    const int64_t t0 = now_ns();
    try {
      const int64_t* ids = sp_ids_.load(std::memory_order_acquire);
      const int64_t rows = std::min<int64_t>(n, sp_max_.load(std::memory_order_acquire));
      if (ids) serve(ids, sp_w_.load(std::memory_order_acquire), sp_s_.load(std::memory_order_acquire), rows);
    } catch (const std::exception& e) {
      die_loudly(e.what());  // the staging rows may be partial: never let the GPU go on with them
    } catch (...) {
      die_loudly("unknown error in the lookup proxy");
    }
    const int64_t dt = now_ns() - t0;
    served_.fetch_add(1);
    serve_ns_.fetch_add(dt);
    if (dt > serve_max_ns_.load()) serve_max_ns_.store(dt);
    __atomic_store_n(&ctl_[kDone], 1, __ATOMIC_RELEASE);
    return true;
  }

  std::vector<int64_t> stats() const { return {0, rows_.load() + fb_rows_.load()}; }  // (hits, misses) like the image
  std::vector<int64_t> spin_stats() const { return {served_.load(), serve_ns_.load(), serve_max_ns_.load(), errors_.load()}; }

  std::map<std::string, int64_t> remote_stats() const {
    std::map<std::string, int64_t> m{{"requests", reqs_.load()},
                                     {"rows", rows_.load()},
                                     {"bytes", bytes_.load()},
                                     {"remote_ns", remote_ns_.load()},
                                     {"remote_max_ns", remote_max_ns_.load()},
                                     {"fallbacks", fallbacks_.load()},
                                     {"fallback_rows", fb_rows_.load()},
                                     {"fallback_unregistered", fb_unreg_.load()},
                                     {"transport_errors", transport_errs_.load()},
                                     {"refusals", refusals_.load()},
                                     {"reconnects", reconnects_.load()},
                                     {"down", down_.load() ? 1 : 0}};
    for (int k = 0; k < 32; k++) m["hist_log2ns_" + std::to_string(k)] = hist_[k].load();
    return m;
  }

  std::string mode() const {
    return std::string("remote ") + endpoint_ + (spin_ ? " (spin proxy)" : " (host node)") +
           (bounce_ ? ", bounce copy" : ", direct placement") + ", exact fallback: " + local_.mode();
  }

  std::string describe() {
    std::lock_guard<std::mutex> g(client_mu_);
    return describe_locked();
  }

  std::string identity() const { return identity_hex_; }

 private:
  struct Job { EngramRemoteTable* t; int64_t ids, w, s, n; };
  static constexpr int kDone = 32;
  static void CUDART_CB host_fn(void* p) {
    auto* j = static_cast<Job*>(p);
    try {
      j->t->serve(reinterpret_cast<const int64_t*>(j->ids), reinterpret_cast<uint8_t*>(j->w), reinterpret_cast<uint8_t*>(j->s), j->n);
    } catch (const std::exception& e) {
      j->t->die_loudly(e.what());
    } catch (...) {
      j->t->die_loudly("unknown error in the lookup host node");
    }
  }

  // serve() threw (only an allocation failure can make it): the rows of this lookup may be partial and the GPU reads
  // them as soon as the proxy answers, so stop the process instead of answering (lane rule: never stale rows)
  [[noreturn]] void die_loudly(const char* why) {
    rlog("FATAL %s: lookup failed inside the proxy (%s); aborting rather than returning partial rows", name_.c_str(), why);
    std::abort();
  }

  std::string describe_locked() const {  // caller holds client_mu_
    return name_ + ": " + (down_.load() ? std::string("DOWN (local reads)") : client_.describe()) + ", identity " +
           identity_hex_;
  }

  // n rows of ids into (w, s): remote when possible, else (or on any failure) the image's exact local reader.
  void serve(const int64_t* ids, uint8_t* w, uint8_t* s, int64_t n) {
    if (n <= 0) return;
    if (w == st_w_ && s == st_s_ && n <= max_rows_ && !down_.load(std::memory_order_acquire)) {
      std::unique_lock<std::mutex> l(client_mu_, std::try_to_lock);  // held only while reconnecting
      if (l.owns_lock() && client_.usable()) {
        ofl_result res{};
        std::string msg;
        const int64_t t0 = now_ns();
        const int rc = client_.lookup(ids, n, timeout_ns_, &res, msg);
        if (rc == OFL_OK && bounce_) {
          std::memcpy(w, dst_w_, (size_t)n * engram::kRowBytes);
          std::memcpy(s, dst_s_, (size_t)n * engram::kScaleBytes);
        }
        const int64_t dt = now_ns() - t0;
        if (rc == OFL_OK) {
          maybe_log(t0);
          reqs_.fetch_add(1, std::memory_order_relaxed);
          rows_.fetch_add(n, std::memory_order_relaxed);
          bytes_.fetch_add(n * (int64_t)(engram::kRowBytes + engram::kScaleBytes), std::memory_order_relaxed);
          remote_ns_.fetch_add(dt, std::memory_order_relaxed);
          if (dt > remote_max_ns_.load(std::memory_order_relaxed)) remote_max_ns_.store(dt, std::memory_order_relaxed);
          int k = 0;
          while (k < 31 && (1LL << (k + 1)) <= dt) k++;
          hist_[k].fetch_add(1, std::memory_order_relaxed);
          return;
        }
        if (rc == OFL_EREMOTE) {
          refusals_.fetch_add(1);
          client_.close();  // the server refused (e.g. it restarted with other tables): re-OPEN from scratch
        } else {
          transport_errs_.fetch_add(1);  // the connection is broken (QP in the error state)
        }
        down_.store(true, std::memory_order_release);  // until the reconnector re-OPENs
        l.unlock();
        if (rc != OFL_EREMOTE) {
          // the QP is in the error state; let any write the NIC had already accepted settle before overwriting
          std::this_thread::sleep_for(std::chrono::microseconds(2000));
        }
        log_fallback(rc, msg, n, dt);
      }
    } else if (w != st_w_ || s != st_s_) {
      fb_unreg_.fetch_add(1);
    }
    fallbacks_.fetch_add(1, std::memory_order_relaxed);
    fb_rows_.fetch_add(n, std::memory_order_relaxed);
    local_.read_rows(reinterpret_cast<int64_t>(ids), reinterpret_cast<int64_t>(w), reinterpret_cast<int64_t>(s), n);
  }

  // one line of counters every log_ns_ while lookups run (VLLM_ENGRAM_REMOTE_LOG_S, 0 = off)
  void maybe_log(int64_t now) {
    if (log_ns_ <= 0 || now - last_stats_log_ < log_ns_) return;
    const bool first = last_stats_log_ == 0;
    last_stats_log_ = now;
    if (first) return;
    const int64_t r = reqs_.load(), rows = rows_.load();
    int64_t p50 = 0, acc = 0;
    for (int k = 0; k < 32 && acc * 2 < r; k++) { acc += hist_[k].load(); p50 = 1LL << k; }
    rlog("%s: %lld remote lookups, %lld rows, %.1f MB, mean %.1f us, p50 bucket <= %.1f us, max %.1f us; fallbacks %lld "
         "(%lld rows), transport errors %lld, refusals %lld, reconnects %lld", name_.c_str(), (long long)r, (long long)rows,
         bytes_.load() / 1e6, r ? remote_ns_.load() / 1e3 / r : 0.0, 2 * p50 / 1e3, remote_max_ns_.load() / 1e3,
         (long long)fallbacks_.load(), (long long)fb_rows_.load(), (long long)transport_errs_.load(),
         (long long)refusals_.load(), (long long)reconnects_.load());
  }

  void log_fallback(int rc, const std::string& msg, int64_t n, int64_t dt) {
    const int64_t now = now_ns();
    if (now - last_fb_log_ > 1000000000LL) {
      last_fb_log_ = now;
      rlog("%s: remote lookup of %lld rows failed after %.2f ms (rc %d: %s); exact local rows used, reconnecting "
           "(fallbacks so far %lld)", name_.c_str(), (long long)n, dt / 1e6, rc, msg.c_str(), (long long)fallbacks_.load() + 1);
    }
  }

  void reconnect_loop() {
    int64_t last = now_ns();
    while (!stop_.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      if (!down_.load() || now_ns() - last < retry_ns_) continue;
      last = now_ns();
      std::lock_guard<std::mutex> g(client_mu_);
      if (st_w_ == nullptr) continue;
      std::string msg;
      bool ok = client_.reconnect(msg);  // broken connection: new QP + re-OPEN; closed client: connect + OPEN
      if (!ok) {  // a fresh connection from scratch (e.g. the first one never came up)
        client_.close();
        ok = client_.open(cfg_, desc_, (uint64_t)row_start_, (uint64_t)num_rows_, msg) &&
             client_.register_staging(dst_w_, (size_t)max_rows_ * engram::kRowBytes, dst_s_,
                                      (size_t)max_rows_ * engram::kScaleBytes, msg);
      }
      if (ok) {
        reconnects_.fetch_add(1);
        down_.store(false, std::memory_order_release);
        rlog("%s: remote rows again from %s (%s)", name_.c_str(), endpoint_.c_str(), client_.describe().c_str());
      } else {
        const int64_t now = now_ns();
        if (now - last_rc_log_ > 60000000000LL) {
          last_rc_log_ = now;
          rlog("%s: reconnect to %s failed: %s (retrying; local rows meanwhile)", name_.c_str(), endpoint_.c_str(), msg.c_str());
        }
      }
    }
  }

  EngramDiskTable local_;  // the image's reader: the exact fallback
  int64_t row_start_, num_rows_;
  std::string name_, endpoint_, identity_hex_;
  bool required_;
  engram::TableDesc desc_;
  engram::ClientConfig cfg_;
  engram::EngramClient client_;
  std::mutex client_mu_;
  int64_t timeout_ns_ = 100000000LL, retry_ns_ = 5000000000LL;
  uint8_t* st_w_ = nullptr;  // the engine's staging
  uint8_t* st_s_ = nullptr;
  uint8_t* dst_w_ = nullptr;  // where the NIC writes (= staging, or the bounce buffers)
  uint8_t* dst_s_ = nullptr;
  bool bounce_ = false;
  int64_t max_rows_ = 0;
  std::atomic<bool> down_{true}, stop_{false};
  std::thread reconnector_;
  int64_t last_fb_log_ = 0, last_rc_log_ = 0, last_stats_log_ = 0;
  int64_t log_ns_ = (int64_t)(atof(getenv("VLLM_ENGRAM_REMOTE_LOG_S") ? getenv("VLLM_ENGRAM_REMOTE_LOG_S") : "60") * 1e9);
  // spin mode (as EngramDiskTable)
  bool spin_ = false;
  int32_t* ctl_ = nullptr;
  CUdeviceptr req_dev_ = 0, done_dev_ = 0;
  std::mutex jobs_mu_;
  std::map<std::tuple<int64_t, int64_t, int64_t, int64_t>, std::unique_ptr<Job>> jobs_;
  std::atomic<const int64_t*> sp_ids_{nullptr};
  std::atomic<uint8_t*> sp_w_{nullptr}, sp_s_{nullptr};
  std::atomic<int64_t> sp_max_{0};
  // counters
  std::atomic<int64_t> served_{0}, serve_ns_{0}, serve_max_ns_{0}, errors_{0};
  std::atomic<int64_t> reqs_{0}, rows_{0}, bytes_{0}, remote_ns_{0}, remote_max_ns_{0};
  std::atomic<int64_t> fallbacks_{0}, fb_rows_{0}, fb_unreg_{0}, transport_errs_{0}, refusals_{0}, reconnects_{0};
  std::atomic<int64_t> hist_[32] = {};
};

void RemoteProxy::loop() {
  std::vector<EngramRemoteTable*> local;
  uint64_t seen = 0;
  int64_t last = now_ns();
  for (;;) {
    if (version_.load(std::memory_order_acquire) != seen) {
      std::unique_lock<std::mutex> l(mu_);
      local = tables_;
      seen = version_.load(std::memory_order_acquire);
      acked_.store(seen, std::memory_order_release);
      if (local.empty()) {
        cv_.wait(l, [&] { return version_.load(std::memory_order_acquire) != seen; });
        continue;
      }
      last = now_ns();
    }
    bool busy = false;
    for (EngramRemoteTable* t : local) busy |= t->poll();
    const int64_t now = now_ns();
    if (busy) {
      last = now;
    } else if (now - last > spin_ns_.load(std::memory_order_relaxed)) {
      timespec ts{0, 50000};
      nanosleep(&ts, nullptr);
    } else {
      cpu_relax();
    }
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<EngramRemoteTable>(m, "EngramRemoteTable")
      .def(pybind11::init<const std::string&, int64_t, const std::string&, int64_t, int64_t, int64_t, int64_t, int64_t,
                          int64_t, int, const std::string&, const std::string&, int64_t, int64_t, const std::string&, bool>(),
           pybind11::arg("weight_file"), pybind11::arg("weight_offset"), pybind11::arg("scale_file"),
           pybind11::arg("scale_offset"), pybind11::arg("row_start"), pybind11::arg("num_rows"),
           pybind11::arg("row_bytes"), pybind11::arg("scale_bytes"), pybind11::arg("cache_rows") = 0,
           pybind11::arg("threads") = 32, pybind11::arg("endpoint") = "", pybind11::arg("manifest") = "",
           pybind11::arg("timeout_ms") = 100, pybind11::arg("retry_ms") = 5000, pybind11::arg("name") = "",
           pybind11::arg("required") = true)
      .def("register_staging", &EngramRemoteTable::register_staging, pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("read_rows", &EngramRemoteTable::read_rows, pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("enqueue_read", &EngramRemoteTable::enqueue_read)
      .def("enable_spin", &EngramRemoteTable::enable_spin, pybind11::arg("spin_ms") = 1000,
           pybind11::arg("queue_depth") = 256, pybind11::arg("keepalive_us") = 2000, pybind11::arg("pool_rows") = 512)
      .def("serve_rows", &EngramRemoteTable::serve_rows, pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("stats", &EngramRemoteTable::stats)
      .def("spin_stats", &EngramRemoteTable::spin_stats)
      .def("remote_stats", &EngramRemoteTable::remote_stats)
      .def("describe", &EngramRemoteTable::describe)
      .def("identity", &EngramRemoteTable::identity)
      .def("mode", &EngramRemoteTable::mode);
}
