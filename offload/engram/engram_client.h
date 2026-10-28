// engram_client.h: the client half of "engram-v1" (header-only; uses ofl + engram_proto). Shared by the Spark-side
// EngramRemoteTable (offload/engram/spark/engram_remote.cpp) and the standalone exactness/bench client
// (offload/engram/tests/engram_check.cpp), so both exercise the same code.
//
// One EngramClient = one ofl connection = one (table, rank range). The engine's staging buffers are registered once
// as two destination regions: weights [max_rows, 256] and scales [max_rows, 8]; a lookup of n rows lands in rows
// [0, n) of both, exactly the layout the engine's lookup kernel reads.
#ifndef ENGRAM_CLIENT_H
#define ENGRAM_CLIENT_H

#include <stdio.h>
#include <string.h>

#include <string>

#include "engram_proto.h"
#include "ofl.h"

namespace engram {

struct ClientConfig {
  int backend = OFL_BACKEND_RDMA;
  std::string server_ip, local_ip, rdma_device;
  uint16_t port = kDefaultPort;
  int gid_index = -1;
  uint32_t max_rows = 49152;
  int64_t connect_timeout_ms = 3000;
  std::string name;  // shows in the server log ("sparkhead rank 0 layer 1")
};

class EngramClient {
 public:
  EngramClient() = default;
  EngramClient(const EngramClient&) = delete;
  EngramClient& operator=(const EngramClient&) = delete;
  ~EngramClient() { close(); }

  // Connect and OPEN (identity + range checked by the server). false + msg on failure.
  bool open(const ClientConfig& cfg, const TableDesc& d, uint64_t row_start, uint64_t num_rows, std::string& msg) {
    cfg_ = cfg;
    desc_ = d;
    identity_digest(d, identity_);
    row_start_ = row_start;
    num_rows_ = num_rows;
    if (num_rows_ == 0 || num_rows_ >= kZeroRow || row_start + num_rows > d.rows()) {
      msg = "row range outside the table";
      return false;
    }
    return connect_and_open(msg);
  }

  // Register the destination buffers (weights: >= max_rows*256 bytes, scales: >= max_rows*8). Once.
  bool register_staging(void* w, size_t wlen, void* s, size_t slen, std::string& msg) {
    if (!c_) { msg = "not connected"; return false; }
    if (registered_ && w == w_ && s == s_) return true;  // already (re-)registered by connect_and_open
    if (wlen < (size_t)cfg_.max_rows * kRowBytes || slen < (size_t)cfg_.max_rows * kScaleBytes) {
      msg = "staging smaller than max_rows";
      return false;
    }
    if (ofl_client_reg(c_, w, wlen, &w_region_) != OFL_OK || ofl_client_reg(c_, s, slen, &s_region_) != OFL_OK) {
      msg = std::string("register staging: ") + ofl_client_error(c_);
      return false;
    }
    w_ = w;
    s_ = s;
    registered_ = true;
    return true;
  }

  // Look up n engine row ids (int64, relative to row_start; < 0 or >= num_rows read as zeros, like the disk reader)
  // into staging rows [0, n). OFL_OK, or an ofl error (OFL_EREMOTE: refused, res->status = engram error code; any
  // other error leaves the connection broken, with no late write possible).
  int lookup(const int64_t* ids, int64_t n, int64_t timeout_ns, ofl_result* res, std::string& msg) {
    if (!c_ || !registered_) { msg = "not open"; return OFL_EBROKEN; }
    if (n <= 0 || n > (int64_t)cfg_.max_rows) { msg = "row count outside [1, max_rows]"; return OFL_EINVAL; }
    uint32_t cap = 0;
    uint8_t* p = static_cast<uint8_t*>(ofl_client_payload_buf(c_, &cap));
    if (!p) { msg = ofl_client_error(c_); return OFL_EBROKEN; }
    LookupHdr h{};
    h.handle = handle_;
    h.n = (uint32_t)n;
    memcpy(h.identity, identity_, 32);
    memcpy(p, &h, sizeof(h));
    uint32_t* out = reinterpret_cast<uint32_t*>(p + sizeof(h));
    const int64_t lim = (int64_t)num_rows_;
    for (int64_t i = 0; i < n; i++) {
      const int64_t v = ids[i];
      out[i] = (v < 0 || v >= lim) ? kZeroRow : (uint32_t)v;
    }
    const ofl_seg segs[2] = {{w_region_, 0, 0, (uint64_t)n * kRowBytes}, {s_region_, 0, 0, (uint64_t)n * kScaleBytes}};
    const uint32_t plen = (uint32_t)(sizeof(h) + 4 * (size_t)n);
    ofl_result r{};
    const int rc = ofl_client_call(c_, kOpLookup, 0, p, plen, segs, 2, &r, timeout_ns);
    if (res) *res = r;
    if (rc == OFL_OK && r.resp_len != (uint32_t)(n * (kRowBytes + kScaleBytes))) {
      msg = "short lookup response";
      return OFL_EPROTO;
    }
    if (rc != OFL_OK) msg = rc == OFL_EREMOTE ? "server refused the lookup, status " + std::to_string(r.status)
                                              : std::string(ofl_client_error(c_));
    return rc;
  }

  // Lookup with an arbitrary identity (tests: a mismatched identity must be refused).
  int lookup_with_identity(const uint8_t identity[32], const int64_t* ids, int64_t n, int64_t timeout_ns,
                           ofl_result* res, std::string& msg) {
    uint8_t keep[32];
    memcpy(keep, identity_, 32);
    memcpy(identity_, identity, 32);
    const int rc = lookup(ids, n, timeout_ns, res, msg);
    memcpy(identity_, keep, 32);
    return rc;
  }

  // Raw lookup of already packed uint32 ids (tests: ids the client would never send, e.g. >= num_rows).
  int lookup_raw32(const uint32_t* ids, uint32_t n, int64_t timeout_ns, ofl_result* res, std::string& msg) {
    uint32_t cap = 0;
    uint8_t* p = static_cast<uint8_t*>(ofl_client_payload_buf(c_, &cap));
    if (!p) { msg = ofl_client_error(c_); return OFL_EBROKEN; }
    LookupHdr h{};
    h.handle = handle_;
    h.n = n;
    memcpy(h.identity, identity_, 32);
    memcpy(p, &h, sizeof(h));
    memcpy(p + sizeof(h), ids, 4 * (size_t)n);
    const ofl_seg segs[2] = {{w_region_, 0, 0, (uint64_t)n * kRowBytes}, {s_region_, 0, 0, (uint64_t)n * kScaleBytes}};
    const int rc = ofl_client_call(c_, kOpLookup, 0, p, (uint32_t)(sizeof(h) + 4 * (size_t)n), segs, 2, res, timeout_ns);
    if (rc != OFL_OK) msg = ofl_client_error(c_);
    return rc;
  }

  // Re-connect, re-register the staging buffers (new keys; same regions) and re-OPEN.
  bool reconnect(std::string& msg) {
    if (!c_) return connect_and_open(msg);
    char err[512];
    if (ofl_client_reconnect(c_, err, sizeof(err)) != OFL_OK) { msg = err; return false; }
    return do_open(msg);
  }

  bool usable() const { return c_ && !ofl_client_broken(c_) && registered_; }
  std::string describe() const { return c_ ? std::string(ofl_client_describe(c_)) + ", " + open_desc_ : "closed"; }
  const uint8_t* identity() const { return identity_; }
  uint64_t num_rows() const { return num_rows_; }
  void close() {
    if (c_) ofl_client_close(c_);
    c_ = nullptr;
    registered_ = false;
    open_registered_ = false;
  }

 private:
  bool connect_and_open(std::string& msg) {
    ofl_client_config oc{};
    oc.backend = cfg_.backend;
    oc.server_ip = cfg_.server_ip.c_str();
    oc.server_port = cfg_.port;
    oc.local_ip = cfg_.local_ip.c_str();
    oc.rdma_device = cfg_.rdma_device.empty() ? nullptr : cfg_.rdma_device.c_str();
    oc.gid_index = cfg_.gid_index;
    oc.service = kService;
    oc.max_req_bytes = (uint32_t)(sizeof(LookupHdr) + 4ull * cfg_.max_rows);
    oc.max_resp_bytes = (uint32_t)((uint64_t)cfg_.max_rows * (kRowBytes + kScaleBytes));
    oc.max_inflight = 1;
    oc.connect_timeout_ms = cfg_.connect_timeout_ms;
    char err[512];
    c_ = ofl_client_connect(&oc, err, sizeof(err));
    if (!c_) { msg = err; return false; }
    if (w_ && s_) {  // a fresh client after close(): register the same staging again
      std::string m2;
      if (!register_staging(w_, (size_t)cfg_.max_rows * kRowBytes, s_, (size_t)cfg_.max_rows * kScaleBytes, m2)) {
        msg = m2;
        return false;
      }
    }
    return do_open(msg);
  }

  bool do_open(std::string& msg) {
    OpenReq q{};
    q.version = kProtoVersion;
    q.layer_id = (uint32_t)desc_.layer_id;
    q.row_start = row_start_;
    q.num_rows = num_rows_;
    q.row_bytes = kRowBytes;
    q.scale_bytes = kScaleBytes;
    memcpy(q.identity, identity_, 32);
    snprintf(q.client, sizeof(q.client), "%s", cfg_.name.c_str());
    OpenResp a{};
    if (!open_registered_) {  // once per client (ofl_client_reconnect keeps regions)
      if (ofl_client_reg(c_, &open_buf_, sizeof(open_buf_), &open_region_) != OFL_OK) { msg = ofl_client_error(c_); return false; }
      open_registered_ = true;
    }
    const ofl_seg seg{open_region_, 0, 0, sizeof(OpenResp)};
    ofl_result r{};
    const int rc = ofl_client_call(c_, kOpOpen, 0, &q, sizeof(q), &seg, 1, &r, 2000000000LL);
    if (rc != OFL_OK) {
      msg = rc == OFL_EREMOTE ? "OPEN refused by the server, status " + std::to_string(r.status)
                              : std::string("OPEN: ") + ofl_client_error(c_);
      return false;
    }
    memcpy(&a, &open_buf_, sizeof(a));
    if (r.resp_len != sizeof(OpenResp) || a.version != kProtoVersion || memcmp(a.identity, identity_, 32) != 0) {
      msg = "OPEN answered with a different table identity";
      return false;
    }
    handle_ = a.handle;
    a.desc[sizeof(a.desc) - 1] = 0;
    open_desc_ = a.desc;
    return true;
  }

  ClientConfig cfg_;
  TableDesc desc_;
  uint8_t identity_[32] = {0};
  uint64_t row_start_ = 0, num_rows_ = 0;
  ofl_client* c_ = nullptr;
  uint32_t handle_ = 0, w_region_ = 0, s_region_ = 0;
  void* w_ = nullptr;
  void* s_ = nullptr;
  bool registered_ = false;
  OpenResp open_buf_{};
  uint32_t open_region_ = 0;
  bool open_registered_ = false;
  std::string open_desc_;
};

}  // namespace engram
#endif
