// ofl transport implementation (see ofl.h). Builds on x86_64 (x86host) and aarch64 (DGX Spark, inside or outside the
// serve container) with g++ >= 9 and libibverbs >= 28. The QP setup follows the plugin's RDMA ring
// (vllm_exl3_moe/csrc/exl3_rdma.cu, itself from TensorFold): RC, path MTU = the ports' active MTU, timeout 14,
// retry 7, RNR retry 7, no relaxed ordering (the response header must land after the response bytes).
#include "ofl.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

static_assert(sizeof(ofl_hdr) == 64, "ofl_hdr must be 64 bytes");
static_assert(sizeof(ofl_wire_seg) == 16, "ofl_wire_seg must be 16 bytes");
#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "ofl assumes little-endian hosts (x86_64, aarch64)"
#endif

namespace {

constexpr uint32_t kSegArea = 64 + 16 * OFL_MAX_SEGS;  // request slot: header + room for every segment, then payload
constexpr uint64_t kWrSend = 1ull << 62;                // client wr_id tags
constexpr uint64_t kWrHdr = 1ull << 62;                 // server wr_id tags: header write (low bits: buffer)
constexpr uint64_t kWrMark = 1ull << 61;                //                    periodic signaled data write
constexpr int kSignalEvery = 32;                        // server: one signaled data write per this many

struct Hello {  // bootstrap message payload, both directions
  uint32_t backend;
  uint32_t max_req_bytes, max_resp_bytes, max_inflight;
  uint32_t qpn, psn;
  uint8_t gid[16];
  uint32_t mtu;  // enum ibv_mtu
  int32_t status;
  char service[64];
  char text[128];  // client: host name; ack: reject reason
};
static_assert(sizeof(Hello) == 240, "Hello layout");

inline int64_t now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

inline void cpu_relax() {
#if defined(__aarch64__)
  asm volatile("isb" ::: "memory");
#elif defined(__x86_64__)
  __builtin_ia32_pause();
#endif
}

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, f);
  vsnprintf(buf, sizeof(buf), f, ap);
  va_end(ap);
  return buf;
}

void copy_err(char* err, size_t errlen, const std::string& msg) {
  if (err && errlen) snprintf(err, errlen, "%s", msg.c_str());
}

bool parse_ipv4(const char* s, in_addr* out) { return s && *s && inet_pton(AF_INET, s, out) == 1; }

// ---- sockets: non-blocking, every operation against a deadline --------------------------------------------------

int wait_fd(int fd, short ev, int64_t deadline) {  // 1 ready, 0 deadline passed, -1 error
  for (;;) {
    int64_t left = deadline - now_ns();
    if (left <= 0) return 0;
    timespec ts{left / 1000000000LL, left % 1000000000LL};
    pollfd p{fd, ev, 0};
    int r = ppoll(&p, 1, &ts, nullptr);
    if (r > 0) return 1;
    if (r == 0) return 0;
    if (errno != EINTR) return -1;
  }
}

// 0 ok, OFL_ETIMEDOUT, OFL_ESYS (msg), OFL_EBROKEN (peer closed)
int read_full(int fd, void* buf, size_t len, int64_t deadline, std::string& msg) {
  uint8_t* p = static_cast<uint8_t*>(buf);
  while (len) {
    ssize_t r = ::recv(fd, p, len, MSG_DONTWAIT);
    if (r > 0) { p += r; len -= (size_t)r; continue; }
    if (r == 0) { msg = "peer closed the connection"; return OFL_EBROKEN; }
    if (errno == EINTR) continue;
    if (errno != EAGAIN && errno != EWOULDBLOCK) { msg = fmt("recv: %s", strerror(errno)); return OFL_ESYS; }
    int w = wait_fd(fd, POLLIN, deadline);
    if (w == 0) { msg = "timed out reading"; return OFL_ETIMEDOUT; }
    if (w < 0) { msg = fmt("poll: %s", strerror(errno)); return OFL_ESYS; }
  }
  return 0;
}

int write_iov(int fd, iovec* iov, int n, int64_t deadline, std::string& msg) {
  while (n > 0) {
    msghdr mh{};
    mh.msg_iov = iov;
    mh.msg_iovlen = (size_t)n;
    ssize_t r = ::sendmsg(fd, &mh, MSG_DONTWAIT | MSG_NOSIGNAL);
    if (r < 0) {
      if (errno == EINTR) continue;
      if (errno != EAGAIN && errno != EWOULDBLOCK) { msg = fmt("send: %s", strerror(errno)); return OFL_ESYS; }
      int w = wait_fd(fd, POLLOUT, deadline);
      if (w == 0) { msg = "timed out writing"; return OFL_ETIMEDOUT; }
      if (w < 0) { msg = fmt("poll: %s", strerror(errno)); return OFL_ESYS; }
      continue;
    }
    size_t done = (size_t)r;
    while (n > 0 && done >= iov->iov_len) { done -= iov->iov_len; iov++; n--; }
    if (n > 0) { iov->iov_base = static_cast<uint8_t*>(iov->iov_base) + done; iov->iov_len -= done; }
  }
  return 0;
}

void tune_socket(int fd) {
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one));
  int big = 8 << 20;  // large socket buffers: the TCP backend moves multi-MB responses
  setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &big, sizeof(big));
  setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &big, sizeof(big));
}

int tcp_connect(const char* local_ip, const char* server_ip, uint16_t port, int64_t deadline, std::string& msg) {
  in_addr la{}, ra{};
  if (!parse_ipv4(local_ip, &la)) { msg = fmt("bad local_ip '%s'", local_ip ? local_ip : ""); return -1; }
  if (!parse_ipv4(server_ip, &ra)) { msg = fmt("bad server_ip '%s'", server_ip ? server_ip : ""); return -1; }
  int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (fd < 0) { msg = fmt("socket: %s", strerror(errno)); return -1; }
  sockaddr_in l{};
  l.sin_family = AF_INET;
  l.sin_addr = la;
  if (::bind(fd, reinterpret_cast<sockaddr*>(&l), sizeof(l)) != 0) {
    msg = fmt("bind %s: %s", local_ip, strerror(errno));
    ::close(fd);
    return -1;
  }
  tune_socket(fd);
  sockaddr_in r{};
  r.sin_family = AF_INET;
  r.sin_addr = ra;
  r.sin_port = htons(port);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&r), sizeof(r)) != 0 && errno != EINPROGRESS) {
    msg = fmt("connect %s:%u: %s", server_ip, port, strerror(errno));
    ::close(fd);
    return -1;
  }
  int w = wait_fd(fd, POLLOUT, deadline);
  int soerr = 0;
  socklen_t sl = sizeof(soerr);
  if (w <= 0 || getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &sl) != 0 || soerr != 0) {
    msg = fmt("connect %s:%u from %s: %s", server_ip, port, local_ip,
              w == 0 ? "timed out" : strerror(soerr ? soerr : errno));
    ::close(fd);
    return -1;
  }
  return fd;
}

// ---- verbs -------------------------------------------------------------------------------------------------------

bool gid_is_ipv4(const ibv_gid& g, const in_addr& a) {
  for (int i = 0; i < 10; i++)
    if (g.raw[i]) return false;
  return g.raw[10] == 0xff && g.raw[11] == 0xff && memcmp(&g.raw[12], &a, 4) == 0;
}

struct Dev {
  ibv_context* ctx = nullptr;
  ibv_pd* pd = nullptr;
  std::string name;
  int port = 1;
  int gid_index = -1;
  ibv_gid gid{};
  ibv_mtu mtu = IBV_MTU_1024;

  void close() {
    if (pd) ibv_dealloc_pd(pd);
    if (ctx) ibv_close_device(ctx);
    pd = nullptr;
    ctx = nullptr;
  }
};

// Find (device, gid index) whose RoCE v2 GID is the IPv4-mapped `ip` (live discovery); check explicit choices.
bool open_dev(const char* ip, const char* want_dev, int port, int want_gid, Dev& d, std::string& msg) {
  in_addr a{};
  if (!parse_ipv4(ip, &a)) { msg = fmt("bad IPv4 address '%s'", ip ? ip : ""); return false; }
  if (port <= 0) port = 1;
  int n = 0;
  ibv_device** list = ibv_get_device_list(&n);
  if (!list || n <= 0) { msg = "no RDMA devices"; if (list) ibv_free_device_list(list); return false; }
  std::string tried;
  for (int i = 0; i < n && !d.ctx; i++) {
    const char* name = ibv_get_device_name(list[i]);
    if (want_dev && *want_dev && strcmp(want_dev, name) != 0) continue;
    ibv_context* ctx = ibv_open_device(list[i]);
    if (!ctx) { tried += fmt(" %s(open failed)", name); continue; }
    ibv_port_attr pa{};
    if (ibv_query_port(ctx, (uint8_t)port, &pa) != 0) { ibv_close_device(ctx); tried += fmt(" %s(no port %d)", name, port); continue; }
    int found = -1;
    for (int g = 0; g < pa.gid_tbl_len && found < 0; g++) {
      if (want_gid >= 0 && g != want_gid) continue;
      ibv_gid_entry e{};
      if (ibv_query_gid_ex(ctx, (uint32_t)port, (uint32_t)g, &e, 0) != 0) continue;
      if (e.gid_type == IBV_GID_TYPE_ROCE_V2 && gid_is_ipv4(e.gid, a)) found = g;
    }
    if (found < 0) {
      tried += fmt(" %s", name);
      ibv_close_device(ctx);
      continue;
    }
    if (pa.state != IBV_PORT_ACTIVE) {
      msg = fmt("%s port %d holds %s but is not active (state %s)", name, port, ip, ibv_port_state_str(pa.state));
      ibv_close_device(ctx);
      ibv_free_device_list(list);
      return false;
    }
    d.ctx = ctx;
    d.name = name;
    d.port = port;
    d.gid_index = found;
    d.mtu = pa.active_mtu;
    ibv_query_gid(ctx, (uint8_t)port, found, &d.gid);
  }
  ibv_free_device_list(list);
  if (!d.ctx) {
    msg = want_gid >= 0 ? fmt("no RoCE v2 GID index %d carries %s (checked:%s)", want_gid, ip, tried.c_str())
                        : fmt("no RDMA device has a RoCE v2 GID for %s (checked:%s)", ip, tried.c_str());
    return false;
  }
  d.pd = ibv_alloc_pd(d.ctx);
  if (!d.pd) { msg = fmt("ibv_alloc_pd on %s failed", d.name.c_str()); d.close(); return false; }
  return true;
}

ibv_qp* make_qp(Dev& d, ibv_cq* scq, ibv_cq* rcq, uint32_t send_wr, uint32_t recv_wr, uint32_t send_sge,
                uint32_t inline_bytes, uint32_t* inline_out, std::string& msg) {
  ibv_qp_init_attr at{};
  at.send_cq = scq;
  at.recv_cq = rcq;
  at.qp_type = IBV_QPT_RC;
  at.cap.max_send_wr = send_wr;
  at.cap.max_recv_wr = recv_wr;
  at.cap.max_send_sge = send_sge;
  at.cap.max_recv_sge = 1;
  at.cap.max_inline_data = inline_bytes;
  at.sq_sig_all = 0;
  ibv_qp* qp = ibv_create_qp(d.pd, &at);
  if (!qp) { msg = fmt("ibv_create_qp on %s: %s", d.name.c_str(), strerror(errno)); return nullptr; }
  if (inline_out) *inline_out = at.cap.max_inline_data;
  ibv_qp_attr a{};
  a.qp_state = IBV_QPS_INIT;
  a.pkey_index = 0;
  a.port_num = (uint8_t)d.port;
  a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
  if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
    msg = fmt("QP to INIT: %s", strerror(errno));
    ibv_destroy_qp(qp);
    return nullptr;
  }
  return qp;
}

bool qp_connect(Dev& d, ibv_qp* qp, uint32_t my_psn, const Hello& peer, std::string& msg) {
  ibv_qp_attr r{};
  r.qp_state = IBV_QPS_RTR;
  r.path_mtu = std::min<ibv_mtu>(d.mtu, static_cast<ibv_mtu>(peer.mtu));
  r.dest_qp_num = peer.qpn;
  r.rq_psn = peer.psn;
  r.max_dest_rd_atomic = 1;
  r.min_rnr_timer = 12;
  r.ah_attr.is_global = 1;
  memcpy(r.ah_attr.grh.dgid.raw, peer.gid, 16);
  r.ah_attr.grh.sgid_index = (uint8_t)d.gid_index;
  r.ah_attr.grh.hop_limit = 64;
  r.ah_attr.port_num = (uint8_t)d.port;
  if (ibv_modify_qp(qp, &r, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
    msg = fmt("QP to RTR: %s", strerror(errno));
    return false;
  }
  ibv_qp_attr s{};
  s.qp_state = IBV_QPS_RTS;
  s.timeout = 14;
  s.retry_cnt = 7;
  s.rnr_retry = 7;
  s.sq_psn = my_psn;
  s.max_rd_atomic = 1;
  if (ibv_modify_qp(qp, &s, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                                IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
    msg = fmt("QP to RTS: %s", strerror(errno));
    return false;
  }
  return true;
}

bool qp_error(ibv_qp* qp) {  // from here on the NIC drops everything addressed to this QP; false if that failed
  if (!qp) return true;
  ibv_qp_attr a{};
  a.qp_state = IBV_QPS_ERR;
  return ibv_modify_qp(qp, &a, IBV_QP_STATE) == 0;
}

uint32_t rand_psn() {
  static std::mt19937 g{std::random_device{}()};
  static std::mutex mu;
  std::lock_guard<std::mutex> l(mu);
  return g() & 0xffffffu;
}

void* alloc_aligned(size_t n) {
  void* p = nullptr;
  if (posix_memalign(&p, 4096, std::max<size_t>(n, 4096)) != 0) return nullptr;
  memset(p, 0, std::max<size_t>(n, 4096));
  return p;
}

const char* mtu_str(ibv_mtu m) {
  switch (m) {
    case IBV_MTU_256: return "256";
    case IBV_MTU_512: return "512";
    case IBV_MTU_1024: return "1024";
    case IBV_MTU_2048: return "2048";
    case IBV_MTU_4096: return "4096";
    default: return "?";
  }
}

void fill_hdr(ofl_hdr& h, uint16_t kind) {
  memset(&h, 0, sizeof(h));
  h.magic = OFL_MAGIC;
  h.version = OFL_VERSION;
  h.kind = kind;
}

bool hdr_ok(const ofl_hdr& h, uint16_t kind) { return h.magic == OFL_MAGIC && h.version == OFL_VERSION && h.kind == kind; }

}  // namespace

// =================================================================================================================
// client

struct ofl_client {
  // config (owned copies)
  int backend = 0;
  std::string server_ip, local_ip, rdma_device, service;
  uint16_t server_port = 0;
  int rdma_port = 1, gid_index = -1;
  uint32_t want_req = 0, want_resp = 0, want_inflight = 1;
  int64_t connect_timeout_ns = 5000000000LL;

  // connection
  int fd = -1;
  bool broken = true;
  std::string err, desc;
  uint64_t conn_gen = 0;
  uint32_t max_req = 0, max_resp = 0, inflight = 1;

  struct Region { uint8_t* addr; size_t len; ibv_mr* mr; };
  std::vector<Region> regions;

  enum { FREE = 0, PENDING = 1, DONE = 2 };
  struct Slot {
    int state = FREE;
    uint64_t req_id = 0;
    bool sent = false, answered = false;
    int rc = 0;
    int64_t t0 = 0;
    uint32_t resp_max = 0;
    std::vector<ofl_seg> segs;
    ofl_result res{};
  };
  std::vector<Slot> slots;
  uint64_t next_id = 1;   // next request id
  uint64_t done_upto = 0; // every id <= this has completed (or failed)
  uint32_t idle_polls = 0;

  // TCP payload staging (ofl_client_payload_buf)
  std::vector<uint8_t> tcp_payload;

  // RDMA
  Dev dev;
  ibv_cq* cq = nullptr;
  ibv_qp* qp = nullptr;
  uint32_t psn = 0;
  uint8_t* send_mem = nullptr;
  size_t send_slot = 0;
  ibv_mr* send_mr = nullptr;
  uint8_t* rhdr_mem = nullptr;
  ibv_mr* rhdr_mr = nullptr;
  uint32_t recv_posted = 0;

  int fail(int code, const std::string& m) {
    err = m;
    return code;
  }
  // Break the connection: QP to error first (no late write can land), then drop everything.
  void break_conn(const std::string& why) {
    if (!err.empty() && broken) return;
    err = why;
    broken = true;
    if (!qp_error(qp) && qp) {  // could not stop the QP: destroy it (also ends delivery into our memory)
      ibv_destroy_qp(qp);
      qp = nullptr;
    }
    if (fd >= 0) { ::close(fd); fd = -1; }
    for (auto& s : slots)
      if (s.state == PENDING) { s.state = DONE; s.rc = OFL_EBROKEN; }
  }
  void teardown() {
    qp_error(qp);
    if (qp) ibv_destroy_qp(qp);
    qp = nullptr;
    if (cq) ibv_destroy_cq(cq);
    cq = nullptr;
    for (auto& r : regions)
      if (r.mr) { ibv_dereg_mr(r.mr); r.mr = nullptr; }
    if (send_mr) ibv_dereg_mr(send_mr);
    if (rhdr_mr) ibv_dereg_mr(rhdr_mr);
    send_mr = rhdr_mr = nullptr;
    // send_mem / rhdr_mem stay allocated for the client's lifetime (stable ofl_client_payload_buf across
    // reconnects); ofl_client_close frees them
    dev.close();
    if (fd >= 0) ::close(fd);
    fd = -1;
    broken = true;
  }
  int connect_now(std::string& m);
  int progress(int64_t deadline, uint64_t want);
};

static bool post_client_recvs(ofl_client* c, uint32_t n) {
  for (uint32_t i = 0; i < n; i++) {
    ibv_recv_wr wr{}, *bad = nullptr;
    wr.wr_id = 0;
    if (ibv_post_recv(c->qp, &wr, &bad) != 0) return false;
    c->recv_posted++;
  }
  return true;
}

int ofl_client::connect_now(std::string& m) {
  const int64_t deadline = now_ns() + connect_timeout_ns;
  err.clear();
  fd = tcp_connect(local_ip.c_str(), server_ip.c_str(), server_port, deadline, m);
  if (fd < 0) return OFL_ESYS;
  Hello h{};
  h.backend = (uint32_t)backend;
  h.max_req_bytes = want_req;
  h.max_resp_bytes = want_resp;
  h.max_inflight = want_inflight;
  snprintf(h.service, sizeof(h.service), "%s", service.c_str());
  gethostname(h.text, sizeof(h.text) - 1);
  if (backend == OFL_BACKEND_RDMA) {
    if (!open_dev(local_ip.c_str(), rdma_device.c_str(), rdma_port, gid_index, dev, m)) return OFL_ERDMA;
    cq = ibv_create_cq(dev.ctx, (int)(4 * want_inflight + 16), nullptr, nullptr, 0);
    if (!cq) { m = "ibv_create_cq failed"; return OFL_ERDMA; }
    qp = make_qp(dev, cq, cq, 2 * want_inflight + 8, 2 * want_inflight + 8, 2, 0, nullptr, m);
    if (!qp) return OFL_ERDMA;
    psn = rand_psn();
    send_slot = (size_t)kSegArea + want_req;
    send_slot = (send_slot + 63) & ~size_t(63);
    if (!send_mem) send_mem = static_cast<uint8_t*>(alloc_aligned(send_slot * want_inflight));  // page-aligned
    if (!rhdr_mem) rhdr_mem = static_cast<uint8_t*>(alloc_aligned(64 * (size_t)want_inflight));
    if (!send_mem || !rhdr_mem) { m = "out of memory for RDMA slots"; return OFL_ENOMEM; }
    send_mr = ibv_reg_mr(dev.pd, send_mem, send_slot * want_inflight, IBV_ACCESS_LOCAL_WRITE);
    rhdr_mr = ibv_reg_mr(dev.pd, rhdr_mem, 64 * (size_t)want_inflight, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!send_mr || !rhdr_mr) { m = fmt("ibv_reg_mr of the request slots failed: %s", strerror(errno)); return OFL_ERDMA; }
    for (auto& r : regions) {
      r.mr = ibv_reg_mr(dev.pd, r.addr, r.len, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
      if (!r.mr) { m = fmt("ibv_reg_mr of a %zu-byte region failed: %s", r.len, strerror(errno)); return OFL_ERDMA; }
    }
    recv_posted = 0;
    if (!post_client_recvs(this, 2 * want_inflight + 4)) { m = "ibv_post_recv failed"; return OFL_ERDMA; }
    h.qpn = qp->qp_num;
    h.psn = psn;
    memcpy(h.gid, dev.gid.raw, 16);
    h.mtu = dev.mtu;
  }
  ofl_hdr hd;
  fill_hdr(hd, OFL_K_HELLO);
  hd.payload_len = sizeof(Hello);
  iovec v[2] = {{&hd, sizeof(hd)}, {&h, sizeof(h)}};
  int rc = write_iov(fd, v, 2, deadline, m);
  if (rc) return rc;
  ofl_hdr ah;
  Hello ack{};
  if ((rc = read_full(fd, &ah, sizeof(ah), deadline, m))) return rc;
  if (!hdr_ok(ah, OFL_K_HELLO_ACK) || ah.payload_len != sizeof(Hello)) { m = "bad handshake reply"; return OFL_EPROTO; }
  if ((rc = read_full(fd, &ack, sizeof(ack), deadline, m))) return rc;
  ack.text[sizeof(ack.text) - 1] = 0;
  if (ack.status != 0) { m = fmt("server refused: %s", ack.text); return OFL_EREJECT; }
  if (ack.max_req_bytes < want_req || ack.max_resp_bytes < want_resp || ack.max_inflight < 1 ||
      ack.max_inflight > want_inflight) {
    m = fmt("server granted req %u resp %u inflight %u for req %u resp %u inflight %u", ack.max_req_bytes,
            ack.max_resp_bytes, ack.max_inflight, want_req, want_resp, want_inflight);
    return OFL_EREJECT;
  }
  max_req = want_req;
  max_resp = want_resp;
  inflight = ack.max_inflight;
  if (backend == OFL_BACKEND_RDMA) {
    if (!qp_connect(dev, qp, psn, ack, m)) return OFL_ERDMA;
    desc = fmt("rdma %s port %d gid %d mtu %s qpn %u -> %s:%u qpn %u (inflight %u)", dev.name.c_str(), dev.port,
               dev.gid_index, mtu_str(std::min<ibv_mtu>(dev.mtu, (ibv_mtu)ack.mtu)), qp->qp_num, server_ip.c_str(),
               server_port, ack.qpn, inflight);
  } else {
    desc = fmt("tcp %s -> %s:%u (inflight %u)", local_ip.c_str(), server_ip.c_str(), server_port, inflight);
    if (tcp_payload.size() < want_req) tcp_payload.assign(want_req, 0);  // kept across reconnects
  }
  slots.assign(want_inflight, Slot{});
  done_upto = next_id - 1;
  conn_gen++;
  broken = false;
  return OFL_OK;
}

extern "C" {

ofl_client* ofl_client_connect(const ofl_client_config* cfg, char* err, size_t errlen) {
  if (!cfg || (cfg->backend != OFL_BACKEND_TCP && cfg->backend != OFL_BACKEND_RDMA) || !cfg->server_ip ||
      !cfg->local_ip || !cfg->server_port) {
    copy_err(err, errlen, "ofl_client_connect: backend, server_ip, server_port and local_ip are required");
    return nullptr;
  }
  if (cfg->max_inflight > 64) { copy_err(err, errlen, "ofl_client_connect: max_inflight above 64"); return nullptr; }
  auto* c = new ofl_client();
  c->backend = cfg->backend;
  c->server_ip = cfg->server_ip;
  c->local_ip = cfg->local_ip;
  c->rdma_device = cfg->rdma_device ? cfg->rdma_device : "";
  c->service = cfg->service ? cfg->service : "";
  c->server_port = cfg->server_port;
  c->rdma_port = cfg->rdma_port > 0 ? cfg->rdma_port : 1;
  c->gid_index = cfg->gid_index;
  c->want_req = cfg->max_req_bytes;
  c->want_resp = cfg->max_resp_bytes;
  c->want_inflight = cfg->max_inflight ? cfg->max_inflight : 1;
  if (cfg->connect_timeout_ms > 0) c->connect_timeout_ns = cfg->connect_timeout_ms * 1000000LL;
  std::string m;
  int rc = c->connect_now(m);
  if (rc != OFL_OK) {
    copy_err(err, errlen, m);
    c->teardown();
    free(c->send_mem);
    free(c->rhdr_mem);
    delete c;
    return nullptr;
  }
  return c;
}

int ofl_client_reconnect(ofl_client* c, char* err, size_t errlen) {
  if (!c) return OFL_EINVAL;
  c->teardown();
  std::string m;
  int rc = c->connect_now(m);
  if (rc != OFL_OK) {
    c->err = m;
    copy_err(err, errlen, m);
    c->teardown();
  }
  return rc;
}

int ofl_client_reg(ofl_client* c, void* addr, size_t len, uint32_t* region) {
  if (!c || !addr || !len || !region) return OFL_EINVAL;
  ofl_client::Region r{static_cast<uint8_t*>(addr), len, nullptr};
  if (c->backend == OFL_BACKEND_RDMA && !c->broken) {
    r.mr = ibv_reg_mr(c->dev.pd, addr, len, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!r.mr) return c->fail(OFL_ERDMA, fmt("ibv_reg_mr of %zu bytes at %p: %s", len, addr, strerror(errno)));
  }
  c->regions.push_back(r);
  *region = (uint32_t)(c->regions.size() - 1);
  return OFL_OK;
}

void* ofl_client_payload_buf(ofl_client* c, uint32_t* capacity) {
  if (!c || c->broken) return nullptr;
  if (capacity) *capacity = c->max_req;
  if (c->backend == OFL_BACKEND_TCP) return c->tcp_payload.data();
  const uint32_t slot = (uint32_t)((c->next_id - 1) % c->inflight);
  return c->send_mem + (size_t)slot * c->send_slot + kSegArea;
}

int64_t ofl_client_submit(ofl_client* c, uint16_t op, uint64_t generation, const void* payload, uint32_t payload_len,
                          const ofl_seg* segs, uint32_t nseg) {
  if (!c) return OFL_EINVAL;
  if (c->broken) return c->fail(OFL_EBROKEN, c->err.empty() ? "not connected" : c->err);
  if (payload_len > c->max_req) return c->fail(OFL_ETOOBIG, fmt("payload %u > max_req %u", payload_len, c->max_req));
  if (nseg > OFL_MAX_SEGS || (nseg && !segs)) return c->fail(OFL_EINVAL, "too many destination ranges");
  uint64_t cap = 0;
  for (uint32_t i = 0; i < nseg; i++) {
    if (segs[i].region >= c->regions.size()) return c->fail(OFL_EINVAL, fmt("unknown region %u", segs[i].region));
    const auto& r = c->regions[segs[i].region];
    if (segs[i].offset > r.len || segs[i].len > r.len - segs[i].offset)
      return c->fail(OFL_EINVAL, fmt("range %u outside region %u", i, segs[i].region));
    cap += segs[i].len;
  }
  if (cap > c->max_resp) return c->fail(OFL_ETOOBIG, fmt("destination %llu > max_resp %u", (unsigned long long)cap, c->max_resp));
  const uint64_t id = c->next_id;
  const uint32_t si = (uint32_t)((id - 1) % c->inflight);
  auto& s = c->slots[si];
  if (s.state != ofl_client::FREE) return c->fail(OFL_EBUSY, "too many requests in flight (wait first)");
  s = ofl_client::Slot{};
  s.state = ofl_client::PENDING;
  s.req_id = id;
  s.resp_max = (uint32_t)cap;
  s.segs.assign(segs, segs + nseg);
  s.t0 = now_ns();
  ofl_hdr h;
  fill_hdr(h, OFL_K_REQ);
  h.op = op;
  h.req_id = id;
  h.generation = generation;
  h.payload_len = payload_len;
  h.resp_max = (uint32_t)cap;
  std::string m;
  if (c->backend == OFL_BACKEND_TCP) {
    iovec v[2] = {{&h, sizeof(h)}, {const_cast<void*>(payload), payload_len}};
    int rc = write_iov(c->fd, v, payload_len ? 2 : 1, now_ns() + 10000000000LL, m);
    if (rc) { c->break_conn("send request: " + m); s.state = ofl_client::FREE; return rc == OFL_ETIMEDOUT ? OFL_ETIMEDOUT : OFL_EBROKEN; }
    c->next_id++;
    return (int64_t)id;
  }
  uint8_t* base = c->send_mem + (size_t)si * c->send_slot;
  h.nseg = (uint16_t)nseg;
  h.rhdr_addr = (uint64_t)(uintptr_t)(c->rhdr_mem + 64 * (size_t)si);
  h.rhdr_rkey = c->rhdr_mr->rkey;
  memset(c->rhdr_mem + 64 * (size_t)si, 0, 64);
  memcpy(base, &h, sizeof(h));
  auto* ws = reinterpret_cast<ofl_wire_seg*>(base + 64);
  for (uint32_t i = 0; i < nseg; i++) {
    const auto& r = c->regions[segs[i].region];
    ws[i].addr = (uint64_t)(uintptr_t)(r.addr + segs[i].offset);
    ws[i].rkey = r.mr->rkey;
    ws[i].len = (uint32_t)segs[i].len;
  }
  uint8_t* pl = base + kSegArea;
  if (payload_len && payload != pl) memcpy(pl, payload, payload_len);
  ibv_sge sge[2] = {{(uint64_t)(uintptr_t)base, 64 + 16 * nseg, c->send_mr->lkey},
                    {(uint64_t)(uintptr_t)pl, payload_len, c->send_mr->lkey}};
  ibv_send_wr wr{}, *bad = nullptr;
  wr.wr_id = kWrSend | id;
  wr.sg_list = sge;
  wr.num_sge = payload_len ? 2 : 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;
  if (ibv_post_send(c->qp, &wr, &bad) != 0) {
    s.state = ofl_client::FREE;
    c->break_conn(fmt("ibv_post_send: %s", strerror(errno)));
    return OFL_ERDMA;
  }
  c->next_id++;
  return (int64_t)id;
}

}  // extern "C"

// Drive completions until request `want` is done or the deadline passes. 0, or an error (connection broken).
int ofl_client::progress(int64_t deadline, uint64_t want) {
  std::string m;
  while (done_upto < want) {
    const uint64_t id = done_upto + 1;
    Slot& s = slots[(id - 1) % inflight];
    if (s.state == DONE) { done_upto++; continue; }
    if (s.state != PENDING || s.req_id != id) { break_conn("internal: slot out of order"); return OFL_EPROTO; }
    if (backend == OFL_BACKEND_TCP) {
      ofl_hdr h;
      int rc = read_full(fd, &h, sizeof(h), deadline, m);
      if (rc) { break_conn("read response: " + m); return rc == OFL_ETIMEDOUT ? OFL_ETIMEDOUT : rc; }
      if (!hdr_ok(h, OFL_K_RESP) || h.req_id != id) {
        break_conn(fmt("bad response header (req %llu)", (unsigned long long)id));
        return OFL_EPROTO;
      }
      if (h.payload_len > s.resp_max) { break_conn("response larger than its destination"); return OFL_EPROTO; }
      uint32_t left = h.payload_len;
      for (const auto& g : s.segs) {
        if (!left) break;
        const uint32_t n = (uint32_t)std::min<uint64_t>(g.len, left);
        rc = read_full(fd, regions[g.region].addr + g.offset, n, deadline, m);
        if (rc) { break_conn("read response body: " + m); return rc == OFL_ETIMEDOUT ? OFL_ETIMEDOUT : rc; }
        left -= n;
      }
      s.res.status = h.status;
      s.res.resp_len = h.payload_len;
      s.res.req_id = id;
      s.res.generation = h.generation;
      s.res.server_ns = (int64_t)h.t_server_ns;
      s.res.elapsed_ns = now_ns() - s.t0;
      s.rc = h.status < 0 ? OFL_EREMOTE : OFL_OK;
      s.state = DONE;
      done_upto++;
      continue;
    }
    // RDMA: poll; every completion applies to some pending slot
    ibv_wc wc[16];
    const int n = ibv_poll_cq(cq, 16, wc);
    if (n < 0) { break_conn("ibv_poll_cq failed"); return OFL_ERDMA; }
    if (n == 0) {
      if (now_ns() > deadline) {
        break_conn(fmt("timed out waiting for response %llu", (unsigned long long)id));
        return OFL_ETIMEDOUT;
      }
      // the server never writes to the bootstrap socket after the handshake: readable = it closed (died / restarted),
      // so fail now instead of at the deadline (checked every 1024 empty polls, ~50-100 us)
      if ((++idle_polls & 1023) == 0) {
        pollfd p{fd, POLLIN, 0};
        if (fd >= 0 && poll(&p, 1, 0) > 0) {
          break_conn(fmt("server closed the connection while request %llu was pending", (unsigned long long)id));
          return OFL_EBROKEN;
        }
      }
      cpu_relax();
      continue;
    }
    for (int i = 0; i < n; i++) {
      if (wc[i].status != IBV_WC_SUCCESS) {
        break_conn(fmt("work completion: %s (opcode %d)", ibv_wc_status_str(wc[i].status), (int)wc[i].opcode));
        return OFL_ERDMA;
      }
      if (wc[i].opcode == IBV_WC_SEND) {
        const uint64_t rid = wc[i].wr_id & ~kWrSend;
        Slot& t = slots[(rid - 1) % inflight];
        if (t.req_id == rid) t.sent = true;
      } else if (wc[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
        recv_posted--;
        if (!post_client_recvs(this, 1)) { break_conn("ibv_post_recv failed"); return OFL_ERDMA; }
        const uint32_t imm = ntohl(wc[i].imm_data);
        // responses come in request order; find the pending slot whose id matches the immediate
        Slot* t = nullptr;
        for (auto& x : slots)
          if (x.state == PENDING && (uint32_t)x.req_id == imm) t = &x;
        if (!t) { break_conn(fmt("response for an unknown request (imm %u)", imm)); return OFL_EPROTO; }
        const size_t si = (size_t)(t - slots.data());
        ofl_hdr h;
        memcpy(&h, rhdr_mem + 64 * si, sizeof(h));
        if (!hdr_ok(h, OFL_K_RESP) || h.req_id != t->req_id || h.payload_len > t->resp_max) {
          break_conn(fmt("bad response header for request %llu", (unsigned long long)t->req_id));
          return OFL_EPROTO;
        }
        t->answered = true;
        t->res.status = h.status;
        t->res.resp_len = h.payload_len;
        t->res.req_id = t->req_id;
        t->res.generation = h.generation;
        t->res.server_ns = (int64_t)h.t_server_ns;
        t->res.elapsed_ns = now_ns() - t->t0;
        t->rc = h.status < 0 ? OFL_EREMOTE : OFL_OK;
      } else {
        break_conn(fmt("unexpected completion opcode %d", (int)wc[i].opcode));
        return OFL_EPROTO;
      }
    }
    // a slot is complete once its request buffer is free again and its response is in
    for (auto& x : slots)
      if (x.state == PENDING && x.sent && x.answered) x.state = DONE;
  }
  return OFL_OK;
}

extern "C" {

int ofl_client_wait(ofl_client* c, int64_t ticket, ofl_result* res, int64_t timeout_ns) {
  if (!c || ticket <= 0) return OFL_EINVAL;
  const uint64_t id = (uint64_t)ticket;
  if (id >= c->next_id) return c->fail(OFL_EINVAL, "unknown ticket");
  auto& s = c->slots[(id - 1) % c->inflight];
  if (s.req_id != id || s.state == ofl_client::FREE) return c->fail(OFL_EINVAL, "ticket already waited for");
  if (s.state == ofl_client::PENDING) {
    if (c->broken) { s.state = ofl_client::FREE; return OFL_EBROKEN; }
    int rc = c->progress(now_ns() + timeout_ns, id);
    if (rc != OFL_OK) {
      if (s.req_id == id) s.state = ofl_client::FREE;
      return rc;
    }
  }
  if (res) *res = s.res;
  const int rc = s.rc;
  s.state = ofl_client::FREE;
  if (rc == OFL_EREMOTE) c->err = fmt("server returned status %d", s.res.status);
  return rc;
}

int ofl_client_call(ofl_client* c, uint16_t op, uint64_t generation, const void* payload, uint32_t payload_len,
                    const ofl_seg* segs, uint32_t nseg, ofl_result* res, int64_t timeout_ns) {
  const int64_t t = ofl_client_submit(c, op, generation, payload, payload_len, segs, nseg);
  if (t < 0) return (int)t;
  return ofl_client_wait(c, t, res, timeout_ns);
}

int ofl_client_broken(const ofl_client* c) { return !c || c->broken; }
const char* ofl_client_error(const ofl_client* c) { return c ? c->err.c_str() : "null client"; }
const char* ofl_client_describe(const ofl_client* c) { return c ? c->desc.c_str() : ""; }

void ofl_client_close(ofl_client* c) {
  if (!c) return;
  c->teardown();
  free(c->send_mem);
  free(c->rhdr_mem);
  delete c;
}

}  // extern "C"

// =================================================================================================================
// server

struct ofl_server;

struct Conn {
  ofl_server* s = nullptr;
  uint64_t id = 0;
  int fd = -1;
  int backend = 0;
  std::thread thr;
  std::atomic<bool> finished{false};
  std::string peer;
  uint32_t max_req = 0, max_resp = 0, inflight = 1;
  // TCP
  std::vector<uint8_t> req_buf, resp_buf;
  // RDMA
  ibv_comp_channel* ch = nullptr;
  ibv_cq* scq = nullptr;
  ibv_cq* rcq = nullptr;
  ibv_qp* qp = nullptr;
  uint32_t psn = 0;
  uint8_t* recv_mem = nullptr;
  size_t recv_slot = 0;
  ibv_mr* recv_mr = nullptr;
  uint8_t* resp_mem = nullptr;
  ibv_mr* resp_mr = nullptr;
  std::vector<int> resp_free;
  std::mutex post_mu;    // posting on the SQ and polling the send CQ
  uint64_t posted = 0;   // send WRs posted
  uint64_t retired = 0;  // send WRs known complete
  uint32_t sq_depth = 0;
  uint32_t max_inline = 0;
  bool sq_failed = false;
  std::string fail_why;
  ~Conn();
  void main();
  bool handshake();
  void run_tcp();
  void run_rdma();
  bool handle_recv(const ibv_wc& w);  // false = the connection must end
  bool drain_send(bool block);        // false = a send completion failed
};

struct ofl_req {
  Conn* c = nullptr;
  ofl_hdr hdr{};
  const ofl_wire_seg* segs = nullptr;
  uint32_t nseg = 0;
  const uint8_t* payload = nullptr;
  uint32_t payload_len = 0;
  uint8_t* resp = nullptr;
  uint32_t cap = 0, len = 0;
  int buf = -1;  // RDMA response buffer index
  std::mutex mu;
  std::vector<std::pair<uint32_t, uint32_t>> sent;  // ranges already posted
  int err = 0;
};

struct ofl_server {
  std::string bind_ip, rdma_device, service, desc;
  uint16_t port = 0;
  int backends = 0, rdma_port = 1, gid_index = -1;
  uint32_t max_req = 0, max_resp = 0, max_inflight = 2, max_conns = 8;
  int64_t spin_ns = 2000000000LL;
  ofl_handler_fn fn = nullptr;
  void* user = nullptr;
  int lfd = -1;
  Dev dev;  // shared device + PD (RDMA)
  bool have_dev = false;
  std::atomic<bool> running{false};
  std::thread acceptor;
  std::mutex mu;
  std::vector<std::unique_ptr<Conn>> conns;
  uint64_t next_conn = 1;
  std::atomic<uint64_t> n_conns{0}, n_reqs{0}, n_bytes{0}, n_errs{0};
  void accept_loop();
  void handshake(int fd);
};

Conn::~Conn() {
  if (thr.joinable()) thr.join();
  qp_error(qp);
  if (qp) ibv_destroy_qp(qp);
  if (scq) ibv_destroy_cq(scq);
  if (rcq) ibv_destroy_cq(rcq);
  if (ch) ibv_destroy_comp_channel(ch);
  if (recv_mr) ibv_dereg_mr(recv_mr);
  if (resp_mr) ibv_dereg_mr(resp_mr);
  free(recv_mem);
  free(resp_mem);
  if (fd >= 0) ::close(fd);
}

namespace {

// Map response bytes [off, off + len) onto the request's destination ranges and post one RDMA WRITE per piece.
// Caller holds c->post_mu.
int post_range_locked(ofl_req* r, uint32_t off, uint32_t len) {
  Conn* c = r->c;
  uint64_t pos = 0;
  for (uint32_t i = 0; i < r->nseg && len; i++) {
    const uint64_t s0 = pos, s1 = pos + r->segs[i].len;
    pos = s1;
    if (off >= s1 || off + (uint64_t)len <= s0) continue;
    const uint64_t a = std::max<uint64_t>(off, s0), b = std::min<uint64_t>(off + (uint64_t)len, s1);
    while (c->posted - c->retired + 2 > c->sq_depth)
      if (!c->drain_send(true)) return OFL_ERDMA;
    ibv_sge sge{(uint64_t)(uintptr_t)(r->resp + a), (uint32_t)(b - a), c->resp_mr->lkey};
    ibv_send_wr wr{}, *bad = nullptr;
    const uint64_t seq = ++c->posted;
    wr.wr_id = kWrMark | seq;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = (seq % kSignalEvery == 0) ? IBV_SEND_SIGNALED : 0;
    wr.wr.rdma.remote_addr = r->segs[i].addr + (a - s0);
    wr.wr.rdma.rkey = r->segs[i].rkey;
    if (ibv_post_send(c->qp, &wr, &bad) != 0) {
      c->posted--;
      return OFL_ERDMA;
    }
  }
  return OFL_OK;
}

}  // namespace

bool Conn::drain_send(bool block) {
  ibv_wc wc[32];
  for (;;) {
    const int n = ibv_poll_cq(scq, 32, wc);
    if (n < 0) { sq_failed = true; fail_why = "ibv_poll_cq (send) failed"; return false; }
    for (int i = 0; i < n; i++) {
      if (wc[i].status != IBV_WC_SUCCESS) {
        sq_failed = true;
        fail_why = fmt("send completion: %s", ibv_wc_status_str(wc[i].status));
        return false;
      }
      const uint64_t w = wc[i].wr_id;
      if (w & kWrHdr) {
        const uint64_t seq = (w >> 16) & 0xffffffffffull;
        resp_free.push_back((int)(w & 0xffff));
        retired = std::max(retired, seq);
      } else if (w & kWrMark) {
        retired = std::max(retired, w & ~kWrMark);
      }
    }
    if (n > 0 || !block) return true;
    if (!s->running.load(std::memory_order_relaxed)) { fail_why = "server stopping"; return false; }
    cpu_relax();
  }
}


static int32_t run_handler(ofl_server* s, ofl_req* r) {
  if (r->hdr.op == OFL_OP_ECHO) {
    const uint32_t n = std::min(r->payload_len, r->cap);
    if (n) memcpy(r->resp, r->payload, n);
    r->len = n;
    return 0;
  }
  return s->fn ? s->fn(s->user, r) : -1;
}

void Conn::run_tcp() {
  std::string m;
  while (s->running.load(std::memory_order_relaxed)) {
    const int w = wait_fd(fd, POLLIN, now_ns() + 100000000LL);
    if (w == 0) continue;
    if (w < 0) break;
    ofl_hdr h;
    const int64_t dl = now_ns() + 10000000000LL;
    if (read_full(fd, &h, sizeof(h), dl, m)) break;
    const int64_t t0 = now_ns();
    if (!hdr_ok(h, OFL_K_REQ) || h.payload_len > max_req) { s->n_errs++; break; }
    if (read_full(fd, req_buf.data(), h.payload_len, dl, m)) break;
    ofl_req r;
    r.c = this;
    r.hdr = h;
    r.payload = req_buf.data();
    r.payload_len = h.payload_len;
    r.resp = resp_buf.data();
    r.cap = std::min<uint32_t>(h.resp_max, max_resp);
    int32_t st = run_handler(s, &r);
    if (r.len > r.cap) st = OFL_ETOOBIG;
    ofl_hdr o;
    fill_hdr(o, OFL_K_RESP);
    o.op = h.op;
    o.req_id = h.req_id;
    o.generation = h.generation;
    o.status = st;
    o.payload_len = st < 0 ? 0 : r.len;
    o.t_server_ns = (uint64_t)(now_ns() - t0);
    iovec v[2] = {{&o, sizeof(o)}, {resp_buf.data(), o.payload_len}};
    if (write_iov(fd, v, o.payload_len ? 2 : 1, now_ns() + 10000000000LL, m)) break;
    s->n_reqs++;
    s->n_bytes += o.payload_len;
    if (st < 0) s->n_errs++;
  }
}

void Conn::run_rdma() {
  int64_t last = now_ns();
  uint32_t idle_iter = 0;
  ibv_wc wc[8];
  while (s->running.load(std::memory_order_relaxed) && !sq_failed) {
    int n = ibv_poll_cq(rcq, 8, wc);
    if (n < 0) { fail_why = "ibv_poll_cq (recv) failed"; break; }
    if (n == 0) {
      {
        std::lock_guard<std::mutex> g(post_mu);
        if (!drain_send(false)) break;
      }
      if (now_ns() - last < s->spin_ns) {
        if ((++idle_iter & 0xffff) == 0) {  // every few ms while spinning: is the client still there?
          pollfd p{fd, POLLIN, 0};
          if (poll(&p, 1, 0) > 0) { fail_why = "client closed the bootstrap socket"; break; }
        }
        cpu_relax();
        continue;
      }
      // idle: arm the completion channel, re-check (a completion may have raced the arm), then sleep on the channel
      // and the bootstrap socket (the client never writes to it after the handshake: readable = closed)
      if (ibv_req_notify_cq(rcq, 0) != 0) { fail_why = "ibv_req_notify_cq failed"; break; }
      n = ibv_poll_cq(rcq, 8, wc);
      if (n < 0) { fail_why = "ibv_poll_cq (recv) failed"; break; }
      if (n == 0) {
        pollfd p[2] = {{ch->fd, POLLIN, 0}, {fd, POLLIN, 0}};
        const int r = poll(p, 2, 100);
        if (r > 0 && (p[1].revents & (POLLIN | POLLHUP | POLLERR))) { fail_why = "client closed the bootstrap socket"; break; }
        if (r > 0 && (p[0].revents & POLLIN)) {
          ibv_cq* evcq = nullptr;
          void* evctx = nullptr;
          if (ibv_get_cq_event(ch, &evcq, &evctx) == 0) ibv_ack_cq_events(evcq, 1);
        }
        continue;
      }
    }
    bool ok = true;
    for (int i = 0; i < n && ok; i++) ok = handle_recv(wc[i]);
    if (!ok) break;
    last = now_ns();
    idle_iter = 0;
  }
  if (!fail_why.empty() && s->running.load() && fail_why != "client closed the bootstrap socket")
    fprintf(stderr, "ofl server: connection %llu (%s) ends: %s\n",
                                                      (unsigned long long)id, peer.c_str(), fail_why.c_str());
}

bool Conn::handle_recv(const ibv_wc& w) {
  if (w.status != IBV_WC_SUCCESS) { fail_why = fmt("recv completion: %s", ibv_wc_status_str(w.status)); return false; }
  if (w.opcode != IBV_WC_RECV) { fail_why = fmt("unexpected recv opcode %d", (int)w.opcode); return false; }
  const int64_t t0 = now_ns();
  const uint32_t k = (uint32_t)w.wr_id;
  if (k >= inflight) { fail_why = "bad receive slot"; return false; }
  uint8_t* base = recv_mem + (size_t)k * recv_slot;
  ofl_req r;
  memcpy(&r.hdr, base, sizeof(ofl_hdr));
  const ofl_hdr& h = r.hdr;
  if (!hdr_ok(h, OFL_K_REQ) || h.nseg > OFL_MAX_SEGS || h.payload_len > max_req ||
      w.byte_len != 64u + 16u * h.nseg + h.payload_len) {
    s->n_errs++;
    fail_why = fmt("malformed request (%u bytes)", w.byte_len);
    return false;
  }
  r.c = this;
  r.segs = reinterpret_cast<const ofl_wire_seg*>(base + 64);
  r.nseg = h.nseg;
  r.payload = base + 64 + 16 * h.nseg;
  r.payload_len = h.payload_len;
  uint64_t cap = 0;
  for (uint32_t i = 0; i < r.nseg; i++) cap += r.segs[i].len;
  if (cap != h.resp_max) { s->n_errs++; fail_why = "request ranges do not add up to resp_max"; return false; }
  {
    std::lock_guard<std::mutex> g(post_mu);
    while (resp_free.empty())
      if (!drain_send(true)) return false;
    r.buf = resp_free.back();
    resp_free.pop_back();
  }
  r.resp = resp_mem + (size_t)r.buf * ((size_t)max_resp + 64);
  r.cap = (uint32_t)std::min<uint64_t>(cap, max_resp);
  int32_t st = run_handler(s, &r);
  if (st >= 0 && r.len > r.cap) st = OFL_ETOOBIG;
  const uint32_t len = st < 0 ? 0 : r.len;
  std::lock_guard<std::mutex> g(post_mu);
  if (r.err) { fail_why = "posting response writes failed"; sq_failed = true; return false; }
  if (len) {  // everything the handler did not already send
    std::sort(r.sent.begin(), r.sent.end());
    uint32_t pos = 0;
    for (const auto& x : r.sent) {
      if (x.first > pos && post_range_locked(&r, pos, std::min(x.first, len) - pos)) { fail_why = "ibv_post_send failed"; return false; }
      pos = std::max(pos, x.first + x.second);
      if (pos >= len) break;
    }
    if (pos < len && post_range_locked(&r, pos, len - pos)) { fail_why = "ibv_post_send failed"; return false; }
  }
  // the receive slot is free again (the range table was read by the posts above; the header fields come from the
  // copy in r.hdr): repost it before the header, so the client's next request never meets an empty receive queue
  ibv_sge rs{(uint64_t)(uintptr_t)base, (uint32_t)recv_slot, recv_mr->lkey};
  ibv_recv_wr rw{}, *rbad = nullptr;
  rw.wr_id = k;
  rw.sg_list = &rs;
  rw.num_sge = 1;
  if (ibv_post_recv(qp, &rw, &rbad) != 0) { fail_why = "ibv_post_recv failed"; return false; }
  while (posted - retired + 2 > sq_depth)
    if (!drain_send(true)) return false;
  ofl_hdr* o = reinterpret_cast<ofl_hdr*>(r.resp + max_resp);  // the buffer's header area (registered)
  fill_hdr(*o, OFL_K_RESP);
  o->op = h.op;
  o->req_id = h.req_id;
  o->generation = h.generation;
  o->status = st;
  o->payload_len = len;
  o->t_server_ns = (uint64_t)(now_ns() - t0);
  ibv_sge sge{(uint64_t)(uintptr_t)o, 64, resp_mr->lkey};
  ibv_send_wr wr{}, *bad = nullptr;
  const uint64_t seq = ++posted;
  wr.wr_id = kWrHdr | (seq << 16) | (uint64_t)r.buf;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED | (max_inline >= 64 ? IBV_SEND_INLINE : 0);
  wr.imm_data = htonl((uint32_t)h.req_id);
  wr.wr.rdma.remote_addr = h.rhdr_addr;
  wr.wr.rdma.rkey = h.rhdr_rkey;
  if (ibv_post_send(qp, &wr, &bad) != 0) { posted--; fail_why = fmt("ibv_post_send (header): %s", strerror(errno)); return false; }
  s->n_reqs++;
  s->n_bytes += len;
  if (st < 0) s->n_errs++;
  return true;
}

bool Conn::handshake() {
  std::string m;
  const int64_t dl = now_ns() + 5000000000LL;
  ofl_hdr h;
  Hello hi{};
  if (read_full(fd, &h, sizeof(h), dl, m) || !hdr_ok(h, OFL_K_HELLO) || h.payload_len != sizeof(Hello) ||
      read_full(fd, &hi, sizeof(hi), dl, m)) {
    fail_why = "bad or missing handshake: " + m;
    return false;
  }
  hi.service[sizeof(hi.service) - 1] = 0;
  hi.text[sizeof(hi.text) - 1] = 0;
  peer = hi.text;
  Hello ack{};
  auto reply = [&](int32_t status, const std::string& why) {
    ack.status = status;
    snprintf(ack.text, sizeof(ack.text), "%s", why.c_str());
    snprintf(ack.service, sizeof(ack.service), "%s", s->service.c_str());
    ofl_hdr o;
    fill_hdr(o, OFL_K_HELLO_ACK);
    o.payload_len = sizeof(Hello);
    iovec v[2] = {{&o, sizeof(o)}, {&ack, sizeof(ack)}};
    std::string m2;
    return write_iov(fd, v, 2, dl, m2) == 0;
  };
  backend = (int)hi.backend;
  std::string why;
  if (s->service != hi.service) why = fmt("service '%s' asked, this is '%s'", hi.service, s->service.c_str());
  else if ((backend != OFL_BACKEND_TCP && backend != OFL_BACKEND_RDMA) || !(s->backends & (1 << backend)))
    why = fmt("backend %d not offered", backend);
  else if (hi.max_req_bytes > s->max_req || hi.max_resp_bytes > s->max_resp)
    why = fmt("limits req %u resp %u exceed the server's %u / %u", hi.max_req_bytes, hi.max_resp_bytes, s->max_req, s->max_resp);
  else if (backend == OFL_BACKEND_RDMA && !s->have_dev) why = "no RDMA device on this server";
  if (!why.empty()) {
    reply(OFL_EREJECT, why);
    fail_why = "refused " + peer + ": " + why;
    return false;
  }
  max_req = hi.max_req_bytes;
  max_resp = hi.max_resp_bytes;
  inflight = std::max<uint32_t>(1, std::min(hi.max_inflight ? hi.max_inflight : 1, s->max_inflight));
  if (backend == OFL_BACKEND_TCP) {
    req_buf.assign(std::max<uint32_t>(max_req, 1), 0);
    resp_buf.assign(std::max<uint32_t>(max_resp, 1), 0);
  } else {
    Dev& d = s->dev;
    sq_depth = 1024;
    ch = ibv_create_comp_channel(d.ctx);
    rcq = ch ? ibv_create_cq(d.ctx, (int)inflight + 8, nullptr, ch, 0) : nullptr;
    scq = ibv_create_cq(d.ctx, (int)sq_depth + 16, nullptr, nullptr, 0);
    if (!ch || !rcq || !scq) { reply(OFL_ERDMA, "completion queue setup failed"); fail_why = "cq setup failed"; return false; }
    fcntl(ch->fd, F_SETFL, fcntl(ch->fd, F_GETFL) | O_NONBLOCK);
    qp = make_qp(d, scq, rcq, sq_depth, inflight + 2, 1, 64, &max_inline, m);
    if (!qp) qp = make_qp(d, scq, rcq, sq_depth, inflight + 2, 1, 0, &max_inline, m);  // no inline support
    if (!qp) { reply(OFL_ERDMA, m); fail_why = m; return false; }
    recv_slot = ((size_t)kSegArea + max_req + 63) & ~size_t(63);
    recv_mem = static_cast<uint8_t*>(alloc_aligned(recv_slot * inflight));
    resp_mem = static_cast<uint8_t*>(alloc_aligned(((size_t)max_resp + 64) * inflight));
    if (!recv_mem || !resp_mem) { reply(OFL_ENOMEM, "out of memory"); fail_why = "out of memory"; return false; }
    recv_mr = ibv_reg_mr(d.pd, recv_mem, recv_slot * inflight, IBV_ACCESS_LOCAL_WRITE);
    resp_mr = ibv_reg_mr(d.pd, resp_mem, ((size_t)max_resp + 64) * inflight, IBV_ACCESS_LOCAL_WRITE);
    if (!recv_mr || !resp_mr) { reply(OFL_ERDMA, "ibv_reg_mr failed"); fail_why = fmt("ibv_reg_mr: %s", strerror(errno)); return false; }
    for (uint32_t k = 0; k < inflight; k++) {
      resp_free.push_back((int)k);
      ibv_sge rs{(uint64_t)(uintptr_t)(recv_mem + k * recv_slot), (uint32_t)recv_slot, recv_mr->lkey};
      ibv_recv_wr rw{}, *bad = nullptr;
      rw.wr_id = k;
      rw.sg_list = &rs;
      rw.num_sge = 1;
      if (ibv_post_recv(qp, &rw, &bad) != 0) { reply(OFL_ERDMA, "ibv_post_recv failed"); fail_why = "ibv_post_recv failed"; return false; }
    }
    psn = rand_psn();
    if (!qp_connect(d, qp, psn, hi, m)) { reply(OFL_ERDMA, m); fail_why = m; return false; }
    ack.qpn = qp->qp_num;
    ack.psn = psn;
    memcpy(ack.gid, d.gid.raw, 16);
    ack.mtu = d.mtu;
  }
  ack.backend = (uint32_t)backend;
  ack.max_req_bytes = max_req;
  ack.max_resp_bytes = max_resp;
  ack.max_inflight = inflight;
  if (!reply(0, "ok")) { fail_why = "could not send the handshake reply"; return false; }
  s->n_conns++;
  return true;
}

void Conn::main() {
  if (handshake()) {
    if (backend == OFL_BACKEND_TCP)
      run_tcp();
    else
      run_rdma();
  } else if (s->running.load()) {
    fprintf(stderr, "ofl server: %s\n", fail_why.c_str());
  }
  finished.store(true);
}

void ofl_server::accept_loop() {
  while (running.load()) {
    pollfd p{lfd, POLLIN, 0};
    const int r = poll(&p, 1, 100);
    {
      std::lock_guard<std::mutex> g(mu);  // reap finished connections
      conns.erase(std::remove_if(conns.begin(), conns.end(), [](const std::unique_ptr<Conn>& c) { return c->finished.load(); }),
                  conns.end());
    }
    if (r <= 0) continue;
    sockaddr_in a{};
    socklen_t al = sizeof(a);
    const int fd = accept4(lfd, reinterpret_cast<sockaddr*>(&a), &al, SOCK_NONBLOCK | SOCK_CLOEXEC);
    if (fd < 0) continue;
    tune_socket(fd);
    std::lock_guard<std::mutex> g(mu);
    if (conns.size() >= max_conns) {
      fprintf(stderr, "ofl server %s: connection limit %u reached, closing a new one\n", desc.c_str(), max_conns);
      ::close(fd);
      continue;
    }
    auto c = std::make_unique<Conn>();
    c->s = this;
    c->id = next_conn++;
    c->fd = fd;
    Conn* cp = c.get();
    conns.push_back(std::move(c));
    cp->thr = std::thread([cp] { cp->main(); });
  }
}

extern "C" {

ofl_server* ofl_server_create(const ofl_server_config* cfg, ofl_handler_fn fn, void* user, char* err, size_t errlen) {
  in_addr a{};
  if (!cfg || !parse_ipv4(cfg->bind_ip, &a) || !cfg->port || !cfg->backends) {
    copy_err(err, errlen, "ofl_server_create: bind_ip, port and backends are required");
    return nullptr;
  }
  if (a.s_addr == htonl(INADDR_ANY)) { copy_err(err, errlen, "ofl_server_create: refusing to bind 0.0.0.0"); return nullptr; }
  auto* s = new ofl_server();
  s->bind_ip = cfg->bind_ip;
  s->port = cfg->port;
  s->backends = cfg->backends;
  s->rdma_device = cfg->rdma_device ? cfg->rdma_device : "";
  s->rdma_port = cfg->rdma_port > 0 ? cfg->rdma_port : 1;
  s->gid_index = cfg->gid_index;
  s->service = cfg->service ? cfg->service : "";
  s->max_req = cfg->max_req_bytes;
  s->max_resp = cfg->max_resp_bytes;
  s->max_inflight = cfg->max_inflight ? std::min<uint32_t>(cfg->max_inflight, 64) : 2;
  s->max_conns = cfg->max_conns ? cfg->max_conns : 8;
  if (cfg->spin_us > 0) s->spin_ns = cfg->spin_us * 1000LL;
  s->fn = fn;
  s->user = user;
  std::string m;
  if (s->backends & (1 << OFL_BACKEND_RDMA)) {
    if (!open_dev(cfg->bind_ip, s->rdma_device.c_str(), s->rdma_port, s->gid_index, s->dev, m)) {
      copy_err(err, errlen, "RDMA: " + m);
      delete s;
      return nullptr;
    }
    s->have_dev = true;
  }
  s->lfd = ::socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  int one = 1;
  setsockopt(s->lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in l{};
  l.sin_family = AF_INET;
  l.sin_addr = a;
  l.sin_port = htons(cfg->port);
  if (s->lfd < 0 || ::bind(s->lfd, reinterpret_cast<sockaddr*>(&l), sizeof(l)) != 0 || ::listen(s->lfd, 16) != 0) {
    copy_err(err, errlen, fmt("listen %s:%u: %s", cfg->bind_ip, cfg->port, strerror(errno)));
    if (s->lfd >= 0) ::close(s->lfd);
    s->dev.close();
    delete s;
    return nullptr;
  }
  s->desc = fmt("%s:%u service '%s' backends%s%s", cfg->bind_ip, cfg->port, s->service.c_str(),
                (s->backends & (1 << OFL_BACKEND_TCP)) ? " tcp" : "", s->have_dev ? "" : "");
  if (s->have_dev)
    s->desc += fmt(" rdma(%s port %d gid %d mtu %s)", s->dev.name.c_str(), s->dev.port, s->dev.gid_index, mtu_str(s->dev.mtu));
  return s;
}

int ofl_server_start(ofl_server* s) {
  if (!s || s->running.load()) return OFL_EINVAL;
  s->running.store(true);
  s->acceptor = std::thread([s] { s->accept_loop(); });
  return OFL_OK;
}

void ofl_server_stop(ofl_server* s) {
  if (!s) return;
  s->running.store(false);
  if (s->acceptor.joinable()) s->acceptor.join();
  std::vector<std::unique_ptr<Conn>> cs;
  {
    std::lock_guard<std::mutex> g(s->mu);
    cs.swap(s->conns);
  }
  cs.clear();  // each destructor joins its thread (they see running == false within ~100 ms)
}

void ofl_server_destroy(ofl_server* s) {
  if (!s) return;
  ofl_server_stop(s);
  if (s->lfd >= 0) ::close(s->lfd);
  s->dev.close();
  delete s;
}

const char* ofl_server_describe(const ofl_server* s) { return s ? s->desc.c_str() : ""; }

void ofl_server_stats(const ofl_server* s, uint64_t out[4]) {
  out[0] = s->n_conns.load();
  out[1] = s->n_reqs.load();
  out[2] = s->n_bytes.load();
  out[3] = s->n_errs.load();
}

uint16_t ofl_req_op(const ofl_req* r) { return r->hdr.op; }
uint64_t ofl_req_id(const ofl_req* r) { return r->hdr.req_id; }
uint64_t ofl_req_generation(const ofl_req* r) { return r->hdr.generation; }
int ofl_req_backend(const ofl_req* r) { return r->c->backend; }
uint64_t ofl_req_conn_id(const ofl_req* r) { return r->c->id; }
const void* ofl_req_payload(const ofl_req* r, uint32_t* len) {
  if (len) *len = r->payload_len;
  return r->payload;
}
void* ofl_req_resp_buf(ofl_req* r, uint32_t* capacity) {
  if (capacity) *capacity = r->cap;
  return r->resp;
}
void ofl_req_set_len(ofl_req* r, uint32_t len) { r->len = len; }
int ofl_req_ready(ofl_req* r, uint32_t off, uint32_t len) {
  if (!r || (uint64_t)off + len > r->cap) return OFL_EINVAL;
  if (r->c->backend != OFL_BACKEND_RDMA || !len) return OFL_OK;
  std::lock_guard<std::mutex> g(r->c->post_mu);
  if (r->err) return r->err;
  const int rc = post_range_locked(r, off, len);
  if (rc) { r->err = rc; return rc; }
  r->sent.emplace_back(off, len);
  return OFL_OK;
}

}  // extern "C"
