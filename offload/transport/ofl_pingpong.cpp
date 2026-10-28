// ofl_pingpong: transport self-test and latency/throughput microbenchmark for both ofl backends.
//
//   ofl_pingpong server --bind 10.202.28.1 --port 18600 [--backends tcp,rdma] [--max-req B] [--max-resp B]
//                       [--spin-us N] [--seconds N]
//   ofl_pingpong client --backend rdma|tcp --server 10.202.28.1 --port 18600 --local 10.202.28.2
//                       [--req B] [--resp B] [--segs N] [--iters N] [--warmup N] [--check] [--label S]
//
// op 0 (built-in echo) when --resp is 0 or equal to --req; op 1 ("pattern") otherwise: the server fills --resp bytes
// with a pattern derived from the request id and byte offset, marking 1 MiB pieces ready as it goes (pipelined RDMA
// writes), and the client checks every byte with --check. The destination is split into --segs ranges of one
// registered buffer (like the Engram weights block + scales block). Prints one JSON line of results.
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <string>
#include <vector>

#include "ofl.h"

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int) { g_stop = 1; }

static int64_t now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static inline uint64_t pat64(uint64_t id, uint64_t w) { return ((w + 1) * 0x9E3779B97F4A7C15ull) ^ ((id << 32) | id); }
static inline uint8_t pattern(uint64_t id, uint64_t off) { return (uint8_t)(pat64(id, off >> 3) >> ((off & 7) * 8)); }

static const char* arg(int argc, char** argv, const char* name, const char* def) {
  for (int i = 2; i + 1 < argc; i++)
    if (!strcmp(argv[i], name)) return argv[i + 1];
  return def;
}
static bool flag(int argc, char** argv, const char* name) {
  for (int i = 2; i < argc; i++)
    if (!strcmp(argv[i], name)) return true;
  return false;
}

static int32_t handler(void*, ofl_req* r) {
  if (ofl_req_op(r) != 1) return -22;
  uint32_t plen = 0;
  const void* p = ofl_req_payload(r, &plen);
  if (plen < 4) return -22;
  uint32_t want = 0;
  memcpy(&want, p, 4);
  uint32_t cap = 0;
  uint8_t* out = static_cast<uint8_t*>(ofl_req_resp_buf(r, &cap));
  if (want > cap) return -27;
  const uint64_t id = ofl_req_id(r);
  const uint32_t piece = 1u << 20;
  for (uint32_t off = 0; off < want; off += piece) {
    const uint32_t n = std::min(piece, want - off);
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {  // off is a multiple of 8
      const uint64_t v = pat64(id, (off + i) >> 3);
      memcpy(out + off + i, &v, 8);
    }
    for (; i < n; i++) out[off + i] = pattern(id, off + i);
    if (ofl_req_ready(r, off, n) != OFL_OK) return -5;
  }
  ofl_req_set_len(r, want);
  return 0;
}

static int run_server(int argc, char** argv) {
  ofl_server_config c{};
  c.bind_ip = arg(argc, argv, "--bind", nullptr);
  c.port = (uint16_t)atoi(arg(argc, argv, "--port", "18600"));
  std::string be = arg(argc, argv, "--backends", "tcp,rdma");
  c.backends = (be.find("tcp") != std::string::npos ? 1 << OFL_BACKEND_TCP : 0) |
               (be.find("rdma") != std::string::npos ? 1 << OFL_BACKEND_RDMA : 0);
  c.gid_index = atoi(arg(argc, argv, "--gid", "-1"));
  c.rdma_device = arg(argc, argv, "--dev", nullptr);
  c.service = "ofl-pingpong";
  c.max_req_bytes = (uint32_t)atol(arg(argc, argv, "--max-req", "1048576"));
  c.max_resp_bytes = (uint32_t)atol(arg(argc, argv, "--max-resp", "33554432"));
  c.max_inflight = 4;
  c.spin_us = atol(arg(argc, argv, "--spin-us", "2000000"));
  char err[512];
  ofl_server* s = ofl_server_create(&c, handler, nullptr, err, sizeof(err));
  if (!s) { fprintf(stderr, "server: %s\n", err); return 1; }
  signal(SIGINT, on_sig);
  signal(SIGTERM, on_sig);
  ofl_server_start(s);
  fprintf(stderr, "ofl_pingpong server up: %s\n", ofl_server_describe(s));
  const int64_t secs = atol(arg(argc, argv, "--seconds", "0"));
  const int64_t t0 = now_ns();
  while (!g_stop && (secs <= 0 || now_ns() - t0 < secs * 1000000000LL)) usleep(100000);
  uint64_t st[4];
  ofl_server_stats(s, st);
  fprintf(stderr, "ofl_pingpong server done: conns %llu requests %llu bytes %llu errors %llu\n",
          (unsigned long long)st[0], (unsigned long long)st[1], (unsigned long long)st[2], (unsigned long long)st[3]);
  ofl_server_destroy(s);
  return 0;
}

static int run_client(int argc, char** argv) {
  ofl_client_config c{};
  std::string be = arg(argc, argv, "--backend", "rdma");
  c.backend = be == "tcp" ? OFL_BACKEND_TCP : OFL_BACKEND_RDMA;
  c.server_ip = arg(argc, argv, "--server", nullptr);
  c.server_port = (uint16_t)atoi(arg(argc, argv, "--port", "18600"));
  c.local_ip = arg(argc, argv, "--local", nullptr);
  c.gid_index = atoi(arg(argc, argv, "--gid", "-1"));
  c.rdma_device = arg(argc, argv, "--dev", nullptr);
  c.service = "ofl-pingpong";
  const uint32_t req = (uint32_t)atol(arg(argc, argv, "--req", "64"));
  uint32_t resp = (uint32_t)atol(arg(argc, argv, "--resp", "0"));
  const bool echo = resp == 0 || resp == req;
  if (echo) resp = req;
  const uint32_t nseg = (uint32_t)std::max(1, std::min(OFL_MAX_SEGS, atoi(arg(argc, argv, "--segs", "1"))));
  const int iters = atoi(arg(argc, argv, "--iters", "1000"));
  const int warmup = atoi(arg(argc, argv, "--warmup", "100"));
  const bool check = flag(argc, argv, "--check");
  const char* label = arg(argc, argv, "--label", "");
  c.max_req_bytes = std::max<uint32_t>(req, 4);
  c.max_resp_bytes = resp;
  c.max_inflight = 1;
  char err[512];
  ofl_client* cl = ofl_client_connect(&c, err, sizeof(err));
  if (!cl) { fprintf(stderr, "client: %s\n", err); return 1; }
  fprintf(stderr, "connected: %s\n", ofl_client_describe(cl));
  std::vector<uint8_t> dst(std::max<uint32_t>(resp, 64) + 4096);
  uint8_t* d = reinterpret_cast<uint8_t*>(((uintptr_t)dst.data() + 4095) & ~uintptr_t(4095));
  uint32_t region = 0;
  if (ofl_client_reg(cl, d, std::max<uint32_t>(resp, 64), &region) != OFL_OK) {
    fprintf(stderr, "reg: %s\n", ofl_client_error(cl));
    return 1;
  }
  std::vector<ofl_seg> segs(nseg);
  uint64_t pos = 0;
  for (uint32_t i = 0; i < nseg; i++) {
    const uint64_t n = i + 1 == nseg ? resp - pos : resp / nseg;
    segs[i] = ofl_seg{region, 0, pos, n};
    pos += n;
  }
  std::vector<uint8_t> payload(std::max<uint32_t>(req, 4));
  std::vector<int64_t> lat;
  lat.reserve(iters);
  uint64_t bad = 0;
  int64_t server_ns = 0;
  const int64_t t_start = now_ns();
  int64_t t_meas = t_start;
  for (int it = 0; it < warmup + iters && !g_stop; it++) {
    if (it == warmup) t_meas = now_ns();
    if (echo) {
      for (uint32_t i = 0; i < req; i++) payload[i] = (uint8_t)(i * 13 + it);
    } else {
      memcpy(payload.data(), &resp, 4);
    }
    if (check) memset(d, 0xEE, resp);
    ofl_result res{};
    const int rc = ofl_client_call(cl, echo ? OFL_OP_ECHO : 1, (uint64_t)it, payload.data(), echo ? req : std::max<uint32_t>(req, 4),
                                   segs.data(), nseg, &res, 2000000000LL);
    if (rc != OFL_OK) {
      fprintf(stderr, "call %d failed: %d %s\n", it, rc, ofl_client_error(cl));
      return 2;
    }
    if (res.resp_len != resp) { fprintf(stderr, "call %d: %u bytes, want %u\n", it, res.resp_len, resp); return 2; }
    if (check) {
      uint32_t i = 0;
      if (!echo)
        for (; i + 8 <= resp; i += 8) {
          const uint64_t v = pat64(res.req_id, i >> 3);
          if (memcmp(d + i, &v, 8)) bad++;
        }
      for (; i < resp; i++) {
        const uint8_t want = echo ? (uint8_t)(i * 13 + it) : pattern(res.req_id, i);
        if (d[i] != want) bad++;
      }
    }
    if (it >= warmup) {
      lat.push_back(res.elapsed_ns);
      server_ns += res.server_ns;
    }
  }
  const int64_t wall = now_ns() - t_meas;
  std::sort(lat.begin(), lat.end());
  auto pct = [&](double p) { return lat.empty() ? 0.0 : lat[std::min(lat.size() - 1, (size_t)(p * lat.size()))] / 1000.0; };
  double mean = 0;
  for (auto v : lat) mean += v;
  mean = lat.empty() ? 0 : mean / lat.size() / 1000.0;
  const double mbps = wall > 0 ? (double)resp * lat.size() / (wall / 1e9) / 1e6 : 0;
  printf("{\"label\":\"%s\",\"backend\":\"%s\",\"desc\":\"%s\",\"req\":%u,\"resp\":%u,\"segs\":%u,\"iters\":%zu,"
         "\"check\":%s,\"bad_bytes\":%llu,\"lat_us\":{\"mean\":%.2f,\"p50\":%.2f,\"p90\":%.2f,\"p99\":%.2f,\"max\":%.2f,"
         "\"min\":%.2f},\"server_us_mean\":%.2f,\"resp_MBps\":%.1f}\n",
         label, be.c_str(), ofl_client_describe(cl), req, resp, nseg, lat.size(), check ? "true" : "false",
         (unsigned long long)bad, mean, pct(0.5), pct(0.9), pct(0.99), lat.empty() ? 0 : lat.back() / 1000.0,
         lat.empty() ? 0 : lat.front() / 1000.0, lat.empty() ? 0 : server_ns / 1000.0 / lat.size(), mbps);
  ofl_client_close(cl);
  return bad ? 3 : 0;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: ofl_pingpong server|client ...\n");
    return 1;
  }
  if (!strcmp(argv[1], "server")) return run_server(argc, argv);
  if (!strcmp(argv[1], "client")) return run_client(argc, argv);
  fprintf(stderr, "usage: ofl_pingpong server|client ...\n");
  return 1;
}
