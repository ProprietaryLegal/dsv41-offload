// R1 CPU-only experiment: x86host -> x86host, own ofl listener, no worker or Spark requests.
// Link to the existing libofl.so; this file does not include verbs or create a CUDA context.
#include "ofl.h"

#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <time.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr const char* kIp = "10.202.28.1";
constexpr const char* kService = "r1-isolated-cpu";
constexpr uint32_t kMaxBytes = 16u << 20;

int64_t now_ns() {
  timespec t{};
  if (clock_gettime(CLOCK_MONOTONIC, &t)) throw std::runtime_error("clock_gettime");
  return t.tv_sec * 1000000000LL + t.tv_nsec;
}

void pin(int cpu) {
  if (cpu < 0) return;
  if (cpu >= CPU_SETSIZE) throw std::runtime_error("CPU exceeds CPU_SETSIZE");
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(cpu, &mask);
  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask))
    throw std::runtime_error("cannot pin own thread");
}

struct HandlerState { int cpu; bool pinned = false; };
int32_t handler(void* user, ofl_req* r) {
  auto& s = *static_cast<HandlerState*>(user);
  try {
    if (!s.pinned) { pin(s.cpu); s.pinned = true; }
    uint32_t n = 0, cap = 0;
    const auto* p = static_cast<const uint8_t*>(ofl_req_payload(r, &n));
    auto* out = static_cast<uint8_t*>(ofl_req_resp_buf(r, &cap));
    if (n < 32 || cap < 24) return -1;
    // Return markers from three distant locations: catches stale/incorrect source buffers.
    std::memcpy(out, p, 8);
    std::memcpy(out + 8, p + n / 2, 8);
    std::memcpy(out + 16, p + n - 8, 8);
    ofl_req_set_len(r, 24);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "handler: %s\n", e.what());
    return -2;
  }
}

double cpu_us() {
  rusage r{};
  if (getrusage(RUSAGE_SELF, &r)) throw std::runtime_error("getrusage");
  return (r.ru_utime.tv_sec + r.ru_stime.tv_sec) * 1e6 + r.ru_utime.tv_usec + r.ru_stime.tv_usec;
}

double percentile(const std::vector<int64_t>& sorted, double q) {
  const auto i = std::min(sorted.size() - 1, static_cast<size_t>(q * (sorted.size() - 1)));
  return static_cast<double>(sorted[i]) / 1000.0;
}

void run_case(ofl_client* client, const ofl_seg& seg, uint8_t* reply,
              uint32_t bytes, bool direct, int iters, int period_us) {
  std::vector<uint8_t> source(bytes, 0x5a);
  uint32_t capacity = 0;
  auto* in_place = static_cast<uint8_t*>(ofl_client_payload_buf(client, &capacity));
  if (!in_place || capacity < bytes) throw std::runtime_error("payload buffer too small");
  std::memcpy(in_place, source.data(), bytes);  // represents producer writing its final destination
  auto* payload = direct ? in_place : source.data();
  std::vector<int64_t> times;
  times.reserve(iters);
  double cpu_start = 0;
  int64_t wall_start = 0, server_ns = 0;
  constexpr int warmup = 12;
  for (int i = 0; i < warmup + iters; ++i) {
    if (i == warmup) { cpu_start = cpu_us(); wall_start = now_ns(); }
    const uint64_t marker = static_cast<uint64_t>(i + 1);
    for (uint32_t pos : {0u, bytes / 2, bytes - 8}) std::memcpy(payload + pos, &marker, 8);
    if (period_us) {
      timespec delay{period_us / 1000000, (period_us % 1000000) * 1000L};
      while (nanosleep(&delay, &delay)) {
        if (errno != EINTR) throw std::runtime_error("nanosleep");
      }
    }
    const auto t0 = now_ns();
    ofl_result result{};
    const int rc = ofl_client_call(client, 1, marker, payload, bytes, &seg, 1, &result, 1000000000LL);
    const auto elapsed = now_ns() - t0;
    if (rc || result.status || result.generation != marker || result.resp_len != 24)
      throw std::runtime_error("ofl call/echo: " + std::to_string(rc) + " " + ofl_client_error(client));
    for (int j = 0; j < 3; ++j) {
      uint64_t got = 0;
      std::memcpy(&got, reply + j * 8, 8);
      if (got != marker) throw std::runtime_error("stale or corrupt reply marker");
    }
    if (i >= warmup) { times.push_back(elapsed); server_ns += result.server_ns; }
  }
  const double cpu = cpu_us() - cpu_start;
  const double wall_us = static_cast<double>(now_ns() - wall_start) / 1000.0;
  std::sort(times.begin(), times.end());
  std::printf("{\"bytes\":%u,\"in_place\":%s,\"iters\":%d,\"period_us\":%d,"
              "\"p50_us\":%.3f,\"p99_us\":%.3f,\"max_us\":%.3f,\"server_mean_us\":%.3f,"
              "\"cpu_us_per_round\":%.3f,\"cpu_cores\":%.4f,\"bad\":0}\n",
              bytes, direct ? "true" : "false", iters, period_us, percentile(times, .5),
              percentile(times, .99), percentile(times, 1), static_cast<double>(server_ns) / iters / 1000.0,
              cpu / iters, cpu / wall_us);
  std::fflush(stdout);
}
}  // namespace

int main(int argc, char** argv) {
  ofl_server* server = nullptr;
  ofl_client* client = nullptr;
  try {
    if (argc != 7) throw std::runtime_error("usage: ofl_loopback PORT ITERS PERIOD_US SPIN_US CLIENT_CPU SERVER_CPU");
    const int port = std::stoi(argv[1]), iters = std::stoi(argv[2]), period = std::stoi(argv[3]);
    const int spin = std::stoi(argv[4]), client_cpu = std::stoi(argv[5]), server_cpu = std::stoi(argv[6]);
    if (port < 18670 || port > 18679 || iters < 1 || iters > 2000 || period < 0 || period > 100000 || spin < 1)
      throw std::runtime_error("arguments outside isolated experiment bounds");
    HandlerState state{server_cpu};
    ofl_server_config sc{};
    sc.bind_ip = kIp; sc.port = static_cast<uint16_t>(port); sc.backends = 1 << OFL_BACKEND_RDMA;
    sc.service = kService; sc.max_req_bytes = kMaxBytes; sc.max_resp_bytes = 64;
    sc.max_inflight = 1; sc.max_conns = 1; sc.spin_us = spin; sc.gid_index = -1;
    char err[512]{};
    server = ofl_server_create(&sc, handler, &state, err, sizeof(err));
    if (!server) throw std::runtime_error(std::string("server create: ") + err);
    if (ofl_server_start(server)) throw std::runtime_error("server start");
    ofl_client_config cc{};
    cc.backend = OFL_BACKEND_RDMA; cc.server_ip = kIp; cc.local_ip = kIp;
    cc.server_port = static_cast<uint16_t>(port); cc.service = kService; cc.max_req_bytes = kMaxBytes;
    cc.max_resp_bytes = 64; cc.max_inflight = 1; cc.gid_index = -1;
    client = ofl_client_connect(&cc, err, sizeof(err));
    if (!client) throw std::runtime_error(std::string("client connect: ") + err);
    // Pin AFTER server creation: only the benchmark caller/connection thread are pinned.
    pin(client_cpu);
    std::fprintf(stderr, "%s; %s; client_cpu=%d server_cpu=%d spin_us=%d\n",
                 ofl_server_describe(server), ofl_client_describe(client), client_cpu, server_cpu, spin);
    std::array<uint8_t, 64> reply{};
    uint32_t region = 0;
    if (ofl_client_reg(client, reply.data(), reply.size(), &region)) throw std::runtime_error("reply registration");
    const ofl_seg seg{region, 0, 0, reply.size()};
    for (uint32_t bytes : {184416u, 737568u, 15728864u}) {
      if (period && bytes > 737568) continue;
      // ABBA order limits drift from favouring either method.
      for (bool direct : {false, true, true, false}) run_case(client, seg, reply.data(), bytes, direct, iters, period);
    }
    ofl_client_close(client); client = nullptr;
    ofl_server_destroy(server); server = nullptr;
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    if (client) ofl_client_close(client);
    if (server) ofl_server_destroy(server);
    return 1;
  }
}
