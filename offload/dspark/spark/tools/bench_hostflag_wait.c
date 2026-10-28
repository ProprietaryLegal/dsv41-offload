// Wake latency and CPU cost of the async proxy's word wait (remote_dspark/hostflag.c dsv41_hf_wait_geq) per spin / nap
// setting, with the default and a 1 us timer slack (R2-RUNTIME). CPU only; one writer thread stores the word after a
// random 0.5-3 ms gap (a GPU writing `gathered` after a forward), the waiter measures store -> return and its own CPU
// time. Build and run (any node, e.g. the Spark head node for GB10 numbers):
//   cc -O2 -Wall -o bench_hostflag_wait offload/dspark/spark/tools/bench_hostflag_wait.c
//      offload/dspark/spark/remote_dspark/hostflag.c -lpthread && ./bench_hostflag_wait [samples]   (one line)
// Controller (x86, EPYC 7402), 1,500 samples: spin-only 100 % of a core, wake p50 0.6 us; spin 50 / nap 20 us with the
// default 50 us slack: p50 41 us, 12 % CPU; with a 1 us slack (set by the R2 proxy): p50 14 us, p90 26 us, 23 % CPU.
#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/prctl.h>
#include <time.h>

void* dsv41_hf_alloc(size_t bytes);
void dsv41_hf_release_max(uint32_t* addr, uint32_t value);
int dsv41_hf_wait_geq(const uint32_t* addr, uint32_t value, int64_t timeout_ns, int64_t spin_ns, int64_t nap_ns);

static int64_t mono_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static int64_t thread_cpu_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

typedef struct {
  uint32_t* word;
  int n;
  volatile int64_t* stamp;  // store time of value i at stamp[i]
  unsigned seed;
} writer_arg;

static void* writer(void* p) {
  writer_arg* a = (writer_arg*)p;
  for (int i = 1; i <= a->n; i++) {
    int64_t gap = 500000 + (int64_t)(rand_r(&a->seed) % 2500000);
    struct timespec ts = {0, (long)gap};
    nanosleep(&ts, NULL);
    a->stamp[i] = mono_ns();
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    dsv41_hf_release_max(a->word, (uint32_t)i);
  }
  return NULL;
}

static int cmp64(const void* x, const void* y) {
  int64_t a = *(const int64_t*)x, b = *(const int64_t*)y;
  return (a > b) - (a < b);
}

static void run(int n, int64_t spin_ns, int64_t nap_ns, unsigned long slack_ns) {
  uint32_t* word = (uint32_t*)dsv41_hf_alloc(4096);
  int64_t* stamp = (int64_t*)calloc((size_t)n + 1, sizeof(int64_t));
  int64_t* lat = (int64_t*)calloc((size_t)n, sizeof(int64_t));
  if (slack_ns) prctl(PR_SET_TIMERSLACK, slack_ns, 0, 0, 0);
  writer_arg a = {word, n, stamp, 12345u};
  pthread_t th;
  pthread_create(&th, NULL, writer, &a);
  int64_t c0 = thread_cpu_ns(), w0 = mono_ns();
  for (int i = 1; i <= n; i++) {
    dsv41_hf_wait_geq(word, (uint32_t)i, 10000000000LL, spin_ns, nap_ns);
    int64_t t = mono_ns();
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    lat[i - 1] = t - stamp[i];
  }
  int64_t cpu = thread_cpu_ns() - c0, wall = mono_ns() - w0;
  pthread_join(th, NULL);
  if (slack_ns) prctl(PR_SET_TIMERSLACK, 50000UL, 0, 0, 0);
  qsort(lat, (size_t)n, sizeof(int64_t), cmp64);
  printf("spin %5lld us  nap %4lld us  slack %5s : wake p50 %6.1f  p90 %6.1f  p99 %6.1f  max %7.1f us   waiter CPU %5.1f %%\n",
         (long long)(spin_ns / 1000), (long long)(nap_ns / 1000), slack_ns ? "1us" : "deflt",
         lat[n / 2] / 1e3, lat[n * 9 / 10] / 1e3, lat[n * 99 / 100] / 1e3, lat[n - 1] / 1e3, 100.0 * cpu / wall);
  free(stamp);
  free(lat);
  free(word);
}

int main(int argc, char** argv) {
  int n = argc > 1 ? atoi(argv[1]) : 2000;
  static const int64_t cfg[][2] = {{1000000, 0}, {50000, 50000}, {50000, 20000}, {200000, 20000}, {50000, 5000}};
  for (size_t k = 0; k < sizeof(cfg) / sizeof(cfg[0]); k++) {
    run(n, cfg[k][0], cfg[k][1], 0);
    if (cfg[k][1] > 0) run(n, cfg[k][0], cfg[k][1], 1000);
  }
  return 0;
}
