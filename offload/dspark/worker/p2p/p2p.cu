// Two-GPU collectives over NVLink peer access for the TP = 2 DSpark worker (lane E). One process drives both GPUs;
// each rank owns a state block in its own device memory, the peer writes into it directly (remote stores over
// NVLink, UVA pointers, peer access enabled both ways). Every collective is "one-shot push":
//   1. write this rank's payload into the peer's receive buffer (remote stores),
//   2. __syncthreads, thread 0: __threadfence_system + st.release.sys of the peer's flag = epoch,
//   3. thread 0 spins (ld.acquire.sys) until its own flag == epoch, bounded by a %globaltimer timeout,
//   4. combine its own payload with the received one (L1 bypassed) and write the result,
//   5. publish a consumption acknowledgment and wait for the peer's acknowledgment before completing.
// Epochs: one counter per rank in its header, read at the start of every collective (epoch = counter + 1) and stored
// by the last block to finish, so the next collective in stream order sees it. Both ranks run the same sequence of
// collectives with the same grids, so their counters stay equal. Buffers and flags are double-buffered by epoch
// parity: a rank reaches epoch e + 2 only after passing epoch e + 1's wait, which the peer signals only after it has
// finished reading epoch e's buffer (stream order). Explicit acks also make peer validation errors visible before
// a successful collective completes. Epoch zero is reserved: exhaustion fails closed instead of aliasing unused flags.
// Timeout: the waiting block sets bit 0 of its own error word and bit 1 of the peer's, records the epoch, and goes on
// with its OWN data only (never the receive buffer: after a missed signal it holds an older epoch's payload, e.g.
// activations where an exchange expects token ids). Error state = local only: a collective that starts with a set
// error word skips the remote stores, the signal and the wait, and combines from its own data (ar_sum: its partial;
// the exchanges: its own candidate, always a valid id), so a step after a failure costs no further timeouts and a rank
// in error never writes into its peer again. Received ids outside [0, vocab) are refused the same way (bit 2). The
// host refuses the step (TPDrafter checks the error word with every readback) and resets both states once both devices
// are idle. Nothing here allocates; every kernel is capturable in a CUDA graph (static arguments only).
//
// Combine rules (identical bits on both ranks):
//   ar_sum:      own + peer in fp32 (IEEE addition is commutative, so rank 0's p0 + p1 equals rank 1's p1 + p0),
//                optionally rounded to bf16 (round to nearest even, as torch's .to(bfloat16)).
//   xchg_argmax: (v0, i0) of rank 0 (vocabulary [0, V0)) and (v1, i1) of rank 1: i0 if v0 >= v1 else i1 (ties go
//                to the lower id, as the one-card kernels' left tie-break), written into out[:, col].
//   xchg_margin: local (top1, top2, argmax) per rank -> global argmax as above and top1 - top2 of the union.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <limits>

namespace {

constexpr int HDR_EPOCH = 0;      // u32 words of the header
constexpr int HDR_DONE = 1;
constexpr int HDR_ERROR = 2;      // bit 0: this rank timed out; bit 1: the peer timed out; bit 2: bad id received
constexpr int HDR_ERR_EPOCH = 3;  // epoch of the first timeout
constexpr int HDR_COUNT = 4;      // collectives completed (diagnostics)
constexpr int MAX_BLOCKS = 128;
constexpr int64_t FLAGS_OFF = 256;                                  // bytes: flags [2][MAX_BLOCKS] u32
constexpr int64_t ACKS_OFF = 1536;                                  // bytes: consumed [2][MAX_BLOCKS] u32
constexpr int64_t RECV_OFF = 4096;                                  // bytes: recv [2][max_words] u32
constexpr int THREADS = 512;

__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void st_release_sys(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ uint32_t ld_volatile(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

struct Ctx {
  uint32_t* own_hdr;
  uint32_t* peer_hdr;
  uint32_t* own_flags;   // [2][MAX_BLOCKS]
  uint32_t* peer_flags;
  uint32_t* own_acks;
  uint32_t* peer_acks;
  uint32_t* own_recv;    // [2][max_words]
  uint32_t* peer_recv;
  int64_t max_words;
  int64_t timeout_ns;
};

__device__ __forceinline__ Ctx make_ctx(char* own, char* peer, int64_t max_words, int64_t timeout_ns) {
  Ctx c;
  c.own_hdr = reinterpret_cast<uint32_t*>(own);
  c.peer_hdr = reinterpret_cast<uint32_t*>(peer);
  c.own_flags = reinterpret_cast<uint32_t*>(own + FLAGS_OFF);
  c.peer_flags = reinterpret_cast<uint32_t*>(peer + FLAGS_OFF);
  c.own_acks = reinterpret_cast<uint32_t*>(own + ACKS_OFF);
  c.peer_acks = reinterpret_cast<uint32_t*>(peer + ACKS_OFF);
  c.own_recv = reinterpret_cast<uint32_t*>(own + RECV_OFF);
  c.peer_recv = reinterpret_cast<uint32_t*>(peer + RECV_OFF);
  c.max_words = max_words;
  c.timeout_ns = timeout_ns;
  return c;
}

// thread 0 of every block: epoch of this collective, and whether this rank is already in error (then the collective
// is local only: no remote store, no signal, no wait)
__device__ __forceinline__ uint32_t begin(const Ctx& c, bool* degraded) {
  __shared__ uint32_t s_epoch, s_err;
  if (threadIdx.x == 0) {
    const uint32_t previous = ld_acquire_sys(c.own_hdr + HDR_EPOCH);
    s_epoch = previous == UINT32_MAX ? previous : previous + 1u;
    if (previous == UINT32_MAX) {
      atomicOr_system(c.own_hdr + HDR_ERROR, 16u);
      atomicOr_system(c.peer_hdr + HDR_ERROR, 16u);
    }
    s_err = ld_acquire_sys(c.own_hdr + HDR_ERROR);
  }
  __syncthreads();
  *degraded = s_err != 0u;
  return s_epoch;
}

// Only local elapsed time is compared: %globaltimer values from different GPUs are never exchanged.
// Cancellation/error is sticky until a host-confirmed idle reset, so neither a delayed publisher nor a new graph
// replay can turn a failed generation into success.
__device__ __forceinline__ bool wait_epoch(const Ctx& c, const uint32_t* flag, uint32_t epoch) {
  const uint64_t start = globaltimer();
  while (true) {
    if (ld_acquire_sys(c.own_hdr + HDR_ERROR)) return false;
    if (ld_acquire_sys(flag) == epoch) return true;
    if (globaltimer() - start > (uint64_t)c.timeout_ns) {
      if (atomicOr_system(c.own_hdr + HDR_ERROR, 1u) == 0u)
        st_release_sys(c.own_hdr + HDR_ERR_EPOCH, epoch);
      atomicOr_system(c.peer_hdr + HDR_ERROR, 2u);
      return false;
    }
    __nanosleep(32);
  }
}

// after this block's remote stores: signal the peer, wait for the peer's signal. Returns (in every thread) whether the
// peer's payload arrived; false when degraded or on a timeout (then the caller combines from its own data only).
__device__ __forceinline__ bool handshake(const Ctx& c, uint32_t epoch, bool degraded) {
  __shared__ int s_ok;
  if (degraded) return false;
  __threadfence_system();      // every thread's remote stores before the barrier (thread 0's release is cumulative too)
  __syncthreads();
  if (threadIdx.x == 0) {
    const int p = epoch & 1u;
    __threadfence_system();
    st_release_sys(c.peer_flags + p * MAX_BLOCKS + blockIdx.x, epoch);
    const uint32_t* f = c.own_flags + p * MAX_BLOCKS + blockIdx.x;
    s_ok = wait_epoch(c, f, epoch);
  }
  __syncthreads();
  return s_ok != 0;
}

// a received id outside [0, vocab): refuse it (error bit 2), keep the rank's own candidate
__device__ __forceinline__ bool id_ok(const Ctx& c, int64_t id, int64_t vocab) {
  if (id >= 0 && id < vocab) return true;
  atomicOr_system(c.own_hdr + HDR_ERROR, 4u);
  atomicOr_system(c.peer_hdr + HDR_ERROR, 4u);
  return false;
}

// the last block to finish publishes the epoch for the next collective in stream order
__device__ __forceinline__ void finish(const Ctx& c, uint32_t epoch, bool received) {
  // All consumer reads and mirrored error stores precede the acknowledgment. No receive buffer can be reused
  // while a peer still reads it. Failed rounds drain locally and the host refuses their outputs.
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    if (received) {
      const int p = epoch & 1u;
      st_release_sys(c.peer_acks + p * MAX_BLOCKS + blockIdx.x, epoch);
      (void)wait_epoch(c, c.own_acks + p * MAX_BLOCKS + blockIdx.x, epoch);
    }
    __threadfence();
    const uint32_t d = atomicAdd(c.own_hdr + HDR_DONE, 1u);
    if (d == gridDim.x - 1) {
      c.own_hdr[HDR_DONE] = 0u;
      c.own_hdr[HDR_COUNT] += 1u;
      __threadfence();
      atomicExch(c.own_hdr + HDR_EPOCH, epoch);
    }
  }
}

template <bool BF16_OUT>
__global__ void __launch_bounds__(THREADS) ar_sum_kernel(const float4* __restrict__ x, void* __restrict__ out,
                                                         int64_t n4, char* own, char* peer, int64_t max_words,
                                                         int64_t timeout_ns) {
  const Ctx c = make_ctx(own, peer, max_words, timeout_ns);
  bool degraded;
  const uint32_t epoch = begin(c, &degraded);
  const int p = epoch & 1u;
  const int64_t per = (n4 + gridDim.x - 1) / gridDim.x;
  const int64_t a = (int64_t)blockIdx.x * per;
  const int64_t b = a + per < n4 ? a + per : n4;
  if (!degraded) {
    float4* dst = reinterpret_cast<float4*>(c.peer_recv + p * max_words);
    for (int64_t i = a + threadIdx.x; i < b; i += blockDim.x) dst[i] = x[i];
  }
  const bool ok = handshake(c, epoch, degraded);
  const float4* src = reinterpret_cast<const float4*>(c.own_recv + p * max_words);
  for (int64_t i = a + threadIdx.x; i < b; i += blockDim.x) {
    const float4 v = x[i];
    const float4 r = ok ? __ldcg(src + i) : make_float4(0.f, 0.f, 0.f, 0.f);   // failed: own partial only
    const float4 s = make_float4(v.x + r.x, v.y + r.y, v.z + r.z, v.w + r.w);
    if constexpr (BF16_OUT) {
      __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(out) + 2 * i;
      o[0] = __floats2bfloat162_rn(s.x, s.y);
      o[1] = __floats2bfloat162_rn(s.z, s.w);
    } else {
      reinterpret_cast<float4*>(out)[i] = s;
    }
  }
  finish(c, epoch, ok);
}

// combine of an argmax exchange: rank 0's (v0, i0) vs rank 1's (v1, i1), ties to rank 0 (the lower ids); a failed
// exchange or an out-of-range received id keeps the rank's own candidate (always a valid id)
__device__ __forceinline__ int64_t pick_of(const Ctx& c, bool ok, const uint32_t* src, float mv, int64_t mi,
                                           int64_t vocab, int is_rank0) {
  if (!ok) return mi;
  const float tv = __uint_as_float(__ldcg(src));
  const int64_t ti = (int64_t)(((uint64_t)__ldcg(src + 2) << 32) | (uint64_t)__ldcg(src + 1));
  if (!id_ok(c, ti, vocab)) return mi;
  const float v0 = is_rank0 ? mv : tv, v1 = is_rank0 ? tv : mv;
  const int64_t i0 = is_rank0 ? mi : ti, i1 = is_rank0 ? ti : mi;
  return v0 >= v1 ? i0 : i1;
}

__device__ __forceinline__ void put_candidate(uint32_t* dst, float mv, int64_t mi) {
  dst[0] = __float_as_uint(mv);
  dst[1] = (uint32_t)((uint64_t)mi & 0xffffffffull);
  dst[2] = (uint32_t)((uint64_t)mi >> 32);
}

// payload per row: [value bits, idx lo, idx hi, unused]
__global__ void xchg_argmax_kernel(const float* __restrict__ val, const int64_t* __restrict__ idx, int64_t v_off,
                                   int64_t vocab, int B, int64_t* __restrict__ out, int64_t out_stride, int col,
                                   int is_rank0, char* own, char* peer, int64_t max_words, int64_t timeout_ns) {
  const Ctx c = make_ctx(own, peer, max_words, timeout_ns);
  bool degraded;
  const uint32_t epoch = begin(c, &degraded);
  const int p = epoch & 1u;
  const int t = threadIdx.x;
  float mv = 0.f;
  int64_t mi = 0;
  if (t < B) {
    mv = val[t];
    mi = idx[t] + v_off;
    if (!degraded) put_candidate(c.peer_recv + p * max_words + 4 * t, mv, mi);
  }
  const bool ok = handshake(c, epoch, degraded);
  if (t < B) out[t * out_stride + col] = pick_of(c, ok, c.own_recv + p * max_words + 4 * t, mv, mi, vocab, is_rank0);
  finish(c, epoch, ok);
}

// Fused Markov step: per row b, the shard's best (value, index) over the block partials of the cached or uncached
// kernel (chosen by whether the previous token has a cached bias row), first maximum = lowest index (exactly
// tp2._argmax_finish_vi_kernel's tl.max left tie-break), then the argmax exchange. One warp per row, B <= 4.
__global__ void __launch_bounds__(128) xchg_markov_kernel(
    const int64_t* __restrict__ prev, int64_t prev_stride, const int* __restrict__ slot_of_token,
    const float* __restrict__ vals_c, const int64_t* __restrict__ idx_c, int nc,
    const float* __restrict__ vals_u, const int64_t* __restrict__ idx_u, int nu, int64_t v_off, int64_t vocab, int B,
    int64_t* __restrict__ out, int64_t out_stride, int col, int is_rank0, char* own, char* peer, int64_t max_words,
    int64_t timeout_ns) {
  const Ctx c = make_ctx(own, peer, max_words, timeout_ns);
  bool degraded;
  const uint32_t epoch = begin(c, &degraded);
  const int p = epoch & 1u;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  float mv = 0.f;
  int64_t mi = 0;
  if (warp < B) {
    int64_t tok = prev[warp * prev_stride];
    if (tok < 0 || tok >= vocab) tok = 0;        // never index the table out of range (the step is refused anyway)
    const bool cached = slot_of_token[tok] >= 0;
    const float* v = cached ? vals_c + (int64_t)warp * nc : vals_u + (int64_t)warp * nu;
    const int64_t* ix = cached ? idx_c + (int64_t)warp * nc : idx_u + (int64_t)warp * nu;
    const int n = cached ? nc : nu;
    float bv = -INFINITY;
    int bj = 0x7fffffff;
    for (int j = lane; j < n; j += 32) {
      const float x = v[j];
      if (x > bv) { bv = x; bj = j; }
    }
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const int oj = __shfl_xor_sync(0xffffffffu, bj, o);
      if (ov > bv || (ov == bv && oj < bj)) { bv = ov; bj = oj; }
    }
    mv = bv;
    mi = (bj < n ? ix[bj] : 0) + v_off;
    if (lane == 0 && !degraded) put_candidate(c.peer_recv + p * max_words + 4 * warp, mv, mi);
  }
  const bool ok = handshake(c, epoch, degraded);
  if (warp < B && lane == 0)
    out[warp * out_stride + col] = pick_of(c, ok, c.own_recv + p * max_words + 4 * warp, mv, mi, vocab, is_rank0);
  finish(c, epoch, ok);
}

// payload per row: [top1 bits, top2 bits, idx lo, idx hi]
__global__ void xchg_margin_kernel(const float* __restrict__ top2, const int64_t* __restrict__ idx, int64_t v_off,
                                   int64_t vocab, int B, int64_t* __restrict__ out, int64_t out_stride,
                                   float* __restrict__ mg, int64_t mg_stride, int col, int is_rank0, char* own,
                                   char* peer, int64_t max_words, int64_t timeout_ns) {
  const Ctx c = make_ctx(own, peer, max_words, timeout_ns);
  bool degraded;
  const uint32_t epoch = begin(c, &degraded);
  const int p = epoch & 1u;
  const int t = threadIdx.x;
  float m1 = 0.f, m2 = 0.f;
  int64_t mi = 0;
  if (t < B) {
    m1 = top2[2 * t];
    m2 = top2[2 * t + 1];
    mi = idx[t] + v_off;
    if (!degraded) {
      uint32_t* dst = c.peer_recv + p * max_words + 4 * t;
      dst[0] = __float_as_uint(m1);
      dst[1] = __float_as_uint(m2);
      dst[2] = (uint32_t)((uint64_t)mi & 0xffffffffull);
      dst[3] = (uint32_t)((uint64_t)mi >> 32);
    }
  }
  const bool ok = handshake(c, epoch, degraded);
  if (t < B) {
    float t1 = -INFINITY, t2 = -INFINITY;            // failed exchange / bad id: own candidate and margin only
    int64_t ti = mi;
    if (ok) {
      const uint32_t* src = c.own_recv + p * max_words + 4 * t;
      const int64_t ri = (int64_t)(((uint64_t)__ldcg(src + 3) << 32) | (uint64_t)__ldcg(src + 2));
      if (id_ok(c, ri, vocab)) {
        t1 = __uint_as_float(__ldcg(src));
        t2 = __uint_as_float(__ldcg(src + 1));
        ti = ri;
      }
    }
    const float a1 = is_rank0 ? m1 : t1, a2 = is_rank0 ? m2 : t2;
    const float b1 = is_rank0 ? t1 : m1, b2 = is_rank0 ? t2 : m2;
    const int64_t ia = is_rank0 ? mi : ti, ib = is_rank0 ? ti : mi;
    float top, second;
    int64_t pick;
    if (a1 >= b1) {
      top = a1; second = fmaxf(a2, b1); pick = ia;
    } else {
      top = b1; second = fmaxf(b2, a1); pick = ib;
    }
    out[t * out_stride + col] = pick;
    mg[t * mg_stride + col] = top - second;
  }
  finish(c, epoch, ok);
}

char* ptr(int64_t p) { return reinterpret_cast<char*>(p); }

void check_state(const torch::Tensor& st, int64_t max_words) {
  TORCH_CHECK(st.is_cuda() && st.dtype() == torch::kUInt8 && st.is_contiguous(), "p2p: state must be a contiguous "
              "uint8 CUDA tensor");
  TORCH_CHECK(max_words > 0 && max_words % 4 == 0, "p2p: max_words ", max_words, " must be a positive multiple of 4 "
              "(the parity-1 receive buffer is read as float4)");
  TORCH_CHECK(st.numel() >= RECV_OFF + 2 * max_words * 4, "p2p: state too small for ", max_words, " words");
}

void check_devices(const torch::Tensor& base, std::initializer_list<torch::Tensor> tensors) {
  for (const auto& t : tensors)
    TORCH_CHECK(t.device() == base.device(), "p2p: every tensor must be on ", base.device(), ", got ", t.device());
}

__global__ void cancel_kernel(char* state) {
  atomicOr_system(reinterpret_cast<uint32_t*>(state) + HDR_ERROR, 8u);
}

}  // namespace

int64_t state_bytes(int64_t max_words) {
  TORCH_CHECK(max_words > 0 && max_words % 4 == 0 &&
              max_words <= (std::numeric_limits<int64_t>::max() - RECV_OFF) / 8,
              "p2p: state size needs a positive, 4-aligned, non-overflowing max_words");
  return RECV_OFF + 2 * max_words * 4;
}

// Peer state must be a legacy CUDA allocation. exllamav3's Python import changes the Torch allocator at runtime
// (without setting PYTORCH_CUDA_ALLOC_CONF). Torch expandable/VMM allocations are not mapped by
// cudaDeviceEnablePeerAccess. A tensor from that allocator is therefore not a valid raw peer-write destination.
// Own this small block independently of the model allocator, on the exact device, for its entire graph lifetime.
torch::Tensor allocate_state(int64_t max_words, int64_t device) {
  const int64_t bytes = state_bytes(max_words);
  c10::cuda::CUDAGuard guard((c10::DeviceIndex)device);
  cudaStreamCaptureStatus status;
  C10_CUDA_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(), &status));
  TORCH_CHECK(status == cudaStreamCaptureStatusNone, "p2p: allocate state outside capture");
  void* data = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&data, bytes));
  auto tensor = torch::from_blob(data, {bytes}, [device](void* p) {
    // A poisoned context is never cleared here. Report a failed free; process exit releases its remaining memory.
    try {
      c10::cuda::CUDAGuard guard((c10::DeviceIndex)device);
      const cudaError_t error = cudaFree(p);
      if (error != cudaSuccess) TORCH_WARN("p2p state free failed; restart required: ", cudaGetErrorString(error));
    } catch (const std::exception& error) {
      TORCH_WARN("p2p state device guard failed during teardown; restart required: ", error.what());
    }
  }, torch::TensorOptions().device(torch::kCUDA, device).dtype(torch::kUInt8));
  C10_CUDA_CHECK(cudaMemsetAsync(data, 0, bytes, at::cuda::getCurrentCUDAStream()));
  return tensor;
}

// cudaDeviceEnablePeerAccess(peer) from `dev`'s context; already enabled is fine
void enable_peer(int64_t dev, int64_t peer) {
  c10::cuda::CUDAGuard g((c10::DeviceIndex)dev);
  // Do not clear a pre-existing launch fault when handling the benign AlreadyEnabled status below.
  C10_CUDA_CHECK(cudaPeekAtLastError());
  int can = 0;
  C10_CUDA_CHECK(cudaDeviceCanAccessPeer(&can, (int)dev, (int)peer));
  TORCH_CHECK(can, "p2p: device ", dev, " cannot access device ", peer);
  cudaError_t e = cudaDeviceEnablePeerAccess((int)peer, 0);
  if (e == cudaErrorPeerAccessAlreadyEnabled) {
    (void)cudaGetLastError();
    return;
  }
  C10_CUDA_CHECK(e);
}

void cancel(torch::Tensor own) {
  c10::cuda::CUDAGuard g(own.device());
  check_state(own, 4);
  cancel_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(static_cast<char*>(own.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void reset(torch::Tensor own) {
  c10::cuda::CUDAGuard g(own.device());
  check_state(own, 4);
  C10_CUDA_CHECK(cudaPeekAtLastError());
  C10_CUDA_CHECK(cudaMemsetAsync(own.data_ptr(), 0, RECV_OFF, at::cuda::getCurrentCUDAStream()));
}

void ar_sum(torch::Tensor x, torch::Tensor out, torch::Tensor own, int64_t peer, int64_t max_words, int64_t nblocks,
            int64_t timeout_ns) {
  c10::cuda::CUDAGuard g(x.device());
  check_state(own, max_words);
  TORCH_CHECK(peer != 0 && timeout_ns > 0, "p2p: null peer or nonpositive timeout");
  TORCH_CHECK(x.dtype() == torch::kFloat32 && x.is_contiguous() && out.is_contiguous(), "p2p.ar_sum: x fp32 contiguous");
  TORCH_CHECK(out.device() == x.device() && own.device() == x.device(), "p2p.ar_sum: devices");
  const int64_t n = x.numel();
  TORCH_CHECK(out.numel() == n && n % 4 == 0 && n <= max_words, "p2p.ar_sum: n ", n, " (max ", max_words, ")");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16
              == 0, "p2p.ar_sum: 16-byte alignment");
  TORCH_CHECK(nblocks >= 1 && nblocks <= MAX_BLOCKS, "p2p.ar_sum: nblocks");
  auto stream = at::cuda::getCurrentCUDAStream();
  if (out.dtype() == torch::kBFloat16) {
    ar_sum_kernel<true><<<nblocks, THREADS, 0, stream>>>(reinterpret_cast<const float4*>(x.data_ptr()),
                                                         out.data_ptr(), n / 4, ptr((int64_t)own.data_ptr()),
                                                         ptr(peer), max_words, timeout_ns);
  } else {
    TORCH_CHECK(out.dtype() == torch::kFloat32, "p2p.ar_sum: out fp32 or bf16");
    ar_sum_kernel<false><<<nblocks, THREADS, 0, stream>>>(reinterpret_cast<const float4*>(x.data_ptr()),
                                                          out.data_ptr(), n / 4, ptr((int64_t)own.data_ptr()),
                                                          ptr(peer), max_words, timeout_ns);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void xchg_argmax(torch::Tensor val, torch::Tensor idx, int64_t v_off, int64_t vocab, torch::Tensor out, int64_t col,
                 bool is_rank0, torch::Tensor own, int64_t peer, int64_t max_words, int64_t timeout_ns) {
  c10::cuda::CUDAGuard g(val.device());
  check_state(own, max_words);
  check_devices(val, {idx, out, own});
  TORCH_CHECK(peer != 0 && timeout_ns > 0 && vocab > 0, "p2p: peer/timeout/vocab");
  const int B = (int)val.numel();
  TORCH_CHECK(val.dtype() == torch::kFloat32 && idx.dtype() == torch::kInt64 && out.dtype() == torch::kInt64 &&
              val.is_contiguous() && idx.is_contiguous() && idx.numel() == B && B >= 1 && B <= 32 &&
              out.dim() == 2 && out.size(0) == B && out.stride(1) == 1 && col >= 0 && col < out.size(1) &&
              4 * B <= max_words, "p2p.xchg_argmax: shapes");
  xchg_argmax_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      val.data_ptr<float>(), idx.data_ptr<int64_t>(), v_off, vocab, B, out.data_ptr<int64_t>(), out.stride(0), (int)col,
      is_rank0 ? 1 : 0, ptr((int64_t)own.data_ptr()), ptr(peer), max_words, timeout_ns);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void xchg_margin(torch::Tensor top2, torch::Tensor idx, int64_t v_off, int64_t vocab, torch::Tensor out,
                 torch::Tensor mg, int64_t col, bool is_rank0, torch::Tensor own, int64_t peer, int64_t max_words,
                 int64_t timeout_ns) {
  c10::cuda::CUDAGuard g(top2.device());
  check_state(own, max_words);
  check_devices(top2, {idx, out, mg, own});
  TORCH_CHECK(peer != 0 && timeout_ns > 0 && vocab > 0, "p2p: peer/timeout/vocab");
  const int B = (int)idx.numel();
  TORCH_CHECK(top2.dtype() == torch::kFloat32 && top2.is_contiguous() && top2.numel() == 2 * B &&
              idx.dtype() == torch::kInt64 && idx.is_contiguous() && B >= 1 && B <= 32 &&
              out.dtype() == torch::kInt64 && out.dim() == 2 && out.size(0) == B && out.stride(1) == 1 &&
              mg.dtype() == torch::kFloat32 && mg.dim() == 2 && mg.size(0) == B && mg.stride(1) == 1 &&
              col >= 0 && col < out.size(1) && col < mg.size(1) && 4 * B <= max_words, "p2p.xchg_margin: shapes");
  xchg_margin_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      top2.data_ptr<float>(), idx.data_ptr<int64_t>(), v_off, vocab, B, out.data_ptr<int64_t>(), out.stride(0),
      mg.data_ptr<float>(), mg.stride(0), (int)col, is_rank0 ? 1 : 0, ptr((int64_t)own.data_ptr()), ptr(peer),
      max_words, timeout_ns);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void xchg_markov(torch::Tensor prev, torch::Tensor slot_of_token, torch::Tensor vals_c, torch::Tensor idx_c,
                 torch::Tensor vals_u, torch::Tensor idx_u, int64_t v_off, int64_t vocab, torch::Tensor out,
                 int64_t col, bool is_rank0, torch::Tensor own, int64_t peer, int64_t max_words, int64_t timeout_ns) {
  c10::cuda::CUDAGuard g(vals_c.device());
  check_state(own, max_words);
  check_devices(vals_c, {prev, slot_of_token, vals_u, idx_c, idx_u, out, own});
  TORCH_CHECK(peer != 0 && timeout_ns > 0 && vals_c.dim() == 2 && vals_u.dim() == 2 &&
              vals_c.size(1) > 0 && vals_u.size(1) > 0, "p2p.xchg_markov: peer/timeout/partials");
  const int B = (int)vals_c.size(0);
  TORCH_CHECK(prev.dtype() == torch::kInt64 && prev.dim() == 1 && prev.size(0) == B && slot_of_token.dtype() ==
              torch::kInt32 && slot_of_token.is_contiguous() && vals_c.dtype() == torch::kFloat32 &&
              vals_u.dtype() == torch::kFloat32 && idx_c.dtype() == torch::kInt64 && idx_u.dtype() == torch::kInt64 &&
              vals_c.is_contiguous() && vals_u.is_contiguous() && idx_c.is_contiguous() && idx_u.is_contiguous() &&
              vals_u.size(0) == B && idx_c.sizes() == vals_c.sizes() && idx_u.sizes() == vals_u.sizes() && B >= 1 &&
              B <= 4 && out.dtype() == torch::kInt64 && out.dim() == 2 && out.size(0) == B && out.stride(1) == 1 &&
              col >= 0 && col < out.size(1) && 4 * B <= max_words && slot_of_token.numel() >= vocab && vocab > 0,
              "p2p.xchg_markov: shapes");
  xchg_markov_kernel<<<1, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      prev.data_ptr<int64_t>(), prev.stride(0), slot_of_token.data_ptr<int>(), vals_c.data_ptr<float>(),
      idx_c.data_ptr<int64_t>(), (int)vals_c.size(1), vals_u.data_ptr<float>(), idx_u.data_ptr<int64_t>(),
      (int)vals_u.size(1), v_off, vocab, B, out.data_ptr<int64_t>(), out.stride(0), (int)col, is_rank0 ? 1 : 0,
      ptr((int64_t)own.data_ptr()), ptr(peer), max_words, timeout_ns);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("allocate_state", &allocate_state);
  m.def("cancel", &cancel);
  m.def("xchg_markov", &xchg_markov);
  m.def("state_bytes", &state_bytes);
  m.def("enable_peer", &enable_peer);
  m.def("reset", &reset);
  m.def("ar_sum", &ar_sum);
  m.def("xchg_argmax", &xchg_argmax);
  m.def("xchg_margin", &xchg_margin);
}
