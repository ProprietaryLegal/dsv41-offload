// Diagnostic only: exercise the pinned donor's scalar and vector decoders.
// Donor headers are MIT, (c) turboderp / cuda-exl3 authors; not vendored here.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "exl3_common.cuh"
#include "exl3_dq.cuh"

template<int K, bool VECTOR>
__global__ void decode_kernel(const uint32_t* tr, half* out, int nt, int tiles) {
    int tile = blockIdx.x;
    if (tile >= tiles) return;
    int lane = threadIdx.x;
    const uint32_t* p = tr + tile * 8 * K;
    half vals[8];
    if constexpr (VECTOR) {
        FragB f0, f1;
        // Use exactly the dispatcher invoked by the donor GEMM, including its
        // two dq4 fragments at K5/K6 (generic dq8 is invalid across three words).
        dq_dispatch<K, 2>(p, lane * 8, f0, f1);
        vals[0] = __low2half(f0[0]); vals[1] = __high2half(f0[0]);
        vals[2] = __low2half(f0[1]); vals[3] = __high2half(f0[1]);
        vals[4] = __low2half(f1[0]); vals[5] = __high2half(f1[0]);
        vals[6] = __low2half(f1[1]); vals[7] = __high2half(f1[1]);
    } else {
        for (int i = 0; i < 8; ++i) vals[i] = dq<K, 2>(p, lane * 8 + i);
    }
    for (int i = 0; i < 8; ++i) {
        int r = 2 * (lane % 4) + i % 2 + 8 * ((i / 2) % 2);
        int c = 2 * (lane / 8) + (lane % 8) / 4 + 8 * (i / 4);
        out[(tile / nt * 16 + r) * (nt * 16) + tile % nt * 16 + c] = vals[i];
    }
}

at::Tensor w1_decode(const at::Tensor& tr, bool vector) {
    TORCH_CHECK(tr.is_cuda() && tr.is_contiguous() && tr.scalar_type() == at::kShort && tr.dim() == 3,
                "decode requires contiguous CUDA int16 trellis");
    int bits = tr.size(2) / 16;
    TORCH_CHECK(tr.size(2) == bits * 16 && (bits == 2 || bits == 5 || bits == 6), "K2/K5/K6 only");
    auto out = at::empty({tr.size(0) * 16, tr.size(1) * 16}, tr.options().dtype(at::kHalf));
    auto stream = at::cuda::getCurrentCUDAStream();
    int nt = tr.size(1), tiles = tr.size(0) * nt;
    const uint32_t* p = (const uint32_t*)tr.data_ptr();
    half* o = (half*)out.data_ptr();
    #define RUN(K) if (vector) decode_kernel<K, true><<<tiles, 32, 0, stream>>>(p,o,nt,tiles); \
                   else decode_kernel<K, false><<<tiles, 32, 0, stream>>>(p,o,nt,tiles)
    if (bits == 2) { RUN(2); }
    if (bits == 5) { RUN(5); }
    if (bits == 6) { RUN(6); }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
