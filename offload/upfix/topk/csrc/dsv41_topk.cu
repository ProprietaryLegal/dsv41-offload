// dsv41 R-UPF (UPS1 #5): vLLM's persistent_topk with the upstream fixes #62467 (no float4 tail lanes
// past the row end), #62936 (exact on candidate overflow), #63475 (CTA sync around radix barrier
// arrivals) and #63118 (NaN-safe), as a separately named JIT op `torch.ops.dsv41_topk.persistent_topk`.
// The vendored kernels also carry #59645 (cooperative candidate overflow, #62936's prerequisite) and #60721
// (header consolidation): all six commits that touch these files between 6e517b15 and 3090266d8.
// Kernels (*.cuh) are vendored unchanged from vllm-project/vllm 3090266d88b6c3abbe18483fa5bbc7de293d8ef0
// (Apache-2.0). This launcher is csrc/topk.cu at that commit (blob d5ff2a83) ported from the libtorch
// stable ABI to ATen; the schema is identical to the image's `_C::persistent_topk` (6e517b15), and the
// low-shared-memory fallback calls the image's own `_C::top_k_per_row_decode` through the dispatcher.
// Default OFF: only offload/upfix/topk/overlay/.../indexer_topk.py with VLLM_DSV41_TOPK_FIX=1 calls it.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <algorithm>

#include "persistent_topk.cuh"
#include "sampled_topk.cuh"

namespace {

// Same schema at 6e517b15 and 3090266d8: top_k_per_row_decode(Tensor logits, int next_n,
// Tensor seq_lens, Tensor! indices, int numRows, int stride0, int stride1, int topK) -> ()
void image_top_k_per_row_decode(const at::Tensor& logits, int64_t next_n,
                                const at::Tensor& lengths, at::Tensor& output,
                                int64_t num_rows, int64_t stride0,
                                int64_t stride1, int64_t top_k) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("_C::top_k_per_row_decode", "")
          .typed<void(const at::Tensor&, int64_t, const at::Tensor&,
                      at::Tensor&, int64_t, int64_t, int64_t, int64_t)>();
  op.call(logits, next_n, lengths, output, num_rows, stride0, stride1, top_k);
}

}  // anonymous namespace

namespace {

template <int TopK>
void launch_persistent_topk(const at::Tensor& logits,
                            const at::Tensor& lengths,
                            at::Tensor& output,
                            const at::Tensor& workspace,
                            int64_t max_seq_len) {
  namespace P = vllm::persistent;

  const c10::cuda::CUDAGuard device_guard(logits.device());
  const int64_t num_rows = logits.size(0);
  const int64_t stride = logits.stride(0);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  static int num_sms = 0;
  static int max_smem_per_block = 0;
  if (num_sms == 0) {
    const cudaDeviceProp* device_prop = at::cuda::getCurrentDeviceProperties();
    num_sms = device_prop->multiProcessorCount;
    max_smem_per_block = device_prop->sharedMemPerBlockOptin;
  }

  // Allow static fallback storage in addition to the 128 KiB dynamic buffer.
  if (num_rows > 64 &&
      max_seq_len >= vllm::sampled_topk::kMinSampledLength<TopK> &&
      max_smem_per_block >= 144 * 1024) {
    auto kernel = vllm::sampled_topk::sampled_topk_kernel<TopK>;
    constexpr size_t smem_size =
        vllm::filtered_topk::FILTERED_TOPK_SMEM_DYNAMIC;
    cudaError_t status = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    TORCH_CHECK(status == cudaSuccess,
                    "sampled_topk smem failed: ", cudaGetErrorString(status));
    kernel<<<num_rows, vllm::sampled_topk::kThreads, smem_size, stream>>>(
        logits.const_data_ptr<float>(), lengths.const_data_ptr<int32_t>(),
        output.mutable_data_ptr<int32_t>(), stride,
        static_cast<int>(std::min(max_seq_len, logits.size(1))));
  } else if (num_rows > 32 && max_smem_per_block >= 128 * 1024) {
    cudaError_t status =
        vllm::FilteredTopKRaggedTransform<float, int32_t, TopK>(
            logits.const_data_ptr<float>(), output.mutable_data_ptr<int32_t>(),
            lengths.const_data_ptr<int32_t>(), static_cast<uint32_t>(num_rows),
            static_cast<uint32_t>(TopK), static_cast<uint32_t>(stride), stream);
    TORCH_CHECK(status == cudaSuccess,
                    "FilteredTopK failed: ", cudaGetErrorString(status));
  } else {
    TORCH_CHECK(workspace.is_cuda(), "workspace must be CUDA tensor");
    TORCH_CHECK(
        workspace.scalar_type() == at::kByte,
        "workspace must be uint8");

    int effective_max_smem;
    if (num_rows <= 4) {
      effective_max_smem =
          std::min(max_smem_per_block, static_cast<int>(P::kSmemMedium));
    } else if (num_rows <= 8) {
      constexpr int kSmemCapMedium = 48 * 1024;
      effective_max_smem = std::min(max_smem_per_block, kSmemCapMedium);
    } else {
      effective_max_smem = max_smem_per_block;
    }

    size_t available_for_ordered =
        static_cast<size_t>(effective_max_smem) - P::kFixedSmemLarge;
    uint32_t max_chunk_elements =
        static_cast<uint32_t>(available_for_ordered / sizeof(uint32_t));

    // Views like x[:, 1:] keep the stride but shift the base.
    const auto base =
        reinterpret_cast<uintptr_t>(logits.const_data_ptr<float>());
    uint32_t vec_size = 1;
    if (stride % 4 == 0 && base % 16 == 0)
      vec_size = 4;
    else if (stride % 2 == 0 && base % 8 == 0)
      vec_size = 2;

    max_chunk_elements = (max_chunk_elements / vec_size) * vec_size;
    uint32_t min_chunk = vec_size * P::kThreadsPerBlock;
    if (max_chunk_elements < min_chunk) max_chunk_elements = min_chunk;

    uint32_t ctas_per_group =
        (static_cast<uint32_t>(stride) + max_chunk_elements - 1) /
        max_chunk_elements;
    uint32_t chunk_size =
        (static_cast<uint32_t>(stride) + ctas_per_group - 1) / ctas_per_group;
    chunk_size = ((chunk_size + vec_size - 1) / vec_size) * vec_size;
    if (chunk_size > max_chunk_elements) chunk_size = max_chunk_elements;

    size_t smem_size = P::kFixedSmemLarge + chunk_size * sizeof(uint32_t);
    if (smem_size < P::kSmemMedium) smem_size = P::kSmemMedium;

    // Query occupancy for the instantiation that will actually launch;
    // overestimating it deadlocks the cooperative barrier.
    int occupancy = 1;
    cudaError_t occ_err = cudaSuccess;
    if (vec_size == 4) {
      occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, P::persistent_topk_kernel<TopK, 4>, P::kThreadsPerBlock,
          smem_size);
    } else if (vec_size == 2) {
      occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, P::persistent_topk_kernel<TopK, 2>, P::kThreadsPerBlock,
          smem_size);
    } else {
      occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, P::persistent_topk_kernel<TopK, 1>, P::kThreadsPerBlock,
          smem_size);
    }
    TORCH_CHECK(occ_err == cudaSuccess,
                    "persistent_topk occupancy query failed: ",
                    cudaGetErrorString(occ_err));
    if (occupancy < 1) occupancy = 1;

    // The cooperative spin-wait barrier only runs when at least one row hits
    // the radix path (seq_len > RADIX_THRESHOLD). Below that, non-CTA-0 CTAs
    // early-exit, so oversubscription can't deadlock and headroom is wasted.
    const bool needs_cooperative =
        static_cast<uint32_t>(max_seq_len) > P::RADIX_THRESHOLD;

    const uint32_t hw_resident_cap =
        static_cast<uint32_t>(num_sms) * static_cast<uint32_t>(occupancy);
    uint32_t max_resident_ctas = hw_resident_cap;
    if (needs_cooperative) {
      // Reserve one CTA per SM when occupancy allows; fall back to a single
      // CTA when occupancy == 1 (the most deadlock-prone case — any straggler
      // kernel that takes the only slot on one SM hangs the barrier). Never
      // drop below one full group's worth.
      uint32_t headroom = (occupancy > 1) ? static_cast<uint32_t>(num_sms) : 1u;
      if (max_resident_ctas >= headroom + ctas_per_group) {
        max_resident_ctas -= headroom;
      }
    }
    uint32_t num_groups = std::min(max_resident_ctas / ctas_per_group,
                                   static_cast<uint32_t>(num_rows));
    if (num_groups == 0) num_groups = 1;
    uint32_t total_ctas = num_groups * ctas_per_group;

    // If the cooperative launch wouldn't fit, use the generic decode kernel on
    // low-smem devices or FilteredTopK where its 128 KiB requirement is met.
    if (needs_cooperative && total_ctas > hw_resident_cap) {
      if (max_smem_per_block < 128 * 1024) {
        const int64_t next_n = lengths.dim() == 2 ? lengths.size(1) : 1;
        image_top_k_per_row_decode(logits, next_n, lengths, output, num_rows,
                                   logits.stride(0), logits.stride(1), TopK);
        return;
      }
      cudaError_t status =
          vllm::FilteredTopKRaggedTransform<float, int32_t, TopK>(
              logits.const_data_ptr<float>(),
              output.mutable_data_ptr<int32_t>(),
              lengths.const_data_ptr<int32_t>(),
              static_cast<uint32_t>(num_rows), static_cast<uint32_t>(TopK),
              static_cast<uint32_t>(stride), stream);
      TORCH_CHECK(status == cudaSuccess, "FilteredTopK fallback failed: ",
                      cudaGetErrorString(status));
      return;
    }

    size_t state_bytes = num_groups * sizeof(P::RadixRowState);
    TORCH_CHECK(workspace.size(0) >= static_cast<int64_t>(state_bytes),
                    "workspace too small, need ", state_bytes, " bytes");

    // Zero the per-group RadixRowState region before launch.
    //
    // Issued UNCONDITIONALLY so the memset is captured as its own node in
    // the cudagraph (a separate cudaMemsetAsync node, sequenced before the
    // persistent_topk_kernel launch on the same stream). The previous
    // host-side guard `if (needs_cooperative)` was evaluated at capture time;
    // when capture-time max_seq_len <= RADIX_THRESHOLD (always true under
    // FULL_DECODE_ONLY with max_model_len < 32 K) the memset would NOT be
    // captured, leaving the workspace state to accumulate across replays.
    // That's a latent correctness bug if the runtime data ever takes the
    // radix path, and removes one variable while debugging hangs in the
    // decode/medium paths.
    //
    // Cost is sub-microsecond: state_bytes = num_groups * sizeof(RadixRowState)
    // is ~3 KB per group, ~100 KB for the largest grids on this hardware.
    //
    // Why the memset is required (regardless of which path the kernel takes):
    //   1. arrival_counter accumulates within a launch and is never reset,
    //      so a prior call leaves it at a large positive value. Without this
    //      reset, the very first wait_ge in the next call sees counter >>
    //      target and returns instantly, breaking the barrier.
    //   2. The previous in-kernel init only ran in CTA-0 with intra-CTA
    //      __syncthreads(), so it had no happens-before edge to CTA-1+'s
    //      first red_release. cudaMemsetAsync is stream-ordered: the zero
    //      is globally visible before any CTA runs.
    {
      cudaError_t mz_err = cudaMemsetAsync(
          workspace.mutable_data_ptr<uint8_t>(), 0, state_bytes, stream);
      TORCH_CHECK(mz_err == cudaSuccess,
                      "row_states memset failed: ", cudaGetErrorString(mz_err));
    }

    P::PersistentTopKParams params;
    params.input = logits.const_data_ptr<float>();
    params.output = output.mutable_data_ptr<int32_t>();
    params.lengths = lengths.const_data_ptr<int32_t>();
    params.num_rows = static_cast<uint32_t>(num_rows);
    params.stride = static_cast<uint32_t>(stride);
    params.top_k = static_cast<uint32_t>(TopK);
    params.chunk_size = chunk_size;
    params.row_states = reinterpret_cast<P::RadixRowState*>(
        workspace.mutable_data_ptr<uint8_t>());
    params.ctas_per_group = ctas_per_group;
    params.max_seq_len = static_cast<uint32_t>(max_seq_len);

  #define LAUNCH_PERSISTENT(TOPK_VAL, VS)                                     \
    do {                                                                      \
      auto kernel = &P::persistent_topk_kernel<TOPK_VAL, VS>;                 \
      cudaError_t err = cudaFuncSetAttribute(                                 \
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);    \
      TORCH_CHECK(err == cudaSuccess,                                     \
                      "Failed to set smem: ", cudaGetErrorString(err));       \
      kernel<<<total_ctas, P::kThreadsPerBlock, smem_size, stream>>>(params); \
    } while (0)

    if (vec_size == 4) {
      LAUNCH_PERSISTENT(TopK, 4);
    } else if (vec_size == 2) {
      LAUNCH_PERSISTENT(TopK, 2);
    } else {
      LAUNCH_PERSISTENT(TopK, 1);
    }
  #undef LAUNCH_PERSISTENT
  }

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
                  "persistent_topk failed: ", cudaGetErrorString(err));
}
}  // anonymous namespace

void persistent_topk(const at::Tensor& logits,
                     const at::Tensor& lengths,
                     at::Tensor& output,
                     const at::Tensor& workspace, int64_t k,
                     int64_t max_seq_len) {
  TORCH_CHECK(logits.is_cuda(), "logits must be CUDA tensor");
  TORCH_CHECK(lengths.is_cuda(), "lengths must be CUDA tensor");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
  TORCH_CHECK(logits.scalar_type() == at::kFloat,
                  "Only float32 supported");
  TORCH_CHECK(lengths.scalar_type() == at::kInt,
                  "lengths must be int32");
  TORCH_CHECK(output.scalar_type() == at::kInt,
                  "output must be int32");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
  TORCH_CHECK(lengths.dim() == 1 || lengths.dim() == 2,
                  "lengths must be 1D or 2D");
  TORCH_CHECK(lengths.is_contiguous(), "lengths must be contiguous");
  TORCH_CHECK(output.dim() == 2, "output must be 2D");

  const int64_t num_rows = logits.size(0);

  TORCH_CHECK(lengths.numel() == num_rows, "lengths size mismatch");
  TORCH_CHECK(output.size(0) == num_rows && output.size(1) == k,
                  "output size mismatch");
  TORCH_CHECK(
      k == 512 || k == 1024 || k == 2048,
      "persistent_topk supports k=512, k=1024, or k=2048, got k=", k);

  const c10::cuda::CUDAGuard device_guard(logits.device());

  if (k == 512) {
    launch_persistent_topk<512>(logits, lengths, output, workspace,
                                max_seq_len);
  } else if (k == 1024) {
    launch_persistent_topk<1024>(logits, lengths, output, workspace,
                                 max_seq_len);
  } else {
    launch_persistent_topk<2048>(logits, lengths, output, workspace,
                                 max_seq_len);
  }
}

TORCH_LIBRARY(dsv41_topk, m) {
  m.def(
      "persistent_topk(Tensor logits, Tensor lengths, Tensor! output, "
      "Tensor workspace, int k, int max_seq_len) -> ()");
}

TORCH_LIBRARY_IMPL(dsv41_topk, CUDA, m) {
  m.impl("persistent_topk", &persistent_topk);
}
