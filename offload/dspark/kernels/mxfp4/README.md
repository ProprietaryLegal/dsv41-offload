Status 2026-10-23: Native SM86 component integrated into opt-in native arms; reconstruction tests do not substitute for N4 pair/GB10 qualification. See [live roadmap](../../../../documents/plans/roadmap.md) and [measured record](../../../results/COMPARISON.md); campaign review decisions are in `LANES.md` / lane reports outside the repo.

# Native MXFP4, standalone SM86 candidate

Experimental, default off; no worker or service changes. Native E2M1 is not EXL3 K4.
This component owns packed expert linear algebra, not routing selection, model
QDQ, distributed reductions, or target sampling. N1 owns model integration.

## API contract

Canonical checkpoint bytes are uint8 `[E,N,K/2]`, low nibble at even K;
scales are uint8 `[E,N,K/32]`, value `2**(code-127)` (255 is NaN).
`PackedExperts.from_checkpoint` checks the actual code/scale pairs on CPU,
then transposes bytes to `[E,K/2,N]` and scales to `[E,K/32,N]` for coalesced
N loads. This permutation loses no values. Load one final TP shard at a time;
never expand the checkpoint on a GPU to prepare this kernel.

Activations are BF16 `[rows,K]`; IDs are int32 `[rows,top_k]` (top_k <= 3),
routing coefficients FP32. Outputs and accumulation are FP32. `linear_into`
uses device IDs directly; no CPU count/readback, sort, atomics, or dynamic
allocation. Workspaces and outputs belong to the caller. The MoE pipeline
supports fused gate/up and staged gate/up; clamp/SwiGLU then down and fixed
slot-order routing combination. The caller must apply the official trained
activation QDQ at its prescribed boundaries; these GEMMs do not remove it.

TP2 gate/up: K=5120,N=1152; down K=1152,N=5120. Input splits must start/end
on group-32 boundaries; output splits preserve their own scale rows. Padding
is masked, not read from uninitialized bytes. No cross-device access here.

## Donor audit

Pinned sources are recorded in `donors.lock.json`. Source inspected, not run:

* [Strata](https://github.com/ro99/strata/tree/015b075079c51a7aec670ee24924f920f5e7bb2b),
  `backend_kernels.cuh`: fragment-order nibble prepack, `__byte_perm` table
  decoding, `mma.sync.aligned.m16n8k16...f32.bf16.bf16.f32`, fused gate/up and
  split-K publication. Its `(scale_code << 7)` shortcut needs scale-0/255
  admission and its sign suppression canonicalizes negative zero. We use
  integer BF16 field construction instead, preserving signed zero and BF16
  subnormals in the decoded representation. Tensor-core denormal arithmetic
  is a separate admission question (tested, not inferred from decode identity).
* [vLLM](https://github.com/vllm-project/vllm/tree/2d40c0609fd672e845ddfef0324ab6d8dd616af4),
  `marlin_utils_fp4.py`, Marlin `dequant.h` and dispatch: MXFP4 per-32 scales
  are distinct from NVFP4's per-16 scale conversion. FP4 on Ampere is decoded
  to BF16/FP16 for supported MMA, not a native FP4 tensor instruction. Scale
  permutation `[0,2,1,3]` is layout-specific, not checkpoint semantics.

Two concrete implementations are retained: original masked Triton kernels with
a simple lossless transpose, and a minimal Strata-derived `regfed.cu` with its
lossless fragment prepack and register-fed BF16 MMA. The latter compacts up to
64 routed slots into groups of eight matching experts on the device and reuses
weight fragments across those rows. It uses 40 registers (single projection) or
48 (fused gate/up), zero shared memory, zero stack/local memory, explicit SM86.
No entire donor engine is imported. `NOTICE` and `LICENSE.strata` preserve the
Apache attribution. The register path explicitly excludes scale code zero;
all actual native scales are in 111..127. The integer Triton decoder covers
the complete code/scale representation, including scale-zero subnormals.
Build target must report SM86; generated PTX is checked for BF16 FP32 MMA and
absence of native FP4/FP8/TMA instructions. The existing pinned lane E environment
provides torch 2.10.0+cu126 and Triton 3.6.0; no engine upgrade or new dependency.

Decode identity is stronger than arithmetic identity: FP32 GEMM reduction order
can differ from a CPU FP32 reference. Report both with actual errors, and never
claim full draft/target acceptance from standalone kernel tests.
