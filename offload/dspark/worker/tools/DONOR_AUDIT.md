# W1 donor and operation audit — source verified, GPU execution pending

This is source inspection and an isolated experimental adapter, **not** a measured
kernel improvement. `W1_GPU_OK` was absent throughout this pass. No CUDA context,
compiler run, donor kernel, worker listener or production RPC was started.

## Identity, revisions and licensing

| Source | Pinned revision | Finding |
|---|---|---|
| sybil-solutions/trellis-serve | `a1f72be2ff6319b9b2ea9bf6768c99664b1ca892` | Authoritative current GitHub identity. `0xSero/trellis-serve` redirects to this same repository ID **1392606873**; not two independent forks. Both API queries say `fork:false`. This is the follow-on queue's exact revision. |
| Zeuss5/cuda-exl3 | `6a1ffc34866e23f484574ce1922a8bca93eb33b2` | Exact follow-on-queue revision. |
| turboderp-org/exllamav3 v1.6.4 | `da68d57de61781e5b74ad68698ac4caf36408d7a` | Compared to our v1.6.0 `151539c77abc7ab7425d30da7a4e8e3c5c154e7b`. |

Public code clones are under `/opt/dsv41/data/donors/`. `donors.lock.json`
records 60 source/license digests. Trellis' repository license is MIT, but its
`cuda/NOTICE` explicitly identifies Marlin-derived C++ as Apache-2.0; preserve both
sets of notices when porting. cuda-exl3's actual LICENSE is MIT with ExLlamaV3
attribution (GitHub's API labels it `NOASSERTION`). ExLlamaV3 is MIT. No donor
source is vendored into the worker, and no engine/plugin package is installed.

The local evidence file `W1-evidence/github-identities.json` records API redirect,
repository ID and license metadata. Pin-specific source links:
[trellis](https://github.com/sybil-solutions/trellis-serve/tree/a1f72be2ff6319b9b2ea9bf6768c99664b1ca892/cuda),
[cuda-exl3](https://github.com/Zeuss5/cuda-exl3/tree/6a1ffc34866e23f484574ce1922a8bca93eb33b2),
[ExLlamaV3 diff](https://github.com/turboderp-org/exllamav3/compare/v1.6.0...v1.6.4).

## Actual shape and dispatch findings

Our manifest has one K6 head (5120→129280), 22 K5 matrices including main_proj,
and 1152 K2 expert matrices (three stages × 128 experts × three projections).
All input/output widths are multiples of 128. Stage dense shapes are:
wq_a 5120→1280, wkv 5120→512, wq_b 1280→32768, wo_b 8192→5120,
shared w1/w3 5120→2304 and w2 2304→5120. Main_proj is 15360→5120.

* Trellis `kernels/marlin.py:supports_matrix` and C++ `repack_trellis` accept
  K3/K4/K5/K6, **not K2**. The MoE front's `check_stack` accepts K3/K4 layouts;
  its generated families contain those bitrates. Merely changing a build variable
  to 2 does not create K2 unpack/layout/template support. This donor cannot
  replace our exact K2 expert path as shipped.
* `sglang_glue/linear.py` rejects Marlin preparation when summed output width is
  above `SGLANG_EXL3_MARLIN_MAX_N` (default 65536). Our 129280-wide head therefore
  falls back under the donor's default engine dispatch. Separately, the runtime
  dense-materialisation path has its own 65536 limit. These are different checks.
* The raw dense C++ interface does not contain that 65536 preparation cutoff.
  `TrellisDense.prepare(..., allow_wide=True)` explicitly tests the full head;
  the default refuses it. It calls raw extension entry points and never an
  automatic backend. Source revision, build receipt and binary digest are checked.
* K5 repacks keep original bytes. K6 expands resident packed storage from six to
  eight bits per weight (overlapping lane windows). A lossless repack is **not**
  yet proof of kernel-decoded identity; test all weights through identity rows
  of the rotated GEMM and compare fp16 bits against our independent CPU decoder.
* Trellis's linear entry point emits fp16 or bf16, while our Exl3Linear asks for
  fp32 before downstream bf16 conversion. A dense switch can alter rounding,
  routing, drafts and confidences even when decoded weights match exactly.
  Our FP32 main_proj path and power-of-two prescale must be retained.

`donor_adapter.py` implements direct dense entry points for both donors, explicit
dispatch rejection, pin checks and receipt-verified binaries. `build_donor.py`
builds only reviewed C++ sources in a W1 directory with SM86 and two compiler
jobs; it never runs upstream setup scripts/launchers. Trellis uses MUL1 K5/K6
and <=8/16/32 row families. Its unsupported K2 MoE path is deliberately not built.
**Neither build nor adapter has been executed on CUDA in this pass.**

## cuda-exl3 and 1.6.4

cuda-exl3 `gemm.cu` instantiates bits 1–8 for all three codebooks. This covers our
MUL1 K2/K5/K6 syntactically, but kernel, scale and shard parity remain untested.
The native dense op can be called through `torch.ops.cuda_exl3_C` without vLLM.
The adapter passes `split_k=False`: its fp32-atomic split-K path has a variable
summation order and is inappropriate as an exact repeatability baseline.
The native op still changes accumulation/output rounding relative to our worker.

Mechanisms worth screening after the current profile: one weight decode per tile
reused across rows; separate input Hadamard to avoid a cooperative grid barrier;
input conversion fused with per-shard scale/Hadamard; output conversion fused
with the output Hadamard. Keep each shard's distinct suh. Tiny expert occupancy
and grouping overhead may erase reuse benefits at 5–20 body rows/top-3. Include
1–4 direct op rows and populated-expert synthetic cases, sorting and unsorting.

The v1.6.0→1.6.4 inspected EXL3 changes add shape/empty-input validation and
ROCm occupancy safety. The GEMM kernel/map and coop-MoE kernel headers used here
are unchanged. hgemm now uses a dedicated cuBLAS handle/workspace; this is a
lifetime/concurrency fix, not evidence of a 3090 GEMM speed gain. No isolated
SM86 arithmetic speed change was found. The full saved diff is in W1 evidence;
do not upgrade the worker wholesale for a claimed benefit without measurement.

## Current operations already contain several proposed optimisations

* The v1.6.0 EXL3 pipeline already issues cp.async in `ptx.cuh`; this is not a new
  Ampere feature to add. coop MoE already avoids a grid barrier according to our
  wrapper's traced entry points. Distinguish that from dense cooperative GEMMs.
* `fused.wo_a_fp8` already launches one grouped grid `(groups,R/BN)` per stage,
  converts FP8 bytes/scales exactly in registers and emits bf16. It is not eight
  Python bmm launches. Screen tile geometry/cast adjacency against this kernel,
  preserving inverse RoPE→FP8 dequant→FP32 accumulation→bf16→wo_b ordering.
* `markov_argmax_into` already combines Markov add and reduction, using cached
  and uncached kernels plus a final partial reduction (three launches). Preserve bf16 base-plus-bias rounding, lowest-ID tie
  handling and sequential previous-token dependence. Confidence still consumes
  the actual selected previous-token embeddings. Do not fuse away reusable base
  logits or replace full-vocabulary search with a candidate list.
* Window attention already reads its ring directly; insert fuses norm/RoPE/QDQ
  and ring-store. mHC already has split-K and collapse/norm fusion. Donors for
  long-context sparse indexers do not correspond to this window-128 path.
* Server `GpuBackend.flush` waits for staging-buffer lifetime; `draft` allocates
  host metadata, uploads it, casts ids/confidences to fp64 and synchronously
  copies a packed reply. That D2H dependency is required before replying; only
  avoidable allocations/conversions and between-graph gaps are candidates.

SM86 has 48 resident warps, 64K 32-bit registers per SM, a 255-register thread
ceiling, 100 KiB shared per SM and 99 KiB per block, requiring opt-in above
48 KiB. Check actual ptxas resource use and launch occupancy rather than A100
limits. The [NVIDIA Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)
also confirms asynchronous global→shared copies and FP32 accumulation support.

## Required qualification, in order

1. Open W1 GPU window; profile current code at burst and decode cadence, save
   active clocks/power, actual expert counts and paired samples. No baseline
   numbers from this pass exist yet.
2. Build pinned standalone donor libraries. Run decode identity on every real
   K5/K6 shard and any candidate K2 implementation, MUL1 windows including cyclic
   wrap and the double-Hadamard regression. An unsupported K2 is a failed coverage
   gate for a whole-MoE replacement, not a pass or a fallback timing.
3. On saved synthetic inputs compare each op with the independent fixed-order
   reference and the present worker separately. Report exact-bit agreement,
   error and nonfinite counts. Check exponent zero/subnormal/special conventions
   before accepting an integer exponent shortcut. Do not narrow decoded fp16
   weights to bf16 and call the result identical.
4. Compare full-step ids and confidence bits for B1–4, every n_out, graph/eager,
   prefix/depth row independence and hit/miss cases. Accumulation differences
   must remain explicit; decoder identity alone never qualifies a switch.
5. Rank op trials by the **new** critical path, including routing/sorting,
   transforms, casts, staging and waits. Retain only repeated measured winners,
   behind a default-off switch, after full-step parity. Deployment belongs to
   the coordinator.
