# Native-expanded SM86 kernels (KX-GROUPED)

Offline component for N1's NativeExpanded/Hybrid TP2 profile. No worker switch,
checkpoint loader, target change, or deployment. Never expand EXL3 K2 and call it
native. Callers supply *verified* native decoded tensors and retain trained QDQ,
routing, norms and residual semantics outside this component.

## Contract

`ExpandedExperts(gate, up, down, rows, top_k=3)` owns all scratch, accepts
`x[rows,H]`, expert IDs `[rows,3]` int32, routing weights FP32, and writes an
FP32 `[rows,H]` channel-shard partial. Weights are contiguous output-major
`gate/up[E,J,H]`, `down[E,H,J]`; both BF16 and FP16 are explicit profiles.
Current channel split: H=5120, J=1152, E=128. No counts or IDs are read by CPU.
Inputs and weights have the same 16-bit dtype; GEMM outputs and accumulation are
FP32. Default `trained_qdq=True` preserves 32-wide FP8 E4M3 activation QDQ with
power-of-two scales, native BF16 gate/up rounding, clamp/SwiGLU, routing weights
before BF16 activation rounding, then down-input FP8 QDQ. Software conversion
uses SM86 FP32 arithmetic, not unsupported FP8 instructions. `trained_qdq=False`
is an explicitly different expanded arithmetic control.

`run()` returns the unrounded channel-shard diagnostic. For native TP2, use the
per-route FP32 down partials in `state[15]`: sum corresponding partials across
both GPUs, then call `combine_after_tp(ids)` to round complete down outputs to
BF16 and add them in ascending expert-ID order, matching the official MoE. Do
not round individual rank partials before that collective. N1 owns integration.
Expanded FP32 dot accumulation order can differ from official per-32-K scaled
FP8/FP4 MMA; numerical drift must be classified by V1. The release values and
trained quantization operations are retained.

Device-only stable segment construction groups routes in expert-ID then original
row/slot order. Compact storage is O(rows*top_k), not E times that. Invalid IDs
poison outputs with NaNs and set device `status`; never substitute expert zero.
Caller must check status at its existing reply boundary. Workspace is per instance,
stream ordered, non-reentrant; do not overlap calls on different streams.
No allocations or host reads occur during run/capture; instantiate and warm first.

CUTLASS uses SM80-class MMA supported by SM86, compiled *only* for sm_86, with
FP32 accumulator/output. Device-only scheduler requires no host problem sizes.
Small-row alternative uses explicit FP32 CUDA-core reductions. Select mode at
setup, never autotune in production.

Dense API uses the same grouped tensor-core kernel for output-major weights
`[groups,N,K]` and input `[groups,M,K]`, FP32 output. Group=1 covers main_proj,
attention and vocabulary shard. wo_a uses group=4, K=4096, N=1024. Head widths
64640 (balanced native split), 64768/64512 (existing 256-tile EXL3 split) must be
measured separately; existing TP2 is *not* an exact half split.

`HeadEpilogue.run(base, bias)` is the native default: FP32 base logits plus an
exact FP32 cached/computed Markov row, followed by FP32 argmax. There is no BF16
rounding in this epilogue; ties select the lowest global ID.
`run_engine_bf16_control(base, bias)` explicitly selects the old engine arithmetic:
BF16(base), BF16(bias), BF16(sum), then argmax with the same tie rule. This control
can choose a different token at near ties and is not the official native head. It returns
local winner and value for the existing TP2 collective; confidence remains with
its owning model, with original feature order and FP32 dot.

## Donor provenance

CUTLASS 3.9.2: `ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e`, reviewed
`examples/24_gemm_grouped/gemm_grouped.cu`, `gemm/device/base_grouped.h`, and
`gemm/kernel/default_gemm_grouped.h`. Use headers only; no upstream launcher or
installation. FlashInfer SegmentGEMM reviewed as an alternative; not installed
and no performance claim. Preserve CUTLASS's upstream BSD license in the donor.

Run commands, numerical limits and measurements will be added after validation.

## Integration (N1)

Initialize the extension with `build.load_extension(DONOR, BUILD)` under the lane's
GPU lock for offline tests. Deployment builds are separately owned by N1/coordinator.
`run_slots(x, ids, rw)` projects and returns preallocated FP32 `[T,3,5120]` in
original row/slot order, with no early BF16 down rounding. Reduce these slots
across TP2, round each expert result to BF16, then sum in ascending expert-ID order;
add the shared expert last. `project()` instead leaves down partials in grouped
order in `state[15]`, with `state[2]` mapping original route to grouped route.
`combine_after_tp()` is provided for that grouped layout. All layouts are stable
under replay when routing changes; do not reuse outputs until consumers finish.

`input_is_qdq=True` skips redundant input QDQ when N1 already supplies the exact
native QDQ tensor. It does not skip the down-input quantizer. Gate/up are separate
contiguous arrays to allow direct streaming into their final shards; avoid a
second stacked w13 copy during load. Shared experts use the dense component with
the same clamp/round/QDQ ordering; they are not routed or added by this component.

Dense always returns FP32. The model applies BF16 output rounding on replicated
projections, and after TP reduction on channel-sharded wo_b/shared-down outputs.
Dense owns a stable input buffer, so run includes its copy; QDQ is caller owned.
The native BF16 head must not use FP8 activation QDQ. Its logits can feed
`HeadEpilogue` with either an exact cached Markov row or a computed row. Confidence
uses the model's unmodified original features and remains outside this API.

FP16 weight admission does not admit FP16 embeddings/router/Markov tensors or
arbitrary activations. N1's inventory says every native FP4/FP8 decoded weight is
FP16-exact, while some BF16 tensors are not. BF16 is the default native profile.
An FP16 execution profile must separately prove activation/QDQ equality and keep
native BF16 intermediate rounding; the benchmark checks its actual input QDQ.

`GroupedWoA` consumes `[T,groups*4096]` directly and writes
`[T,groups*1024]`, using strided CUTLASS pointer/leading-dimension metadata. It
avoids grouped-bmm transpose/materialization passes and keeps the model's flat
layout. No input/output transpose is hidden in its timing.

## Static dispatch

The default `auto` profile is a fixed setup-time table, not an autotuner. Experts
use the dedicated FP32 warp kernel for the measured 5120/1152/top-3 shape at
1–3 rows; otherwise grouped CUTLASS. Real live T is5/10/15/20, so the ordinary
native drafter selects grouped. The all-distinct synthetic case can favor the
dedicated kernel slightly; no per-layer CPU count read or dynamic host dispatch
is introduced to chase that case.

Dense `auto` uses the installed Torch2.10 cuBLAS FP32-output API for measured
BF16 main_proj rows 8/32/128; wq_a, wkv, wo_b rows 5/10/15/20; and flat grouped wo_a
rows 5/10/15/20. wq_b and vocabulary shards retain CUTLASS. Unmeasured shapes
retain CUTLASS. Explicit `cutlass` and `cublas` overrides remain available for
V1 review. cuBLAS setup refuses reduced-precision-reduction flags; the owning
model must set them false. It uses preallocated outputs and captures successfully.
No EXL3 operation is used as an implicit fallback in a native profile.

## FIX1 head regression probe

`probe_head.py DONOR BUILD CHECKPOINT OUTPUT CAPTURE [CAPTURE ...]` runs under
`run_locked.sh` on X86host's GPU-7744b44b only (14-minute acquisition bound).
CAPTURE is a V1 reference dump with `head_hidden` and `variants.own.prev`.
It reads native safetensors directly, computes FP32 Markov rows from the actual
preceding tokens, and checks exact `(value, global ID)` equality to Torch FP32
`(base + bias).max(-1)`. It covers both halves of the balanced 64640 split and
the 64768/64512 split, with native CUTLASS logits and official FP32 linear logits,
both policies and graph replay. Constructed near ties at every width prove that
the default preserves FP32 distinctions and exact ties pick the lowest ID.
Keep all captures/weights and probe results on X86host/Controller outside git.
