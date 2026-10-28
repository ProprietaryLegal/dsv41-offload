Status 2026-10-23: Numerical checks delivered; M20/M128 main_proj performance target unmet. Keep BF16 main_proj baseline; native whole-arm gates still apply. See [live roadmap](../../../../documents/plans/roadmap.md) and [measured record](../../../results/COMPARISON.md); campaign review decisions are in `LANES.md` / lane reports outside the repo.

# Native FP8 dense kernels (KX-FP8DENSE)

Experimental SM86-only E4M3/E8M0 block-scaled W8A16 dense GEMM. This lane
owns this directory only; N1 integrates, V1 reviews, coordinator deploys.

`DenseFP8.from_checkpoint(codes, full_scales, torch.device('cuda:0'),
row_offset=..., col_offset=..., groups=1)` consumes an official rank shard
and the FULL tensor's scale grid. Cuts must align to 32x32 blocks. For
`wo_a`, pass four consecutive groups as `[4096,4096]` with `groups=4`;
input is `[M,16384]`, output `[M,4096]`. The load prepack permutes bytes
into BF16 MMA fragment order without expanding storage. Scales remain
E8M0 bytes. Admission checks every present code/scale pair on the CPU;
NaNs and nonzero BF16 subnormal/overflow values fail loudly. Runtime
weight mutation is unsupported.

`linear_into(xq, out, scratch=..., splits=..., row_tiles=...)` takes BF16
input already quantize-dequantized with N1's `formats.act_qdq`. It never
allocates, synchronizes, compiles or reads device values on the host.
Outputs are BF16 (RNE) or FP32 channel partials. `splits>1` requires stable
caller-owned FP32 `[splits,M,G,N]` scratch; reduction is ascending split
order without atomics. All kernels use the current device stream, no
process-global GPU pointers. Construction/build must precede capture.

FP32 accumulation order may differ from N1 Triton and the FP64 oracle;
qualification must classify BF16 midpoint cases and FP32 relative error.
No fast-math, native FP8 MMA, TF32, FP16 accumulation, or service dependency.
Build uses the existing torch extension environment with explicit sm_86.

Offline execution on x86host only:

```bash
source ~/.config/ndm/paths.sh
export KX_FP8DENSE_ROOT=$HOME/dsv41-offload/kx-fp8dense/repo
"$KX_FP8DENSE_ROOT/offload/dspark/kernels/fp8dense/run_locked.sh" \
  -m offload.dspark.kernels.fp8dense.screen --output /tmp/fp8-screen.json
```

The runner pins GPU-7744b44b, acquires its exclusive flock, and bounds work
to 830 seconds plus <=5 seconds teardown. Release between sub-items.
Recorded timing includes the linear/reduction, excludes activation QDQ,
prepack/load and model/network/collectives; no engine speedup is implied.

For the measured register dispatch, obtain `(splits,row_tiles) =
weight.configuration(M)`, allocate scratch `[splits,M,G,N]` if needed, and
pass both explicitly to `linear_into`. The default remains split=1 and
requires no scratch; it is a correctness control, not the tuned path.

`layout='simt'` is a separate experimental raw-byte tiny-M control (M<=20,
K%128=0, E8M0<=134, 8-byte-aligned input rows); it avoids tensor-tile padding.
Do not choose a control based on isolated synthetic numbers without reviewing
its all-stage/both-rank report. The tiled Triton files are offline controls,
not automatic production fallbacks.

Qualification at implementation commit 989d99c: 198 official-tensor cases,
all three stages/both rank cuts, every required M. Minimum BF16 agreement
with NativeDense is 99.9414%; all candidate FP32 relative L2 errors versus
CPU FP64 are <=5.535e-7 (FP32 partial operations <=8.554e-8). This is an L2
bound; pointwise relative 1e-6 fails at cancellation. Disagreement locators,
values, reference sums and BF16 midpoints are retained in the campaign JSON.
Final focused suite: 19 passed. Full native domain census: 25 tensors,
564,658,176 values, 1,679 present pairs, E8M0 115..121, all exact BF16.

Selected register timings below are median across stages/rank cuts, us,
locked GPU-7744b44b, unchanged 325 W cap, clock settling + CUDA graphs:

|Shape|M5|M20|
|---|---:|---:|
|wq_a|11.95|16.72|
|wkv|7.04|8.63|
|wq_b|27.62|38.63|
|wo_b FP32|30.59|40.99|
|shared w1|10.39|15.49|
|shared w3|10.39|15.67|
|shared w2 FP32|9.06|15.41|
|four-group wo_a|25.06|34.63|

Main projection M8/32/128: 98.07/171.03/569.22 us. Separate raw-byte Triton
control reached 394.65 us at M128, still slower than the supplied 328.76 us
BF16 cuBLAS baseline. Neither qualifies for the 110 us M128 target. The
isolated linear median sums are 394..399 us per rank across the three stages
at M5, above the 350 us request. The requested performance acceptance is
**not met overall**. Do not promote this component as a qualified full-model
speedup. N1 owns selection/integration; V1/coordinator own whole-drafter and
engine review. Detailed report/evidence:
`/opt/dsv41/campaigns/DSV41-OFFLOAD/KX-FP8DENSE.report.md`.
