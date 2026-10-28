# Native packed MXFP4 SM86 screen

A standalone register-fed kernel is implemented and tested. Native checkpoint
values survive packing exactly in the admitted BF16 domain. Model integration,
TP communication, trained activation QDQ, captured-tap fidelity and target
acceptance are separate qualification gates owned by N1/V1/coordinator. Default
enabling remains off (`dispatch.json`). No service or target was changed.

## Provenance and exactness

Official 1017 revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, N1-acquired
shards 44/45/46, locally rehashed independently:

| shard | SHA256 |
|---|---|
|44|3e8ac315c2bcca3e83c5e6e25c5758a75c03732063a6ff0c1b0e24b9ff49f6e6|
|45|9c73b27c4343236790008979d5e0b565f687aa64440031b9e096fed1ab8039c5|
|46|1dccc19a3741972c661de734572dc7d4b899e6e99bfbdf3ae4e057c707e5b339|

Full CPU census: all 1152 routed matrices, 13,589,544,960 native values,265
code/scale pairs, scale 111..127. All values exactly BF16 representable; zero
nonfinite, overflow or subnormal pairs. CPU census 69.46 s, 8 threads. Raw shape
and code/scale counts are in the campaign evidence, not model tensors.

Triton decode covers all 4096 E2M1/E8M0 pairs: exact BF16 bits including signed
zero, scale-zero subnormals, signed overflow infinity; scale 255 emits NaN.
Load admission explicitly refuses unsupported arithmetic, preserving source
values. Register decode covers all 16 codes × scales 111..127 (272 pairs, including
all 265 present pairs) with exact bits. Source code zero is excluded by the
register multiplier and is absent from the native census.

Projection checks use independent CPU FP32 and FP64 dot products. Maximum
absolute gate error vs FP64 is6.30e-7 for register-fed; all nine rank1 stage/
projection checks pass vs CPU FP32 (maximum7.16e-7). Arithmetic reduction order
is explicitly different from the official scaled-per32 dot; decoded-value
identity alone does not prove full model parity. Gate/up BF16 boundaries,
pre-down routing/QDQ, per-route TP sum, BF16 w2 boundary and ascending-expert
combination remain explicit in API.md.

## Timing boundary and conditions

2026-10-23, x86host development GPU7744b44b-8a13-b06d-a67e-f958fd74b8fc, exclusively
locked, one visible SM86 RTX3090, owner325W unchanged. PyTorch2.10.0+cu126,
Triton 3.6.0, nvcc12.6.85, explicit sm86, no fast-math/FMA contraction. Seven
CUDA-event graph samples, 20 pipeline iterations per sample, three warmups.
Stage 0 full 128-expert channel shards resident: gate/up K5120,N1152, down
K1152,N5120. Native bytes; seeded synthetic BF16 inputs and explicit spread or
three-expert-reuse routes. No real captured routing histogram claimed.

Pipeline includes device scheduling, gate/up, clamp/SwiGLU/routing, down,
FP32 slot combine and sticky-ID-status reset. It excludes trained activation
QDQ, native final w2 rounding/order, shared expert, attention, TP/NVLink,
head/Markov, network, Spark consumption and target verification. This is
neither full worker latency nor a TP2/accepted-token result. Telemetry snapshots
are taken after each timing run; short-run utilization/power can lag. The raw
samples record each snapshot, including clock, power, temperature and cap.

| rows/top3 | route reuse | Triton MMA staged µs | register staged µs | register fused gate/up µs |
|---|---|---:|---:|---:|
|1|spread|293.38|138.29|111.05|
|5|spread|1004.90|215.60|193.99|
|20|spread|3505.51|742.60|708.61|
|5|three experts|918.84|148.84|122.88|
|20|three experts|3509.86|168.60|152.47|

These separate screens did not lock clocks; comparison is directional, not a
paired engine speedup. Register snapshots: SM 1695..1950MHz, memory 9501MHz,
325 W cap; Triton snapshots: SM 1695..1935MHz. Fused register gate/up won every
screened shape and is the proposed explicit static choice. Three independent
projection calls per selected expert are never issued from Python: both
implementations schedule all route slots in device kernels.

Resident packed+scale bytes 1,203,240,960 (1147.5MiB) for all three matrices of
one stage on one TP shard. Register peak allocated 1,215,342,592 bytes
(1159.04 MiB), reserved 1,235,222,528 (1178 MiB). This includes qualification
workspace/transients and is not a complete worker or full-three-stage load
peak. Conversion streams one expert into its final shard, never expands a
whole bank. Decoded routed storage for all stages remains the separately
verified 25,920 MiB, 12,960 MiB per balanced GPU; packed is fourfold smaller before
scales. No pair allocated in this lane.

## What did not work

The original simple Triton row-per-route MMA wasted most of its16-row tile and
re-decoded weights for repeated experts. Its SIMT control was somewhat faster
at spread 20 rows (~3268µs) but still slow. Direct register decoding and reuse
removed most of that cost. We retain both controls and do not enable them
silently. The first exhaustive test also found malformed overflow bit patterns
in integer decode; corrected to signed infinity and retained load refusal.

Register fusion covers gate/up; activation and down are staged to preserve
explicit rounding/QDQ/TP boundaries. No giant fused whole-MoE claim, no Marlin
engine upgrade, no tensor download, no service/clock/power change, no Spark
access, push, merge, publication or e-mail occurred.

## Reproduce

Use the existing x86host lane-E venv, no new package installs:

```
export KX_MXFP4_ROOT=$HOME/dsv41-offload/kx-mxfp4/src
$KX_MXFP4_ROOT/offload/dspark/kernels/mxfp4/run_locked.sh -m pytest -q offload/dspark/kernels/mxfp4/test_mxfp4.py
$KX_MXFP4_ROOT/offload/dspark/kernels/mxfp4/run_locked.sh -m offload.dspark.kernels.mxfp4.qualify --regfed --output /tmp/regfed.json
$KX_MXFP4_ROOT/offload/dspark/kernels/mxfp4/run_locked.sh -m offload.dspark.kernels.mxfp4.native_identity --output /tmp/native-identity.json
```

CPU census (no GPU context) uses the same module with `--census`,
`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=8`. Run outputs belong in campaign
scratch. Do not run while the coordinator reserves the development card.

## Final verification

Final suite: **17 passed in 2.96 s** (two CPU tests, fifteen GPU cases), including
native BF16 final-projection rounding and ascending-expert combine order.
Exhaustive actual-native register-prepack/MMA reconstruction: **212,336,640
values, zero mismatches**, 36 complete matrices (stages 0/1/2, experts 0/127,
w1/w2/w3, both TP channel shards), 20.44 s. The one-hot sweep covers every input
column and output row of every selected shard, including both sides of every
scale-group/shard boundary. Fixtures exercise masked padding and graph input
mutation, invalid IDs, repeated experts beyond eight rows, and fused/staged
agreement. Source weights and client captures are not embedded in the report.

The measured binary SHA256 was
`1cac07cd830b833edf8cd1152c0f0bc9a22e1f0358295dd9ee0c0ac7da2b23ce`.
Final bounds-check host wrappers produce binary SHA256
`0cee7ada2a277fe00a2df06754290c41de8341539bd73adf99c69588b0f57bc0`.
**Their complete cuobjdump SASS is byte-identical**, SHA256
`3123ab07a8c834423199b963a7c99232d771bcabf99e4f2b5c27c0a3ff5d3ca3`.
Timing GPU code is unchanged. The final loader also drops transient source
references earlier; quoted memory peaks describe the measured screen before
that bounded staging-lifetime cleanup, not a new full-worker measurement.

Pinned Triton 3.6.0 CPU interpreter truncates FP32->BF16 casts in `cast_impl`;
it is unsuitable as a BF16-rounding oracle. An ordering-only CPU diagnostic
passed with already-rounded inputs; offline PTX compilation explicitly emitted
`cvt.rn.bf16.f32`, and the unchanged unrounded-input GPU regression passed.
No failing assertion was weakened and no dependency was patched.

All GPU work ran through `run_locked.sh` with the development UUID only. Jobs
queued during the coordinator's reservation, completed when the exclusive lock
became available, then exited and released it. No pair acquisition occurred.
