Status 2026-10-23: Native packed expert component delivered; original performance targets met in 4/6 cases only. N4 whole-arm pair/engine qualification remains pending. See [live roadmap](../../../../documents/plans/roadmap.md) and [measured record](../../../results/COMPARISON.md); campaign review decisions are in `LANES.md` / lane reports outside the repo.

# KX-FUSEDMOE

Experimental SM86 native MXFP4 routed experts for N1. Retains the KX-MXFP4
register prepack without requantization. A bounded device schedule groups up to
eight routes per decoded expert tile. The default uses CTA-local split-K2, FP32 compensation per64-product chunk,
and a high/low FP32 join retaining each partition correction.
Each warp owns 16 output channels; adjacent warps share the QDQ maximum across
32 channels. Gate/up rounds both projections to BF16, applies full-precision
expf SwiGLU and route weight, rounds to BF16, then performs official per-32 E4M3
activation QDQ.
Down writes FP32 rank partials to caller-owned SLOT storage; no BF16 down rounding.

No worker/service dispatch is changed. Input xq is already QDQ'd. Supported
T=1..20, top-3, int32/int64 IDs, explicit CUDA SM86 device. Every invocation
uses only caller-owned buffers, with no host reads or tensor allocations. Output
may be the first three slots of [T,4,5120]. Invalid IDs set sticky status bit 0
and poison only their slots with NaNs, without reading a weight out of bounds.
Status is caller-cleared outside the invocation. Buffers must not overlap and
must remain alive until their stream completes. Construct/prewarm outside capture.

The weight prepack and BF16 register decoder derive from KX-MXFP4/regfed.cu;
see ../mxfp4/{NOTICE,LICENSE.strata,donors.lock.json} for pinned Apache attribution.
No new dependencies. Compile only sm_86, --fmad=false, no fast math. Qualification
and exact reproduction commands/results are recorded in the campaign report.

## API and variants

```python
from offload.dspark.kernels.fusedmoe import FusedMoE, Workspace
# Existing native.kx.KXRegisterBank has gate/up/down RegisterExperts:
moe = FusedMoE.from_register_banks(bank.gate, bank.up, bank.down, variant=131072)
workspace = Workspace.allocate(T, 1152, torch.device('cuda:0'))
# status, xq, ids, rw, buf are preallocated on that same device.
moe.forward_into(xq, ids, rw, buf[:, :3], status, workspace)
```

Default 131072 passes the full native all-stage/all-rank numerical gates; speed
qualification is reported separately. Variant 16: four 16-channel warps per CTA, adjacent warps share only eight
amax scalars for exact 32-channel QDQ; ordinary FP32 MMA accumulation. Variant
64: same layout with Kahan FP32 accumulation of 64-product MMA chunks. This
changes only FP32 summation order and costs additional time. Neither mode is
an implicit fallback for the other. Variants 4/8 use larger 32-channel register
tiles; 32/128 use eight warps for the ordinary/compensated shared-amax kernels.
These controls remain available to reproduce the tuning screen.

Workspace is `(1+51*T)*4 + (3*T*1152)*2` bytes: 142,324 B at T20.
Weights retain exactly the original regfed prepack, 1,203,240,960 B per stage
per rank. No decoded-weight bank or new serving weight allocation is needed
when wrapping existing banks. Qualification holds an additional raw packed
oracle bank; that is offline test memory, not a serving requirement.

Scratch pointers and shapes are checked without device reads. Input/status/
workspace/output buffers must be disjoint. A workspace cannot be used by
concurrent independent streams. Prewarm both ID dtypes/variants outside capture;
keep status sticky until its caller explicitly clears/checks it. The exact-
official TP reduction, per-expert BF16 down rounding, ascending-ID combination
and shared expert remain N1's responsibility after this rank's SLOT outputs.

Fine compensation modes 4096/8192 use K2/K4 CTA partitions and compensate each
16-product MMA result, then join high sums and their corrections in FP32.
They preserve both BF16 rounding boundaries and the identical activation QDQ.
Mode 16384 uses eight partitions and two output warps per CTA, with a two-CTA
occupancy hint. All are explicit static choices, with the same caller workspace;
no host counts, runtime autotuning, FP64 arithmetic, or wider weight storage.
Modes 256/512 are plain K2/K4 accumulation; 2048/1024 compensate K64 chunks.
Native all-stage numerical results and full live-routing timing determine which
choices are qualified; a passing small synthetic test is not that qualification.


Mode 65536 shares decoded fragments across sixteen routed occurrences of an
expert (two B8 MMA tiles), skipping the second tile when empty. It retains the
plain split-K2 order of each route, and keeps three launches. Caller schedule
storage reserves seventeen integers per potential task for this optional mode;
other modes use the first nine per task. Mode 131072 preserves the low-order
K64 compensation of each K2 partition during the final FP32 join. These are
explicit profile options; qualification/speed results are in the lane report.
