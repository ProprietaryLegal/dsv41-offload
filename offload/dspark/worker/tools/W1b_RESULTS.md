# W1b measured results and reproduction

The current worker has now been profiled and both donors executed on x86host's
GPU-7744b44b at the unchanged325-W cap. Neither donor nor decoded-weight
materialisation qualifies an exact replacement. The only retained implementation
is optional captured reply packing, with about0.006–0.012ms saved per backend
call. That is well below the campaign's~5% promotion threshold; no service change
or deployment is recommended from these isolated results alone.

## Current baseline

See W1b_PROFILE.md for all conditions and canonical B1–4 numbers. There are80
cases in each of two runs (burst/42-ms cadence),3200 measured samples per run,
plus warmups. Graph/eager replies agree. Timing separates H2D, insert, draft,
packing and required D2H wait, using paired CPU timestamps/CUDA events. The
separate CUPTI kernel totals are a diagnostic, not a critical-path subtraction.
Full expert row-count vectors and Markov hit vectors are retained per case.

## Numerical gates

Trellis decodes all23 real K5/K6 matrices exactly, including the full129280-wide
head, through actual identity-row GEMMs. K2 is unsupported: all1152 expert
matrices are explicitly excluded, never timed through fallback. cuda-exl3 scalar
and actual GEMM vector dispatchers decode all1175 matrices exactly. Every weight
is checked against the installed decoder, with independent CPU corner tiles on
every shard; no decoded tensor is narrowed to BF16.

The op tests compare both donors with the current operators and an independent
CPU-FP64 reference containing both128-wide Hadamards and each shard's scales.
Trellis has92 and cuda-exl3 has200 row1–4 op cases. All outputs are finite;
cuda-exl3's maximum relative Frobenius error versus FP64 is0.000607. Small
FP64 errors do not mean identical SM86 accumulation. main_proj controls were
repeated with its actual FP32-accumulation extension and prescale; those corrected
four-row records supersede the main_proj entries in the original op files.
Full-worker tests always retained main_proj's required arithmetic.

Full-step mismatch cases below are out of160 (80 saved cases ×eager/graph),
including B1–4, n_out1–5, hit/miss and both insert sizes:

| donor / replacement | draft ID mismatch cases | confidence bit mismatch cases |
|---|---:|---:|
| trellis small dense | 82 | 156 |
| trellis all stage dense | 46 | 156 |
| trellis head | 20 | 16 |
| cuda-exl3 small dense | 72 | 156 |
| cuda-exl3 all stage dense | 72 | 156 |
| cuda-exl3 head | 20 | 16 |

The standalone trellis build is bounded to<=32 rows; full-step qualification
captures8/16/32 insert buckets for the saved matrix's<=24 real rows. Its initial
128-row insert warmup correctly refused an unbuilt family. Wider insert dispatch
is not qualified. No donor is exposed as a production switch.

## Ranked operation screens

Rank follows the new baseline's critical-path candidates; gains below are direct
paired measurements at the stated op boundary, not additive whole-worker gains.

1. **Routed experts:** current kernel diagnostic0.695–2.077ms over B1–4. The
   standalone cuda-exl3 grouping experiment includes GPU sorting/gather, distinct
   gate/up scales, clamped SwiGLU, down, scatter and fixed-order top-k combine.
   At1/2/3/4 distinct input rows it adds0.315/0.664/1.163/1.374ms per stage versus
   the current coop path, and changes outputs. Static synthetic group counts are
   prepared outside capture; this is not a dynamic routing implementation.
   Populated/shared patterns and full query-row counts5/10/15/20 were also tested.
2. **Dense projections:** trellis small wkv/wq_a saves~5–7us/op at rows1–4,
   but fails full-step parity. Installed EXL3 bundled shared gate/up saves
   10.3/9.8/9.8/11.3us for1/2/3/4 rows; outputs differ. Neither is retained.
3. **Head/Markov boundary:** trellis full-width head is~0.11ms slower at rows1–4;
   cuda-exl3 is~0.025–0.049ms faster in that op screen, but both head-only
   replacements fail full-step parity. Existing exact full-vocabulary Markov
   add/reduction, BF16 rounding, lowest-ID ties and confidence inputs are retained.
4. **wo_a/cast/QDQ:** six grouped tile variants across5/10/15/20 rows preserve
   output bits in these samples, but do not offer a consistent improvement.
   AtB1 they add1.55–24.58us per stage. Required operation ordering is unchanged.
5. **Window128 attention/mHC/norm:** tested20 attention,12 norm and12 mHC
   variants. Faster attention variants change outputs; mHC split variants change
   sums or slow down. Norm8-warps saves~0.46–1.06us/op in the B1/B4 samples;
   no full-step win beyond noise was established, so no switch is added.
6. **Materialisation:** all four actual budgets and full-draft timings are in
   MATERIALISATION_BUDGET.md. Tiny dense subset: -0.002/+0.012/+0.018/+0.050ms
   B1–4 but confidence drift in all four. Expanded subsets and head are slower.
7. **Reply packing:** exact, small positive result below, retained default off.

## Optional captured reply packing

`DSPARK_GRAPH_REPLY=1` makes server.make_core capture packed float64 reply twins
and GpuBackend use them. Default0 retains the original path. A missing packed
graph raises, rather than silently falling back. Draft arithmetic, metadata upload,
plain/margin layouts, synchronous D2H completion and NumPy wire conversions stay
the same. The original graph API is retained for diagnostics. Capture count grows
from45 to85 (40 reply twins); observed process use8286MiB, with the2-GiB Markov
cache still resident. No GPU service is changed by setting this in the source.

Backend timing includes real metadata allocation/upload, graph input copies,
replay, reply staging, synchronous D2H and NumPy conversions. It excludes insert,
transport and frame handling. Three interleaved trials,40 measured pairs each,
alternating baseline/candidate order; cadence sleeps occur outside timing.
Numbers are the median of the three **within-pair median savings**, not the
difference of independent medians:

| requests / n_out | burst saving ms |42-ms cadence saving ms |
|---|---:|---:|
|1 /5 |0.0066 |0.0089 |
|2 /5 |0.0070 |0.0115 |
|3 /4 |0.0074 |0.0097 |
|4 /4 |0.0059 |0.0077 |

Cadence trial medians range0.0008–0.0204ms; the signal is small and noisy.
All24 final trial medians are positive. Final screen checks80 distinct byte-exact
reply cases (160 backend calls), including margins/cache modes. GPU tests also
cover consecutive anchors, all n_out prefixes, mixed per-request depths and
missing-graph refusal. Focused final suite:102 passed, no skips.

The coordinator owns deployment. To test promotion, set DSPARK_GRAPH_REPLY=1 in
an isolated worker's environment, run the same serve workload and check complete
round latency/acceptance/memory. Rollback is unset/the value0 plus a coordinated
worker restart. These instructions do not authorize this lane to restart a unit.

## Reproduce on the assigned test card

Source ~/.config/ndm/paths.sh; set CUDA_VISIBLE_DEVICES to
GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc; require~/dsv41-offload/W1_GPU_OK.
Use the existing~/dsv41-offload/o2/venv/bin/python, cwd
~/dsv41-offload/w1/repo/offload/dspark. The guarded profiler commands are in
W1.md. For subsequent phases use:

```
python -m worker.tools.build_donor trellis DONOR BUILD
python -m worker.tools.build_donor cuda-exl3 DONOR BUILD
python -m worker.tools.qualify_donor decode DONOR_KIND ROOT DONOR BUILD OUT.jsonl
python -m worker.tools.qualify_donor op DONOR_KIND ROOT DONOR BUILD OUT.jsonl
python -m worker.tools.qualify_donor full DONOR_KIND ROOT DONOR BUILD OUT.jsonl --baseline BASELINE --subset small
python -m worker.tools.screen_ops reply ROOT BASELINE OUT.jsonl
```

Use fresh outputs. `screen_ops` also has experts, mgemm, wo, misc and material
phases; experts requires --donor-root/--build. Each run must remain detached,
record assigned-card clocks/power and stay under20GiB. Existing immutable runner
scripts and all raw measurements/inputs remain on x86host under~/dsv41-offload/w1/.
Campaign evidence and the final report are in Controller's DSV41-OFFLOAD directory.

Pinned donor/architecture details remain in DONOR_AUDIT.md and donors.lock.json.
The original cuda-exl3 MLA build failure (SM89 FP8 instructions), corrected
vector-probe failure, bounded trellis insert-family refusal and diagnostic case
count correction are retained in raw evidence. No package/engine upgrade, Spark
access, production traffic, service/power/clock change, email, push or merge.
