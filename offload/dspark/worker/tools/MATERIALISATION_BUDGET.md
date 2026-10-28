# W1 selective decoded-weight budget

Planning arithmetic from the actual S2 `TENSORS.json`; not a CUDA allocation or
performance measurement. Inventory evidence is local to the campaign at
`W1-evidence/tensor-memory.json`. Store decoded codebook values in **FP16**:
silently narrowing those values to BF16 is not exact materialisation.

| Tensor group | Matrices | Packed trellis MiB | Decoded FP16 MiB |
|---|---:|---:|---:|
| full K6 head | 1 | 473.438 | 1262.5 |
| K2 expert projections | 1152 | 3240 | 25920 |
| K5 stage dense | 21 | 229.688 | 735 |
| K5 main_proj | 1 | 46.875 | 150 |

Packed columns exclude suh/svh/pointer tables, which must remain in the budget.
These numbers come from shape × dtype bytes, not an estimate from nominal bpw.
The full expert expansion alone is 25.3125 GiB, larger than the entire 3090.
The full head expands bandwidth demand by 2.667× relative to its six-bit trellis.

## Process and card envelopes

The recorded production worker budget is **8152 MiB** with 45 CUDA graphs,
including the ~2020.5 MiB Markov cache, weights, windows, graph pool, CUDA context
and module overhead (see worker/MEMORY_BUDGET.md). A read-only W1 observation
showed **8172 MiB** already used on the assigned card; this is a card-total
snapshot, not a newly measured test-instance footprint.

The brief caps W1's own process below **9216 MiB**, independently of card capacity.
For screening, reserve an additional 15 MiB device staging (512 × 15360 BF16
elements), 64 MiB bounded transform/GEMM/repack workspace, and **512 MiB safety
margin** beyond the recorded base. The existing 15 MiB pinned *host* staging is
separate. No full-table registration or additional Markov cache is budgeted.

| Candidate, retaining original packed tensors | Extra FP16 MiB | Planning total incl. reserves | Under 9 GiB? |
|---|---:|---:|---|
| three wq_a + three wkv | 52.5 | 8795.5 MiB | yes |
| above + all shared w1/w3/w2 | 255 | 8998 MiB | yes |
| above + main_proj | 405 | 9148 MiB | narrowly; measure real allocator peak first |
| all stage dense + main_proj | 885 | 9628 MiB | no |
| whole decoded head | 1262.5 | 10005.5 MiB | no |
| one full expert in each stage | 202.5 | 8945.5 MiB | yes; routing benefit unknown |

The margin is a reservation, not permission to allocate up to 9216 and then add
more scratch. Observe process footprint and peak allocations after all capture,
staging and op warm-up; fail closed rather than shrinking the margin silently.
Temporary FP32 decoded copies cost twice the FP16 column and are not permitted
implicitly. Decode slices without retaining a second full decoded copy.

With a live ~8172 MiB worker and a maximum 9216 MiB W1 process, combined use is
~17388 MiB, leaving ~7188 MiB of a nominal 24 GiB card before other owners. This
does **not** relax the separate W1 ceiling or establish current free memory:
query assigned-UUID free/total memory immediately before starting. Only one W1
model instance should exist; compare candidates sequentially against saved
inputs/replies. Do not load baseline and candidate models together inside W1.
Host had 304 GiB available and zero swap used at initial inventory; recheck free
RAM before loading. Pinned staging remains bounded at 15 MiB per test backend.

Trellis K6 repacking itself changes the full head to eight resident bits per
weight: **631.25 MiB**, +157.812 MiB if replacing the compact trellis, or +631.25
MiB if retained alongside it. Keeping both in the full worker exceeds the above
screening envelope. A head-only standalone op experiment can use a much smaller
process. Any later full-step replacement must release all old tensor/LinearEXL3
references before capture; allocator reservation and stale graph pointers still
need checking. No production weight is deleted or altered.

## Numerical and speed gates

Materialising the exact rotated-basis FP16 codebook tensor removes decoding;
it does not restore lost quantisation precision. Preserve both 128-wide
Hadamards, per-shard suh/svh, input FP16 conversion, main_proj power-of-two
prescale and its FP32 accumulation. Using a dense GEMM can change accumulation
order relative to our existing kernels even with identical stored weights.

Screen the 52.5/255 MiB subsets first only if the new profile confirms fixed
small-GEMM overhead matters. Compare complete transforms + matmul + downstream
casts, not only GEMM throughput. Retain solely measured full-step winners with
draft/confidence parity and a default-off switch. Every B1–4 saving and every
materialisation parity result is **pending** because the GPU window is closed.

## W1b measured update — second card, <20 GiB ceiling

W1b replaces the original 9-GiB lane ceiling with 20 GiB on GPU-7744b44b.
The live production worker remains on c4e7a371, so its ~8.1 GiB is not subtracted
from this card. W1b uses one model at a time and retains the original packed
weights. Host RAM was301 GiB available / swap0 immediately before the screens.

Actual sequential allocations and paired complete-transform/full-draft screens:

| subset | additional FP16 decoded MiB | maximum process MiB | full-draft GPU saving B1/B2/B3/B4, ms | decision |
|---|---:|---:|---|---|
| wq_a+wkv, all stages | 52.5 | 8550 | -0.002 / +0.012 / +0.018 / +0.050 | reject: all confidence vectors differ |
| above + shared w1/w3/w2 | 255 | 8768 | -0.166 / -0.148 / -0.198 / -0.013 | reject: all confidence vectors differ |
| all stage dense | 735 | 9248 | -0.563 / -0.478 / -0.531 / -0.102 | reject: slower and confidence drift |
| head alone | 1262.5 | 9730 | -0.920 / -0.931 / -0.937 / -0.961 | reject: much slower |

These are medians of within-pair CUDA-event savings, at canonical n_out5/5/4/4,
synthetic anchor0 and full128-row context. The four canonical head cases have
identical drafts/confidences; this is not the full80-case parity gate. The other
subsets fail exactness even where their draft IDs happen to agree.

Storage is exact rotated-basis FP16, reconstructed by the verified installed
kernel, preserving both Hadamards, each matrix's suh/svh, FP16 input conversion
and FP32 output. main_proj and its prescale/FP32 accumulation are retained.
cuBLAS accumulation changes the result relative to the current compressed GEMM;
exact stored weights alone do not establish op identity. No decoded BF16 weights,
second model, whole-expert expansion or unbudgeted FP32 weight copy was used.

All measured peaks leave over10 GiB below the20-GiB ceiling. Retaining a512-MiB
safety reservation,15-MiB device staging and64-MiB workspace remains within that
ceiling. Graph pools and CUDA module/context overhead are included in the
nvidia-smi process peaks; allocator peaks are also recorded separately. The
full production Markov cache remains resident. The reply-packing screen separately
measures85 graphs (45 original plus40 reply twins) at8286 MiB process use.
Any deployment would need its own peak check, but no materialisation candidate
is being proposed. Raw per-sample pairs and active telemetry are on x86host under
`w1/results/screen-material-{small,shared,dense,head}*`.
