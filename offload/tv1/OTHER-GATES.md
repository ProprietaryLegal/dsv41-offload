# Item 2e — no additional exact >=2% B1 candidate established

The explored remaining mechanisms have no justified exact >=2% B1 reduction:

- SGLang #42864: more SM120 split-K parallelism and tree rather than serial slice
  reduction. Explicit SM120-only guard excludes SM121; new rounding order. No
  bitwise-old-path oracle, and our mHC path already uses different overlap/boundary
  kernels. No exact port justified from the measured bucket alone.
- #42867: BF16 skinny-head batch invariance; our target head uses quantized EXL3
  mrow. Native drafter divergence belongs to NB1. No new target-work reduction.
- spark-ds41f dead-row expert skip: already implemented as our depth policy; not
  compact rows. Existing implementation is default OFF. No duplicate switch.
- spark-ds41f fused block rejection: different sampler, no measured material share
  to justify exact FMA/reduction reconstruction here. Our standard sampler stays.
- RS1 paged-direct candidate scoring is item 2a. Its DCP/NVFP4 implementation and
  upstream 1M performance are not a second independent feature.

This decision does not dismiss broader future work. It refuses to describe a
format/reduction/sampler change as an already exact >=2% optimization without the
required old-path oracle and a local time budget. Nothing beyond items 2a/2b was
built into the engine; item 2c found no removable production indexer syncs and item
2d failed its explicit build gate.
