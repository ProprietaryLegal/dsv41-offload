# PF1: what the cap comparison proves

Status: CPU source audit, **cause unresolved**. A cap-dependent reservation is present;
a cap-dependent repeated prefill sweep has **not** been found. Do not promote a purported
35% fix on this evidence. Sources are local pinned copies, not current upstream behavior.

Reproduce the numeric evidence (no requests, GPU imports or network):

```bash
source ~/.config/ndm/paths.sh
"$HARNESS_PYTHON" -m offload.prefill_maxlen.evidence \
  --results /opt/dsv41/repos/dsv41-deploy/offload/results/K1 \
  --output /opt/dsv41/campaigns/DSV41-OFFLOAD/PF1-evidence.json
```

## Recorded measurements

S18's final report is supported by the raw cold, uncached B1 requests. At 32K, compare
rep 1 in both arms (R1 rep 0 has first-use loading); at 150K use both reps' median.

| MEASURED proxy | R1, cap 327680 | R5, cap 524288 | Extra per 4096-token equivalent |
|---|---:|---:|---:|
| 32000-token TTFT | 14.551026 s | 19.759385 s | 0.666670 s |
| 150000-token TTFT | 68.923200 s | 93.355367 s | 0.667161 s |

These are whole-request TTFTs normalized to 4096 tokens, **not measured chunk times**.
The deltas are 0.162761 and 0.162881 ms/token. They do not prove a linear dependence
on the cap; there are only two caps and one post-warmup 32K sample per arm. An intermediate
393216 cap predicts 16.287145 / 77.067256 s at 32K / 150K **if** linear (ESTIMATED).

Both verify logs record K1 overlay `559c5867ba42`, 13 files, identical five switches.
KV lines record 2502600 / 2820756 capacity tokens at the two caps, group sizes ending
in 192, LCM 384. Capacity tokens are not prompt lengths. R5 load memory is 92.34 GiB
vs R1 92.24 GiB. The raw `extra/lines-*` records establish the load difference but
contain no kernel timing. S18 records equal clocks and unchanged decode; no new GPU
measurement was made here.

## Active call path and cap-sensitive objects

Paths below are relative to `/opt/dsv41/data/src-image`, unless marked K1.
The executed family is `vllm/models/deepseek_v41`, **not** its v4.0 sibling. K1's
`overlay/.../nvidia/model.py` retains the replay cut: full chunks in layers 0–19,
full KV write at 20, then the final 128 rows through 20–39. Top-k source layers
share indices with consumers; do not count 40 independent full-row scans.

`deepseek_v41/attention.py` (K1 1507–1563) passes `cap/ratio` and
`factor*cap/ratio` into `SparseAttnIndexer`. In
`model_executor/layers/sparse_attn_indexer.py:566–704`, the prefill path gets shared
K/scale workspace views at the reserved size, then **slices to
chunk.max_local_total_seq_lens** before gathering and scoring. The metadata builder
`v1/attention/backends/mla/indexer.py:1470–1567,1775–1916` derives those rows from
the current requests' CPU sequence-length upper bounds (exact for prefill).
Its 128-MiB logits limit determines query sub-chunks. Query continuations reuse
the same gathered keys. MQA and top-k get these current extents, not cap.

| Cap-sensitive object (ESTIMATED bytes, per rank unless stated) | 327680 | 524288 | Recurring cost per 4096 at each cap |
|---|---:|---:|---|
| Shared indexer K/scale reservation, factor 4, ratio 1, 132 B/row | 165 MiB | 264 MiB | View construction only; no full-buffer memset/copy in Python. Actual gather/score unchanged for matched chunk. Proven cap-extra sweep: 0 / 0. |
| Ratio-2 reservation (same shared owner, not additive to ratio-1 peak) | 82.5 MiB | 132 MiB | Same reasoning. |
| Two FP32 RoPE caches, 64 columns, cap+2048 positions | 161 MiB | 257 MiB | Constructed at load. Runtime kernels index current positions; 0 / 0 extra full-cache sweeps. |
| Expanded indexer block table, per ratio-1 builder, 4096 rows × ceil(cap/64) × int32 | 80 MiB | 128 MiB | Initialized once. Filled on decode; no use in pure prefill. Do not multiply by model layers without builder-sharing accounting. |
| Request all-token IDs, four seats × cap × int32 | 5 MiB | 8 MiB | Admission writes actual prompt; input extraction reads scheduled tokens. 0 / 0 extra cap-row copies per chunk. |
| Per-group input block tables (two device copies; width depends on group spec) | ≤2×4×5120×4 B per 64-token full-length group | ≤2×4×8192×4 B | Slot mapping uses current positions. Reorder can copy table columns; tens to hundreds of KiB, not a full KV/window scan. |
| Sparse SWA/FlashMLA prefill workspace formula | `ceil(cap/ratio)+128+4096`, times seat/chunk budget | same expression at larger cap | **Inactive on SM121 FlashInfer path**. Its Python chunk plan returns actual widths anyway. |
| Generic `flashmla_sparse.py` workspace, 5×cap×576×2 | 1800 MiB | 2880 MiB | **Inactive backend**; do not attribute to served V4.1 FlashInfer. |
| Paged MQA logits maximum output width, cap/ratio | cap/ratio | cap/ratio | **Decode only**, not the prefill MQA call. |

The 80/128-MiB expanded-table numbers assume a ratio-1 64-token kernel page.
Other builders' widths follow their own cache specs. A native implementation may
inspect strides/shapes for dispatch; its behavior cannot be inferred just from the
Python wrapper. The reference tree lacks the image's compiled cache-op, DeepGEMM
and FlashInfer native source/binary timing, so this audit cannot certify every
native loop bound. `cp_gather_indexer_k_quant_cache` is an opaque C++/CUDA call
here (`vllm/_custom_ops.py:3071`), with actual-sized destinations as arguments.

## Why the proposed buffer explanation does not quantitatively close

At an illustrative **ESTIMATED** 225 GB/s (SP1 budget assumption), even a nonexistent
extra entire factor-4 sweep would cost 0.769 / 1.230 ms, a delta of **0.461 ms**,
not 667 ms. Twenty such extra sweeps would add 9.23 ms. Explaining the observed
delta this way requires about 1447 extra sweeps per chunk, or approximately 150 GB
of additional traffic at that assumed bandwidth. No such sweeps appear in the
Python call path. This calculation is a plausibility check, not a bandwidth bound
on unknown native kernels or evidence against allocator/TLB effects.

MoE scratch under 0076d/e is sized by actual routed rows/slots/dimensions, not cap
(`offload/kvpack/overlay/vllm_exl3_moe/tf_prompt.py:94–140`, `tf_experts.py`).
0076b removes a batch-sized residual buffer. Neither changes arithmetic with cap.
FlashInfer SM120 uses a constant shared 128-MiB workspace and actual sparse indices/
lens; its prefill wrapper has no cap argument. RoPE and block-table reservations,
workspace address/scale separation, allocator layout, native dispatch and page
placement remain plausible **unmeasured hypotheses**. Equal clocks do not prove
equal achieved memory throughput.

The `max_num_batched_tokens=4096` allocations do not change between R1/R5:
shared top-k indices are 4096×512×int32 = **ESTIMATED 8 MiB**; SWA's
4096×128 int32 prefill indices are **ESTIMATED 2 MiB** (per owning metadata
buffer); the disabled MTP residual would be **ESTIMATED 160 MiB**; TF's
transient/aliased prompt scratch is **UPSTREAM CLAIM approximately 750 MiB**
at a full TP2 4096-row call, plus its small persistent metadata. Indexer logits
are bounded by the configured 128-MiB budget and the FlashInfer workspace is
128 MiB. Activation/projection buffers scale with current rows, head count and
hidden dimensions. None supplies an extra cap-proportional per-token loop in
the inspected Python. These memory numbers are not additive peak RSS estimates.

There is **no defensible set of measured component costs summing to +0.667 s**.
The gap remains the full observed delta. The required next measurement is a
matched early full-chunk profile on both TP ranks, paired with actual M/N/stride,
kernel-dispatch and allocator observations. A narrow patch can isolate workspace
view layout; it must not be described as an established regression fix.

## Donor claims, separated from local evidence

Pinned donor `afe4d5e41ec6fa4db17c5cc88758af439a106913`, README and patches
0033a/b, 0036, 0051 and 0076: **UPSTREAM CLAIM** factor 4 changes no chunk splits
with four seats, TF transient/alias preserves bits, and the changes recover about
813 MiB on its setup. Donor 512K/1M needles and prefill rates are not measurements
of legal-v2a, the offloaded deployment or this patch. SP1's 1.90-s chunk allocation
is an illustrative **ESTIMATED** budget; its attention bucket is not a measurement
that can be subtracted from R5 TTFT.

Baseline CPU checks: K1's 13-file rebuild and manifest match; 27 existing stage/
arithmetic/config tests pass (their deliberate corrupted-file test prints FAILED).
