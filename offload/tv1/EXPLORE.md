# TV1 exploration — reference mechanisms, our paths, bounded estimates

Verdict: build candidate scoring and compact verification as experimental OFF arms.
Do not build prefill graphs: measured idle is below the 5% gate. No donor launcher
was executed and no donor module will be imported or vendored. Donor revisions,
PR heads, complete diff hashes and SP1 numerical aggregates are in exploration-pins.json.

All engine references below are relative to `/opt/dsv41/data/src-image/`.
K1 replaces attention.py and model_runner.py; its overlay is the build base, never edited.
The seven PR diffs were read including their kernels and tests. SGLang main is not
assumed to contain the open PR heads. References to added open-PR code mean the pinned
diff's new file coordinates. The RS1 recipe was additionally shallow-cloned and its
experimental indexer implementation and tests inspected as reference only (AGPL).

## Candidate scoring: SGLang #43401 and RS1

SGLang `python/sglang/kernels/ops/attention/dsv4/fp4_indexer.py` in #43401
adds `_fp4_index_logits_candidates_kernel` near line 900. A persistent worker
iterates candidate blocks, expands eight positions per block, maps logical positions
through the request table and scores those positions. It first initializes visible
non-candidates to -inf. The same `_fp4_index_logits_tile` is shared with the dense
scan; this is why its bitwise test is meaningful. The test exercises new pages,
lengths, candidates, zero-length rows and captured replay, not just one static input.
The current tile (main lines 369–432) has BF16 dot/product/head-sum rounding; that
arithmetic is NOT our FP8 DeepGEMM arithmetic.

Our dense consumer is `vllm/model_executor/layers/sparse_attn_indexer.py:825–864`:
full paged logits, THEN candidate mask. Source layer 20 publishes; 24/28/32/36 consume
(`vllm/models/deepseek_v41/attention.py:1435` onward). Short-context all-selected
bypass already exists in that model's `takes_short_path` (image :1650; K1 :1686).
Candidate filtering itself already exists; candidate-only scoring does not execute.

The misleading existing switch is `vllm/v1/attention/backends/mla/sparse_indexer.py:111–145`:
requires MXFP4, SM100 family, varlen metadata and DeepSelect. K1 attention.py:1369
explicitly rejects `indexer_sparse_logits` with packed FP8 pages. The image wrapper
`vllm/utils/deep_gemm.py:308,758–766` labels sparse kernels SM100 only. The real
DeepGEMM source `csrc/apis/attention.hpp:306–439` dispatches these APIs exclusively to
`launch_sm100_sparse_mqa_logits`; its `impls/sm100_sparse_mqa_logits.hpp:38–128`
uses TMEM/UMMA and SM100 shared memory. There is NO SM121 sparse implementation
in that source. The image snapshot omits the external DeepGEMM binary/source: this
is a source-level finding, not a claim to have probed the running container.
Our new path must read segregated FP8 values/scales from the CURRENT paged layout
and preserve the existing top-k consumer, including its tie policy.

RS1's `release/experimental/indexer_fp4/README.md:198–220` describes paged-direct,
then candidate-only scoring; its cited 81.7 → 67.5 ms at 1.04M is an UPSTREAM CLAIM.
It uses NVFP4 key records and DCP. Our FP8 records and K1 page stride differ. We
use the idea, not that code, licence or numerical claim as our implementation.

## Compact verification: #42718 and spark-ds41f

#42718 is primarily a correctness fix for pre-existing compact geometry, not a
complete width-policy implementation. `ragged_verify.py:228` maps packed token rows
by searchsorted over cumulative widths; empty requests own no rows; graph tail maps
to a dummy request. `engram.py:357` uses per-request starts and never commits
verify history. Tests compare each packed request with its own uniform verify and
change staged layouts after capture, across candidate-filtered/unfiltered variants.

Our V2 engine ALREADY supports ragged query starts/logits and rejection sampling:
`vllm/v1/worker/gpu/model_runner.py:1320–1610`, `input_batch.py` combine/expand,
`spec_decode/rejection_sampler.py` chunk verification. The scheduler's :2025–2036
chooses MAX accepted-EMA-derived width across requests; AsyncScheduler :35–59 writes
uniform placeholders for the next step. Remote confidences already arrive on BOTH
TP ranks (`offload/dspark/spark/remote_dspark/speculator.py:307–309` in our repo).
The remote speculator refuses built-in adaptive verification (:65–66), which needs
local draft profiling; simply flipping that config is invalid.

Engine-side compact widths can use the actual broadcast confidences with stable
request-id ownership, trim only causal suffix rows before forward metadata is built,
and retain scheduler reservations as upper bounds. Scheduler :2122–2157 rolls back
ALL originally reserved drafts minus accepted drafts; worker GPU accounting uses the
actual compact rows. These two counts must not be conflated. Engram in OUR engine
is position/slot based (`models/deepseek_v41/common/engram.py:1–35`), not a uniform
repeat_interleave. Remote gather uses query_start_loc_np and per-request k_window;
it already supports unequal spans. Mixed prefill and grammar batches need separate
qualification; the initial isolated arm must retain those batches unchanged.
Ragged full graph replay on SM121 is not established: initial compact ragged forwards
must use eager dispatch, retain all metadata mappings and expose that cost honestly.

spark-ds41f `patches/vllm/0010-dspark-dead-verification-rows.patch` marks low-survival
rows as MoE padding, caps accepted length and leaves dense/attention rows in place.
Our depth_policy.py:1–55 and model_runner.py:1532–1542,1695–1700 ALREADY implement
that mechanism (OFF). It is distinct from removing target rows. Its current TileLang
indexer (`experiments/2026-10-26-retire-b12x/bundles/compare-ea60f8afe/overlay/vllm/models/deepseek_v4_1/tilelang/indexer.py:1–39,117–372`)
uses MXFP4 MMA, BF16 rounding and ordered head sums, so is not portable to our FP8 path.
Its patch 0420 in `experiments/2026-10-25-spark-ds41f-candidate/vllm/` fuses local
statistics, cumulative log probabilities and BLOCK rejection, with PTX-matched FMA
ordering. Our standard rejection sampler is kept; block verification's #61161
backport is irrelevant unless that sampler is adopted. No sampler is ported here.

## Prefill syncs: #41659

The donor fuses Q/RoPE/quant/head weights, writes top-k directly to raw output,
skips unused physical-index mapping and sort, replaces pageable list uploads with
pinned nonblocking copies, supplies repeat_interleave output_size and changes sparse
schedule IDs from whole requests to row pairs (BLOCK_Q=2 bounds a quadratic walk).
`v41_indexer/scoring.py:163–227`, `sparse_table.py:365–387`, `types.py:24`,
`engram.py:313` and associated prefill/row-pair tests were read in the diff.

Our CUDA forward `model_executor/layers/sparse_attn_indexer.py:399–912` has NO .item,
.cpu, .tolist, repeat_interleave or explicit synchronize. The .item at :1276 is CPU
fallback only. CUDA prefill already writes top-k directly into topk_indices_buffer,
uses fused Q/RoPE/quant, and bounds logits allocation in chunks. Metadata builder
`v1/attention/backends/mla/indexer.py:1142,1174,1256,1344–1359,1568,1604–1605,1801,1844–1845,1889–1895`
reads CPU mirrors: those .item calls do NOT cause device sync. :1837–1838 are real
GPU .item syncs but only under DCP > 1, OFF in production. PCP and TP shard helpers
also use CPU mirrors. Engram uses slot/position hashing, no donor repeat_interleave
sync. Remote Engram host wait is necessary completion ordering, not pure metadata
overhead. A traced CPU-path audit will accompany item 2c; do not invent removable
production syncs or use DCP-only savings as a production claim.

## Prefill graphs: #43733/#43736

#43733 keeps dense/source scoring in eager breaks over LIVE context and captures
candidate consumers with persistent bridge buffers; replays refresh lengths/pages
in place. It removes the old 16K admission limit only where context-independent
capture is correct. Tests check full AND tail metadata at a 156672-token live width.
#43736 adds a fixed-capacity bounded-replay tail, dummy scatter row, eager metadata
enter/exit breaks and real-tail hidden-state indices for DSpark. Its capacity is
request-cap × SWA window; stale tail state is cleared on decode. This is a multi-
subsystem build, not removal of one max_seq_len check.

Our `v1/worker/gpu/cudagraph_utils.py:74–75` and dispatch keep prefill eager;
`models/deepseek_v41/nvidia/model.py:792–843` already bounds late layers to the
128-row replay suffix. SP1 completed trace (post-FW2, pins JSON) MEASURED:
32K early prefill 1854.5 ms, idle rank0/1 0.983/1.035%, host-gap 0.486/0.662%;
150K late prefill 1886.2 ms, idle 0.596/0.470%, host-gap 0.095/0.085%.
Even all observed idle is below 5%. Gate FAIL: graphs not built. A new measurement
window is neither needed nor requested for this decision.

## Other donors: #42864/#42867

#42864 uses more split-K slices only on SM120 decode and vectorizes the slice
reducer. It explicitly EXCLUDES SM121 from both dispatch choices. Its tests permit
numerical tolerance to the unfused/FP64 reference; batch identity is within the new
kernel, not bitwise equality to the old reduction. Our mHC uses DeepGEMM/EXL3
boundary kernels (`models/deepseek_v41/nvidia/ops/mhc.py:48–67`, :130 onward).
No validated ≥2% B1 saving with unchanged bits follows. Do not transplant a split-K
reduction change as an exact optimization.
#42867 fixes arithmetic tiling at BF16 LM-head M<=16. Our target head is EXL3 mrow
up to eight rows; different format, different kernel. Batch invariance of the native
remote drafter belongs to NB1. No TV1 head port; ESTIMATED time saving here zero.

## Size estimates (not benchmark results)

MEASURED post-FW2 SP1 per-step input medians: short B1 38.455/38.457 ms;
32K B1 38.265/38.269; 150K B1 39.578/39.577; 32K B4 75.374/75.372.
Attention shares are respectively 5.2–5.35%, 5.4%, 5.33%, 4.88%. No donor's
1M/GB300 gains are substituted for these. Numbers below are ESTIMATED saved ms
per round, may be negative after dispatch overhead; intervals are planning bounds.

| Mechanism | short B1 | 32K B1 | 150K B1 | 32K B4 |
|---|---:|---:|---:|---:|
| Candidate-only FP8 consumers | 0 (already bypass) | 0–0.15 | 0–0.60 | 0–0.35 |
| Compact verify, unequal-request benefit ONLY | 0 | 0 | 0 | 0–7.5 |
| Compact verify, also shorter B1 confidence width | 0–2 | 0–2 | 0–2 | included above |
| Pure prefill host sync removal in CURRENT TP2 path | 0 | 0 | 0 | 0 |
| Prefill graphs (gate failed; decode effect) | 0 | 0 | 0 | 0 |
| mHC/head donors (no qualified exact port) | 0 | 0 | 0 | 0 |
| Existing dead-row MoE skip (already implemented, OFF) | not a new TV1 build | same | same | same |

Candidate bound reserves at most ~0.6/39.58 = 1.52% at 150K for four consumers;
full attention includes other work and is an absolute ceiling. Compact B4 example:
widths [5,1,4,0] drafts remove 6 of 20 scheduled rows (30%); assuming 1/3 of
~(59% MoE + 17% dense) target budget scales after expert reuse gives 7.5 ms (10%).
This is NOT measured per-row cost or expected speedup: eager fallback/synchronization
may erase it. B1 has no cross-request padding to remove. Shorter B1 policy widths
trade acceptance for fewer rows and need per-cell tokens/round and ms/round together.
For prefill graphs, even magically removing all observed idle saves at most
9–19 ms of ~1.85–1.89 seconds (<1.04%); no ≥5% host-gap opportunity exists.
