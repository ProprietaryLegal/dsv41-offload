# PF1-CAUSE item 2: the confirmation spans and the one short window that reads them

Status: CPU-built and CPU-tested on Controller; **not run, not GB10-qualified**. PF1-CAUSE-WIRE adds opt-in runner wiring (see §5).
Every duration below is ESTIMATED from the MEASURED window-2 file timestamps unless marked.

## Plain language

- Item 1 put the whole extra cost in rank 1's MoE half of layers 0-19 but could not say *which* part, because nothing
  inside the MoE half is traced. These spans time it directly, on **both** ranks: the whole MoE call, the mHC pre-mix
  before it, the routed-expert (TensorFold prompt) call and each of its kernel launches, plus the attention call as a
  control. One extra number per layer gives each rank's own MoE time with the all-reduce wait removed.
- They reuse the PF1-R2 trace runtime unchanged, so they only exist in ON arms and default to off twice (overlay not
  mounted; runtime env off). With the switch off the builder emits the exact K1 bytes.
- One short window answers it: the window-2 shape (traced 327,680 then 524,288, 150K cold ×1, ~27 min HOLD).
  An optional third arm (~+9.5 min) runs 524,288 with the routed-expert scratch held from start-up instead of
  allocated per call: if the extra time disappears there, H1 is confirmed and the arm itself is the fix candidate.

## 1. What is added (offload/prefill_cause)

| File | Role |
|---|---|
| `build_overlay.py` | instruments 3 K1 files (bases pinned in `BASES.json` = K1 MANIFEST overlay hashes = the window's mounted hashes); `--enabled` off ⇒ exact bytes |
| `runtime.py` → `vllm/v1/worker/gpu/dsv41_prefill_cause.py` | host-only allocator counters for the routed-expert row, only while a PF1-R2 chunk is open |
| `OVERLAY.sha256` | the 4 reviewed ON outputs (3 sources + helper) |
| `localize.py` | item-1 attribution plus a `cause_labels` table (per label and row count, matched-prefix medians, and `moe.local`) |
| `../launch/configs/PF1-cause.env` | PF1-window.env sourced unchanged, then: ON swaps the 3 production K1 mounts for the span sources (exactly 3, read-only, refused otherwise) and adds the helper; `PF1C_TF_TRANSIENT=0` flips exactly `VLLM_EXL3_TF_PROMPT_TRANSIENT` and `_ALIAS` to 0 |

Spans (CUDA events on the current stream, the PF1-R2 `_pf.call`; label: metadata):

| Label | Where | Metadata (numbers only) |
|---|---|---|
| `attn.total` | K1 `nvidia/model.py` `self.attn(...)` in `DeepseekV4DecoderLayer.forward` (includes the attention all-reduce) | layer, rows |
| `mhc.ffn_pre` | same, the FFN-side `mhc_shifted_post_pre(... pre_mix=attn_pre ...)` | layer, rows |
| `moe.ffn` | same, `self.ffn(...)` (router, shared expert, routed experts **and the FFN all-reduce** nested) | layer, rows |
| `moe.routed.tf` | K1 `tf_prompt.py` both `T.routed(...)` calls | rows, slots, xd_ptr, allocator: retries, device allocs/frees, reserved/active bytes, segments |
| `tf.<kernel>[.gateup/.down]` | K1 `tf_experts.py` `routed()`: group_count, group_place, group, rot_in (+xg_ptr), work_list, grouped_mma/_rows/(mma2/mma3/plain), gateup_epilogue, grouped_down3, combine_y, down_epilogue, down_combine (+out_ptr) | rows, slots_total |

Derived per layer: `moe.local = moe.ffn − (FFN all-reduce nested in it)` = that rank's own MoE-half time (m0, m1 of
CAUSE.md §1, measured instead of inferred). `moe.local − moe.routed.tf − mhc` (approximately) = router + shared expert.

Invariants (CPU-tested, `test_overlay.py`, `test_localize.py`, `test_pf1_cause_config.py`):

- OFF output = the K1 bytes; bases refused on any hash drift; every anchor must appear exactly as often as listed (a
  changed base cannot silently lose a span); outputs refused inside the K1 / PF1-R2 / image trees.
- Removing `_pf.call(label, meta, fn, ...)` → `fn(...)` and the two imports gives the **whole original AST** of each file.
  Metadata expressions are side-effect free (only literals, shapes, `data_ptr()`, the helper's host-only stats read).
- The real K1 `tf_prompt.routed` + `tf_experts.routed` (TRANSIENT=1, ALIAS=1 as in production) run on CPU against a
  recording kernel stub give the **same launches, in the same order, with the same argument shapes/dtypes/strides and
  scalar values, and the same output tensor** with and without the spans; spans pass straight through outside an open
  chunk (decode, capture) and the helper then reads nothing.
- Config: OFF arm = PF1-window.env OFF (production launch) except the label; ON = PF1-window ON + exactly 3 read-only
  production mounts removed + 4 hash-checked overlays (10 unique targets); T0 changes exactly two env tokens.
- Not covered on CPU: native CUDA timing overhead, in-container imports (`vllm.v1.worker.gpu.dsv41_prefill_trace` must be
  mounted, i.e. PF1_TRACE=1), graph capture interplay (spans only run under an open eager prefill chunk, as PF1-R2).

Overhead (ESTIMATED): ~480 more event pairs per full chunk (12 spans × 40 layers) on top of PF1-R2's ~240; at a few
µs host time each and no device sync, ≲ 5 ms per 1.8-2.5 s chunk. PF1-R2 itself showed none measurable (window 2
traced TTFT 69.81 / 91.48 s vs window-1 OFF medians 70.38 / 95.33 s, MEASURED). JSONL ~25 MB per rank per arm
(window 2: ~8 MB with 11.5K rows; +~23K rows).

## 2. The window (one HOLD; ON-only; no OFF pair needed)

The cap gap is already REPRODUCED twice (S18; window 1 OFF gate, `PF1-W1-20261025T234254Z/summary.json`). This window
only attributes it. It uses window 2's sequence, which ran COMPLETE (`PF1-W2-.../summary.json`).

| Order | Arm | Config | Requests |
|---|---|---|---|
| 1 | `cause-327` | PF1-cause.env, PF1_CAP=327680, PF1_TRACE=1 | warm-up 32K (64 tokens) + 150K cold ×1, 1,500 output, window-2 prompts/salts |
| 2 | `cause-524` | same, PF1_CAP=524288 | same |
| 3 (optional) | `cause-524-t0` | same + PF1C_TF_TRANSIENT=0 | same |
| — | restore | `W8/restore.sh start/check/close` (greedy 6/6), unchanged | |

Guards as window 2 (4 GiB MemAvailable stop line, swap, telemetry liveness, foreign demand, drafter positions ≤ 301,500).
Read-only additions to preflight, per Spark, before each start: `uname -r`, `nvidia-smi --query-gpu=driver_version`,
`/sys/kernel/mm/transparent_hugepage/{enabled,defrag}`, `/proc/buddyinfo`, `/proc/vmstat` compaction counters (host
parity and free-memory fragmentation for H1's "why rank 1"). No writes, no sudo.

Arm 3 memory (ESTIMATED): +1,014 MiB held per rank from start-up (K1 README). Window-2 prefill MemAvailable at 524,288
was 6.00 GiB (sparkhead) / 6.77 GiB (sparkworker), MEASURED → ~5.0 / ~5.8 GiB, above the 4 GiB stop line with ~1 GiB margin on
sparkhead. Idle before start was ~9 GiB on both (≥ 5 GiB start rule holds). The window's existing stop line protects it.

### ESTIMATED timeline (MEASURED inputs: window 2 cmd file times, HOLD 01:09:13Z → close 01:35:19Z = 26.1 min)

| Step | Window 2 MEASURED | This window ESTIMATED |
|---|---:|---:|
| preflight, HOLD, identity snapshot | 0.35 min | 0.5 min (+ host-parity reads) |
| arm start (stop → ready) | 6.9 (327) / 6.3 (524) min | same |
| identity + pin checks | 0.2 min | 0.2 min |
| warm-up + 150K cold + collect | 2.5 min | 2.6 min (span overhead ≲ 1 %) |
| restore start/check/close | 7.4 min | same |
| **HOLD, arms 1-2** | **26.1 min** | **≈ 27 min** (production down ≈ 26) |
| **HOLD, arms 1-3** | — | **≈ 36.5 min** (production down ≈ 36) |

## 3. Reading the result (localize.py on the four or six trace files)

| Observation (rank 1, 524,288 − 327,680, per full chunk) | Meaning | Next |
|---|---|---|
| `moe.local:4092` ≈ +600 ms on rank 1, ≈ 0 on rank 0 | confirms item 1 directly | — |
| …and `tf.*:4092` carry it (same rows/slots/args), `tf.*:128` do not | **H1 or H3**: same kernels slower on rank 1 | arm 3 decides |
| arm 3 removes it (moe.local and TTFT back to ~327,680 level) | **H1 confirmed**: per-call scratch placement | FIX.md path A |
| arm 3 does not remove it | H3 (persistent placement) or a kernel-level effect | FIX.md path B |
| `moe.local − moe.routed.tf − mhc.ffn_pre` carries it | **H2** router/shared expert | FIX.md path C |
| `mhc.ffn_pre` carries it | H4 | FIX.md path C |
| `moe.local` flat on both ranks but rank 0 still waits at the FFN all-reduce | the wait is not local work (H5) | nsys window (FIX.md path D) |
| allocator counters (`alloc_retries`, `device_allocs`) step up per chunk at 524,288 only | mapping churn per call (H1 variant) | path A |

## 4. What a positive answer looks like in numbers (from window 2, MEASURED)

Rank 0's FFN all-reduce grows 27-35 ms per full layer (sum 606.8 ms/chunk). If H1 holds, rank 1's `moe.local:4092`
grows by the same per-layer amounts and `tf.grouped_*` / `tf.rot_in` / `tf.down_combine` (the kernels touching the
750-MiB scratch) carry most of it, with `xg_ptr`/`out_ptr` differing between caps or ranks in allocator segment.

## 5. Runner wiring (PF1-CAUSE-WIRE; CPU qualified, coordinator execution only)

`pf1_window.py --phase cause --gate-from <window-1-dir>` uses PF1-cause.env and stages both hash-checked generated
source trees, refusing any different existing tree. Both ranks verify **18** unique engine paths: K1 13 + PF1-R2's
4 new paths + the cause helper; the five existing targets are replaced with their selected overlay hashes. Every
arm compares its env, mounted hashes, image, engine, weights and argv against its own rank's production snapshot.
`--cause-t0` adds **instrumented** cause-524-t0; exactly the TRANSIENT and ALIAS env values become 0. Each arm runs
32K warm-up (64 tokens) followed by 150K cold ×1 (1,500 output). Existing phases and thresholds are unchanged.

Cause uses window 1's REPRODUCED gate, restored close and pinned artifacts. The ancestor rule checks that the
runtime launch/K1/PF1-R2 inputs did not change: the cause-only allowlist admits the reviewed cause additions, runner,
verifier, offline analysis, D2 files (not used by the target), and documentation. The original ON-phase runner-only
allowlist remains unchanged. No unrelated execution-source drift is accepted. Host-parity reads and >=5 GiB checks
precede every cause start; the shared >=4 GiB/swap/telemetry/demand/deadline guards and W8 restoration still apply.
`arm-memory.json` and summary `arm_memory_min_kb` retain both hosts' observed minima even after a STOP; completed
arms also carry `checks.mem_available_min_kb`. These are 1 s sample minima, not a continuous-memory bound.

No production, K1 package, restore script or promotion gate changes. No window was run by this lane.

## 6. Pre-registered T0 reading (written before any cause window)

CPU command (numbers and hashes only; reads complete, restored cause evidence, validates the pinned request artifacts,
trace completeness, unique 150K size mapping on both ranks, cap/run identity and paired prompts/salts):

```bash
python3 -m offload.prefill_cause.analyze \
  --cause-window "$PF1_CAUSE_OUT" \
  --off-window /opt/dsv41/campaigns/DSV41-OFFLOAD/PF1-W1-20261025T234254Z \
  --output "$PF1_CAUSE_OUT/cause-analysis.json"
```

Output includes both ranks' per-layer `attn.total`, `mhc.ffn_pre`, `moe.ffn`, and `moe.local` (FFN minus nested
all-reduce), with high-minus-low deltas over matched full-chunk prefixes, separated by layer and row count. Aggregated
routed/kernel spans are retained as well. T0 is compared separately against cause-327, never pooled with cause-524.
Every input summary/request/trace file receives a SHA256 in the output. Missing/incomplete spans refuse analysis.

**MEASURED OFF references** (window-1 `summary.json`, pinned artifacts verified locally): 150K median TTFT
**70.384044684 s** at 327,680, **95.330878192 s** at 524,288. OFF arms were uninstrumented: there are **no measured
OFF per-chunk CUDA totals**. The command explicitly reports those unavailable; it does not infer them from TTFT.
Per-chunk comparison therefore uses the current **cause-327 measured full-chunk totals** on each rank.

Fixed decision, applied only to a complete, restored, valid cause window:

- **H1 CONFIRMED**: T0's 150K TTFT differs from the OFF-327 median by **at most 5%**, and on **both** ranks its median
  full-chunk CUDA total falls inclusively within the **min–max range of cause-327 totals over the same prefixes**.
- **H1 RULED OUT**: T0 TTFT differs from OFF-524 by **at most 5%**, and is outside the OFF-327 5% band.
- **AMBIGUOUS**: every other result (including overlap of the TTFT bands or restored TTFT without restored chunks).
- **NOT RUN**: no T0 arm. Attribute the cap gap using the first two arms; do not assign an H1 verdict.

These are pre-registered experiment readings, not a promotion or proof of physical page placement. Interpret alongside
rank-1 `moe.local` and TensorFold kernel deltas (§3) and allocator counters. The OFF baseline is from an earlier HOLD;
report any baseline drift and the instrumented controls' TTFT before relying on a cross-window conclusion. Native
CUDA equivalence, the separate greedy/logit-floor candidate checks in FIX.md, and K1's existing memory/promotion rows
remain coordinator work before any production adoption. ESTIMATED HOLD: ≈27 min without T0, ≈36.5 min with T0.
