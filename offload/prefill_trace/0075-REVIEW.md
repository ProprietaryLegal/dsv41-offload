# PF1-R2: tap-aware 0075 and bounded time-budget chunking

UP2 update (2026-10-24): the default-off CPU candidate now closes the successor
reservation and real-suffix gates described below. See
[successor implementation](../upstream_adopt/SUCCESSOR.md),
[CONTRACT amendment draft](../upstream_adopt/CONTRACT-0075-DRAFT.md) and
[replacement isolated GB10 recipe](../upstream_adopt/UP2-GB10-TEST-PLAN.md).
The remaining-work list below records the PF1 intake state; hardware/source
qualification still belongs to the coordinator. No GB10 gain is measured.

Decision: keep 0075 and chunk/time-budget changes OFF during the cap diagnosis.
The process-aware UP1 composition is the current CPU candidate, not X1's old
gather-only patch. **MEASURED CPU**: its seven replay tests pass on Controller,
including interpreted fused gather, real tap suffixes, mixed rejection/resync,
invalid provenance and the process proxy positional ABI. No GB10 result or
prefill gain is claimed. Existing X1's estimated ~0.1s/chunk saving remains
**ESTIMATED**, not a measured amount that explains the cap penalty.

## What is already present, and what must survive integration

Read/review inputs: `offload/spark_leads/01-replay-skip.md`,
`patches/0075-v5.patch`; `offload/upstream_adopt/README.md`,
`patches/0075-process.patch`, its composed overlay and tests/test_replay.py;
production remote_dspark `state.py`, `roundcore.py`, `gather.py`; target replay
and model-state code. Existing source identities belong to UP1's inputs.lock
and MANIFEST; do not bypass their zero-fuzz composition/hash checks.

The target still processes all rows through the last full-KV source and writes
L20 full/source KV before trimming. Disposable intermediate suffix work in
L20–39 is shortened only when the next consumer can be proven independent.
Layers/taps37/38/39 on discarded rows are **not real computed taps**. A single
kept intermediate row cannot be relabeled as a128-row DSpark context.

Required chain, all in the current process-aware composition:

1. Model-state resets `replay_tap_lens_np=None` every preparation, including
   graph paths; an eager trimmed batch publishes actual kept suffix lengths
   in original request order. Prompt-logprobs/MM exclusions stay intact.
   The L20 first SWA consumer preserves its original bound; later replay
   consumers use the actual kept suffix, not a fictional128 rows.
2. `InputBatch` owns this optional engine metadata. `gather.batch_view` copies
   it into `roundcore.BatchView.tap_lens`, **after max_q**. max_q's positional
   field is used by the process descriptor and cannot move. tap_lens is engine
   metadata; it need not be serialized into the process proxy descriptor once
   valid counts/positions have been generated on the engine stream.
3. BOTH Torch and fused gather cap candidate rows using real suffix lengths.
   Invalid/zero/oversized provenance refuses. A request that NEED_DRAFTs cannot
   supply fewer than min(query,128) real tap rows. The actual valid count also
   subtracts rejected rows; preserve first_pos, anchor_pos, sampled anchor,
   request-state slot and generation. Do not turn speculative/rejected rows
   into accepted history or assume all rows in a mixed batch are prefill.
4. Device ring and rank0 WindowRange update from the same real rows. A gap
   resets the contiguous range. A backward overwrite rolls back rows at/after
   first_pos; a zero-row rollback cannot recreate missing older taps. Preserve
   every still-real earlier row allowed by the128 window. NEVER widen lo to
   anchor−128 merely because a consumer wants128 rows. The final full suffix
   must actually be recomputed before drafting/resync can require it.
5. Slot reuse changes generation; disconnect/watchdog/lost reply marks state
   dirty. Async metadata keeps its own64-slot round ownership. Prefix restore,
   rejection, rollback, late reply and resync must use only that generation's
   contiguous rows, not a later engine staging buffer. RoundCore keeps STEP
   headers/resync rows private; do not restore the old shared-staging write.
   Prefix-hit rewind h−128 and the final128-row suffix must rebuild the real
   worker context. Deeper backtracking below a skipped gap needs replay from
   a valid target checkpoint, never a fabricated ring range.

The new CPU regression executes the **production WindowRange**: after a skipped
jump its range contains one row; rolling backward produces no invented history;
only a real128-row replay restores a full context. Existing UP1's fused/Torch
mixed tests retain their original safety assertions. This is boundary/state
validation, not native kernel or end-to-end numerical proof.

## Remaining work to finish 0075

- Coordinator composes against the actual current K1/process-proxy sources,
  retains full async package and verifies all produced/mounted hashes on both
  ranks. Update CONTRACT's optional suffix-provenance semantics alongside the
  candidate; no unreviewed wire/version change. Do not mount two copies of
  model_state/gather/roundcore over one another.
- Close the **successor liveness condition**. Existing `_dsv41_replay_min_chunk`
  and helper return0 when remaining>128 and proposed q≤128. The randomized
  tests eventually offer a larger budget; they do not prove progress if the
  budget stays64/128. The CPU regression now demonstrates eight consecutive
  zero decisions on such a budget. This is a conditional integration risk,
  not a claim that the production OFF path currently hangs.
- Preserve a per-request obligation after skipping: the next late-layer
  consumer must trim (>128) unless a real suffix rebuild occurs. Reserve that
  positive minimum in a bounded future turn BEFORE marking the predecessor
  skippable. Both scheduler running/waiting paths must honor it. A time budget
  may be exceeded by this minimum; it cannot silently return0 indefinitely.
  If resource/KV/restore constraints invalidate the obligation, recover by
  recomputing real suffix history before allowing a smaller consumer. A
  one-step arithmetic helper alone cannot guarantee scheduler fairness.
- Replay-enabled qualification: off/on/off isolated at150/300K first, then
  400/450/literal500K under524288 after attribution. Same fixed chunk policy,
  sampling, target/worker identity and prefix salts; no combined grid yet.
  Include final remainders1/127/128/129 and remaining129/256/257, repeated
  leftover budgets64/128, mixed decode, prompt-logprobs/MM exclusions,
  partial/full hits, async restores, cancel/slot reuse, preempt/backtrack below
  the real ring lo, disconnect before/after the final chunk, watchdog and
  late reply. An unadmittable waiter must not freeze admitted work.
- Assert q/kept/first_pos/valid/anchor/generation agreement across ranks;
  all source KV writes retained; final real128-row suffix before drafting;
  no fake taps/no stale resync; unchanged fixed-prefix target behavior with
  A/A numerical floor, needles, visible task completion, and decode p95/p99
  stalls. Intermediate one-row activations may differ but must be unconsumed.
  Stop on provenance/rank mismatch, missing final KV, unexplained drift,
  nonprogress, fallback growth or repeatable decode regression. Keep HOLD
  unless full current qualification and useful repeated benefit pass.

## Small later chunk/threshold grid

After the isolated 0075 decision, hold that decision constant across a SEPARATE
chunk arm. Legal priorities10 and auxiliary admission/fairness are unchanged.
The current `serve_exp.sh` hardcodes MAX_BATCH=4096 and threshold=MAX_BATCH/4;
there is no validated8192 flag to turn on. A later reviewed experiment-only
launcher must expose and validate both knobs, rebuild shape-sized buffers,
and keep rollback to the exact original launcher. Do not edit production now.

| Initial cell | Max new rows/step | Long-prefill threshold | Purpose |
|---|---:|---:|---|
| current control | 4096 | 1024 | existing baseline |
| threshold-only | 4096 | 512 | separates threshold from chunk size |
| smaller chunk | 2048 | 512 | compare at identical threshold |

Do not test2048/1024 as a fairness-preserving cell: existing policy requires
threshold≤MAX_BATCH/4. This three-cell grid avoids that invalid combination.
Initially use B1 cold/partial plus B1 decode overlapped with long prefill and
B2/B4 progress/p95/p99 stalls. Keep indexer TP-row-shard threshold2048 fixed,
record its actual firing (2048 versus4096 can change sharding), and distinguish
that dispatch threshold from long-prefill scheduling. Do not attribute a
scheduling change to sparse-attention arithmetic alone.

**ESTIMATED per-rank component sizes**, not an additive peak-memory forecast:

| Object | 2048 | 4096 | 8192, conditional only |
|---|---:|---:|---:|
| one BF16 residual [rows,4,5120] | 80MiB | 160MiB | 320MiB |
| shared top-k [rows,512] int32 | 4MiB | 8MiB | 16MiB |
| SWA indices [rows,128] int32 per builder | 1MiB | 2MiB | 4MiB |
| expanded ratio1 int32 table at524288, per builder | 64MiB | 128MiB | 256MiB |
| ratio1 indexer workspace at cap524288, factor4 | 264MiB | 264MiB | 264MiB |

Indexer logits budget stays128MiB; FlashInfer workspace stays128MiB. At a
500000-key illustrative ratio1 gather, the logits planner can use at most67
query rows per subchunk: approximately31/62/123 score subchunks for full
2048/4096/8192 query rows. These are **ESTIMATED extents**, not measured traffic
or elapsed work; causal masks, ratios and TP sharding change actual counts.
Persistent builder sharing and transient lifetimes prevent multiplying all
rows by40 layers or blindly summing the table.

8192 is not in the initial grid. It needs measured two-rank headroom with
all allocations/live workspaces accounted for, no new swap, usable≥5GiB
pre-profile memory, native TensorFold/FlashInfer/indexer/Engram kernel and
buffer row limits verified, compilation/capture shapes rebuilt, RPC/tap gather
bounds checked, and a lower-chunk passing numerical/stall result first.
Its larger minimum step can worsen decode stalls despite better aggregate
prefill. Never try it by changing max_model_len or max_num_seqs alone.

## Default-off bounded time-budget design

`bounded_chunk.py` is an executable **detached CPU proposer**, not a scheduler
patch or deployed feature. No environment flag is read by production. Default
`enabled=False` returns the upstream choice unchanged. A later integration may
use an explicit default0 time-budget option only after the following contract
is implemented/tested. It borrows the DeepInfer new-token + existing-prefix
cost-model concept described in expert continuation §A.9/[R10], not its cluster
scheduler or a transferable performance claim.

Prediction: `fixed + token*q + prefix_coeff*q*(prefix+(q−1)/2)/100000`.
Fit nonnegative costs from measured new-row/prefix points on our target/ranks,
with separate models/threshold segments if residuals require. The4096 fixed-row
fit delivered in item3 diagnoses cap effects but **cannot identify q scaling**;
use the later2048/4096 measurements and at least two smaller supported probe
sizes before fitting this proposer. No training on headline traces followed
by calling the same traces held-out. Use conservative residual margins and the
slower rank; misprediction adjusts the next budget, never drops valid work.

The proposer binary-searches the largest admissible q within a soft budget.
If even its floor exceeds the budget, it returns that positive floor and
records the overrun. Default floor1 may be raised to a hardware-supported
minimum; after a skipped0075 predecessor it is at least129, with a real
reservation. Invalid resources/obligations fail explicitly. It never changes
attention, caches, target sampling or tap metadata.

Required scheduler integration (still open; no speedup claim):

1. Apply only AFTER real KV admission, async restore readiness, encoder and
   source-KV constraints. Reserve decode rows and the progress floor for every
   ready already-admitted request; new/unadmittable waiters cannot consume
   those reservations or toggle a global foreground/background exclusion.
2. Maintain a bounded age/deficit queue within current priority classes. Every
   ready admitted request receives its reserved floor within at most N ready
   request turns under fixed resources, even with an over-budget prediction.
   Cap opportunistic extras, then rotate. No global winner holds other work
   forever; keep legal priority semantics and auxiliary admission controls.
3. Pending restores keep ownership and their asynchronous completion path;
   they are not counted as newly admitted repeatedly. Bound restore timeout
   and surface recovery; physical missing KV/resources cannot have an honest
   unconditional token-progress guarantee. Do not busy-wait or fabricate
   progress while a restore is incomplete. Ready peers continue meanwhile.
4. Couple replay skip eligibility to actual guaranteed successor reservations.
   Keep must-trim obligations across scheduler turns/preemption; clear only
   after the real required suffix exists. Changing chunk size cannot erase
   rollback/ring history or publish padded rows as taps. MM/logprob/prefix
   exclusions retain upstream behavior.
5. Test deterministic persistent64/128 budgets, zero/underestimated timing,
   already-admitted decode and prefill, multiple classes, async restore then
   ready, KV eviction/preemption/recovery and an unadmittable top waiter.
   Inspect the actual installed scheduler on both paths; the detached CPU
   tests prove positive proposal/floor properties, **not full engine liveness**.

CPU validation commands (no Spark/GPU access):

```bash
source ~/.config/ndm/paths.sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 TRITON_INTERPRET=1 "$HARNESS_PYTHON" \
  -m pytest -q -p no:cacheprovider offload/upstream_adopt/tests/test_replay.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 "$HARNESS_PYTHON" \
  -m pytest -q -p no:cacheprovider offload/prefill_trace/test_bounded_chunk.py
```
