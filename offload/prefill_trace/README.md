# PF1-R2 eager prefill instrumentation (default OFF)

CPU-qualified diagnostic, **not deployed or GB10-qualified**. It observes K1 v5
without changing model inputs/outputs, workspace allocation, chunk plans,
top-k arguments, dispatch policy or sampling. Five exact bases are pinned in
BASES.json. The K1 runner and FlashInfer bases also match the current local
`offload-x86host` tree inspected during this lane. No merge was performed.

```bash
source ~/.config/ndm/paths.sh
CUDA_VISIBLE_DEVICES='' "$HARNESS_PYTHON" -m offload.prefill_trace.build_overlay \
  --enabled --output-root offload/prefill_trace/generated
(cd offload/prefill_trace/generated && sha256sum -c ../OVERLAY.sha256)
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 "$HARNESS_PYTHON" -m pytest \
  -q -p no:cacheprovider offload/prefill_trace
```

Omitting `--enabled` emits exact original base bytes, without a helper. The
launch config `PF1-R2-trace.env` defaults PF1_TRACE=0: its mounts, Docker env
and Docker args equal D-kv6-async-pack byte for byte; only the selected cap and
label are set. It does not require generated files while off. Enabling requires
PF1_TRACE=1, unique PF1_TRACE_RUN, checked generated hashes, and explicit
per-node writable trace directories. Two K1 targets are replaced exactly once;
the rest of K1 and the process proxy are retained. No launcher is invoked by
this build or by config tests. Runtime env defaults OFF as a second guard.

The **enabled** source inserts narrow call spans; removes none of the original
arguments/operations. CPU tests strip all telemetry and compare the entire
original AST, execute the actual prefill indexer against the existing FP8/FP4
CPU references, and run the actual chunk planner through workspace overflow,
request grouping and logits subchunking. These preserve output bytes/plans at
the Python boundary; native CUDA parity and timing overhead need a window.

Each JSONL row contains run/rank, an opaque request hash, chunk identity
(hash:existing-prefix:new-tokens:occurrence), per-member batch row bounds and
max_model_len. Mixed batches carry all prefill members; do not divide shared
batch intervals among requests. Prefix is from the host-known computed
prefill count; query lengths are from the original host query offsets. Request
names, prompts, tensor contents and generated output are never serialized.
Occurrence disambiguates repeated prefix work following rollback; no state is
written back into the model. Both ranks use the same run salt.

Recorded intervals and work counters:

- `chunk.total`: wall enqueue span and current-stream CUDA-event interval from
  before attention metadata through the eager execute_model return. Input-batch
  preparation before that point is excluded; TTFT remains separate.
- `metadata.prepare` and `metadata.indexer_chunk`: host/CUDA span of actual
  metadata builders. The block-table rows/columns/elements are a **scan bound**,
  not a claim that every reserved entry is scanned by a native kernel.
- `indexer.plan`: exact host request/query slices, compressed-key rows,
  workspace rows, logits budget, all requested key rows, grouping due to
  workspace pressure, query subchunking, and single-request overflow. The
  last case is labeled `single_request_over_workspace`; it does not invent a
  new fallback or truncate keys. A native refusal remains a refusal.
- `indexer.gather`: compressed value/scale view shapes, elements/bytes and
  local/max gathered row counts. Reused KV subchunks have skip_kv_gather=true;
  they produce score/top-k spans but no false gather interval.
- `indexer.mqa`, `indexer.topk`, candidate selection/masking: row bounds,
  score/logit work elements and bytes, top-k width, value/scale sizes,
  indexer layer name where present. These are logical/native-boundary extents,
  not measured hardware DRAM traffic or effective top-k comparisons.
- `sparse.metadata_scan` records global sparse-index construction;
  `sparse.dispatch.flashinfer` / `.triton` identify the dispatch actually taken,
  with query/sparse-index sizes. FlashInfer's deeper native kernel choice is
  opaque here and must come from its timeline, never a guessed threshold.
- GroupCoordinator all-reduce/all-gather/reduce-scatter and variable variants,
  plus the indexer layout exchange, record `collective.*` wall and CUDA spans.
  These intervals include transfer AND wait. Pure exposed wait requires
  comparing both rank timelines/dependencies; fused/native collectives that
  bypass these Python entries are **not individually covered**. A lack of
  collective records is not evidence of zero wait.

CUDA events are recorded on the current stream. Completion uses query(), then
elapsed_time() only for completed events, never synchronize() or tensor.item()
on device data. A span on a caller stream can include its waits for auxiliary
streams but does not time independent work that is never joined. Events add
launch/host overhead: use traces for attribution and separate OFF runs for
headline latency. Span times overlap; never sum them as total GPU work.

The queue holds at most roughly 8192 pending pairs (plus nested open spans);
chunk identities are bounded at 32768. Exhaustion, bad mode, write failure or
hash drift fails loudly. No samples are silently discarded. Only eager
prefills are supported; a graph-mode prefill refuses rather than changing
execution mode. Dummy/decode steps do no timing; they still drain completed
samples. Every request must generate multiple steps after its prefill so the
last chunk events are drained naturally. Check `chunk.total` completeness on
both ranks before fitting. A final outstanding event is missing data, never
0 ms. Files use exclusive create with rank+PID names; rotate the run directory
for each fresh process/window. Recorder I/O is bounded in memory but JSONL
storage must be budgeted on local disk, never /tmp.

For a coordinator window use the replacement recipe in
`../prefill_maxlen/GB10-TEST-PLAN.md`. Do not combine this overlay with the old
request-workspace candidate or UP1: pins deliberately require baseline K1.
Those are separate later arms after the baseline slowdown is understood.
