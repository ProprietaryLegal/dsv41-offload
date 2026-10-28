# S51 FIX 3 — production-derived candidate qualification

Implementation complete; hardware qualification remains mandatory and unperformed.
No SM121 equality or speed result is claimed. Exact selected-index inequality raises
STOP. Both workers must supply valid installed-kernel and actual-forward receipts
before candidate warmup/timing. Missing coverage is also STOP.

The original six synthetic shapes have been replaced by 288 pinned cases:
consumer layers 24/28/32/36 × B1/B2/B3/B4 × raw occupied contexts
1024/8193/32000/150000/327679/327680 × random/exact-tie/near-tie arithmetic.
Six row layouts rotate across the matrix: single, native2, flattened widths3/4/6,
and flattened6 with a graph-padding row. All four consumers share layer20's
ratio-one cache. Their compressed allocation width is therefore 327680 even for a
short occupied context; the matrix distinguishes allocation and visible widths.

The installed oracle executes the **installed**
`DeepseekV32IndexerMetadataBuilder._prepare_decode_tensors`, including its uniform
Triton expansion and padded-tail expansion. It then executes `score_decode`, the
same metadata adapter used by the engine patch. Dense installed DeepGEMM and
installed persistent top-k are the reference. Every case checks eager equality
and captured replay after changing queries, per-row lengths, pages and candidates.
Candidates include real tail blocks, empty rows, -1 padding, and full capacity2048.
Ties and adjacent fp32-scale decisions use exact array equality, without tolerance
or index-set overlap.

The pinned K1 kernel view is page64, stride46080 bytes, cache offset37440 bytes;
manager pages192 divide the common138240-byte block stride into three kernel
pages. Physical IDs begin47000, so addresses cross2GiB; this is not an arbitrary
extra256-byte synthetic stride. One ~2.4GB uninitialized address-span allocation is
reused, and only referenced high pages are initialized/read. No model tensors,
weights, prompts or generated text are serialized. Runtime geometry deviations
fail loudly.

Actual candidate forwards additionally compare dense versus candidate selection
once per `(consumer layer, request B, context bucket)`, bounded to64 checks/worker.
The owning layer is resolved from the actual attention/indexer-op identity in
static forward context; the shared cache's layer20 prefix cannot masquerade as
consumer identity. A candidate-only dispatch hook temporarily requests eager for
unchecked real decode buckets so production FULL graph replay cannot silently
skip the guard. Checked buckets immediately resume normal graph dispatch. Dummy
runs/captures and the short all-selected bypass never force this eager guard.
A known limitation is that this CPU bookkeeping/dispatch integration still needs
qualification in the real engine; any missing receipt prevents timing.

The new dry-by-default `offload.tv1.qualification` driver reuses the existing
public prompt cuts and sends B1 at nominal150K, B2/B3/B4 at nominal32K, and B4 at
nominal150K. It forces complete outputs and uses fresh per-stream salts. It never
profiles. ESTIMATED admission budget735s per candidate start includes all five
cells. Original cut tolerances mean actual context minimums are147000 and31360;
receipts record the actual minimum positive per-row context, rather than nominal
labels. The separately installed oracle supplies exact327680 boundary coverage;
no actual300K/B4 model-forward claim is made.

Native `next_n>2` is unqualified on SM121 and refused by the adapter. Packed ragged
native Q and varlen `indices` explicitly use the unchanged dense path; FP4, DCP
and PCP are excluded. Flattened ragged rows and graph tails remain supported via
the installed metadata builder. Prefill stays on the original dense path.

## Runner APIs

- `candidate_cases.validate_oracle_receipt(receipt)` checks exact ordered case
  identities and every parameter, backend, capability and eager/replay flag.
- `candidate_cases.validate_engine_receipts(records)` checks one worker's parsed
  `TV1_ENGINE_ORACLE {json}` log records. Required actual forwards: all four
  consumers at B1/B4≥147000, B2/B3≥31360, plus a separate B4 32K bucket. It validates
  row ownership, stride/offset, allocation width, page, capacity and backend.
- Invoke `python -m offload.tv1.qualification --execute --data-dir ... --cuts ...
  --deadline-utc FLOAT --telemetry PATH PATH --capture-floor-kb N
  --cache-salt-prefix STR --output PATH` only through the coordinator runner.
  Invoke without `--execute` for a local dry plan.

OFF continues to build exactly the two original engine files, byte-identical to
their pinned baselines. Candidate builds six files, compact three. Combined
candidate/compact composition uses its own pinned zero-fuzz diff because both
features now touch the runner. All standalone and combined builds are tested.
No donor expression or launcher is imported or vendored.

## Provenance

- Model configuration: `offload/spark_leads/metadata/official-1017.json`,
  `compress_ratios`/index heads/top-k/source/capacity/block-size fields;
  SHA256 `45bde9d6d87f3afbffbb6c93b98b311ad2946453fbc4cf496dcbda4538aaadf6`.
- K1 allocation: `offload/kvpack/kv_arith.py:7` and `STRIDE`, plus
  `offload/kvpack/overlay/vllm/v1/kv_cache_interface.py:376` (paired view split),
  SHA256 `36ebd7e679c0771a5568026dc530546a0e7ce2ec85e8eeec29a50cf00f6c8755`.
- S8 real allocator views: campaign `S8.report.md` N-3, lines177–206;
  SHA256 `7624472d793ff20d93b51e7fddc30282ba7223cd2ed921a36cfe9fc5e7675b33`.
- Installed builder: read-only image
  `vllm/v1/attention/backends/mla/indexer.py:938,1126–1240,1738`;
  actual patch adapter/guard source and patches are pinned in `inputs.lock.json`.

## CPU validation

Controller only, required miniconda interpreter, CUDA visibility empty, TMUX unset,
private TMPDIR/TMUX_TMPDIR and pytest/mypy caches in authorized TV1 scratch.
TV1 suite after the dispatch change:61 passed,8 hardware-only skips; one retained
CPU-profiler warning. Strict mypy:25 source files clean. Targeted final checks
cover strict receipt omissions/substitutions/duplicates, B3, stride and context
drift, ties/replay flags, metadata row ownership, explicit unsupported layouts,
actual-op owner resolution, mismatch STOP, capture refusal, eager-to-graph return,
public qualification dry mode, all overlay compositions and OFF byte identity.
No hardware/remote request, installation, process kill, production change, email,
push, merge, memory or session-log write was performed for this sub-item.
