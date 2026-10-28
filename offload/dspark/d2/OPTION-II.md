# D2 option ii: generation-time top20

Candidate revision `d2-decode-capture-option-ii-v2`. Coordinator integration and
independent check precede use. This lane makes no Spark/x86host requests or service
changes. All qualification here is CPU source inspection and synthetic rehearsal.

## Pinned-image wire proof

Paths below are relative to `/opt/dsv41/data/src-image/`, inspected on
Controller without connecting to the serving hosts:

- `vllm/config/model.py:280-291`: max_logprobs defaults to **20**, mode to
  **raw_logprobs**, before temperature/top-p/processors. The deployment documents
  the cap as 20 (`DATASET.md`). `vllm/sampling_params.py:857-870` rejects a request
  exceeding the cap. Request **20**, never 21. No server setting changes are needed.
- `vllm/v1/sample/sampler.py:335-360`: gather the top20 and concatenate the sampled
  ID/LP first, also computing its vocabulary rank internally.
  `vllm/v1/sample/rejection_sampler.py:238-264` gathers accepted-token target
  logprobs on the speculative path. Raw/processed selection is at lines 158-175.
- `vllm/logprobs.py:188-210`: sampled entry first, then top20, with dictionary
  deduplication. There are 20 unique entries when the sample is inside the selected
  top20; 21 when it is outside, including ties at the boundary.
- `vllm/entrypoints/openai/completion/serving.py:379-389` passes streaming delta IDs
  and their logprobs to `_create_completion_logprobs`. Lines 693-721 serialize
  `tokens`, `token_logprobs`, and `top_logprobs`; the inclusive `20 >= i` retains
  all 21 entries. `return_tokens_as_token_ids=true` produces `token_id:<decimal>`
  labels and dictionary keys, avoiding decoded-text collisions. Thus completions
  **does not lose the true twentieth entry**. The formatter clamps LPs at -9999;
  the collector refuses that sentinel rather than calling it exact.

The source/default mode is raw_logprobs; no local launch override was found. This
is a source/configuration finding, not a fresh live runtime probe. The coordinator
must retain that mode with the already qualified image and capture derivative.

**Rank limitation:** completions does not serialize the internally computed
vocabulary rank. `tok_rank` follows the shard contract: 1..20 is the position in
the stored top20 ordered by descending LP, then ID; **21 means outside top20**,
not an exact vocabulary rank. Boundary ties preserve the server-selected top20
membership. Exact outside vocabulary rank would require a different API contract;
no server/image modification is included here.

## Coordinator W12 generation command

After integration into a reviewed immutable checkout, retain W12's existing
approval, HOLD/PAUSE/drain/preflight and tmux supervision. In its `gen` step,
replace only the generator invocation with the following command. The exact state,
run, PAUSE and capture-error paths match the existing W12 d2pass2-r1 script; the
new generator source SHA/hashes must be recorded separately from capture SOURCE
07fc7d6. **This lane did not edit or execute W12.**

```bash
source ~/.config/ndm/paths.sh
: "${HARNESS_PYTHON:?}"
cd /opt/dsv41/repos/dsv41-deploy/offload/dspark
HOURS="$($HARNESS_PYTHON -c 'import json; print(float(json.load(open("/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/APPROVED.json"))["spark_hours"]))')"
"$HARNESS_PYTHON" -m d2.generate run \
  --prompts /opt/dsv41/data/d2/prompts/prompts.jsonl \
  --state /opt/dsv41/data/d2/gen-decode-07fc7d6-d2pass2-r1 \
  --spark-hours "$HOURS" --streams 3 --top20 generation \
  --pause-file /opt/dsv41/campaigns/DSV41-OFFLOAD/D2-GEN2.PAUSE \
  --capture-error-host sparkhead \
  --capture-error-file /home/rank0user/dsv41-offload-run/d2-decode-d2pass2-r1/ERROR_COUNT.json
```

Require a NEW second-pass state on first launch. Resuming that state must retain
`--top20 generation`; switching modes refuses. Default `--top20 off` preserves the
old request body byte for byte. Temperature 1, top_p 1, priority20, slot cap3,
foreign-demand cancellation, HOLD, capture-error monitoring and budget are retained.

Each token chunk is validated before its IDs/rows become committed under one
lock. Required checks: equal array lengths, matching committed ID and exact own LP,
20 or 21 distinct vocabulary IDs, finite nonpositive unclamped LPs, all twenty
ranked entries present, an outside sample no higher than the twentieth, and top20
mass <= 1 + 1e-4 (the replay, shard and trainer tolerance). Residual is `max(0, 1 - sum(exp(top20)))` in the saved
row, the shard contract: real rows exceed 1 by ~1e-7 when the top-1 LP rounds to exactly 0 (S32 FIX-1).

Missing/invalid rows immediately create PAUSE exclusively (preserving an existing
reason), close that stream and record a `top20_invalid` event and segment error.
The gate cancels remaining own streams. It never removes PAUSE or repairs a bad
row. A rejected chunk is not committed; the previously validated prefix remains.
Every segment save and running checkpoint stores generated IDs and rows in the
same atomic JSON replacement. Abort/resume and crash-checkpoint recovery retain
1:1 alignment; an uncheckpointed suffix after a process death is unclaimed.

## Freeze, export and build

After PAUSE/cancel and clean generator exit, follow the existing coordinator
FLUSH/drain/copy procedure for capture files. Do not export from a live driver;
the exporter acquires its LOCK and refuses unreaped `running` states. Restart and
stop cleanly to resolve a crash checkpoint first. The offline command below writes
a NEW private tree; it makes no network calls and never alters source state:

```bash
# Coordinator: choose NEW output paths for this frozen collection.
"$HARNESS_PYTHON" -m d2.generation_export \
  --state /opt/dsv41/data/d2/gen-decode-07fc7d6-d2pass2-r1 \
  --out "$TOP20_TREE" \
  --image-smoke /opt/dsv41/campaigns/DSV41-OFFLOAD/W10/private/window-smoke-r6-20261024T203207Z/smoke/report.json
"$HARNESS_PYTHON" -m d2.decode_dataset \
  --top20 "$TOP20_TREE" --captures "$FROZEN_CAPTURES" --out "$DECODE_PLAN" \
  --top20-source generation_verified --verification "$TOP20_TREE/VERIFIED.json"
"$HARNESS_PYTHON" -m d2.dataset build \
  --replay "$DECODE_PLAN" --out "$SHARDS" --taps-source decode_capture \
  --prompt-manifest "$PROMPT_MANIFEST" --sequence-context "$SECOND_PASS_CONTEXT"
"$HARNESS_PYTHON" -m d2.g1 plumbing --shards "$SHARDS"
```

Define TOP20_TREE, FROZEN_CAPTURES, DECODE_PLAN, SHARDS, PROMPT_MANIFEST and
SECOND_PASS_CONTEXT as absolute coordinator-selected paths. Preserve the verified
context's target, encoder, quantization, sampling and evidence receipts. Copy the
whole top20 tree when moving assembly hosts; its absolute plan/src/receipt paths
must remain accessible at their recorded paths. Do not rewrite paths or hashes
by hand; assemble on Controller or provide the exact recorded mount paths.

Layout: `seqs/<sid>.json` frozen state, `plan.jsonl` accepted by replay.read_plan,
`logprobs/<sid>.json` with seq_id/request_id/L/gen_start/from/tokens_sha256,
`logprobs/<sid>.npz` with top_ids int32 [L-gen_start,20], top_lp float32 of that
shape, tok_lp float32, tok_rank int32 and resid float32 [L-gen_start]. Metadata's
`d2g.top20...` request_id identifies the multi-segment export; original d2c request
IDs remain in frozen segment records. VERIFIED.json binds validator source,
plan, each state/array/metadata file and the real image/capture smoke receipt.

The R6 **chat receipt alone is inadequate** for completions. It still supplies
real image/capture provenance (checked locally against the unchanged smoke gate).
The collector's online validation and export-time revalidation of *all* frozen
rows replace chat's top20-format proof. The receipt binds those rows and generated
IDs; prepare checks every file, and build rechecks each sequence's hashes. A bare
R6 receipt is refused as generation verification. Replay-source checks are unchanged.
This does not prove exact equality to inaccessible full target logits or measure
live cost. The coordinator should inspect the first frozen segments and their
`top20_invalid`/PAUSE status before continuing full collection.

Capture coverage, newest-writer hashes, flush nonce, exact committed IDs, source
identities and the served-data context gate still apply. Assembly drops the final
sample together with its probability row because it has no required feedback tap.
No replay window is needed for successfully collected/verified generation top20.
A failed generation-mode collection is paused; it is never silently filled by replay.
The explicit replay mode remains available as a separate coordinator decision.

G1(a) plumbing still must pass on the resulting shards. G1(b)'s binding acap
sanity gate remains on original native decode acap trajectories; this collection
cannot replace or relabel them. Real GPU G1/training remains coordinator work.

## CPU evidence and costs

MEASURED: synthetic fake streams cover samples inside/outside top20, boundary
ties, malformed/missing row PAUSE, midstream foreign cancellation, checkpoints,
crash/resume, frozen export, decode preparation, synthetic capture build_record
and shard reading. Tests never create a production-usable synthetic smoke receipt.
The external-smoke gate is replaced only within the test; restoring it refuses
the synthetic tree. Tests: `test_generation_top20.py`, `test_generation_export.py`,
`test_option_ii.py`; existing D2 suites also run CPU-only.

MEASURED on Controller/Python 3.13.15, synthetic rows only: 30,000 row validations in
0.97024 s (32.34 us/row). Encoding one 16,384-row synthetic checkpoint took
0.1903 s and produced 10,846,239 bytes, excluding disk I/O/fsync. Synthetic mean
saved-row JSON was 657 B and wire-logprobs JSON 895.33 B. These are fixture sizes,
not legal-data or live inference measurements.

ESTIMATED per 1M generated tokens: normalized arrays are 172,000,000 bytes
(164.03 MiB) plus small NPZ/metadata headers; saved JSON roughly 0.7–1.3 GB and
additional wire JSON roughly 0.9–1.6 GB depending on IDs/float text. Frozen state
copies add another saved-JSON copy alongside active state. Exported arrays are
additional to it. At three full-length active streams and 30 s checkpoints, the
fixture implies ~32.5 MB rewritten per checkpoint and ~0.57 s serialization per
checkpoint (serial CPU). At 100 aggregate tok/s, row validation alone estimates
~0.32% of one CPU core. Parsing, allocations, disk and server-side logprob
serialization/gather add cost; no live throughput claim is made.
