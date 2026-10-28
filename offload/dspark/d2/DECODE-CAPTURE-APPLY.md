# D2 generation-time decode capture — coordinator apply plan

Prepared on Controller, CPU only. No apply, inference, service or GPU operation was
performed. Deepseek / coordinator owns independent review and every live command.
The smoke runner defaults to a synthetic offline rehearsal. Never run its execute
mode from a lane. It does not start/stop a service or manage a x86host worker.

## Pinned base and the actual change

Read-only inspection on 2026-10-24 found sparkhead main clean at
`21bed313ac803d38327c4de1ff2e5235f329c716`, tag `prod-k1-327-20261023`.
The live image is `sha256:435a5236ce3a7db9d45f5a2c483817d60ea629c4ea0a0b3af33da2ef8122334b`.
Base SHA256s:

- dsv41.env: `faa9d3e0019a8665f0b61dbf46a79134680731d43b9a9cb99ca2fa5af85373a8`
- start.sh: `fa172986fcb3e9ee66fd62b6dc11435ec7a56abdc60247ad1e73d85fa561ad5d`
- deployed remote speculator: `441be6b7d245114974c5dcb307c2efebd29516b958cd363bc70e24bcd73bdcf3`
- K1 manifest: `559c5867ba42aa7daa7b4a7c8231d15ee1d35be40ae5b715b013f973193c58de`

The capture candidate is production D: existing target, TP2 x86host drafter, async
process proxy, Engram REQUIRED=0, K1 pair3/drop-MTP/transient/alias/indexer4,
327680 context, 6442450944 KV bytes/rank, sequences4, batch4096 and the existing
speculation schedule. No worker/unit/GPU setting changes. Pin the reviewed lane
commit as SOURCE, install its three source files read-only in
`$HOME/dsv41-decode/SOURCE/{capture,spark/remote_dspark}` on BOTH ranks, and verify
bytes against the immutable Controller source. The remote file's pre-hook prefix is
byte-identical to the deployed SHA above. Do not replace the live remote directory.

The production diff is an append to dsv41.env only (SOURCE and RUN below stand for
the full reviewed SHA and a NEW safe run ID). start.sh stays byte-identical:

```diff
+# D2 reviewed decode-capture derivative; original K1 profile unchanged.
+OFFLOAD_DOCKER_ENV="$OFFLOAD_DOCKER_ENV VLLM_DSPARK_DECODE_CAPTURE_DIR=/d2-decode"
+OFFLOAD_DOCKER_ARGS="$OFFLOAD_DOCKER_ARGS -v \$HOME/dsv41-decode/SOURCE/capture/speculator_replay.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:ro -v \$HOME/dsv41-decode/SOURCE/capture/decode_capture.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/dspark/decode_capture.py:ro -v \$HOME/dsv41-decode/SOURCE/spark/remote_dspark/speculator.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/remote_dspark/speculator.py:ro -v \$HOME/dsv41-offload-run/d2-decode-RUN:/d2-decode:rw"
```

Render the exact concrete diff/hash with the reviewed module, against a COPY of
the pinned production root for review:

```bash
python -m offload.dspark.d2.decode_apply --root /review/base-copy --source "$SOURCE" --run "$RUN"
```

The unchanged K1 startup guard checks the promoted config hash. A naive env edit
would refuse and invoke its fallback. `decode_apply --apply --review receipt.json`
therefore also journals a reviewed derivative in logs/k1-promotion/files.json:
its original/profile fields stay intact, promoted.env becomes the rendered hash,
and d2_decode_derivative records the base config/record, source and review hashes.
It saves byte-exact env/start.sh/K1-record backups in NEW logs/d2-decode-RUN. The review
receipt must be `d2.decode-apply-review.v1`, approved=true,
coordinator=Deepseek/coordinator, source=SOURCE, base_commit=21bed313…,
base_env_sha256=the base hash above and capture_env_sha256=the rendered hash.
No tool fabricates this approval. Any other base or existing run refuses.

Stage/install sources separately as the coordinator before changing config; verify
all three destination file hashes and keep them immutable. Pre-create fresh
host-user-owned mode700 capture dirs on BOTH ranks (never let Docker create them),
check free space with df, and retain at least 3 GiB MemAvailable throughout smoke.
Collection is about 31 GB per million tokens plus ids/ledger; the former free-space
observation is not a current capacity promise. In-flight stages share a 256 MiB byte budget (64 MiB per-stage maximum),
including the writer's active stage. CPU prefill offsets trim transfers below r0;
writer work needs additional transient CPU memory. Three sequences have phased
256 MiB fdatasync/fadvise boundaries to avoid simultaneous syncs. Measure this in
the smoke rather than assuming CPU tests establish GB10 headroom.

## Opening window and three-prompt qualification on :8001

PAUSE any generation driver, confirm own streams aborted/saved and both production
gauges drained. Obtain 300 continuous seconds of zero running AND waiting gauges;
missing/nonfinite/negative metrics refuse. Record samples at intervals <=10 seconds.
Use the established coordinator HOLD/placeholder window protocol; it reserves both
Sparks. No production worker or x86host pair restart is needed.

In the opening window, start the experiment with the reviewed checkout's
`offload/launch/configs/D-decode-capture.env`, D2_DECODE_RUN=NEW_SMOKE_RUN. Its base
is D-kv6-async-pack with the two production env differences corrected, plus the
capture mounts. It is :8001 only; never start it beside a loaded production serve.

Prepare private `d2.decode-smoke-input.v1` JSON. `requests` has exactly three rows,
each `{endpoint, body, prompt_ids}`. prompt_ids are the exact server-rendered IDs,
including for chat. Bodies use stream=false, return_token_ids=true, <=256 output
for tagged requests (<=512 for the greedy reference) and <=8192 prompt tokens. First two use T=1/top_p=1, priority20 and tags
`d2c.smoke0.0.r0.prompt_length` / `d2c.smoke1.0.r0.prompt_length`. First is /v1/completions with raw prompt IDs;
second is /v1/chat/completions with the matching messages/template settings,
logprobs=true, top_logprobs=20 and return_tokens_as_token_ids=true. Third is an
untagged greedy request, T=0/top_p=1, matching the selected K1 production capture
(including its priority0/10 and completion or chat endpoint).

`greedy_reference` supplies {path, sha256, row, request_receipt,
request_receipt_sha256}. The capture file is the existing private K1 production
JSONL (for example campaign K1-production/greedy-post.jsonl). The request receipt
contains profile=K1-327680, capture_sha256, row, verified=true and the exact original
request body and endpoint. Request identity ignores only request_id, token-ID
serialization flags and an explicit stream=false default; all model, prompt/
template and sampling fields must match. Chat controls compare message.content;
completion controls compare choice.text. The historical JSONL alone contains output,
not input identity: recover and verify the original command/request before writing
this receipt. No guessed prompt is an acceptable greedy control.

The same SHA-bound request receipt must also contain cached_prompt_tokens and
latency_metrics={before, after}, the raw K1 metrics snapshots bracketing only that
exact greedy request. Both snapshots must include the sum and count series for
vllm:time_to_first_token_seconds and vllm:inter_token_latency_seconds. TTFT count
must rise by exactly one; the step count must rise by at least one. The pinned
image records inter-token latency once per engine output step (accepted drafts
share a step), so this measures mean step latency rather than dividing wall time
by accepted tokens. The live smoke records the same metrics around the untagged
control and requires its cached-token count to match the reference. Missing,
nonfinite, reset or contaminated timing telemetry refuses qualification. TTFT
and mean step latency must each be at most 1.05 times K1 (5% regression bound).
The report includes both timings, sample counts and ratios; raw metric snapshots
and timing evidence stay in private output. Existing output-only K1 JSONL is
insufficient for this performance qualification; supply verified timing evidence
from the matching K1 request before executing the smoke.

The window JSON contains coordinator, source, created_unix and drain_samples
[{t, metrics},…] proving that continuous five-minute interval; it must be <1h old.
It is evidence of the preceding drain, not permission to send production traffic.
The runner also checks HOLD, both rank runtime identities/hashes, every K1 overlay
file, correct legal-v2a root/context and zero :8001 gauges before each request.
Only the coordinator, from a clean immutable source with all required modules:

```bash
export D2_COORDINATOR=Deepseek/coordinator
bash offload/dspark/d2/decode_smoke.sh --execute --source "$SOURCE" --run "$NEW_SMOKE_RUN" \
  --inputs /private/smoke-input.json --window /private/window.json \
  --out /opt/dsv41/campaigns/DSV41-OFFLOAD/NEW_SMOKE_OUTPUT
```

The runner sends exactly three prompts. It supervises its owned HTTP children with
both-rank 3-GiB memory/STOP/time limits, checks raw echoed IDs, complete per-token
finite top-20 distributions ranked by value (chat returns exactly top_logprobs entries
with the sampled token first; a row whose sample is strictly the lowest entry may have lost
the true 20th and refuses, so an option (ii) collector must request top_logprobs 21) and
committed LPs, and requires DSpark's
draft-round counter to increase for option (ii). HTTP400/422/501 for option (ii),
missing logprobs or target-only execution select option (i), with no claim that
option (ii) works. Other refusals, ID mismatches or malformed responses fail.

After requests drain, it requests a nonce-bound durable flush, requires zero
writer errors, copies only the two tagged sequences (or one if option ii HTTP
rejected) using O_DIRECT reads, and checks every ledger/range/hash/ID/prompt-tail.
A captured untagged request fails. The untagged result must be byte-identical to
the selected K1 greedy capture; there is no noise tolerance. A mismatch is a HOLD
for review, even if historical long-context greedy runs sometimes diverged.
It rechecks both container IDs before writing a real receipt. Raw responses/taps
stay in private output, never git/report. Synthetic receipts cannot qualify data.
No restore or window-close command is hidden in this runner.

## Two production transitions for collection, and exact rollback

Collection has two coordinator production transitions: install/start the capture
configuration on :8000, then restore/start the original K1 configuration. The :8001
qualification adds an experiment launch in the opening window; do not describe
that extra launch as free or run concurrent Spark serves.

1. With HOLD retained after smoke, stop the experiment using the established
   coordinator script, create NEW capture dirs for collection (not smoke dirs),
   verify SOURCE/source hashes and review the concrete production diff. On
   sparkhead, run the reviewed CPU apply module against ~/dsv41-deploy with
   --apply --source SOURCE --run RUN --review the approved receipt. It changes no
   service. Verify the recorded derivative and original files; then use unchanged
   ./start.sh. Its K1 guard and K1 verify remain active. Inspect both actual mounts,
   image, capture activation, capacity/async process logs, K1 switches, KV/context,
   memory and expected code hashes. Save the apply/runtime receipt before releasing
   HOLD. Do not call a worker/unit command or replace the production x86host pair.
2. Close the window only after production checks pass. Start the reviewed generator
   on Controller with the frozen prompts and a NEW private state path such as
   /opt/dsv41/data/d2/gen-decode-SOURCE-RUN, streams3 and the explicitly
   approved Spark-hours cap. Pass --capture-error-host sparkhead and
   --capture-error-file /home/rank0user/dsv41-offload-run/d2-decode-RUN/ERROR_COUNT.json.
   The required background monitor polls every 0.5 s, latches the existing PAUSE
   file on the first error (or unavailable/malformed telemetry), and never removes
   PAUSE automatically. The coordinator can poll that same small count file.
   Do not copy/reuse first-pass state or budget. Resume
   only this second-pass state with the same capture tree. Normal priority0/10 work
   remains untagged and is never copied by capture; the reviewed driver cancels its
   own work on foreign demand. The prefix cache can omit prompt-tail rows: assembly
   refuses any holes rather than substituting replay data.
3. At completion or abort, PAUSE/cancel/save/drain tagged requests; request FLUSH.json
   with a NEW nonce and wait for FLUSHED.json matching nonce and the error-ledger count. Freeze the
   generation state and capture ledger. Copy exact files to a NEW private x86host tree
   using existing copy_to_x86host O_DIRECT/direct-link/idle/memory/rate-cap safeguards,
   hash both source stream and destination, and retain the flush controls/header/
   ledger and ERRORS.jsonl (if present) with taps/ids. Do not use replay.copy unmodified: it expects DONE.json,
   which decode mode deliberately does not produce. FLUSHED's ledger hash/file sizes
   plus newest-writer per-row hashes are the decode copy/assembly contract.
   An overflow drops only its stage and names seq_ids in ERRORS.jsonl; capture
   continues. Assembly refuses those sequences, and refuses all sequences for
   unattributable flush/ownership errors or a mismatched flush/error count. New tagged writes
   invalidate the snapshot. Newer segments may replace older-segment rows at or beyond the newer prompt
   length, with the superseded segment recorded. Prompt rows stay first-wins.
   Assembly verifies current rows against their newest writer, then requires
   ids.bin == final committed tokens on the entire required interval; never relabel data.
4. Open/drain the restoration window. Stop only this Spark deployment through the
   coordinator's normal protocol. Run the reviewed CPU rollback module:
   `python -m offload.dspark.d2.decode_apply --root ~/dsv41-deploy --run RUN --rollback`.
   It restores byte-identical dsv41.env AND the original K1 approval record, refuses
   unrelated edits and leaves backup/review evidence. start.sh is unchanged. Run
   unchanged ./start.sh and the existing production verification/greedy/memory
   checks; confirm no decode capture env/mount, then close HOLD. The mounted source
   files and capture data may remain inert for audit/copy; delete only lane-owned
   temporary smoke copies after review.

On any apply/start/smoke failure retain HOLD and use the rollback above after
stopping the candidate. The original K1 fallback is still active; if it triggers,
use the explicit fallback-recovery mode after stopping the candidate:

```bash
python -m offload.dspark.d2.decode_apply --root ~/dsv41-deploy --run RUN --rollback-after-k1-fallback
python offload/promote/k1/manage.py verify --root ~/dsv41-deploy --profile 327680
```

This mode proves that both current env/start.sh hashes match the pre-K1 originals
and the current K1 receipt is the exact reviewed derivative, validates all saved
K1 bytes, then restores env/start.sh/receipt byte-for-byte. It changes no service.
Unrelated edits or corrupt backups refuse before any restore. This revised apply
backs up start.sh too; older transaction folders without that saved file cannot
use this mode and require verified recovery material from the coordinator.
Capture also refuses tagged rows if enable_adaptive_verification is true,
records the named-sequence error and leaves the original proposer running.

Never claim an SSH disconnect restored production. No shared process killing, e-mail,
push, merge, x86host GPU/unit or placement operation is part of this plan.

## CPU assembly and training gate boundary

The later [OPTION-II.md](OPTION-II.md) collector/exporter supersedes the
replay-only collection guidance below when the coordinator selects reviewed
`--top20 generation`. It requires the completions-validation receipt as well as
the real image/capture smoke. The replay path below remains the explicit fallback.

Until generation logprobs are actually qualified and collected/normalized, use a
separate coordinator replay window for TOP20 ONLY with the second-pass exact token
states. Requests without d2r tags can use the existing replay parser on a normal
K1 replay configuration without the tap overlay; do not save/reuse prefill taps.
The replay parser's cache/refusal/echo/probability checks still apply. Prepare NEW
normalized metadata/arrays/plan as documented in DATASET.md, then d2.decode_dataset
and dataset build --taps decode_capture with the verified second-pass context.
Do not claim this path eliminates the top20 replay window or shortens the outage.

The optional generation_verified assembler input consumes that same normalized
array contract plus a successful real image-bound smoke receipt. The reviewed
generator changes its tag and adds capture-error PAUSE monitoring, and does not
collect streaming generation logprobs;
option (i) is the complete collection path delivered here. Option (ii)'s transport
and normalization must be reviewed before choosing it for full collection.

No equivalence gate applies to decode taps. G1(a) shard/CaptureSet plumbing still
must pass. G1(b) must use the original native acap decode evaluation and its own
sampling/context receipt; this assembler refuses to relabel old acap prefill rows
or newly generated answers. Real G1 and training remain coordinator work.
