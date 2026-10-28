# UP2 low-cap coordinator handover

Runner built and rehearsed offline; **never executed on hardware by this lane**.
Production qualification and speed remain NOT RUN. Keep replay OFF/HOLD.
Use W8's direct-start pattern, with owned process supervision, rather than WR1:
WR1's required WINDOW_HOOK adapter is not implemented. This runner directly
implements starts, runtime checks, API probes, cold cells and summaries. It does
not implement native scheduler/fault injection. Default qualification remains
fail-closed: missing native evidence records NOT RUN/HOLD and stops. The explicit
`--screen` path below continues missing native checks as NOT RUN, labels all
request/needle results, ledgers, checkpoints and summaries SCREEN with
`screen: true, adoptable: false`, and produces only PASS-SCREEN qualification.
Every existing API safety/numerical/needle gate remains hard. A supplied native
packet is always validated; invalid supplied evidence still stops. The native
contract and its 12 checks are unchanged. Stage B builds the native apparatus
and fully qualifies only if the measured 300K screen gain exceeds OFF-A/OFF-B
spread. Production stays K1, replay OFF, regardless of the screen outcome.

Read [identity/equality table](UP2-IDENTITY.md), [failure rehearsal](UP2-REHEARSAL.md)
and [owning recipe](../../upstream_adopt/UP2-GB10-TEST-PLAN.md). The newer lane
brief's3GiB resource floor and safety-only hard gates govern this runner.
Timing, batch-shape numerical differences and batch text comparisons are records,
never automatic STOP/adoption gates. The recipe's matched numerical floor,
isolated stable short greedy check, needles and provenance are correctness gates.

## Preparation and exact coordinator commands

Only Deepseek / coordinator executes. Independently review/integrate the lane,
then stage the identical reviewed checkout to sparkhead's `~/dsv41-deploy-offload`.
Source/config hashes and the full expected experiment commit must agree; a dirty
local or remote tracked tree refuses. Source input drift requires owner review and a new
pinned build, never forced pins or worker replacement. Review the stats-flush
helper too: it requests the existing experiment proxy's FLUSH_REQ/ACK, bounded2s,
without reading/writing tensors or importing torch/CUDA. This avoids the known
1024-round stale-file trap. Engine audit counters still need native evidence.
X86host is read only throughout; its worker/GPU units/clocks are never changed.

First, on Controller CPU in the reviewed integration checkout:

```bash
source ~/.config/ndm/paths.sh
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
bash offload/launch/windows/up2_low.sh --dry-run
```

### SCREEN: diagnostic, then 300K, then optional 150K windows

These are coordinator commands for later review and execution, **not commands
executed by this lane**. First independently integrate/review and stage the same
commit on both sides. Reserve/open the diagnostic HOLD using the existing
coordinator procedure, with the exclusive experiment tunnel, paused/finished D2,
saved K1 rollback and no other producer. `--screen` is required on EVERY window.
Omit the absent native directory and leave native wait at its default zero:

```bash
source ~/.config/ndm/paths.sh
export WINDOW_COORDINATOR=Deepseek/coordinator
export UP2_REVIEWED_COMMIT=$(git rev-parse HEAD)
export UP2_SCREEN_DIAG_OUT=/opt/dsv41/campaigns/DSV41-OFFLOAD/UP2-screen-diagnostic-$(date -u +%Y%m%dT%H%M%SZ)
bash offload/launch/windows/up2_low.sh --execute --screen --phase diagnostics \
  --expected-experiment-commit "$UP2_REVIEWED_COMMIT" \
  --out "$UP2_SCREEN_DIAG_OUT"
```

Inspect PASS-SCREEN, `native: "NOT RUN"`, `adoptable: false` and both
`diag-{off,on}-native-NOT-RUN.json` files (all 12 checks NOT RUN). Missing native
q uses **ESTIMATED** CPU q4092 and tails147313/147439/147440/147441, explicitly
unverified in each `*-tails.json`; those chat lengths alone do not prove final
scheduler remainders. The reviewed engine/proxy telemetry exposes audited-round
counts, not chunk q, so it cannot establish a MEASURED q. The saved proxy snapshot
and source description explain the fallback. If measured q.json is supplied via
`--native-evidence`, its q is validated and used with that path cited; a missing
complete receipt still leaves native NOT RUN/tails unverified. Never infer q from
MAX_BATCH, a round count or the CPU expectation. Future q telemetry needs a
reviewed extractor before it can replace this fallback.

Restore/check/close the diagnostic window using the printed W8 commands and the
named restoration gates below. Reserve/open a NEW HOLD, keeping the identical
reviewed source/config/worker and the actual diagnostic path from above:

```bash
source ~/.config/ndm/paths.sh
export WINDOW_COORDINATOR=Deepseek/coordinator
bash offload/launch/windows/up2_low.sh --execute --screen --phase measured --contexts 300000 \
  --expected-experiment-commit "$UP2_REVIEWED_COMMIT" \
  --qualification "$UP2_SCREEN_DIAG_OUT" \
  --out /opt/dsv41/campaigns/DSV41-OFFLOAD/UP2-screen-300K-$(date -u +%Y%m%dT%H%M%SZ)
```

Restore/check/close the 300K window before considering the optional 150K window.
Reserve/open another NEW HOLD, retaining the same reviewed commit and diagnostic
qualification; do not update either variable to a different source/run:

```bash
source ~/.config/ndm/paths.sh
export WINDOW_COORDINATOR=Deepseek/coordinator
bash offload/launch/windows/up2_low.sh --execute --screen --phase measured --contexts 150000 \
  --expected-experiment-commit "$UP2_REVIEWED_COMMIT" \
  --qualification "$UP2_SCREEN_DIAG_OUT" \
  --out /opt/dsv41/campaigns/DSV41-OFFLOAD/UP2-screen-150K-$(date -u +%Y%m%dT%H%M%SZ)
```

Keep one fixed run-directory salt prefix inside each window. Each context window
contains its own OFF-A/ON/OFF-B arms and all three repetitions. Never combine
arms or contexts from different windows into a runner summary or Stage-B result.
A 150K-only summary records Stage B INCOMPLETE; only the complete 300K window can
establish it. `--contexts` accepts a unique nonempty subset of150000/300000,
requires `--phase measured`, and defaults to both when omitted (legacy commands
retain their schedule). The diagnostic matrix is unchanged.

The measured phase rechecks the qualification's source/config/worker/commit,
artifact hashes, missing-check records and any supplied native receipts. Without
`--screen`, PASS-SCREEN is refused before remote operations. Each measured cell
and the summary remain SCREEN/non-adoptable even if supplied native checks pass.
Inspect results using [the SCREEN result template](UP2-RESULTS-TEMPLATE.md), then
restore/check/close. No screen outcome authorizes adoption or a larger cap.

### Full native qualification (unchanged default)

Once the native apparatus described below is actually available, prepare its
private output root and reserve/open the diagnostic HOLD through the existing
coordinator procedure. The runner **never** opens/closes HOLD. It reads the
existing HOLD timestamp and cannot renew the75min submission/90min hard bounds.
The reserved experiment tunnel must already be127.0.0.1:18001→experiment:8001.
No other producer may use either endpoint during this experiment; D2 must be
finished or paused. Save/verify current K1 before the window using W8's named
production gates. The runner additionally saves both-rank production image,
target metadata, Engram shim/source/manifest/JIT/transport hashes, env and complete launch argv in its rollback manifest.

Diagnostic window, exact invocation from that reviewed checkout:

```bash
source ~/.config/ndm/paths.sh
export WINDOW_COORDINATOR=Deepseek/coordinator
export UP2_REVIEWED_COMMIT=$(git rev-parse HEAD)
export UP2_NATIVE_EVIDENCE=/opt/dsv41/campaigns/DSV41-OFFLOAD/UP2-native-qualification
export UP2_DIAG_OUT=/opt/dsv41/campaigns/DSV41-OFFLOAD/UP2-low-diagnostic-$(date -u +%Y%m%dT%H%M%SZ)
bash offload/launch/windows/up2_low.sh --execute --phase diagnostics \
  --expected-experiment-commit "$UP2_REVIEWED_COMMIT" \
  --native-evidence "$UP2_NATIVE_EVIDENCE" --native-wait-seconds 900 \
  --out "$UP2_DIAG_OUT"
```

Native apparatus owns only its deterministic experiment probes. It watches the
run-local `diag-{off,on}-await-{q,complete}.json` checkpoints, containing actual
container IDs, source digest, requested packet and deadline. While the runner
waits, memory/STOP/hard-bound supervision continues. Publish each packet atomically
within the explicit900s checkpoint bound and the original submission deadline.
Default wait0 refuses missing evidence immediately. No receipt can grant a later
deadline. Ordinary native fault probes must be matched OFF/ON; no production
worker kill, unit edit, random fault traffic or unreviewed PF1/SO2 composition.

After diagnostics, inspect the evidence and restore/check/close with the printed
W8 commands. Use a **new separately reserved HOLD** for measured-only execution,
on the exact same experiment source/config/worker. Keep UP2_DIAG_OUT pointing to
the successful diagnostic run (not a guessed path):

```bash
source ~/.config/ndm/paths.sh
export WINDOW_COORDINATOR=Deepseek/coordinator
bash offload/launch/windows/up2_low.sh --execute --phase measured \
  --expected-experiment-commit "$UP2_REVIEWED_COMMIT" \
  --qualification "$UP2_DIAG_OUT" \
  --out /opt/dsv41/campaigns/DSV41-OFFLOAD/UP2-low-measured-$(date -u +%Y%m%dT%H%M%SZ)
```

Prior source/config/worker/experiment identity and diagnostic artifact hashes are
rechecked; both native receipts and their original evidence hashes are revalidated.
`--phase all` expresses the complete sequence but is unsuitable for one90min
reservation. Use the separately reserved windows, including for SCREEN.

**ESTIMATED window budgets, corrected against S30's measured inputs:**
S30.report.md (campaign, sections5/6) records start-to-drafter-ready6.5–6.8min,
150K cold TTFT68.6–69.4s, 300K140.2s, decode52–56tok/s and prompt construction
19–33s per request (20–23 tokenize calls). These are **MEASURED historical inputs**,
not new inference measurements. Control-plane10–20s/request and audit overhead
remain **ESTIMATED / NOT MEASURED**. No prompt caching is implemented.

| Window | CPU schedule: submissions / prompt tokens / max output | Total ESTIMATED | Margin to75min /90min at upper estimate |
|---|---|---|---|
| SCREEN diagnostic |28 /4,246,152 /2,106|62–68min|7min /22min|
| SCREEN measured300K first |12 /2,796,000 /13,596|55–59min|16min /31min|
| SCREEN optional measured150K |12 /1,446,000 /13,596|43–48min|27min /42min|
| Legacy measured both contexts |21 /4,146,000 /27,096|75–81min|exceeds submission cutoff by6min /9min|

Counts are **MEASURED CPU synthetic schedule counts**, not inference results.
Diagnostic includes2 starts; each measured window includes3 starts/3 warmups and
9 cold cells. S30 estimates include decode, repeated tokenization and checks.
Add approximately7min **ESTIMATED** plus restore checks for restoration. At the
upper estimates, diagnostic+restore75min leaves15min to90min before extra checks;
300K+restore66min leaves24min; 150K+restore55min leaves35min. Audit and operational
variance can consume that margin, so no estimate guarantees completion.
The legacy combined measured window is unsuitable: its upper estimate exceeds
75min, and81+7=88min leaves only2min for restore checks. Diagnostic plus300K is
117–127min **ESTIMATED** before restoration, unsuitable for one90min HOLD.

Never extend deadlines or reduce output/repetitions to force completion. The
existing HOLD timestamp owns75min submissions and90min hard supervision; request
wall bound remains900s. The runner never restores automatically. Stop early enough
to drain/cancel owned work and restore inside the reservation, keeping partial
cells missing. A local abort does not prove remote cancellation. Use a new HOLD
for each window, same reviewed commit/config/worker/qualification and a unique
fresh output directory; no splicing. Full native qualification's waits/apparatus
require a separate budget. No native wait is budgeted for SCREEN.

## Native evidence contract — apparatus not supplied

Provide `<native-root>/diag-off/{q,qualification}.json` and the matching diag-on
files. Every packet requires schema `up2-native-v1`, classification `MEASURED`,
actual `arm`, both actual `container_ids`, `manifest_sha256`, an integer observed
`q`, and a nonempty `evidence` array of `{path,sha256}`. Paths must be inside the
native evidence root. The q packet describes the solo chunk observed in that
start, not the CPU expectation4092. The runner derives36*q+{1,127,128,129}, sends
those exact chat lengths, and sends327675+5 with EOS ignored for a length finish.

A complete packet additionally requires `tail_sizes` equal to the sent derived
sizes, positive `audited_rounds`, actual `request_events` and `checks` mapping
**every** name below to `PASS`. `request_events` uses the same send_intent,
http_opened, completed/failed/canceled/not_sent events as the runner; unresolved
native requests refuse. Include failed/unsupported probes explicitly, rather
than constructing passing receipts. Any NOT RUN check retains HOLD.

- rank_q_kept_position_valid_anchor_owner_agreement
- final_need_draft_real_complete
- source_kv_writes_retained
- step_generation_echo_and_uniform_broadcast
- near_cap_k4_k5_sync_async_clamp
- remaining129_256_257_and_leftover64_128_peer_progress
- partial_full_hits_async_restore_real_suffix
- cancel_slot_reuse_preempt_backtrack
- disconnect_before_after_final_watchdog_late_reply
- mixed_rejected_rows_mm_logprobs_exclusions
- native_dispatch_unchanged
- no_local_drafter

Attach actual observations, not blanket approval prose: two-rank q/kept/position/
valid/anchor/owner metadata; real final suffix and KV-write comparisons; STEP
generation echoes; K4/K5 sync/async length-boundary scheduling where supported;
ready-successor/decoder progress; cache/async-restore/cancel/reuse/preempt/backtrack/
disconnect/watchdog/late-reply results; rejected/MM/two prompt-logprobs exclusions.
The built-in API near-cap probe alone covers the current production execution
mode; it does not establish all K/mode cases. Unsupported modes are NOT RUN/HOLD
until the owning qualification plan explicitly resolves them.

Receipt checking binds evidence to the current start and proves file integrity;
it does not independently implement or certify those native tests. There is no
fixture-to-live bypass. With today's absent apparatus, the usable deliveries are
the dry plan, offline rehearsal and fail-closed runner; a full native result is
not yet available. The explicit SCREEN exception above records missing evidence
and never changes that qualification claim.

## Outputs and recovery

All outputs use exclusive create in a fresh campaign directory. Expect:
`deadlines.json`, `rollback.json`, `coordinator-next.sh`, continuous both-rank
memory JSONL, numbered private command logs,42-file verifier evidence per rank,
per-start identity, native checkpoints/receipts, numerical/needle summaries,
per-cell hashes/usage/TTFT/channel counts/finish, durable `requests.jsonl`,
`checkpoints.jsonl`, final `accounting.json`, `summary.json`, and successful
`qualification.json`. On failure: STOP plus completed cell files/checkpoints;
no missing cell is promoted to complete. Submission-cutoff STOP names the exact
submission deadline, including refusals inside the request subprocess; hard-bound
STOP names the90-minute hard deadline. Unrelated child failures retain their own
error label. Cold timing uses fixed salts across
OFF-A/ON/OFF-B, same public corpus/header/seed42, exact150000/300000,1500 output,
thinking high/75, T1/top_p1/priority10. Warmup32K generates32 tokens and is excluded.

Per-request summaries retain no prompt or generated text. Do not publish private
command logs or original native captures. A transport intent without HTTP headers
is counted uncertain; the coordinator reconciles admission/cancellation with its
own metrics. Optional4-way needles, mixed delivery-event stalls and a completed
representative task are not automated here; retain NOT RUN until separately run.

Every exit prints a production revision/diff check, then:

```bash
bash /opt/dsv41/campaigns/DSV41-OFFLOAD/W8/restore.sh start
bash /opt/dsv41/campaigns/DSV41-OFFLOAD/W8/restore.sh check
bash /opt/dsv41/campaigns/DSV41-OFFLOAD/W8/restore.sh close
```

These commands are printed, never executed by the runner. Inspect rollback.json,
drain/explicitly cancel only window-owned work, and require K1 START VERIFIED,
cap327680, KV2502600, greedy6/6 against K1-production/greedy-post.jsonl and x86host
PRODUCTION HOLD OK. Close only once all restore gates and the existing short
reasoning/tool/structured smoke pass. No worker/unit changes or promotion.

## Stage B and later qualification

Stage B (native apparatus, full qualification, and consideration of adoption)
proceeds only if the **MEASURED** 300000-token ON prefill/TTFT gain is greater
than the paired OFF-A/OFF-B spread. Use all three usable matched repetitions:
`ratio_i = ON_i / mean(OFF-A_i, OFF-B_i)`,
`gain = 1 - median(ratio_i)`, and
`spread = median(abs(OFF-A_i - OFF-B_i) / mean(OFF-A_i, OFF-B_i))`.
Report the ratios, gain and spread with exact sources; require positive gain
strictly greater than spread. Missing/unusable pairs mean the criterion is
incomplete, not passed. TTFT remains an end-to-end prefill proxy, not a GPU timer.
`summary.json` records this criterion under `stage_b` using only the current
window's measured300K cells; missing300K or any unusable pair is INCOMPLETE.
`contexts`, `window_id` and `expected_cells` identify the selected schedule.
Summaries reject absent/mixed window identities, unexpected contexts and duplicate
cells, and continue to require paired prompt/salt hashes across arms.
This is the coordinator's Stage-B planning criterion, never a runner safety STOP
or automatic adoption gate. A null/below-spread screen does not justify Stage B.

Before any later adoption, pass every native/source/worker/resource/needle/
numerical gate, all restoration gates, decoder-only A/A and mixed per-stream
delivery-event checks, and representative completed-task review. No screen
result is adoptable. The upstream478K/949K claims do not establish local gain or
authorize a larger cap. Explicit524288 profile/window authorization remains
required;400/450/500K execution is **not built here**.
