# PF1-R2 replacement GB10 window recipe

This **replaces** the original PF1 recipe. Only coordinator Deepseek / coordinator
may execute its Spark/API/service commands, after D2 generation/native W8 and
other owners release the window. PF1-R2 executed none of them. Keep 524288 on
HOLD. No 1M experiment, attention change, 0075, workspace candidate, factor-1,
clock, concurrency or drafter change belongs in this attribution arm.

The shared public family is the existing CourtListener opinions + RECAP corpus,
fixed filenames, seed42 and order at every size. One common instruction/header
precedes the truncated public material. Actual complete-chat /tokenize and
server usage must be exactly 150000/300000/400000/450000/500000. There is no
approximate length fallback. Corpus hashes and prompt hashes are recorded,
never prompt/generated text. Same seed, priority10, temperature1, top_p1,
thinking=true/high, 1500 output reserve at every measured point. The full
500K request uses at most 501500 total; remaining context is 22788. Output
may end in reasoning; record that honestly and use separate task-completion
qualification before promotion.

Each point has: cold (fresh salt, cache0), full warm (same prompt/salt as cold),
and partial warm (fresh salt, prime at point−100000, then extend to full point).
Prime uses 64 output tokens and is **not a measured generation comparison**.
Actual cached/uncached counts are asserted; alignment/replay may leave up to
102048 uncached at the partial point. Full warm may leave up to2048. Investigate
an out-of-bound hit instead of relabeling it cold. B1 only. All admitted work
must finish; idle check is not protection against future arrivals, so the
coordinator must own the exclusive benchmark window and retain demand guard.

## Preparation (Controller, CPU only)

Use a reviewed experiment checkout containing these lane files and the current
K1/process proxy. PF1-R2 made no merge. Do not overwrite files used by D2. If
any pinned K1/image base changed, rebuild/review pins against the actual
composed source first; never force hashes to pass.

```bash
source ~/.config/ndm/paths.sh
set -euo pipefail
cd /opt/dsv41/repos/.worktrees/dsv41-pf1
PF1_RUN_DIR=$(mktemp -d /opt/dsv41/campaigns/DSV41-OFFLOAD/PF1-R2-window.XXXXXX)
printf '%s\n' "$PF1_RUN_DIR" > /opt/dsv41/campaigns/DSV41-OFFLOAD/PF1-R2-window-current.txt
CUDA_VISIBLE_DEVICES='' "$HARNESS_PYTHON" -m offload.prefill_trace.build_overlay \
  --enabled --output-root offload/prefill_trace/generated
(cd offload/prefill_trace/generated && sha256sum -c ../OVERLAY.sha256)
"$HARNESS_PYTHON" -m offload.prefill_trace.emit_verify > "$PF1_RUN_DIR/verify-off.py"
"$HARNESS_PYTHON" -m offload.prefill_trace.emit_verify --enabled > "$PF1_RUN_DIR/verify-on.py"
sha256sum offload/prefill_trace/BASES.json offload/prefill_trace/OVERLAY.sha256 \
  offload/launch/configs/PF1-R2-trace.env /opt/dsv41/data/bench-data/{opinions_cl,filings_recap}.jsonl \
  > "$PF1_RUN_DIR/sources.sha256"
# Plan only: no SSH, API, tokenize, GPU or service call.
"$HARNESS_PYTHON" -m offload.prefill_trace.window_requests --cap 327680 \
  --contexts 150000 300000 --trace-run plan --output "$PF1_RUN_DIR/plan-only.jsonl"
```

Default-off config needs no generated sources. Enabling replaces two K1 target
mounts, never overlays them twice. The diagnostic sources refuse non-eager
prefills; first try the normal K1 policy. If any real prefill has graph mode
other than NONE, stop that diagnostic. Use the separately matched eager pair
below on **both** caps and controls, rather than silently changing one arm.
The existing model's full4096-row prefill path normally is eager; this still
needs runtime verification, especially the final short tail.

Only after coordinator review/staging authorization, copy this small code and
its generated sources to the experiment checkout (not weights/model files):

```bash
source ~/.config/ndm/paths.sh
set -euo pipefail
cd /opt/dsv41/repos/.worktrees/dsv41-pf1
tar -C offload --exclude=__pycache__ --exclude=.pytest_cache \
  -czf "$PF1_RUN_DIR/trace-code.tgz" prefill_trace
scp "$PF1_RUN_DIR/trace-code.tgz" sparkhead:dsv41-deploy/offload/
ssh sparkhead 'tar -xzf "$HOME/dsv41-deploy/offload/trace-code.tgz" -C "$HOME/dsv41-deploy/offload"'
scp offload/launch/configs/PF1-R2-trace.env sparkhead:dsv41-deploy/offload/launch/configs/
```

Stage only current sources, excluding __pycache__/test scratch. Coordinator
must reconcile its current production ledger first: target manifest/image,
K1 packing/factor4/KV6, native kernel revisions, TP2 worker source/adapter label,
process proxy and clocks. Require worker position support≥524288 already;
this recipe **does not change or restart the x86host pair**. If that is not
available, stop and return a dependency to the coordinator, not a worker edit.

## Window and arms (coordinator on sparkhead)

```bash
source ~/.config/ndm/paths.sh
set -euo pipefail
cd "$HOME/dsv41-deploy"
bash offload/kvpack/stage_overlay.sh verify-checkout
bash offload/launch/window.sh open 'PF1-R2 matched prefill attribution; no attention changes'
# Primary normal-policy OFF arms, sequentially, each followed by its own measurements.
PF1_CAP=327680 PF1_TRACE=0 bash offload/launch/start_exp.sh offload/launch/configs/PF1-R2-trace.env
# AFTER collecting this arm, start the second:
# PF1_CAP=524288 PF1_TRACE=0 bash offload/launch/start_exp.sh offload/launch/configs/PF1-R2-trace.env
```

Do not run consecutive starts without collecting the first arm. Read actual
readiness/identity logs; a sleep is not readiness. Use `/v1/models` to require
selected cap524288 or327680, correct alias/root; save exact KV/load and complete
container argv/env on both ranks. For OFF, feed verify-off.py into both containers;
for ON use verify-on.py. K1's original verify-running hashes deliberately differ
for two trace targets; do not describe that checker as passed on an ON arm.

On Controller (coordinator only; no forwarder daemon), open a separate terminal:

```bash
source ~/.config/ndm/paths.sh
PF1_RUN_DIR=$(cat /opt/dsv41/campaigns/DSV41-OFFLOAD/PF1-R2-window-current.txt)
# Set once after open, before the first arm; exclusive creation prevents renewal.
(set -o noclobber; date -u -d '+150 minutes' +%s > "$PF1_RUN_DIR/submission-deadline.txt")
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18001:127.0.0.1:8001 sparkhead
```

For **each arm**, select PF1_ARM explicitly on Controller (e.g. off-327-a,
off-524, on-327, on-524, off-327-return):

```bash
source ~/.config/ndm/paths.sh
set -euo pipefail
cd /opt/dsv41/repos/.worktrees/dsv41-pf1
PF1_RUN_DIR=$(cat /opt/dsv41/campaigns/DSV41-OFFLOAD/PF1-R2-window-current.txt)
PF1_ARM=off-327-a
PF1_CAP=327680
PF1_VERIFY=off
for PF1_NODE in sparkhead sparkworker; do
  ssh "$PF1_NODE" 'docker exec -i dsv41-serve python3 -' < "$PF1_RUN_DIR/verify-$PF1_VERIFY.py"
  ssh "$PF1_NODE" 'docker inspect --format "{{json .Config}}" dsv41-serve' > "$PF1_RUN_DIR/$PF1_ARM-$PF1_NODE-config.json"
  ssh "$PF1_NODE" 'awk "/MemAvailable|SwapFree|SwapTotal/{print}" /proc/meminfo'
done
curl -fsS http://127.0.0.1:18001/v1/models > "$PF1_RUN_DIR/$PF1_ARM-models.json"
curl -fsS http://127.0.0.1:18001/metrics | awk '
  /^vllm:num_requests_running\{/{r=1; total+=$NF}
  /^vllm:num_requests_waiting\{/{w=1; total+=$NF}
  END{exit !(r && w && total==0)}'
```

Before first timed requests, discard one symmetric 32K warmup on every start,
using the existing bounded capture helper with phase none. Verify its actual
usage/worker rounds; an EOS one-token output is not a1500-token warmup.

```bash
source ~/.config/ndm/paths.sh
"$HARNESS_PYTHON" -m offload.gb10_budget.capture --execute --ctx 32000 \
  --batch 1 --phase none --output "$PF1_RUN_DIR/$PF1_ARM-warmup.json"
# Low-cap arm: two points. High-cap arm: all five. Select after readiness.
PF1_POINTS=(150000 300000)
if [ "$PF1_CAP" = 524288 ]; then PF1_POINTS=(150000 300000 400000 450000 500000); fi
"$HARNESS_PYTHON" -m offload.prefill_trace.window_requests --execute \
  --cap "$PF1_CAP" --contexts "${PF1_POINTS[@]}" --reps 2 \
  --trace-run "$PF1_ARM" --deadline-utc "$(cat "$PF1_RUN_DIR/submission-deadline.txt")" \
  --output "$PF1_RUN_DIR/$PF1_ARM-requests.jsonl"
```

For returned baseline run just `--contexts 150000 300000 --modes cold --reps 2`.
Require the same corpus/prompt hashes at identical points across caps/arms.
The request driver is default dry-plan and will not contact anything without
--execute. It writes numeric usage, SSE delivery times, opaque trace request ID
and source hashes only. It enforces literal500K, output reserve and cap identity.
Its idle check is necessary but the coordinator demand guard remains mandatory.

## Both-Spark telemetry (coordinator only, each arm)

Run the following **on each Spark**, in a unique coordinator-owned tmux session,
before starting that arm's requests. The script is stdlib only: no torch/CUDA
context, model build or service change. Keep the existing coordinator memory
watcher/demand guard too. Its errors must surface to the window owner.

```bash
# On EACH node: select this arm's unique name; same name as API --trace-run.
source ~/.config/ndm/paths.sh
PF1_ARM=off-327-a
mkdir -p "$HOME/dsv41-offload-run/pf1-trace/$PF1_ARM"
tmux new-session -d -s "pf1-telemetry-$PF1_ARM" \
  "python3 \$HOME/dsv41-deploy/offload/prefill_trace/telemetry.py --seconds 7200 --output \$HOME/dsv41-offload-run/pf1-trace/$PF1_ARM/telemetry.jsonl 2> \$HOME/dsv41-offload-run/pf1-trace/$PF1_ARM/telemetry-error.log"
```

Records each second: UTC/monotonic time, MemAvailable/SwapFree/SwapTotal, UUID,
SM/memory clocks, power/limit, temperature, utilization, active throttle mask.
If a query field is unsupported, this script refuses visibly: query supported
names with nvidia-smi --help-query-gpu and review a replacement before continuing;
do not silently omit throttle data. No power/clock changes. Sample before and
after timed intervals; synchronize hosts with their standing time service and
record clock offset read-only (`chronyc tracking`, if installed). No NTP changes.

## Instrumented cap pair, isolated from headline runs

After uninstrumented OFF runs, for each ON arm prepare that unique directory on
BOTH nodes, then start sequentially on sparkhead:

```bash
source ~/.config/ndm/paths.sh
set -euo pipefail
cd "$HOME/dsv41-deploy"
# ON327; collect fully, then ON524. Do not reuse RUN across process starts.
for PF1_NODE in sparkhead sparkworker; do
  ssh "$PF1_NODE" 'mkdir -p "$HOME/dsv41-offload-run/pf1-trace/on-327"'
done
PF1_CAP=327680 PF1_TRACE=1 PF1_TRACE_RUN=on-327 \
  bash offload/launch/start_exp.sh offload/launch/configs/PF1-R2-trace.env
# After collecting on-327, prepare on-524 dirs and start:
# PF1_CAP=524288 PF1_TRACE=1 PF1_TRACE_RUN=on-524 \
#   bash offload/launch/start_exp.sh offload/launch/configs/PF1-R2-trace.env
```

Measure ON arms with the same driver command, --reps1 initially, PF1_VERIFY=on,
and matching --trace-run. Repeating only when residual/noise demands it keeps
profiling bounded. Need≥5GiB MemAvailable per node at start; no new swap.
Collect both node directories with scp into fresh per-node paths under
PF1_RUN_DIR. Do not infer GPU-rank identity from local_rank (both hosts local0).
Use actual JSON rank fields0/1. The driver sets request_id explicitly; this
installed API passes a single `chatcmpl-…` ID through to the engine, matching
its returned SSE id. Verify the numeric artifact's hashed trace_request against
trace members; if none match, inspect the installed API ID path and stop fitting.

If graph tails refuse, use a separate diagnostic config on sparkhead:

```bash
source ~/.config/ndm/paths.sh
cd "$HOME/dsv41-deploy"
cat > offload/launch/configs/PF1-R2-eager-window.env <<'SH'
. offload/launch/configs/PF1-R2-trace.env
EXP_EXTRA_ARGS="${EXP_EXTRA_ARGS:+$EXP_EXTRA_ARGS }--enforce-eager"
SH
# Run eager OFF327/OFF524 first, then eager ON327/ON524 using the same flags
# and driver commands as above, fresh arm names/directories and this config.
```

Eager OFF must reproduce the cap effect before its trace explains the normal
policy loss. If it eliminates/changes the loss, report that branch explicitly:
compiled/graph/native dispatch is now implicated, not proven. Use SP1's
bounded two-rank timeline recipe rather than silently extrapolating eager
measurements to production. Remove this temporary config after the window.

## Fit and interpret (Controller CPU only)

After copying fresh trace files to `PF1_RUN_DIR/on-{327,524}/{sparkhead,sparkworker}/`,
run (paths contain no private text):

```bash
source ~/.config/ndm/paths.sh
set -euo pipefail
cd /opt/dsv41/repos/.worktrees/dsv41-pf1
"$HARNESS_PYTHON" -m offload.prefill_trace.fit_cost \
  --traces "$PF1_RUN_DIR"/on-{327,524}/{sparkhead,sparkworker}/prefill-r*-p*.jsonl \
  --requests "$PF1_RUN_DIR"/on-{327,524}-requests.jsonl \
  --output "$PF1_RUN_DIR/chunk-cost-fit.json"
```

The fit requires both ranks for every selected full4096-token B1 cold chunk,
matching begins/completed CUDA totals and≥3 overlapping prefix points at both
caps. It excludes primes/hits/partial tails/mixed batches from the regression;
their traces still verify controlled cache behavior. Repeated requests at a
cap/prefix are reduced to a median before fitting. It fits separately per rank,
on **common prefix support**:

`chunk_ms = a + b*(prefix/100K) + c*high_cap + d*high_cap*(prefix/100K)`.

b identifies within-cap growth; c identifies a cap offset at matched prefix;
d identifies a changed slope. All high-cap long-only points remain in the
output to inspect curvature/thresholds; they cannot alone identify c. Inspect
per-request curves, both-rank skew, residual spikes and exact dispatch/work
sizes. A step discontinuity with a dispatch/subchunking change calls for a
piecewise follow-up, not a forced linear explanation. Report family coverage
ONE and correlated requests/chunks; no IID chunk error bars or universal law.

Compare host total vs CUDA interval for enqueue/gap evidence; compare gather,
MQA/top-k, metadata bounds, workspace pressure and actual sparse dispatch at
matching prefix/layer. Overlay memory/power/clocks/throttle. A group collective
interval includes transfer and waiting; two-rank arrival/dependency traces are
needed for pure exposed wait, including fused paths absent from Python spans.
Use wall-time unions/dependencies, never sums of nested/overlapping events.
Headline speed comes from OFF TTFT and full-response timing, never trace sums.

## Durations, stop rules, restoration and expected artifacts

**ESTIMATED**, using saved ~0.46–0.64ms/token cold costs: OFF pair all three
cache modes/two repetitions ~90–110min including generation/tokenization;
ON pair one repetition ~45–60min plus instrumentation; startup ~4–8min each.
Request HTTP timeout900s. Split into separately reserved windows if needed;
never claim a completed matrix across mismatched target/worker identities.
Hard wall budget180min per reservation; stop new submissions at150min (the driver enforces the shared --deadline-utc), let the
current request finish under its900s bound, then restore. Long shared-corpus
tokenization can dominate CPU/API time; retain that outside TTFT, not as prefill.
If this budget cannot fit all arms, complete and record the measured subset,
restore production, and schedule the next window. Required checks remain open.

Stop immediately on demand guard/foreign request, MemAvailable<1.5GiB on either
node, new swap traffic, thermal fault, numerical/provenance error, lost worker,
rank mismatch, timeout, missing telemetry or pending-event overflow. Do not
kill an unrelated process. Coordinator uses its reviewed request cancellation
boundary if necessary; restoration is its responsibility. Do not leave HOLD
or experimental service active after a failed command.

Expected: request JSONL for each cap/mode/repetition, corpus+prompt hashes,
actual tokenize/usage counts, two rank telemetry files and prefill JSONL, source
verification per rank, complete launch/env/worker identities, fit residuals and
matched-prefix deltas, and returned baseline. No GB10 numbers are supplied by
this lane. An unexplained cap penalty keeps524288 HOLD. Attribution alone is
not promotion: reuse S18/K1 correctness, prefixes, eviction, concurrent and
regression gates; re-run those affected by any later candidate. No attention
or chunk change is authorized merely because this recipe exists.

Coordinator restores on sparkhead, using the current production ledger:

```bash
source ~/.config/ndm/paths.sh
set -euo pipefail
cd "$HOME/dsv41-deploy"
./stop.sh
./start.sh
bash offload/kvpack/stage_overlay.sh verify-running --expect \
  'VLLM_DSV41_KV_PAIR=3 VLLM_DSV41_DROP_MTP_BUFFER=1 VLLM_EXL3_TF_PROMPT_TRANSIENT=1 VLLM_EXL3_TF_PROMPT_ALIAS=1 VLLM_INDEXER_PREFILL_BUFFER_FACTOR=4' spark sparkworker
curl -fsS http://127.0.0.1:8000/v1/models
bash offload/launch/window.sh close
```

Require production327680, real readiness/healthy pair, request smoke, correct
unadapted/adapted label and demand guard released only after restoration. Retain
the current reviewed rollback tags/profile receipts; never replace production
files with this experiment config. Collect telemetry before its7200s deadline
or stop only the coordinator-owned telemetry session after collection. Delete
only this window's generated/scratch copies after hashing/retaining numeric
results; no broad `/tmp`, process or shared-worktree cleanup.
