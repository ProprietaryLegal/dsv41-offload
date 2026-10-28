# UP2: 0075 isolated GB10 qualification recipe — NOT EXECUTED

Only coordinator Deepseek / coordinator executes remote/API/start commands, in
an exclusive reserved window after independent review. This lane used Controller
CPU only. Production continues unchanged; all delivered code defaults OFF.
The 524,288 profile remains HOLD. Never restart or modify x86host's production
pair/units for this recipe. If its current worker identity/position support is
insufficient, return that dependency to its owner and stop the relevant arm.

UPSTREAM CLAIM: HeNryous afe4d5e reports +4.9/+5.9% at 478K and +6.5% at 949K;
first-token KL .0017 versus .0015 fresh/fresh. These are another deployment's
results. Our first points are 150K/300K, where smaller or no benefit is expected;
no positive local result is assumed. 949K and 1M are outside this recipe.

## Preconditions and immutable preparation

Use the reviewed integration checkout containing UP2 plus the ACTUAL current
K1/process proxy. Resolve its source/worker identities against the coordinator's
current production ledger. Historical UP1 configs name an earlier worker; a
mismatch is a STOP, never permission to overwrite that worker or silently change
identity. Reconcile reviewed config/source pins before the window. All controls
and replay arms must have the SAME target weights, quantization, kernel versions,
worker/checkpoint/adapters, speculation/depth, clocks, TP layout, proxy policy,
Engram backend, priorities and sampling. Preserve reasoning high/effort75 and
all current legal request behavior. No SO2 scheduler composition: UP2 and SO2 need a hand-composed scheduler and fresh independent check.
The startup replay guard rejects image/SO2/missing/duplicate scheduler mounts and hash drift before remote actions.
No separate split, fused-SWA, CPU affinity,
dual-ring, SO1, RP1, time-budget chunking or workspace experiment in this matrix.

On Controller CPU, from that reviewed checkout:

```bash
source ~/.config/ndm/paths.sh
set -euo pipefail
UP2_RUN_DIR=$(mktemp -d /opt/dsv41/campaigns/DSV41-OFFLOAD/UP2-window.XXXXXX)
printf '%s\n' "$UP2_RUN_DIR" > /opt/dsv41/campaigns/DSV41-OFFLOAD/UP2-window-current.txt
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 TRITON_INTERPRET=1 \
  "$HARNESS_PYTHON" -m pytest -q -p no:cacheprovider offload/upstream_adopt/tests \
  --basetemp="$UP2_RUN_DIR/pytest" \
  offload/prefill_trace/test_bounded_chunk.py offload/prefill_trace/test_window.py
# Isolated child interpreters can create bytecode despite the environment flag.
# Remove ONLY generated caches inside this reviewed output tree before hash checks.
"$HARNESS_PYTHON" - <<'PY'
from pathlib import Path
root = Path('offload/upstream_adopt/overlay')
for path in root.rglob('*.pyc'):
    path.unlink()
for path in root.rglob('__pycache__'):
    path.rmdir()
PY
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' "$HARNESS_PYTHON" -m offload.upstream_adopt.build_overlay
PYTHONDONTWRITEBYTECODE=1 "$HARNESS_PYTHON" -m offload.upstream_adopt.make_configs --check
"$HARNESS_PYTHON" offload/upstream_adopt/verify_runtime.py --emit > "$UP2_RUN_DIR/verify-runtime.py"
sha256sum offload/upstream_adopt/{MANIFEST.json,inputs.lock.json} \
  offload/launch/configs/UP2-0075.env > "$UP2_RUN_DIR/source-identity.sha256"
# Dry plan: no network, tokenize, model load or service change.
"$HARNESS_PYTHON" -m offload.prefill_trace.window_requests --cap 327680 \
  --contexts 150000 300000 --modes cold --reps 3 --trace-run plan \
  --cache-salt-prefix up2-low-pair1 --output "$UP2_RUN_DIR/plan.jsonl"
```

Stage only reviewed code/manifests/config through the existing coordinator
procedure. Rebuild on integration-tip input drift; do not force a pin or mount
a stale package. Both rank image-base hashes must match. The 42 delivered files
have one mount per destination. All 13 K1 files remain byte-identical. Preserve
the unchanged Engram JIT and complete async package; no new CUDA/JIT extension.

Before opening, save the ACTUAL production checkout/revision, applied K1 profile,
launch env/argv, overlay hashes, readiness and worker identity to the window's
rollback manifest. `prod-async`/3c1dc6b in the older UP1 plan is historical: do not
restore it over the current K1 production profile. Set UP2_PROD_CHECKOUT to the
verified current production checkout with its exact saved configuration. Use the named
W8/restore.sh production gates at opening and after restore: `K1 START VERIFIED`,
`/v1/models` max_model_len327680, KV capacity2,502,600 tokens, greedy byte check
against `K1-production/greedy-post.jsonl`, and x86host `PRODUCTION HOLD OK`.
A missing/failed gate is STOP; never substitute an older rollback profile.

## Fixed policy, diagnostics and ordered starts

Fixed policy: MAX_BATCH=4096, long-prefill threshold=1024, four active slots,
6 GiB KV/rank and factor4, unchanged clocks and speculation/depth. The installed
policy gives a solo prefill the whole budget and caps under contention. Record
ACTUAL q per chunk; UP2's reserved 129-row safety floor overrides small leftover
budgets only when needed. Never call a changed effective chunk distribution an
isolated replay-layer kernel gain. All source-KV writes remain required.

Open/close the reserved window with the existing coordinator runbook/demand
guard; verify zero production running/waiting before each traffic/start phase.
After opening, on sparkhead in its reviewed experiment checkout, one start at a
time (these are instructions, not commands executed by UP2):

```bash
source ~/.config/ndm/paths.sh
set -euo pipefail
cd "$HOME/dsv41-deploy"
bash offload/launch/window.sh open 'UP2 isolated 0075; restore current K1 production'
# Diagnostic start first. Its measurements are EXCLUDED from headline timing.
UP2_CAP=327680 UP2_REPLAY=0 UP2_AUDIT=1 \
  bash offload/launch/start_exp.sh offload/launch/configs/UP2-0075.env
# After OFF diagnostic gates, repeat the SAME probes with UP2_REPLAY=1, AUDIT=1.
# Only after all diagnostic gates pass, start and MEASURE each arm separately:
# UP2_CAP=327680 UP2_REPLAY=0 UP2_AUDIT=0 bash offload/launch/start_exp.sh offload/launch/configs/UP2-0075.env
# UP2_CAP=327680 UP2_REPLAY=1 UP2_AUDIT=0 bash offload/launch/start_exp.sh offload/launch/configs/UP2-0075.env
# UP2_CAP=327680 UP2_REPLAY=0 UP2_AUDIT=0 bash offload/launch/start_exp.sh offload/launch/configs/UP2-0075.env
```

After EACH start: readiness, actual `/v1/models` cap, identical mounted hashes on
both ranks using verify-runtime.py, exact launch argv/env (including both audit
flags), process-proxy health, required remote Engram, absence of local drafter,
no proxy_torch_loaded and actual KV capacity. Never infer readiness from a sleep.
Verify all new/other experimental flags are zero in OFF. Both ranks MUST have
the same AUDIT setting: the diagnostic adds a TP CPU-group collective.

Run normal-policy diagnostics symmetrically; no one-arm eager/graph override.
If diagnostic synchronization is incompatible with a captured path, stop and
review an explicitly matched diagnostic execution mode. Keep headline policy
unchanged. Warm identical shapes per start with a bounded 32K request, verifying
actual generated tokens/worker rounds/finish reasons; a one-token EOS does not
prove warmup. Discard warmup timing. Retain clocks/thermal state on both ranks.

Diagnostic workload and required evidence:

| Probe | Required result |
|---|---|
| Short/32K + final tails 1/127/128/129 | q/kept/first_pos/valid/anchor and engine owner epoch agree across ranks; all final NEED_DRAFT windows real and complete |
| Exact cold sizes 147313/147439/147440/147441 | CPU MEASURED solo q=4092 with DSpark K=5 (4 drafting slots): 36×4092 + {1,127,128,129}. Derive sizes from first diagnostic measured q and verify final tails; GB10 NOT MEASURED |
| Near-cap decode (cap327680, prompt327675, output budget5), K4/K5, sync/async where supported | Drive generation to the length boundary with drafts; ON/OFF clamp to available rows identically, no reservation exception, normal length finish and healthy engine; verify actual occupied lengths and scheduled drafts |
| Remaining 129/256/257; repeated leftover64/128 with decode | Next ready admitted successor gets >128, or documented real prefix suffix replay; peer decode continues; no indefinite zero |
| Partial/full hits + async restore | Rewind/recompute actual suffix; no missing rows promoted from cached-token counts; ready peers run during restore |
| Cancel after skip, slot reuse, KV preempt/backtrack below ring lo | New owner/generation, real suffix rebuild; no old-owner data or fabricated widened range |
| Disconnect before final, after final; watchdog, late reply | Real final suffix or real retained ring restores worker; old round cannot publish into new staging; no stale-generation usable draft |
| Mixed rejected decode; MM exclusion; two prompt-logprobs requests | Rejected rows excluded, baseline exclusions preserved, all source-KV writes retained |

Use the existing scheduler/fault injection qualification apparatus in the reserved
experiment; do not kill a production worker or treat random traffic as a fault
probe. CPU regressions are ready-made schedules, not proof these native cases
have passed. If a native deterministic fault/scheduler probe is unavailable,
mark that gate NOT RUN and retain HOLD, rather than claim it passed from text.
The optional audit records counts/error strings, never private tensors/text.
It compares engine owner epochs; actual STEP generation echoes and uniform
broadcast fallback/drafts remain separate required checks. Any PARTIAL_WINDOW
under 0075 is a stop even if target verification eventually completes.

## Primary off/on/off matrix (timing audit OFF)

1. Under production cap327680, run 150000 and 300000 first, three matched cold
repetitions per point per arm, sequential OFF-A / ON / OFF-B. No high point at
this cap. Compare OFF-A/OFF-B to establish drift/noise before crediting ON.
2. Only after correctness/state/resource gates pass, and with coordinator's
524288 window/profile authorization, repeat off/on/off at400000,450000,500000.
Use a SEPARATE bounded window per high point if needed. Include150/300K matched
524288 controls to identify cap dependence separately. Do not mix cap changes
into a replay ratio. The profile remains HOLD even if this arm completes.
3. Warm/full/controlled100K-suffix cases are separate cells after cold; one paired
repeat each is sufficient to detect state errors, not to establish a speed gain.
4. Run a short existing representative completed task symmetrically with its
original reasoning/output budget. A1500-token all-reasoning result is a timing
probe with zero visible answer, not evidence of completed task quality.

1500 output tokens are reserved at every measured cold point. Literal500000
occupied prompt +1500 output =501500≤524288; actual tokenize and usage must match.
No approximate500K or reasoning truncation to make a request fit. Existing public
CourtListener + RECAP corpus/order/seed42/header is shared across sizes/arms;
missing/short corpus fails. Token counts include the complete chat template.

Coordinator on Controller after it owns the reserved experiment tunnel
127.0.0.1:18001→experiment:8001. For EACH already-started arm:

```bash
source ~/.config/ndm/paths.sh
set -euo pipefail
UP2_RUN_DIR=$(cat /opt/dsv41/campaigns/DSV41-OFFLOAD/UP2-window-current.txt)
UP2_ARM=off-a
UP2_CAP=327680
UP2_POINTS=(150000 300000)
# Create ONCE per bounded window; never renew to overrun the reservation.
(set -o noclobber; date -u -d '+75 minutes' +%s > "$UP2_RUN_DIR/submission-deadline.txt")
"$HARNESS_PYTHON" -m offload.prefill_trace.window_requests --execute \
  --url http://127.0.0.1:18001 --cap "$UP2_CAP" --contexts "${UP2_POINTS[@]}" \
  --modes cold --reps 3 --trace-run "$UP2_ARM" --cache-salt-prefix up2-low-pair1 \
  --deadline-utc "$(cat "$UP2_RUN_DIR/submission-deadline.txt")" \
  --output "$UP2_RUN_DIR/$UP2_ARM-cold.jsonl"
```

For ON/OFF-B use new output/trace-run names with EXACTLY the same salt prefix.
Execute the noclobber deadline line only once, before OFF-A. Each start clears
the experiment cache; driver asserts cold cached_tokens=0 on every request.
The fixed cache salts are cache keys, not prompt content. Require identical
prompt_sha256 and cache_salt_sha256 per point/rep across the three arms. Fresh
window uses a new prefix. For a high-point window set cap524288, one literal
high point, and a new fixed prefix shared across its three starts. Never reuse
an existing output file. Record partial/full cases with separate output files,
`--modes full partial` and a separate shared salt prefix; primes are not timed.

**ESTIMATED** from historical PF1 timing, not new measurements: low cold matrix
roughly45–70 minutes including starts; each high-point off/on/off roughly45–75
minutes. Diagnostics/mixed/numerical tests need their own window. Use a75-minute
submission deadline,900-second per-request wall bound and90-minute reservation
hard bound; partition when needed. Stop new submissions at deadline, drain or
explicitly cancel only window-owned requests through the coordinator's existing
procedure, restore production. Missing cells remain missing; no deadline renewal.

## Numerical, needles and concurrent decoder stalls

Measure A/A on the SAME build, prefix, batch shape and execution mode before ON.
Compare final prefix/taps/KV-source writes and first-token target distributions;
intermediate discarded one-row activations are intentionally unconsumed. Use the
UP1/PF1 numeric gate: top20+OTHER coarse KL≤max(.005,2×A/A), common captured mass
≥.99, stable short greedy outputs/finish reasons. API coarse KL is a lower bound;
ambiguous/out-of-gate cases require bounded full-logit/teacher-forced diagnosis
before a decision. Do not call stochastic text identity a distribution proof.
Full tensors/logits remain owner-controlled scratch outside git and reports.

Coordinator examples for numerical floor and needles at the low cap (separate
from headline timing, each followed by metrics/health checks):

```bash
source ~/.config/ndm/paths.sh
"$HARNESS_PYTHON" offload/kvpack/prefix_identity.py --url http://127.0.0.1:18001 \
  --n 32000 150000 300000 --block 384 --out "$UP2_RUN_DIR/$UP2_ARM-prefix-local.json"
"$HARNESS_PYTHON" offload/kvpack/needle.py --url http://127.0.0.1:18001 \
  --ctx 150000 300000 --needles 6 --concurrent 1 --reps 1 \
  --out "$UP2_RUN_DIR/$UP2_ARM-needle-local.json"
"$HARNESS_PYTHON" offload/bench/bench2.py --url http://127.0.0.1:18001 \
  --label "UP2-$UP2_ARM-mixed" --ctx 150000 300000 --workload legal --filler real \
  --data-dir /opt/dsv41/data/bench-data --concurrency 1 4 --mixed \
  --mixed-short-ctx 8000 --max-tokens 1500 --reps 3 --thinking \
  --reasoning-effort high --priority 10 --temperature 1 --top-p 1 --seed 42 \
  --cache-mode cold --out "$UP2_RUN_DIR/$UP2_ARM-mixed-local.jsonl"
```

Needles require6/6 and zero foreign codes in every arm. Add4-way low-cap needle
isolation if admission/headroom supports it. At524288, use each authorized high
point B1 first; no four-way500K assumption. These legacy needle/numerical helpers
report actual lengths that can be approximate: they are retrieval/numerical
probes, NEVER the literal-size throughput evidence. Their local outputs can
contain generated text; export ONLY aggregate scores/numeric comparisons, never
copy their raw files into git or reports. No proprietary weights/prompts/taps.

The mixed driver retains per-stream step_times_ms. Compute consecutive delivery
event gaps per already-decoding stream; sort and use nearest-rank percentiles
ceil(.95*n) and ceil(.99*n), plus max gap, event count, delivered tokens and wall
span. Report these as DELIVERY-EVENT stalls: speculative SSE bursts are not one
original target token each. Server correlated round/accepted-token telemetry is
required to report engine/token stalls. Do not relabel event p99 as token p99.
Compare decoder-only A/A first, then identical arrivals with long-prefill at
150/300K and each approved high point. Report per-stream p95/p99, never just an
aggregate throughput. No new total concurrency or priority policy is introduced.

Collect both-rank source identities, q/kept distributions, prefix/uncached counts,
host/CUDA prefill timings if already safely available, TTFT first generated,
first visible answer, reasoning/answer counts, total task time, remote phase
waits, accepted drafts/round, actual fallback counters, MemAvailable/swap,
clocks/throttling/temp/power and native kernel dispatch. Use PF1 telemetry's
existing bounded coordinator procedure, outside headline profiling. No PF1
source overlay may cover UP2's model_state/runner without reviewed composition.
TTFT is an end-to-end boundary, not a pure GPU prefill timer.

## Stop rules, decision and expected artifacts

Stop immediately for missing final source-KV, invalid/zero/fake taps, rank or
owner-epoch mismatch, stale-generation usable reply, incorrect anchor, NaN/inf,
hang/nonprogress, health failure, new swap, either Spark MemAvailable<5GiB,
unexplained numerical drift or unexpected kernel/fallback growth. Normal arms
require zero new fallback_internal_error, replay_suffix_incomplete, provenance
errors and PARTIAL_WINDOW. Fault probes compare ordinary failures with matched
OFF probes, but any stale read, rank disagreement or invented suffix is a stop.
No gain is credited to target-only fallback or missing/abandoned requests.

Repeatable concurrent decoder throughput loss>1% or p95/p99 delivery-stall growth
>max(5%,measured A/A noise) stops adoption; investigate thermal/start drift rather
than selecting a convenient repetition. Needles, completed-task behavior and all
numerical gates must pass. Preserve all admitted requests and per-stream progress.

For useful benefit retain UP1's threshold: ≥5% repeatable cold-prefill/TTFT gain,
or≥3% with verified replay work reduction and a completed-task win; all safety,
decode and quality gates also pass. Below-noise150/300K results are an honest
null result and may still justify the already-authorized longer screening after
correctness passes. Do not sum upstream percentages or claim a cap-level effect
as a replay effect. No deployment/promotion occurs from this recipe alone.

Expected window outputs (all initially NOT RUN):

| Artifact | Required contents |
|---|---|
| identity / rollback manifest | Current production identity and exact restore;42 mounted hashes/rank;flags/cap;unchanged worker identity |
| diagnostic receipt | q/kept/position/valid/anchor/owner agreement counts;STEP-generation echoes;full final suffix;source-KV retention;fault/recovery outcomes |
| cold JSONL per OFF-A/ON/OFF-B | Exact prompt/uncached counts,matching hashes/salts,sampling,TTFT/channel counts,total/finish;no text |
| numerical / needles summary | A/A and ON/OFF floor,probability-mass scope,6/6 recall,no foreign codes,missing tests explicit |
| decoder summary | Per-stream event p95/p99/max/count,throughput,engine-stall scope,arrivals,accepted-token/round counts |
| telemetry / decision | Two-rank memory/swap/clocks/temps/throttle,paired effect/noise,complete/failed/canceled accounting,HOLD or candidate decision |

After EACH window drain the experiment and restore from the saved CURRENT
production checkout/config, with replay/audit OFF; do not start from UP2 configs
or restore the historical pre-K1 deployment. Coordinator only:

```bash
source ~/.config/ndm/paths.sh
set -euo pipefail
: "${UP2_PROD_CHECKOUT:?set to the verified current production rollback checkout}"
cd "$UP2_PROD_CHECKOUT"
./stop.sh
./start.sh
# Require W8/restore.sh: K1 START VERIFIED; /v1/models 327680; KV 2,502,600;
# greedy byte check vs K1-production/greedy-post.jsonl; x86host PRODUCTION HOLD OK.
# Verify both-rank saved production hashes/cap327680, unchanged worker,
# queue health, short reasoning/tool/structured smoke, then close window.
# Follow the existing coordinator runbook for HOLD and window close.
```

No x86host restart/unit edit, private model copy, publication, push, merge or email
is authorized by these instructions. Rollback evidence is part of the decision;
production must be healthy before relinquishing the reserved window.
