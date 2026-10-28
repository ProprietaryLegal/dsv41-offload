# SO2 coordinator-only GB10 OFF/ON/OFF recipe — DO NOT RUN AS A LANE

No step here was executed on a Spark or x86host. Independent code review and a
reserved/drained coordinator window are prerequisites. Production remains live.
The qualified bound is DESIGN.md's runnable mixed-decode envelope; an unconditional
all-state bound is unsupported and remains HOLD. Do not silently broaden it.

## Preconditions — blocked; no current window

Coordinator decision (SO2-R2, S28 review): merge default OFF only. UP2 merges
first. SO2 remains roadmap P3 until a non-aborting background client exists;
there is no SO2 GB10 window or joint SO2 + UP2/0075 arm authorized now. D2 is
the only production priority-20 client and already aborts on foreign demand
within about 0.5 s (S28 report, existing client poll contract). The sustained
resident-background scenario below is hypothetical, not a production forecast.

Before any future window, all of these gates must be satisfied:

- **Composition (S27 O6 / S28):** SO2 and UP2/0075 both replace the scheduler.
  They do not compose through a clean Git merge or mount ordering. A single
  hand-composed scheduler and fresh independent check are required. Replay
  owners (`_dsv41_replay_end`) must retain UP2's exemption from deferral and
  SO2 pause/quantum; reserved replay floors must win after caps. The current
  builder deliberately refuses the UP2 base. Do not change its pins to bypass
  this refusal. The recipe below applies only to a compatible isolated base.
- **Slot/KV residency (S28 O1):** before introducing any non-aborting background
  client, test later foreground arrivals with all four active slots occupied
  and under KV pressure. Paused backgrounds retain slots and KV, which can
  delay admission. Record per-request slot-wait time (submission to admission),
  first-scheduled time, active-slot occupancy and KV residency; distinguish slot
  wait from KV refusal. Require a reviewed admission/residency policy before
  claiming sustained foreground benefit.
- **0074 policy cost (S28 O2):** first measure one lone cold 60000-token prefill
  with 1024 versus 4096 chunks, equal work/cache/sampling/config and paired
  repeats. Record actual chunk counts, per-step latency, TTFT and completion;
  isolate the fixed per-prefill-step cost before interpreting mixed workloads.
  The solo cap is a policy trade. Existing simulations assume zero fixed cost;
  their sensitivity values are ESTIMATED, not measurements. The <=3% all-work
  gate below remains binding.
- **Missing tools (S28 item 5):** a reviewed mixed-priority 0/10/20 driver and
  reviewed `so2_stats` plus per-request progress trace hook do not exist yet.
  Both must be built and checked before a runnable window; bench2 alone is
  insufficient. Include the slot-wait metric above. Missing traces fail the gate.
- **Restore identity (S28 item 5):** the recorded restore target is
  `K1-327680`, status `RECORDED_PRODUCTION` in
  `offload/operations/profiles.v1.json`, KV capacity 2502600 tokens, K1 manifest
  `offload/kvpack/MANIFEST.json` SHA256
  `559c5867ba42aa7daa7b4a7c8231d15ee1d35be40ae5b715b013f973193c58de`.
  Freeze the actual current launch/environment and hashes before the window;
  reconcile any drift with that record. Restore through the operations runbook
  and verify profile, mount/env hashes, capacity and health after every restore.
  These are recorded identities, not a new live verification by this lane.
- **Reachability before timing (S28 item 5):** a short ON qualification must
  show `mixed > 0` under the actual DSpark config (lookahead 1, draft slots 4,
  five speculative tokens and dynamic schedule). Also observe admission and
  maintenance in the directed cases. All-native counters fail qualification;
  do not spend a timing window measuring only the independent 0074 cap.

## Prepare and freeze (Controller CPU)

1. Record reviewed SO2 commit plus the current production profile, exact image,
   scheduler/Request/config/queue/async/utils hashes, legal-v2a manifest, remote
   DSpark/Engram identities, thinking/effort and launch arguments. Verify source
   pins against live files with coordinator-authorized lightweight reads. SO2
   certified the local src-image snapshot, not the current live container.
2. Run README CPU/real-KV/strict typing commands. Build using
   `python -m offload.bg_prefill.build_overlay --output <new-stage-directory>`.
   Verify both file hashes against MANIFEST.json. Stage these two files in the
   reviewed deployment checkout as `offload/bg_prefill/generated/scheduler.py`
   (take the generated vllm/v1/core/sched/scheduler.py) and
   `offload/bg_prefill/generated/so2_policy.py`. Never stage test shims/donor code.
3. Use the exact current reviewed production configuration as BASE; save rollback
   with its hashes. Do not assume the historical D-kv6-async-process-nap-lazy
   profile is still the current x86host pair/checkpoint configuration. Reject any
   BASE with an existing scheduler.py mount, including SO1/UP1/UP2/0075: composition must
   be rebuilt/reviewed, not resolved by mount order. Keep priority policy, seqs 4,
   budget 4096, threshold 1024, floor 5, cadence 24, 327680 context and current memory,
   offload/worker/depth/clock settings identical in every arm. No x86host unit change.

Create three inert config files on the coordinator's staged deployment checkout,
with this body (BASE is the saved exact baseline config path; choose FLAG 0/1/0
and LABEL SO2-OFF1/SO2-ON/SO2-OFF2):

```bash
source "${SO2_BASE_CONFIG:?saved exact baseline config path}"
# Assert no destination duplicates before appending; do not replace a scheduler mount.
SO2_PREFIX=/usr/local/lib/python3.12/dist-packages
OVERLAYS="$OVERLAYS offload/bg_prefill/generated/scheduler.py=$SO2_PREFIX/vllm/v1/core/sched/scheduler.py offload/bg_prefill/generated/so2_policy.py=$SO2_PREFIX/so2_policy.py"
EXP_DOCKER_ENV="$EXP_DOCKER_ENV VLLM_SO2_BG_PREFILL=${SO2_ENABLE:?0 or 1} VLLM_SO2_PERIOD=24 VLLM_SO2_QUANTUM=32"
LABEL=${SO2_ARM_LABEL:?SO2-OFF1 or SO2-ON or SO2-OFF2}
```

Pass SO2_BASE_CONFIG, SO2_ENABLE and SO2_ARM_LABEL explicitly for each arm.
Inspect expanded env/mount uniqueness and `bash -n` each. Every container mount is
read-only via the existing start_exp.sh; OFF arms use the SAME mounted code with
flag 0. Separately compare an unmounted baseline against mounted-OFF before timing
ON, to prove inertness. No scheduler cadence 100000 or local depth-policy mount.

## Coordinator window sequence (not executed)

- Use the existing window/drain/HOLD/restore procedure from COMMON.md. Capture
  running/waiting 0 before changing the experiment. Preserve the production alias,
  existing reserved production GPUs and their services. Only the coordinator runs
  `bash offload/launch/start_exp.sh <SO2-OFF1.env>`, then ON, then OFF2, with drain,
  restore checks and identical warmup at each transition. Do not change units.
  Existing `./start.sh` plus the saved exact production environment is the rollback;
  confirm the actual current production procedure before opening the window.
- Verify both mounted code hashes and scheduler flags in each arm. Capture final
  `so2_stats` from the scheduler process with a reviewed trace hook; do not add an
  unreviewed live monkeypatch. Missing per-request progress/counter traces is a
  failed measurement gate. ON should show mixed, admission and maintenance calls;
  OFF should have no SO2 state. Native/fallback calls are not mixed-mode service.
- Warm once, then at least three paired repetitions per cell. Preallocate IDs and
  record submission/first-scheduled/completion/cancel timestamps and all work.
  Use one foreground 0, then a separate foreground 10 series; backgrounds 20. Use
  one/two cold 60000-token background prompts and one foreground decode stream
  with 600 delivered target tokens. Keep identical representative local legal
  task/prompt identities, output caps, seed, sampling, thinking/effort and cache
  salts within paired arms. Verify actual `/tokenize` and server usage counts.
  Keep raw prompts/responses and token traces outside git and reports.
- Test both orders: foreground is already decoding when backgrounds arrive;
  foreground arrives during solo background prefill (and while two prefills run).
  The CPU scenario starts with all streams already admitted; include the actual
  admission transient separately. Background admission must be observed, not
  inferred from submitted HTTP requests. Complete all required background work
  after foreground ends; do not count cancellation as throughput.
- Existing bench2 uses ONE `--priority` for its entire group. Its `--mixed` is two
  decoders plus a prefill arrival, not this split-priority scenario. Do not claim
  bench2 alone reproduces this test. Coordinator needs its existing per-request
  client or a separately reviewed mixed-priority driver with explicit 0/10/20.
  Ordinary bench2 short/c32/c150 and mixed controls can still detect regressions.
- D2 integration is a DISTINCT short test with its unmodified priority 20,
  max-three-stream driver. Arrive foreground 0/10; verify first positive foreign
  poll aborts auxiliary streams, plus grace/chained-completion behavior. Record
  client cancellation separately from scheduler skips/service. Sustained overlap
  uses separate background clients, not a disabled D2 foreign-demand abort.

## Correctness, liveness and decision gates

Before performance credit, stable greedy OFF/OFF controls must agree, then
OFF/ON tokens/counts/finish reasons must match under stable deterministic settings.
Preserve nonzero-temperature sampling settings; this lane supplies no GPU
numerical/distribution proof. No lost requests, bad logits, TP disagreement,
worker fallback/watchdog, service errors or altered target/draft identity.

Exercise admitted runner vs infeasible fresh waiter (S15), ready and completed
restore KV holders vs infeasible winner (S19), cancellation while paused, native
capacity escape and numeric-priority KV preemption. All must progress under their
native resource conditions. Separately trace continuous runnable foreground
with one/three resident backgrounds through at least 10 full service intervals.
In mixed mode each runnable background advances every <=24 scheduler calls;
record actual per-request gaps and native allocation refusals/preemptions, not
maintenance-attempt counts alone. Keep foreground legal priorities unchanged.
Holding/dependency fallback is native behavior, not proof of an N-step guarantee.

Report foreground delivered tok/s, accepted+bonus per round, round and ITL
p50/p95/p99/max, first generated/first visible token, full foreground task time,
background progress rate/service gap and completion latency, ALL-work completion,
CPU overhead and both-rank headroom. Separate pure-prefill cost from decode stalls
and actual async in-flight chunk depth. No throughput credit from withheld work.

Stop and roll back immediately on a stalled runnable request, unexpected preemption
victim, lost/corrupt outputs, rank drift, new watchdog/fallback, health failure,
MemAvailable <5 GiB on either rank or swap growth. Keep at least matched baseline
safety margins. Do not reset the pair, move GPU assignments or change power/clocks.

Consider isolated adoption only with repeatable >=5% foreground task/rate gain
beyond OFF1/OFF2 noise, positive measured background minimum share, and no more
than 3% ALL-work slowdown. A background completion SLA still needs an explicit
coordinator decision; 32 tokens/24 calls can be very slow under endless foreground.
Keep OFF if workload falls outside the supported envelope, if the requested
unconditional guarantee is required, or if tails/aggregate cost fail. No combined
SO1/UP1/RP1 promotion is authorized by this recipe.
