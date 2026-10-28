# Coordinator window runners

These scripts are prepared on Controller, CPU only. The default and `--dry-run` print the full
command plan without executing commands, reading a Spark, starting processes or creating results.
Only Deepseek / coordinator runs `--execute`, after independent review and staging the merged
checkout. No runner opens/closes the Spark window, restores production or promotes a candidate.

```bash
source ~/.config/ndm/paths.sh
bash offload/launch/windows/native_pair.sh --dry-run r000008_b0,r000009_b0
bash offload/launch/windows/native.sh --dry-run offload/launch/configs/D-kv6-async-process-nap-lazy.env 1604926 1604926 /private/measured-native-timing.json
```

KV counts in this example are expectations to verify against the coordinator's selected config,
not measurements by WR1. Use the production-equivalent K2 configuration and exact KV figure for
that window (packing must match between arms). Native uses the calibrated staged config
`offload/launch/configs/WR1-E4-native-kx2-async-calibrated.env`. The runner never edits tracked configs.

For execution set `WINDOW_COORDINATOR=Deepseek/coordinator`, `WINDOW_HOOK=/absolute/executable`,
and optionally `WINDOW_OUT` to a **new** directory inside the campaign. Optional `WINDOW_HEAD`,
`WINDOW_PEER`, `WINDOW_X86HOST`, `WINDOW_REMOTE_REPO` (relative to remote home) and `WINDOW_URL`
select reviewed deployments. The experiment launcher currently owns port 8001 and container
`dsv41-serve`; changing the benchmark URL does not change its health check port.

`common.sh` is shared. Every start requires idle evidence, `start_exp.sh` success, exact KV
startup lines, health, and both-rank runtime/identity evidence. Memory supervision starts before
every active step, including startup: both Sparks must retain at least **5 GiB MemAvailable**;
swap growth from the run baseline, missing/stale telemetry, command failure, timeout or an
external `STOP` file aborts. Both allocated 3090 UUIDs have **1 s** memory/power/temperature
traces. These are plan settings, not WR1 hardware measurements. Processes use new sessions;
cleanup signals only processes started by this supervisor. Remote services remain coordinator
owned; a local abort does not certify that an SSH-started engine stopped. The printed next
steps require coordinator inspection/restoration.

Each benchmark block measures short/32K/150K, B1/B3/B4 and mixed arrival with real local corpus,
fixed output budget, high reasoning, priority 10 and seed 42. Outer C/x/C repeats provide at
least three interleaved repetitions. Warm mode has a priming repetition before its measured
warm repeat. JSON summaries preserve cell, accepted tokens/round, delivered-token rate,
client-event step-gap p50, TTFT and nullable first-visible time. Client gaps are an observation,
not GPU-event verify-only timings. Full-task quality and numerical/distribution gates still
need the owning plan's review; no automatic performance pass is claimed.

## Coordinator adapter and evidence contract

Some required GB10 operations have no generic executable probe in the repository: selecting
and owning transient native workers, independent live rank-payload capture, stale mmap injection,
aligned corruption, exact MoE microchecks and full-logit/fault qualification. The runners require
an explicit **coordinator adapter** for these operations. This is a required execution input,
not a supplied mock or an assumed pass. Dry runs show every invocation; do not use a placeholder
adapter for execution. The adapter can dispatch existing reviewed campaign probes.

Invocation: `WINDOW_HOOK GATE ARM RUN_DIR RECEIPT [ARGUMENT ...]`. It must exit nonzero on
failure, write diagnostic evidence under RUN_DIR, and write the receipt **after** the checks.
`checks.py REQUIRED` is the exact per-gate checklist. `preflight` verifies the exclusive HOLD,
production drain, saved rollback revision/env/worker, D2 pause, source/image pins and process
ownership. `runtime` checks both ranks' mounted bytes, env, model, HELLO/worker identity, KV
layout, process proxy/mlock/no torch, no local draft tensors, required Engram and errors.
`clean` captures before/after counters and BOTH rank logs (including all runtime since startup)
and rejects fallback/watchdog/replay provenance stops. It must also check health. Run
`checks.py counters BEFORE AFTER` and `checks.py logs LOG` inside that adapter; absent sparse
counters can mean zero, absent counter maps cannot. Hooks receive the arm/config/URL in their
argument list. Gates describing numeric thresholds must implement the owning GB10 plan's
exact thresholds, not turn a boolean true merely on successful request completion.

Receipt format (all numbers shown here are schema examples, not measurements):

```json
{"schema":"wr1-gate-v1","gate":"idle","arm":"native-r1-C1",
 "run_dir":"/opt/dsv41/campaigns/DSV41-OFFLOAD/WR1-example",
 "checks":{"experiment_idle":true,"no_foreign_requests":true},
 "evidence":[{"path":"/opt/dsv41/campaigns/DSV41-OFFLOAD/WR1-example/idle-metrics.txt","sha256":"FULL_SHA256"}]}
```

Every named check must be boolean true. Receipt identity/run directory and SHA256 of fresh,
run-local evidence are checked. This proves evidence integrity and explicit acknowledgement;
the receipt is not an independent implementation of its GPU/fault test. Preserve raw prompts,
responses and private captures only outside git. `runtime` and `clean` must include the
config/worker and both-rank artifacts, not a blanket approval text.

S20 M3 timing input is a JSON object with `classification: "MEASURED"`,
`arm: "native-hybrid-kx2"`, finite nonnegative `draft_ms`, six positive nondecreasing
`round_ms`, `overlap_audited: true`, `verify_only: true`, and nonempty
`sources: [{"path": "...", "sha256": "..."}]`. Use pair-derived effective nonoverlapped
remote cost and GB10 verify-only costs, not isolated worker time. The `recalibrate` hook
validates sources/build identity/fit-heldout scope and stages the new config on both ranks.
Use N3 `depth_config` to derive costs from measured depth rows; wrap its `fit_effective_remote_ms`
and `fit_verify_only_ms` in this evidence envelope. No inherited draft timing is accepted as
recalibration. Pair qualification uses `prod_window.sh pair COORD`, then N4's own bounded
lock acquisition; do not wrap N4 in the pre-promotion `pair_window.sh run` hold.

## CPU verification

```bash
source ~/.config/ndm/paths.sh
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 "$HARNESS_PYTHON" -m unittest discover -s offload/launch/windows/tests -v
"$HARNESS_PYTHON" -m mypy --strict offload/launch/windows/checks.py
shellcheck -x offload/launch/windows/*.sh
```

Recorded benchmark/KV/stats sources under `offload/results` exercise real schemas without
copying any prompt/response into new fixtures. Malformed and missing evidence fails closed.

UP1: `bash offload/launch/windows/up1.sh --dry-run EXPECTED_PACKED_KV`.
It runs K/C/K three times before C/x/C for split, split+pin, pin-only, replay, K5,
dual-ring, then 0067 copy/SWA/both separately. No combined `all` arm runs. Every
UP1 start pipes the locally emitted pinned runtime hash checker into **each** rank.
Split gates require bitwise equality at actual rows 33/42/43/63 on each rank,
per-row oracle for 43/63 and unchanged dispatch controls 32/64. Replay requires
before/after stats, both stop strings absent, successor/remainder/fault/prefix and
final KV/logit gates. Dual-ring checks both IPv4 RoCE v2 GID index 3 preconditions,
then exact beginning-of-line rank prefixes and no NCCL-only fallback; collective
bytes and both-link traffic still require the adapter's measured microcheck.
The adapter implements the detailed acceptance/value gates in the UP1 plan before
issuing any passing receipt; summary rows alone do not establish adoption.

RP1: `bash offload/launch/windows/rp1.sh --dry-run P0_KV P1_P2_KV MANIFEST_SHA256 VERIFIED_IO_CPUS`.
No P3/P4 or larger CPU tier runs. P0 starts with no connector; P1 uses both stores
for independent valid-group SHA/CRC before reload; P2 uses a single writer and a
256 MiB landing tier with configured cap 29,568. Those are plan settings, not
measurements. Each enabled start passes the exact checkpoint digest, audited CPU
list, .2 load fraction and verifies both running overlays plus K1. Three fresh
saved-packet restart trials use 24K/75K/150K with 300 s command bounds and compare
against fresh compute. `rp1-restart` is a pre-start operation: wait for stores,
record old engine, stop only the owned experiment, prove old processes/maps gone,
remove only unused engine-specific shm and preserve disk. After the runner's next
start, `rp1-empty` proves a new empty engine and no intervening packet salts.
Stale/CRC/sidecar/namespace/concurrency gates must implement the exact owning plan
procedures and their own bounded safe stop/restart sequence. Their final receipt
must leave true identity restored. The runner then starts ordinary P2 again and
runs needles, prefix checks and three decode C/x/C cycles.

Depth: `bash offload/launch/windows/depth.sh --dry-run EXPECTED_KV CONTROL_CONFIG SP2_CANDIDATE_CONFIG [SP2_CANDIDATE_CONFIG ...]`.
It takes SP2's actual config paths; the depth gate checks source/target/worker,
trained maximum five, cost/overlap evidence, held-out groups and unchanged
sampling/output budget. Each candidate receives three C/x/C repetitions with
greedy control/control and candidate/control identity checks. Stage selected SP2
configs on the coordinator's reviewed remote checkout first. Different KV layouts
or target/worker changes require a separate experiment, not this comparison.

The preflight adapter must also save `RUN_DIR/coordinator-next.sh`: an exact,
reviewable bash restoration script using the saved production checkout, revision,
environment and worker state. Include the worker close/check with
`PRODUCTION HOLD OK`, production start/verification and final Spark window close.
The runner syntax-checks this script and prints it on exit; it **never executes**
it. This makes the coordinator's next actions concrete even if a later step fails.

## UP2 low-cap

`bash offload/launch/windows/up2_low.sh --dry-run` is network free. This runner
uses direct W8-style operations rather than WR1's unsupplied WINDOW_HOOK.
See [UP2 handover](UP2-LOW.md), [identity table](UP2-IDENTITY.md),
[CPU failure rehearsal](UP2-REHEARSAL.md) and [results template](UP2-RESULTS-TEMPLATE.md).
Native q/scheduler/fault apparatus is not supplied: missing evidence retains
NOT RUN/HOLD and stops before measured cells. No native result or speedup is claimed.
