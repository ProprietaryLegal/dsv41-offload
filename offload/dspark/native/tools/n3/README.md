Status 2026-10-23: N3 integration retained; original failed-pilot conclusions are superseded by N4 d1dfe13 and S20 MERGE. Native physical pair and GB10 A/B remain pending; post-TP2 ownership uses prod_window.sh. See [live roadmap](../../../../../documents/plans/roadmap.md) and [measured record](../../../../results/COMPARISON.md); campaign review decisions are in `LANES.md` / lane reports outside the repo.

# N3 native TP2 serving and coordinator handoff

The TP2 server selects one arm with `DSPARK_WORKER_ARM`: `k2-tp2` (default),
`native-packed`, `native-expanded`, `native-hybrid-kx`, `native-hybrid-kx2`.
All four native serving arms retain the target-shared **EXL3 K6 draft head**.
Unknown arms, missing/mismatched official shards, prepack failures and identity
mismatches refuse startup/HELLO. There is no K2 fallback. Native arms require
`--tp 2`. The four active slots, protocol, 1M RoPE cap, negotiated session clamp,
prewarm, status, RX poison refusal and backend failure handling remain in force.

No script here changes a service, Spark, power/clock setting or weight file.
Only the coordinator executes physical pair commands. Proprietary dumps,
fixtures and detailed logs stay under the lane's external results directory.

## Source and identity

From the final merged checkout on Controller, recompute each identity, including
its runtime source/kernel hashes (the coordinator's R3/S13 merge changes them):

```bash
source ~/.config/ndm/paths.sh
$HARNESS_PYTHON offload/dspark/spark/tools/identity.py \
  /opt/dsv41/campaigns/DSV41-OFFLOAD/N3-DSPARK_COPY_DONE.json \
  --arm native-hybrid-kx2 --json-out /opt/dsv41/campaigns/DSV41-OFFLOAD/N3-final-kx2.identity.json
```

Repeat for packed, expanded and hybrid-kx. The tool prints the exact
`VLLM_DSPARK_REMOTE_IDENTITY` and `VLLM_DSPARK_REMOTE_MARKOV_IDENTITY` assignments.
Update the matching E2/E3/E4 config's identity before starting it. Do not reuse
N3's pre-merge identity after editing/merging runtime sources. Copy exactly that
checkout's `offload/{dspark,transport}` to a fresh X86host lane tree. Record its
commit in `<tree>/SOURCE_COMMIT` and export `DSPARK_SOURCE_COMMIT` for dumps.
The worker independently verifies the retained checkpoint delivery and official
shards before deriving the same identity. No weight path/content is inferred
from a requested identity.

The native recipe covers the official 1017 manifest/receipt and verified shard
hashes, complete arm config, K6 label and retained checkpoint provenance, plus
runtime `.py/.cu/.cuh/.cpp/.h/.c/.json` sources under worker/native/kernels.
Tools, tests, caches and result directories are excluded. K2 retains its
existing §6.7 identity and wire bytes.

## Solo qualification

On X86host the lane alias/root is `~/dsv41-offload/n3`. Launch long work detached:

```bash
source ~/.config/ndm/paths.sh
tmux new-session -d -s n3-solo-example \
  'source ~/.config/ndm/paths.sh; bash ~/dsv41-offload/n3/repo/offload/dspark/native/tools/n3/run_locked.sh solo-example -m offload.dspark.native.tools.n3.shard native-hybrid-kx2 --rank 0 --out /home/operator/dsv41-offload/n3/results/solo-example.json'
```

Each wrapper re-reads the coordinator's live LANES log, refuses an existing
PAIR_OK, takes the shared 7744b44b flock, rechecks PAIR_OK, pins the UUID, refuses
an occupied card, starts 1s telemetry, and bounds one process to830s+5s cleanup.
It terminates its own job if PAIR_OK appears. It releases between steps.

`shard ARM --rank 0|1 --out ...` measures each real shard's load memory/time,
full40 B1–4 × n_out1–5 × margin on/off graph prewarm, then2000 replays.
Rank1 replay executes the rank1 graph but has no rank0 confidence output;
its output checks are explicitly labelled absent. These are local-collective
shard measurements, not paired serving measurements.

`soak ARM --out ...` exercises the actual server factory/core/backend in full
TP2 emulation:2000 protocol rounds dump-off, observer installation/recapture,
2000 dump-on, all40 graph shapes, incremental state and generation/resync,
HELLO identity/config refusals, valid outputs/tails, and exact reply digest
equality. It emits selected V1 records and requires each traced eager replay
to equal its actual graph reply. Expanded full emulation is refused before
allocation: two~17GiB shards do not fit a24GiB card. Its two solo shard checks
and coordinator pair soak are the required substitutes, explicitly labelled.

## V1 dumps and entry gates

Serving debug flags, disabled by default:

```bash
export DSPARK_V1_DUMP_DIR=/home/operator/dsv41-offload/n3/results/debug-native
export DSPARK_V1_ROUNDS=1,5,40
```

Rounds count backend draft calls; `wire` records the actual session/seq. Dumps
contain `n_out`, drafts/confidence, full vocabulary base logits at the consumed
positions, live `kv_window`, actual insertion-graph `main_x_P`, window/source
metadata, s0–s2 attention/FFN/stream/routing, normalized head hidden, K6 identity,
source revision and UUIDs. Selected graph replies are compared to a traced eager
replay on the same state; mismatch fails the round. Off mode installs no
observers and imports no dump module. Dumps deliberately synchronize and are
**not** latency measurements. Incomplete n_out<5 records are refused by the V1
entry-gate tool, though they remain useful protocol diagnostics.

V1 capture replay uses the independent `verify.captures.CaptureSet.arm_inputs`
contract. It preserves the native-tap vs engine_mx-only prefill row provenance
at every position. This typed engine_mx injection is a replay-only API; it is
not an additional wire payload. It emits the exact requested keys, grouped by
captured round, through the same server factory/backend:

```bash
python -m offload.dspark.native.tools.n3.replay native-hybrid-kx2 \
  --verify-root /home/operator/dsv41-offload/n3/verify-source \
  --keys r000008_b0,r000009_b0 \
  --out /home/operator/dsv41-offload/n3/results/capture-example
```

Run under the solo wrapper or coordinator pair wrapper, never unguarded. V1's
independent package is staged separately under `verify-source/verify`; N3 does
not edit it. Copy private captured dumps to Controller, then run there:

```bash
source ~/.config/ndm/paths.sh
$HARNESS_PYTHON -m offload.dspark.native.tools.n3.classify \
  --verify-root /opt/dsv41/repos/.worktrees/dsv41-v1-verify/offload/dspark \
  --arm-dir /opt/dsv41/data/n3/capture-native-hybrid-kx2 \
  --arm-name native-hybrid-kx2 --expected-keys @/opt/dsv41/campaigns/DSV41-OFFLOAD/N3-v1-keys.txt \
  --dumps /opt/dsv41/data/v1/dumps \
  --captures /opt/dsv41/data/v1/captures/A-capture-20261022 \
  --ckpt /opt/dsv41/data/v1/native-1017 \
  --official /opt/dsv41/data/ref-ds41/inference \
  --out /opt/dsv41/campaigns/DSV41-OFFLOAD/N3-kx2-gates.json
```

The wrapper calls V1 `classify --ref-mode mirror --band-mode mirror64
--representation native` without altering its arithmetic or classifications.
It refuses missing/extra requested keys or incomplete traces, then requires:
zero state defects (including agree-but-state-flag), zero unexplained cases,
s0.attn_out median excess<3×, KL median≤3× the measured summation-band median.
A bounded key set is only a bounded pilot; expand it before a full entry verdict.
K6 head representation/Markov differences may fail these strict gates; report
that failure to V1, do not suppress it or claim GB10 readiness.

## Coordinator-only physical pair sequence

Preconditions: drain/own the production pair window using the campaign's
existing coordinator procedure; stop the existing worker yourself; ensure the
permitted pair is unoccupied. The script performs no service actions. Read the
current PAIR_OK/LANES and use the current R3/RX p2p source and matching pinned
prebuilt extension (`DSPARK_P2P_PREBUILT`, defaults to `~/dsv41-offload/n3/p2p`).
N3 stages the verified existing R3 binary for source08e1903b0a43feff; the pair
script verifies its pin before taking locks and never silently JIT-builds it. The loader refuses a
prebuilt/source/hash mismatch. An intentional poison test remains a separate
RX/coordinator operation; do not run it during normal traffic.

Stage `pair.sh` as `~/dsv41-offload/n3/n3_pair.sh`, then on X86host, detached:

```bash
source ~/.config/ndm/paths.sh
# Put every selection inside tmux's command; it may not inherit caller exports.
tmux new-session -d -s n3-pair-window \
  'source ~/.config/ndm/paths.sh; export N3_PAIR_BY_COORD=1; export N3_PAIR_ARMS="native-packed native-expanded native-hybrid-kx native-hybrid-kx2"; export N3_V1_KEYS="$(paste -sd, "$HOME/dsv41-offload/n3/v1-keys.txt")"; bash ~/dsv41-offload/n3/n3_pair.sh > ~/dsv41-offload/n3/logs/pair-console.log 2>&1'
# Or omit the override when PAIR_OK begins N3 / N3-NATIVE-SERVE.
# An existing marker's withdrawal/replacement always stops the owned job.
```

Each arm runs separate bounded holds (≤820s+10s teardown), releasing between:
(1) transient TP2 worker, wait for its own ready PID/status/identity, local TCP
fake-client2000round soak, stop; (2) same with debug dumps, compare on/off reply
digests, stop; (3) captured V1 dump replay via the identical worker factory after
the listener has stopped. Both flocks are acquired nonblocking together; the
script never holds one while sleeping for the other. Authorization is watched
through each hold; withdrawal stops only its own processes. The job verifies
actual inherited flocks, not just an environment flag. Per-card1s telemetry
covers load/prewarm and serving; all output goes to a new timestamped directory.
Any failure stops the script. Do not reuse a partial output directory as a pass.

After pair and V1/RX gates pass, start exactly one transient worker for each
coordinator engine arm (the coordinator owns both locks and production drain):

```bash
source ~/.config/ndm/paths.sh
cd ~/dsv41-offload/n3/repo
export PYTHONPATH=$PWD
export OFL_PY_DIR=$HOME/dsv41-offload/n3/lib
export OFL_LIB=$OFL_PY_DIR/libofl.so
export DSPARK_P2P_PREBUILT=$HOME/dsv41-offload/n3/p2p
export CUDA_VISIBLE_DEVICES=GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93,GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc
export DSPARK_WORKER_ARM=native-hybrid-kx2
export DSPARK_SOURCE_COMMIT=$(cat SOURCE_COMMIT)
unset DSPARK_V1_DUMP_DIR DSPARK_WORKER_TP_EMULATE
~/dsv41-offload/e/venv/bin/python -u -m offload.dspark.worker.server \
  --tp 2 --root /opt/dsv41/x86host-data/dspark --max-model-len 1048576 \
  --ofl 10.202.28.1:18620 --status-file /home/operator/dsv41-offload/n3/results/engine-kx2.status.json
```

Run the corresponding Spark config through the existing coordinator launcher:
E2 `E2-native-packed-async.env`, E3 `E3-native-expanded-async.env`,
E4 `E4-native-kx2-async.env`; `E4-native-kx-async.env` retains the prior hybrid.
Matching `*-sync.env` files set `VLLM_DSPARK_REMOTE_ASYNC=0` with distinct labels;
all four native arms have both sync and async files. Recompute identities from the final merged source first. Check startup
arm/head/identity, max length and all4slots, then HELLO agreement and refusal of
K2/native mismatch. On GB10 check process proxy running, self-test/mlock,
proxy_torch_loaded=false, watchdog/preempted/skipped=0, no internal/TP errors,
correct target-only fallback, and reconnect after the coordinator stops/restarts
its own transient worker. Keep raw captures off during timed engine runs.

Compare short/32K/150K, B1 and B3/4, ≥3 interleaved repetitions. Record actual
accepted/committed tokens, complete remote critical path, target verify/round
period, latency/TTFT, both cards' load/serving peaks and full runtime identity.
K2 agreement is diagnostic; target acceptance and full useful throughput decide.
The inherited E-config depth numbers are provisional one-card screening values,
not a claimed native pair recalibration.

## Per-arm depth recalibration recipe

For each of the four arms, after V1 qualification, use its own captured dumps:

```bash
PYTHONPATH=/path/to/v1/dspark python -m verify.depth_calib \
  --source arm --arm-dir /private/full-arm-dumps --out /private/ARM-confidence.json
```

Measure fixed depths0..5, B1 and B3/4, occupied short/32K/150K cells separately,
with full server-side round IDs. Keep the trained maximum5. V1 `roundlog` CUDA
events are sampled/harvested without per-round sync; preserve adjacent round
records. Include depth vector, planned/graph rows, occupied context, cache,
salted task-family groups and request hashes. Split whole matters/task families:
≥3 fit and≥2 held-out groups per cell; the current4-request capture is a pilot
and cannot qualify the fit. Keep sync (`two_card_tp`) and async (`remote_async`)
records separate. For async, explicitly measure remote/verification overlap at
the first consumer; V1's default0 for missing overlap is not a measurement.

```bash
PYTHONPATH=/path/to/v1/dspark python -m verify.depth_rows /private/ARM-rank0.jsonl \
  --mode remote_async --build 'ARM:FULL_DRAFTER_ID:MERGED_REVISION' \
  --salt "$PRIVATE_FIXED_SALT" --out /private/ARM-depth-rows.jsonl
python -m offload.dspark.native.tools.n3.depth_config /private/ARM-cell-rows.jsonl \
  --mode remote_async --build 'ARM:FULL_DRAFTER_ID:MERGED_REVISION' \
  --requests 1 --context 32768 --overlap-audited --out /private/ARM-depth-config.json
```

Filter the V1 rows into exact occupied-context/cache/request cells first, retaining
all depths/groups. Exporter refuses mixed identities/modes/cells, sparse groups,
queued samples, invalid/missing costs, or a nonmonotonic measured verify table.
`ROUND_MS` is the fit median verify-only target forward cost at each depth0..5.
`DRAFT_MS` is the measured non-overlapped remote prep cost (`prep_ms-overlap_ms`)
over depths1..5, not raw worker compute and not another copy of old remote cost.
It prints the measured lines and reports held-out complete-round prediction error:

```text
VLLM_DSPARK_REMOTE_DRAFT_MS=<measured effective remote milliseconds>
VLLM_DSPARK_REMOTE_ROUND_MS=<verify-only d0>,<d1>,<d2>,<d3>,<d4>,<d5>
```

A scalar is only an approximation; do not promote if held-out errors or context/
concurrency variation make it unsuitable. Retain per-cell tables for X1's direct
round-cost path where available. Shadow-log predictions/depth decisions and
promote only after held-out useful throughput/acceptance and tail checks. N3
provides no invented pair/GB10 cost numbers, and does not deploy any policy.

## Observed packed V1 pilot failure

N3's first12-key serving capture passes graph/eager equality but fails strict V1:
2head/logit selfcheckflags, s0 attention median excess600.402x and KL3.41x the
summation band. CPU recomposition with the declared BF16 Markov/sum policy
reproduces all60actual drafts; V1's official FP32 recomposition differs on2samples.
The inherited attention kernel also uses one global softmax/BF16P rounding,
while the official mirror uses online64-key blocks and per-block BF16P. A bounded
CPU counterfactual demonstrates this contribution; it does not explain every
residual. This is a failed pilot, not a gate exemption. Native math and V1 remain
unchanged. Before GB10, obtain a deliberate attention/head-policy fidelity fix
and V1review with fresh identities/captures. Other-arm/pair results are separate.

The staged OFL binary is the existing qualified E/prod pin
`9f11b6e8d4f459c07ac35745eec95e5e54fcf705d5085e454faab7386520356d`.
The existing R3 prebuilt p2p binary is
`a5ac180b2e9b7b8f93f0dd4baefc57c2a04daebb2636f1690275870258a8b6bc`,
bound to p2p.cu
`08e1903b0a43feffa5ee5fb465ea163bbe6725bc05290d05f2f837042a6844b9`.
These are copied locally into N3's own directories, not rebuilt or downloaded.
A coordinator merge changing p2p.cu requires a matching new prebuilt/pin first.

After the coordinator has staged the final merged source/identities and obtained
V1/RX gates, these are the existing launcher commands **on sparkhead only**,
inside its already-open production-drained HOLD window (not run by N3):

```bash
source ~/.config/ndm/paths.sh
cd ~/dsv41-deploy
bash offload/launch/start_exp.sh offload/launch/configs/E2-native-packed-async.env
# Or E2-native-packed-sync.env, E3-native-expanded-{async,sync}.env,
# E4-native-kx2-{async,sync}.env, E4-native-kx-{async,sync}.env, one at a time.
```

The current12-key pilots fail, so this is a handoff command, not authorization
to bypass the failed entry gate. Keep the unchanged K2 production configuration
as rollback under the coordinator's normal window/worker procedure.
