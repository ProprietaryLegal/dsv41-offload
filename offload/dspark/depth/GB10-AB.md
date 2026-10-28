Status 2026-10-23: SP2 CPU-qualified experiment plan only; no new depth optimum measured. Coordinator must capture paired fixed-depth TP2/async costs before policy changes. See [live roadmap](../../../documents/plans/roadmap.md) and [measured record](../../results/COMPARISON.md); campaign review decisions are in `LANES.md` / lane reports outside the repo.

# SP2 GB10 A/B and recalibration plan (coordinator only; not executed by SP2)

All thresholds below are proposed TEST CRITERIA, not measured results. Configuration literals identify arms.
Use the existing window/prod_window/pair_window ownership protocol. This CPU lane has no Spark access.

## Ready configurations

`offload/launch/configs/D-kv6-async-depth-*.env` source the pinned, self-contained control. The control reproduces
main `3c1dc6b`'s complete active OFFLOAD_DSPARK_ENV + OFFLOAD_DOCKER_ENV, including process-proxy supervision,
Engram REQUIRED=0 fallback, identities, timeout/backoff, six GiB KV and maximum context 327680. Only experiment
overlay/mount locations and API port differ through start_exp. All candidates keep five draft query positions.
They have no K1/UP1/native overlay, target-weight, temperature, reasoning, parser or kernel-setting change.
Existing production work on other branches is untouched. If production changes, diff and freeze a NEW control first.

| suffix | only policy difference from control | purpose |
|---|---|---|
| control | none; inherited remote scalar 4.4 ms and verify table | C in this plan = TP2 + production async config |
| residual0 | remote scalar zero | ESTIMATED full-overlap bound; not a measured zero-cost RPC |
| residual08 | remote scalar .8 ms | ESTIMATED B1-centric initial cost arm |
| residual15 | remote scalar 1.5 ms | ESTIMATED residual-cost sensitivity, especially larger batches |
| cap3 | scalar .8 ms, schedule [[1,4,3]] | cap-three screening arm; scheduler/individual adaptation still on |
| cap5 | scalar .8 ms, schedule [[1,4,5]] | tests uncensored fifth position at larger batch sizes |
| fixed0 … fixed5 | fixed per-request depth, scheduler adaptation OFF, common cap five | calibration only, not promotion candidates |

No scalar/table is called a measured recalibration. The control's inherited table also remains unqualified for new
TP2 async cells. The fixed arms preserve the full five-position noncausal backbone and n_out five, isolating verify
masking without accidental query-block changes. Because ordinary adaptive caps can use n_out four at B3/B4, record
this cost difference; do not reuse a fixed-n_out-five draft cost blindly in the adaptive n_out-four arm.

## Freeze and qualify C before X

1. Coordinator records exact repo/runtime/image, target and drafter identities, TP2 worker revision, GPU/power policy,
   native arm (must be k2-tp2), config hashes, occupied tokens and cache state. Arrange the physical NVLink pair under
   its existing hold protocol; verify expected HELLO and TP2 stats. Do not assume two cards are active from an env name.
2. Drain and open the coordinator's existing window; no independent production traffic. Start C on the experiment
   port with start_exp.sh. The snippets below run ON THE COORDINATOR'S Spark checkout, never in this lane:

   ```bash
   offload/launch/start_exp.sh offload/launch/configs/D-kv6-async-depth-control.env
   ```

3. Confirm startup reports async host-flag self-test, mlock ok, proxy process, correct TP2 connection, confidence
   policy and effective six-value cost table, scheduler caps, no silent sync fallback. Save both ranks' current
   counters. Warm short AND 32K paths identically after every restart; exclude those from measurements.
4. Before measuring performance: six established greedy prompts, thinking off, fresh salts, C/C then C/X/C,
   token-level identity including EOS at short/32K/150K and B1/B3/B4. Store token sequences privately, export only
   equality counts and first mismatch position. Existing `engram/tests/logit_ab.py` can supply fixed-prefix probes;
   reuse W5's greedy set too. If C/C itself drifts, diagnose fixed-prefix target logprobs/tokens against the established
   engine numerical floor rather than declaring X correct because stochastic text looks similar. Any unexplained
   X-specific mismatch stops promotion. No approximate acceptance/typical-posterior setting is permitted.

## Balanced performance sequence

For EACH candidate, run three separately warmed **C / X / C blocks**, preserving paired prompt/task seeds, sampling,
fixed output budgets, cache salts/prefix policy and hardware conditions. At least three repeats PER arm PER context
and batch cell. Do not count B3's three simultaneous requests as three independent repetitions. Start with residual08,
then residual0/residual15 as needed. Compare cap3/cap5 to residual08 as the matching-cost control so the cap comparison
has only one changed factor; also retain the original C bracket for total benefit. No concurrent benchmark owners.

Each block's concrete matching matrix (coordinator chooses a fresh LABEL and local OUTDIR):

```bash
# Run in a coordinator-owned drained window only, after start_exp and identical warm-ups.
# These are commands for the coordinator; SP2 did NOT execute them.
for ctx in 0 32000 150000; do
  case "$ctx" in 0) work=legal ;; 32000) work=summary ;; 150000) work=synth ;; esac
  python3 offload/bench/bench2.py --url http://127.0.0.1:8001 --label "$LABEL" \
    --ctx "$ctx" --workload "$work" --concurrency 1 3 4 --reps 3 --max-tokens 1500 \
    --cache-mode cold --thinking --reasoning-effort high --temperature 1 --top-p 1 --seed 42 \
    --priority 10 --data-dir /opt/dsv41/data/bench-data --out "$OUTDIR/$LABEL-$ctx.jsonl"
done
```

Cold salts must produce matching occupied tokens and task text across labels; verify prompt hashes privately.
Measure warm prefix separately, never mix cold/warm observations. Retain B2/32K as a diagnostic for the missing TP2
cell. Screen short/32K first; send viable arms through all 150K cells, a warm-prefix repeat, mixed long-prefill/decode
and one full representative task at the SAME effort. Do not rank by end-to-end rate containing different prefills.
No need to run 256K/300K until a candidate wins the requested matrix.

Primary metric: ACTUAL delivered target tokens / steady decode wall time, aggregate and each stream, with reasoning
and visible answer counts/timing separate. Also report completion latency, TTFT/first visible, ms/engine step, actual
commits and accepted drafts per request-round, depth histograms, active batch/graph rows, n_out, cache hits,
failures/retries, queueing, worker prep/RPC, overlap at first consumer, memory headroom and both ranks' stats.
The acceptance fraction of proposed tokens alone cannot decide a winner. Compare paired C brackets; bootstrap whole
requests/task-family groups, not token events. Report all repetitions and paired uncertainty, not best-of-three.

## Capture the missing measured depth tables

In a separate calibration pass, start fixed0 through fixed5, retaining the same TP2 async proxy and target settings.
Capture quiet steady full-B cells at B1/B2/B3/B4, occupied short/32K/150K, with whole task-family groups split in
advance into at least three fit and two disjoint held-out groups. Do not infer full batch from requested concurrency:
select trace rows with that actual active batch. Keep engine round/request-generation IDs in PRIVATE traces only.

Use the V1 paired CUDA/CPU marker path referenced by N3's README; record actual verify start/end, draft preparation
and reply-ready/first-consumer timestamps, host submission, graph/planned/live rows, insertion and queue costs,
actual commits and context. Confirm overlap exists in the trace, not just a default zero field. Do not subtract
independent histogram p50s, worker p50s or clocks from separate runs to infer overlap (R1's warning).
Construct `dsv41.depth-round.v1` per cell and run N3's strict export:

```bash
python3 -m offload.dspark.native.tools.n3.depth_config "$PRIVATE_CELL_ROWS" \
  --build "$ARM_FULL_ID_RUNTIME_REV" --mode remote_async --requests "$B" --context "$CTX" \
  --overlap-audited --out "$PRIVATE_FIT_JSON"
```

Fit verifies only on fit groups. Require held-out median absolute round error <=5% and maximum <=15% as proposed
screening gates, no batch/context bias, and actual useful-tps validation. N3's scalar fit includes draft overhead at
zero depth too; isolate target-only no-proposer diagnostics separately. If one scalar/table cannot fit the cells,
retain current policy and recommend the already-planned per-cell cost path (`spark_leads/05-depth-cost-plan.md`),
not arbitrary table averaging. Do not promote a B1 fit to B4 or native arms. New measured fit gets a separately named
config and measured source/receipt. Re-run useful throughput after installing it.

## Stop lines, decision and rollback

- Stop immediately for identity mismatch, state/rollback/branch contamination, unverified output, unexplained greedy
  drift, TP failure, Xid, lost rank or target settings changing. Preserve private evidence, return to C.
- Stop for unexpected steady decode fallback/watchdog/respawn after warm-up; fail a supposedly async arm if it falls
  back to sync. Known, deliberately induced failure tests are separate from timing and never improve the score.
- Stop if MemAvailable <1.5 GiB on either Spark, active swap-in/out, growing memory across repeats, request timeout or
  another owner's traffic. Use existing telemetry; no new services or system policy changes.
- Stop screening X if two paired blocks show >5% completed-task or useful decode regression in any required cell,
  or visible-answer latency/per-stream fairness degrades materially. Inspect queue/cache/prefill cause before retries.
- Promote only repeatable useful decode gain around >=5% beyond the paired noise at matched settings, with no
  meaningful required-cell regression, legal-task behavior unchanged and zero correctness faults. If gain is below
  threshold, keep C and report no qualified win. Small CPU-only predictions are not an exception.
- Restore experiment control using the control start_exp command above. At window close restore the coordinator's
  EXACT pre-window production revision and config with its existing `./stop.sh` / `./start.sh` sequence, verify alias,
  warm-up, health, identity, async/TP2 counters and no pending requests, then close the ownership window normally.
  Keep the already-tested TP2-to-one-card worker rollback available; do not invent a lane-owned kill command.

SP2 performs none of these GPU, API, SSH, process or service actions. No new permission is needed to review the ready
configs; hardware execution belongs exclusively to the coordinator.
