# Decode second-pass pilot staging

The coordinator must use this order (amendment 7b / S35 FIX-2):

1. Run `decode_close.sh pause` and wait for its return with `D2-GEN2-EXIT` seen.
2. Immediately capture the final `/metrics` snapshot as `metrics-<UTC>.prom`
   and save the generator's last `rate.jsonl` row (the forced exit row) as
   `rate-<same UTC>.json`. Record the pause time. Capture within 120 seconds of
   the generator's exit; the existing 120-second freshness check is unchanged.
3. Collect the attestation evidence with `pass2_collect.sh --execute` while the
   derivative still serves, so the mounts reflect the captured deployment.
4. Flush with a NEW nonce, then copy and record that nonce at copy time.
5. Roll back. Only then resolve/review the evidence and stage attestation/context,
   transfer/hash-check, assemble/G1(a), build the live reference, score K2/G1(b),
   and stage the pilot after the K2 holds release their locks.

Do not delay the final snapshot or exit row until after collection, flush, copy
or rollback: the paused generator writes no periodic rate rows, and a later
snapshot cannot recover this measurement.

Prepared on Controller, CPU only. These commands have **not** run on x86host. The
script never connects to a host; the coordinator runs it locally on x86host after
reviewing and transferring this checkout by the existing coordinator process.
No service, generator, capture writer, W12 or production-pair operation is part
of this procedure. [PILOT.md](PILOT.md) and plan amendments 3–6 remain binding.

`pilot_stage` is dry by default, including on Controller. It renders all commands
without reading the default x86host data paths, invoking git, tmux or CUDA, or
creating files. `--check-inputs` explicitly enables CPU validation of a LOCAL
receipt tree. `--execute` requires `D2_COORDINATOR=Deepseek/coordinator`, a local
hostname of x86host/x86host, and the exact reviewed second-pass data path.
Do not use `--execute` in a lane rehearsal.

## Amendment-7 prerequisites

The exact assembled root is `/opt/dsv41/x86host-data/d2/decode-d2pass2-r1-c2`, the verified copy retry
(the first attempt's `decode-d2pass2-r1` tree stopped at the Spark memory guard, has no receipt and is never used).
Staging verifies ASSEMBLED.json hashes, copy binding, generation-time option-ii
verification, decode identity, shard structure/probabilities and G1(a) plumbing.
Any synthetic marker refuses. At least 20,000 unique training blocks are required.

The pilot DEV population is **dev + heldout_legal**. Original acap is historical
and cannot admit training. Supply the amendment-7b binding `dev` G1(b) PASS,
the live-reference receipt and all of its hash-bound local snapshot/rate sources,
and one exact K2 JSONL file per DEV split. See [G1-LIVE.md](../G1-LIVE.md).
The G1 reference must be VALID (foreign share <=2%), with the gross-error tripwire passed: dev proxy position-0 match rate
>= 0.5 times live position-0 acceptance. The ±0.03 rule is withdrawn. Wrong splits, missing/changed live sources, historical or synthetic G1
receipts refuse. Stage recomputes the exact dev proxy from current shards/K2 and
compares the entire G1 receipt, including MANIFEST/K2/live reference identities.
A fresh assembly receipt must cover the final MANIFEST; do not append splits.
The existing genuine 50-sample qualified reference parity receipt is also required.

Amendment-4 sampling remains seed **20261024**, at most **1,000 complete rounds
per DEV split**, drawn once before training. Amendment 3–6 numerical verdict
rules are unchanged. Historical acap analytical helpers remain for regression
fixtures and history; real plan/sample admissions require amendment-7 DEV splits.
The reserved TEST population and heldout_general are untouched.

## Coordinator procedure, local x86host only

First review and transfer the committed checkout using the coordinator's existing
process. It must be clean and HEAD must equal the full reviewed integration
commit. There is no remote-transfer command in this script. Source inspection,
GPU facts and venv selection here come from existing repository launchers/plans,
not from a live host probe by this lane.

From that checkout, replace `REVIEWED_FULL_40_HEX_COMMIT` and the three exact G1
paths. Do not reuse the earlier replay pilot's source/output namespace.

```bash
set -euo pipefail
source ~/.config/ndm/paths.sh
export PYTHONPATH="$PWD:$PWD/offload/dspark"
export CUDA_VISIBLE_DEVICES=''
D2_STAGE_PYTHON=$HOME/dsv41-offload/e/venv/bin/python
pilot_stage_args=(
  --source "$PWD" --commit REVIEWED_FULL_40_HEX_COMMIT
  --snapshot "$HOME/dsv41-offload/d2/pilot/decode-d2pass2-r1"
  --output /opt/dsv41/x86host-data/d2/pilot/decode-d2pass2-r1
  --g1 /EXACT/SECOND_PASS/dev.json
  --k2-dev /EXACT/SECOND_PASS/dev.k2_drafts.jsonl
  --k2-legal /EXACT/SECOND_PASS/heldout_legal.k2_drafts.jsonl)
"$D2_STAGE_PYTHON" -m offload.dspark.d2.train.pilot_stage "${pilot_stage_args[@]}"
# Explicit LOCAL CPU admission check, still no launch:
"$D2_STAGE_PYTHON" -m offload.dspark.d2.train.pilot_stage "${pilot_stage_args[@]}" --check-inputs
```

Inspect free RAM and the existing exact `dsv41-powerlog` tmux session locally
before launch, as PILOT.md requires. No new power logger or service is started.
The pilot launcher checks each selected solo's unchanged **225 W** power limit,
leases it with nonblocking flock, and supervises child RSS <=128 GiB/no swap.
The existing interpreter is `~/dsv41-offload/e/venv/bin/python`, dependency record
`requirements-x86host.lock`; native checkpoint is `/opt/dsv41/x86host-data/native-1017`.
No install, GPU power change or model download is needed.

Execute one phase at a time, only after checking the preceding phase's artifacts:

```bash
export D2_COORDINATOR=Deepseek/coordinator
"$D2_STAGE_PYTHON" -m offload.dspark.d2.train.pilot_stage "${pilot_stage_args[@]}" --execute --phase stage
"$D2_STAGE_PYTHON" -m offload.dspark.d2.train.pilot_stage "${pilot_stage_args[@]}" --execute --phase prepare
# Wait for prepare coordinator log to finish successfully and plan.json.sample to exist.
"$D2_STAGE_PYTHON" -m offload.dspark.d2.train.pilot_stage "${pilot_stage_args[@]}" --execute --phase train
# Wait for all three complete arms/*/arm.json receipts.
"$D2_STAGE_PYTHON" -m offload.dspark.d2.train.pilot_stage "${pilot_stage_args[@]}" --execute --phase evaluate
# Wait for dev.part.json AND legal.part.json; inspect logs for failures.
"$D2_STAGE_PYTHON" -m offload.dspark.d2.train.pilot_stage "${pilot_stage_args[@]}" --execute --phase finish
```

Stage uses local `git archive` into a NEW snapshot and writes `SOURCE_COMMIT` and
private `STAGED.json`, binding source, inputs, output namespace and exact command
bodies. No git environment override is used; inherited `GIT_*` variables are
removed from child git operations. Every later phase rechecks inputs and source
bytes. Keep the reviewed checkout available and clean until completion.

Prepare uses the REF lease, then freezes the sample on CPU (seed **20261024**,
up to 1,000 complete fixed-K2 rounds per split) BEFORE launching any arm. It is
once-only. Training launches REF on
`GPU-240876ad-ea42-47de-1b8d-f6153810b7e1` concurrently with FAST then FAST-B on
`GPU-bc62df2a-59cf-9828-7d43-558d1355bab7`. The other UUIDs refuse even in dry mode.
FAST-B starts only after FAST exits successfully. Evaluation launches dev on
240876ad and legal on bc62df2a, after all three arms finish. Finish merges the
two reference split evaluations and runs the unchanged Amendment-6 verdict.

Session names are `d2pilot-d2pass2-r1-{prepare,ref,fast,eval-dev,eval-legal,finish}`.
Scripts and coordinator logs live in the snapshot (`NAME.sh`, `logs/NAME.coordinator.log`);
leased child logs are `logs/{prepare,REF,FAST,FAST-B,eval-dev,eval-legal}.log`.
Inspect/join only these new sessions. Existing names refuse; no session is killed.
A detached launch is not a completion signal: inspect logs and the named receipts.

The normal checkpoint cadence is 300 seconds plus the exact 2,500/5,000/10,000/
20,000 block boundaries (20/40/80/159 updates). `arms/ARM/latest.pt` and immutable
`latest.checkpoints/step-NNNNNNNNN.pt` persist optimizer/scheduler/RNG state.
On interruption, preserve the sampled plan, source and namespace. Inspect the
saved boundary and rerun ONLY the interrupted arm's rendered locked command
with `--resume`, in a newly named coordinator-owned detached session. For the
FAST chain, resume FAST only if incomplete, then launch FAST-B normally if it
never started, or resume FAST-B only if that was interrupted. Never repeat
prepare/sample or a completed arm. The stage launcher deliberately refuses
relaunching an already staged job script; it does not guess resume state.

Final result: `/opt/dsv41/x86host-data/d2/pilot/decode-d2pass2-r1/receipt.json`.
Only a genuine real PASS authorizes the separate full FAST training and full G2
procedure in [FULL-RUN.md](FULL-RUN.md) (PILOT.md step 5). Nothing here launches full training, packages an adapter, passes final
G2/G3, changes a gate, or deploys anything. On FAIL/INCONCLUSIVE report the exact
receipt to the coordinator; no automatic extra seed or sample redraw.

## Cost and CPU limits

**ESTIMATED** from PILOT.md's recorded historical synthetic throughput:
20,000 REF blocks / 2.7 blocks/s = 2.058 GPU-hours; two FAST arms at 190 blocks/s
add 0.0585 GPU-hours. Total training ≈2.116 GPU-hours. Thirteen reference passes
over <=2,000 sampled rounds at the historical 0.5046 seconds/round add <=3.644
GPU-hours. Combined pilot ≈5.760 GPU-hours; ideal parallel wall time ≈3.880 hours
(REF training followed by balanced split evaluation). Add source/input hashing,
I/O, model loads, diagnostics and scheduling overhead. These are estimates,
not measurements on decode-capture data. Final full-set G2/fresh test and worker
qualification have separate costs; no target replay is launched by this procedure.

**MEASURED CPU rehearsal:** assembly's synthetic capture writer/copy/export/shard
helpers feed D2Blocks; rank 21 remains a present top20+tail target with a sampled
label outside top20. Sample freezes before four REF optimizer updates and four
FAST explicit bucket-loss/backward updates; both reference scoring split paths
validate exact sample coverage. Tiny synthetic weights and a test-only input
view reduce dimensions; shard labels, positions, source/request identities and
probabilities remain unchanged. Full FAST CUDA/Triton admission still refuses
CPU. CPU checks cannot establish GPU arithmetic, throughput, real G1 or pilot PASS.


Current full-run planning uses the coordinator's MEASURED real pilot rates
(FAST approximately .85 seconds/update, REF approximately 40 seconds/update);
the historical synthetic pilot estimates above are retained as history.
Full G2 scores all rounds and has a separate approximately 51 GPU-hour estimate;
see FULL-RUN.md before budgeting or launching it.
