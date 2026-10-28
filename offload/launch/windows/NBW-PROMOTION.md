# NBW conditional promotion package — NOT APPLIED

No NBW serving window has run. Native is not a measured CANDIDATE yet. This is
an attended-apply design, usable only after a complete, restored NBW window
returns CANDIDATE and NDM separately approves promotion. A speed result does
not remove N4/N5's outstanding fidelity qualifications. A rejected or incomplete
window stops this package without any service, unit or config change.

The candidate evidence packet must contain the reviewed source commit and both
native identities, hashes of all cell/greedy artifacts, the four mandatory
complete starts, the fixed K-to-K-spread decision, and a restored production
receipt with all six distinct IDENTICAL lines. Include TTFT and acceptance as
well as tok/s. Missing actual-batch/confidence/exposed-wait traces limit causal
explanations. The paired-window runner is not a production deployment script.

## Proposed change, only on that later approval

| Location | Present | Proposed |
|---|---|---|
| X86host current link | `~/dsv41-offload/prod/dspark-worker/current -> fb68c52` | Immutable release containing exactly the tested NB1 runtime bytes and full SOURCE_COMMIT |
| X86host enabled TP2 unit | `dsv41-dspark-worker-tp2-prod.service`, K2 | Same enabled unit, same two UUID locks, TP2 and HELLO bound; reviewed native startup drop-in |
| X86host native settings | K2 default | DSPARK_WORKER_ARM=native-hybrid-kx2, DSPARK_NATIVE_BATCH_INVARIANT=2, DSPARK_SOURCE_COMMIT=tested full commit |
| Spark production env | K2 drafter / Markov identities | Tested native drafter / Markov identities, no other model/policy/Engram/KV/cap changes |

Stage with the existing `worker/run/prod_tp2.sh stage` mechanism into a new
release directory; never replace fb68c52 or its binary pins. Before HOLD,
prepare and review a byte-exact old/new unit/drop-in and Spark dsv41.env diff.
Keep those backups and hashes outside the production source directories.

The native drop-in needs more than the invariant env variable. It must use the
same successful window interpreter (`~/dsv41-offload/e/venv/bin/python`), root
PYTHONPATH (the release root, importing `offload.dspark.worker.server`), N4
OFL/p2p/extension/Triton caches and tune cache. Use NBW's reviewed prebuilt-only
NATIVE_BOOTSTRAP startup wrapper. Preserve the unit's paired `gpu_locked.sh`
ExecStart boundary, listener `10.202.28.1:18620`, root delivery, TP2/1048576,
production status/log paths, Restart behavior and 48 GiB memory limit. Do not
use the old unit's `-m worker.server` with the new root accidentally. Never let
a missing cache compile or fall back during startup. Copy the wrapper outside
the runtime-source tree (or under tools) and pin its own SHA256 in the packet.

The installed unit file was not read by NBW; the repository service template
is the reference. Compare it to the actual installed unit/drop-ins at attended
review. If different, revise the prepared diff before stopping anything.

## Attended apply checklist

1. Verify CANDIDATE, zero failed rounds, six native greedy matches at each I
   start, explicit promotion approval and disposition of the N4/N5 limitations.
   Reconcile current production against the packet; if it moved, stop and
   re-plan. Save the exact current-link target, complete unit/drop-ins, both
   Sparks' config/argv/mounted hashes and both identities. Confirm rollback
   release fb68c5291792ab90c0f468bd94010917967799f3 exists with its original pins.
2. Pause producers, open a fresh coordinator HOLD, drain, and retain HOLD until
   every apply or rollback gate passes. Reserve recovery time before changing
   either side. No concurrent lane may own the pair or experiment endpoint.
3. Stop the Spark serve while drained, then stop the TP2 worker through the
   pair-window mechanism. Prove the owned experimental native launcher absent.
   With both GPU locks free, atomically switch current to the prepared immutable
   release and install the reviewed TP2-only drop-in. Preserve unit enablement;
   the one-card unit stays disabled. Reload systemd. No weight, image, power,
   clock or firmware change.
4. Patch only the two Spark production identity values, preserving all K1,
   OPS1 warm-up, target and sampling settings. Restart the TP2 unit and inspect
   its actual cwd, source, UUIDs, level=2, TP2, HELLO bounds, fresh status, source
   digest, native/Markov identities and lock ownership. A ready log alone fails
   the gate. Close the pair marker only when production owns both locks and
   `prod_window.sh check` reports PRODUCTION HOLD OK.
5. Start the Sparks through the K1 production guard; require K1 START VERIFIED,
   /v1/models 327680, engine KV 2502600 on rank 0, exact saved target/mount/Engram
   hashes, native identities, async process proxy, no log refusals and >=4 GiB
   available on both nodes. Require all OPS1 warm-up completion gates, not WARN.
6. Run the six W5 greedy prompts and compare to the existing K1-post reference.
   Count exactly indices 0..5 IDENTICAL, no duplicates or divergence, even if the
   comparison program exits zero. Run the matched short/32K and B2/B3 checks
   while HOLD remains. Confirm zero failures and acceptable measured behavior.
   The operations registry must describe the new native release; prepare its
   reviewed update and require live configuration matching before release.
7. Only then remove the placeholder/HOLD with `window.sh close`. Archive the
   apply/restore receipts. If any gate fails, carry out the rollback below;
   do not close HOLD on partial success.

## Rollback to the present TP2 K2 worker

Keep HOLD. Stop only the attended apply's own work and native TP2 unit. Restore
exactly the saved TP2 unit/drop-ins and current link to fb68c52; restore the saved
Spark production env (including both original K2/Markov identities) on the
appropriate host. Remove only the newly created native override, never unrelated
drop-ins. Reload systemd and run `prod_window.sh close`, requiring PRODUCTION
HOLD OK and fb68c5291792ab90c0f468bd94010917967799f3 on the original UUID pair.
Do not use the historical prod_tp2.sh one-card rollback recipe.

On Controller run W8/restore.sh start, then check; verify the return code AND count
all six distinct `greedy K1-post vs restored: N IDENTICAL M` lines, reject any
DIVERGES, require PRODUCTION HOLD OK and CHECK: DONE. Only then run close and
require CLOSE: window closed. Reconcile the saved OPS1/K1 production configuration
and its registry. If any gate fails, HOLD stays open for attended recovery.

Do not run W8's restore helper as an apply validator while native is intended to
remain production: its `start` deliberately restarts the saved K1 production.
The trial-window runner always returns to the existing K2 production and never
executes this promotion design.
