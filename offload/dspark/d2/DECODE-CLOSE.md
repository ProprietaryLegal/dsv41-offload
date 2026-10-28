# D2 second-pass close-out handover

Prepared and rehearsed on Controller CPU only. No host access, production operation,
GPU operation or real capture read was performed by D2-CLOSE. Deepseek /
coordinator owns review, code staging, every live command and production recovery.
This implements transition 2 of [DECODE-CAPTURE-APPLY.md](DECODE-CAPTURE-APPLY.md).

The campaign artifact is
`/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/decode_close.sh` (outside git).
Reviewed D2-CLOSE-R2 SHA256:
`6a6924cebf933295289e7ea8e52113aff5c30f611dfba09164bd43efb5ea80bd`.
Original versions remain in `W12/private/close-versions/decode_close.v{1,2,3}.sh`.
R2 versions are `decode_close.r2-fix{1,2,3,4}.sh` and `decode_close.r2-final.sh`;
r2-final is the delivery. Runtime evidence goes in its mode700
`evidence-d2pass2-r1/` subtree. No production script was edited.

## Coordinator preparation

Independently review this branch's final commit and stage its code on x86host in a
NEW `~/dsv41-offload/d2-close` checkout. Retain `offload/dspark/copy_to_x86host.py`
from that revision too: the new module imports its reviewed transfer safeguards.
Do not replace any running worker checkout. On x86host, confirm `ssh controller`
authenticates noninteractively and the existing direct-link sparkhead SSH identity
in `copy_to_x86host.SSH` works; D2-CLOSE did not probe either connection. No package
install, numpy, torch or CUDA is needed for the copy/rollback tools.

The rollback checkout remains `~/dsv41-deploy-offload` on sparkhead at `8174771`,
matching `decode_prod.sh`. The script verifies that revision and tracked-file
cleanliness before invoking the standard-library rollback module. The deployment
root is `~/dsv41-deploy`; RUN is `d2pass2-r1`; immutable capture SOURCE is
`07fc7d6abc369640e1d0129ced197d4c21e892df`. Base env/start/K1 manifest hashes match
`decode_prod.sh`. Do not substitute another run or reuse a dataset destination.

Choose a NEW safe nonce (letters, digits, underscore or hyphen, at most 120
characters), record it, and use that same nonce for **all** phases. Freeze any
option-ii top20 producer separately before copying its tree; `--top20` accepts an
absolute Controller path and has no dependency on D2-OPT2 code. The generator path is
`/opt/dsv41/data/d2/gen-decode-07fc7d6-d2pass2-r1`. No first-pass state is
used. Ensure storage for captures, state, top20 and one partial destination; a
previously existing x86host destination always refuses, even if apparently complete.

## Exact phase order

On Controller, the coordinator uses one command at a time and checks its exit status.
`--dry` prints every external/local action in order and performs no phase action,
SSH, HTTP, filesystem mutation, tmux or service operation. It does not qualify
production. Rehearse all eight subcommands this way first if desired.

```bash
export D2_COORDINATOR=Deepseek/coordinator
C=/opt/dsv41/campaigns/DSV41-OFFLOAD
S=$C/W12/decode_close.sh
N=NEW_SAFE_NONCE
bash "$S" --nonce "$N" pause
bash "$S" --nonce "$N" flush
bash "$S" --nonce "$N" copy
# Or: bash "$S" --nonce "$N" --top20 /absolute/frozen/Controller/top20-tree copy
bash "$S" --nonce "$N" rollback
bash "$S" --nonce "$N" post
bash "$S" --nonce "$N" close
```

1. `pause` latches `D2-GEN2.PAUSE` and allows 10 seconds for tmux `d2-gen2`
   to end naturally. If still present, it sends `tmux send-keys -t =d2-gen2: C-c` (the colon makes it a pane target; tmux 3.6 rejects `=d2-gen2`, S33)
   to this coordinator-owned generator. Its SIGINT handler saves state and exits.
   The phase then waits at most 120 seconds for BOTH the absent session and
   `D2-GEN2-EXIT` in `D2-GEN2.out.log`. It requires the generator state directory
   and zero running/waiting production gauges. It never signals an unrelated
   process. Missing, nonfinite or nonzero
   gauges refuse. Expected: `NEXT: flush ...`. Keep PAUSE latched permanently for
   this frozen capture. No writer or tagged generator may resume.
2. `flush` requires the pause phase marker, checks idle, then runs the copy module
   on x86host with `--flush`. It atomically writes a NEW `FLUSH.json` nonce on
   sparkhead and waits up to 30 seconds for the matching `FLUSHED.json`, as in the
   smoke. Reusing an already requested/acknowledged nonce refuses. It then checks
   error ledger/count, sequence inventory, ledger hashes and payload sizes.
   Expected: `FLUSH VERIFIED NONCE`, then `NEXT: copy ...`. Error count need not be
   zero for an audit copy, but it must match both persisted error telemetry and
   the ledger. Sequence usability remains an assembly decision.
3. `copy` requires successful flush, PAUSE and absent generator tmux. The module
   handles production demand, including demand before its first inventory. On x86host it runs:

   ```bash
   cd ~/dsv41-offload/d2-close
   D2_COORDINATOR=Deepseek/coordinator python3 -m offload.dspark.d2.decode_copy \
     --source-root /home/rank0user/dsv41-offload-run/d2-decode-d2pass2-r1 \
     --nonce NEW_SAFE_NONCE --run d2pass2-r1 \
     --generator-root /opt/dsv41/data/d2/gen-decode-07fc7d6-d2pass2-r1
   # Optional: --top20-root /absolute/frozen/Controller/top20-tree
   ```

   Destination is NEW mode700 `/opt/dsv41/x86host-data/d2/decode-d2pass2-r1`.
   Subtrees are `capture/`, `generator/`, optional `top20/`. Every source and
   destination is snapshotted/verified before and after transfer. Symlinks,
   unexpected files/directories, new tagged sequences, changed ledger, size drift
   and same-size file writes invalidate the copy. Spark metadata is read using
   O_DIRECT too. Payloads use direct-link O_DIRECT, decimal 1,000,000,000 B/s cap,
   source 4 GiB MemAvailable floor, x86host 16 GiB floor, production idle checks
   before transfers and every 0.5 seconds during transfers and inventory reads.
   Valid nonzero demand stops the owned transfer immediately when detected.
   No source reads or copies run while paused. After 30 seconds of continuous
   idle, every established source inventory is revalidated before resuming.
   The interrupted file restarts at byte zero; completed files still undergo
   final snapshot and destination verification. A cumulative wait cap defaults
   to four hours for the whole invocation, including idle qualification, and can
   be set through the module's `--pause-wait-cap-seconds`. The success receipt
   records the cap, idle interval, total wait and pause events. Exhaustion leaves
   no receipt. Missing/malformed metrics, HTTP/SSH errors, memory guards and all
   snapshot/nonce/ledger mismatches remain hard aborts. Only owned SSH/dd
   children are stopped on guard failure. Destination is hashed from the stream
   and re-read independently by the transfer helper and again by the snapshot
   orchestrator. Expected: `COPY VERIFIED .../COPY_RECEIPT.json`.
   A failed tree has **no** completed receipt; do not treat its partial files as
   certified or retry into it after a hard abort. Demand-only pauses resume
   within the same invocation. After a hard abort, preserve the partial tree for
   inspection and choose a reviewed new destination/run argument for a retry.
4. Every rollback mode checks PAUSE/absent generator and runs a read-only
   preflight BEFORE opening a window or stopping production. It checks tool
   revision/cleanliness, transaction identity and reconstructed hashes, review,
   backup hashes, K1 manifest and the current env/start/record against exactly
   the states that mode accepts. A mismatch leaves production untouched by this
   invocation. Ordinary rollback accepts base/derivative files, including partial
   rollback states. It then invalidates old post/rollback evidence, opens the
   normal `window.sh open` drain/HOLD protocol, and runs `./stop.sh`. In the
   reviewed tool checkout ordinary rollback invokes:

   ```bash
   python3 -m offload.dspark.d2.decode_apply \
     --root ~/dsv41-deploy --run d2pass2-r1 --rollback
   ```

   It checks original env/start/manifest hashes, runs `manage.py verify --profile
   327680`, runs unchanged `./start.sh` and requires `K1 START VERIFIED` in this
   invocation's fresh `start.log`. Every failed command stops the phase; no hidden
   fallback or close follows it. Expected: `NEXT: post ...`. Recovery remains
   available if flush/copy failed: there is deliberately no copy-success gate
   preventing restoration of production. Missing copy evidence blocks dataset
   use, rather than blocking service recovery.
5. `post` requires successful rollback. It checks legal-v2a and 327680 through
   `/v1/models`, rank-0 `GPU KV cache size` 2,502,600 (same authoritative log check
   as `decode_prod.sh`), and both ranks' actual inspect env/mounts have no decode
   capture. Both ranks must show at least 3 GiB MemAvailable. It runs W5's fixed
   greedy requests, W5's comparison, and an additional exact six-row/index/text
   check so `zip` truncation cannot pass. Then x86host's existing
   `prod_window.sh check` must exit zero and print `PRODUCTION HOLD OK`.
   Expected: `POST VERIFIED: greedy 6/6 IDENTICAL ...`. All raw outputs remain
   private; they are not printed in reports.
6. `close` requires the same nonce's post marker, invokes `window.sh close`, and
   records success. Expected: `CLOSE: original K1-327680 restored ...`. Leave PAUSE
   latched and the frozen capture/source files inert. Do not restart workers or
   delete captured data as part of close-out.

Two explicit recovery modes cover different K1 fallback states. Neither is
selected automatically; both prove the exact state before opening/stopping.

- `rollback-after-k1-fallback`: use only when K1 fell back while the capture
  derivative was applied. Current env/start must be the pre-K1 originals and
  the record must match the transaction's DERIVATIVE record. This invokes
  `decode_apply --rollback-after-k1-fallback`, verifies restored K1 and starts it.
- `rollback-after-close-fallback`: use when the close's own K1 start fell back
  AFTER ordinary rollback restored the BASE record. Current env/start must be
  pre-K1 originals and the record must be the BASE record. Keep HOLD; production
  is serving the previous configuration. After preflight/window/stop, this mode
  repeats its preflight, restores `logs/d2-decode-d2pass2-r1/{dsv41.env,start.sh}`
  using `cp -p`, checks BASE_ENV / BASE_START / K1 manifest with `sha256sum -c`,
  runs `manage.py verify --profile 327680`, and starts K1. It requires
  `K1 START VERIFIED`, then the same separate `post` and `close` phases.

For a fallback during the close's own start, after inspecting `start.log`:

```bash
bash "$S" --nonce "$N" rollback-after-close-fallback
bash "$S" --nonce "$N" post
bash "$S" --nonce "$N" close
```

The wrong mode refuses before touching production. Keep HOLD until post passes.
Operational failures print `STOP` with phase, tracked step and recovery guidance.
Before stop, production is untouched by that phase; fix and rerun, or close the
window. After stop and before verified start, production may be DOWN: rerun
`rollback` or use `W8/restore.sh start|check|close` after inspecting the state.
If start fails, read the fresh `evidence-d2pass2-r1/start.log`.
`K1 REFUSED; previous configuration restarted` identifies serving pre-K1 fallback;
`K1 FALLBACK FAILED` means DOWN: keep HOLD and inspect `logs/k1-guard-*.log`.

## Failure table

| Failure | Result and coordinator action |
|---|---|
| Generator still running, missing state, busy/malformed metrics | Stop before flush/copy; leave PAUSE latched, inspect cancellation/save and production activity. Never kill an unrelated process. |
| Nonce reused, flush timeout, count mismatch | Stop; inspect FLUSH/FLUSHED/ERROR_COUNT/ERRORS, drain own tagged work and issue a genuinely new nonce. Do not certify a mismatched snapshot. |
| Ledger hash, payload size, source stat/inventory changes | Copy refuses; freeze all writers, inspect the capture and flush again with a new nonce. Never resume into the certified tree. |
| Production traffic during copy | Owned transfer pauses; after 30 s continuous idle and snapshot revalidation, restart the interrupted file at byte zero. Four-hour cumulative wait cap by default; cap exhaustion aborts without a receipt. |
| Low memory, unavailable/malformed gauges, transport failure during copy | Hard abort; partial tree is invalid, no completion receipt. Inspect before any fresh copy. |
| Destination already exists, stream hash or destination re-read mismatch | Refuse; preserve evidence, inspect storage/source and use a NEW destination. No overwrite/resume/reuse shortcut. |
| Tool revision/cleanliness, transaction, saved material or current mode-state differs | Read-only preflight refuses BEFORE window-open/stop; production untouched by this invocation. Inspect verified recovery material. |
| Start fails or lacks `K1 START VERIFIED` | HOLD stays; inspect fresh start log. Fallback during close start (BASE record): `rollback-after-close-fallback`, then post/close. Fallback before close (DERIVATIVE record): `rollback-after-k1-fallback`, then post/close. `K1 FALLBACK FAILED`: DOWN; retain HOLD and read logs/k1-guard-*.log. |
| Models/KV/env/mount/memory/greedy/x86host HOLD check fails | No post marker; retain HOLD, diagnose the failed check, rerun post only after correction. Do not close. |
| SSH disconnect | No restoration claim: inspect actual state and evidence before continuing. |

## Dataset evidence and limits

`COPY_RECEIPT.json` schema `d2.decode-copy.v1` must be present, parse completely,
match the requested nonce and have `frozen=true`. It records source paths, optional
top20 path, exact before/after-verified source inventories (sizes, inode,
mtime/ctime, metadata hashes), and each file's stream hash, re-read-verified hash,
size, transfer provenance and rate cap, plus pause policy, wait cap, total wait
and pause events. Match every copied file to its receipt hash and
size before dataset construction. Treat the receipt itself as private provenance;
it is not a substitute for an approved source/smoke/context receipt.

Retain `FLUSH.json`, `FLUSHED.json`, `ERROR_COUNT.json`, `ERRORS.jsonl` when present,
every `CAPTURE.json`, `ledger.jsonl`, `taps.bin`, `ids.bin`, frozen generator state
and optional normalized top20 tree. Require exact sequence membership, nonce,
ledger hashes and sizes again at assembly. Assembly must check newest-writer
per-row hashes, replacement/superseded segments, prompt first-wins rules, required
interval coverage and committed token IDs. Named writer errors quarantine those
sequences; global/unattributed errors refuse all. This audit copy intentionally
preserves errors rather than claiming those sequences are usable. No `DONE.json`
is required or synthesized, no prefill taps are substituted, and no G1/training
qualification is claimed here. Keep this capture frozen after copying; any later
tagged writes invalidate the receipt for that later snapshot.

ESTIMATED: 1,000,000 rows × 30,720-byte taps = 30.720 GB; four-byte IDs add 0.004 GB.
At the 1.000 GB/s cap the nominal payload transfer is 30.724 seconds. This excludes
ledger/header/state/top20 bytes, SSH/startup, remote/destination storage and
independent disk verification. For example, each additional 30.724-GB disk pass at
an assumed 0.5 GB/s adds about 61.4 seconds; the two independent re-reads add
about 122.9 seconds at that assumed speed. These are arithmetic estimates, not
measured bandwidth or an end-to-end duration promise. The live receipt records
per-file transfer/verification timing and whole-copy elapsed time including
snapshot checks and both re-reads when the coordinator executes it.

## CPU rehearsal

From this worktree, with scratch on the campaign disk:

```bash
PYTHONPATH=.:offload/dspark CUDA_VISIBLE_DEVICES='' python3 -m pytest -q \
  offload/dspark/d2/tests \
  --basetemp=/opt/dsv41/campaigns/DSV41-OFFLOAD/scratch/D2-CLOSE-R2/pytest
```

Copy tests use synthetic files and fake hosts, forbid live subprocess/network
calls, exercise the exact remote scan/flush code and the reused source guards.
Close tests include S31 fake-host rehearsals with the real decode_apply module.
They prove corruption refusal before window/stop, interrupted-stop retry, both
fallback recoveries and the generator SIGINT/exit-marker handshake.
Close tests require the campaign script to be present at its designated path;
they run every `--dry` phase with fatal SSH/HTTP/tmux/filesystem command stubs,
validate remote command order, Python payload quoting, nonce markers, idle
refusals and mount refusals. S31 also tests synthetic CPU-writer trees with real
local O_DIRECT reads and
real dataset assembly. Tests do not qualify actual SSH authentication, production
filesystem behavior, live log formatting, service restart, live greedy
results or hardware headroom. Those remain coordinator integration checks.
