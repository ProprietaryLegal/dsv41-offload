Status 2026-10-23: Async process proxy promoted in W5 (main 3c1dc6b / prod-async-20261023); TP2 worker promotion remains pending under the evening plan. Original package instructions below retain their baseline. See [live roadmap](../../../documents/plans/roadmap.md) and [measured record](../../results/COMPARISON.md); campaign review decisions are in `LANES.md` / lane reports outside the repo.

# Promotion package: TP2 worker + async rounds through the process proxy (R3-PROMOTE, 2026-10-23)

**Partially applied:** the async process proxy is in production since Window 5, main 3c1dc6b, tag `prod-async-20261023`. The TP2 production-worker switch is still pending in the evening plan. See [Window 5 production record](../../results/COMPARISON.md#production-switch-async-on-the-one-card-worker-13491358z).

**Historical package baseline:** this package was originally written against main 6aa71ca before S12 review and Window 5. The original application/rehearsal steps below describe that baseline; do not reapply the async patches blindly to current main. TP2 promotion follows the current reviewed coordinator switch and retains separate attribution from K1.

What window 4 measured (COMPARISON.md "Window 4"): TP2 worker + process proxy + nap, CUDA_MODULE_LOADING lazy:
−3.9 to −6.3 % per decode round (corrected 2026-10-23; COMPARISON.md Window 4) against the production configuration (one-card worker, synchronous rounds), TTFT at
parity, 0 watchdog fires. R3 fixed what S11 required before production (P-1 late-proxy row corruption, P-2 start
fallback, P-3 stall detection, P-4 respawn, P-6 shutdown); evidence in R3-PROMOTE.report.md.

## What changes

| where | original production baseline (main 6aa71ca, tag prod-d-20261023) | package target (async applied W5; TP2 pending) |
|---|---|---|
| `dsv41.env` OFFLOAD_DSPARK_ENV | configuration D, synchronous rounds | the same + `ASYNC=1 ASYNC_PROXY=process ASYNC_SPIN_US=50 ASYNC_NAP_US=20 ASYNC_RESPAWNS=3 ASYNC_RESPAWN_RESET_S=3600` (all `VLLM_DSPARK_REMOTE_` prefixed); the sync line stays, commented, as the rollback (`dsv41.env.patch`) |
| `start.sh` | — | one warm-up request after the serve answers, when ASYNC=1 (`start.sh.patch`, `offload/launch/warmup_serve.sh`) |
| remote_dspark package (staged by `stage_overlay.sh` from the checkout) | 6aa71ca's | the merged R3 package |
| x86host worker | `dsv41-dspark-worker` (one card, GPU-c4e7a371), `current -> d275f6b` | `dsv41-dspark-worker-tp2-prod` (pair c4e7a371 + 7744b44b), `current -> <merged commit>`, prebuilt p2p (`p2p-prebuilt.txt`) |
| x86host GPU ownership | 7744b44b = development card (lanes by flock), pair windows by `pair_window.sh` | 7744b44b = production rank 1: the TP2 unit's own worker process holds both GPU locks (`worker/run/gpu_locked.sh`); `pair_window.sh` refuses; lanes only in `prod_window.sh` training / pair windows (plan `documents/plans/2026-10-23-dual3090-plan.md` §"GPU ownership after the TP2 promotion") |
| unchanged | model, image, KV 6 GiB/rank, Engram (B), CUDA_MODULE_LOADING lazy, hostflag.c (sha256 10490b58…, aarch64 pin 0ada6aee…), Spark libofl pin 4daef6c2…, x86host libofl pin 9f11b6e8…, venv `~/dsv41-offload/o2/venv` (same torch 2.10.0+cu126 and exllamav3 ext hashes as lane E's measured venv) | |

The patches apply cleanly to main 6aa71ca (`git apply --check`, checked 2026-10-23).

GPU ownership after promotion (S12 F1). GPU-7744b44b leaves the development pool. Lanes get it only in coordinator
training or pair windows (`offload/launch/prod_window.sh`). The protocol is in `documents/plans/2026-10-23-dual3090-plan.md`
§"GPU ownership after the TP2 promotion", with the swap commands both ways and their checks.
- The hold is the production unit's own worker process. Its ExecStart runs `worker/run/gpu_locked.sh`, which takes
  `flock -n` on `gpu-c4e7a371.lock` and `gpu-7744b44b.lock` and then execs the worker. The hold is never a tmux
  session or a deleted lock file.
- `pair_window.sh` refuses open, hand, hold and close while the TP2 production unit is enabled or running.
- R3's first version of this README said `pair_window.sh close` "restarts the one-card unit". That was wrong (S12 N3):
  it stopped at the c4e7a371-in-use check after dropping the hold. It now refuses.

## Window 5 (GB10, supervised, before promotion)

Setup:
- Experiment launcher on :8001 with the merged code:
  `offload/launch/start_exp.sh offload/launch/configs/D-kv6-async-process-nap-lazy.env`.
- The TP2 worker in a pair window, as in W4-E1p.
- Every check below must pass. Any failure is a stop, with the log to R3.

Shorthands used below:
```
L()  { docker logs dsv41-serve 2>&1; }                              # the serve log on sparkhead
J()  { docker exec dsv41-serve cat /tmp/dsv41-remote-dspark.json; }  # the stats file (light read)
K()  { docker exec dsv41-serve kill -"$1" "$2" || docker exec dsv41-serve python3 -c "import os; os.kill($2, $1)"; }
                                                                    # the second form if the image has no /bin/kill
```

**When the stats file is current (S12 F3).**
- Who writes it:
  - the proxy, only at each 1,024-round mark (`LOG_EVERY=1024`), at the moment it logs
    `(remote-dspark proxy pid=<pid>) ... remote DSpark capacity: N rounds, ok ...`;
  - the engine, only at a proxy loss and at close.
- What that means:
  - there is no file before the first 1,024 decode rounds; start-up and the warm-up never reach that mark;
  - right after a drill, the file still shows the moment of loss (the old pid, `async_dead` set) until the new proxy's
    next 1,024-round line. On d103ba3, the code window 5 runs, even `async_respawn` there is the dead proxy's stale copy
    (`live`, the old budget); 23c89ec makes the engine write its own current values (`waiting`, then `exhausted` if
    the budget is spent). The rule below holds for both.
- **Rule: read the stats only after a `remote DSpark capacity: N rounds` line from the pid under test, logged after
  the event.** Then confirm in the file that `async_proxy_process.pid` is that pid. After a drill, also confirm
  `rounds` > `async_proxy_losses[-1].round`.
- **Count rounds, not tokens.** A round emits 1 + accepted tokens per request, so 1,100 decode tokens can be far fewer
  than 1,024 rounds. At concurrency 1, keep one generation running (max_tokens 6,000) until the line appears.

1. **Startup**, rank 0 (`L | grep "remote DSpark"`).
   - Present: `async mode on (host-flag hand-off self-test passed`, `remote DSpark proxy process running`,
     `through proxy process pid`.
   - Absent: `async mode off`, `did not start`, `async proxy process did not start`.
   - Note the first proxy pid:
     `pid1=$(L | grep -oE 'through proxy process pid [0-9]+' | tail -1 | grep -oE '[0-9]+$')`.
2. **Warm-up and first requests** (S12 9). Production start.sh does the warm-up itself; here run it by hand right after
   the serve answers: `offload/launch/warmup_serve.sh 8001 <served name>`.
   - It prints `warm-up request: N s (prompt_tokens:... completion_tokens:32)`.
   - Then the first real request at concurrency 1: its TTFT must be at the W4-E1p level (no first-request penalty).
   - Then the first burst of 3 concurrent requests: each TTFT within the noise of later bursts (R3 open question 6).
3. **The compact benchmark set against W4-E1p.**
   - Decode per round and TTFT within run-to-run noise.
   - **Acceptance parity** (S12 2): accepted draft tokens per draft token over the set, from the `vllm:spec_decode_*`
     counters on `:8001/metrics` (deltas across the set). Must be within the noise of W4-E1p's. Wrong drafts would not
     show in milliseconds per round.
4. **Greedy output check** (S12 3).
   - Run 4 to 8 fixed prompts at temperature 0 (max_tokens 256). Compare with the same prompts on W4-E1p's recorded
     outputs or on synchronous D.
   - Expected: identical text. Any divergence is a stop: report the prompt, the first differing token index and both
     outputs. Drafts must never change outputs.
5. **Stats, after check 3** (at least 1,024 decode rounds since start; the old check 2). `J` once the pid1 line
   `remote DSpark capacity: N rounds` is in the log. Expected values:
   - `async` true;
   - `async_proxy` {mode process, pid = pid1, generation 0, region_mlock ok};
   - `async_proxy_process` {pid = pid1, generation 0, start_round 1};
   - `proxy_torch_loaded` false; `async_dead` null; `async_respawns` 0;
   - `async_respawn` {state live, limit 3, left 3, failed_starts 0, reset_s 3600};
   - `async_stall_detection` {heartbeat_limit_ms 400, stall_fires 3};
   - after the benchmark (S12 8): `async_watchdog.fires` 0, and none of the counters `watchdog_preempted`,
     `fallback_preempted`, `proxy_lost_*`, `fallback_proxy_dead`;
   - `latency.rank0_submit_us` p50 about 0.3 to 0.6 ms (S11 P-5).

   Record for check 8: `rounds_ok`, and the region fds of the rank-0 engine (the proxy's parent):
   ```
   eng=$(docker exec dsv41-serve ps -o ppid= -p $pid1 | tr -d ' ')
   docker exec dsv41-serve sh -c "ls -l /proc/$eng/fd | grep -c memfd:dsv41-remote-dspark"
   ```
6. **Kill drill under concurrency 3** (S12 4; spends respawn 1).
   - Load: one long generation (max_tokens 6,000) plus two other requests. Record accepted tokens per round over about
     200 rounds before the kill (`:8001/metrics` deltas every 2 s).
   - Kill: `K 9 $pid1`.
   - Expected log lines, in order:
     - `async proxy process <pid1> lost (proxy process exited with -9): target-only rounds ... (3 respawn(s) left)`;
     - about 1 s later, `starting proxy process <pid2> (respawn 1, session 0x..., rounds from N)`;
     - `(remote-dspark proxy pid=<pid2>) ... proxy process running`;
     - `async proxy process <pid2> running (respawn 1, ...): drafting resumes at round N`;
     - the new proxy's `worker ... ready at o1rdma://...`.
   - **pid2 comes from the log, never from the stats file** (still the old pid):
     `pid2=$(L | grep -oE 'starting proxy process [0-9]+' | tail -1 | grep -oE '[0-9]+$')`.
   - All three requests complete, with valid output.
   - Accepted tokens per round over about 200 rounds after the respawn: back to the pre-kill level within about 128
     positions per request (the partial windows refill: R3 open question 3).
   - **Worker side** (x86host, S12 5; the TP2 worker's log and status file: lane E's unit `~/dsv41-offload/e/logs/worker.log`
     and `~/dsv41-offload/e/run/worker.status.json`, or the production paths under `~/dsv41-offload/prod/`):
     - `ofl server: connection <n> (...) ends: ...` for the dead client's connection, then
       `HELLO new session 0x... from sparkhead tp0`;
     - no restart: the same `pid`, `uptime_s` still growing, `new_sessions` +1.
     - This is the first test of the RDMA dead-client path; S12 covered only the ofl TCP backend on CPU.
   - **Stats**, per the rule above (the pid2 line `remote DSpark capacity: N rounds` after the kill):
     - `async_respawns` 1, `async_respawn.left` 2;
     - `async_proxy.pid` = `async_proxy_process.pid` = pid2, `async_proxy.generation` 1;
     - `async_proxy_losses[0].kind` exit;
     - counters `proxy_lost_exit` 1, `fallback_proxy_dead` > 0, `rounds_ok` larger than in check 5. Counters carry
       across a proxy only as of its last stats write (S12 N1).
7. **Stall drill** (spends respawn 2). During a generation: `K 19 $pid2` (SIGSTOP).
   - Expected: `lost (proxy process stalled: 3 watchdog fires with no round finished ...)` or
     `lost (proxy process heartbeat ... ms old (limit 400 ms ...)`, within about 0.6 s of decoding.
   - Then respawn 2 as in check 6 (the engine SIGKILLs the stopped process):
     `pid3=$(L | grep -oE 'starting proxy process [0-9]+' | tail -1 | grep -oE '[0-9]+$')`.
   - Worker side as in check 6.
   - Stats, per the rule: `async_respawns` 2, `left` 1, and `proxy_lost_fires` or `proxy_lost_heartbeat` 1.
8. **Process hygiene after both drills** (S12 6).
   - `docker exec dsv41-serve ps -eo stat,pid,cmd` shows no `T` and no `Z` entry: the stopped proxy is gone.
   - The rank-0 engine holds the same number of `memfd:dsv41-remote-dspark` fds as in check 5:
     `eng` = `ps -o ppid= -p $pid3`, which must be the same engine pid.
9. **Affinity.** `docker exec dsv41-serve taskset -p $pid3` shows the wide affinity (W4: 0-19).
10. **Idle and resume** (S12 7). No request for at least 10 min, then one request.
    - It completes.
    - No `lost (` line and no `proxy_lost_heartbeat` since the idle began:
      `docker logs --since <idle start> dsv41-serve 2>&1 | grep -E "lost \(|proxy_lost"` prints nothing.
    - This exercises the "the engine did not run" rule on GB10.
11. **Rollback rehearsal** (S12 11). It backs the "< 10 min" claim.
    - **x86host, the one-card flip:** `pair_window.sh close` (stops the TP2 unit, starts the one-card worker). Time from
      the command to the one-card `worker ready in` line plus `HELLO new session ... from sparkhead`. Expected
      about 70 s or less.
    - **Spark, the synchronous restart:** restart the :8001 serve with the synchronous D config. Time from stop to the
      serve answering.
    - Then the post-rollback checks below ("After a rollback"), on :8001.
12. **For the production window, not window 5** (S12 12, after F1): production step 5 runs
    `prod_window.sh check`. It must print `PRODUCTION HOLD OK`:
    - `flock -n ~/dsv41-offload/locks/gpu-7744b44b.lock true` fails;
    - `pair_window.sh open X` refuses;
    - `systemctl --user is-enabled` shows `dsv41-dspark-worker` disabled and the TP2 production unit enabled.

The drills spend 2 of the 3 respawns. The budget returns after 3600 s of service, or at the next serve start.

## Before the production window (no service impact)

On Controller (coordinator's usual promote flow, as promote-d):
```
git switch -c promote-async main                    # main = 6aa71ca
git merge <reviewed offload-x86host commit>           # carries offload/r3-promote; stops on ONE conflict, a comment:
#   CONFLICT (content): Merge conflict in offload/dspark/spark/stage_overlay.sh
# Resolve the comment-only conflict in stage_overlay.sh by taking offload-x86host's side (S12 F2). main's only change to
# that file is the comment and the pinned OFL_SO_SRC line, and offload-x86host's side has that line too:
git checkout --theirs offload/dspark/spark/stage_overlay.sh
grep -cF 'HOSTFLAG_SO_SHA=0ada6aee896842b2196734ec128749edfff52b0b54685fcca27501484f8906c4' offload/dspark/spark/stage_overlay.sh  # 1
grep -cF 'OFL_SO_SRC:-$HOME/dsv41-offload-prod/pinned/libofl.so' offload/dspark/spark/stage_overlay.sh                           # 1
# -F (fixed strings): in Claude Code shells on Controller `grep` may be ugrep, which reads a mid-pattern $ as an anchor
git add offload/dspark/spark/stage_overlay.sh && git commit --no-edit
git apply offload/promote/async-tp2/dsv41.env.patch offload/promote/async-tp2/start.sh.patch
bash offload/dspark/spark/test_stage_overlay.sh        # stage_overlay.sh hostflag tests: 7/7 passed
(cd offload/dspark/spark && CUDA_VISIBLE_DEVICES="" python3 -m pytest tests -q)   # 223 passed, 1 skipped
git commit ...; then bring it to sparkhead ~/dsv41-deploy as for promote-d
```
Rehearsed 2026-10-23 in a scratch clone: main 6aa71ca, merging offload-x86host f79f8d4 with this branch's F1 commits on top.
- `git merge` stopped on that one conflict and nothing else. Resolved as above.
- `HOSTFLAG_SO_SHA=0ada6aee…` is present, and both patches apply.
- `test_stage_overlay.sh` 7/7; CPU suite 222 passed, 1 skipped at the time; 223 passed, 1 skipped with 23c89ec (its new
  test), as S12's rehearsal on offload-x86host 6b99476 + 057f937 confirmed.
On sparkhead (head node), the async staging requires the pinned hostflag build:
```
sha256sum ~/dsv41-offload-prod/pinned/libdsv41_hostflag.so    # must be 0ada6aee896842b2196734ec128749edfff52b0b54685fcca27501484f8906c4
# if missing: sha256sum ~/dsv41-offload-run/ofl/libdsv41_hostflag.so (same value), then
# install -m 644 ~/dsv41-offload-run/ofl/libdsv41_hostflag.so ~/dsv41-offload-prod/pinned/
```
On x86host (worker layout; nothing enabled):
```
# Controller: offload/dspark/worker/run/prod_tp2.sh stage <merged commit>
cd ~/dsv41-offload/prod/dspark-worker/<commit>/offload/dspark
# the prebuilt p2p extension: either install R3's build (p2p-prebuilt.txt) ...
S=~/dsv41-offload/r3/prodtest/dspark-worker/p2p/08e1903b0a43feff; D=~/dsv41-offload/prod/dspark-worker/p2p/08e1903b0a43feff
(cd $S && sha256sum -c dspark_p2p.so.sha256) && mkdir -p $D && install -m 755 $S/dspark_p2p.so $D/ \
  && install -m 644 $S/dspark_p2p.so.sha256 $S/BUILD.txt $D/ && (cd $D && sha256sum -c dspark_p2p.so.sha256)
# ... or build one here: bash worker/run/prod_tp2.sh build-p2p <commit>   (record its sha256 in the window log)
bash worker/run/prod_tp2.sh install-unit
bash worker/run/prod_tp2.sh check <commit>          # CHECK OK except "current -> d275f6b" (flipped in the window);
                                                    # includes "production hold script ..." and
                                                    # "unit holds both GPU locks (gpu_locked.sh)"
```
The reviewed window scripts. pair_window.sh refuses after promotion and runs pair windows under the hold; prod_window.sh
runs the windows after promotion. Copy them from Controller's merged checkout, then compare on x86host:
```
# Controller, in the merged checkout:
scp offload/launch/pair_window.sh offload/launch/prod_window.sh x86host:dsv41-offload/
sha256sum offload/launch/pair_window.sh offload/launch/prod_window.sh
# x86host: the same two sums (a stale ~/dsv41-offload/pair_window.sh has no post-promotion guard)
sha256sum ~/dsv41-offload/pair_window.sh ~/dsv41-offload/prod_window.sh
```
Tell every lane that uses GPU-7744b44b (N2, N3, D2-DATA, D2-TRAIN, reviewers) before the window. From production step 3
on, the card is production rank 1:
- stop GPU work on it before step 2;
- afterwards use it only inside a training or pair window that names the lane, with `flock -n` (plan §"GPU ownership
  after the TP2 promotion").

## The production window (about 10 min of serve downtime)

1. sparkhead: `cd ~/dsv41-deploy-offload && offload/launch/window.sh open "promote D-async + TP2"`.
2. x86host, pair window (`offload/launch/pair_window.sh open coordinator hold`; nothing else on GPU-7744b44b). Qualify
   the promoted p2p binary on the pair, with the prebuilt path the unit uses:
   ```
   cd ~/dsv41-offload/prod/dspark-worker/<commit>/offload/dspark
   CUDA_VISIBLE_DEVICES=GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93,GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc \
   DSPARK_P2P_PREBUILT=$HOME/dsv41-offload/prod/dspark-worker/p2p \
     ~/dsv41-offload/o2/venv/bin/python -m pytest -q -s worker/tests/test_p2p.py "worker/tests/test_tp2.py::test_pair_equals_emulation"
   ```
   (the one-card worker is stopped by the pair window; this is lane E's pair suite step 1 against the pinned build).
3. x86host: hand both cards from the pair window to the TP2 production unit.
   - **The hold.** The unit's worker holds both GPU locks itself, for its whole life. `worker/run/gpu_locked.sh` takes
     `flock -n` on `gpu-c4e7a371.lock` and `gpu-7744b44b.lock`, then execs the worker. Because the unit is enabled,
     this happens from boot too. No tmux session and no lock-file change is involved.
   - **Lanes cannot share the cards.** While the worker runs, a lane's flock on either card fails. If a lane holds a
     lock, the worker refuses to start: the worker log gets `gpu_locked: ... is held by another process ... refusing`,
     and the unit retries every 10 s.
   ```
   ln -sfn <commit> ~/dsv41-offload/prod/dspark-worker/current
   systemctl --user disable dsv41-dspark-worker            # stopped already by the pair window
   systemctl --user enable dsv41-dspark-worker-tp2-prod    # enable only; from here pair_window.sh refuses
   fuser -v ~/dsv41-offload/locks/gpu-*.lock               # only dsv41-pairhold's flock/sleep: no lane waiting
   tmux kill-session -t dsv41-pairhold; rm -f ~/dsv41-offload/PAIR_OK ~/dsv41-offload/locks/.pairhold-acquired
   bash ~/dsv41-offload/prod_window.sh close               # waits for both locks, starts the TP2 unit, ready line,
                                                           # tp 2, then its production checks
   ```
   - `close` must end with `PRODUCTION HOLD OK` (window 4: ready 68.6 s after start). Its check reads the holder from
     the TP2 main pid's own `/proc/<pid>/fdinfo` lock lines and fails on any process waiting in flock on either lock
     (`/proc/locks` `->` lines). lslocks is not used: util-linux 2.39 on x86host names a dead flock pid (S12 R1, R2).
   - If a lane takes a lock in between, `close` names the holder. Stop that lane, then run `close` again.
   - **If the TP2 unit fails to start for any other reason** (the worker log shows a p2p, venv or model error, not
     `gpu_locked: ... refusing`), start the one-card production worker at once:
     `systemctl --user start dsv41-dspark-worker`. `Conflicts=` ends the TP2 unit's retry loop, and the Spark drafts
     again within seconds. Then fix the cause, or roll back (below). The TP2 unit stays enabled, so a later
     `prod_window.sh close` (or a reboot) brings TP2 back.
   - From here GPU-7744b44b is production rank 1. Never run `pair_window.sh close` after promotion: it refuses, and
     `prod_window.sh` replaces it.
4. sparkhead: `cd ~/dsv41-deploy && ./stop.sh && ./start.sh` (stages the R3 package and the hostflag pin; lazy
   start about 6 min; the last start.sh line is `warm-up request: N s (prompt_tokens:... completion_tokens:32)`).
5. Checks on :8000, in this order:
   - Window 5 check 1 (startup lines);
   - the start.sh log's `warm-up request: N s (...completion_tokens:32)` line;
   - one short and one 32K request (TTFT at the W4 level, no first-request penalty);
   - check 9 (affinity, on pid1);
   - check 5's stats, only once the log has the proxy's first `remote DSpark capacity: 1024 rounds` line (the stats-file
     rule in Window 5). Then on x86host, `bash ~/dsv41-offload/prod_window.sh check` must print `PRODUCTION HOLD OK`. Each line:
   - `systemctl --user is-enabled`: `dsv41-dspark-worker` disabled, `dsv41-dspark-worker-tp2-prod` enabled; the TP2
     unit active, the one-card unit inactive;
   - `flock -n ~/dsv41-offload/locks/gpu-7744b44b.lock true` FAILS, and the same for c4e7a371. The TP2 unit's main
     pid holds both: its own `/proc/<pid>/fdinfo` has a `lock:` line for each file's dev:inode (lslocks is not used,
     S12 R1);
   - no process waits in flock on either lock (`/proc/locks` `->` lines). A waiter would take the card in the unit's
     10 s restart gap (S12 R2);
   - `pair_window.sh open X` refuses and writes no PAIR_OK;
   - no TRAIN_OK and no PAIR_OK;
   - the status file has `tp` 2. The drills (window 5 items 4 and 5) are not repeated in production.
6. `offload/launch/window.sh close` (refuses unless :8000 serves /srv/dsv41/legal-v2a).

## Rollback (under 10 min; x86host and the Sparks in parallel)

* x86host, TP2 -> one card (about 70 s; the serve runs target-only rounds meanwhile and reconnects by itself):
  ```
  rm -f ~/dsv41-offload/d2/TRAIN_OK ~/dsv41-offload/PAIR_OK        # if a window is open: its lane stops within 60 s
  systemctl --user disable --now dsv41-dspark-worker-tp2-prod
  ln -sfn d275f6b ~/dsv41-offload/prod/dspark-worker/current
  systemctl --user enable dsv41-dspark-worker && systemctl --user restart dsv41-dspark-worker
  ```
  **The 7744b44b hold after a rollback.** `disable --now` stops the TP2 worker, and its exit releases both GPU locks: the
  hold ends with it, and nothing else holds them.
  - 7744b44b is the development card again (plan §"GPU ownership protocol", the pre-promotion rules).
  - `pair_window.sh` works again, because the TP2 unit is disabled; `prod_window.sh` refuses ("not promoted").
  - Tell the lanes.
  - `restart`, not `enable --now`: in a training window the one-card unit already runs the merged code, and only a
    restart loads d275f6b.
* Sparks, async -> synchronous rounds with the R3 package kept: in `~/dsv41-deploy/dsv41.env` put `#` before the async
  `OFFLOAD_DSPARK_ENV=` line and remove it from the line below, then `./stop.sh && ./start.sh` (about 7 min). The R3
  package then runs synchronous rounds (its two-part STEP send is in that path too: CPU suite, ofl o1tcp test and the
  x86host RDMA loopback probe all exact).
* Sparks, everything back to the exact production code: `cd ~/dsv41-deploy && git checkout prod-d-20261023 && ./stop.sh
  && ./start.sh` (detached at the tag: 6aa71ca's dsv41.env, start.sh and package; about 7 min); move main back with a
  revert commit afterwards.
* Full rollback = the x86host step plus either Spark step, run at the same time: under 10 min, bounded by the serve start.

**After a rollback** (S12 11; also window 5 check 11), every one of these must hold:
- **Spark:**
  - the new serve's `docker logs dsv41-serve 2>&1 | grep "remote DSpark"` has no `async mode on` line (synchronous
    rounds);
  - the serve answers `/v1/models`.
- **x86host:**
  - `systemctl --user is-active dsv41-dspark-worker` is active and `is-enabled` shows enabled;
  - `dsv41-dspark-worker-tp2-prod` is inactive and disabled;
  - `readlink ~/dsv41-offload/prod/dspark-worker/current` is d275f6b;
  - `~/dsv41-offload/prod/logs/dspark-worker.log` has a new `worker ready in` line (one card, no `TP=2`), followed by
    `HELLO new session ... from sparkhead`;
  - the status file has no `tp` key.
- **One request with usable drafts:** the `vllm:spec_decode_*` accepted-tokens counter on `:8000/metrics` grows over
  one 256-token request.
- **The 7744b44b hold is gone:**
  - `flock -n ~/dsv41-offload/locks/gpu-7744b44b.lock true` now succeeds, and so does c4e7a371's: the hold ended with
    the TP2 worker;
  - `prod_window.sh status` shows both locks free.
  - 7744b44b is the development card again (plan §"GPU ownership protocol"), and `pair_window.sh` works again. Tell the
    lanes.

## Files

| file | what |
|---|---|
| `dsv41.env.patch` | the D-async OFFLOAD_DSPARK_ENV line + the commented sync line, against main 6aa71ca |
| `start.sh.patch` | the warm-up hook, against main 6aa71ca |
| `../../launch/warmup_serve.sh` | the warm-up request (~6,000 prompt tokens, 32 greedy tokens; curl only) |
| `p2p-prebuilt.txt` | provenance and sha256 of the prebuilt p2p extension (on x86host, not in git) |
| `../../dspark/worker/run/gpu_locked.sh` | the production hold: `flock -n` on both GPU lock files, then exec of the TP2 worker (the unit's ExecStart) |
| `../../launch/prod_window.sh` | windows after promotion: `train`, `pair`, `close`, `check` (`PRODUCTION HOLD OK`), `status` |
| `../../launch/pair_window.sh` | pre-promotion pair windows. The coordinator's hold keeps both locks for the whole window; the owner's GPU commands go through `pair_window.sh run OWNER -- CMD` (window-5 hand-off starvation fix). It refuses open/hand/run/hold/close once the TP2 production unit is enabled or running |
| `../../launch/tests/test_pair_window_handoff.sh` | regression for the hand-off: real hold and real queued `flock -w` lanes under stubs (33 checks: the hand-off, close order, escaped job, close race, stdin, orphan; run on Controller and on x86host) |
| `../../launch/tests/test_prod_gpu_hold.sh` | regression for the three scripts and the unit (stubs plus real lock files; 51 checks, run on Controller and on x86host) |
