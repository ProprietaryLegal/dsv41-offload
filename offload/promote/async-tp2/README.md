# Promotion package: TP2 worker + async rounds through the process proxy (R3-PROMOTE, 2026-10-23)

**Written, not applied.** Nothing here has touched production, `~/dsv41-deploy`, `~/dsv41-offload/prod` or a unit.
Apply only after the independent review (S12) of `offload/r3-promote`, its merge into offload-x86host, and a GB10 window
that runs the checks in "Window 5" below on the merged code.

What window 4 measured (COMPARISON.md "Window 4"): TP2 worker + process proxy + nap, CUDA_MODULE_LOADING lazy:
−3.9 to −6.5 % per decode round against the production configuration (one-card worker, synchronous rounds), TTFT at
parity, 0 watchdog fires. R3 fixed what S11 required before production (P-1 late-proxy row corruption, P-2 start
fallback, P-3 stall detection, P-4 respawn, P-6 shutdown); evidence in R3-PROMOTE.report.md.

## What changes

| where | production now (main 6aa71ca, tag prod-d-20261023) | after |
|---|---|---|
| `dsv41.env` OFFLOAD_DSPARK_ENV | configuration D, synchronous rounds | the same + `ASYNC=1 ASYNC_PROXY=process ASYNC_SPIN_US=50 ASYNC_NAP_US=20 ASYNC_RESPAWNS=3 ASYNC_RESPAWN_RESET_S=3600` (all `VLLM_DSPARK_REMOTE_` prefixed); the sync line stays, commented, as the rollback (`dsv41.env.patch`) |
| `start.sh` | — | one warm-up request after the serve answers, when ASYNC=1 (`start.sh.patch`, `offload/launch/warmup_serve.sh`) |
| remote_dspark package (staged by `stage_overlay.sh` from the checkout) | 6aa71ca's | the merged R3 package |
| x86host worker | `dsv41-dspark-worker` (one card, GPU-c4e7a371), `current -> d275f6b` | `dsv41-dspark-worker-tp2-prod` (pair c4e7a371 + 7744b44b), `current -> <merged commit>`, prebuilt p2p (`p2p-prebuilt.txt`) |
| unchanged | model, image, KV 6 GiB/rank, Engram (B), CUDA_MODULE_LOADING lazy, hostflag.c (sha256 10490b58…, aarch64 pin 0ada6aee…), Spark libofl pin 4daef6c2…, x86host libofl pin 9f11b6e8…, venv `~/dsv41-offload/o2/venv` (same torch 2.10.0+cu126 and exllamav3 ext hashes as lane E's measured venv) | |

The patches apply cleanly to main 6aa71ca (`git apply --check`, checked 2026-10-23).

Decisions the coordinator owns before applying (R3 open questions): GPU-7744b44b leaves the development pool for good
(lanes such as N2 need pair windows instead); `offload/launch/pair_window.sh close` restarts the one-card unit, which
`Conflicts=` with the TP2 production unit, so after promotion it must not be used unchanged.

## Window 5 (GB10, supervised, before promotion)

Experiment launcher on :8001 with the merged code: `offload/launch/start_exp.sh offload/launch/configs/D-kv6-async-process-nap-lazy.env`
and the TP2 worker in a pair window (as W4-E1p). Every check must pass; any failure is a stop, with the log to R3.

1. Startup, rank 0 (`docker logs dsv41-serve 2>&1 | grep -E "remote DSpark"`):
   `async mode on (host-flag hand-off self-test passed`, `remote DSpark proxy process running`,
   `through proxy process pid`. Absent: `async mode off`, `did not start`, `async proxy process did not start`.
2. Stats, `docker exec dsv41-serve cat /tmp/dsv41-remote-dspark.json` (light read):
   `async` true; `async_proxy` {mode process, pid, generation 0, region_mlock ok}; `async_proxy_process`
   {pid, generation 0, start_round 1}; `proxy_torch_loaded` false; `async_dead` null; `async_respawns` 0;
   `async_respawn` {state live, limit 3, left 3, failed_starts 0, reset_s 3600}; `async_stall_detection`
   {heartbeat_limit_ms 400, stall_fires 3}; `async_watchdog.fires` 0; no counters `watchdog_preempted`,
   `fallback_preempted`, `proxy_lost_*`, `fallback_proxy_dead`.
3. The compact benchmark set against W4 (decode per round and TTFT within run-to-run noise of W4-E1p).
4. Deliberate proxy kill, while a long generation runs (the respawn advances only on rounds):
   `pid=$(docker exec dsv41-serve python3 -c 'import json; print(json.load(open("/tmp/dsv41-remote-dspark.json"))["async_proxy"]["pid"])')`
   then `docker exec dsv41-serve kill -9 $pid`. Expect in the log, in order: `async proxy process <pid> lost (proxy
   process exited with -9): target-only rounds ... (3 respawn(s) left)`; about 1 s later `starting proxy process <pid2>
   (respawn 1, session 0x..., rounds from N)`; `(remote-dspark proxy pid=<pid2>) ... proxy process running`;
   `async proxy process <pid2> running (respawn 1, ...): drafting resumes at round N`; the new proxy's `worker ... ready
   at o1rdma://...`. The generation completes; its output is valid (drafts never change outputs). Stats:
   `async_respawns` 1, `async_respawn.left` 2, `async_proxy.pid` = pid2, `async_proxy.generation` 1,
   `async_proxy_losses[0].kind` exit, counters `proxy_lost_exit` 1 and `fallback_proxy_dead` > 0, `rounds_ok` still
   counting up (counters are carried across proxies).
5. Deliberate stall: during a generation `docker exec dsv41-serve kill -STOP <pid2>`. Expect `lost (proxy process
   stalled: 3 watchdog fires with no round finished ...)` or `lost (proxy process heartbeat ... ms old (limit 400 ms
   ...)` within about 0.6 s of decoding, then respawn 2 as in 4 (the engine SIGKILLs the stopped process). Stats:
   `async_respawns` 2, `left` 1, `proxy_lost_fires` or `proxy_lost_heartbeat` 1.
6. `taskset -p` of the new proxy shows the wide affinity (W4: 0-19).

The drills spend 2 of the 3 respawns; the budget returns after 3600 s of service or at the next serve start.

## Before the production window (no service impact)

On Controller (coordinator's usual promote flow, as promote-d):
```
git switch -c promote-async main                    # main = 6aa71ca
git merge <reviewed offload-x86host commit>           # carries offload/r3-promote
git apply offload/promote/async-tp2/dsv41.env.patch offload/promote/async-tp2/start.sh.patch
(cd offload/dspark/spark && CUDA_VISIBLE_DEVICES="" python3 -m pytest tests -q)
bash offload/dspark/spark/test_stage_overlay.sh        # 7/7 on offload/r3-promote
git commit ...; then bring it to sparkhead ~/dsv41-deploy as for promote-d
```
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
bash worker/run/prod_tp2.sh check <commit>          # CHECK OK except "current -> d275f6b" (flipped in the window)
```

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
3. x86host: `ln -sfn <commit> ~/dsv41-offload/prod/dspark-worker/current && systemctl --user disable dsv41-dspark-worker
   && systemctl --user enable --now dsv41-dspark-worker-tp2-prod`; wait for the ready line in
   `~/dsv41-offload/prod/logs/dspark-worker.log` and `tp 2` in `~/dsv41-offload/prod/run/dspark-worker.status.json`
   (window 4: ready 68.6 s after start). End the pair window without `pair_window.sh close` (it would restart the
   one-card unit, which `Conflicts=` with this one): `rm ~/dsv41-offload/PAIR_OK`; keep GPU-7744b44b out of lane use
   from now on (the coordinator's decision how: keep `dsv41-pairhold` holding both locks, or retire the lock file).
4. sparkhead: `cd ~/dsv41-deploy && ./stop.sh && ./start.sh` (stages the R3 package and the hostflag pin; lazy
   start about 6 min; the last start.sh line is `warm-up request: N s (prompt_tokens:... completion_tokens:32)`).
5. Checks: Window 5 items 1, 2 and 6 on :8000; one short and one 32K request (TTFT at the W4 level, no first-request
   penalty). The drills (window 5 items 4 and 5) are not repeated in production.
6. `offload/launch/window.sh close` (refuses unless :8000 serves /srv/dsv41/legal-v2a).

## Rollback (under 10 min; x86host and the Sparks in parallel)

* x86host, TP2 -> one card (about 70 s; the serve runs target-only rounds meanwhile and reconnects by itself):
  ```
  systemctl --user disable --now dsv41-dspark-worker-tp2-prod
  ln -sfn d275f6b ~/dsv41-offload/prod/dspark-worker/current
  systemctl --user enable --now dsv41-dspark-worker
  ```
* Sparks, async -> synchronous rounds with the R3 package kept: in `~/dsv41-deploy/dsv41.env` put `#` before the async
  `OFFLOAD_DSPARK_ENV=` line and remove it from the line below, then `./stop.sh && ./start.sh` (about 7 min). The R3
  package then runs synchronous rounds (its two-part STEP send is in that path too: CPU suite, ofl o1tcp test and the
  x86host RDMA loopback probe all exact).
* Sparks, everything back to the exact production code: `cd ~/dsv41-deploy && git checkout prod-d-20261023 && ./stop.sh
  && ./start.sh` (detached at the tag: 6aa71ca's dsv41.env, start.sh and package; about 7 min); move main back with a
  revert commit afterwards.
* Full rollback = the x86host step plus either Spark step, run at the same time: under 10 min, bounded by the serve start.

## Files

| file | what |
|---|---|
| `dsv41.env.patch` | the D-async OFFLOAD_DSPARK_ENV line + the commented sync line, against main 6aa71ca |
| `start.sh.patch` | the warm-up hook, against main 6aa71ca |
| `../../launch/warmup_serve.sh` | the warm-up request (~6,000 prompt tokens, 32 greedy tokens; curl only) |
| `p2p-prebuilt.txt` | provenance and sha256 of the prebuilt p2p extension (on x86host, not in git) |
