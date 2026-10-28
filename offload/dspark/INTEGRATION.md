Status 2026-10-23: Original integration checklist retained; capacity/async production confirmed W5. Use promotion README and roadmap for current TP2/native gates. See [live roadmap](../../documents/plans/roadmap.md) and [measured record](../results/COMPARISON.md); campaign review decisions are in `LANES.md` / lane reports outside the repo.

# Remote DSpark drafter — Spark-side integration checklist (lane O3 → coordinator)

Everything below is for the coordinator (session coordinator). Lanes never restart the serve. Every step runs inside
an open window (`offload/launch/window.sh open …`), through `offload/launch/start_exp.sh <config>` on :8001. To roll
back, run `./start.sh`, or start any config without the overlays.

Wire/state contract: `offload/dspark/CONTRACT.md` (v1.8). Code: `offload/dspark/spark/` (lane O3, branch
`offload/o3-proxy`). Worker: lane O2, `offload/dspark/worker/` (`server.py`, CONTRACT §6–§9).

## 0. What the overlay changes, and what it does not

- **One image file is replaced:** `vllm/v1/worker/gpu/spec_decode/__init__.py`. The replacement is the image file plus
  one guarded branch: when `VLLM_DSPARK_REMOTE` is non-empty, `method == "dspark"` builds the remote speculator.
  - Unset, the image path runs unchanged. `tests/test_speculator.py::test_overlay_is_image_plus_guarded_branch`
    pins the diff.
  - The base file's sha256 `61b3acba…aa1b7e8` equals the live container's file (checked 2026-10-22 on sparkhead).
- **One package is added:** `vllm/v1/worker/gpu/spec_decode/remote_dspark/`, the contents of
  `offload/dspark/spark/remote_dspark/`.
- **Unchanged:**
  - The target model, verifier, rejection sampler, depth policy, 0040 schedule, 0047 adaptive window, the scheduler,
    and the API/aliases.
  - SPEC JSON, `--kv-cache-memory-bytes` and all production env vars stay as they are, unless a step below says
    otherwise.
- **Both ranks** need the same overlays and the same env: TP ranks must make identical decisions. Only rank 0
  connects to x86host; rank 1 receives drafts through one NCCL broadcast on the TP group.
- **Compatible with the coordinator's capture overlay** (`offload/dspark/capture/speculator.py`, mounted over
  `dspark/speculator.py`). The two can run together; for timing, use neither.

## 1. Preparation

### 1.1 Files (from the merged `offload-x86host` checkout on sparkhead)
`start_exp.sh` copies `OVERLAYS` to `~/dsv41-offload-run/overlay/` on both nodes and bind-mounts them read-only. The
ready configs in `offload/dspark/spark/launch-configs/` already list both overlays:
```
offload/dspark/spark/remote_dspark=/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/remote_dspark
offload/dspark/spark/overlay/vllm/v1/worker/gpu/spec_decode/__init__.py=/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/__init__.py
```
Copy the config you want into `offload/launch/configs/`, or pass its path directly.

### 1.2 Identities
`python3 offload/dspark/spark/tools/identity.py <S2's DSPARK_COPY_DONE>` prints the two values the configs carry
(live legal-v2a, 2026-10-22):
```
VLLM_DSPARK_REMOTE_IDENTITY=be2c1436fa9f9e66b398e1b1da2e140e6fddc82a11939e2ac4240a0ca79543bd
VLLM_DSPARK_REMOTE_MARKOV_IDENTITY=0b6e71e638abd5abce2db433965bb5ff523cbbe92ed7ae54324a50ccaeffa19a
```
The worker must report the same values at HELLO; otherwise rank 0 disables the remote and logs both digests.

### 1.3 Transport
The endpoint scheme selects the protocol. The worker has to speak the same one.

- `o1rdma://10.202.28.1:18620?bind=10.202.28.2`: lane O1's `ofl`, RDMA RC. This is what all four configs use, and
  what O2's server is expected to serve (an ofl server, service `dspark-v1`).
  - Measured Spark A ↔ x86host p50: echo 5.6 µs, 75 KB 33 µs (O1 API.md).
  - `o1tcp://…` selects ofl's TCP backend on the same server and port, for debugging.
  - Both need `libofl.so` + `ofl.py` in `VLLM_DSPARK_REMOTE_OFL_DIR`; the configs mount `~/dsv41-offload-run/ofl` at
    `/opt/dsv41-ofl`. Build on sparkhead's host, aarch64 (O1 API.md "Build"), and put the same directory on sparkworker
    so the bind mount source exists there too; rank 1 never loads it:
    ```
    cd ~/dsv41-deploy/offload/transport && make OUT=$HOME/dsv41-offload-run/ofl $HOME/dsv41-offload-run/ofl/libofl.so \
      && cp ofl.py $HOME/dsv41-offload-run/ofl/
    ssh sparkworker mkdir -p dsv41-offload-run/ofl && scp -q $HOME/dsv41-offload-run/ofl/{libofl.so,ofl.py} sparkworker:dsv41-offload-run/ofl/
    ```
  - O1 commit f256568 or later gives a stable `payload_buf`. The proxy does not use it yet: each STEP is copied into
    the ofl send slot once, ≈ 20–40 µs per decode round and ≈ 0.5–1 ms per 4-request prefill step.
- `tcp://10.202.28.1:18620?bind=10.202.28.2`: the contract's own TCP stub (length-prefixed frames). It is a
  **different wire protocol** from ofl's TCP backend. Only the reference `mock_worker` (TCP-stub mode) or a worker
  that implements the stub speaks it. It needs no library.
- `remote_dspark/mock_worker.py` can stand in for the worker on x86host, for plumbing checks without the GPU drafter.
  It serves either protocol: `--ofl-dir <dir with libofl.so + ofl.py>`, or the stub. Its drafts are fake, so use
  shadow policy only.

### 1.4 Offline checks (Controller, CPU)
```
cd offload/dspark/spark && python3 -m pytest -q tests        # 160 pass (2026-10-22, O3-FIX)
```
- An end-to-end engine simulator checks exact drafts through: chunked prefill with trimmed taps, decode with random
  k and rejections, slot reuse, re-add/replay, timeouts, worker restart, frame and per-request errors, and dropped
  connections.
- Also covered: the speculators with vLLM stubs, the overlay diff and dispatch, and the proxy over ofl's TCP backend
  (`.build/ofl`).

### 1.5 Pre-flight against the real worker (before the experiment start)
Run inside an open window or with the serve idle. The probe refuses unless `/metrics` shows 0 running and 0 waiting.
During a window nothing listens on :8000, so pass `--no-metrics-check` only when the serve is down.

Run the probe on sparkhead's host with the system python3. It needs no numpy and no torch (checked with numpy
hidden): it uses only `remote_dspark/{wire,transport,o1_adapter}.py` from the merged checkout.
```
VLLM_DSPARK_REMOTE_OFL_DIR=$HOME/dsv41-offload-run/ofl python3 ~/dsv41-deploy/offload/dspark/spark/tools/probe.py \
  --endpoint 'o1rdma://10.202.28.1:18620?bind=10.202.28.2' \
  --identity be2c1436fa9f9e66b398e1b1da2e140e6fddc82a11939e2ac4240a0ca79543bd \
  --markov-identity 0b6e71e638abd5abce2db433965bb5ff523cbbe92ed7ae54324a50ccaeffa19a --hello --step-rows 128
```
Expected:
- `hello_status: OK`, with the worker's identity and markov digests equal to ours;
- `ping_us` (TCP ~30–60 µs, RDMA ~6 µs);
- `step_status: OK` and `step_req_status: OK` for a 128-row window. `step_worker_us` is O2's draft time for B=1.

`--hello` opens a new session, which drops an engine session's windows on the worker. Use it only while no engine
is connected; the engine would recover by itself (NO_SESSION → HELLO → resync).

## 2. Order of in-engine checks

| step | config | purpose | pass criterion |
|---|---|---|---|
| 1 | `C-shadow.env` | wiring + parity. Production drafts unchanged; the remote runs alongside | the worker connects (log `remote DSpark worker '…' ready`); `rounds_ok ≈ rounds`; `shadow.full_block_agree_rate` high; mismatches are near-ties (`examples[].remote_margin` small) |
| 2 | `C-shadow-remote.env` | acceptance with remote proposals (local still compared) | `vllm:spec_decode_num_accepted_tokens_total / num_drafts_total` within noise of A on the same prompts |
| 3 | timing inputs | read `latency.local_propose_us` and `latency.rank0_round_us` from the shadow stats | replace the PLACEHOLDER `REMOTE_ROUND_MS` / `REMOTE_DRAFT_MS` in C-capacity / C-resilience (§4) |
| 4 | `C-capacity.env` | the capacity mode: no local drafter | starts; reclaimed memory measured (§5); outputs sane; speed vs A (bench2) |
| 4b | `C-capacity-async-cal1.env` (lane A2's hand-off; `CUDA_MODULE_LOADING=EAGER` stays: the proxy needs the GIL, and pybind11 extension launches hold it — S9 P1) | same, with async rounds (the CPU never blocks; the engine stream waits on a host word the proxy thread writes with no CUDA call) | log `remote DSpark async mode on (host-flag hand-off self-test passed; …)`; stats `async: true`, `async_handoff`; speed vs step 4. Deploy `libdsv41_hostflag.so` next to `libofl.so` (§1.3) or let rank 0 compile `hostflag.c` at startup |
| 4c | `A-local.env` vs `C-capacity.env` (then `C-capacity-nodrop.env` if needed) | prefix-cache hits without the drafter's KV layers (§5, "KV group annotation") | the second request of a repeated ≥ 32K prompt has the same `cached_tokens` as under A |
| 5 | `C-capacity(-async).env` with `KV_BYTES_OVERRIDE` | re-sized KV after step 4's measurement | larger `GPU KV cache size` / `Maximum concurrency` in the log; the 256K/300K runs of plan §4.5 |
| 6 | `C-resilience.env` | per-round fallback | stop the worker mid-run → `local_fallback_rounds` grows, outputs continue; restart it → remote resumes (`resync_rounds`) |
| 7 | failure drill in capacity | stop/restart the worker mid-run | the target-only rounds are counted (`fallback_not_ready`, `fallback_timeout`); after restart `hellos` +1, `resync_rounds` +1, `rounds_ok` grows again |

Step 1 shows rank 0's gather in `gather_us`, and the network plus worker time in `rtt_us` / `worker_us`. In shadow
mode the step is slower than A because both drafters run; do not benchmark shadow.

## 3. What to read

- **Rank 0 stats file:** `docker exec dsv41-serve cat /tmp/dsv41-remote-dspark.json` on sparkhead. It is written
  every `VLLM_DSPARK_REMOTE_LOG_EVERY` rounds; a log line is emitted at the same time.
  - `counters`: `rounds_ok`, `rounds_failed`, `rounds_some_failed`, `fallback_<reason>` (`not_ready`, `disabled`,
    `reconnected` (the connection changed after the round's entries were built; nothing sent), `cooldown`, `timeout`, `transport_error`, `bad_reply`, `frame_<status>`, `internal_error`), `req_status_<status>`,
    `resync_rounds`, `resync_rows`, `rows_sent`, `bytes_sent`, `bytes_received`, `hellos`, `disconnects`,
    `connect_failures`, `local_fallback_rounds` (resilience), `rounds_n_out_cut` (rounds sent with n_out < N),
    `breaker_trips`, `breaker_recoveries`, `fallback_cooldown` (rounds skipped by an open breaker).
  - `breaker`: `state` (`closed` / `open` / `half_open`), `consecutive_failures`, `cooldown_left`, `trips`, and the
    configured `trip` / `cooldown` (CONTRACT §10). Log lines: `remote DSpark circuit breaker open (…)` and
    `… closed: the probe round succeeded`.
  - `last_bad_reply`: why the last reply was refused. `fallback_bad_reply` counts replies refused for their framing
    or their values (CONTRACT §6.3 v1.4: draft ids outside the vocabulary, confidences not in [0, 1], bad window or
    status, nonzero n_out tail). Any nonzero count is a worker bug to report to lane O2; the round ran target-only
    (capacity) or local (resilience) and the slots resynced.
  - `stale_generation_new_sessions` (and `req_status_stale_generation`): the worker reported a slot generation
    newer than rank 0's. Rank 0 never sends an older one, so this means the worker's state diverged; rank 0 started a
    new session (the worker dropped every window) and resynced (CONTRACT §9.2, v1.5). Expected 0; report any other
    value.
  - `latency` (µs, p50/p90/p99/max):
    - `rank0_round_us`: rank 0's whole remote round. This is the remote cost the depth policy should get.
    - `gather_us`: device gather + D2H + sync.
    - `rtt_us`: network + worker.
    - `worker_us`: the worker's own report.
    - `local_propose_us`: shadow only; the local drafter, CUDA events.
  - `shadow`: `agree_rate_by_position`, `first_mismatch_hist` (index 5 = whole block agrees), `full_block_agree_rate`,
    `max_conf_abs_diff`, `examples` (local vs remote ids, worker margins).
  - `worker`, `policy`, `endpoint`, `rank0_buffers` (45 MiB device, 15 MiB pinned), `async` /
    `async_fallback_reason` / `async_handoff` (mechanism, library path, spin/nap), and `fused_gather` (`on`, `check`).
  - Async counters: `fallback_resync_pending` (a resync waited one round for its staged windows),
    `resync_windows_staged` (slot windows the engine stream copied for the proxy); `rank0_submit_us` (the engine
    thread's enqueue cost per round).
    - The one-launch Triton row gather (`remote_dspark/fused_gather.py`) replaces ~40 eager torch ops per round on
      rank 0's stream.
    - It is self-checked at startup against the torch path on the GPU (bit-identical rows, ring and metadata) and
      turned off with a warning on any mismatch or error. It was verified offline under `TRITON_INTERPRET=1`.
    - `VLLM_DSPARK_REMOTE_FUSED_GATHER=0` forces the torch path.
  - `n_out_by_reqs`, `n_out_reason` (CONTRACT 1.3, §6.3): the draft positions requested from the worker by request
    count, and why.
    - In capacity and resilience, with the live `conf` policy and schedule, this is `[0, 5, 5, 4, 4]`: n_out 4 at 3–4
      requests, which saves the worker ≈ 0.55 ms per round at 4 requests (O2).
    - Shadow always shows `null` (all N, so parity covers every position).
    - `VLLM_DSPARK_REMOTE_N_OUT=0` turns it off.
    - A `VLLM_DSPARK_DEPTH_FILE`, or a `fixed:`/`off` depth policy, also turns it off.
- **Serve logs:**
  - `remote DSpark (<policy> policy) rank 0 -> <endpoint>`
  - `remote DSpark worker … ready … (session …, epoch N)`: the epoch rises on every reconnect.
  - `remote DSpark disabled for this engine run: …`: identity, config or version refused. Fix it and restart.
  - In remote modes, the `DSpark depth policy conf (k=5), round cost ms by depth […]` line must show the remote table.
- **vLLM `/metrics`:** as bench2 reads them: `vllm:spec_decode_num_drafts_total`, `…num_draft_tokens_total`,
  `…num_accepted_tokens_total`, `…num_accepted_tokens_per_pos_total`.
- **Worker side:** O2's STATS op (`tools/probe.py --hello` prints it before the run) and its own logs.

## 4. Depth-policy costs in remote modes (recalibration)

The live `VLLM_DSPARK_ROUND_MS=33.26,…` is "verify forward + drafter" by depth, with `VLLM_DSPARK_DRAFT_MS=0`. Its
cost already includes the local drafter. In capacity and resilience the remote round must be an explicit input:
- `VLLM_DSPARK_REMOTE_ROUND_MS` replaces `VLLM_DSPARK_ROUND_MS` with a verify-only table. First estimate: each live
  entry minus shadow `local_propose_us` p50 (in ms). Better: measure with fixed depths, the way the live table was
  made.
- `VLLM_DSPARK_REMOTE_DRAFT_MS` replaces `VLLM_DSPARK_DRAFT_MS` with the remote round: shadow `rank0_round_us` p50 in
  ms (it includes gather, network and worker; add ~0.05 ms for the TP broadcast).
- The overlay writes both into the process env before the model runner builds the depth policy. Both ranks get the
  same env, so the tables stay identical across TP ranks.
- With the depth policy on (live: `conf`), capacity and resilience refuse to start unless **both** `REMOTE_DRAFT_MS`
  and `REMOTE_ROUND_MS` are set. The policy prices depth d as `ROUND_MS[d] + DRAFT_MS`, so either one alone mixes the
  live table (local drafter included) with a remote cost. The error names what is missing, and nothing is
  overridden. Checked at startup:
  - `REMOTE_DRAFT_MS`: one finite value ≥ 0;
  - `REMOTE_ROUND_MS`: exactly k + 1 = 6 finite values > 0 (depth 0..5), non-decreasing. The launch configs'
    placeholders pass.
- `VLLM_DSPARK_DEPTH_POLICY=confr:<rows ms>` is refused in capacity and resilience. Its rows table is part of the
  policy string, measured with the local drafter, and the remote inputs cannot replace it. Use `conf` (live) or
  `confb`.
- A `VLLM_DSPARK_DEPTH_FILE` can replace the cost table at runtime (`conf:<ms…>`, `confb:<ms…>`, `confr:<ms…>`). In
  remote modes, any table in that file must be a remote-mode measurement. The overlay cannot check it.
- `fixed:<d>` never prices a depth, so it needs neither value unless a `VLLM_DSPARK_DEPTH_FILE` is set (the file
  can switch the mode to `conf` at runtime). This lets you measure the remote verify table with fixed depths without
  dummy costs.
- With the depth policy off, nothing reads either value. They are only parsed: a value given alone is written, and
  a malformed one is refused.
- Shadow ignores both and keeps the local measurements.

## 5. Capacity mode: what disappears, and how to measure it

Not allocated on either Spark:
- the drafter weights. `mtp.*` is 3.77 GiB on disk, TP-sharded, so roughly 1.9 GiB per rank plus the replicated
  `main_proj` (47 MiB) and Markov head (126 MiB);
- the Markov bias cache (4,096 rows);
- the draft CUDA graphs (FULL_DECODE_ONLY captures of the query block);
- the draft workspace lane;
- the three draft attention layers. They are never registered, so `get_kv_cache_spec` has no draft SWA layers and
  the KV-cache groups change.

Measure, A vs C-capacity, both ranks:
1. Log lines: `Model loading took X GiB`, `Graph capturing finished … took X GiB`, `GPU KV cache size: N tokens`,
   `Maximum concurrency …`, and the KV-cache group printout (count of groups/layers).
2. `free -g` / MemAvailable on each Spark after warmup, plus `nvidia-smi` per-process usage.
3. With `--kv-cache-memory-bytes` unchanged (4026531840), the token capacity should rise a little: the same bytes
   now serve fewer layers. The freed non-KV memory is what `KV_BYTES_OVERRIDE` can then take. Keep the margin the
   production fit was sized with; Sparks run at 5–7 GiB MemAvailable.
4. Retained for the remote path: rank 0 holds 45 MiB of device buffers (15 MiB ring + two 15 MiB stages), a 15 MiB
   pinned message buffer and the ofl send slot (16 MiB). Rank 1 holds a < 1 KiB broadcast tensor. The stats file
   reports `rank0_buffers`.

Also note:
- **KV group annotation and prefix-cache hits (check in the capacity window).**
  - What happens:
    - SPEC still says `dspark`, so `use_eagle_block_drop()` stays true (`config/speculative.py:1947-1949`).
    - For `deepseek_v41`, `_uses_trailing_mtp_layers` turns on `_annotate_eagle_groups`' positional fallback
      (`v1/core/kv_cache_utils.py:2155-2171`, `:2234-2242`). That fallback flags the group holding the last
      registered layer as the "eagle" group.
    - Under A, that group is the drafter's SWA group (the three `mtp.*` layers register last). In capacity mode the
      draft layers do not exist, so it is the group holding the **last target layer**: a target group.
    - If the partition check fails, no group is flagged. The coordinator then flags **every** group
      (`v1/core/kv_cache_coordinator.py:104-110`).
    - A flagged group loses its trailing scheduler block from prefix-cache hits, the "EAGLE last-block drop"
      (comment at `kv_cache_coordinator.py:112-122`). Hits land on scheduler-block boundaries, so a hit can come
      out up to one scheduler block shorter than under A, and the request prefills that much more.
    - Output is never affected, only cached tokens and prefill time.
  - How to check, in the window. Run the same steps under A (`A-local.env`) and under `C-capacity.env`:
    1. Startup log of both ranks: `kv cache group sizes [...]` and `kv lcm block sizes N`. Record how capacity
       differs from A; the drafter's layers are gone, so a group may disappear or merge. N is the scheduler block,
       the most a hit can lose. No log line names the flagged group, so step 2 is the real test.
    2. Send one long legal prompt (≥ 32K tokens, from bench2) twice in a row on an idle serve. Read the second
       response's `usage.prompt_tokens_details.cached_tokens`; production returns it.
    3. Read the deltas of `vllm:prefix_cache_queries_total` and `vllm:prefix_cache_hits_total` on `/metrics` across
       those two requests.
    4. Compare A with capacity. Equal cached tokens: nothing to do. Capacity lower by about N tokens per hit: the
       fallback flagged a target group.
  - Remedy, capacity mode only: `launch-configs/C-capacity-nodrop.env`, which is C-capacity with
    `"disable_eagle_block_drop": true` added to `SPEC_OVERRIDE`.
    - The drop exists so that a prefix hit never reuses a draft-KV slot written with the chunk's lookahead token
      (`kv_cache_coordinator.py:112-122`). Capacity mode has no local draft KV, so dropping a target block protects
      nothing.
    - Rerun step 2 under it: cached tokens should match A.
    - **Never in resilience or shadow:** they keep the local draft KV and need the drop.
- **Wasted 160 MiB per rank in all modes, not touched here.** The target allocates `_mtp_hidden_buffer`
  (4096 × 20480 bf16) because `use_eagle()` includes dspark (`nvidia/model.py:874-886`), and copies the hidden states
  into it every step. DSpark never reads it: the aux taps exist (`dflash/speculator.py:376-381`). Freeing it needs a
  model-side change: skip the buffer for dspark, and return a zero-width view from `get_mtp_target_hidden_states`,
  which `model_runner.py:2336-2338` slices. That would reclaim 160 MiB per rank in A as well.

## 6. Failure policies (CONTRACT §10)

- **Circuit breaker, all policies (CONTRACT §10 v1.6).**
  - After `VLLM_DSPARK_REMOTE_TRIP` failed attempts in a row (default 3), the next `VLLM_DSPARK_REMOTE_COOLDOWN`
    rounds (default 256) skip the network. A failed attempt is a STEP that timed out, was refused or got a frame
    error, or a STEP where every request that needed drafts failed. Then one probe round.
  - Rounds without a connection do not count, and neither do successful rounds with nothing to draft (prefill
    appends only, v1.8). A round where only some requests failed is a success.
  - Why it exists: without it, a worker that answers but fails every round costs each round its RTT, and a worker
    that misses every deadline costs one 50 ms stall per reconnect (≈ 5 %). With it, the cost is one attempt per
    ≈ 9 s.
  - `VLLM_DSPARK_REMOTE_TRIP=0` turns it off. That is useful in a failure drill when every attempt should hit the
    worker.
  - Drill expectation (step 7): with the worker stopped, rounds are `fallback_not_ready` while the reconnect backoff
    runs and never trip the breaker, since no STEP is sent. A worker that is up but broken trips it after 3 rounds.
- **capacity.**
  - A failed round sends zero drafts with confidence 0, so the depth policy verifies nothing that round.
  - Rounds never wait on a connect: a daemon thread reconnects with backoff (1 s doubling to 30 s), and while it is
    down rounds fail in ~µs.
  - On recovery: HELLO, then the whole window of every live slot is resent from rank 0's ring. Drafts are exact
    again from the next round.
  - `VLLM_DSPARK_REMOTE_ON_FAIL=raise`, benchmarks only: any failed round raises on both ranks (engine-fatal), and
    connect + HELLO must succeed at startup.
  - A clean failure of a single request is not possible from inside the speculator.
- **resilience.** After a failed round both ranks run the local drafter that round, as decided by the broadcast
  status. Requests that failed on the worker get local drafts; the others keep remote drafts. The local context KV is
  kept current every round with an eager precompute of the new rows. Nothing is freed.
- **shadow.** Both run. Local drafts are used unless `VLLM_DSPARK_REMOTE_SHADOW_USE=remote` is set; with it, failed
  remote requests fall back to local.
- **Never** a hang: every network wait has a deadline (`VLLM_DSPARK_REMOTE_TIMEOUT_MS`, default 50 ms), and an
  exception on rank 0 still broadcasts a failed status so rank 1 never waits forever.

## 7. Rollback

- **Immediate:** `./start.sh` (production config; no overlay, no remote env).
- **In an experiment:** start the same config without `OVERLAYS`/`EXP_DOCKER_ENV`, e.g. `A-local.env`. Without
  `VLLM_DSPARK_REMOTE` the overlay is inert, but the plain image is the reference.
- **Worker:** stop O2's unit. A connected engine then runs per its policy: capacity → target-only, resilience →
  local drafter, shadow → unchanged.

## 8. Known limits

- **Greedy drafts only** (live: local argmax). Probabilistic drafting, `dspark_draft_topk`, watermarking, adaptive
  verification (`enable_adaptive_verification`, off live), PP, DP and context parallelism are refused at startup with
  a message.
- **Sync rounds are a host round trip on rank 0.** The rank-0 CPU waits for the target step, then for the network
  and the worker, so the next step's CPU preparation is no longer overlapped with GPU work, as it is with the local
  drafter. That is a fixed per-step cost of ≈ 0.5–2 ms on top of the drafter time.
  - In shadow it is visible as `rank0_round_us` minus `rtt_us`.
  - Async rounds (`VLLM_DSPARK_REMOTE_ASYNC=1`, capacity only) remove it. Lane A2 design, v2 after review S9
    (`remote_dspark/streamsync.py`, `hostflag.c`): one C-owned page of registered, device-mapped host memory holds
    three 32-bit words (`gathered`, `released`, `claimed`) and the reply. The engine stream writes `gathered = t`
    after the gather's copies (`cuStreamWriteValue32`, fenced), waits for `released >= t` (`cuStreamWaitValue32`),
    then copies the reply from the page (`cuMemcpyHtoDAsync`) and broadcasts it. The proxy thread acquire-spins on
    `gathered`, runs the round on the CPU and the network, builds the reply in a private buffer and publishes it in
    one GIL-free C call: claim `t`, copy the reply into the page, release `t` (monotonic). It makes **no CUDA or torch
    call**: the first async design (side-stream H2D + device flag written by the proxy thread) deadlocked on GB10
    against a Triton module load on the engine thread while the stream was parked
    (`offload/results/C/lazy-repro/RESULT.md`); a host word is released by a CPU store, which needs no CUDA lock. The
    engine's Engram spin reader uses the same handshake on GB10.
  - A native watchdog thread (C: no Python, no CUDA) fails and releases a round that was gathered but not released
    within the deadline; the claim word makes it and the proxy's publish mutually exclusive per round. Releases never
    move backwards (cyclic max), so a late proxy cannot undo a later release (S9 F3). If the proxy thread ends for any
    reason other than close(), every pending and later round is failed and released at once (far-ahead claim) and
    later rounds skip the hand-off (zero drafts) (S9 F2). close() fails everything still pending the same way and
    logs an error if the thread did not stop within 5 s.
  - Resync rows in async mode come from ring windows the engine stream copies to pinned host memory on request:
    the first round that needs a resync asks for them and fails (zero drafts), the next one is exact.
    `fallback_resync_pending` counts those rounds (one per failure/reconnect event; two if the engine had already
    submitted the next round).
  - The C helper `libdsv41_hostflag.so` (v2: page allocation + mlock, acquire wait with GIL released, monotonic
    release, claim, publish, fail-release, watchdog). Production staging (`spark/stage_overlay.sh`, S9 F8) copies a
    pinned aarch64 build next to `libofl.so` and refuses when async is configured and the build is missing or not the
    pinned one (the pins: source `HOSTFLAG_C_SHA`, build `HOSTFLAG_SO_SHA`, set after building on the head node:
    `cc -O2 -Wall -shared -fPIC -o ~/dsv41-offload-run/ofl/libdsv41_hostflag.so
    offload/dspark/spark/remote_dspark/hostflag.c -lpthread`). Experiments may leave it out: rank 0 then compiles
    the overlay's `hostflag.c` with `cc` into a private temp directory at startup (`async_handoff.lib` says which).
    No C compiler and no library: async is refused with a warning (synchronous rounds).
    `VLLM_DSPARK_REMOTE_HOSTFLAG_LIB` names an explicit path. A v1 library is refused (version check).
  - The proxy thread spins on one rank-0 core while a round is pending (as `cudaEventSynchronize` did);
    `VLLM_DSPARK_REMOTE_ASYNC_NAP_US` > 0 sleeps between checks after `VLLM_DSPARK_REMOTE_ASYNC_SPIN_US`.
  - Startup self-test (`HostFlagSync.self_test`): the CPU sees the copied data with `gathered`; the stream does not
    pass the wait before the publish, and its reply copy after the wait reads the bytes the publish wrote (the
    production path). Its rescue is a CPU store; on failure the mode turns off with the reason. If the stream still
    does not drain after the rescue (a device that never observes host stores) it raises `StreamStuckError`, which
    stops startup instead of falling back to a stream that is parked for good (S9 F4).
  - Tested (lane A2, 2026-10-23): CPU suites (a profiler test fails on any torch call from the proxy thread), and on
    an RTX 3090 the GPU harness `tests_gpu/gpu_async_harness.py` with the reference worker on localhost: the first
    design deadlocked on a fresh `cuModuleLoadData` during a parked round (killed at 90 s); this one completed 30/30
    such rounds (each load waited ~41 ms = the injected worker delay, then returned) and a 10,000-round soak
    pipelined 3 deep with 270 module loads during parked rounds and 49 injected faults: 19,082 drafts bit-exact,
    9,319 failed rows zero, 73 resyncs from staged windows (124 `resync_pending` rounds), proxy round p50 664 µs
    vs 861 µs for the first design without module loads. Evidence: `$CAMPAIGN_LOGS_ROOT/DSV41-OFFLOAD/A2-evidence/`.
    GB10 is first run by the coordinator.
  - `CUDA_MODULE_LOADING=EAGER` stays in the async configs (S9 F1). The proxy needs the GIL, and pybind11 extension
    launches (every `vllm_exl3_moe` extension: plain `m.def`, no `gil_scoped_release`) hold it; under lazy loading
    such a launch waits for the parked stream while holding the GIL (S9 P1). EAGER moves those loads to startup.
  - The native watchdog (hostflag v2, S9 F2/F5) is the backstop for what EAGER cannot cover (an extension first
    imported while serving, a GC finalizer on the proxy thread making a CUDA call, proxy death, interpreter exit): a
    round the stream gathered but nobody released within `VLLM_DSPARK_REMOTE_ASYNC_WATCHDOG_MS` (auto
    `max(4 x TIMEOUT_MS, 200)`) is failed (zero drafts) and released from C, with no Python and no CUDA. Counters:
    `async_watchdog.fires`, `watchdog_preempted` (the proxy found its round already failed), `fallback_proxy_dead` /
    `async_dead` (the proxy thread ended: every later round is target-only, no hand-off).
  - The proxy thread shares the GIL with the engine thread, so async mode lowers the interpreter switch interval to
    500 µs (`VLLM_DSPARK_REMOTE_SWITCH_INTERVAL_US`) to bound the waits.
- **No local timing claims.** The overlay ran only offline (CPU, Controller). Every number on GB10 comes from the
  coordinator's runs.
