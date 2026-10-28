# Spark-side overlap of the async remote round (R2-RUNTIME, 2026-10-23)

Traced in the engine sources (`/opt/dsv41/data/src-image`, vLLM 0.31.1.dev0 dsv41gb10: `v1/worker/gpu/
model_runner.py`, `async_utils.py`, `spec_decode/{utils,depth_policy}.py`, `distributed/parallel_state.py`) and this
package. Results cited from `offload/results/COMPARISON.md` (commits 2d53280, 8c0ab70).

## 1. One proposal / verify cycle, host side (rank 0)

`execute_model(step k)` enqueues the forward; `sample_tokens(step k)`:

1. `self.sample(...)`: sampler + rejection sampler on the main stream S (async; no host read).
2. `AsyncOutput(...)`: its copy stream waits on S (event recorded here) and copies the sampled tokens to pinned host
   memory. Recorded before the remote round, so it never waits for it.
3. `postprocess_sampled(...)`: device updates on S.
4. `speculator.propose(...)` -> `RemoteRound.run()`:
   * thread proxy `_submit_async` / process proxy `_submit_process`: build the BatchView (host numpy), enqueue the
     gather + D2H (+ requested windows), hand the round over (queue item / descriptor + `desc_head`), enqueue
     `gathered = t`, `wait released >= t`, the reply H2D; return. No host synchronization: the only host-blocking path in
     `run()` is the synchronous mode's gather sync, which async mode does not take. `read_status()` (a host read of the
     device reply) is called only with `ON_FAIL=raise`, which async mode refuses (config.py).
   * TP>1: `broadcast_fn(bcast)` = `torch.distributed.broadcast` on the device group (NCCL): enqueue only (the default
     non-blocking-wait mode makes S wait on the collective's event; the host does not block).
5. Back in `sample_tokens`: `req_states.draft_tokens[idx] = drafts`, `depth_policy.record(...)` (all on the device; the
   depth policy keeps `depth` as a device tensor), `draft_tokens_handler.set_draft_tokens(...)` (a host copy of the
   drafts only with structured-output requests), return `async_output`.

So the runner returns right after the enqueue. The first real consumer of the drafts is step k+1's forward on S
(`combine_sampled_and_draft_tokens`, the rejection sampler, `depth.dead_rows`): a device-side stream-order wait, which
is the only wait. Host-side, the round is waited for only (a) by `DraftTokensHandler.get_draft_tokens` when a request
uses structured outputs (grammar validation of drafts), and (b) implicitly by any engine-thread call that blocks on S
or the device while S is parked on the round (ORDERING.md §2).

Rank 1: `run()` only joins the broadcast (enqueue); its S waits on the collective. Both ranks overlap the same way.

## 2. What the GB10 numbers say

Async vs sync on D-kv6 (one-card worker): decode −3.3 to −4.5 % per round; TTFT +3.5 % (32K) to +4–8 % (150K); 31
watchdog fail-closed rounds, identical with switch interval 0.5 ms and 5 ms (si0 retest); E1 (TP2 worker + async):
−4.0 to −4.7 % per round, TTFT cost unchanged, watchdog_preempted +45 over the set.

* The switch interval is not the TTFT cause (si0). A switch interval only affects threads that run bytecode.
* The 31 fires are structural (identical counters). The watchdog clock starts at GPU readiness (hostflag.c `wd_main`),
  so each fire is a round the proxy thread did not publish within 200 ms of the stream parking on it: the thread could
  not run at all. While a round is parked the GPU stream idles; a fire costs 200 ms of GPU time on that rank (and on
  rank 1, which waits on the broadcast). 31 x 200 ms = 6.2 s. The si0 TTFT deltas over the set, excluding the first
  request after boot: 32K x 3: +0.55 / +0.50 s; 32K x 3 concurrent: +2.98 s; 150K x 2: +3.41 / +2.89 s; sum 10.3 s.
  The fires can account for about 60 % of it if they fall in prefill steps (the per-phase counters below decide this).
* Likely mechanism (inference, not yet observed on GB10): prefill steps run long eager forwards with many launches; the
  host runs ahead of S while S is parked on round t; an engine-thread native call that waits on S (a launch blocking on
  a full queue, a device-synchronizing call, ...) while holding the GIL keeps the proxy thread from publishing until the
  watchdog fires (S9 P1). Decode steps are graph replays with few launches, which fits fires in prefill-bearing steps.
* The remaining ~4 s: candidates are the proxy thread spinning a core through every forward (`ASYNC_NAP_US=0`, GB10
  CPU and GPU share power and memory) and GIL hand-offs on the prefill path. Both are measurable (§3).

## 3. Tests for the next GB10 window (in order)

1. `D-kv6-async-process`: D-kv6-async-si0 + `VLLM_DSPARK_REMOTE_ASYNC_PROXY=process` (stage the new package; hostflag.c
   unchanged, so the pinned libdsv41_hostflag.so stays valid). Expect `async_watchdog.fires` and `watchdog_preempted`
   ~0, and the TTFT delta vs sync to shrink by the fires' share. Check `proxy_torch_loaded: false`, `async_proxy.mode:
   process` in the stats file, and `watchdog_events` (empty or attributed by phase).
2. Same + `VLLM_DSPARK_REMOTE_ASYNC_SPIN_US=50 VLLM_DSPARK_REMOTE_ASYNC_NAP_US=50`: the spin's share of TTFT, and what
   the nap costs per decode round.
3. Same as 1 with `CUDA_MODULE_LOADING=LAZY`: EAGER is no longer needed for liveness with the process proxy; this
   measures whether it costs anything (compare with the coordinator's D-kv6-prod-eager).
4. Thread proxy, unchanged config, for attribution only: the new counters `rounds_{decode,prefill}`,
   `watchdog_preempted_{decode,prefill}`, `watchdog_skipped_rounds` and `watchdog_events` (wall time, phase, rows,
   longest query) say which stage the fires come from.
