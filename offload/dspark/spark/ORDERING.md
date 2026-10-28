# Async remote round: GPU ordering and buffer lifetimes (R2-RUNTIME, 2026-10-23)

Scope: rank 0's async round (`VLLM_DSPARK_REMOTE_ASYNC=1`, thread or process proxy) and its consumers on both TP ranks,
traced in this package and in the engine sources (`/opt/dsv41/data/src-image`, vLLM 0.31.1.dev0 dsv41gb10).
NDM 2026-10-23 §3: "CUDA stream-memory waits are not automatically visible to CUDA's scheduler; express required GPU
producer/consumer relationships with supported events and validate lifetimes. Keep the supported host-staged RDMA
path. Neither a CPU atomic nor an indiscriminately added FLUSH flag proves the complete ordering contract."

## 1. One round on the engine stream S (rank 0)

`S` is the model runner's current stream when `propose()` runs (`model_runner.py` sample_tokens, the main stream).

| # | op on S | producer -> consumer | why it is ordered |
|---|---|---|---|
| 1 | gather kernels (fused Triton or torch) | taps (forward on S) -> device ring, stage, meta_dev | same stream |
| 2 | D2H rows -> STEP message rows, metadata -> metadata slot `t % K` | S -> host (proxy) | thread: `Tensor.copy_(non_blocking)` into pinned memory; process: driver `cuMemcpyDtoHAsync` into the registered region (never a pageable, synchronous copy) |
| 3 | D2H of requested resync windows | S -> host (proxy) | as 2 |
| 4 | `cuStreamWriteValue32(gathered = t, DEFAULT)` | S -> proxy, watchdog | executes after 1-3 complete (stream order); DEFAULT puts a stream-scoped system fence before the write, so 2-3 are visible to the CPU before the word; the proxy reads them only after an acquire load of `gathered >= t` (hostflag.c `wait_geq`) |
| 5 | `cuStreamWaitValue32(released >= t, GEQ)` | proxy / watchdog -> S | the reply bytes are stored before the release store (hostflag.c publish / fail_release: memcpy, then CAS with RELEASE); the GPU observes the word only after the bytes are globally visible |
| 6 | `cuMemcpyHtoDAsync(bcast <- reply area)` | host -> S | enqueued after 5, so the copy engine reads the page only after the wait passed |
| 6a | `cuStreamWriteValue32(consumed = t, DEFAULT)` | S -> proxy | explicit consumer acknowledgment (a5c847b): written after the copy completed; the proxy requires `consumed >= t-1` before it writes round t's reply |
| 7 | TP>1: `torch.distributed.broadcast(bcast)` (vLLM GroupCoordinator, NCCL) | S -> NCCL stream -> rank 1's S | ProcessGroupNCCL makes its stream wait on an event recorded on S after 6 (an event completes only after every earlier op on S, the stream-memory wait included); rank 1's S waits on the collective's event |
| 8 | speculator copies `res.drafts` / `res.conf` into its buffers; model runner `req_states.draft_tokens[idx] = drafts`; depth policy `record()` | S | same stream |
| 9 | `DraftTokensHandler.set_draft_tokens` (only with structured-output requests) | S -> its copy stream -> host | `copy_stream.wait_stream(S)` = event after 8; host waits on `copy_event` (the only host read of drafts, grammar validation) |
| 10 | next step's forward (`combine_sampled_and_draft_tokens`, rejection sampler, `depth.dead_rows`) | S | same stream: the first real consumer, a device-side wait only |

Not dependent on the round: `AsyncOutput` (sampled tokens) is created before `propose()`; its copy stream waits on an
event recorded before step 1, so the host's `get_output()` never waits for the remote round. The depth policy keeps
its state on the device (`depth.index_copy_`); nothing on the host reads drafts or confidences except item 9.

So every GPU producer/consumer pair that crosses streams (7, 9) is expressed with events by the engine (torch.distributed,
`wait_stream`), and those events are recorded after the stream-memory wait, which they therefore include. Items 4-6
are the only host<->stream hand-offs and are ordered by the words, not by an event; no event is needed there and none
can replace them (the host side is not a CUDA stream). No `FLUSH` flag is needed on the wait: `CU_STREAM_WAIT_VALUE_FLUSH`
orders remote writes into DEVICE memory; the reply is host memory read by the copy engine after the wait. The RDMA
path stays host-staged: the NIC reads the STEP message from host memory and writes the reply into ofl's own registered
host buffer; nothing is RDMA'd into device memory.

## 2. What "not visible to the scheduler" costs, and what the process proxy changes

A stream parked on (5) looks busy to everything else. Any engine-thread call that waits for S, or for the whole device,
blocks until the round is released: a lazy module load (needs the device idle; S9's GB10 deadlock), `cudaFree` and
other device-synchronizing calls, a launch that blocks because the stream's work queue is full behind the parked
op, `cudaStreamSynchronize` / event syncs after (5). With the thread proxy the release needs the engine's GIL; if that
blocking call holds the GIL the round can only be released by the watchdog (deadline >= 200 ms; S9 P1). The GB10 async
runs fit this signature: 31 watchdog fires, identical with switch interval 0.5 ms and 5 ms (a switch interval only
preempts a thread running bytecode, not one blocked in native code). Which engine call it is has not been identified
on GB10; the per-phase counters and `watchdog_events` (05c1366) attribute the fires.

With the process proxy (VLLM_DSPARK_REMOTE_ASYNC_PROXY=process) the release needs nothing from the engine process: the
proxy waits on host words in its own interpreter and publishes with one C call. Such engine-side waits then last one
round (worker RTT), not the watchdog deadline (CPU test `test_process_proxy_releases_while_the_engine_holds_the_gil`
and a 20-round run of it on Controller: engine thread blocked in a GIL-holding native wait, worker delay 40 ms, every round
released by the proxy in 41.8-49.2 ms, 20/20 rounds ok, 0 watchdog fires; the same probe on the thread proxy is
released only by the watchdog, `test_thread_proxy_needs_the_watchdog_when_the_engine_holds_the_gil`). It also removes the liveness reason for
`CUDA_MODULE_LOADING=EAGER` (S9 F1): a module load behind a parked stream now waits one round instead of deadlocking.
EAGER stays in the configs until a GB10 run shows LAZY + process proxy clean.

## 3. Buffer lifetimes

| buffer | written by | read by | reuse rule |
|---|---|---|---|
| shared region / hostflag page | — | — | mapped and registered for the life of the process; `close()` and the atexit hook fail every round far ahead but never unmap or unregister (the stream may still copy during teardown) |
| STEP message rows (msg) | S, round t (2); nothing else (R3, S11 P-1) | proxy: `client.step` (NIC send) in round t | round t+1's D2H runs only after release(t). The proxy only READS the staging: the STEP headers and entries go into a private buffer and resync rows into another, and the message is sent as [headers, rows] (`client.step(rows=)` / `RpcClient.request_parts`: one sendmsg on TCP, straight into ofl's send slot on RDMA, so still one copy). Released by the proxy: it is done with the rows (an ofl timeout breaks the QP before the call returns, so no late NIC read). Released by the watchdog while the proxy is still inside the round: a check of the claim right before the send skips the network (`fallback_preempted`); past it the rows may change under the send; either way the proxy loses the claim, discards the reply and marks every slot dirty, so the next round re-announces whole windows (NEW_GEN) and the worker's state is replaced. Before R3 the late proxy wrote its headers (and resync rows) into the staging the stream had refilled for round t+1: with a fresh request in round t+1 those rows became the worker's window and every later draft of it was wrong while the status said OK (S11 Q1; `tests/test_late_proxy.py`) |
| gather metadata | S, round t, into slot `t % K` (K = 64; process: the descriptor ring size) | proxy, round t | slot reused by round t+K. Process proxy: the engine writes descriptor/metadata slot t+K only after `desc_tail >= t` (the proxy finished round t); a proxy K rounds behind is declared lost. Thread proxy: a proxy K rounds behind treats the ring of that round's slots as unknown (`watchdog_skipped_unknown_ring`) |
| staged resync windows | S, a round that requested them | proxy, the same round (copied into its private resync rows) | rewritten only by a later round's staging, i.e. after this round's release; same watchdog case as the rows |
| reply area | proxy publish / watchdog / fail-all, round t, only after `gathered >= t` | S (6) | `gathered >= t` means S finished round t-1's copy (stream order 6 < 1 of the next round); the proxy also checks the explicit acknowledgment `consumed >= t-1` and stops (fail-all, target-only) if it is missing. Exception: a fail-all at proxy death or close may overwrite the area while a previous round's copy runs; that copy can mix a usable reply with zeros. Both ranks still receive the same device tensor (7), so the decision stays identical across ranks, and drafts never change outputs (target verification); accepted |
| bcast (device) | S (6) | S (8), NCCL (7) | stream / event order |
| descriptor slot t % K (process) | engine thread | proxy | as the metadata slot |

### The late proxy (fixed in 05c1366)

Before this change a proxy that reached round t after the watchdog had completed it (the GB10 fires, or any stall
longer than the deadline) still read round t's rows and metadata. By then S had passed release(t) and could already
have written round t+1's gather into the same staging, so the round's ring bookkeeping and the rows sent to the worker
could belong to a later round. Now the proxy checks the claim word right after `gathered >= t`: a round already taken
is skipped (no row read, no network), its ring bookkeeping runs from its own metadata slot, and its slots resend their
windows. CPU tests: process proxy SIGSTOPped past the watchdog then resumed, thread proxy made late on three rounds;
both exact afterwards. A round preempted mid-flight (watchdog fires while the proxy is inside the step) keeps the
previous rule (claim lost: reply discarded, every slot dirty), and since R3 it never writes into the staging (table
above).

## 4. Completion ownership (one winner)

Every round t has exactly one completion: the claim word (`dsv41_hf_claim`, CAS) is taken by the proxy's publish, the
watchdog's fail-release or a fail-all (FAR_AHEAD), and only the winner writes the reply and releases. Releases are
monotonic (cyclic max), so nothing moves the word backwards. The proxy checks the claim before it reads staging (§3)
and at publish. The watchdog deadline runs from the first poll that sees `gathered > released` for that round, i.e.
from GPU readiness, not from enqueue (hostflag.c `wd_main`).

The one exception to "nothing moves a word backwards" is the respawn's reset (§5), done only with no proxy process
alive and no stream operation pending, in an order that never shows the running watchdog a pending round.

## 5. The process proxy's life (R3: S11 P-2, P-3, P-4, P-6)

* **Start** (`round.make_remote_round`): a proxy that does not start (exit, start failure, 60 s start timeout) gives
  synchronous rounds with `async_fallback_reason`, as a refused self-test does; a strict start
  (`VLLM_DSPARK_REMOTE_STRICT_START`) raises instead. Every refusal (`streamsync.release_hand_off`) stops the
  watchdog, unregisters the region from CUDA and unmaps it (both fds, the mapping). `StreamStuckError` still
  re-raises without touching the region.
* **Lost** (`RemoteRound._proxy_health`, once per submit): exited; descriptor ring would overflow; `stall_fires()`
  (3) watchdog fires with no round finished (`C_DONE` still); heartbeat (`C_HEARTBEAT`, written on every housekeeping
  poll, <= 100 ms apart while waiting) older than `heartbeat_limit_ms()` = max(2 x deadline, timeout + 250 ms) =
  400 ms in production, judged only when the engine thread itself ran in that time. A stall costs at most about three
  deadlines (before: a ring of 64). On loss: fail-all far ahead, SIGKILL without waiting, `async_dead`, a loss record.
* **Respawn** (`_respawn_step`, once per submit, never blocking): when the lost process is reaped, an event recorded
  at the loss has completed (the stream passed every operation of the lost proxy's rounds) and 1 s has passed, every
  hand-off word is reset to the last round handed out (`_reset_hand_off`: gathered, claimed, released, consumed, then
  the control words) and a new proxy is started at base + 1 with a new session id (the worker drops its windows; the
  empty slot table re-announces every slot, NEW_GEN). Target-only rounds meanwhile copy the failed reply and gather
  nothing, so requests in flight draft from the rows gathered since the respawn (partial window, refilled within W
  positions). Budget `VLLM_DSPARK_REMOTE_ASYNC_RESPAWNS` (3), restored after `..._RESPAWN_RESET_S` (3600 s) of
  service; spent: target-only for the rest of the run, one error line. Stats: `async_dead`, `async_respawns`,
  `async_respawn`, `async_proxy_losses`, counters `proxy_lost_{exit,ring,fires,heartbeat}`, `fallback_proxy_dead`.
* **Shutdown**: the speculator's finalizer calls `close()` when vLLM drops it (after its device synchronize) or at
  exit: proxy stopped (stats flushed), every round failed far ahead; the hand-off's atexit fail-ahead stays.

## 6. Open items

* LAZY module loading with the process proxy: GB10 test (EAGER is no longer needed for liveness, only measured).
* Proxy wait policy on GB10: the default `ASYNC_SPIN_US=1000, ASYNC_NAP_US=0` spins one core from submit until the
  GPU writes `gathered` (through every forward, prefill included). Compare `SPIN_US=50, NAP_US=50` (1 us timer slack
  is now set in the proxy) on TTFT and decode.
* Rounds counted `rounds_ok` by a proxy that then lost the claim (mid-flight preemption) are failed rounds for the
  engine; the counters keep the reviewed accounting, `watchdog_preempted` says how many.
