Status 2026-10-23: R1 decomposition delivered; process-proxy remedy measured W4 and async promoted W5. Historical synchronous/thread investigations remain evidence. See [live roadmap](../../documents/plans/roadmap.md) and [measured record](../results/COMPARISON.md); campaign review decisions are in `LANES.md` / lane reports outside the repo.

# R1 round-trip experiments

These tools observe x86host or run isolated CPU-only RDMA loopback on x86host. They do not open a CUDA context,
contact the live worker, change services, or send traffic to a Spark. Results belong outside the repository.

## Passive observation

Copy `observe_x86host.sh` to `~/dsv41-offload/r1/` on x86host. Run detached:

```bash
source ~/.config/ndm/paths.sh
tmux new-session -d -s dsv41-r1-observe \
  'bash ~/dsv41-offload/r1/observe_x86host.sh ~/dsv41-offload/r1/sample-next 45'
```

The script requires exactly one live O2 worker, discovers the direct-link **netdevs by IP** (the `rocep...` names
are RDMA devices), and captures process/thread CPU, scheduling delay, IRQs, governors, idle states, GPU telemetry,
memory and worker counter snapshots. It fails on missing prerequisites. Sample for <=60 seconds at a time and
correlate with worker STEP deltas: a quiet interval cannot establish active contention. `schedstat` time unit is ns.

## Copy, pinning and poll/event prototype

Use existing `g++` and the pinned O2 transport; no new packages. Copy `ofl_loopback.cpp`, `run_x86host.sh` and
`../transport/ofl.h` to x86host's `~/dsv41-offload/r1/`. On x86host:

```bash
source ~/.config/ndm/paths.sh
cp ~/dsv41-offload/o2/lib/libofl.so ~/dsv41-offload/r1/libofl.so
sha256sum ~/dsv41-offload/r1/libofl.so
g++ -std=c++17 -O3 -Wall -Wextra -Werror -pthread \
  -I ~/dsv41-offload/r1 ~/dsv41-offload/r1/ofl_loopback.cpp \
  -L ~/dsv41-offload/r1 -Wl,-rpath,/home/operator/dsv41-offload/r1 -lofl \
  -o ~/dsv41-offload/r1/ofl_loopback
tmux new-session -d -s dsv41-r1-loopback 'bash ~/dsv41-offload/r1/run_x86host.sh'
```

The binary accepts `PORT ITERS PERIOD_US SPIN_US CLIENT_CPU SERVER_CPU`. It binds only `10.202.28.1`, restricts
ports to 18670–18679, starts its own server, verifies three changing markers and the generation/length of every
reply, and destroys both client and server on success/failure. A port conflict fails; never stop another owner.
It uses ABBA order for the ordinary-copy vs `ofl_client_payload_buf` variants, 12 warmups per case, exact sample
percentiles, 184,416 / 737,568 / 15,728,864-byte requests and 24-byte replies. The two small sizes also run at a
42-ms request cadence. Total resident payload/transport buffers stay below 0.2 GiB. No model tensors are used.

Pinning applies only to the benchmark caller and its own server connection thread, after server creation.
The default example chooses otherwise quiet, separate physical cores 2/3; recheck the collector before reuse.
The `cpu_cores` metric includes both benchmark threads over each case's wall interval; other processes are excluded.

This is **NIC loopback**, not a measurement of either physical cable or Spark staging. The in-place variant assumes
the producer can write the transport buffer directly. On a Spark that additionally requires CUDA pin registration,
ownership across reconnects, and a safe lifetime through SEND completion; no such integration is claimed here.
O2's live library SHA256 during R1: `9f11b6e8d4f459c07ac35745eec95e5e54fcf705d5085e454faab7386520356d`.

## Coordinator's next Spark experiment

Use the existing `C-capacity-async.env` with the **current calibrated C-capacity costs**, identical KV allocation,
depth policy and workload. The checked-in async template explicitly contains placeholder costs: do not compare it
unchanged to calibrated C. Confirm the log says async self-test passed and stats have `async: true`; otherwise
the run is synchronous fallback. Only the coordinator restarts/benchmarks Sparks.

Useful phase markers for a coordinator profiling overlay:

1. CUDA events just before/after the fused gather and its D2H copies, recorded on the engine stream. Resolve after
   the existing gather event, or on a later completed round; never add a global sync just for timing.
2. CPU timestamps before enqueue, before/after the existing `event.synchronize`, before/after STEP packing,
   before/after `ofl_client_call`, after reply validation and after reply H2D enqueue.
3. CUDA events around H2D and TP broadcast on **both ranks** (the existing rank-0 round statistic excludes broadcast
   and measures H2D enqueue, not H2D completion).
4. A CPU marker when model_runner returns its already-created AsyncOutput, and CPU/CUDA markers around the next
   target graph launch. This shows the scheduling work that local drafting overlaps and synchronous RPC serializes.
5. On a separate released worker, CUDA events around staging H2D, insert graph, draft graph and reply D2H; retain
   per-round CPU handling time. Do not profile or send new sessions to the live worker.

Store paired per-round deltas, request count, inserted rows, `n_out` and context bucket. Existing Stats percentiles
are upper edges of logarithmic bins (8/octave, up to 9.05% wide); subtracting two of them cannot measure a phase.
Means can be subtracted only for matching sample populations and nested timing scopes. Async mode moves the host
wait to the helper; it still orders GPU work after the remote reply and cannot verify dependent draft tokens early.
