# PP4 Round 8: physical-pair probe stall analysis

SOURCE / CPU ONLY, starting at `399d8b7`. No hardware run or qualification.
The coordinator's `PP4-399-pair-W-20261027T044707Z` correctness child had an
empty log and exceeded its 240-second subprocess limit. The earlier
`5fdc17a` child reached graph capture after roughly two minutes, then failed
with an empty graph and selector error. An empty log alone cannot locate the
new failure: the starting probe prints nothing, including on successful cases.
Round 6 added no progress output.

## Ranked hypotheses before the coordinator diagnostic

| Rank | Hypothesis | Fit to empty log and 240 seconds | Distinguishing evidence |
|---|---|---|---|
| 1 | CPU compilation, Triton launcher/module loading, or accumulated 32-case runtime exceeds the child bound | Strong. Even successful cases are silent; a previous run already spent about 120 seconds reaching its first capture. A cold specialization or a later case can consume the remaining budget. This is not evidence that 240 seconds is sufficient or that the run is healthy. | Traceback inside compiler/launcher build/load; phase output locates case, warmup position or capture. |
| 2 | First paired eager launch loads a CUDA module after its peer starts waiting | Plausible source gap. `kernel.warmup()` returns a CompiledKernel without initializing its CUDA handles in installed Triton. First `kernel[grid]` enters `.run` initialization. Loading rank 1 after rank 0 launches can serialize with outstanding work. This gap predates Round 6. | Stack in `_init_handles`/`load_binary` on the first eager position; distinguish from compilation. |
| 3 | CUDA stream/event/graph operation blocks in the driver | Plausible. Explicit rank-local streams repair the old default-stream error but graph enter also synchronizes/collects/empties cache. Event creation/record, replay and CPU copies precede or lie outside the host idle polling bound. | Last phase plus faulthandler stack inside graph entry/exit, event record, replay, allocation or copy. |
| 4 | Unprimed Triton specialization during capture or eager launch | Possible. Manual primed key is only rank/position/base stride/previous stride. Real Triton keys also include tensor dtype/alignment, constants/options and device. Warmup uses the same actual tensor arguments, so ordinary alignment should agree; capture inputs differ in the whole-worker path and require review. | Compiler stack during a capture phase or launch after a warmup said complete. A primed set membership alone cannot prove a real cache hit. |
| 5 | Peer flags/ack, epoch or P2P address visibility wedges execution | Weaker for a 240-second silent stall by itself: healthy waits are bounded and sticky errors are never cleared. A CUDA/driver failure can still make host operations block. | Stack at a CUDA wait with GPU samples; next result check should fail on error/sticky. Requires physical evidence. |
| 6 | Capture-intent guard or clearing itself spins | Very weak. Guard does finite Python set comparisons and per-device capture queries. `finally` clears intent on success or failure, with no lock/polling. | Wrong stream/rank produces a RuntimeError before launching, covered by CPU two-device tests. A driver stall in the capture query belongs to rank 3 above. |

These rankings are hypotheses, not physical findings. A later primary diagnostic
must supersede them explicitly rather than relabel a CPU check as hardware proof.

## Complete path and blocking boundaries

1. **Before qualify:** Python imports torch/Triton before the old main body.
   `checked_output`, coordinator/UUID/flock guard and thread setup precede
   `ext().enable_peer(0,1)` then `(1,0)`. Extension import/build and CUDA context
   creation or peer enable can block before fixture construction. Both peer
   directions are enabled before allocating buffers or launching peer kernels;
   no in-flight kernel should exist at this point. An already enabled peer is
   accepted but unrelated CUDA errors are not cleared.
2. **Fixtures/state:** deterministic CPU random tensors are copied to the two
   real devices, then both logical solo shards to device 0. Context creation
   allocates fixed candidate/error/epoch/inbox/q buffers, and update copies keys
   and mask. CPU random generation, allocator/driver calls and copies can block;
   there is no timeout or progress here in the starting probe.
3. **Eager warmup:** twice, each position calls `select((0,1),...)`. All argument
   views and both peers' pointer values are prepared before launch. For each
   missing key, rank 0 and then rank 1 call Triton warmup; only then are kernels
   launched in order `(0,1)`. The intent guard is inactive here. Each subsequent
   position can introduce constants/alignment/stride specializations while
   earlier asynchronous work is outstanding. No full-logit tensor is added.
   Compilation has no 5-second host bound. Warmup is compile-only, not a proof
   of loaded launch handles (local Triton 3.7 source `runtime/jit.py`,
   `compiler/compiler.py`).
4. **Idle before capture:** `wait_idle(peer.devs,5)` records an event on each
   current stream, then starts its monotonic deadline. Polling is bounded;
   device/current-stream lookup and event construction/record are outside that
   deadline. This helper neither checks selector error flags nor resets poison.
5. **Capture:** explicit streams are constructed for both physical devices.
   Each rank is captured separately, `(0,)` then `(1,)`; graph capture records
   the peer kernel without executing it. Intent is set before graph entry and
   cleared after exit, including exceptions. `select` requires the declared
   rank and its own current device stream to be capturing, and refuses missing
   primed keys. There is no peer handshake during correct capture. PyTorch
   graph enter includes synchronize/GC/empty-cache before capture begins;
   graph exit instantiates. Neither is bounded by `wait_idle`.
6. **Replay:** changed inputs and RNG counters are copied at fixed addresses,
   then graphs are launched `(1,0)` without an intervening wait. Both CUDA
   submissions must return for `wait_idle(peer.devs,5)` to begin. Reversed replay
   order is also the real worker's convention. A capture missing a peer launch
   should become sticky timeout, not a valid result. Solo reference runs only
   after the paired idle wait, then both result checks and exact CPU comparisons
   occur; copies/hash construction can also block in driver operations.
7. **Case teardown:** graph/context destruction and `gc.collect` can enter CUDA
   memory teardown. All 32 cases and 320 changed-input replays run before the
   old main writes a result. No per-case result is a full qualification.
8. **Deliberate timeout children:** both specializations are primed/launched,
   idle/result checked; one peer alone is then launched; idle and result must
   report sticky failure; a both-peer retry must remain failed. Process exit,
   not state clearing/reuse, is the teardown boundary. Their 60-second child
   limits are independent of correctness's 240 seconds.

## Kernel timing versus host timing

The brief calls `50_000_000` an iteration spin. Actual `_clock()` uses
`mov.u64 %globaltimer`; `_wait` compares elapsed nanoseconds with TIMEOUT.
Thus each flag/ack wait is approximately **50 ms**, not 50 million iterations.
The last retiring CTA per batch row executes it; other tiles do not wait for
unscheduled tiles. Error propagation/sticky state disables subsequent healthy
handshakes. No timeout adjustment or measured kernel change is justified by an
empty host log. The timer cannot bound compilation, module loading, arbitrary
driver waits, memory operations or a whole 32-case qualification.

The real K2 `_capture_one` already used explicit per-device capture streams
before Round 6; Round 6 added only intent around those captures. Its two eager
warmups run on side streams, wait back to the caller, then synchronize each
device. Capture guard failure clears intent. OFF/greedy spans are unchanged.
The FusedContext compilation/launch boundary is shared by probe and worker, so
any confirmed fix there needs both source/CPU regressions and fresh physical
pair qualification by the coordinator.

## Next evidence

Add flushed BEGIN/END/FAIL phase lines, including per-position eager/captured
submission and both replay ranks, and arm a repeating traceback timer before
torch import. The next coordinator run must retain the log and original stack
with elapsed/GPU samples. The diagnostics must leave kernels, numerical gates,
240/60-second child bounds, HOLD ownership and restoration unchanged. Hardware
requalification remains outstanding even if a deterministic source defect is
repaired.

## Coordinator diagnostic read at 06:32 UTC

Read-only inspection of the coordinator's completed
`/opt/dsv41/x86host-data/pp4/diag-20261027T062939Z` found `PROBE rc=0`.
`correctness.json` has all 32 ordered cases, both exact flags true and ten
changed-input replays each. The open log starts at 06:29:40Z and close starts
at 06:31:08Z; their 88-second span includes wrapper work and is not an exact
probe duration. No 420-second faulthandler dump occurred. Four sparse GPU
samples show 0% utilization; they cannot rule out kernels between samples.
The close log reports `PRODUCTION HOLD OK`; this lane did not restore or
independently test the service.

This primary diagnostic **does not pin the earlier hang's cause**. It is
consistent with cache/compilation differences or a transient, but proves neither.
The child JSON has no source commit/hash binding and is only the selector
portion, not the complete physical receipt (isolated timeout children,
whole-K2/N4 gate, strict W8 and post-close watch are separate). It must not
be used as a QUALIFIED admission or measurement budget receipt. No previous
failure is relabelled.

Read-only primary SHA256 values:

- correctness.log: `9bd3d3052d98a4e67939c2a5a523e799a0c2e75bcbec20eb99007719c96c0853`
- correctness.json: `0b18e289d65e0975cf1bee69949c7639f603968596b2b6aa0dd41e8c2bb990dc`
- smi.log: `82696c84355b9dc9218d4763823181189b23e5d24a72e78cb0f264c998d0653e`

The strongest bounded source correction to evaluate separately is preloading
both warmed kernels' CUDA handles before either peer launches. Installed
Triton proves compile-only warmup leaves handles lazy; it does not prove that
this caused the original timeout. Any such change is **HYPOTHESIS**, requiring
independent review and fresh coordinator physical-pair qualification.

## Separate HYPOTHESIS correction

`FusedContext.select` now reads the returned CompiledKernel's `.run` property
inside the same per-device warmup context, before recording the key primed.
Installed Triton's property initializes its launcher/module handles but does
not invoke the launcher. Both ranks finish this setup before either peer
kernel submission. None/noncallable handles refuse; load exceptions propagate
and the failed key is not primed. Already primed calls retain the original
two launches without repeated loading. Emulation and greedy paths are unchanged.

Three CPU regressions execute the actual select with a lazy-handle compiler
model: successful both-device initialization, rank0 load failure, rank1 load
failure. All fail against399d8b7's host ordering, all pass with the correction;
neither failure permits a peer launch. This proves a setup ordering change,
not that a real CUDA loader blocked the coordinator's earlier run. The complete
Triton kernel bytes and original worker bytes remain identical, and historical
receipt reconstruction removes exactly this host-only delta without rehashing
any receipt. The commit is labelled HYPOTHESIS so it can be reviewed separately
from the diagnostic instrumentation. No source staging or hardware trial was
performed by PP4.
