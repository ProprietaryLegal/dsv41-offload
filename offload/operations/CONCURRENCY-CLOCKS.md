# P3 controlled concurrency and clock plans (OPS1)

**Plans only. No setting, unit, GPU clock, protocol limit, service or request was
changed or exercised by OPS1.** The coordinator owns future exclusive windows.
Finish D2 fairness fixes, native outcome checks and server-side target profiling
first, per expert continuation §10–11. Do not combine concurrency, depth-policy,
prefill-cadence or clock changes in one attribution arm.

## Current limits and the prerequisite work

Recorded production K1-327680 has pool **2,502,600** nominal tokens, per-request
cap **327,680**, four sequences, chunk budget4096 and target TP2; x86host is the
K2 TP2 worker via the async process proxy. K1-524288 / pool2,820,756 is **HOLD**.
These configuration/capacity values are **MEASURED** in the K1 startup and S18
records, not proof that every mix with a smaller sum fits.

Source audit below refers to the current lane base / production worker release
fb68c52. These are deterministic source facts, not hardware measurements:

| Boundary | Current source | Required before B6; then B8 |
|---|---|---|
| Admission / request length | `serve.sh` MAX_SEQS=4; cap/pool from K1 profile | Keep cap and KV allocation fixed; allocator-aware prompt+reasoning+answer+tool reserves, unique-prefix cold cases, fragmentation and partial-prefix bounds. Do not set max_num_seqs alone |
| Protocol session | `worker/server.py:expected_hello` max_slots4 / max_rows_per_step512; Spark `_hello` advertises max_num_seqs ×128 | Coordinated versioned worker/proxy contract for slots6/8 and rows768/1024. Current worker rejects larger HELLO; treat target-only fallback as failure, not successful scale-up |
| Request transport | worker/server.py MAX_REQ=16 MiB; remote_dspark/o1_adapter.py DEFAULT_MAX_REQ=16 MiB; TCP MAX_FRAME=64 MiB | Full-window resync exceeds16 MiB at B6/B8. Resize **both** registered endpoints/pools with a checked bound (32 MiB covers the current B8 frame), or independently review bounded multi-frame resync and its state semantics. Preserve inflight=1 ordering and safe retries |
| Reply transport | server.py MAX_RESP / o1_adapter.py DEFAULT_MAX_RESP=64 KiB; current ID/conf replies <1KiB | Validate complete frame lengths and every failure reply; resize engine broadcast/shared reply from M,N. A later probabilistic q reply is a different protocol/capacity experiment; do not import today's tiny reply budget |
| Spark gather | remote_dspark/fused_gather.py MAX_REQ=4, explicitly unrolled prefix offsets; gather.py auto-disables above4 | Extend/test the fused kernel to6/8 including ragged rows and cancellation, or make an explicit separately timed torch-gather control. No accidental fallback or false fusion claim. Preserve resync ring and late-proxy metadata ownership |
| Shared region | remote_dspark/shmregion.py RegionLayout scales M and max_rows; RING64 descriptor slots | Recompute actual memfd, pinned/registered buffers, device staging and /dev/shm limits; preserve64 **round descriptors**, not64 request slots. Exercise overflow/stall/respawn boundaries; no unbounded pinned growth |
| Drafter slots / graphs | worker/server.py allocates slots8 but HELLO still4; capture only1..exp.max_slots, k5, n_out1..5; TPDrafter scratch slot=self.n_slots | Capture B1..6 then1..8 on both ranks, retain scratch slot separately, all insertion buckets8/16/32/64/128 and cancellation/resync tails. Eight allocated slots does not establish eight-request support |
| Drafter compute / collectives | worker/tp2.py max_rows32 coop scratch; comm_words() fixed32×hidden; backbone always B×k | B6×5=30 is inside the nominal32-row bound but needs oracle/shape tests. B8×5=40 exceeds it: coordinated scratch, kernel-plan, collective buffer and graph changes first. Reducing n_out affects the head, **not** the five-query backbone. Do not silently change the architecture/block length |
| Target dispatch / graphs | existing k schedule [1..2→5,3..4→4], mHC overlap max24, several dense fast paths max32; full-decode graphs | Installed dynamic schedule carries the last k forward beyond its last range; make B5..8 policy explicit anyway. At k4 the upper verify rows are30/40 for B6/B8 (k5:36/48), plus mixed prefill rows. Audit every dispatch, row limit and graph key on both ranks; no silent fallback-based gain |
| Timeouts / recovery | remote RPC50ms, derived async watchdog max(4×50,200)=200ms; pair default2000ms (source, not re-read live) | Measure p95/p99 RPC+staging/queue/worker latency and tail full-resync before selecting bounds. Preserve bounded target-only recovery, stale-reply/cancel/session checks, GPU flocks and process respawn3/3600s. Raising deadlines alone can stall target streams; do not conceal timeout fallback |

Source files: [worker server](../dspark/worker/server.py), [TP2 model](../dspark/worker/tp2.py),
[wire](../dspark/spark/remote_dspark/wire.py), [transport adapter](../dspark/spark/remote_dspark/o1_adapter.py),
[gather](../dspark/spark/remote_dspark/gather.py), [fused gather](../dspark/spark/remote_dspark/fused_gather.py),
[shared region](../dspark/spark/remote_dspark/shmregion.py), [reply layout](../dspark/spark/remote_dspark/roundcore.py).
The installed dynamic schedule helper is in the read-only engine snapshot at
`vllm/v1/spec_decode/dynamic/utils.py`. These are prerequisites for a future
implementation/review lane, not an instruction to modify the live pair now.

### Frame and memory planning arithmetic

**ESTIMATED allocation/traffic requirements**, calculated on CPU from the current
wire/RegionLayout formulas; not observed wire traffic or measured peak memory.
Row bytes =3 taps ×5120 ×2=30,720; worst per-step window rows=M×128.
STEP frame =32+align64(32+32M)+rows×30720. At k5, ID/conf wire reply
=32 app header+32 reply header+8M+40M; margins add20M.
Shared reply broadcast =4×(2+M+2M×5). Full process region includes message,
resync windows, metadata, descriptor ring and alignment; it excludes separate
CUDA staging, worker buffers and graph pools.

| Requests M | Full-window rows | STEP frame bytes | ID/conf reply bytes (with margin) | Shared broadcast bytes | Process region bytes | Backbone rows k5 |
|---:|---:|---:|---:|---:|---:|---:|
|4|512|15,728,864|256 (336)|184|31,612,928|20|
|6|768|23,593,248|352 (472)|272|47,349,760|30|
|8|1024|31,457,632|448 (608)|360|63,086,592|40|

Steady incremental steps may transfer fewer rows; cancellation/reconnect/full
prefix-hit resync must still fit. The current16 MiB limit is16,777,216 bytes;
32 MiB is33,554,432. Size bounds for both links, duplicate pinned pools,
independent peer allocations and Engram shared RAM/NIC contention must be
rechecked; no Spark cudaMalloc-backed arbitrary RDMA registration.

### Admission examples, before allocator/block overhead

**ESTIMATED reserves**, illustrative not permission to serve: reserve16,384
completion/tool tokens per request, including hidden reasoning. For the current
327680 profile, 8×(150000+16384)=1,331,072 and
6×(300000+16384)=1,898,304 fit the nominal pool and individual cap.
8×(300000+16384)=2,531,072 exceeds2,502,600 and must be refused even though
each request is within the cap. Eight full327680 requests total2,621,440 and
also exceed it. A live allocator/block budget can reject an apparently fitting
sum; do not count shared prefix savings unless actually observed.

On the **held**524288 profile, 6×(500000+16384)=3,098,304 already exceeds
2,820,756, and eight is4,131,072. A literal500000-token prompt has only24,288
total tokens left before reasoning, visible output, tools and any extra
formatting/reserves. All prompt lengths must come from actual `/tokenize`,
not nominal string length. No500K request is permitted at today's327680 cap.
No1M experiment is part of this plan.

## Staged B4 / B6 / B8 measurement

1. **Freeze and review.** Save the exact target/manifests, worker/Markov/arm,
   overlay hashes, graph keys, k policy, clocks/power, env and pool/cap.
   Record outstanding D2/native/target-profile results. Register a separate
   candidate version and tested rollback to K1-327680/fb68c52. No existing
   production pair/solo grant is expanded by this plan.
2. **CPU and worker qualification first.** Exercise slot7/8 including generation
   wrap guards, maximum frames, wrong-size/overflow replies, ragged B1..B8,
   simultaneous full128-window resync, prefix-hit/cancel/reuse, late replies,
   missing peer and restart. Required independent review of the larger transport
   and B8>32-row compute change precedes any Spark request. Under a coordinator
   pair window, verify both ranks' graph/eager fixed-input oracle at B5/B6, then
   B7/B8; log dispatch/capture keys, traffic, memory and collective errors.
3. **Warm and screen B6 only.** Exclusive drained window; pause training and
   background producers, retain all required admitted work. Keep target
   T=1/top_p=1, reasoning effort, target weights/quantization, output budgets,
   priority10 and prefill cadence/throttle unchanged. Use explicit B1..B6
   graph warm-up; require fresh actual worker/verify counts and shape evidence.
   First short/32K public workloads at B4/B6, including staggered arrivals and
   finish/cancel tails. A failing shape/memory/state/fairness row ends the arm.
4. **Repeated matched workload.** For each passing arm use the same fixed
   queue of24 request IDs/families and generation budgets at offered concurrency4
   then6 (later8), at least three interleaved control/candidate/control repeats.
   Include32K and150K cold salts, a separate warm-prefix case, and a mixed long
   prefill arriving during decode. Use actual prompt counts/cached tokens.
   Reserve output within the unchanged cap. Keep one matched full task with
   reasoning/answer/tool behavior as a semantic regression; fixed-budget
   microbenchmarks do not establish completed-task quality.
5. **B8 only after B6 passes.** Require the B8 scratch/collective/graph contract
   above and a measured memory margin on both Sparks/x86host cards before the
   same screen/repeated workload. Do not test all300K requests at B8 with the
   illustrated reserve. No forced swap, memory eviction of unrelated services,
   second full Spark deployment or unbounded admission.
6. **Fairness as a separate controlled arrival case.** First use a public
   legal-class priority10 probe against bounded auxiliary priority20 arrivals,
   with the installed scheduler priority semantics tested by D2-FIX1. Stop
   auxiliary submissions on legal demand; use its explicit active-job cancel/
   resume boundaries. Every admitted request must progress or report a bounded
   cancel, including long-prefill waiters and async prefix restores. Preserve
   SO1/S15/S19 failures as liveness regression cases. Do not enable cadence100000
   or0073/0074 or revive exclusion/winner policies as part of this arm.

### Timing and decision record

For each request report prompt/cached/uncached counts, hidden reasoning and
visible answer counts separately (unknown stays unknown), TTFT, first visible
answer, completion latency, delivered token rate, and p50/p95/p99/max
**inter-event stalls** between token-bearing SSE events. SSE gaps are delivery
measurements; multiple tokens per event and network buffering prevent equating
them with GPU rounds. First-token latency is separate from decode gaps.
Report sample counts; a p99 from too few events is not a robust tail estimate.

Also report aggregate delivered target tokens divided by common wall time
(first submission to last completion), decode-only throughput with an explicit
steady interval, per-request useful token rates and completed tasks/hour.
Rejected drafts are excluded. Server-side sampled round IDs/verify rows and
non-blocking GPU timing on **both ranks** separate target work, overlapped
worker time, queueing and host/graph gaps. Report accepted draft A excluding
bonus, and full-round C; R=(1+E[A])/C, keeping their request/round populations
matched. Longer accepted depth may increase C. Never infer a kernel saving
from an SSE median or sum nested timers as independent costs.

Before the window, coordinator records the maximum legal p95/p99 stalls and
maximum time without progress acceptable for the workload, based on its B4
baseline/SLO. This plan does not invent an owner-approved latency allowance.
A candidate cannot be promoted without that recorded budget. Report throughput
and the worst-request tails together, with paired request/family-clustered
uncertainty; do not pool correlated events as independent request samples.
Require a repeatable ~5% useful throughput/capacity gain, no semantic/state/
transport failures and no fairness/SLO breach. If throughput improves at a
latency cost, present both and retain B4 absent a preregistered tradeoff.
Unqualified/no gain is a valid close; never keep other required work waiting
indefinitely to manufacture a scheduling result.

Stop immediately on state divergence, stale/wrong reply, collective error,
new timeout/fallback/watchdog bursts, stalled admitted work, a thermal/power
limit breach, unexpected dispatch fallback, swap use or loss of the reviewed
memory margin. Start with ≥5GiB available on each Spark and require a margin
budget that remains above the campaign hard stop; do not relax the existing
1.5GiB emergency floor. Coordinator restores exact saved configuration and
production hold, validates health/greedy/accepted drafts, then releases HOLD.

## Separate 2150 MHz clock comparison

Lower priority, after the stable warm profile is chosen and without training
or concurrency/prefill/depth changes. `dsv41.env` records a **configured**2150MHz
cap and donor fallback2000. OPS1 did not read actual GPU clocks. A configured
cap is not a measured sustained frequency or owner approval of a new setting.

Coordinator first records the owner's approved clock/power/temperature envelope
for both Sparks, supported device clocks, actual current caps, sustained
clocks, voltage/power/temperature and throttle reasons. No alternative executes
without that approval. Candidate comparison: current2150 vs2000MHz **only if**
2000 is supported and within that envelope; no higher cap, voltage/power raise,
persistent change or x86host clock change. If constraints are not established,
close clock testing as not authorized, retaining2150. Donor benchmark gains
are UPSTREAM/PUBLISHED CLAIMS, never a result for these machines.

Use an interleaved2150→candidate→2150 sequence, same profile, actual prompt
counts, cold/warm cache class, reasoning/target sampling, worker release and
representative B1/B4 workload. Warm to a stable thermal state before collecting;
record stabilization interval (planning minimum five minutes, longer if
clocks/temperature still drift) and both ranks' sustained distributions,
throttling duty cycle, power, peak temperature and memory alongside full-task
throughput, TTFT and per-request p95/p99 stalls. Keep instrumentation and
ambient/load conditions comparable. A brief cold peak does not qualify.

Stop/revert to the saved2150 settings on instability, errors, unexpected
throttling/power/thermal behavior or violated envelope. Repeat the control
and validate production after restoring the saved nonpersistent cap on **both**
Sparks; never leave only one rank changed. Retain2150 unless a repeatable
whole-engine benefit within the workload/SLO and approved limits survives
repetition. Report no difference honestly; clocks do not change native
fidelity, acceptance semantics or the target model.
