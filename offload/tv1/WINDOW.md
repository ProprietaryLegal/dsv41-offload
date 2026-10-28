# TV1 coordinator sheet — review before any real action

## Round 9 — isolated installed oracle; compact CLOSED

The candidate at dae5f9e STARTED in `TV1-candidate-W-20261027T145246Z`.
Its installed-oracle allocation then failed with CUDA OOM beside the live
candidate server; cmd0153 passed 8GiB MemAvailable but Normal had no free
buddy blocks at order>=7. This run remains INCOMPLETE, restore 6/6 / watch PASS.
The separate compact run at 141057Z failed greedy at token64 with 0.75 nats,
its second greedy-gate failure. **Compact is CLOSED / NOT A CANDIDATE.**
Do not schedule another compact run or change any greedy/near-tie rule.

Candidate startup now runs the unchanged 288-case numerical matrix in a
short-lived container on each SM121 node **after stop and overlay staging,
before JIT/model/server startup**. `start_exp.sh` has one candidate-only hook;
unset/OFF takes its original path. `oracle_prestart.py` uses the same pinned
image by immutable ID and the exact assembled engine/overlay mounts and
candidate environment. No weights are loaded and no network listener is
started. Every mounted K1/candidate source is hashed before importing CUDA.
The oracle still imports the installed metadata adapter/dense/top-k, requires
SM121 and exact eager/changed-input replay equality in every original case.

The high-offset allocation is exactly 2,401,735,488 bytes (2.237GiB):
`(47000+5120)*46080 +37440 +64*132`. The first live byte is
`47000*46080+37440=2,165,797,440`, above signed-int32. The span includes
all 5120 logical pages through 327680 positions and the existing one-page guard;
the last referenced page requires 2,401,689,408 bytes. Reducing it to the
~43MB of records would remove the addressing test. No geometry was reduced.
Sources: `candidate_cases.py:15-19`, `gb10_oracle.py:31-46`; the unchanged
scorer promotes physical page IDs to int64 before multiplying at
`candidate_kernel.py:37-38`.
The oracle checks CUDA free bytes plus 512MiB scratch before allocating once;
allocation exceptions report the required span and retain the original cause.

Nested SSH uses the existing scoped start owner and immutable deadline.
The isolated container uses `--rm --name dsv41-serve --network none` so
existing experiment-stop/quiet recovery covers a lost launch reply. Each
node must have no existing server/container or compute process before run,
and no container/compute process afterward. A stale receipt refuses replay.
Only both exact receipts permit startup to continue; the runner retrieves
and validates `~/dsv41-offload-run/tv1-oracle-<run>.json` after serving startup
and saves `arms/s2-candidate/prestart-oracles.json`. It no longer executes
the oracle in a live server. Either oracle failure prevents server startup.
The two 120s remote bounds consume the existing 240s allowance; candidate
estimate 5055s and 85/86/98 ceilings are unchanged. Hardware execution and
duration remain unqualified until the coordinator's next reviewed window.
All older instructions below to run the installed oracle after candidate
startup are superseded by this section. Actual-forward qualification still
runs in the real candidate engine before timing; all decision gates remain.

### Round 9 pre-HOLD memory admission

Candidate preflight reads `/proc/buddyinfo`, the MemAvailable line and the
host's actual base page size on both nodes, with stdlib only. It saves each
raw probe and assessment as `oracle-memory-prehold-<host>[.probe].json`
**before opening HOLD**. The existing 8GiB floor remains; admission also
requires node0 Normal's largest already-free physical buddy block to be at
least order7 (128 base pages). DMA availability cannot substitute. Missing,
ambiguous or unsupported topology/page geometry refuses. The recorded
cmd0153 snapshot fails this new gate despite 9386836kB available.

This is a conservative physical-contiguity screen, not a measurement of the
largest possible CUDA virtual allocation. CUDA maps many physical blocks;
a 2.4GB virtual span does not require one 2.4GB physical block. Order7 is an
explicit admission policy selected to reject the observed fragmentation,
not a proven NVIDIA driver granule. The receipt reports the measured
largest-free block in bytes/order and `cuda_allocation_guaranteed=false`.
Reclaim/compaction and concurrent host activity can change it; the screen
may refuse a host that would recover after server stop. No compaction,
cache drop or GPU probe beside production is performed. Only the actual
isolated allocation and all 288 exact oracle cases can qualify SM121.
The next reviewed window must measure this path on both nodes and stop on
allocation, equality, receipt, absence or time failure. No budget extension.

## Round 8 — candidate initialization repair (source/CPU review)

The candidate window `TV1-candidate-W-20261027T124301Z` at5b2de7c failed during
s2 memory profiling on both ranks: `consumer_layer()` called the current-config
accessor outside its load-time scope. That run remains INCOMPLETE. Its complete
engine logs and restored6/6/watchPASS evidence are read only.

The resolver now uses the active forward context's `no_compile_layers`, which
contains the same owning-module map even for dummy profiling. Uniqueness,
layers24/28/32/36 refusals and per-op caching remain. Read
[CANDIDATE-INIT.md](CANDIDATE-INIT.md) for the image path/line proof across
profile/dummy, CUDA-graph capture and prefill/decode serving. No candidate
scoring, cuts, near-tie rule, schedule or recovery change is included.
The CPU regression executes the composed forward method with the image's real
forward-context functions and no current config for all four consumers in
profile/dummy, FULL/PIECEWISE capture, prefill and decode. Original5b2de7c helper
bytes fail all20 cases at the observed assertion; fixed code passes. See the
complete accessor audit in CANDIDATE-INIT.md. These are binding checks only.
GB10 initialization and numerical qualification still require coordinator evidence.

Candidate preflight now locally builds/verifies the candidate overlay and runs
`candidate_init_smoke.check()` **before Supervisor startup, transport or HOLD**.
Its isolated30s CPU child imports the three real composed helper files using
installed package names, supplies a synthetic owning-layer map with metadata
None and no current config, and proves four bindings plus cached repeats.
Receipt `candidate-init-source-smoke.json` records helper hashes/input-lock,
`SOURCE/CPU ONLY`, and `cuda_or_engine_init_qualified=false`. Import/binding,
receipt IO or timeout failure ends preflight before any host action. Compact's
existing source smoke is unchanged. No CUDA context, model, kernel or
extra service is started; this cannot replace actual GB10 initialization.

The next coordinator candidate window must show s2 completes memory profiling,
kernel warmup, KV initialization and graph capture on **both ranks**, with no
owning-layer assertion/refusal and correct runtime hashes. Then the unchanged
288-case installed oracle, actual-forward receipt matrix, near-tie greedy and
timing gates must pass. Preserve full bounded engine logs before every stop
and startup-failure removal; strict W8 six IDENTICAL plus post-close watch stay
mandatory. Nothing in this source repair qualifies the old incomplete window.

## Prior registration and coordinator controls

Verdict: Round7 registers `tv1-r7-nbw-near-tie` for NEW experiments, using the unchanged NBW checker.
The compact window at2aea31f started ON but stopped under its strict identity rule; it remains NOT A CANDIDATE.
Two default-OFF experimental arms remain for source/CPU review. The original
scorer has Ampere reference tests; new engine mapping/guards and GB10 numerics
remain unqualified. Only Deepseek / coordinator runs this runner or preflight.
The TV1 lane has not contacted either Spark or any serving endpoint.
The first compact window stopped at ON startup; the later strict window stopped at greedy reference0.
Neither is a completed comparison. See the Round7 registration below before another window.

## One feature per HOLD

Run candidate and compact in SEPARATE windows. Each uses exactly three fresh starts:
OFF → feature → OFF (C/arm/C). Five starts exceed the85min planning ceiling
even with the reduced cell sets (arithmetic below). There is no dynamic shortening. No combined arm is included. There is no
prefill-sync arm (no removable current indexer sync), no graph arm (measured gate
failed), and no mHC/head/sampler port.

Each start has:

1. Source, mount, env, model, KV, Engram and remote worker identity checks against
   the current OPS1 production snapshot. K1 327680, 6 GiB/rank, 2,502,600 KV tokens, async
   process proxy, remote Engram and remote capacity drafter remain the baseline.
2. Candidate starts ONLY: on each SM121 rank, run gb10_oracle.py against the ACTUAL
   installed DeepGEMM dense implementation and installed top-k. The pinned288 production-derived
   cases, changed pages/lengths/query/candidates under graph replay, followed by
   both-rank actual-metadata qualification before timing (Round2 below). Every
   selected index must match bit for bit. Any mismatch STOPs immediately.
3. Six indexed W5/K1 greedy checks through `offload/launch/windows/nbw_cells.py`, unchanged.
   Experimental text must be identical or pass that checker's first-divergence rule:
   both tokens in target top-20, gap<=0.501 nats, better token<=0.501 nats from the top.
   `greedy-references.json` still pins the original K1 reference bytes/hashes.
   Partial/final receipts and summaries retain position, margin_nats and gap_to_top_nats for each
   diagnosed divergence, including a failed check. No prompt, response text or tensor is saved.
4. A discarded 32K/64-token warmup. Paired public-corpus cells: candidate uses150K B1
   plus short B1; compact uses32K B4 plus short B1. Each forces 512 output tokens (ignore_eos=true), same T=1/top_p=1/seed42/priority10.
   Prompt cuts are built once and reused; fresh per-start salts force cold prefills.
5. Each unprofiled B1 cell runs four independent sequential requests; B4 runs one
   batch of four. Each request forces512 tokens. Pool raw counters across these
   four requests, using distinct per-repetition cold salts. Unprofiled cells first; then those same two cells with decode-only captures.
   The torch profiler is configured in every arm, idle during control cells and
   greedy checks. Both ranks, 16-iteration captures, SP1 visibility/overhead rules.
6. Flushed proxy counters and worker progress around every cell, telemetry, memory
   minima, trace hashes and bounded copies. Finish checks and clean profile dirs.

The decision metric is **unprofiled server request-decode milliseconds per
speculative request-round**. Recompute `1000 * sum(decode_seconds) / sum(rounds)`
from the retained raw metrics snapshots across all four control requests. These
are request durations and request-round counters, including terminal effects;
they are not an engine iteration timer. Pool accepted/proposed/round counters by
summing numerators/denominators, never by averaging request acceptance fractions.
Require exactly four completed/decode-histogram requests per control cell; missing,
reset, nonfinite or foreign/unfinished-request counters STOP qualification.

Profiles remain diagnostics/coverage gates: the analyzer refuses missing rank,
wrong active request count, fewer than eight complete steps, mixed/prefill or
tail-only captures, hidden graph kernels, coverage<90%, or overhead>10%. Their GPU
period cannot substitute for the server decision metric. SSE events remain a
client proxy, never CUDA timings. TTFT and actual aggregate decode tok/s remain
separate observations, with no averaging of sequential B1 request spans as if
simultaneous. No generated output text is retained.

## Bounds and estimate

Round4 supersedes earlier full-matrix estimates. Both fixed three-start schedules
fit **ESTIMATED <=85min including12min recovery**. Candidate84.25min; compact57min.
The post-close15min shared watch is outside HOLD (99.25/72min estimated elapsed
including watch). These are extrapolations; this lane has run no TV1 GB10 window.

MEASURED inputs, read from saved records only:

- Spark starts ~388–404s (coordinator Round4 brief; S49's W8/PF1 records corroborate
  ~388–407s). Use420s/start ESTIMATED, without assuming any faster ON startup.
- NBW-W3 = campaign `NBW-W-20261026T231445Z/s{1..5}-*-short-b1.json`:
  short TTFT1.447–2.409s, decode52.649–56.820tok/s, total27.999–30.804s at1500 outputs.
  Scaling only decode to511 intervals yields ESTIMATED ~11–13s at512 outputs;
  allow15s/control request,20s/profiled short request.
- Same saved window's32k-b1: TTFT14.813–16.737s; decode49.308–54.568tok/s.
  Warmup64 outputs: ESTIMATED30s (includes allowance). Six unchanged300-token
  K1 references: ESTIMATED60s total, not a new measured identity duration.
- Same saved window's32k-b3: maximum total103.051s at1500 outputs, TTFT up to52.101s,
  per-stream decode down to24.427tok/s. B4 is an extrapolation, not a measured B4
  timing: allow100s for four512-token controls together and115s for their profile.
- Cold150K TTFT ~69–71s (SP1 report / PF1 off-327); allow85s/control request,
  95s/profiled request, including512 outputs and profiling allowance.

All seconds below are **ESTIMATED planning allowances**, except qualification's
735s is its unchanged configured estimate. Installed-oracle runtime is UNMEASURED;
240s budgets both existing120s command bounds. Exceeding a bound STOPs; none of
these allowances waives a receipt, equality check, memory floor or complete request.

| Per start | Candidate window | Compact window |
|---|---:|---:|
| Fresh startup |420s|420s|
| Six K1 identities |60s|60s|
| Warmup64 outputs |30s|30s|
| Four short B1 controls |60s|60s|
| Target controls: four sequential150K B1 / four concurrent32K B4 |340s|100s|
| Short plus target profiles |20+95=115s|20+115=135s|
| Source/mount/env checks, counters, bounded trace copies and finish allowance |75s|75s|
| Common start block |1100s|880s|
| Candidate ON only: full real-forward qualification+both installed oracles |735+240=975s|0s|

One-time snapshot/staging30s plus prompt-build30s; recovery720s. Prompt building
runs after the first start, so first-start admission explicitly reserves its30s
as still-pending work (not as time already spent before admission). Candidate:
`60 + 3*1100 + 975 + 720 = 5055s =84.25min`. Compact:
`60 + 3*880 + 720 =3420s =57min`. Five-start alternatives are
`60+5*1100+2*975+720=8230s=137.17min` and
`60+5*880+720=5180s=86.33min`, respectively. Candidate has only45s spare
outside its allowances; slippage can refuse its next start. No completion guarantee.

- Before EVERY start, runner requires ALL remaining start blocks, including any
  qualification/oracles AND recovery720s, to fit original HOLD+85. An overrun
  cannot borrow the closing OFF, cut requests, or admit a partial result.
- A stricter measurement deadline is HOLD+73 (85 minus12min recovery). Driver
  deadlines, qualification, full repeated-cell checks and supervised call timeouts
  use this bound. Expiry enters the unchanged fenced W8 recovery path.
- Original submission+85, Supervisor abort+86 and immutable recovery+98 remain
  outer safety ceilings. Normal planning aims for restored/closed by+85;
  actual emergency recovery still shares the original+98 deadline. No renewal.
- Candidate's8GiB floor independently prevents admission at ordinary5–7GiB
  availability. Installed/actual-forward qualification remains complete, including
  B2/B3/B4/150K traffic outside the reduced timing matrix.
- Existing PF1 guards: >=5GiB before start; >=4GiB always, no post-ready swap growth
  >256MiB, fresh 1s telemetry, no foreign demand, STOP file/deadline guards.
- Captures require >=4.5GiB; a skipped required decode capture STOPs TV1.
- Candidate oracle requires >=8GiB before its short extra CUDA process. No model
  weights are read by it; temporary synthetic tensors are bounded and not recorded.
- SP1 trace caps/copy/O_DIRECT hashes/removal are unchanged. Uncollected traces are
  retained as evidence; only this window's empty directories are removed.

## Commands (COORDINATOR ONLY)

Use a reviewed clean checkout at the final TV1 commit (or its integration descendant).
No push, merge, deploy or production promotion is performed by the runner.

```
source ~/.config/ndm/paths.sh
# Dry plan: no network, files, preflight or service operations.
"$HARNESS_PYTHON" -m offload.launch.windows.tv1_window --feature candidate --dry-run
"$HARNESS_PYTHON" -m offload.launch.windows.tv1_window --feature compact --dry-run

# Coordinator only, after independent review and a reserved window:
WINDOW_COORDINATOR=Deepseek/coordinator "$HARNESS_PYTHON" -m offload.launch.windows.tv1_window \
  --feature candidate --execute --expected-experiment-commit <40-char-reviewed-SHA> \
  --remote-repo dsv41-deploy-offload --stage-bundle \
  --out /opt/dsv41/campaigns/DSV41-OFFLOAD/TV1-CANDIDATE-W-<UTC>

# After successful restoration, CPU-only analysis:
"$HARNESS_PYTHON" -m offload.tv1.analyze \
  --window /opt/dsv41/campaigns/DSV41-OFFLOAD/TV1-CANDIDATE-W-<UTC> \
  --output /opt/dsv41/campaigns/DSV41-OFFLOAD/TV1-CANDIDATE-reading-<UTC>.json
```

Replace candidate with compact for its separate HOLD. Execution refuses a missing
coordinator identity, existing output, dirty tracked source, wrong full commit, or
production checkout name. Preflight and every service command live inside --execute.
All source build operations are local and hash-pinned. Generated overlays are staged
only in the isolated experiment checkout and passed through a guard before stop.sh.
Duplicate, shadowing, writable, missing or tampered feature mounts are refused.

## Pre-registered reading (Round7 changes experimental greedy only)

Hard gates: six ordered K1 experimental checks under `tv1-r7-nbw-near-tie`; candidate installed top-k oracle on
both ranks; expected source/env/mounts; no new fallback/watchdog/communication error;
all required profiles valid; W8 restored; post-close shared watch PASS. No numerical tolerance or overlap criterion
can substitute for a bitwise selected-index mismatch. A failed reference is a STOP.

OFF drift: per-cell maximum/minimum server request-round time <=2%; maximum/minimum TTFT <=3%.
For fixed C/arm/C, compare the ONE ON with EACH OFF separately. Both comparisons
must pass every applicable threshold, with no averaging of the controls. This is
one ON observation bracketed by two OFF observations; it does not establish
independent repeatability across ON starts. Missing closing OFF, an extra start,
a mismatched feature/cell set, or a partial run cannot qualify.

- Candidate: >=2% lower150K B1 unprofiled server request-round time in BOTH comparisons.
- Compact: >=2% lower32K B4 server request-round time AND >=5% higher
  unprofiled32K B4 aggregate client decode rate in BOTH comparisons.
- Every cell: no >2% decode round or client-rate regression; no >3% TTFT regression.
- Acceptance is pooled over >=4 requests per cell in EACH start; accepted/request-round
  and proposed counts are reported. The proxy `(1+pooled accepted/round)/server ms`
  may not regress>2% against EACH OFF's pooled acceptance separately. This is a supporting
  acceptance check, not measured aggregate tok/s; compact also needs its actual5%
  B4 throughput gain. Faster rounds cannot hide lost useful output.

Below threshold or noisy = HOLD, no speed claim. Passing = WINDOW_PASS_FOR_REVIEW,
never automatic promotion. Compact's explicit small-vector host synchronization and
eager shortened forwards may lose; B1 has no cross-request padding benefit. Six
short-reference checks are not proof of every long-context/mixed/grammar behavior;
those batches retain the original widths, and broader adoption needs their own review.

## Restoration (S48)

After any possible HOLD write, success, STOP, crash or interrupt, the runner uses the
existing W8/restore.sh in finally: `start` if production may have stopped, then `check`,
then `close`. `check` counts EXACTLY indices 0,1,2,3,4,5 once each on IDENTICAL lines;
six copies of one line, a missing line, any DIVERGES, nonzero rc or missing completion
marker fails restoration. Closing is withheld after a failed check.

Required markers: K1 START VERIFIED; six `greedy K1-post vs restored: i IDENTICAL n`;
PRODUCTION HOLD OK; CHECK: DONE; CLOSE: window closed. Failure leaves HOLD and exits2;
the coordinator inspects cmd/restore logs and runs the original W8 sequence. Neither
this lane nor this patch changes W8, production code/config, image, unit or worker.

## CPU rehearsal

The real TV1Live + PF1 Supervisor and SP1 capture collection run through fake commands,
with real bounded CPU telemetry helpers only. Tests cover both fixed three-start sequences,
late HOLD failure, start/source/identity/mount/oracle/greedy failure and duplicate-index
restoration rejection. Existing PF1/SP1 regression suites remain part of final checks.

## Round 2 — S51 FIX 1 (supersedes inherited recovery description)

TV1 owns HOLD-open and each experiment start before submission, using the reviewed
KL1/NBW lock, persistent ABORT and PID/starttime/boot identity pattern. A TV1 SSH
shim registers EVERY nested SSH command on its destination; SCP uses the same
registered transport. This covers recipe stop/run commands on both Sparks, even
when the original SSH reply is lost. No process-name reaper is used by TV1.
Recovery fences each phase on sparkhead, sparkworker-wired and x86host before reaping its
recorded groups, and requires explicit absence proof. Failed fence/reap or JIT
absence stops before W8 and retains HOLD. W8's local process group is owned, and
its start/check/close SSH hops receive separate persistent ownership phases.
Each recovery phase must be fenced and absent before the next step. The original
W8 script and strict six-index IDENTICAL check are unchanged.
W8 close's final `rm -f HOLD` is deferred by the phase's private transport shim;
all original close checks still run. Only successful W8 close PLUS all-node
fence/absence evidence permits the final, one-shot unlink of the unchanged owned
HOLD under the registry lock. A lost close reply or failed close cleanup leaves
HOLD present. Docker/JIT query failure is a hard failure, never an empty listing.

## Round 2 — S51 FIX 2: current production and exact OFF control

TV1 preflight pins full production HEAD
`596ba1cf4e3b11be29b41a57e7f69a5d8efa4799` and tag
`prod-k1-327-ops1-20261026`, including the warmup launch files in the clean-file
check. The rollback receipt records this current ledger. TV1's config reads
production's OFFLOAD environment/mounts directly; it no longer sources PF1/SP1's
historical REQUIRED override. Engram remains `VLLM_ENGRAM_REMOTE_REQUIRED=0` in
EVERY arm. All container Config.Env entries are captured and compared exactly,
except the fixed TV1 switches. Image, worker, engine/target metadata, Engram JIT,
K1 source hashes, mount hashes/read-only mode, port-only argv change and reviewed
idle profiler settings remain mandatory. PF1's historical comparator and tests
are unchanged. Executable frozen OPS1 fixtures pass; REQUIRED=1 control fails.

## Round 2 — S51 FIX 3: actual candidate consumers and row mapping

The six-shape oracle is superseded by the exact288-case production-derived matrix
in `ROUND2-CANDIDATE.md`: layers24/28/32/36, B1–B4 (includingB3), raw boundaries
through327680, K1 page64/stride46080/offset37440, physical offsets beyond2GiB,
installed metadata expansion, empty/tail/padded candidates and tie/near-tie cases.
Both workers must pass exact installed dense/top-k eager and changed-input graph
replay. The runner validates every case identity/parameter; counts alone fail.
The high-offset allocation requires **8GiB available** before the extra process,
including the4GiB serving floor and runtime margin. Ordinary5–7GiB availability
is insufficient: STOP rather than attempt the allocation or weaken the gate.

Before candidate timing, additional unprofiled public-corpus qualification runs
B1 150K, B2/B3/B4 32K and B4 150K (ESTIMATED735s per candidate start). Actual
forwards compare the SAME patched row mapping against installed dense/top-k.
Both ranks must supply the specific consumer/B/context receipts. Unchecked real
decode buckets temporarily request eager; checked buckets return to production
graph dispatch before timing. Native ragged packed Q/varlen-index paths retain
dense fallback; unsupported FP4/DCP/PCP/native-next_n>2 refuse. No tolerance or
overlap criterion replaces equality. Cut tolerance and all exclusions are explicit
in `ROUND2-CANDIDATE.md`. No actual327680 or SM121 result is claimed here.

The previous75–90min estimate excludes this new qualification and is superseded:
candidate windows add about24.5min over two ON starts, plus unmeasured installed
oracle time. Completion within the fixed HOLD ceiling is uncertain; admission
includes the full735s qualification budget. STOP on insufficient time/headroom,
retain current production, and return an incomplete review result. The lane does
not authorize an extended HOLD or an alternative live window.

## Round 2 — S51 FIX 4: enforced recovery ceiling

One frozen HOLD-based budget sets submission+85min, abort+86min, recovery+98min.
Abort moved four minutes earlier to reserve12min for fence/reap/JIT/W8. No HOLD
renewal/reopening exists. Even a lost HOLD-open reply establishes the budget in
finally; an unreadable reply uses the earlier admission time conservatively.
Every unsupervised cleanup, proof, profiler cleanup and W8 step receives at most
the remaining shared time. Exhaustion prevents the next command and reports
failed restoration; it never grants another full per-step timeout.

Each mutation's remote registration carries a lease bounded by its own local
call timeout and the shared deadline. Nested SSH inherits that exact lease. A
run-owned guardian ends stragglers, including TERM-ignoring children, before the
shared ceiling; guardian failures are logged and fail the absence gate. Final
HOLD unlink checks its lease while holding the persistent fence lock, immediately
before unlinking. A persistent commit-cancel fence refuses a delayed first commit
or retry. A lost final acknowledgement triggers fenced retention of the ORIGINAL
HOLD bytes/timestamp (no renewal, no overwrite of another owner). If transport or
budget exhaustion prevents confirming retention, the report explicitly requires
coordinator inspection; it cannot claim RESTORED.

MEASURED CPU fake-clock tests exhaust the shared budget between recovery steps,
prevent check/close submission after exhaustion, reject already-expired recovery,
and preserve the original lost-reply HOLD timestamp. Owned process tests exercise
late registration/final unlink after lease expiry and a TERM-ignoring straggler.
These are control-flow tests, not measured GB10 startup/recovery times.
Final unlink admission additionally reserves90s for the entire commit/cancel/
retention sequence. Each of those calls is capped20s, leaving at least30s margin
even when both acknowledgements use their full allowance. Below90s, HOLD removal
is refused before submission. A fake-clock regression covers20s-left refusal and
90s-left lost commit/cancel replies followed by confirmed original-HOLD retention.

## Round 3 — shared cgroup ownership (supersedes Round 2 FIX1 ownership)

TV1 now imports OWN1's `scoped_remote.py`, `hold_admission.py`,
`spark_quiet.py` and `post_close_watch.py` from cleared OWN1 `1aaaa59`,
merged via `offload-x86host` `8192430`. The private TV1 registry is deleted.
Registration occurs inside real cgroup-v2 user scopes before business commands;
GNU timeout, setsid, nested receiving-host SSH/SCP and private tmux placeholders
remain contained. Persistent ABORT prevents delayed registration/retry. Scopes
are an ownership boundary for this reviewed chain; arbitrary new daemon units
are outside it. No process-name or `/proc/*/environ` scan remains in TV1.

HOLD admission drains first, then publishes complete bytes by exclusive hard
link under the shared global lock. A nonce plus saved inode/hash identifies the
winning admission. Every arm's first stop checks that exact receipt under the
same lock. S55's opening ceiling and placeholder lease freeze together only at
the one-shot HOLD submission, after preflight. Publication outside that opening
ceiling refuses before any target stop; the accepted placeholder lease covers
HOLD+98. Neither the phase nor the HOLD can be renewed.

Recovery fences ALL destinations (sparkhead, sparkworker-wired, x86host, localhost)
before reaping ANY; separate absence proof is required on every destination.
It stops only the reviewed experiment via its own scoped stop.sh and uses the
shared quiet helper to wait for named serving/experiment containers, anonymous
expert/Engram JIT containers and GPU compute absence. Query/inspection failures
retain HOLD. Unknown JIT/compute work is waited for, never killed. W8 itself and
all its receiving-host hops run through the same shared scopes. After W8 close,
all phases are fenced/proved absent again before the one-shot HOLD unlink.
Lost close acknowledgements cancel the commit and retain the original HOLD.
Six distinct indexed IDENTICAL lines remain mandatory; no near-tie exception.

The original frozen +85/+86/+98 limits and 90s final transaction reserve stay.
Each cleanup/control/W8 call is capped to the original recovery remainder.
Recovery scopes expire at recovery-minus90s; no new recovery scope starts after
that reserve. At exhaustion the runner reports failure and requires coordinator
inspection. Runtime and real startup durations remain unmeasured by this lane.

After a verified close, the CLI runs the shared 15-minute production watch:
30s model/priority20 one-token probes, busy/unknown health is inconclusive,
two confirmed failures admit at most one scoped W8 recovery. A failed watch
fails the window's review eligibility. This is additional monitoring after the
unchanged HOLD deadline, not an extension of that HOLD. Nothing runs on import
or dry-run. Historical Round2 ownership prose describes retired code.

## Round 3 — decision metric (LANES.md LESSON)

Server duration/round deltas and >=4-request acceptance replace profiled GPU
period as the primary decision. The raw source definitions are image
`vllm/v1/metrics/stats.py:565` (last minus first token duration),
`vllm/v1/metrics/loggers.py:1224` (request histogram observation), and
`vllm/v1/spec_decode/metrics.py:41` (one observed request draft increments rounds).
B1 uses four sequential512-token requests; B4 uses four concurrent512-token
requests. Every trial's request completion count, decode histogram count and
speculative counters are checked and retained. Candidate targets150K B1, compact
targets32K B4 round time plus aggregate tok/s. Round4 below supersedes the two-ON schedule; paired prompts and all numerical
gates remain required.


## Round 4 — fixed feature cells, C/arm/C and budget fit

The current sections above supersede historical Round2/3 scheduling estimates.
`cells.order(feature)` fixes exactly warmup, short/target controls, short/target
profiles; no other timing cell is executed. `schedule.py` owns the frozen estimate
and remaining-schedule admission. The three starts, exact cell set and C/arm/C-v4
design are checked by the analyzer; the single ON must beat each OFF separately.
Candidate requires >=2% lower150K B1 server ms/request-round against each OFF;
compact requires >=2% lower32K B4 server ms/request-round AND >=5% greater32K B4
aggregate tok/s against each OFF. Both cells retain2% round/client and3% TTFT
regression guards, OFF drift bounds and >=4-request raw-counter pooling.
Four independent B1 controls or one concurrent B4 group force512 outputs/request.
Matching profiles remain diagnostics with the same coverage/overhead/rank gates.

CPU regressions cover exact feature cell sets/three starts, full schedules at
84.25/57min, five-start refusal arithmetic, admission at/just beyond the remaining
budget, startup slippage refusing a second start before any stop/run command,
measurement timeout clamping at+73, and rejection when an averaged OFF would pass
but one individual OFF comparison fails. Shared OWN1, OPS1, candidate geometry,
patches, exact oracle validators and original85/86/98 deadline module are unchanged.
This sheet requires independent review and coordinator prerequisites before execution;
source/CPU evidence is not GB10 numerical, speed or live-recovery clearance.

## Round 5 — restore evidence (item 1)

Recovery command logs are now `cmd/restore-NNNN-<phase>.log`, with a serial that
increases across all local recovery calls. Repeated fence/reap/absence passes
before and after W8 close preserve separate observations. CPU end-to-end window
rehearsals use the real `TV1Live.local()` subprocess/log path on a fresh output
directory for successful completion, STOP before experiment start and failed ON
start; each reaches verified close without a filename collision. Remote transport
and W8 responses in those tests are owned CPU fixtures, not live recovery proof.

## Round 5 — complete startup logs before removal (item 2)

Before each replacement experiment start, preserve the previous experiment's logs
from BOTH ranks; the first start has no prior experiment. After a failed start,
recovery first fences/proves absence of all launchers, then preserves both ranks
before `experiment-stop`. Successful final recovery captures before that stop too.
The reader checks the experiment-only `/sp1-profile` mount, pins the immutable
Docker container ID, and uses `docker logs --timestamps ID` without a tail filter.
It does not import torch/CUDA, export container environment or read production logs.

Evidence goes to `engine-logs/NNNN-<reason>/<host>.log` and its JSON receipt. Up to
8MiB/rank is retained. Larger streams retain first/last4MiB and their stored hash,
plus a SHA256 of the entire consumed stream, total byte count and explicit
`truncated=true, complete=false`. Read timeout20s, transport<=30s/rank and every
call remains inside the original measurement/recovery deadline. Timeout/transport,
missing container, wrong mount, Docker error and malformed evidence are recorded,
never passed as complete logs. Such failures stop a subsequent experiment. During
recovery both rank reads are attempted and errors cannot block W8; production
restoration is reported separately and the window becomes INCOMPLETE, ineligible
for analysis. No recovery deadline is extended. Log I/O is part of the existing
75s/start overhead allowance (ESTIMATED, unmeasured); slower I/O can refuse the
next start under remaining-schedule admission. Recovery reads consume the existing
shared recovery allowance.

Keep these local campaign logs private: they can contain model diagnostic output.
Do not paste them into git or the report. Report only the exception and source
location after checking both rank files and completeness/hash receipts.

## Round 5 — compact initialization (item 3)

`INIT-DIAGNOSIS.md` records the actual startup path and two reproduced source
blockers: missing `os` import, then the compact guard's rejection of production's
plain confidence depth policy. Initialization includes real scheduler warmup,
not only dummy profiling/graph capture. Both standalone feature overlays and the
combined overlay bind os; candidate's numerical/decision logic is unchanged.
Compact now composes only with plain conf, retaining the original masks and
rejection bookkeeping. Other depth compositions still refuse. OFF is unchanged.

Before any remote command or HOLD, compact preflight locally builds/verifies its
pinned overlay and runs the source/CPU smoke; inspect
`compact-init-source-smoke.json`. This proves the named Python bindings/guards
against the recorded configuration, not full engine or CUDA initialization.
There is no safe full-model init probe alongside production under this lane's
headroom/access limits. Full both-rank startup logs must still be reviewed in the
next coordinator window. A corrected source does not establish that the lost
traceback had only these causes, or that GB10 startup/greedy/performance passes.


## Round 7 — new registration before new data

Registration id: **tv1-r7-nbw-near-tie**. Coordinator decision in the binding Round7
brief, 2026-10-27, after observing the earlier strict compact STOP and before any
new Round7 data. This openly changes the experimental greedy gate. The historical
strict registration is preserved by the source history through2aea31f and by the
unchanged read-only `TV1-compact-W-20261027T085424Z` evidence. That window started
compact, diverged on K1 reference0, stopped and restored6/6; it remains
**STOP / NOT A CANDIDATE**. Its divergence was not diagnosed as a near-tie, and no
such retrospective finding is claimed. The new analyzer refuses old/unregistered
summaries; do not relabel or reanalyze old evidence under this rule.

Every experimental start (OFF and ON) delegates to `nbw_cells.greedy()` and its
`near_tie()` implementation, with no copied checker or changed threshold. The same
six W5 prompts, temperature0, max_tokens300, thinking=False and K1 reference text
bytes remain. Text stays in memory. At the FIRST differing token, NBW tokenizes
both outputs and submits a one-token target completion for their shared prefix,
requesting top20 logprobs. Pass requires both tokens in that top20, absolute
logprob gap<=0.501 nats and the better token<=0.501 nats below the top. A missing
token, larger gap, prefix/length-only difference or failed diagnostic request STOPs.
This check diagnoses numerical behavior; it does not prove universal equivalence.

`greedy.json` carries registration_id/greedy_rule and the unchanged NBW numeric
checks schema. Each diagnosed divergence records token_index (zero-based),
margin_nats and gap_to_top_nats, plus near_tie true/false. NBW rounds recorded gaps
to4 decimals AFTER its full-precision decision; that decision remains authoritative.
A missing top20 token records null margins and fails. An unavailable diagnostic
records an explicit diagnostic_error; no position/gap is fabricated. Partial
receipts survive STOP into `summary.json` at starts[arm].greedy, including the
failed arm before any cells. Summaries and analysis name the new rule. Both the
new wrapper and unchanged NBW source are pinned before HOLD.

Feature patches, OFF/control composition, cells, target/candidate exact oracles,
performance thresholds, C/arm/C schedule and5055s/3420s estimates, immutable
85/86/98 ceilings and restoration are unchanged. Diagnostic RPCs consume the
existing greedy-command timeout/identity allowance; slow checks may STOP within
that budget. W8/restore.sh and its six-index IDENTICAL gate remain strict.

## Round 10 — NDM's 0.9-nat near-tie limit (registration `tv1-r10-near-tie-0p9`)

NDM 2026-10-27 ~18:20Z: "You can loosen the nats to .9". Read narrowly, this changes ONE number for NEW TV1 windows:
the NBW first-divergence threshold 0.501 → **0.901 nats** (0.9 plus the same 1e-3 rounding allowance). Everything else
is the Round 7 rule: same six W5 prompts and K1 reference bytes, temperature 0, max_tokens 300, thinking off; at the
first differing token both tokens must be in the target's top-20, their gap ≤ 0.901 nats and the better one ≤ 0.901
nats below the top; a missing token, larger gap, prefix/length-only difference or failed diagnostic request STOPs.
W8 restore stays strict six IDENTICAL. `nbw_cells.greedy(..., near_tie_nats=)` / `near_tie(..., threshold=)` take the
threshold explicitly; their DEFAULT stays NBW's 0.501, so NBW and every other caller are unchanged. `greedy.json`,
the cell CLI preview and the window summary carry `registration_id`, `greedy_rule` and (in greedy.json)
`near_tie_nats` = 0.901; `validate()` and the analyzer refuse any other registration or threshold.

Old evidence is NOT reclassified. `TV1-compact-W-20261027T141057Z` (registered tv1-r7) stopped at K1 prompt 0, token
64, margin 0.75 nats, and remains a STOP under its own registration. A NEW compact window under r10 would admit that
same 0.75-nat flip; its speed has never been measured, so it needs a new window. A 1.0-nat flip still STOPs.
