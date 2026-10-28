Status 2026-10-23: S17 round 2 ADOPT FOR GB10 TEST; ordered isolated gates remain mandatory. 0078 absent, SO1 shelved; no production adoption asserted. See [live roadmap](../../documents/plans/roadmap.md) and [measured record](../results/COMPARISON.md); campaign review decisions are in `LANES.md` / lane reports outside the repo.

# UP1 coordinator GB10 qualification plan

UP2 supersedes the P2 replay recipe with [UP2-GB10-TEST-PLAN.md](UP2-GB10-TEST-PLAN.md):
reserved successor progress, full real suffix, optional rank audit, low-cap150/300K
first, high-cap400/450/500K separately. This older plan remains the other UP1
arm reference; its historical production rollback identity must be refreshed.

Only the coordinator runs this plan, after independent merge review and an idle reserved window under COMMON.md.
UP1 itself used no GPU, Spark traffic, SSH, restart, window, JIT build, model download or service change.
None of the pass lines below has been exercised on GB10 by this lane.

## Baselines and invariants

Use the same legal-v2a manifest, target quant, current x86host TP2 worker/checkpoint, sampling and seed, thinking
high/effort75, request priority10,4 sequences,6GiB KV/rank,327680 context,4096 batch and current speculation policy.
Use existing representative LOCAL legal-writing/review prompts; never replace them with donor code or repeated
filler to obtain a win. Keep raw responses and proprietary telemetry local/outside git. No weight downloads.

A = D-kv6-async-process-nap-lazy. K = D-kv6-async-pack (K1 reference). C = D-kv6-async-up1-control (all UP1 OFF).
First qualify A/K using K1's plan if not already done on this exact build. C must equal K operationally before
any candidate can be credited. C keeps K1's Engram bytes and JIT identity; P0 checks the gated Python files.
Whole-overlay rollback is K only. Whole-experiment rollback is the recorded **prod-async production configuration
(main 3c1dc6b)**, with its saved pre-window launch environment and worker state; K is not production rollback.

Save exact overlay/input/source manifests, all live env flags and launch arguments, CPU affinity by actual
thread/TID, GPU clocks/temperatures, memory and NIC counters for BOTH Sparks and x86host's two allocated3090s.
Verify remote DSpark capacity+async process mode, proxy_torch_loaded=false, no local DSpark tensors/KV groups,
and required remote Engram on both ranks. No latency improvement may count a watchdog fallback or missing work.
Warm and cold prefix/Engram conditions are separate cells; record actual occupied/token and delivered-token counts.

Stop immediately for corrupt/stale bytes or taps, rank disagreement, NaN/invalid logits, hang, watchdog/fallback,
swap growth, service health failure, unexplained output distribution drift or each Spark MemAvailable<5GiB.
This is a conservative gate on our offload stack, not upstream's1.5GiB watchdog. Existing live commitments win.
No runtime change is promoted solely because startup fits or max context is configured.

## Local preparation and immutable stage checks

Run on Controller, CPU only, before the window:

```bash
source ~/.config/ndm/paths.sh
CUDA_VISIBLE_DEVICES= "$HARNESS_PYTHON" -m offload.upstream_adopt.catalogue --check --audit-image
CUDA_VISIBLE_DEVICES= "$HARNESS_PYTHON" -m offload.upstream_adopt.build_overlay
"$HARNESS_PYTHON" -m offload.upstream_adopt.make_configs --check
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 "$HARNESS_PYTHON" -m pytest -q offload/upstream_adopt/tests
"$HARNESS_PYTHON" offload/upstream_adopt/verify_runtime.py --emit > /opt/dsv41/campaigns/DSV41-OFFLOAD/UP1-runtime-check.py
```

The coordinator stages only these reviewed files and versioned experiment configs through the existing deployment
procedure. Do not copy the donor checkout or run donor launchers. Confirm the actual v5 image base hashes match
MANIFEST image_base_sha256s before mounts; the donor v24 source mirror is NOT the base image. Preserve K1's packing,
record formats and memory settings. Source configs locally to inspect their actual expansion before start.

All arms retain K1's O1 root, shim, extension sources and JIT identity. No UP1 Engram prebuild or private
source-root staging is required. The0067 Triton module warms during the isolated engine's startup dummy runs;
confirm eligible fusion and absence of request-time compilation before timing it.

Coordinator-only START, from the reviewed repo checkout on sparkhead in an ALREADY OPEN reserved window:

```bash
source ~/.config/ndm/paths.sh
bash offload/launch/start_exp.sh offload/launch/configs/D-kv6-async-up1-control.env
# Select a tested suffix below for the next arm, e.g. split, replay, vor.
# UP1_VOR=copy bash offload/launch/start_exp.sh offload/launch/configs/D-kv6-async-up1-vor.env
```

start_exp stops the current serve and stages mounts on both ranks: never run this outside the coordinator's window.
Use its actual experiment container name; pipe UP1-runtime-check.py into `docker exec -i CONTAINER python3 -` on
EACH rank. No torch/CUDA import occurs in that check. Validate process launch env separately; both ranks must see
the same replay/SWA switches. Each container path occurs once in OVERLAYS. K1's original O1 root is unchanged.
Keep upstream/display/NVFP4/0062/0065/0079/0080 switches OFF and local drafter absent.

## Compact run order and per-item pass lines

Order: K1 window 6 and its independent S18 results check first; P0 C/K; split alone, then split+cpu-pin
(the upstream +9% is joint), then replay. RP1 follows its own P0–P2 gates after K1/P0. K5, dual-ring and
0067 microchecks may run any time after P0. Combine only individually passing flags. 0078 is not scheduled.

Three paired repetitions, A/B/B/A ordering across starts where practical (at least a reference repeat at the end).
Fix existing clock caps, allow comparable cooling, and log thermal throttling. Compare per-round time AND accepted
and delivered target tokens, first generated TTFT, first VISIBLE answer, per-stream ITL p50/p95/p99, task completion,
prefill, memory and proxy/Engram phase waits. Report confidence/noise; do not average streams as independent reps.

| Order / arm | Why / measured upstream ONLY | Smallest useful GB10 measurement | Pass line / classification |
|---|---|---|---|
| P0 C vs K | Default-off equivalence; no upstream speed claim | Short T0 target output/logits +32K cold/warm,1/4 streams, mixed150K; remote faults/cancel/reuse | Identical short output for stable K/K controls; full request/finish accounting. No extra retries/fallbacks, corruption or hidden local drafter. Median round/decode change within1% after repeated pairs; otherwise investigate control before continuing. |
| P1 split | 0064+pin upstream sustained+9%, shares inseparable | C/split/C. Direct actual33,42,43,63 row MoE microchecks vs unchanged eager path, plus mixed150K with2 decoders and4-stream sustained90s; rows32/64 controls | Source dispatch hits split only33–63; correct real partial output at every row, no host expert-count sync in band. Exact per-row bitwise gate below for m = 33/42/43/63 is mandatory before the band-vs-tier rounding gate. For value, >=5% sustained OR >=10% shorter exposed mixed-step stall with <=1% decode loss and <=5% task regression; no cold/warm TTFT loss>1.5%. Run split+cpu-pin immediately after split alone; isolate pin-only P6 as well. |
| P2 replay | 0075 upstream+4.9–6.5% at478K/949K | C/replay/C, cold150K then256K/~300K if viable; final remainders1/127/128/129; actual budget leftovers<=128 beside decoders; 1/4 streams | Zero fallback_internal_error and zero replay-provenance stop strings (below). Both scheduler paths honor a reserved successor>128 or actual suffix replay. Count late-layer replay rows128→1 ONLY eligible intermediate chunks; all L20 KV-source writes retained. Remote STEP carries actual suffix, NEED_DRAFT=0 on partial prefill, full required window before drafting. >=5% repeatable cold-prefill/TTFT gain OR >=3% with verified reduction in replay work and task win; <=1% concurrent decoder loss; final KV/logit/needle gates. |
| P3 vor | 0067 joint with local0066 was-0.66%/round, NOT isolated | C/vor(COPY then SWA then both)/C; actual target group count in capacity mode;1/4 streams short + mixed arrival | Compare EVERY valid/lens/index bit against stock for padded rows,invalid slots,replay bounds,seq up to300K. Mixed/capture/noncausal/microbatch mismatches take stock; no hidden exception-disable. Fused eligible path actually launches once/step, replay copy only on change. Accept >=0.05ms repeatable metadata critical-path reduction AND no round regression>0.5%; below timing resolution = no proven gain, default OFF. |
| P4 k5 | Config0040 upstream+1.27% tok/s; remote costs differ | C/k5/C at3/4 streams,32K +150K, one prose/review/tool cell;1 stream control | Five actual proposals remain available;24 target rows supported; same output budget/settings. >=2% repeatable delivered-token gain beyond paired noise with no cell<-1.5%, no task loss>5%; preserve accepted/round statistics. Never import upstream local round-ms table. |
| P6 dual-ring | Upstream-1.20%/round WITH0051, shares not separated | Confirm the verified IPv4 RoCE v2 GID index3 on both f1 functions on both Sparks and both exact rank log lines below; C/dual-ring/C, tiny allgather micro +1/4-stream32K | Stop on missing rank log or NCCL-only fallback. Correct collective bytes and nonzero traffic on both intended functions; >=1% round reduction beyond paired noise, no decode or task loss>1.5%. No change to Spark↔x86host ports, second link or NCCL rails. If no valid second function, retain single, report inapplicable. |
| P6 cpu-pin | Upstream WITH0064 sustained+9%, shares not separated | C/cpu-pin/C then split with pin0/1. Inspect all actual thread affinities; async process proxy/Engram RPC waits and p99 ITL | >=2% repeatable round/sustained gain beyond noise with no individual cell<-1.5%, no new queue/Engram/proxy tail. Do not move RDMA poller from its known efficiency core by assumption. If inherited pin affects proxy negatively, retain original affinity. No CPU-frequency/service changes. |
| P7 core / all | Interaction validation, no additive gain assumed | Only individually passing flags. Core, then beneficial config flags, repeat 32K/150K/mixed and one full local legal task. | Preserve ALL prior numerical/resource gates; >=5% end-to-end win beyond noise or demonstrated useful capacity with<=5% speed loss. Each existing mixed stream >=control delivered rate; no task loss>5%; zero errors/fallbacks. Reject any unproven flag from final combination. |

Do not automatically execute the all config: it is a switchable screening artifact. If only one item wins, retain
only that item. For0067 a useful independently measured micro-saving may be carried without inventing a5% end-to-end
claim; retain OFF if the saving cannot be resolved. All percentage signs above are relative to matched C, not target-only.

Example measured screen from Controller AFTER the coordinator has started the chosen arm (never run by this lane):

```bash
source ~/.config/ndm/paths.sh
"$HARNESS_PYTHON" offload/bench/bench2.py --url http://HEAD_IP:8001 --label UP1-split --ctx 32000 150000 --workload legal --filler real --data-dir /opt/dsv41/data/bench-data --concurrency 1 4 --mixed --mixed-short-ctx 8000 --max-tokens 1500 --reps 3 --thinking --reasoning-effort high --priority 10 --temperature 1 --top-p 1 --seed 42 --cache-mode cold --out /opt/dsv41/campaigns/DSV41-OFFLOAD/UP1-split-cold.jsonl
```

Use the matched C command changing ONLY label/output path; repeat warm separately and retain a C/C noise floor.
Missing/insufficient existing corpus is a failure, never permission to download/private-upload or substitute filler.
For microchecks a matched fixed512-token generation budget is sufficient; final full task uses its original budget
and semantics. Never count rejected proposals, truncate thinking, or replace reasoning with nonthinking output.

## Numerical and state safety gates

0064 exact gate (S17 FIX-4a): for **m = 33/42/43/63**, the split output must equal the same rows computed
through one `exl3_smoe` call (where that row count fits) or per-row decode calls, **bit for bit**, on each rank.
Use identical inputs, expert IDs, weights, permutations, padding and activation limit; compare every real output
row before TP reduction. m=43/63 exceed the normal 42-row capacity, so use the per-row decode oracle rather than
raising the kernel's capacity. Keep rows32/64 as unchanged dispatch controls. Any bit mismatch stops the arm.
Only after this passes does band-vs-exllamav3-tier rounding remain for the existing numerical/distribution gate;
a BF16 tolerance or CPU mathematical comparison does not replace this exact gate. Teacher-force identical
prefixes at identical batch shapes for that subsequent gate.

For0067, require byte/index identity. For0075, ignore intentionally unconsumed intermediate one-row activations;
compare FINAL prefix/logits, all retained KV-source writes and the full final drafter window.

Dual-ring gate (S17 FIX-4b): the coordinator verified via read-only sysfs at **2026-10-23 15:39Z** that on
**both Sparks**, **rocep1s0f1 and roceP2p1s0f1** have an **IPv4 RoCE v2 GID at index 3**, on the
**10.202.26.x / 10.202.27.x** fabrics respectively. This lane did not repeat remote access. Record/reconfirm
that precondition before the arm: rdma.py uses one GID index for all devices. Stop unless both ranks log a line that
BEGINS with (prefix match: the real line continues `, gid 3, proxy on cpu …`; S17 round 2 note 1):

```text
EXL3 RDMA all-reduce: rank 0 of 2 on rocep1s0f1,roceP2p1s0f1
EXL3 RDMA all-reduce: rank 1 of 2 on rocep1s0f1,roceP2p1s0f1
```

A `RDMA all-reduce off … NCCL stays` warning means an NCCL-only fallback; stop and do not credit a dual-ring result.
Correct collective bytes and measured traffic on both intended functions are still required after the log gate.

Replay provenance gate (S17 FIX-4c): capture per-arm before/after engine stats and both rank logs, including the
**fallback_internal_error** counter and counts of **invalid replay tap provenance** and
**replay skip would remove taps**. Require zero new occurrences/increments throughout qualification. During fault
probes, which can raise fallback_internal_error for unrelated reasons, compare that counter with the control arm
under the same probe; there the two provenance log strings are the decisive signal (S17 round 2 note 2). Any occurrence is a stop: gather fails closed and the round silently runs target-only. A normal
request completion or plausible text cannot hide that failure; preserve the counter/log evidence.

Use fixed salts, partial prefix hits, A-B-A cache pressure and1/4 requests with K1's384-token cache granularity:

```bash
source ~/.config/ndm/paths.sh
"$HARNESS_PYTHON" offload/kvpack/prefix_identity.py --url http://HEAD_IP:8001 --n 32000 150000 --block 384 --out /opt/dsv41/campaigns/DSV41-OFFLOAD/UP1-prefix.json
"$HARNESS_PYTHON" offload/kvpack/needle.py --url http://HEAD_IP:8001 --ctx 32000 150000 300000 --needles 6 --concurrent 4 --reps 1 --out /opt/dsv41/campaigns/DSV41-OFFLOAD/UP1-needles.json
```

Needles require6/6 with zero foreign codes. In a short stable C/C case, byte-preserving variants must match token
IDs/finish reasons. Long-context/full-prefix comparisons must measure C/C floor first: require first-token
coarse top20+OTHER KL<=max(.005,2*C/C floor), common captured probability mass>=.99, and no material legal/task
regression. API top20 KL is a LOWER bound, not full-vocabulary validation. If ambiguous or outside the gate, capture
full logits/teacher-forced target distributions before accepting. Changed greedy text alone neither passes nor
fails a known rounding change; diagnose it. Never waive a systematic error as "nondeterminism".

Replay-specific faults: cancel after skipped chunk then reuse slot; preempt/prefix-hit bounded replay; worker
reconnect just before final chunk, timeout then resync, ragged partial prefill next to a rejected decode. No invalid
zero-filled tap may be certified. Both fused and torch gather must agree. MM/image and prompt-logprobs requests
must keep baseline replay; at least one MM exclusion smoke test and two actual prompt-logprobs requests.

## Prefix integration and rollback

RP1 has no overlapping source files, but enabling its connector introduces cache-load interactions: re-run RP1's
rank-byte equality, identities, corruption/load-cap/concurrent reload gates and the above replay/prefix probes.
Do not take its .45 donor cap, skip load synchronization or claim a cache hit based only on plausible text.

To disable one feature at next start, override its UP1_* variable with0 on that arm. To remove all UP1 files:

```bash
# Coordinator, inside the reserved window, on sparkhead reviewed checkout:
source ~/.config/ndm/paths.sh
bash offload/launch/start_exp.sh offload/launch/configs/D-kv6-async-pack.env
# Whole experiment: use the separate recorded PRODUCTION checkout/config (prod-async, main 3c1dc6b).
# Verify its saved pre-window revision/env first; do NOT run start.sh from the UP1 checkout.
# From that verified production checkout, after draining the experiment:
# ./stop.sh
# ./start.sh
# Restore the recorded x86host worker state via the coordinator window runbook as needed.
```

After the UP1-only rollback, verify BOTH nodes serve the K1 manifest bytes. After whole-experiment rollback,
verify BOTH nodes serve the recorded prod-async production bytes/config on :8000 (main 3c1dc6b). In either case,
verify original O1 root/JIT identity, process proxy and required remote Engram; complete a short reasoning/tool/structured-output smoke, health/queue drain and previous-workload check.
Close the reserved window through its existing procedure only when production is healthy and no request is lost.
No service unit, public port, model alias, local-law behavior or model file is changed by adopting this package.

Staging note (S17 round 2 note 3): the remote_dspark package in this overlay is a full pinned copy. Rebuild the
overlay from the merged offload-x86host tip before every window (`build_overlay` + `make_configs --check`); a later
remote_dspark change makes staging refuse on input-hash drift, never run stale code.
