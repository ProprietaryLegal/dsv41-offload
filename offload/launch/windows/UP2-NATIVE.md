# UP2 native qualification feasibility — item 1, STOP/HOLD

2026-10-25. Lane `offload/up2-native`, inspected base
`39eebf2aab079280207618ea5436ea56bf915c08`. CPU source/artifact inspection only;
no engine execution, endpoint traffic, Spark/x86host access or GPU use.

**The full apparatus cannot be supplied as an API-only driver.** At least eight
checks (3–10 below) require substantive new engine/proxy control or observation
hooks, in addition to a shared metadata exporter. Together these cross scheduler,
KV/replay, engine-rank, process-proxy and transport boundaries. This exceeds a
single bounded launcher/CPU-rehearsal lane. The binding UP2-NATIVE brief says:
“If the feasibility table shows most checks need engine hooks larger than one
lane, stop after item 1, commit it, and report: the coordinator decides.” This
is that stop. Items 2–5 are not implemented/executed; no partial launcher, test
hook patch, live receipt or PASS claim is delivered. All 12 native checks remain
NOT RUN/HOLD. The runner, validator, names, floors and fail-closed default are
unchanged.

## Evidence inspected

Paths below are relative to the checkout unless absolute. These are source
observations, not proof the code executed on hardware.

- [Runner](up2_low.py): `sequence`, `Live.native`, `Live.start`, `Live.clean`.
  [Validator](up2_checks.py): `NATIVE_CHECKS`, `native`, `runtime`, `accounting`.
  [Protocol and window limits](UP2-LOW.md), [identity](UP2-IDENTITY.md),
  [CPU rehearsal](UP2-REHEARSAL.md), and
  [owning workload table](../../upstream_adopt/UP2-GB10-TEST-PLAN.md).
- `offload/upstream_adopt/patches/0075-{process,suffix,successor,ready-waiter}.patch`,
  composed `overlay/vllm/v1/core/sched/scheduler.py`,
  `overlay/vllm/models/deepseek_v41/{replay_skip.py,nvidia/model_state.py,nvidia/model.py}`.
- Under `offload/upstream_adopt/overlay/offload/dspark/spark/remote_dspark/`:
  `suffix_provenance.py`, `round.py`, `speculator.py`, `roundcore.py`, `gather.py`,
  `client.py`, `state.py`, `proxy_proc.py`, `config.py`, `mock_worker.py`.
  `CONTRACT-0075-DRAFT.md` and `SUCCESSOR.md` distinguish metadata validation
  from real target execution and real KV allocation.
- Read-only pinned source `/opt/dsv41/data/src-image/`: the DeepSeek
  `decoder_replay_layers.py`, NVIDIA `model.py` / `model_state.py`, scheduler
  and GPU-worker source search. The image's full-batch `write_batch_kv` precedes
  replay-row gathering; that call structure does not prove native write coverage.
- Campaign root `/opt/dsv41/campaigns/DSV41-OFFLOAD/`:
  all three `UP2-screen-diagnostic-20261025T{064342,075003,120635}Z` directories'
  await files, NOT-RUN records and request ledgers; successful diagnostic summary,
  tails and near-cap summaries; `UP2-screen-300K-20261025T131508Z/summary.json` and
  ledger; `S30.report.md` timing inputs. W13 captures are under `W13-UP2/`,
  not a `W13.report.md`; successful raw aggregate results are in the named SCREEN
  directories. No prompt, generated text or tensor is reproduced here.

MEASURED saved-record facts: the successful diagnostic has 28 send intents,
28 opened responses and 28 completions. Both arms' tails say q4092 ESTIMATED and
`tails_verified: false`; all 12 native checks say NOT RUN. Both near-cap API
requests report prompt327675, completion5, total327680, finish `length`. This
covers the configured mode's API result only. The two earlier diagnostics each
completed 28 requests but stopped at short-greedy comparison gates; neither is
native evidence. The 300K screen has 12/12 completed submissions and Stage-B
PASS: median ON/OFF0.9317489404, gain6.8251%, paired spread0.1051%. It remains
SCREEN/non-adoptable and supplies no missing native observations.

## Twelve-check feasibility table

“NEW” means proposed default-OFF instrumentation/control, **not an existing
switch or delivered patch**. “Bound not established” means no deterministic
native case completion or 900-second proof exists today. A short request's
completion does not turn that into a measured native bound. Every schedule must
be repeated OFF/ON with the same workload, mode and hooks; replay is the arm
variable. A missing subcase keeps the entire named check NOT RUN/HOLD.

| # / exact check name | Deterministic drive on experiment engine | Evidence needed and current gap | Fits 900s checkpoint / 75/90min window? |
|---|---|---|---|
| 1 `rank_q_kept_position_valid_anchor_owner_agreement` | Existing `VLLM_DSV41_REPLAY_AUDIT=1` on both ranks; solo32K warmup, then runner-derived tails. NEW bounded per-round metadata export at `_audit_suffix` / `suffix_agree`, tagged by case and engine round. | Existing `SuffixAudit.observe` constructs q/kept/first/valid/anchor/owner epoch and `agree` checks rank equality. Rows are discarded; only `replay_provenance_audited_rounds` increments. Need paired rank records/digests, equality counts and actual solo q, with observation coverage. Never log raw anchor token IDs. MAX_BATCH or a round count cannot supply q. | Small exporter is plausible within one lane; no native overhead measurement. Warmup can supply q without another long prefill, but current apparatus cannot extract it. Conditional fit, not proven. |
| 2 `final_need_draft_real_complete` | Existing audit plus RoundCore full-real-ring guard; runner sends36*q+{1,127,128,129}. NEW export correlates actual final scheduling/NEED_DRAFT with gathered real suffix and ring range. | `SuffixAudit` checks metadata continuity; `RoundCore._after_gather` refuses incomplete ON suffix. Need positive case coverage: final q/kept/valid/range, expected window and NEED_DRAFT, absence of PARTIAL_WINDOW, both arms. Successful HTTP output and absence of an error do not establish all four final tails. | Can share #1 exporter/probes; current tail requests already budgeted. Coverage/overhead unmeasured; conditional fit only. |
| 3 `source_kv_writes_retained` | NEW diagnostic observer at actual source-KV write boundary, before replay gather, with request/position/slot coverage and write completion; bounded native comparison of OFF/ON source-KV at matched inputs. | `DecoderReplayLayers.__call__` calls full-batch `write_batch_kv` before `index_select`; source audit alone proves intent. Existing suffix audit observes taps, not source-KV writes/data. Need all original nonpadding source rows written, no holes, and validated numerical/write comparison including mixed/excluded requests. Captures stay private outside git. | NEW model/KV hook and native diagnostic design; bound not established. Cannot reuse PF1 overlay without reviewed composition. |
| 4 `step_generation_echo_and_uniform_broadcast` | NEW scoped export in real client STEP/reply and engine broadcast consumption, across both ranks and normal/recovery rounds. Inject an old generation only through #9's owned experiment fault seam. | `client.step` validates session/seq/step_id and generation echoes; engine owner epoch is explicitly a different field. Need per-case generation-echo comparisons, identical per-rank draft/fallback decisions and publication ownership, without exporting draft token IDs. Current metadata audit runs before remote submission; it cannot certify post-reply broadcast. | NEW cross-process and post-broadcast observations, with TP ordering review. Bound not established. |
| 5 `near_cap_k4_k5_sync_async_clamp` | Exact327675+5 length probe, with real K5 at B1–2 and K4 at B3–4. NEW scheduler-boundary schedule/barrier and actual depth/allowance telemetry. Separate coordinator-reviewed sync execution configuration or explicit owner resolution of unsupported mode is necessary. | Need occupied lengths, per-turn allowance/clamp, actual scheduled drafts and normal finish for K4/K5 × sync/async, OFF/ON. Current runner starts async/process only and forbids env drift except its narrow allowlist. One serial API probe does not exercise the matrix; staggered HTTP requests do not guarantee batch size at the boundary. | NEW schedule controls plus a mode/identity-plan dependency. Cannot certify all modes within the existing two starts/packet identity. Bound not established; do not silently change `runtime()` equality. |
| 6 `remaining129_256_257_and_leftover64_128_peer_progress` | NEW engine-round scheduler script pins the target predecessor/end and competing offered work so repeated leftovers are exactly64/128. Exercise running and waiting admission paths, real remaining129/256/257 and a ready decoding peer. | Need per-turn budgets/floors, remaining, scheduled q, readiness, obligation/replay state and peer progress. `test_successor.py` mutates fake requests/work each turn with FakeKV; it is no live control surface. API priorities/arrival sleeps cannot guarantee these states. | NEW scheduler controls, barriers and telemetry; short contexts make a future bounded design plausible, but native bound and full-window cost are not established. |
| 7 `partial_full_hits_async_restore_real_suffix` | API prime/reuse can target full/partial hits; NEW cache-hit observations plus deterministic async-restore hold/release at the installed readiness fence, keeping a peer ready. Use real cache/block ownership. | Need verified actual hit/uncached lengths, replay_start, completed restored KV, newly computed suffix ranges and peer progress while restore waits. `_mark_prefix_replay` and ready-waiter code exist; the CPU test fakes connector/readiness. A cached-token count alone cannot prove any of this. No experiment connector fixture/control is supplied by the runner. | NEW connector/KV readiness integration and schedule control. Bound not established; cannot call a synthetic computed-count edit native restore. |
| 8 `cancel_slot_reuse_preempt_backtrack` | NEW one-shot barrier after a confirmed actual skip, owner-scoped cancel, forced real slot reuse, safe KV allocation/preemption fault and backtrack below recorded ring lo. Resume through installed replay route after completion/free fences. | Need old/new opaque owner/generation, actual cancel acknowledgment/retirement, slot reuse, preempt/free fence, backtracked position below prior lo and real suffix rebuild. CPU successor/suffix tests cover state logic only. Closing SSE at a wall-clock time does not guarantee cancel-after-skip or allocator reuse. | NEW scheduler/KV and owner lifecycle controls; cannot induce reliably via bounded traffic under unchanged memory/slots. Bound not established. |
| 9 `disconnect_before_after_final_watchdog_late_reply` | NEW local experiment proxy/transport seam keyed by case + owner + engine round: drop only owned traffic before/after confirmed final suffix; hold an actual response beyond watchdog release, then release it after newer staging ownership. Never kill/reconfigure the shared worker. | Need phase acknowledgment, real sent/applied/replied generations, retained ring restore, watchdog completed round, late publication refusal and newer staging intact. `mock_worker.Inject` exists for synthetic WorkerCore; it is not an installed native fault switch. Natural round1 watchdog events are incidental and do not prove the requested interleaving. | NEW process-proxy/transport control, cleanup and round-ownership evidence; possible without worker edits, but not implemented or bounded. A mock reply is not a native inference reply. |
| 10 `mixed_rejected_rows_mm_logprobs_exclusions` | Two API prompt-logprobs requests and a supported local MM input can request exclusions. NEW sampler/gather observer and deterministic native rejection schedule/control must guarantee rejected decode rows alongside prefill. Verify actual MM feature presence; never fake a text-only request as MM. | Need actual `_keeps_rows_np`/`_has_mm_np`, q/kept, rejection counts and gathered valid ranges, full source-KV coverage, all subcases OFF/ON. `skip_mask` exclusions and CPU tests exist. A stochastic decode may accept every draft, and HTTP logprobs output alone does not show the excluded replay path. MM load/support must be established before the window. | NEW sampler/gather telemetry/control plus #3 and MM fixture review. Bound not established; unsupported MM is NOT RUN/HOLD pending owner resolution. |
| 11 `native_dispatch_unchanged` | Observe real graph/eager/kernel dispatch for matched shapes and execution modes with a bounded native trace or NEW safe dispatch-summary observer; distinguish intended replay row reduction from backend changes. | Source/JIT/mount hashes are necessary identities, not evidence of kernel dispatch. Existing q/rank audit does not record dispatch. Need matched path/backend/kernel identifiers and unexpected fallback counts with no PF1/SO2 overlay composition. | A separate trace facility may suffice after owner review; existing runner has none. Native trace cost/capture compatibility unmeasured; bound not established. |
| 12 `no_local_drafter` | Existing capacity policy/drop-MTP config, runtime equality, process-proxy startup, proxy-torch and remote-round checks; collect corresponding engine startup/model-load evidence in each start. | `Live.start`, `ck.runtime`, `ck.counters` provide usable pieces. Retain positive no-local-MTP/load evidence and zero local-fallback growth with remote round progress; flag/env presence alone is insufficient. No deterministic fault hook needed for normal-mode evidence. | Existing per-start checks, minimal extraction cost; plausible inside current budget. Native PASS still NOT RUN by this lane. |

The common exporter for #1/#2 must not serialize `SuffixRow` blindly: it contains
an anchor token. Compare privately and emit equality/results and aggregate
metadata only. Likewise, do not export a broadcast buffer or tap/KV contents.

## Why a receipt writer is insufficient

`up2_checks.native()` checks schema/classification, arm/container IDs, manifest,
q, safe evidence paths/digests, all12 PASS strings, audited-round positivity,
request accounting and derived tail sizes. It does not interpret raw engine
observations. A CPU fake can produce a packet which passes that validator while
never exercising the native engine. The future apparatus must decide each PASS
from actual observations plus case-coverage assertions; fixture packets are
strictly CPU evidence. Lowering checks or renaming unsupported subcases would
hide the missing work.

At q checkpoint, runner warmup has finished; an observer must retain that start's
solo-q evidence, or send its own owned solo probe. At complete checkpoint, tails,
near-cap, numerical and needle requests have finished. The apparatus must have
observed them already, or account for any repeated requests and their extra cost.
It cannot inject a fault into a request that has already completed. Runner-owned
requests and apparatus-owned fault probes must be distinguishable and never
silently overlap. The current sequence calls `idle()`/`clean()` around API
requests; expected fault counters may only be scoped after a verified complete
native receipt. Plan barriers and cleanup accordingly.

`Live.native` waits until `min(time_at_call + native_wait, submission_deadline)`,
with max wait900s and ongoing supervisor checks. Await JSON contains submission
deadline but no explicit checkpoint creation time or 900s expiry. A future
watcher must conservatively bound arrival (and handle restart/stale files),
never invent a later deadline. Publish exclusive fresh artifacts and atomic
packets under the evidence root, binding actual start IDs and manifest. Keep the
original75min submission and90min hard bounds. No packet authorizes renewal.

## Window feasibility and coordinator decision

MEASURED historical inputs from `S30.report.md` §§5–6: start6.5–6.8min,
150K TTFT68.6–69.4s, 300K140.2s, decode52–56tok/s, exact prompt construction19–33s.
Control overhead10–20s/request and audit overhead are ESTIMATED/unmeasured.
S30's SCREEN diagnostic estimate is62–68min; restore approximately7min plus
checks. Only7–13min remains before75min submissions at that estimate.

MEASURED saved W13 successful diagnostic ledger: first send at HOLD+7.708min,
last completion at HOLD+57.801min, 50.093min between them. These exclude final
validation/restoration, and include no native apparatus. Saved near-cap TTFTs
are153.058s OFF and146.241s ON. They are timing inputs, not full mode coverage.

ESTIMATED incremental costs, illustrative serial probes only: a32K request with
32 output tokens costs roughly0.74–1.14min using S30's scaled prefill estimate
(~15s), decode(~0.6s), construction and control overhead. Eight such additional
probes cost5.9–9.1min **per arm**, before hooks, fault waits or cache primes.
Each extra327675+5 probe costs about3.1–3.4min using W13 OFF TTFT plus S30
construction/control/decode. Three extra mode/depth probes add9.3–10.3min per
arm, if those modes can be exercised at all without restart. Together that
illustration adds30.4–38.8min across OFF/ON, putting S30's diagnostic at roughly
92–107min before restore. It is not a complete probe inventory, a guaranteed
lower bound, or an approved execution schedule; concurrency and instrumentation
may change cost. Each additional engine start separately costs6.5–6.8min.

Four waits (q/complete × OFF/ON) could consume up to60min, but the window can
expire sooner; those waits are not extra reservation time. Per-case/fault timings
are unmeasured. Therefore **the full diagnostic does not have an established
one-HOLD fit**, nor an established900s complete-packet fit. Publishing PASS
from an incomplete probe set to meet the clock is prohibited. Native mode
qualification across extra starts also conflicts with the present two-start
packet/container binding and needs explicit coordinator design, without relaxing
the validator.

Coordinator decision needed: allocate a coordinated default-OFF hook effort
(shared observation schema; scheduler/KV lifecycle controls; process-proxy fault
seam; dispatch/MM/mode review), then independently review composition and CPU
negative tests before reserving hardware. Pin new engine files/manifests through
the normal build/identity process. Keep the production worker unchanged. Resolve
sync/MM/restore support and an honest partitioned window plan explicitly; do not
splice partial arms or copy PASS receipts between starts. An unsupported mode
remains NOT RUN/HOLD until its owning plan resolves it.

No executable coordinator command is supplied for nonexistent apparatus. The
existing command in UP2-LOW.md will correctly stop at missing native evidence;
it is not a full-qualification recipe until the dependencies above are delivered.
