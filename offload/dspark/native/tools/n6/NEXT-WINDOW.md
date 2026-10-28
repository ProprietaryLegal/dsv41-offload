# N6 coordinator-only next-window measurement

This is a measurement prescription, not a launch script or authorization to
interrupt production. Native remains opt-in. Existing N4/N5 qualification
limits remain in force. N6 sends no serving request.

The first diagnostic should distinguish proposal quality from confidence and
cost selection. Prioritize steady B2/B3 at 32K, where W8 lost, with B1/32K as a
control. Keep checkpoint, target sampling (T=1/top_p=1), effort, prompts, output
budget, priority, cache state, trained five-query block and identities fixed.
Record actual active B rather than inferring it from nominal concurrency.

1. Obtain matched request-state captures at B1/B2/B3, including each complete
   window and taps. Replay each state alone and packed through K2 and native.
   Score both proposal streams against the same saved realized target
   continuation where fully known; preserve censoring when it is unknown.
   This supplies the missing B2/B3 states. Keep all tokens/tensors private.
2. Run interleaved/reversed K2/native controls with uniform fixed verification
   depths 0..5. Disable adaptive K, confidence masking and its online calibration during
   this cost/quality arm; hold the trained draft body at five. Where n_out
   truncation saves work, identify it separately from a changed verification
   depth. Keep the dynamic scheduler cap explicit so a fifth-position score is
   never silently censored at B3. Use the coordinator's reversible experiment
   mechanism, restore the original cap afterwards, and do not infer benefit
   from the uncapped arm without its actual complete-round cost.
3. For each private request-round record: task group, actual active B, steady
   context, graph dimensions, n_out, raw confidence per position, chosen depth, per-request accepted-draft EMA, scheduler window,
   reached/verified masks, accepted length, committed correction/bonus count,
   and actual delivered-token count. Measure verification-only time by depth,
   full round wall time, remote staging/transport/dispatch, and audited async
   overlap at the first draft consumer. Keep prefill/mixed/terminal rounds
   separate from steady decode; retain failures instead of dropping them.
4. Fit per-position native confidence reliability using reached, verified
   observations. Do not fit conditional acceptance from adjacent aggregate
   survival ratios. Fit cost from uniform-depth traces independently of the
   confidence calibration. Separate K2 and native fits; retain B/context/graph
   cells even if the production interface exports one approximate scalar.
5. Use at least three fit and two disjoint held-out task-family groups, with
   repeated temporal blocks. Compare the unchanged native policy, native-only
   confidence fit, native-only cost fit, and the combined fit on held-out tasks.
   Select using actual committed tokens / complete round time and completed
   task latency, alongside TTFT and per-stream/aggregate throughput. Keep K2
   before/after controls to measure drift and uncertainty at the run/task level.

Interpretation:

- If proposal quality loses at uniform fixed depth on matched B2/B3 states,
  calibration alone has not explained the loss; retain that as a native/D2
  quality lead. Diagnose batch/state coupling only if same-input controls show
  it, without restarting a general arithmetic-fidelity loop.
- If proposal quality holds but confidence chooses poor depths, use the
  native-only reliability fit and held-out objective to choose a policy.
- If cost slopes change by B/graph shape, the existing scalar timing interface
  may be inadequate. Keep the per-cell measurements rather than silently
  averaging or extrapolating missing cells.
- If the repeated task-level difference vanishes, report the observed noise
  with those independent blocks. W8's two blocked runs cannot establish that
  conclusion by themselves.

## NB1 arm: native kx2 with batch-invariant dispatch (written by lane NB1 2026-10-26; not run)

Coordinator window only: the NVLink pair is production's drafter. This arm adds one question to the window above —
does making native's packed drafts equal its B1 drafts recover W8's B2/B3 loss? — and can run alone or inside the
N6 window (then `I` joins N6's fixed-depth controls as a third drafter). Evidence: NB1 `INVARIANT.md`,
`MICROBATCH.md` and `evidence/item2/` in `offload/dspark/native/tools/nb1/`.

What NB1 established offline (MEASURED, solo-card TP2 emulation on bc62df2a, 12 captured N4/V1 + 12 synthetic
states):
- Packed native drafts differ from B1 first at stage-0 `wo_a`: the FP8 split-K changes 8→4 from M10, and M20
  falls back to cuBLAS BF16. Act-QDQ and top-3 routing flips amplify the gap. On captured states the inherited depth policy then
  picks 3→2 on 6 / 72 B2 and 18 / 216 B3 placements. Window insertion is also batch-dependent (5 rows per
  request, 32 rows at B2).
- `DSPARK_NATIVE_BATCH_INVARIANT=2` makes IDs, confidence and inserted windows bit-identical alone vs packed at
  B2–B4. When unset it is byte-identical to frozen N4. At level 1 the B1 draft is unchanged; at level 2 (this arm) B1
  insertions above 8 rows change by design, so 12 of 24 single-request results differ (4 in token IDs), with B1
  insertion no less accurate (FP64 error ratio ≤ 1.0011; `tools/nb1/INVARIANT.md`). The B1 control cell measures it.
- Per-shard cost: draft +0.7–2.8 % at B2/B3 (≤ 0.09 ms), +6–7 % at B4. Insertion +0.35 ms per steady flush of
  ≤ 32 rows, ≈ +2.1 ms per 128-row prefill chunk. Microbatching (each request its own B1 graph) costs 1.5–1.9× the
  packed draft at B2/B3 (+1.3 / +2.8–2.9 ms) for the same draft invariance, so this arm uses the switch, not
  microbatching.
- Not established offline: whether these drafts are accepted more by the target, and whether any of the extra
  worker time is exposed at the consumer deadline. Those are this arm's questions.

**Arms.** `K` = production TP2 K2 worker (unchanged unit). `N` = native-hybrid-kx2, switch unset. `I` = the same
native worker with `DSPARK_NATIVE_BATCH_INVARIANT=2` (draft dispatch pinned to B1's, window insertion in 8-row
sub-batches). `N` and `I` run from one source tree of `offload/nb1` at the NB1 item-2 commit (python-only change on
top of N4: N4's prebuilt `lib`/`build`/`p2p` and Triton caches apply unchanged; no item-2 run at any level compiled
a new Triton kernel), so the two native arms differ only in the
environment variable; NB1 proved the unset path byte-identical to frozen N4 on every probe output. Their drafter
identities differ from W8's `5c3be5c7…` (the native identity hashes runtime sources; `I`'s recipe also names the
level); record both at start. Launch as W8 (`W8/native_ab.sh` line 71) with the tree path changed and the
variable exported (`I`) or explicitly unset (`N`). The worker reads it once at build; any value other than unset/0/1/2
refuses to start.

**Load.** Steady decode at 32K context with 1, 2 and 3 equal streams (B1 control, then B2, B3), W8's `c32` suite
settings: legal workload, T=1/top_p=1, seed 42, effort high, 1,500 output tokens, priority 10, cold cache, real
filler; 3M (mixed) and 150K optional, not decided on. Record actual active B per round, not nominal streams.

**Matched depth settings.** All arms: n_out cap `[0,5,5,4,4]`, the same scheduler window rule (EMA α .3), the same
adaptive-K / confidence policy and inherited prices (`DRAFT_MS=4.4`, `ROUND_MS=28.79…43.68`), trained block 5,
KV 1,604,926, no K1. No native-specific fit in this arm (that is N6's arm; if the window also runs N6's uniform
fixed-depth 0..5 controls with adaptive K and masking off, run them for `I` as well).

**Counters.** Everything N6 item 3 lists, per private request-round: task group, actual active B, steady context,
graph dimensions, n_out, raw confidence per position, chosen depth, per-request accepted-draft EMA, scheduler
window, reached/verified masks, accepted length, committed correction/bonus count, delivered tokens;
verification-only time by depth, full round wall time, remote staging/transport/dispatch, and the audited async
overlap at the first draft consumer (exposed draft wait). Prefill/mixed/terminal rounds separate from steady
decode; failures retained. Plus W8's run summaries (per-request and aggregate tok/s, accepted/round, step p50,
drafter round trip p50, failed rounds, pair memory and power) and the greedy/near-tie check per block. If the
per-round trace counters N6 found missing are not exported yet, run anyway and say so: the decision below uses
only run-summary metrics and the exposed wait; the missing traces only limit the explanation.

**Order (≥ 3 interleaved reps per arm).** Blocks `K N I K I N K N I K` (K brackets every native pair; within the
pairs the order alternates `N I`, `I N`, `N I`). Each block: `start_exp` for matched cold state, then the
32K B1/B2/B3 cells × 2 consecutive reps, then the greedy capture. Native→native swaps restart the worker (the
switch is read at build). If time is short, the pre-registered reduced order is `K I K I K I K` (no `N`): it still
decides against K2 but cannot attribute the result.

**Pre-registered decision rule (fixed before the window; metrics per steady-32K cell).** Arm value = mean of its
block means (each block mean of its 2 reps). Noise δ(cell) = the largest |difference| between any two `K` block
means at that cell, as a fraction of the `K` mean, floored at 1.0 %. Primary metric: per-request decode tok/s.
- **CANDIDATE** — `I ≥ K − δ` at B2 and at B3, `I ≥ K − δ` at B1 (the control: W8 native was +5.63 % there),
  zero failed rounds in `I`, exposed draft wait in `I` not above `K`'s by more than δ of `K`'s step time, and every
  greedy divergence within one target grid step (W8's near-tie criterion). Then report to NDM as a candidate for
  a separate promotion decision; it does not lift N4/N5's fidelity qualifications or rule 27.
- **PARTIAL** — `I > N + δ` at B2 or B3 but `I < K − δ` there: batch-dependent arithmetic explains part of the
  loss; the N6 calibration fit proceeds on `I`, not `N`.
- **NO EFFECT** — `|I − N| ≤ δ` at B2 and B3: batch-dependent arithmetic is not W8's cause; close NB1's lead and
  continue N6 calibration on `N`.
- **HARM** — `I < N − δ` at any of B1/B2/B3, or failed rounds/startup refusal in `I`: the switch stays off.
- Accepted/round is the mechanism check (W8: native −0.156/−0.122 at B2/B3); it does not decide alone.
With the reduced order only CANDIDATE vs not-CANDIDATE is decided.

**ESTIMATED minutes** (from W8's MEASURED runner log, 2026-10-24: `start_exp` to drafter ready 6.5–6.8 min;
production stop to native worker ready 5.5 min; native stop to production worker back 0.2 min; one 32K B1+B2+B3
rep 3.7 min; greedy capture 0.3 min):

|block|overhead|load (2 reps)|greedy|minutes|count|subtotal|
|---|---|---|---|---|---|---|
|K|6.7 (+0.2 after a native block)|7.4|0.3|~14.5|4|~58|
|N or I|5.5 worker + 6.6 `start_exp`|7.4|0.3|~19.8|6|~119|
|restore production + checks|||||1|~5|
|**full order**||||||**~182 (≈ 3 h)**|
|**reduced order `K I K I K I K`**||||||**~122 (≈ 2 h)**|

Prerequisites (no production change): the `offload/nb1` source tree on x86host with N4's prebuilt binaries, a solo
start of the native worker with `DSPARK_NATIVE_BATCH_INVARIANT=2` on bc62df2a (identity printed, no requests)
before the window, and the coordinator's reversible experiment mechanism for the cap/scheduler settings (unchanged
here). Restore production exactly as W8 did and verify the K2 drafter identity `be2c1436…` afterwards.
