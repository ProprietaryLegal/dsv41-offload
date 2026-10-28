# Two decoding streams plus a 150K arrival

Design only, no live scheduling change. Source pin: our v5 scheduler.py / gpu/model_runner.py at engine
6e517b15.dsv41gb10, O3 at21bc3ab; HeNryous afe4d5e 0072/0074 are contrasts, not transplanted policy.
Affected op: scheduling a prefill chunk alongside or between two already-decoding priority10 streams.
Mechanism: budget elapsed decoder service debt, limit non-preemptible prefill chunks and give the long request a
measurable minimum share. Priority labels, effort75, target tokens/context/sampling and output budgets stay matched.

Problem decomposition:
- A GPU kernel/forward already running cannot be preempted by scheduler policy. A4096-row prefill can dominate the
  whole round; changing HTTP priorities does not shorten it. Async admission may also have one or more batches
  already queued. Both non-preemptible duration and queue depth bound best-case decoder response.
- Smaller chunks change TensorFold/expert dispatch, attention/indexer launches, graph eligibility, replay (128
  late-layer rows/chunk unless0075), and Engram/proxy overhead. More chunks can worsen total prefill time and change
  arithmetic. 0064 may matter for33–63-row tails, but do not combine it with this experiment initially.
- Shared model bandwidth means simultaneous prefill/decoder maxima may be physically impossible. The policy must
  report a missed service budget rather than silently starve the150K job to satisfy decoder tps.

Proposed controls (all default off, coordinator-owned future scheduler overlay):
1. Reserve rows for every ready decode request FIRST within its existing priority cohort, based on current allowed
   verify depth and capture shape; limit prefill to leftover token/KV/encoder budgets. Retain FIFO/age ordering among
   ready prefills and among decoding peers. Do not change request.priority, global inter-cohort ordering or effort.
   Record the scheduler phase distinction explicitly; it is a service policy change, not a free throughput gain.
2. Maintain per-decoder last-service completion timestamp and delivered-token count from actual emitted tokens,
   not proposed drafts. A cohort budget T_gap and optional per-stream rolling delivery floor define service debt.
   Track actual GPU completion/emission timestamps via output records; scheduler submission time is insufficient.
3. Profile no-prefill decode-only round cost D(n,depth,context) and the cost P(c,n,depth,context,backend) of candidate
   chunks c∈{256,512,1024,2048,4096}, actual graph/eager shape. Use current legal workload rolling p90/95 event times,
   with uncertainty margins; direct table cells only. No extrapolated constant tokens/ms at150K.
4. At each scheduling decision choose largest measured c whose predicted non-preemptible duration plus already
   queued work plus decoder debt fits T_gap. Initial screening:256/512 chunks; purely illustrative T_gap100ms,
   not a performance promise. If chunk256 cannot fit, choose a minimum256 service quantum explicitly and log a
   budget violation; there is no exact policy guarantee until measured kernels can satisfy the bound.
5. If debt remains after a mixed/prefill step, run bounded decode-only service steps to repay it, then admit the
   next prefill chunk. Cap consecutive decode-only steps by a long-request max-wait budget, e.g. initially500ms,
   and enforce a nonzero measured prefill share/quantum. More prefills share those quanta round-robin with age.
   Use deficit accounting/capped debt: debt cannot accumulate without bound and monopolize future scheduling.
6. Hysteresis: decrease chunk one bucket on budget miss, increase only after multiple satisfied intervals and no
   old prefills overdue; minimum dwell time prevents oscillation. Stop/freeze adaptation at sparse/unmeasured cells
   and use last qualified static cap. Do not invent timestamps or fit from ratios of unrelated run medians.
7. Admission: if max_num_seqs/KV pool lacks fit, queue new work preserving original arrival/priority/salt and report
   waiting TTFT. Do not silently reject long legal tasks, shorten context, reserve almost all KV for existing streams,
   or pause equal-priority jobs indefinitely. Decode budget also applies to decoding the admitted long request later.

0075 interaction: every chunk after a skipped replay must trim. Apply `_reserve_prefill_lookahead`, encoder limits,
then replay min-chunk rule on both running/waiting paths. Do not consume leftover<=128 rows when>128 prompt rows
remain. Final remainder<=128 keeps the previous full suffix; a long request's minimum quantum cannot force a tiny
successor across a skipped boundary. If budgets cannot yield129+rows, defer the prefill and account its wait; eventually
reserve a full minimum256 quantum. Prompt-logprobs/MM exclusions stay intact. No kernel/page/cache-format changes.

Async pipeline: the control must price ALL already-enqueued batches. Cap outstanding prefills to one while decoders
carry service debt. A CPU clock at enqueue cannot certify GPU completion or cancel an already-captured batch. Input
preparation and remote drafter overlap remain subject to A2 ordering; no second proposal/status stream is introduced.

Fairness feasibility gate:
For each stream i estimate expected committed tokens c_i per decode service (including correction/EOS behavior) and
its measured maximum inter-service interval G_i. c_i/G_i is a planning bound, not delivered tps. Example: c_i≈2,
100ms gaps gives≈20tok/s; a150ms minimum prefill already prevents that100ms budget. If the long-request minimum
share and decoder budget cannot coexist, report the measured Pareto curve (prefill TTFT vs decoder gaps/delivery)
and keep the best useful bounded profile; never claim both requirements were met by hiding prefill starvation.

Coordinator run plan:
- Baseline current D+K1, two legal decoder streams at priority10, effort75/T1/high exactly as the matched suite,
  then third150K cold legal prompt at fixed offset; also warm hit, fourth arrival, cancellation and near-capacity KV.
  Continuous arrival variant checks age/queue stability; after third prompt prefills it becomes another decoder.
- First static screening c4096 vs1024 vs512 vs256 in interleaved repeated runs, SAME output budgets. Change no other
  optimization. Then compare budget policy only against the best static cap; selective0064 or0075 separate arms.
- Per stream: actual occupied input, accepted/delivered reasoning+answer tokens, every emission timestamp, p50/p95/
  p99/max token gap, rolling5s/15s delivery, longest starvation interval, first token/first visible/full-task latency.
  Long job: tokens prefilling each step, wall-time progress curve, oldest wait, TTFT, completion, min share violations.
  Aggregate: delivered tokens, model utilization/queue depth, rows/graph/chunk, direct verify/prep/round cost, fallbacks,
  memory/paging/thermals and remote worker state. S3 bench2 can collect streaming gaps; engine progress log is needed
  for chunk/admission attribution. No need to replace the existing benchmark harness.
- Compare full distributions at matched shapes independently (item6), prefix/cancel/resume correctness, current
  depth control and fixed-depth diagnostic. Smaller chunks may change arithmetic; do not dismiss large shifts.
- Stop on starvation >configured max-wait+one measured non-preemptible quantum, missing long-request progress,
  rank/state/ordering defects, memory pressure, unexplained logit shift or repeatable aggregate loss>5% without an
  explicit useful concurrency gain. Keep baseline rollback. A decoder-only percentage is insufficient promotion.

No target service budget or speed number is measured by X1. This is the concrete scheduler experiment specification;
the coordinator owns implementation/window choice after existing current campaign gates.
