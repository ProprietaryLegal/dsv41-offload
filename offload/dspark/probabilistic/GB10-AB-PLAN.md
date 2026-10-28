# PP1 bounded GB10 A/B plan — coordinator execution only

Status: **NOT RUN; all sampled paths OFF**. PP1 is a CPU lane and has no Spark,
GPU or production-pair access. This is the handoff for expert-continuation §6,
reconciliation B8. It does not authorize a restart, allocate the pair or add a
serving flag. The rational reference passed; the GPU sampler/transport adapter
described below has not been implemented or qualified.

## Prerequisites and implementation boundary

1. Coordinator closes source-faithful native pair qualification/W8, records the
   selected exact native profile, weights/Markov/head identity, worker SHA and
   graph/precision receipts. Keep source-faithful, adapted and production K2
   controls distinct. Do not infer a native qualification from PP1 CPU tests.
2. Finish/review a v2 worker q producer and standard verifier adapter per
   [TRANSPORT.md](TRANSPORT.md). Required seams: TP2 `heads`/Markov selection,
   wire HELLO/STEP/commit, process-proxy reply layout and bounded pools,
   speculator q-ready/lifetime, GPU runner request-row mapping and sampler.
   Today's guard in `remote_dspark/speculator.py` must keep refusing sampled
   mode until all these capabilities negotiate. No sampled IDs with missing q.
3. Add explicit OFF-by-default, versioned experimental profiles G/S/K and a
   reviewed stage/check/restore adapter using existing window conventions.
   Bind the final source SHA, image overlay hashes, buffer sizes, verifier
   standard mode, q ABI, draft temperature/support, fixed depth, target identity
   and processor policy in a machine-readable run receipt. Do not invent CLI
   flags for unimplemented controls or reuse a greedy profile label.
4. Complete CPU/sharded emulation and GPU toy-distribution tests before Spark
   serving. Compare q byte/count identity, normalization, selected tokens and
   residual probabilities against `reference.py`, including masks, zero-mass
   shards, duplicate IDs, NaN/inf, padding, vocabulary boundaries, TP ownership,
   fixed depths 1..5 and B1..B4. Exercise graph capture and dynamic request-row
   reordering. RNG streams must be separate, correctly seeded and request-owned.
5. Coordinator opens an exclusive drained test window after D2's authorized
   generation is paused/checkpointed or complete. D2 training on separately
   granted cards may continue only if its shared-resource interference is fixed
   across arms. PP1 does not cancel/restart the independent D2 pilot. The live
   production pair and its units remain untouched until the coordinator holds
   its existing pair/process ownership gates and has a working rollback.

## Matched arms and controls

Every comparison uses the **same native NVLinked pair**, same released tensors,
Markov/cache implementation, precision, graph shapes, process proxy and Engram
backend. No adapter in the primary experiment. Target legal-v2a, target T=1,
top_p=1, all processors, high reasoning effort, prompt formatting, priority 10,
output budget, context/KV allocation and maximum sequence count stay identical.
No target-only control replaces the native greedy baseline.

| Arm | Proposal policy | q seen by standard verifier | Purpose |
|---|---|---|---|
| G | Existing native greedy argmax | Point mass at d | Primary control on the same pair |
| S | Full-vocabulary sampled draft, calibrated draft temperature | Actual dyadic24 full q sampled on worker | Probability-aware dense arm |
| K | Global top-k draft, initially k=32, same selected draft temperature | Normalized dyadic24 sparse q, exactly zero outside transmitted support | Bounded communication candidate |

Choose **fixed depth 5 before all draws** for the first comparison, B1 then B2.
Disable confidence/depth-policy pruning and retrospective adaptive-verify masks
for every arm, including G. Backbone/query shapes stay the same. Before B3/B4,
prove the required 20 draft + 4 bonus target rows, buffers, graph captures,
timeouts and kernels support depth 5; the existing batch-size cap may use four
drafts at B3/B4 and must not silently change this experiment. If depth five is
unsupported, register a separate all-arm fixed-depth-four experiment and repeat
its G baseline. No concurrency above four or larger context allocation here.

Only the draft policy changes. Since the installed local sampled path ties q
temperature to target temperature, the new adapter must explicitly decouple
them; verify effective target parameters at runtime, not merely in the client
request. Use standard tokenwise verification, never block or synthetic mode.
Record “unchanged native weights, experimental proposal policy,” rather than
claiming that sampled/quantized q is the unchanged official inference policy.

## Bounded calibration, then frozen evaluation

Development screen: eight public requests from at least four source families,
spanning reasoning/answer segments and short/32K prefixes. Compare draft T in
{0.7, 1.0, 1.3}; keep support 32 initially. If useful, one k=16/64 screen at the
selected T, within the same fixed budget. Select by delivered-token benefit
under measured whole-round cost, with per-family acceptance and tails reported.
No unbounded grid or selection on final requests.

For acceptance diagnostics, evaluate p on **the prefix actually drawn by each
policy**. The K2 committed trajectory's teacher logits are not valid after a
sampled alternative chain diverges. Use correctly conditioned target calls or
the real verifier; a shared target feature capture alone does not repair that.
Any teacher/probability captures remain private scratch and outside git/reports.

Freeze T, support, q ABI and profiles before the final set. Reserve eight new
public requests across at least four unseen source families (never used for
temperature or support selection). This is a bounded feasibility experiment,
not a universal quality claim. Repeated seeds on one request are not independent
families. If intervals are inconclusive, permit one extra interleaved repeat;
then report INCONCLUSIVE rather than expanding indefinitely.

Final schedule: warm each profile with verified real worker rounds and graph
shapes; record cold-start separately. Interleave G/S/K/G, then rotate arm order,
at least three complete repetitions with matched seeds/budgets and public
request identities. Do not require the stochastic texts to match by seed.

| Stage | Workload | Advance rule |
|---|---|---|
| Semantics | Toy sampler, then short fixed-prefix and structured/stop/tool/reasoning requests | Every correctness/state row below passes |
| Screen | Short + 32K, B1/B2, fixed 1500-token budget | No semantic failures; complete round cost is measured |
| Final | Frozen public families, 32K and 150K; cold/warm prefix; B1 and actual B3/B4; mixed prefill/decode | Same arms/controls; no lost requests or unfair stalls |
| Full task | One representative complete public legal analysis per arm at 150K | Same reasoning/output requirements, no truncation or effort reduction |
| Optional long check | Winning arm vs G, 256K and ~300K occupied | Reserves fit current qualified 327680 profile; no 524288/500K requalification here |

The current `offload/bench/bench2.py` already has matching client controls.
After the coordinator has staged and attested an arm, this is a representative
client invocation **for the coordinator only**, never run by this CPU lane:

```bash
source ~/.config/ndm/paths.sh
"$HARNESS_PYTHON" offload/bench/bench2.py \
  --url http://HEAD_IP:8001 --label PP1-G-fixed5 \
  --ctx 32000 --workload summary --concurrency 1 --reps 3 \
  --max-tokens 1500 --max-model-len 327680 --thinking --reasoning-effort high \
  --priority 10 --temperature 1 --top-p 1 --seed 42 --cache-mode cold \
  --filler real --data-dir /opt/dsv41/data/bench-data \
  --out /opt/dsv41/campaigns/DSV41-OFFLOAD/PP1-G-fixed5-c32.jsonl
```

This command selects client workload, **not** the server's proposal arm; arm
selection/receipts are a prerequisite. Repeat matched invocations for S and K
only after verifying their effective profiles. Do not use `--keep-text`.
Use exact `/tokenize` counts and reserve reasoning/answer space. Preserve source
families, IDs/hashes and aggregate timing, never client prompts/generated text.

## Semantics and state checks

| Check | Required evidence |
|---|---|
| Categorical semantics | Exact toy p/q law and conditional joint distributions against the CPU oracle; measured per-token frequency/acceptance with preregistered simultaneous bounds; q=p accepts all, disjoint support accepts none |
| Precision | Producer and verifier use identical q counts for every consumed row; FP32 dyadic validation exact; probability/residual differences against rational reference classified without loosening gates to force a pass |
| Effective p | Same target fixed-prefix processed logits/probabilities against target-only sampling under A-A numerical floor, including grammar, penalties, bad words, minimum tokens and stop rules |
| Constrained output | Real engine structured JSON, tool schema/calls, reasoning parser and EOS/stop paths; target grammar advances on matching proposed prefix, then rolls back rejected suffix; unsupported processors refuse sampled mode before a draw |
| Lifecycle | Request permutation, mixed lengths, cancel, same slot/new generation, prefix reuse, eviction, reconnect, late/duplicate reply, missing q and timeout; no reuse of stale distributions or cached Markov dependencies |
| Failure | Worker/proxy failure drills only under coordinator-owned test processes; both ranks discard unverified q/IDs, preserve committed prefix, complete target-only or fail explicitly; recovery establishes a new generation |
| Target greedy | T=0 diagnostic uses exact target argmax semantics and the established A-A near-tie policy; does not stand in for stochastic distribution tests |
| Selection | Profile/log proves fixed depth selected before proposal draws; no confidence or later-token-dependent exclusion of earlier positions |

Predeclare toy GPU tolerances using binomial simultaneous confidence bounds and
numeric reference error; do not choose tolerances after inspecting samples.
CPU exact-law tests are necessary but do not qualify GPU RNG, grammar objects,
TP row mapping, graph lifetimes or the model's real output distribution.

## Timing, uncertainty and decision

Collect server-side request/generation/round IDs, accepted draft count A,
actual target verification rows, actual delivered tokens and EOS truncations.
Sample non-blocking CUDA-event spans on **both Spark ranks and both worker
cards**: target verify, sampler, draft backbone/head/Markov, q build, proposal
sampling, D2H/H2D, queue/transport, TP collectives and exposed waits. Correlate
IDs; do not add overlapping intervals or synchronize every round to time it.
Keep profiling overhead measurements separate from headline runs.

Report whole-engine delivered tokens/s, p50/p95/p99 complete round time C,
per-request and aggregate throughput, TTFT, first visible answer, full-task
latency, answer/reasoning tokens, acceptance by position/family/segment, and
transport bytes. Record both Sparks' available memory and peaks, both 3090s'
load/serving peaks, sustained power/clocks/throttling, bounded pinned pools,
Engram latency, link errors/retries and watchdog/fallback counts. SSE timing is
a delivery measurement, not the sole GPU/engine timer.

For steady nonterminal rounds, R=(1+E[A])/C; A excludes correction/bonus and
rejected proposals. For terminal rounds use actual delivered-token counts
because an accepted EOS can omit the bonus. More acceptance is a loss if its
q/verification/transport cost grows faster. A standalone drafter microbenchmark
does not measure the cost exposed through the process proxy's overlap.

Pair comparisons by request, source family and repetition. Use a seeded
family-clustered paired bootstrap (2000 resamples), report 90% intervals and
per-family results, plus p95/p99 stalls. Report small-family uncertainty rather
than treating thousands of rounds as independent observations.

Promotion candidate: all semantics/state gates pass; ≥5% repeatable delivered
throughput gain or equivalent full-task latency reduction over G with the paired
90% lower bound above zero; no >5% TTFT, visible-answer or p95/p99 stall regression
on any measured workload class, no starvation, and no unexplained failures or
memory pressure. No capacity trade is claimed: this experiment does not increase
context or concurrency limits. Coordinator still owns independent review and
promotion; PP1 cannot promote by meeting its own paper plan.

**LOSS/REJECT** if it violates semantics, changes target policy, starves legal
work, loses requests, increases memory/failure risk, or a throughput/latency
regression is demonstrated. **NO PROMOTION / INCONCLUSIVE** if it only improves
acceptance/microbenchmarks, its whole-engine gain is below the practical bar,
or repeated-run uncertainty includes zero. Preserve results and native greedy
G; do not lower target temperature, disable reasoning or substitute a model.

Only after independent S/K and D2 decisions, optionally test one frozen passing
adapter combined with the winning q policy. Repeat matched G/S/K controls under
that adapter identity; invalidate affected caches. Keep the native and adapted
profiles distinct. A PP1 loss does not cancel the D2 distillation pilot.

Rollback: coordinator restores the saved exact pre-window Spark configuration,
overlay hashes and production worker ref/profile using the existing reviewed
window/restore scripts; checks the actual baseline unit, both-card ownership,
health, greedy regression, target sampling settings and fresh counter deltas;
closes window markers only after the service is usable. Save a restoration
receipt. No q feature is enabled on production merely by merging this directory.
