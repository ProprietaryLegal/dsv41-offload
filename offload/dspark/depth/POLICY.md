# SP2: what the deployed depth policy actually does

Source audit, 2026-10-23. CPU only. This directory contains analysis and opt-in experiments; no serving change.
Numerals describing source constants are configuration/code facts, not performance measurements. Performance
numbers below are labelled MEASURED or ESTIMATED. Source image root: `/opt/dsv41/data/src-image`.

Repo convention is `R = (1 + E[A]) / C`: A excludes bonus; C includes complete critical-path cost.
The policy ratio is an estimated instance of that objective, not a measured engine throughput.
See [metric definitions and paper comparisons](../../evidence/THROUGHPUT.md).

## Three different depths

1. `offload/launch/serve_exp.sh` sets maximum trained block size five and scheduler caps
   `[[1,2,5],[3,4,4]]`. The remote worker HELLO is also `num_spec=5` (`worker/server.py:expected_hello`).
   Depths six through eight are **unsupported**, not an extrapolation opportunity. Keeping five query positions is
   necessary even when consuming fewer: the draft block attention is noncausal. Shortening the block changes its
   first predictions. `n_out` trims only the head/Markov/confidence tail (CONTRACT §6.3).
2. Image `vllm/v1/core/sched/scheduler.py::_dsv41_adaptive_k` (patch 0047) sets the common next window.
   With unknown EMA, a partial prefill, or ANY request's accepted-draft EMA >= `AK_HI=1.6`, keep the batch cap.
   Otherwise take max over requests of ceil(EMA + `AK_MARGIN=1`), bounded below by `AK_MIN=2` and above by cap.
   The default applies through four requests. Raising a cap alone need not raise actual verification depth.
3. Image `vllm/v1/worker/gpu/spec_decode/depth_policy.py::DSparkDepthPolicy.record` sets a per-request depth.
   `conf` forms conditional acceptance predictions a_i from the confidence head, clamps them to [0,1], forms
   prefix survivals S_i = product(a_1..a_i), and selects the FIRST maximum of
   `(1 + sum_{i=1..d} S_i) / (ROUND_MS[d] + DRAFT_MS)`, including depth zero.
   It zeros confidence beyond the batch cap before choosing. Scheduler window and per-request depth both limit
   what reaches the verifier. Rows beyond depth become routed-expert padding and invalid drafts; dense/graph rows
   need not disappear. This is why fewer live rows do not imply a proportional reduction in step time.

The extra one is the target correction/bonus token on a steady, nonterminal round. EOS, budget clipping,
prefill-only and failed rounds must be counted by actual commits for final throughput qualification.

## Timing interface and sync versus async

`round_cost_ms()` adds the same scalar `DRAFT_MS` at EVERY candidate depth, including zero. `ROUND_MS` is one
single-stream verify-only table indexed by depth zero through five. If short, the image extends its last slope;
the remote config deliberately rejects lengths other than six. These are policy prices, not online measurements.

`remote_dspark/config.py::apply_cost_overrides` runs before constructing the depth policy. In capacity/resilience
mode it requires BOTH `REMOTE_ROUND_MS` and `REMOTE_DRAFT_MS` for cost-based policies, copies them to the image
variables, validates positivity/monotonicity and rejects `confr`. Shadow keeps the local cost model. A fixed policy
without a reload file does not price depths. `confb` jointly chooses depths using total live rows; its extended
single-stream slope is an estimate, not a measured B4 table. `ctx=` adds interpolated context overhead only in joint
modes, never in `conf`. This interface cannot represent independent cost tables for every batch/context/graph cell.

Sync: host waits for the RPC before returning the model runner's asynchronous output; worker/transport and lost CPU
preparation overlap may both be exposed. Async process mode: the proxy handles RPC while the engine can submit
independent preparation; the GPU must still wait at the FIRST consumer of draft ids/confidences, then TP-broadcast
the same reply. Async does not allow next verification to ignore that dependency. Treat
`effective_remote_ms = prep_ms - audited_overlap_ms` as the exposed cost, including staging/transport/dispatch,
not the isolated GPU draft graph. Source: `spark/{ORDERING,OVERLAP}.md`, remote speculator `_make_round` and N3 recipe.
Changing ASYNC alone does not change the depth formula or automatically recalibrate its prices.

`main:dsv41.env` at this audit has async process/nap/LAZY, capacity, six GiB KV, inherited
`REMOTE_ROUND_MS=28.79,31.86,34.44,37.76,41.56,43.68` and `REMOTE_DRAFT_MS=4.4`.
These are HISTORICAL CONFIG VALUES, not newly measured TP2+async costs. W5-prod was one-card async; W4/W5-E1p were
TP2 async. Do not call the W5-prod data TP2. Read main with `git show main:dsv41.env`: this lane's root config predates it.
S20.report.md M3 specifically warns against transplanting these prices to native arms (different costs/identities).

Counterintuitive consequence: lowering a depth-independent DRAFT_MS makes expensive extra verification rows
RELATIVELY MORE expensive. Hidden draft cost alone is not a mathematical reason to deepen. If verify cost is
unchanged, adding position d helps only when S_d > E_{d-1} * delta_cost / previous_cost. A faster verify slope or
better acceptance can reverse that conclusion. Measure the whole round rather than optimizing a guessed scalar.

## Acceptance, calibration and exactness

Optional `VLLM_DSPARK_CONF_CALIB=1` scales each position's predicted probability using accepted/predicted EMA,
separately for greedy and sampled target requests, after 256 reached observations; alpha .02, scale clamped .3..1.2.
It uses only reached, verified, non-lookup positions. Async does not change this rule. New experiments keep it off
unless a separately identified calibration arm is being tested, to avoid mixing cost and confidence changes.

The remote adapter accepts only GREEDY draft ids and confidence, not full proposal distributions. The target remains
T=1, top_p=1. In image `rejection_sampler_utils.py`, a point-mass draft d is accepted with p_target(d); rejection
resamples with d excluded. Greedy target requests must match target decisions subject to the measured engine
numerical floor. Sampled requests preserve the target distribution, not necessarily identical seeded text.
Confidence/window selection never authorizes an unverified token. Tree verification or probabilistic proposals need
explicit algorithm/protocol work; changing a verifier flag does not create extra acceptance for point masses.

`accepted_per_position / rounds` in bench2 is ALREADY a prefix survival under the policy actually run. Do not multiply
these values again. Ratios of adjacent values are conditional on survival AND being verified, not uncensored
acceptance predictions. Scheduler caps, dynamic windows, confidence masking and changing active batch sizes censor
the tail. Small fifth-position counts in nominal B3/B4 runs can come from requests finishing and the batch shrinking.
`spec.rounds` counts request-rounds, not engine steps; multiplying observed aggregate throughput by B again is wrong.

## Measured recalibration gate

N3 `native/tools/n3/{depth_config.py,README.md}` exports per-cell prices from paired server traces. Require uniform
fixed depths zero through five; B1/B3/B4 (B2 for completeness); short/32K/150K steady cache cells; same build/identity;
quiet queues; at least three fit and two disjoint held-out task-family groups. For async explicitly audit overlap at
first consumer. The default zero in a trace without overlap is not a measurement. Exporter refuses sparse or mixed
cells and nonmonotonic costs. Do not silently clamp, impute or label the screening estimates as this qualified fit.
The scalar export is an approximation: retain per-cell and actual graph/live-row data, and check held-out complete
round prediction error and actual delivered tokens/s before adoption. No suitable fixed-depth trace exists in the
requested compact suites; this lane supplies candidates and the missing-measurement plan, not a promoted calibration.
