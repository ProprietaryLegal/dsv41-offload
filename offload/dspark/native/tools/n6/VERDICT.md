# N6 disposition: CALIBRATION, with a bounded negative defect finding

No native batching state defect was established. On the twelve available N4
captured states, K2 and native kx2 preserve every proposal ID under identical-input
B1 versus B2/B3 packing. Graph replay, n_out prefix, same-size companion/order
controls and window metadata are separate checks in the aggregate receipts.
Numerical batch-shape sensitivity exists in both drafters, especially on
synthetic stress inputs. Native confidence is more batch-sensitive on this
captured scope even when proposal IDs agree: the frozen-policy simulation
changes depth 3→2 on one of twelve states at both B2/n5 and B3/n4, while
K2 changes none. These are estimated policy decisions, not target acceptance.
This supports measuring native
policy calibration; it does not prove W8's loss is caused entirely by calibration.

The inherited policy combines a request-count output cap, a common scheduler
window chosen from per-request acceptance EMAs, and a per-request confidence /
cost depth decision. The scheduler intentionally couples verification windows
across requests. W8 does not retain the per-request EMA/depth/verified-position
trace needed to separate those effects from proposal quality. The saved
`rounds_n_out_cut` counts only request-count caps and is not that missing trace.
Native needs its own held-out cost/confidence fit; no new schedule is promoted.

W8's B2/B3 native deficits are -0.155903/-0.121823 accepted drafts per
request-round (mean of the two run rates, versus mean controls). The control
return drifts are -0.002182/-0.008689. That does not support dismissing the loss
as demonstrated NOISE. Conversely, two temporal run windows per cell do not
supply a reliable task-level confidence interval: the B2 controls' individual
run rates vary substantially, and all runs follow different sampled trajectories.
The diagnostic cluster-SE ratios retained in the evidence are descriptive only.
A run/task-block repetition is needed before a noise verdict.

The first four positions account for 85.66%/98.87% of the pooled B2/B3 deficit.
Changing only B3's fifth-position cap cannot explain the observed pattern.
The full matched fixed-depth and adaptive-K-disabled diagnostic is specified in
[NEXT-WINDOW.md](NEXT-WINDOW.md), followed by independent cost/confidence fits
and held-out actual-token/complete-round evaluation. Existing fixed-depth
configs already disable adaptive K; preserve that distinction in the experiment.

This is a completed batch-dependence investigation with limits. It is not a
native-fidelity waiver, a broad absence-of-bugs proof, a physical-pair protocol
qualification, a D2-adapter qualification, or a production-promotion result.
The narrow captured scope is twelve early states of one request; permutations
increase coverage of row placement, not independent statistical sample size.
The synthetic ring tests use production shapes/absolute positions, not occupied
long legal tasks. The pair's real collectives and production engine were not
run by this lane. N4/N5's unresolved fidelity limits remain unchanged.

No runtime fix is committed because no semantic defect was isolated; changing
native arithmetic would exceed this lane's stop-fix-loops boundary. The branch
contains reproducible probes, source controls and aggregate evidence. No target
weights, request sampling, trained block size, production service, GPU power cap
or deployment identity changes.
