# N6 depth and output-cap audit

`window8.py` reads the completed coordinator files on Controller; it sends no
requests. Its committed evidence whitelists numerical run summaries and caps.
All performance numbers here are **MEASURED** from W8-native, unless labelled
otherwise. The depth formula and constants below are source/configuration facts.

The remote table `[0,5,5,4,4]` maps request count to a head-output limit. B1/B2
consume at most five positions; B3/B4 consume four. It does not shorten the
noncausal five-query draft body. `TP2Backend.draft` computes the largest requested
n_out in the batch; `DraftCore` masks each request's unused tail independently.
The runtime paths are shared by K2 and native. `roundcore.n_out_for` emits zero
when the whole block is available, otherwise four. Its `rounds_n_out_cut` counter
increments once per successful **engine round** with that request-count cut,
including prefill-only rounds. It does not count confidence rejections, request
rounds, rejected draft positions, or the depth policy's decisions.

The saved snapshots are all at 10,240 engine rounds. C1/X/C2 have
3,358/3,163/3,226 capped rounds (32.793%/30.889%/31.504%). They differ in request
mix and prefill/decode trajectories, and precede the end of each full suite.
X being lower does not mean native confidence is being cut harder or less hard.
No B2 round is shortened by this table while it actually contains two requests.
A nominal B3 benchmark includes the periods when requests finish and B shrinks.

There is a separate **intended cross-request policy coupling** in the scheduler.
`_dsv41_adaptive_k` uses a per-request accepted-draft EMA (alpha .3), margin 1,
minimum 2 and `AK_HI=1.6`, enabled through four requests. Any unknown/prefill
request or ANY EMA >=1.6 keeps the common batch cap. Otherwise it uses the
maximum ceil(EMA+1), clipped to the cap. CPU execution of the actual source
method (`scheduler_probe.py`, source hash in `evidence/scheduler.json`) confirms:
a request at EMA 1.59 gets window 3 alone or with another low-EMA request; adding
a companion at 1.60 changes the common window to 5 in B2 or 4 in B3. This is
verification-window policy, not changing the drafter's trained five-query body.
The EMA input adds one on fully accepted windows, so W8's overall A=1.55 cannot
be substituted for a request's EMA to claim the threshold fired. Lower native
acceptance could interact with this threshold; actual EMA/window traces are
missing. A fixed-depth diagnostic must also disable adaptive K, as the existing
fixed-depth experiment configs do.

The image's `DSparkDepthPolicy.record`, with policy `conf`, masks confidence
beyond the cap, forms prefix products, and takes the first maximum of
`(1 + sum(prefix_survival[:d])) / (ROUND_MS[d] + DRAFT_MS)` over depths 0..5.
Each request is priced independently; there is no native-specific threshold or
per-batch verification-cost table. The optional confidence calibration is off
unless explicitly selected. The inherited `DRAFT_MS=4.4` and
`ROUND_MS=28.79,31.86,34.44,37.76,41.56,43.68` are historical prices, not W8
TP2/async/native measurements. Replacing 4.4 with an RTT percentile is invalid:
exposed remote cost must include staging, dispatch and audited overlap at the
first consuming GPU operation. Lowering a depth-independent draft price can
favor shallower verification when verification slopes are unchanged.

From the two complete equal-32K runs per arm (mean of run rates):

|Nominal batch|K2 C1|Native X|K2 C2|X − mean controls|C2 − C1|
|---|---:|---:|---:|---:|---:|
|B1|1.600924|1.716017|1.481622|+0.174743|-0.119302|
|B2|1.817209|1.660215|1.815027|-0.155903|-0.002182|
|B3|1.675894|1.549727|1.667205|-0.121823|-0.008689|

The per-position decomposition uses **pooled accepted counts / request-rounds
inside each arm**, then averages the two controls; it intentionally differs
slightly from the mean-run table. These quantities are already prefix survival
counts, censored by the policy actually run. Do not multiply them again or call
them uncensored conditional acceptance probabilities.

|Nominal batch|Position 1 delta|Position 2|Position 3|Position 4|Position 5|Total|
|---|---:|---:|---:|---:|---:|---:|
|B2|-0.013057|-0.031334|-0.054604|-0.032440|-0.021999|-0.153435|
|B3|-0.043998|-0.032793|-0.030208|-0.013572|-0.001376|-0.121948|

85.66% of B2's deficit and 98.87% of B3's are in the first four positions.
Both drafters already have B3's same fifth-position cap. A fifth-position-only
change cannot account for this pattern. X proposes 0.063432 fewer tokens per
request-round at B2 and 0.050740 fewer at B3 than the mean controls. Confidence
selection could contribute, but these totals cannot separate selected depth
from proposal quality: no W8 raw confidence, chosen-depth/reached-denominator,
actual draft-batch or verify-shape trace is retained by these files.

A **native-calibrated schedule is needed before tuning or promotion**, but no
replacement schedule is established here. Measure separate native and K2
confidence reliability on reached, verified positions, and the actual per-cell
complete round cost. Use fixed depths 0..5 while keeping the trained block at
five; hold B and context steady, disable online calibration initially, and
separate cost fitting from confidence fitting. Keep B1/B2/B3, short/32K/150K,
actual graph shape/live rows and async overlap explicit. Use at least three fit
and two disjoint held-out task-family groups, following the existing N3 depth
exporter's coverage/refusal rules. Compare the candidate against the unchanged
policy using actual delivered tokens and complete round time on held-out tasks.
N6's frozen-policy confidence decisions are estimates on replay inputs; they
are not a new target acceptance measurement or a calibrated production fit.

Reproduce on Controller:

```bash
source ~/.config/ndm/paths.sh
cd /opt/dsv41/repos/.worktrees/dsv41-n6
CUDA_VISIBLE_DEVICES='' "$HARNESS_PYTHON" -m offload.dspark.native.tools.n6.window8 \
  --root /opt/dsv41/repos/dsv41-deploy/offload/results/W8-native \
  --out offload/dspark/native/tools/n6/evidence/window8.json
```
