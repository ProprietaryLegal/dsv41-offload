# N6 identical-input batch probe

This lane tests batch dependence, not official-arithmetic fidelity. It changes
no serving source, weights, deployment configuration or identity. N4/N5's
unresolved qualification gates remain unresolved.

`batch_probe.py` runs both actual TP2 channel shards on X86host's assigned solo
3090, with local Torch collectives. The frozen N4 runtime is used read-only;
prebuilt CUDA extensions are imported directly and existing Triton cache entries
are looked up without writing. Missing binaries/caches refuse. A Python audit
hook rejects filesystem mutations except opening the existing assigned-card
lock and `/dev/null`; bytecode and CUDA disk caches are disabled. The runner
uses `flock -n`, checks that the card is idle at its verified 225 W cap, pins the
full UUID, and bounds each owned process to 830 seconds plus five seconds of
teardown. It never accesses a Spark or starts a listener/service.

Seven fixtures comprise twelve early captured states from N4's existing V1
scope and nine synthetic states. The captured fixtures are chronological states
of one original request, placed in independent slots; they are not independent
legal-task groups. Synthetic state covers short/64/128 boundaries, ring wrap,
32K and 150K absolute positions with the production hidden/window/head shapes.
Synthetic random main_x values are stress inputs, not a quality benchmark.

For each fixture, every request is first run alone. The probe then runs every
ordered pair and every triple permutation at n_out 4/5, comparing each request
against its unchanged B1 state. It separately checks same-B companion/order
changes, graph/eager equality, n_out prefix equality, zero unused tails and
unchanged window storage. Head confidence differences feed a separately labelled
**estimated** application of the inherited depth formula, with the same cap for
both compared confidence vectors. No target inference/acceptance is measured.

A separate insertion check compares graph insertions per request against packed
insertions at 1/2/5 new rows per request. This deliberately crosses graph bucket
sizes; source-dependent numerical differences are reported rather than silently
classified as a state defect. It uses synthetic embedding-derived taps, not a
new capture. The draft comparison holds window bytes fixed to isolate draft
packing from projection/insertion dispatch.

The captured source policies are N4 d1dfe13, matching W8's active native path.
This lane's parent adds three inert D2 hooks: optional adapter_wo_a, optional
adapter_guard checks, and an explicit evaluation-core entry point. None is
selected by these native/K2 arms. Source and prebuilt-extension hashes are
retained in receipts. All probe tools live under an identity-excluded `tools/`
directory; native runtime identities are unchanged.

Reproduce from this worktree, launching the long job detached on Controller:

```bash
source ~/.config/ndm/paths.sh
cd /opt/dsv41/repos/.worktrees/dsv41-n6
tmux new-session -d -s dsv41-n6-reproduce \
  'source ~/.config/ndm/paths.sh; bash offload/dspark/native/tools/n6/run_readonly.sh /opt/dsv41/campaigns/DSV41-OFFLOAD/N6-reproduce'
```

Then reduce each successful log (require the runner's `.exit` to be zero):

```bash
source ~/.config/ndm/paths.sh
cd /opt/dsv41/repos/.worktrees/dsv41-n6
for arm in k2-tp2 native-hybrid-kx2; do
  "$HARNESS_PYTHON" -m offload.dspark.native.tools.n6.collect_probe \
    "/opt/dsv41/campaigns/DSV41-OFFLOAD/N6-reproduce/$arm.log" \
    --out "offload/dspark/native/tools/n6/evidence/$arm.json"
done
```

The logs and committed evidence contain aggregate comparisons and source hashes;
no token IDs, captured tensors, prompts, generated text or weights are emitted.
Raw inputs stay in their existing private locations. Measured same-input results
are recorded in the lane report and `evidence/{k2-tp2,native-hybrid-kx2}.json`.
The confidence/depth audit and W8 arithmetic are in [DEPTH.md](DEPTH.md).

Source audit of batch-dependent paths:

|Path|What varies with batch|Isolation finding|
|---|---|---|
|Block inputs / attention|B×5 rows; native online64 launch grid; each program reads its own slot/lo/last and block offset|Five-query body is retained; no cross-request softmax or chronological ring reorder|
|Native dense dispatch|M5/M10/M15 choose prepacked FP8; selected projections use split-K 8 at M5 and 4 at M10/M15; insertion main_proj uses FP8 at M8 and BF16 at other buckets|Arithmetic reduction schedule can vary, while weight values and the per-32 QDQ rule are preserved|
|Native expert dispatch|Stage 1 at ten rows selects mode 256; other stage/row cells select 131072|Scheduling groups routes by expert, but output addresses retain original row/slot; measured companion/order control tests this|
|K2 control|EXL3 plans key on exact body rows; insert plans key on 8/16/32/64/128; optional fused pairs remain off|K2 also permits batch-shape arithmetic variation|
|Windows / insertion|Both ranks own per-slot position/KV rings; real inserts are padded into a scratch slot|Proposal calls do not commit window state; insertion is checked separately|
|Head / Markov|Same K6 kernel bucket at every B/n_out; head takes B×n_out rows; previous IDs and bias lookup remain per request|No shared previous-token index or flat row-offset coupling; native retains FP32 head/bias sum, K2 keeps deployed BF16 policy|
|Graphs / server|Exact (B,5,n_out,margin) graph keys; server zeroes each request's unused tail after a batch-max n_out call|Graph/eager and n_out-prefix checks distinguish dispatch from stale buffers|

The source audit identifies reachable batch-dependent arithmetic paths; it does
not attribute every confidence difference to one projection or reopen N4's
arithmetic-fidelity work. N6's scope is request packing and policy selection.

Final measured results (2026-10-24, existing Torch 2.10.0+cu126 environment):

|Input scope|Arm|B2 equal positions 1..5 / 48 or 36 comparisons|B3 equal positions 1..5 / 72 or 54 comparisons|
|---|---|---|---|
|12 captured states|K2|48,48,48,48,48 / 48|72,72,72,72,72 / 72|
|12 captured states|Native kx2|48,48,48,48,48 / 48|72,72,72,72,72 / 72|
|9 synthetic stress states|K2|28,28,20,16,12 / 36|42,42,36,18,12 / 54|
|9 synthetic stress states|Native kx2|32,32,28,28,28 / 36|48,48,42,42,42 / 54|

The table uses n_out=5 to expose every position at both B values. n_out=4
preserves the same first-four agreement; all output-prefix comparisons are
exact. Counts include repeated placements of the same states, not 48/72
independent samples. Graph/eager: 168/168 per arm; n_out prefix: 84/84 per arm;
unchanged rank-window snapshots across proposals: 14/14 per arm. Same-B
companion/order IDs **and confidence** are exact in all 126 B2 and 210 B3
non-self comparisons per arm, across both input scopes.

Captured B1-versus-packed maximum confidence changes are .00758338/.00623515
for K2 B2/B3, and .05238831/.05238831 for native. Under the **estimated inherited
policy** at the production cap (B2/n5, B3/n4), native changes depth **3→2 on one
of the twelve states**, repeated at all four/six placements (4/48 and 6/72).
K2 changes depth on none. The B3 comparison masks BOTH confidence vectors at
four before depth choice; this isolates confidence sensitivity from the cap.
This is a concrete policy-calibration lead, not W8's measured causal attribution.

Insertion metadata is exact throughout. One or two rows/request yield exact KV
values in both arms. Five rows/request cross graph bucket 8→16 and change 499
KV elements/rank in K2 (maximum absolute .125), versus 67/rank in native
(maximum .0625). Both ranks give the same counts. No arithmetic tolerance is
introduced and no native fidelity gate is rerun or waived.

Final K2/native job durations are 7.43/460.89 seconds including load, graph
capture and checks, **not draft latency**. Torch allocation high-water is
9.060/12.499 GiB on the solo card, **not physical-pair serving memory**.
Both final jobs exit zero and release their locks. The last native release is
02:01:47Z; SMI then shows 1 MiB on the assigned card at the unchanged 225 W cap.
The final logs are `/opt/dsv41/campaigns/DSV41-OFFLOAD/N6-final-results/`.
All 194 recorded source hashes match the frozen N4 d1dfe13 tree exactly; both
arms used the same source set. Prebuilt extension hashes are in each receipt.
No dependencies were installed.
