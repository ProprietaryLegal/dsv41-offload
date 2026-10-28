# KX-GROUPED results — 2026-10-23

The native-expanded component works on SM86 and is ready for N1 integration and
V1 review. It retains released native values, trained activation QDQ, BF16
rounding boundaries and FP32 accumulation. Grouping pays when experts repeat;
the dedicated kernel helps tiny row counts. Native dense execution needs a
shape-specific choice, and the native BF16 head remains a significant byte cost.
No complete remote round, pair collective, acceptance or engine speedup is claimed.

## Conditions and evidence

X86host, RTX3090 GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc only, exclusive
`gpu-7744b44b.lock`, pinned CUDA visibility, acquisitions bounded to850s. No pair
access. Owner power cap325 W throughout; no clock/power changes. Driver580.126.20,
CUDA12.6, Torch2.10.0+cu126, CUTLASS3.9.2
`ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e`; sm_86-only cubin, FP32 accumulation,
no fast-math. Numerical CPU work uses 16 threads. Model source:
`deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170`,
N1's existing local shards with pinned index/full-shard hashes. No weight download
by this lane. Inputs and captures never leave X86host/Controller.

Numbers below are median of 3 CUDA-event trial averages,20 captured replays per
trial, after 3 warmups; no per-replay synchronization. Includes device routing,
QDQ, projections and output layout for expert SLOT runs; excludes TP all-reduce,
shared expert, network, target verification and engine scheduling. Dense includes
input QDQ except BF16 head, and a copy into stable input storage. Dense/head
inputs are synthetic BF16 (seed321); expert inputs are recorded live K2-control
features. These are warm repeated operation timings, not full-model cache behavior.
No timing run overlaps another user of the development card.

Telemetry samples every0.5s. Dense/cuBLAS screen: SM210–1980 MHz (active samples
1620–1980, median1695),32.8–261.29 W,45–53 C. Expert screen:
SM420–1920 MHz (active1770–1920, median1837.5),67.41–325.11 W,44–55 C.
All-stage reuse screen: SM435–1980 MHz,96.59–201.43 W,47–53 C; sampling missed
its short active bursts, so those are whole-run ranges, not kernel-only clocks.
Memory clock9501 MHz in all captured samples. Raw samples preserve timestamps.

Evidence on Controller:
`/opt/dsv41/campaigns/DSV41-OFFLOAD/KX-GROUPED-evidence/`.
Full CPU decoder evidence: `KX-GROUPED.cpu-native.json` alongside that directory.
X86host source/build/logs/results: `~/dsv41-offload/kx-grouped/`.
Proprietary route features remain in `routes-final.pt` on X86host, not in git.

## Numerical and capture gates

- 44 focused checks pass: both dtypes, grouped/dedicated experts, stable device
segments, empty experts, changing IDs under replay, SLOT order, invalid-ID poison,
nonfinite propagation, strided wo_a, cuBLAS FP32 output, automatic dispatch,
unsafe reduction-flag refusal, all head shard/tail widths and lowest-ID ties.
- 1,177 official scaled tensors (1,152 expert matrices plus25 dense/wo_a/main_proj)
match N1's independently implemented decoders, including signed zeros. Every
value survives BF16 and FP16 round trips exactly. This does not admit narrowing
other native BF16 tensors or arbitrary activations.
- 2,304 expert matrix/cut cases (3 stages ×128 experts ×3 projections ×2 cuts)
compare every output channel with independent CPU FP64 dots. Native QDQ matches
N1's `formats.act_qdq` exactly on live inputs from all3 stages.
- 5,701,632 projection elements:99.9997544563% BF16 agreement;14 disagreements
(11 gate,3 up,0 down), listed with values, indices and rounding midpoints in
`qualify-final.json`. One reference is exactly on a BF16 midpoint; the others
are within8.94e-8 of one. Maximum relative FP64 error1.49078e-7;
maximum relative FP32-reference disagreement1.86344e-7. These are diagnosed
rounding-boundary differences, not a claim of bit identity to official MMA order.
- Dense FP32 relative errors are <=6.76e-6 in the measured synthetic samples.
Head epilogue is exact against its specified BF16 base/bias/sum policy.

## Real reuse and expanded expert timing

Replayed all453 captured rounds sequentially through the frozen K2 worker:
443 live,8 warmup,2 mid-prefill. There are337 live T5 rounds and106 live T10
rounds; no recorded T15/T20 rounds. Across all live rounds, unique expert counts
are stage 0:4/11/22, stage 1:4/8/17, stage 2:3/5/12 (min/median/max).
This describes the K2 control; native routing must be measured independently.

Actual per-stage native BF16 rank-0 weights, actual live inputs/IDs/weights,
median-reuse case at each recorded T, SLOT output; microseconds:

|stage|T|unique experts|grouped|dedicated|
|---|---:|---:|---:|---:|
|0|5|10|437.09|609.64|
|1|5|7|313.24|609.28|
|2|5|5|239.20|609.08|
|0|10|16|687.82|1204.89|
|1|10|11|481.38|1208.12|
|2|10|7|318.26|1210.42|

At maximum observed unique counts, grouped T5 is605.54/480.82/394.39 us
(stages0/1/2); grouped T10 is927.85/727.76/550.25 us.
Live-derived1–3-row slices favor dedicated: BF16 row1 grouped153.96 vs137.21 us;
row3 grouped392.55 vs377.60 us. Row4 grouped475.60 vs495.72 us.
Thus the fixed expert dispatch uses dedicated at1–3 rows and grouped otherwise.

T15/T20 controls repeat a real five-row input, preserving14 unique experts:
BF16 combined-partial timing609.33/613.58 us grouped versus1815.19/2415.10 us
dedicated. These are labeled repetitions, not measured three/four-request traffic.
Low-reuse FP16 SLOT controls use real activations but synthetic IDs: at T20,
all60 experts distinct gives2485.96 us grouped /2418.43 us dedicated;
42 distinct gives1749.71 us /2418.89 us. Grouping cannot avoid native weight bytes
when every route is unique. No CPU count read or dynamic host dispatch is added.
BF16/FP16 live-derived expert results agree within2.58e-8 relative Frobenius
between grouped and dedicated in the recorded screen. BF16 remains the default.

## Dense and head screen

Same inputs and weights within each native comparison; microseconds:

|operation / rows|CUTLASS native|cuBLAS native|EXL3 control|
|---|---:|---:|---:|
|main_proj /8|228.71|196.66|146.38 K5|
|main_proj /32|238.54|206.75|191.84 K5|
|main_proj /128|491.57|328.76|630.12 K5|
|wq_a /5|56.01|24.52|20.99 K5|
|wq_a /20|57.80|27.34|29.08 K5|
|wkv /5|51.30|14.75|20.83 K5|
|wkv /20|52.07|19.51|28.42 K5|
|wq_b /5|57.04|56.67|28.36 K5|
|wq_b /20|58.62|66.71|44.95 K5|
|wo_b /5|66.87|59.90|36.92 K5|
|flat grouped wo_a /5|59.24|49.92|33.2 FP8 control*|
|flat grouped wo_a /20|65.02|54.37|58.7 FP8 control*|
|head64640 /1|810.29|824.68|434.43 K6|
|head64640 /4|802.25|785.46|429.41 K6|
|head64640 /20|810.96|849.15|897.84 K6|

*FP8 control numbers are from the first screen and omit native input QDQ; the
expanded wo_a timings include it. These are not an equal-arithmetic speed ratio.
K5/K6 controls deliberately retain different checkpoint representations. Main
control: `Exl3Linear`, FP32-accumulator extension, power-of-two prescale,
force_shape0; other controls: frozen TP2 `Exl3Mat`/`KernelPlan`. They do not
establish a full-worker improvement. No norm or confidence work is included.

Existing EXL3 TP2 uses64768/64512 vocabulary columns, not a balanced64640 split.
Both widths were also executed explicitly. At4 rows, native CUTLASS is809.22 /
806.20 us and K6 is315.28 /299.06 us. Head epilogue is approximately12.5–29.9 us
for1–20 rows. The native BF16 head is a separate fidelity arm, not a replacement
for the target's head. Keep target taps, target head and sampler unchanged.

Static native dense dispatch selects cuBLAS for measured main_proj/wq_a/wkv/
wo_b/flat-wo_a shapes; CUTLASS for wq_b/head and unmeasured shapes. It never
silently falls back to EXL3. The best hybrid profile needs N1/V1 worker-level
comparison, retaining trained QDQ for any native packed dense path.

## Memory and limits

One stage's128 routed expert channel shards occupy4,529,848,320 bytes =4320 MiB;
measured load allocator peak equals that final storage. Three stages imply
12960 MiB =12.65625 GiB per rank for routed expert weights only. This is not a
whole-worker fit claim. T20 per-instance scratch/SLOT/output storage is4,531,028
bytes; graphs/driver/module/communication/dense/shared/cache reserves are separate.
All-stage reuse process peak allocated4,532,220,928 bytes. Dense screen peak
2,285,586,432 bytes. The streamed numerical oracle peaks44,900,864 bytes.
The optional BF16→FP16 admission probe temporarily retains both representations
and full FP32 comparison temporaries: peak15,863,255,552 bytes. That is a
qualification cost, not the proposed production loading strategy. N1 must stream
directly into final shards, and owns complete pair load/serving memory budgets.

The first dense attempt failed with `ModuleNotFoundError: worker.tp2` because
W1's old staging copy predates TP2. Fixed by staging the current worker source
in this lane's own reference directory; subsequent complete screens pass.
CUTLASS alone is not the fastest dense choice. Expanded bytes dominate the head
and low-reuse experts. No source fidelity or acceptance conclusion is inferred
from agreement with EXL3 K2. Full TP2 model integration, exact per-slot collective,
V1 independent full-prefix/logit review and coordinator E3/E4 engine benchmarks
remain with their owning lanes. No services, Spark files, sampler, shared target
weights, other GPUs, power/clock settings or external applications changed.

## Reproduce

On X86host, source `~/.config/ndm/paths.sh`. The lane directory contains the pinned
source, build receipt, frozen current-worker reference and N1 formats snapshot.

```bash
source ~/.config/ndm/paths.sh
grouped_root=$HOME/dsv41-offload/kx-grouped
grouped_python=$HOME/dsv41-offload/o2/venv/bin/python
grouped_donor=/opt/dsv41/x86host-data/donors/cutlass-v3.9.2
"$grouped_root/src/run_locked.sh" "$grouped_python" -u "$grouped_root/src/verify.py" \
  "$grouped_donor" "$grouped_root/build" "$grouped_root/verify-reproduce.json"
"$grouped_root/src/finish_queued.sh"
```

The second command sequences focused tests, capture replay, dense/head and
experts screens, and the all-stage oracle. It releases the GPU lock between
sub-items, retries only lock contention, and stops on any actual failure.
For all-stage actual reuse timing use bench.py phase `reuse` with the same
arguments as phase `experts`. CPU decoder comparison requires
`CUDA_VISIBLE_DEVICES=` and cpu_validate.py (see its CLI). N1 integration uses
README.md's SLOT/QDQ/rounding contract. This lane installs no service: rollback
is to keep the existing worker/profile and not select this new component.
