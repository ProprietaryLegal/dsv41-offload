# SO2 CPU scenario estimates — not GB10 performance measurements

Reproduce with `python -m offload.bg_prefill.simulate --output <new-json-path>`
under the configured interpreter, `CUDA_VISIBLE_DEVICES=` and `PYTHONPATH=.`.
The JSON contains eight matched scenarios plus 24 sensitivity scenarios. It holds
aggregates only. Every arm executes actual pinned image schedule statements
through D2-FIX1's CPU loader; no network, teacher, model forward or GPU is used.

## Inputs and accounting

**MEASURED source, not measured by SO2:** COMPARISON.md, Window 8 C1/C2, single
stream short/c32/c150 decode steps ~37–41 ms; 150K prefill ~69 s in the async
comparisons implies ~2170 prompt tok/s, rounded to the brief's ~2200. Installed
batch budget 4096, long-prefill threshold 1024, throttle floor 5/cadence 24, four
request slots and up to four additional draft slots/request. 4096 is the total
budget, not each mixed request's native chunk size: native mixed chunks are 1024.

**ESTIMATED:** use 39 ms per foreground decode round plus prompt tokens/2200,
additively; 2 delivered target tokens/round, including bonus/correction, excluding
rejected proposals. This is an assumption (roughly W8 short 1.01 accepted+1), not a
new acceptance measurement. Keep one synthetic anchor pending in the CPU fixture;
virtual delivered-token accounting is separate. No inference tokens are produced.
Use one foreground already decoding next to one/two admitted 60000-token priority 20
prefills, 300 foreground rounds (600 estimated delivered tokens). Native and SO2
use the SAME costs, acceptance, initial work, cadence and budgets. Backgrounds
finish after prefill in this workload; background decode has separate CPU tests.

## Results (all ESTIMATED)

Foreground rate over all 600 output tokens, including prefill stalls during decode:

| Foreground priority | Background 60K prefills | Native tok/s | SO2 tok/s | Native background work before foreground ends | SO2 background work before foreground ends |
|---|---:|---:|---:|---:|---:|
|0|1|34.71|50.53|12288|384|
|0|2|26.23|49.80|24576|768|
|10|1|15.40|50.53|60000|384|
|10|2|9.06|49.80|120000|768|

For priority 10, native 0031 floor 5 does not throttle auxiliary prefill at all:
while background is still prefilling, the native overlap-only rates are 3.99/2.08;
once that finite prefill finishes, foreground returns to solo speed. Reporting
only that overlap-only native rate would exaggerate the finite-task comparison.
Priority 0 is protected by native cadence 24; its background stays in prefill for
the entire foreground output window in both arms.

SO2 services each background on calls 24/48/.../288, 32 tokens each time. Maximum
observed service gap is 24 calls under continuous foreground decode. The smoother
foreground comes from **less background work during that interval**, not a faster
model. At those default costs the shared-round increment on each due call is
14.55 ms per background, 29.09 ms for two. Foreground-only ideal is 51.28 tok/s.

After foreground finishes, drain EVERY background prompt in every arm. All arms
process exactly 60000/120000 background tokens plus 600 foreground tokens. The
zero-fixed-overhead additive model gives IDENTICAL full-work makespans of 38.973 /
66.245 s. There is no aggregate throughput gain under this model. Background
completion is delayed. With continuous foreground, 32 tokens/24 calls represents
roughly 34 background prompt tok/s/request at these costs: bounded progress is a
small guaranteed share, not comparable background throughput.

Sensitivity (foreground 0, 37/41 ms decode and 0/20/55 ms **assumed**, not measured,
fixed cost per scheduled background prefill batch):

| Backgrounds | Native foreground tok/s range | SO2 foreground tok/s range | Native all-work s range | SO2 all-work s range |
|---|---:|---:|---:|---:|
|1|32.35–35.96|45.68–53.22|38.37–40.89|38.37–43.48|
|2|24.20–26.94|42.95–52.41|65.65–73.34|65.65–74.66|

Those ranges span multiple assumptions, not confidence intervals. Small quanta
can cost aggregate time if per-batch overhead is material; no promotion follows.
Nonlinear chunk efficiency, expert reuse, TP/remote drafter work, acceptance,
async pipeline depth, prefix-length attention cost, worker graph shapes, scheduler
CPU time and first visible answer are not modeled. Actual throughput may differ
substantially. A lower call gap is not a proven wall-clock or ITL bound.

## What the donor claim would mean here

**UPSTREAM CLAIM:** 31.3/22.5 → 57.4/56.5 tok/s, one/two background 60K prefills;
ratios 1.834x/2.511x. The donor used different priorities, worker, clock, costs and
cadence 100000, with indefinite background pause; 0073/cadence shares were not
isolated. The pinned 0074 patch also quotes 58.5/59.1 in a related test; these are
not interchangeable replicates of the brief's 57.4/56.5. We reproduce neither.

Normalize donor rates by its quoted 55.9 solo control and multiply by this model's
51.28 solo: **ESTIMATED normalized illustration**28.71/20.64 →52.66/51.83.
The slight above-solo on-values inherit donor noise/acceptance differences; they
are not feasible predictions in our fixed-acceptance additive model. A literal
57 tok/s at 39 ms needs ~2.22 delivered tokens/round, rather than the assumed 2.
Our bounded policy estimates 34.71/26.23 → 50.53/49.80 for foreground 0, a smaller
relative gain with positive background progress. Priority 10 requires its own
control because the installed floor 5 affords it no cadence protection.

## Foreground arrival limit

0074 reduces a solo background's existing chunk from 4096 to 1024. At 2200 tok/s,
that is 1.862 → 0.465 s of prompt compute per in-flight chunk; two async chunks
would be 3.724 → 0.931 s before other costs. These are arithmetic ceilings under
a linear cost assumption, not measured TTFT bounds. A later foreground arrival
cannot recall an already-running chunk. When a foreground waiter is visible,
SO2's admission mode allows each existing background 32 tokens (~14.55 ms) rather
than excluding it; native admission can still refuse KV. This is deliberately a
bounded residual cost, not a promise of zero waiting.
