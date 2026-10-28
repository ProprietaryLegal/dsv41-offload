# STOP: measured primitives do not qualify an exposed production win

All values below are labelled. No live request, Spark access, pair work or
production-K2 full-model replay occurred. [Frozen PP2 inputs](pp2-frozen.json)
were read with git show from e1befc9, never refitted. [Cost model](estimate.py)
and [machine-readable result](cost-estimate.json) preserve unknown costs as null.

MEASURED on x86host's solo bc62df2a at225W, required miniconda CPython3.13,
Torch2.12.1/cu130 and Triton3.7.1; graph-replayed primitive timings:

| Batch | Depth4 top32 median/p95 ms | Depth5 top32 median/p95 ms | Depth5 torch argmax median ms |
|---|---:|---:|---:|
| B1 | .143155/.143258 | .155443/.155546 | .130970 |
| B2 | .141926/.142029 | .177357/.177459 | .133325 |
| B3 | .150221/.150323 | .187699/.187786 | .133318 |
| B4 | .166912/.167110 | .208589/.208691 | .134144 |

These time already-conditioned synthetic logits, not Markov evaluation, the
current production fused selector, K2 backbone, real TP2 exchange or eager
sampled serving. B1 incremental selector cost versus this torch microbenchmark
is .024474ms; B4 is .074445ms. Neither grants replacement credit against the
production selector. The producer uses two launches/position and misses the
absolute .05ms/round goal at every measured shape. Its prototype model hook
still runs the original greedy selector and materializes extra Markov logits;
there is no whole-K2 or exposed-pair timing for that path.

MEASURED B1depth5, full target vocabulary129280, actual prepared standard
rejection kernels on the same solo: full q-aware verifier p95 .014029ms versus
point-mass verifier .013005ms; sparse1280-byte D2H .003859ms, H2D .004307ms;
dense-q zero+scatter .004506ms. These exclude target model/processor time,
CPU validation, slot index-copy and GB10 differences. Five toy-law tests pass
at family alpha1e-5; the numerical test is not a hardware timing transfer proof.

ESTIMATED framing: B1depth5 is1416B/reply including app header, sparse payload,
semantic header/mask and CRC; B4 is5412B. At the previously measured40Gb/s link
negotiation, B1 serialization is .0002832ms, or .0005664ms charging two serial
copies at40Gb/s. This is shape arithmetic, not a network-latency measurement.

Adding the full producer, full q-aware verifier, two transfer proxies, dense
expansion and two wire copies gives **.182813ms** in known B1 proxies. This
charges full verifier cost rather than only its delta, gives no greedy credit
and does not assign a value to missing work. Summing p95 operations is a
conditional cost model, not a measured p95 of a pipeline or a statistical upper
bound. Unknowns include extra Markov work, eager launches, full vocabulary peer
staging, CPU CRC/q/proxy work, target slot copies/TP synchronization, GB10 work,
quantizer differences from PP2 and future sampled-policy occupancy.

PP2's pooled nominal top32/.7 fixed5 expectation rises from3.091813 to3.305514
retained tokens/round; paired delta interval .175370–.254434. With its archived
48.08ms fixed5 policy price (not an observed matched mean), the pooled lower
interval allows2.727129ms total added cost. After known proxies,2.544316ms
remains for unknowns. Family lower-interval remaining budgets are2.320812 and
2.369660ms; the16–32K proxy is tighter at1.365677ms. PP2 used float64 normalized
q, whereas this implementation samples emitted dyadic24 q; its acceptance is
not newly replay-qualified. There is no150K acceptance capture.

For the requested B1 short/32K/150K view, the following are **ESTIMATED
sensitivities**, not forecasts. Each transplants the relevant fixed5 replay's
relative token-count gain onto a different historical adaptive W8 time/acceptance
population. The32K row uses the16–32K proxy;150K additionally extrapolates pooled
anchors no longer than47K. Changing the unknown-cost assumption changes the
answer substantially. No actual expected live rate is reported.

| B1 context | Historical adaptive tok/s | Conditional tok/s with unknown extra0ms | With extra1ms | With extra3ms |
|---|---:|---:|---:|---:|
| Short |53.56|58.01 (+8.32%)|56.52 (+5.52%)|53.74 (+.34%)|
|32K proxy|62.64|65.31 (+4.25%)|63.71 (+1.71%)|60.74 (−3.03%)|
|150K extrapolation|55.23|58.78 (+6.42%)|57.34 (+3.82%)|54.66 (−1.03%)|

The0ms column illustrates omitted costs; it is not the decision model. Actual
net rate and150K acceptance remain null. The current prototype fails the
single-launch/absolute timing objectives and lacks real K2/pair/GB10/lifecycle
qualification. Preserve production; finish those gates before the separately
pre-registered [69-minute coordinator window](LIVE-AB-PLAN.md).
