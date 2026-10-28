# W1b current baseline — 2026-10-23 UTC

Revision f4d7898, isolated GPU-7744b44b, 325 W unchanged. Both runs completed
80 B1–4/n_out1–5/one-or-six insert rows per request/Markov hit-or-miss cases,
20 measured samples per mode, 3 warmups. All graph/eager replies identical.
Synthetic inputs, replies, expert counts and raw paired samples are retained on
x86host in `~/dsv41-offload/w1/results/profile-w1b-{burst,cadence}`.

Canonical cases below use n_out 5/5/4/4, six inserted rows per request, and
first-position cache hits. Later positions use actual generated previous IDs.

| B | burst wall ms | burst draft GPU ms | cadence wall ms | cadence draft GPU ms |
|---|---:|---:|---:|---:|
| 1 | 2.931 | 2.497 | 3.317 | 2.763 |
| 2 | 3.738 | 3.287 | 3.993 | 3.424 |
| 3 | 4.355 | 3.824 | 4.453 | 3.911 |
| 4 | 5.290 | 4.757 | 5.346 | 4.830 |

Burst first-position misses: wall3.313/3.995/4.555/5.359 ms. Insert GPU
0.304/0.326/0.403/0.395 ms for hit cases. Per-sample wall-minus-GPU residual
median0.022 ms; this is not pure CPU time. Reply packing CPU submission ~0.077 ms.
D2H CPU time2.47–4.83 ms includes the required preceding GPU execution wait;
it cannot be added to the draft GPU time as a separate transfer cost.
Graph/eager distributions are reported separately, never subtracted as savings.

Separate CUPTI diagnostic (not a critical-path decomposition): experts
0.695/1.224/1.615/2.077 ms, dense0.578/0.626/0.685/0.933, head0.604/0.618/0.626/0.634,
wo_a0.144/0.151/0.155/0.219. Expert/dense/head are first experimental priorities;
reply packing is a smaller exact candidate.

Own post-capture footprint8184 MiB (allocator peak is a different measure).
Active telemetry (utilization >=10%): burst SM median1845 MHz, range1710–1980,
power median286.18 W; cadence SM median1695 MHz, range1695–1980,
power median129.58 W. Memory clock9501 MHz throughout, cap325 W. Cadence sampling
includes idle portions; the thresholds do not isolate every kernel. This is a
different card and duty cycle from O2's historical ~1935–1950-MHz run, so the
historical timing difference is not attributed to a code change.
