# PF1-R2: timing boundaries of the reported long-context loss

All values below are **MEASURED from existing files**, not new inference. Reproduce:

```bash
source ~/.config/ndm/paths.sh
"$HARNESS_PYTHON" -m offload.prefill_maxlen.request_evidence \
  --results /opt/dsv41/repos/dsv41-deploy/offload/results/K1 \
  --production /opt/dsv41/campaigns/DSV41-OFFLOAD/K1-production \
  --output /opt/dsv41/campaigns/DSV41-OFFLOAD/PF1-R2-evidence.json
```

The numeric-only artifact contains 152 request records, 73 run windows, 16 serve
excerpt/memprobe inventories and SHA256s of 39 source files. R0/R1/R1p/R2p/R5
suite request JSONL, production K1 suite request JSONL, R5 needle and prefix rows
are included. Suite logs repeat run summaries; do not count those as independent
requests. The original files remain unchanged. No prompt or generated text is
copied. Missing measurements remain null.

| Saved request (B1 cold) | Actual uncached prompt | TTFT s | Full response s | After first token s | Reasoning / answer tokens |
|---|---:|---:|---:|---:|---:|
| R1 c150 rep0 | 150000 | 68.990248 | 96.988528 | 27.998280 | 1500 / 0 |
| R1 c150 rep1 | 150000 | 68.856152 | 98.015233 | 29.159081 | 1500 / 0 |
| R1 c300 rep0 | 300000 | 140.214732 | 167.089770 | 26.875038 | 1500 / 0 |
| R5 c150 rep0 | 150000 | 93.341295 | 120.227054 | 26.885759 | 1500 / 0 |
| R5 c150 rep1 | 150000 | 93.369439 | 120.585874 | 27.216435 | 1500 / 0 |
| R5 c500 rep0 | 500000 | 322.426173 | 353.557386 | 31.131213 | 1500 / 0 |
| K1 production c150 rep0 | 150000 | 68.623818 | 96.891978 | 28.268161 | 1500 / 0 |
| K1 production c150 rep1 | 150000 | 68.669066 | 95.991242 | 27.322176 | 1500 / 0 |

R5 c500 has three literal 500000-token requests (B1 once, B2 once); server usage
and recorded occupied chat counts agree, total 501500 each, within 524288.
This closes literal occupancy only: the 1500-token budget ended in reasoning,
so it is not evidence of a completed synthesis or useful visible answer.
B2 timings include interleaving; do not interpret them as isolated prefill.

**TTFT is a prefill proxy, not a GPU prefill timer.** It contains queueing,
scheduler/metadata, host/network and first-generation work. The R1/R5 c150
median TTFT difference is 24.432167 s (+35.45% latency, ~26.17% fewer prompt
 tokens/s), while generation delivery does not slow correspondingly. Together
with the short-prompt result in DIAGNOSIS.md, this puts the regression before
first delivery. It does not identify attention, a native kernel or a collective.
The literal 500K cold B1 proxy is 1550.74 prompt tokens/s (500000/322.426173).
Its full-response ratio is 1414.20 prompt tokens/s; 8.81% of its full elapsed
occurs after TTFT. Neither ratio is an engine-throughput measurement.

| Non-streaming R5 needle | Actual uncached prompt | Full elapsed s | Completion (reasoning / other) |
|---|---:|---:|---:|
| nominal 450K rep0 | 449613 | 290.19 | 0 / 42 |
| nominal 450K rep1 | 449597 | 287.73 | 0 / 43 |
| nominal 500K rep0 | 498245 | 319.94 | 0 / 43 |
| nominal 500K rep1 | 497825 | 319.85 | 0 / 43 |

These have **no TTFT or segment timestamps**. Their prompt/elapsed ratios
(~1549–1563 tokens/s) include generation, so the reported approximately 28%
450–500K prefill drop is not established as a measured GPU prefill drop.
There is no matched 327680-cap 450–500K request (it would not fit). Comparing
shorter R1 with these R5 requests also changes prompt family, prefix length
and generation budget. No precise fraction of that 28% can be attributed to
prefill from these rows. The separate matched c150/c32 TTFT loss is measured.

R5 prefix identity: actual prompt 249865 (cold/control uncached 249865; hit
uncached 265), elapsed cold/hit/control 157.80/1.76/157.64 s. At actual 499942:
uncached 499942/358/499942, elapsed 321.98/2.23/321.93 s. Usage/segment counts
and TTFT are absent, so a hit cannot be subtracted from a cold response to
produce exact GPU prefill. The partial-prefix row records 399744 cached and
399994 common tokens but omits actual total prompt count and timing; nominal
500000 must not be used to invent the uncached suffix.

Per-request `first_by_kind_s`, `last_by_kind_s`, delivery spans, SSE arrival
times/gap percentiles and token sources are preserved when available. R5 c500
B1 has 727 content-bearing events; delivery-gap p50 42.564 ms, max 92.521 ms.
These are client delivery events, **not authoritative engine/verify steps**.
The old bench comment calling them direct round timers is not an evidence
contract. No saved event during prefill exposes a 4096-token chunk boundary.

The twelve `K1/extra/lines-*.txt` files contain only load/KV/env lines, not
per-request serve traces. The four memprobe files contain memory snapshots
(`ev/layer/tag/toks/before/peak_over/net/reserved/t`); the artifact also preserves phase/step host timestamps, n/kind/toks/reqs,
consecutive logged-step host end spacing (only when n increments by1), and
Engram reader served counts, service-time sums and running maxima where
present. These are real saved timing counters. The timestamps are taken after
enqueue, not synchronized GPU completion. Reader service counters do not
measure the exposed wait on the target critical path. They have no request or
prefix locator and cover R1p/R2p at327680, not a matched R5 long-context arm.
They cannot establish the fraction of the R5 drop spent in prefill. There is no
saved per-chunk CUDA time, dispatch duration or exposed collective wait to
extract. S18's unchanged clock assessment is useful historical context;
two-rank power/throttle/clock time series aligned to these requests are still
needed. All kernel/threshold/prefix-growth attribution requires the new window.


The extracted memprobe inventory is 9416 timestamped records: 1672 step
records and 7744 phase records across both ranks/R1p/R2p. Each file has418
steps and1936 phases. There are384 consecutively numbered prefill-end spacings
per file; their medians are0.8010s (R1p rank1),0.8015s (R1p rank0),0.8050s
(R2p both ranks). **MEASURED host record spacing**, across varying step shapes
and requests, not GPU chunk durations and not an R5 comparison. Engram reader
counter arrays occur on all1672 step records and are retained numerically.
