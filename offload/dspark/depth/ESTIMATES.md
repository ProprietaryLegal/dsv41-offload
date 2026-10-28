# SP2 aggregate screen: assumptions and recommendation

**No measured new optimum is identifiable from the available data.** Keep the production depth policy until the
coordinator measures the fixed-depth cells. Test cost-only changes first, then a cap-three arm, and retain cap five
as a countercheck for censored late-position acceptance. Do not infer that async requires deeper verification.

`TABLES.md` and `screen.json` are generated from MEASURED bench2 counters and direct SSE step timers, using only
aggregate numerical fields. Every candidate throughput is ESTIMATED. The JSON preserves each input filename and
SHA256; it contains no prompts, tokens, outputs or tensors. Context zero denotes the suite's short prompt, not an
actually empty context. Input target contexts are nominal; retain actual occupied tokens in the coordinator run.

## Reproduce (CPU only, no server access)

```bash
source ~/.config/ndm/paths.sh
CUDA_VISIBLE_DEVICES='' "$HARNESS_PYTHON" -m offload.dspark.depth.screen \
  --results-root offload/results \
  --coordinator-results-root /opt/dsv41/repos/dsv41-deploy/offload/results \
  --out-dir offload/dspark/depth
CUDA_VISIBLE_DEVICES='' "$HARNESS_PYTHON" -m pytest -q offload/dspark/depth/test_screen.py
CUDA_VISIBLE_DEVICES='' "$HARNESS_PYTHON" -m mypy --strict --follow-imports=skip offload/dspark/depth/screen.py
```

Requested MEASURED input families:

- `offload/results/W4-E1p/W4-E1p-{short,c32,c32x3,c150}.jsonl`: TP2 + async process/nap.
- `offload/results/W5-E1p/W5-E1p-{short,c32,c32x3,c32x4,c150}.jsonl`: TP2 + async, R3 runtime.
- `offload/results/W5-prod/W5-prod-{short,c32}.jsonl`: ONE-CARD async production, reference only.
- Coordinator tree `offload/results/K1/R1-{short,c32,c150,cap4}.jsonl`: ONE-CARD async with KV packing, separate
  reference. MEASURED telemetry in R1-c32 shows c4e7a371 at 8,352 MiB and 7744b44b at one MiB; the W6 runner logs
  refer to the one-card production worker. It is not a TP2 replication.
- `offload/results/n2-pair-w5/summary.json`, `k2-tp2.draft_p50_us`: MEASURED graph p50 at B1/B2/B3/B4 is
  2.0275 / 2.5122 / 2.8961 / 3.7219 ms for k5/n_out5. At n_out4 it is
  2.0002 / 2.4433 / 2.8196 / 3.2798 ms. These exclude some full STEP costs; they are not exposed async latency.
  N1's earlier `COMPARISON.md` pair table is consistent (MEASURED 1.94 / 2.44 / 2.75 / 3.62 ms).

The available pure cells cover all four batch sizes, but not their full cross-product with context and worker.
B2 comes only from K1; B4 short/150K and B2 short/150K are absent. Never fill them with invented measurements.
Mixed prefill/decode runs are deliberately excluded from this steady decode model, and must be checked live.

## Equations and limits

Repo convention: `R = (1 + E[A]) / C`, with accepted drafts A excluding bonus and complete-round C.
The batch formula below extends the numerator across active requests; its SSE-derived cost remains an ESTIMATED
proxy. See [metric boundaries and bonus-inclusive paper comparisons](../../evidence/THROUGHPUT.md).

For each run, S_i = accepted_position_i / request_rounds is MEASURED, already a prefix survival. Across repetitions
we pool counts/rounds; we never multiply S_i again. An ESTIMATED steady full-batch numerator at depth d is
`E_B(d) = B * (1 + sum(S_i, i<=d))`. This is a truncation proxy on the OLD policy's trajectories. Confidence and
scheduler censorship omit acceptance outside verified positions; it is not an uncensored static-depth experiment.
Changing depth also changes future anchor positions. Missing exposure denominators prevent correcting that bias.
Observed dynamic tail is consequently a conservative numerator proxy, not a proven lower bound on new full-run tps.

Let T_B be the median of each run's median request step-p50. It is a MEASURED summary of SSE event gaps, not a
pooled GPU-event median or a fixed-depth target forward. All following assignments are ESTIMATED assumptions:

```text
c_B = 5 at B1/B2, 4 at B3/B4                    (nominal cap anchoring; actual depth may be smaller)
D_B = measured TP2 graph p50 above              (held constant across consumed depths; block stays five)
e_ref = .25 * D_B + .30 ms                     (75% assumed overlap + dispatch allowance)
V_B(d) = T_B - e_ref + .5 * B * (H[d] - H[c_B])
round_B(d) = V_B(d) + (1 - overlap) * D_B + overhead
rate_B(d) = 1000 * E_B(d) / round_B(d)
H = [28.79,31.86,34.44,37.76,41.56,43.68] ms   (historical CONFIG slope only, not new MEASURED prices)
```

Nominal cap anchoring is deliberately explicit: it makes a screen possible, but can overstate savings when the old
policy already masked rows. `.5` estimates only half of the historical live-row slope survives graph/dense padding;
B multiplication assumes marginal routed-expert work scales with live rows. Neither is measured here. At the nominal
cap, the central rate is exactly the measured-time/censored-numerator proxy; the remote cost is removed and added
ONCE. No raw graph cost is added to an end-to-end time that already contains it. W5-prod and K1 timings still belong
to ONE-CARD runs; their candidate columns are only a hypothetical TP2-cost sensitivity anchored to those times,
not predicted TP2 speedups over one card. Their values must not be pooled with TP2 suites.

The brackets enumerate all observed run survival vectors and step-p50 extrema, overlap fractions zero to one,
overhead zero to .6 ms, and row-slope multipliers zero to one. They are ESTIMATED assumption envelopes, not a
statistical confidence interval or a rigorous bound on unobserved acceptance/cost. Unknown counterfactual tail,
active-batch transitions and graph thresholds can fall outside them. At flat verify cost, maximum depth wins every
time; that sensitivity test prevents a false universal recommendation for shallower depth.

Five is the trained block and HELLO maximum in this stack. Columns six through eight are UNSUPPORTED; there is no
throughput estimate. No query-block truncation, target temperature change or unverified acceptance is proposed.

## Candidate ranking and expected benefit

All rates and gains in this paragraph are ESTIMATED, taken from screen.json. W5-E1p short favors depth two:
57.3 aggregate tok/s, envelope 46.9–70.1; nominal-cap proxy gain +7.6%, sensitivity −6.9% to +27.1%.
At 32K, B1 favors depth three: 65.2 [55.9–74.1], only +1.0% vs nominal cap (−7.3% to +11.2%).
B3 favors depth three: 111.1 [99.1–123.4], +3.0% (−5.1% to +12.7%); B4 also depth three:
125.8 [111.0–141.4], +3.0% (−5.9% to +14.2%). B1 at 150K: depth three 58.6 [49.4–68.9],
+5.0% (−4.1% to +16.3%). These are steady decode aggregate rates, not prefill-inclusive task rates.
B2 K1's hypothetical screen favors depth three: 90.3 [76.3–106.8], +3.0% (−7.0% to +16.1%).

The CURRENT policy is dynamic, not fixed at the nominal cap used for this comparison. Thus NONE of those gains is
an estimate validated against C's actual depths. Practical planning expectation: **ESTIMATED zero to three percent
incremental benefit over already-adaptive C**, possibly more on short prompts, with material regression risk. This
is an engineering budget, not a measurement or a promise. The broad envelopes justify screening, not promotion.
The raw late-position counters favor shallower candidates precisely where censoring is strongest; cap five remains
necessary as a falsification arm. Do not claim a five-percent win from this analysis.

Suggested policy candidates (configuration values, all unqualified): keep existing caps and replace inherited
4.4-ms remote scalar with 0 / .8 / 1.5 ms as overlap sensitivity arms; use .8 ms for the cap-three / cap-five
comparison. Central modeled exposed remote cost is about .8–1.2 ms across B1–B4. One scalar cannot fit every batch;
the zero arm is the full-overlap bound, never described as a measured zero RPC. Keep the verify table unchanged in
this first cost-only isolation. A defensible new verify table requires N3's paired fixed-depth traces; replace the
screening scalars/table with its export only AFTER the held-out error and throughput gates pass.
