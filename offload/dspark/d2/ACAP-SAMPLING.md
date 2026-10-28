# Window-1 A-capture sampling evidence

D2-FIX1-R2, read-only verification on Controller, 2026-10-23. The files on disk support
**temperature 1, top_p 1, thinking enabled, reasoning_effort high** for the four
A-capture requests. They also record seed 42 and priority 10. These are the capture
settings, independent of D2 generation's historical priority 5 or reviewed priority 20.

Exact primary-source JSONL lines (the values are in `settings`):

| Source | temperature | top_p | chat_template_kwargs | seed | priority |
|---|---|---|---|---|---|
| `offload/results/capture/cap-short.jsonl:1` | 1.0 | 1.0 | `{"thinking": true, "reasoning_effort": "high"}` | 42 | 10 |
| `offload/results/capture/cap-8k.jsonl:1` | 1.0 | 1.0 | same | 42 | 10 |
| `offload/results/capture/cap-8k-c2.jsonl:1` | 1.0 | 1.0 | same | 42 | 10 |
| `offload/results/capture/cap-8k-c2.jsonl:2` | 1.0 | 1.0 | same | 42 | 10 |

`offload/launch/configs/A-capture.env:4-5` identifies the capture overlay and its
environment. `/opt/dsv41/campaigns/DSV41-OFFLOAD/start-A-capture.log:6`
records those mounts/environment for the window; line 7 records the image and
target root. This configuration does not itself specify per-request sampling.
`offload/dspark/capture/speculator.py:351` saves the request's actual temperature
in each round. No top_p or reasoning setting is stored in that round record.

The four exports in `/opt/dsv41/data/d2/acap` carry `capture_req`, but no
sampling receipt. Matching those IDs against the local CPU-loaded round files in
`/opt/dsv41/data/v1/captures/A-capture-20261022` gives:

| Export | Request/round appearances | Distinct recorded temperatures |
|---|---:|---|
| acap0 | 152 | 1.0 |
| acap1 | 111 | 1.0 |
| acap2 | 106 | 1.0 |
| acap3 | 182 | 1.0 |

All 551 appearances agree. Counts are appearances of these four requests in round
records, including batched requests; they are not 551 independent samples. JSONL
request rows do not retain the capture request IDs, so top_p/reasoning attribution
uses the uniform settings of all four window-1 benchmark requests. Temperature
additionally has direct per-request evidence from the captured rounds.

The evidence summary and benchmark-file SHA256s are in the campaign artifact
`D2-FIX1-R2.acap-settings.json`. Validation asserted all four request rows' settings
and matched every exported capture ID to captured temperature 1. No model requests,
capture writes, or live generation-state writes were involved.

The required T=1/top_p=1 check is satisfied by the recorded settings. This evidence
summary is **not** a complete `d2.sequence-context.v1` receipt. Before building real
data, the coordinator still supplies verified original capture context in
`per_sequence[seq_id]`: target/quantization/engine and replay identities, tokenizer,
encoder and template hashes, actual tools/schema/stop processors, effective
top_k/min_p/defaults (including generation-config absence/hash), and seed 42.
The builder must retain its existing refusal when that context is absent or
contradictory. No missing fields or receipts were fabricated here.
