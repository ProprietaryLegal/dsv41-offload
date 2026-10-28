# PC1 arm-specific KV identity

Round3 adds8192 and16384 to the same formula: derived1,934,490 and1,330,447
tokens respectively. See [HEADROOM.md](HEADROOM.md) for arithmetic, activation
admission/refusal and measured telemetry. The historical Round2 derivation below
continues to apply; the fixed pool and allocator are unchanged.

Round2 changes only PC1's expected capacity and per-start summary. The existing
shared log validator takes its existing `kv=` override; production snapshot and
restore still require exactly2,502,600. No allocator, KV reservation, launcher,
counter, deadline, greedy, TTFT decision or recovery policy changes.

The checked K1 allocator uses the widest group page sum as the shared pool
stride: `offload/kvpack/overlay/vllm/v1/core/kv_cache_utils.py:1624` (sum/alignment)
and `:1789` (integer pool division). Production's fp8_ds_mla ratio-2 group has
three37,440-byte MLA +8,640-byte indexer pages, hence138,240bytes/block. The
ratio-1 group uses three64-state kernel pages per192-token manager block,
also138,240bytes: `offload/kvpack/overlay/vllm/models/deepseek_v41/attention.py:138`
and `:145`. The existing K1 derivation `offload/kvpack/kv_arith.py:6` documents the
same page sizes/group layout. Forty SWA layers grouped three at a time give14
SWA groups; the remaining per-request pool consumers are one ratio-2 group,
one packed ratio-1 group and one compressor-ring group.

Source for the changed reserve is the checked
`offload/kvpack/overlay/vllm/v1/kv_cache_interface.py:899`:
SWA blocks/request/group = `ceil(min(127 + in_flight, max_model_len)/64) + 1`.
Its `:922` converts those blocks to bytes. Full attention uses `:594`, and the
one-block compressor ring uses `:958`. In the read-only image reference,
`/opt/dsv41/data/src-image/vllm/config/vllm.py:593` and `:605` set
async PP1 concurrent batches to2 and in-flight tokens to twice max batch tokens.

At327,680 max model length and6,442,450,944KV bytes/rank:

```text
pool_blocks = floor(6,442,450,944 / 138,240) = 46,603
ratio2_blocks = ceil(327,680 / 128) = 2,560
ratio1_blocks = ceil(327,680 / 192) = 1,707
ring_blocks = 1

control/threshold: in_flight=8,192; SWA=131
required = 2,560 + 1,707 + 1 + 14*131 = 6,102
reported = int((46,603 / 6,102) * 327,680) = 2,502,600

chunk2048: in_flight=4,096; SWA=67
required = 2,560 + 1,707 + 1 + 14*67 = 5,206
reported = int((46,603 / 5,206) * 327,680) = 2,933,321
```

The final float division/int truncation matches actual
`offload/kvpack/overlay/vllm/v1/core/kv_cache_utils.py:1106` and `:2486`.
The fixed pool is unchanged; the reported token equivalent grows because its
per-request admission denominator shrinks. This does not measure more allocated
KV bytes or a reduction in indexer scratch. Production's
`VLLM_INDEXER_PREFILL_BUFFER_FACTOR=4` remains unchanged:
`/opt/dsv41/data/src-image/vllm/v1/attention/backends/mla/indexer.py:893`
through`:910` returns `max_model_len * factor`, independent of batch chunk size.
All other production environment settings remain subject to existing equality
checks. Threshold changes do not alter this capacity formula.

`capacity.expectation()` calculates the expected value from the registered arm;
there is no chunk2048 capacity literal in runtime code. `tests/test_capacity.py`
verifies the K1 source hashes against `inputs.json`, extracts the actual pure
allocator/spec/config function bodies and executes them against production-shaped
spec adapters for all four registered settings. These tests do not import vLLM,
torch, CUDA, or create model objects; grouping/model construction is represented
by the documented fixed production layout, not claimed as a new hardware run.
The source functions independently reproduce both capacities and verify factor4.

Historical measured evidence, read only:
`/opt/dsv41/campaigns/DSV41-OFFLOAD/PC1-chunk2048-W-20261027T100908Z/`.
Its `cmd/0108-s2-chunk2048-logs-sparkhead.log` reports `[2933321]`.
The control's final checks report `[2502600]`; the saved summary stops on the
old capacity check and records6/6 exact restoration plus PASS post-close watch.
These are observations from the coordinator's earlier window. This fix has only
CPU evidence; it supplies no new TTFT or live restoration result.

Each summary start's `capacity` object contains the inputs, computed expectation
and `reported_tokens` by host. Empty rank1 lists are retained as observed.
A wrong, missing or mixed rank0 capacity stops before that start's timing and
retains its raw values in the summary. Capacity does not enter TTFT selection;
fresh per-request salts and zero-cache-hit gates remain intact.
