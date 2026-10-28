# Item 2c: no pure host sync to remove from the production prefill indexer

Verdict: no runtime patch or fake no-op switch. The current CUDA indexer forward
already avoids the donor's host syncs. Removing required remote/stream ordering
would introduce races. The CPU rehearsal and explicit source inventory are reproducible:

```
source ~/.config/ndm/paths.sh
CUDA_VISIBLE_DEVICES= /home/operator/miniconda3/bin/python3 -m offload.tv1.prefill_sync_audit --output <new-TV1-scratch-directory>
```

The actual `build_prefill_chunk_metadata` Python function executes on CPU inputs at
32K and 150K with 4096 query rows. Only its GPU launch is stubbed. MEASURED:
8 `aten::item` / 8 `aten::_local_scalar_dense` calls, all on CPU tensors;
44 explicit scalar/list/stream/repeat sites inventoried across the selected engine
call paths. This is a CPU metadata rehearsal, not a GB10 full-forward trace and not
a measurement of CUDA synchronization latency. The profiler prints a no-visible-GPU
device-count diagnostic despite CPU-only activities; it does not execute CUDA work.

The source audit includes these branches (image-root-relative):

| Path/site | Production relevance | Disposition |
|---|---|---|
| `vllm/model_executor/layers/sparse_attn_indexer.py:399–912` | CUDA forward, prefill and decode | No host scalar/list/device reads. Q/RoPE/quant already fused; top-k writes the destination directly. |
| same file `:1276` | CPU fallback | CPU `.item`, no CUDA synchronization. |
| `vllm/v1/attention/backends/mla/indexer.py:1142,1174,1256,1344–1359,1568,1604–1605,1801,1844–1845,1889–1895` | Host metadata from named CPU mirrors | Keep scalar reads: no GPU->CPU operation is removed by replacing `.item` with `int`. |
| same file `:1837–1838` | DCP>1 ONLY | Real device scalar reads in that branch. Production DCP=1. Reworking this is not a production prefill win. |
| same file `:226–243,1290,1505` | CPU NumPy/sharding helper | CPU work, not a device sync. TP sharding threshold 65536 keeps it off at 4096 rows. |
| `vllm/models/deepseek_v41/common/engram.py`, remote overlay | Slot/position hashes + required remote completion | No donor GPU repeat_interleave length sync. Transport completion and copy stream dependencies must stay. |
| `vllm_exl3_moe/forward.py:181,232` | Expert-count `.tolist` fallback | Real sync, bypassed by `TF_PROMPT` at >=64 rows (:116–121). For <64-row tails it drives host dispatch/reconstruction and is required by that fallback. Cannot delete without replacing the algorithm. |
| same file `:95–100` / stats branches | Diagnostic checks OFF | Scalar logging, not production overhead. |
| same file `:119–120,174–175` | Shared/routed streams and shared lock buffer | Required ordering, `wait_stream` is a device dependency rather than a host scalar synchronization. Keep. |
| `vllm_exl3_moe/kernels.py:402` | Initialization | Startup JIT warmup synchronization, outside steady prefill. |
| `vllm_exl3_moe/hcap.py:51`, `moe_method.py` loader sites | Capture OFF / weight loading | Diagnostic save or initialization, outside production prefill. |
| `vllm_exl3_moe/sparse_attn.py:201–203` | Reference/debug implementation | CPU list construction in reference code, not served FlashInfer path. |

This covers explicit Python synchronization opportunities in the engine paths read.
It cannot certify absence of hidden waits inside closed/imported libraries. Existing
SP1 GPU traces are the end-to-end bound: <1.04% prefill idle, including other causes.
No bitwise comparison is needed for a runtime change that was not made; current engine
bytes and control manifests remain the exact baseline.
