# PF1-CAUSE2 item 1: what the cause overlay changes in the MoE half besides timing

Lane PF1-CAUSE2, 2026-10-26, CPU only on Controller. Inputs: the cause overlay as built by
`offload/prefill_cause/build_overlay.py` from the pinned K1 bases (= the bytes window CW mounted: its `cause-src`
hashes equal `OVERLAY.sha256`), the PF1-R2 runtime it records into (`offload/prefill_trace/runtime.py`, mounted as
`dsv41_prefill_trace.py`), and the saved window files of W1 (`PF1-W1-20261025T234254Z`), W2 (`PF1-W2-20261026T010854Z`),
CW (`PF1-CW-20261026T024340Z`) and W3 (`PF1-W3-20261026T032022Z`). MEASURED = read from a saved file; INFERRED =
argued from MEASURED values; ESTIMATED = arithmetic.

## Plain language

- The cause overlay changes **no computation, allocation, stream, synchronization or copy** in the MoE half. Its only
  edits are three imports and 25 call wrappers (4 in the decoder layer, 2 in the TensorFold prompt call, 19 TensorFold
  kernel launches). Removing the wrappers gives back the production source exactly (AST equality), and a census of
  every stream/event/sync/host-read/allocation/copy call outside the wrappers is identical to production.
- What the wrappers do while a prefill chunk runs is the whole difference. Per full 4,092-row chunk the GPU command
  stream gets **+960 CUDA timing-event records** (12 spans × 40 layers × 2; 24 per layer, 20 of them between MoE-half
  kernels), and the host spends longer issuing each layer (**10.2 → 16.4 ms per layer**, MEASURED) and runs less far
  ahead of the GPU. The host also reads the caching allocator's counters once per MoE call and writes ~3× more trace
  rows between steps. None of this touches a tensor.
- CW's arms ran exactly production's engine, weights, argv (port/cap only) and env plus the declared trace switches
  (and, for T0, exactly the two TRANSIENT/ALIAS values). Production itself was identical in W1, W2, CW and W3.
- Ranked: the **event records in the MoE half's command stream** are the only GPU-visible difference and fit every
  observation; host pacing is second; the shallower host lead is measured but does not fit (the slowdown is flat from
  layer 0, where the queue is short in both). The physical mechanism on GB10 is **not identified** by any CPU
  evidence. Item 2 turns the top-ranked difference into a default-OFF switch with no instrumentation, so one matched
  window separates "event records" from "host-side trace work".

Reproduce (CPU; numbers and hashes only):

```bash
python3 -m offload.prefill_cause2.inventory diff --output "$S/diff.json"        # sha256 8c125246... (this lane)
python3 -m offload.prefill_cause2.inventory lead --output "$S/lead.json" --traces \
  /opt/dsv41/campaigns/DSV41-OFFLOAD/PF1-{W2,CW}-*/traces/*/*/*.jsonl  # sha256 a6261611... (this lane)
```

## 1. Every edit (MEASURED from the rebuilt bytes; `inventory.py diff`)

| File (base K1 sha → overlay sha) | Added imports | Spans `_pf.call(label, metadata, fn, *args)` | Metadata calls |
|---|---|---|---|
| `nvidia/model.py` (bc3aa8b0 → b5595484) | `dsv41_prefill_trace as _pf` | `DeepseekV4DecoderLayer.forward`: `attn.total` (self.attn), `mhc.ffn_pre` (FFN-side `mhc_shifted_post_pre`), `moe.ffn` (self.ffn), `moe.ffn_unfinalized` (self.ffn.forward_unfinalized) | none (layer index, `positions.shape[0]`) |
| `vllm_exl3_moe/tf_prompt.py` (8d66ae29 → 8c1628b8) | `_pf`, `dsv41_prefill_cause as _pc` | `routed`: `moe.routed.tf` around both `T.routed(...)` returns (TRANSIENT on / off) | `_pc.alloc(...)`, `s.xd.data_ptr()` |
| `vllm_exl3_moe/tf_experts.py` (bde1d0b4 → 622392b5) | `_pf` | `routed`: every launch: group_count, group_place, group, rot_in, work_list, grouped_mma3/mma2/mma/rows/grouped (gate/up and down), gateup_epilogue, grouped_down3, combine_y, down_epilogue, down_combine | `s.xg.data_ptr()` (rot_in), `out.data_ptr()` (down_combine) |
| helper `dsv41_prefill_cause.py` (new, edb72e0c) | — | — | `torch.cuda.memory_stats()` only when a traced chunk is open |

Census of behaviour-relevant calls **outside** the wrappers (Stream/stream/current_stream, wait_stream, wait_event,
record_stream, Event/record, synchronize, item/tolist/cpu, empty/zeros(_like), contiguous, clone, copy_, to,
memory_stats, empty_cache, data_ptr, is_current_stream_capturing): identical to the base in all three files
(`inventory.py` refuses otherwise; CPU-tested with an injected `torch.cuda.synchronize()`).

In a full chunk of the 150K request the TensorFold path runs 8 launches per MoE call (group_count, group_place,
rot_in, work_list, grouped_mma.gateup, gateup_epilogue, grouped_rows.down, down_combine; MEASURED labels in CW at
4,092 and 128 rows), so each of the 40 layer passes adds 12 spans: CW full chunks carry 700 (rank 0) / 705 (rank 1)
timed spans vs W2's 220 / 225 (MEASURED, `lead.json`).

## 2. What a wrapper does at run time (source: `prefill_trace/runtime.py` `call` / `Recorder.call`)

Outside an open traced prefill chunk (decode, CUDA-graph capture, dummy/profile runs, ENABLED off) `fn(*args)` is called
directly. Inside one, per span: `torch.cuda.is_current_stream_capturing()` (host), two `torch.cuda.Event(enable_timing=True)`
objects, a host dict of scalars, `start.record()` on the **current** stream, `fn(*args, **kwargs)`, `end.record()` on
the current stream (also on exception), and the pair is appended to a pending deque (bounded 8,192; drained with
non-blocking `query()` + `elapsed_time()` and written to JSONL only at the step boundary, `begin`/`execution`).
Arguments are evaluated before the call exactly as in production (the metadata dict first: side-effect-free scalar
and `data_ptr()` reads), are not retained after the call, and the return value is passed through.

## 3. Behavioural differences, each checked

| # | Difference in the MoE half (CW vs W2/production) | Evidence | Verdict |
|---|---|---|---|
| D1a | **+24 timing-event records per layer pass on the current stream** (+960 per full chunk), 20 of them between MoE-half kernels (mHC pre-mix, router/shared expert/routed call, each TensorFold launch), the others at the attention/FFN boundary | §1, §2; span counts MEASURED | **Kept, rank 1** |
| D1b | Shallower host lead (a consequence of D1a's extra commands and of D2): GPU work still queued when the host issues layer 19's FFN all-reduce: W2 1,037-1,045 ms (327,680) / 1,411-1,416 ms (524,288) vs CW 888-894 ms (all three arms) | `lead.json` MEASURED | Kept, rank 3 (see §4) |
| D2 | Slower host issue in the layer stack: 10.1-10.3 ms per layer (W2) vs 16.1-16.8 ms (CW) before launch back-pressure; host still ahead of the GPU in both (layer 9's FFN all-reduce issued at 127-129 ms vs 168-174 ms, while the GPU reaches it at ~0.8-1 s, ESTIMATED in CAUSE.md §1) | `lead.json` MEASURED | Kept, rank 2 |
| D3 | `torch.cuda.memory_stats()` once per traced MoE call (host copy of the allocator's counters under its mutex) | helper source | Kept, rank 5: no device work, no allocation |
| D4 | ~1,400 more CUDA event objects alive per chunk until the step's drain (driver event pool) | §2 | Kept, rank 4: host/driver only |
| D5 | ~3× JSONL bytes per step (written at the step boundary, after the forward's host sync) | trace sizes 18.4 vs 8.4 MB per arm | Kept, rank 6: between steps |
| — | Tensor allocations, retained references, caching-allocator reuse | wrapper keeps no argument; census equal; CW allocator counters MEASURED flat in every arm (`device_allocs` +16 over 860 full-chunk calls, `alloc_retries` 0) | **Excluded** |
| — | Streams, stream joins, `wait_event` | census equal; events recorded on the current stream, never waited on | **Excluded** |
| — | Host syncs / device reads (`.item()`, `synchronize`) | census equal; `query()` is non-blocking and runs only at the step boundary | **Excluded** |
| — | Contiguity, copies, argument order, kernel choice | AST strip identity (`prefill_cause/test_overlay.py`) + recording-stub launch identity (same launches, args, strides, scalars, output) | **Excluded** |
| — | Env-gated paths (TRANSIENT/ALIAS, others) | §5: env identical to production except the declared switches | **Excluded** |
| — | Import-time effects | `dsv41_prefill_trace` is already imported by W2's model_runner; the helper imports it only; no CUDA call at import | **Excluded** |

## 4. Ranking against the four observations

| Observation (MEASURED) | D1a event records | D2 host pacing | D1b host lead | D4 / D3 / D5 |
|---|---|---|---|---|
| Gap present with production (W1, W3) and PF1-R2 attention-side spans (W2), absent with CW at 524,288 | fits: only CW has records in the MoE half | fits | fits | fit |
| Cost is in rank 1's MoE half (GPU kernels slower at equal clocks, lower power, CAUSE.md §1) | fits: the only difference that is GPU-visible and located there | weak: host work cannot slow GPU kernels except through the shared memory system | weak (same) | weak: host-only |
| Flat per chunk, layer-specific 27-35 ms from layer 0 on | fits: fixed records per layer | fits | **misfit**: at layer 0 the queue is a few layers deep in both W2 and CW, at layer 19 ~10 vs ~8; the cost does not grow with depth | n/a |
| Independent of T0 within CW (69.20 vs 69.77 s; per-chunk 1,846 vs 1,855 ms) | fits: same records with persistent scratch | fits | fits (888-894 ms in all CW arms) | fit |

**Rank 1: D1a.** It is the only change the GPU sees and it sits where the cost is. How a timing-event record could
remove a rank-1, cap-dependent slowdown of the same kernels is **not known**: a record is a semaphore release with a
GPU timestamp, executed in stream order after all preceding work; it changes the command stream between the MoE
kernels but not their arguments. No CPU test can reproduce it; only a window can.
**Rank 2: D2** (host pacing), **3: D1b** (lead, disfavoured by the flat-from-layer-0 shape), **4-6: D4, D3, D5**.

The item-2 switch reproduces D1a exactly (same event kind, same positions, same nesting, current stream) and with it
D1b's command-count part, without D2's host work, D3 or D5. In the matched window (item 3): fix ≈ cause ⇒ the effect
is in the event records (D1); fix ≈ off with cause fast ⇒ host-side trace work (D2-D5); cause slow ⇒ the CW result
was state, not code (next: the alternating off/cause series).

## 5. CW identities (MEASURED; `cause-*-identity.json` vs the same window's `rollback.json`)

- Both ranks of every CW arm: image `sha256:435a5236…`, engine `0.31.1.dev0+g6e517b15.dsv41gb10`, weights/target files,
  Engram identity all equal production. Mounted code = production's mounts with **10** paths replaced or added, each equal
  to `prefill_trace/OVERLAY.sha256` ∪ `prefill_cause/OVERLAY.sha256` (5 replaced: model_runner, flashinfer_sparse,
  model.py, tf_prompt, tf_experts; 5 added: parallel_state, sparse_attn_indexer, mla/indexer, the two helpers).
- Env vs production: `VLLM_ENGRAM_REMOTE_REQUIRED` 0 → 1 (every PF1 arm), `VLLM_DSV41_PREFILL_TRACE{,_DIR,_RUN}` added;
  T0 additionally `VLLM_EXL3_TF_PROMPT_TRANSIENT` and `_ALIAS` 1 → 0. Nothing else. Argv: `--port 8001` and, at
  524,288, `--max-model-len`; same token count.
- CW cause-524 vs W2 on-524: the 3 MoE-half mounts differ and the helper is added; env identical except the run label.
  W1 off-524 vs W3 off-524: identical except container id. Production snapshots (rollback.json) of W1, W2, CW, W3: equal.
- Limit: the identity capture records env keys with prefixes VLLM_, EXL3_, NCCL_, PYTORCH_, CUDA_ plus PYTHONPATH and
  DSV41_ENGRAM_OVERLAY only (`pf1_window.py` `capture`). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is among them.
- KV capacity per cap equal across W1/W2/CW/W3 (2,502,600 / 2,820,756 tokens); host parity (kernel 6.17.0-1026-nvidia,
  driver 580.159.03, THP madvise) equal on both Sparks before every CW arm.

## 6. Numbers used (MEASURED unless marked)

| Arm | 150K cold TTFT (s) | full-chunk GPU ms (r0/r1) | timed spans per full chunk (r0/r1) | host ms/layer before back-pressure | GPU ms queued at layer 19 FFN AR (r0/r1) |
|---|---:|---:|---:|---:|---:|
| W1 off-327 / off-524 | 71.16, 69.60 / 95.78, 94.88 | untraced | — | — | — |
| W2 on-327 | 69.81 | 1,844 / 1,844 | 220 / 225 | 10.2 / 10.1 | 1,045 / 1,037 |
| W2 on-524 | 91.48 | 2,475 / 2,475 | 220 / 225 | 10.1 / 10.3 | 1,416 / 1,411 |
| CW cause-327 | 70.37 | 1,844 / 1,854 | 700 / 705 | 16.4 / 16.3 | 888 / 890 |
| CW cause-524 | 69.20 | 1,846 / 1,846 | 700 / 705 | 16.4 / 16.8 | 888 / 894 |
| CW cause-524-t0 | 69.77 | 1,855 / 1,855 | 700 / 705 | 16.1 / 16.7 | 889 / 893 |
| W3 off-327 / off-524 | 68.94, 68.98 / 91.85, 90.84 | untraced | — | — | — |

CW rank-1 per-chunk MoE-half components at 4,092 rows (20 layers): `moe.ffn` 876.0 / 875.8 / 879.1 ms, of which the
TensorFold routed call 718.0 / 719.0 / 720.0 ms (cause-327 / cause-524 / T0); rank 0 equal within 1 ms.
