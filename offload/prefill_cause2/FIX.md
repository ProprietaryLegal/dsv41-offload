# PF1-CAUSE2 item 2: the fix candidate (default OFF, no instrumentation)

Status: CPU-built and CPU-tested on Controller; **not run, not GB10-qualified, not a promotion**.

## Plain language

- Item 1 ranked the CUDA event records that the PF1-CAUSE spans put into the MoE half's command stream as the one
  GPU-visible difference between the slow (production, W2) and fast (CW) 524,288 runs. The fix candidate keeps
  exactly those records and nothing else: the same 25 call sites, the same event kind (timing-enabled), the same
  order and nesting, on the current stream, for every eager pass of at least 64 rows (prompt chunks and the 128-row
  replay tail). No span rows, no file, no clock reads, no allocator reads, no trace runtime.
- It is off unless `VLLM_DSV41_MOE_FENCE=1`. Without the switch the mounted sources call straight through (one Python
  call per site); without the overlay production's bytes run. Nothing reads the events, so the model's arithmetic,
  kernels, streams and memory are those of production.
- If the matched window (item 3) shows fix-524 as fast as cause-524, the gap fix is this switch (production-ready
  after the usual memory/promotion rows, which it barely touches: CUDA events use no caching-allocator memory). If
  fix-524 stays as slow as off-524 while cause-524 is fast, the effect is in the trace runtime's host work instead
  (item 1 D2-D5).

## Files (offload/prefill_cause2)

| File | Role |
|---|---|
| `build_fix.py` | instruments the 3 K1 files at `prefill_cause.build_overlay.ANCHORS` (same matching, same counts: a changed base cannot lose a site); `--enabled` off ⇒ exact K1 bytes; refuses base drift and outputs inside the K1 / PF1-R2 / PF1-CAUSE / image trees |
| `fence.py` → `vllm/v1/worker/gpu/dsv41_moe_fence.py` | `call(rows, fn, /, *args, **kwargs)`: if `VLLM_DSV41_MOE_FENCE=1`, `rows >= 64` and not capturing: `Event(enable_timing=True).record()` before and after `fn` (also on exception); pairs kept in a 4,096-pair ring (released long after completion) |
| `BASES.json` | = `prefill_cause/BASES.json` = K1 MANIFEST overlay hashes (bc3aa8b0 / 8d66ae29 / bde1d0b4) |
| `OVERLAY.sha256` | the 4 reviewed ON outputs: model.py 74fbc525, tf_prompt 553d90a4, tf_experts b7ef5b7b, helper 8fb264d7 |
| `../launch/configs/PF1-fix.env` | PF1-window.env OFF sourced unchanged (PF1_TRACE must be 0), then exactly 3 read-only production MoE mounts swapped for the fence sources + the helper (4 overlays), `VLLM_DSV41_MOE_FENCE=1` appended; refuses tampered sources, a traced arm, a production env that already sets the switch, inherited overlays |

Build (CPU): `CUDA_VISIBLE_DEVICES= python3 -m offload.prefill_cause2.build_fix --enabled --output-root offload/prefill_cause2/generated`

## Invariants (CPU-tested: `test_fix.py`, `test_pf1_fix_config.py`)

- OFF build = K1 bytes; ON build = `OVERLAY.sha256`; the helper = `fence.py` bytes.
- Removing `_mf.call(rows, fn, ...)` → `fn(...)` and the import gives the **whole original AST** of each file. The row
  argument is a name or shape read already in scope (`positions.shape[0]`, `m`, `R`): no call, no side effect. The
  wrapper's own parameters are positional-only, so a callee keyword can never be captured.
- **Same sites as the cause spans**: in every file the wrapped callees, in source order, equal the cause overlay's.
- **Same records as the cause spans, same outputs as production** (the real K1 `tf_prompt.routed` → `tf_experts.routed`,
  TRANSIENT=ALIAS=1 as in production and T0, against a recording kernel stub whose launches and the event records of
  both runtimes go into one log): launches, argument shapes/dtypes/strides/scalars and the output tensor equal
  production's; the interleaving of event records and launches with the fence equals the one inside an open PF1
  trace chunk with the cause overlay (18 records for a 96-row call: 8 launches + the routed call, ×2); production
  records none.
- Gates: switch off / rows < 64 / stream capturing ⇒ no record, plain call; exception ⇒ end record then re-raise;
  ring bounded; switch defaults off.
- Not covered on CPU: native CUDA behaviour of the records (the point of the window), the model.py sites at run time
  (AST-checked only), in-container import (`vllm.v1.worker.gpu.dsv41_moe_fence` must be mounted: the 4-overlay set).

## What the candidate reproduces of the CW difference (item 1 numbering)

| Item-1 difference | Fix candidate |
|---|---|
| D1a: 24 timing-event records per layer pass on the current stream, MoE-half positions | **reproduced exactly** (positions, kind, order, nesting) |
| D1b: shorter host lead from the extra commands | reproduced in its command-count part (same records); host-time part not |
| D2: slower host issue (~6 ms per layer of Python span bookkeeping) | **not** reproduced (fence: two event creations + records per site, no dict/clock/JSON) |
| D3: `torch.cuda.memory_stats()` per MoE call | not reproduced |
| D4: event objects alive until the step's drain | approximated (ring; released after completion) |
| D5: JSONL writes between steps | not reproduced |
| Startup dummy/profile and warm-up passes ≥ 64 rows (eager) | fenced (the cause spans record only inside an open traced chunk); events only, no memory |
| Small prefill passes (< 64 rows) | not fenced (cause spans would record there); no 150K chunk or tail pass is below 64 |
| PF1-R2 attention-side spans | not present (W2 shows they do not remove the gap) |

## Greedy identity and logit floor

The records cannot change arithmetic by construction (no tensor is read or written by them). The window verifies it
anyway on the fix arm against the off arm of the same HOLD (item 3): the six W5 greedy prompts (temperature 0, thinking
off, 300 tokens), text **hashes** 6/6 equal; and an 8,192-token cold prompt's first-token top-20 distribution compared
with the up2 numerical rule (`up2_checks.numerical`: KL(off ‖ fix) ≤ max(0.005, 2 × the off A-A KL) and common mass ≥
0.99), the A-A pair being two cold off-arm requests with different cache salts.

## Memory and decode

No caching-allocator memory (CUDA events), no new tensor; host memory ~8K event objects. Decode windows are below
64 rows and graph-replayed: never fenced. ESTIMATED GPU cost: ~960 records per full chunk at a few µs each ≲ 5 ms per
1.85 s chunk (CW: cause-327 70.37 s vs W3 off-327 68.94-68.98 s, with the far heavier spans).
