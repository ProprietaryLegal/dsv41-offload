# offload/upfix — upstream robustness and top-k fixes (R-UPF; UPS1 #4 and #5)

Both items are **off unless an arm turns them on**. Production and every existing config are untouched: nothing here is
mounted by `dsv41.env`, by K1's promotion package, or by any other config. CPU-built and CPU-tested on Controller; no
Spark, x86host or production contact.

| Item | What | Where | Switch |
|---|---|---|---|
| UPS1 #4 | henryous patch 0091: six vLLM fixes (#59096 file-tier `mark_miss`, #59880 spec-decode blocks per KV group, #61174 stale block-table rows, #63165 busy-reader notify growth, #61000 SHM buffer ownership, #59504 KV zeroing next to async loads) | `build_0091.py` → `overlay/` (4 files) and `overlay-0075/` (scheduler composed onto the 0075 replay-skip overlay); `MANIFEST-0091.json` | arm `UPF_0091=1`; inside, `VLLM_DSV41_SPEC_RESERVE_PER_GROUP=0` turns #59880 back off |
| UPS1 #5 | vLLM persistent top-k with #62467 (tail lanes), #62936 (candidate overflow), #63475 (CTA-sync race), #63118 (NaN-safe) as a separate JIT op `torch.ops.dsv41_topk.persistent_topk`; the vendored kernels also carry #59645 (cooperative candidate overflow, #62936's prerequisite) and #60721 (header consolidation) | `topk/csrc` (kernels vendored unchanged from vLLM 3090266d8; launcher ported to ATen), `topk/build_overlay.py` → `topk/overlay/.../indexer_topk.py`, `topk/jit_build.py`, `topk/bitcompare.py` | arm `UPF_TOPK=1` → `VLLM_DSV41_TOPK_FIX=1` + `VLLM_DSV41_TOPK_FIX_SO` (default `0`: the image's `_C` kernel) |

## Composition with K1 and UP2

- None of the four 0091 files, and not `indexer_topk.py`, is among K1's 15 production mounts
  (`offload/promote/k1/dsv41.env.327680.patch`) or under one of them (tested).
- `scheduler.py` is also replaced by the UP2 0075 overlay (`offload/upstream_adopt/overlay`). `overlay-0075/` is that
  file plus exactly the same 0091 lines as the image variant (tested); `UPF.env` with `UPF_BASE=0075` **replaces** the
  UP2 scheduler mount in place instead of mounting a second scheduler.
- 0091's scheduler hunk 1 (the #59880 switch) anchors on donor header lines our v5 image does not have; it is placed
  right after the existing `import os as _dsv41_os` line. Every other hunk applies with `patch --fuzz=0` unchanged.

## Commands (repository root; Controller, CPU only)

```
python3 -m offload.upfix.build_0091 check            # overlays + MANIFEST-0091.json equal a fresh build
python3 -m offload.upfix.topk.build_overlay check    # vendored blobs, indexer_topk overlay, MANIFEST-topk.json
python3 -m offload.upfix.configs arms                # arm -> UPF.env environment
python3 -m offload.upfix.topk.bitcompare             # the SM121 bit-compare plan (no --execute)
CUDA_VISIBLE_DEVICES= python3 -m pytest offload/upfix
```

Windows, gates, HOLD estimates and the in-window build/bit-compare commands: `WINDOW.md` (designs as data in
`windows.py`). Licences: `NOTICE`.
