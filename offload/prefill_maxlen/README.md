# PF1 CPU candidate: request-sized indexer workspace views

This patch removes max-model-len from the **runtime prefill workspace view extent
and scale-buffer offset** on the existing TP-only path. Its default is off. It
does **not** establish the cause of the 35% regression or an achieved speedup.
Read [DIAGNOSIS.md](DIAGNOSIS.md) before evaluating it.

Build, verify and CPU-test locally:

```bash
source ~/.config/ndm/paths.sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 "$HARNESS_PYTHON" -m pytest \
  -q -p no:cacheprovider offload/prefill_maxlen
"$HARNESS_PYTHON" -m mypy --strict --explicit-package-bases \
  --cache-dir /opt/dsv41/campaigns/DSV41-OFFLOAD/PF1-mypy-cache \
  offload/prefill_maxlen/*.py
"$HARNESS_PYTHON" -m offload.prefill_maxlen.build_overlay \
  --output-root offload/prefill_maxlen/generated
```

No vLLM module, CUDA extension, model, HTTP client or launcher is imported by the
builder. It accepts only the exact image-source SHA in `MANIFEST.json`, applies
the patch with zero fuzz in temporary lane-tree scratch, checks the patched SHA
and Python syntax, and emits one full file ready for a coordinator bind mount.
It refuses a foreign base or a different pre-existing output. It writes no
reference source. Generated artifacts are ignored by git; the patch and manifest
are the reviewable source of truth. No dependency installation is needed.

The additional target is
`/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py`.
It overlaps **none** of K1's 13 mounted paths. Keep all K1, Engram and remote
DSpark mounts and switches from the reviewed base. Enable the candidate only by
adding `VLLM_DSV41_INDEXER_REQUEST_WORKSPACE=1` to the experimental container's
environment, with the overlay mounted on both ranks. Restore `=0` to run the
identical mounted file with legacy behavior.

## Mechanism and limits

Before: `_gather_workspace_shapes(total_seq_lens,...)` uses factor×cap/ratio,
then each loop slices values/scales to the current chunk. The FP8 scale region
starts after factor×cap/ratio×128 bytes, even for a short request.

After: find the maximum `chunk.max_local_total_seq_lens` on the host and create
both views once using that maximum. Every existing per-chunk slice, gather,
query sub-chunk, key order, logits extent, score mask and top-k call stays the
same. Reused gathers retain the same values **and** scales throughout the loop.
The helper fails if required rows exceed the profiled reservation; it never
truncates. Empty chunk lists request zero rows. There is no new `.item()`,
synchronization, quantization, arithmetic order, or token scheduling change.

For one 32K ratio-1 request, ESTIMATED FP8 scale offset becomes 4096000 bytes
at **both** caps, vs 167772160 / 268435456 bytes. Total required view bytes
become approximately 4.03 MiB at both caps, vs 165 / 264 MiB. At 150K the
required views are approximately 18.88 MiB, again equal at both caps. This
changes **view layout**, not necessarily traffic: existing kernels already
receive slices with these row counts. Native kernels may be sensitive to
addresses/placement; the benchmark must determine that.

The dummy/profile path still reserves the original worst case. WorkspaceManager
locks its allocation before serving and retains that larger byte buffer. Thus
this patch **does not free the 99-MiB cap-dependent reservation**, remove RoPE
growth, or eliminate the rest of cap-dependent memory layout. Dynamic allocation
after lock, profile-underbudgeting and freeing graph-owned addresses would be
unsafe shortcuts. PCP/DCP deliberately retain their legacy layout; decode code
is unchanged. No numerical claim is made about unexecuted native kernels.

CPU validation executes the actual patched prefill function, actual shape helper
and actual aligned WorkspaceManager view constructor extracted through AST.
Reference gather/scoring/top-k kernels replace CUDA boundaries. Poisoned scratch,
duplicate page order, shorter later gathers, continuation reuse, FP8/MXFP4,
reserved output rows and two different caps produce identical bytes/logits/indices.
PCP/DCP fallback tests check legacy layout, and an AST comparison proves the
dummy/profile and decode branches are unchanged. The real upstream chunker also
produces identical plans at the two caps on matched supported request lengths.
These tests establish buffer/control-flow equivalence, **not native GPU parity**.

## Buffer-factor alternative

`K1_INDEXER_FACTOR=1` is a useful **separate coordinator experiment**, not a proven
speed fix. At 524288 it reserves 66 MiB instead of 264 MiB (ESTIMATED 198-MiB
reduction in the shared ratio-1 requirement). It still holds an individual
maximum-length request and the chunker can split multi-request groups. Query
sub-chunks continue to score all keys; there is no approximate top-k. Grouping
may change at high concurrency, so require the same correctness/concurrency
gates. Never use zero, negative or fractional factors: the unmodified function
does not validate them and a single request can exceed its nominal workspace.

Neither factor 1 nor request-sized views explain the +0.667-s delta in advance.
They isolate different effects: factor changes the persistent reserve, views
change the relative addresses while preserving reserve. Test them separately.
