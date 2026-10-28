# Round 8: candidate owning-layer binding

SOURCE/CPU fix; GB10 startup and numerics remain unqualified. The read-only
`TV1-candidate-W-20261027T124301Z/engine-logs/0002-start-failed/` evidence
captures both ranks failing in `consumer_layer()` during memory profiling.
The current-config context is absent at forward time. The complete log hashes
are sparkhead `20dbd0e615bdf119beeafdae644775b86ece4cdd4bc333faa58a2a0f2a2cec1b`
and sparkworker-wired `57e11cc1e03dbee96a545d1365b83b55d5c977e889a5dcbf25faae2075f83aae`.
The saved failed run stays incomplete, with no candidate performance result.

## State available on every reaching path

All citations below are read-only image sources rooted at
`/opt/dsv41/data/src-image/vllm/`; the patched runner uses the same
unchanged forward-context scope in the K1 base.

| Path | Evidence |
|---|---|
| Model construction | `models/deepseek_v41/attention.py:419-429` registers each owning attention module under its own prefix in `compilation_config.static_forward_context`; its `indexer.indexer_op` is the consumer instance. Shared K cache prefixes refer to layer20 and cannot identify a consumer. |
| Memory profile | `v1/worker/gpu_worker.py:566-575` sets current config only around model loading. `determine_available_memory:618,655` later calls `profile_run` without that scope. `v1/worker/gpu/model_runner.py:1032-1051` calls `_dummy_run`; `_dummy_run:925-942` calls `execute_model(dummy_run=True)`. |
| Profile/dummy forward | `v1/worker/gpu/model_runner.py:2101-2133` wraps eager/piecewise model execution in `set_forward_context(attn_metadata, self.vllm_config, ...)`, including dummy execution. Metadata may be None; the layer map is still present. |
| Graph capture | `v1/worker/gpu/cudagraph_utils.py:790-817` wraps FULL and PIECEWISE capture model calls in `set_forward_context` with its explicit config. FULL replay (`:827-840`) replays captured GPU work and does not call the Python resolver again. Piecewise/eager execution retains the runner scope above. |
| Serving prefill and decode | Both use the same `execute_model` model-call scope above; prefill and decode metadata differ, but the owning map does not. Short all-selected consumers can bypass the op; those that reach it already require forward context at `models/deepseek_v41/attention.py:1527` and `model_executor/layers/sparse_attn_indexer.py:430`. |
| Context creation | `forward_context.py:245-260` always passes the explicit config's `static_forward_context` as `ForwardContext.no_compile_layers`, independently of attention metadata. `set_forward_context:347-365` installs it with `override_forward_context`; `get_forward_context:208-214` refuses missing context. No current-config accessor is needed. |

The resolver now reads `get_forward_context().no_compile_layers`. It retains
identity-based ownership matching, exactly-one-match refusal, the name regex,
the qualified layers24/28/32/36 refusal and the existing per-op cache. No dummy
special case, guessed layer, shared-cache fallback, tensor read or changed
candidate/greedy/decision logic is introduced. OFF and compact install no
candidate helper; OFF remains byte-identical.

## Round 8 accessor audit and regression

The CPU regression imports the composed candidate helpers under their installed
`vllm_exl3_moe.tv1_*` names and executes the composed `SparseAttnIndexer.forward_cuda`
method. The custom-op tensor work is an inert CPU callback. The config accessor,
ForwardContext dataclasses, creation, set/override/get context managers are
extracted unchanged from the real image source; optional platform/DP/CUDA work
is replaced with explicit fixtures. `get_current_vllm_config()` asserts before,
inside and after every forward scope. Cases cover all four consumers in profile
(metadata None), FULL/PIECEWISE capture, prefill and decode, plus uncached-owner
refusals and repeat cache hits. These are lifetime/binding checks, not actual
model or CUDA execution. Replacing only the metadata helper with5b2de7c bytes
makes all20 reaching-path cases fail at the original current-config assertion.

Every context/group accessor in the six-file candidate overlay was audited.
The following locators name the composed overlay (image locators for inherited
code differ only by patch insertions). The regression compares its complete
inventory to the untouched pinned bases: the sole addition must be
`candidate_metadata.consumer_layer -> get_forward_context`.

| File / call sites | Lifetime / disposition |
|---|---|
| `sparse_attn_indexer.py:140` `_merge_dcp_topk_global`; `:153,155` `dcp_gather_kv_rows` → `get_dcp_group` | Inherited process-lifetime parallel groups; candidate refuses DCP. No current-config lookup. |
| `sparse_attn_indexer.py:365` `PrefillRowShard.exchange_topk`; `:388` `_reserve_row_shard_exchange_transient` → `get_tp_group` | Inherited process-lifetime TP group, not a lexical config context (`distributed/parallel_state.py:1547-1549`). |
| `sparse_attn_indexer.py:380` `_reserve_row_shard_exchange_transient` → `get_current_vllm_config_or_none` | Inherited optional probe explicitly returns immediately for None (`image :380-382`). The accessor itself is a nonasserting read (`config/vllm.py:3634-3635`). No change to profiling reservation. |
| `sparse_attn_indexer.py:431` `sparse_attn_indexer`; `:1063` `cp_kv_cache_interleave_size`; `:1227` `forward_cpu` → `get_forward_context` | Inherited forward-only state. Uses the same model-call scope proved above. CPU path is outside GB10 execution. |
| `sparse_attn_indexer.py:521` `sparse_attn_indexer` → `get_pcp_group` | Inherited process-lifetime PCP group; candidate refuses PCP. |
| `sparse_attn_indexer.py:1017` `SparseAttnIndexer.__init__` → `get_current_vllm_config`; `:1022` → `get_dcp_group` | Constructor inside model-load config scope, not a forward call. Already binds topk backend and parallel scalars on the op. |
| `model_runner.py:236,237` `GPUModelRunner.__init__` → `get_pp_group`; `:262` → `get_dcp_group` | Inherited process-lifetime groups during runner construction. Candidate runner addition only reads owned CPU request metadata; no context accessor added. |
| `tv1_candidate_metadata.py:33` `consumer_layer` → `get_forward_context` | Sole new context lookup; owning map remains available with metadata None. Cached repeats avoid the lookup. |

`candidate.py`, `candidate_kernel.py` and `candidate_cases.py` have **zero**
vLLM context accessors. `check_actual` uses CUDA's capture query while executing
a real CUDA forward; the kernel launcher scopes its device to `q.device`.
Neither needs a vLLM config scope. Its imported `get_indexer_topk(topk_backend)`
factory always passes an explicit string. The factory (`image
model_executor/layers/indexer_topk.py:138-140`) forwards it to the constructor;
that constructor's config fallback (`:152-155`) runs only for None. Installed
oracle similarly passes explicit `'auto'`. The explicit backend and persistent512
refusals remain unchanged. Candidate helper imports and the kernel itself add
no other context-only accessor. Compact and combined's compact-only accessors
are unchanged by this round and are not candidate initialization fixes.

## Local pre-HOLD smoke and limits

`candidate_init_smoke.check(tree)` launches a bounded30s child with the current
required interpreter, hidden CUDA and disabled bytecode. It imports the real
composed metadata/candidate/cases helpers by installed package names and
requires their file identities. With a synthetic forward map, metadata None
and an accessor that refuses any current-config read, it binds all four
consumers and repeats through the per-op cache. It imports real CPU torch,
checks CUDA remains uninitialized and refuses an unexpected kernel import.
It deliberately does not import full vLLM or the Triton kernel. The child keeps
its synthetic vLLM modules outside the coordinator interpreter. Recorded helper
hashes and the verified build's input-lock hash identify the checked source.

The runner performs this check before `sup.begin()` and every transport/HOLD
operation. Missing imports, wrong binding, subprocess failure/timeout or receipt
IO error stop preflight. CPU tests restore the original lookup and verify its
failure; missing helper, import failure, unqualified layer and timeout also
refuse. A full-window fake asserts a failed smoke submits no remote operation,
opens no HOLD and starts no experiment or W8 restoration.

This catches Python import/binding regressions cheaply, using synthetic module
ownership. The separate source-execution regression above checks real image
forward-context semantics. Neither can prove CUDA kernels, memory headroom,
TP2 startup, graph capture or numerics. A full second model cannot safely run
beside production under this lane's access/headroom limits. The next coordinator
window's candidate start must complete profile/dummy, warmup, KV allocation and
graph capture on both ranks, then pass unchanged exact installed/actual-forward
oracles and the registered experimental greedy/performance gates. Complete
engine logs must survive before removal if any phase fails. Old failed evidence
and S77's source/CPU-only clearance remain historical; no speed result is claimed.
