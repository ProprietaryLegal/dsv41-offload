# Round 5 compact startup diagnosis

**Two source defects are fixed. The failed window's actual engine-core traceback
was lost, so a successful GB10 startup is still unqualified.** No new live window,
GPU execution or serving request was performed by this lane.

## Evidence and the startup path

Read-only coordinator evidence: campaign
`TV1-COMPACT-W-20261027T031304Z`, reviewed runner7509c44. The s1 OFF completed
its cells; s2 compact failed. `cmd/0128-s2-compact-s2-compact-start.log` retains
only the launcher's final40 container lines. They contain the API process's
`Engine core initialization failed` wrapper, rather than the engine-worker root
cause. The Docker containers were subsequently removed. The wrapper is raised
by image `vllm/v1/engine/utils.py:1363` after a watched process exits; an empty
failed-process dictionary does not identify the failed Python operation.
The coordinator's binding brief reports no OOM/Xid in the relevant kernel logs;
this lane did not make a live kernel query.

A startup is not exclusively dummy execution. The image's
`vllm/v1/worker/gpu_worker.py:912` invokes v2 `warmup_kernels` when
`enable_jit_warmup` is true (default, `vllm/config/kernel.py:287`). That warmup
constructs real SchedulerOutputs and calls the worker at
`vllm/v1/worker/gpu/warmup.py:351–359,410–419`, then normal runner
`execute_model` through `gpu_worker.py:1318`. These calls have `dummy_run=False`.
The saved s1/prod argv use `--kernel-config '{"enable_flashinfer_autotune":false}'`,
with no JIT-warmup override. Separate profiling `_dummy_run` and target graph
capture do skip the new compact initialization hook; relying on those paths
alone misses the startup warmup.

## Defect 1: missing module import (SOURCE/CPU reproduced)

The saved compact overlay's runner uses `os.environ` at1384/1836/2441, but neither
it nor its K1 base imports/binds `os`. Its first non-dummy hook therefore raises
`NameError: name 'os' is not defined`. The old gather test explicitly supplied
`os` in its execution namespace, masking this defect; py_compile cannot find it.

All feature patch variants now add the single stdlib `import os`. The standalone
candidate patch had the same missing dependency; its only executable change this
round is that import. Candidate scoring, metadata mapping, qualification, decision
rules and thresholds are unchanged. OFF still builds the original two source
files byte for byte. Lock hashes and combined zero-fuzz patch coordinates are
regenerated. The repaired gather test binds only actual module imports.

## Defect 2: unconditional depth-policy refusal (SOURCE/CPU reproduced)

The saved production and s1 Config.Env records explicitly contain
`VLLM_DSPARK_DEPTH_POLICY=conf`. The K1 initializer creates `DSparkDepthPolicy`
at runner687–734: remote capacity has confidence support and disables the
separate adaptive-verification manager (remote `speculator.py:275–276`).
The old compact guard at saved runner1841 rejected every non-null depth policy.
After repairing the import, that guard would raise its compact mode ValueError
on the same recorded configuration. Changing the arm's environment to switch
off production's policy would confound the experiment and is not the fix.

`require_depth_policy` now permits only the recorded **plain conf** policy or
None. Calibration, prompt lookup, context adjustments, file reload and all other
policy modes remain refused. Existing policy state and the original `dead_rows`,
sampler invalid-draft union, `record` and `observe` hooks remain intact.

For reserved drafts r, compact width c<=r and policy depth d, the final live prefix
is min(c,d). The target correction/bonus row at that position remains computed;
rows after d stay masked and cannot be accepted. The actual installed mask and
observation methods are exercised on CPU for every r=0..5, c=0..r, d=0..r and
every possible acceptance0..min(c,d). Observation already clips live drafts to
min(actual drafts, depth). This is a source/CPU composition check, not a numerical
GPU sampler qualification. The unchanged rejection sampler and GB10 greedy gates
are still mandatory.

## Other candidate causes examined

| Path | Source conclusion and remaining limit |
|---|---|
| Env parsing | Mounted hooks use exact string1; tau0.5 is finite and in(0,1]. No compact flag is parsed by vLLM's env registry. Actual tau parser is now rehearsed before HOLD. Invalid tau refuses. |
| Graph sizes | Compact adds no capture sizes, graph pools or graph configuration. Dummy gather returns None before compact mapping; added need_eager remains false for dummy graph dispatch. Real shortened forwards request eager. Uncommon eager shapes still need GB10 execution. |
| SM121 kernel availability | Compact introduces no custom CUDA/Triton kernel. It uses ordinary torch stream/event, pinned float32 staging, tensor broadcast/copy and existing target kernels. No new architecture-specific kernel is a proven cause. Driver/stream failure remains possible and needs the full logs. |
| Scheduler/KV | Scheduler reservations and K1 cache allocation remain unchanged; only actual per-request worker counts shrink. CP/DP/PP/LoRA/batch sharding and local adaptive verification remain refused. The recorded depth-policy incompatibility is fixed as above. |
| Memory | New resources are one CUDA stream/event plus small confidence staging (at most4x5 float32 values per record in this configuration). There is no new model weight, KV pool or graph reservation. Allocator/driver overhead and actual low-memory startup still require live evidence; absence of kernel OOM alone cannot exclude an allocation failure. |

## Safe source smoke and the next window

Compact preflight now builds/verifies the pinned overlay locally and runs
`init_smoke.py` **before the first remote command or HOLD**. It executes the actual
stdlib bindings and init/threshold/mode/dispatch expressions, resolves the actual
NamedTuple, and checks the recorded plain-conf configuration. It does not inject
`os`, import the model/torch, create a CUDA context or stop production. The receipt
`compact-init-source-smoke.json` states `SOURCE/CPU ONLY` and
`cuda_or_engine_init_qualified=false`. Negative controls catch the original missing
import and blanket depth refusal. This is a cheap defect precheck; it is not a
full engine initialization self-test. A full TP2 model-load/warmup/graph test
cannot safely coexist with live production's5–7GiB headroom under this lane's rules.

The coordinator's next independently reviewed compact window must retain both
rank logs and receipts from before removal (see WINDOW.md). On failure inspect
the first worker traceback, including rank, operation, Python path:line and CUDA
error; identify whether it fails during load, cache initialization, scheduler
warmup, graph capture or readiness. Check container State/ExitCode/OOMKilled and
complete/hash fields. Oversize or timeout evidence is explicitly incomplete.
Do not infer a root cause from another API tail. Keep all six greedy checks,
source/env/mount/KV identities, original recovery ceilings and post-close watch.
No speed/identity or successful-start claim is made until those gates execute.
