# Round 3 larger-chunk admission

8K is a conditional source/CPU candidate. 16K is registered for arithmetic and
inspection, but REFUSED before host action and by the experiment launcher.
Neither has a measured native startup, peak, correctness result or speed gain.
Use one C/X/C HOLD per admitted arm. Do not run a 16K window at this revision.

## Allocator identity

[CAPACITY.md](CAPACITY.md) gives the unchanged checked allocator locators and
production environment. Fixed bytes remain6,442,450,944/rank, pool stride138,240,
pool blocks46,603, model cap327,680. Async PP1 in-flight tokens remain twice the
registered max-num-batched-tokens. Full groups require2,560+1,707+1 blocks.

| Registered pair | In-flight | SWA blocks/group | Request blocks | Derived token equivalent |
|---|---:|---:|---:|---:|
| chunk8192:8192/1024 |16384|259|7894|1934490|
| chunk16384:16384/1024 |32768|515|11478|1330447|

`SWA = ceil((127+2*chunk)/64)+1`, request blocks=`4268+14*SWA`,
tokens=`int(46603/request_blocks*327680)`. Both exceed the configured cap and
the150K+512 request. These are source-derived admission equivalents, not more
allocated KV or cache reuse. Existing tests execute the actual checked allocator
function bodies for all six pairs. Cold salts/zero-cache-hit gates stay required.

## Activation arithmetic and measured evidence

Production K1 uses TF prompt/transient/alias. Source:
`offload/kvpack/overlay/vllm_exl3_moe/tf_prompt.py:27-39,82-92,109-122` and
`tf_experts.py:379-387,417-421`. At R rows, six routed slots, H5120, I1152/rank,
the aliased MoE scratch is `R*6*(4*5120 + 2*1152 + 8*1152)` bytes:
750MiB at4096,1500MiB at8192,3000MiB at16384. The two rotated fp16 inputs
also hold the down fp32 partials; no second down buffer is added. Persistent
expert membership tables and work lists grow too (`tf_experts.py:313,407-409`).

MEASURED K1 R1p probes, the same instrumented records inventoried by SP1:
`/opt/dsv41/repos/dsv41-deploy/offload/results/K1/extra/memprobe/{sparkhead,sparkworker-wired}/dsv41-memprobe-R1p_r{0,1}.jsonl`.
Both ranks'4092-row full prefill phases have max FFN875MiB, attention579MiB,
Engram547MiB peak-over-entry; startup4096-row FFN max885MiB (attention567MiB).
Producer `offload/kvpack/overlay/vllm/v1/worker/gpu/dsv41_memprobe.py:124-139`
reports integer MiB, relative to phase entry, not total device residency or RSS.
Model's actual FFN bracket is `.../models/deepseek_v41/nvidia/model.py:632-637`.
The probe affects timing; none of these numbers is used for a speed comparison.

The estimate charges **885MiB per additional4096 rows**, larger than the750MiB
core scratch slope, plus **256MiB** allocator/metadata contingency per new arm.
This assumes roughly linear activation growth over this grid, including the
measured phase overhead. It is an admission estimate, not a measured upper bound.
It grants no credit for hypothetical allocator reuse or smaller KV allocation.
The gathered indexer buffer remains `327680*4` entries, independent of chunk
(`src-image/vllm/v1/attention/backends/mla/indexer.py:893-910`); logits are chunked
under their existing byte ceiling (`:1327-1365`). No factor or ceiling is changed.

MEASURED MemAvailable minima (kB), within ledger send-through-completion ranges;
these include decode too, so do not claim an isolated activation measurement:

| Saved window / cell | sparkhead | sparkworker-wired |
|---|---:|---:|
| PC1-chunk2048-W-20261027T100908Z/control150K |7036016|8045492|
| same/control mixed150K+2decoders |7003268|8041908|
| SP1-W-20261026T144153Z/unprofiled150K |6749280|7872596|
| SP1-W-20261026T160012Z/unprofiled150K |7100424|7878980|
| SP1-W-20261026T191247Z/unprofiled150K |7107216|7899796|

Each window's `telemetry-{host}.jsonl` and `requests-ledger.jsonl` are read-only
inputs, SHA256-pinned in `headroom-inputs.json`. The historical PC1 arm never
reached candidate requests. SP1 profiled150K sparkhead minima6425784/6676268/6686600kB
include profiler overhead and are not the baseline for this profiler-free arm.
Older PF1 minima4.60GiB (SP1's PROFILE-WINDOW-RECONCILE.md:193-204) demonstrate
host-state variation: historical headroom alone cannot authorize a later start.

ESTIMATED projection from the worst unprofiled150K minimum per host:

| Arm | Extra reserve(kB) | sparkhead projected(kB/GiB) | sparkworker projected(kB/GiB) | Admission |
|---|---:|---:|---:|---|
|8192|1168384|5580896 /5.322|6704212 /6.393|conditional|
|16384|2980864|3768416 /3.594|4891732 /4.665|REFUSED|

5GiB =5242880kB. Before8192 starts, **both ranks' entire fresh first-control
minimum** must be at least6411264kB (floor+reserve). Missing evidence or a1kB
shortfall refuses before candidate start and restores. A PC1-only Supervisor
subclass retains shared telemetry/ownership and strengthens the live floor to
5GiB throughout the larger-chunk window; all original4GiB, swap, staleness,
demand and recovery checks still run. The shared monitor is unchanged.
The 1s telemetry cannot rule out subsecond dips. Failure of the estimate/guard
invalidates the candidate; this is not an allocation trace or a guarantee.

## Window cost

MEASURED earlier PC1 launcher readiness:353s(control),337s(chunk2048), from
`cmd/0030-s1-control-start.log` and `0094-s2-chunk2048-start.log` in the named
read-only PC1 window. Previous campaign starts388/404s remain the conservative
startup basis. Larger-chunk startup is UNMEASURED: add180s ESTIMATED allowance
only to its candidate block, including the same immutable premeasurement limit.

For each new arm, planning only: `120+1214+1424+1214+720=4692s=78.20min` HOLD.
16K has the same cost estimate but cannot execute because memory admission fails.
73min submission/abort,85min restore, all-remaining-start reservation, six greedy
checks/start, unchanged W8 and15min post-close watch outside HOLD are preserved.
No extra retries or deadline renewal is permitted. Mixed load's existing145s/group
allowance and new-start variance remain estimates; a slow arm can STOP the screen.
