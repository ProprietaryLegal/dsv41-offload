# SP1-WINDOW item 1: the SP1 profiling recipe reconciled with today's production

Lane SP1-WINDOW, 2026-10-26, CPU only on Controller. This lane did not access sparkhead, sparkworker or x86host, and sent nothing to
:8000, :8001 or :18001. Every **MEASURED** value comes from a saved file named next to it. Values the runner must re-read
live are marked **RUNNER CHECKS**. Source line numbers refer to the read-only image copy `/opt/dsv41/data/src-image/`
(below, `img/`) or to this branch.

## Plain language

- **Verdict: the window can run safely.** One experiment start profiles today's production at 327,680. The runner
  always ends with the W8 restore. Items 2-6 follow.
- **The old SP1 recipe cannot run as written** (SP1.report.md "Item 4"):
  - Its base config predates K1 production and mounts the experiment checkout's drifted remote-drafter client, so it
    would not profile today's production.
  - It runs `start_exp.sh` from the production checkout, so it would write `logs/` there.
  - It restores with a bare `./stop.sh; ./start.sh`, with no greedy or x86host check.
  - It has no memory, disk, deadline or foreign-demand guard.
- **The profiled arm is PF1's OFF arm at 327,680**, plus two additions:
  - one `--profiler-config` flag, which keeps the profiler off until `/start_profile`;
  - one read-write trace directory mount.

  PF1's OFF arm reads production's own `dsv41.env` and mounts. It starts from the separate checkout
  `~/dsv41-deploy-offload` on port 8001. Five PF1 windows have already started it, and production was restored each time.
- **CUDA graphs.** Decode runs inside CUDA graphs. Prefill does not.
  - The image's profiler records kernels through CUPTI.
  - CUPTI reports each kernel of a replayed graph as its own record, carrying the correlation id of the graph launch.
  - The only local proof is an analog: torch 2.10 on x86host, where W1's CUPTI breakdown of graph replays returned
    per-kernel times. The image runs torch 2.13 / CUDA 13.0.2, which has not been checked.
  - So the window's first capture tests this, and the analysis tool flags it (`graph_visibility`).
  - If graph kernels turn out invisible, decode still yields step time, host gaps, launch cost and coverage.
    Prefill is fully attributable either way.
- **No `--enforce-eager` arm.**
  - Its start time has never been measured on GB10. The often-quoted 1,456 s belongs to a different switch,
    `CUDA_MODULE_LOADING=EAGER`.
  - Eager decode would replace the very host and launch gaps the window must measure.
  - It stays a contingency HOLD, used only if `graph_visibility` comes back `invisible`.
- **Memory is the binding risk, not disk.**
  - Sparkhead at 327,680 sits at 4.9-5.9 GiB MemAvailable while serving (MEASURED).
  - Its deeper dips (4.17-4.81 GiB) came from building long prompts with repeated `/tokenize` bisection.
  - The window therefore builds every prompt before the first capture, with at most four `/tokenize` calls each (`window_cells.MAX_COUNT_CALLS`).
  - It also refuses to start a capture below 4.5 GiB, and keeps PF1's 4 GiB STOP at all times.
- **Nsight is not in scope.** Nothing local shows `nsys` in the image, and launching under it needs a launcher change.
  It stays a follow-up (`PROFILE-WINDOW.md` §6). The precheck only reports whether `nsys` exists.

## 1. Which arm is profiled

| Item | Evidence | Window rule |
|---|---|---|
| Production today | K1 at 327,680. Sparkhead main `21bed31`, tag `prod-k1-327-20261023`. KV 2,502,600, four sequences, TP2 K2 drafter on the x86host NVLink pair, async process proxy, remote Engram. Sources: brief; `pf1_window.py:44-48`; LANES 2026-10-26 13:11 | **RUNNER CHECKS** (PF1 `Live.preflight`): ledger HEAD = 21bed31 = tag; `dsv41.env`/`serve.sh`/`start.sh` clean; `:8000/v1/models` = one model, 327,680, `/srv/dsv41/legal-v2a`; worker fb68c52, TP2, `--max-model-len 1048576` |
| Production-equivalent arm | `offload/launch/configs/PF1-window.env`. OFF (`PF1_TRACE=0`) sources production's `OFFLOAD_DOCKER_ENV`, `OFFLOAD_DSPARK_ENV` and `OFFLOAD_DOCKER_ARGS` in a subshell (`:21-28`). It refuses an inherited `EXP_EXTRA_ARGS`/`EXTRA_DOCKER_ARGS` (`:9-12`) and any ledger other than `6442450944:327680:1:327680` (`:30-31`). It differs from production only in port 8001, the cap and `VLLM_ENGRAM_REMOTE_REQUIRED=1` (PF1-WINDOW-RECONCILE.md §2) | new tracked `offload/launch/configs/SP1-profile.env` sources PF1-window.env with `PF1_CAP=327680 PF1_TRACE=0 PF1_EAGER=0` **first**, then appends the two additions below. Nothing else changes |
| Experiment checkout | `start_exp.sh:9` does `cd` to the checkout that contains it and writes `logs/` there (`:50-53`). PF1 uses `~/dsv41-deploy-offload`, bundle-staged at the reviewed commit (`pf1_window.py:806-815, 829-839`) | same checkout, same staging, same pin check. **Never** `~/dsv41-deploy` (runner refuses `--remote-repo dsv41-deploy`) |
| Port and tunnel | `start_exp.sh:12` sets `PORT=8001`. The PF1 tunnel is `127.0.0.1:18001 → sparkhead 127.0.0.1:8001`, owned by the runner, which refuses a pre-existing listener (`pf1_window.py:508-522`) | same |
| Profiler flag on both ranks | `EXP_EXTRA_ARGS` joins `EXTRA` (`serve_exp.sh:29-30`). It is written into each rank's `/serve.sh` (`dsv41_tp2_offload.sh:77-87`). The headless rank 1 gets the same line (`:86`) | `--profiler-config '<JSON>'` (exact JSON in §2), the same on both ranks |
| Trace mount on both ranks | `EXTRA_DOCKER_ARGS` is spliced into `docker run` on each node (`dsv41_tp2_offload.sh:95`). `\$HOME` expands on that node, as PF1's trace mount does (`PF1-window.env:79`) | `-v $HOME/dsv41-offload-run/sp1-profile:/sp1-profile:rw` on each node. **RUNNER CHECKS** both containers have exactly this read-write destination. The directory must be absent before the window and is created empty by the runner |
| Identity versus production | `capture_window_identity.sh` hashes mounts only under `dist-packages`, `/opt/dsv41-ofl` and `/serve.sh`. A `/sp1-profile` mount does not enter `mounted_code_sha256`. PF1 `runtime()` requires argv to equal production except `--port`/`--max-model-len` (`pf1_window.py:261-269`) | **RUNNER CHECKS** PF1 `runtime()` against the production snapshot after removing exactly one `--profiler-config` whose JSON equals the reviewed JSON byte for byte. Env, mounts, image, engine, Engram identity and KV are all equal to the OFF arm. The 13-pin in-container verify passes. The rank-0 log contains vLLM's "Profiler with mode 'torch' is enabled in the API server" line |
| GPU clocks | `start_exp.sh:31` re-applies `GPU_CLOCK_CAP` from the experiment checkout's `dsv41.env`, the same value production's start applies (PF1-WINDOW-DESIGN.md "What every arm holds equal") | no clock or power change; the telemetry records the actual clocks |
| TP2 K2 drafter on x86host | The same worker serves production and the experiment. PF1 reads it only through its status file and `/proc` (`X86HOST_PROBE`, `pf1_window.py:210-221`) | never restarted, reconfigured or sent a request. The only x86host contacts are the read-only probe and `W8/restore.sh`'s own `prod_window.sh check` |

What the old recipe did, line by line, and what replaces it:

| SP1.report.md | Old action | Replacement |
|---|---|---|
| 412-413 | `prepare_profile --base offload/launch/configs/D-kv6-async-pack.env`. That base mounts the checkout's `offload/dspark/spark/remote_dspark` and K1 copies (`D-kv6-async-pack.env` OVERLAYS line). Its drafter env differs from production (`SPIN_US=50 NAP_US=20`, no respawn settings, own identities). PF1-WINDOW-RECONCILE.md §2 found that checkout's `speculator.py` drifted and `suffix_provenance.py` extra | `SP1-profile.env` on top of `PF1-window.env`: production's bytes and env |
| 417, 436 | `scp` to `sparkhead:/tmp/SP1-profile.env`, then `start_exp.sh` from `cd "$HOME/dsv41-deploy"`: writes `logs/` into production and runs production's older launcher copy | tracked config in the staged experiment commit; `start_exp.sh` from `~/dsv41-deploy-offload` |
| 432-434 | `window.sh open` by hand; `mkdir` on both nodes by hand | the runner's `window.sh open` with a bounded drain; per-node `mkdir` of a directory that must be absent |
| 463-497 | `capture.py` and `bench2.py` per cell. `bench2.make_prompt` bisects with `/tokenize` (`bench2.py:270-296`, about 20 calls of the full prompt). `bench2.stream_request` re-tokenizes generated text afterwards (`:176-182`). `capture.py` keeps no client times for start/stop | new cell driver `offload/gb10_budget/window_cells.py`: all prompts built before any capture, at most 4 `/tokenize` calls each; the server's usage gives exact counts; client times of start/stop are recorded (§3) |
| 532-535 | `scp -r` of whole trace directories, no size bound | per-capture listing, size cap and disk check, sha256 on both sides, then removal of the verified files on the Spark (§4) |
| 597-602 | `./stop.sh; ./start.sh; window.sh close` | `W8/restore.sh start|check|close`, checked against its real output (unchanged PF1 `restored()`, `pf1_window.py:447-456`). **RUNNER CHECKS** restore.sh sha256 = `d1c4479d…` (verified unchanged today) |

## 2. Profiler fields, endpoints and the iteration bound (verified against the image source)

The reviewed JSON (`SP1-profile.env`), and the image source for each field:

| Field | Value | Image source | Why |
|---|---|---|---|
| `profiler` | `"torch"` | `img/vllm/config/profiler.py:43-48`; kinds `torch`, `cuda`, `proton` (`:16`) | torch/CUPTI needs no launcher change |
| `torch_profiler_dir` | `"/sp1-profile"` | `:50-53`; required when `profiler=torch` (`:208-209`); made absolute (`:213-216`) | the read-write mount |
| `torch_profiler_activities` | `["CPU","CUDA"]` | `:55-59`; validated against the platform (`wrapper.py:644-672`) | CPU gives iteration annotations and launch calls; CUDA gives kernels |
| `torch_profiler_with_stack` | `false` | `:90-93`, **default true** | no Python stacks: smaller traces, no source text |
| `torch_profiler_with_flops` | `false` | `:95-96` | — |
| `torch_profiler_record_shapes` | `false` | `:104-105` | — |
| `torch_profiler_with_memory` | `false` | `:107-109` | — |
| `torch_profiler_use_gzip` | `true` | `:98-99`; handler `wrapper.py:224-228` | traces are written `.pt.trace.json.gz` |
| `torch_profiler_dump_cuda_time_total` | `false` | `:101-102`, **default true**; when on, `_stop` builds a `key_averages` table and prints it on rank 0 (`wrapper.py:340-361`) | torch (2.12 source on Controller) parses events into Python objects lazily; the table would force that parse for every event inside the stop RPC. Off saves host memory and the stall |
| `ignore_frontend` | `true` | `:122-127`; otherwise AsyncLLM records its own CPU trace (`img/vllm/v1/engine/async_llm.py:213-232`) | front end not needed; no overhead |
| `detailed_trace_annotation` | `true` | `:116-120`; annotation `execute_<tokens>_context_<n>(sq…sk…sqsq…sqsk…)_generation_<n>(…)` (`gpu_worker.py:1148-1229`) | gives per-iteration scheduled tokens and the prefix (`sk`) of prefill chunks |
| `max_iterations` | `16` | `:134-137` | bounds each capture (next row) |
| `delay_iterations`, `warmup_iterations`, `wait_iterations` | defaults (0) | `:129-158` | no schedule; recording starts at `/start_profile` |
| `capture_torch_profiler`, `proton_*` | defaults | `:111-114`, `:61-88` | no capture-time profiling; no Proton |

The config fields were checked against the image's `ProfilerConfig` (lines 39-158) and are pinned by a CPU test. The
config is parsed from JSON like the other `--*-config` flags (`img/vllm/engine/arg_utils.py:184, 756, 1847`).

**Endpoints.**

- `POST /start_profile` and `POST /stop_profile` are registered only when a profiler is configured
  (`img/vllm/entrypoints/serve/profile/api_router.py:21-46`, attached by `entrypoints/serve/__init__.py:30-34`).
- Each call awaits `engine_core.profile_async` (`async_llm.py:1076-1086`). That is a `collective_rpc("profile")` to
  every worker (`img/vllm/v1/executor/abstract.py:272-273`; `core.py:822-823`), rank 1 included, so both global ranks
  start and stop together.
- The HTTP call returns only after every worker answers. So the trace files of a stopped capture exist once
  `/stop_profile` returns: the export runs inside the stop (`wrapper.py:340-342` → the trace handler).
- Utility calls are served between engine steps, so a start or stop can lag by up to one step. During prefill one
  step is one ~1.85 s chunk.

**Trace names.**

- The trace is `dp0_pp0_tp<r>_dcp0_ep<e>_rank<g>.<time_ns>.pt.trace.json.gz` (`gpu_worker.py:1365-1387`;
  `img/vllm/distributed/utils.py:689-730`; torch `tensorboard_trace_handler`).
- Each capture writes one file per rank on that rank's node; the nanosecond suffix keeps repeated captures apart.
- `local_rank` is 0 on both nodes. The runner identifies ranks by the global `rank<g>` in the name and by the node,
  never by `local_rank`.

**Iteration bound.**

- `step()` runs at the start of every annotated `execute_model` (`gpu_worker.py:1135-1142, 1317`).
- The profiler stops itself once the counted steps exceed `max_iterations` (`wrapper.py:103-136`). With 16, a capture
  records 16 annotated iterations and exports at the 17th. The last iteration's period is incomplete, so a decode
  capture gives **15 complete step periods**.
- An explicit `/stop_profile` after that is a logged no-op (`wrapper.py:149-158`).
- Prefill captures are bounded instead by the driver's stop after ≤ 5 s: two or three chunks.
- The trace export synchronizes the device and stalls one step on both ranks. Profiled requests are
  "profile-affected", never promotion timings.

**What the annotation covers.**

- Only `execute_model` is wrapped. `sample_tokens` runs outside the annotation (`gpu_worker.py:1248-1250`).
- The tool therefore assigns every kernel to an iteration through its launch call. A launch between annotation *i* and
  annotation *i+1*, in any worker thread, belongs to iteration *i*.
- The step period is measured on the GPU timeline, from the first kernel of iteration *i* to the first kernel of
  iteration *i+1*.

## 3. CUDA graphs: what this build can and cannot show

| Fact | Evidence |
|---|---|
| Decode runs in full CUDA graphs; prefill and mixed steps run eager | `cudagraph_mode FULL_DECODE_ONLY`, capture sizes 1-48 (MEASURED engine config line, `UP2-screen-300K-20261025T131508Z/0230-sparkhead.log`); launcher default `GRAPH` (`dsv41_tp2_offload.sh:27`) |
| Image torch/CUDA | torch 2.13.0+cu130, CUDA 13.0.2 (`OUTLINE-for-SWE-review-20261022T1917Z.md:22`; `O1.progress.md:19`) |
| Kernels inside a replayed graph appear in a torch/CUPTI trace | **Analog MEASURED, not this build.** W1's CUPTI diagnostic profiled `draft_graphed` (a CUDA-graph replay, `offload/dspark/worker/model.py:574-597`) with `torch.profiler` CUDA activity (`offload/dspark/worker/tools/bench.py:73-90`, `profile_current.py:184-186`). It reported per-group kernel totals for B1-B4 (`W1.report.md:241-243`), built with torch 2.10.0+cu126 / CUDA 12.6 on an RTX 3090 (`W1.report.md:256`). CUPTI reports each kernel node of a replayed graph as a kernel record carrying the launch's correlation id |
| No per-layer CPU scopes in the model | no `record_function` in `img/vllm/models/deepseek_v41/` or `img/vllm_exl3_moe/` (grep) |

So attribution is by **kernel name**, not Python scope. In graph replays every kernel shares the `cudaGraphLaunch`
correlation id. The tool's buckets are kernel-name patterns read from the image's own symbols (item 4).

The tool computes `graph_visibility` per capture:

| Value | Meaning |
|---|---|
| `visible` | every graph launch has kernels |
| `invisible` | none has |
| `partial` | some have |
| `no_graphs` | no graph launch was captured |

If decode turns out `invisible`, the window still yields, per decode step:

- wall time from the iteration annotations;
- CPU launch cost: graph-launch API time and the eager kernels outside the graph (input prep, sampler and drafter
  hand-off when they run eagerly);
- the host gaps between steps;
- coverage: the share of the step with any attributable kernel.

It cannot then split MoE, dense, attention and collectives *inside* the graph for decode. Two more things stay
available:

- **Prefill.** It is eager, so it is fully attributable: per-chunk MoE (TensorFold grouped), dense, indexer/attention,
  NCCL and ring collectives, Engram, host gaps. That holds for both 32K and 150K and for both ranks.
- **Collectives in decode.** They are opaque in that case too. Rank skew can still be read from the stop/start
  alignment of the two ranks' step periods.

**`--enforce-eager` arm: not justified as a default.**

- Its start time on GB10 is **not measured** anywhere in the campaign. The commonly cited 1,456 s is
  `CUDA_MODULE_LOADING=EAGER` (COMPARISON.md "E0 + EAGER isolation", line 236), which is a different switch.
- Eager decode replaces graph replay with per-kernel Python launches. It would therefore measure a different host path,
  exactly the gaps this window is meant to measure. Its kernels would be the same as the graph's, but its exposed
  shares would not be production's.
- It adds a second start (≥ 7 min ESTIMATED, from the 6.8 min graph start) and a second set of cells. Prefill gains
  nothing, because prefill is already eager.
- Decision: **contingency HOLD only**, run if and only if this window reports decode `graph_visibility = invisible`
  (`PROFILE-WINDOW-DESIGN.md` "Contingency").

**Nsight Systems.**

- No campaign file shows `nsys` inside the image, so item 1 cannot prove it is there.
- `nsys profile` must wrap `vllm serve` at process creation (SP1.report.md 571-591). That requires changing
  `dsv41_tp2_offload.sh`'s `docker run` entry point, which is not a trivial launcher change.
- It is listed as a follow-up. The precheck reports `command -v nsys` inside the production container as INFO only.

## 4. Trace size, host memory and disk

**Host memory, MEASURED on the 327,680 OFF arm.**

These are sparkhead's MemAvailable minima per phase, from the 1 s telemetry: PF1-W1 and PF1-W3, `telemetry-sparkhead.jsonl`
and `requests-ledger.jsonl`. sparkworker stayed at or above 6.83 GiB throughout.

| Phase (sparkhead) | W1 | W3 |
|---|---:|---:|
| ready, before the warm-up | 5.92 | 5.34-5.39 |
| 150K cold prefill (first / second request) | 5.54 / 5.90 | 4.60 / 5.03 |
| 150K decode | 5.90-5.92 | 4.98-5.03 |
| next prompt being built (`/tokenize` bisection) | 4.81-4.87 | **4.17**-4.36 |

The deepest dips came while the runner was building the next long prompt, through repeated `/tokenize` calls on the API
server. They were not during prefill or decode. PF1 MATCH stopped at `sparkhead MemAvailable 4181536 kB below 4 GiB`
during `off-524-checks` (MEASURED, `PF1-MATCH-20261026T045943Z/summary.json`), at 524,288, which this window does not
use.

Rules:

1. **STOP floor.** MemAvailable ≥ 4 GiB on both Sparks, polled every 1 s, from before the HOLD to the restore.
   This is PF1's `Supervisor` unchanged. Swap growth > 256 MiB after readiness is also a STOP.
2. **Start floor.** ≥ 5 GiB on both Sparks before the experiment start (PF1).
3. **Capture floor.** ≥ 4.5 GiB on both Sparks in the latest telemetry sample (≤ 3 s old) immediately before every
   `/start_profile`. Below it, that capture is **skipped and recorded** (`SKIPPED_MEMORY`); no profiler is started.
   The basis is ESTIMATED: profiler footprint ≤ 0.4 GiB per rank for one capture.
   - Kineto/CUPTI activity buffers.
   - CPU op records for ≤ 3 eager chunks or 16 decode steps (≤ ~60K events).
   - The JSON export, which streams to the container's `/tmp` and then gzips.
   - With `dump_cuda_time_total=false` there is no `key_averages` parse.
   - 4.5 − 0.4 leaves ≥ 0.1 GiB above the STOP floor. W3's steady 4.9-5.05 GiB passes.
4. **Prompt construction** happens once, before the first capture, with ≤ 4 `/tokenize` calls per prompt. It never
   bisects. No 300K prompt is built, so the dip that tripped PF1 is avoided.
5. **No 150K B4 cell.** Four 150K prefills is the unmeasured case SP1 itself flagged. It is out of the brief's list.

**Trace size (ESTIMATED; MEASURED at the first capture).**

| Capture | Events per rank | JSON | gzip |
|---|---|---|---|
| Decode, 16 steps, graph kernels visible | about 1.2-2.4K kernels per step plus ≤ 1K CPU/runtime events → ≤ 55K | ≤ 20 MB | ≤ 4 MB |
| Prefill, ≤ 5 s (2-3 chunks of 4,092 tokens, eager) | about 15-25K per chunk (CPU op, launch, kernel) → ≤ 75K | ≤ 30 MB | ≤ 6 MB |

**Hard caps** (runner). After each `/stop_profile`, the runner lists the profile directory on each node and requires:

| Check | Requirement |
|---|---|
| New files | exactly one new `rank<g>` file per node (rank 0 on sparkhead, rank 1 on sparkworker) |
| Size of each new file | ≤ **128 MiB** |
| Total profile directory per node | ≤ **512 MiB** |
| Free home disk afterwards | ≥ **8 GiB** |

Any breach is a STOP, and the restore runs.

**Disk.**

- Precheck and preflight require ≥ 10 GiB free in each Spark's home.
- `~/dsv41-offload-run/sp1-profile` must not exist before the window.
- After each capture, the files are copied to Controller and sha256-compared on both sides.
- The verified files are then removed from the Spark. The directory is user-owned, so a plain `rm` of the named files
  works, even though the container writes them as root.
- The empty directory is removed at the end.
- On Controller the traces live in the window directory; ESTIMATED ≤ 100 MB per window.

## 5. Conclusion

The window can run safely on today's production. Item 2 (`PROFILE-WINDOW-DESIGN.md`) designs it under the rules above.
Nothing here edits production files, the K1 package or the restore script.
