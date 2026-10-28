# KL1 stage B: one microbenchmark window, then at most two kernel builds (written 2026-10-26, NOT RUN)

Stage A (KL1.report.md, outside git) found 0–10 % ESTIMATED decode headroom at B1 and 0–6 % at B4, with no single
number pinned down. The traces cannot tell whether smoe at B1 runs near the rate of production's other EXL3 streams
(216–235 GB/s, LM head) or 20–25 % below it. This window measures isolated smoe curves and infers an expert count by
matching production spans. The resulting rate is the bench's own rate at that count; it cannot independently measure
production bandwidth or prove that production is slower per byte. It also measures build flags, fused MoE and mrow32
variants under the scoped bitwise oracles below. Nothing is built for serving until the gates pass.

## What runs

- **One container per Spark, both Sparks at once** (sparkhead = rank-0 shapes, sparkworker = rank-1 shapes), started from
  the pinned production image during a coordinator window with production down. Both now use DGX OS 7.6,
  kernel `7.0.0-1019-nvidia`, driver `580.178.04`. With synthetic data, `--rank` changes nothing: both shards are
  1152 wide and get the same random inputs. The Sparks are a **hardware replicate**, not distinct routing samples.
  No engine, no model, no weights, no network.
- **Inputs.**
  - Synthetic EXL3 banks: random trellis words and random signs/scales. EXL3 decode is data-independent arithmetic,
    so bytes and time match real weights.
  - Laid out by `vllm_exl3_moe.bank.Exl3Bank`, with legal-v2a's real per-layer K from
    `/srv/dsv41/legal-v2a/exl3_moe_plan.json`, mounted read-only as a single file. Fallback: `synthetic_plan` with the
    logged histogram (6095/7075/2125/65).
  - Eight layers (0, 6, 11, 17, 22, 28, 33, 39), cycled over 40 calls so that L2 never holds a layer.
- **Routing.**
  - No routing capture exists, and making one needs an engine hook. Instead, `smoe_bench.cases()` sweeps the number of
    distinct experts:
    - B1: rows 3/4/6, 6–24 distinct experts. This brackets stage A's bound of at most 14–19 at 273 GB/s.
    - B4: rows 16/20 over 4 requests, independent or with a shared core.
  - `analyze --window ... --rank ...` selects one SP1 B1 window/rank from `KL1-bandwidth.json` and checks its saved
    stack log against the bench. Only post-FW1/FW2 data on `7.0.0-1019-nvidia` / `580.178.04` qualifies. Of the
    existing traces, only `SP1-W-20261026T160012Z` rank 0 qualifies (45 B1 steps); sparkworker and the earlier window are
    cross-stack and refused. The other Spark's bench supplies hardware replication via `--replicate-bench`, never
    pooled production spans. A future same-stack SP1 trace is needed for rank-1-specific production inference.
- **Kernels.**
  - `gpu`: smoe (rot + gate|up + down, shared expert as a slot) in a CUDA graph of 40 calls. Variants are
    `smoe_bench.VARIANTS`:
    - prod `-DSMOE_RUNFLAGS=1`;
    - PIPE 4/12/16;
    - OBI 1/4;
    - L2 hint 256;
    - run-flag sleep 64 ns;
    - run flags off.
  - `gpu-fused`: `smoe.fused` against `router_topk` + smoe, at 2–8 rows.
  - `gpu-dense`: the four exl3_mrow groups of a layer at 4/16/20 rows, for EXL3_MROW32_VARIANT ring/lean/ring1 ×
    EXL3_MROW32_MIN_ROWS 9 (production) / 17.
- **Bitwise oracle.**
  - Every smoe variant's 40-layer output digest must equal prod's for every case. Every variant keeps each column's
    single-block k loop and the pick-order fp32 sum, so any mismatch is a bug, not rounding.
  - The fused launch must equal `router_topk` + smoe in output, ids and weights. This bench oracle does not qualify
    serving: production uses `Exl3GateLinear` with `EXL3_ROUTER=1`, vLLM DSv4 top-k, then smoe; serving never calls
    `router_topk`. Routing scores and normalization order still need full-task and target-distribution gates.
  - Dense inputs are drawn once per row count and reused across all configurations. Each bank/group records
    `info32` so the slice plan is explicit. Dense digests are compared within one plan. mrow (min 17) vs mrow32 (min 9) at 9–16 rows are different k-slice
    plans: report the mismatch, and adopt only behind the distribution gate.

## Coordinator sheet (reviewed runner only; NOT RUN by KL1B-RUNNER)

`offload/launch/windows/kl1_window.py` (`KL1Live(pf.Live)`) and `kl1_window.sh` implement the window. Do not run bare
Docker commands by hand. Default/dry mode performs no command, network request or file write. `--execute` and the
read-only `--precheck` require `WINDOW_COORDINATOR=Deepseek/coordinator`, a fresh local campaign output directory, a
full reviewed local commit and a clean separate remote experiment checkout. `--precheck` cannot stage a bundle.
`--execute --stage-bundle` may use PF1's reviewed bundle staging if the separate experiment checkout already exists
but differs. No production checkout or W8 script is edited.

The adapter inherits PF1's restore pin, D2-pause, production ledger/no-HOLD, reviewed-source pins, K1-health, worker,
telemetry and snapshot/idle checks. It additionally verifies both nodes' image, equal O_DIRECT plan hashes, local disk
and stack before the HOLD. It checks no HOLD again immediately before open, opens/drains, saves production identity,
explicitly calls `./stop.sh`, and requires no GPU compute apps and ≥40 GiB MemAvailable on both Sparks before staging.

Sources are sha256-checked on **each node's own disk**, under that node's `~/dsv41-offload-run/kl1/src`; sparkworker's home
belongs to `rank1user`, so there is no mount of sparkhead's checkout over NFS. The run directory must be absent
and its parent must be on a local filesystem. Each node runs one `dsv41-kl1-bench` container with `timeout 1500s`,
network disabled, host memory capped at 32 GiB with no cgroup swap, a 24 GiB PyTorch CUDA allocator cap, and the plan
mounted read-only as a **single file**. Output/build mounts are user-owned and isolated from production's JIT cache.
The allocator cap bounds the bench's tensor allocations; the host cgroup and live MemAvailable guard cover builds and
other allocations on unified-memory GB10. smoe variants, all mrow/mrow32 K/variant instantiations, router and l2pf are
prebuilt; every child build's status is observed. `prod` runs first and last, surrounding the tuning/fused/dense runs.
Variant failures are recorded and the remaining variants run; any recorded failure makes the completed window fail.

One supervised local step launches both benches in parallel. Throughout measurement, MemAvailable ≥4 GiB, STOP,
telemetry liveness and the HOLD+40-minute abort deadline apply. On collection, every JSON is size- and sha256-checked
against an O_DIRECT remote digest before any copies from that node are deleted. Unverified results stay as evidence.
Every post-HOLD exit masks signals before closing the supervisor. The inline, source-pinned `kl1_remote.py` launcher
records its own remote process group, start time and boot identity under each node's
`~/dsv41-offload-run/kl1-control/<tag>/launcher.pid`. Registration and ABORT share a lock; the launcher checks ABORT
again immediately before Docker. `timeout --foreground` keeps its children in that recorded group. Restore first
writes ABORT on **both** nodes, then terminates and waits for only this run's identity-verified groups. It removes
the named containers and proves launcher absence, container absence and GPU idle on both nodes before W8
`start|check|close`. A stale/reused PID, surviving launcher or missing proof blocks W8 and fails visibly. The tiny
control directories and ABORT fences deliberately remain after disposable source/build cleanup: a delayed SSH
session must still refuse this run, even after restoration. Run tags cannot be reused.
If production was never stopped, cleanup permits its compute apps and W8 only
checks/closes. Restore commands are bounded by the remaining HOLD+55 budget. Any missed deadline, missing cleanup or
missing exact W8 success string fails visibly, leaves the HOLD open and prints the manual restore commands. The exact
strings are `K1 START VERIFIED`, six `greedy K1-post vs restored: N IDENTICAL M` lines for N=0–5, no `DIVERGES`,
`PRODUCTION HOLD OK`, `CHECK: DONE`, and `CLOSE: window closed`.

Only the coordinator runs the following from the **reviewed merge checkout on Controller**. These commands are a sheet,
not authorization for this headless lane to contact any host. Supply the full reviewed SHA in `REVIEWED` first.

```bash
source "$HOME/.config/ndm/paths.sh"
: "${REVIEWED:?set REVIEWED to the full 40-hex reviewed merge commit}"
CAMPAIGN=/opt/dsv41/campaigns/DSV41-OFFLOAD
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$CAMPAIGN/KL1-W-$STAMP"
PRE="$CAMPAIGN/KL1-PRECHECK-$STAMP"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1

# Local dry plan (no network or file).
/home/operator/miniconda3/bin/python3 -m offload.launch.windows.kl1_window --dry-run

# Coordinator only: read-only remote precheck; no HOLD, no stop and no bench.
WINDOW_COORDINATOR=Deepseek/coordinator /home/operator/miniconda3/bin/python3 \
  -m offload.launch.windows.kl1_window --precheck --out "$PRE" \
  --expected-experiment-commit "$REVIEWED"

# Coordinator only: supervised HOLD/window/restore; fresh OUT. Optional --stage-bundle as described above.
WINDOW_COORDINATOR=Deepseek/coordinator /home/operator/miniconda3/bin/python3 \
  -m offload.launch.windows.kl1_window --execute --out "$OUT" \
  --expected-experiment-commit "$REVIEWED"

# Afterwards, CPU only. Explicit smoe files: do not feed dense/fused/failures JSON to analyze.
variants=(prod-first pipe4 pipe12 pipe16 obi1 obi4 l2h256 sleep64 norunflags prod-last)
rank0=(); rank1=()
for v in "${variants[@]}"; do
  rank0+=("$OUT/results/sparkhead/$v.json")
  rank1+=("$OUT/results/sparkworker-wired/$v.json")
done
/home/operator/miniconda3/bin/python3 -m offload.kl1.smoe_bench analyze \
  --window SP1-W-20261026T160012Z --rank 0 \
  --bench "${rank0[@]}" --replicate-bench "${rank1[@]}" \
  --bandwidth "$CAMPAIGN/KL1-bandwidth.json" --out "$OUT/analysis-rank0.json"
# A future rank-1 analysis needs its OWN same-stack SP1 window and --rank 1, with rank1 primary and rank0 replicate.
```

The analysis records its selected stack-log path and sha256, bench window, both hardware ranks, oracle differences,
per-rank drift/gain medians and the joint gate. Omitting the hardware replicate reports an incomplete gate and can
never pass. Pooling windows/ranks, differing stacks/plans/seeds/banks, missing cases or prod endpoints are refused.

## ESTIMATED minutes (production down)

| Step | Minutes |
|---|---:|
| Stop production, HOLD checks and GPU-idle/memory checks | 2–3 |
| Container start, 9 smoe builds (3 at a time), all mrow/mrow32 variants × K2–6, router/l2pf | 6–9 |
| `gpu` × 9 variants plus closing `prod` control (19 GB random banks, 41 cases/run) | 10–11 |
| `gpu-fused` (7 row counts × 2 graphs) + `gpu-dense` (6 configs × 3 row counts) | 3–5 |
| Copy, verify, delete private scratch | 1–2 |
| W8/restore.sh start + check + close (MEASURED prior PF1/SP1, estimate here) | 12–14 |
| **Total production down, including mrow/router builds** | **≈34–44 ESTIMATED** |

Both Sparks run at once and on the same new stack. These are estimates, not a measured duration. Abort measurement
at HOLD+40 and complete restore by HOLD+55 even if a build, benchmark or copy takes longer than expected.

## Gates (decided on the analysis, CPU)

1. **Oracle.** Any smoe variant with a digest mismatch is discarded and reported as a bug. The fused launch moves on
   only if it is bitwise equal to the bench split at every row count; that equality alone never qualifies serving.
2. **B1 smoe.** Rate label: **"smoe isolated rate at production's inferred expert count"**.
   - Use the median over `where=inside` steps, report the clamped/flat/unsupported counts, and never pool production
     windows or ranks. Fit a non-decreasing cumulative-max curve; report flat stretches as non-identifiable counts.
     Missing row curves are explicit, not silently skipped. No in-range data or a missing hardware replicate is
     incomplete and cannot qualify a serving change.
   - Production's `moe` L2 prefetch warms the ≈11 MB shared expert, biasing the uncorrected count about **1.7 experts
     low (ESTIMATED)**. Report the +1.7 correction and compare gains at both raw/corrected points. This is a stated
     sensitivity estimate, not a measured bound; corrected points outside the sweep cannot qualify a variant.
     SP1's kineto overhead remains in the spans.
   - A variant qualifies only if its in-range median gain exceeds **3 % AND twice prod-first/prod-last drift on both
     Sparks**, with a clean oracle on every case. Gain uses the faster production endpoint; the drift is endpoint
     spread divided by their mean. Use the worse raw/corrected gain and gain-minus-twice-drift medians.
   - Median isolated rate ≥215 GB/s (the LM head's low end) and no qualifying variant → **STOP** B1 smoe work for
     the measured range. This is an isolated-kernel decision, not proof of production bandwidth. Report incomplete
     or clamped correction coverage before any STOP conclusion.
   - A qualifying variant → serve A/B of that build flag only, in its own window, with the existing full-task and
     target-distribution gates. No new kernel code.
   - Median isolated rate <200 GB/s with a wave step near 16 runs (one smoe_gu wave = 144 blocks = 16 runs) → consider
     stage-B build 1: smoe grid shaping or occupancy. The B1 sweep reaches at most 25 runs, so it cannot establish a
     32-run step. Keep the k loop per column exact. A grid change still needs the serving gates.
3. **B4 dense.**
   - mrow32 at 16/20 rows below 180 GB/s while some variant reaches ≥ 1.2× → stage-B build 2: an mrow32 variant
     change. It must be bitwise equal to ring at the same plan, or pass the target-distribution gate.
   - Otherwise STOP.
4. **Fused launch.** Bitwise equal to `router_topk` + smoe and ≥ 3 % faster per layer at 3–6 rows → an engine
   lane that must pass the full-task and target-distribution gates. The routing floating-point chain changes from
   vLLM top-k to EXL3 top-k: equality in this bench does not prove equality to serving. The measured gain is against
   `router_topk` + smoe, a lower bound on gain against production's split (which has a separate vLLM top-k launch).
   `deepseek_v41` would need to call `fused_moe_forward` (today only `deepseek_v4/model.py:1034/1111` does).
   Hash-routed layers are excluded. The kernel author's 2026-10-11 log (`moe_method.py:53`) recorded only 1.02–1.03×
   at 4–6 rows (0.98× at one row), already near this threshold; these are prior results, not measurements of this window.
5. **16-row runs (B4 shared-expert re-read).** Only if NDM wants B4. First a B4 control, then a build.

## CPU tests for the harness

`offload/kl1/test_kl1.py` (`env -u TMUX TMUX_TMPDIR=<private> python3 -m pytest offload/kl1`) covers:

- the variant flag rules (PIPE even, OBI divides 40, only known flags);
- exact distinct counts per request and per case, deterministic in the seed;
- per-row distinct picks;
- exact layer bytes, including 8-row runs and shared-expert runs;
- monotone curve inversion, flat stretches and invalid-point refusals;
- single-window/rank and saved-stack selection, endpoint drift, both-Spark qualification, median inference and
  shared-expert prefetch sensitivity;
- dense identical-input CPU fake-kernel oracle and recorded per-group plans.

`offload/launch/windows/tests/test_kl1_window.py` runs the real KL1 adapter and Supervisor against fake commands and
owned local telemetry processes: full sequence, failed start/stop/stage, crash/hang timeout, low memory, STOP, deadline,
signals including a second signal after supervisor close, copy mismatch, variant failure, cleanup/restore failure and
mutations skipping cleanup or W8. No remote command, precheck or real GPU bench is exercised by these tests.
The Round 2 regression also runs the real generated launch/fence/reap shells with fake Docker/GPU tools on two local
fake nodes: remote-owned sessions paused before registration or container creation, and an active bench, all abort
before W8. Releasing delayed work at W8 cannot create a bench. A mutation forging launcher absence is caught at W8;
reused identities are never signalled, and a surviving group cannot produce an absence marker.

Use the lane brief’s isolated scratch environment and explicit miniconda interpreter for pytest and
`mypy --strict --explicit-package-bases` on changed Python. The broad baseline already has three unrelated failures,
recorded in `KL1B-RUNNER.report.md` outside git; the lane does not suppress them. The `gpu*` subcommands import torch and
`vllm_exl3_moe` only when run. They have not been run anywhere: no GB10, no image on Controller.
