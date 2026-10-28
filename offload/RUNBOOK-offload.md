# DSV41-OFFLOAD runbook (x86host offloads for the two-Spark DeepSeek-V4.1-Flash-1017 serve)

Where: sparkhead `~/dsv41-deploy` = production checkout (`main`); `~/dsv41-deploy-offload` = experiment checkout
(`offload-x86host`); Controller `/opt/dsv41/repos/dsv41-deploy` = working clone (origin = sparkhead). Plan and lane briefs:
`documents/plans/`. Results: `offload/results/` (COMPARISON.md is the summary).

## The configurations

| | file | runs |
|---|---|---|
| A (production) | `dsv41.env` + `serve.sh` on `main` | `cd ~/dsv41-deploy && ./stop.sh && ./start.sh` |
| A via the experiment launcher | `offload/launch/configs/A-local.env` | port 8001, identical settings |
| A, speculation off (diagnostic) | `configs/A-target-only.env` | port 8001 |
| A + drafter capture (parity data) | `configs/A-capture.env` | port 8001; rank 0 writes `~/.cache/dsv41-serve/dspark-captures/*.pt` |
| B, remote Engram | `configs/B-remote-engram.env` | port 8001; needs x86host `dsv41-engram` up + overlay staged + JIT prebuilt |
| C, remote drafter | lane O3's `offload/dspark/` configs (C-shadow, C-capacity, C-resilience) | port 8001; needs x86host `dsv41-dspark-worker` |

## A maintenance window (only one coordinator at a time)

```bash
# on sparkhead, experiment checkout
cd ~/dsv41-deploy-offload && git pull --ff-only 2>/dev/null || true   # (Controller pushes offload-x86host here)
offload/launch/window.sh open "<purpose>"   # waits for an idle serve, writes ~/dsv41-deploy/HOLD (pauses local-law K24),
                                             # placeholder `vllm serve [DSV41-OFFLOAD hold]` on both nodes (NVO drama3 waiter)
offload/launch/start_exp.sh offload/launch/configs/<CONFIG>.env      # stops the serve, starts <CONFIG> on :8001
# ... measure (Controller): offload/bench/suite.sh <LABEL> http://HEAD_IP:8001 offload/results/<LABEL> [stages]
cd ~/dsv41-deploy && ./stop.sh && ./start.sh                         # production back on :8000 (tested rollback)
cd ~/dsv41-deploy-offload && offload/launch/window.sh close          # refuses unless :8000 serves /srv/dsv41/legal-v2a
```

Tell the local-law orchestrator session when a window opens and closes (it honours HOLD; a running K24 job finishes
unless drained).

## Remote Engram (B)

- x86host server: `ssh x86host '~/dsv41-offload/o1/src/offload/engram/server/engram_ctl.sh status|start|stop'` during
  experiments; for production the systemd --user unit `dsv41-engram.service` (see `offload/engram/INTEGRATION.md`).
  Loads + sha256-verifies both tables (~70 s), 189 GiB resident (THP), listens only on 10.202.28.1 / 10.202.29.1 :18610.
- Spark side: `offload/engram/spark/stage_overlay.sh spark sparkworker` (copies into `~/.cache/dsv41-serve/dsv41-offload`,
  checks sha256 of the copies and of the image's engram files), `offload/engram/spark/prebuild_jit.sh spark sparkworker`
  (serve stopped), then start with the B env (`VLLM_ENGRAM_REMOTE=rdma://10.202.28.1:18610@10.202.28.2,
  rdma://10.202.29.1:18610@10.202.29.2`, `VLLM_ENGRAM_REMOTE_REQUIRED=0` in production, `=1` for experiments).
- Inspect: per node `docker logs dsv41-serve 2>&1 | grep engram_remote` — one line per table per 60 s with lookups,
  rows, mean/p50/max µs, `fallbacks`, transport errors, refusals, reconnects. A benchmark run is valid only with 0 fallbacks.
- Exactness tools: `offload/engram/tests/spark_exact.sh <rank>` (on each Spark, under `idle_gate.sh`),
  `offload/engram/tests/logit_ab.py run|compare` (fixed-prefix outputs vs A).

## Benchmarks

`offload/bench/bench2.py` (one group), `offload/bench/suite.sh` (the screening suite: short, 32K × conc 1/2/3 + mixed,
warm 32K, 150K + mixed, warm 150K, full task; stages c256 and c300 for finalists), `offload/bench/summ.py` (quick view),
`offload/bench/report.py` (mean ± sd tables). Corpus: `/opt/dsv41/data/bench-data/` (= `data/*.jsonl` on the
Sparks, sha256 in `data/SHA256SUMS`). Every group checks the serve is idle first; speculation counters are server-global.

## Rollback, all levels

- Any experiment → production: `cd ~/dsv41-deploy && ./stop.sh && ./start.sh` (tested 3× on 2026-10-22; launch files and
  container env verified identical to the baseline each time).
- Production B → A: comment out `OFFLOAD_DOCKER_ENV` in `dsv41.env`, `./stop.sh && ./start.sh`.
- Repository: tag `baseline-20261022` is the deployment as found (git history before this campaign did not exist).
- x86host: stop `dsv41-engram` / `dsv41-dspark-worker` (frees 189 GiB RAM / the 3090); nothing on the Sparks depends on them
  when they are not configured.

## KV page packing + head-node memory (lane K1, `offload/kvpack/`)

- What: an overlay of 13 engine/plugin files (HeNryous afe4d5e patches 0051/0053/0057/0059-code/0060/0076 on image v5).
  It is inert until switched. With `VLLM_DSV41_KV_PAIR=3` D-kv6 holds 2,502,600 tokens at 327,680 (v5: 1,604,926) and
  3,154,934 at max-model-len 1,048,576 (v5: 1,850,243). The 0076 switches and indexer factor 4 free about 0.6–0.7 GiB
  per rank (upstream measured +714 / +813 MiB for the set). At 1M that about pays for the larger RoPE caches and
  indexer buffer (net ≈ today's free memory); without them 1M costs ~1.05 GiB: what makes 1M fit.
- Before any start that uses it: on sparkhead `offload/kvpack/stage_overlay.sh verify-checkout`; then
  `offload/kvpack/stage_overlay.sh check spark sparkworker`. After every start:
  `offload/kvpack/stage_overlay.sh verify-running --expect "<switches>" spark sparkworker`.
- Experiments: `offload/launch/configs/D-kv6-async-1m-k1mem.env` (1M + 0076b/d/e + factor 4), `D-kv6-async-pack.env`,
  `D-kv6-async-pack-1m.env`, on the async production candidate `D-kv6-async-process-nap-lazy.env` (the GB10 plan runs
  these); `D-kv6-1m-k1mem.env`, `D-kv6-pack.env`, `D-kv6-pack-1m.env` are the same on the sync D-cal1-kv6 (the control)
  (`D-kv6-1m.env` is the plain 1M run: stock v5 code + factor 4, no K1 overlay). Every item is a start-time
  switch: `K1_KV_PAIR`, `K1_DROP_MTP`, `K1_TF_TRANSIENT`, `K1_TF_ALIAS`, `K1_INDEXER_FACTOR`, `K1_MEMPROBE`, `K1_NODROP`.
  Empty means off, e.g. `K1_TF_ALIAS= offload/launch/start_exp.sh offload/launch/configs/D-kv6-async-pack.env`. 1M configs
  need the x86host worker started with `--max-model-len 1048576`.
- Test order and pass criteria: `offload/kvpack/GB10-TEST-PLAN.md` (R0 inertness first; stop if MemAvailable < 1.5 GiB).
- Production: `offload/kvpack/stage_overlay.sh stage spark sparkworker`; the launcher's docker arguments get
  `$(offload/kvpack/stage_overlay.sh args)` and `OFFLOAD_DOCKER_ENV` the switches. Rollback: drop both, then
  `./stop.sh && ./start.sh`.
- Integrity: `offload/kvpack/build_overlay.sh` (overlay == rebuild from v5 + pinned donor) and
  `offload/kvpack/test_stage_overlay.sh` (27 offline tests).
