# DeepSeek-V4.1-Flash-1017 — production operations on the Sparks and x86host

Status 2026-10-23: last recorded production is **main 3c1dc6b / prod-async-20261023**: legal-v2a target on both Sparks, remote Engram, one-card x86host capacity drafter and the supervised **async process proxy with nap**. Source: [Window 5 production switch](offload/results/COMPARISON.md#production-switch-async-on-the-one-card-worker-13491358z). DOC1 did not contact the hosts; verify the saved production receipt before acting. This worktree carries experiment/development configuration and is not the production checkout.

**Tonight:** TP2 K2 x86host worker promotion retains async; K1 packing at 524,288 is a separate restart after final S18 and S22 checks. Until those receipts exist, use the recorded one-card/327,680 state. The [documentation index](documents/README.md) maps components and decisions; the [one live roadmap](documents/plans/roadmap.md) orders the gates. Preserve legal-v2a throughout routine runtime rollback; changing back to stock model weights is a separate choice.

OpenAI-compatible API: **http://HEAD_IP:8000/v1** (LAN). Also reachable at http://localhost:8000/v1 on sparkhead
and through Tailscale at TAILNET_IP:8000. Model name: `DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2`. No API key is configured: keep port 8000 on a private network.
Port 8000 accepts connections only from localhost, LAN_SUBNET/24, Tailscale, the fabric link and Docker networks
(`firewall.sh`).

## Daily use

| | command (run on sparkhead, in `~/dsv41-deploy`) |
|---|---|
| start (W5 async: ≈6 min plus warm-up) | `./start.sh` from the verified production checkout |
| stop (only this deployment's containers; coordinator window) | `./stop.sh` |
| status / health | `./status.sh` |
| logs | `./logs.sh head 100` · `./logs.sh worker 100` · `./logs.sh head 50 -f` (follow) |
| benchmark | `./bench.py --ctx 0 --workload legal --reps 3` · `./bench.py --filler real --ctx 150000 --workload legal` |
| functional checks | `./functest.py --long 32000` |
| quality suite | `./quality.py --long 150000` |
| firewall | `./firewall.sh show` · `./firewall.sh remove` |

`start.sh` refuses to start if the model is incomplete on either node, the image is missing, or a GPU process is already
running (override that check with `FORCE=1`). It sets the GPU clock cap, applies the firewall rule, prebuilds or loads the
JIT kernels on both nodes, then starts rank 1 (sparkworker) and rank 0 (sparkhead) and waits for `/v1/models`.

**After a reboot** of either Spark: restore/check x86host services for the recorded profile, then run `./stop.sh`, then `./start.sh` from the verified production checkout. The containers use `--restart no` on purpose: a
rank that restarts on its own can never rejoin its TP pair. The clock cap and the firewall rule are re-applied by `start.sh`.

**If one Spark rank dies**, the other can hang. In the coordinator's drained window, `./stop.sh && ./start.sh` recovers both. A x86host drafter outage is different: capacity mode uses target-only rounds and reconnects; a supervised proxy loss has its own bounded respawn budget. Use the component checks below before deciding a Spark restart is needed.

## Current offload operation

These commands are instructions for the coordinator on the named host; this documentation lane executed none of them. Use [Engram integration](offload/engram/INTEGRATION.md), [remote integration](offload/dspark/INTEGRATION.md) and the [async/TP2 promotion runbook](offload/promote/async-tp2/README.md) for their full checks and failure policy.

| Host / action | Command or record |
|---|---|
| sparkhead: verify production revision | `cd ~/dsv41-deploy && git log -1 --oneline` and inspect saved `dsv41.env`; :8000 `/v1/models` must serve `/srv/dsv41/legal-v2a` |
| x86host: Engram status / start | `systemctl --user status dsv41-engram.service` / `systemctl --user start dsv41-engram.service`; allow verified table load before starting dependent tests |
| x86host: currently recorded one-card worker | `systemctl --user status dsv41-dspark-worker.service`; start with `systemctl --user start dsv41-dspark-worker.service` when this is the selected production profile |
| x86host: after TP2 promotion | `systemctl --user status dsv41-dspark-worker-tp2-prod.service`; the TP2 unit must be enabled+active and the one-card unit disabled+inactive outside a window |
| x86host: stop the selected worker, only in an owned window | `systemctl --user stop <selected-unit>`; Sparks run target-only until a worker reconnects. Use the window helpers for routine swaps rather than manually releasing locks |
| sparkhead: async startup | `docker logs dsv41-serve` must show async self-test/mlock success, proxy process running, and connection through that proxy; no async-off/start-failed line |
| sparkhead: async stats | `docker exec dsv41-serve cat /tmp/dsv41-remote-dspark.json`; read only after a fresh 1,024-round stats line from the current proxy pid, especially after a respawn |
| x86host: TP2 production-hold check after promotion | `bash ~/dsv41-offload/prod_window.sh check` → `PRODUCTION HOLD OK` |

The selected async settings are process mode, spin 50 µs / nap 20 µs, lazy CUDA module loading, three respawns with a 3,600 s budget reset. A short startup warm-up absorbs the short-prompt penalty. W5 still measured about 1.7 s extra on the first 32K prefill after restart; do not promise that every long-prefill module is warmed. The stats file may be absent before the first 1,024 rounds, or stale at loss until the new proxy emits its next mark. Match proxy pid and round count as described in the [promotion stats rule](offload/promote/async-tp2/README.md#window-5-gb10-supervised-before-promotion).

## Maintenance windows and GPU ownership

Only coordinator Deepseek / coordinator integrates, restarts or benchmarks the Sparks. [window.sh](offload/launch/window.sh) owns the Spark maintenance window: on sparkhead's reviewed experiment checkout, `offload/launch/window.sh open "<purpose>"`, perform the reviewed experiment, restore the recorded production checkout on :8000, then `offload/launch/window.sh close`. Open drains running/waiting requests, writes `~/dsv41-deploy/HOLD` and holds both nodes against the narration waiter; close refuses unless :8000 serves legal-v2a. Pause D2 generation before a later window. Preserve the coordinator's BENCH_ACTIVE/telemetry protocol. Never update the checkout while a runner uses it.

Before TP2 promotion, pair windows use [pair_window.sh](offload/launch/pair_window.sh) and its retained coordinator hold. After promotion, the production worker itself holds both GPU locks through `gpu_locked.sh`; use [prod_window.sh](offload/launch/prod_window.sh) on x86host:

| Action after TP2 promotion | Effect / completion check |
|---|---|
| `bash ~/dsv41-offload/prod_window.sh status` | Units, markers, lock holders and queued waiters |
| `bash ~/dsv41-offload/prod_window.sh pair COORD` | Both production draft cards assigned to the named pair window; Spark target stays up and runs target-only |
| `bash ~/dsv41-offload/prod_window.sh train <LANE>` | Optional fallback to one-card production so the named owner can use 7744b44b; normal D2 work now uses the solo cards |
| `bash ~/dsv41-offload/prod_window.sh close` | Removes markers, waits for lane locks, restores TP2 and ends with `PRODUCTION HOLD OK` |

After promotion `pair_window.sh` refuses. Lanes use nonblocking locks only, stop/checkpoint when their marker is withdrawn, and never replace a lock file. If TP2 fails for a p2p/venv/model cause, restore the one-card unit as the promotion runbook directs; do not keep production target-only while debugging. Detailed ownership and checks: [dual-3090 plan](documents/plans/2026-10-23-dual3090-plan.md), [promotion rollback](offload/promote/async-tp2/README.md#rollback-under-10-min-x86host-and-the-sparks-in-parallel).

## Logs and evidence

| Location | What to inspect |
|---|---|
| Sparks `./logs.sh head 100` / `./logs.sh worker 100` in production checkout | Engine startup, advertised maximum, KV pool, Engram fallback/errors, async proxy and watchdog/respawn lines |
| Spark rank 0 `/tmp/dsv41-remote-dspark.json` inside `dsv41-serve` | Aggregate current-pid remote/proxy state after a fresh stats mark |
| x86host `~/dsv41-offload/prod/logs/dspark-worker.log` and `~/dsv41-offload/prod/run/dspark-worker.status.json` | Production worker readiness, HELLO session, arm/identity, `tp: 2` after TP2 promotion, errors |
| x86host `journalctl --user -u dsv41-engram.service` and selected drafter unit | Service lifecycle; see component guides for configured file logs |
| `offload/results/<run>/` | Benchmark JSONL, startup/stats/telemetry aggregates; [COMPARISON](offload/results/COMPARISON.md) owns measured conclusions |
| Controller `/opt/dsv41/campaigns/DSV41-OFFLOAD/` | `LANES.md`, lane progress/reports, W5/W6/W6b runner logs and promotion/review receipts; keep outside git |

## Runtime rollback

Open/retain the coordinator window, pause generation and restore the exact saved configuration. Tags refer to Spark checkout/runtime state; x86host worker state is a separate rollback step. Full commands, lock transitions and health checks are in the [promotion runbook](offload/promote/async-tp2/README.md#rollback-under-10-min-x86host-and-the-sparks-in-parallel).

| Rollback target | Meaning / procedure |
|---|---|
| `prod-async-20261023` (main 3c1dc6b at W5) | Known async Spark runtime with legal-v2a and 6 GiB KV/rank; one-card worker was the recorded W5 partner. Restore tag files in the production checkout and restart both Spark ranks inside the window; restore the matching saved x86host worker state separately |
| `prod-d-20261023` (6aa71ca) | Exact pre-async D runtime: remote Engram + one-card capacity drafter, synchronous rounds. Promotion runbook gives the tested `git checkout prod-d-20261023` stop/start and x86host current→d275f6b restoration; follow with a normal revert commit rather than resetting a shared branch |
| TP2 x86host worker → one card | Disable/stop `dsv41-dspark-worker-tp2-prod`, restore recorded `current` and enable/restart `dsv41-dspark-worker`; confirm new ready+HELLO and usable accepted drafts. Tag checkout alone does not switch worker units |
| K1 promotion → previous profile | Use the separately reviewed K1 package's restore receipt to restore exact saved pre-K1 files and limits, then verify both-rank runtime bytes/KV/API. Do not infer K1 promotion from this roadmap |

W5 measured the x86host pair→one-card rollback at 12.2 s and the async Spark start at about 363 s; the package's under-10-minute full rollback expectation is bounded by the Spark restart, not an unconditional service guarantee. After rollback verify :8000/legal-v2a, expected KV/max length, selected worker ready/session, accepted-draft counters and the appropriate lock/unit state; close the Spark window only after health passes.

## Requests

```bash
curl http://HEAD_IP:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2", "temperature": 1.0, "top_p": 1.0,
  "chat_template_kwargs": {"thinking": false},
  "messages": [{"role": "user", "content": "..."}]}'
```

- **Thinking** is on by default. Control it only through `chat_template_kwargs`: `{"thinking": false}` or
  `{"reasoning_effort": "none"|"low"|"high"|"xhigh"|"max"}`. All of these were verified on this server. A top-level
  `reasoning_effort` is ignored.
- **Tool calling** (`tools`, `tool_choice: auto`) and `response_format: {"type": "json_object"}` both work (verified).
- **Background jobs:** send `"priority": 5` (or header `X-Vllm-Priority: 5`). Their prefill then yields to interactive
  requests (patch 0031).
- **Untrusted text:** literal DeepSeek special tokens in user text (`<｜…｜>`) become real special tokens. Insert a
  zero-width space after `<` before sending, for example for quoted e-mails or OCR text.
- **Recorded limits (W5 production):** max context 327,680 tokens; 4 concurrent sequences; KV pool **1,604,926 tokens** at 6 GiB/rank. The original A pool was 989,460 at 3.75 GiB/rank. Pending K1 candidate: max 524,288 and expected pool 2,820,756, subject to final promotion checks. Model maximum, pool capacity and successful occupied-context tests are different quantities.
- **Cache:** the prefix cache is on. Repeated long prompts prefill almost instantly the second time. Add
  `"cache_salt": "<random>"` to force a cold prefill.

## Served model

| | legal v2a (default, adopted 2026-10-21) | baseline (rollback) |
|---|---|---|
| path (same on both nodes) | `/srv/dsv41/legal-v2a` | `/srv/dsv41/model` |
| head | symlink → `~/models/DeepSeek-V4.1-Flash-1017-EXL3-legal-v2a` (hardlink tree of `~/dsv41-legal/models/legal-candidate-v2a`) | symlink → `~/models/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark` |
| worker | hardlink tree (baseline files + 32 re-encoded layer parts), local NVMe | real dir, local NVMe |
| what differs | 1,414 routed experts in 16 layers (2-10, 12, 14-18, 25) moved bits toward legal-leaning experts and were re-encoded on the baseline's own calibration (sudoingX cal_v3). Same total bytes; everything else byte-identical | — |
| KLD vs native 1017 | legal 0.0249 (−9.0%), general 0.0204 (+1.5%; math +7.6%, extraction/preservation +3-4%) | legal 0.0274, general 0.0201 |
| plan sha256 | `ebf315ef…` (`exl3_moe_plan.json`, identical on both nodes) | |
| publishing | **local only, never publish or upload** (plan derives from the user's local grpo-builder opinions; `CANDIDATE_MANIFEST.json`) | public HF weights |

Served under the same model name, so clients need no change; `/v1/models` → `root` shows which directory is loaded.
Details: `~/dsv41-legal/reports/` and `~/dsv41-legal/STATUS.md`.

## Machines and software (verified 2026-10-20)

| | sparkhead (head, rank 0) | sparkworker / sparkworker (worker, rank 1) |
|---|---|---|
| hardware | DGX Spark GB10, 128 GB unified (121 GiB visible), aarch64 | same |
| OS / kernel | Ubuntu 24.04.4, 6.17.0-1026-nvidia | same |
| driver | 580.159.03 (CUDA 13.0) | same |
| Docker / NVIDIA CTK | 29.2.1 / 1.19.1 | 29.2.1 |
| model dir | `/srv/dsv41/legal-v2a` (served) and `/srv/dsv41/model` (baseline), see "Served model" | same paths, local NVMe |
| disk | Samsung 4 TB NVMe | 1 TB NVMe (+ Crucial X9 USB, not used by the model) |
| ssh alias | `spark` (127.0.0.1) | `sparkworker` → 10.202.26.1, user rank1user |

**Important:** on sparkworker, `~/models` and `/home/rank0user` are NFS mounts of sparkhead. The model must not be
read through them. That is why the node-neutral path `/srv/dsv41/model` exists.

## Network

- **Fabric (TP2 traffic):** ConnectX-7 port **f1** on both nodes. `enp1s0f1np1` / `rocep1s0f1` is
  10.202.26.2 (head) ↔ 10.202.26.1 (worker). The second PCIe function, `enP2p1s0f1np1` / `roceP2p1s0f1`, is 10.202.27.x.
  Both f1 PCIe functions run at 200 Gb/s (4X HDR). RoCE v2 GID index 3. **The old Oct 20 f0-down statement is superseded:** f0 now carries each Spark's direct 40 Gb/s x86host link, separate from target TP traffic (Oct 22 inventory).
- NCCL uses both f1 PCIe functions (`NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`). The baseline plugin ring uses `rocep1s0f1`; UP1's dual-ring arm remains a separately gated experiment.
- **Offload links:** x86host 10.202.28.1 ↔ Spark A 10.202.28.2; x86host 10.202.29.1 ↔ Spark B 10.202.29.2, 40 Gb/s each, MTU 9000, RoCE v2 GID 3. Engram uses each rank's link; DSpark uses rank 0's link with TP broadcast. [Transport API](offload/transport/API.md).
- **LAN:** sparkhead HEAD_IP (wired) / .147 (Wi-Fi); sparkworker LAN_IP (wired).

## Image provenance and original baseline (reproducibility)

The table retains image-v5 and original launcher provenance. Current production adds remote Engram/capacity drafting and the W5 async package as recorded above; consult the exact production revision/env and component manifests rather than treating the original serve-example settings as current.

| item | value |
|---|---|
| quantized weights | `HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark` @ `86f852533ec94d1a537fea41fd9c01d6b0f6eb50` |
| Engram shards | `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e1088993b9b82de08f59de5be5a34e42170`, `model-00047/00048-of-00048.safetensors` (101,535,150,936 / 101,537,926,640 bytes) |
| verification | `verify_hashes.py`: all 192 files sha256/size-identical to the pinned HF trees on **both** nodes (`manifest.json`) |
| image | `ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v5`, repo digest `sha256:8fc2c99a3562e99d81d2a05b8345d12726def0a027410d14d0eb0763de8309e4`, image id `sha256:435a5236…` (same on both nodes; sparkworker got it by `docker save \| docker load` over the fabric because ghcr pulls kept failing) |
| inside the image | vLLM `0.31.1.dev0+g6e517b15.dsv41gb10` (sudoingX recipe v1.1 + HeNryous patches 0022–0050), torch 2.13.0+cu130, CUDA 13.0.2 |
| deployment repo | `HeNryous/deepseek-v41-1017-dgx-spark` @ `b1726644b89fbb57b6468d33880cddac55a3c527` (`deepseek-v41-1017-dgx-spark/`) |
| launcher | `sudoingX/dgx-spark-deepseek-v41` @ `cc59c3609e8e671418f39eb92aa16b0e4f770b77` (`~/dgx-spark-deepseek-v41`) |
| serve config | `serve.sh` = the author's `serve/serve_example.sh` with 3 lines changed (`serve.sh.diff`): f1 fabric names, f1 HCAs, `MODEL_DIR=/srv/dsv41/model`. Everything else is the tested v5 configuration: max-len 327,680, 4 seqs, batch 4,096, KV 3.75 GiB/rank, util 0.86, DSpark k=5 at 1–2 streams / k=4 at 3–4 streams, adaptive verify (0047), RDMA all-gather (0045), mrow head/dense (0042b/0046), mHC fused (0048/0050), priority scheduling (0031) |
| JIT kernel cache | `~/.cache/dsv41-serve/torch_extensions` on each node (built once; kept across starts) |
| GPU clock | capped at 2,150 MHz (`nvidia-smi -lgc 0,2150`, not persistent) for heat; see BENCHMARKS.md. Stock (no cap) boosts to ~2,450–2,490 MHz under load |

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `fabric not ready: …gid` at start | the fabric link or IP is down. Run `ip -br addr show enp1s0f1np1` and `ibdev2netdev` on both nodes; the link must be Up with 10.202.26.x |
| `/srv/dsv41/model missing on sparkworker` | the worker dir was moved or wiped. Re-copy it: `./sync_worker.sh`, then `scp verify_hashes.py sparkworker:/tmp/ && ssh sparkworker python3 /tmp/verify_hashes.py /srv/dsv41/model` |
| a container exits during load with an OOM error | something else holds memory (GB10 CUDA cannot reclaim page cache or other processes' RSS). Check `free -g` on both nodes, stop big copies/downloads/other GPU jobs, `sync; echo 3 \| sudo tee /proc/sys/vm/drop_caches`, then restart. Steady state leaves ~5–7 GiB available per node; long prefills dip by ~0.5 GiB |
| start hangs at the JIT step / a stale lock | a previous JIT build was killed. Remove stale `*.lock` files under `~/.cache/dsv41-serve/torch_extensions` on that node and start again. Never build JIT next to a loaded model |
| `image missing on sparkworker` | `docker save ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v5 \| ssh sparkworker docker load` (≈4 min over the fabric) |
| decode suddenly slower | check `./status.sh`: another client running (`num_requests_running`), GPU temperature (~80 °C+), clock. With 2–4 streams, aggregate throughput goes up but per-stream speed goes down |
| first token slow on long prompts | expected: ~2,000–2,300 tok/s prefill (150K ≈ 73 s, 310K ≈ 161 s). Repeats of the same prefix hit the prefix cache |
| GPU >80 °C | lower the cap: `GPU_CLOCK_CAP=2000 ./start.sh`, or live: `ssh <node> sudo nvidia-smi -lgc 0,2000` on both nodes |
| server reachable from a new subnet is needed | add the subnet to `ALLOW` in `firewall.sh` and run `./firewall.sh apply` |

## Do not

- Don't copy, hash or download large files on a node while it serves, unless you evict them from the page cache
  (`dd iflag=direct`, or `~/dsv41-legal/scripts/evict_cache.py`). MemAvailable drops and the serve can run out of memory.
- Don't start other GPU workloads on either node while it serves.
- Don't move or edit `/srv/dsv41/model` or `/srv/dsv41/legal-v2a` on either node. They share inodes (hardlinks): never
  write into a model file in place.
- Don't expose port 8000 to the internet (no router port forward). The API has no authentication.

## Other workspaces

- `~/dsv41-legal/` holds the legal-domain requant (phase 2). Pause it with `touch ~/dsv41-legal/PAUSE` and resume with
  `~/dsv41-legal/resume.sh`. Its weight fetch (`scripts/fetch_supervisor.sh`) pauses itself when memory is low.
