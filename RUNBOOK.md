# DeepSeek-V4.1-Flash-1017 EXL3 2.75 bpw — TP2 on sparkhead + sparkworker

**Served model since 2026-10-21: legal v2a** (adopted by the user; see "Served model" below). The baseline is kept
unchanged and is one command away: `./stop.sh && MODEL_DIR=/srv/dsv41/model ./start.sh`.

OpenAI-compatible API: **http://HEAD_IP:8000/v1** (LAN). Also reachable at http://localhost:8000/v1 on sparkhead
and through Tailscale at TAILNET_IP:8000. Model name: `DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2`. No API key is configured: keep port 8000 on a private network.
Port 8000 accepts connections only from localhost, LAN_SUBNET/24, Tailscale, the fabric link and Docker networks
(`firewall.sh`). Port 25000, the two ranks' coordination port, accepts only localhost and the fabric (IPv6: ::1 only;
SPARKFIX1022, 2026-10-22). `start.sh` re-applies both rules at every start; `./firewall.sh remove` drops them.

## Daily use

| | command (run on sparkhead, in `~/dsv41-deploy`) |
|---|---|
| start (≈8 min until ready) | `./start.sh` |
| stop (only this deployment's containers) | `./stop.sh` |
| status / health | `./status.sh` |
| logs | `./logs.sh head 100` · `./logs.sh worker 100` · `./logs.sh head 50 -f` (follow) |
| benchmark | `./bench.py --ctx 0 --workload legal --reps 3` · `./bench.py --filler real --ctx 150000 --workload legal` |
| functional checks | `./functest.py --long 32000` |
| quality suite | `./quality.py --long 150000` |
| firewall | `./firewall.sh show` · `./firewall.sh remove` |

`start.sh` refuses to start if the model is incomplete on either node, the image is missing, or a GPU process is already
running (override that check with `FORCE=1`). It sets the GPU clock cap, applies the firewall rule, prebuilds or loads the
JIT kernels on both nodes, then starts rank 1 (sparkworker) and rank 0 (sparkhead) and waits for `/v1/models`.

**After a reboot** of either node: run `./stop.sh`, then `./start.sh`. The containers use `--restart no` on purpose: a
rank that restarts on its own can never rejoin its TP pair. The clock cap and the firewall rule are re-applied by `start.sh`.

**If one rank dies**, the other one hangs. `./stop.sh && ./start.sh` recovers both.

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
- **Limits:** max context 327,680 tokens; 4 concurrent sequences; KV pool 989,460 tokens (3 full-length requests).
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
  Both run at 200 Gb/s (4X HDR) and are ACTIVE. RoCE v2 GID index 3 on both nodes.
- **Port f0 is NOT the fabric.** Since 2026-10-22 each node's f0 (`enp1s0f0np0`, second function `enP2p1s0f0np0`) is a
  40 Gb/s direct link to x86host's ConnectX-5: sparkhead 10.202.28.2 ↔ x86host 10.202.28.1, sparkworker 10.202.29.2 ↔ x86host
  10.202.29.1 (verified 2026-10-22, SPARKFIX1022). The upstream recipe defaults are `enp1s0f0np0` / `rocep1s0f0`. While f0
  was unplugged a launch with those defaults stopped at "fabric not ready"; now it would pick the x86host links, which do not
  connect the two Sparks. Always start with `./start.sh`: it reads `dsv41.env` (FAB_IF / FAB_HCA = f1), and `serve.sh`
  pins the same f1 names.
- NCCL uses both PCIe functions (`NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`). The plugin's RDMA ring uses `rocep1s0f1`.
- **LAN:** sparkhead HEAD_IP (wired; local-law and every client use this) / .147 (Wi-Fi, fallback only);
  sparkworker LAN_IP (wired) / .5 (Wi-Fi).

## What is deployed (reproducibility)

| item | value |
|---|---|
| quantized weights | `HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark` @ `86f852533ec94d1a537fea41fd9c01d6b0f6eb50` |
| Engram shards | `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e1088993b9b82de08f59de5be5a34e42170`, `model-00047/00048-of-00048.safetensors` (101,535,150,936 / 101,537,926,640 bytes) |
| verification | `verify_hashes.py`: all 192 files sha256/size-identical to the pinned HF trees on **both** nodes (`manifest.json`) |
| image | `ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v5`, repo digest `sha256:8fc2c99a3562e99d81d2a05b8345d12726def0a027410d14d0eb0763de8309e4`, image id `sha256:435a5236…` (same on both nodes; sparkworker got it by `docker save \| docker load` over the fabric because ghcr pulls kept failing) |
| inside the image | vLLM `0.31.1.dev0+g6e517b15.dsv41gb10` (sudoingX recipe v1.1 + HeNryous patches 0022–0050), torch 2.13.0+cu130, CUDA 13.0.2 |
| deployment repo | `HeNryous/deepseek-v41-1017-dgx-spark` @ `b1726644b89fbb57b6468d33880cddac55a3c527` (`deepseek-v41-1017-dgx-spark/`) |
| launcher | `sudoingX/dgx-spark-deepseek-v41` @ `cc59c3609e8e671418f39eb92aa16b0e4f770b77` (`~/dgx-spark-deepseek-v41`) |
| serve config | `serve.sh` = the author's `serve/serve_example.sh` with 3 lines changed (`serve.sh.diff`): f1 fabric names, f1 HCAs, `MODEL_DIR=/srv/dsv41/model` (plus the DSV41-OFFLOAD hooks; since 2026-10-23 configuration D: KV 6.0 GiB/rank, drafter on x86host — see the DSV41-OFFLOAD section). Base v5 configuration: max-len 327,680, 4 seqs, batch 4,096, KV 3.75 GiB/rank, util 0.86, DSpark k=5 at 1–2 streams / k=4 at 3–4 streams, adaptive verify (0047), RDMA all-gather (0045), mrow head/dense (0042b/0046), mHC fused (0048/0050), priority scheduling (0031) |
| JIT kernel cache | `~/.cache/dsv41-serve/torch_extensions` on each node (built once; kept across starts) |
| GPU clock | capped at 2,150 MHz (`nvidia-smi -lgc 0,2150`, not persistent) for heat; see BENCHMARKS.md. Stock (no cap) boosts to ~2,450–2,490 MHz under load |

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `fabric not ready: …gid` at start | the fabric link or IP is down. Run `ip -br addr show enp1s0f1np1` and `ibdev2netdev` on both nodes; the link must be Up with 10.202.26.x |
| `scp: dest open "…/dsv41-serve.sh/tmp.…": Permission denied` at start | a directory sits where the generated launch file goes (a docker bind mount of a missing file creates a root-owned one; happened 2026-10-22 under /tmp). The file now lives in `~/dsv41-deploy/run/` on each node (`SERVE_DIR` in `dsv41.env`, read by a local patch to the recipe launcher — `recipe-servedir.patch` here, applied in `~/dgx-spark-deepseek-v41`, which stays at cc59c36 plus that diff; the launcher refuses with a clear message when a directory is in the way; unset, it uses /tmp as upstream does). Remove the empty directory (`sudo rmdir <path>`) and start again |
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

## DSV41-OFFLOAD: remote Engram from x86host (configuration B)

Since <promotion date>, both ranks read their Engram rows from x86host's RAM instead of their own NVMe: rank 0 (this node)
over its port f0 RDMA link 10.202.28.2 → x86host 10.202.28.1:18610, rank 1 over 10.202.29.2 → 10.202.29.1:18610. The rows
are byte-identical to the local shards (checked at boot against the table identity, and in testing row by row); the
local shards stay mounted and are read automatically for any lookup x86host cannot answer within 100 ms. Measured effect:
prompt reading ~5.5 % faster at 32K–150K, decode unchanged, no change in Spark memory (offload/results/COMPARISON.md).

| | command |
|---|---|
| switch | `OFFLOAD_DOCKER_ENV=...` line in `dsv41.env` (B); `start.sh` then stages the overlay and builds its JIT extension before the model loads |
| roll back to A | comment out that line, then `./stop.sh && ./start.sh` (the staged files under `~/.cache/dsv41-serve/dsv41-offload` are inert) |
| x86host side | `ssh x86host 'systemctl --user status dsv41-engram'` (unit enabled; loads + sha256-verifies both tables, ~70 s, 189 GiB) |
| is B really in use | `./logs.sh head 400 \| grep -E "remote rows from\|engram_remote"` on each node: per table every 60 s, lookups / mean µs / `fallbacks 0` |
| x86host down | the serve keeps answering with local rows (one ≤ 100 ms timeout per table, then local reads at A's speed); it re-attaches by itself within 5 s of x86host's server returning |

Code and evidence: `offload/` (engram/, transport/, results/, documents/plans/2026-10-22-x86host-offload-plan.md).

### Configuration D (remote DSpark drafter on x86host + 6 GiB KV), 2026-10-22
- What: `dsv41.env` OFFLOAD_DSPARK_ENV / OFFLOAD_DOCKER_ARGS / PROD_KV_BYTES / PROD_MAX_LEN. `start.sh` stages the overlay
  (`offload/dspark/spark/stage_overlay.sh`, sha-checked) into `~/dsv41-offload-prod/dspark` on both nodes; `serve.sh`
  uses `offload/launch/dsv41_tp2_offload.sh` (recipe launcher + EXTRA_DOCKER_ARGS) to bind-mount it.
- Needs: x86host `dsv41-engram` and `dsv41-dspark-worker` (systemd --user, both enabled). The worker runs from
  `~/dsv41-offload/prod/dspark-worker/current` (-> commit dir with SOURCE_COMMIT; unit from
  `offload/dspark/worker/dsv41-dspark-worker.prod.service`, GPU-c4e7a371, 10.202.28.1:18620, --max-model-len 1048576);
  status `~/dsv41-offload/prod/run/dspark-worker.status.json`, log `~/dsv41-offload/prod/logs/dspark-worker.log`.
  Worker down → the serve keeps answering with target-only rounds and reconnects within ~5 s of the worker being ready.
- Spark staging: `~/dsv41-offload-prod/pinned/libofl.so` on the head node is the pinned transport build (sha 4daef6c2…).
- Inspect: `docker logs dsv41-serve 2>&1 | grep -E "remote DSpark|GPU KV cache size"`; rank-0 stats
  `docker exec dsv41-serve cat /tmp/dsv41-remote-dspark.json` (rounds_ok, fallback_*, rtt_us, breaker).
- Rollback to B: comment out the four D lines in `dsv41.env`, `./stop.sh && ./start.sh` (KV back to 3.75 GiB, local
  drafter). Rollback to A: also comment out OFFLOAD_DOCKER_ENV.
