Status 2026-10-23: Exact remote Engram in production since D; measured cold-prefill benefit and exactness in COMPARISON. Original integration steps retain history. See [live roadmap](../../documents/plans/roadmap.md) and [measured record](../results/COMPARISON.md); campaign review decisions are in `LANES.md` / lane reports outside the repo.

# Remote Engram (configuration B) — integration, in-engine test, rollback

Lane O1, DSV41-OFFLOAD, 2026-10-22. For the coordinator (coordinator), who alone restarts and benchmarks the serve.

## What B changes

Today (A) each Spark rank reads its Engram rows per lookup from the local NVMe shards (`EngramDiskTable`, spin mode,
O_DIRECT AIO / 256-thread pread pool). In B each rank reads the same rows from the x86host table server, which holds both
tables resident in RAM, over its own direct link with RDMA. The engine's code path is the same from the hash ids to
the output. The same pinned staging layout, the same Triton lookup kernel, the same TP all-gather and the same
CUDA-graph structure are used. Only the row reader behind `enqueue_read` changes. The image's own `EngramDiskTable`
stays inside B as the exact fallback: on a timeout or error, rows come from the local shard, never zeros.
B frees no Spark memory: the tables were never resident on the Sparks, and B replaces the 2 × 13 MB pinned staging per
rank with 2 × 13 MB NIC-registered staging. Any gain is lookup latency.

| piece | where | what |
|---|---|---|
| table server | x86host, tmux `dsv41-engram` (or systemd --user unit, not enabled), `~/dsv41-offload/o1/bin/engram_server` | both tables resident (188.8 GiB RSS, all THP), sha256-verified against `manifest.json` at load, listeners `10.202.28.1:18610` (Spark A / rank 0) and `10.202.29.1:18610` (Spark B / rank 1), TCP + RDMA |
| overlay | `offload/engram/spark/engram.py` | image `common/engram.py` (sha256 `bf95ad9a…05e5`) + 8 small changes, each marked `DSV41-OFFLOAD` (import sys, `_IMAGE_DIR` for the image's engram_disk.cpp, the remote helpers, the constructor branch, the staging branch); without `VLLM_ENGRAM_REMOTE` it behaves as the image's (tested: A = C bit-identical) |
| remote table | `offload/engram/spark/engram_remote.cpp` (JIT, like `engram_disk.cpp`) | `EngramRemoteTable`, EngramDiskTable's interface; compiles the image's `engram_disk.cpp` in as the exact fallback |
| loader | `offload/engram/spark/py/sitecustomize.py` | with `DSV41_ENGRAM_OVERLAY` set, loads the overlay in place of the image module in every container process; no launcher edit, no extra bind mount |
| transport | `offload/transport/ofl.{h,cpp}` | see `offload/transport/API.md` |

## Proven so far (where, how)

- Every row is byte-exact in four places:
  - over both links and both backends, from x86host, against a local copy with all corner cases (synthetic tables);
  - against the real resident tables on x86host;
  - from both Sparks against their own `/srv/dsv41/legal-v2a` shards (`tests/spark_exact.sh`, results in
    `offload/results/o1/spark_exact_*`; see O1.report.md for the run status);
  - in the engine's own `ParallelEngramEmbedding`, where the overlay with remote rows gives bf16 outputs bit-identical
    to the image's disk path, for layers 1 and 14, TP ranks 0 and 1 (`tests/test_overlay_module.py`).
- Spin-mode stream memops, stream ordering, CUDA graph replay and failure injection (frozen, killed and restarted
  server) are proven **on an RTX 3090 (x86_64, x86host GPU-240876ad), not on GB10.** The GB10 proof is the first boot
  below. Two things differ on GB10, and neither is new to this container:
  - The NIC registration of `cudaHostRegister`ed anonymous memory is exactly how the production RDMA ring
    (vllm_exl3_moe `pinned()`) allocates its buffers.
  - The Triton lookup kernel is the image's unchanged kernel. On SM86 the module test had to emulate its sorted path,
    because Triton has no fp8e4nv on Ampere.

**Update 2026-10-22 ~20:00Z (coordinator, GB10):**
- `spark_exact.sh` passed on both Spark ranks: 839,040 lookup entries per rank (corrected 2026-10-23 from the raw logs; earlier 838,880 was wrong), 0 bad, 12/12 refusals (logs in
  `offload/results/o1/spark_exact_rank{0,1}_20261022.log`; count reconciliation in [COMPARISON Exactness](../results/COMPARISON.md#exactness-b)).
- B boots on GB10 (:8001), with both ranks on rocep1s0f0 GID 3, the right identities, 0 fallbacks and a mean lookup of
  50–60 µs.
- `logit_ab.py` puts B vs A inside the A-vs-A floor. The short prompt is identical; at 8K and 32K both A-A and A-B
  differ, because the engine is nondeterministic at long context under greedy decoding.

## Preconditions (before the window)

```bash
# x86host: server up, both tables verified, both listeners
ssh x86host '~/dsv41-offload/o1/src/offload/engram/server/engram_ctl.sh status'
#   ready=True ... verified=True (x2), listening 10.202.28.1:18610 and 10.202.29.1:18610
# Sparks: link GIDs unchanged (rocep1s0f0, RoCE v2 GID 3 = 10.202.28.2 / 10.202.29.2, MTU 4096)
ssh spark /tmp/o1/bin/ofl_probe | grep -A5 'rocep1s0f0 '; ssh sparkworker /tmp/o1/bin/ofl_probe | grep -A5 'rocep1s0f0 '
```

## 1. Stage the overlay on both nodes (serve may keep running; A never reads these files)

```bash
cd /opt/dsv41/repos/dsv41-deploy            # the integrated offload-x86host checkout
offload/engram/spark/stage_overlay.sh spark sparkworker
#   -> ~/.cache/dsv41-serve/dsv41-offload/{manifest.json, offload/transport/ofl.*, offload/engram/...}
#      = /root/.cache/dsv41-offload/... in the container; checks sha256 of the copies and of the image's engram files
```

## 2. JIT prebuild (serve stopped, both nodes; as serve.sh does for the plugin's kernels)

The build takes about 1 min and needs about 0.3 GB. Do it before the model loads (building next to the weights is
what serve.sh avoids). The module name hashes the sources, so later boots only load it.

```bash
for n in spark sparkworker; do ssh $n 'docker run --rm --gpus all --device /dev/infiniband -v $HOME/.cache/dsv41-serve:/root/.cache \
  -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions -e MAX_JOBS=4 \
  -e PYTHONPATH=/root/.cache/dsv41-offload/offload/engram/spark/py \
  -e DSV41_ENGRAM_OVERLAY=/root/.cache/dsv41-offload/offload/engram/spark/engram.py \
  -e VLLM_ENGRAM_REMOTE_ROOT=/root/.cache/dsv41-offload \
  --entrypoint python3 ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v5 -c "
import vllm.models.deepseek_v41.common.engram as e
print(\"overlay:\", e.__file__); print(\"ext:\", e._load_engram_remote_ext().__file__)"' & done; wait
```
Expected output: `overlay: /root/.cache/dsv41-offload/offload/engram/spark/engram.py` and
`ext: /root/.cache/torch_extensions/vllm_engram_remote_<hash>/...so`.

## 3. Start B (experimental port 8001 or the production port; everything else exactly as A)

Add to `DOCKER_ENV` in `serve.sh` (or export before `start.sh`). The values contain no spaces, because the launcher
splits `DOCKER_ENV` on whitespace:

```
PYTHONPATH=/root/.cache/dsv41-offload/offload/engram/spark/py
DSV41_ENGRAM_OVERLAY=/root/.cache/dsv41-offload/offload/engram/spark/engram.py
VLLM_ENGRAM_REMOTE_ROOT=/root/.cache/dsv41-offload
VLLM_ENGRAM_REMOTE=rdma://10.202.28.1:18610@10.202.28.2,rdma://10.202.29.1:18610@10.202.29.2
VLLM_ENGRAM_REMOTE_TIMEOUT_MS=100 VLLM_ENGRAM_REMOTE_RETRY_MS=5000 VLLM_ENGRAM_REMOTE_REQUIRED=1 VLLM_ENGRAM_REMOTE_LOG_S=60
# optional: VLLM_ENGRAM_REMOTE_BOUNCE=1 (CPU copy into staging; see "If B's outputs differ")
```

- `VLLM_ENGRAM_REMOTE` is indexed by the rank's Engram head shard (= TP rank): entry 0 is Spark A's link, entry 1
  Spark B's. The same env goes to both containers.
- `--engram-config` (disk_offload, disk_threads 256), `VLLM_ENGRAM_DISK_SPIN_MS=1000` and all other knobs stay as in
  A. B needs disk_offload: the local shard is the fallback.
- `VLLM_ENGRAM_REMOTE_REQUIRED=1` makes the boot fail loudly if x86host is unreachable or refuses the table identity.
  With `0` it boots on the local reader and attaches when x86host appears.
- `VLLM_ENGRAM_REMOTE=tcp://...` selects the TCP backend (debug; about 4x slower per lookup).

## 4. Check that B is really B

```bash
ssh spark 'docker logs dsv41-serve 2>&1 | grep -E "engram_remote|Engram table rows from the remote|Engram remote staging"'
ssh sparkworker 'docker logs dsv41-serve 2>&1 | grep -E "engram_remote|Engram table rows from the remote|Engram remote staging"'
#   per rank, for each of the 2 tables: "remote rows from rdma://... (rdma rocep1s0f0 port 1 gid 3 mtu 4096 ...
#   layer 1|14 rows [start, end) ...)" and "Engram remote staging (49152 rows)"; identities
#   layer 1 6ef189d0323b24cd..., layer 14 d463615d5e5fe68b...
ssh x86host '~/dsv41-offload/o1/src/offload/engram/server/engram_ctl.sh status'     # opens=4, lookups growing
```
Every `VLLM_ENGRAM_REMOTE_LOG_S` seconds of traffic, each table logs one line with: remote lookups, rows, MB,
mean/p50/max latency, fallbacks, transport errors, refusals and reconnects. **Fallbacks must be 0 in a benchmark run.**
A non-zero count means that run measured a mix of A and B.

## 5. Fixed-prefix outputs, A vs B (before any speed numbers)

```bash
T=offload/engram/tests/logit_ab.py; D=data/opinions_cl.jsonl     # on sparkhead (data/ is there)
# with A up on the test port:
$T run --url http://HEAD_IP:8001 --label A1 --out logs/ab_A1.jsonl --data $D
$T run --url http://HEAD_IP:8001 --label A2 --out logs/ab_A2.jsonl --data $D
$T compare logs/ab_A1.jsonl logs/ab_A2.jsonl        # noise floor of A against itself
# restart with the B env, then:
$T run --url http://HEAD_IP:8001 --label B1 --out logs/ab_B1.jsonl --data $D
$T compare logs/ab_A1.jsonl logs/ab_B1.jsonl        # must look like the A-A floor (identical tokens expected)
```
The test runs greedy with thinking off, a fresh `cache_salt` (so every prompt position goes through the lookups) and
priority 10. It uses three prompts: short, ~8K and ~32K tokens of public opinions. It records the top-20 logprobs per
generated token and, if the serve accepts it, `prompt_logprobs`. B's Engram rows are byte-identical to A's, so any
difference beyond the A-A floor is a bug to stop on, not noise.

## 6. Then the B vs A benchmarks (bench2.py, coordinator)

Expected effect: lookup latency per table per rank goes from local disk to about 0.03 ms (decode, 288 rows) and about
3 ms (prefill chunk, 49,152 rows) on RDMA; measured numbers are in O1.report.md. Layer 1's lookup has only layer 0's
compute to hide behind, so this is where a prefill gain, if any, comes from.

## If B's outputs differ from A's on GB10 (things that only GB10 can show)

- **NIC writes into GPU-read staging.** In A the CPU writes the staging rows; in B the NIC does (RDMA WRITE straight into
  the registered staging), and the GPU reads them through UVA after the proxy sets the done flag. GB10 should be
  hardware-coherent for device DMA, and the plugin's RDMA ring relies on the same property. If logit_ab shows a
  difference anyway, restart with `VLLM_ENGRAM_REMOTE_BOUNCE=1`: the NIC then writes into a separate registered buffer
  and the proxy copies the rows into staging with the CPU, exactly as the disk reader does. The cost is about +0.15 ms
  at 4,096 rows on x86host (memcpy) and nothing measurable at decode sizes. The mode string in the log says
  `bounce copy` or `direct placement`.
  `VLLM_ENGRAM_REMOTE=tcp://...` is the other CPU-written path; it is slower.
- **Fork after registration.** The Spark kernels are ≥ 5.13 (copy-on-fork for pinned pages), so a fork after the MRs
  exist cannot redirect NIC writes. The plugin's ring registers its rings in the same processes.
- **A refused OPEN at boot** (`OPEN refused by the server, status -101`) means the Spark's shard header or manifest
  entry does not match the table x86host loaded. Stop and compare `engram_check --mode ident` on the Spark with x86host's
  `engram_ctl.sh status` identities. Never override it.

## Rollback

- **Back to A:** remove the `PYTHONPATH`, `DSV41_ENGRAM_OVERLAY` and `VLLM_ENGRAM_REMOTE*` entries from `DOCKER_ENV` and
  restart (production `dsv41.env` / `serve.sh` as committed on `main`). Without `DSV41_ENGRAM_OVERLAY` the
  sitecustomize does nothing and the image module loads. The staged files can stay; remove them with
  `ssh NODE rm -rf ~/.cache/dsv41-serve/dsv41-offload`.
- **Without a restart:** if x86host or its server stops, B keeps serving with the exact local rows. That costs one ≤ 100 ms
  timeout per table (or ~2–10 ms when the server's socket closes), then local reads at A's speed until a reconnect
  succeeds (every `VLLM_ENGRAM_REMOTE_RETRY_MS`). Tested: frozen, killed and restarted server.
- **Stop the server:** `ssh x86host '~/dsv41-offload/o1/src/offload/engram/server/engram_ctl.sh stop'`. It frees the
  188.8 GiB on x86host; a restart takes about 70 s to load and verify.

## Resource notes

- **Spark, per rank:**
  - 2 × 12.98 MB NIC-registered staging (replaces A's 2 × 12.98 MB pinned staging);
  - 2 × ~200 KB request slots;
  - the fallback reader's file descriptors and pread pool (as in A).
  - One spinning proxy thread, which replaces A's spin proxy: the fallback table never enables spin.
- **x86host:**
  - 188.8 GiB anonymous THP (never registered, never pinned);
  - per connection 2 × 13 MB registered response buffers;
  - 8 gather threads plus one thread per connection, which busy-poll for 2 s after a request and then sleep on the
    completion channel.
- **Ports:** 18610 on 10.202.28.1 and 10.202.29.1 only (no 0.0.0.0).
