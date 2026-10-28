# RP1 Stage A GB10 acceptance plan (coordinator only)

> **MC1 amendments (2026-10-27, branch exp/MC1-matter-tier1).** The plan below still governs P0–P2, with these
> changes; MC1 ran nothing on a Spark. Details in [MATTER.md](MATTER.md).
> * **Persistence is matter-gated.** The configs' only persistent tier is `Dsv41MatterTier` at
>   `/root/.cache/kv_matter_mc1` (was the unscoped `fs` tier at `kv_prefix_rp1`; patch 0101 now refuses that in
>   multinode mode). Only requests with `kv_transfer_params {matter_ns, persist: true}` store or read.
>   `restart_probe.py` sends that tag for saved runs (salt = synthetic namespace) and never for `--fresh`. Packets
>   prepared before MC1 (32-hex salts) must be re-prepared.
> * **Stage before every enabled start:** the hash-only exclusion file at
>   `~/.cache/dsv41-serve/dsv41-matter/matter-cache-excluded-ns.txt` (0600, dir 0700) on sparkhead (and sparkworker), and
>   `PREFIX_MATTER_EXCLUDED_SHA256` = first field of its private `.sha256`. For the exclusion test, derive a window copy
>   with `matter/derive_excluded_ns.py --out <window file> --add-synthetic-ns <random 64-hex>`, stage that instead, and
>   send one saved case with `--matter-ns <that hex>`: the audit must show `refuse` / `excluded` and the store root
>   must gain 0 bytes for it. Also: a start with a wrong sha must log the gate CLOSED and persist nothing.
> * **The 29,568-token cap is gone for tagged requests.** `PREFIX_STREAM=1` lands a file-tier hit window by window
>   (two windows of half of min(64 MiB, 0.2 x CPU tier) resident per request). Demand the full 75K/150K restores;
>   untagged requests keep the 0081 cap. 150K restore TTFT <= 10 s is MC2's M2 line (ESTIMATED ~1–5 s here).
> * **P1 byte identity and the M2 byte-equality gate** use the per-block hash dump instead of manual capture:
>   start with `PREFIX_HASH_DUMP=/root/.cache/mc1-hash/<run>` (never in timing runs: it synchronizes per store step);
>   `blockhash_compare.py ranks` (rank 0 vs rank 1, store lines) and `blockhash_compare.py restore` (in-memory store
>   lines vs restored load lines, per rank) are the pass/fail tools.
> * `verify_overlay.py running` now also checks the MC1 switches (`--stream`, `--stream-mb`, `--hash-dump`) and all
>   25 mounted files (K1 13 + RP1/MC1 8 patched + 4 new).
> * Disk bounds now exist: `PREFIX_MATTER_MAX_GB` (8), `PREFIX_MATTER_NS_MAX_GB` (4), TTL 30 days, `matter/purge.py`.
>   Size the bound from the measured free space before the window.

This lane did not contact a Spark or allocate a GPU. Only coordinator Deepseek / coordinator may run the commands
below, inside its normal exclusive drained window. Stage A passes before any Stage B engine integration. No model,
weights, public upload, service replacement, network reconfiguration or permanent GPU allocation is needed.

## Hard stop and prerequisites

Keep each Spark **MemAvailable >=5 GiB (5,242,880 kB)** throughout every run, including startup and prefill peaks.
Any reading below this line, swap growth, NaN, illegal access, foreign needle code, timeout, CRC accepted as a hit,
rank-1 live-payload mismatch, or unexpected mounted bytes: stop the experiment and restore the preceding known config.
Do not lower the threshold to obtain a pass. The K1 plan's 1.5 GiB stop line does not apply to RP1.

Use the existing campaign memory/power logger and record raw 5-second MemAvailable/SwapFree from both hosts. Check
production `/metrics` has zero running/waiting requests before every Spark-NIC burst and before opening the window;
stop lane traffic if requests appear. Record unchanged model/checkpoint identity, clocks, effort, concurrency, cache
state and current async worker. Leave production ownership and its rollback procedure with the coordinator.

On Controller, before a window:

```bash
source ~/.config/ndm/paths.sh
python3 offload/prefixstore/build_overlay.py
python3 offload/prefixstore/verify_overlay.py tree /opt/dsv41/data/src-image --base
python3 offload/prefixstore/make_configs.py --check
python3 offload/kvpack/make_configs.py --check
bash offload/kvpack/build_overlay.sh
bash offload/kvpack/test_stage_overlay.sh
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 "$HARNESS_PYTHON" -m unittest discover -s offload/prefixstore/tests -v
```

Pass: seven-file fuzz-0 rebuild, 20 base files match, 3+6 generated configs match, K1 13 files match and 27 tests pass;
RP1 CPU suite passes. Use the final branch commit from the campaign report; independently verify before merge.
The donor and source paths exist only on Controller. On sparkhead's reviewed checkout, run
`python3 offload/prefixstore/build_overlay.py --checkout-only` before start_exp copies files. After every start,
from Controller run `python3 offload/prefixstore/verify_overlay.py running sparkhead sparkworker-wired` (P0 adds `--control`;
P1 adds `--verify-stores`). This reads hashes/environment only and proves all 20 mounted paths. Also run K1's
`stage_overlay.sh verify-running --expect` with its ordinary packed switches to certify the inherited K1 settings.

The seven base files must also match image v5 on both hosts; the saved source check is not proof of a changed image.
Use a small disposable image check during the coordinator's window, never pull an image. Extend K1's `docker run
--rm --pull never --entrypoint sha256sum` check with RP1's seven MANIFEST `container_path` entries and compare to
`base_sha256`. Any mismatch requires a new rebuild/audit, not bypassing the check.

Before enabled starts, set `PREFIX_MODEL_SHA256` to SHA256 of the exact verified legal-v2a checkpoint manifest.
Both nodes must have that manifest hash and the already-verified corresponding weight bytes. Use existing S2 and
candidate receipts; do not hash/read large weight files through the Spark page cache. This digest deliberately
stays outside committed configs. K1+RP1 layout digest is generated automatically. Confirm A725 CPU IDs/allowed
cpuset cover `0-4,10-13`; otherwise choose verified allowed efficiency cores via PREFIX_IO_CPUS and pass the same
value to verify_overlay `--io-cpus`. Do not guess CPU pinning from this document.

The filesystem tier uses the existing node-local cache mount `/root/.cache/kv_prefix_rp1`, O_DIRECT, 2 read and 1
write thread, and an 8 MiB gloo staging buffer. If O_DIRECT is unsupported, enabled startup must fail. Do not accept
buffered fallback. Record free disk space and bound the experiment's new disk footprint (e.g. 8 GiB). Expire only
inactive RP1 namespaces while stopped; this tier does not implement automatic disk quota/eviction.

## Runs in order

Every start below runs on sparkhead from the reviewed checkout in the coordinator's already-open window:

| Run | Start command | Required observation |
|---|---|---|
| P0 | `PREFIX_ENABLE=0 offload/launch/start_exp.sh offload/launch/configs/D-kv6-async-pack-prefix.env` | No connector CLI or RP1 switches. Same K1 pool/groups and same fixed-workload outputs as D-kv6-async-pack. Timed no-probe control; >=5 GiB on each Spark. |
| P1 | `PREFIX_MN_VERIFY=1 offload/launch/start_exp.sh offload/launch/configs/D-kv6-async-pack-prefix.env` | Offloading config: two one-head selected groups, 128/192 tokens/chunk, one-copy replicated layout. CPU stride 139,264 B (raw 138,240), 1,927 chunks, cap **29,568**. Both ranks store; logs show gloo init. Capture independently stored valid group payload for byte identity before any load; see below. Not a timing run. |
| P2 | `offload/launch/start_exp.sh offload/launch/configs/D-kv6-async-pack-prefix.env` | Same cap; rank-0 stores on, rank-1 stores off. Gloo replaces rank-1 CPU chunks before GPU copy. Run restart/corruption/large-prefix/concurrency gates below. |
| P3 (optional) | `offload/launch/start_exp.sh offload/launch/configs/D-kv6-async-pack-prefix-1g.env` | Only after >=6 GiB baseline headroom plus measured startup allowance. CPU tier 7,710 chunks, cap **118,272**; repeat restart75K, four concurrent loads and mixed decode. Stop at5 GiB. |
| P4 (optional) | `offload/launch/start_exp.sh offload/launch/configs/D-kv6-async-pack-prefix-1m.env` | Only after K1's 1M tests pass and RP1 P2 passes. Full1M worker already required by K1. Cap stays29,568; test prefix150K first, then selected long cases if >=5GiB holds. No new capacity claim merely from configuring1M. |

P1 rank identity: take a synthetic 24K needle with one cache salt, let all store jobs finish, and record the newly
created engine ID, mmap filename and CPU chunk IDs mapped to the two selected groups. P1 allows rank1 to store its
own computed KV. Compare rank0/rank1 valid payload for those chunks before any reload; capture SHA256 and CRC for
both nodes. The complete packed backing stride can include inactive/padding bytes: compare the actual group's valid
live bytes and record any excluded ranges from the worker's mapping. Fail on any selected-payload difference.
A one-head type predicate or a post-broadcast match is not evidence of independently computed byte identity.
Run this once with the production async drafter and remote Engram enabled as generated.

## Exact saved-prompt restart test

Use the same packet and salts at every step, on Controller. Output paths must be new (the probe refuses overwrite).
All prompt/output evidence stays in the campaign directory, never git. These are public-corpus needle diagnostics
with thinking off, distinct from production timing. Keep the production timing suite's reasoning settings unchanged.

```bash
source ~/.config/ndm/paths.sh
RP1_RESULTS=/opt/dsv41/campaigns/DSV41-OFFLOAD/RP1-GB10
mkdir -p "$RP1_RESULTS"
python3 offload/prefixstore/restart_probe.py prepare --url http://HEAD_IP:8001 --model DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2 --ctx 24000 75000 150000 --out "$RP1_RESULTS/packet.json"
timeout 300s python3 offload/prefixstore/restart_probe.py run --packet "$RP1_RESULTS/packet.json" --url http://HEAD_IP:8001 --out "$RP1_RESULTS/P2-cold.json"
```

Wait until store jobs finish and capture file-tier successful-store counters, paths and sidecars for this namespace.
Record the exact old experiment engine ID. Coordinator stops the experiment and verifies both processes are gone;
remove only that stopped experiment's unused `/dev/shm/vllm_offload_<id>.mmap` files after confirming no live process
maps them. Preserve disk files. Restart P2 normally (new engine ID and empty GPU/CPU cache). Verify runtime bytes.
There must be no intervening request with the packet salts. Then:

```bash
timeout 300s python3 offload/prefixstore/restart_probe.py run --packet "$RP1_RESULTS/packet.json" --url http://HEAD_IP:8001 --out "$RP1_RESULTS/P2-reload.json"
timeout 300s python3 offload/prefixstore/restart_probe.py run --packet "$RP1_RESULTS/packet.json" --url http://HEAD_IP:8001 --fresh --out "$RP1_RESULTS/P2-fresh.json"
python3 offload/prefixstore/restart_probe.py compare --packet "$RP1_RESULTS/packet.json" --cold "$RP1_RESULTS/P2-cold.json" --reload "$RP1_RESULTS/P2-reload.json" --fresh "$RP1_RESULTS/P2-fresh.json" --out "$RP1_RESULTS/P2-compare.json"
```

Pass: cold and fresh controls cached0; reload cached>0 and <=cap (plus separately identified local hits, which are
absent after the controlled restart); successful file-tier reads and gloo-copied chunks increase; 6/6 codes for every
request, no foreign code, chosen first token equal to fresh compute. Coarse first-token KL must be <=max(.005,2x the
cold/fresh floor), with common top-K probability mass >=.99. The tool emits these checks; require every `pass:true`.
This top20+OTHER KL is a lower bound, not full-vocabulary KL. For ambiguous/non-peaked first tokens or unexplained
drift, capture full logits with the existing engine diagnostic before acceptance. A missing logprob/usage field is a
hard test failure, not a silent skip.

24K should fit wholly within the cap (apart from scheduler alignment and final token). 75K/150K on P2 deliberately
exercise partial load + recompute. **Never demand a full150K file hit at256MiB.** Repeat full75K at P3 if memory allows.
A full150K at fraction.2 requires about1.3GiB CPU landing memory/rank plus staging and >=5GiB remaining; do not run
that optional variation unless budgeted and measured. Record actual loaded tokens, TTFT and first-visible-token time.
Three repeated comparable restart trials are the promotion evidence. Require median restart TTFT improvement >=5%
for the admitted24K prefix, no unexplained150K regression >5%, and bounded completion (<300s per command). Report
partial150K gains separately. Upstream1.3–2.1s at75K is a donor result, never our pass expectation.

## Rank-1 stale-block, corruption and failure gates

| Gate | Procedure | Pass / fail |
|---|---|---|
| Stale rank1 | After a controlled P2 restart and before any packet reload, production drained and experiment idle with all store jobs complete, coordinator confirms the new rank1 experiment engine ID and that its region is not a production/other-engine region. Overwrite only that new rank1 CPU mmap with a recognizable stale sentinel in <=64KiB chunks (<=the configured256MiB region, no large buffer). Keep rank0 CPU/file payload untouched. Repeat the saved reload and compare to fresh. Observe `_mn_sync` chunk copy before GPU load; instrument/hash selected CPU chunks at the copy boundary in a diagnostic run if needed. | Rank1 sentinel is replaced for every submitted chunk; selected rank1 bytes equal rank0 after sync, codes6/6, first token/KL within floor. A zero-initialized restart alone does not exercise stale data. A postload answer without byte/order evidence is insufficient. Disable MN_VERIFY for this gate so rank1 cannot overwrite the injected sentinel before reload. |
| CRC byte corruption | With experiment stopped, select a recorded RP1 disk block used by packet24K, preserve its valid sidecar, flip one byte using aligned O_DIRECT I/O; restart P2 emptyCPU/GPU, reload. Use a page-aligned mmap4096 buffer with `os.open(O_RDWR|O_DIRECT)`, `preadv`/`pwritev` at offset0; assert4096 bytes each, changeonebyte. Never read entire block through page cache. | Log names exact CRC mismatch and removes block+sidecar. No invalid chunk completion; recomputed answer codes/logits pass. A fast answer alone does not establish miss/fallback. |
| Missing sidecar | Regenerate the clean packet store, stop, remove only its recorded `.crc`, restart emptyCPU/GPU, reload. | MissingCRC is a miss and block is removed/recomputed. |
| Namespace mismatch | Stop, start with a different *test* PREFIX_MODEL_SHA256 (same actual weights for this negative test only), request original packet. Restore true identity immediately afterward. Repeat with changed layout identity in a diagnostic config. | Old file namespace never read, cached0 in emptyGPU/CPU; recompute passes. Never promote or write real records under the false test namespace. |
| Oversized prefix / four reloads | Pre-store four distinct150K needle packets; restart and submit them concurrently, same stored salts, using four `restart_probe run` commands with separate output files. Observe file reads/load caps and pending requests until completion. | Each completes<300s, loaded<=cap29,568 on P2, codes6/6, no foreigncode; >=5GiB eachSpark. No persistent pending/self-evict loop. Also test explicit request `kv_transfer_params.max_load_tokens=1000000`: scheduler must clamp down to configuredcap. |
| Peer/link failure | Isolated diagnostic window only: use an invalid RP1 gloo interface at startup (never change live networking), or stop only the coordinator-owned experiment peer. | Startup/request fails clearly, never accepts stale chunks; timeout abort and restore baseline. StageA does not promise transparent recovery from broken gloo. Do not interrupt unrelated peers. |

The small CPU test already fills rank1 with stale bytes and executes actual `_mn_sync` through sparse/staged and
contiguous paths with fake collective transport. That is a regression check, not a substitute for this two-node gate.
Coordinator may add diagnostic instrumentation on a separate reviewed overlay; no live monkeypatching of production.

## Decode, interaction and promotion

Run existing probe-free `offload/bench/suite.sh` for P0 and P2 with matched worker, clocks, sampling, effort and exact
workload: `short c32 c150`, one/three requests as existing suite defines; include the mixed background150K/foreground
decode case. Repeat a small A/B/A to measure noise. Pass: accepted target tokens/s and visible-answer throughput lose
<=1%, no isolated cell worse than5% without explanation, and end-to-end representative tasks do not regress>5%.
Record gloo time, file bytes, cap/admitted tokens, CPU jobs, remote Engram/drafter timeouts and per-Spark memory.
A prefix-only speedup does not excuse a persistent decode regression. Upstream+0.27% is reference context only.

Run existing `needle.py --ctx 8000 30000 50000 100000 --concurrent 4 --needles 6` under P2: pass16/16 requests6/6,
zero foreign codes. Use `prefix_identity.py --block 384` ordinary GPU-hit rows as well, preserving K1's cold-control
assertions. These use fresh salts and complement, rather than prove, the saved-salt SSD restart path.

Promotion requires every P0/P1/P2 gate, >=5GiB throughout, rank-byte/stale/corruption evidence, matched decode and
restart timing and an independent merge check. P3/P4 are optional and promoted only if individually gated. Before
long-running use add bounded inactive SSD-namespace eviction/operational resource ownership; do not let this unbounded
file tier fill a system drive. StageB remains unmounted and has separate trusted-catalogue/tier-adapter gates.

Rollback is the coordinator's normal stop/start of the known pre-RP1 config; restore D-kv6-async-pack/production args,
remove RP1 mounts/switches/connector CLI, and verify model health and existing workloads. Leave other lanes' processes
alone. Delete only this experiment's unused shared-memory regions and inactive disk namespace when safe. Restoring
production is also required after any failed gate. No push, merge or service installation is authorized by this plan.
