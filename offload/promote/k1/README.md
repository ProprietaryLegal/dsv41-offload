# K1 production promotion package

CPU-tested on Controller; **not deployed**. S18's FINAL report (2026-10-23,
checker `070cc0a`) clears K1 at **327,680** and holds **524,288**. Only the
coordinator applies this package during a drained maintenance window, after
TP2 + async production is complete and healthy.

| Explicit profile | max_model_len | KV pool tokens | S18 gate / current status |
|---|---:|---:|---|
| `327680` | 327,680, unchanged | 2,502,600 | final window6 PROMOTE / **PROMOTE** |
| `524288` | 524,288 | 2,820,756 | final window6 + window6b (R5) PROMOTE / **HOLD** |

There is no default profile. Both use **6,442,450,944 bytes per rank**. Capacity
figures agree with `offload/kvpack/kv_arith.py:capacity(bytes, max_len, 3, 14)`;
every production start must confirm the selected pair in the new engine's log
and `/v1/models`. The context cap includes output space.

## Promote 327,680 first

S18 found the pool grows from 1,604,926 to 2,502,600 tokens (+56%) at the
unchanged per-request cap and budget. Cold prefill was unchanged (150K: 68.92
versus 69.02 seconds), with decode within the approved bounds. Every window-6
acceptance row passes; reviewed flags and data gaps are recorded in S18's report.

The patches target main `3c1dc6b`; recheck current main before promotion.
`dsv41.env.327680.patch` leaves `PROD_MAX_LEN=327680` and
`PROD_KV_BYTES=6442450944` untouched. It appends the 13 literal read-only mounts
from `offload/kvpack/FILES`, pinned to manifest
`559c5867ba42aa7daa7b4a7c8231d15ee1d35be40ae5b715b013f973193c58de`,
and exactly S18's Production form switches:

```
VLLM_DSV41_KV_PAIR=3
VLLM_DSV41_DROP_MTP_BUFFER=1
VLLM_EXL3_TF_PROMPT_TRANSIENT=1
VLLM_EXL3_TF_PROMPT_ALIAS=1
VLLM_INDEXER_PREFILL_BUFFER_FACTOR=4
```

The only other env additions are the startup guard `PROD_K1=1` and selected
profile `PROD_K1_PROFILE=327680`. Existing async DSpark, Engram fallback policy,
mounts and production settings retain their exact bytes; `serve.sh` and
`stop.sh` stay unchanged. Literal mounts ensure sourcing the env never stages
files. No x86host worker, GPU, service-unit, image or weight changes are included.

### Receipt and prechecks

S18/coordinator supplies an attestation bound to the final report hash. This
schema example is **not an approval receipt**; do not use synthetic test receipts
for production:

```json
{
  "schema": "dsv41-k1-promotion-v1",
  "reviewer": "S18",
  "final": true,
  "window6": "PROMOTE",
  "window6b": "HOLD",
  "max_model_len": 327680,
  "kv_tokens": 2502600,
  "manifest_sha256": "559c5867ba42aa7daa7b4a7c8231d15ee1d35be40ae5b715b013f973193c58de",
  "report_sha256": "<sha256 of S18's final report>"
}
```

For `327680`, window6 must be final PROMOTE; window6b is not a gate and can
remain HOLD or be omitted. For `524288`, window6b must explicitly be PROMOTE as
well. The gate checks exact field types, capacities, manifest and report hash;
it records an independent review attestation, without interpreting report prose
or authenticating a cryptographic signature. A correct startup alone cannot
supply the missing R5 approval.

On Controller, prepare a promotion branch based on current main. Carry this package
and the reviewed `offload/kvpack` helper/overlay; do not merge the entire
experiment branch into main. Recheck current production files and async package:

```bash
source ~/.config/ndm/paths.sh
git log -4 --oneline main
git apply --check offload/promote/k1/dsv41.env.327680.patch offload/promote/k1/start.sh.patch
CUDA_VISIBLE_DEVICES='' python3 -m pytest -q -p no:cacheprovider offload/promote/k1/test_promotion.py
CUDA_VISIBLE_DEVICES='' bash offload/kvpack/test_stage_overlay.sh
```

Run the patch check on the main-based promotion branch. The CPU suite separately
archives current main and checks both profiles with local host-boundary stubs.

Coordinator on sparkhead, `~/dsv41-deploy`, with the reviewed package:

```bash
python3 offload/promote/k1/manage.py check --profile 327680 \
  --receipt /path/to/S18.final.receipt.json --report /path/to/S18.final.report.md
bash offload/promote/k1/stage.sh spark sparkworker
```

Staging uses `--pull never`, checks the image base and pinned checkout, stages
versioned node-local copies, and verifies every hash. It does not restart or load
the model. Check free memory/disk and the TP2 production hold/worker health as in
the async runbook. Confirm the TP2 + async switch passed its greedy and stats
checks. A different manifest, image base or staged copy is a hard refusal.

### Production window at 327,680

Only the coordinator runs these steps. Pause D2 generation and all producers,
coordinate with local-law, and confirm running/waiting request gauges are both
zero. Keep HOLD and node placeholders through the post-start probes; opening a
window alone does not pause API clients.

1. On Controller, save two pre-switch greedy references using the same corpus and
   budgets. Keep proprietary output outside git/reports:

   ```bash
   mkdir -p /opt/dsv41/campaigns/DSV41-OFFLOAD/K1-production
   python3 offload/engram/tests/logit_ab.py run --url http://HEAD_IP:8000 \
     --label pre1 --data /opt/dsv41/data/bench-data/opinions_cl.jsonl \
     --out /opt/dsv41/campaigns/DSV41-OFFLOAD/K1-production/pre1.jsonl
   # Repeat as pre2 / pre2.jsonl for the engine's A-A floor.
   ```

2. On sparkhead in `~/dsv41-deploy`:

   ```bash
   offload/launch/window.sh open 'K1 packing, unchanged 327680 production cap'
   python3 offload/promote/k1/manage.py apply --profile 327680 \
     --receipt /path/to/S18.final.receipt.json --report /path/to/S18.final.report.md
   ./stop.sh
   ./start.sh
   ```

   Apply saves the actual pre-switch files/modes, report and receipt under
   `logs/k1-promotion/`, records profile `327680` in `files.json`, then patches
   both files atomically. It changes no service. Retain the backup even after
   committing patched files. Earlier state is never overwritten; archive a
   completed attempt explicitly before retrying or selecting another profile.

3. Require **K1 START VERIFIED**. Before staging, the guard checks the applied
   profile against `PROD_K1_PROFILE`, the saved receipt/report and file hashes.
   After launch it verifies both nodes' live overlay hashes and exact switches,
   then requires **KV 2,502,600 / max_model_len 327,680** in the fresh engine log
   and API. Existing async warm-up remains. The launcher's early `ready` line is
   insufficient. Evidence: `logs/k1-runtime.*` and `logs/k1-guard-*.log`.

4. With HOLD open, check on Controller:

   ```bash
   curl --fail --silent http://HEAD_IP:8000/v1/models
   python3 offload/kvpack/needle.py --url http://HEAD_IP:8000 \
     --ctx 250000 --needles 6 --reps 2 \
     --out /opt/dsv41/campaigns/DSV41-OFFLOAD/K1-production/needle250.jsonl
   python3 offload/engram/tests/logit_ab.py run --url http://HEAD_IP:8000 \
     --label post --data /opt/dsv41/data/bench-data/opinions_cl.jsonl \
     --out /opt/dsv41/campaigns/DSV41-OFFLOAD/K1-production/post.jsonl
   python3 offload/engram/tests/logit_ab.py compare \
     /opt/dsv41/campaigns/DSV41-OFFLOAD/K1-production/pre1.jsonl \
     /opt/dsv41/campaigns/DSV41-OFFLOAD/K1-production/post.jsonl
   ```

   Require the unchanged alias, cap 327680, all needles recalled and no foreign
   code. Strict greedy identity uses W5's `greedy_check.py` short prompts
   (temperature 0, thinking off). Judge long-context logit A/B against the
   pre1/pre2 floor as S18's `check_w6.py` does: the engine itself can diverge at
   32K token 4. A difference outside that floor stops promotion. Reinspect the
   exact KV/cap log pair; never waive a difference based only on an exit status.

5. Read `/tmp/dsv41-remote-dspark.json` on sparkhead only after the current proxy's
   new stats-write log line (async runbook freshness rule). Require async true,
   healthy proxy, accepted drafts and no new failure/watchdog/fallback counters.
   Compare saved pre-switch stats and x86host worker health without changing units.
   Watch both Sparks' MemAvailable and SwapFree; the campaign memory stop lines
   still bind. Confirm short/c32/c150 performance within S18's bounds, including
   cold TTFT; warm up lazy modules first. At this profile no request past 327680
   is permitted or required.

6. After all probes pass, record aggregates and elapsed start time, close with
   `offload/launch/window.sh close`, and resume producers. local-law's advertised
   context remains 327680. Any later context change requires its own measured
   server change and coordinated metadata update.

## 524,288 remains HOLD

S18 did **not** approve R5. Cold prefill rose **35.45% at 150K** (93.36 versus
68.92 seconds) and **35.79% at 32K** (19.76 versus 14.55 seconds); corruption
waves slowed 22–32%. This breaks NDM's approximately 5% speed-loss rule even
though decode and long-context correctness passed. The cause is unproven. Keep
this form for later investigation and a new explicit final R5 PROMOTE receipt.

`dsv41.env.524288.patch` retains the former 524288 form, with explicit profile
metadata. The gate will refuse S18's current HOLD. Do not relabel HOLD as PROMOTE
or reuse the 327680 receipt. If a later review clears it, use the same procedure
with `manage.py check/apply --profile 524288`, a receipt with `max_model_len`
524288, `kv_tokens` 2820756, both windows PROMOTE and the new final report hash.
Select `dsv41.env.524288.patch` for the main-based patch check.

For that future window, require KV **2,820,756 / max_model_len 524,288** and
replace the 250K needle with 450K and 500K probes. The x86host worker must already
support at least 524288. S22 found R5 GPU evidence used the one-card worker;
TP2 long-position evidence was missing. With the production TP2 worker live,
require fresh stats showing drafts accepted beyond 327680. Repeat the 500K
prefix-identity pair against its cold/cold floor, and watch memory during long
concurrent prompts (S18 observed sparkhead at 3.09 GiB during 2 × 500K). Update
local-law's advertised cap only after every production check passes.

## Rollback

The guard stops a failed candidate, restores the saved files byte for byte with
their modes, and restarts the actual prior configuration. It returns failure
even if recovery succeeds. Corrupted backups or later file edits cause a loud
refusal; HOLD stays open for inspection. A repeated start refuses before the
fallback trap if production already runs.

For a failed post-start probe, coordinator on sparkhead, after draining:

```bash
offload/launch/window.sh open 'rollback K1'
timeout --kill-after=15s 45s bash ./stop.sh </dev/null
python3 offload/promote/k1/manage.py restore
timeout --kill-after=15s 600s bash ./start.sh </dev/null
```

ESTIMATED recovery target: under ten minutes; a live rehearsal remains required.
The 600-second start limit avoids falsely rejecting a slow healthy load. Stop
and start limits plus both kill graces total **675 seconds (11 min 15 sec)**,
before restore/health checks, so the timeout bounds do not guarantee the target.
S22 measured experimental ready times around 360 seconds and estimated a full
production start at 410–440 seconds; this CPU lane makes no live timing claim.
A timed-out fallback stays failed, with HOLD retained.

Verify restored hashes, the snapshot's old cap/KV pair, API alias, healthy async
proxy, accepted drafts and unchanged TP2 worker/hold. Do not assume a stale tag's
settings. Unused versioned overlay files can stay on disk. Close HOLD only after
health and a short greedy probe pass; archive snapshot and evidence.

## While K1 is applied

Every start rechecks the saved profile/receipt and guarded file hashes. Editing
`dsv41.env` or `start.sh` after apply fails verification, and fallback refuses to
discard those edits. Run `manage.py restore` before the async runbook's sync
rollback edits or a checkout of `prod-d-20261023`. The coordinator owns any
explicit closeout of the permanent guard after acceptance.

`PROD_K1=0` or a commented cap is not a complete rollback; use the saved snapshot.
Never leave `K1_START_INNER=1` in the shell: it skips the outer fallback guard.
Apply only this package and reviewed dependencies to main, without merging the
whole campaign branch. A future UP1 promotion must replace overlapping K1
mounts rather than stacking a second overlay on the same paths.
