# AUD1 item 3: OPS1 status against today's production (21bed31) and offload-x86host

**Verdict:**
- **offload-x86host:** OPS1 is already merged (`bde828d merge offload/ops1 d57893b`). Its files there are
  byte-identical to `159b5a1`, and its 19 CPU tests pass (9 warm-up, 10 profile).
- **Production main 21bed31:** `159b5a1` merges cleanly (`git merge-tree`, no conflicts). Only the **warm-up** (four
  files under `offload/launch/`) needs to be on the production checkout. The profile registry and the read-only
  checker run from Controller. `d57893b` only adds a roadmap row, and main has no roadmap (modify/delete conflict), so
  skip it.
- **Stale after K1-327 + TP2:** one thing, the registry's source-control record. Sparkhead committed and tagged the
  applied K1 files as 21bed31 / `prod-k1-327-20261023`. The checker therefore reports today's production as
  `DRIFT_OR_UNKNOWN` (HEAD, dirty state and diff all differ). Replayed offline below. A one-line-per-field patch is
  ready (`offload/audit/ops1/registry-21bed31.patch`).
- **Risk found:** the warm-up runs **inside K1's 600 s start guard**. A slow warm-up can trip the guard, and the
  guard's fallback **restores the pre-K1 configuration**. Bound it in `warmup.py`, never in `start.sh` (see
  section 3).
- Nothing else in OPS1 is broken by K1-327 or the TP2 worker.

## 1. Facts (MEASURED from git on Controller, 2026-10-26)

| Check | Result |
|---|---|
| Brief's OPS1 base | `f4bc392` is OPS1's first commit; the branch forks from offload-x86host at `cc9515f`. Branch head today `d57893b` (= `159b5a1` + 2 roadmap lines) |
| OPS1 in offload-x86host? | Yes: `f4bc392`, `159b5a1` and `d57893b` are ancestors of eb60342 via merge `bde828d`. `offload/operations/*` and `offload/launch/warmup*` blobs are identical to `159b5a1` (e.g. profiles.v1.json `ef7487a2`, warmup.py `9ac6dd2e`) |
| OPS1 in production main? | No. None of its commits are ancestors of 21bed31 |
| `159b5a1` onto 21bed31 (merge base `cc9515f`) | `git merge-tree --write-tree` exit 0, tree `25c76ba4`, 9 files, +1,815/−22. Only `warmup_serve.sh` exists in 21bed31, and it equals the base (`cde8c85`), so it changes cleanly. The other 8 files are new |
| `d57893b` onto 21bed31 | CONFLICT (modify/delete) `documents/plans/roadmap.md`. Main does not carry the roadmap: skip it |
| 21bed31 vs what OPS1 observed serving | `git diff ad2a4a0 21bed31 -- dsv41.env start.sh` has sha256 `e7ef68d6…a793`, the very `applied_diff_sha256` OPS1 recorded on 2026-10-23 22:39Z. 21bed31 commits exactly the configuration that was serving |
| Tag | `prod-k1-327-20261023` = tag object `54db31f5…` → commit 21bed31. OPS1 recorded it as `PREPARED_NOT_RUN` |
| Worker | Roadmap (2026-10-24 01:38Z): x86host `current` → `fb68c52`, which OPS1's registry also records. Not re-read: no x86host access |
| OPS1 tests on offload-x86host (this worktree, `PYTHONDONTWRITEBYTECODE=1`) | 10/10 profile tests OK. 9/9 warm-up tests OK (extracted `159b5a1` tree in scratch) |
| OPS1 tests on main | `test_registered_files_and_git_refs` would fail on a main checkout: `offload/launch/configs/E4-native-kx2-async.env` is offload-x86host only. Every other registered file hash and rollback ref matches in the merged tree. So do not copy `offload/operations/` to main |
| Checker on OPS1's 10-23 receipt | `CONFIGURATION_MATCH` (still, for that snapshot) |
| Checker on the same receipt edited to today's committed state (HEAD 21bed31, clean, empty diff) | `DRIFT_OR_UNKNOWN`: "sparkhead HEAD differs", "dirty state differs", "applied diff differs" (exit 2) |
| Same, with `registry-21bed31.patch` applied | `CONFIGURATION_MATCH`. The 10-23 receipt now correctly reports drift. OPS1's 10 tests pass with the patch (its one fixture line updated: the "dirty drift" probe becomes `[" M dsv41.env"]`, because the clean state is now `[]`) |

## 2. What is stale after K1-327 + TP2, and what is not

Stale:
- `profiles.v1.json` K1-327680 `source_control`. It records `allowed_heads` [ad2a4a0], `dirty` [" M dsv41.env", " M
  start.sh"] and `diff_sha256` e7ef68d6…, but production is now the clean 21bed31. Also stale: `pending_record`
  (`PREPARED_NOT_RUN`) and `rollback_refs`, which lacks `prod-k1-327-20261023`. Fixed by the patch.
- Wording in `offload/operations/README.md:67` ("Sparkhead source record/tag: Prepared, not run") and the
  procedure at `:114-124`: done since 2026-10-24. K1-524288 is "HOLD" there; NDM closed 524K on 2026-10-26 and
  reopened it only for the bounded PF2-LOWKV experiment. HOLD remains the right checker state (not production).
  Docs only, no behaviour change.

Not stale (checked against 21bed31 and the K1 guard):
- The warm-up waves are concurrency 1, 2, 3, 4: within `max_num_seqs` 4 and the TP2 worker's 4 HELLO slots.
- Wave 1 must exceed 4,096 prompt tokens, which is one full chunk at `MAX_BATCH` 4096. The old warm-up's prompt was
  5,041 tokens.
- A per-start uuid in the prompt text means no cache hit under K1's 384-token blocks.
- Raw `/v1/completions` with `min_tokens = max_tokens = 32` and `ignore_eos`, priority 20.
- The idle gauges are checked before each wave.
- The call signature `warmup_serve.sh PORT SERVED` is unchanged, so 21bed31 `start.sh:36` calls it as before.
- K1's guard does not hash the warm-up files: `manage.py verify` checks only `dsv41.env`/`start.sh` and the overlay
  manifest (`manage.py:53-57, 106-120`; `stage_overlay.sh verify-checkout`). It does not check git HEAD, so a new
  commit on main does not trip it.
- The concurrency-6/8 and clock plans (`CONCURRENCY-CLOCKS.md`) are still plans gated on source limits. Nothing in
  21bed31 changes those limits.

## 3. The start-budget risk (K1 guard)

`start.sh` → `offload/promote/k1/start_checked.sh` runs `K1_START_INNER=1 timeout --kill-after=15s 600s bash
./start.sh`, and the warm-up is the last step inside it (21bed31 `start.sh:31-38`). On a timeout, the ERR trap stops
the serve, runs `manage.py restore` (the **pre-K1** `dsv41.env`/`start.sh`) and restarts that previous configuration.

- MEASURED: the K1-327 production apply took 432 s from "patched; restarting production" (20:45:23Z) to "K1 START
  VERIFIED" (20:52:35Z), including a 5.6 s warm-up (`K1-production/k1-prod.log`). That leaves about 168 s of margin.
- The new warm-up runs 4 sequential waves (10 requests, at most 320 forced tokens).
  - ESTIMATED typical: 10-25 s.
  - Worst case: each request has a 60 s socket timeout and the barrier 5 s, and the first failure ends the
    warm-up, so up to about 4 × 65 = 260 s if every wave is slow but succeeds. That exceeds the margin. The guard
    would then treat a slow warm-up as a failed K1 start and roll production back to pre-K1.
- Mitigation (coordinator's choice, before the apply):
  - Add a total deadline in `offload/launch/warmup.py`: for example, a `DEADLINE = 90` s checked between waves and
    passed as each `urlopen` timeout. A deadline overrun then becomes the existing WARN in `start.sh`.
  - Or apply the warm-up only together with a re-measured start time.
- **Never edit `start.sh` to wrap the warm-up:** `manage.py verify` refuses to start when `start.sh` changed since
  the K1 apply ("promoted start.sh changed since apply; refusing start"), which would trigger the same fallback.

## 4. Exact coordinator steps (not run by AUD1)

**A. Registry (offload-x86host, Controller clone; no service contact):**
```bash
cd /opt/dsv41/repos/dsv41-deploy            # offload-x86host checkout
sha256sum <AUD1 worktree>/offload/audit/ops1/registry-21bed31.patch   # 70d4241a638fd54559c15062f067b3d5159780e72de20ef5d26da1f1fd44571b
git apply --check <AUD1 worktree>/offload/audit/ops1/registry-21bed31.patch && git apply <same>
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= python3 -m unittest discover -s offload/operations -p test_live_profile.py   # 10 OK
git add offload/operations/profiles.v1.json offload/operations/test_live_profile.py
git commit -m "ops1 registry: K1-327680 recorded at 21bed31 (prod-k1-327-20261023), clean tree"
CUDA_VISIBLE_DEVICES= python3 offload/operations/live_profile.py --live    # fixed read-only ssh reads; expect exit 0 CONFIGURATION_MATCH
```
After any later production commit (for example step B), add its SHA to `allowed_heads`, or the checker reports HEAD
drift again.

**B. Warm-up onto production main (optional; no restart; takes effect at the next production start):**
1. Decide the section 3 mitigation first. If the deadline is added, regenerate the patch from that commit.
2. Preconditions on sparkhead: `cd ~/dsv41-deploy && git rev-parse HEAD` = `21bed313ac803d38327c4de1ff2e5235f329c716`;
   `git status --porcelain` empty; tag `prod-k1-327-20261023` present; host `python3 -c 'import sys;
   assert sys.version_info >= (3, 10)'` (warmup.py uses `X | None` at runtime).
3. Copy `offload/audit/ops1/warmup-onto-21bed31.patch` (sha256
   `c643b2c1269995007b3adee9b9c5030f26992bd4957f0c739f5241ecad14a61e`; equal to `git diff 21bed31 159b5a1 --
   offload/launch/{warmup.py,warmup_serve.sh,OPERATIONS.md,tests/test_warmup.py}`) to sparkhead, then:
   ```bash
   cd ~/dsv41-deploy && git apply --check ~/warmup-onto-21bed31.patch && git apply ~/warmup-onto-21bed31.patch
   git ls-files -s offload/launch/warmup.py offload/launch/warmup_serve.sh offload/launch/OPERATIONS.md offload/launch/tests/test_warmup.py
   #   9ac6dd2e… warmup.py, 29b302fe… warmup_serve.sh (100755), f2a0cd35… OPERATIONS.md, 845258d1… test_warmup.py
   #   (after `git add`; or compare `git hash-object` of each file)
   PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s offload/launch/tests -p test_warmup.py   # 9 OK
   bash -n offload/launch/warmup_serve.sh
   git add offload/launch/warmup.py offload/launch/warmup_serve.sh offload/launch/OPERATIONS.md offload/launch/tests/test_warmup.py
   git commit -m "warm-up: OPS1 bounded 4-wave internal warm-up (159b5a1); start.sh/dsv41.env unchanged"
   ```
   `dsv41.env` and `start.sh` must stay byte-identical (K1 guard). Check with `python3 offload/promote/k1/manage.py
   verify --profile 327680`, a read-only verify that prints "…verified".
4. The running serve is untouched. Do **not** restart for this. Qualify it at the next planned production start
   (for example after the SP1/PF2 windows), with producers held:
   - the start log has 4 `"warmup": "completed"` lines with `requested_concurrency` 1-4, each request
     `completion_tokens` 32 and `finish_reason` "length";
   - wave 1 `prompt_tokens` > 4096 and `cached_tokens` 0/null;
   - "K1 START VERIFIED" inside 600 s, and the total start time recorded against the 432 s baseline;
   - fresh x86host worker steps/drafted-request deltas;
   - then `live_profile.py --live` CONFIGURATION_MATCH (after step A, with the new HEAD added).
5. Rollback (no restart): `git revert --no-edit <warm-up commit>`. That restores the old one-request `curl`
   warm-up, which takes effect at the next start. If a start is in progress and the guard has fallen back, follow
   the K1 runbook (`manage.py restore` receipt) as for any K1 fallback.

**C. Docs:** update `offload/operations/README.md:67,114-124` to "run 2026-10-24, 21bed31, tag object 54db31f5", and
record K1-524288 as CLOSED (PF2-LOWKV experiment only). No behaviour change.
