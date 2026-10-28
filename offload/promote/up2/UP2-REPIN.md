# UP2-PROMOTE re-pin list (item 3; no edits made)

What a PASSED 0075 promotion changes, and which runner pins must follow. Nothing here is edited by this lane: re-pins
happen only after a real PASS, and each owning lane re-pins its own runner (PROMO1/PP4 practice). Rollback/control arms
and historical evidence keep their old identity. Refs scanned 2026-10-27 from local Git objects only.

## What changes in production on PASS

| Identity | Before (ops1) | After |
|---|---|---|
| Production commit/tag | 596ba1c / `prod-k1-327-ops1-20261026` (or PROMO1's `prod-k1-327-low-*`) | hooks commit (offload/promote/up2/hooks.py, before the HOLD) + the dsv41.env commit, tag `prod-up2-YYYYMMDD` |
| dsv41.env | — | `OFFLOAD_DOCKER_ENV` + `VLLM_DSV41_REPLAY_SKIP=1`; remote_dspark source `~/dsv41-offload-prod/up2/ad3dc5be5eaf/remote_dspark`; four new file mounts; `PROD_UP2=ad3dc5be5eaf` |
| Running engine | — | env gains the one flag; mounted bytes: scheduler.py, input_batch.py, model_state.py, replay_skip.py (new), remote package (gather/round/roundcore/speculator changed, suffix_provenance.py new) |
| Greedy references | `K1-production/greedy-post.jsonl` (K1-post) | `K1-production/greedy-post-up2-ad3dc5be5eaf.jsonl` (three identical runs recorded in the HOLD; equal to K1-post only if 0075 changes no greedy text) |
| W8 | `W8/restore.sh` (sha d1c4479d…) | `W8/restore-up2.sh` (coordinator copies `offload/promote/up2/restore-up2.sh`); restore.sh stays the rollback restore, never replaced |
| K1 record | promoted dsv41.env hash | rebased dsv41.env hash; snapshot in `logs/up2-switch` |

## Repo plan (coordinator 2026-10-27): remotes and the clock fold

* **Remotes.** `origin` is now the private GitHub repo ProprietaryLegal/dsv41-deploy. `sparkhead` is the production
  deploy remote. Every push of a production commit (the hooks commit, then the accepted dsv41.env commit) names
  `sparkhead` explicitly, never `origin`.
* **Clock fold.** The next production commit must carry `prod/clock-fold-2300` at 420be24. That commit changes the
  dsv41.env default from `GPU_CLOCK_CAP=${GPU_CLOCK_CAP-2150}` to `${GPU_CLOCK_CAP-2300}`.
  * **If 0075 promotes first**, the apply order is 420be24 → hooks → push `sparkhead` → fold-clock → W8 start:
    1. Put 420be24 on top of the live production commit, then the UP2 hooks commit (`hooks.py apply`), then push to
       `sparkhead`.
    2. Run fold-clock on sparkhead right after the push. The K1 start guard hashes dsv41.env, so after the one-line
       fold `manage.py verify` (and every production start) refuses until the K1 record is re-recorded. The step
       reuses PP4-PROMOTE's own action from 8548a7c, carried here byte for byte as
       `offload/launch/windows/pp4_promote_switch.py`, not re-implemented. Run it from the staged UP2-ADOPT checkout
       (`~/dsv41-deploy-up2a`, UP2A-STAGE.sh), and only after its bytes are proved to be the reviewed tool (S97 FIX 5;
       sha256 of blob 9dcb9eb, `promote_window.FOLD_TOOL_SHA256`):
       `ssh sparkhead 'cd ~/dsv41-deploy-up2a && echo "e722fc56913673e4ac23b9c19b64401f81c5115f5650863c309078be9715ee49  offload/launch/windows/pp4_promote_switch.py" | sha256sum -c --quiet && python3 offload/launch/windows/pp4_promote_switch.py fold-clock --root ~/dsv41-deploy'`.
       Never copy that file into the production tree: PP4-PROMOTE's own
       package apply refuses a tree that already contains it. The action proves dsv41.env differs from the
       K1-verified one by exactly the clock line, keeps a files.json copy and ledger under `logs/clock-fold`, and is
       safe to run twice.
    3. Run the W8 start (`restore.sh start`/`check`) on the folded production, then the UP2 promotion HOLD. Its
       preflight (`promote_window.CLOCK_FOLD_CHECK`) refuses unless dsv41.env has the folded 2300 default and
       `logs/clock-fold/transaction.json` exists.
    * The 0075 switch composes with the fold. On 420be24's dsv41.env, `base_of` = ops1, and `edit_env` changes only
      `OFFLOAD_DOCKER_ENV` / `OFFLOAD_DOCKER_ARGS`, so the 2300 line is kept. This was checked on CPU.
    * The switch rebases the K1 dsv41.env hash from the INSTALLED (folded, fold-clock-recorded) file. Its snapshot,
      and so every UP2 rollback, returns the folded production. This is tested: fold-clock (twice), then
      spark-apply, then spark-restore, which yields 420be24's env byte for byte.
    * After the post-close watch passes, delete `GPU_CLOCK_CAP` from `/etc/environment` on both Sparks. The fold
      then supplies 2300.
    * Before any W8 rollback to a pre-fold commit (`restore.sh` on 596ba1c, whose default is 2150), first re-add
      `GPU_CLOCK_CAP=2300` to `/etc/environment` on both Sparks.
  * **If PP4 promotes first:** PP4's production commit already carries the fold, and PP4's runner already ran
    fold-clock (`logs/clock-fold` exists). This package rebases on that production. The PP4 base itself stays refused until `pp4-composed` is measured.
  * PROMO1's `dsv41.env.327680-low.patch` also edits the `GPU_CLOCK_CAP` line (2150 → 2300). On a folded production
    its first hunk no longer applies (fail-closed). That is PROMO1's re-pin, not part of this package.

## Shared pins every start_exp/PF1-style runner inherits

| Pin | Where | Needed after PASS |
|---|---|---|
| W8 path + sha | `offload/launch/windows/pf1_window.py:39-40` (`RESTORE`, `RESTORE_SHA256`), `post_close_watch.py:39-40`, `offload/dspark/d2/smoke_window.py:30`, `replay_window.py` | restore-up2.sh and its sha (post_close_watch takes `restore_sha256=`, the PROMO1 extension this lane carries) |
| W8 greedy line | runners parse `greedy K1-post vs restored: N IDENTICAL` (restore-up2.sh keeps that literal prefix) | none (format kept) |
| In-window greedy references | `offload/launch/windows/nbw_cells.py:24` `REFERENCE = K1-production/greedy-post.jsonl` (UP3, PC1, TV1, PP4, NBW, CC1 greedy) | point at greedy-post-up2-ad3dc5be5eaf.jsonl if it differs from K1-post |
| Replay guard in start_exp | `offload/launch/start_exp.sh:13-19` runs `offload.upstream_adopt.replay_guard` whenever EXP_DOCKER_ENV carries `VLLM_DSV41_REPLAY_SKIP=1` | Production's env now carries it, so every arm built from production env trips the guard; `replay_guard.verify_production` (this lane, selected when dsv41.env sets PROD_UP2) accepts exactly production's staged scheduler/remote mounts and refuses any experiment overlay on a 0075 file. Runners must carry this replay_guard version. |
| Production identity snapshot | PF1 `capture()`/`runtime()` compare an arm with the saved production identity | no edit: the snapshot is taken live; arms inherit the flag and mounts |

## Runner-specific production constants

| Runner (ref) | File:line | Pin |
|---|---|---|
| PP4 stage7 (941a3c3) | `offload/launch/windows/pp4_window.py:38-39` | PROD_COMMIT 596ba1c, PROD_TAG |
| PP4-PROMOTE (offload/pp4-promote 0f0d61e) | `pp4_promote_switch.py` edit_env | expects production's remote_dspark mount source; on a 0075 production it would find `up2/<id>/remote_dspark` and refuse (correct: PP4 then needs `pp4-composed`, which is unmeasured). Order: promote one, then re-qualify the other on top. |
| TV1 (25faa84) | `offload/tv1/production.py:14-15` | PROD_COMMIT, PROD_TAG |
| PC1 (a3aa2a7 / Round3 44081bd) | `offload/pc1/build.py:12-13`, `offload/pc1/inputs.json`, `offload/pc1/budget.json` | PRODUCTION, TAG, pinned production bytes (remote package hashes change) |
| UP3 run7 (ac23d4f) and this lane | `offload/up3/build.py:16-17`, `inputs.json`, `manifests/*.json`, `qualification.lock.json`, `budget.json`, `window.env` (KV/cap/K1 check; replay arm removes production's remote mount), `up3_window.py` plan text | PRODUCTION, TAG, every production byte pin; the `replay` group becomes obsolete (production is the arm) |
| CC1 (8cdf7a4) | `offload/launch/windows/cc1_window.py` via pf1 pins; ops1 constants in `nbw_window.py` imports | as the shared pins above |
| PROMO1 runner (4740ce9) | `offload/promote/k1/promo1_window.py` `BASE`, `OLD_SHA`, `materialize_repin.py`, `repin/*.json`; `dsv41.env.327680-low.patch` | BASE 596ba1c, and the low patch carries `OFFLOAD_DOCKER_ARGS` as CONTEXT: on a 0075 production `git apply` refuses (correct, fail-closed). **Order: PROMO1 first, then UP2** (this switch is tested on PROMO1's low env); UP2 first means regenerating PROMO1's patch on the 0075 env and a re-review. |
| NBW (5a6a65e) | `offload/launch/windows/nbw_window.py:32-33, 675` | PROD_COMMIT, PROD_TAG, plan text |
| Maintenance | `offload/launch/maintenance/ops1_apply.sh:8-21`, `offload/operations/profiles.v1.json:70,72,349,446,508` | OPS1 tag/commit and the production remote mount path |
| WIN2 forks in progress | offload/{nbw,pc1,pp4,up3}-win2 | re-scan with the command below at promotion time |

Re-scan command (local objects only):

```bash
git -C /opt/dsv41/repos/dsv41-deploy grep -n -E '596ba1c|prod-k1-327-ops1-20261026|W8/restore\.sh|greedy-post\.jsonl|K1-post|dsv41-offload-prod/dspark/remote_dspark|VLLM_DSV41_REPLAY_SKIP|replay_guard' <ref> -- '*.py' '*.sh' '*.env' '*.json' ':!**/tests/**' ':!**/test_*' ':!documents/**'
```

## Generated inventory (code files only; tests follow their runner)

### offload-x86host @ 93980b1
- `offload/dspark/d2/replay_window.py`: W8 (lines 5, 343, 413)
- `offload/dspark/d2/smoke_prepare.py`: greedy refs (lines 22, 86)
- `offload/dspark/d2/smoke_window.py`: W8 (lines 30)
- `offload/launch/configs/D-kv6-async-up1-all.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-control.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-core.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-cpu-pin.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-dual-ring.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-k5.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-replay.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-split.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-vor.env`: replay flag (lines 36)
- `offload/launch/maintenance/fw1_sparkhead.sh`: W8, greedy refs (lines 11, 137, 139, 140, 142)
- `offload/launch/maintenance/fw2_sparkworker.sh`: W8, greedy refs (lines 13, 137, 139, 140, 142)
- `offload/launch/maintenance/fw3_sparkworker_firmware.sh`: W8, greedy refs (lines 15, 130, 132, 133, 135)
- `offload/launch/maintenance/fw3b_sparkworker_ec_pd.sh`: W8, greedy refs (lines 20, 154, 156, 157, 159)
- `offload/launch/maintenance/ops1_apply.sh`: W8, commit/tag, greedy refs (lines 8, 9, 11, 12, 14, 21, 56, 61, 62, 98)
- `offload/launch/start_exp.sh`: replay flag (lines 15, 16)
- `offload/launch/windows/kl1_window.py`: W8, greedy refs (lines 352, 353)
- `offload/launch/windows/nbw_cells.py`: greedy refs (lines 24)
- `offload/launch/windows/nbw_window.py`: W8, commit/tag (lines 32, 33, 675, 685)
- `offload/launch/windows/pf1_window.py`: W8, greedy refs (lines 39, 454, 799)
- `offload/launch/windows/pf2_window.py`: greedy refs (lines 527)
- `offload/launch/windows/post_close_watch.py`: W8, greedy refs (lines 39, 55, 57)
- `offload/launch/windows/sp1_window.py`: greedy refs (lines 394)
- `offload/launch/windows/up2_checks.py`: replay flag (lines 261)
- `offload/launch/windows/up2_low.py`: W8, greedy refs, replay flag (lines 26, 340, 652)
- `offload/operations/profiles.v1.json`: commit/tag, remote mount (lines 70, 72, 349, 446, 508)
- `offload/upstream_adopt/catalogue.json`: replay flag (lines 312)
- `offload/upstream_adopt/make_configs.py`: replay flag (lines 44)
- `offload/upstream_adopt/overlay/offload/dspark/spark/remote_dspark/roundcore.py`: replay flag (lines 36)
- `offload/upstream_adopt/overlay/vllm/models/deepseek_v41/replay_skip.py`: replay flag (lines 22, 33)
- `offload/upstream_adopt/overlay/vllm/v1/core/sched/scheduler.py`: replay flag (lines 103)

### offload/pp4 (stage7 941a3c3) @ 941a3c3
- `offload/dspark/d2/replay_window.py`: W8 (lines 5, 343, 413)
- `offload/dspark/d2/smoke_prepare.py`: greedy refs (lines 22, 86)
- `offload/dspark/d2/smoke_window.py`: W8 (lines 30)
- `offload/launch/configs/D-kv6-async-up1-all.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-control.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-core.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-cpu-pin.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-dual-ring.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-k5.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-replay.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-split.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-vor.env`: replay flag (lines 36)
- `offload/launch/maintenance/fw1_sparkhead.sh`: W8, greedy refs (lines 11, 137, 139, 140, 142)
- `offload/launch/maintenance/fw2_sparkworker.sh`: W8, greedy refs (lines 13, 137, 139, 140, 142)
- `offload/launch/maintenance/fw3_sparkworker_firmware.sh`: W8, greedy refs (lines 15, 130, 132, 133, 135)
- `offload/launch/maintenance/fw3b_sparkworker_ec_pd.sh`: W8, greedy refs (lines 20, 154, 156, 157, 159)
- `offload/launch/maintenance/ops1_apply.sh`: W8, commit/tag, greedy refs (lines 8, 9, 11, 12, 14, 21, 56, 61, 62, 98)
- `offload/launch/start_exp.sh`: replay flag (lines 15, 16)
- `offload/launch/windows/kl1_window.py`: W8, greedy refs (lines 352, 353)
- `offload/launch/windows/nbw_cells.py`: greedy refs (lines 24)
- `offload/launch/windows/nbw_window.py`: W8, commit/tag (lines 32, 33, 675, 685)
- `offload/launch/windows/pf1_window.py`: W8, greedy refs (lines 39, 454, 799)
- `offload/launch/windows/pf2_window.py`: greedy refs (lines 527)
- `offload/launch/windows/post_close_watch.py`: W8, greedy refs (lines 39, 55, 57)
- `offload/launch/windows/pp4_cells.py`: greedy refs (lines 29)
- `offload/launch/windows/pp4_window.py`: commit/tag (lines 38, 39)
- `offload/launch/windows/sp1_window.py`: greedy refs (lines 394)
- `offload/launch/windows/up2_checks.py`: replay flag (lines 261)
- `offload/launch/windows/up2_low.py`: W8, greedy refs, replay flag (lines 26, 340, 652)
- `offload/operations/profiles.v1.json`: commit/tag, remote mount (lines 70, 72, 349, 446, 508)
- `offload/upstream_adopt/catalogue.json`: replay flag (lines 312)
- `offload/upstream_adopt/make_configs.py`: replay flag (lines 44)
- `offload/upstream_adopt/overlay/offload/dspark/spark/remote_dspark/roundcore.py`: replay flag (lines 36)
- `offload/upstream_adopt/overlay/vllm/models/deepseek_v41/replay_skip.py`: replay flag (lines 22, 33)
- `offload/upstream_adopt/overlay/vllm/v1/core/sched/scheduler.py`: replay flag (lines 103)

### offload/tv1 @ 25faa84
- `offload/dspark/d2/replay_window.py`: W8 (lines 5, 343, 413)
- `offload/dspark/d2/smoke_prepare.py`: greedy refs (lines 22, 86)
- `offload/dspark/d2/smoke_window.py`: W8 (lines 30)
- `offload/launch/configs/D-kv6-async-up1-all.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-control.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-core.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-cpu-pin.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-dual-ring.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-k5.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-replay.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-split.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-vor.env`: replay flag (lines 36)
- `offload/launch/maintenance/fw1_sparkhead.sh`: W8, greedy refs (lines 11, 137, 139, 140, 142)
- `offload/launch/maintenance/fw2_sparkworker.sh`: W8, greedy refs (lines 13, 137, 139, 140, 142)
- `offload/launch/maintenance/fw3_sparkworker_firmware.sh`: W8, greedy refs (lines 15, 130, 132, 133, 135)
- `offload/launch/maintenance/fw3b_sparkworker_ec_pd.sh`: W8, greedy refs (lines 20, 154, 156, 157, 159)
- `offload/launch/maintenance/ops1_apply.sh`: W8, commit/tag, greedy refs (lines 8, 9, 11, 12, 14, 21, 56, 61, 62, 98)
- `offload/launch/start_exp.sh`: replay flag (lines 15, 16)
- `offload/launch/windows/kl1_window.py`: W8, greedy refs (lines 352, 353)
- `offload/launch/windows/nbw_cells.py`: greedy refs (lines 24)
- `offload/launch/windows/nbw_window.py`: W8, commit/tag (lines 32, 33, 675, 685)
- `offload/launch/windows/pf1_window.py`: W8, greedy refs (lines 39, 454, 799)
- `offload/launch/windows/pf2_window.py`: greedy refs (lines 527)
- `offload/launch/windows/post_close_watch.py`: W8, greedy refs (lines 39, 55, 57)
- `offload/launch/windows/sp1_window.py`: greedy refs (lines 394)
- `offload/launch/windows/tv1_window.py`: W8 (lines 692)
- `offload/launch/windows/up2_checks.py`: replay flag (lines 261)
- `offload/launch/windows/up2_low.py`: W8, greedy refs, replay flag (lines 26, 340, 652)
- `offload/operations/profiles.v1.json`: commit/tag, remote mount (lines 70, 72, 349, 446, 508)
- `offload/tv1/greedy-references.json`: greedy refs (lines 2)
- `offload/tv1/production.py`: commit/tag (lines 14, 15)
- `offload/upstream_adopt/catalogue.json`: replay flag (lines 312)
- `offload/upstream_adopt/make_configs.py`: replay flag (lines 44)
- `offload/upstream_adopt/overlay/offload/dspark/spark/remote_dspark/roundcore.py`: replay flag (lines 36)
- `offload/upstream_adopt/overlay/vllm/models/deepseek_v41/replay_skip.py`: replay flag (lines 22, 33)
- `offload/upstream_adopt/overlay/vllm/v1/core/sched/scheduler.py`: replay flag (lines 103)

### offload/pc1 @ a3aa2a7
- `offload/dspark/d2/replay_window.py`: W8 (lines 5, 343, 413)
- `offload/dspark/d2/smoke_prepare.py`: greedy refs (lines 22, 86)
- `offload/dspark/d2/smoke_window.py`: W8 (lines 30)
- `offload/launch/configs/D-kv6-async-up1-all.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-control.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-core.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-cpu-pin.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-dual-ring.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-k5.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-replay.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-split.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-vor.env`: replay flag (lines 36)
- `offload/launch/maintenance/fw1_sparkhead.sh`: W8, greedy refs (lines 11, 137, 139, 140, 142)
- `offload/launch/maintenance/fw2_sparkworker.sh`: W8, greedy refs (lines 13, 137, 139, 140, 142)
- `offload/launch/maintenance/fw3_sparkworker_firmware.sh`: W8, greedy refs (lines 15, 130, 132, 133, 135)
- `offload/launch/maintenance/fw3b_sparkworker_ec_pd.sh`: W8, greedy refs (lines 20, 154, 156, 157, 159)
- `offload/launch/maintenance/ops1_apply.sh`: W8, commit/tag, greedy refs (lines 8, 9, 11, 12, 14, 21, 56, 61, 62, 98)
- `offload/launch/start_exp.sh`: replay flag (lines 15, 16)
- `offload/launch/windows/kl1_window.py`: W8, greedy refs (lines 352, 353)
- `offload/launch/windows/nbw_cells.py`: greedy refs (lines 24)
- `offload/launch/windows/nbw_window.py`: W8, commit/tag (lines 32, 33, 675, 685)
- `offload/launch/windows/pf1_window.py`: W8, greedy refs (lines 39, 454, 799)
- `offload/launch/windows/pf2_window.py`: greedy refs (lines 527)
- `offload/launch/windows/post_close_watch.py`: W8, greedy refs (lines 39, 55, 57)
- `offload/launch/windows/sp1_window.py`: greedy refs (lines 394)
- `offload/launch/windows/up2_checks.py`: replay flag (lines 261)
- `offload/launch/windows/up2_low.py`: W8, greedy refs, replay flag (lines 26, 340, 652)
- `offload/operations/profiles.v1.json`: commit/tag, remote mount (lines 70, 72, 349, 446, 508)
- `offload/pc1/budget.json`: commit/tag (lines 3)
- `offload/pc1/build.py`: commit/tag (lines 12, 13)
- `offload/pc1/inputs.json`: commit/tag (lines 2)
- `offload/pc1/window.py`: greedy refs (lines 561, 562)
- `offload/upstream_adopt/catalogue.json`: replay flag (lines 312)
- `offload/upstream_adopt/make_configs.py`: replay flag (lines 44)
- `offload/upstream_adopt/overlay/offload/dspark/spark/remote_dspark/roundcore.py`: replay flag (lines 36)
- `offload/upstream_adopt/overlay/vllm/models/deepseek_v41/replay_skip.py`: replay flag (lines 22, 33)
- `offload/upstream_adopt/overlay/vllm/v1/core/sched/scheduler.py`: replay flag (lines 103)

### offload/up3 (run7) @ ac23d4f
- `offload/dspark/d2/replay_window.py`: W8 (lines 5, 343, 413)
- `offload/dspark/d2/smoke_prepare.py`: greedy refs (lines 22, 86)
- `offload/dspark/d2/smoke_window.py`: W8 (lines 30)
- `offload/launch/configs/D-kv6-async-up1-all.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-control.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-core.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-cpu-pin.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-dual-ring.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-k5.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-replay.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-split.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-vor.env`: replay flag (lines 36)
- `offload/launch/maintenance/fw1_sparkhead.sh`: W8, greedy refs (lines 11, 137, 139, 140, 142)
- `offload/launch/maintenance/fw2_sparkworker.sh`: W8, greedy refs (lines 13, 137, 139, 140, 142)
- `offload/launch/maintenance/fw3_sparkworker_firmware.sh`: W8, greedy refs (lines 15, 130, 132, 133, 135)
- `offload/launch/maintenance/fw3b_sparkworker_ec_pd.sh`: W8, greedy refs (lines 20, 154, 156, 157, 159)
- `offload/launch/maintenance/ops1_apply.sh`: W8, commit/tag, greedy refs (lines 8, 9, 11, 12, 14, 21, 56, 61, 62, 98)
- `offload/launch/start_exp.sh`: replay flag (lines 15, 16)
- `offload/launch/windows/kl1_window.py`: W8, greedy refs (lines 352, 353)
- `offload/launch/windows/nbw_cells.py`: greedy refs (lines 24)
- `offload/launch/windows/nbw_window.py`: W8, commit/tag (lines 32, 33, 675, 685)
- `offload/launch/windows/pf1_window.py`: W8, greedy refs (lines 39, 454, 799)
- `offload/launch/windows/pf2_window.py`: greedy refs (lines 527)
- `offload/launch/windows/post_close_watch.py`: W8, greedy refs (lines 39, 55, 57)
- `offload/launch/windows/sp1_window.py`: greedy refs (lines 394)
- `offload/launch/windows/up2_checks.py`: replay flag (lines 261)
- `offload/launch/windows/up2_low.py`: W8, greedy refs, replay flag (lines 26, 340, 652)
- `offload/launch/windows/up3_window.py`: commit/tag, greedy refs (lines 579, 580, 663)
- `offload/operations/profiles.v1.json`: commit/tag, remote mount (lines 70, 72, 349, 446, 508)
- `offload/up3/budget.json`: commit/tag (lines 3)
- `offload/up3/build.py`: commit/tag (lines 16, 17)
- `offload/up3/host_gates.py`: commit/tag (lines 35)
- `offload/up3/inputs.json`: commit/tag (lines 2)
- `offload/up3/manifests/control.json`: commit/tag (lines 2)
- `offload/up3/manifests/cpu-pin.json`: commit/tag (lines 2)
- `offload/up3/manifests/dual-ring.json`: commit/tag (lines 2)
- `offload/up3/manifests/k5.json`: commit/tag (lines 2)
- `offload/up3/manifests/split.json`: commit/tag (lines 2)
- `offload/up3/manifests/vor.json`: commit/tag (lines 2)
- `offload/up3/proc_census.py`: commit/tag (lines 19)
- `offload/up3/qualification.lock.json`: commit/tag (lines 3)
- `offload/upstream_adopt/catalogue.json`: replay flag (lines 312)
- `offload/upstream_adopt/make_configs.py`: replay flag (lines 44)
- `offload/upstream_adopt/overlay/offload/dspark/spark/remote_dspark/roundcore.py`: replay flag (lines 36)
- `offload/upstream_adopt/overlay/vllm/models/deepseek_v41/replay_skip.py`: replay flag (lines 22, 33)
- `offload/upstream_adopt/overlay/vllm/v1/core/sched/scheduler.py`: replay flag (lines 103)

### offload/cc1 @ 8cdf7a4
- `offload/dspark/d2/replay_window.py`: W8 (lines 5, 343, 413)
- `offload/dspark/d2/smoke_prepare.py`: greedy refs (lines 22, 86)
- `offload/dspark/d2/smoke_window.py`: W8 (lines 30)
- `offload/launch/configs/D-kv6-async-up1-all.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-control.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-core.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-cpu-pin.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-dual-ring.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-k5.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-replay.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-split.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-vor.env`: replay flag (lines 36)
- `offload/launch/maintenance/fw1_sparkhead.sh`: W8, greedy refs (lines 11, 137, 139, 140, 142)
- `offload/launch/maintenance/fw2_sparkworker.sh`: W8, greedy refs (lines 13, 137, 139, 140, 142)
- `offload/launch/maintenance/fw3_sparkworker_firmware.sh`: W8, greedy refs (lines 15, 130, 132, 133, 135)
- `offload/launch/maintenance/fw3b_sparkworker_ec_pd.sh`: W8, greedy refs (lines 20, 154, 156, 157, 159)
- `offload/launch/maintenance/ops1_apply.sh`: W8, commit/tag, greedy refs (lines 8, 9, 11, 12, 14, 21, 56, 61, 62, 98)
- `offload/launch/start_exp.sh`: replay flag (lines 15, 16)
- `offload/launch/windows/kl1_window.py`: W8, greedy refs (lines 352, 353)
- `offload/launch/windows/nbw_cells.py`: greedy refs (lines 24)
- `offload/launch/windows/nbw_window.py`: W8, commit/tag (lines 32, 33, 683, 693)
- `offload/launch/windows/pf1_window.py`: W8, greedy refs (lines 39, 454, 802)
- `offload/launch/windows/pf2_window.py`: greedy refs (lines 527)
- `offload/launch/windows/post_close_watch.py`: W8, greedy refs (lines 39, 55, 57)
- `offload/launch/windows/sp1_window.py`: greedy refs (lines 394)
- `offload/launch/windows/up2_checks.py`: replay flag (lines 261)
- `offload/launch/windows/up2_low.py`: W8, greedy refs, replay flag (lines 26, 340, 652)
- `offload/operations/profiles.v1.json`: commit/tag, remote mount (lines 70, 72, 349, 446, 508)
- `offload/upstream_adopt/catalogue.json`: replay flag (lines 312)
- `offload/upstream_adopt/make_configs.py`: replay flag (lines 44)
- `offload/upstream_adopt/overlay/offload/dspark/spark/remote_dspark/roundcore.py`: replay flag (lines 36)
- `offload/upstream_adopt/overlay/vllm/models/deepseek_v41/replay_skip.py`: replay flag (lines 22, 33)
- `offload/upstream_adopt/overlay/vllm/v1/core/sched/scheduler.py`: replay flag (lines 103)

### offload/promo1 runner @ 4740ce9
- `offload/dspark/d2/replay_window.py`: W8 (lines 5, 343, 413)
- `offload/dspark/d2/smoke_prepare.py`: greedy refs (lines 22, 86)
- `offload/dspark/d2/smoke_window.py`: W8 (lines 30)
- `offload/launch/configs/D-kv6-async-up1-all.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-control.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-core.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-cpu-pin.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-dual-ring.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-k5.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-replay.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-split.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-vor.env`: replay flag (lines 36)
- `offload/launch/maintenance/fw1_sparkhead.sh`: W8, greedy refs (lines 11, 137, 139, 140, 142)
- `offload/launch/maintenance/fw2_sparkworker.sh`: W8, greedy refs (lines 13, 137, 139, 140, 142)
- `offload/launch/maintenance/fw3_sparkworker_firmware.sh`: W8, greedy refs (lines 15, 130, 132, 133, 135)
- `offload/launch/maintenance/fw3b_sparkworker_ec_pd.sh`: W8, greedy refs (lines 20, 154, 156, 157, 159)
- `offload/launch/maintenance/ops1_apply.sh`: W8, commit/tag, greedy refs (lines 8, 9, 11, 12, 14, 21, 56, 61, 62, 98)
- `offload/launch/start_exp.sh`: replay flag (lines 15, 16)
- `offload/launch/windows/kl1_window.py`: W8, greedy refs (lines 352, 353)
- `offload/launch/windows/nbw_cells.py`: greedy refs (lines 24)
- `offload/launch/windows/nbw_window.py`: W8, commit/tag (lines 32, 33, 675, 685)
- `offload/launch/windows/pf1_window.py`: W8, greedy refs (lines 39, 454, 799)
- `offload/launch/windows/pf2_window.py`: greedy refs (lines 527)
- `offload/launch/windows/post_close_watch.py`: W8, greedy refs (lines 39, 55, 57)
- `offload/launch/windows/sp1_window.py`: greedy refs (lines 394)
- `offload/launch/windows/up2_checks.py`: replay flag (lines 261)
- `offload/launch/windows/up2_low.py`: W8, greedy refs, replay flag (lines 26, 340, 652)
- `offload/operations/profiles.v1.json`: commit/tag, remote mount (lines 70, 72, 349, 446, 508)
- `offload/promote/k1/materialize_repin.py`: W8, commit/tag (lines 26, 156, 220)
- `offload/promote/k1/promo1_window.py`: W8, commit/tag, greedy refs (lines 38, 39, 41, 50, 299)
- `offload/promote/k1/restore-327low.sh`: W8, greedy refs (lines 37, 46, 56)
- `offload/upstream_adopt/catalogue.json`: replay flag (lines 312)
- `offload/upstream_adopt/make_configs.py`: replay flag (lines 44)
- `offload/upstream_adopt/overlay/offload/dspark/spark/remote_dspark/roundcore.py`: replay flag (lines 36)
- `offload/upstream_adopt/overlay/vllm/models/deepseek_v41/replay_skip.py`: replay flag (lines 22, 33)
- `offload/upstream_adopt/overlay/vllm/v1/core/sched/scheduler.py`: replay flag (lines 103)

### offload/nbw @ 5a6a65e
- `offload/dspark/d2/replay_window.py`: W8 (lines 5, 343, 413)
- `offload/dspark/d2/smoke_prepare.py`: greedy refs (lines 22, 86)
- `offload/dspark/d2/smoke_window.py`: W8 (lines 30)
- `offload/launch/configs/D-kv6-async-up1-all.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-control.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-core.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-cpu-pin.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-dual-ring.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-k5.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-replay.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-split.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-vor.env`: replay flag (lines 36)
- `offload/launch/maintenance/fw1_sparkhead.sh`: W8, greedy refs (lines 11, 137, 139, 140, 142)
- `offload/launch/maintenance/fw2_sparkworker.sh`: W8, greedy refs (lines 13, 137, 139, 140, 142)
- `offload/launch/maintenance/ops1_apply.sh`: W8, commit/tag, greedy refs (lines 8, 9, 11, 12, 14, 21, 56, 61, 62, 98)
- `offload/launch/start_exp.sh`: replay flag (lines 15, 16)
- `offload/launch/windows/kl1_window.py`: W8, greedy refs (lines 352, 353)
- `offload/launch/windows/nbw_cells.py`: greedy refs (lines 24)
- `offload/launch/windows/nbw_window.py`: W8, commit/tag (lines 28, 29, 558, 568)
- `offload/launch/windows/pf1_window.py`: W8, greedy refs (lines 39, 454, 799)
- `offload/launch/windows/pf2_window.py`: greedy refs (lines 527)
- `offload/launch/windows/sp1_window.py`: greedy refs (lines 394)
- `offload/launch/windows/up2_checks.py`: replay flag (lines 261)
- `offload/launch/windows/up2_low.py`: W8, greedy refs, replay flag (lines 26, 340, 652)
- `offload/operations/profiles.v1.json`: commit/tag, remote mount (lines 70, 72, 349, 446, 508)
- `offload/upstream_adopt/catalogue.json`: replay flag (lines 312)
- `offload/upstream_adopt/make_configs.py`: replay flag (lines 44)
- `offload/upstream_adopt/overlay/offload/dspark/spark/remote_dspark/roundcore.py`: replay flag (lines 36)
- `offload/upstream_adopt/overlay/vllm/models/deepseek_v41/replay_skip.py`: replay flag (lines 22, 33)
- `offload/upstream_adopt/overlay/vllm/v1/core/sched/scheduler.py`: replay flag (lines 103)

### offload/pp4-promote @ 6275221
- `offload/dspark/d2/replay_window.py`: W8 (lines 5, 343, 413)
- `offload/dspark/d2/smoke_prepare.py`: greedy refs (lines 22, 86)
- `offload/dspark/d2/smoke_window.py`: W8 (lines 30)
- `offload/launch/configs/D-kv6-async-up1-all.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-control.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-core.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-cpu-pin.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-dual-ring.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-k5.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-replay.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-split.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-vor.env`: replay flag (lines 36)
- `offload/launch/maintenance/fw1_sparkhead.sh`: W8, greedy refs (lines 11, 137, 139, 140, 142)
- `offload/launch/maintenance/fw2_sparkworker.sh`: W8, greedy refs (lines 13, 137, 139, 140, 142)
- `offload/launch/maintenance/fw3_sparkworker_firmware.sh`: W8, greedy refs (lines 15, 130, 132, 133, 135)
- `offload/launch/maintenance/fw3b_sparkworker_ec_pd.sh`: W8, greedy refs (lines 20, 154, 156, 157, 159)
- `offload/launch/maintenance/ops1_apply.sh`: W8, commit/tag, greedy refs (lines 8, 9, 11, 12, 14, 21, 56, 61, 62, 98)
- `offload/launch/start_exp.sh`: replay flag (lines 15, 16)
- `offload/launch/windows/kl1_window.py`: W8, greedy refs (lines 352, 353)
- `offload/launch/windows/nbw_cells.py`: greedy refs (lines 24)
- `offload/launch/windows/nbw_window.py`: W8, commit/tag (lines 32, 33, 675, 685)
- `offload/launch/windows/pf1_window.py`: W8, greedy refs (lines 39, 454, 799)
- `offload/launch/windows/pf2_window.py`: greedy refs (lines 527)
- `offload/launch/windows/post_close_watch.py`: W8, greedy refs (lines 39, 55, 57)
- `offload/launch/windows/pp4_cells.py`: greedy refs (lines 29)
- `offload/launch/windows/pp4_promote_switch.py`: commit/tag (lines 72)
- `offload/launch/windows/pp4_promote_window.py`: commit/tag, greedy refs (lines 3, 6, 45, 474)
- `offload/launch/windows/pp4_window.py`: commit/tag (lines 38, 39)
- `offload/launch/windows/restore-pp4.sh`: W8, greedy refs (lines 3, 5, 6, 58)
- `offload/launch/windows/sp1_window.py`: greedy refs (lines 394)
- `offload/launch/windows/up2_checks.py`: replay flag (lines 261)
- `offload/launch/windows/up2_low.py`: W8, greedy refs, replay flag (lines 26, 340, 652)
- `offload/operations/profiles.v1.json`: commit/tag, remote mount (lines 70, 72, 349, 446, 508)
- `offload/upstream_adopt/catalogue.json`: replay flag (lines 312)
- `offload/upstream_adopt/make_configs.py`: replay flag (lines 44)
- `offload/upstream_adopt/overlay/offload/dspark/spark/remote_dspark/roundcore.py`: replay flag (lines 36)
- `offload/upstream_adopt/overlay/vllm/models/deepseek_v41/replay_skip.py`: replay flag (lines 22, 33)
- `offload/upstream_adopt/overlay/vllm/v1/core/sched/scheduler.py`: replay flag (lines 103)

### offload/up2-adopt (this lane) @ 889841b
- `offload/dspark/d2/replay_window.py`: W8 (lines 5, 343, 413)
- `offload/dspark/d2/smoke_prepare.py`: greedy refs (lines 22, 86)
- `offload/dspark/d2/smoke_window.py`: W8 (lines 30)
- `offload/launch/configs/D-kv6-async-up1-all.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-control.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-core.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-cpu-pin.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-dual-ring.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-k5.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-replay.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-split.env`: replay flag (lines 36)
- `offload/launch/configs/D-kv6-async-up1-vor.env`: replay flag (lines 36)
- `offload/launch/maintenance/fw1_sparkhead.sh`: W8, greedy refs (lines 11, 137, 139, 140, 142)
- `offload/launch/maintenance/fw2_sparkworker.sh`: W8, greedy refs (lines 13, 137, 139, 140, 142)
- `offload/launch/maintenance/fw3_sparkworker_firmware.sh`: W8, greedy refs (lines 15, 130, 132, 133, 135)
- `offload/launch/maintenance/fw3b_sparkworker_ec_pd.sh`: W8, greedy refs (lines 20, 154, 156, 157, 159)
- `offload/launch/maintenance/ops1_apply.sh`: W8, commit/tag, greedy refs (lines 8, 9, 11, 12, 14, 21, 56, 61, 62, 98)
- `offload/launch/start_exp.sh`: replay flag (lines 15, 16)
- `offload/launch/windows/kl1_window.py`: W8, greedy refs (lines 352, 353)
- `offload/launch/windows/nbw_cells.py`: greedy refs (lines 24)
- `offload/launch/windows/nbw_window.py`: W8, commit/tag (lines 32, 33, 675, 685)
- `offload/launch/windows/pf1_window.py`: W8, greedy refs (lines 39, 454, 799)
- `offload/launch/windows/pf2_window.py`: greedy refs (lines 527)
- `offload/launch/windows/post_close_watch.py`: W8, greedy refs (lines 39, 55, 57)
- `offload/launch/windows/sp1_window.py`: greedy refs (lines 394)
- `offload/launch/windows/up2_checks.py`: replay flag (lines 261)
- `offload/launch/windows/up2_low.py`: W8, greedy refs, replay flag (lines 26, 340, 652)
- `offload/launch/windows/up3_window.py`: commit/tag, greedy refs (lines 579, 580, 663)
- `offload/operations/profiles.v1.json`: commit/tag, remote mount (lines 70, 72, 349, 446, 508)
- `offload/up3/budget.json`: commit/tag (lines 3)
- `offload/up3/build.py`: commit/tag (lines 16, 17)
- `offload/up3/host_gates.py`: commit/tag (lines 35)
- `offload/up3/inputs.json`: commit/tag (lines 2)
- `offload/up3/manifests/control.json`: commit/tag (lines 2)
- `offload/up3/manifests/cpu-pin.json`: commit/tag (lines 2)
- `offload/up3/manifests/dual-ring.json`: commit/tag (lines 2)
- `offload/up3/manifests/k5.json`: commit/tag (lines 2)
- `offload/up3/manifests/split.json`: commit/tag (lines 2)
- `offload/up3/manifests/vor.json`: commit/tag (lines 2)
- `offload/up3/proc_census.py`: commit/tag (lines 19)
- `offload/up3/qualification.lock.json`: commit/tag (lines 3)
- `offload/upstream_adopt/catalogue.json`: replay flag (lines 312)
- `offload/upstream_adopt/make_configs.py`: replay flag (lines 44)
- `offload/upstream_adopt/overlay/offload/dspark/spark/remote_dspark/roundcore.py`: replay flag (lines 36)
- `offload/upstream_adopt/overlay/vllm/models/deepseek_v41/replay_skip.py`: replay flag (lines 22, 33)
- `offload/upstream_adopt/overlay/vllm/v1/core/sched/scheduler.py`: replay flag (lines 103)

