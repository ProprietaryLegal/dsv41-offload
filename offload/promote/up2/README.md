# UP2-PROMOTE — 0075 replay skip into production (item 2 of UP2-ADOPT)

Modelled on PROMO1 (`offload/promote/k1`: switch, versioned restore, runner structure) and PP4-PROMOTE. Turns 0075 ON
through exactly the flag and bytes the qualification window measured (`offload/up3/UP2-ADOPT-WINDOW.md`). Nothing here
has touched a host; every path is CPU-tested.

| File | Role |
|---|---|
| `package.py`, `packages/*.json` | Content-addressed production package. `qualified` (id `ad3dc5be5eaf`) = the UP3 replay arm's 24 files; `pp4-composed` (id `25d875445e93`, 26 files) = PP4 941a3c3's remote package with 0075 applied, never measured. |
| `patches/0075-suffix-on-pp4-941a3c3.patch` | 0075-suffix re-anchored to PP4's `round.py` (PP4 added sampled temperature/seed parameters to `run()`); the same added/removed lines, different context (tested). |
| `switch.py` | Spark env transaction on sparkhead: `spark-apply` / `spark-restore` (byte-exact snapshot in `logs/up2-switch`, K1 dsv41.env hash rebased), `verify-running` (both ranks: four files, remote package and flag equal the staged package), `base`, `edit-env`. |
| `hooks.py` | Two inert production hooks: the K1 fallback never restores the pre-K1 backup over a 0075 production; `verify.sh` runs `switch.py verify-running` when `PROD_UP2` is set. Copies the switch and package manifests into the production tree. |
| `stage.sh` | Pre-HOLD staging of the immutable package to `~/dsv41-offload-prod/up2/<id>/` on both Sparks. |
| `restore-up2.sh` | Versioned W8 for a 0075 production (new references `K1-production/greedy-post-up2-<id>.jsonl`). |
| `promote_window.py` | The promotion runner (one owned HOLD; any failure restores the exact prior production and its W8). |
| `UP2-REPIN.md` | Item 3: runner pins that follow a PASS. |

Also changed for the promotion: `offload/upstream_adopt/replay_guard.py` gains `verify_production`. A 0075 production
exports `PROD_UP2`, so every experiment arm built from its env carries the flag. The guard then accepts exactly
production's staged scheduler, the three other 0075 files and the remote package, and refuses any experiment overlay on
a 0075 path. `offload/launch/windows/{post_close_watch,scoped_remote}.py` are byte-identical to PROMO1's
(`restore_sha256`, `close_owner`), as PP4-PROMOTE also carries them.

## Composition (explicit; anything else refuses)

Order: image v5 → K1 overlay (13 files) → PROMO1 low profile (dsv41.env only) → PP4 (remote package + runner files) →
0075. Shared files: none with K1 or PROMO1; with PP4 the remote package directory, where gather/round/roundcore/
speculator carry both changes and suffix_provenance.py is byte-identical.

| Live production | Switch | Why |
|---|---|---|
| ops1 596ba1c (K1 327680) | applies `qualified` | the measured bytes |
| PROMO1 327680-low | applies `qualified` (switch only) | env-only overlap; tested on PROMO1's low env. The promotion runner refuses this base until PROMO1 carries the 420be24 fold (S97 FIX 4) |
| PP4 (`PROD_PP4`) | REFUSES | `pp4-composed` was never measured; needs its own qualification window first |
| anything else, or UP2 already applied | REFUSES | unknown combination |

PROMO1's own low patch carries `OFFLOAD_DOCKER_ARGS` as context, so PROMO1 cannot follow 0075 without a regenerated
patch. **PROMO1 → UP2** works for the switch only; the runner refuses PROMO1-low until PROMO1 carries the fold. PP4-PROMOTE's switch expects production's own remote mount,
so after UP2 it refuses (fail-closed) until PP4 is re-qualified on top.

## Coordinator sequence (after an ADOPT CANDIDATE qualification summary and independent review)

```bash
source ~/.config/ndm/paths.sh
export WINDOW_COORDINATOR=Deepseek/coordinator
# 1. Before any HOLD (apply order 420be24 -> hooks -> push sparkhead -> fold-clock -> W8 start; UP2-REPIN.md):
git -C /opt/dsv41/repos/dsv41-deploy worktree add --detach /opt/dsv41/repos/.worktrees/up2-prod "$LIVE_PRODUCTION_COMMIT"
git -C /opt/dsv41/repos/.worktrees/up2-prod cherry-pick 420be244a524b34a904a0d7cca0ec954057af00b   # unless PP4 already folded
PYTHONPATH=$PWD:$PWD/offload/dspark /home/operator/miniconda3/bin/python3 -m offload.promote.up2.hooks apply \
  --tree /opt/dsv41/repos/.worktrees/up2-prod      # commit the printed paths there
git -C /opt/dsv41/repos/.worktrees/up2-prod push sparkhead HEAD:main      # production remote only, never origin
ssh sparkhead 'cd ~/dsv41-deploy-up2a && echo "e722fc56913673e4ac23b9c19b64401f81c5115f5650863c309078be9715ee49  offload/launch/windows/pp4_promote_switch.py" | sha256sum -c --quiet && python3 offload/launch/windows/pp4_promote_switch.py fold-clock --root ~/dsv41-deploy'
#    (the staged UP2-ADOPT checkout; the reviewed fold tool's sha256 is checked before it writes the K1 record)
#    then W8 start/check on the folded production (restore.sh), in its own coordinator window
bash offload/promote/up2/stage.sh --execute qualified
# 2. Copy restore-up2.sh to $C/W8/restore-up2.sh (reviewed), then the promotion HOLD:
PYTHONPATH=$PWD:$PWD/offload/dspark /home/operator/miniconda3/bin/python3 -m offload.promote.up2.promote_window --execute \
  --production-commit "$HOOKED_PRODUCTION_COMMIT" --base ops1 \
  --rollback-restore /opt/dsv41/campaigns/DSV41-OFFLOAD/W8/restore.sh \
  --rollback-restore-sha256 d1c4479dbfbd90bc2036c7bf1561db5b564c2b162a7cde4a121e46bfb9f86a49 \
  --qualification-summary "$UP2A_OUT/summary.json" \
  --out /opt/dsv41/campaigns/DSV41-OFFLOAD/UP2-PROMOTE-W-$(date -u +%Y%m%dT%H%M%SZ)
```

PROMO1-low: refused until PROMO1's low patch is re-pinned onto the 420be24-folded env (S97 FIX 4: PROMO1's 2300 line is
PROMO1's own text, so `fold-clock` cannot write the ledger the runner's clock-fold preflight requires). `switch.py` keeps
its PROMO1-low support for that later re-pin.
ESTIMATED HOLD 63.6 min (starts and W8 MEASURED, the rest ESTIMATED); `--execute` is refused before any host action
when the estimate exceeds 85 min. A promotion never runs inside a chain.
