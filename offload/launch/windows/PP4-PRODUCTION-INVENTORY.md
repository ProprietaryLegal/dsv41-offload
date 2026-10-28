# PP4 production inventory — what promotion changes, from qualified bytes only

Written 2026-10-27 for PP4-PROMOTE (brief `documents/plans/lanes/PP4-PROMOTE.md`, NDM: "approve teh drafting fix" /
"promote it" / "1.1 nat ceiling for that"). Sources are this branch (from offload/pp4 941a3c3; re-pinned 2026-10-27 to
Round15 85b0d96, which changes only pp4_cells.py's near-tie locator, then Round16 90f0278, which changes only the
admission client seam) and production 596ba1c
(`git show 596ba1c:<path>`). No host was read; installed files on x86host/Sparks are reconciled by the runner's preflight.

## Read this first: the measurement does not compare PP4 with production

The PP4 windows compare **fixed5 greedy drafting (A) vs fixed5 sampled drafting (B)** with the adaptive policies OFF in
both arms (`PP4-sampled.env:24`, `pp4_window.py:100-102`; summary field `comparison: 'CONTROLLED fixed5 PP4 runtime OFF
vs ON; not production-exact'`, `pp4_window.py:856`). Production drafts with the 0040 schedule and the 0047 adaptive
verify window (`serve.sh:34` SPEC `[[1,2,5],[3,4,4]]`; `serve.sh:58` `VLLM_DSV41_ADAPTIVE_K=1` + AK settings, at 596ba1c).
A PP4 measurement win therefore does NOT establish a gain over production. The promotion window's same-HOLD control on
the live production (cells `ctl-short-b1`, `ctl-32k-b1`, `ctl-150k-b1`, two repetitions) is the only production
comparison, and the runner refuses to keep PP4 if any cell loses beyond max(2 %, control spread).

## Spark side (both ranks, serve container)

| Item | Production 596ba1c | PP4 production (qualified bytes) | Source |
|---|---|---|---|
| Verify window | SPEC `{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,2,5],[3,4,4]]}` hard-coded in `serve.sh:34` | `{"method":"dspark","num_speculative_tokens":5}` (fixed5) | `pp4_window.py:103` SPEC; `PP4-sampled.env:59` |
| Adaptive window / depth | `VLLM_DSV41_ADAPTIVE_K=1` + AK_* (`serve.sh:58`); depth policy/lookup/margin from `OFFLOAD_DSPARK_ENV` defaults | `VLLM_DSV41_ADAPTIVE_K=0 VLLM_DSPARK_DEPTH_POLICY=fixed:5 VLLM_DSPARK_DEPTH_FILE= VLLM_DSPARK_LOOKUP= VLLM_DSPARK_REMOTE_WANT_MARGIN=0` | `pp4_window.py:100-102`; `PP4-sampled.env:24` |
| Sampled drafts | absent (greedy remote contract) | `VLLM_DSPARK_SAMPLED_TOP32=1` | `PP4-sampled.env:24`, `pp4_window.py:110` |
| Drafter identities | `VLLM_DSPARK_REMOTE_IDENTITY=be2c1436…`, `_MARKOV_IDENTITY=0b6e71e6…` (`dsv41.env:37`) | identical values (admission seed `worker_identity`/`markov_identity`) | `scratch/PP4-coord/admission-seed-*.json` |
| Round trace | absent | NOT in production: `VLLM_PP4_ROUND_DIR` unset keeps `dsv41_pp4_round_trace` a no-op (`pp4_round_trace.py:1-3`, `begin()` returns without the env) | `pp4_round_trace.py` |
| remote_dspark | staged from the checkout by `offload/dspark/spark/stage_overlay.sh` into `~/dsv41-offload-prod/dspark/remote_dspark` at every start (`start.sh:26`; the stage `rm -rf`s that directory, `stage_overlay.sh:60`) and mounted (`dsv41.env:39`) | 90f0278's (= 941a3c3's) `offload/dspark/spark/remote_dspark/*.py` (adds `sampled_wire.py`, `probability_buffers.py`, `suffix_provenance.py`; changes `wire.py`, `round.py`, `roundcore.py`, `speculator.py`, `shmregion.py`, `proxy_proc.py`), staged to a SEPARATE immutable directory so `stage_overlay.sh`'s `rm -rf` cannot touch it and a rollback (env restore) mounts production's own package again | `git diff 596ba1c 941a3c3 -- offload/dspark/spark/remote_dspark` |
| model_runner.py | K1 overlay (`dsv41.env:39`, kvpack 559c5867ba42, sha 682eb895…) | ordinary PP4 verifier runner = K1 runner + PP3 probability mask + PP4 timing anchors, sha `e0a11b89…` (`pp4_prepare.py:40-71`; seed `verifier_sha256`) | `pp4_prepare.py`, admission seed |
| rejection_sampler_utils.py | image file | PP3 utilities, sha `1468ec72…` | same |
| dsv41_pp4_round_trace.py | absent | helper, sha `5b9165ad…` (inert without `VLLM_PP4_ROUND_DIR`) | same |
| Everything else | — | unchanged: image v5, weights, K1 overlay (other 12 files), Engram, ofl/hostflag, KV bytes/profile, clocks | `PP4-sampled.env:10-21,34-46` keep every other production mount/env |

Admission-only files (`dsv41_pp4_admission.py`, the admission verifier variant) are NEVER used in production.

### K1 start-guard couplings (new glue code is unavoidable; it is NOT qualified by any PP4 window)

The PP4 windows started arms with `offload/launch/start_exp.sh`, not production `start.sh`, so the K1 guard never saw
PP4. Production `start.sh` with `PROD_K1=1` runs `offload/promote/k1/start_checked.sh` and then `verify.sh`:
1. `manage.py verify` refuses any `dsv41.env` whose sha differs from `logs/k1-promotion/files.json` `promoted`
   (`manage.py:117-130`). The PP4 switch must rebase that hash (PROMO1's `switch_profile.py` pattern).
2. `verify.sh:12` → `kvpack/stage_overlay.sh verify-running` requires the running container's 13 K1 paths to equal the
   K1 manifest (`stage_overlay.sh:113-128`). PP4 replaces `model_runner.py` → this check FAILS on every PP4 start unless
   it is told the exact PP4 runner sha. Hook: `KVPACK_PP4_RUNNER_SHA256`, honoured only while
   `pp4_promote_switch.py active` validates the switch (S91 #4); inert when unset.
3. `start_checked.sh:16-31` fallback calls `manage.py restore` on ANY start failure, which restores the PRE-K1
   `dsv41.env`/`start.sh` backup. With the PP4 switch applied that is a wrong-version restore. PROMO1 added a delegation for
   its own transaction only. Hook: delegate while `dsv41.env` selects PP4 or the switch transaction is open; after an
   exact rollback (`logs/pp4-switch/restored.json`) the original K1 fallback applies again (S91 #4).
4. `serve.sh:34` hard-codes SPEC. Hook: `PROD_SPEC_OVERRIDE`, applied only with a validated active switch; an override
   without one refuses the start (S91 #4); inert when unset.
These four hooks are the only production-code edits; `pp4_promote_package.py` applies them by exact anchors to either
596ba1c or PROMO1's accepted tree and refuses anything else. They need an independent review before the window.

## X86host side (TP2 K2 drafter, `dsv41-dspark-worker-tp2-prod.service`)

| Item | Production (unit file `offload/dspark/worker/dsv41-dspark-worker-tp2.prod.service`) | PP4 production | Source |
|---|---|---|---|
| Code | `~/dsv41-offload/prod/dspark-worker/current -> fb68c52` (`:34-35`) | release `~/dsv41-offload/prod/dspark-worker/90f0278` via `worker/run/prod_tp2.sh stage` (git archive, `prod_tp2.sh:41-52`); `current` is NOT flipped (the one-card unit and train windows keep fb68c52) | PP4 worker tree = 4292855's, R11 probe-cleared |
| Interpreter | `~/dsv41-offload/o2/venv/bin/python` (`:35`) | `/home/operator/miniconda3/bin/python3` (CPython 3.13, torch 2.12.1+cu130, CUDA 13.0) | `pp4_prerequisites.py:23`, `pp4_window.py:378-380` |
| EXL3 runtime | o2 venv exllamav3 | `PYTHONPATH=<rt>/stock:<rt>/facc:<release>/offload/dspark`; qualified `.so` digests stock `6c77786a…`, facc `2582a363…` | `pp4_window.py:241`, seed `runtime_artifact_sha256` |
| P2P prebuilt | `%h/dsv41-offload/prod/dspark-worker/p2p` (`:31`) | `<rt>/p2p` with `08e1903b0a43feff/dspark_p2p.so` `92d1b8c0…` | `pp4_window.py:240` |
| Tune cache | `%h/dsv41-offload/prod/run/exl3_tune_cache_tp2.bin` (`:32`) | reviewed `exl3_tune_cache-reviewed.bin` `937e9633…` copied to a production path | `pp4_window.py:46-47` |
| Mode | default arm | `DSPARK_WORKER_ARM=k2-tp2 DSPARK_SAMPLED_TOP32=1 DSPARK_SOURCE_COMMIT=<full>`; graphs ON (R10 removed the sampled override) | `pp4_window.py:242-244`; X86host probe PP4-server-R11-20261027T132720Z |
| Unchanged | GPUs/UUIDs, `gpu_locked.sh` two-lock ExecStart boundary, `--tp 2 --root /opt/dsv41/x86host-data/dspark --ofl 10.202.28.1:18620 --max-model-len 1048576`, status file, log, Restart, `MemoryMax=48G`, libofl pin | same | `:27-39`, `pp4_window.py:246-251` |

Applied as ONE systemd drop-in (`dsv41-dspark-worker-tp2-prod.service.d/50-pp4.conf`) that resets `ExecStart=` and
`WorkingDirectory=` to the release and adds an `ExecStartPre` attestation (interpreter, torch, CUDA and the three
qualified `.so` digests). Rollback = remove exactly that drop-in (hash-checked), `daemon-reload`, restart through
`prod_window.sh`.

The PP4 runtime is copied out of `/opt/dsv41/x86host-data/pp4/source/runtime` into an immutable production directory
`~/dsv41-offload/prod/pp4-runtime/<id>/` with a full `SHA256SUMS`, so experiment scratch can never change production.

## Risks the promotion cannot remove

- **Shared interpreter.** Production would run on the user miniconda3 Python that other x86host work also uses. A package
  change there changes the drafter. The `ExecStartPre` attestation turns drift into a refused start (Spark then runs
  target-only rounds) rather than silent change. A dedicated frozen venv is the follow-up.
- **Train windows.** `prod_window.sh train` starts the ONE-CARD unit from `current` (fb68c52, greedy). Against a sampled
  Spark config that pairing is unqualified. Train windows are forbidden while PP4 is production (D2 is paused).
- **Greedy references change.** PP4 production reproduces DEP1-fixed2's flips on prompts 0–2 (margins 0.0/0.5/1.0 nats;
  byte-identical outputs). The old W8 (K1-post references) can never pass on PP4 production. The promotion records new
  references (`K1-production/greedy-post-pp4.jsonl`, three identical runs) and ships a versioned W8 that checks them.
- **No production-path start was ever measured.** Startup cost of the graphs-ON sampled worker at production start is
  estimated from the window, not measured on the unit.
</content>
</invoke>
<invoke name="Bash">
<parameter name="command">cd /opt/dsv41/repos/.worktrees/dsv41-pp4-promote; grep -n "SPEC\|COMMON_ENV\|comparison" offload/launch/windows/pp4_window.py | head -5; grep -n "SPEC_OVERRIDE\|EXP_DOCKER_ENV=\"\$EXP" offload/launch/windows/PP4-sampled.env | head; mkdir -p /opt/dsv41/campaigns/DSV41-OFFLOAD/scratch/PP4PROMO; cat > /opt/dsv41/campaigns/DSV41-OFFLOAD/PP4-PROMOTE.progress.md <<'EOF'
# PP4-PROMOTE progress (fork of Deepseek coordinator; brief b3da114)
- 2026-10-27 item 1 inventory written: offload/launch/windows/PP4-PRODUCTION-INVENTORY.md (K1-guard couplings found: env hash, verify-running model_runner sha, pre-K1 fallback restore, hard-coded SPEC; fixed5-vs-adaptive measurement gap).
EOF
echo ok