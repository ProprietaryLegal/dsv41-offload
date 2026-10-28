# PP4-PROMOTE re-pin note — what must be re-pinned after a PASS (nothing here is edited by this lane)

A PASS changes five things every existing window runner assumes about production. Until the affected runners are
re-pinned (PROMO1's `offload/promote/k1/materialize_repin.py` pattern, extended with the PP4 constants below), each one
must REFUSE at preflight, not run against PP4 production.

## The five changed facts

| Fact | Before | After PP4 PASS |
|---|---|---|
| Production commit / tag | 596ba1c `prod-k1-327-ops1-20261026` (or PROMO1's `prod-k1-327-low-*`) | the PP4-PROMOTE acceptance commit, tag `prod-pp4-YYYYMMDD` (`logs/pp4-switch/{accepted-commit,production-tag}` on sparkhead) |
| Rollback W8 | `$C/W8/restore.sh` sha d1c4479d… (`pf1_window.py` `RESTORE`/`RESTORE_SHA256`, `post_close_watch.py:39-40`) or PROMO1's `restore-327low.sh` | `$C/W8/restore-pp4.sh` (copy of `offload/launch/windows/restore-pp4.sh`, sha recorded at copy time) |
| Greedy references | `K1-production/greedy-post.jsonl` (K1-post) | `K1-production/greedy-post-pp4.jsonl`. The old W8 can NEVER pass on PP4 production: prompts 0–2 flip deterministically (DEP1-fixed2 / PP4 admission byte-identical outputs). |
| Clock | `dsv41.env` default `${GPU_CLOCK_CAP-2150}` + `GPU_CLOCK_CAP=2300` in `/etc/environment` on both Sparks | the production commit carries 420be24's `${GPU_CLOCK_CAP-2300}` default (K1 record rebased by `pp4_promote_switch.py fold-clock`); the 2300 override line is removed after the post-close watch PASSes (an explicit dial-back line stays); a rollback keeps an override before its W8 start |
| Production env/mounts | adaptive window, greedy remote contract, K1 `model_runner.py` | `PROD_PP4`, fixed5 `PROD_SPEC_OVERRIDE`, sampled env tokens, PP4 `remote_dspark` + verifier mounts (`pp4_promote_switch.edit_env`) |

## Pin sites on this branch (every file that pins one of those facts)

- Production commit/tag, W8 path/sha: `offload/launch/windows/{pf1_window,pf2_window,sp1_window,kl1_window,nbw_window,
  pp4_window,up2_low,post_close_watch,pf1_precheck,pf2_precheck,sp1_precheck}.py`; D2 `offload/dspark/d2/{replay_window,
  smoke_window,smoke_prepare}.py`; maintenance `offload/launch/maintenance/{fw1_sparkhead,fw2_sparkworker,fw3_sparkworker_firmware,
  fw3b_sparkworker_ec_pd,ops1_apply}.sh`.
- K1-post references used by in-window greedy gates: `nbw_cells.py:24` and `pp4_cells.py:29` (`REFERENCE`). A control
  arm started from PP4 production compares against these: it must use the PP4 references after promotion.
- The literal `greedy K1-post vs restored:` W8 line prefix (`post_close_watch.py:55-57`, `pf1_window.py:454`) is kept by
  `restore-pp4.sh` on purpose, so the parsers need no change.
- Production profile drift guards that hard-code `6442450944:327680:1:327680`: `PP4-sampled.env:14`,
  `configs/PF1-window.env:30`, `configs/NBW-native.env:16` (already a PROMO1 re-pin item; PP4 adds the env/mount set).

## Current runner heads (PROMO1 manifest at 4740ce9, plus the newer heads queued 2026-10-27)

shared 8bb5481 · pp4 4292855 → 941a3c3 → 85b0d96 → 90f0278 (R16, the promoted candidate) · tv1 5b2de7c → dae5f9e (R9 a78a99c, R10 pending) · pc1 fd1123e · up3 06be7c0 →
ac23d4f · cc1 8cdf7a4 · nbw 5a6a65e. Each re-pinned head needs its own independent check before a window.

## Operating rules while PP4 is production

- No `prod_window.sh train` windows: the one-card unit runs `current` (fb68c52, greedy) against a sampled Spark config;
  that pairing is unqualified. Keep `$C/D2-GEN.PAUSE` in place.
- Pair windows (`prod_window.sh pair`/`close`) stay valid: `close` restarts the same TP2 unit with the PP4 drop-in.
- Rollback to the prior production at any time = the runner's finish() path: Spark `spark-restore`, x86host drop-in removal
  through `prod_window.sh pair/close`, prior W8 start/check/close.
