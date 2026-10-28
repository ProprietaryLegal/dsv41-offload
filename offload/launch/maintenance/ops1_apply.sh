#!/usr/bin/env bash
# OPS1 apply (coordinator only, 2026-10-26; NDM "sure apply"): put the bounded warm-up on production main and qualify it
# BEFORE releasing HOLD (S48 FIX 2). Candidate = branch prod/ops1-warmup at REVIEWED (21bed31 + the four warm-up files).
# Run on Controller, one step at a time, in order; each step stops at its first failed gate and leaves HOLD in place:
#   ops1_apply.sh pre       production serving: no HOLD, sparkhead main = 21bed31 clean + rollback tag, host python3 >= 3.10
#   ops1_apply.sh open      window.sh open (drain, HOLD, placeholders)
#   ops1_apply.sh apply     push the candidate as a ref, fast-forward main to exactly REVIEWED, verify files + K1 guard
#   ops1_apply.sh start     W8/restore.sh start (stop.sh + start.sh: the new warm-up runs inside K1's start guard)
#   ops1_apply.sh check     W8/restore.sh check: six greedy IDENTICAL + PRODUCTION HOLD OK (strict)
#   ops1_apply.sh qualify   the fresh start log: 4 completed waves 1/2/3/4, 10 x 32-token length replies, wave 1 > 4096
#                           prompt tokens uncached, no WARN, within budget; then tag prod-k1-327-ops1-20261026 on sparkhead
#   ops1_apply.sh close     W8/restore.sh close (releases HOLD) — only after qualify
# Rollback (HOLD kept): ssh sparkhead 'cd ~/dsv41-deploy && ./stop.sh && git status --porcelain' must show nothing, then
#   'git reset --hard prod-k1-327-20261023', then W8/restore.sh start|check|close. Never manage.py restore (that is pre-K1).
set -euo pipefail
C=/opt/dsv41/campaigns/DSV41-OFFLOAD
D=$C/OPS1-APPLY; mkdir -p "$D"
LOG=$D/ops1.log
BASE=21bed313ac803d38327c4de1ff2e5235f329c716
REVIEWED=${OPS1_REVIEWED:?set OPS1_REVIEWED to the full reviewed candidate SHA}
TAG=prod-k1-327-ops1-20261026
REPO=/opt/dsv41/repos/dsv41-deploy
die() { echo "OPS1 STOP: $*" | tee -a "$LOG" >&2; exit 1; }
q() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
r() { ssh -o BatchMode=yes -o ConnectTimeout=10 sparkhead "$@"; }
need() { [ -e "$D/$1.done" ] || die "run $1 first"; }
[ -e "$D/close.done" ] && die "OPS1 already applied and closed ($D/close.done)"
[ "$(git -C $REPO rev-parse prod/ops1-warmup)" = "$REVIEWED" ] || die "prod/ops1-warmup is not the reviewed $REVIEWED"
case "${1:-}" in
  pre)
    r 'test -f ~/dsv41-deploy/HOLD' && die "a HOLD exists on sparkhead; another window is open"
    [ "$(r 'cd ~/dsv41-deploy && git rev-parse HEAD')" = "$BASE" ] || die "sparkhead main is not $BASE"
    [ -z "$(r 'cd ~/dsv41-deploy && git status --porcelain --untracked-files=no')" ] || die "sparkhead checkout has tracked changes"
    r 'cd ~/dsv41-deploy && git rev-parse -q --verify refs/tags/prod-k1-327-20261023 >/dev/null' || die "rollback tag missing"
    r 'python3 -c "import sys; assert sys.version_info >= (3, 10)"' || die "host python3 < 3.10"
    [ "$(git -C $REPO diff --name-only $BASE $REVIEWED | sort | tr '\n' ' ')" = "offload/launch/OPERATIONS.md offload/launch/tests/test_warmup.py offload/launch/warmup.py offload/launch/warmup_serve.sh " ] \
      || die "candidate changes files other than the four warm-up files"
    curl -sf -m 10 http://HEAD_IP:8000/v1/models | grep -q '"max_model_len":327680' || die "production is not serving 327680"
    q "PRE DONE: sparkhead at $BASE clean, rollback tag present, python3 >= 3.10, candidate = 4 warm-up files"; touch "$D/pre.done" ;;
  open)
    need pre
    r 'cd ~/dsv41-deploy && offload/launch/window.sh open "OPS1 warm-up apply (NDM sure apply 2026-10-26)"' | tee -a "$LOG" || die "window.sh open refused"
    q "OPEN DONE: HOLD written"; touch "$D/open.done" ;;
  apply)
    need open
    git -C $REPO push -q sparkhead "$REVIEWED:refs/heads/ops1-candidate" || die "push of the candidate ref failed"
    r "cd ~/dsv41-deploy && git merge --ff-only -q $REVIEWED" >> "$LOG" 2>&1 || die "fast-forward to $REVIEWED failed"
    [ "$(r 'cd ~/dsv41-deploy && git rev-parse HEAD')" = "$REVIEWED" ] || die "sparkhead HEAD is not $REVIEWED after the fast-forward"
    [ -z "$(r 'cd ~/dsv41-deploy && git status --porcelain --untracked-files=no')" ] || die "tracked changes after the fast-forward"
    r "cd ~/dsv41-deploy && git diff --quiet $BASE HEAD -- dsv41.env start.sh serve.sh stop.sh offload/promote" || die "protected files differ from $BASE"
    r 'cd ~/dsv41-deploy && python3 offload/promote/k1/manage.py verify --profile 327680' >> "$LOG" 2>&1 || die "K1 manage.py verify failed"
    q "APPLY DONE: sparkhead main = $REVIEWED; dsv41.env/start.sh/serve.sh/stop.sh/K1 package unchanged; K1 verify OK"; touch "$D/apply.done" ;;
  start)
    need apply
    r 'ls -1t ~/dsv41-deploy/logs/start-*.log 2>/dev/null | head -1' > "$D/start-log-before.txt" || true
    bash $C/W8/restore.sh start | tee "$D/restore-start.txt" | tee -a "$LOG"
    grep -q "K1 START VERIFIED" "$D/restore-start.txt" || die "no K1 START VERIFIED (HOLD stays; consider rollback)"
    q "START DONE: K1 START VERIFIED"; touch "$D/start.done" ;;
  check)
    need start
    bash $C/W8/restore.sh check | tee "$D/restore-check.txt" | tee -a "$LOG"
    [ "$(grep -cE 'greedy K1-post vs restored: [0-9]+ IDENTICAL [0-9]+$' "$D/restore-check.txt")" = 6 ] || die "not six greedy IDENTICAL lines"
    grep -q "PRODUCTION HOLD OK" "$D/restore-check.txt" || die "no PRODUCTION HOLD OK"
    q "CHECK DONE: six greedy IDENTICAL, PRODUCTION HOLD OK (HOLD still open)"; touch "$D/check.done" ;;
  qualify)
    need check
    newest=$(r 'ls -1t ~/dsv41-deploy/logs/start-*.log | head -1')
    [ "$newest" != "$(cat "$D/start-log-before.txt")" ] || die "no new start log was written by this start"
    r "cat $newest" > "$D/start-log.txt" || die "cannot read $newest"
    grep -q "WARN: the warm-up request failed" "$D/start-log.txt" && die "warm-up WARN in the start log (warm-up failed)"
    python3 - "$D/start-log.txt" <<'PY' | tee -a "$LOG" || die "warm-up qualification failed (read $D/start-log.txt)"
import json, sys
waves = []
for line in open(sys.argv[1], encoding='utf-8', errors='replace'):
    line = line.strip()
    if line.startswith('{') and '"warmup"' in line:
        rec = json.loads(line)
        if rec.get('warmup') == 'completed':
            waves.append(rec)
assert [w['requested_concurrency'] for w in waves] == [1, 2, 3, 4], f'waves {[w.get("requested_concurrency") for w in waves]}'
reqs = [q for w in waves for q in w['requests']]
assert len(reqs) == 10, f'{len(reqs)} requests'
assert all(q.get('completion_tokens') == 32 and q.get('finish_reason') == 'length' for q in reqs), 'not 10 x 32-token length replies'
first = waves[0]['requests'][0]
assert isinstance(first.get('prompt_tokens'), int) and first['prompt_tokens'] > 4096, 'wave 1 prompt <= 4096'
assert first.get('cached_tokens') in (0, None), 'wave 1 reused cached tokens'
total = sum(float(w['elapsed_s']) for w in waves)
assert total < 90.0, f'warm-up waves took {total:.1f} s'
print(f'QUALIFIED: 4 waves 1/2/3/4, 10 x 32-token replies, wave-1 prompt {first["prompt_tokens"]} uncached, waves {total:.1f} s')
PY
    (cd $REPO && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python3 offload/operations/live_profile.py --live) > "$D/live-profile.txt" 2>&1 \
      || die "live_profile --live is not CONFIGURATION_MATCH (read $D/live-profile.txt)"
    grep -q "CONFIGURATION_MATCH" "$D/live-profile.txt" || die "live_profile did not print CONFIGURATION_MATCH"
    r "cd ~/dsv41-deploy && git tag -a $TAG -m 'K1 327680 + OPS1 bounded warm-up (qualified 2026-10-26)' $REVIEWED" || die "tag failed"
    q "QUALIFY DONE: warm-up qualified; tagged $TAG at $REVIEWED"; touch "$D/qualify.done" ;;
  close)
    need qualify
    bash $C/W8/restore.sh close | tee "$D/restore-close.txt" | tee -a "$LOG"
    grep -q "CLOSE: window closed" "$D/restore-close.txt" || die "window did not close"
    q "CLOSE DONE: HOLD released; production = $REVIEWED ($TAG)"; touch "$D/close.done" ;;
  *) sed -n '2,16p' "$0"; exit 2 ;;
esac
