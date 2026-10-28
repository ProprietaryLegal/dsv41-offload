#!/usr/bin/env bash
# Versioned W8 for 0075 production (UP2-PROMOTE, replay skip, package ad3dc5be5eaf). The coordinator copies it to
# $C/W8/restore-up2.sh after review; restore.sh / restore-327low.sh stay the rollback restores and are never replaced.
# Greedy references: K1-production/greedy-post-up2-ad3dc5be5eaf.jsonl, recorded by the promotion window from three
# identical runs on the switched production. Output keeps the literal "greedy K1-post vs restored:" prefix and the
# START/CHECK/CLOSE lines post_close_watch.restore_verified parses.
# Bases: ops1 (596ba1c) and PROMO1's 327680-low; the K1 profile is read from the installed record on sparkhead.
# A PP4 production refuses here: PP4+0075 is an unmeasured composition and would also need PP4's worker check.
set -Eeuo pipefail
cd /opt/dsv41/repos/dsv41-deploy
C=/opt/dsv41/campaigns/DSV41-OFFLOAD
R=offload/results/W8-up2
mkdir -p "$R"
LOG=$R/restore.log
U=http://HEAD_IP:8000
PY=/home/operator/miniconda3/bin/python3
ID=ad3dc5be5eaf
REF=$C/K1-production/greedy-post-up2-$ID.jsonl
q() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
die() { q "STOP: $*"; exit 1; }
start() {
  if tmux has-session -t =dsv41-w8 2>/dev/null; then die 'tmux dsv41-w8 still exists'; fi
  ssh -o BatchMode=yes x86host 'bash ~/dsv41-offload/prod_window.sh check' > "$R/prod-check-pre.log" 2>&1 || true
  grep -qx 'PRODUCTION HOLD OK' "$R/prod-check-pre.log" || die 'production worker hold not verified'
  ssh -o BatchMode=yes sparkhead "cd ~/dsv41-deploy && grep -qx 'PROD_UP2=$ID' dsv41.env && ! grep -q '^PROD_PP4=' dsv41.env && \
    profile=\$(python3 -c \"import json;print(json.load(open('logs/k1-promotion/files.json'))['profile'])\") && \
    python3 offload/promote/k1/manage.py verify --profile \"\$profile\" && ./stop.sh </dev/null && ./start.sh </dev/null" \
    > "$R/start.log" 2>&1 || true
  grep -q 'K1 START VERIFIED' "$R/start.log" || die 'missing K1 START VERIFIED'
  grep -qx 'UP2 RUNNING VERIFIED' "$R/start.log" || die 'missing UP2 RUNNING VERIFIED (both ranks)'
  q 'START: K1 START VERIFIED; UP2 RUNNING VERIFIED'
}
check() {
  # The latest guarded start (this W8's, the promotion's or a watchdog's) must carry both verifications.
  ssh -o BatchMode=yes sparkhead 'cd ~/dsv41-deploy && latest=$(ls -t logs/k1-guard-*.log | head -1) && test -n "$latest" && cat "$latest"' \
    > "$R/start.log" 2>&1 || true
  grep -q 'K1 START VERIFIED' "$R/start.log" || die 'missing K1 START VERIFIED in latest start output'
  grep -qx 'UP2 RUNNING VERIFIED' "$R/start.log" || die 'missing UP2 RUNNING VERIFIED in latest start output'
  test -s "$REF" || die "UP2 greedy references $REF missing"
  "$PY" "$C/W5/greedy_check.py" "$U" "$R/greedy.jsonl" > "$R/greedy-run.log" 2>&1 || die 'greedy run failed'
  "$PY" "$C/W5/greedy_check.py" --compare "$REF" "$R/greedy.jsonl" > "$R/compare.log" 2>&1 || die 'greedy compare failed'
  "$PY" - "$R/compare.log" <<'PY' || die 'six distinct IDENTICAL greedy rows required'
import pathlib, re, sys
text = pathlib.Path(sys.argv[1]).read_text()
rows = text.splitlines()
matches = re.findall(r'^([0-9]+) IDENTICAL ([0-9]+)\s*$', text, re.M)
if len(rows) != 6 or len(matches) != 6 or sorted(int(i) for i, _ in matches) != list(range(6)):
    raise SystemExit('STOP: six distinct IDENTICAL greedy rows required')
PY
  sed 's/^/greedy K1-post vs restored: /' "$R/compare.log" | tee -a "$LOG"
  ssh -o BatchMode=yes x86host 'bash ~/dsv41-offload/prod_window.sh check' > "$R/prod-check.log" 2>&1 || true
  tee -a "$LOG" < "$R/prod-check.log"
  grep -qx 'PRODUCTION HOLD OK' "$R/prod-check.log" || die 'production worker hold not verified'
  q 'CHECK: DONE'
}
close() {
  ssh -o BatchMode=yes sparkhead 'cd ~/dsv41-deploy && offload/launch/window.sh close' >> "$LOG" 2>&1
  q 'CLOSE: window closed, production 0075 replay skip serving'
}
case "${1:-}" in start) start ;; check) check ;; close) close ;; *) echo 'usage: restore-up2.sh start|check|close' >&2; exit 2 ;; esac
