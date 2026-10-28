#!/usr/bin/env bash
# Versioned W8 for PP4 production (PP4-PROMOTE, sampled K2 drafting, PP4 Round16 90f0278e900c). The coordinator copies
# it to $C/W8/restore-pp4.sh after review; restore.sh and restore-327low.sh stay the rollback restores, never replaced.
# Greedy references: K1-production/greedy-post-pp4.jsonl, recorded by the promotion window from three identical runs
# (PP4 production reproduces DEP1-fixed2's flips on prompts 0-2, so K1-post can never match it). Output keeps the literal
# "greedy K1-post vs restored:" prefix and the START/CHECK/CLOSE lines post_close_watch.restore_verified parses.
# Profile-agnostic: the K1 profile (327680 or PROMO1's 327680-low) is read from the installed record on sparkhead.
# `check` verifies the CURRENT configuration (S91 #5), never a historical log: the validated active switch, the installed
# K1 record and the full K1 + PP4 running verification of the serving containers, whose ids must not change across the
# greedy check; and the x86host unit's drop-in bytes, attested runtime and running worker process.
set -Eeuo pipefail
cd /opt/dsv41/repos/dsv41-deploy
C=/opt/dsv41/campaigns/DSV41-OFFLOAD
R=offload/results/W8-pp4
mkdir -p "$R"
LOG=$R/restore.log
U=http://HEAD_IP:8000
PY=/home/operator/miniconda3/bin/python3
REF=$C/K1-production/greedy-post-pp4.jsonl
ID=90f0278e900c
DROPIN=.config/systemd/user/dsv41-dspark-worker-tp2-prod.service.d/50-pp4.conf
DROPIN_SHA256=f7b3cb8bd2c08426372cf6253ff7963ce68739024c6c808211e44f4ea8e261dd
q() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
die() { q "STOP: $*"; exit 1; }
pp4_worker() {  # the TP2 production unit runs the attested sampled worker from exactly the reviewed PP4 drop-in, now
  ssh -o BatchMode=yes x86host bash -s > "$R/pp4-worker.log" 2>&1 <<EOF || true
set -eu
u=dsv41-dspark-worker-tp2-prod.service
d=\$HOME/$DROPIN
test "\$(sha256sum < "\$d" | cut -d' ' -f1)" = $DROPIN_SHA256
systemctl --user show -p DropInPaths --value \$u | tr ' ' '\n' | grep -qx "\$d"
test "\$(systemctl --user show -p NeedDaemonReload --value \$u)" = no
sh \$HOME/dsv41-offload/prod/pp4-runtime/$ID/attest.sh
pid=\$(systemctl --user show -p MainPID --value \$u)
test "\$pid" -gt 0
test "\$(tr '\\0' '\\n' < /proc/\$pid/cmdline | head -1)" = /home/operator/miniconda3/bin/python3
test "\$(readlink /proc/\$pid/cwd)" = "\$HOME/dsv41-offload/prod/dspark-worker/90f0278/offload/dspark"
tr '\\0' '\\n' < /proc/\$pid/environ | grep -qx DSPARK_SAMPLED_TOP32=1
tr '\\0' '\\n' < /proc/\$pid/environ | grep -qx DSPARK_SOURCE_COMMIT=90f0278e900c5446d627fd5f42db15246cd544b5
rt=\$HOME/dsv41-offload/prod/pp4-runtime/$ID
tr '\\0' '\\n' < /proc/\$pid/environ \
  | grep -qx "PYTHONPATH=\$rt/stock:\$rt/facc:\$HOME/dsv41-offload/prod/dspark-worker/90f0278/offload/dspark"
echo 'PP4 WORKER SAMPLED'
EOF
  grep -qx 'PP4 WORKER SAMPLED' "$R/pp4-worker.log" || die 'PP4 sampled production worker not verified (bytes/runtime/process)'
}
containers() {  # the serving dsv41-serve container ids on both ranks, now
  ssh -o BatchMode=yes sparkhead 'cd ~/dsv41-deploy && set -a && . ./dsv41.env >/dev/null && set +a && for n in "$HEAD" "$WORKER"; do
    ssh -o BatchMode=yes "$n" "docker inspect -f {{.Id}} dsv41-serve"; done' | sort
}
start() {
  if tmux has-session -t =dsv41-w8 2>/dev/null; then die 'tmux dsv41-w8 still exists'; fi
  ssh -o BatchMode=yes x86host 'bash ~/dsv41-offload/prod_window.sh check' > "$R/prod-check-pre.log" 2>&1 || true
  grep -qx 'PRODUCTION HOLD OK' "$R/prod-check-pre.log" || die 'production worker hold not verified'
  pp4_worker
  ssh -o BatchMode=yes sparkhead "cd ~/dsv41-deploy && grep -qx 'PROD_PP4=$ID' dsv41.env && \
    profile=\$(python3 -c \"import json;print(json.load(open('logs/k1-promotion/files.json'))['profile'])\") && \
    python3 offload/promote/k1/manage.py verify --profile \"\$profile\" && ./stop.sh </dev/null && ./start.sh </dev/null" \
    > "$R/start.log" 2>&1 || true
  grep -q 'K1 START VERIFIED' "$R/start.log" || die 'missing K1 START VERIFIED'
  grep -qx 'PP4 RUNNING VERIFIED' "$R/start.log" || die 'missing PP4 RUNNING VERIFIED (both ranks)'
  q 'START: K1 START VERIFIED; PP4 RUNNING VERIFIED'
}
check() {
  test -s "$REF" || die "PP4 greedy references $REF missing"
  # Fresh and container-bound: the active switch, the installed K1 record, then K1's own verify.sh against the serving
  # containers (K1 overlay with the declared PP4 runner, verifier/remote_dspark bytes, effective env, fixed5 argv).
  ssh -o BatchMode=yes sparkhead "cd ~/dsv41-deploy && python3 offload/launch/windows/pp4_promote_switch.py active && \
    profile=\$(python3 -c \"import json;print(json.load(open('logs/k1-promotion/files.json'))['profile'])\") && \
    python3 offload/promote/k1/manage.py verify --profile \"\$profile\" && bash offload/promote/k1/verify.sh" \
    > "$R/config.log" 2>&1 || die 'current PP4 production configuration not verified (see config.log)'
  grep -qx "PP4 ACTIVE $ID" "$R/config.log" || die 'PP4 switch not active'
  grep -qx 'PP4 RUNNING VERIFIED' "$R/config.log" || die 'missing fresh PP4 RUNNING VERIFIED (both ranks)'
  grep -q '^K1 runtime evidence: ' "$R/config.log" || die 'K1 runtime verification incomplete'
  verified=$(sed -n 's/^PP4 RUNNING VERIFIED [^ ]* container \([0-9a-f]\{64\}\)$/\1/p' "$R/config.log" | sort)
  [ "$(grep -c . <<<"$verified")" = 2 ] || die 'fresh verification did not name both serving containers'
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
  containers > "$R/containers-after.log" 2>&1 || die 'serving containers unreadable after the greedy check'
  [ "$(cat "$R/containers-after.log")" = "$verified" ] || die 'serving containers changed since the configuration check'
  sed 's/^/greedy K1-post vs restored: /' "$R/compare.log" | tee -a "$LOG"
  ssh -o BatchMode=yes x86host 'bash ~/dsv41-offload/prod_window.sh check' > "$R/prod-check.log" 2>&1 || true
  tee -a "$LOG" < "$R/prod-check.log"
  grep -qx 'PRODUCTION HOLD OK' "$R/prod-check.log" || die 'production worker hold not verified'
  pp4_worker
  q 'CHECK: DONE'
}
close() {
  ssh -o BatchMode=yes sparkhead 'cd ~/dsv41-deploy && offload/launch/window.sh close' >> "$LOG" 2>&1
  q 'CLOSE: window closed, production PP4 sampled drafting serving'
}
case "${1:-}" in start) start ;; check) check ;; close) close ;; *) echo 'usage: restore-pp4.sh start|check|close' >&2; exit 2 ;; esac
