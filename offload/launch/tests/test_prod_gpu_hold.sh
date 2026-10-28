#!/usr/bin/env bash
# Regression for S12 F1 (R3-PROMOTE): after the TP2 promotion nothing may hand GPU-7744b44b (or c4e7a371) to a lane while
# the production TP2 worker runs on it.
#   1. worker/run/gpu_locked.sh: the command holds both locks for its own lifetime (same pid, a real python too); a lock
#      held elsewhere -> refusal (exit 75, one line), the command never runs, nothing stays locked.
#   2. the TP2 production unit runs its worker under gpu_locked.sh on both lock files.
#   3. pair_window.sh refuses open / open hold / hand / hold / close while the TP2 production unit is enabled or running
#      (no unit, tmux or PAIR_OK change); before promotion it still opens a window.
#   4. prod_window.sh against stub systemctl / nvidia-smi whose TP2 unit runs the REAL gpu_locked.sh on real lock files:
#      check -> train (one-card production, TRAIN_OK, 7744b44b free) -> a lane holds 7744b44b -> close waits for it and
#      restores TP2 with the hold -> pair (no worker, PAIR_OK) -> close; refusals (window open, not promoted, a lane that
#      keeps the lock).
# CPU only, scratch HOME; nothing on x86host and no real unit is touched.   bash offload/launch/tests/test_prod_gpu_hold.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(cd "$HERE/../.." && pwd)
GL=$REPO/offload/dspark/worker/run/gpu_locked.sh
UNITF=$REPO/offload/dspark/worker/dsv41-dspark-worker-tp2.prod.service
PW=$HERE/pair_window.sh; PRW=$HERE/prod_window.sh
T=$(mktemp -d "${TMPDIR:-/tmp}/prod-gpu-hold.XXXXXX")
PASS=0; FAIL=0
cleanup() { for f in "$T"/state/*.pid; do [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null; done; jobs -p | xargs -r kill 2>/dev/null; rm -rf "$T"; }
trap cleanup EXIT
ok() { echo "ok   $*"; PASS=$((PASS + 1)); }
bad() { echo "FAIL $*"; [ -n "${out:-}" ] && sed "s/^/     | /" <<<"$out" | tail -12; FAIL=$((FAIL + 1)); }
expect() { local what=$1; shift; if "$@"; then ok "$what"; else bad "$what"; fi; }

# ---------------------------------------------------------------- 1. gpu_locked.sh
A=$T/a.lock; B=$T/b.lock
sh "$GL" "$A" "$B" -- sh -c 'echo $$ > "$0"; sleep 2' "$T/child.pid" 2> "$T/gl.err" & gl=$!
sleep 0.5
expect "gpu_locked: lock A held while the command runs" bash -c "! flock -n '$A' true"
expect "gpu_locked: lock B held while the command runs" bash -c "! flock -n '$B' true"
expect "gpu_locked: the command runs in the same pid (exec)" test "$(cat "$T/child.pid")" = "$gl"
expect "gpu_locked: that pid's own fdinfo has the lock line for B's dev:inode (what prod_window.sh check reads)" \
  bash -c "for fd in /proc/$gl/fd/*; do [ \"\$(readlink \$fd)\" = '$B' ] && grep -q '^lock:.* [0-9a-f]*:[0-9a-f]*:$(stat -c %i "$B") ' /proc/$gl/fdinfo/\${fd##*/} && exit 0; done; exit 1"
wait "$gl"
expect "gpu_locked: both locks free after the command exits" bash -c "flock -n '$A' true && flock -n '$B' true"
sh "$GL" "$A" "$B" -- python3 -c 'import time; time.sleep(1.5)' 2>/dev/null & gl=$!
sleep 0.7
expect "gpu_locked: a python worker keeps holding (fds inherited through exec)" bash -c "! flock -n '$B' true"
wait "$gl"
flock "$B" sleep 3 & lane=$!
sleep 0.3
rc=0; sh "$GL" "$A" "$B" -- touch "$T/ran" 2> "$T/gl.err" || rc=$?
expect "gpu_locked: refuses with exit 75 when a lane holds a lock" test "$rc" = 75
expect "gpu_locked: refusal line names the lock" grep -q "$B is held by another process.*refusing" "$T/gl.err"
expect "gpu_locked: the command never ran" test ! -e "$T/ran"
expect "gpu_locked: the first lock is not left held" flock -n "$A" true
kill "$lane" 2>/dev/null; wait "$lane" 2>/dev/null
rc=0; sh "$GL" -- true 2>/dev/null || rc=$?
expect "gpu_locked: no lock -> usage error" test "$rc" = 2

# ---------------------------------------------------------------- 2. the unit
expect "TP2 production unit: ExecStart runs gpu_locked.sh on both locks before the worker" \
  grep -Eq '^ExecStart=/bin/sh %h/dsv41-offload/prod/dspark-worker/current/offload/dspark/worker/run/gpu_locked\.sh %h/dsv41-offload/locks/gpu-c4e7a371\.lock %h/dsv41-offload/locks/gpu-7744b44b\.lock -- %h/dsv41-offload/o2/venv/bin/python -m worker\.server --tp 2 ' "$UNITF"
expect "TP2 production unit: still Conflicts= with both other worker units" \
  grep -q '^Conflicts=dsv41-dspark-worker.service dsv41-dspark-worker-tp2.service$' "$UNITF"
expect "TP2 production unit: Restart=on-failure (a refused start retries)" grep -q '^Restart=on-failure$' "$UNITF"

# ---------------------------------------------------------------- stubs (state in $T/state)
H=$T/home; S=$T/state; BIN=$T/bin
mkdir -p "$BIN" "$S" "$H/dsv41-offload/locks" "$H/dsv41-offload/prod/logs" "$H/dsv41-offload/prod/run" "$H/dsv41-offload/d2"
LK=$H/dsv41-offload/locks; LOGF=$H/dsv41-offload/prod/logs/dspark-worker.log; ST=$H/dsv41-offload/prod/run/dspark-worker.status.json
touch "$LOGF"
cat > "$BIN/systemctl" <<EOS
#!/usr/bin/env bash
# stub: units = dsv41-dspark-worker.service (ONE) and dsv41-dspark-worker-tp2-prod.service (PROD, runs the real
# gpu_locked.sh on the real lock files); enabled set in $S/enabled; running pids in $S/<unit>.pid
S=$S; LK=$LK; LOGF=$LOGF; ST=$ST; GL=$GL
echo "systemctl \$*" >> \$S/calls
a=("\$@"); [ "\${a[0]}" = --user ] && a=("\${a[@]:1}")
q=0; [ "\${a[1]:-}" = --quiet ] && { q=1; a=("\${a[0]}" "\${a[@]:2}"); }
u=\${a[-1]}; case "\$u" in *.service) ;; *) u=\$u.service ;; esac
alive() { [ -f \$S/\$1.pid ] && kill -0 "\$(cat \$S/\$1.pid)" 2>/dev/null; }
st() { if alive \$1; then echo active; elif [ -f \$S/\$1.failed ]; then echo failed; else echo inactive; fi; }
stop() { if [ -f \$S/\$1.pid ]; then p=\$(cat \$S/\$1.pid); kill \$p 2>/dev/null
           for _ in \$(seq 50); do kill -0 \$p 2>/dev/null || break; sleep 0.1; done; rm -f \$S/\$1.pid; fi; }
case "\${a[0]}" in
  is-active) s=\$(st \$u); [ \$q = 1 ] || echo \$s; [ \$s = active ] ;;
  is-enabled) if grep -qx \$u \$S/enabled 2>/dev/null; then [ \$q = 1 ] || echo enabled; exit 0; fi
              [ \$q = 1 ] || echo disabled; exit 1 ;;
  enable) echo \$u >> \$S/enabled ;;
  disable) grep -vx \$u \$S/enabled > \$S/e2 2>/dev/null; mv \$S/e2 \$S/enabled ;;
  show) alive \$u && cat \$S/\$u.pid || echo 0 ;;
  stop) stop \$u ;;
  start)
    rm -f \$S/\$u.failed
    if [ \$u = dsv41-dspark-worker-tp2-prod.service ]; then
      stop dsv41-dspark-worker.service                                     # Conflicts=
      ( exec sh \$GL \$LK/gpu-c4e7a371.lock \$LK/gpu-7744b44b.lock -- sleep 1000 ) </dev/null >>\$LOGF 2>&1 &
      echo \$! > \$S/\$u.pid; disown; sleep 0.4
      if alive \$u; then echo "TP=2 worker ready in 1.0 s" >> \$LOGF; echo '{"tp": 2, "worker": "stub"}' > \$ST
      else rm -f \$S/\$u.pid; touch \$S/\$u.failed; exit 0; fi
    else
      stop dsv41-dspark-worker-tp2-prod.service                            # Conflicts= (both ways)
      ( exec sleep 1000 ) </dev/null >/dev/null 2>&1 & echo \$! > \$S/\$u.pid; disown
      echo "worker ready in 1.0 s" >> \$LOGF; echo '{"worker": "stub", "gpu_mem_bytes": 1}' > \$ST
    fi ;;
esac
exit 0
EOS
cat > "$BIN/nvidia-smi" <<EOS
#!/usr/bin/env bash
S=$S
alive() { [ -f \$S/\$1.pid ] && kill -0 "\$(cat \$S/\$1.pid)" 2>/dev/null; }
case "\$*" in
  *compute-apps*)
    alive dsv41-dspark-worker-tp2-prod.service && { echo "GPU-c4e7a371, 1, python, 9000 MiB"; echo "GPU-7744b44b, 1, python, 9000 MiB"; }
    alive dsv41-dspark-worker.service && echo "GPU-c4e7a371, 2, python, 8000 MiB" ;;
  *query-gpu*) echo "GPU-c4e7a371, 1 MiB"; echo "GPU-7744b44b, 1 MiB" ;;
esac
exit 0
EOS
cat > "$BIN/tmux" <<EOS
#!/usr/bin/env bash
echo "tmux \$*" >> $S/calls
case "\$1" in
  has-session) grep -qx "\$3" $S/tmux 2>/dev/null ;;
  kill-session) grep -vx "\$3" $S/tmux > $S/t2 2>/dev/null; mv $S/t2 $S/tmux; exit 0 ;;
  new-session) echo dsv41-pairhold >> $S/tmux; date -u > $LK/.pairhold-acquired ;;
esac
EOS
chmod +x "$BIN"/*
run() { HOME=$H PATH=$BIN:$PATH PAIR_WINDOW_SH=$PW PROD_WINDOW_READY_S=20 "$@"; }

# ---------------------------------------------------------------- 3. pair_window.sh guard
PROD=dsv41-dspark-worker-tp2-prod.service
for cfg in "enabled+active" "enabled, stopped (training window)" "disabled, running"; do
  for f in "$S"/*.pid; do [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null; done; rm -f "$S"/*.pid "$S"/*.failed "$S/enabled" "$S/tmux"
  touch "$S/enabled"
  case "$cfg" in
    enabled+active) run systemctl --user enable $PROD; run systemctl --user start $PROD ;;
    "enabled, stopped"*) run systemctl --user enable $PROD ;;
    "disabled, running") run systemctl --user start $PROD ;;
  esac
  for args in "open N2" "open N2 hold" "hand N2" "hold" "close"; do
    echo dsv41-pairhold > "$S/tmux"; echo N2 > "$H/dsv41-offload/PAIR_OK"; [ "${args%% *}" = open ] && rm -f "$H/dsv41-offload/PAIR_OK"
    pok_before=$(cat "$H/dsv41-offload/PAIR_OK" 2>/dev/null || echo absent); prod_before=$(run systemctl --user is-active $PROD)
    rm -f "$S/calls"
    rc=0; out=$(run bash "$PW" $args 2>&1) || rc=$?
    changed=$(grep -E "systemctl --user (start|stop|enable|disable)|tmux (new-session|kill-session)" "$S/calls" 2>/dev/null || true)
    pok_after=$(cat "$H/dsv41-offload/PAIR_OK" 2>/dev/null || echo absent)
    if [ "$rc" = 1 ] && grep -q "refusing" <<<"$out" && [ -z "$changed" ] && [ "$pok_before" = "$pok_after" ] \
       && [ "$(run systemctl --user is-active $PROD)" = "$prod_before" ] && grep -qx dsv41-pairhold "$S/tmux"; then
      ok "pair_window.sh $args refuses ($cfg), nothing changed"
    else bad "pair_window.sh $args ($cfg): rc $rc, changes [$changed], PAIR_OK $pok_before -> $pok_after: $out"; fi
  done
done
for f in "$S"/*.pid; do [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null; done; rm -f "$S"/*.pid "$S/enabled" "$S/tmux" "$H/dsv41-offload/PAIR_OK" "$S/calls"
touch "$S/enabled"
rc=0; out=$(run bash "$PW" open N2 2>&1) || rc=$?
expect "pair_window.sh open N2 before promotion (TP2 production unit not installed) still opens the window" \
  bash -c "[ $rc = 0 ] && [ \"\$(head -1 '$H/dsv41-offload/PAIR_OK')\" = N2 ] && grep -q 'tmux new-session' '$S/calls'"
rm -f "$H/dsv41-offload/PAIR_OK" "$S/tmux"

# ---------------------------------------------------------------- 4. prod_window.sh
rm -f "$S/enabled"; touch "$S/enabled"
rc=0; out=$(run bash "$PRW" train D2-TRAIN 2>&1) || rc=$?
expect "prod_window.sh train refuses before promotion" bash -c "[ $rc = 1 ] && grep -q 'not promoted' <<<'$out'"
# promotion state: one-card disabled, TP2 production enabled + running under the real gpu_locked.sh
run systemctl --user enable $PROD; run systemctl --user start $PROD
out=$(run bash "$PRW" check 2>&1); rc=$?
expect "prod_window.sh check passes on the promoted state" bash -c "[ $rc = 0 ] && grep -q 'PRODUCTION HOLD OK' <<<\"\$1\"" _ "$out"
expect "check: flock -n on 7744b44b fails, holder = the TP2 worker's pid" \
  grep -q "ok: flock -n $LK/gpu-7744b44b.lock fails; held by the TP2 worker" <<<"$out"
expect "check: pair_window.sh open refuses" grep -q "ok: pair_window.sh open refuses" <<<"$out"
expect "a lane's flock -n on 7744b44b fails while production runs" bash -c "! flock -n '$LK/gpu-7744b44b.lock' true"

# a lane queued in flock -w on 7744b44b would take the card the moment TP2 stops (window-5 starvation): train refuses
flock -w 30 "$LK/gpu-7744b44b.lock" sh -c "echo got > '$T/queued.got'" & qlane=$!
sleep 0.4
rc=0; out=$(run bash "$PRW" check 2>&1) || rc=$?
expect "check FAILS while a process waits in flock on 7744b44b and names it (S12 R2: the 10 s restart gap)" \
  bash -c "[ $rc = 1 ] && grep -q 'FAIL: pid(s) [0-9 ]* wait in flock on $LK/gpu-7744b44b.lock' <<<\"\$1\" && \
           grep -q 'ok: no process waits on $LK/gpu-c4e7a371.lock' <<<\"\$1\"" _ "$out"
out=$(run bash "$PRW" status 2>&1)
expect "status lists the queued process" bash -c "grep -q 'queued on the locks: [0-9]*($LK/gpu-7744b44b.lock)' <<<\"\$1\"" _ "$out"
rc=0; out=$(run bash "$PRW" train D2-TRAIN 2>&1) || rc=$?
expect "train refuses while a lane is queued on a GPU lock; TP2 untouched; the lane still waits" \
  bash -c "[ $rc = 1 ] && grep -q 'queued on the GPU locks' <<<\"\$1\" && [ ! -e '$T/queued.got' ] && \
           [ \"\$(HOME=$H PATH=$BIN:\$PATH systemctl --user is-active $PROD)\" = active ]" _ "$out"
kill "$qlane" 2>/dev/null; wait "$qlane" 2>/dev/null
out=$(run bash "$PRW" train D2-TRAIN 2>&1); rc=$?
expect "train: rc 0" test "$rc" = 0
expect "train: TRAIN_OK names D2-TRAIN" test "$(head -1 "$H/dsv41-offload/d2/TRAIN_OK" 2>/dev/null)" = D2-TRAIN
expect "train: one-card production worker running, TP2 stopped, TP2 still enabled" \
  bash -c "[ \"\$(HOME=$H PATH=$BIN:\$PATH systemctl --user is-active dsv41-dspark-worker)\" = active ] && \
           [ \"\$(HOME=$H PATH=$BIN:\$PATH systemctl --user is-active $PROD)\" = inactive ] && grep -qx $PROD '$S/enabled'"
expect "train: GPU-7744b44b lock free for the lane" flock -n "$LK/gpu-7744b44b.lock" true
rc=0; out=$(run bash "$PRW" pair N3 2>&1) || rc=$?
expect "pair refuses while the training window is open" bash -c "[ $rc = 1 ] && grep -q 'a window is open' <<<\"\$1\"" _ "$out"
out=$(run bash "$PRW" check 2>&1); rc=$?
expect "check fails inside a training window (TP2 not running, TRAIN_OK present)" test "$rc" = 1
# the trainer holds 7744b44b and exits 2 s after TRAIN_OK disappears (the D2-TRAIN rule: within 60 s)
flock "$LK/gpu-7744b44b.lock" sh -c "while [ -e '$H/dsv41-offload/d2/TRAIN_OK' ]; do sleep 0.2; done; sleep 2" & lane=$!
sleep 0.3
t0=$SECONDS; out=$(run bash "$PRW" close 2>&1); rc=$?
expect "close: waited for the trainer, then TP2 production back with the hold (rc 0, PRODUCTION HOLD OK)" \
  bash -c "[ $rc = 0 ] && grep -q 'PRODUCTION HOLD OK' <<<\"\$1\" && [ $((SECONDS - t0)) -ge 2 ]" _ "$out"
expect "close: TRAIN_OK removed" test ! -e "$H/dsv41-offload/d2/TRAIN_OK"
wait "$lane" 2>/dev/null

out=$(run bash "$PRW" pair N3 2>&1); rc=$?
expect "pair: PAIR_OK names N3, no worker runs, both locks free" \
  bash -c "[ $rc = 0 ] && [ \"\$(head -1 '$H/dsv41-offload/PAIR_OK')\" = N3 ] && \
           [ \"\$(HOME=$H PATH=$BIN:\$PATH systemctl --user is-active dsv41-dspark-worker)\" = inactive ] && \
           [ \"\$(HOME=$H PATH=$BIN:\$PATH systemctl --user is-active $PROD)\" = inactive ] && \
           flock -n '$LK/gpu-c4e7a371.lock' true && flock -n '$LK/gpu-7744b44b.lock' true"
out=$(run bash "$PRW" close 2>&1); rc=$?
expect "close after the pair window: PRODUCTION HOLD OK" bash -c "[ $rc = 0 ] && grep -q 'PRODUCTION HOLD OK' <<<\"\$1\"" _ "$out"

# a lane that keeps the lock past close's wait: close fails loudly, production is NOT started next to it
run bash "$PRW" train D2-TRAIN > /dev/null 2>&1
flock "$LK/gpu-7744b44b.lock" sleep 30 & lane=$!
sleep 0.3
rc=0; out=$(PROD_WINDOW_WAIT_S=3 run bash "$PRW" close 2>&1) || rc=$?
expect "close: a lane holding 7744b44b past the wait -> rc 1, names the holder, TP2 not started" \
  bash -c "[ $rc = 1 ] && grep -q 'still held' <<<\"\$1\" && \
           [ \"\$(HOME=$H PATH=$BIN:\$PATH systemctl --user is-active $PROD)\" != active ]" _ "$out"
# the TP2 unit started anyway (e.g. at boot) while the lane holds the card: it refuses and logs it
run systemctl --user start $PROD
expect "TP2 unit start while a lane holds 7744b44b: refused (failed), refusal line in the worker log" \
  bash -c "[ \"\$(HOME=$H PATH=$BIN:\$PATH systemctl --user is-active $PROD)\" = failed ] && \
           grep -q 'gpu-7744b44b.lock is held by another process' '$LOGF'"
pkill -P "$lane" 2>/dev/null; kill "$lane" 2>/dev/null; wait "$lane" 2>/dev/null   # its sleep child holds the lock too

# x86host rebooted inside a training window: the enabled TP2 unit started at boot and holds both locks, TRAIN_OK is stale.
# close must not wait PROD_WINDOW_WAIT_S for locks production itself holds (S12 note): straight to check.
run systemctl --user start $PROD
printf '%s\n' D2-TRAIN "training window opened (before the reboot)" > "$H/dsv41-offload/d2/TRAIN_OK"
t0=$SECONDS; rc=0; out=$(PROD_WINDOW_WAIT_S=30 run bash "$PRW" close 2>&1) || rc=$?
expect "close after a reboot inside a window: rc 0 at once, 'already runs and holds both locks', PRODUCTION HOLD OK" \
  bash -c "[ $rc = 0 ] && [ $((SECONDS - t0)) -lt 10 ] && grep -q 'already runs and holds both locks' <<<\"\$1\" && \
           grep -q 'PRODUCTION HOLD OK' <<<\"\$1\" && [ ! -e '$H/dsv41-offload/d2/TRAIN_OK' ]" _ "$out"

echo "test_prod_gpu_hold: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
