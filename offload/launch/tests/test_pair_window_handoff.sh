#!/usr/bin/env bash
# Regression for the window-5 hand-off starvation (coordinator, 2026-10-23): `pair_window.sh hand N2` dropped the
# coordinator's hold, the lanes already blocked in `flock -w ... gpu-7744b44b.lock` took the card at once, and N2's
# runner, polling both locks with `flock -n`, starved. Now the hold stays for the whole window and the owner's commands
# run under it (`pair_window.sh run OWNER -- CMD`). S12's re-check of 151504e (R3, R4): close removes PAIR_OK first,
# waits for the run, and refuses with the hold kept while any compute process is on either card (an escaped job); run
# re-checks PAIR_OK under its run lock (the close race) and gives CMD stdin from /dev/null (SIGTTIN from a terminal).
# Stubs as in S12's probe (systemctl, nvidia-smi), except that the tmux stub RUNS the hold's command, so the hold takes
# the real lock files and queued lanes are real `flock -w` processes. CPU only, scratch HOME; nothing on x86host touched.
#   bash offload/launch/tests/test_pair_window_handoff.sh [path/to/pair_window.sh]
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
PW=${1:-$HERE/pair_window.sh}
T=$(mktemp -d "${TMPDIR:-/tmp}/pair-handoff.XXXXXX")
H=$T/home; S=$T/state; BIN=$T/bin; LK=$H/dsv41-offload/locks
mkdir -p "$BIN" "$S" "$LK" "$H/dsv41-offload/prod/logs"
touch "$LK/gpu-c4e7a371.lock" "$LK/gpu-7744b44b.lock" "$H/dsv41-offload/prod/logs/dspark-worker.log" "$S/active" "$S/enabled"
PASS=0; FAIL=0
cleanup() {
  for f in "$S"/tmux-*.pgid; do [ -f "$f" ] && kill -KILL -- "-$(cat "$f")" 2>/dev/null; done
  jobs -p | xargs -r kill 2>/dev/null; rm -rf "$T"
}
trap cleanup EXIT
ok() { echo "ok   $*"; PASS=$((PASS + 1)); }
bad() { echo "FAIL $*"; [ -n "${out:-}" ] && sed "s/^/     | /" <<<"$out" | tail -8; FAIL=$((FAIL + 1)); }
expect() { local what=$1; shift; if "$@"; then ok "$what"; else bad "$what"; fi; }

cat > "$BIN/systemctl" <<EOS
#!/usr/bin/env bash
S=$S; LOGF=$H/dsv41-offload/prod/logs/dspark-worker.log
a=("\$@"); [ "\${a[0]}" = --user ] && a=("\${a[@]:1}")
q=0; [ "\${a[1]:-}" = --quiet ] && { q=1; a=("\${a[0]}" "\${a[@]:2}"); }
u=\${a[-1]}; case "\$u" in *.service) ;; *) u=\$u.service ;; esac
case "\${a[0]}" in
  is-active) if grep -qx \$u \$S/active; then [ \$q = 1 ] || echo active; exit 0; fi; [ \$q = 1 ] || echo inactive; exit 3 ;;
  is-enabled) if grep -qx \$u \$S/enabled; then [ \$q = 1 ] || echo enabled; exit 0; fi; [ \$q = 1 ] || echo disabled; exit 1 ;;
  stop) grep -vx \$u \$S/active > \$S/a2; mv \$S/a2 \$S/active ;;
  start) echo \$u >> \$S/active; echo "worker ready in 1.0 s" >> \$LOGF ;;
esac
exit 0
EOS
cat > "$BIN/nvidia-smi" <<EOS
#!/usr/bin/env bash
# a GPU job = a live pid in $S/gpujob.pid, on GPU-7744b44b
case "\$*" in
  *compute-apps*) [ -f $S/gpujob.pid ] && kill -0 "\$(cat $S/gpujob.pid)" 2>/dev/null \
                    && echo "GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc, \$(cat $S/gpujob.pid), python, 9000 MiB" ;;
  *query-gpu*) echo "GPU-c4e7a371, 1 MiB"; echo "GPU-7744b44b, 1 MiB" ;;
esac
exit 0
EOS
# flock wrapper: with DELAY_RUNLOCK_TAKE set, `flock -n 9` (run taking its run lock) starts that much later, to place a
# close between run's first checks and its run lock (S12's race D)
REALFLOCK=$(command -v flock)
cat > "$BIN/flock" <<EOS
#!/usr/bin/env bash
[ "\$*" = "-n 9" ] && [ -n "\${DELAY_RUNLOCK_TAKE:-}" ] && sleep "\$DELAY_RUNLOCK_TAKE"
exec $REALFLOCK "\$@"
EOS
cat > "$BIN/tmux" <<EOS
#!/usr/bin/env bash
# a session = one process group running the command for real (the hold takes the real lock files)
S=$S
case "\$1" in
  has-session) [ -f \$S/tmux-\$3.pgid ] && kill -0 -- "-\$(cat \$S/tmux-\$3.pgid)" 2>/dev/null ;;
  kill-session) [ -f \$S/tmux-\$3.pgid ] || exit 1
                kill -HUP -- "-\$(cat \$S/tmux-\$3.pgid)" 2>/dev/null; rm -f \$S/tmux-\$3.pgid; sleep 0.2 ;;
  new-session) set -m; bash -c "\$5" </dev/null >/dev/null 2>&1 & echo \$! > \$S/tmux-\$4.pgid ;;
esac
EOS
chmod +x "$BIN"/*
pw() { HOME=$H PATH=$BIN:$PATH bash "$PW" "$@"; }
HELD="! flock -n '$LK/gpu-c4e7a371.lock' true && ! flock -n '$LK/gpu-7744b44b.lock' true"   # both lock files held
held() { bash -c "$HELD"; }
lane() {  # a lane runner blocked in flock -w on 7744b44b, as S13's and N3's steps were in window 5
  flock -w 120 "$LK/gpu-7744b44b.lock" sh -c "echo $1 >> '$S/gpu_users'; sleep 1" &
}
pok() { head -1 "$H/dsv41-offload/PAIR_OK" 2>/dev/null || echo absent; }

out=$(pw open N2 2>&1); rc=$?
expect "open N2: rc 0, PAIR_OK names N2, the hold has both locks, the one-card worker stopped" \
  bash -c "[ $rc = 0 ] && [ '$(pok)' = N2 ] && ! grep -qx dsv41-dspark-worker.service '$S/active'"
expect "open N2: both lock files really held" held
lane S13; lane N3
sleep 0.5
out=$(pw status 2>&1)
expect "status names the two lanes queued on 7744b44b" bash -c "[ \$(grep -o 'gpu-7744b44b.lock' <<<\"\$1\" | wc -l) = 2 ]" _ "$out"

# --- the window-5 failure: hand must not open a gap the queued lanes take
out=$(pw hand N2 2>&1); rc=$?
sleep 1.5
expect "hand N2: rc 0, PAIR_OK names N2" bash -c "[ $rc = 0 ] && [ '$(pok)' = N2 ]"
expect "hand N2: the queued lanes did NOT get 7744b44b (no gap)" test ! -s "$S/gpu_users"
expect "hand N2: both locks still held by the hold" held

# --- the owner's commands run under the hold
out=$(pw run N2 -- sh -c "echo \"\$DSV41_PAIR_WINDOW\" > '$S/env'; sleep 1.5; [ ! -s '$S/gpu_users' ] && echo clean > '$S/during'; exit 3" 2>&1); rc=$?
expect "run N2: the command's exit status (3) comes back" test "$rc" = 3
expect "run N2: DSV41_PAIR_WINDOW=N2 in the command's environment" test "$(cat "$S/env" 2>/dev/null)" = N2
expect "run N2: no lane got 7744b44b while the command ran" test "$(cat "$S/during" 2>/dev/null)" = clean
expect "run N2: no lane got 7744b44b afterwards either (the hold stays)" bash -c "sleep 1; [ ! -s '$S/gpu_users' ]"
out=$(pw run N3 -- true 2>&1); rc=$?
expect "run N3 refused: PAIR_OK names N2" bash -c "[ $rc = 1 ] && grep -q 'PAIR_OK names N2, not N3' <<<\"\$1\"" _ "$out"

# --- one run at a time; hand / hold refuse during it
pw run N2 -- sleep 4 > "$T/bg.out" 2>&1 & bg=$!
sleep 1
out=$(pw run N2 -- true 2>&1); rc=$?
expect "a second run is refused while one is active" bash -c "[ $rc = 1 ] && grep -q 'another run is active' <<<\"\$1\"" _ "$out"
for c in "hand N3" "hold"; do
  out=$(pw $c 2>&1); rc=$?
  expect "$c refused while a run is active" bash -c "[ $rc = 1 ] && grep -q 'a run is active' <<<\"\$1\"" _ "$out"
done
expect "after the refusals the window is unchanged (PAIR_OK N2, hold kept)" bash -c "[ '$(pok)' = N2 ] && $HELD"
out=$(pw status 2>&1)
expect "status shows the active run" bash -c "grep -q '^run: N2, pid' <<<\"\$1\"" _ "$out"
wait "$bg"; rc=$?
expect "the background run ends with rc 0" test "$rc" = 0

# --- the hold disappears in the middle of a run: the command is stopped (its GPUs are no longer protected)
pw run N2 -- sh -c "echo \$\$ > '$S/victim.pid'; sleep 60; echo survived > '$S/victim.out'" > "$T/v.out" 2>&1 & bg=$!
sleep 1
HOME=$H PATH=$BIN:$PATH tmux kill-session -t dsv41-pairhold
t0=$SECONDS; wait "$bg"; rc=$?
expect "hold gone mid-run: run exits 1 within 5 s and says why" \
  bash -c "[ $rc = 1 ] && [ $((SECONDS - t0)) -le 5 ] && grep -q 'the hold is gone' '$T/v.out'"
expect "hold gone mid-run: the command (process group) was stopped" \
  bash -c "! kill -0 \$(cat '$S/victim.pid') 2>/dev/null && [ ! -e '$S/victim.out' ]"
expect "hold gone: the queued lanes now take 7744b44b (what the hold kept them from)" \
  bash -c "for _ in 1 2 3 4 5 6; do [ -s '$S/gpu_users' ] && exit 0; sleep 0.5; done; exit 1"

# --- recover, then close
out=$(pw hold 2>&1); rc=$?
expect "hold: re-taken after the lanes' steps, PAIR_OK COORD" bash -c "[ $rc = 0 ] && [ '$(pok)' = COORD ] && $HELD"
out=$(pw run COORD -- true 2>&1); rc=$?
expect "run COORD under the re-taken hold: rc 0" test "$rc" = 0

# --- close during a run (S12 R3 order): PAIR_OK goes first, the run stops its command, close waits for the run, and the
# hold goes only after: the queued lane gets the card after the command died, never next to it
lane S13b; sleep 0.3; : > "$S/gpu_users.t"
pw run COORD -- sh -c "echo \$\$ > '$S/gpujob.pid'; trap 'date +%s.%N > \"$S/job.dead\"; exit 0' TERM; sleep 60 & wait" \
  > "$T/r.out" 2>&1 & bg=$!
flock -w 120 "$LK/gpu-7744b44b.lock" sh -c "date +%s.%N > '$S/lane.t'" & tl=$!
sleep 1
out=$(pw close 2>&1); rc=$?
wait "$bg"; brc=$?; wait "$tl"
expect "close during a run: rc 0; the run ended (rc 1, stopped); the command died BEFORE a lane got 7744b44b" \
  bash -c "[ $rc = 0 ] && [ $brc = 1 ] && [ -s '$S/job.dead' ] && [ -s '$S/lane.t' ] && \
           awk -v d=\$(cat '$S/job.dead') -v l=\$(cat '$S/lane.t') 'BEGIN{exit !(d < l)}'"
rm -f "$S/gpujob.pid"

# --- an escaped job (S12 case B): CMD starts its GPU job in a new session without fd 9 and returns; close refuses with the
# hold kept while nvidia-smi lists it on either card
pw open N2 > /dev/null 2>&1
lane S13c; sleep 0.3; rm -f "$S/gpu_users"
out=$(pw run N2 -- python3 -c "
import subprocess, time
subprocess.Popen(['sh', '-c', 'echo \$\$ > $S/gpujob.pid; exec sleep 60'], start_new_session=True)
time.sleep(0.5)" > "$T/esc.out" 2>&1); rc=$?   # to a file: the job inherits stdout and would hold a $(...) pipe open
expect "escaped job: run returns rc 0 while the job lives" bash -c "[ $rc = 0 ] && kill -0 \$(cat '$S/gpujob.pid')"
out=$(pw close 2>&1); rc=$?
sleep 0.5
expect "escaped job: close refuses (names the compute process), the hold stays, no lane on the card" \
  bash -c "[ $rc = 1 ] && grep -q 'compute processes still on the pair: GPU-7744b44b-.*, $(cat "$S/gpujob.pid"), python' <<<\"\$1\" && \
           $HELD && [ ! -s '$S/gpu_users' ]" _ "$out"
kill "$(cat "$S/gpujob.pid")"; sleep 0.3
out=$(pw close 2>&1); rc=$?
expect "escaped job stopped: close again -> rc 0" test "$rc" = 0
rm -f "$S/gpujob.pid"; sleep 2.5

# --- the close race (S12 case D): run passes its first checks, close removes PAIR_OK and closes before run takes its run
# lock; run must re-check under the lock and refuse, the command never runs
pw open N2 > /dev/null 2>&1
DELAY_RUNLOCK_TAKE=2 HOME=$H PATH=$BIN:$PATH bash "$PW" run N2 -- touch "$S/race.ran" > "$T/race.out" 2>&1 & bg=$!
sleep 0.5
out=$(pw close 2>&1); crc=$?
wait "$bg"; rc=$?
expect "close race: close rc 0; the run that took its lock afterwards refuses and never runs its command" \
  bash -c "[ $crc = 0 ] && [ $rc = 1 ] && grep -q 'PAIR_OK no longer names N2' '$T/race.out' && [ ! -e '$S/race.ran' ]"

# --- stdin (S12 R4): run from a terminal, CMD reads stdin: it gets EOF from /dev/null instead of stopping on SIGTTIN
if command -v script > /dev/null; then
  pw open N2 > /dev/null 2>&1
  timeout -k 1 10 script -qec "HOME=$H PATH=$BIN:$PATH bash $PW run N2 -- sh -c 'read x; echo eof-ok > $S/stdin.ok'" \
    /dev/null > "$T/e.out" 2>&1; rc=$?
  expect "run from a terminal, CMD reads stdin: gets EOF, run rc 0 (no SIGTTIN stop)" \
    bash -c "[ $rc = 0 ] && [ -s '$S/stdin.ok' ]"
  pw close > /dev/null 2>&1
fi

# --- an orphan that kept fd 9 (S12 case A): status names it, close waits (bounded) and refuses with the hold kept
pw open N2 > /dev/null 2>&1
pw run N2 -- sh -c "setsid sh -c 'echo \$\$ > $S/orphan.pid; exec sleep 30' & sleep 0.3" > /dev/null 2>&1
out=$(pw status 2>&1)
expect "status names the orphan holding the run lock" \
  bash -c "grep -q \"^run: active, holder unknown: no run process; pid(s) holding the run lock: .*\$(cat '$S/orphan.pid')\" <<<\"\$1\"" _ "$out"
rc=0; out=$(PAIR_CLOSE_WAIT_S=2 pw close 2>&1) || rc=$?
expect "orphan: close waits PAIR_CLOSE_WAIT_S, refuses naming it, the hold stays" \
  bash -c "[ $rc = 1 ] && grep -q 'a run is still active after 2 s' <<<\"\$1\" && $HELD" _ "$out"
kill "$(cat "$S/orphan.pid")"; sleep 0.3
out=$(pw close 2>&1); rc=$?
expect "close: rc 0, PAIR_OK removed, hold gone, one-card worker started, locks free" \
  bash -c "[ $rc = 0 ] && [ '$(pok)' = absent ] && grep -qx dsv41-dspark-worker.service '$S/active' && \
           flock -n '$LK/gpu-c4e7a371.lock' true && flock -n '$LK/gpu-7744b44b.lock' true"
out=$(pw run N2 -- true 2>&1); rc=$?
expect "run with no window open: refused" bash -c "[ $rc = 1 ] && grep -q 'no pair window is open' <<<\"\$1\"" _ "$out"

# --- after promotion run refuses like the other subcommands
echo dsv41-dspark-worker-tp2-prod.service > "$S/enabled"
echo N2 > "$H/dsv41-offload/PAIR_OK"
out=$(pw run N2 -- true 2>&1); rc=$?
expect "run refuses while the TP2 production unit is enabled" \
  bash -c "[ $rc = 1 ] && grep -q \"refusing 'run'\" <<<\"\$1\"" _ "$out"

echo "test_pair_window_handoff: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
