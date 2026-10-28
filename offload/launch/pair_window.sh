#!/usr/bin/env bash
# pair_window.sh: coordinator PAIR WINDOW on x86host's NVLink pair (GPU-c4e7a371 + GPU-7744b44b), per the GPU ownership
# protocol in documents/plans/2026-10-23-dual3090-plan.md. Run ON x86host; open, hand, hold and close are coordinator-only.
#   pair_window.sh open OWNER         take both GPU locks (waits for the lane holding 7744b44b to finish its step), stop
#                                     the one-card production worker, write ~/dsv41-offload/PAIR_OK naming OWNER. The
#                                     coordinator's hold (tmux session dsv41-pairhold) keeps both locks for the WHOLE
#                                     window: OWNER runs its GPU commands with `run`, never with its own flock (a released
#                                     hold goes to the lanes queued in `flock -w`, not to a polling owner: window 5). A
#                                     third argument ("hold", the old form) is accepted and changes nothing.
#   pair_window.sh hand OWNER         re-name the window's owner; the hold stays (no gap a queued lane could take)
#   pair_window.sh run OWNER -- CMD   OWNER (PAIR_OK must name it; COORD for the coordinator) runs CMD while the hold
#                                     keeps both locks. One run at a time; CMD gets DSV41_PAIR_WINDOW=OWNER; CMD is stopped
#                                     (SIGTERM to its process group, SIGKILL 30 s later) when the hold disappears or
#                                     PAIR_OK stops naming OWNER; exits with CMD's status
#   pair_window.sh hold               re-take both locks for the coordinator inside an open window (PAIR_OK -> COORD)
#   pair_window.sh close              in this order (S12 R3): remove PAIR_OK (no run can start or continue); wait until
#                                     no run is active (at most PAIR_CLOSE_WAIT_S, default 60 s: a run stops its command
#                                     within ~31 s once PAIR_OK is gone); stop lane E's TP2 unit if active; WITH THE HOLD
#                                     STILL HELD require that no compute process runs on either card (else refuse and
#                                     keep the hold: an escaped job, e.g. Popen(start_new_session=True), is still there);
#                                     only then drop the hold, start the one-card worker and wait for its ready line.
#                                     After a refusal: stop what it names, run close again (or `open COORD` to go on)
#   pair_window.sh status             PAIR_OK, holder, the active run (or an orphan holding the run lock), lanes queued on
#                                     the locks, units, both cards
# hand and hold refuse while a run is active. A child that leaves the run's process group but keeps its fd 9 (the run
# lock) is not stopped by run: close refuses while it lives and status names it; the coordinator stops it by hand. The Spark engine runs target-only rounds while no worker answers and
# reconnects by itself (backoff cap 5 s).
# BEFORE PROMOTION ONLY: open, hand, run, hold and close refuse while the TP2 production unit
# (dsv41-dspark-worker-tp2-prod) is enabled or running, because GPU-7744b44b is then production rank 1 and the unit
# itself holds both locks; windows after promotion go through offload/launch/prod_window.sh (S12 F1).
set -euo pipefail
D=$HOME/dsv41-offload; L=$D/locks; PAIR_OK=$D/PAIR_OK; MARK=$L/.pairhold-acquired
LOCKS=("$L/gpu-c4e7a371.lock" "$L/gpu-7744b44b.lock")
RUNLOCK=$L/.pairrun.lock; RUNINFO=$L/.pairrun
ONE=dsv41-dspark-worker.service; TP2=dsv41-dspark-worker-tp2.service; PROD_TP2=dsv41-dspark-worker-tp2-prod.service
ONE_LOG=$D/prod/logs/dspark-worker.log
CLOSE_WAIT_S=${PAIR_CLOSE_WAIT_S:-60}
log() { echo "$(date -u +%FT%TZ) pair_window: $*"; }
refuse() { echo "pair_window: refusing $*" >&2; exit 1; }

not_promoted() {   # refuse before touching anything once the TP2 production unit owns the pair
  local en=no st
  systemctl --user is-enabled --quiet $PROD_TP2 2>/dev/null && en=yes
  st=$(systemctl --user is-active $PROD_TP2 2>/dev/null || true)
  case "$en/$st" in
    yes/*|*/active|*/activating|*/deactivating|*/reloading|*/refreshing)
      echo "pair_window: refusing '$1': $PROD_TP2 is enabled ($en) / $st: GPU-7744b44b is production rank 1 and that" \
           "unit holds both GPU locks; use offload/launch/prod_window.sh (training and pair windows after promotion," \
           "documents/plans/2026-10-23-dual3090-plan.md)" >&2
      exit 1 ;;
  esac
}

take_hold() {
  tmux has-session -t dsv41-pairhold 2>/dev/null && { log "hold already taken"; return; }
  rm -f "$MARK"
  tmux new-session -d -s dsv41-pairhold \
    "flock $L/gpu-c4e7a371.lock flock $L/gpu-7744b44b.lock sh -c 'date -u > $MARK; exec sleep infinity'"
  log "waiting for both GPU locks (a lane step may hold 7744b44b for up to 15 min)"
  for _ in $(seq 1 1800); do [ -f "$MARK" ] && { log "both locks held by dsv41-pairhold"; return; }; sleep 1; done
  tmux kill-session -t dsv41-pairhold; echo "pair_window: locks not acquired in 30 min" >&2; exit 1
}
drop_hold() {
  tmux kill-session -t dsv41-pairhold 2>/dev/null && log "hold released" || true
  rm -f "$MARK"
}
hold_alive() {   # the hold's session runs AND both locks are actually held
  local l
  tmux has-session -t dsv41-pairhold 2>/dev/null && [ -f "$MARK" ] || return 1
  for l in "${LOCKS[@]}"; do flock -n "$l" true && return 1; done
  return 0
}
run_active() { [ -e "$RUNLOCK" ] && ! flock -n "$RUNLOCK" true; }
run_desc() {   # the active run, or what holds the run lock when its run process is gone (an orphan that kept fd 9)
  local h
  if [ -s "$RUNINFO" ]; then cat "$RUNINFO"; return; fi
  h=$(lock_holders "$RUNLOCK")
  echo "active, holder unknown: no run process; pid(s) holding the run lock: ${h:-none found} (an orphan that kept fd 9)"
}
no_run() { run_active && refuse "'$1': a run is active ($(run_desc)); wait for it"; return 0; }
# Waiters come from the kernel ("->" lines for the lock file's dev:inode in /proc/locks), never from lslocks (S12 R1:
# util-linux 2.39 on x86host misreports flock holders)
lock_key() {   # dev:inode as /proc/locks prints it (major:minor in hex, the kernel's encoding of st_dev)
  local d i
  d=$(stat -c %d "$1" 2>/dev/null) && i=$(stat -c %i "$1") || return 1
  printf '%02x:%02x:%s' $(( (d >> 8) & 0xfff )) $(( (d & 0xff) | ((d >> 12) & 0xfff00) )) "$i"
}
lock_holders() {   # lock_holders PATH: pids with their own fd on PATH carrying a "lock:" line for its dev:inode
  local key f pid
  key=$(lock_key "$1") || return 0
  find /proc/[0-9]*/fd -maxdepth 1 -lname "$1" 2>/dev/null | while read -r f; do
    pid=${f#/proc/}; pid=${pid%%/*}
    grep -q "^lock:.* $key " "/proc/$pid/fdinfo/${f##*/}" 2>/dev/null && echo "$pid"
  done | sort -un | tr '\n' ' ' | sed 's/ $//'
}
waiters() {   # "pid(path)" for each process queued in flock on a pair lock
  local l key p
  for l in "${LOCKS[@]}"; do
    key=$(lock_key "$l") || continue
    for p in $(awk -v k="$key" '$2 == "->" && $3 == "FLOCK" && $7 == k {print $6}' /proc/locks | sort -un); do
      printf '%s(%s) ' "$p" "$l"
    done
  done | sed 's/ $//'
}
cards() {
  nvidia-smi --query-gpu=uuid,memory.used,power.draw,power.limit,clocks.sm,temperature.gpu --format=csv,noheader \
    | grep -E "c4e7a371|7744b44b"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | grep -E "c4e7a371|7744b44b" || true
}

case "${1:-}" in
  open|hand|run|hold|close) not_promoted "$1" ;;
esac

case "${1:-}" in
  open)
    owner=${2:?owner (COORD or a lane name)}
    [ -e "$PAIR_OK" ] && { echo "pair_window: already open: $(cat "$PAIR_OK")" >&2; exit 1; }
    take_hold
    systemctl --user stop $ONE; log "one-card worker stopped"
    printf '%s\n' "$owner" "opened $(date -u +%FT%TZ) by the coordinator (pair_window.sh)" > "$PAIR_OK"
    log "PAIR_OK -> $owner; the hold keeps both locks: $owner runs GPU commands with pair_window.sh run $owner -- CMD"
    cards ;;
  hand)
    owner=${2:?owner}; [ -e "$PAIR_OK" ] || { echo "pair_window: not open" >&2; exit 1; }
    no_run hand
    systemctl --user is-active --quiet $TP2 && { echo "pair_window: stop the TP2 unit first (stop_tp2.sh)" >&2; exit 1; }
    take_hold                                  # normally a no-op: the hold stays for the whole window
    printf '%s\n' "$owner" "handed $(date -u +%FT%TZ) by the coordinator (pair_window.sh)" > "$PAIR_OK"
    log "PAIR_OK -> $owner (the hold stays; $owner runs GPU commands with pair_window.sh run $owner -- CMD)" ;;
  run)
    owner=${2:?owner}; shift 2; [ "${1:-}" = -- ] && shift
    [ "$#" -gt 0 ] || { echo "usage: pair_window.sh run OWNER -- CMD [ARG...]" >&2; exit 2; }
    [ -e "$PAIR_OK" ] || refuse "run: no pair window is open"
    named=$(head -1 "$PAIR_OK")
    [ "$named" = "$owner" ] || refuse "run: PAIR_OK names $named, not $owner"
    hold_alive || refuse "run: the coordinator's hold does not hold both locks (pair_window.sh hold)"
    if [ "$owner" != COORD ] && systemctl --user is-active --quiet $TP2; then
      refuse "run: the TP2 unit runs on the pair (stop_tp2.sh first)"
    fi
    exec 9>>"$RUNLOCK"                         # the run lock; CMD inherits it, so it lasts as long as CMD does
    flock -n 9 || refuse "run: another run is active ($(run_desc))"
    # again under the run lock: a close that removed PAIR_OK meanwhile waits for this lock, so it must see us refuse
    [ "$(head -1 "$PAIR_OK" 2>/dev/null)" = "$owner" ] || refuse "run: PAIR_OK no longer names $owner (the window is closing)"
    hold_alive || refuse "run: the coordinator's hold does not hold both locks (pair_window.sh hold)"
    printf '%s\n' "$owner, pid $$, since $(date -u +%FT%TZ): $*" > "$RUNINFO"
    log "run for $owner under the hold: $*"
    set -m                                     # CMD in its own process group, so a stop reaches its children
    DSV41_PAIR_WINDOW=$owner "$@" </dev/null &  # stdin /dev/null: a background group reading a tty stops (SIGTTIN)
    cpid=$!
    set +m
    stop_cmd() {
      kill -TERM -- "-$cpid" 2>/dev/null || true
      for _ in $(seq 1 30); do kill -0 "$cpid" 2>/dev/null || break; sleep 1; done
      kill -KILL -- "-$cpid" 2>/dev/null || true
      wait "$cpid" 2>/dev/null || true
      rm -f "$RUNINFO"
    }
    trap 'echo "pair_window: run for $owner interrupted: stopping the command" >&2; stop_cmd; exit 130' INT TERM HUP
    while kill -0 "$cpid" 2>/dev/null; do
      if ! hold_alive || [ "$(head -1 "$PAIR_OK" 2>/dev/null)" != "$owner" ]; then
        echo "pair_window: run for $owner: the hold is gone or PAIR_OK no longer names $owner: stopping the command" >&2
        stop_cmd; exit 1
      fi
      sleep 1
    done
    rc=0; wait "$cpid" || rc=$?
    rm -f "$RUNINFO"
    log "run for $owner ended (rc $rc)"
    exit "$rc" ;;
  hold)
    [ -e "$PAIR_OK" ] || { echo "pair_window: not open" >&2; exit 1; }
    no_run hold
    take_hold
    printf '%s\n' "COORD" "held $(date -u +%FT%TZ) by the coordinator (pair_window.sh)" > "$PAIR_OK" ;;
  close)
    rm -f "$PAIR_OK"; log "PAIR_OK removed: no run can start, a running one stops its command"
    t0=$SECONDS
    while run_active; do
      [ $((SECONDS - t0)) -lt "$CLOSE_WAIT_S" ] || refuse "close: a run is still active after $CLOSE_WAIT_S s" \
        "($(run_desc)); the hold stays: stop it, then run close again (or open COORD to go on)"
      sleep 1
    done
    if systemctl --user is-active --quiet $TP2; then systemctl --user stop $TP2; log "TP2 unit stopped"; fi
    hold_alive || log "WARNING: the hold does not hold both locks (a lane may already be on 7744b44b)"
    apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader) \
      || refuse "close: nvidia-smi cannot list compute processes; the hold stays"
    apps=$(grep -E "c4e7a371|7744b44b" <<<"$apps" || true)
    [ -z "$apps" ] || refuse "close: compute processes still on the pair: $(tr '\n' ';' <<<"$apps")" \
      "the hold stays: stop them (an escaped job of the window?), then run close again (or open COORD to go on)"
    drop_hold
    before=$(grep -c "worker ready in" "$ONE_LOG" 2>/dev/null || true)
    systemctl --user start $ONE
    for _ in $(seq 1 240); do
      [ "$(grep -c 'worker ready in' "$ONE_LOG" 2>/dev/null || true)" -gt "${before:-0}" ] && break
      systemctl --user is-active --quiet $ONE || { echo "pair_window: one-card worker died" >&2; tail -20 "$ONE_LOG" >&2; exit 1; }
      sleep 1
    done
    grep "worker ready in" "$ONE_LOG" | tail -1; cards ;;
  status)
    echo "PAIR_OK: $( [ -e "$PAIR_OK" ] && tr '\n' ' ' < "$PAIR_OK" || echo absent)"
    echo "hold: $(tmux has-session -t dsv41-pairhold 2>/dev/null && cat "$MARK" 2>/dev/null || echo none)"
    echo "run: $(run_active && run_desc || echo none)"
    echo "queued on the locks: $(w=$(waiters); echo "${w:-none}")"
    echo "$ONE: $(systemctl --user is-active $ONE || true) | $TP2: $(systemctl --user is-active $TP2 || true)" \
         "| $PROD_TP2: $(systemctl --user is-active $PROD_TP2 2>/dev/null || true)," \
         "$(systemctl --user is-enabled $PROD_TP2 2>/dev/null || true)"
    cards ;;
  *) sed -n 2,30p "$0"; exit 2 ;;
esac
