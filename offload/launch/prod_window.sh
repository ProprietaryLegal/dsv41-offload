#!/usr/bin/env bash
# prod_window.sh: coordinator windows on x86host's NVLink pair AFTER the TP2 promotion (R3-PROMOTE, S12 F1), per
# "GPU ownership after the TP2 promotion" in documents/plans/2026-10-23-dual3090-plan.md. Coordinator-only; run ON x86host.
# Production rank 0 + 1 = dsv41-dspark-worker-tp2-prod (enabled; its worker holds both GPU locks through
# worker/run/gpu_locked.sh). It stays ENABLED through every window, so a reboot brings TP2 production back on its own.
#   prod_window.sh train OWNER   training window: stop the TP2 worker (its locks come free), start the one-card production
#                                worker on GPU-c4e7a371 (same `current`, same Spark config: no Spark restart), wait for its
#                                ready line, write ~/dsv41-offload/d2/TRAIN_OK naming OWNER. OWNER may use GPU-7744b44b.
#   prod_window.sh pair OWNER    pair window: stop the TP2 worker, no worker at all (the Spark runs target-only rounds
#                                until close), write ~/dsv41-offload/PAIR_OK naming OWNER. OWNER may use both cards.
#   prod_window.sh close         remove TRAIN_OK and PAIR_OK, wait until both GPU locks are free (a lane exits within 60 s
#                                of its marker's removal; at most PROD_WINDOW_WAIT_S, default 900 s), stop the one-card
#                                worker if it runs, start the TP2 worker, wait for its ready line and `tp 2`, then check.
#                                A TP2 unit that already runs and holds both locks (started at boot inside the window)
#                                goes straight to the check
#   prod_window.sh check         read-only production checks: units (TP2 enabled + active, one-card disabled + inactive),
#                                no window markers, `flock -n` on both locks FAILS and the TP2 unit's main pid holds both
#                                (its /proc/<pid>/fdinfo lock lines), no process waits in flock on either lock (the 10 s
#                                restart gap), `pair_window.sh open` refuses, status file `tp 2`
#   prod_window.sh status        markers, units, lock holders, lanes queued on the locks, both cards
# train and pair refuse while any process is queued in flock on a pair lock: a queued `flock -w` takes the card the
# moment the TP2 worker stops, before OWNER (the window-5 starvation; lanes use flock -n only after promotion).
# The Spark engine runs target-only rounds while no worker answers and reconnects by itself (backoff cap 5 s); both units
# share the listener, status file and log, so the Spark side never changes.
set -euo pipefail
D=$HOME/dsv41-offload; L=$D/locks
PAIR_OK=$D/PAIR_OK; TRAIN_OK=$D/d2/TRAIN_OK
LOCKS=("$L/gpu-c4e7a371.lock" "$L/gpu-7744b44b.lock")
ONE=dsv41-dspark-worker.service; PROD=dsv41-dspark-worker-tp2-prod.service
LOG=$D/prod/logs/dspark-worker.log; STATUS=$D/prod/run/dspark-worker.status.json
PW=${PAIR_WINDOW_SH:-$D/pair_window.sh}
WAIT_S=${PROD_WINDOW_WAIT_S:-900}; READY_S=${PROD_WINDOW_READY_S:-300}
log() { echo "$(date -u +%FT%TZ) prod_window: $*"; }
die() { echo "prod_window: $*" >&2; exit 1; }

state() { systemctl --user is-active "$1" 2>/dev/null || true; }
enabled() { local o; o=$(systemctl --user is-enabled "$1" 2>/dev/null) || true; echo "${o:-not-installed}"; }
need_promoted() {
  systemctl --user is-enabled --quiet $PROD 2>/dev/null \
    || die "refusing '$1': $PROD is not enabled ($(enabled $PROD)): not promoted; before promotion use pair_window.sh"
}
no_window() {
  local m
  for m in "$TRAIN_OK" "$PAIR_OK"; do
    [ -e "$m" ] && die "refusing '$1': a window is open: $m ($(head -1 "$m")); prod_window.sh close first"
  done
  return 0
}
lock_free() { flock -n "$1" true; }
# Lock holders and waiters come from the kernel, never from lslocks (S12 R1): gpu_locked.sh (and flock(1)) take a lock
# through a short-lived flock process, /proc/locks records that process's pid, and util-linux 2.39's lslocks (x86host)
# then names a dead pid. A holder is a process with its own fd on the lock file whose /proc/<pid>/fdinfo has a "lock:"
# line for the file's dev:inode; a waiter is a "->" line for that dev:inode in /proc/locks (S12 R2).
lock_key() {   # dev:inode as /proc/locks prints it (major:minor in hex, the kernel's encoding of st_dev)
  local d i
  d=$(stat -c %d "$1" 2>/dev/null) && i=$(stat -c %i "$1") || return 1
  printf '%02x:%02x:%s' $(( (d >> 8) & 0xfff )) $(( (d & 0xff) | ((d >> 12) & 0xfff00) )) "$i"
}
lock_holders() {   # lock_holders PATH [PID]: the pids (of PID only, if given) holding PATH, space separated
  local key f pid
  key=$(lock_key "$1") || return 0
  find /proc/${2:-[0-9]*}/fd -maxdepth 1 -lname "$1" 2>/dev/null | while read -r f; do
    pid=${f#/proc/}; pid=${pid%%/*}
    grep -q "^lock:.* $key " "/proc/$pid/fdinfo/${f##*/}" 2>/dev/null && echo "$pid"
  done | sort -un | tr '\n' ' ' | sed 's/ $//'
}
lock_waiters() {   # lock_waiters PATH: the pids blocked in flock on PATH
  local key
  key=$(lock_key "$1") || return 0
  awk -v k="$key" '$2 == "->" && $3 == "FLOCK" && $7 == k {print $6}' /proc/locks | sort -un | tr '\n' ' ' | sed 's/ $//'
}
holders() { local h; h=$(lock_holders "$1"); echo "${h:-no live holder}"; }
waiters() {   # every lock: "pid(path)" for each process queued on it
  local l p
  for l in "${LOCKS[@]}"; do for p in $(lock_waiters "$l"); do printf '%s(%s) ' "$p" "$l"; done; done | sed 's/ $//'
}
wait_locks_free() {   # wait_locks_free SECONDS WHY
  local t0=$SECONDS f
  while :; do
    f=0; for l in "${LOCKS[@]}"; do lock_free "$l" || f=1; done
    [ "$f" = 0 ] && return 0
    if [ $((SECONDS - t0)) -ge "$1" ]; then
      for l in "${LOCKS[@]}"; do lock_free "$l" || echo "  held: $l by $(holders "$l")" >&2; done
      die "$2: GPU locks still held after $1 s"
    fi
    [ $(((SECONDS - t0) % 60)) -lt 2 ] && for l in "${LOCKS[@]}"; do
      lock_free "$l" || log "waiting for $l (held by $(holders "$l"))"; done
    sleep 2
  done
}
ready_count() { grep -c "worker ready in" "$LOG" 2>/dev/null || true; }
wait_ready() {   # wait_ready UNIT BEFORE
  local t0=$SECONDS st
  while [ "$(ready_count)" -le "${2:-0}" ]; do
    st=$(state "$1")
    case "$st" in active|activating) ;; *) tail -20 "$LOG" >&2; die "$1 is $st before its ready line";; esac
    [ $((SECONDS - t0)) -ge "$READY_S" ] && { tail -20 "$LOG" >&2; die "$1: no ready line in $READY_S s"; }
    sleep 1
  done
  grep "worker ready in" "$LOG" | tail -1
}
status_tp() {
  python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("tp", 1))' "$STATUS" 2>/dev/null || echo unknown
}
stop_prod() {
  if [ "$(state $PROD)" != inactive ]; then systemctl --user stop $PROD; log "$PROD stopped"; fi
  wait_locks_free 40 "after stopping $PROD"
  log "both GPU locks free"
}
cards() {
  nvidia-smi --query-gpu=uuid,memory.used,power.draw,power.limit,clocks.sm,temperature.gpu --format=csv,noheader \
    | grep -E "c4e7a371|7744b44b" || true
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader \
    | grep -E "c4e7a371|7744b44b" || true
}
mark() {   # mark FILE OWNER WHAT
  mkdir -p "$(dirname "$1")"
  printf '%s\n' "$2" "$3 $(date -u +%FT%TZ) by the coordinator (prod_window.sh)" > "$1.tmp"
  mv "$1.tmp" "$1"
}

check() {
  local ok=1 l pid main out rc
  main=$(systemctl --user show -p MainPID --value $PROD 2>/dev/null || echo 0)
  echo "$PROD: $(state $PROD), $(enabled $PROD) (main pid $main) | $ONE: $(state $ONE), $(enabled $ONE)"
  [ "$(state $PROD)" = active ] && [ "$(enabled $PROD)" = enabled ] || { echo "FAIL: $PROD not enabled + active"; ok=0; }
  [ "$(state $ONE)" = inactive ] || [ "$(state $ONE)" = failed ] || { echo "FAIL: $ONE is $(state $ONE)"; ok=0; }
  [ "$(enabled $ONE)" = disabled ] || { echo "FAIL: $ONE is $(enabled $ONE), not disabled"; ok=0; }
  for l in "$TRAIN_OK" "$PAIR_OK"; do [ -e "$l" ] && { echo "FAIL: window marker $l present"; ok=0; }; done
  for l in "${LOCKS[@]}"; do
    if lock_free "$l"; then echo "FAIL: flock -n $l true SUCCEEDED: production does not hold it"; ok=0; continue; fi
    if [ "$main" != 0 ] && [ -n "$(lock_holders "$l" "$main")" ]; then
      echo "ok: flock -n $l fails; held by the TP2 worker (main pid $main, its fdinfo lock line)"
    else echo "FAIL: $l is held, but not by the TP2 worker (main pid $main): holders $(holders "$l")"; ok=0; fi
    w=$(lock_waiters "$l")
    if [ -z "$w" ]; then echo "ok: no process waits on $l"
    else echo "FAIL: pid(s) $w wait in flock on $l: they would take the card in the TP2 unit's 10 s restart gap" \
              "(after promotion lanes use flock -n under a marker only)"; ok=0; fi
  done
  if [ -f "$PW" ]; then
    rc=0; out=$(timeout 20 bash "$PW" open PROD-CHECK 2>&1) || rc=$?
    if [ "$rc" != 0 ] && grep -q "refusing" <<<"$out" && [ ! -e "$PAIR_OK" ]; then
      echo "ok: pair_window.sh open refuses (rc $rc)"
    else
      echo "FAIL: pair_window.sh open did not refuse (rc $rc): $out"; ok=0
      tmux kill-session -t dsv41-pairhold 2>/dev/null || true
    fi
  else echo "FAIL: no $PW (install the reviewed offload/launch/pair_window.sh there)"; ok=0; fi
  [ "$(status_tp)" = 2 ] && echo "ok: status file tp 2" || { echo "FAIL: status file tp $(status_tp)"; ok=0; }
  [ "$ok" = 1 ] && echo "PRODUCTION HOLD OK" || { echo "CHECK FAILED"; return 1; }
}

case "${1:-}" in
  train|pair)
    what=$1; owner=${2:?owner (a lane name)}
    need_promoted "$what"; no_window "$what"
    w=$(waiters)
    [ -z "$w" ] || die "refusing '$what': processes queued on the GPU locks would take the card the moment the TP2 worker
  stops, not $owner: $w (after promotion lanes use flock -n under a marker only; stop them first)"
    stop_prod
    if [ "$what" = train ]; then
      before=$(ready_count); systemctl --user start $ONE; log "$ONE starting on GPU-c4e7a371"
      wait_ready $ONE "$before"
      lock_free "${LOCKS[1]}" || die "GPU-7744b44b lock is held after the swap: $(holders "${LOCKS[1]}")"
      mark "$TRAIN_OK" "$owner" "training window opened"
      log "TRAIN_OK -> $owner (GPU-7744b44b only; production = one-card worker on GPU-c4e7a371)"
    else
      [ "$(state $ONE)" = inactive ] || { systemctl --user stop $ONE; log "$ONE stopped"; }
      mark "$PAIR_OK" "$owner" "pair window opened"
      log "PAIR_OK -> $owner (both cards; no production worker: the Spark runs target-only rounds until close)"
    fi
    cards ;;
  close)
    need_promoted close
    for m in "$TRAIN_OK" "$PAIR_OK"; do [ -e "$m" ] && { rm -f "$m"; log "removed $m"; }; done
    main=$(systemctl --user show -p MainPID --value $PROD 2>/dev/null || echo 0)
    if [ "$(state $PROD)" = active ] && [ "$main" != 0 ] && [ -n "$(lock_holders "${LOCKS[0]}" "$main")" ] \
       && [ -n "$(lock_holders "${LOCKS[1]}" "$main")" ]; then
      log "$PROD already runs and holds both locks (main pid $main; e.g. started at boot inside the window)"
      check; exit $?
    fi
    wait_locks_free "$WAIT_S" "close"
    if [ "$(state $ONE)" != inactive ]; then systemctl --user stop $ONE; log "$ONE stopped"; fi
    before=$(ready_count); systemctl --user start $PROD; log "$PROD starting"
    wait_ready $PROD "$before"
    for _ in $(seq 1 15); do [ "$(status_tp)" = 2 ] && break; sleep 1; done
    cards
    check ;;
  check) check ;;
  status)
    for m in "$TRAIN_OK" "$PAIR_OK"; do echo "$m: $([ -e "$m" ] && tr '\n' ' ' < "$m" || echo absent)"; done
    echo "$PROD: $(state $PROD), $(enabled $PROD) | $ONE: $(state $ONE), $(enabled $ONE)"
    for l in "${LOCKS[@]}"; do echo "$l: $(lock_free "$l" && echo free || echo "held by $(holders "$l")")"; done
    echo "queued on the locks: $(w=$(waiters); echo "${w:-none}")"
    echo "status file tp: $(status_tp)"
    cards ;;
  *) sed -n 2,24p "$0"; exit 2 ;;
esac
