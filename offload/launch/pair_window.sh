#!/usr/bin/env bash
# pair_window.sh: coordinator PAIR WINDOW on x86host's NVLink pair (GPU-c4e7a371 + GPU-7744b44b), per the GPU ownership
# protocol in documents/plans/2026-10-23-dual3090-plan.md. Coordinator-only; run ON x86host.
#   pair_window.sh open OWNER [hold]  take both GPU locks (waits for the lane holding 7744b44b to finish its step), stop
#                                     the one-card production worker, write ~/dsv41-offload/PAIR_OK naming OWNER. With
#                                     "hold" the coordinator keeps both locks (tmux session dsv41-pairhold) for its own
#                                     units; without it the locks are released for OWNER's own flock runner.
#   pair_window.sh hand OWNER         re-name the window's owner; drops the coordinator's hold (OWNER takes the locks)
#   pair_window.sh hold               re-take both locks for the coordinator inside an open window
#   pair_window.sh close              remove PAIR_OK, stop the TP2 unit if active, drop the hold, start the one-card worker
#                                     and wait for its ready line
#   pair_window.sh status             PAIR_OK, holder, units, both cards
# The Spark engine runs target-only rounds while no worker answers and reconnects by itself (backoff cap 5 s).
set -euo pipefail
D=$HOME/dsv41-offload; L=$D/locks; PAIR_OK=$D/PAIR_OK; MARK=$L/.pairhold-acquired
ONE=dsv41-dspark-worker.service; TP2=dsv41-dspark-worker-tp2.service
ONE_LOG=$D/prod/logs/dspark-worker.log
log() { echo "$(date -u +%FT%TZ) pair_window: $*"; }

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
cards() {
  nvidia-smi --query-gpu=uuid,memory.used,power.draw,power.limit,clocks.sm,temperature.gpu --format=csv,noheader \
    | grep -E "c4e7a371|7744b44b"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | grep -E "c4e7a371|7744b44b" || true
}

case "${1:-}" in
  open)
    owner=${2:?owner (COORD or a lane name)}
    [ -e "$PAIR_OK" ] && { echo "pair_window: already open: $(cat "$PAIR_OK")" >&2; exit 1; }
    take_hold
    systemctl --user stop $ONE; log "one-card worker stopped"
    printf '%s\n' "$owner" "opened $(date -u +%FT%TZ) by the coordinator (pair_window.sh)" > "$PAIR_OK"
    [ "${3:-}" = hold ] || drop_hold
    log "PAIR_OK -> $owner"; cards ;;
  hand)
    owner=${2:?owner}; [ -e "$PAIR_OK" ] || { echo "pair_window: not open" >&2; exit 1; }
    systemctl --user is-active --quiet $TP2 && { echo "pair_window: stop the TP2 unit first (stop_tp2.sh)" >&2; exit 1; }
    drop_hold
    printf '%s\n' "$owner" "handed $(date -u +%FT%TZ) by the coordinator (pair_window.sh)" > "$PAIR_OK"
    log "PAIR_OK -> $owner" ;;
  hold)
    [ -e "$PAIR_OK" ] || { echo "pair_window: not open" >&2; exit 1; }
    take_hold
    printf '%s\n' "COORD" "held $(date -u +%FT%TZ) by the coordinator (pair_window.sh)" > "$PAIR_OK" ;;
  close)
    rm -f "$PAIR_OK"; log "PAIR_OK removed"
    if systemctl --user is-active --quiet $TP2; then systemctl --user stop $TP2; log "TP2 unit stopped"; fi
    drop_hold
    others=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader | grep c4e7a371 || true)
    [ -z "$others" ] || { echo "pair_window: c4e7a371 still in use: $others" >&2; exit 1; }
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
    echo "$ONE: $(systemctl --user is-active $ONE || true) | $TP2: $(systemctl --user is-active $TP2 || true)"
    cards ;;
  *) sed -n 2,14p "$0"; exit 2 ;;
esac
