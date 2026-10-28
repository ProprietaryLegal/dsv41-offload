#!/usr/bin/env bash
# idle_gate.sh: run a short command that sends traffic over a Spark NIC only while the production serve is idle
# (COMMON.md rule 2). Waits until /metrics shows vllm:num_requests_running 0 and num_requests_waiting 0 on N
# consecutive polls, runs the command, and kills it the moment a request appears (exit 75 = preempted, retry later).
#   idle_gate.sh [-n polls] [-w max_wait_s] -- command args...
# Polls from wherever it runs (Controller / x86host / a Spark host): http://HEAD_IP:8000/metrics.
set -u
METRICS=${METRICS:-http://HEAD_IP:8000/metrics}
need=3
max_wait=1800
while [ $# -gt 0 ]; do
  case "$1" in
    -n) need=$2; shift 2 ;;
    -w) max_wait=$2; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
[ $# -gt 0 ] || { echo "usage: idle_gate.sh [-n polls] [-w max_wait_s] -- command..." >&2; exit 64; }

busy() {  # 0 = idle, 1 = busy, 2 = metrics unreadable (treated as busy)
  local m r w
  m=$(curl -s -m 2 "$METRICS") || return 2
  r=$(printf '%s\n' "$m" | awk '/^vllm:num_requests_running\{/ {s+=$NF} END {print s+0}')
  w=$(printf '%s\n' "$m" | awk '/^vllm:num_requests_waiting\{/ {s+=$NF} END {print s+0}')
  [ -n "$r" ] && [ -n "$w" ] || return 2
  awk -v r="$r" -v w="$w" 'BEGIN {exit !(r == 0 && w == 0)}' && return 0
  return 1
}

start=$(date +%s)
ok=0
while [ "$ok" -lt "$need" ]; do
  if busy; then ok=$((ok + 1)); else ok=0; fi
  [ $(( $(date +%s) - start )) -lt "$max_wait" ] || { echo "idle_gate: serve not idle within ${max_wait}s" >&2; exit 75; }
  [ "$ok" -ge "$need" ] || sleep 1
done
echo "idle_gate: serve idle (${need} polls), running: $*" >&2
"$@" &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
  if ! busy; then
    echo "idle_gate: a request appeared, stopping pid $pid" >&2
    kill -TERM "$pid" 2>/dev/null; sleep 0.3; kill -KILL "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    exit 75
  fi
  sleep 0.5
done
wait "$pid"
