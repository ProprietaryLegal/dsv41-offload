#!/bin/sh
# gpu_locked.sh LOCK... -- CMD [ARG...]: the production hold on x86host's NVLink pair (R3-PROMOTE, S12 F1).
# Opens every LOCK, takes an exclusive flock on each WITHOUT waiting, then execs CMD in this same process. CMD (the TP2
# production worker, the unit's main pid) therefore holds the locks for exactly its own lifetime: while production runs a
# lane's `flock -n ~/dsv41-offload/locks/gpu-7744b44b.lock` fails, and the locks come free when the worker exits (stop,
# crash, Conflicts= with the one-card unit). When any LOCK is held by another process (a lane, a pair window) it refuses:
# one line on stderr (the worker log), exit 75, CMD never runs, nothing stays locked; the unit's Restart=on-failure tries
# again every RestartSec, so production never shares a card and says so every 10 s until the card is free.
# Used by dsv41-dspark-worker-tp2-prod.service; tested by offload/launch/tests/test_prod_gpu_hold.sh. At most 7 LOCKs
# (fds 3-9: dash redirects single-digit fds only).
set -eu
fd=3
held=
while [ "$#" -gt 0 ] && [ "$1" != -- ]; do
  [ "$fd" -le 9 ] || { echo "gpu_locked: more than 7 locks" >&2; exit 2; }
  eval "exec $fd>>\"\$1\""
  if ! flock -n "$fd"; then
    echo "$(date -u +%FT%TZ) gpu_locked: $1 is held by another process (a lane or a pair window): refusing to start" >&2
    exit 75
  fi
  held="$held $1"
  fd=$((fd + 1))
  shift
done
[ "$#" -ge 2 ] && [ -n "$held" ] || { echo "usage: gpu_locked.sh LOCK... -- CMD [ARG...]" >&2; exit 2; }
shift
echo "$(date -u +%FT%TZ) gpu_locked: pid $$ holds$held" >&2
exec "$@"
