#!/bin/bash
# Start the TP = 2 DSpark worker on x86host (run ON x86host). Installs the unit if needed; never enables it.
# Refuses while anything runs on either card of the pair (the one-card unit, a vision service, a benchmark), while the
# one-card unit is active (they share the port and GPU-c4e7a371), or without the pinned transport / direct link.
set -euo pipefail
UUIDS="GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93 GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc"
UNIT=dsv41-dspark-worker-tp2.service
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p ~/.config/systemd/user ~/dsv41-offload/e/logs ~/dsv41-offload/e/run
if ! cmp -s "$here/$UNIT" ~/.config/systemd/user/$UNIT; then
  install -m 644 "$here/$UNIT" ~/.config/systemd/user/$UNIT
  systemctl --user daemon-reload
fi
if systemctl --user is-active --quiet $UNIT; then echo "$UNIT already active"; exit 0; fi
if systemctl --user is-active --quiet dsv41-dspark-worker.service; then
  echo "refusing: the one-card worker dsv41-dspark-worker.service is active (stop it first: run/stop.sh)"; exit 1; fi
for u in $UUIDS; do
  others=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | grep "$u" || true)
  if [ -n "$others" ]; then echo "refusing: GPU $u is in use:"; echo "$others"; exit 1; fi
done
[ -f ~/dsv41-offload/e/lib/libofl.so ] && [ -f ~/dsv41-offload/e/lib/ofl.py ] || {
  echo "refusing: transport not pinned (OFL_PIN_DST=\$HOME/dsv41-offload/e/lib $here/pin_ofl.sh)"; exit 1; }
ip -4 addr show | grep -q "inet 10.202.28.1/" || { echo "refusing: 10.202.28.1 (direct link to Spark A) is not configured"; exit 1; }
READY_BEFORE=$(grep -c "worker ready in" ~/dsv41-offload/e/logs/worker.log 2>/dev/null || true)
systemctl --user start $UNIT
echo "started; waiting for the ready line (two-card load + CUDA graphs, ~30-90 s)"
ready=0
for i in $(seq 1 240); do
  if grep -q "worker ready in" ~/dsv41-offload/e/logs/worker.log 2>/dev/null && \
     [ "$(grep -c 'worker ready in' ~/dsv41-offload/e/logs/worker.log)" -gt "${READY_BEFORE:-0}" ]; then ready=1; break; fi
  systemctl --user is-active --quiet $UNIT || { echo "unit died:"; tail -30 ~/dsv41-offload/e/logs/worker.log; exit 1; }
  sleep 1
done
tail -3 ~/dsv41-offload/e/logs/worker.log
[ "$ready" = 1 ] || { echo "no ready line after 240 s (the unit is still running: check status_tp2.sh / the log)"; exit 1; }
