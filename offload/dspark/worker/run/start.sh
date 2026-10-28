#!/bin/bash
# Start the DSpark worker on x86host (run ON x86host). Installs the unit if needed; never enables it.
# Refuses if another process holds the 3090 GPU-c4e7a371 (e.g. x86host-3090-vision-8114.service, pinned to the same card).
set -euo pipefail
UUID=GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93
UNIT=dsv41-dspark-worker.service
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p ~/.config/systemd/user ~/dsv41-offload/o2/logs ~/dsv41-offload/o2/run
if ! cmp -s "$here/$UNIT" ~/.config/systemd/user/$UNIT; then
  install -m 644 "$here/$UNIT" ~/.config/systemd/user/$UNIT
  systemctl --user daemon-reload
fi
if systemctl --user is-active --quiet $UNIT; then echo "$UNIT already active"; exit 0; fi
others=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | grep "$UUID" || true)
if [ -n "$others" ]; then echo "refusing: GPU $UUID is in use:"; echo "$others"; exit 1; fi
[ -f ~/dsv41-offload/o2/lib/libofl.so ] && [ -f ~/dsv41-offload/o2/lib/ofl.py ] || { echo "refusing: transport not pinned (run $here/pin_ofl.sh)"; exit 1; }
ip -4 addr show | grep -q "inet 10.202.28.1/" || { echo "refusing: 10.202.28.1 (direct link to Spark A) is not configured"; exit 1; }
READY_BEFORE=$(grep -c "worker ready in" ~/dsv41-offload/o2/logs/worker.log 2>/dev/null || true)
systemctl --user start $UNIT
echo "started; waiting for the ready line (model load + CUDA graphs, ~10-30 s)"
ready=0
for i in $(seq 1 120); do
  if grep -q "worker ready in" ~/dsv41-offload/o2/logs/worker.log 2>/dev/null && \
     [ "$(grep -c 'worker ready in' ~/dsv41-offload/o2/logs/worker.log)" -gt "${READY_BEFORE:-0}" ]; then ready=1; break; fi
  systemctl --user is-active --quiet $UNIT || { echo "unit died:"; tail -30 ~/dsv41-offload/o2/logs/worker.log; exit 1; }
  sleep 1
done
tail -3 ~/dsv41-offload/o2/logs/worker.log
[ "$ready" = 1 ] || { echo "no ready line after 120 s (the unit is still running: check status.sh / the log)"; exit 1; }
