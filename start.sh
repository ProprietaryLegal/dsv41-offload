#!/usr/bin/env bash
# Start the TP2 serve (both ranks). Safe: refuses if model/image missing or a GPU workload already runs.
set -euo pipefail; cd "$(dirname "$0")"; set -a; . ./dsv41.env; set +a
for n in $HEAD $WORKER; do
  ssh -o BatchMode=yes $n "test -f $MODEL_DIR/model-00048-of-00048.safetensors && test -f $MODEL_DIR/config.json" || { echo "model incomplete on $n"; exit 1; }
  ssh $n "docker image inspect $IMAGE >/dev/null" || { echo "image missing on $n"; exit 1; }
  busy=$(ssh $n "nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader" | grep -v '^$' || true)
  [ -z "$busy" ] || [ "${FORCE:-}" = 1 ] || { echo "GPU busy on $n: $busy (FORCE=1 to override)"; exit 1; }
done
if [ -n "${GPU_CLOCK_CAP:-}" ]; then for n in $HEAD $WORKER; do ssh $n "sudo -n nvidia-smi -lgc 0,$GPU_CLOCK_CAP >/dev/null && echo $n clock cap $GPU_CLOCK_CAP MHz"; done; fi
./firewall.sh apply || echo "WARN: firewall rule not applied"
mkdir -p logs; ts=$(date +%Y%m%dT%H%M%S)
HEAD=$HEAD WORKER=$WORKER RECIPE=$RECIPE MODEL_DIR=$MODEL_DIR bash ./serve.sh 2>&1 | tee logs/start-$ts.log
