#!/usr/bin/env bash
# Start the TP2 serve (both ranks). Safe: refuses if model/image missing or a GPU workload already runs.
set -euo pipefail; cd "$(dirname "$0")"; set -a; . ./dsv41.env; set +a
# K1 startup is guarded, including staging failures and the original launcher.
if [ "${PROD_K1:-0}" = 1 ] && [ "${K1_START_INNER:-0}" != 1 ]; then
  exec bash offload/promote/k1/start_checked.sh
fi
for n in $HEAD $WORKER; do
  ssh -o BatchMode=yes $n "test -f $MODEL_DIR/model-00048-of-00048.safetensors && test -f $MODEL_DIR/config.json" || { echo "model incomplete on $n"; exit 1; }
  ssh $n "docker image inspect $IMAGE >/dev/null" || { echo "image missing on $n"; exit 1; }
  busy=$(ssh $n "nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader" | grep -v '^$' || true)
  [ -z "$busy" ] || [ "${FORCE:-}" = 1 ] || { echo "GPU busy on $n: $busy (FORCE=1 to override)"; exit 1; }
done
if [ -n "${GPU_CLOCK_CAP:-}" ]; then for n in $HEAD $WORKER; do ssh $n "sudo -n nvidia-smi -lgc 0,$GPU_CLOCK_CAP >/dev/null && echo $n clock cap $GPU_CLOCK_CAP MHz"; done; fi
./firewall.sh apply || echo "WARN: firewall rule not applied"
# DSV41-OFFLOAD B: stage the remote-Engram overlay from this checkout and build/load its JIT extension before the model
# loads (no-op unless dsv41.env sets OFFLOAD_DOCKER_ENV with DSV41_ENGRAM_OVERLAY)
if [[ "${OFFLOAD_DOCKER_ENV:-}" == *DSV41_ENGRAM_OVERLAY* ]]; then
  offload/engram/spark/stage_overlay.sh $HEAD $WORKER || { echo "remote-Engram overlay staging failed" >&2; exit 1; }
  offload/engram/spark/prebuild_jit.sh $HEAD $WORKER || exit 1
fi
# DSV41-OFFLOAD D: stage the remote-DSpark overlay + ofl transport into ~/dsv41-offload-prod/dspark on both nodes
# (no-op unless dsv41.env sets OFFLOAD_DSPARK_ENV); the x86host worker must be up (systemd dsv41-dspark-worker), else
# capacity mode serves target-only rounds until it is.
if [ -n "${OFFLOAD_DSPARK_ENV:-}" ]; then
  offload/dspark/spark/stage_overlay.sh $HEAD $WORKER || { echo "remote-DSpark overlay staging failed" >&2; exit 1; }
fi
mkdir -p logs; ts=$(date +%Y%m%dT%H%M%S)
HEAD=$HEAD WORKER=$WORKER RECIPE=$RECIPE MODEL_DIR=$MODEL_DIR bash ./serve.sh 2>&1 | tee logs/start-$ts.log
if [ "${PROD_K1:-0}" = 1 ]; then
  bash offload/promote/k1/verify.sh
fi
# DSV41-OFFLOAD D-async (R3-PROMOTE): one warm-up request takes the first request's lazy module loads
# (CUDA_MODULE_LOADING stays LAZY; EAGER costs ~18 min per start). The serve is up either way.
if [[ "${OFFLOAD_DSPARK_ENV:-}" == *VLLM_DSPARK_REMOTE_ASYNC=1* ]]; then
  offload/launch/warmup_serve.sh "$PORT" "$SERVED" 2>&1 | tee -a logs/start-$ts.log \
    || echo "WARN: the warm-up request failed (the first real request pays the module loads)" | tee -a logs/start-$ts.log
fi
