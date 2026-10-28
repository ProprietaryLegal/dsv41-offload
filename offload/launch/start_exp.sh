#!/usr/bin/env bash
# DSV41-OFFLOAD: start one experiment configuration in an open window (run on sparkhead from ~/dsv41-deploy).
#   offload/launch/start_exp.sh offload/launch/configs/<NAME>.env
# The env file may set: LABEL, PORT (default 8001), SPEC_OVERRIDE, KV_BYTES_OVERRIDE, MAX_LEN_OVERRIDE, ENGRAM, EXP_DOCKER_ENV,
# EXTRA_DOCKER_ARGS, MODEL_DIR (must stay /srv/dsv41/legal-v2a for every comparison), and OVERLAYS: space-separated
# `repo/relative/path=/container/path` pairs. Each overlay is copied to every node's ~/dsv41-offload-run/overlay/ (node-local
# disk: sparkworker's NFS view of this home is not readable by the docker daemon) and bind-mounted read-only at its container
# path. Production returns with ./start.sh.
set -euo pipefail; cd "$(dirname "$0")/../.."
cfg=${1:?config env file}; [ -f "$cfg" ] || { echo "no $cfg" >&2; exit 1; }
[ -f "$HOME/dsv41-deploy/HOLD" ] || { echo "open a window first: offload/launch/window.sh open '<purpose>'" >&2; exit 1; }
set -a; . ./dsv41.env; PORT=8001; . "$cfg"; set +a
# Fail before any SSH, stop or staging if a skipping worker lacks UP2's scheduler.
for up2_replay_env in ${EXP_DOCKER_ENV:-}; do
  if [ "$up2_replay_env" = VLLM_DSV41_REPLAY_SKIP=1 ]; then
    python3 -m offload.upstream_adopt.replay_guard
    break
  fi
done
SERVE_DIR=dsv41-offload-run   # per-node home dir, outside the production checkout
[ "$MODEL_DIR" = /srv/dsv41/legal-v2a ] || { echo "MODEL_DIR must stay /srv/dsv41/legal-v2a (got $MODEL_DIR)" >&2; exit 1; }
for n in $HEAD $WORKER; do
  ssh -o BatchMode=yes $n "test -f $MODEL_DIR/model-00048-of-00048.safetensors && test -f $MODEL_DIR/config.json" || { echo "model incomplete on $n"; exit 1; }
  ssh $n "docker image inspect $IMAGE >/dev/null" || { echo "image missing on $n"; exit 1; }
done
./stop.sh
for n in $HEAD $WORKER; do
  busy=$(ssh $n "nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader" | grep -v '^$' || true)
  [ -z "$busy" ] || { echo "GPU busy on $n after stop: $busy" >&2; exit 1; }
done
if [ -n "${GPU_CLOCK_CAP:-}" ]; then for n in $HEAD $WORKER; do ssh $n "sudo -n nvidia-smi -lgc 0,$GPU_CLOCK_CAP >/dev/null && echo $n clock cap $GPU_CLOCK_CAP MHz"; done; fi
# firewall.sh re-reads PORT=8000 from dsv41.env: it (re)builds the DSV41-API chain; the experiment port joins that chain
./firewall.sh apply >/dev/null || { echo "firewall chain could not be applied" >&2; exit 1; }
sudo -n iptables -C INPUT -p tcp --dport "$PORT" -j DSV41-API 2>/dev/null || sudo -n iptables -I INPUT 1 -p tcp --dport "$PORT" -j DSV41-API
echo "port $PORT restricted by chain DSV41-API"
ovl_args=""
if [ -n "${OVERLAYS:-}" ]; then
  for n in $HEAD $WORKER; do ssh -o BatchMode=yes $n "rm -rf dsv41-offload-run/overlay && mkdir -p dsv41-offload-run/overlay"; done
  i=0
  for pair in $OVERLAYS; do
    src=${pair%%=*}; dst=${pair#*=}
    [ -e "$src" ] || { echo "overlay source $src missing" >&2; exit 1; }
    name=$i-$(basename "$src"); i=$((i + 1))
    for n in $HEAD $WORKER; do scp -rq -o BatchMode=yes "$src" "$n:dsv41-offload-run/overlay/$name"; done
    ovl_args="$ovl_args -v \$HOME/dsv41-offload-run/overlay/$name:$dst:ro"
  done
fi
export EXTRA_DOCKER_ARGS="${EXTRA_DOCKER_ARGS:-}$ovl_args"
# TV1 alone qualifies its installed kernels while no model/server is resident.
# Unset is the historical path; every nested command remains in the start scope.
if [ "${TV1_ROLE-}" = candidate ]; then
  python3 -m offload.tv1.oracle_prestart
fi
echo "experiment ${LABEL:-exp}: port $PORT, docker args:${EXTRA_DOCKER_ARGS:- none}, extra env: ${EXP_DOCKER_ENV:-none}"
mkdir -p logs; ts=$(date +%Y%m%dT%H%M%S)
cp "$cfg" "logs/exp-$ts-${LABEL:-exp}.env"
HEAD=$HEAD WORKER=$WORKER RECIPE=$RECIPE MODEL_DIR=$MODEL_DIR PORT=$PORT SERVE_DIR=$SERVE_DIR \
  bash offload/launch/serve_exp.sh 2>&1 | tee "logs/exp-start-$ts-${LABEL:-exp}.log"
