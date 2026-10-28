#!/usr/bin/env bash
# Detached tmux caller; lock released after each bounded job.
set -euo pipefail
source ~/.config/ndm/paths.sh
root=${N4_ROOT:-$HOME/dsv41-offload/n4}
name=${1:?job name}; shift
mkdir -p "$root/logs" "$root/results"
exec >> "$root/logs/$name.log" 2>&1
tree=${N4_TREE:-repo}
[[ $tree == repo || $tree == before ]] || { echo "invalid N4_TREE" >&2; exit 1; }
cd "$root/$tree"
marker=$root/STOP
# PAIR_OK reserves the NVLink pair; this assigned solo card is outside it.
if [[ -e $marker ]]; then echo "N4 STOP present: solo step refused: $(cat "$marker")" >&2; exit 1; fi
ssh -o BatchMode=yes controller 'source ~/.config/ndm/paths.sh; tail -4 /opt/dsv41/campaigns/DSV41-OFFLOAD/LANES.md' > "$root/logs/$name.lanes.txt"
cat "$root/logs/$name.lanes.txt"
nvidia-smi -L
free -g
exec 9> "$HOME/dsv41-offload/locks/gpu-bc62df2a.lock"
echo "$(date -u +%FT%TZ) waiting $name"
flock -w 3600 9
[[ ! -e $marker ]] || { echo 'N4 STOP appeared while waiting; refusing' >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=GPU-bc62df2a-59cf-9828-7d43-558d1355bab7
export TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2 OMP_NUM_THREADS=8
export TORCH_EXTENSIONS_DIR="$root/build" TRITON_CACHE_DIR="$root/triton" PYTHONPATH="$root/$tree"
export KX_FP8DENSE_LOCK_FD=9 KX_FP8DENSE_LOCKED=1 KX_FUSEDMOE_LOCKED=1
export DSPARK_WORKER_TP_EMULATE=1 TZ=UTC
export DSPARK_SOURCE_COMMIT=$(cat "$root/$tree/SOURCE_COMMIT")
apps=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)
[[ -z $apps ]] || { echo 'Authorized card is occupied; refusing' >&2; exit 1; }
limit=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=power.limit --format=csv,noheader,nounits)
[[ $limit == 225.00 ]] || { echo "Expected coordinator 225W cap, found $limit; refusing" >&2; exit 1; }
echo "$(date -u +%FT%TZ) acquired $name"
# Use the exact selected GPU only; one-second sampled memory includes load/prewarm/serve.
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,uuid,memory.used,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu --format=csv,noheader -l 1 > "$root/results/$name.smi.csv" &
sampler=$!
start_token() { [[ ! -r /proc/$1/stat ]] || awk '{print $22}' "/proc/$1/stat"; }
sampler_start=$(start_token "$sampler")
child=
child_start=
signal_owned() {
  [[ -z $2 || $(start_token "$1") != "$2" ]] || kill -TERM "$1" 2>/dev/null || true
}
cleanup() {
  [[ -z $child ]] || { signal_owned "$child" "$child_start"; wait "$child" 2>/dev/null || true; }
  signal_owned "$sampler" "$sampler_start"
  wait "$sampler" 2>/dev/null || true
}
trap cleanup EXIT
set +e
timeout --signal=TERM --kill-after=5s 830s "$HOME/dsv41-offload/e/venv/bin/python" -u "$@" &
child=$!
child_start=$(start_token "$child")
monitor_failed=0
while [[ -n $child_start && $(start_token "$child") == "$child_start" ]] && kill -0 "$child" 2>/dev/null; do
  if [[ -e $marker ]]; then
    echo 'N4 STOP appeared: terminating owned solo process' >&2
    signal_owned "$child" "$child_start"
    break
  fi
  if [[ -z $sampler_start || $(start_token "$sampler") != "$sampler_start" ]]; then
    echo 'memory sampler exited unexpectedly; terminating owned job' >&2
    monitor_failed=1
    signal_owned "$child" "$child_start"
    break
  fi
  sleep 1
done
wait "$child"
rc=$?
[[ $monitor_failed == 0 ]] || rc=1
child=
set -e
echo "$(date -u +%FT%TZ) released $name rc=$rc"
exit "$rc"
