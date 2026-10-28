#!/usr/bin/env bash
# Coordinator-only pair workflow. This implementation lane does not execute it.
set -euo pipefail
source ~/.config/ndm/paths.sh
root=${N2_ROOT:-$HOME/dsv41-offload/n2}
marker=$HOME/dsv41-offload/PAIR_OK
marker_seen=0
marker_value=
if [[ -e $marker ]]; then marker_seen=1; marker_value=$(cat "$marker"); fi
allowed() { [[ ${N2_PAIR_BY_COORD:-0} == 1 ]] || grep -q '^N2\([[:space:]-]\|$\)' "$marker" 2>/dev/null; }
window_valid() {
  allowed || return 1
  [[ $marker_seen == 0 ]] || { [[ -e $marker ]] && [[ $(cat "$marker") == "$marker_value" ]]; }
}
allowed || { echo 'PAIR_OK must name N2 or coordinator must set N2_PAIR_BY_COORD=1' >&2; exit 1; }
mkdir -p "$root/logs" "$root/results"
cd "$root/repo"
run() (
  name=$1
  shift
  exec >> "$root/logs/pair_$name.log" 2>&1
  exec 8> "$HOME/dsv41-offload/locks/gpu-c4e7a371.lock"
  exec 9> "$HOME/dsv41-offload/locks/gpu-7744b44b.lock"
  acquired=0
  for ((attempt=0; attempt<600; attempt++)); do
    if flock -n 8; then
      if flock -n 9; then acquired=1; break; fi
      flock -u 8
    fi
    sleep 1
  done
  [[ $acquired == 1 ]] || { echo 'Pair locks unavailable after 600s' >&2; exit 1; }
  window_valid || { echo 'Pair authorization withdrawn while waiting' >&2; exit 1; }
  export CUDA_VISIBLE_DEVICES=GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93,GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc
  export TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2 OMP_NUM_THREADS=8
  export TORCH_EXTENSIONS_DIR="$root/build" TRITON_CACHE_DIR="$root/triton" PYTHONPATH="$root/repo"
  echo "$(date -u +%FT%TZ) acquired $name"
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu --format=csv,noheader
  timeout --signal=TERM --kill-after=5s 830s "$HOME/dsv41-offload/e/venv/bin/python" -u "$@" &
  child=$!
  trap 'kill -TERM "$child" 2>/dev/null || true' EXIT
  while kill -0 "$child" 2>/dev/null; do
    if ! window_valid; then
      echo 'PAIR_OK withdrawn: terminating owned process' >&2
      kill -TERM "$child"
      break
    fi
    sleep 1
  done
  set +e
  wait "$child"
  rc=$?
  set -e
  trap - EXIT
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu --format=csv,noheader
  echo "$(date -u +%FT%TZ) released $name rc=$rc"
  exit "$rc"
)
for arm in ${N2_PAIR_ARMS:-k2-tp2 native-hybrid-kx native-hybrid-kx2}; do
  run "bench-$arm" -m offload.dspark.native.tools.native_bench "$arm" pair "$root/results/bench_${arm}_pair.json" 0 200
  run "parity-$arm" -m offload.dspark.native.tools.native_parity "$arm" pair "$root/results/parity_${arm}_pair.json"
done
