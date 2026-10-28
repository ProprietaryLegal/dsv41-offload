#!/usr/bin/env bash
# Nonblocking solo lease. Run this wrapper in detached tmux on X86host.
set -euo pipefail
source ~/.config/ndm/paths.sh
task_root=${D2_TRAIN_ROOT:?immutable source snapshot required}
task_gpu=${D2_PILOT_GPU:?full permitted solo UUID required}
case "$task_gpu" in
  GPU-240876ad-ea42-47de-1b8d-f6153810b7e1) task_id=240876ad ;;
  GPU-bc62df2a-59cf-9828-7d43-558d1355bab7) task_id=bc62df2a ;;
  *) echo 'Refusing GPU: pilot permits only the two named solo cards' >&2; exit 2 ;;
esac
mkdir -p "$task_root/logs"
name=$1
shift
exec >> "$task_root/logs/$name.log" 2>&1
cd "$task_root/repo"
exec 9> "$HOME/dsv41-offload/locks/gpu-$task_id.lock"
flock -n 9
export D2_LOCK_FD=9 CUDA_VISIBLE_DEVICES="$task_gpu" D2_PILOT_LEASE=1
export OMP_NUM_THREADS=8 PYTHONPATH="$task_root/repo${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="$task_root/triton-$task_id" TORCH_EXTENSIONS_DIR="$task_root/build-$task_id"
task_limit=$(nvidia-smi --id="$task_gpu" --query-gpu=power.limit --format=csv,noheader,nounits)
[[ "$task_limit" == '225.00' ]] || { echo "Unexpected solo cap: $task_limit" >&2; exit 2; }
task_python=${D2_PYTHON:-$HOME/dsv41-offload/e/venv/bin/python}
exec "$task_python" -u -m offload.dspark.d2.train.lease -- "$task_python" -u "$@"
