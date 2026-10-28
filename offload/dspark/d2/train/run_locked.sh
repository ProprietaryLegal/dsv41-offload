#!/usr/bin/env bash
# Start from detached tmux. One acquisition; always release between jobs.
set -euo pipefail
source ~/.config/ndm/paths.sh
task_root=${D2_TRAIN_ROOT:?Set D2_TRAIN_ROOT to the immutable X86host source snapshot}
mkdir -p "$task_root/logs"
name=$1
shift
exec >> "$task_root/logs/$name.log" 2>&1
cd "$task_root/repo"
exec 9> "$HOME/dsv41-offload/locks/gpu-240876ad.lock"
flock -w 3600 9
export D2_LOCK_FD=9 CUDA_VISIBLE_DEVICES=GPU-240876ad-ea42-47de-1b8d-f6153810b7e1
export OMP_NUM_THREADS=8 PYTHONPATH="$task_root/repo${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="$task_root/triton" TORCH_EXTENSIONS_DIR="$task_root/build" TORCHINDUCTOR_CACHE_DIR="$task_root/inductor"
query() { nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,uuid,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,memory.used --format=csv,noheader; }
query
task_python=${D2_PYTHON:-$HOME/dsv41-offload/e/venv/bin/python}
set +e
"$task_python" -u -m offload.dspark.d2.train.lease -- "$task_python" -u "$@"
rc=$?
set -e
query
exit "$rc"
