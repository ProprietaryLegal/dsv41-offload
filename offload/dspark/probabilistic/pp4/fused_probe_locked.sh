#!/usr/bin/env bash
# Start only in an owned PP4 named tmux with TMUX_TMPDIR set before tmux starts.
set -euo pipefail
source ~/.config/ndm/paths.sh
: "${PP4_SOURCE_ROOT:?set the staged PP4 source root}"
PP4_KERNEL_ROOT=/opt/dsv41/x86host-data/pp4/kernel
export CUDA_VISIBLE_DEVICES=GPU-bc62df2a-59cf-9828-7d43-558d1355bab7
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MAX_JOBS=8
export TMPDIR="$PP4_KERNEL_ROOT/tmp" TMUX_TMPDIR="$PP4_KERNEL_ROOT/tmux"
export TRITON_CACHE_DIR="$PP4_KERNEL_ROOT/triton" TORCH_EXTENSIONS_DIR="$PP4_KERNEL_ROOT/extensions"
export CUDA_CACHE_PATH="$PP4_KERNEL_ROOT/cuda-cache" XDG_CACHE_HOME="$PP4_KERNEL_ROOT/xdg-cache"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PP4_SOURCE_ROOT:$PP4_SOURCE_ROOT/offload/dspark"
mkdir -p "$TMPDIR" "$TMUX_TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"
exec 9>"$HOME/dsv41-offload/locks/gpu-bc62df2a.lock"
flock -w 3600 9
export PP4_LOCK_FD=9
[[ $(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=power.limit --format=csv,noheader,nounits) == 225.00 ]]
PP4_KERNEL_USED=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits)
if (( PP4_KERNEL_USED > 100 )); then echo "PP4 solo occupied ($PP4_KERNEL_USED MiB)" >&2; exit 3; fi
exec taskset -c 12-15,28-31 nice -n 19 /home/operator/miniconda3/bin/python3 \
  -m offload.dspark.probabilistic.pp4.fused_probe "$@"
