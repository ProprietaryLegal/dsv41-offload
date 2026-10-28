#!/usr/bin/env bash
# Launch from a PP4-owned detached tmux session on x86host. No production control.
set -euo pipefail
source ~/.config/ndm/paths.sh
umask 077
PP4_RUNTIME_ROOT=/opt/dsv41/x86host-data/pp4/runtime
: "${PP4_SOURCE_ROOT:?set the staged PP4 source root}"
: "${PP4_EXL3_PATH:?set qualified CPython3.13 stock/facc extension directories}"
export CUDA_VISIBLE_DEVICES=GPU-bc62df2a-59cf-9828-7d43-558d1355bab7
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MAX_JOBS=8
export PYTHONDONTWRITEBYTECODE=1 DSPARK_SAMPLED_TOP32=1
export TMPDIR="$PP4_RUNTIME_ROOT/tmp" TMUX_TMPDIR="$PP4_RUNTIME_ROOT/tmux"
export TRITON_CACHE_DIR="$PP4_RUNTIME_ROOT/triton" TORCH_EXTENSIONS_DIR="$PP4_RUNTIME_ROOT/torch_extensions"
export CUDA_CACHE_PATH="$PP4_RUNTIME_ROOT/cuda_cache"
export CUDA_HOME="$PP4_RUNTIME_ROOT/cuda13/nvidia/cu13"
export PYTHONPATH="$PP4_SOURCE_ROOT:$PP4_SOURCE_ROOT/offload/dspark:$PP4_EXL3_PATH"
mkdir -p "$TMPDIR" "$TMUX_TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH"
exec 9>"$HOME/dsv41-offload/locks/gpu-bc62df2a.lock"
flock -w 3600 9
export KX_FP8DENSE_LOCK_FD=9
if [[ $(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=power.limit --format=csv,noheader,nounits) != 225.00 ]]; then
  echo 'The solo GPU must already be at225W; no power change is authorized.' >&2
  exit 3
fi
PP4_RUNTIME_USED=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits)
if (( PP4_RUNTIME_USED > 100 )); then
  echo "Solo GPU occupied ($PP4_RUNTIME_USED MiB); refusing" >&2
  exit 3
fi
exec taskset -c 12-15,28-31 nice -n 19 /home/operator/miniconda3/bin/python3 \
  -m offload.dspark.probabilistic.pp4.full_worker "$@"
