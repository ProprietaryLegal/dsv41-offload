#!/usr/bin/env bash
set -euo pipefail
source ~/.config/ndm/paths.sh
: "${KX_FP8DENSE_ROOT:?set to lane checkout containing offload/}"
exec 9>"$HOME/dsv41-offload/locks/gpu-7744b44b.lock"
flock -w 1800 9
export CUDA_VISIBLE_DEVICES=GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc
export KX_FP8DENSE_LOCKED=1
export OMP_NUM_THREADS=8 MAX_JOBS=2
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6
export TORCH_EXTENSIONS_DIR="$HOME/dsv41-offload/kx-fp8dense/extensions"
cd "$KX_FP8DENSE_ROOT"
exec timeout --signal=TERM --kill-after=5 830 "$HOME/dsv41-offload/e/venv/bin/python" "$@"
