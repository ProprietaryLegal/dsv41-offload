#!/usr/bin/env bash
# Run only from PP4's named detached tmux on x86host after EXL3 qualification.
set -euo pipefail
source ~/.config/ndm/paths.sh
umask 077
PP4_ACCEPTANCE_ROOT=/opt/dsv41/x86host-data/pp4/acceptance
: "${PP4_SOURCE_ROOT:?set the staged PP4 source root}"
# A qualified scratch-only CPython3.13 extension path may be provided. Never
# substitute an interpreter or add CPython3.12 site-packages.
: "${PP4_EXL3_PATH:?set the qualified CPython3.13 EXL3 module directory}"
export CUDA_VISIBLE_DEVICES=GPU-bc62df2a-59cf-9828-7d43-558d1355bab7
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1 MAX_JOBS=8
export TMPDIR="$PP4_ACCEPTANCE_ROOT/tmp" TMUX_TMPDIR="$PP4_ACCEPTANCE_ROOT/tmux"
export TRITON_CACHE_DIR="$PP4_ACCEPTANCE_ROOT/triton" TORCH_EXTENSIONS_DIR="$PP4_ACCEPTANCE_ROOT/torch_extensions"
export CUDA_CACHE_PATH="$PP4_ACCEPTANCE_ROOT/cuda-cache" XDG_CACHE_HOME="$PP4_ACCEPTANCE_ROOT/cache"
export PYTHONPATH="$PP4_SOURCE_ROOT:$PP4_SOURCE_ROOT/offload/dspark:$PP4_EXL3_PATH"
mkdir -p "$TMPDIR" "$TMUX_TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"
exec 9>"$HOME/dsv41-offload/locks/gpu-bc62df2a.lock"
flock -w 3600 9
export KX_FP8DENSE_LOCK_FD=9
test "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=power.limit --format=csv,noheader,nounits)" = "225.00"
PP4_ACCEPTANCE_USED=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits)
if (( PP4_ACCEPTANCE_USED > 100 )); then
  echo "Solo GPU occupied ($PP4_ACCEPTANCE_USED MiB); refusing" >&2
  exit 3
fi
exec taskset -c 12-15,28-31 nice -n 19 /home/operator/miniconda3/bin/python3 \
  -m offload.dspark.probabilistic.pp4.acceptance_replay \
  --production-root "$HOME/dsv41-offload/prod/dspark-worker/fb68c52" \
  --manifest /opt/dsv41/x86host-data/d2/decode-d2pass2-r1-c2/shards/MANIFEST.json \
  --weights /opt/dsv41/x86host-data/dspark \
  --frozen-dir "$PP4_ACCEPTANCE_ROOT/frozen" "$@"
