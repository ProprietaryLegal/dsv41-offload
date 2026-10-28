#!/usr/bin/env bash
set -euo pipefail
source ~/.config/ndm/paths.sh
PP3_ROOT=/opt/dsv41/x86host-data/pp3
cd "$PP3_ROOT"
export CUDA_VISIBLE_DEVICES=GPU-bc62df2a-59cf-9828-7d43-558d1355bab7
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MAX_JOBS=8
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PP3_ROOT/tmp" TRITON_CACHE_DIR="$PP3_ROOT/triton" TORCH_EXTENSIONS_DIR="$PP3_ROOT/extensions"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
exec 9>"$HOME/dsv41-offload/locks/gpu-bc62df2a.lock"
flock -n 9
export PP3_LOCK_FD=9
test "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=power.limit --format=csv,noheader,nounits)" = 225.00
PP3_USED=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits)
if (( PP3_USED > 100 )); then echo "PP3 solo occupied ($PP3_USED MiB)" >&2; exit 3; fi
PP3_MODULE=offload.dspark.worker.tools.pp3_probe
if [[ ${1:-} == --oracle ]]; then
  shift
  PP3_MODULE=offload.dspark.probabilistic.pp3.oracle_bridge
elif [[ ${1:-} == --qualify ]]; then
  shift
  PP3_MODULE=offload.dspark.probabilistic.pp3.gpu_qualify
fi
exec taskset -c 12-15,28-31 nice -n 19 /home/operator/miniconda3/bin/python3 \
  -m "$PP3_MODULE" "$@"
