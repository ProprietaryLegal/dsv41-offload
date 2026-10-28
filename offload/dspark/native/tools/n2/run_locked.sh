#!/usr/bin/env bash
# Invoke from detached tmux; one bounded acquisition per process.
set -euo pipefail
source ~/.config/ndm/paths.sh
root=${N2_ROOT:-$HOME/dsv41-offload/n2}
name=$1
shift
mkdir -p "$root/logs" "$root/results"
exec >> "$root/logs/$name.log" 2>&1
cd "$root/repo"
exec 9> "$HOME/dsv41-offload/locks/gpu-7744b44b.lock"
echo "$(date -u +%FT%TZ) waiting $name"
flock -w 3600 9
export CUDA_VISIBLE_DEVICES=GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc
export TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2 OMP_NUM_THREADS=8
export KX_FP8DENSE_LOCKED=1 KX_FP8DENSE_LOCK_FD=9
export KX_FUSEDMOE_LOCKED=1
export PYTHONPATH="$root/repo${PYTHONPATH:+:$PYTHONPATH}"
# Shared compiled caches are NOT rewritten; N2 owns its build artifacts.
export TORCH_EXTENSIONS_DIR="$root/build" TRITON_CACHE_DIR="$root/triton"
query() { nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,uuid,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,memory.used --format=csv,noheader; }
echo "$(date -u +%FT%TZ) acquired $name"
query
set +e
timeout --signal=TERM --kill-after=5s 830s "$HOME/dsv41-offload/e/venv/bin/python" -u "$@"
rc=$?
set -e
query
echo "$(date -u +%FT%TZ) rc=$rc $name"
exit "$rc"
