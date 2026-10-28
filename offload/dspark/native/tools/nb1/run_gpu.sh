#!/usr/bin/env bash
# NB1 GPU job, run ON X86HOST inside a detached `tmux -L nb1` session:
#   run_gpu.sh LOG PROBE.py [probe args...]
# Solo GPU-bc62df2a by UUID under its own flock -n, unchanged 225 W cap, nice 19, pinned to CPUs 12-15,28-31
# (production's Engram server and drafter worker are unpinned; this keeps NB1 on 8 logical CPUs), a 48 GiB memory
# scope, scratch-only writes (TMPDIR and Triton cache under the NB1 scratch). Only the log is written outside it.
set -euo pipefail
log=${1:?log path}; probe=${2:?probe script}; shift 2
scratch=/opt/dsv41/x86host-data/nb1
uuid=GPU-bc62df2a-59cf-9828-7d43-558d1355bab7
mkdir -p "$scratch/tmp" "$scratch/triton" "$(dirname "$log")"
exec > "$log" 2>&1
exec 9> "$HOME/dsv41-offload/locks/gpu-bc62df2a.lock"
flock -n 9 || { echo 'assigned GPU lock busy'; exit 73; }
export CUDA_VISIBLE_DEVICES=$uuid
export KX_FP8DENSE_LOCK_FD=9 KX_FP8DENSE_LOCKED=1 KX_FUSEDMOE_LOCKED=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_CACHE_DISABLE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_CUDA_ARCH_LIST=8.6 TRITON_CACHE_DIR=$scratch/triton TMPDIR=$scratch/tmp
unset TMUX
free -g
nvidia-smi -L
[[ $(nvidia-smi -i "$uuid" --query-gpu=power.limit --format=csv,noheader,nounits) == 225.00 ]] || { echo 'cap changed'; exit 74; }
[[ -z $(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader) ]] || { echo 'card busy'; exit 75; }
echo "acquired-$(date -u +%FT%TZ) env DSPARK_NATIVE_BATCH_INVARIANT=${DSPARK_NATIVE_BATCH_INVARIANT-<unset>}"
set +e
systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0 \
  timeout --signal=TERM --kill-after=10s "${NB1_TIMEOUT:-5400}s" \
  nice -n 19 taskset -c 12-15,28-31 "$HOME/dsv41-offload/e/venv/bin/python" -u "$probe" "$@"
rc=$?
set -e
echo "released-$(date -u +%FT%TZ)-rc=$rc"
nvidia-smi -i "$uuid" --query-gpu=uuid,power.limit,memory.used --format=csv,noheader
echo "$rc" > "$log.exit"
exit "$rc"
