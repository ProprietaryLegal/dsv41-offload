#!/usr/bin/env bash
# Execute on x86host. Each acquisition ends within 14 minutes including compilation.
set -euo pipefail
source ~/.config/ndm/paths.sh
[[ $(hostname) == x86host ]] || { echo 'must run on x86host' >&2; exit 2; }
mkdir -p ~/dsv41-offload/locks
exec 9>~/dsv41-offload/locks/gpu-7744b44b.lock
flock -w 1800 -E 75 9
export KX_GROUPED_LOCK_FD=9
export CUDA_VISIBLE_DEVICES=GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc
exec timeout --signal=TERM --kill-after=10 830 "$@"
