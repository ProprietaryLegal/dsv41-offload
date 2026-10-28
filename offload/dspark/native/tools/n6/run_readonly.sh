#!/usr/bin/env bash
# Launch THIS script in a detached Controller tmux session. Only logs write locally.
set -euo pipefail
source ~/.config/ndm/paths.sh
out=${1:?absolute Controller aggregate-log directory}
[[ $out == /* ]] || { echo 'absolute output directory required' >&2; exit 2; }
mkdir -p "$out"
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for arm in k2-tp2 native-hybrid-kx2; do
  set +e
  ssh -o BatchMode=yes x86host "source ~/.config/ndm/paths.sh
set -euo pipefail
exec 9> ~/dsv41-offload/locks/gpu-bc62df2a.lock
flock -n 9 || { echo 'assigned GPU lock busy' >&2; exit 73; }
export CUDA_VISIBLE_DEVICES=GPU-bc62df2a-59cf-9828-7d43-558d1355bab7
export KX_FP8DENSE_LOCK_FD=9 KX_FP8DENSE_LOCKED=1 KX_FUSEDMOE_LOCKED=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_CACHE_DISABLE=1 OMP_NUM_THREADS=4
export TORCH_CUDA_ARCH_LIST=8.6 TRITON_CACHE_DIR=\$HOME/dsv41-offload/n4/triton
export PYTHONPATH=\$HOME/dsv41-offload/n4/repo
cd ~/dsv41-offload/n4/repo
free -g
nvidia-smi -L
[[ \$(nvidia-smi -i \$CUDA_VISIBLE_DEVICES --query-gpu=power.limit --format=csv,noheader,nounits) == 225.00 ]]
[[ -z \$(nvidia-smi -i \$CUDA_VISIBLE_DEVICES --query-compute-apps=pid --format=csv,noheader) ]]
echo acquired-\$(date -u +%FT%TZ)
set +e
timeout --signal=TERM --kill-after=5s 830s ~/dsv41-offload/e/venv/bin/python -u - $arm
rc=\$?
set -e
echo released-\$(date -u +%FT%TZ)-rc=\$rc
nvidia-smi -i \$CUDA_VISIBLE_DEVICES --query-gpu=uuid,power.limit,memory.used --format=csv,noheader
exit \$rc" < "$here/batch_probe.py" > "$out/$arm.log" 2>&1
  rc=$?
  set -e
  echo "$rc" > "$out/$arm.exit"
  [[ $rc == 0 ]] || { echo "$arm failed: $out/$arm.log" >&2; exit "$rc"; }
done
