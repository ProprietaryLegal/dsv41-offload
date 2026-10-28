#!/usr/bin/env bash
# Coordinator-only PP4-PROMOTE; dry by default, the aggregate HOLD estimate refuses before any execution.
set -euo pipefail
source "$HOME/.config/ndm/paths.sh"
cd "$(dirname "$0")/../../.."
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/offload/dspark"
exec /home/operator/miniconda3/bin/python3 -m offload.launch.windows.pp4_promote_window "$@"
