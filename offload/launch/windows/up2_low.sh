#!/usr/bin/env bash
# Coordinator only. Default is a network-free plan. No window/restore operations.
set -euo pipefail
source "$HOME/.config/ndm/paths.sh"
: "${HARNESS_PYTHON:?}"
cd "$(dirname "$0")/../../.."
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
exec "$HARNESS_PYTHON" -m offload.launch.windows.up2_low "$@"
