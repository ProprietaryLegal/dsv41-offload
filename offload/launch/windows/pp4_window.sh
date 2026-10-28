#!/usr/bin/env bash
# Dry by default. Only the attended coordinator may execute after all hash-bound admission gates pass.
set -euo pipefail
source "$HOME/.config/ndm/paths.sh"
: "${HARNESS_PYTHON:?}"
cd "$(dirname "$0")/../../.."
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
exec "$HARNESS_PYTHON" -m offload.launch.windows.pp4_window "$@"
