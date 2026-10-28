#!/usr/bin/env bash
# Offline by default. --execute is reserved for Deepseek/coordinator in an open window.
set -euo pipefail
source "$HOME/.config/ndm/paths.sh"
: "${HARNESS_PYTHON:?}"
cd "$(dirname "$0")/../../.."
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
exec "$HARNESS_PYTHON" -m offload.dspark.d2.decode_smoke "$@"
