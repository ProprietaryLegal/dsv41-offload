#!/usr/bin/env bash
# Coordinator only. PF2-LOWKV window runner; dry plan unless --execute (offload/launch/windows/PF2-LOWKV.md).
set -euo pipefail
source "$HOME/.config/ndm/paths.sh"
: "${HARNESS_PYTHON:?}"
cd "$(dirname "$0")/../../.."
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
exec "$HARNESS_PYTHON" -m offload.launch.windows.pf2_window "$@"
