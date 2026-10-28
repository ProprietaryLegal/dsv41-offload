#!/usr/bin/env bash
# Coordinator only. Read-only PF1-WINDOW precheck (sparkhead, sparkworker-wired, x86host status file, Controller); writes nothing.
set -euo pipefail
source "$HOME/.config/ndm/paths.sh"
: "${HARNESS_PYTHON:?}"
cd "$(dirname "$0")/../../.."
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
exec "$HARNESS_PYTHON" -m offload.launch.windows.pf1_precheck "$@"
