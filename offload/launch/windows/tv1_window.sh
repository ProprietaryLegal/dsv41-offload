#!/usr/bin/env bash
set -euo pipefail
source ~/.config/ndm/paths.sh
: "${HARNESS_PYTHON:?}"
cd "$(dirname "$0")/../../.."
exec "$HARNESS_PYTHON" -m offload.launch.windows.tv1_window "$@"
