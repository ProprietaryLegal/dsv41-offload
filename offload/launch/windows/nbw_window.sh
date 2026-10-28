#!/usr/bin/env bash
set -euo pipefail
source "$HOME/.config/ndm/paths.sh"
cd "$(dirname "$0")/../../.."
exec /home/operator/miniconda3/bin/python3 -m offload.launch.windows.nbw_window "$@"
