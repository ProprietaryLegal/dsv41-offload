#!/usr/bin/env bash
# W9 entry point. Default is offline dry-run. Execution is coordinator-only.
set -euo pipefail
source "$HOME/.config/ndm/paths.sh"
repo=${D2_REPO:-/opt/dsv41/repos/.worktrees/dsv41-d2-replay}
cd "$repo"
exec "$HARNESS_PYTHON" -m offload.dspark.d2.replay_window "$@"
