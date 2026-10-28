#!/usr/bin/env bash
# W9 post-window entry point. Default is offline dry-run; only coordinator executes.
set -euo pipefail
source "$HOME/.config/ndm/paths.sh"
repo=${D2_REPO:-/opt/dsv41/repos/.worktrees/dsv41-d2-replay}
cd "$repo"
exec "$HARNESS_PYTHON" -m offload.dspark.d2.after_replay "$@"
