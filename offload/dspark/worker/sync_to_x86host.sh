#!/bin/bash
# Copy this worktree's offload/ tree to x86host (code only). Default destination: lane O2's tree, from which the
# one-card worker runs (~/dsv41-offload/o2/repo/offload/dspark); lane E syncs to its own tree with
# DSPARK_SYNC_DEST=dsv41-offload/e/repo/offload/ (the TP = 2 unit runs from there).
set -euo pipefail
here=$(cd "$(dirname "$0")/../../.." && pwd)
dest=${DSPARK_SYNC_DEST:-dsv41-offload/o2/repo/offload/}
rsync -a --delete --exclude __pycache__ --exclude '*.pyc' "$here/offload/" "x86host:$dest"
