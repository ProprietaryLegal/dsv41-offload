#!/bin/bash
# Stop the DSpark worker (run ON x86host). The Spark proxy treats the closed connection as a failed round (CONTRACT 10).
set -euo pipefail
systemctl --user stop dsv41-dspark-worker.service || true
systemctl --user is-active dsv41-dspark-worker.service || true
