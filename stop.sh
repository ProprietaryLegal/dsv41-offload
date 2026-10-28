#!/usr/bin/env bash
# Stop only this deployment's containers (name dsv41-serve) on both nodes.
set -euo pipefail; cd "$(dirname "$0")"; set -a; . ./dsv41.env; set +a
HEAD=$HEAD WORKER=$WORKER bash "$RECIPE/scripts/launch/dsv41_tp2.sh" down
