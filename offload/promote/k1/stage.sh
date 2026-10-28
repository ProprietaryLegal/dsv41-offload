#!/usr/bin/env bash
# Coordinator-only, on the head node. Never called by status/stop/env sourcing.
set -euo pipefail
cd "$(dirname "$0")/../../.."
. offload/promote/k1/constants.sh
[ "$#" = 2 ] && [ "$1" != "$2" ] || { echo 'K1 staging requires two distinct nodes' >&2; exit 1; }
for key in ${!KVPACK_TEST_@}; do
  echo "K1 production staging refuses test hook $key" >&2; exit 1
done
[ "$(sha256sum offload/kvpack/MANIFEST.json | cut -d' ' -f1)" = "$K1_MANIFEST_SHA" ] \
  || { echo 'K1 manifest is not the reviewed production manifest; refusing staging' >&2; exit 1; }
offload/kvpack/stage_overlay.sh verify-checkout
offload/kvpack/stage_overlay.sh stage "$@"
