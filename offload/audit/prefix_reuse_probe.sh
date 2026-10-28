#!/usr/bin/env bash
# AUD1 item 2 wrapper. Dry plan by default (no network). Coordinator-only execution, with production idle:
#   offload/audit/prefix_reuse_probe.sh --execute --base-url http://HEAD_IP:8000 \
#       --out /opt/dsv41/campaigns/DSV41-OFFLOAD/AUD1-prefix-probe-$(date -u +%Y%m%dT%H%M%SZ).json   (absolute: the wrapper cds first)
# Sends exactly two chat requests (same ~150K-token public document, two questions, max_tokens 8, sequential) and
# reads /metrics before and after each; refuses to send while any request runs or waits. Writes numbers and hashes only.
# The coordinator's host needs the pinned corpus: --data PATH (sha256 80461203...b6a1; sparkhead: ~/dsv41-deploy/data).
set -euo pipefail
cd "$(dirname "$0")"
exec python3 -I prefix_reuse_probe.py "$@"
