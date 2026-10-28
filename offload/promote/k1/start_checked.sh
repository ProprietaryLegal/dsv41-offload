#!/usr/bin/env bash
# Coordinator-only; entered by the start.sh patch. The production window/HOLD
# stays open until independent post-start probes pass. No success on fallback.
set -Eeuo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p logs
log="logs/k1-guard-$(date -u +%Y%m%dT%H%M%S).log"
set -a; . ./dsv41.env; set +a
# A repeated ./start.sh must never stop an already-running production serve.
# Fail before installing the fallback trap if a container is up or inspection fails.
for node in "$HEAD" "$WORKER"; do
  running=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$node" \
    "docker ps --filter 'name=^/dsv41-serve$' --format '{{.Names}}'")
  [ -z "$running" ] || { echo "K1 start refuses: $node already runs $running; drain/stop in a window first" >&2; exit 1; }
done
fallback() {
  local rc=$1 line=$2
  trap - ERR
  echo "K1 START FAILED (status $rc, line $line); stopping candidate and restoring previous config" | tee -a "$log" >&2
  # Only this deployment's containers; no process-name matching or unrelated kills.
  if ! timeout --kill-after=15s 45s bash ./stop.sh </dev/null >> "$log" 2>&1; then
    echo "K1 FALLBACK FAILED: could not stop candidate; keep HOLD and inspect $log" >&2; exit 1
  fi
  if ! python3 offload/promote/k1/manage.py restore >> "$log" 2>&1; then
    echo "K1 FALLBACK FAILED: backup/edits need operator review; keep HOLD and inspect $log" >&2; exit 1
  fi
  if timeout --kill-after=15s 600s bash ./start.sh </dev/null >> "$log" 2>&1; then
    echo "K1 REFUSED; previous configuration restarted. Inspect $log; keep HOLD until rollback checks pass." >&2
  else
    echo "K1 FALLBACK FAILED: previous start failed/timed out; keep HOLD and inspect $log" >&2
    exit 1
  fi
  exit "$rc"
}
trap 'fallback "$?" "$LINENO"' ERR
python3 offload/promote/k1/manage.py verify --profile "${PROD_K1_PROFILE:?missing applied K1 profile}" 2>&1 | tee -a "$log"
bash offload/promote/k1/stage.sh "$HEAD" "$WORKER" 2>&1 | tee -a "$log"
# Bound the existing launcher's unbounded /v1/models wait. timeout owns only
# the start process group it launches. The fallback has its own bounded start.
K1_START_INNER=1 timeout --kill-after=15s 600s bash ./start.sh </dev/null 2>&1 | tee -a "$log"
echo 'K1 START VERIFIED; complete the production-window probes before releasing HOLD' | tee -a "$log"
