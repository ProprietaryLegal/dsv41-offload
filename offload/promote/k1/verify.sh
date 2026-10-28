#!/usr/bin/env bash
# Coordinator-only, after the launcher answers, before start.sh warm-up/success.
set -euo pipefail
cd "$(dirname "$0")/../../.."
set -a; . ./dsv41.env; set +a
. offload/promote/k1/constants.sh
k1_select_profile "${PROD_K1_PROFILE:?missing applied K1 profile}"
for key in ${!KVPACK_TEST_@}; do
  echo "K1 production verification refuses test hook $key" >&2; exit 1
done
python3 offload/promote/k1/manage.py verify --profile "${PROD_K1_PROFILE:?missing applied K1 profile}"
offload/kvpack/stage_overlay.sh verify-running --expect "$K1_EXPECT" "$HEAD" "$WORKER"
# Fresh files only, obtained from this container (docker rm/run resets its logs).
evidence=$(mktemp -d logs/k1-runtime.XXXXXXXX)
ssh -o BatchMode=yes -o ConnectTimeout=10 "$HEAD" 'docker logs dsv41-serve 2>&1' > "$evidence/engine.log"
curl --fail --silent --show-error --max-time 15 "http://localhost:$PORT/v1/models" > "$evidence/models.json"
python3 offload/promote/k1/check_runtime.py --profile "$PROD_K1_PROFILE" --log "$evidence/engine.log" --models "$evidence/models.json" --served "$SERVED"
echo "K1 runtime evidence: $evidence"
