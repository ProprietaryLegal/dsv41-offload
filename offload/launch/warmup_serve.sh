#!/usr/bin/env bash
# Internal startup requests only. The coordinator holds/drains producers first.
# No changes to server defaults or production sampling. See OPERATIONS.md.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ -f dsv41.env ]; then set -a; . ./dsv41.env; set +a; fi
# Hard process-wide bound (S48 FIX 1): the helper's own 90 s deadline cannot stop a trickling HTTP read or a waiting
# worker thread, so the whole process is ended at 95 s (KILL 5 s later). Exit 124/137 is a failure, which start.sh
# turns into its existing WARN; K1's 600 s start guard is never reached by the warm-up (last K1 start 432 s).
exec timeout --kill-after=5s "${WARMUP_HARD_S:-95}s" python3 offload/launch/warmup.py --port "${1:-${PORT:-8000}}" \
  --model "${2:-${SERVED:?served model name (argument 2 or SERVED)}}"
