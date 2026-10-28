#!/usr/bin/env bash
# DSV41-OFFLOAD failure drill (coordinator only, inside an open window): one long single-stream request on the C/D serve
# while the x86host drafter worker is stopped mid-run and started again. The request must finish without error; the
# remote-DSpark stats must show the target-only rounds and the resync.
#   offload/bench/drill_worker.sh <LABEL> <URL> <OUTDIR> [stop_after_s=25] [down_s=30]
set -euo pipefail; cd "$(dirname "$0")/../.."
L=${1:?label}; U=${2:?url}; O=${3:?outdir}; STOP_AFTER=${4:-25}; DOWN=${5:-30}
mkdir -p "$O"
stats() { ssh -o BatchMode=yes sparkhead "docker exec dsv41-serve cat /tmp/dsv41-remote-dspark.json"; }
stats > "$O/$L-drill-stats-before.json"
python3 offload/bench/bench2.py --url "$U" --label "$L-drill" --thinking --reasoning-effort high --priority 10 \
  --temperature 1 --top-p 1 --data-dir "${BENCH_DATA:-/opt/dsv41/data/bench-data}" --ctx 0 --workload legal \
  --concurrency 1 --reps 1 --max-tokens 6000 --cache-mode cold --out "$O/$L-drill.jsonl" &
bench=$!
sleep "$STOP_AFTER"
echo "$(date -u +%FT%T.%3NZ) stop worker" | tee "$O/$L-drill-events.txt"
ssh -o BatchMode=yes x86host 'systemctl --user stop dsv41-dspark-worker && systemctl --user is-active dsv41-dspark-worker || true'
stats > "$O/$L-drill-stats-down.json" || true
sleep "$DOWN"
echo "$(date -u +%FT%T.%3NZ) start worker" | tee -a "$O/$L-drill-events.txt"
ssh -o BatchMode=yes x86host 'systemctl --user start dsv41-dspark-worker && systemctl --user is-active dsv41-dspark-worker'
wait "$bench"; rc=$?
echo "$(date -u +%FT%T.%3NZ) request finished rc=$rc" | tee -a "$O/$L-drill-events.txt"
sleep 5
stats > "$O/$L-drill-stats-after.json"
python3 - "$O/$L-drill-stats-before.json" "$O/$L-drill-stats-after.json" <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
def flat(d, p=""):
    for k, v in d.items():
        if isinstance(v, dict): yield from flat(v, p + k + ".")
        elif isinstance(v, (int, float)): yield p + k, v
fa, fb = dict(flat(a)), dict(flat(b))
for k in sorted(fb):
    if any(s in k for s in ("rounds", "fallback", "hello", "resync", "breaker", "reconnect", "fail", "timeout")):
        d = fb[k] - fa.get(k, 0)
        if d: print(f"{k}: +{d:g}")
EOF
exit "$rc"
