#!/usr/bin/env bash
# RX-TP2SAFE tests only. No service mutation. Run ON X86HOST; locks cover context creation, execution and telemetry.
set -euo pipefail
source "$HOME/.config/ndm/paths.sh"
mode=${1:?usage: run_owned.sh solo|pair LOG command...}
log=${2:?log path required}
shift 2
[[ $(hostname) == x86host ]] || { echo 'run_owned.sh must run on x86host' >&2; exit 2; }
lane=RX-TP2SAFE
base="$HOME/dsv41-offload"
solo=GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc
prod=GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93
pair_owned() { [[ -f "$base/PAIR_OK" ]] && rg -q '(^|[^A-Z0-9-])RX-TP2SAFE([^A-Z0-9-]|$)' "$base/PAIR_OK"; }
if [[ $mode == pair ]]; then
  pair_owned || { echo 'PAIR_OK does not name RX-TP2SAFE' >&2; exit 3; }
  exec 9>"$base/locks/gpu-c4e7a371.lock"
  flock -w 1800 9
  exec 8>"$base/locks/gpu-7744b44b.lock"
  flock -w 1800 8
  pair_owned || { echo 'PAIR_OK revoked while waiting for locks' >&2; exit 3; }
  export CUDA_VISIBLE_DEVICES="$prod,$solo"
elif [[ $mode == solo ]]; then
  exec 8>"$base/locks/gpu-7744b44b.lock"
  flock -w 1800 8
  export CUDA_VISIBLE_DEVICES="$solo"
else
  echo "unknown mode: $mode" >&2; exit 2
fi
mkdir -p "$(dirname "$log")"
export DSPARK_P2P_BUILD="$base/rx-tp2safe/build"
export PYTHONPATH="$base/rx-tp2safe/src/offload/dspark"
{
  date -u +%FT%TZ
  printf 'lane=%s mode=%s CUDA_VISIBLE_DEVICES=%s\n' "$lane" "$mode" "$CUDA_VISIBLE_DEVICES"
  printf 'command:'; printf ' %q' "$@"; printf '\n'
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,name,power.limit,clocks.max.sm --format=csv
} > "$log"
# A new process group contains only this lane's test and children. Never signal shared services/processes.
setsid timeout --signal=TERM --kill-after=5 840 "$@" >> "$log" 2>&1 &
job=$!
cleanup() {
  if kill -0 "$job" 2>/dev/null; then kill -TERM -- "-$job" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM
revoked=0
while kill -0 "$job" 2>/dev/null; do
  if [[ $mode == pair ]] && ! pair_owned; then
    echo 'PAIR_OK revoked: terminate own test process group immediately' >> "$log"
    cleanup
    revoked=1
    break
  fi
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,uuid,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu,memory.used --format=csv,noheader >> "$log.telemetry.csv"
  sleep 0.2
done
set +e
wait "$job"
rc=$?
set -e
if (( revoked )); then rc=3; fi
printf 'EXIT=%s\n' "$rc" >> "$log"
exit "$rc"
