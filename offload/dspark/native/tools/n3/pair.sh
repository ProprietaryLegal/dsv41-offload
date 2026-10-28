#!/usr/bin/env bash
# COORDINATOR ONLY. No service, Spark, power/clock or deployment changes.
set -euo pipefail
source ~/.config/ndm/paths.sh
root=${N3_ROOT:-$HOME/dsv41-offload/n3}
marker=$HOME/dsv41-offload/PAIR_OK
marker_seen=0; marker_value=
if [[ -e $marker ]]; then marker_seen=1; marker_value=$(cat "$marker"); fi
allowed() { [[ ${N3_PAIR_BY_COORD:-0} == 1 ]] || grep -q '^N3\([[:space:]-]\|$\)' "$marker" 2>/dev/null; }
window_valid() {
  allowed || return 1
  [[ $marker_seen == 0 ]] || { [[ -e $marker ]] && [[ $(cat "$marker") == "$marker_value" ]]; }
}
allowed || { echo 'PAIR_OK must name N3 or coordinator must set N3_PAIR_BY_COORD=1' >&2; exit 1; }
[[ ${N3_V1_KEYS:-} =~ ^r[0-9]+_b[0-9]+(,r[0-9]+_b[0-9]+)*$ ]] || {
  echo 'N3_V1_KEYS must explicitly name comma-separated V1 sample keys before any pair job' >&2; exit 2;
}
requested=0
for arm in ${N3_PAIR_ARMS:-native-packed native-expanded native-hybrid-kx native-hybrid-kx2}; do
  requested=$((requested+1))
  case "$arm" in native-packed|native-expanded|native-hybrid-kx|native-hybrid-kx2) ;; *) echo "unknown arm $arm" >&2; exit 2;; esac
done
[[ $requested -gt 0 ]] || { echo 'N3_PAIR_ARMS contains no arms' >&2; exit 2; }
export DSPARK_P2P_PREBUILT=${DSPARK_P2P_PREBUILT:-$root/p2p}
key=$(sha256sum "$root/repo/offload/dspark/worker/p2p/p2p.cu" | cut -c1-16)
pin_dir=$DSPARK_P2P_PREBUILT/$key
[[ -f $pin_dir/dspark_p2p.so && -f $pin_dir/dspark_p2p.so.sha256 ]] || {
  echo "missing pinned p2p for source $key under $DSPARK_P2P_PREBUILT; no JIT fallback" >&2; exit 2;
}
(cd "$pin_dir" && sha256sum --check dspark_p2p.so.sha256) || exit 2
runid=${N3_PAIR_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
[[ $runid =~ ^[[:alnum:]_.-]+$ ]] || { echo 'invalid run id' >&2; exit 1; }
output=$root/results/pair-$runid
mkdir "$output"
cd "$root/repo"
run() (
  name=$1; shift
  exec > "$output/$name.log" 2>&1
  exec 8> "$HOME/dsv41-offload/locks/gpu-c4e7a371.lock"
  exec 9> "$HOME/dsv41-offload/locks/gpu-7744b44b.lock"
  acquired=0
  for ((attempt=0; attempt<600; attempt++)); do
    window_valid || { echo 'pair authorization withdrawn' >&2; exit 1; }
    if flock -n 8; then
      if flock -n 9; then acquired=1; break; fi
      flock -u 8
    fi
    sleep 1
  done
  [[ $acquired == 1 ]] || { echo 'pair locks unavailable after 600s' >&2; exit 1; }
  window_valid || { echo 'pair authorization withdrawn while waiting' >&2; exit 1; }
  export CUDA_VISIBLE_DEVICES=GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93,GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc
  export TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2 OMP_NUM_THREADS=8 TZ=UTC
  export PYTHONPATH="$root/repo" TORCH_EXTENSIONS_DIR="$root/build" TRITON_CACHE_DIR="$root/triton"
  export N3_PAIR_LOCKED=1 DSPARK_SOURCE_COMMIT=$(cat "$root/repo/SOURCE_COMMIT")
  unset DSPARK_WORKER_TP_EMULATE DSPARK_V1_DUMP_DIR
  nvidia-smi -L
  free -g
  apps=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)
  [[ -z $apps ]] || { echo 'pair occupied: coordinator must stop existing worker before this script' >&2; exit 1; }
  echo "$(date -u +%FT%TZ) acquired $name"
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,uuid,memory.used,power.draw,power.limit --format=csv,noheader -l 1 > "$output/$name.smi.csv" &
  sampler=$!
  start_token() { [[ ! -r /proc/$1/stat ]] || awk '{print $22}' "/proc/$1/stat"; }
  sampler_start=$(start_token "$sampler")
  signal_owned() {
    [[ -z $2 || $(start_token "$1") != "$2" ]] || kill -TERM "$1" 2>/dev/null || true
  }
  timeout --signal=TERM --kill-after=10s 820s "$HOME/dsv41-offload/e/venv/bin/python" -u "$@" &
  child=$!
  child_start=$(start_token "$child")
  trap 'signal_owned "$child" "$child_start"; wait "$child" 2>/dev/null || true; signal_owned "$sampler" "$sampler_start"; wait "$sampler" 2>/dev/null || true' EXIT
  monitor_failed=0
  while [[ -n $child_start && $(start_token "$child") == "$child_start" ]] && kill -0 "$child" 2>/dev/null; do
    if ! window_valid; then
      echo 'PAIR_OK withdrawn: terminating owned process' >&2
      signal_owned "$child" "$child_start"
      break
    fi
    if [[ -z $sampler_start || $(start_token "$sampler") != "$sampler_start" ]]; then
      echo 'memory sampler exited unexpectedly; terminating owned job' >&2
      monitor_failed=1
      signal_owned "$child" "$child_start"
      break
    fi
    sleep 1
  done
  set +e; wait "$child"; rc=$?; set -e
  [[ $monitor_failed == 0 ]] || rc=1
  signal_owned "$sampler" "$sampler_start"
  wait "$sampler" 2>/dev/null || true
  trap - EXIT
  echo "$(date -u +%FT%TZ) released $name rc=$rc"
  exit "$rc"
)
for arm in ${N3_PAIR_ARMS:-native-packed native-expanded native-hybrid-kx native-hybrid-kx2}; do
  case "$arm" in native-packed|native-expanded|native-hybrid-kx|native-hybrid-kx2) ;; *) echo "unknown arm $arm" >&2; exit 1;; esac
  for mode in off on; do
    run "$arm-$mode" -m offload.dspark.native.tools.n3.pair_job "$arm" "$mode" --out "$output/$arm-$mode"
  done
  "$HOME/dsv41-offload/e/venv/bin/python" - "$output" "$arm" <<'PY'
import json,sys
from pathlib import Path
root,arm=Path(sys.argv[1]),sys.argv[2]
a=json.loads((root/f'{arm}-off/client.json').read_text())
b=json.loads((root/f'{arm}-on/client.json').read_text())
assert a['digest']==b['digest'], 'dump on/off output mismatch'
(root/f'{arm}-on-off-equal.json').write_text(json.dumps({'equal':True,'rounds_per_mode':a['rounds']})+'\n')
PY
  # The V1 capture set contains engine_mx-only prefill rows; its typed injection
  # cannot be sent through the raw-tap wire protocol. Replay in the identical
  # worker factory/backend in a separate, bounded hold after stopping the listener.
  run "$arm-capture" -m offload.dspark.native.tools.n3.replay "$arm" \
    --verify-root "${N3_VERIFY_ROOT:-$root/verify-source}" \
    --keys "${N3_V1_KEYS:?set explicit V1 sample keys (N3-v1-keys.txt or full capture)}" \
    --out "$output/$arm-capture"
done
echo "Pair qualification completed: $output (V1 classify still required before GB10)"
