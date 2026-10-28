#!/usr/bin/env bash
# RP1 P0 -> independent-rank P1 -> single-writer P2; 256 MiB only.
# shellcheck source=offload/launch/windows/common.sh
source "$(dirname -- "$0")/common.sh"
w_init rp1 "$@"
[[ ${#W_ARGS[@]} == 4 ]] || w_die 'usage: rp1.sh [--dry-run|--execute] P0_KV P1_P2_KV MODEL_MANIFEST_SHA256 VERIFIED_IO_CPUS'
p0kv=${W_ARGS[0]} kv=${W_ARGS[1]} model=${W_ARGS[2]} cpus=${W_ARGS[3]}
[[ $model =~ ^[0-9a-f]{64}$ && $cpus =~ ^[0-9,-]+$ ]] || w_die 'explicit manifest SHA256 and verified allowed CPU list required'
w_uint "$p0kv"; w_uint "$kv"
cfg=offload/launch/configs/D-kv6-async-pack-prefix.env
w_rp1_verify() {
  local arm=$1; shift
  w_step "$arm-rp1-runtime" "$W_PY" offload/prefixstore/verify_overlay.py running "$W_HEAD" "$W_PEER" --io-cpus "$cpus" "$@"
  w_k1 "$arm"
}
w_p2_start() {
  w_start "$1" "$cfg" "$kv" PREFIX_ENABLE=1 PREFIX_MN_VERIFY=0 PREFIX_CPU_BYTES=268435456 PREFIX_LOAD_FRACTION=0.2 PREFIX_MODEL_SHA256="$model" PREFIX_IO_CPUS="$cpus"
  w_rp1_verify "$1"
}
w_open_checks
w_gate rp1-preflight window "$model" "$cpus" 268435456
# Stage only reviewed local generated overlays, never fetch a donor or mutate production.
w_head rp1-checkout python3 offload/prefixstore/build_overlay.py --checkout-only
w_start P0 "$cfg" "$p0kv" PREFIX_ENABLE=0
w_rp1_verify P0 --control
w_gate rp1-p0 P0
w_greedy P0
w_measure P0
w_start P1 "$cfg" "$kv" PREFIX_ENABLE=1 PREFIX_MN_VERIFY=1 PREFIX_CPU_BYTES=268435456 PREFIX_LOAD_FRACTION=0.2 PREFIX_MODEL_SHA256="$model" PREFIX_IO_CPUS="$cpus"
w_rp1_verify P1 --verify-stores
w_step P1-needle24k "$W_PY" offload/kvpack/needle.py --url "$W_URL" --ctx 24000 --reps 1 --needles 6 --out "$W_OUT/P1-needle.jsonl"
# Mandatory independent valid live group payload SHA+CRC BEFORE any reload, not mmap padding/post-broadcast.
w_gate rp1-rank P1 "$model" 268435456 29568
for trial in 1 2 3; do
  arm=P2-t$trial
  w_p2_start "$arm-cold"
  packet=$W_OUT/$arm-packet.json
  w_step "$arm-prepare" "$W_PY" offload/prefixstore/restart_probe.py prepare --url "$W_URL" --model DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2 --ctx 24000 75000 150000 --out "$packet"
  w_step "$arm-cold" timeout 300s "$W_PY" offload/prefixstore/restart_probe.py run --packet "$packet" --url "$W_URL" --out "$W_OUT/$arm-cold.json"
  # Hook waits for stores, records old engine ID, stops owned experiment, proves no maps/processes, removes ONLY unused shm.
  w_gate rp1-restart "$arm" "$packet"
  w_p2_start "$arm-reload"
  w_gate rp1-empty "$arm" "$packet"
  w_step "$arm-reload" timeout 300s "$W_PY" offload/prefixstore/restart_probe.py run --packet "$packet" --url "$W_URL" --out "$W_OUT/$arm-reload.json"
  w_step "$arm-fresh" timeout 300s "$W_PY" offload/prefixstore/restart_probe.py run --packet "$packet" --url "$W_URL" --fresh --out "$W_OUT/$arm-fresh.json"
  w_step "$arm-compare" "$W_PY" offload/prefixstore/restart_probe.py compare --packet "$packet" --cold "$W_OUT/$arm-cold.json" --reload "$W_OUT/$arm-reload.json" --fresh "$W_OUT/$arm-fresh.json" --out "$W_OUT/$arm-compare.json"
  w_gate rp1-reload-evidence "$arm" "$packet"
 done
for gate in rp1-stale rp1-crc rp1-sidecar rp1-namespace rp1-concurrency; do
  # Each adapter operation must arrange its own documented stop/restart/fresh stores and restore true identity.
  w_gate "$gate" P2 "$cfg" "$model" "$cpus" 268435456 29568 "$W_URL"
 done
# Prove the ordinary enabled config again after the negative/fault operations.
w_p2_start P2-final
w_greedy P2-final
w_compare P0 P2-final
w_step P2-corruption-needles "$W_PY" offload/kvpack/needle.py --url "$W_URL" --ctx 8000 30000 50000 100000 --concurrent 4 --needles 6 --out "$W_OUT/P2-needles.jsonl"
w_step P2-prefix "$W_PY" offload/kvpack/prefix_identity.py --url "$W_URL" --block 384 --n 36016 100017 250000 --partial 250000:200000 --out "$W_OUT/P2-prefix.jsonl"
# A/B/A decode regression, three starts each, matched cold/warm/mixed cells.
for rep in 1 2 3; do
  for phase in C1 X C2; do
    arm=rp1-r$rep-$phase
    if [[ $phase == X ]]; then w_p2_start "$arm"
    else w_start "$arm" "$cfg" "$p0kv" PREFIX_ENABLE=0; w_rp1_verify "$arm" --control; fi
    w_measure "$arm"
  done
 done
w_summary
