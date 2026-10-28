#!/usr/bin/env bash
# UP2-PROMOTE staging (coordinator only, ON Controller, BEFORE the HOLD). Creates NEW immutable directories only; never
# touches a running service, dsv41.env, production's ~/dsv41-offload-prod/dspark (stage_overlay.sh rm -rf's that one at
# every start) or an existing directory (each step refuses one).
#   Sparks (sparkhead + sparkworker-wired): ~/dsv41-offload-prod/up2/<id>/{vllm/...,remote_dspark/}, SHA256SUMS, package.json
#       built by `offload.promote.up2.package materialize` (the committed manifest is re-derived from the pinned inputs
#       first; a drift refuses). <id> = sha256(SHA256SUMS)[:12].
# Usage: WINDOW_COORDINATOR=Deepseek/coordinator bash offload/promote/up2/stage.sh --execute [qualified|pp4-composed]
set -euo pipefail
cd "$(dirname "$0")/../../.."
[ "${1-}" = --execute ] || { sed -n 2,8p "$0"; exit 0; }
[ "${WINDOW_COORDINATOR-}" = Deepseek/coordinator ] || { echo "coordinator only" >&2; exit 2; }
NAME=${2-qualified}
case "$NAME" in qualified|pp4-composed) ;; *) echo "unknown package $NAME" >&2; exit 2 ;; esac
PY=/home/operator/miniconda3/bin/python3
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/offload/dspark CUDA_VISIBLE_DEVICES=
ID=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["id"])' "offload/promote/up2/packages/$NAME.json")
[[ "$ID" =~ ^[0-9a-f]{12}$ ]] || { echo "bad package id $ID" >&2; exit 2; }
scratch=/opt/dsv41/campaigns/DSV41-OFFLOAD/scratch/UP2PROMO/stage-$ID
[ ! -e "$scratch" ] || { echo "refusing: $scratch exists" >&2; exit 2; }
mkdir -p "$(dirname "$scratch")"
trap 'rm -rf "$scratch"' EXIT
$PY -m offload.promote.up2.package materialize --name "$NAME" --out "$scratch"
(cd "$scratch" && sha256sum -c --quiet SHA256SUMS)
for h in sparkhead sparkworker-wired; do
  ssh -o BatchMode=yes "$h" "test ! -e ~/dsv41-offload-prod/up2/$ID && test ! -e ~/dsv41-offload-prod/up2/$ID.tmp"
  tar -C "$scratch" -cf - . | ssh -o BatchMode=yes "$h" "mkdir -p ~/dsv41-offload-prod/up2/$ID.tmp && \
    tar -x -C ~/dsv41-offload-prod/up2/$ID.tmp && cd ~/dsv41-offload-prod/up2/$ID.tmp && sha256sum -c --quiet SHA256SUMS && \
    chmod -R a-w . && mv ~/dsv41-offload-prod/up2/$ID.tmp ~/dsv41-offload-prod/up2/$ID && echo \"\$(hostname): staged UP2 $ID\""
done
echo "UP2 STAGED $NAME $ID: Sparks ~/dsv41-offload-prod/up2/$ID"
