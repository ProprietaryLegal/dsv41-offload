#!/usr/bin/env bash
# K1 (P1): the KV-packing overlay as bind mounts for BOTH the engine-core process (scheduler: kv_cache_utils,
# kv_cache_interface) and the TP workers (attention, model, runner, plugin). A bind mount covers every process in the
# container, so one mechanism serves both. Coordinator-only (lanes never write on the Sparks). Run from the repo root.
#   offload/kvpack/stage_overlay.sh verify-checkout       this checkout's overlay/ files == MANIFEST.json overlay_sha256
#                                                          (run on sparkhead before a start: start_exp copies from there)
#   offload/kvpack/stage_overlay.sh check NODE...         the image on NODE is the overlay's base (MANIFEST base_sha256)
#                                                          and lacks the overlay's new files; refuses otherwise
#   offload/kvpack/stage_overlay.sh stage NODE...         check, then copy overlay/ to NODE:~/dsv41-kvpack/<id>/ and verify
#                                                          the copies (<id> = MANIFEST sha256[:12]; never overwritten)
#   offload/kvpack/stage_overlay.sh verify-running [--expect "<K1 switches>"] NODE...
#                                                          after a start: sha256 of the 13 paths INSIDE the running
#                                                          dsv41-serve == overlay_sha256, and the live K1 switches printed;
#                                                          with --expect, they must equal the given set ("" = all off)
#   offload/kvpack/stage_overlay.sh args                  docker -v arguments for the staged <id> (production docker args)
#   offload/kvpack/stage_overlay.sh overlays              the OVERLAYS pairs for offload/launch/start_exp.sh configs
# The overlay is inert until a switch is set (MANIFEST.json switches_default_off). Remove: drop the -v arguments /
# OVERLAYS and restart; `ssh NODE rm -rf ~/dsv41-kvpack/<id>`. docker run never pulls (--pull never): a missing image
# tag fails instead of downloading GBs onto a serving node.
# Test hooks (offline only): KVPACK_TEST_IMAGE_ROOT=<dir> stands in for the image, KVPACK_TEST_RUNNING_ROOT=<dir> and
# KVPACK_TEST_RUNNING_ENV=<file of KEY=VALUE lines> for the running container; the node name `local` runs on this host.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
IMAGE=${IMAGE:-ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v5}
CONTAINER=${CONTAINER:-dsv41-serve}
PREFIX=/usr/local/lib/python3.12/dist-packages/
# Every switch the overlay adds (vLLM-image switches such as VLLM_DSV41_AK_* / _MHC_* are production's own and not listed)
K1_SWITCHES="VLLM_DSV41_KV_PAIR VLLM_DSV41_KV_SWA_FILL VLLM_DSV41_KV_NVFP4 VLLM_DSV41_INDEXER_DEFER VLLM_DSV41_DISPLAY_KV_MIB
VLLM_DSV41_DROP_MTP_BUFFER VLLM_EXL3_TF_PROMPT_TRANSIENT VLLM_EXL3_TF_PROMPT_ALIAS VLLM_DSV41_MEMPROBE
VLLM_DSV41_MEMPROBE_LAYERS VLLM_DSV41_MEMPROBE_MIN_TOKENS VLLM_DSV41_MEMPROBE_MAX VLLM_INDEXER_PREFILL_BUFFER_FACTOR"
manifest="$here/MANIFEST.json"
id=$(sha256sum "$manifest" | cut -c1-12)
mapfile -t ROWS < <(python3 -c '
import json, sys
for e in json.load(open(sys.argv[1]))["files"]:
    print(e["path"], e["base_sha256"] or "NEW", e["overlay_sha256"])' "$manifest")

on() {  # on NODE CMD: run CMD on NODE (local = this host)
  local n=$1; shift
  if [ "$n" = local ]; then bash -c "$*"; else ssh -o BatchMode=yes "$n" "$*"; fi
}

compare() {  # compare NODE WHAT "<sha path>\n…" "<got sha…>": every expected sha matched, in order
  local n=$1 what=$2 want=$3 got=$4 bad=0 i=0 line h exp p
  mapfile -t W <<<"$want"; mapfile -t G <<<"$got"
  [ "${#G[@]}" = "${#W[@]}" ] || { echo "$n: $what: read ${#G[@]} of ${#W[@]} files" >&2; return 1; }
  for line in "${W[@]}"; do
    exp=${line%% *}; p=${line#* }; h=${G[$i]}; i=$((i + 1))
    [ "$h" = "$exp" ] || { echo "$n: $what $p sha256 ${h:-<none>}, expected $exp" >&2; bad=1; }
  done
  return $bad
}

image_check() {  # image_check NODE: the image's base files == manifest bases, new files absent
  local n=$1 bad=0 row path base want="" paths="" new=() got out
  for row in "${ROWS[@]}"; do
    read -r path base _ <<<"$row"
    if [ "$base" = NEW ]; then new+=("$path"); else want+="$base $path"$'\n'; paths+=" $PREFIX$path"; fi
  done
  want=${want%$'\n'}
  if [ -n "${KVPACK_TEST_IMAGE_ROOT:-}" ]; then
    got=$(cd "$KVPACK_TEST_IMAGE_ROOT" && while read -r _ p; do sha256sum "$p" | cut -d' ' -f1; done <<<"$want")
  else
    got=$(on "$n" "docker run --rm --pull never --entrypoint sha256sum $IMAGE$paths" | awk '{print $1}') \
      || { echo "$n: docker run on $IMAGE failed (image missing? it is never pulled)" >&2; return 1; }
  fi
  compare "$n" image "$want" "$got" || bad=1
  # the overlay's new files must not exist in the image; a docker failure is reported as one, not as "exists"
  local probe="for p in ${new[*]/#/$PREFIX}; do if [ -e \"\$p\" ]; then echo EXISTS \$p; else echo ABSENT \$p; fi; done"
  if [ -n "${KVPACK_TEST_IMAGE_ROOT:-}" ]; then
    out=$(for p in "${new[@]}"; do if [ -e "$KVPACK_TEST_IMAGE_ROOT/$p" ]; then echo "EXISTS $p"; else echo "ABSENT $p"; fi; done)
  else
    out=$(on "$n" "docker run --rm --pull never --entrypoint sh $IMAGE -c '$probe'") \
      || { echo "$n: docker run on $IMAGE failed while probing new files" >&2; return 1; }
  fi
  [ "$(grep -c '^ABSENT ' <<<"$out")" = "${#new[@]}" ] || { echo "$n: new-file probe: $(grep -v '^ABSENT ' <<<"$out" | tr '\n' ' ')" >&2; bad=1; }
  [ $bad = 0 ] || { echo "$n: the image is not the overlay's base; not staging" >&2; return 1; }
  echo "$n: image matches the overlay base ($((${#ROWS[@]} - ${#new[@]})) files, ${#new[@]} new) ($IMAGE)"
}

stage() {  # stage NODE
  local n=$1 dst="\$HOME/dsv41-kvpack/$id" row path ovl sums=""
  image_check "$n"
  if on "$n" "test -d $dst"; then echo "$n: $dst exists (staged before); verifying"; else
    on "$n" "mkdir -p $dst.tmp"
    (cd "$here/overlay" && tar cf - $(for row in "${ROWS[@]}"; do echo "${row%% *}"; done)) | on "$n" "tar xf - -C $dst.tmp"
    on "$n" "mv $dst.tmp $dst"
  fi
  for row in "${ROWS[@]}"; do read -r path _ ovl <<<"$row"; sums+="$ovl  $path"$'\n'; done
  on "$n" "cd $dst && sha256sum -c --quiet -" <<<"$sums" || { echo "$n: staged copies differ from MANIFEST" >&2; return 1; }
  echo "$n: staged ${#ROWS[@]} files at ~/dsv41-kvpack/$id, sha256 verified"
}

verify_checkout() {
  local row path ovl want="" got
  for row in "${ROWS[@]}"; do read -r path _ ovl <<<"$row"; want+="$ovl $path"$'\n'; done
  want=${want%$'\n'}
  got=$(cd "$here/overlay" && while read -r _ p; do sha256sum "$p" 2>/dev/null | cut -d' ' -f1 || echo missing; done <<<"$want")
  compare "$(hostname) checkout" overlay "$want" "$got" || { echo "this checkout's overlay is not the reviewed one" >&2; return 1; }
  echo "$(hostname): checkout $(git -C "$here" rev-parse --short HEAD 2>/dev/null || echo '?') overlay == MANIFEST ($id), ${#ROWS[@]} files"
}

verify_running() {  # verify_running EXPECT_SET(or __none__) NODE
  local expect=$1 n=$2 row path ovl want="" paths="" got envs live
  for row in "${ROWS[@]}"; do read -r path _ ovl <<<"$row"; want+="$ovl $path"$'\n'; paths+=" $PREFIX$path"; done
  want=${want%$'\n'}
  if [ -n "${KVPACK_TEST_RUNNING_ROOT:-}" ]; then
    got=$(cd "$KVPACK_TEST_RUNNING_ROOT" && while read -r _ p; do sha256sum "$p" 2>/dev/null | cut -d' ' -f1 || echo missing; done <<<"$want")
    envs=$(cat "$KVPACK_TEST_RUNNING_ENV")
  else
    got=$(on "$n" "docker exec $CONTAINER sha256sum$paths" | awk '{print $1}') \
      || { echo "$n: docker exec $CONTAINER sha256sum failed (not running, or a path is missing)" >&2; return 1; }
    envs=$(on "$n" "docker inspect $CONTAINER --format '{{range .Config.Env}}{{println .}}{{end}}'") \
      || { echo "$n: docker inspect $CONTAINER failed" >&2; return 1; }
  fi
  compare "$n" running "$want" "$got" || { echo "$n: the running container does not serve the reviewed overlay" >&2; return 1; }
  live=$(for k in $K1_SWITCHES; do grep -E "^$k=" <<<"$envs" || true; done | sort | tr '\n' ' ' | sed 's/ $//')
  echo "$n: running $CONTAINER serves the reviewed overlay ($id, ${#ROWS[@]} files); K1 switches: ${live:-none}"
  if [ "$expect" != __none__ ]; then
    local want_set; want_set=$(tr ' ' '\n' <<<"$expect" | grep -v '^$' | sort | tr '\n' ' ' | sed 's/ $//')
    [ "$live" = "$want_set" ] || { echo "$n: K1 switches '${live}' != expected '${want_set}'" >&2; return 1; }
    echo "$n: K1 switches as expected"
  fi
}

case ${1:-} in
  verify-checkout) verify_checkout ;;
  check) shift; for n in "$@"; do image_check "$n"; done ;;
  stage) shift; for n in "$@"; do stage "$n"; done ;;
  verify-running)
    shift; expect=__none__
    if [ "${1:-}" = --expect ]; then expect=${2-}; shift 2; fi
    [ $# -gt 0 ] || { echo "verify-running needs NODE..." >&2; exit 2; }
    rc=0; for n in "$@"; do verify_running "$expect" "$n" || rc=1; done; exit $rc ;;
  args)  for row in "${ROWS[@]}"; do p=${row%% *}; printf ' -v $HOME/dsv41-kvpack/%s/%s:%s%s:ro' "$id" "$p" "$PREFIX" "$p"; done; echo ;;
  overlays) for row in "${ROWS[@]}"; do p=${row%% *}; printf '%s ' "offload/kvpack/overlay/$p=$PREFIX$p"; done | sed 's/ $//'; echo ;;
  *) sed -n '2,24p' "$0" >&2; exit 2 ;;
esac
