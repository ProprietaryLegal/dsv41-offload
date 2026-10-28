#!/usr/bin/env bash
# stage_overlay.sh NODE [NODE...]: production staging of the remote-DSpark overlay (configuration D). Copies into
# NODE:~/dsv41-offload-prod/dspark/ the remote_dspark package, the spec_decode __init__.py overlay and the ofl transport
# (libofl.so + ofl.py, plus libdsv41_hostflag.so for async rounds), verifies every copy, and checks that the image's own
# spec_decode/__init__.py is the file the overlay was made from. dsv41.env's OFFLOAD_DOCKER_ARGS bind-mounts these paths
# read-only into the serve container.
# libdsv41_hostflag.so (S9 F8): the aarch64 build of remote_dspark/hostflag.c that async rounds load from
# VLLM_DSPARK_REMOTE_OFL_DIR. Required (missing / not the pinned build / source not the pinned source = refusal) when
# OFFLOAD_DSPARK_ENV sets VLLM_DSPARK_REMOTE_ASYNC=1 or HOSTFLAG_REQUIRED=1; otherwise staged when it is the pinned
# build and skipped with a note when not (synchronous rounds never load it). Build on the head node, then pin:
#   cc -O2 -Wall -shared -fPIC -o ~/dsv41-offload-run/ofl/libdsv41_hostflag.so \
#      offload/dspark/spark/remote_dspark/hostflag.c -lpthread && sha256sum ~/dsv41-offload-run/ofl/libdsv41_hostflag.so
# Coordinator-only. Run from the repo root on the head node (start.sh does it when OFFLOAD_DSPARK_ENV is set):
#   offload/dspark/spark/stage_overlay.sh spark sparkworker
# Remove:  ssh NODE rm -rf ~/dsv41-offload-prod/dspark   (nothing reads it once OFFLOAD_DSPARK_ENV is unset)
set -euo pipefail
cd "$(dirname "$0")/../../.."
PKG=offload/dspark/spark/remote_dspark
INIT=offload/dspark/spark/overlay/vllm/v1/worker/gpu/spec_decode/__init__.py
OFL_PY=offload/transport/ofl.py
# libofl.so is an aarch64 build product (offload/dspark/INTEGRATION.md 1.3); the staged copy must be this exact build.
# Production reads production-owned pinned copies (S7 F3), so a rebuild into the experiment folder cannot change or block it.
OFL_SO_SRC=${OFL_SO_SRC:-$HOME/dsv41-offload-prod/pinned/libofl.so}
OFL_SO_SHA=4daef6c28f4d777817d2bc8e2af1a9053f2dea7e54bbf8871f97c762bcf8d8e5
# hostflag v2 (lane A2 after S9: monotonic release, GIL-free publish, native watchdog). The v1 pins were source
# 6eb15a07... / aarch64 build 2d4f1f9f...; the v2 build's sha256 is set here after it is built on the head node.
HOSTFLAG_C=offload/dspark/spark/remote_dspark/hostflag.c
HOSTFLAG_C_SHA=10490b58aeabc1fbe45e22afcbe4d0a5ee25b1b9c677789da6949e0ee73c781a
HOSTFLAG_SO_SRC=${HOSTFLAG_SO_SRC:-$HOME/dsv41-offload-prod/pinned/libdsv41_hostflag.so}
HOSTFLAG_SO_SHA=0ada6aee896842b2196734ec128749edfff52b0b54685fcca27501484f8906c4   # v2, built 2026-10-23 on both Sparks (identical)
IMAGE_INIT_SHA=61b3acba401c1079039496f72445c28bbc6119d0b508f9a75d231f305aa1b7e8
IMAGE=${IMAGE:-ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v5}
IMG_INIT=/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/__init__.py
DEST=dsv41-offload-prod/dspark
[ "$(sha256sum < "$OFL_SO_SRC" | cut -d' ' -f1)" = "$OFL_SO_SHA" ] \
  || { echo "$OFL_SO_SRC is not the pinned libofl.so build ($OFL_SO_SHA)" >&2; exit 1; }
hf_required=${HOSTFLAG_REQUIRED:-0}
[[ "${OFFLOAD_DSPARK_ENV:-}" == *VLLM_DSPARK_REMOTE_ASYNC=1* ]] && hf_required=1
hf_problem=""
if [ "$(sha256sum < "$HOSTFLAG_C" | cut -d' ' -f1)" != "$HOSTFLAG_C_SHA" ]; then
  hf_problem="$HOSTFLAG_C is not the pinned hostflag source ($HOSTFLAG_C_SHA): rebuild libdsv41_hostflag.so from it and update both pins"
elif [ ! -f "$HOSTFLAG_SO_SRC" ]; then
  hf_problem="$HOSTFLAG_SO_SRC is missing (build command in this script's header)"
elif [ "$(sha256sum < "$HOSTFLAG_SO_SRC" | cut -d' ' -f1)" != "$HOSTFLAG_SO_SHA" ]; then
  hf_problem="$HOSTFLAG_SO_SRC ($(sha256sum < "$HOSTFLAG_SO_SRC" | cut -d' ' -f1)) is not the pinned libdsv41_hostflag.so build ($HOSTFLAG_SO_SHA)"
fi
if [ -n "$hf_problem" ]; then
  [ "$hf_required" = 1 ] && { echo "async rounds requested but $hf_problem" >&2; exit 1; }
  echo "note: libdsv41_hostflag.so not staged ($hf_problem); synchronous rounds do not load it"
fi
stage=$(mktemp -d); trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/remote_dspark" "$stage/ofl"
(cd "$PKG" && find . -name '*.py' -not -path '*/__pycache__/*' -print0 | cpio -0pdm --quiet "$stage/remote_dspark")
cp "$INIT" "$stage/spec_decode__init__.py"
cp "$OFL_PY" "$OFL_SO_SRC" "$stage/ofl/"
[ -z "$hf_problem" ] && cp "$HOSTFLAG_SO_SRC" "$stage/ofl/libdsv41_hostflag.so"
local_sums=$(cd "$stage" && find . -type f | LC_ALL=C sort | xargs sha256sum)
for n in "$@"; do
  echo "== $n"
  ssh -o BatchMode=yes "$n" "rm -rf \$HOME/$DEST && mkdir -p \$HOME/$DEST"
  tar cf - -C "$stage" . | ssh -o BatchMode=yes "$n" "tar xf - -C \$HOME/$DEST"
  remote_sums=$(ssh -o BatchMode=yes "$n" "cd \$HOME/$DEST && find . -type f | LC_ALL=C sort | xargs sha256sum")
  [ "$local_sums" = "$remote_sums" ] || { echo "$n: staged copies differ" >&2; exit 1; }
  echo "  staged $(printf '%s\n' "$local_sums" | wc -l) files, sha256 match"
  img=$(ssh -o BatchMode=yes "$n" "docker run --rm --entrypoint sha256sum $IMAGE $IMG_INIT" | cut -d' ' -f1)
  [ "$img" = "$IMAGE_INIT_SHA" ] || { echo "$n: the image's spec_decode/__init__.py ($img) is not the overlay's base" >&2; exit 1; }
  echo "  image spec_decode/__init__.py matches the overlay's base"
done
