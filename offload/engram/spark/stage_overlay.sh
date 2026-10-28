#!/usr/bin/env bash
# stage_overlay.sh NODE [NODE...]: copy the remote-Engram overlay into NODE:~/.cache/dsv41-serve/dsv41-offload/ (the
# serve container mounts ~/.cache/dsv41-serve at /root/.cache, so no launcher change is needed), verify the copies,
# and check that the running image's engram.py / engram_disk.cpp are the ones the overlay was made from.
# Coordinator-only (lane rules forbid lanes to write under ~/.cache/dsv41-serve). Run from the repo root on Controller:
#   offload/engram/spark/stage_overlay.sh spark sparkworker        (ssh aliases of dsv41.env HEAD / WORKER)
# Remove:  ssh NODE rm -rf ~/.cache/dsv41-serve/dsv41-offload   (A does not read it; JIT builds stay in torch_extensions)
set -euo pipefail
cd "$(dirname "$0")/../../.."
FILES=(manifest.json
       offload/transport/ofl.h offload/transport/ofl.cpp
       offload/engram/engram_proto.h offload/engram/engram_proto.cpp offload/engram/engram_client.h
       offload/engram/spark/engram_remote.cpp offload/engram/spark/engram.py offload/engram/spark/py/sitecustomize.py)
IMAGE_ENGRAM_PY=bf95ad9a61b0f5f211f00c21e4dc1fd46d2404fcfae54ab53690ec2e951b05e5
IMAGE_ENGRAM_DISK=3ec20a0111b15c05d7c41d33b45acce4392a46743057f9535e1919e21b45ef48
IMG_DIR=/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/common
local_sums=$(sha256sum "${FILES[@]}")
for n in "$@"; do
  echo "== $n"
  ssh -o BatchMode=yes "$n" "mkdir -p \$HOME/.cache/dsv41-serve/dsv41-offload"
  tar cf - "${FILES[@]}" | ssh -o BatchMode=yes "$n" "tar xf - -C \$HOME/.cache/dsv41-serve/dsv41-offload"
  remote_sums=$(ssh -o BatchMode=yes "$n" "cd \$HOME/.cache/dsv41-serve/dsv41-offload && sha256sum ${FILES[*]}")
  [ "$local_sums" = "$remote_sums" ] || { echo "$n: staged copies differ" >&2; exit 1; }
  echo "  staged ${#FILES[@]} files, sha256 match"
  img=$(ssh -o BatchMode=yes "$n" "docker run --rm --entrypoint sha256sum ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v5 $IMG_DIR/engram.py $IMG_DIR/engram_disk.cpp" | awk '{print $1}' | tr '\n' ' ')
  [ "$img" = "$IMAGE_ENGRAM_PY $IMAGE_ENGRAM_DISK " ] || { echo "$n: the image's engram files are not the ones the overlay was made from: $img" >&2; exit 1; }
  echo "  image engram.py / engram_disk.cpp match the overlay's base"
done
