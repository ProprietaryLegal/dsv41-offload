#!/usr/bin/env bash
# prebuild_jit.sh NODE [NODE...]: build (or just load, when already built) the remote-Engram JIT extension on each node
# before the model loads, exactly as INTEGRATION.md step 2; the module name hashes the staged sources, so a re-stage of
# new sources builds once, then later starts only load it. Needs the overlay staged (stage_overlay.sh) and the serve
# stopped (building next to loaded weights can run a GB10 node out of memory). Fails loudly on any node.
set -euo pipefail
IMAGE=${IMAGE:-ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v5}
pids=()
for n in "$@"; do
  ssh -o BatchMode=yes "$n" "docker run --rm --gpus all --device /dev/infiniband -v \$HOME/.cache/dsv41-serve:/root/.cache \
    -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions -e MAX_JOBS=4 \
    -e PYTHONPATH=/root/.cache/dsv41-offload/offload/engram/spark/py \
    -e DSV41_ENGRAM_OVERLAY=/root/.cache/dsv41-offload/offload/engram/spark/engram.py \
    -e VLLM_ENGRAM_REMOTE_ROOT=/root/.cache/dsv41-offload \
    --entrypoint python3 $IMAGE -c 'import vllm.models.deepseek_v41.common.engram as e; print(\"$n overlay:\", e.__file__); print(\"$n ext:\", e._load_engram_remote_ext().__file__)'" &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ $rc = 0 ] || { echo "remote-Engram JIT prebuild failed on at least one node" >&2; exit 1; }
