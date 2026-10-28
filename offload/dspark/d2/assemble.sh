#!/bin/bash
# LOCAL offline procedure. No SSH, HTTP, service, capture or generator control.
set -euo pipefail
execute=0
if [[ ${1:-} == --execute ]]; then
  execute=1
  shift
fi
if [[ ${1:-} == --help || $# == 0 ]]; then
  printf '%s\n' 'Usage: assemble.sh [--execute] --root COPIED_TREE --image-smoke COPIED_R6_JSON --smoke-sha256 SHA256 --prompt-manifest JSON --sequence-context JSON [--path-map OLD_ABSOLUTE=ROOT_RELATIVE]...'
  printf '%s\n' 'Dry by default. Execute requires D2_COORDINATOR=Deepseek/coordinator. Run locally on x86host only; see ASSEMBLE.md.'
  exit 0
fi
if (( execute )) && [[ ${D2_COORDINATOR:-} != Deepseek/coordinator ]]; then
  printf '%s\n' 'REFUSED: --execute requires D2_COORDINATOR=Deepseek/coordinator' >&2
  exit 2
fi
# Exact existing x86host interpreter, per stage_replay_inputs.sh and g1.py.
python=${D2_ASSEMBLY_PYTHON:-$HOME/dsv41-offload/o2/venv/bin/python}
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHONPATH="$repo/offload/dspark:$repo"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
command=("$python" -m d2.assemble "$@")
if (( ! execute )); then
  printf 'DRY CPU: '
  printf '%q ' "${command[@]}"
  printf '\n%s\n' 'Order: verify COPY_RECEIPT -> export -> prepare -> build shards -> G1(a) -> aggregate -> ASSEMBLED.json. No commands executed.'
  exit 0
fi
umask 077
exec "${command[@]}"
