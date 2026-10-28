#!/usr/bin/env bash
# MC1: every CPU check of the matter cache branch, in one command (Controller; no host contact, no GPU).
#   bash offload/prefixstore/check_all.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/home/operator/miniconda3/bin/python3}
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
"$PY" offload/prefixstore/build_overlay.py
"$PY" offload/prefixstore/make_configs.py --check
"$PY" offload/prefixstore/verify_overlay.py tree /opt/dsv41/data/src-image --base
"$PY" offload/prefixstore/compose_audit.py >/dev/null
bash offload/kvpack/test_stage_overlay.sh | tail -1
"$PY" -m pytest -q -p no:cacheprovider offload/prefixstore/tests offload/prefixstore/matter/tests
"$PY" -m mypy --strict --follow-imports=skip offload/prefixstore/{build_overlay,make_configs,verify_overlay,restart_probe,compose_audit,blockhash_compare}.py
"$PY" -m mypy --strict --follow-imports=skip --ignore-missing-imports offload/prefixstore/matter/*.py \
  offload/prefixstore/matter/tests/*.py offload/prefixstore/tests/{test_matter_gate,test_compose_audit,test_vllm_cases,test_blockhash_compare,vllm_tree}.py
"$PY" -m mypy --strict --follow-imports=silent --ignore-missing-imports offload/prefixstore/tests/vllm_cases/*.py
(cd offload/prefixstore/overlay && "$PY" -m mypy --strict --follow-imports=skip --ignore-missing-imports \
  --explicit-package-bases --namespace-packages $(cd .. && "$PY" -c 'import json; m=json.load(open("MANIFEST.json")); print(" ".join(e["path"] for e in m["new_files"]))'))
echo "MC1 check_all: PASS"
