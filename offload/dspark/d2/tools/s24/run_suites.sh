#!/usr/bin/env bash
# S24 item 6: run every CPU test directory of an exported tree (git archive) and record red node ids.
# usage: run_suites.sh TREE_ROOT OUT_PREFIX     (CUDA hidden; scratch under the campaign dir)
set -uo pipefail
tree=$1 out=$2
dirs=(offload/bench offload/dspark/d2/tests offload/dspark/d2/train/tests offload/dspark/depth
      offload/dspark/kernels/fp8dense offload/dspark/kernels/fusedmoe offload/dspark/kernels/mxfp4
      offload/dspark/native/tests offload/dspark/native/tools/n2 offload/dspark/native/tools/n5
      offload/dspark/probabilistic/tests offload/dspark/spark/tests offload/dspark/worker/tests offload/engram/tests
      offload/evidence offload/gb10_budget offload/launch/tests offload/launch/windows/tests offload/operations
      offload/prefill_maxlen offload/prefixstore/remote offload/prefixstore/tests offload/promote/k1 offload/roundtrip
      offload/spark_leads/tests offload/transport offload/upstream_adopt/tests)
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
: > "$out.summary"; : > "$out.red"
for d in "${dirs[@]}"; do
  [[ -d "$tree/$d" ]] || { echo "$d ABSENT" >> "$out.summary"; continue; }
  bt=$(mktemp -d "${TMPDIR:?}/s24-bt-XXXXXX")
  log=$(cd "$tree" && PYTHONPATH="$tree" timeout 1800 python3 -m pytest "$d" -q -p no:cacheprovider -rfE \
        --basetemp="$bt" -W ignore::UserWarning 2>&1)
  rc=$?
  rm -rf "$bt"
  echo "$d rc=$rc $(printf '%s\n' "$log" | grep -E '^[0-9]+ (passed|failed|error)|^=+ .*(passed|failed|error|skipped).* =+$|no tests ran' | tail -1)" >> "$out.summary"
  printf '%s\n' "$log" | grep -E '^(FAILED|ERROR) ' | sed -E 's/ - .*$//' >> "$out.red"
done
sort -o "$out.red" "$out.red"
