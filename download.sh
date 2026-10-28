#!/usr/bin/env bash
# Pinned-revision wrapper around the author's download_model.sh logic (same files, same size checks).
set -euo pipefail
export PATH=$HOME/.venvs/hfcli/bin:$PATH
DIR=$HOME/models/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark
QREV=86f852533ec94d1a537fea41fd9c01d6b0f6eb50   # HeNryous quant, 2026-10-20
BREV=1bae0e1088993b9b82de08f59de5be5a34e42170   # deepseek-ai 1017 base
mkdir -p "$DIR"
echo "== $(date -Is) quant"; hf download HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark --revision $QREV --local-dir "$DIR"
echo "== $(date -Is) engram"; hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 model-00047-of-00048.safetensors model-00048-of-00048.safetensors --revision $BREV --local-dir "$DIR"
for f in model-00047-of-00048.safetensors:101535150936 model-00048-of-00048.safetensors:101537926640; do
  n=${f%%:*}; w=${f##*:}; g=$(stat -c %s "$DIR/$n"); [ "$g" = "$w" ] || { echo "SIZE MISMATCH $n $g != $w"; exit 1; }; done
echo "== $(date -Is) DONE $(ls "$DIR" | wc -l) files $(du -sh "$DIR" | cut -f1)"
