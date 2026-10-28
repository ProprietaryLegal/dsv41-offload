#!/bin/bash
# N1-NATIVE: fetch ONLY the official 1017 draft (mtp.*) shards + config/index/reference code, pinned revision, sha256 receipts
set -euo pipefail
REV=1bae0e1088993b9b82de08f59de5be5a34e42170
B=https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/$REV
cd /opt/dsv41/x86host-data/native-1017
for f in config.json model.safetensors.index.json; do curl -sfL --retry 5 -o $f "$B/$f"; done
declare -A SHA=([model-00044-of-00048.safetensors]=3e8ac315c2bcca3e83c5e6e25c5758a75c03732063a6ff0c1b0e24b9ff49f6e6 [model-00045-of-00048.safetensors]=9c73b27c4343236790008979d5e0b565f687aa64440031b9e096fed1ab8039c5 [model-00046-of-00048.safetensors]=1dccc19a3741972c661de734572dc7d4b899e6e99bfbdf3ae4e057c707e5b339)
for f in "${!SHA[@]}"; do
  [ -f "$f" ] && [ "$(sha256sum "$f" | cut -d" " -f1)" = "${SHA[$f]}" ] && { echo "have $f"; continue; }
  curl -sfL --retry 5 -C - -o "$f.part" "$B/$f"
  got=$(sha256sum "$f.part" | cut -d" " -f1)
  [ "$got" = "${SHA[$f]}" ] || { echo "SHA MISMATCH $f $got"; exit 1; }
  mv "$f.part" "$f"; echo "ok $f $got"
done
date -u +%FT%TZ > RECEIPT.txt
echo "repo deepseek-ai/DeepSeek-V4.1-Flash-1017 revision $REV" >> RECEIPT.txt
sha256sum config.json model.safetensors.index.json model-0004[4-6]-of-00048.safetensors >> RECEIPT.txt
echo FETCH-DONE
