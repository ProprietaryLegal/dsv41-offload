#!/bin/bash
# N1-NATIVE fetch 2: official shared draft embed (shard 2 = embed.weight only) and head (shard 43 = head.weight +
# norm.weight), pinned revision, sha256-verified against the LFS oids. Needed for the native BF16 draft-head arm and to
# prove the worker embed is byte-identical to the official tensor.
set -euo pipefail
REV=1bae0e1088993b9b82de08f59de5be5a34e42170
B=https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/$REV
cd /opt/dsv41/x86host-data/native-1017
declare -A SHA=([model-00002-of-00048.safetensors]=43616eebd88dd4f1f1b642772356fc160478d1d9d04edd240431e7d065d2eda7 [model-00043-of-00048.safetensors]=4ae12b941c3facf8616ce62dd45bc6868076c2aeb82106f98123ea6c5bb64182)
for f in "${!SHA[@]}"; do
  curl -sfL --retry 5 -C - -o "$f.part" "$B/$f"
  got=$(sha256sum "$f.part" | cut -d" " -f1)
  [ "$got" = "${SHA[$f]}" ] || { echo "SHA MISMATCH $f $got" >&2; exit 1; }
  mv "$f.part" "$f"; echo "ok $f $got"
done
{ date -u +%FT%TZ; echo "fetch2 (shared embed/head) revision $REV"; sha256sum model-00002-of-00048.safetensors model-00043-of-00048.safetensors; } >> RECEIPT.txt
echo FETCH2-DONE
