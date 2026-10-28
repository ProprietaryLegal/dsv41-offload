#!/usr/bin/env bash
# Copy the model folder to the worker's local NVMe over the direct CX-7 link (10.202.26.1), 6 files in parallel.
set -euo pipefail
SRC=/home/rank0user/models/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark
DST=/srv/dsv41/model
cd "$SRC"
find . -path ./.cache -prune -o -type f -print | sed 's#^\./##' | sort |
  xargs -P 6 -I{} rsync -a --relative -e "ssh -c aes128-gcm@openssh.com -o Compression=no" "./{}" sparkworker:"$DST/"
echo "== $(date -Is) sync pass done: $(ssh sparkworker "find $DST -type f | wc -l") files on worker, $(ssh sparkworker "du -sh $DST | cut -f1")"
