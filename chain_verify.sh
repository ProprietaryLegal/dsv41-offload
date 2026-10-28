#!/usr/bin/env bash
set -uo pipefail; cd ~/dsv41-deploy
while pgrep -f download.sh >/dev/null; do sleep 20; done
tail -3 logs/download.log
echo "== $(date -Is) verify head"; ./verify_hashes.py /home/rank0user/models/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark manifest.json; echo "head verify rc=$?"
while pgrep -f sync_worker.sh >/dev/null; do sleep 10; done
echo "== $(date -Is) sync pass 2"; ./sync_worker.sh
scp -q verify_hashes.py sparkworker:/tmp/verify_hashes.py
echo "== $(date -Is) verify worker"; ssh sparkworker 'python3 /tmp/verify_hashes.py /srv/dsv41/model'; echo "worker verify rc=$?"
echo "== $(date -Is) chain done"
