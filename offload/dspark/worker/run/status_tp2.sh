#!/bin/bash
# TP = 2 worker status (run ON x86host): unit state, both GPUs, NVLink, the listener, the last status snapshot (5 s).
UNIT=dsv41-dspark-worker-tp2.service
en=$(systemctl --user is-enabled $UNIT 2>/dev/null); echo "unit: $(systemctl --user is-active $UNIT) (${en:-not-installed})"
for u in GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93 GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc; do
  nvidia-smi --id=$u --query-gpu=uuid,memory.used,memory.total,utilization.gpu,power.draw,power.limit,temperature.gpu --format=csv,noheader
done
nvidia-smi nvlink -s -i GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc | tail -4
ss -ltnp 2>/dev/null | grep -E "10.202.2[89].1:186[2-9][0-9]" || echo "no listener on 10.202.28.1/10.202.29.1:1862x"
f=~/dsv41-offload/e/run/worker.status.json
[ -f "$f" ] && { echo "status ($(stat -c %y "$f" | cut -c1-19)):"; python3 -m json.tool "$f"; } || echo "no status file"
[ -f ~/dsv41-offload/e/lib/PINNED.txt ] && { echo "transport:"; cat ~/dsv41-offload/e/lib/PINNED.txt; } || echo "transport not pinned"
