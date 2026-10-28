#!/bin/bash
# Worker status (run ON x86host): unit state, the GPU, the listener, the last status snapshot (JSON, every 5 s).
UNIT=dsv41-dspark-worker.service
en=$(systemctl --user is-enabled $UNIT 2>/dev/null); echo "unit: $(systemctl --user is-active $UNIT) (${en:-not-installed})"
nvidia-smi --id=GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93 --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
ss -ltnp 2>/dev/null | grep -E "10.202.2[89].1:186[2-9][0-9]" || echo "no listener on 10.202.28.1/10.202.29.1:1862x"
f=~/dsv41-offload/o2/run/worker.status.json
[ -f "$f" ] && { echo "status ($(stat -c %y "$f" | cut -c1-19)):"; python3 -m json.tool "$f"; } || echo "no status file"
[ -f ~/dsv41-offload/o2/lib/PINNED.txt ] && { echo "transport:"; cat ~/dsv41-offload/o2/lib/PINNED.txt; } || echo "transport not pinned"
