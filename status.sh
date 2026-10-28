#!/usr/bin/env bash
# Container state, memory, GPU temp/clock on both nodes + API health.
cd "$(dirname "$0")"; set -a; . ./dsv41.env; set +a
for n in $HEAD $WORKER; do
  echo "== $n: $(ssh $n "docker ps -a --filter name=dsv41-serve --format '{{.Status}}'") | $(ssh $n "free -g | awk '/Mem:/{print \"avail\",\$7,\"GiB\"}'") | $(ssh $n "nvidia-smi --query-gpu=temperature.gpu,clocks.gr,power.draw,utilization.gpu --format=csv,noheader")"
done
curl -s -m 5 http://localhost:$PORT/v1/models | python3 -c 'import sys,json; print("API OK:", [m["id"] for m in json.load(sys.stdin)["data"]])' 2>/dev/null || echo "API not responding on :$PORT"
curl -s -m 5 http://localhost:$PORT/metrics | grep -E '^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|spec_decode_num_accepted_tokens_total|spec_decode_num_draft_tokens_total)' | sed 's/{[^}]*}//' 2>/dev/null
