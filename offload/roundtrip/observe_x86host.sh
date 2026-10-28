#!/usr/bin/env bash
# Passive x86host-only sampling. No requests, signals, affinity writes or CUDA contexts.
set -euo pipefail
source "$HOME/.config/ndm/paths.sh"
out=${1:?usage: observe_x86host.sh OUTPUT_DIR [SECONDS]}
seconds=${2:-45}
[[ "$seconds" =~ ^[1-9][0-9]*$ ]] && (( seconds <= 60 )) || exit 2
mkdir -p "$out"
exec >"$out/inventory.txt" 2>&1
date -u --iso-8601=seconds
hostname
free -g
uptime
lscpu
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE,MAXMHZ,MINMHZ
nvidia-smi -L
nvidia-smi topo -m
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
ps -eo pid,ppid,psr,pcpu,pmem,nlwp,stat,args --sort=-pcpu | head -n 45
mapfile -t workers < <(pgrep -f '^/home/operator/dsv41-offload/o2/venv/bin/python -m worker.server ')
(( ${#workers[@]} == 1 )) || { echo "Expected exactly one live worker, found ${#workers[@]}"; exit 3; }
worker=${workers[0]}
printf '%s\n' "$worker" >"$out/worker.pid"
capture_threads() {
  local task
  for task in /proc/"$worker"/task/*; do
    printf '%s ' "${task##*/}"
    cat "$task/schedstat"
  done
}
cat "/proc/$worker/status"
capture_threads >"$out/threads.before"
cat "$HOME/dsv41-offload/o2/run/worker.status.json" >"$out/worker-status.before.json"
cat "/proc/$worker/schedstat" >"$out/schedstat.before"
cat /proc/interrupts >"$out/interrupts.before"
cat /proc/softirqs >"$out/softirqs.before"
cat /proc/vmstat >"$out/vmstat.before"
cat /proc/meminfo >"$out/meminfo.before"
mapfile -t nics < <(ip -o -4 addr show | awk '$4 == "10.202.28.1/30" || $4 == "10.202.29.1/30" {print $2}')
(( ${#nics[@]} == 2 )) || { echo "Expected both x86host direct-link NICs"; exit 4; }
for nic in "${nics[@]}"; do
  printf '\nNIC %s\n' "$nic"
  readlink -f "/sys/class/net/$nic/device"
  cat "/sys/class/net/$nic/device/numa_node" "/sys/class/net/$nic/device/local_cpulist"
  for irq in /sys/class/net/"$nic"/device/msi_irqs/*; do
    n=${irq##*/}
    printf 'IRQ %s ' "$n"
    cat "/proc/irq/$n/smp_affinity_list"
  done
done
for policy in /sys/devices/system/cpu/cpufreq/policy*; do
  printf '%s ' "$policy"
  cat "$policy/scaling_governor" "$policy/scaling_cur_freq" | tr '\n' ' '
  printf '\n'
done
for state in /sys/devices/system/cpu/cpu0/cpuidle/state*; do
  printf '%s ' "$state"
  cat "$state/name" "$state/disable" "$state/latency" | tr '\n' ' '
  printf '\n'
done
cat /proc/pressure/{cpu,memory,io}
pidstat -u -r -w -t -p "$worker" 1 "$seconds" >"$out/worker.pidstat" &
p1=$!
pidstat -u -r -w -p ALL 1 "$seconds" >"$out/all.pidstat" &
p2=$!
mpstat -P ALL 1 "$seconds" >"$out/mpstat.txt" &
p3=$!
nvidia-smi dmon -s pucvmet -d 1 -c "$seconds" >"$out/gpu.dmon" 2>"$out/gpu.dmon.stderr" &
p4=$!
vmstat 1 "$seconds" >"$out/vmstat.txt" &
p5=$!
for job in "$p1" "$p2" "$p3" "$p4" "$p5"; do
  wait "$job"
done
cat "/proc/$worker/schedstat" >"$out/schedstat.after"
capture_threads >"$out/threads.after"
cat "$HOME/dsv41-offload/o2/run/worker.status.json" >"$out/worker-status.after.json"
cat /proc/interrupts >"$out/interrupts.after"
cat /proc/softirqs >"$out/softirqs.after"
cat /proc/vmstat >"$out/vmstat.after"
cat /proc/meminfo >"$out/meminfo.after"
cat /proc/pressure/{cpu,memory,io}
date -u --iso-8601=seconds
printf 'Completed passive sample\n'
