#!/usr/bin/env python3
"""Read-only S1 inventory; SSH uses management routes, never model file reads."""
import concurrent.futures
import json
import pathlib
import subprocess
import time

COMMAND = r'''
hostname
free -g
ip -details -br link
ip -br address
ip route show table all
lspci -tv
sudo -n lspci -vv -d 15b3:
ibv_devinfo -v
for d in /sys/class/infiniband/*; do
 echo "DEVICE $d PCI $(readlink -f "$d/device")"
 cat "$d/device/numa_node"
 for p in "$d"/ports/*; do
  for g in "$p"/gids/*; do
   v=$(cat "$g"); case "$v" in 0000:0000:0000:0000:0000:0000:0000:0000) continue;; esac
   i=${g##*/}; echo "GID $i $v TYPE $(cat "$p/gid_attrs/types/$i") NETDEV $(cat "$p/gid_attrs/ndevs/$i")"
  done
  for c in "$p"/counters/*; do echo "$c $(cat "$c")"; done
 done
done
for n in /sys/class/net/*; do
 i=${n##*/}; case "$i" in enp*|enP2*) ethtool "$i"; ethtool -i "$i"; ethtool -S "$i";; esac
done
command -v ib_send_lat ib_write_lat ib_read_lat ib_write_bw ib_read_bw iperf3
ib_write_bw --version
ib_write_bw --help
'''


def capture(host: str) -> str:
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', host, 'bash', '-s'], input=COMMAND, text=True, capture_output=True, timeout=45)
    row = dict(host=host, utc=time.strftime('%FT%TZ', time.gmtime()), command=COMMAND, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    out = pathlib.Path('offload/results/s1') / f'inventory-{host}.jsonl'
    out.write_text(json.dumps(row) + '\n')
    return f'{host}: rc={p.returncode}, {len(p.stdout)} bytes, {p.stderr[:300]}'


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        print('\n'.join(pool.map(capture, ['x86host', 'sparkhead', 'sparkworker'])))
