#!/usr/bin/env python3
"""Coordinator load generator. Starts idle; --during-benchmark permits later traffic.

Example: link_load.sh --link both --seconds 10 --rate-gbps 5 --size 77824
Rate is per link, decimal Gbit/s. All listener sockets bind X86host direct IPs.
Read/Writes are host-memory RC verbs; no GPUDirect claim. No model execution.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from s1_links import LINKS, METRICS, copy_ready, rdma_pair, run_pairs


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--link',choices=['a','b','both'],required=True)
    p.add_argument('--seconds',type=int,default=5)
    p.add_argument('--rate-gbps',type=float,required=True)
    p.add_argument('--size',type=int,default=77824)
    p.add_argument('--direction',choices=['x86host-to-spark','spark-to-x86host'],default='x86host-to-spark')
    p.add_argument('--metrics-url',choices=[METRICS,METRICS.replace(':8000/',':8001/')],default=METRICS,help='Coordinator only: choose the experimental port 8001 gate when benchmarking that serve; default production port 8000')
    p.add_argument('--during-benchmark',action='store_true',help='Coordinator only: allow requests AFTER idle startup check; default stops on any request')
    a=p.parse_args()
    if not (1<=a.seconds<=60 and 0<a.rate_gbps<=40 and 1<=a.size<=67108864):
        p.error('Require seconds 1..60, rate-gbps >0..40, size 1..67108864')
    links=['a','b'] if a.link=='both' else [a.link]
    if 'a' in links: copy_ready()
    row=run_pairs([rdma_pair(LINKS[n],'ib_write_bw',a.size,a.direction=='x86host-to-spark',duration=a.seconds,rate=a.rate_gbps) for n in links],allow_busy=a.during_benchmark,deadline=a.seconds+15,metrics_url=a.metrics_url)
    achieved: dict[str,float] = {}
    for process in row['processes']:
        # Client reports throughput even when server sends (--reversed).
        if process['host']=='x86host': continue
        lines=re.findall(r'^\s*'+str(a.size)+r'\s+\d+\s+[\d.]+\s+([\d.]+)\s+[\d.]+\s*$',process['output'],re.M)
        if lines: achieved[process['host']]=float(lines[-1])
    row.update({'link': a.link, 'size': a.size, 'seconds_requested': a.seconds, 'rate_gbps_per_link': a.rate_gbps, 'direction': a.direction, 'allow_busy': a.during_benchmark, 'achieved_gbps': achieved})
    if len(achieved)!=len(links):
        row['ok']=False
        row['error']=str(row['error'])+' Missing measured bandwidth; inspect process outputs.'
    print(json.dumps(row),flush=True)
    if not row['ok']: sys.exit(1)

if __name__=='__main__': main()
