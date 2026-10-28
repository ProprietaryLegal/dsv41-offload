#!/usr/bin/env python3
"""Regenerate LINKS.md from last attempts in raw S1 JSONL; never reuse failed results."""
from __future__ import annotations
import json
from pathlib import Path
import re
from typing import cast
from s1_links import RunResult, SIZES

ROOT=Path(__file__).resolve().parents[1]/'results/s1'


def numeric_line(output: str, size: int) -> list[float]:
    matches=re.findall(r'^\s*'+str(size)+r'\s+\d+\s+.*$',output,re.M)
    if not matches: raise ValueError(f'No numeric result for size {size}')
    return [float(v) for v in matches[-1].split()]


def client_rows(row: RunResult) -> list[tuple[str,list[float]]]:
    return [(p['host'],numeric_line(p['output'],row['size'])) for p in row['processes'] if p['host']!='x86host']


def tcp_rows(row: RunResult) -> list[tuple[str,dict[str,float]]]:
    result: list[tuple[str,dict[str,float]]]=[]
    for p in row['processes']:
        if p['host']=='x86host':continue
        for line in p['output'].splitlines():
            if line.startswith('{'):
                data=cast(dict[str,float],json.loads(line))
                if 'payload_gbps' in data:result.append((p['host'],data))
    if not result:raise ValueError('No TCP result')
    return result


def iperf_rates(row: RunResult) -> list[tuple[str,float]]:
    result: list[tuple[str,float]]=[]
    for process in row['processes']:
        if process['host']=='x86host':continue
        data=cast(dict[str,dict[str,dict[str,float]]],json.loads(process['output']))
        result.append((process['host'],data['end']['sum_received']['bits_per_second']/1e9))
    return result


def main() -> None:
    rows: list[tuple[str,RunResult]]=[]
    failed: list[str]=[]
    for path in sorted(ROOT.glob('*.jsonl')):
        if path.name.startswith(('inventory-','tp-counters-','summary')):continue
        attempts=[cast(RunResult,json.loads(line))for line in path.read_text().splitlines()]
        if not attempts:
            failed.append(f'- `{path.name}` is empty: no completed result; rate remains pending.')
            continue
        row=attempts[-1]
        if 'tool' not in row:continue
        for i,attempt in enumerate(attempts,1):
            if not attempt['ok']:
                errors=[line.strip() for p in attempt['processes'] for line in p['output'].splitlines() if any(t in line for t in ['Failed to','Couldn\'t','Unable to','Incompatibility'])]
                failed.append(f'- `{path.name}` attempt {i}: '+('; '.join([attempt['error'],*errors]).strip('; ') or 'Peer exited nonzero; inspect raw output')+(' (later recovered)' if row['ok'] else ''))
        if row['ok']:rows.append((path.stem,row))
    index={name:row for name,row in rows}
    lines=['# S1 measured links — 2026-10-22','',
           'Both cables negotiate **40 Gb/s**. All measurements use host memory and direct-bound sockets. These short idle-only bursts are link diagnostics; inference interference remains coordinator-owned. K/M sizes use binary multipliers and all tables list exact bytes. perftest 6.20 is recorded in the inventory; no dependencies were installed.','',
           '| Host/path | HCA / PCIe | Address / MTU | Firmware |',
           '|---|---|---|---|',
           '| x86host → A | rocep130s0f1 / 82:00.1, shared Gen4 x16 CX-5 Ex | 10.202.28.1 / 9000 | 16.32.1010 |',
           '| x86host → B | rocep130s0f0 / 82:00.0, same adapter | 10.202.29.1 / 9000 | 16.32.1010 |',
           '| Spark A/B direct | rocep1s0f0 / 0000:01:00.0, Gen5 x4 | .28.2 / .29.2, 9000 | 28.45.4028 |',
           '| Spark A/B twin direct | roceP2p1s0f0 / 0002:01:00.0, second Gen5 x4 | IPv6 link-local only, **1500** | 28.45.4028 |',
           '| Spark A/B TP rail 1 | rocep1s0f1 / 0000:01:00.1, shares first uplink with direct f0 | 10.202.26.2 / .1, 9000, 200G | 28.45.4028 |',
           '| Spark A/B TP rail 2 | roceP2p1s0f1 / 0002:01:00.1, shares second uplink with twin f0 | 10.202.27.2 / .1, 9000, 200G | 28.45.4028 |','',
           'All HCA NUMA nodes report -1 (no specific NUMA association). Normal direct and TP RoCEv2 use discovered IPv4 GID3; twin link-local probes use X86host GID1 and Spark GID1 or configured GID3 with --ipv6. Each twin/netdev f0 pair shares physical port p0 and the same switch ID. X86host direct interfaces have addrgenmode none, no configured IPv6 address or link-local route despite device GID1. No configuration changes were attempted.','',
           'Initial serve environment: EXL3_RDMA_DEVICES=rocep1s0f1; NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1; TP/Gloo/NCCL socket interface enp1s0f1np1. Initial counters corroborate traffic on BOTH f1 rails: first rail ~364 GB, second ~106 GB transmitted on each Spark (port_xmit_data units ×4). Initial f0 RDMA data counters were zero. Counters identify used rails; no claim of a live TP throughput measurement is made.','']
    for link in ['b','a']:
        lines += [f'## Spark {link.upper()} — RDMA','',
                  'Latency is perftest average µs as reported; send/write are ping-pong-derived, read is completion latency. TCP below reports full application RTT and is not directly the same measure. Bandwidth is decimal Gb/s; write/read both directions are measured.','',
                  '| Bytes | send µs | write µs | read µs | write Spark→x86host | write x86host→Spark | read x86host→Spark | read Spark→x86host |',
                  '|---:|---:|---:|---:|---:|---:|---:|---:|']
        for size in SIZES:
            values=[]
            for tool,rev in [('ib_send_lat',False),('ib_write_lat',False),('ib_read_lat',False),('ib_write_bw',False),('ib_write_bw',True),('ib_read_bw',False),('ib_read_bw',True)]:
                measurement=index.get(f'{link}-{tool}-{size}-{"reverse" if rev else "forward"}')
                values.append(f'{client_rows(measurement)[0][1][3 if tool.endswith("_bw") else 5]:.2f}' if measurement else 'pending')
            lines.append('| '+str(size)+' | '+' | '.join(values)+' |')
        lines += ['',f'## Spark {link.upper()} — TCP','',
                  '| Bytes | echo RTT mean µs | echo RTT p99 µs | Spark→x86host Gb/s | x86host→Spark Gb/s |',
                  '|---:|---:|---:|---:|---:|']
        for size in SIZES:
            echo=index.get(f'{link}-tcp-{size}-echo')
            a=index.get(f'{link}-tcp-{size}-forward');b=index.get(f'{link}-tcp-{size}-reverse')
            d=tcp_rows(echo)[0][1] if echo else {}
            values=[f'{d[k]:.2f}' if k in d else 'pending'for k in ['rtt_mean_us','rtt_p99_us']]
            values +=[f'{tcp_rows(r)[0][1]["payload_gbps"]:.2f}'if r else 'pending'for r in [a,b]]
            lines.append('| '+str(size)+' | '+' | '.join(values)+' |')
    lines += ['','## Both links simultaneously','',
              'RDMA uses overlapping three-second duration windows per link, TX depth 16, CQ moderation 1; TCP clients synchronize application start against a shared wall-clock deadline. Values below sum per-link payload rates; individual rates remain in raw output.','',
              '| Bytes | write Spark→x86host aggregate | write x86host→Spark aggregate | read x86host→Spark aggregate | read Spark→x86host aggregate | TCP Spark→x86host aggregate | TCP x86host→Spark aggregate |',
              '|---:|---:|---:|---:|---:|---:|---:|']
    for size in SIZES:
        values=[]
        for tool,rev in [('ib_write_bw',False),('ib_write_bw',True),('ib_read_bw',False),('ib_read_bw',True)]:
            measurement=index.get(f'both-{tool}-{size}-{rev}')
            values.append(f'{sum(v[3]for _,v in client_rows(measurement)):.2f}'if measurement else 'pending')
        for mode in ['forward','reverse']:
            measurement=index.get(f'both-tcp-{size}-{mode}')
            values.append(f'{sum(v["payload_gbps"]for _,v in tcp_rows(measurement)):.2f}'if measurement else 'pending')
        lines.append('| '+str(size)+' | '+' | '.join(values)+' |')
    lines += ['','## iperf3 cross-check','',
              'Single TCP stream, default 128 KiB write buffer, three seconds per link; receiver payload Gb/s.','',
              '| Link | Spark→x86host | x86host→Spark |','|---|---:|---:|']
    for link in ['b','a','both']:
        values=[]
        for reverse in [False,True]:
            measurement=index.get(f'{link}-iperf3-{reverse}')
            values.append(f'{sum(rate for _,rate in iperf_rates(measurement)):.2f}'if measurement else 'pending')
        lines.append('| '+link+' | '+' | '.join(values)+' |')
    lines += ['','Passive counter sample during an existing production request (lane generated no model traffic): rocep1s0f1 transmitted ~114 MB/s on each Spark; roceP2p1s0f1 and both f0 HCAs were unchanged over that one-second sample. The second TP rail had historical traffic but was idle in this decode sample. See tp-counters-*.jsonl.']
    lines += ['','## Failures and limits','']+failed+['',
              'The twin-function link-local path failed to transition QPs to RTR in the unmodified network configuration; no twin bandwidth result is asserted. Optional ethtool netlink queries returned Operation not permitted; base inventory and counters are present. perftest --help returns 1 by design. CPU-frequency warnings remain in raw output; clock governors were not changed.','',
              'Production request gauges were checked before each pair and throughout each burst (poll sleep 150 ms plus HTTP time, timeout 2 s); missing/nonzero gauges abort and close all owned peers. Controller invocations serialize through an advisory file lock. Short iteration-mode single-link tests include warmup/CPU scheduling effects; simultaneous duration-mode bandwidth is a separate measurement regime. TCP includes Python/socket overhead and a final payload check; p99 is based on 200 or 8 samples, so large-size p99 is diagnostic only.','',
              'S2 gate: no Spark A measurement starts before x86host /opt/dsv41/x86host-data/engram/ENGRAM_COPY_DONE exists. No TP workload was driven by S1. Coordinator must measure inference interference using the supplied load script.','',
              '## Reproduce','',
              '```bash\npython3 offload/bench/s1_links.py --link b --resume\npython3 offload/bench/s1_links.py --link a --resume\npython3 offload/bench/s1_links.py --link both --simultaneous\npython3 offload/bench/s1_links.py --link both --tcp --simultaneous\npython3 offload/bench/s1_report.py\noffload/bench/link_load.sh --link both --seconds 10 --rate-gbps 5 --size 77824\n```','',
              'To measure inference interference, the coordinator launches load while idle with --during-benchmark, then begins its own benchmark after connections establish. That explicit flag permits subsequent serve activity; default mode stops when requests appear. The coordinator can select --metrics-url http://HEAD_IP:8001/metrics when benchmarking the experimental serve. S1 matrix commands retain the required production port 8000 gate. Rate is per link. --direction chooses x86host-to-spark (default) or spark-to-x86host. Output includes commands, exact errors, requested and achieved rates. No installation or serve restart required.']
    (ROOT/'LINKS.md').write_text('\n'.join(lines)+'\n')
    print(f'Wrote LINKS.md from {len(rows)} successful latest cases; {len(failed)} retained failed attempts')

if __name__=='__main__':main()
