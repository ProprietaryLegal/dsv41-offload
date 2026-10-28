#!/usr/bin/env python3
"""Resumable S1 remaining measurements. Controller-only, run detached in tmux.

Waits for idle and the S2 marker; aborts on a real test error. Writes progress
only to the campaign's authorized S1 file. Does not restart or benchmark serve.
"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import time
from s1_links import LINKS, copy_ready, idle, rdma_pair, run_pairs, record, matrix

PROGRESS=Path('/opt/dsv41/campaigns/DSV41-OFFLOAD/S1.progress.md')


def note(message: str) -> None:
    line=time.strftime('%FT%TZ',time.gmtime())+' '+message
    print(line,flush=True)
    with PROGRESS.open('a')as f:f.write('\n- '+line+'\n')


def wait_ready(needs_a: bool) -> None:
    last=''
    last_note=0.0
    while True:
        try:
            idle()
            if needs_a:copy_ready()
            return
        except (RuntimeError,OSError,TimeoutError,subprocess.TimeoutExpired)as exc:
            message=str(exc)
            if message!=last or time.monotonic()-last_note>=60:
                note('Waiting: '+message)
                last=message
                last_note=time.monotonic()
            time.sleep(.5)


def probe_twin() -> None:
    while True:
        wait_ready(False)
        try:
            row=run_pairs([rdma_pair(LINKS['b'],'ib_send_lat',64,twin=True,twin_gid=3)])
        except RuntimeError as exc:
            if 'Serve busy:' in str(exc):continue
            raise
        row.update({'link':'b','tool':'ib_send_lat','size':64,'twin':True,'reverse':False})
        record(Path('offload/results/s1'),'b-twin-gid3-ib_send_lat-64-forward',row)
        if 'Serve busy:' in row['error']:continue
        note('Twin configured IPv6 GID3 probe: '+str(row['ok']))
        if row['ok']:matrix(['b'],Path('offload/results/s1'),True,True,3)
        return


def main() -> None:
    stages=[['--link','b','--iperf'],['--link','a'],['--link','a','--tcp'],['--link','a','--iperf'],['--link','both','--simultaneous'],['--link','both','--tcp','--simultaneous'],['--link','both','--iperf','--simultaneous']]
    for args in stages:
        while True:
            wait_ready(args[1]!='b')
            note('Starting '+ ' '.join(args))
            p=subprocess.run([sys.executable,'offload/bench/s1_links.py',*args,'--resume'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            print(p.stdout,flush=True)
            if p.returncode==0:
                note('Completed '+ ' '.join(args));break
            if any(marker in p.stdout for marker in ['Serve busy:','MetricsUnavailable:']):
                note('Traffic stopped because idle could not be verified; incomplete cases will resume.');continue
            raise RuntimeError('Measurement failed; inspect offload/results/s1/.artifacts/S1.finish.log: '+p.stdout[-2000:])
    while True:
        wait_ready(True)
        p=subprocess.run(['offload/bench/link_load.sh','--link','both','--seconds','3','--rate-gbps','1','--size','77824'],text=True,capture_output=True)
        out=Path('offload/results/s1/load-both-1gbps.jsonl')
        if p.stdout.strip():
            with out.open('a')as f:f.write(p.stdout)
        if p.returncode:print(p.stderr,flush=True)
        if not p.returncode:
            result=json.loads(p.stdout)
            note('Both-link idle load validation complete: '+str(result['achieved_gbps']))
            break
        if any(marker in p.stdout+p.stderr for marker in ['Serve busy:','MetricsUnavailable:']):
            note('Load validation stopped on idle guard; retrying after idle returns.')
            continue
        raise RuntimeError(p.stderr+' '+p.stdout)
    subprocess.run([sys.executable,'offload/bench/s1_report.py'],check=True)
    note('All remaining measurement stages complete; final verification/report pending.')

if __name__=='__main__':main()
