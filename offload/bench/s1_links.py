#!/usr/bin/env python3
"""Bounded, idle-gated link measurements. No installation or remote file writes.

Remote supervisors kill only their own process group on SSH stdin EOF. Closing
stdin, including controller exit, stops traffic. Raw commands/output are JSONL.
"""
from __future__ import annotations
import argparse
import base64
import fcntl
from functools import wraps
from typing import Callable, ParamSpec, TypeVar, TypedDict, NotRequired
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.request

SIZES = [64, 4096, 65536, 77824, 1048576, 13631488, 67108864]
METRICS = 'http://HEAD_IP:8000/metrics'
REMOTE = '''import json,os,select,signal,subprocess,sys,time
args=json.loads(sys.argv[1]); p=subprocess.Popen(args,start_new_session=True)
try:
 end=time.monotonic()+float(sys.argv[2])
 while p.poll() is None:
  if time.monotonic()>end: raise TimeoutError("remote deadline")
  if select.select([sys.stdin],[],[],0.05)[0] and not os.read(0,4096): break
finally:
 if p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try: p.wait(timeout=1)
  except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGKILL);p.wait()
sys.exit(p.returncode)
'''

class ProcessResult(TypedDict):
    host: str
    command: list[str]
    returncode: int
    output: str


class RunResult(TypedDict):
    utc: str
    seconds: float
    idle_before: dict[str, float]
    error: str
    processes: list[ProcessResult]
    ok: bool
    link: NotRequired[str]
    tool: NotRequired[str]
    size: NotRequired[int]
    reverse: NotRequired[bool]
    twin: NotRequired[bool]
    mode: NotRequired[str]
    seconds_requested: NotRequired[int]
    rate_gbps_per_link: NotRequired[float]
    direction: NotRequired[str]
    allow_busy: NotRequired[bool]
    achieved_gbps: NotRequired[dict[str, float]]
    metrics_url: NotRequired[str]


@dataclass(frozen=True)
class Link:
    name: str
    host: str
    x86host_ip: str
    spark_ip: str
    hca: str
    port: int

LINKS = {'a': Link('a','sparkhead','10.202.28.1','10.202.28.2','rocep130s0f1',19151),
         'b': Link('b','sparkworker','10.202.29.1','10.202.29.2','rocep130s0f0',19152)}


class MetricsUnavailable(RuntimeError):
    """The idle precondition cannot be verified; all traffic must stop."""


def idle(url: str = METRICS) -> dict[str, float]:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            body = response.read().decode()
    except OSError as exc:
        raise MetricsUnavailable(f'Metrics unavailable: {type(exc).__name__}: {exc}') from exc
    values: dict[str, float] = {}
    for name in ['num_requests_running', 'num_requests_waiting']:
        samples = re.findall(r'^vllm:' + name + r'(?:\{[^\n]*?\})?\s+([\d.eE+-]+)\s*$', body, re.M)
        if not samples:
            raise MetricsUnavailable(f'Missing metric {name}; refusing traffic')
        nums = [float(v) for v in samples]
        if any(not math.isfinite(v) or v != 0 for v in nums):
            raise RuntimeError(f'Serve busy: {name}={nums}; stopping traffic')
        values[name] = sum(nums)
    return values


class Remote:
    def __init__(self, host: str, args: list[str], deadline: float = 25):
        self.host, self.args = host, args
        artifacts=Path(__file__).resolve().parents[1]/'results/s1/.artifacts'
        artifacts.mkdir(parents=True,exist_ok=True)
        self.output = tempfile.TemporaryFile(mode='w+t',dir=artifacts)
        code = "exec(__import__('base64').b64decode(" + repr(base64.b64encode(REMOTE.encode()).decode()) + '))'
        command = shlex.join(['python3', '-u', '-c', code, json.dumps(args), str(deadline)])
        self.process = subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',host,command], stdin=subprocess.PIPE, stdout=self.output, stderr=subprocess.STDOUT, text=True)

    def close(self) -> ProcessResult:
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=4)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=4)
        self.output.seek(0)
        out = self.output.read()
        self.output.close()
        assert self.process.returncode is not None
        return dict(host=self.host, command=self.args, returncode=self.process.returncode, output=out)


def copy_ready() -> None:
    p = subprocess.run(['ssh','-o','BatchMode=yes','x86host','test -f /opt/dsv41/x86host-data/engram/ENGRAM_COPY_DONE'], timeout=8)
    if p.returncode:
        raise RuntimeError('Spark A gated: x86host ENGRAM_COPY_DONE does not exist')
    receipt=Path(__file__).resolve().parents[1]/'results/s1/s2-copy-gate.jsonl'
    if not receipt.exists():
        code='import json,pathlib; p=pathlib.Path("/opt/dsv41/x86host-data/engram/ENGRAM_COPY_DONE"); print(json.dumps(dict(path=str(p),mtime=p.stat().st_mtime,contents=p.read_text())))'
        verified=subprocess.run(['ssh','-o','BatchMode=yes','x86host',shlex.join(['python3','-c',code])],text=True,capture_output=True,check=True,timeout=8)
        receipt.write_text(json.dumps(dict(checked_utc=time.strftime('%FT%TZ',time.gmtime()),host='x86host',proof=json.loads(verified.stdout)))+'\n')


P = ParamSpec('P')
R = TypeVar('R')


def serialized(function: Callable[P, R]) -> Callable[P, R]:
    """Serialize independent S1 invocations, including load, to isolate traffic."""
    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        lock = Path(__file__).resolve().parents[1] / 'results/s1/.controller.lock'
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open('w') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            return function(*args, **kwargs)
    return wrapped


@serialized
def run_pairs(pairs: list[tuple[str,list[str],str,list[str]]], allow_busy: bool = False, deadline: float = 25, metrics_url: str = METRICS) -> RunResult:
    before = idle(metrics_url)
    started = time.monotonic()
    workers: list[Remote] = []
    error = ''
    try:
        for server, cmd, _, _ in pairs:
            workers.append(Remote(server,cmd,deadline))
        # Give bound listeners a moment to start; no Spark traffic yet.
        time.sleep(0.5)
        idle(metrics_url)
        for _, _, client, cmd in pairs:
            workers.append(Remote(client,cmd,deadline))
        while any(w.process.poll() is None for w in workers):
            if not allow_busy:
                idle(metrics_url)
            if time.monotonic()-started > deadline:
                raise TimeoutError('Pair deadline exceeded')
            time.sleep(0.15)
    except (Exception, KeyboardInterrupt) as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        # Stop all endpoints before waiting on any one SSH process.
        for worker in workers:
            if worker.process.stdin and not worker.process.stdin.closed:
                worker.process.stdin.close()
        results = [w.close() for w in reversed(workers)]
    return dict(utc=time.strftime('%FT%TZ',time.gmtime()), seconds=time.monotonic()-started,
                idle_before=before, metrics_url=metrics_url, error=error, processes=results,
                ok=not error and all(r['returncode']==0 for r in results))


def rdma_pair(link: Link, tool: str, size: int, reverse: bool = False,
              twin: bool = False, duration: int | None = None, rate: float | None = None, twin_gid: int = 1) -> tuple[str,list[str],str,list[str]]:
    bw = tool.endswith('_bw')
    count = min(100000,max(10,536870912//size)) if bw else (200 if size < 1048576 else 20)
    common = [tool,'-p',str(link.port),'-s',str(size),'-n',str(count),'-u','10']
    if twin:
        common += ['--ipv6']
    if bw:
        common += ['-t','16','--report_gbits']
        if reverse: common += ['--reversed']
    if tool == 'ib_send_lat': common += ['-r','16']
    if duration is not None:
        common += ['-D',str(duration),'-f','0']
    if rate is not None:
        common += [f'--rate_limit={rate}','--rate_units=g','--rate_limit_type=SW','--burst_size=1']
    # Parameter exchange sockets bind direct IP; all Spark processes connect out.
    server = common + ['-d',link.hca,'-x','1' if twin else '3',f'--bind_source_ip={link.x86host_ip}']
    client = common + ['-d','roceP2p1s0f0' if twin else 'rocep1s0f0','-x',str(twin_gid) if twin else '3',f'--bind_source_ip={link.spark_ip}',link.x86host_ip]
    return 'x86host',server,link.host,client


def record(out: Path, name: str, data: RunResult) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / (name+'.jsonl')).open('a') as f:
        f.write(json.dumps(data)+'\n')
    print(name, 'PASS' if data['ok'] else 'FAIL', str(data.get('error','')), flush=True)
    progress=Path('/opt/dsv41/campaigns/DSV41-OFFLOAD/S1.progress.md')
    if progress.exists():
        with progress.open('a') as f:
            f.write(f'\n- {data["utc"]} {name}: {"PASS" if data["ok"] else "STOP/FAIL"} {data["error"]}\n')


def completed(out: Path, label: str, resume: bool) -> bool:
    previous=out/(label+'.jsonl')
    if not resume or not previous.exists():return False
    lines=previous.read_text().splitlines()
    if not lines:return False  # Opened but no result committed before interruption.
    return bool(json.loads(lines[-1])['ok'])


def matrix(link_names: list[str], out: Path, twin: bool = False, resume: bool = False, twin_gid: int = 1) -> None:
    if 'a' in link_names: copy_ready()
    for name in link_names:
        link = LINKS[name]
        for size in SIZES:
            for tool in ['ib_send_lat','ib_write_lat','ib_read_lat','ib_write_bw','ib_read_bw']:
                for reverse in ([False,True] if tool.endswith('_bw') else [False]):
                    prefix=(f'twin-gid{twin_gid}-' if twin_gid!=1 else 'twin-') if twin else ''
                    label = f'{name}-{prefix}{tool}-{size}-{"reverse" if reverse else "forward"}'
                    if completed(out,label,resume):
                        continue
                    row = run_pairs([rdma_pair(link,tool,size,reverse,twin,twin_gid=twin_gid)])
                    row.update({'link': name, 'tool': tool, 'size': size, 'reverse': reverse, 'twin': twin})
                    record(out,label,row)
                    if row['error']: raise RuntimeError(str(row['error']))
                    if not row['ok']: raise RuntimeError(f'{label}: process failed; inspect raw JSONL')


def tcp_pair(link: Link, size: int, mode: str, start_at: float = 0) -> tuple[str,list[str],str,list[str]]:
    code = Path(__file__).with_name('s1_tcp.py').read_text()
    encoded = "exec(__import__('base64').b64decode(" + repr(base64.b64encode(code.encode()).decode()) + '))'
    n = (200 if size<1048576 else 8) if mode=='echo' else min(20000,max(8,536870912//size))
    common = ['python3','-u','-c',encoded]
    server = common+['server',link.x86host_ip,link.spark_ip,str(link.port+100),str(size),str(n),mode,str(start_at)]
    client = common+['client',link.spark_ip,link.x86host_ip,str(link.port+100),str(size),str(n),mode,str(start_at)]
    return 'x86host',server,link.host,client


def tcp_matrix(links: list[str], out: Path, simultaneous: bool, resume: bool = False) -> None:
    if 'a' in links: copy_ready()
    groups = [links] if simultaneous else [[n] for n in links]
    for group in groups:
        for size in SIZES:
            for mode in ['echo','forward','reverse']:
                label=f'{"both" if simultaneous else group[0]}-tcp-{size}-{mode}'
                if completed(out,label,resume):continue
                start_at = time.time()+2 if simultaneous else 0
                row=run_pairs([tcp_pair(LINKS[n],size,mode,start_at) for n in group])
                row.update({'link': 'both' if simultaneous else group[0], 'tool': 'tcp', 'size': size, 'mode': mode, 'twin': False})
                record(out,label,row)
                if not row['ok']: raise RuntimeError('TCP test failed; inspect raw data')


def iperf_pair(link: Link, reverse: bool) -> tuple[str,list[str],str,list[str]]:
    port=str(link.port+300)
    server=['iperf3','-s','-1','-B',link.x86host_ip,'-p',port,'-J']
    client=['iperf3','-c',link.x86host_ip,'-B',link.spark_ip,'-p',port,'-t','3','-J']
    if reverse:client+=['-R']
    return 'x86host',server,link.host,client


def iperf_matrix(links: list[str], out: Path, simultaneous: bool, resume: bool = False) -> None:
    if 'a' in links:copy_ready()
    groups=[links] if simultaneous else [[n]for n in links]
    for group in groups:
        for reverse in [False,True]:
            label=f'{"both" if simultaneous else group[0]}-iperf3-{reverse}'
            if completed(out,label,resume):continue
            row=run_pairs([iperf_pair(LINKS[n],reverse) for n in group])
            row.update({'link':'both' if simultaneous else group[0],'tool':'iperf3','size':131072,'reverse':reverse})
            record(out,label,row)
            if not row['ok']:raise RuntimeError('iperf3 failed; inspect raw data')


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--link',choices=['a','b','both'],default='b')
    p.add_argument('--out',type=Path,default=Path('offload/results/s1'))
    p.add_argument('--twin',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--tcp',action='store_true')
    p.add_argument('--iperf',action='store_true')
    p.add_argument('--simultaneous',action='store_true')
    args=p.parse_args()
    links=['a','b'] if args.link=='both' else [args.link]
    if args.iperf:
        iperf_matrix(links,args.out,args.simultaneous,args.resume)
    elif args.tcp:
        tcp_matrix(links,args.out,args.simultaneous,args.resume)
    elif args.simultaneous:
        copy_ready()
        for size in SIZES:
            for tool in ['ib_write_bw','ib_read_bw']:
                for reverse in [False,True]:
                    label=f'both-{tool}-{size}-{reverse}'
                    if completed(args.out,label,args.resume):continue
                    row=run_pairs([rdma_pair(LINKS[n],tool,size,reverse,duration=3) for n in ['a','b']])
                    row.update({'link': 'both', 'tool': tool, 'size': size, 'reverse': reverse, 'twin': False})
                    record(args.out,label,row)
                    if not row['ok']: raise RuntimeError('Simultaneous test failed')
    else:
        matrix(links,args.out,args.twin,args.resume)

if __name__=='__main__':
    main()
