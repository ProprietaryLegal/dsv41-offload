"""Read-only, bounded SSH telemetry; no CUDA contexts or service changes."""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import TypeAlias, cast

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None
Object: TypeAlias = dict[str, JSON]

REMOTE = r'''
import json, os, pathlib, re, subprocess, sys
cfg=json.loads(sys.stdin.read())
out={"errors": []}
mem={k:int(v) for k,v in re.findall(r'^(MemAvailable|SwapTotal|SwapFree):\s+(\d+)', pathlib.Path('/proc/meminfo').read_text(), re.M)}
out['memory_kib']=mem
out['swap_used_kib']=mem['SwapTotal']-mem['SwapFree']
query=['uuid','temperature.gpu','clocks.gr','power.draw','memory.used','utilization.gpu']
cmd=['nvidia-smi', '--query-gpu='+','.join(query), '--format=csv,noheader,nounits']
if cfg['x86host']:
    # Resolve the assigned UUID prefixes on each probe; never rely on ordinal indices.
    inv=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True)
    if inv.returncode:
        out['errors'].append('GPU inventory: '+inv.stderr)
        uuids=[]
    else:
        uuids=[u.strip() for u in inv.stdout.splitlines() if u.startswith(('GPU-c4e7a371','GPU-7744b44b'))]
        if len(uuids)!=2: out['errors'].append('expected assigned GPU UUIDs, got '+str(uuids))
    cmd+=['--id='+','.join(uuids)] if uuids else ['--id=missing-assigned-GPU']
r=subprocess.run(cmd,capture_output=True,text=True,timeout=3)
if r.returncode: out['errors'].append('nvidia-smi: '+r.stderr)
else: out['gpu_csv']=r.stdout.strip(); out['gpu_columns']=query
if cfg['x86host']:
    out['processes']=[]
    for p in pathlib.Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid(): continue
        try:
            name=(p/'comm').read_text().strip()
            argv=(p/'cmdline').read_bytes().split(b'\0')
            executable=os.path.basename(argv[0].decode(errors='replace'))
            if name not in cfg['names'] and executable not in cfg['names']: continue
            stat=(p/'stat').read_text().rsplit(')',1)[1].split()
            out['processes'].append({'pid':int(p.name),'name':name,'start_ticks':int(stat[19]),'rss_bytes':int(stat[21])*os.sysconf('SC_PAGE_SIZE'),'major_faults':int(stat[9])})
        except FileNotFoundError: pass  # process exited during observation
        except PermissionError as e: out['errors'].append(str(e))
    out['nics']={}
    for nic in cfg['nics']:
        r=subprocess.run(['ethtool','-S',nic],capture_output=True,text=True,timeout=3)
        if r.returncode: out['errors'].append('ethtool '+nic+': '+r.stderr); continue
        out['nics'][nic]={k:int(v) for k,v in re.findall(r'^\s+([\w]+):\s+(\d+)\s*$',r.stdout,re.M) if any(x in k.lower() for x in ('byte','retrans','error','discard','drop','retry'))}
print(json.dumps(out))
'''


class Telemetry(threading.Thread):
    def __init__(self, sparks: list[str], x86host: str, names: list[str], extra: list[str],
                 nics: list[str] | None = None) -> None:
        super().__init__(daemon=True)
        self.sparks, self.x86host, self.names, self.extra = sparks, x86host, names, extra
        self.nics = nics or ["enp130s0f0np0", "enp130s0f1np1"]
        self.stopped = threading.Event()
        self.samples: list[JSON] = []
        self.errors: list[JSON] = []
        self.previous_nics: dict[str, dict[str, int]] = {}

    def host_sample(self, host: str, x86host: bool) -> Object:
        command = "python3 -c " + shlex.quote(REMOTE)
        config = {"x86host": x86host, "names": self.names, "nics": self.nics}
        process = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", host, command],
                                 input=json.dumps(config), capture_output=True, text=True, timeout=9, check=False)
        if process.returncode:
            raise RuntimeError(f"{host} telemetry exit {process.returncode}: {process.stderr.strip()}")
        result = cast(Object, json.loads(process.stdout))
        if x86host:
            current = cast(dict[str, dict[str, int]], result.get("nics", {}))
            result["nic_delta"] = {nic: {key: value - self.previous_nics[nic][key]
                                       for key, value in counters.items()
                                       if key in self.previous_nics.get(nic, {})}
                                   for nic, counters in current.items()}
            self.previous_nics = current
        return result

    def sample(self) -> None:
        started = time.monotonic()
        row: Object = {"ts": time.time()}
        with ThreadPoolExecutor(max_workers=len(self.sparks) + 1) as pool:
            futures = {host: pool.submit(self.host_sample, host, False) for host in self.sparks}
            futures[self.x86host] = pool.submit(self.host_sample, self.x86host, True)
            for host, future in futures.items():
                try:
                    row[host] = future.result()
                    result = cast(Object, row[host])
                    for error in cast(list[JSON], result.get("errors", [])):
                        self.record_error(f"{host}: {error}")
                except Exception as exc:  # noqa: BLE001 -- telemetry failures are printed and retained
                    self.record_error(f"{host}: {type(exc).__name__}: {exc}")
        for url in self.extra:
            try:
                with urllib.request.urlopen(url, timeout=3) as response:
                    row[url] = response.read().decode()
            except Exception as exc:  # noqa: BLE001 -- telemetry failures are printed and retained
                self.record_error(f"{url}: {type(exc).__name__}: {exc}")
        row["sample_duration_s"] = time.monotonic() - started
        self.samples.append(row)

    def record_error(self, error: str) -> None:
        self.errors.append({"ts": time.time(), "error": error})
        print(f"telemetry error: {error}", file=sys.stderr, flush=True)

    def run(self) -> None:
        while not self.stopped.is_set():
            begin = time.monotonic()
            try:
                self.sample()
            except Exception as exc:  # noqa: BLE001 -- telemetry failures are printed and retained
                self.record_error(f"sampler: {type(exc).__name__}: {exc}")
            self.stopped.wait(max(0, 2 - (time.monotonic() - begin)))

    def finish(self) -> None:
        self.stopped.set()
        self.join(timeout=30)
        if self.is_alive():
            raise RuntimeError("telemetry sampler did not stop within 30 seconds")
        self.sample()  # terminal counters cover the last sampling interval
