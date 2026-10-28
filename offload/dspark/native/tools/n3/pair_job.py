"""One bounded transient pair worker + local CPU fake client, coordinator only.

Called under pair.sh's two flocks and timeout. No service or Spark actions.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from ...identity import compute
from ....spark.tools.identity import identities


class StopRequested(RuntimeError):
    pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('arm')
    ap.add_argument('mode', choices=('off','on'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--root', default='/opt/dsv41/x86host-data/dspark')
    a = ap.parse_args()
    # A direct invocation cannot bypass the coordinator/lock guards in pair.sh.
    if os.environ.get('N3_PAIR_LOCKED') != '1':
        raise RuntimeError('run pair.sh under the coordinator pair window')
    from .guards import require_pair_ownership
    require_pair_ownership()
    a.out.mkdir(parents=True, exist_ok=False)
    def stop(signum: int, frame: Any) -> None:
        raise StopRequested(f'signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    d, markov, files = identities(json.loads(Path(a.root,'DSPARK_COPY_DONE').read_text()))
    ident = compute(a.arm, {'drafter_identity':d,'markov_identity':markov,'files':files})
    status = a.out/'worker.status.json'
    env = dict(os.environ, DSPARK_WORKER_ARM=a.arm, DSPARK_WORKER_ALLOW_LOOPBACK='1')
    env.pop('DSPARK_WORKER_TP_EMULATE',None)
    env.pop('DSPARK_V1_DUMP_DIR',None)
    if a.mode == 'on':
        env.update(DSPARK_V1_DUMP_DIR=str(a.out/'dumps'), DSPARK_V1_ROUNDS='1,2,3,4,5,6,7,8,9,10,40')
    phase = 'load-and-prewarm'
    samples: list[dict[str,Any]] = []
    telemetry_error: list[str] = []
    done = threading.Event()
    def sample() -> None:
        try:
            with (a.out/'memory-1s.jsonl').open('x') as log:
                while not done.is_set():
                    out = subprocess.run(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],
                        '--query-gpu=uuid,memory.used,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu',
                        '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True).stdout
                    for line in out.splitlines():
                        v = [x.strip() for x in line.split(',')]
                        row = {'time':time.time(),'phase':phase,'uuid':v[0],'memory_mib':float(v[1]),
                               'power_w':float(v[2]),'limit_w':float(v[3]),'sm_mhz':float(v[4]),
                               'mem_mhz':float(v[5]),'temp_c':float(v[6])}
                        log.write(json.dumps(row)+'\n'); log.flush(); samples.append(row)
                    done.wait(1)
        except Exception as error:
            telemetry_error.append(str(error))
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    worker = client = None
    start = time.monotonic()
    try:
        with (a.out/'worker.log').open('x') as log:
            worker = subprocess.Popen([sys.executable,'-u','-m','offload.dspark.worker.server','--tp','2',
                '--root',a.root,'--tcp-stub','127.0.0.1:18639','--max-model-len','1048576',
                '--status-file',str(status)],stdout=log,stderr=subprocess.STDOUT,env=env)
            deadline = time.monotonic()+600
            while True:
                if worker.poll() is not None:
                    raise RuntimeError(f'worker startup failed rc={worker.returncode}; see {a.out}/worker.log')
                if time.monotonic() > deadline:
                    raise TimeoutError('worker ready deadline 600s exceeded')
                if status.exists():
                    st = json.loads(status.read_text())
                    if st['pid'] == worker.pid:
                        if st.get('drafter_identity') != ident['drafter_identity'] or st.get('draft_head') != ident['draft_head']:
                            raise RuntimeError('ready status arm/head/identity mismatch')
                        break
                time.sleep(.25)
            startup_s = time.monotonic()-start
            phase = 'serve'
            cenv = dict(env, CUDA_VISIBLE_DEVICES='')
            with (a.out/'client.log').open('x') as clog:
                client = subprocess.Popen([sys.executable,'-u','-m','offload.dspark.native.tools.n3.fake_client',
                    '--drafter',ident['drafter_identity'],'--markov',ident['markov_identity'],
                    '--out',str(a.out/'client.json')],stdout=clog,stderr=subprocess.STDOUT,env=cenv)
                rc = client.wait(timeout=220)
            if rc:
                raise RuntimeError(f'fake client rc={rc}; see {a.out}/client.log and worker.log')
            if worker.poll() is not None:
                raise RuntimeError(f'worker exited during serving: {worker.returncode}')
            phase = 'done'
    finally:
        # Only exact subprocesses created above. No units, pkill or foreign PIDs.
        for proc in (client,worker):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait(timeout=2)
        done.set(); thread.join(timeout=2)
    if telemetry_error or not samples:
        raise RuntimeError(f'telemetry failed: {telemetry_error}')
    peaks: dict[str,dict[str,float]] = {}
    for row in samples:
        key = row['phase']
        card = row['uuid']
        peaks.setdefault(key,{})[card] = max(peaks.get(key,{}).get(card,0),row['memory_mib'])
    result = {'arm':a.arm,'mode':a.mode,'identity':ident['drafter_identity'],'draft_head':ident['draft_head'],
              'startup_s':startup_s,'memory_peak_mib_1s_observed':peaks,'samples':len(samples),
              'client':json.loads((a.out/'client.json').read_text())}
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'client'}),flush=True)


if __name__ == '__main__':
    main()
