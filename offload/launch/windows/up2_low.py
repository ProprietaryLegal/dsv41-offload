"""Direct W8-style UP2 low-cap runner. No window opening/closing or restoration."""
from __future__ import annotations

import argparse
import json
import os
import re
import selectors
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO, Protocol, cast

from offload.prefill_maxlen.evidence import Json, number, record
from offload.launch.windows import up2_checks as ck

CAMPAIGN = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD')
ROOT = Path(__file__).resolve().parents[3]
RESTORE = ('# Inspect rollback.json; drain/cancel only window-owned requests before restoration.\n'
    'ssh sparkhead \'cd ~/dsv41-deploy && test "$(git rev-parse --short=7 HEAD)" = 21bed31 '
    '&& git diff --exit-code HEAD -- dsv41.env serve.sh start.sh\'\n' +
    '\n'.join(f'bash {CAMPAIGN}/W8/restore.sh {step}' for step in ('start','check','close')))
X86HOST_PROBE = '''import json,pathlib,os
root=pathlib.Path.home()/'dsv41-offload/prod'
current=(root/'dspark-worker/current').resolve()
status=json.loads((root/'run/dspark-worker.status.json').read_text())
process=pathlib.Path('/proc')/str(status['pid'])
env=dict(x.split('=',1) for x in (process/'environ').read_text().split('\\0') if '=' in x)
print(json.dumps({'revision':(current/'SOURCE_COMMIT').read_text().strip(),'current':str(current),
 'cwd':os.readlink(process/'cwd'),'cmd':(process/'cmdline').read_text().split('\\0'),
 'gpus':env.get('CUDA_VISIBLE_DEVICES'),'status':status}))
'''
ENGRAM_PROBE = '''import hashlib,json,os,pathlib,subprocess
root=pathlib.Path(os.environ['VLLM_ENGRAM_REMOTE_ROOT'])
files=[pathlib.Path(os.environ['DSV41_ENGRAM_OVERLAY']),root/'manifest.json',
 root/'offload/engram/spark/py/sitecustomize.py',
 pathlib.Path('/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/common/engram_disk.cpp'),
 pathlib.Path('/opt/dsv41-ofl/libofl.so')]
files += [root/p for p in ('offload/engram/spark/engram_remote.cpp',
 'offload/engram/engram_proto.cpp','offload/engram/engram_client.h',
 'offload/engram/engram_proto.h','offload/transport/ofl.cpp','offload/transport/ofl.h')]
jit=set()
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit(): continue
 try: maps=(proc/'maps').read_text()
 except (FileNotFoundError,ProcessLookupError): continue
 for line in maps.splitlines():
  parts=line.split(None,5)
  if len(parts)==6 and '/vllm_engram_remote_' in parts[5] and parts[5].endswith('.so'):
   jit.add(pathlib.Path(parts[5]))
if len(jit)!=1: raise SystemExit('required single loaded Engram JIT identity missing/ambiguous')
files += sorted(jit)
hashes={}
for path in files:
 p=subprocess.Popen(['dd','if='+str(path),'iflag=direct','bs=4096','status=none'],stdout=subprocess.PIPE)
 try: hashes[str(path)]=hashlib.file_digest(p.stdout,'sha256').hexdigest()
 finally: p.stdout.close(); code=p.wait()
 if code: raise SystemExit('Engram O_DIRECT identity read failed: '+str(path))
print(json.dumps(hashes,sort_keys=True))
'''
ARMS = [('diag-off',0,1), ('diag-on',1,1), ('off-a',0,0), ('on',1,0), ('off-b',0,0)]
SSH = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=4','-o','ServerAliveInterval=4','-o','ServerAliveCountMax=1']


def save(path: Path, data: dict[str, Json], *, screen: bool = False) -> None:
    with path.open('x') as output:
        json.dump(ck.screen_record(data,screen), output, indent=2, allow_nan=False)
        output.write('\n')


class Backend(Protocol):
    out: Path
    def start(self, arm: str, replay: int, audit: int) -> None: ...
    def request(self, arm: str, name: str, target: int, budget: int, *, numeric: bool = False,
                force_length: bool = False, rep: int = 0, cold: bool = False) -> dict[str, Json]: ...
    def native(self, arm: str, *, complete: bool) -> int: ...
    def needles(self, arm: str) -> None: ...
    def clean(self, arm: str, audit: int) -> None: ...


def sequence(backend: Backend, phase: str, checkpoint: Callable[[dict[str, Json]], None],
             *, screen: bool = False, contexts: tuple[int, ...] = (150000, 300000)) -> dict[str, Json]:
    """Same orchestration for real execution and offline rehearsal; every cell is durable."""
    def emit(data: dict[str, Json]) -> None:
        checkpoint(ck.screen_record(data,screen))
    ck.measured_contexts(contexts)
    diagnostic_status = 'PASS-SCREEN' if screen else 'PASS'
    cells: dict[str, list[dict[str, Json]]] = {}
    numeric: dict[str, list[dict[str, Json]]] = {}
    if phase in ('all','diagnostics'):
        for arm,replay,audit in ARMS[:2]:
            backend.start(arm,replay,audit)
            backend.request(arm,'warm32',32000,32,force_length=True)
            q = backend.native(arm,complete=False)
            for tail in (1,127,128,129):
                backend.request(arm,f'tail{tail}',36*q+tail,8,force_length=True)
            backend.request(arm,'near-cap',327675,5,force_length=True)
            numerical_rows: list[dict[str, Json]] = []
            for n in (512,32000,150000,300000):
                for rep in range(2 if arm == 'diag-off' else 1):
                    numerical_rows.append(backend.request(arm,f'num{n}-{rep}',n,64,numeric=True,rep=rep))
            numeric[arm] = numerical_rows
            backend.needles(arm)
            backend.native(arm,complete=True)
            backend.clean(arm,audit)
            if arm == 'diag-on':
                comparisons: list[Json] = []
                for i,n in enumerate((512,32000,150000,300000)):
                    a,aa = numeric['diag-off'][2*i:2*i+2]
                    b = numerical_rows[i]
                    ck.require(a['prompt_sha256'] == aa['prompt_sha256'] == b['prompt_sha256'], 'numeric prompt drift')
                    # Greedy text is a record at every context, A/A included: the engine is not run-to-run
                    # deterministic at T=0 (W13 r2: same-start OFF A/A text differed at 512, 32K, 150K and 300K with
                    # first-token KL 0.0 at 512). The numerical floor above is the hard gate.
                    comparisons.append({'context':n, **ck.numerical(a,aa,b),
                        'text_aa_equal_INFORMATIONAL': a['text_sha256'] == aa['text_sha256'] and a['finish'] == aa['finish'],
                        'text_on_equals_off_INFORMATIONAL': a['text_sha256'] == b['text_sha256'] and a['finish'] == b['finish']})
                emit({'numerical':comparisons,'diagnostics':diagnostic_status})
    if phase in ('all','measured'):
        for arm,replay,audit in ARMS[2:]:
            backend.start(arm,replay,audit)
            backend.request(arm,'warm32',32000,32,force_length=True)
            cells[arm] = []
            for n in contexts:
                for rep in range(3):
                    row = ck.screen_record(backend.request(arm,f'cold{n}-{rep}',n,1500,rep=rep,cold=True),screen)
                    ck.require(row.get("window_id") == str(backend.out.resolve()), "cell belongs to another window")
                    ck.cold(row,n,rep)
                    cells[arm].append(row)
                    emit(ck.summarize(cells,contexts))
            backend.clean(arm,audit)
    result = {'phase':phase, 'diagnostics':diagnostic_status if numeric else 'PRIOR QUALIFICATION REQUIRED',
              **ck.summarize(cells,contexts)}
    if screen and numeric:
        result['native'] = 'NOT RUN' if getattr(backend,'native_missing',False) else 'PASS'
    return ck.screen_record(result,screen)


def terminate(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL)
            proc.wait(timeout=3)


class Supervisor:
    """Only owned local process groups. Continuous5s remote read-only memory streams."""
    def __init__(self, out: Path, deadline: float):
        self.out, self.deadline = out, deadline
        self.select = selectors.DefaultSelector()
        self.monitors: list[subprocess.Popen[str]] = []
        self.files: list[IO[str]] = []
        self.pending: dict[int, str] = {}
        self.last: dict[str, float] = {}
        self.active: subprocess.Popen[str] | None = None

    def begin(self) -> None:
        script = ('import time; from pathlib import Path\nwhile True:\n '
            'print(";".join(x for x in Path("/proc/meminfo").read_text().splitlines() '
            'if x.startswith(("MemAvailable:","SwapFree:"))),flush=True); time.sleep(5)')
        for host in ('sparkhead','sparkworker-wired'):
            err = (self.out/f'memory-{host}.stderr').open('x')
            log = (self.out/f'memory-{host}.jsonl').open('x')
            self.files.extend([err,log])
            proc = subprocess.Popen([*SSH,host,'exec python3 -u -c '+shlex.quote(script)],
                stdout=subprocess.PIPE,stderr=err,text=True,start_new_session=True)
            self.monitors.append(proc)
            assert proc.stdout is not None
            os.set_blocking(proc.stdout.fileno(),False)
            self.select.register(proc.stdout,selectors.EVENT_READ,(host,log))
            self.pending[proc.stdout.fileno()] = ''
        start = time.monotonic()
        while len(self.last) < 2:
            ck.require(time.monotonic()-start < 10, 'initial memory telemetry missing')
            self.poll()

    def poll(self) -> None:
        ck.require(not (self.out/'STOP').exists(), 'external STOP file')
        ck.require(time.time() < self.deadline, f'90-minute hard deadline {self.deadline:.6f} reached')
        for key,_ in self.select.select(timeout=.1):
            stream = cast(IO[str],key.fileobj)
            part = os.read(stream.fileno(),65536).decode()
            ck.require(bool(part), 'memory monitor EOF')
            self.pending[stream.fileno()] += part
            while '\n' in self.pending[stream.fileno()]:
                line,self.pending[stream.fileno()] = self.pending[stream.fileno()].split('\n',1)
                host,log = key.data
                available,swap = ck.memory(line.replace(';','\n'))
                self.last[host] = time.monotonic()
                log.write(json.dumps({'time':time.time(),'available_kb':available,'swap_free_kb':swap})+'\n')
                log.flush()
        ck.require(all(p.poll() is None for p in self.monitors), 'memory monitor exited')
        ck.require(all(time.monotonic()-t < 12 for t in self.last.values()), 'stale memory telemetry')

    def run(self, name: str, command: list[str], *, stdin: str | None = None, timeout: float = 900) -> str:
        self.poll()
        start = time.monotonic()
        with (self.out/f'{name}.log').open('x') as log:
            # File stdin avoids a pipe write deadlock for the emitted42-file verifier.
            input_path = self.out/f'{name}.stdin'
            if stdin is not None:
                input_path.write_text(stdin)
            with input_path.open() if stdin is not None else open(os.devnull) as source:
                proc = subprocess.Popen(command,stdin=source,stdout=log,stderr=subprocess.STDOUT,
                                        text=True,start_new_session=True)
                self.active = proc
                try:
                    while proc.poll() is None:
                        self.poll()
                        ck.require(time.monotonic()-start < timeout, f'{name} timeout/nonprogress')
                    if proc.returncode != 0:
                        log.flush()
                        # Preserve the request driver's exact deadline refusal, rather
                        # than guessing from the clock after an unrelated child error.
                        deadline_error = re.search(
                            r'^ValueError: STOP: ((?:needle )?submission deadline [0-9.]+ reached)$',
                            (self.out/f'{name}.log').read_text(), re.M)
                        if deadline_error:
                            ck.require(False, f'{name}: {deadline_error.group(1)}')
                    ck.require(proc.returncode == 0,f'{name} failed rc{proc.returncode}; see private log')
                    self.poll()
                finally:
                    terminate(proc)
                    self.active = None
        return (self.out/f'{name}.log').read_text()

    def close(self) -> None:
        if self.active is not None:
            terminate(self.active)
        for proc in self.monitors:
            terminate(proc)
        self.select.close()
        for handle in self.files:
            handle.close()


class Live:
    screen = False
    native_missing = False

    def __init__(self, out: Path, remote: str, expected: str, deadline: float, submit: float,
                 native_root: Path, data_dir: Path, supervisor: Supervisor, native_wait: float = 0, *, screen: bool = False):
        self.out,self.remote,self.expected = out,remote,expected
        self.submit,self.native_root,self.data_dir = submit,native_root,data_dir
        self.s = supervisor
        self.native_wait = native_wait
        self.screen = screen
        self.tail_q: dict[str,int] = {}
        self.serial = 0
        self.production: list[dict[str, Json]] = []
        self.ids: list[str] = []
        self.previous_ids: set[str] = set()
        self.before: dict[str, Json] | None = None
        self.last_stats: dict[str, Json] = {}
        self.proxy_pid = 0
        self.manifest = ck.digest(ROOT/'offload/upstream_adopt/MANIFEST.json')
        self.verifier = self.s.run('emit-verifier',[sys.executable,'offload/upstream_adopt/verify_runtime.py','--emit'])

    def save(self, path: Path, data: dict[str, Json]) -> None:
        save(path,data,screen=self.screen)

    def call(self, label: str, command: list[str], stdin: str | None = None) -> str:
        self.serial += 1
        return self.s.run(f'{self.serial:04d}-{label}',command,stdin=stdin)

    def ssh(self, host: str, command: str, stdin: str | None = None) -> str:
        return self.call(host,[*SSH,host,command],stdin)

    def health(self, *, production: bool = False) -> None:
        port = 8000 if production else 8001
        self.ssh('sparkhead',f'curl -fsS --max-time 10 http://localhost:{port}/health')
        info = record(cast(Json,json.loads(self.ssh('sparkhead',f'curl -fsS --max-time 10 http://localhost:{port}/v1/models'))))
        models = ck.rows(info['data'])
        ck.require(len(models) == 1 and models[0].get('max_model_len') == ck.CAP and
                   models[0].get('root') == '/srv/dsv41/legal-v2a', 'model identity/cap failure')

    def idle(self, *, production: bool = False) -> None:
        port = 8000 if production else 8001
        metrics = self.ssh('sparkhead',f'curl -fsS --max-time 10 http://localhost:{port}/metrics')
        for key in ('vllm:num_requests_running','vllm:num_requests_waiting'):
            found = [float(x.split()[-1]) for x in metrics.splitlines() if x.startswith((key+'{',key+' '))]
            ck.require(bool(found) and sum(found) == 0,f'{key} absent/busy')

    def worker(self) -> None:
        text = self.ssh('x86host','python3 -c '+shlex.quote(X86HOST_PROBE))
        probe = record(cast(Json,json.loads(text)))
        ck.require(probe.get('revision') == ck.WORKER,'worker source revision mismatch; never overwrite')
        ck.require(str(probe['cwd']).startswith(str(probe['current'])+'/'), 'running worker source differs from current')
        argv = ck.strings(probe['cmd'])
        ck.require(argv[argv.index('--tp')+1] == '2' and argv[argv.index('--max-model-len')+1] == '1048576',
                   'running worker layout/position support mismatch')
        ck.require(probe.get('gpus') == 'GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93,GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc',
                   'running worker GPU identity mismatch')
        data = record(probe['status'])
        ck.require(data.get('tp') == 2 and data.get('tp_comm_errors') == 0 and
                   data.get('listeners') == ['10.202.28.1:18620'], 'worker status identity/error failure')
        import datetime
        stamp = datetime.datetime.fromisoformat(str(data['time']).replace('Z','+00:00')).timestamp()
        ck.require(abs(time.time()-stamp) < 30,'worker status stale')

    def capture(self, host: str) -> dict[str, Json]:
        text = self.ssh(host,'bash -s', (ROOT/'offload/dspark/d2/capture_window_identity.sh').read_text())
        objects = [record(cast(Json,json.loads(line))) for line in text.splitlines()]
        ck.require(len(objects) == 2,'incomplete rank identity capture')
        container,files = objects
        inspect = record(cast(Json,json.loads(self.ssh(host,"docker inspect --format '{{json .Config}}' dsv41-serve"))))
        env: dict[str, Json] = {}
        for entry in ck.strings(inspect['Env']):
            key,_,value = entry.partition('=')
            if key.startswith(('VLLM_','EXL3_','NCCL_','PYTORCH_','CUDA_')) or key in ('PYTHONPATH','DSV41_ENGRAM_OVERLAY'):
                ck.require(key not in env,'duplicate runtime environment key')
                env[key] = value
        # Only target metadata belongs in comparison: composed engine files intentionally differ.
        target = {k:v for k,v in record(files['files']).items() if k.startswith('/srv/dsv41/legal-v2a/')}
        target['generation_config'] = files['generation_config']
        return {'id':container['id'],'state':container['state'],'image':container['image'],
                'engine':files['engine'],'files':target,'env':env,
                'engram_identity':record(cast(Json,json.loads(self.ssh(host,
                    'docker exec -i dsv41-serve python3 -',ENGRAM_PROBE)))),
                'argv':self.ssh(host,'docker exec dsv41-serve cat /serve.sh')}

    def preflight(self) -> None:
        ck.require((CAMPAIGN/'D2-GEN.PAUSE').exists() or subprocess.run(
            ['tmux','has-session','-t','=d2-gen'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode != 0,
            'D2 generation active without pause')
        self.ssh('sparkhead',"test -f ~/dsv41-deploy/HOLD && cd ~/dsv41-deploy && "
            "test \"$(git rev-parse --short=7 HEAD)\" = 21bed31 && "
            "test \"$(git rev-parse HEAD)\" = \"$(git rev-list -n1 prod-k1-327-20261023)\" && "
            "git diff --exit-code HEAD -- dsv41.env serve.sh start.sh")
        self.ssh('sparkhead',f'cd ~/{shlex.quote(self.remote)} && test "$(git rev-parse HEAD)" = {shlex.quote(self.expected)} '
            '&& test -z "$(git status --porcelain --untracked-files=no)"')
        # Source guard on CPU even for OFF. Staged replay guard is called again by start_exp for ON.
        from offload.upstream_adopt.replay_guard import verify
        config = self.call('source-config',['bash','-c',f'set -eu; source {ck.CONFIG}; printf "%s\\n%s\\n" "$OVERLAYS" "$EXTRA_DOCKER_ARGS"'])
        overlays,extra = config.splitlines()
        verify(overlays,extra)
        paths = [ck.CONFIG,'offload/launch/configs/UP2-0075.env',
                 'offload/launch/configs/D-kv6-async-up1-control.env','offload/launch/start_exp.sh',
                 'offload/launch/serve_exp.sh','offload/upstream_adopt/MANIFEST.json']
        remote = self.ssh('sparkhead',f'cd ~/{shlex.quote(self.remote)} && sha256sum '+shlex.join(paths))
        expected = '\n'.join(f'{ck.digest(ROOT/p)}  {p}' for p in paths)
        ck.require(remote.strip() == expected,'staged config/source mismatch')
        self.health(production=True)
        self.idle(production=True)
        self.worker()
        self.production = [self.capture(host) for host in ('sparkhead','sparkworker-wired')]
        ck.require(self.production[0]['files'] == self.production[1]['files'] and
                   self.production[0]['image'] == self.production[1]['image'], 'production rank identities disagree')
        self.previous_ids.update(str(r['id']) for r in self.production)
        self.save(self.out/'rollback.json',{'production_commit':'21bed31','tag':'prod-k1-327-20261023',
            'checkout':'~/dsv41-deploy','worker_source':ck.WORKER,'ranks':cast(Json,self.production)})

    def start(self, arm: str, replay: int, audit: int) -> None:
        ck.require(time.time() < self.submit,f'submission deadline {self.submit:.6f} reached; do not start another arm')
        self.ssh('sparkhead','test -f ~/dsv41-deploy/HOLD')
        self.idle(production=arm in ('diag-off','off-a') and not self.ids)
        self.worker()
        self.ssh('sparkhead',f'cd ~/{shlex.quote(self.remote)} && UP2_CAP=327680 UP2_REPLAY={replay} UP2_AUDIT={audit} '
            f'bash offload/launch/start_exp.sh {ck.CONFIG}')
        self.health()
        current = [self.capture(host) for host in ('sparkhead','sparkworker-wired')]
        for rank,host in enumerate(('sparkhead','sparkworker-wired')):
            ck.runtime(self.production[rank],current[rank],replay,audit)
            ck.require(str(current[rank]['id']) not in self.previous_ids,'start reused a previous container')
            result = self.ssh(host,'docker exec -i dsv41-serve python3 -',self.verifier)
            ck.require('UP1 runtime hashes verified: 42 files;' in result,'mounted42-file check missing')
            logs = self.ssh(host,'docker logs dsv41-serve 2>&1')
            ck.clean_logs(logs)
            ck.require('remote Engram' in logs or 'remote table' in logs or 'Engram remote' in logs,
                       'remote Engram runtime evidence missing')
            if rank == 0:
                # Only rank 0 runs EngineCore, which logs the KV capacity; rank 1 is --headless (workers only).
                found = re.findall(r'GPU KV cache size: ([\d,]+)',logs)
                ck.require(bool(found) and all(int(n.replace(',','')) == ck.KV for n in found),'rank KV capacity mismatch')
                pids = re.findall(r'through proxy process pid (\d+)',logs)
                ck.require(len(pids) == 1 and 'async mode on' in logs, 'async process startup not proven')
                self.proxy_pid = int(pids[0])
        self.ids = [str(r['id']) for r in current]
        self.previous_ids.update(self.ids)
        self.save(self.out/f'{arm}-identity.json',{'ranks':cast(Json,current)})
        self.before = None

    def stats(self) -> dict[str, Json]:
        self.idle()
        return record(cast(Json,json.loads(self.ssh('sparkhead',
            f'docker exec -i dsv41-serve python3 - --pid {self.proxy_pid}',
            (ROOT/'offload/launch/windows/up2_proxy_stats.py').read_text()))))

    def request(self, arm: str, name: str, target: int, budget: int, *, numeric: bool = False,
                force_length: bool = False, rep: int = 0, cold: bool = False) -> dict[str, Json]:
        self.health()
        self.idle()
        # Coordinator fault probes may deliberately exercise ordinary failures.
        # Normal requests must introduce none of the forbidden counter growth.
        before = self.stats()
        output = self.out/f'{arm}-{name}.json'
        # Pair numeric/cold requests across restarted arms. Warmup uses separate salts.
        salt = f'{self.out.name}:{"cold" if cold else name}:{target}:{rep}'
        cmd = [sys.executable,'-m','offload.launch.windows.up2_request','--target',str(target),'--budget',str(budget),
               '--salt',salt,'--output',str(output),'--ledger',str(self.out/'requests.jsonl'),
               '--deadline',str(self.submit),'--data-dir',str(self.data_dir),'--rep',str(rep),
               '--kind','cold' if cold else name]
        if self.screen: cmd.append('--screen')
        if numeric: cmd.append('--numeric')
        if force_length: cmd.append('--force-length')
        self.call(f'{arm}-{name}',cmd)
        result = ck.read(output)
        self.clean(arm,1 if arm.startswith('diag-') else 0, before)
        if cold or name == 'warm32':
            ck.require(number(record(self.last_stats['counters']).get('rounds_ok',0)) >
                       number(record(before['counters']).get('rounds_ok',0)), 'warmup/cold request had no remote drafting')
        return result

    def native(self, arm: str, *, complete: bool) -> int:
        path = self.native_root/arm/('qualification.json' if complete else 'q.json')
        stage = 'complete' if complete else 'q'
        self.save(self.out/f'{arm}-await-{stage}.json',{'arm':arm,'container_ids':cast(Json,self.ids),
            'manifest_sha256':self.manifest,'required_packet':str(path),'submission_deadline':self.submit,
            'native_checks':list(ck.NATIVE_CHECKS) if complete else ['native measured q'],
            'tail_q':self.tail_q.get(arm)})
        end = min(time.time()+self.native_wait,self.submit)
        while not path.is_file() and time.time() < end:
            self.s.poll()
        if not path.is_file():
            self.native_missing = True
            missing = self.out/f'{arm}-native-NOT-RUN.json'
            if not missing.exists():
                self.save(missing,{'status':'NOT RUN' if self.screen else 'NOT RUN/HOLD',
                    'missing':str(path),'checks':{key:'NOT RUN' for key in ck.NATIVE_CHECKS}})
            if not self.screen:
                raise ValueError('NOT RUN/HOLD: native probe evidence unavailable; see UP2-LOW.md')
            if not complete:
                # The reviewed proxy publishes audited-round counts, not per-chunk q.
                # A count or MAX_BATCH is not a measured solo q. Preserve the snapshot
                # so this ESTIMATED fallback can be audited without inventing telemetry.
                self.tail_q[arm] = 4092
                self.save(self.out/f'{arm}-tails.json',{'q':4092,'classification':'ESTIMATED',
                    'source':'CPU schedule; reviewed audit/proxy stats expose no per-chunk q',
                    'proxy_stats':self.last_stats,'tails_verified':False,
                    'tail_sizes':[36*4092+t for t in (1,127,128,129)]})
            return self.tail_q[arm]
        packet = ck.read(path)
        q = ck.native(packet,arm,self.ids,self.manifest,self.native_root,complete=complete)
        if complete:
            ck.require(q == self.tail_q.get(arm),'native receipt q differs from q used for sent tails')
        else:
            self.tail_q[arm] = q
            if self.screen:
                self.save(self.out/f'{arm}-tails.json',{'q':q,'classification':'MEASURED',
                    'source':str(path),'tails_verified':False,
                    'tail_sizes':[36*q+t for t in (1,127,128,129)]})
        self.save(self.out/f'{arm}-native-{"complete" if complete else "q"}.json',packet)
        if complete:
            # Expected ordinary fault failures are scoped by the native receipt.
            # They cannot be used as a baseline for normal requests until verified.
            self.before = self.stats()
        return q

    def needles(self, arm: str) -> None:
        for target in (150000,300000):
            self.idle()
            before = self.stats()
            path = self.out/f'{arm}-needle{target}.json'
            self.call(f'{arm}-needle{target}',[sys.executable,'-m','offload.launch.windows.up2_request',
                '--needle','--target',str(target),'--budget','300','--salt',f'{self.out.name}:needle{target}',
                '--output',str(path),'--ledger',str(self.out/'requests.jsonl'),'--deadline',str(self.submit),
                '--data-dir',str(self.data_dir),*(['--screen'] if self.screen else [])])
            data = ck.read(path)
            ck.require(data.get('recall') == 6 and data.get('of') == 6 and data.get('foreign_codes') == 0,
                       'needle6/6 or foreign-code failure')
            self.clean(arm,1,before)

    def clean(self, arm: str, audit: int, before: dict[str, Json] | None = None) -> None:
        self.health()
        data = self.stats()
        ck.counters(data,audit,before or self.before)
        with (self.out/'proxy-stats.jsonl').open('a') as handle:
            handle.write(json.dumps(ck.screen_record({'arm':arm,'time':time.time(),
                'counters':data['counters'],'async_watchdog':data['async_watchdog']},self.screen),allow_nan=False)+'\n')
        self.last_stats = data
        for rank,host in enumerate(('sparkhead','sparkworker-wired')):
            identity = self.ssh(host,"docker inspect --format '{{.Id}}' dsv41-serve").strip()
            ck.require(identity == self.ids[rank],'engine/container changed during arm')
            ck.clean_logs(self.ssh(host,'docker logs dsv41-serve 2>&1'))
        self.worker()
        self.idle()


def plan(phase: str, *, screen: bool = False, contexts: tuple[int, ...] = (150000, 300000)) -> str:
    ck.measured_contexts(contexts)
    selected = ARMS if phase == 'all' else ARMS[:2] if phase == 'diagnostics' else ARMS[2:]
    lines = ['DRY RUN: no commands executed; no network or output files.',
        'Coordinator Deepseek/coordinator; HOLD already open; current production snapshot + idle + worker + source guard.',
        'Continuous both-rank5s memory supervision >=3GiB; SwapFree recorded; STOP/deadline abort.',
        'Each start: idle, source/worker pins, start_exp, cap/KV, both-rank verify_runtime42 files, argv/env, proxy/Engram/no local drafter.']
    for arm,replay,audit in selected:
        lines.append(f'{arm}: UP2_CAP=327680 UP2_REPLAY={replay} UP2_AUDIT={audit} bash offload/launch/start_exp.sh {ck.CONFIG}')
        lines.append('  exact32000 warmup,32 generated tokens; discard timing')
        if audit:
            lines += ['  native measured q; tails36*q+{1,127,128,129}; prompt327675+5 normal length finish',
                '  matched isolated A/A then ON/OFF numeric floor at512/32000/150000/300000; needles6/6 at150K/300K',
                '  native recovery/scheduler/provenance evidence (missing = NOT RUN/HOLD); no supplied native fault adapter']
        else:
            lines.append('  cold exact'+ '/'.join(map(str,contexts)) +' x3,1500 output reserve; shared salt pairs; cache0; timing/spreads informational')
    if screen:
        lines.insert(1,'SCREEN ONLY; adoptable=false. Missing native evidence = NOT RUN; continue API gates.')
        lines = [line.replace('native measured q;', 'native measured q if supplied; otherwise ESTIMATED CPU q=4092, tails unverified;')
                 .replace('missing = NOT RUN/HOLD','missing = NOT RUN, continue SCREEN') for line in lines]
    return '\n'.join(lines)


def validate_qualification(directory: Path, expected: str, *, screen: bool = False) -> None:
    qualification = ck.read(directory/'qualification.json')
    ck.require((qualification.get('diagnostics') == 'PASS' or
        (screen and qualification.get('diagnostics') == 'PASS-SCREEN' and
         qualification.get('screen') is True and qualification.get('adoptable') is False)) and
        qualification.get('manifest_sha256') == ck.digest(ROOT/'offload/upstream_adopt/MANIFEST.json') and
        qualification.get('config_sha256') == ck.digest(ROOT/ck.CONFIG) and
        qualification.get('worker_source') == ck.WORKER and
        qualification.get('experiment_commit') == expected,
        'prior diagnostics absent/drifted')
    artifacts = record(qualification['artifact_sha256'])
    ck.require(bool(artifacts),'prior qualification evidence absent')
    for name,sha in artifacts.items():
        path = directory/name
        ck.require(path.resolve().is_relative_to(directory.resolve()) and
                   ck.digest(path) == sha,'prior qualification evidence changed')
    for arm in ('diag-off','diag-on'):
        identity = ck.read(directory/f'{arm}-identity.json')
        ids = [str(r['id']) for r in ck.rows(identity['ranks'])]
        for stage,complete in ((('q',False),('complete',True)) if screen else (('complete',True),)):
            receipt = directory/f'{arm}-native-{stage}.json'
            if receipt.is_file():
                ck.native(ck.read(receipt),arm,ids,str(qualification['manifest_sha256']),
                          Path(str(qualification['native_evidence_root'])),complete=complete)
            elif screen and qualification.get('diagnostics') == 'PASS-SCREEN':
                missing = directory/f'{arm}-native-NOT-RUN.json'
                ck.require(missing.name in artifacts,'screen missing native record not bound')
                data = ck.read(missing)
                ck.require(data.get('status') == 'NOT RUN' and data.get('screen') is True and
                           data.get('adoptable') is False and data.get('checks') ==
                           {key:'NOT RUN' for key in ck.NATIVE_CHECKS},'invalid screen NOT RUN record')
            else:
                ck.require(False,'prior native receipt missing')


def coordinator() -> None:
    values = [os.environ[k] for k in ('D2_COORDINATOR','WINDOW_COORDINATOR') if k in os.environ]
    ck.require(bool(values) and all(v == 'Deepseek/coordinator' for v in values), 'coordinator identity required')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true')
    mode.add_argument('--dry-run',action='store_true')
    ap.add_argument('--screen',action='store_true',help='explicit non-adoptable screen; missing native evidence is NOT RUN')
    ap.add_argument('--phase',choices=('all','diagnostics','measured'),default='all')
    ap.add_argument('--contexts',type=int,nargs='+',choices=(150000,300000),default=None,
                    help='measured-only context subset; default both')
    ap.add_argument('--out',type=Path)
    ap.add_argument('--remote-repo',default='dsv41-deploy-offload')
    ap.add_argument('--expected-experiment-commit')
    ap.add_argument('--native-evidence',type=Path)
    ap.add_argument('--native-wait-seconds',type=float,default=0,help='bounded coordinator evidence checkpoint; default fail immediately')
    ap.add_argument('--qualification',type=Path,help='completed prior diagnostic run, for measured-only window')
    ap.add_argument('--data-dir',type=Path,default=Path('/opt/dsv41/data/bench-data'))
    args = argparse.Namespace(out=None,execute=False,screen=False)
    supervisor: Supervisor | None = None
    created_out: Path | None = None
    try:
        args = ap.parse_args()
        ck.require(args.contexts is None or args.phase == 'measured', '--contexts requires --phase measured')
        contexts = tuple(args.contexts) if args.contexts is not None else (150000, 300000)
        ck.measured_contexts(contexts)
        if not args.execute:
            print(plan(args.phase,screen=args.screen,contexts=contexts))
            return 0
        coordinator()
        ck.require(0 <= args.native_wait_seconds <= 900,'native wait must be0..900 seconds')
        ck.require(args.out is not None,'fresh --out required')
        out = cast(Path,args.out)
        ck.require(out.resolve().is_relative_to(CAMPAIGN) and out.resolve() != CAMPAIGN,'output must be private campaign directory')
        ck.require(re.fullmatch('[A-Za-z0-9_./-]+',args.remote_repo) is not None and
                   not args.remote_repo.startswith('/') and '..' not in args.remote_repo,'unsafe remote checkout')
        ck.require(re.fullmatch('[0-9a-f]{40}',args.expected_experiment_commit or '') is not None,'full reviewed experiment commit required')
        local_revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        ck.require(local_revision == args.expected_experiment_commit,'local reviewed checkout revision differs')
        ck.require(not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip(),
                   'local tracked source is dirty; review and commit before execution')
        ck.require(args.screen or args.native_evidence is not None or args.phase == 'measured','native evidence directory required; unavailable = HOLD')
        if args.phase != 'measured' and not args.screen:
            ck.require(args.native_evidence.is_dir(), 'native apparatus/evidence directory unavailable; NOT RUN/HOLD')
        if args.phase == 'measured':
            ck.require(args.qualification is not None,'prior diagnostic qualification required')
            validate_qualification(args.qualification,args.expected_experiment_commit,screen=args.screen)
        # Validate corpus availability on CPU before any start. Never download/substitute filler.
        from offload.prefill_trace.window_requests import corpus
        corpus(args.data_dir)
        out.mkdir(parents=False,exist_ok=False)
        created_out = out
        # Read the already-open HOLD once; its timestamp, not runner start time,
        # owns the reservation. Reinvocation never renews an existing window.
        hold = subprocess.run([*SSH,'sparkhead','cat ~/dsv41-deploy/HOLD'],
            check=True,text=True,capture_output=True,timeout=8).stdout
        opened,submit,hard = ck.window_deadlines(hold,time.time())
        save(out/'deadlines.json',{'opened':opened,'submission':submit,'hard':hard},screen=args.screen)
        (out/'coordinator-next.sh').write_text('#!/usr/bin/env bash\n# Review saved rollback.json first; production still coordinator-owned.\n'+RESTORE+'\n')
        def checkpoint(data: dict[str, Json]) -> None:
            with (out/'checkpoints.jsonl').open('a') as handle:
                handle.write(json.dumps(ck.screen_record(data,args.screen),allow_nan=False)+'\n')
        def interrupted(signum: int, _frame: object) -> None:
            raise RuntimeError(f'interrupted by signal{signum}')
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP): signal.signal(sig,interrupted)
        supervisor = Supervisor(out,hard)
        supervisor.begin()
        live = Live(out,args.remote_repo,args.expected_experiment_commit,hard,submit,
                    args.native_evidence or out,args.data_dir,supervisor,args.native_wait_seconds,screen=args.screen)
        live.preflight()
        result = sequence(live,args.phase,checkpoint,screen=args.screen,contexts=contexts)
        result.update({'manifest_sha256':ck.digest(ROOT/'offload/upstream_adopt/MANIFEST.json'),
            'config_sha256':ck.digest(ROOT/ck.CONFIG),'worker_source':ck.WORKER,
            'experiment_commit':args.expected_experiment_commit,
            'native_evidence_root':str(live.native_root.resolve()),
            'artifact_sha256':{p.name:ck.digest(p) for p in out.iterdir() if p.is_file() and
                not p.name.startswith('memory-') and p.suffix in ('.json','.jsonl')}})
        save(out/'summary.json',result,screen=args.screen)
        if result.get('diagnostics') in ('PASS','PASS-SCREEN'): save(out/'qualification.json',result,screen=args.screen)
        print('Sequence complete. HOLD; inspect private evidence and restore production.')
        return 0
    except (Exception,KeyboardInterrupt) as exc:
        if created_out is not None and not (created_out/'STOP').exists():
            with (created_out/'STOP').open('x') as handle:
                handle.write(f'{type(exc).__name__}: {exc}\n')
        print(str(exc),file=sys.stderr)
        return 1
    finally:
        try:
            if supervisor is not None: supervisor.close()
            if created_out is not None and (created_out/'requests.jsonl').is_file():
                events = [record(cast(Json,json.loads(x))) for x in (created_out/'requests.jsonl').read_text().splitlines()]
                save(created_out/'accounting.json',ck.accounting(events),screen=args.screen)
        finally:
            print('Coordinator restoration commands (runner never executes these):\n'+RESTORE)
            print('Require K1 START VERIFIED; cap327680; KV2502600; greedy6/6 vs K1-production/greedy-post.jsonl; PRODUCTION HOLD OK before close.')


if __name__ == '__main__':
    raise SystemExit(main())
