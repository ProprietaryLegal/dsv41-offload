"""W10 coordinator window orchestration; offline rehearsal is the default.

Restoration is always separate. A failure retains HOLD and all private evidence.
No x86host service/GPU command, production generation, or automatic restore exists.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import tempfile
import time
from typing import Protocol, cast

from .decode_dataset import digest, mapping, obj, text
from .decode_smoke import checked_spec, window_gate
from .replay_window import CAMPAIGN, IMAGE, LiveIO, gauges, require
from .smoke_prepare import PRIVATE

COORDINATOR = 'Deepseek/coordinator'
RANKS = ('sparkhead', 'sparkworker-wired')
FILES = ('capture/speculator_replay.py', 'capture/decode_capture.py', 'spark/remote_dspark/speculator.py')
RESTORE = 'bash /opt/dsv41/campaigns/DSV41-OFFLOAD/W8/restore.sh'
SCRATCH = CAMPAIGN/'scratch/D2-SMOKE-R2'


def write(path: Path, value: object) -> None:
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
    path.chmod(0o600)


def checkout_command(source: str, bundle_sha: str) -> str:
    script = f'''set -euo pipefail
cd "$HOME/dsv41-deploy-offload"
test -z "$(git status --porcelain --untracked-files=all)"
test "$(sha256sum "$HOME/dsv41-offload-run/w10-{source}.bundle" | cut -d' ' -f1)" = {shlex.quote(bundle_sha)}
git fetch --no-tags "$HOME/dsv41-offload-run/w10-{source}.bundle" HEAD
test "$(git rev-parse FETCH_HEAD)" = {source}
git -c advice.detachedHead=false checkout --detach {source}
test "$(git rev-parse HEAD)" = {source}
test -z "$(git status --porcelain --untracked-files=all)"
'''
    return shlex.join(['bash','-c',script])


def stage_command(repo: Path, source: str, run_id: str) -> str:
    remote = (repo/'offload/dspark/spark/remote_dspark/speculator.py').read_bytes()
    prefix = remote.split(b'# Capacity mode does not call DSparkSpeculator.propose.')[0]
    require(hashlib.sha256(prefix[:-1]).hexdigest() ==
            '441be6b7d245114974c5dcb307c2efebd29516b958cd363bc70e24bcd73bdcf3',
            'remote speculator pre-hook prefix differs from pinned production')
    payload = {name: {'sha256':digest(repo/'offload/dspark'/name),
                      'bytes':base64.b64encode((repo/'offload/dspark'/name).read_bytes()).decode()}
               for name in FILES}
    # Only small source files are written. Never read/copy model tensors.
    script = f'''import base64,hashlib,json,os,pathlib,shutil,stat
os.umask(0o077)
root=pathlib.Path.home()/'dsv41-decode'/{source!r}
capture=pathlib.Path.home()/'dsv41-offload-run'/('d2-decode-'+{run_id!r})
assert not root.is_symlink() and not root.parent.is_symlink(), 'source symlink refused'
assert not capture.exists() and not capture.is_symlink(), 'capture run already exists'
assert shutil.disk_usage(capture.parent).free >= 1024**3, 'less than 1 GiB capture disk headroom'
payload=json.loads({json.dumps(payload)!r})
if root.exists():
 for name,entry in payload.items():
  p=root/name
  assert not p.is_symlink() and p.is_file() and p.stat().st_uid==os.getuid(), 'unsafe existing source'
  assert p.stat().st_mode & 0o222 == 0, 'existing source is mutable'
  assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'], 'existing source bytes differ'
else:
 root.mkdir(parents=True,mode=0o700)
 for name,entry in payload.items():
  p=root/name
  p.parent.mkdir(parents=True,mode=0o700,exist_ok=True)
  with p.open('xb') as f: f.write(base64.b64decode(entry['bytes'],validate=True))
  p.chmod(0o444)
  assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'], 'staged source hash differs'
 for p in sorted(root.rglob('*'),reverse=True):
  if p.is_dir(): p.chmod(0o555)
 root.chmod(0o555)
capture.mkdir(mode=0o700)
s=capture.stat()
assert s.st_uid==os.getuid() and stat.S_IMODE(s.st_mode)==0o700 and not list(capture.iterdir()), 'unsafe capture directory'
print(json.dumps({{'source':{source!r},'run':{run_id!r},'capture_mode':oct(stat.S_IMODE(s.st_mode)),
 'free_bytes':shutil.disk_usage(capture).free,'source_sha256':{{k:v['sha256'] for k,v in payload.items()}}}}))
'''
    return shlex.join(['python3','-c',script])


def container_command() -> str:
    script = "import json,subprocess; c=json.loads(subprocess.check_output(['docker','inspect','dsv41-serve']))[0]; print(json.dumps({'id':c['Id'],'image':c['Image'],'state':c['State']['Status'],'mounts':c['Mounts']}))"
    return shlex.join(['python3','-c',script])


def candidate(record: dict[str, object], run_id: str, old_id: str) -> str:
    cid = text(record['id'])
    require(re.fullmatch('[0-9a-f]{64}', cid) is not None and cid != old_id, 'not a new experiment container')
    require(record.get('image') == IMAGE, 'unexpected candidate image')
    mounts = cast(list[dict[str, object]], record['mounts'])
    require(any(m.get('Destination') == '/d2-decode' and m.get('RW') is True and
                text(m.get('Source')).endswith('/dsv41-offload-run/d2-decode-'+run_id) for m in mounts),
            'candidate does not own this run capture mount')
    return cid


class WindowIO(Protocol):
    synthetic: bool
    def now(self) -> float: ...
    def stamp(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...
    def metrics(self) -> str: ...
    def memcheck(self) -> None: ...
    def call(self, name: str, host: str, command: str, timeout: int = 60, supervised: bool = False) -> str: ...
    def bundle(self, repo: Path, source: str, out: Path) -> str: ...
    def smoke(self, repo: Path, source: str, run_id: str, inputs: Path, window: Path, out: Path) -> None: ...


class LiveWindow(LiveIO):
    synthetic = False

    def stamp(self) -> float:
        return time.time()

    def local(self, name: str, argv: list[str], timeout: int) -> None:
        """Supervise only our child group, including memory/STOP during transfers."""
        self.memcheck()
        with (self.out/(name+'.log')).open('x') as log:
            child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = self.now()+timeout
                while child.poll() is None:
                    require(self.now() < deadline, name+' timeout')
                    self.memcheck()
                    self.sleep(.5)
                require(child.returncode == 0, name+' refused; inspect private '+name+'.log')
            finally:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    child.wait(timeout=8)

    def bundle(self, repo: Path, source: str, out: Path) -> str:
        path = out/'source.bundle'
        # A raw SHA alone has no advertised ref and produces an empty bundle.
        # Export HEAD, then prove its advertised object is the reviewed SHA.
        self.local('create-source-bundle',['git','-C',str(repo),'bundle','create',str(path),'HEAD'],60)
        head = subprocess.check_output(['git','bundle','list-heads',str(path),'HEAD'],text=True,timeout=5).strip()
        require(head == source+' HEAD','source HEAD changed while creating bundle')
        # Directory is newly created by this operation, not production storage.
        self.call('bundle-parent','sparkhead','mkdir -p "$HOME/dsv41-offload-run"', supervised=True)
        self.local('transfer-source-bundle',['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=5',str(path),
                        'sparkhead:dsv41-offload-run/w10-'+source+'.bundle'],60)
        return digest(path)

    def smoke(self, repo: Path, source: str, run_id: str, inputs: Path, window: Path, out: Path) -> None:
        # Its existing owned-command supervision covers every HTTP/copy step.
        self.local('decode-smoke',['bash',str(repo/'offload/dspark/d2/decode_smoke.sh'),'--execute',
                '--source',source,'--run',run_id,'--inputs',str(inputs),'--window',str(window),'--out',str(out)],1200)
        report = obj(out/'report.json')
        require(report.get('synthetic') is False and report.get('pass') is True, 'smoke did not qualify')


class DryWindow:
    """No subprocess or network capability. Time and rank identities are synthetic."""
    synthetic = True

    def __init__(self, run_id: str) -> None:
        self.elapsed = 0.0
        self.run_id = run_id
        self.calls: list[str] = []
        self.started = False

    def now(self) -> float:
        return self.elapsed

    def stamp(self) -> float:
        return 1800000000+self.elapsed

    def sleep(self, seconds: float) -> None:
        self.elapsed += seconds

    def metrics(self) -> str:
        return 'vllm:num_requests_running{} 0\nvllm:num_requests_waiting{} 0\n'

    def memcheck(self) -> None:
        self.calls.append('memory-both-ranks')

    def call(self, name: str, host: str, command: str, timeout: int = 60, supervised: bool = False) -> str:
        self.calls.append(name)
        print('DRY-RUN synthetic: '+name+' on '+host)
        if name == 'start-exp': self.started = True
        if name.startswith(('original-container-', 'candidate-container-', 'cleanup-container-')):
            return json.dumps({'id': ('b' if self.started else 'a')*64,'image':IMAGE,'state':'running',
                'mounts':[{'Destination':'/d2-decode','RW':True,
                           'Source':'/home/synthetic/dsv41-offload-run/d2-decode-'+self.run_id}]})
        return 'synthetic success\n'

    def bundle(self, repo: Path, source: str, out: Path) -> str:
        self.calls.append('bundle-transfer')
        return '0'*64

    def smoke(self, repo: Path, source: str, run_id: str, inputs: Path, window: Path, out: Path) -> None:
        self.calls.append('decode-smoke')
        print('DRY-RUN synthetic: decode_smoke.sh --execute (not invoked)')


def run(io: WindowIO, repo: Path, source: str, run_id: str, inputs: Path, out: Path) -> dict[str, object]:
    require(re.fullmatch('[0-9a-f]{40}', source) is not None, 'full reviewed commit SHA required')
    require(re.fullmatch('[A-Za-z0-9_-]{1,80}', run_id) is not None, 'unsafe NEW run ID')
    samples: list[dict[str, object]] = []
    start = io.now()
    while True:
        io.memcheck()
        metrics = io.metrics()
        require(gauges(metrics), 'production busy/invalid during drain; refusing window')
        stamp = io.stamp()
        if samples:
            require(0 < stamp-float(cast(float,samples[-1]['t'])) <= 10, 'drain telemetry gap above ten seconds')
        samples.append({'t':stamp,'metrics':metrics})
        if io.now()-start >= 300 and stamp-float(cast(float,samples[0]['t'])) >= 300:
            break
        io.sleep(5)
    window = {'coordinator':COORDINATOR,'source':source,'created_unix':io.stamp(),'drain_samples':samples}
    window_gate(window,source,io.stamp())
    window_path = out/'window.json'
    write(window_path,window)
    opened, attempted_start = False, False
    original: dict[str, str] = {}
    stopped: list[str] = []
    try:
        # Never borrow another coordinator's already-open HOLD/window.
        io.call('no-existing-hold','sparkhead','test ! -e "$HOME/dsv41-deploy/HOLD"',supervised=True)
        require(gauges(io.metrics()), 'foreign work after drain; refusing open')
        io.call('open-window','sparkhead','cd "$HOME/dsv41-deploy" && bash offload/launch/window.sh open '+
                shlex.quote('W10 D2 decode smoke '+run_id),timeout=30,supervised=True)
        opened = True
        bundle_sha = io.bundle(repo,source,out)
        io.call('reviewed-checkout','sparkhead',checkout_command(source,bundle_sha),timeout=60,supervised=True)
        stage = stage_command(repo,source,run_id)
        for node in RANKS:
            io.call('disk-'+node,node,'df -B1 "$HOME/dsv41-offload-run"',timeout=10,supervised=True)
            io.call('stage-'+node,node,stage,timeout=30,supervised=True)
            original[node] = text(mapping(json.loads(io.call('original-container-'+node,node,container_command(),supervised=True)))['id'])
        require(gauges(io.metrics()), 'foreign work after staging; refusing experiment')
        window_gate(window,source,io.stamp())
        attempted_start = True
        io.call('start-exp','sparkhead', 'cd "$HOME/dsv41-deploy-offload" && D2_DECODE_RUN='+shlex.quote(run_id)+
                ' bash offload/launch/start_exp.sh offload/launch/configs/D-decode-capture.env </dev/null',
                timeout=900,supervised=True)
        for node in RANKS:
            candidate(mapping(json.loads(io.call('candidate-container-'+node,node,container_command(),supervised=True))),run_id,original[node])
        io.smoke(repo,source,run_id,inputs,window_path,out/'smoke')
    finally:
        # Under low-memory/STOP conditions cleanup must still stop ONLY identified
        # candidates. It deliberately bypasses the memory gate for exact-ID stop.
        errors: list[str] = []
        if attempted_start:
            for node in RANKS:
                try:
                    current = mapping(json.loads(io.call('cleanup-container-'+node,node,container_command(),timeout=10)))
                    if current.get('id') == original[node]:
                        continue  # start refused before replacing the original
                    cid = candidate(current,run_id,original[node])
                    stop = ('test "$(docker inspect --format \'{{.Id}}\' dsv41-serve)" = '+shlex.quote(cid)+
                            ' && docker stop --time 30 '+shlex.quote(cid))
                    io.call('stop-owned-'+node,node,stop,timeout=40)
                    stopped.append(node)
                except (ValueError,OSError,subprocess.SubprocessError) as exc:
                    errors.append(node+': '+str(exc))
        write(out/'window-state.json', {'synthetic':io.synthetic,'window_opened':opened,
            'experiment_start_attempted':attempted_start,'stopped_ranks':stopped,'cleanup_errors':errors,
            'restore_separate':True})
        print('HOLD stays open after opening. Restore separately, in order:')
        print(RESTORE+' start\n'+RESTORE+' check\n'+RESTORE+' close')
        require(not errors,'candidate cleanup refused; HOLD retained: '+'; '.join(errors))
    report: dict[str, object] = {'schema':'d2.smoke-window.v1','synthetic':io.synthetic,
        'pass':True,'source':source,'run':run_id,'drain_samples':len(samples),
        'drain_seconds':float(cast(float,samples[-1]['t']))-float(cast(float,samples[0]['t'])),
        'stopped_ranks':stopped,'restore_separate':True}
    write(out/'window-report.json',report)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--dry-run',action='store_true')
    mode.add_argument('--execute',action='store_true')
    ap.add_argument('--source',default='')
    ap.add_argument('--run',default='synthetic-w10')
    ap.add_argument('--inputs',type=Path,default=PRIVATE/'smoke-input-r2-qualified.json')
    ap.add_argument('--out',type=Path)
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[3]
    os.umask(0o077)
    if not a.execute:
        SCRATCH.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='window-dry-',dir=SCRATCH) as scratch:
            print('SYNTHETIC OFFLINE REHEARSAL: no HTTP/SSH/subprocess/service/GPU operation')
            source = a.source or 'f'*40
            report = run(DryWindow(a.run),repo,source,a.run,a.inputs,Path(scratch))
            print(json.dumps(report))
        return
    require(os.environ.get('D2_COORDINATOR') == COORDINATOR,'coordinator-only execute')
    require(a.source and subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip() == a.source,
            'reviewed source differs from local HEAD')
    require(not subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all'],text=True).strip(),
            'local source checkout is dirty')
    require((CAMPAIGN/'D2-GEN.PAUSE').exists(),'pause D2 generation and save/abort own streams before window')
    require(a.inputs.resolve().is_relative_to(PRIVATE) and (PRIVATE.stat().st_mode & 0o777) == 0o700,
            'mode-700 private W10 inputs required')
    checked_spec(obj(a.inputs))
    require(a.out is not None and a.out.resolve().is_relative_to(PRIVATE), 'NEW private output directory required')
    a.out.mkdir(mode=0o700,parents=True,exist_ok=False)
    report = run(LiveWindow(a.out,a.source),repo,a.source,a.run,a.inputs,a.out)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
