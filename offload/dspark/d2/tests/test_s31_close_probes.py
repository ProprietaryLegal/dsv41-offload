"""S31 independent probes of W12/decode_close.sh with fake hosts (no host access).

A COPY of the campaign script runs NON-dry with only two lines rewritten (campaign root and generator state
path, both pointing into tmp). ssh/curl/tmux/docker/git are fake executables; sparkhead and sparkworker commands run
locally against a scratch copy of the reviewed K1 base root, through the real decode_apply.py (8174771) and the
real window.sh/manage.py/start.sh semantics emulated by small hooks. Nothing here can reach a host: every
network tool on PATH is a stub and the remote PATH has fatal ssh/curl stubs.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/decode_close.sh')
BASE_COPY = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/private/base-copy')
REPO = Path(__file__).resolve().parents[4]
DECODE_APPLY = REPO/'offload/dspark/d2/decode_apply.py'
REAL_GREEDY = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD/W5/greedy_check.py')
SCRIPT_SHA = 'a30a7b66b8a5bd4776022aa02327d941fc55a26c2ce8df899ffafb0659e27cfc'
APPLY_SHA = '6cf189790fc09f41fe7869df921ee7ed66eef302907b0e0613abdf5848e8e66e'
RUN = 'd2pass2-r1'
SRC = '07fc7d6abc369640e1d0129ced197d4c21e892df'
SYSPY = '/usr/bin/python3'

pytestmark = pytest.mark.skipif(not SCRIPT.exists() or not BASE_COPY.exists(), reason='campaign artefacts absent')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exe(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o700)


SSH = r'''#!/usr/bin/python3
import json, os, subprocess, sys
from pathlib import Path
T = Path(os.environ['S31_T'])
args = sys.argv[1:]
while args and args[0].startswith('-'):
    args = args[2:] if args[0] in ('-o', '-b', '-c') else args[1:]
host, command = args[0], ' '.join(args[1:])
with (T/'ssh.log').open('a') as log:
    log.write(json.dumps([host, command])+'\n')
state = json.loads((T/'state.json').read_text())
if host == 'x86host':
    if 'prod_window.sh check' in command:
        print('PRODUCTION HOLD OK' if state['prod_window_ok'] else 'PRODUCTION HOLD NOT OK')
        sys.exit(0 if state['prod_window_ok'] else 1)
    if 'decode_copy' in command:
        ok = state.get('copy_ok', True)
        print(('FLUSH VERIFIED x' if '--flush' in command else 'COPY VERIFIED x') if ok else 'RuntimeError: copy refused')
        sys.exit(0 if ok else 1)
    sys.exit('unexpected x86host command: '+command)
if host not in ('sparkhead', 'sparkworker-wired'):
    sys.exit('unexpected host '+host)
home = T/host
rewrites = [('offload/launch/window.sh', '"$S31_HOOKS"/window.sh'), ('./stop.sh', '"$S31_HOOKS"/stop.sh'),
            ('./start.sh', '"$S31_HOOKS"/start.sh'),
            ('python3 offload/promote/k1/manage.py', 'python3 "$S31_HOOKS"/manage.py')]
for old, new in rewrites:
    command = command.replace(old, new)
env = {'HOME': str(home), 'PATH': str(T/'remotebin')+':/usr/bin:/bin', 'S31_T': str(T),
       'S31_HOOKS': str(T/'hooks'), 'S31_HOST': host, 'LC_ALL': 'C'}
sys.exit(subprocess.run(['bash', '-c', command], cwd=home, env=env).returncode)
'''

CURL = r'''#!/usr/bin/python3
import json, os, sys
from pathlib import Path
T = Path(os.environ['S31_T'])
state = json.loads((T/'state.json').read_text())
url = [a for a in sys.argv[1:] if a.startswith('http')][0]
run = state['running']['sparkhead']
if run is None:
    sys.exit(7)
if url.endswith('/metrics'):
    busy = 1 if state['busy'] else 0
    print('vllm:num_requests_running{engine="0",model_name="m"} %d.0' % busy)
    print('vllm:num_requests_waiting{engine="0",model_name="m"} 0.0')
elif url.endswith('/v1/models'):
    print(json.dumps({'object': 'list', 'data': [{'id': 'DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2',
        'root': '/srv/dsv41/legal-v2a', 'max_model_len': 327680 if run['k1'] else 262144}]}))
else:
    sys.exit('unexpected url '+url)
'''

TMUX = r'''#!/usr/bin/python3
import json,os,sys
from pathlib import Path
T=Path(os.environ['S31_T']); s=json.loads((T/'state.json').read_text())
if sys.argv[1:] == ['has-session','-t','=d2-gen2']:
    sys.exit(0 if s['generator'] else 1)
assert sys.argv[1:] == ['send-keys','-t','=d2-gen2:','C-c'], sys.argv   # S33: pane target needs the colon
assert (T/'C/D2-GEN2.PAUSE').exists()
s['generator']=False; s['sigint_saved']=True
(T/'state.json').write_text(json.dumps(s))
(T/'C/D2-GEN2.out.log').write_text('D2-GEN2-EXIT 0\n')
'''

FATAL = '#!/bin/bash\necho "S31 FATAL: real network tool reached: $0 $*" >&2\nexit 97\n'

DOCKER = r'''#!/usr/bin/python3
import json, os, sys
from pathlib import Path
T = Path(os.environ['S31_T'])
host = os.environ['S31_HOST']
state = json.loads((T/'state.json').read_text())
run = state['running'][host]
if sys.argv[1:3] not in (['inspect', 'dsv41-serve'], ['logs', 'dsv41-serve']):
    sys.exit('unexpected docker '+' '.join(sys.argv[1:]))
if run is None:
    print('Error: No such object: dsv41-serve', file=sys.stderr)
    sys.exit(1)
if sys.argv[1] == 'logs':
    print('INFO [kv_cache_utils.py:1] GPU KV cache size: %s tokens' % ('2,502,600' if run['k1'] else '989,460'))
    sys.exit(0)
env = ['PATH=/usr/bin']
mounts = [{'Type': 'bind', 'Source': '/home/u/dsv41-offload-prod/dspark/remote_dspark',
           'Destination': '/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/remote_dspark'}]
if run['capture']:
    env.append('VLLM_DSPARK_DECODE_CAPTURE_DIR=/d2-decode')
    mounts.append({'Type': 'bind', 'Source': '/home/u/dsv41-offload-run/d2-decode-r', 'Destination': '/d2-decode'})
print(json.dumps([{'Config': {'Env': env}, 'Mounts': mounts}]))
'''

GIT = r'''#!/usr/bin/python3
import json, os, sys
from pathlib import Path
state = json.loads((Path(os.environ['S31_T'])/'state.json').read_text())
if sys.argv[1:] in (['diff', '--quiet'], ['diff', '--cached', '--quiet']):
    sys.exit(0 if state['tool_clean'] else 1)
if sys.argv[1:] == ['rev-parse', '--short=7', 'HEAD']:
    print(state['tool_head'])
    sys.exit(0)
sys.exit('unexpected git '+' '.join(sys.argv[1:]))
'''

# Emulation of the production start/stop/window/K1 semantics at 21bed31 (start_checked.sh, manage.py, window.sh).
COMMON = r'''
import hashlib, json, os, shutil, sys
from pathlib import Path
T = Path(os.environ['S31_T'])
ROOT = T/'sparkhead'/'dsv41-deploy'
def load(): return json.loads((T/'state.json').read_text())
def save(s): (T/'state.json').write_text(json.dumps(s))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def record(): return json.loads((ROOT/'logs/k1-promotion/files.json').read_text())
def k1_ok():
    r = record()
    return (r.get('profile') == '327680' and sha(ROOT/'dsv41.env') == r['promoted']['dsv41.env']
            and sha(ROOT/'start.sh') == r['promoted']['start.sh']
            and sha(ROOT/'offload/kvpack/MANIFEST.json') == '559c5867ba42aa7daa7b4a7c8231d15ee1d35be40ae5b715b013f973193c58de')
def capture_env(): return b'VLLM_DSPARK_DECODE_CAPTURE_DIR' in (ROOT/'dsv41.env').read_bytes()
'''

START = COMMON + r'''
s = load()
if any(v is not None for v in s['running'].values()):
    print('K1 start refuses: already running', file=sys.stderr); sys.exit(1)
r = record()
if sha(ROOT/'start.sh') == r['original']['start.sh']:
    # pre-K1 launcher: no K1 guard, no K1 message
    s['running'] = {h: {'capture': capture_env(), 'k1': False} for h in s['running']}; save(s)
    print('previous (pre-K1) configuration started'); sys.exit(0)
if k1_ok() and s['inject_k1_failures'] == 0:
    s['running'] = {h: {'capture': capture_env(), 'k1': True} for h in s['running']}; save(s)
    print('K1 START VERIFIED; complete the production-window probes before releasing HOLD'); sys.exit(0)
if s['inject_k1_failures']:
    s['inject_k1_failures'] -= 1
print('K1 START FAILED (status 1, line 30); stopping candidate and restoring previous config', file=sys.stderr)
# manage.py restore: current files must be original or promoted, then the pre-K1 backups are restored
for name in ('dsv41.env', 'start.sh'):
    if sha(ROOT/name) not in (r['original'][name], r['promoted'][name]):
        save(s); print('K1 FALLBACK FAILED: backup/edits need operator review', file=sys.stderr); sys.exit(1)
for name in ('dsv41.env', 'start.sh'):
    shutil.copy2(ROOT/'logs/k1-promotion'/name, ROOT/name)
s['running'] = {h: {'capture': capture_env(), 'k1': False} for h in s['running']}; save(s)
print('K1 REFUSED; previous configuration restarted. Inspect log; keep HOLD until rollback checks pass.', file=sys.stderr)
sys.exit(1)
'''

STOP = COMMON + r'''
s = load()
if s['stop_partial']:
    s['stop_partial'] = False
    s['running']['sparkhead'] = None; save(s)
    print('dsv41_tp2.sh down: worker ssh failed', file=sys.stderr); sys.exit(1)
s['running'] = {h: None for h in s['running']}; save(s)
'''

WINDOW = COMMON + r'''
s = load()
if sys.argv[1] == 'open':
    if s['running']['sparkhead'] is not None and s['busy']:
        print('serve still busy after 60 min; not opening', file=sys.stderr); sys.exit(1)
    (ROOT/'HOLD').write_text('DSV41-OFFLOAD (session coordinator): '+sys.argv[2]+'\n')
    s['hold_procs'] = True; save(s); print('window open'); sys.exit(0)
if sys.argv[1] == 'close':
    if s['running']['sparkhead'] is None:
        print("production not up on :8000 with legal-v2a (root=''); window stays open", file=sys.stderr); sys.exit(1)
    (ROOT/'HOLD').unlink(missing_ok=True); s['hold_procs'] = False; save(s); print('window closed'); sys.exit(0)
sys.exit('unexpected window '+' '.join(sys.argv[1:]))
'''

MANAGE = COMMON + r'''
args = sys.argv[1:]
assert args[0] == 'verify' and args[args.index('--profile')+1] == '327680' and Path(args[args.index('--root')+1]) == ROOT, args
if not k1_ok():
    print('K1 REFUSED: promoted dsv41.env changed since apply; refusing start', file=sys.stderr); sys.exit(1)
print('K1 profile 327680, promotion record, saved rollback and reviewed checkout verified')
'''

GREEDY = r'''#!/usr/bin/python3
import json, os, sys
from pathlib import Path
real = {}
exec(compile(Path(os.environ['S31_REAL_GREEDY']).read_text(), 'greedy_check', 'exec'), real)
if sys.argv[1] == '--compare':
    real['compare'](sys.argv[2], sys.argv[3]); sys.exit(0)
T = Path(os.environ['S31_T'])
s = json.loads((T/'state.json').read_text())
run = s['running']['sparkhead']
rows = [json.loads(l) for l in (T/'C/K1-production/greedy-post.jsonl').read_text().splitlines()]
with open(sys.argv[2], 'w') as out:
    for r in rows:
        if not (run and run['k1'] and s['greedy_ok']):
            r['text'] += ' diverged'
        out.write(json.dumps(r)+'\n')
'''


class Fake:
    def __init__(self, tmp: Path) -> None:
        assert sha(SCRIPT) == SCRIPT_SHA and sha(DECODE_APPLY) == APPLY_SHA
        self.t = tmp
        self.c = tmp/'C'
        (self.c/'W12').mkdir(parents=True)
        (self.c/'W5').mkdir()
        (self.c/'K1-production').mkdir()
        (self.c/'K1-production/greedy-post.jsonl').write_text(
            ''.join(json.dumps({'i': i, 'text': f'synthetic row {i}', 'usage': {}})+'\n' for i in range(6)))
        exe(self.c/'W5/greedy_check.py', GREEDY)
        self.state_dir = tmp/'gen-state'
        self.state_dir.mkdir()
        text = SCRIPT.read_text()
        old_c = 'C=/opt/dsv41/campaigns/DSV41-OFFLOAD\n'
        old_s = 'STATE=/opt/dsv41/data/d2/gen-decode-07fc7d6-$RUN\n'
        assert text.count(old_c) == 1 and text.count(old_s) == 1
        text = text.replace(old_c, f'C={self.c}\n').replace(old_s, f'STATE={self.state_dir}\n')
        self.script = self.c/'W12/decode_close.sh'
        exe(self.script, text)
        bin_ = tmp/'bin'
        bin_.mkdir()
        exe(bin_/'ssh', SSH)
        exe(bin_/'curl', CURL)
        exe(bin_/'tmux', TMUX)
        exe(bin_/'sleep', '#!/bin/bash\nexit 0\n')
        remote = tmp/'remotebin'
        remote.mkdir()
        exe(remote/'docker', DOCKER)
        exe(remote/'git', GIT)
        exe(remote/'ssh', FATAL)
        exe(remote/'curl', FATAL)
        hooks = tmp/'hooks'
        hooks.mkdir()
        for name, body in (('start.sh', START), ('stop.sh', STOP), ('window.sh', WINDOW), ('manage.py', MANAGE)):
            exe(hooks/name, '#!/usr/bin/python3\n'+body)
        for host in ('sparkhead', 'sparkworker-wired'):
            (tmp/host).mkdir()
        self.root = tmp/'sparkhead/dsv41-deploy'
        self._production_root()
        self.state = {'running': {'sparkhead': {'capture': True, 'k1': True}, 'sparkworker-wired': {'capture': True, 'k1': True}},
                      'busy': False, 'generator': False, 'prod_window_ok': True, 'greedy_ok': True,
                      'tool_clean': True, 'tool_head': '8174771', 'inject_k1_failures': 0, 'stop_partial': False,
                      'hold_procs': False, 'copy_ok': True}
        self.save()
        (self.c/'D2-GEN2.PAUSE').write_text('coordinator\n')

    def _production_root(self) -> None:
        """K1 base (real bytes from W12/private/base-copy) + synthetic pre-K1 originals, then the REAL apply."""
        root = self.root
        (root/'offload/kvpack').mkdir(parents=True)
        (root/'logs/k1-promotion').mkdir(parents=True)
        for rel in ('dsv41.env', 'start.sh', 'offload/kvpack/MANIFEST.json'):
            shutil.copy2(BASE_COPY/rel, root/rel)
        (root/'logs/k1-promotion/dsv41.env').write_text('# synthetic pre-K1 env\n')
        (root/'logs/k1-promotion/start.sh').write_text('# synthetic pre-K1 start\n')
        record = json.loads((BASE_COPY/'logs/k1-promotion/files.json').read_text())
        record['original'] = {n: sha(root/'logs/k1-promotion'/n) for n in ('dsv41.env', 'start.sh')}
        (root/'logs/k1-promotion/files.json').write_text(json.dumps(record, indent=2)+'\n')
        self.base_record = sha(root/'logs/k1-promotion/files.json')
        tool = self.t/'sparkhead/dsv41-deploy-offload/offload/dspark/d2'
        tool.mkdir(parents=True)
        shutil.copy2(DECODE_APPLY, tool/'decode_apply.py')
        (tool/'__init__.py').write_text('')
        render = subprocess.run([SYSPY, '-m', 'offload.dspark.d2.decode_apply', '--root', str(root), '--source', SRC,
                                 '--run', RUN], cwd=self.t/'sparkhead/dsv41-deploy-offload', text=True,
                                capture_output=True, check=True).stdout
        capture_sha = render.strip().splitlines()[-1].split('=')[1]
        review = self.t/'review.json'
        review.write_text(json.dumps({'schema': 'd2.decode-apply-review.v1', 'approved': True,
            'coordinator': 'Deepseek/coordinator', 'source': SRC, 'base_commit': '21bed313ac803d38327c4de1ff2e5235f329c716',
            'base_env_sha256': 'faa9d3e0019a8665f0b61dbf46a79134680731d43b9a9cb99ca2fa5af85373a8',
            'capture_env_sha256': capture_sha}))
        subprocess.run([SYSPY, '-m', 'offload.dspark.d2.decode_apply', '--root', str(root), '--source', SRC, '--run', RUN,
                        '--apply', '--review', str(review)], cwd=self.t/'sparkhead/dsv41-deploy-offload', check=True)
        assert sha(root/'dsv41.env') == capture_sha

    def save(self) -> None:
        (self.t/'state.json').write_text(json.dumps(self.state))

    def load(self) -> dict:
        self.state = json.loads((self.t/'state.json').read_text())
        return self.state

    def phase(self, phase: str, nonce: str = 'n1') -> subprocess.CompletedProcess[str]:
        env = {'PATH': f'{self.t/"bin"}:/usr/bin:/bin', 'HOME': str(self.t), 'S31_T': str(self.t),
               'S31_REAL_GREEDY': str(REAL_GREEDY), 'D2_COORDINATOR': 'Deepseek/coordinator', 'LC_ALL': 'C'}
        result = subprocess.run(['/bin/bash', str(self.script), '--nonce', nonce, phase], env=env, text=True,
                                capture_output=True, timeout=120)
        self.load()
        assert 'S31 FATAL' not in result.stderr
        return result

    def ssh_log(self) -> list[list[str]]:
        path = self.t/'ssh.log'
        return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []

    def files(self) -> dict[str, str]:
        r = self.root
        return {'env': sha(r/'dsv41.env'), 'start': sha(r/'start.sh'), 'record': sha(r/'logs/k1-promotion/files.json')}

    def mark(self, phase: str, nonce: str = 'n1') -> None:
        e = self.c/f'W12/private/close-versions/evidence-{RUN}'
        e.mkdir(parents=True, exist_ok=True)
        (e/f'{phase}.ok').write_text(nonce+'\n')


@pytest.fixture
def fake(tmp_path: Path) -> Fake:
    return Fake(tmp_path)


BASE_ENV = 'faa9d3e0019a8665f0b61dbf46a79134680731d43b9a9cb99ca2fa5af85373a8'
BASE_START = 'fa172986fcb3e9ee66fd62b6dc11435ec7a56abdc60247ad1e73d85fa561ad5d'


def test_happy_path_restores_k1_and_closes(fake: Fake) -> None:
    for phase in ('rollback', 'post', 'close'):
        r = fake.phase(phase)
        assert r.returncode == 0, (phase, r.stdout, r.stderr)
    assert fake.files() == {'env': BASE_ENV, 'start': BASE_START, 'record': fake.base_record}
    assert fake.state['running'] == {h: {'capture': False, 'k1': True} for h in ('sparkhead', 'sparkworker-wired')}
    assert not (fake.root/'HOLD').exists()


def test_preflight_failure_prints_untouched_recovery(fake: Fake) -> None:
    fake.state['tool_head'] = '1234567'
    fake.save()
    before = fake.files()
    r = fake.phase('rollback')
    assert r.returncode != 0
    assert all(v is not None for v in fake.state['running'].values())
    assert not (fake.root/'HOLD').exists()
    assert fake.files() == before
    assert 'STOP' in r.stderr and 'step read-only-preflight' in r.stderr
    assert 'Production untouched' in r.stderr
    assert not any('window.sh open' in cmd or './stop.sh' in cmd for _, cmd in fake.ssh_log())


def test_start_failure_prints_state_specific_recovery(fake: Fake) -> None:
    """K1 fallback during transition-2 start must produce exact recovery guidance."""
    fake.state['inject_k1_failures'] = 1
    fake.save()
    r = fake.phase('rollback')
    assert r.returncode != 0
    assert 'STOP' in r.stderr and 'NEXT' not in r.stdout
    assert 'step start' in r.stderr and 'start.log' in r.stderr
    assert 'K1 FALLBACK FAILED' in r.stderr and 'rollback-after-close-fallback' in r.stderr


def test_close_start_fallback_has_safe_explicit_recovery(fake: Fake) -> None:
    fake.state['inject_k1_failures'] = 1
    fake.save()
    r = fake.phase('rollback')
    assert r.returncode != 0
    # Production runs the PREVIOUS (pre-K1) configuration, no capture; K1 base record restored by rollback.
    assert fake.state['running']['sparkhead'] == {'capture': False, 'k1': False}
    record = json.loads((fake.root/'logs/k1-promotion/files.json').read_text())
    assert fake.files()['record'] == fake.base_record
    assert fake.files()['env'] == record['original']['dsv41.env']
    # The derivative-fallback mode must refuse this base-record state safely:
    r2 = fake.phase('rollback-after-k1-fallback')
    assert r2.returncode != 0
    assert 'not the exact K1 fallback state' in r2.stderr
    # Serving fallback production must survive both wrong recovery modes.
    assert all(v == {'capture': False, 'k1': False} for v in fake.state['running'].values())
    # Ordinary rollback cannot recover either (start.sh is the pre-K1 original).
    r3 = fake.phase('rollback')
    assert r3.returncode != 0 and 'not an accepted rollback state' in r3.stderr
    assert all(v == {'capture': False, 'k1': False} for v in fake.state['running'].values())
    for phase in ('rollback-after-close-fallback', 'post', 'close'):
        result = fake.phase(phase)
        assert result.returncode == 0, result.stderr
    assert fake.files() == {'env': BASE_ENV, 'start': BASE_START, 'record': fake.base_record}
    assert all(v == {'capture': False, 'k1': True} for v in fake.state['running'].values())


def test_k1_fallback_before_close_is_recovered_by_explicit_phase(fake: Fake) -> None:
    """The case the explicit phase does cover: production fell back while the derivative was applied."""
    root = fake.root
    for name in ('dsv41.env', 'start.sh'):
        shutil.copy2(root/'logs/k1-promotion'/name, root/name)
    fake.state['running'] = {h: {'capture': False, 'k1': False} for h in fake.state['running']}
    fake.save()
    for phase in ('rollback-after-k1-fallback', 'post', 'close'):
        r = fake.phase(phase)
        assert r.returncode == 0, (phase, r.stdout, r.stderr)
    assert fake.files() == {'env': BASE_ENV, 'start': BASE_START, 'record': fake.base_record}


def test_rollback_is_rerunnable_after_partial_stop(fake: Fake) -> None:
    fake.state['stop_partial'] = True
    fake.save()
    r = fake.phase('rollback')
    assert r.returncode != 0
    # Mixed state: rank 1 still runs capture; files unchanged; recovery guidance prints.
    assert fake.state['running'] == {'sparkhead': None, 'sparkworker-wired': {'capture': True, 'k1': True}}
    assert 'STOP' in r.stderr and 'step stop' in r.stderr
    r = fake.phase('rollback')
    assert r.returncode == 0, r.stderr
    assert fake.files()['env'] == BASE_ENV


@pytest.mark.parametrize('breakage', ['sparkworker_capture', 'fallback_profile', 'greedy', 'hold_check', 'busy'])
def test_post_refuses(fake: Fake, breakage: str) -> None:
    assert fake.phase('rollback').returncode == 0
    if breakage == 'sparkworker_capture':
        fake.state['running']['sparkworker-wired'] = {'capture': True, 'k1': True}
    elif breakage == 'fallback_profile':
        fake.state['running'] = {h: {'capture': False, 'k1': False} for h in fake.state['running']}
    elif breakage == 'greedy':
        fake.state['greedy_ok'] = False
    elif breakage == 'hold_check':
        fake.state['prod_window_ok'] = False
    else:
        fake.state['busy'] = True
    fake.save()
    r = fake.phase('post')
    assert r.returncode != 0
    assert fake.phase('close').returncode != 0  # no post marker -> no close
    assert (fake.root/'HOLD').exists()


def test_close_requires_same_nonce_post(fake: Fake) -> None:
    for phase in ('rollback', 'post'):
        assert fake.phase(phase, 'n1').returncode == 0
    assert fake.phase('close', 'other').returncode != 0
    assert fake.phase('close', 'n1').returncode == 0


def test_rollback_never_requires_copy_or_flush(fake: Fake) -> None:
    r = fake.phase('rollback')
    assert r.returncode == 0, r.stderr


def test_generator_pause_requires_sigint_to_save_and_exit(tmp_path: Path) -> None:
    """decode_close.sh pause waits for tmux d2-gen2 to END; generate.py only sleeps on PAUSE. SIGINT/SIGTERM
    (d.stop) is the clean exit (own streams aborted, budget saved) used by the close handshake."""
    import threading
    import time
    from d2 import generate as G
    from d2.tests.fakevllm import Fake as FakeVllm
    from d2.tests.test_generate import _args, _prompts
    with FakeVllm() as f:
        a = _args(tmp_path, f.url)
        d = G.Driver(a, _prompts(3))
        t = threading.Thread(target=d.loop, daemon=True)
        t.start()
        time.sleep(0.3)
        Path(a.pause_file).write_text('coordinator\n')
        time.sleep(1.5)
        assert t.is_alive()                    # paused, still looping: tmux d2-gen2 would never end
        d.stop = True                          # what the SIGINT/SIGTERM handler does
        t.join(timeout=30)
        assert not t.is_alive()


@pytest.mark.parametrize('mode', ['rollback', 'rollback-after-k1-fallback', 'rollback-after-close-fallback'])
@pytest.mark.parametrize('damage', ['dirty', 'transaction', 'backup', 'env', 'start', 'record', 'manifest'])
def test_all_recovery_preflights_refuse_without_window_or_stop(fake: Fake, mode: str, damage: str) -> None:
    root = fake.root
    if mode != 'rollback':
        if mode == 'rollback-after-close-fallback':
            assert fake.phase('rollback').returncode == 0
        for name in ('dsv41.env', 'start.sh'):
            shutil.copy2(root/'logs/k1-promotion'/name, root/name)
        fake.state['running'] = {h: {'capture': False, 'k1': False} for h in fake.state['running']}
    if damage == 'dirty':
        fake.state['tool_clean'] = False
    else:
        target = {'transaction': f'logs/d2-decode-{RUN}/transaction.json',
                  'backup': f'logs/d2-decode-{RUN}/start.sh', 'env': 'dsv41.env',
                  'start': 'start.sh', 'record': 'logs/k1-promotion/files.json',
                  'manifest': 'offload/kvpack/MANIFEST.json'}[damage]
        (root/target).write_text('{}')
    fake.save()
    before = fake.files()
    count = len(fake.ssh_log())
    running = fake.state['running'].copy()
    result = fake.phase(mode)
    assert result.returncode != 0 and 'Production untouched' in result.stderr
    assert fake.files() == before and fake.state['running'] == running
    assert not any('window.sh open' in cmd or './stop.sh' in cmd for _, cmd in fake.ssh_log()[count:])


def test_pause_sigints_only_owned_generator_and_requires_exit_marker(fake: Fake) -> None:
    fake.state['generator'] = True
    fake.save()
    result = fake.phase('pause')
    assert result.returncode == 0, result.stderr
    assert fake.state['sigint_saved'] and not fake.state['generator']
    assert (fake.c/'D2-GEN2.PAUSE').exists()
    assert (fake.c/'D2-GEN2.out.log').read_text() == 'D2-GEN2-EXIT 0\n'
    assert not fake.ssh_log()


def test_evidence_setup_failure_prints_untouched_stop(fake: Fake) -> None:
    exe(fake.t/'bin/mkdir', '#!/bin/bash\nexit 81\n')
    result = fake.phase('rollback')
    assert result.returncode == 81
    assert 'STOP' in result.stderr and 'Production untouched' in result.stderr
    assert not fake.ssh_log()


def test_missing_k1_verified_prints_start_log_recovery(fake: Fake) -> None:
    hook = fake.t/'hooks/start.sh'
    hook.write_text(hook.read_text().replace('K1 START VERIFIED;', 'K1 ready;'))
    result = fake.phase('rollback')
    assert result.returncode != 0
    assert 'STOP' in result.stderr and 'step verify-start' in result.stderr
    assert 'start.log' in result.stderr and 'restore.sh start|check|close' in result.stderr
    assert all(v == {'capture': False, 'k1': True} for v in fake.state['running'].values())
