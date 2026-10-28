"""Exercise the protocol runner and coordinator-only refusal without CUDA."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from worker.server import DraftCore, FakeBackend, expected_hello
from offload.dspark.native.tools.n3.soak import rounds


def test_pair_script_refuses_before_any_gpu_or_lock_action(tmp_path):
    config = tmp_path/'.config/ndm'
    config.mkdir(parents=True)
    (config/'paths.sh').write_text(':\n')
    script = Path(__file__).resolve().parents[2]/'native/tools/n3/pair.sh'
    env = {**os.environ,'HOME':str(tmp_path),'N3_PAIR_BY_COORD':'0'}
    result = subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True)
    assert result.returncode==1 and 'PAIR_OK must name N3' in result.stderr
    assert not (tmp_path/'dsv41-offload').exists()
    env['N3_PAIR_BY_COORD']='1'
    env.pop('N3_V1_KEYS',None)
    result = subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True)
    assert result.returncode==2 and 'N3_V1_KEYS must explicitly name' in result.stderr
    assert not (tmp_path/'dsv41-offload').exists()


def test_pair_job_requires_real_inherited_flocks(tmp_path):
    root = tmp_path/'dsv41-offload/locks'
    root.mkdir(parents=True)
    program = '''
import fcntl, os
from pathlib import Path
from offload.dspark.native.tools.n3.guards import require_pair_ownership
streams=[]
for fd,name in ((8,'gpu-c4e7a371.lock'),(9,'gpu-7744b44b.lock')):
    f=open(Path.home()/'dsv41-offload/locks'/name,'a')
    os.dup2(f.fileno(),fd)
    streams.append(f)
try:
    require_pair_ownership()
except RuntimeError as e:
    assert 'was not held' in str(e)
else:
    raise AssertionError('unheld locks accepted')
for fd in (8,9):
    fcntl.flock(fd,fcntl.LOCK_EX)
require_pair_ownership()
print('ownership verified')
'''
    from offload.dspark.native.tools.n3.guards import PAIR
    env={**os.environ,'HOME':str(tmp_path),'CUDA_VISIBLE_DEVICES':PAIR,'N3_PAIR_BY_COORD':'1'}
    import sys
    result=subprocess.run([sys.executable,'-c',program],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'ownership verified' in result.stdout


@pytest.fixture
def fake_pair(tmp_path):
    """All GPU commands/children are fakes; only temporary real flocks are used."""
    import sys
    root = tmp_path/'dsv41-offload/n3'
    for directory in (root/'repo',root/'results',tmp_path/'.config/ndm',tmp_path/'bin',
                      tmp_path/'dsv41-offload/e/venv/bin',tmp_path/'dsv41-offload/locks'):
        directory.mkdir(parents=True,exist_ok=True)
    (tmp_path/'.config/ndm/paths.sh').write_text(':\n')
    (root/'repo/SOURCE_COMMIT').write_text('fake-source\n')
    import hashlib
    cu=root/'repo/offload/dspark/worker/p2p/p2p.cu'
    cu.parent.mkdir(parents=True)
    cu.write_text('synthetic-source\n')
    pin=root/'p2p'/hashlib.sha256(cu.read_bytes()).hexdigest()[:16]
    pin.mkdir(parents=True)
    (pin/'dspark_p2p.so').write_bytes(b'synthetic-fake-only')
    (pin/'dspark_p2p.so.sha256').write_text(hashlib.sha256((pin/'dspark_p2p.so').read_bytes()).hexdigest()+'  dspark_p2p.so\n')
    source = Path(__file__).resolve().parents[4]
    py = tmp_path/'dsv41-offload/e/venv/bin/python'
    py.write_text('#!'+sys.executable+'\n'+'''
import json,os,signal,sys,time
from pathlib import Path
if sys.argv[1]=='-':
    sys.argv=sys.argv[1:]
    exec(compile(sys.stdin.read(),'<digest-check>','exec'))
    sys.exit(0)
sys.path.insert(0,os.environ['N3_TEST_SOURCE'])
from offload.dspark.native.tools.n3.guards import require_pair_ownership
require_pair_ownership()
out=Path(sys.argv[sys.argv.index('--out')+1]);out.mkdir()
trace=Path(os.environ['N3_TEST_TRACE'])
with trace.open('a') as f:f.write(json.dumps({'pid':os.getpid(),'argv':sys.argv})+'\\n')
(out/'client.json').write_text(json.dumps({'rounds':2000,'digest':'synthetic-same'}))
if os.environ.get('N3_TEST_WAIT')=='1':
    def stop(*args):
        trace.with_suffix('.stopped').write_text(str(os.getpid()))
        sys.exit(143)
    signal.signal(signal.SIGTERM,stop)
    while True:time.sleep(.1)
''')
    py.chmod(0o755)
    smi = tmp_path/'bin/nvidia-smi'
    smi.write_text('#!'+sys.executable+'\n'+'''
import os,sys,time
if '--query-compute-apps=pid' in sys.argv and os.environ.get('N3_TEST_SMI_FAIL')=='1':
    print('synthetic NVML query failure',file=sys.stderr)
    sys.exit(9)
if '-l' in sys.argv:
    while True:
        print('2026/10/23 00:00:00.000, fake-uuid, 0 MiB, 0 W, 325 W',flush=True)
        time.sleep(.1)
elif '-L' in sys.argv: print('FAKE GPU enumeration')
''')
    smi.chmod(0o755)
    env = {**os.environ,'HOME':str(tmp_path),'PATH':str(tmp_path/'bin')+':'+os.environ['PATH'],
           'N3_PAIR_BY_COORD':'1','N3_PAIR_ARMS':'native-packed','N3_PAIR_RUN_ID':'fake-test',
           'N3_ROOT':str(root),'DSPARK_P2P_PREBUILT':str(root/'p2p'),
           'N3_V1_KEYS':'r000008_b0','N3_TEST_TRACE':str(tmp_path/'trace.jsonl'),
           'N3_TEST_SOURCE':str(source)}
    script = source/'offload/dspark/native/tools/n3/pair.sh'
    return root,env,script


def test_pair_fake_sequence_and_no_hold_of_first_lock_while_second_is_busy(fake_pair):
    import fcntl
    import time
    root,env,script = fake_pair
    locks = root.parents[0]/'locks'
    with (locks/'gpu-7744b44b.lock').open('a') as second:
        fcntl.flock(second,fcntl.LOCK_EX)
        child = subprocess.Popen(['bash',str(script)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            time.sleep(.4)
            with (locks/'gpu-c4e7a371.lock').open('a') as first:
                # The worker cannot run yet, but rank0's lock must be released.
                fcntl.flock(first,fcntl.LOCK_EX|fcntl.LOCK_NB)
                fcntl.flock(first,fcntl.LOCK_UN)
            fcntl.flock(second,fcntl.LOCK_UN)
            out,err = child.communicate(timeout=15)
            assert child.returncode==0,(out,err)
        finally:
            if child.poll() is None:
                child.terminate();child.wait(timeout=5)
    import json
    jobs=[json.loads(x) for x in Path(env['N3_TEST_TRACE']).read_text().splitlines()]
    assert len(jobs)==3
    assert (root/'results/pair-fake-test/native-packed-on-off-equal.json').is_file()


@pytest.mark.parametrize('action',['remove','replace'])
def test_pair_marker_withdrawal_terminates_owned_fake_child(fake_pair,action):
    import time
    root,env,script = fake_pair
    marker=root.parents[0]/'PAIR_OK'
    marker.write_text('N3 first-window\n')
    env.update(N3_PAIR_BY_COORD='0',N3_TEST_WAIT='1')
    trace=Path(env['N3_TEST_TRACE'])
    child=subprocess.Popen(['bash',str(script)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        deadline=time.monotonic()+10
        while not trace.exists() and time.monotonic()<deadline:
            time.sleep(.05)
        assert trace.exists(),'fake child did not start'
        if action=='remove':marker.unlink()
        else:marker.write_text('N3 replacement-window\n')
        child.communicate(timeout=10)
        assert child.returncode!=0
        assert trace.with_suffix('.stopped').is_file()
        assert len(trace.read_text().splitlines())==1
    finally:
        if child.poll() is None:
            child.terminate();child.wait(timeout=5)


def test_pair_bad_prebuilt_refuses_before_gpu_or_lock(fake_pair):
    root,env,script=fake_pair
    pin=next((root/'p2p').glob('*/dspark_p2p.so'))
    pin.write_bytes(b'tampered')
    result=subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True)
    assert result.returncode==2 and 'FAILED' in result.stdout
    assert not Path(env['N3_TEST_TRACE']).exists()
    assert not list((root/'results').iterdir())


def test_pair_nvml_query_failure_never_starts_a_worker(fake_pair):
    root,env,script=fake_pair
    env['N3_TEST_SMI_FAIL']='1'
    result=subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True)
    assert result.returncode==9
    assert not Path(env['N3_TEST_TRACE']).exists()
    assert 'synthetic NVML query failure' in (root/'results/pair-fake-test/native-packed-off.log').read_text()


def test_solo_nvml_query_failure_never_starts_a_worker(fake_pair):
    root,env,_=fake_pair
    env['N3_TEST_SMI_FAIL']='1'
    stub=Path(env['HOME'])/'bin/ssh'
    stub.write_text('#!/bin/sh\nprintf "fake coordinator state\\n"\n')
    stub.chmod(0o755)
    script=Path(env['N3_TEST_SOURCE'])/'offload/dspark/native/tools/n3/run_locked.sh'
    result=subprocess.run(['bash',str(script),'fake-solo','-m','fake-worker'],env=env,capture_output=True,text=True)
    assert result.returncode==9
    assert not Path(env['N3_TEST_TRACE']).exists()
    assert 'synthetic NVML query failure' in (root/'logs/fake-solo.log').read_text()
