"""Campaign artifact dry rehearsal; external commands are fatal fake executables."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/decode_close.sh')
PHASES = ('pause','flush','copy','rollback','post','close','rollback-after-k1-fallback','rollback-after-close-fallback')


def dry(tmp_path: Path, phase: str) -> str:
    binpath = tmp_path/'bin'
    binpath.mkdir(exist_ok=True)
    hit = tmp_path/'EXTERNAL_COMMAND_WAS_RUN'
    for name in ('ssh','curl','tmux','touch','rm','mkdir','python3','sleep','grep'):
        exe = binpath/name
        exe.write_text('#!/bin/bash\nprintf forbidden > '+shlex.quote(str(hit))+'\nexit 91\n')
        exe.chmod(0o700)
    env = {k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
    env['PATH'] = str(binpath)
    result = subprocess.run(['/bin/bash',str(SCRIPT),'--dry','--nonce','rehearsal',
        '--top20','/synthetic/top20 tree',phase], env=env, text=True, capture_output=True, check=True)
    assert not hit.exists(), result.stderr
    return result.stdout


@pytest.mark.parametrize('phase', PHASES)
def test_dry_has_no_external_commands_or_mutations(tmp_path, phase):
    output = dry(tmp_path, phase)
    assert 'DRY ' in output
    for command in (l[4:] for l in output.splitlines() if l.startswith('DRY ')):
        subprocess.run(['/bin/bash','-n','-c',command], check=True)


def remote_commands(output: str) -> list[tuple[str, str]]:
    result = []
    for line in output.splitlines():
        if line.startswith('DRY ssh '):
            argv = shlex.split(line[4:])
            assert argv[:3] == ['ssh','-o','BatchMode=yes']
            result.append((argv[3], argv[4]))
    return result


def test_exact_restoration_order_and_separate_fallback(tmp_path):
    for phase, flag in [('rollback','--rollback'),('rollback-after-k1-fallback','--rollback-after-k1-fallback')]:
        commands = remote_commands(dry(tmp_path,phase))
        assert len(commands) == 6
        assert all(host == 'sparkhead' for host, _ in commands)
        texts = [command for _, command in commands]
        assert "READ-ONLY PREFLIGHT VERIFIED" in texts[0]
        assert "git diff --quiet" in texts[0]
        texts = texts[1:]
        assert 'window.sh open' in texts[0]
        assert './stop.sh' in texts[1]
        assert 'python3 -m offload.dspark.d2.decode_apply' in texts[2] and texts[2].endswith(flag)
        assert 'sha256sum -c' in texts[3] and 'manage.py verify' in texts[3]
        assert './start.sh' in texts[4]
        assert 'window.sh close' not in '\n'.join(texts)


def test_flush_copy_and_close_commands(tmp_path):
    flush = remote_commands(dry(tmp_path,'flush'))
    copy = remote_commands(dry(tmp_path,'copy'))
    assert flush[0][0] == copy[0][0] == 'x86host'
    assert '--flush' in flush[0][1]
    argv = shlex.split(copy[0][1])
    assert argv[argv.index('--top20-root')+1] == '/synthetic/top20 tree'
    assert argv[argv.index('--run')+1] == 'd2pass2-r1'
    assert remote_commands(dry(tmp_path,'close')) == [('sparkhead','cd ~/dsv41-deploy && offload/launch/window.sh close')]


def python_payloads(output: str):
    for line in output.splitlines():
        if not line.startswith('DRY '):
            continue
        argv = shlex.split(line[4:])
        if '-c' in argv:
            yield argv[argv.index('-c')+1]
        elif argv[0] == 'ssh':
            remote = shlex.split(argv[4])
            if remote[0] == 'python3':
                yield remote[remote.index('-c')+1]


def test_payload_quoting_and_idle_refusals(tmp_path):
    codes = list(python_payloads(dry(tmp_path,'post')))
    assert len(codes) == 7  # model, inspect+memory x2, idle, full greedy rows
    for code in codes:
        compile(code, '<dry command payload>', 'exec')
    code = next(c for c in codes if 'num_requests_' in c)
    for gauges, good in [('vllm:num_requests_running{} 0\nvllm:num_requests_waiting{} 0\n',True),
        ('vllm:num_requests_running{} 1\nvllm:num_requests_waiting{} 0\n',False),
        ('vllm:num_requests_running{} nan\nvllm:num_requests_waiting{} 0\n',False), ('',False)]:
        result = subprocess.run([sys.executable,'-c',code],input=gauges,text=True,capture_output=True,check=False)
        assert (result.returncode == 0) is good, result.stderr


def test_phase_marker_is_actual_nonce_line(tmp_path):
    out = dry(tmp_path,'pause')
    command = next(l[4:] for l in out.splitlines() if l.startswith('DRY printf '))
    destination = tmp_path/'marker'
    command = command.split(' > ')[0]+' > '+shlex.quote(str(destination))
    subprocess.run(['/bin/bash','-euo','pipefail','-c',command],check=True)
    assert destination.read_text() == 'rehearsal\n'


def test_mount_probe_refuses_capture_env_and_either_mount(tmp_path):
    code = next(c for c in python_payloads(dry(tmp_path,'post')) if 'Mounts' in c)
    for env, mounts, good in [([],[],True),
        (['VLLM_DSPARK_DECODE_CAPTURE_DIR=/d2-decode'],[],False),
        ([],[{'Source':'/home/user/dsv41-decode/SOURCE/capture.py','Destination':'/code.py'}],False),
        ([],[{'Source':'/data/capture','Destination':'/d2-decode'}],False)]:
        payload = [{'Config':{'Env':env},'Mounts':mounts}]
        result = subprocess.run([sys.executable,'-c',code],input=json.dumps(payload),text=True,capture_output=True,check=False)
        assert (result.returncode == 0) is good
