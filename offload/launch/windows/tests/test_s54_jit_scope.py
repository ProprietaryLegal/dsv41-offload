"""S54 anonymous dockerd escape: real scopes, file-only Docker backend."""
from __future__ import annotations

import ast
from pathlib import Path
import shlex
import sys
import time

import pytest

from offload.launch.windows import spark_quiet as q
from offload.launch.windows.tests.test_scoped_remote import (
    OPS1, Sandbox, frozen, sandbox as sandbox, wait_for,
)

OLD = 'acb6f1e3bee05c607f35b32f3a69b4d70f61eacb'


def old_predicate(kind: str) -> str:
    source = frozen(OLD, 'offload/launch/windows/' + ('nbw_window.py' if kind == 'NBW' else 'post_close_watch.py'))
    name = 'EXPERIMENT_ABSENT' if kind == 'NBW' else 'quiet'
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            value = ast.literal_eval(node.value)
            assert isinstance(value, str)
            return value
    raise AssertionError('missing exact old cleanup predicate')


@pytest.mark.parametrize('kind', ['NBW', 'WATCH'])
@pytest.mark.parametrize('jit', ['experts', 'engram'])
def test_anonymous_pre_cuda_backend_survives_owner_but_blocks_both_cleanup_paths(
    sandbox: Sandbox, kind: str, jit: str,
) -> None:
    s = sandbox
    root = s.root
    backend_tag, client_tag = s.tag('dockerbackend'), s.tag('jitclient')
    body = ('touch ' + str(root/'backend-ready') + '; while [ ! -e ' + str(root/'request')
            + ' ]; do sleep .02; done; touch ' + str(root/'anonymous') + '; while [ ! -e '
            + str(root/'release') + ' ]; do sleep .02; done; echo late-JIT > ' + str(root/'state')
            + '; rm -f ' + str(root/'anonymous'))
    backend = s.launch('new', backend_tag, body)
    wait_for(root/'backend-ready')
    docker = s.binary/'docker'
    docker.write_text('#!' + sys.executable + '\nimport sys,time,json\nfrom pathlib import Path\n'
        + 'r=Path(' + repr(str(root)) + ')\n'
        + 'if sys.argv[1]=="run":\n'
        + ' (r/"args").write_text(json.dumps(sys.argv[2:])); (r/"request").touch()\n'
        + ' while not (r/"release").exists(): time.sleep(.02)\n'
        + 'elif sys.argv[1]=="ps":\n'
        + ' if (r/"anonymous").exists(): print("patient_babbage" if "{{.Names}}" in sys.argv else "a"*64)\n'
        + 'elif sys.argv[1]=="inspect":\n'
        + ' print(json.dumps([{"Id":"a"*64,"Name":"/patient_babbage","Path":"python3",'
        + '"Args":json.loads((r/"args").read_text()),"Config":{},"State":{"Status":"running"}}]))\n'
        + 'else: raise SystemExit(99)\n')
    docker.chmod(0o700)
    gpu = s.binary/'nvidia-smi'
    gpu.write_text('#!/bin/bash\nexit 0\n')
    gpu.chmod(0o700)
    if jit == 'experts':
        serve = frozen(OPS1, 'serve.sh')
        payload = 'export IMAGE=fixture; ' + serve[serve.index('JIT="'):serve.index('ssh "$WORKER" "$JIT"')]
        payload += '\nwait $JIT_HEAD'
    else:
        payload = shlex.join(['bash', '-c', frozen(OPS1, 'offload/engram/spark/prebuild_jit.sh'), '--', 'sparkworker'])
    client = s.launch('new', client_tag, payload)
    wait_for(root/'anonymous')
    s.close('new', client_tag)
    assert (root/'anonymous').exists() and backend.poll() is None
    old = s.run(old_predicate(kind))
    marker = 'NBW EXPERIMENT ABSENT' if kind == 'NBW' else 'WATCH SPARK QUIET'
    assert old.returncode == 0 and marker in old.stdout  # exact old false proof
    new = s.run(q.command(marker, deadline=time.time() + .3))
    assert new.returncode != 0 and marker not in new.stdout
    assert (root/'anonymous').exists() and backend.poll() is None  # compiler was never killed
    (root/'state').write_text('restored')
    (root/'release').touch()
    assert backend.wait(timeout=10) == 0
    assert (root/'state').read_text().strip() == 'late-JIT'
    assert not (root/'anonymous').exists()
    assert client.wait(timeout=10) != 0
    new = s.run(q.command(marker, deadline=time.time() + 10))
    assert new.returncode == 0 and marker in new.stdout.splitlines()
    s.close('new', backend_tag)
