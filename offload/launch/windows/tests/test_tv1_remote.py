"""Actual TV1 submissions through the shared receiving-host cgroup boundary.

Real local user scopes, private fake homes/SSH only; no production operations.
Historical timeout/close/JIT controls live in the shared OWN1 test modules.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import pytest

from offload.launch.windows import scoped_remote as owner, tv1_window as tv
from offload.launch.windows.tests.test_scoped_remote import Sandbox, sandbox as sandbox, wait_for


def admission(s: Sandbox, tag: str, token: str) -> None:
    home = s.homes['sparkhead']
    hold = home/'dsv41-deploy/HOLD'
    hold.parent.mkdir()
    candidate = home/owner.CONTROL/(tag+'-hold')/('hold-'+token)
    candidate.parent.mkdir(parents=True)
    data = f'original timestamp TV1-WINDOW {tag}: private fixture\n'.encode()
    candidate.write_bytes(data)
    os.link(candidate, hold)
    stat = hold.stat()
    (candidate.parent/('receipt-'+token+'.json')).write_text(json.dumps(
        {'dev': stat.st_dev, 'ino': stat.st_ino, 'sha256': hashlib.sha256(data).hexdigest()}))


@pytest.mark.parametrize('when', ['before-register', 'active'])
@pytest.mark.parametrize('kind', ['stop', 'run'])
def test_actual_tv1_second_hop_refuses_after_restore(sandbox: Sandbox, when: str, kind: str) -> None:
    s = sandbox
    root_tag = s.tag('tv1')
    live = object.__new__(tv.TV1Live)
    live.root_tag, live.owner_tags, live.owner_deadlines = root_tag, [], {}
    live.admission_token = uuid.uuid4().hex
    live.sup = SimpleNamespace(abort_at=time.time()+25)  # type: ignore[assignment]
    submitted: list[str] = []
    def capture(label: str, argv: list[str], *, stdin: str | None = None,
                timeout: float = 120, inflight: int = 0) -> str:
        submitted.append(argv[-1])
        return ''  # lost/empty reply does not erase remembered ownership
    live.call = capture
    admission(s, root_tag, live.admission_token)
    state, pending, release = s.root/'production', s.root/'pending', s.root/'release'
    state.write_text('running')
    gate = s.root/'sparkworker-wired-transport-gate'
    if when == 'before-register':
        gate.touch()
    body = (f'touch {shlex.quote(str(pending))}; while [ ! -e {shlex.quote(str(release))} ]; do sleep .02; done; '
            f'echo {kind} > {shlex.quote(str(state))}')
    # GNU timeout creates a different PGID, reproducing S53 beneath the actual
    # TV1 submission. A cgroup proof must include it and its second SSH session.
    live.ssh('s2-candidate-start', 'sparkhead', 'timeout --kill-after=1s 20s ssh sparkworker '+shlex.quote(body))
    assert len(submitted) == 1 and len(live.owner_tags) == 1
    tag = live.owner_tags[0]
    s.tags.add(tag)
    s.deadlines[tag] = live.sup.abort_at
    with (s.root/'submitted.log').open('w') as log:
        proc = subprocess.Popen(['bash','-c',submitted[0]], env=s.env(), text=True,
                                stdout=log, stderr=log, start_new_session=True)
    s.processes.append(proc); s.groups.add(proc.pid)
    wait_for(s.root/'sparkworker-wired-accepted' if when == 'before-register' else pending)
    s.close('new', tag)
    state.write_text('restored')  # fake W8 closed only AFTER all proofs
    gate.unlink(missing_ok=True); release.touch()
    proc.wait(timeout=10)
    time.sleep(.1)
    assert state.read_text() == 'restored', 'delayed nested timeout/SSH mutation crossed close'
    for host in s.homes:
        assert not any(owner.populated(p) for p in owner.paths(s.homes[host]/owner.CONTROL/tag, tag))
    retry = s.run(submitted[0])
    assert retry.returncode != 0 and state.read_text() == 'restored'


def test_shared_scp_transport_preserves_actual_openssh_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt, transport = tmp_path/'arguments.json', tmp_path/'transport'
    transport.write_text('#!'+sys.executable+'\nimport json,sys\n'
                         f'open({str(receipt)!r},"w").write(json.dumps(sys.argv[1:]))\nraise SystemExit(83)\n')
    transport.chmod(0o700)
    source = owner.source().replace("SSH_BINARY = '/usr/bin/ssh'", 'SSH_BINARY = '+repr(str(transport)))
    monkeypatch.setattr(owner, 'source', lambda: source)
    monkeypatch.setenv('SCOPED_REMOTE_DEADLINE', str(time.time()+30))
    root = tmp_path/'owner'; root.mkdir()
    bindir = owner.install(root, 'scp-fixture')
    payload = tmp_path/'payload'; payload.write_text('public fixture')
    result = subprocess.run([str(bindir/'scp'),'-q',str(payload),'sparkworker:fixture'],
                            text=True,capture_output=True,timeout=10)
    assert result.returncode != 0 and receipt.exists(), result.stderr
    args = json.loads(receipt.read_text())
    assert args[-2] == 'sparkworker'
    remote = shlex.split(args[-1])
    assert remote[:2] == ['env', 'SCOPED_REMOTE_DEADLINE='+format(float(os.environ['SCOPED_REMOTE_DEADLINE']), '.6f')]
    assert remote[5:7] == ['launch', 'scp-fixture'] and remote[-1] == 'scp -t fixture'
