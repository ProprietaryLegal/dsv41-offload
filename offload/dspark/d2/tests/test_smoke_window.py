"""Window rehearsal cannot execute live commands; cleanup owns exact run/IDs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from d2 import smoke_window as W

REPO = Path(__file__).resolve().parents[4]
SOURCE = 'f'*40


def test_default_rehearsal_opens_stages_smokes_stops_without_external_calls(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('dry run attempted subprocess/network')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    io = W.DryWindow('synthetic')
    report = W.run(io, REPO, SOURCE, 'synthetic', tmp_path/'nonexistent-input', tmp_path)
    assert report['synthetic'] is True and report['pass'] is True
    assert report['drain_seconds'] == 300 and report['drain_samples'] == 61
    assert report['stopped_ranks'] == ['sparkhead','sparkworker-wired']
    assert io.calls.index('open-window') < io.calls.index('reviewed-checkout') < io.calls.index('start-exp')
    assert io.calls.index('stage-sparkhead') < io.calls.index('start-exp')
    assert io.calls.index('stage-sparkworker-wired') < io.calls.index('start-exp')
    assert io.calls.index('decode-smoke') < io.calls.index('stop-owned-sparkhead')
    assert not any('restore' in call or 'close' in call for call in io.calls)


@pytest.mark.parametrize('failure', ['busy','missing','negative','nan','gap','memory'])
def test_drain_faults_never_open_window(tmp_path, failure):
    io = W.DryWindow('synthetic')
    if failure == 'memory':
        def low(): raise ValueError('MemAvailable below 3 GiB')
        io.memcheck = low
    elif failure == 'gap':
        io.sleep = lambda seconds: setattr(io,'elapsed',io.elapsed+11)
    else:
        io.metrics = lambda: {'busy':'vllm:num_requests_running{} 1\nvllm:num_requests_waiting{} 0\n',
            'missing':'', 'negative':'vllm:num_requests_running{} -1\nvllm:num_requests_waiting{} 0\n',
            'nan':'vllm:num_requests_running{} NaN\nvllm:num_requests_waiting{} 0\n'}[failure]
    with pytest.raises(ValueError): W.run(io, REPO, SOURCE,'synthetic',tmp_path/'unused',tmp_path)
    assert 'open-window' not in io.calls and 'start-exp' not in io.calls


@pytest.mark.parametrize('failure', ['stage','launch','smoke','wrong-run','wrong-id','cleanup-refused'])
def test_window_failure_retains_hold_and_only_stops_owned_candidates(tmp_path, failure):
    io = W.DryWindow('synthetic')
    call = io.call
    def fault(name,host,command,**kwargs):
        result = call(name,host,command,**kwargs)
        if (failure == 'stage' and name == 'stage-sparkworker-wired') or (failure == 'launch' and name == 'start-exp'):
            raise ValueError('synthetic refusal')
        if name == 'cleanup-container-sparkhead' and failure in ('wrong-run','wrong-id'):
            c=json.loads(result)
            if failure == 'wrong-run': c['mounts'][0]['Source'] += '-other'
            else: c['image']='wrong'
            return json.dumps(c)
        if failure == 'cleanup-refused' and name == 'stop-owned-sparkhead':
            raise ValueError('synthetic stop refusal')
        return result
    io.call = fault
    if failure == 'smoke':
        def refuse(*args): raise ValueError('smoke timeout')
        io.smoke=refuse
    with pytest.raises(ValueError): W.run(io,REPO,SOURCE,'synthetic',tmp_path/'unused',tmp_path)
    state=json.loads((tmp_path/'window-state.json').read_text())
    assert state['window_opened'] is True and state['restore_separate'] is True
    if failure == 'stage': assert 'stop-owned-sparkhead' not in io.calls
    else: assert 'stop-owned-sparkworker-wired' in io.calls
    if failure in ('wrong-run','wrong-id'): assert 'stop-owned-sparkhead' not in io.calls
    assert not (tmp_path/'window-report.json').exists()


def test_stage_is_immutable_exact_source_and_fresh_host_owned_directories():
    command=W.stage_command(REPO,SOURCE,'safe-new-run')
    for path in W.FILES: assert path in command
    assert '0o444' in command and '0o700' in command and 'st_uid==os.getuid()' in command
    assert 'capture run already exists' in command and 'source symlink refused' in command
    assert '1024**3' in command and 'sha256' in command


def test_checkout_refuses_dirty_source_and_pins_full_reviewed_commit():
    command=W.checkout_command(SOURCE,'a'*64)
    assert '--untracked-files=all' in command and '--detach '+SOURCE in command
    assert 'sha256sum' in command and 'git fetch --no-tags' in command
    assert 'git reset' not in command and 'git stash' not in command


def test_execute_without_coordinator_cannot_spawn(monkeypatch):
    monkeypatch.delenv('D2_COORDINATOR', raising=False)
    monkeypatch.setattr('sys.argv',['smoke_window','--execute'])
    def forbidden(*args,**kwargs): raise AssertionError('unauthorized process')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    with pytest.raises(ValueError,match='coordinator-only'):
        W.main()


def test_bundle_handoff_exports_named_head_and_fetches_exact_commit(tmp_path):
    source_repo=tmp_path/'source'
    source_repo.mkdir()
    # Never inherit GIT_DIR/GIT_WORK_TREE etc.: S29 (2026-10-24) ran this with GIT_DIR exported and the init/config/
    # commit below rewrote the shared dsv41-deploy .git/config (core.bare, a synthetic [user]) for 13 minutes.
    clean = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    def git(root,*args):
        return subprocess.check_output(['git','-C',str(root),*args],text=True,env=clean).strip()
    git(source_repo,'init','-q')
    git(source_repo,'config','user.name','Synthetic Test')
    git(source_repo,'config','user.email','test@example.invalid')
    (source_repo/'fixture').write_text('synthetic\n')
    git(source_repo,'add','fixture')
    git(source_repo,'commit','-qm','fixture')
    source=git(source_repo,'rev-parse','HEAD')
    out=tmp_path/'out'
    out.mkdir()

    class LocalBundle(W.LiveWindow):
        def memcheck(self): pass
        def local(self,name,argv,timeout):
            if name == 'create-source-bundle': subprocess.run(argv,check=True,timeout=timeout)
            else: assert name == 'transfer-source-bundle'  # never scp in CPU test
        def call(self,*args,**kwargs): return 'synthetic directory'
    io=LocalBundle(out,source)
    io.bundle(source_repo,source,out)
    receiving=tmp_path/'receiving'
    receiving.mkdir()
    git(receiving,'init','-q')
    git(receiving,'fetch','--no-tags',str(out/'source.bundle'),'HEAD')
    assert git(receiving,'rev-parse','FETCH_HEAD') == source
    bad=tmp_path/'bad'
    bad.mkdir()
    with pytest.raises(ValueError,match='HEAD changed'):
        io.bundle(source_repo,'0'*40,bad)


def test_default_cli_uses_r2_inputs_and_has_no_http_or_process_path(tmp_path, monkeypatch):
    import urllib.request
    monkeypatch.setattr('sys.argv',['smoke_window'])
    monkeypatch.setattr(W,'SCRATCH',tmp_path)
    def forbidden(*args,**kwargs): raise AssertionError('offline CLI tried HTTP/process')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    monkeypatch.setattr(urllib.request,'urlopen',forbidden)
    original = W.run
    def checked(io,repo,source,run_id,inputs,out):
        assert isinstance(io,W.DryWindow)
        assert inputs.name == 'smoke-input-r2-qualified.json'
        return original(io,repo,source,run_id,inputs,out)
    monkeypatch.setattr(W,'run',checked)
    W.main()
    assert not list(tmp_path.iterdir())
