"""S33 probes of the decode_close.sh restore path with the REAL 21bed31 manage.py and real pre-K1 bytes.

Built on S31's fake-host harness (a COPY of the campaign script runs NON-dry against a scratch root, real
decode_apply 8174771). S33 replaces two emulations with real code:
- `manage.py verify` / `manage.py restore` are the 21bed31 git objects (with profiles.py), run against the root;
- the pre-K1 originals are the REAL bytes, reconstructed by reverse-applying 21bed31's K1 patches to the reviewed
  base copy (hashes equal the K1 record's `original`), so the record is the real base record byte for byte.
The start hook follows start_checked.sh's control flow (refuse a running serve, verify, on failure stop + REAL
restore + start the previous configuration, `K1 REFUSED; previous configuration restarted`).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from d2.tests import test_s31_close_probes as s31

BASE_ENV = s31.BASE_ENV
BASE_START = s31.BASE_START
REAL_ORIGINAL = {'dsv41.env': '91041e7334a9d59832b81b83a4711ed213ea5876605397b6bff708bfb65e7d71',
                 'start.sh': 'beb20b21be2b4ed34afa70cc8bb5bfd1e838f585b53e8440e07e8423026c2c84'}
K1_COMMIT = '21bed313ac803d38327c4de1ff2e5235f329c716'

pytestmark = s31.pytestmark

MANAGE = r'''#!/usr/bin/python3
import os, subprocess, sys
from pathlib import Path
root = Path(os.environ['S31_T'])/'sparkhead'/'dsv41-deploy'
sys.exit(subprocess.run([sys.executable, str(root/'offload/promote/k1/manage.py'), *sys.argv[1:]], cwd=root).returncode)
'''

START = s31.COMMON + r'''
import subprocess
s = load()
def manage(*args):
    return subprocess.run([sys.executable, str(ROOT/'offload/promote/k1/manage.py'), *args], cwd=ROOT,
                          capture_output=True, text=True)
if any(v is not None for v in s['running'].values()):
    print('K1 start refuses: already running', file=sys.stderr); sys.exit(1)
guarded = sha(ROOT/'start.sh') == '%s' and b'\nPROD_K1=1\n' in (ROOT/'dsv41.env').read_bytes()
if not guarded:
    s['running'] = {h: {'capture': capture_env(), 'k1': False} for h in s['running']}; save(s)
    print('previous (pre-K1) configuration started'); sys.exit(0)
verified = manage('verify', '--profile', '327680')
print(verified.stdout + verified.stderr)
if verified.returncode == 0 and s['inject_k1_failures'] == 0:
    s['running'] = {h: {'capture': capture_env(), 'k1': True} for h in s['running']}; save(s)
    print('K1 START VERIFIED; complete the production-window probes before releasing HOLD'); sys.exit(0)
if s['inject_k1_failures']:
    s['inject_k1_failures'] -= 1
print('K1 START FAILED (status 1, line 30); stopping candidate and restoring previous config', file=sys.stderr)
s['running'] = {h: None for h in s['running']}; save(s)
restored = manage('restore')
print(restored.stdout + restored.stderr, file=sys.stderr)
if restored.returncode:
    print('K1 FALLBACK FAILED: backup/edits need operator review; keep HOLD', file=sys.stderr); sys.exit(1)
s['running'] = {h: {'capture': capture_env(), 'k1': False} for h in s['running']}; save(s)
print('K1 REFUSED; previous configuration restarted. Inspect log; keep HOLD until rollback checks pass.', file=sys.stderr)
sys.exit(1)
''' % BASE_START


class RealK1(s31.Fake):
    def _production_root(self) -> None:
        root = self.root
        (root/'offload/kvpack').mkdir(parents=True)
        (root/'offload/promote/k1').mkdir(parents=True)
        (root/'logs/k1-promotion').mkdir(parents=True)
        for rel in ('dsv41.env', 'start.sh', 'offload/kvpack/MANIFEST.json', 'logs/k1-promotion/files.json'):
            shutil.copy2(s31.BASE_COPY/rel, root/rel)
        # Real pre-K1 originals: reverse the reviewed K1 patches on scratch copies of the base bytes.
        recon = self.t/'recon'
        recon.mkdir()
        for name in ('dsv41.env', 'start.sh'):
            shutil.copy2(root/name, recon/name)
        patches = []
        for name in ('start.sh.patch', 'dsv41.env.327680.patch'):
            text = subprocess.run(['git', '-C', str(s31.REPO), 'show', f'{K1_COMMIT}:offload/promote/k1/{name}'],
                                  capture_output=True, check=True).stdout
            (self.t/name).write_bytes(text)
            patches.append(str(self.t/name))
        subprocess.run(['git', 'apply', '-R', *patches], cwd=recon, check=True,
                       env={'PATH': '/usr/bin:/bin', 'HOME': str(self.t)})
        for name in ('dsv41.env', 'start.sh'):
            assert s31.sha(recon/name) == REAL_ORIGINAL[name]
            shutil.copy2(recon/name, root/'logs/k1-promotion'/name)
        for name in ('manage.py', 'profiles.py'):
            data = subprocess.run(['git', '-C', str(s31.REPO), 'show', f'{K1_COMMIT}:offload/promote/k1/{name}'],
                                  capture_output=True, check=True).stdout
            (root/'offload/promote/k1'/name).write_bytes(data)
        s31.exe(root/'offload/kvpack/stage_overlay.sh', '#!/bin/bash\n[ "$1" = verify-checkout ] || exit 9\n')
        report = root/'logs/k1-promotion/S18.report.md'
        report.write_text('synthetic S18 report for the scratch gate\n')
        (root/'logs/k1-promotion/S18.receipt.json').write_text(json.dumps({
            'schema': 'dsv41-k1-promotion-v1', 'reviewer': 'S18', 'final': True, 'window6': 'PROMOTE',
            'max_model_len': 327680, 'kv_tokens': 2502600, 'manifest_sha256': s31.sha(root/'offload/kvpack/MANIFEST.json'),
            'report_sha256': s31.sha(report)}))
        record = json.loads((root/'logs/k1-promotion/files.json').read_text())
        assert record['original'] == REAL_ORIGINAL and record['promoted'] == {'dsv41.env': BASE_ENV, 'start.sh': BASE_START}
        self.base_record = s31.sha(root/'logs/k1-promotion/files.json')
        # REAL manage.py verify on the base before the capture apply (K1 base approval holds).
        ok = subprocess.run(['/usr/bin/python3', str(root/'offload/promote/k1/manage.py'), 'verify', '--profile', '327680'],
                            cwd=root, capture_output=True, text=True)
        assert ok.returncode == 0, ok.stderr
        tool = self.t/'sparkhead/dsv41-deploy-offload/offload/dspark/d2'
        tool.mkdir(parents=True)
        shutil.copy2(s31.DECODE_APPLY, tool/'decode_apply.py')
        (tool/'__init__.py').write_text('')
        cwd = self.t/'sparkhead/dsv41-deploy-offload'
        render = subprocess.run([s31.SYSPY, '-m', 'offload.dspark.d2.decode_apply', '--root', str(root), '--source',
                                 s31.SRC, '--run', s31.RUN], cwd=cwd, text=True, capture_output=True, check=True).stdout
        capture_sha = render.strip().splitlines()[-1].split('=')[1]
        review = self.t/'review.json'
        review.write_text(json.dumps({'schema': 'd2.decode-apply-review.v1', 'approved': True,
            'coordinator': 'Deepseek/coordinator', 'source': s31.SRC, 'base_commit': K1_COMMIT,
            'base_env_sha256': BASE_ENV, 'capture_env_sha256': capture_sha}))
        subprocess.run([s31.SYSPY, '-m', 'offload.dspark.d2.decode_apply', '--root', str(root), '--source', s31.SRC,
                        '--run', s31.RUN, '--apply', '--review', str(review)], cwd=cwd, check=True)
        assert s31.sha(root/'dsv41.env') == capture_sha

    def install_real_k1_hooks(self) -> None:
        s31.exe(self.t/'hooks/manage.py', MANAGE)
        s31.exe(self.t/'hooks/start.sh', '#!/usr/bin/python3\n'+START)


@pytest.fixture
def real(tmp_path: Path) -> RealK1:
    f = RealK1(tmp_path)
    f.install_real_k1_hooks()
    return f


def serving(f: RealK1, k1: bool) -> bool:
    return all(v == {'capture': False, 'k1': k1} for v in f.state['running'].values())


def close_fallback_state(f: RealK1) -> None:
    """Transition 2's own K1 start falls back (real manage.py restore): pre-K1 serving, BASE record."""
    f.state['inject_k1_failures'] = 1
    f.save()
    r = f.phase('rollback')
    assert r.returncode != 0
    assert 'step start' in r.stderr and 'rollback-after-close-fallback' in r.stderr
    assert serving(f, k1=False)
    assert f.files() == {'env': REAL_ORIGINAL['dsv41.env'], 'start': REAL_ORIGINAL['start.sh'], 'record': f.base_record}


def test_close_fallback_end_to_end_with_real_manage(real: RealK1) -> None:
    close_fallback_state(real)
    for wrong in ('rollback', 'rollback-after-k1-fallback'):
        count = len(real.ssh_log())
        r = real.phase(wrong)
        assert r.returncode != 0 and 'Production untouched' in r.stderr
        assert not any('window.sh open' in c or './stop.sh' in c for _, c in real.ssh_log()[count:])
        assert serving(real, k1=False)
    for phase in ('rollback-after-close-fallback', 'post', 'close'):
        r = real.phase(phase)
        assert r.returncode == 0, (phase, r.stdout, r.stderr)
    assert real.files() == {'env': BASE_ENV, 'start': BASE_START, 'record': real.base_record}
    assert serving(real, k1=True) and not (real.root/'HOLD').exists()


def test_close_fallback_mode_refuses_derivative_and_normal_states(real: RealK1) -> None:
    before = real.files()
    r = real.phase('rollback-after-close-fallback')       # capture derivative applied, serving
    assert r.returncode != 0 and 'Production untouched' in r.stderr and real.files() == before
    for name in ('dsv41.env', 'start.sh'):                # transition-1 fallback state: DERIVATIVE record
        shutil.copy2(real.root/'logs/k1-promotion'/name, real.root/name)
    real.state['running'] = {h: {'capture': False, 'k1': False} for h in real.state['running']}
    real.save()
    before = real.files()
    r = real.phase('rollback-after-close-fallback')
    assert r.returncode != 0 and 'not the exact K1 fallback state for rollback-after-close-fallback' in r.stderr
    assert real.files() == before and serving(real, k1=False)


def test_close_fallback_failure_after_copy_is_recovered_by_printed_rollback(real: RealK1) -> None:
    """After cp -p the files are K1 + BASE record; a failed verify prints DOWN -> 'Rerun rollback' (works)."""
    close_fallback_state(real)
    report = real.root/'logs/k1-promotion/S18.report.md'
    good = report.read_bytes()
    report.write_text('tampered\n')                       # REAL manage.py verify now refuses
    r = real.phase('rollback-after-close-fallback')
    assert r.returncode != 0 and 'step verify-restored-files' in r.stderr and 'Rerun rollback' in r.stderr
    assert all(v is None for v in real.state['running'].values())           # DOWN
    assert real.files() == {'env': BASE_ENV, 'start': BASE_START, 'record': real.base_record}
    report.write_bytes(good)
    for phase in ('rollback', 'post', 'close'):
        r = real.phase(phase)
        assert r.returncode == 0, (phase, r.stderr)
    assert serving(real, k1=True)


def test_OBSERVATION_partial_stop_in_close_fallback_prints_plain_rollback_which_refuses(real: RealK1) -> None:
    """Partial stop in a fallback mode: the DOWN text says 'Rerun rollback'. Plain `rollback` refuses (and says
    'Production untouched' while production is mixed/down); rerunning the SAME phase, or W8 start, recovers."""
    close_fallback_state(real)
    real.state['stop_partial'] = True
    real.save()
    r = real.phase('rollback-after-close-fallback')
    assert r.returncode != 0 and 'step stop' in r.stderr and 'Rerun rollback' in r.stderr
    assert real.state['running'] == {'sparkhead': None, 'sparkworker-wired': {'capture': False, 'k1': False}}   # mixed
    plain = real.phase('rollback')
    assert plain.returncode != 0 and 'Production untouched by this phase' in plain.stderr
    assert real.state['running']['sparkhead'] is None                       # still down on rank 0
    for phase in ('rollback-after-close-fallback', 'post', 'close'):
        r = real.phase(phase)
        assert r.returncode == 0, (phase, r.stderr)
    assert real.files() == {'env': BASE_ENV, 'start': BASE_START, 'record': real.base_record}
    assert serving(real, k1=True)


def test_derivative_fallback_mode_end_to_end_with_real_manage(real: RealK1) -> None:
    """Transition-1 fallback with the derivative applied: real restore leaves the DERIVATIVE record."""
    root = real.root
    restored = subprocess.run(['/usr/bin/python3', str(root/'offload/promote/k1/manage.py'), 'restore'], cwd=root,
                              capture_output=True, text=True)
    assert restored.returncode == 0, restored.stderr
    real.state['running'] = {h: {'capture': False, 'k1': False} for h in real.state['running']}
    real.save()
    r = real.phase('rollback-after-close-fallback')
    assert r.returncode != 0 and 'Production untouched' in r.stderr
    for phase in ('rollback-after-k1-fallback', 'post', 'close'):
        r = real.phase(phase)
        assert r.returncode == 0, (phase, r.stderr)
    assert real.files() == {'env': BASE_ENV, 'start': BASE_START, 'record': real.base_record}
    assert serving(real, k1=True)


def test_second_k1_fallback_inside_close_fallback_mode_is_again_recoverable(real: RealK1) -> None:
    close_fallback_state(real)
    real.state['inject_k1_failures'] = 1
    real.save()
    r = real.phase('rollback-after-close-fallback')
    assert r.returncode != 0 and 'step start' in r.stderr and 'rollback-after-close-fallback' in r.stderr
    assert serving(real, k1=False)
    for phase in ('rollback-after-close-fallback', 'post', 'close'):
        r = real.phase(phase)
        assert r.returncode == 0, (phase, r.stderr)
    assert serving(real, k1=True)
