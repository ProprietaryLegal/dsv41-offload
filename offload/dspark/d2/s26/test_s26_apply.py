"""S26: decode_apply against a copy of the pinned production tree (sparkhead main 21bed31), with the REAL K1 guard.

The K1 promotion state is rebuilt the way K1 manage.py apply leaves it (pre-K1 backups from 3c1dc6b, files.json
with original/promoted hashes, an S18-shaped receipt bound to a stand-in report). Nothing touches a real deployment.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

import pytest

from d2 import decode_apply as A

REPO = Path(__file__).resolve().parents[4]
GIT = os.environ.get('S26_GIT_ROOT', str(Path(__file__).resolve().parents[4]))  # archived merge trees
SOURCE = '29398bef8bfe992dd7336a14a79d77eee8df9bb0'
RUN = 's26probe'


def git(*args: str) -> bytes:
    return subprocess.run(['git', *args], cwd=GIT, capture_output=True, check=True).stdout


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    r = tmp_path / 'deploy'
    r.mkdir()
    tarfile.open(fileobj=io.BytesIO(git('archive', A.BASE_COMMIT, 'offload/kvpack', 'offload/promote/k1',
                                        'dsv41.env', 'start.sh'))).extractall(r, filter='data')
    state = r / 'logs/k1-promotion'
    state.mkdir(parents=True)
    for name in ('dsv41.env', 'start.sh'):
        (state / name).write_bytes(git('show', f'3c1dc6b:{name}'))
    (state / 'S18.report.md').write_text('stand-in report for the CPU probe\n')
    (state / 'S18.receipt.json').write_text(json.dumps({
        'schema': 'dsv41-k1-promotion-v1', 'reviewer': 'S18', 'final': True, 'window6': 'PROMOTE',
        'max_model_len': 327680, 'kv_tokens': 2502600, 'manifest_sha256': A.K1,
        'report_sha256': sha(state / 'S18.report.md')}))
    record = {'profile': '327680', 'original': {n: sha(state / n) for n in ('dsv41.env', 'start.sh')},
              'promoted': {n: sha(r / n) for n in ('dsv41.env', 'start.sh')}}
    (state / 'files.json').write_text(json.dumps(record, indent=2) + '\n')
    return r


def k1(root: Path, action: str) -> subprocess.CompletedProcess[str]:
    args = ['python3', str(root / 'offload/promote/k1/manage.py'), action, '--root', str(root)]
    if action == 'verify':
        args += ['--profile', '327680']
    return subprocess.run(args, capture_output=True, text=True)


def review(root: Path, tmp: Path) -> Path:
    rendered = A.render((root / 'dsv41.env').read_bytes(), SOURCE, RUN)
    p = tmp / 'review.json'
    p.write_text(json.dumps({'schema': 'd2.decode-apply-review.v1', 'approved': True,
                             'coordinator': 'Deepseek/coordinator', 'source': SOURCE, 'base_commit': A.BASE_COMMIT,
                             'base_env_sha256': A.BASE_ENV,
                             'capture_env_sha256': hashlib.sha256(rendered).hexdigest()}))
    return p


def sourced(root: Path, var: str) -> str:
    return subprocess.run(['bash', '-c', f'set -a; . ./dsv41.env; set +a; printf %s "${var}"'], cwd=root,
                          capture_output=True, text=True, check=True).stdout


def test_pinned_base_and_rendered_diff_is_the_stated_append(root: Path) -> None:
    assert sha(root / 'dsv41.env') == A.BASE_ENV and sha(root / 'start.sh') == A.BASE_START
    assert k1(root, 'verify').returncode == 0, k1(root, 'verify').stderr
    before = (root / 'dsv41.env').read_bytes()
    after = A.render(before, SOURCE, RUN)
    assert after.startswith(before)
    template = (REPO / 'offload/dspark/d2/production-decode.patch').read_text().splitlines()
    added = [x[1:] for x in template if x.startswith('+') and not x.startswith('+++')]
    want = '\n'.join(added).replace('0' * 40, SOURCE).replace('REVIEWED_RUN', RUN) + '\n'
    assert after[len(before):].decode() == want


def test_apply_passes_the_real_k1_guard_and_rollback_is_byte_exact(root: Path, tmp_path: Path) -> None:
    env0, rec0 = (root / 'dsv41.env').read_bytes(), (root / 'logs/k1-promotion/files.json').read_bytes()
    args0 = sourced(root, 'OFFLOAD_DOCKER_ARGS')
    A.apply(root, SOURCE, RUN, review(root, tmp_path))
    assert (root / 'start.sh').read_bytes() == git('show', f'{A.BASE_COMMIT}:start.sh')
    v = k1(root, 'verify')
    assert v.returncode == 0, v.stderr                      # unchanged K1 start guard accepts the derivative
    args1 = sourced(root, 'OFFLOAD_DOCKER_ARGS')
    assert args1.startswith(args0 + ' ')                    # every K1/remote mount kept, in order
    extra = args1[len(args0) + 1:].split(' ')
    assert extra[0::2] == ['-v'] * 4 and len(extra) == 8
    dests = [m.split(':')[1] for m in extra[1::2]]
    assert dests == [A.BASE + '/dspark/speculator.py', A.BASE + '/dspark/decode_capture.py',
                     A.BASE + '/remote_dspark/speculator.py', '/d2-decode']
    assert all(f'$HOME/dsv41-decode/{SOURCE}/' in m for m in extra[1:6:2])
    assert sourced(root, 'OFFLOAD_DOCKER_ENV').endswith(' VLLM_DSPARK_DECODE_CAPTURE_DIR=/d2-decode')
    # No duplicate container destination (docker refuses those); the file mount nests inside the ro remote dir.
    all_dests = [m.split(':')[1] for m in args1.split(' ') if m.count(':') >= 2]
    assert len(all_dests) == len(set(all_dests))
    A.rollback(root, RUN)
    assert (root / 'dsv41.env').read_bytes() == env0
    assert (root / 'logs/k1-promotion/files.json').read_bytes() == rec0
    assert k1(root, 'verify').returncode == 0


def test_apply_refuses_reuse_and_stacking(root: Path, tmp_path: Path) -> None:
    rv = review(root, tmp_path)
    A.apply(root, SOURCE, RUN, rv)
    with pytest.raises(ValueError):
        A.apply(root, SOURCE, RUN, rv)                       # applied env is not the reviewed base
    A.rollback(root, RUN)
    with pytest.raises(FileExistsError):
        A.apply(root, SOURCE, RUN, rv)                       # a run ID is never reused


def test_k1_fallback_after_a_failed_capture_start_leaves_pre_k1_and_d2_rollback_refuses(root: Path,
                                                                                       tmp_path: Path) -> None:
    """Residual, not FIX: start_checked.sh's ERR fallback runs `manage.py restore` - with the derivative record that
    restores the PRE-K1 (3c1dc6b) env/start.sh and restarts it. decode_apply --rollback then refuses."""
    A.apply(root, SOURCE, RUN, review(root, tmp_path))
    assert k1(root, 'restore').returncode == 0
    assert (root / 'dsv41.env').read_bytes() == git('show', '3c1dc6b:dsv41.env')
    with pytest.raises(ValueError, match='base rollback files corrupted'):   # start.sh is pre-K1 now
        A.rollback(root, RUN)
    assert k1(root, 'verify').returncode != 0                # K1 cannot be started again without manual repair


def test_explicit_fallback_recovery_restores_exact_k1_and_guard(root: Path, tmp_path: Path) -> None:
    before = {name:(root/name).read_bytes() for name in ('dsv41.env','start.sh','logs/k1-promotion/files.json')}
    A.apply(root, SOURCE, RUN, review(root,tmp_path))
    assert k1(root,'restore').returncode == 0
    A.rollback(root,RUN,after_k1_fallback=True)
    assert {name:(root/name).read_bytes() for name in before} == before
    verify = k1(root,'verify')
    assert verify.returncode == 0,verify.stderr


@pytest.mark.parametrize('edited', ['dsv41.env','start.sh','logs/k1-promotion/files.json','logs/d2-decode-s26probe/start.sh'])
def test_fallback_recovery_refuses_edits_before_any_restore(root: Path, tmp_path: Path, edited: str) -> None:
    A.apply(root,SOURCE,RUN,review(root,tmp_path))
    assert k1(root,'restore').returncode == 0
    path = root/edited
    path.write_bytes(path.read_bytes()+b'\n# unrelated edit\n')
    before = {name:(root/name).read_bytes() for name in ('dsv41.env','start.sh','logs/k1-promotion/files.json')}
    with pytest.raises((ValueError,json.JSONDecodeError)):
        A.rollback(root,RUN,after_k1_fallback=True)
    assert {name:(root/name).read_bytes() for name in before} == before
