"""CPU-only: source offload/upfix/UPF.env exactly as start_exp.sh does (dsv41.env, PORT=8001, then the config); no host."""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from offload.upfix import build_0091
from offload.upfix import configs as c
from offload.upfix.topk import build_overlay as topk

FIELDS = ('OVERLAYS', 'EXP_DOCKER_ENV', 'EXTRA_DOCKER_ARGS', 'LABEL', 'SPEC_OVERRIDE', 'KV_BYTES_OVERRIDE', 'MAX_LEN_OVERRIDE')


def source(config: str, env: dict[str, str], root: Path = c.ROOT) -> dict[str, str]:
    script = ('set -eu; set -a; . ./dsv41.env; PORT=8001; . "$1"; set +a; printf "%s\\0" '
              + ' '.join(f'"${{{f}-}}"' for f in FIELDS))
    clean = {k: v for k, v in os.environ.items() if not k.startswith(('UPF_', 'UP2_', 'UP1_', 'K1_'))}
    result = subprocess.run(['bash', '-c', script, 'upf-test', config], cwd=root, env={**clean, **env},
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(result.stderr.strip())
    return dict(zip(FIELDS, result.stdout.split('\0')))


def base_of(arm: dict[str, str]) -> dict[str, str]:
    return source(c.BASES[arm['UPF_BASE']], {k: v for k, v in arm.items() if k.startswith('UP2_')})


@pytest.mark.parametrize('name', [n for n, a in c.ARMS.items() if a['UPF_TOPK'] == '0'])
def test_sourced_arm_equals_the_python_composition(name: str) -> None:
    arm = c.ARMS[name]
    got, base = source(c.CONFIG, arm), base_of(arm)
    assert c.pairs(got['OVERLAYS']) == c.compose(c.pairs(base['OVERLAYS']), arm)
    for field in FIELDS:
        if field not in ('OVERLAYS', 'LABEL'):
            assert got[field] == base[field], (name, field)
    if arm['UPF_0091'] == '0':
        assert got == base, 'an arm with every item off must be its base config exactly'
    else:
        assert got['LABEL'] == base['LABEL'] + '-upf00911-topk0'


def test_0091_mounts_are_the_manifest_files_once_each() -> None:
    manifest = json.loads(build_0091.MANIFEST.read_text())
    for name in ('upf-0091', 'upf-0091+0075'):
        arm = c.ARMS[name]
        mounts = c.pairs(source(c.CONFIG, arm)['OVERLAYS'])
        targets = [t for _, t in mounts]
        assert len(targets) == len(set(targets)), 'a container path is mounted twice'
        variant = 'image' if arm['UPF_BASE'] == 'k1' else '0075'
        for row in manifest['files']:
            if row['variant'] == 'image' and row['path'] != build_0091.SCHEDULER or row['variant'] == variant:
                assert (row['overlay'], row['container_path']) in mounts, (name, row['overlay'])
        assert (c.ADOPT_SCHEDULER, c.PY + c.SCHEDULER) not in mounts


def test_0075_composition_keeps_every_other_up2_mount_in_place() -> None:
    arm = c.ARMS['upf-0091+0075']
    base = c.pairs(base_of(arm)['OVERLAYS'])
    got = c.pairs(source(c.CONFIG, arm)['OVERLAYS'])
    assert got[:len(base)] == [('offload/upfix/overlay-0075/' + c.SCHEDULER, t) if s == c.ADOPT_SCHEDULER else (s, t)
                               for s, t in base]


def test_replay_skipping_with_0091_is_refused_until_the_guard_admits_it() -> None:
    with pytest.raises(ValueError, match='replay_guard admits only the UP2 scheduler'):
        source(c.CONFIG, {**c.ARMS['upf-0091+0075'], 'UP2_REPLAY': '1'})
    assert source(c.CONFIG, {**c.ARMS['C-0075'], 'UP2_REPLAY': '1'}) == base_of({**c.ARMS['C-0075'], 'UP2_REPLAY': '1'})


@pytest.mark.parametrize('field,value', [('UPF_BASE', 'prod'), ('UPF_0091', 'yes'), ('UPF_TOPK', '2')])
def test_invalid_switches_are_refused(field: str, value: str) -> None:
    with pytest.raises(ValueError, match=f'invalid {field}'):
        source(c.CONFIG, {**c.ARMS['C-k1'], field: value})


def test_compose_refuses_collisions_and_shadowing() -> None:
    arm = c.ARMS['upf-0091']
    with pytest.raises(ValueError, match='base already mounts'):
        c.compose([('elsewhere/scheduler.py', c.PY + c.SCHEDULER)], arm)
    with pytest.raises(ValueError, match='shadows'):
        c.compose([('elsewhere/sched', c.PY + 'vllm/v1/core/sched')], arm)
    with pytest.raises(ValueError, match='malformed'):
        c.pairs('no-equals-sign')


def test_no_arm_target_is_a_k1_or_up2_mount_except_the_replaced_scheduler() -> None:
    for name, arm in c.ARMS.items():
        base_targets = {t for _, t in c.pairs(base_of(arm)['OVERLAYS'])}
        for _, target in c.added(arm):
            assert target not in base_targets or (target == c.PY + c.SCHEDULER and arm['UPF_BASE'] == '0075'), name


def fake_root(tmp_path: Path, verdict: str = 'PASS', capability: list[int] | None = None,
              build_sha: str | None = None, done: int = 149) -> Path:
    """A copy of the parts of the checkout the topk start gate reads, plus a fake built library and receipts."""
    root = tmp_path / 'repo'
    for rel in (c.CONFIG, 'offload/upfix/topk/MANIFEST-topk.json', 'offload/upfix/topk/overlay/' + c.TOPK_REL):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(c.ROOT / rel, root / rel)
    library = root / c.TOPK_SO
    library.parent.mkdir(parents=True)
    library.write_bytes(b'fake dsv41_topk library')
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    build = root / c.TOPK_BUILD
    (build / 'receipt.json').write_text(json.dumps({'arch': '12.1', 'library_sha256': build_sha or digest}))
    (build / 'bitcompare.json').write_text(json.dumps({'verdict': verdict, 'library_sha256': digest,
                                                       'capability': capability or [12, 1], 'cases': 149,
                                                       'cases_done': done, 'complete': done == 149}))
    return root


def test_topk_gate_passes_only_a_passing_sm121_compare_of_the_same_bytes(tmp_path: Path) -> None:
    assert c.verify_topk(fake_root(tmp_path / 'ok')) == hashlib.sha256(b'fake dsv41_topk library').hexdigest()
    with pytest.raises(ValueError, match="verdict 'FAIL'"):
        c.verify_topk(fake_root(tmp_path / 'fail', verdict='FAIL'))
    with pytest.raises(ValueError, match='incomplete .147 of 149'):
        c.verify_topk(fake_root(tmp_path / 'partial', verdict='INCOMPLETE', done=147))
    with pytest.raises(ValueError, match='not SM121'):
        c.verify_topk(fake_root(tmp_path / 'cap', capability=[12, 0]))
    with pytest.raises(ValueError, match='jit_build receipt'):
        c.verify_topk(fake_root(tmp_path / 'sha', build_sha='0' * 64))
    root = fake_root(tmp_path / 'overlay')
    target = root / 'offload/upfix/topk/overlay' / c.TOPK_REL
    target.write_text(target.read_text() + '# drift\n')
    with pytest.raises(ValueError, match='differs from MANIFEST-topk'):
        c.verify_topk(root)
    with pytest.raises(ValueError, match='built library'):
        c.verify_topk(tmp_path / 'empty')


def test_topk_arm_refuses_to_start_without_a_passing_compare() -> None:
    assert not (c.ROOT / c.TOPK_BUILD / 'bitcompare.json').exists()
    with pytest.raises(ValueError, match='UPF_TOPK start gate'):
        source(c.CONFIG, c.ARMS['topk-fix'])


def test_topk_arm_mounts_overlay_library_and_env(tmp_path: Path) -> None:
    """Source the topk arm in a throwaway copy of the checkout whose gate inputs are faked."""
    root = fake_root(tmp_path)
    for rel in ('dsv41.env', 'offload/__init__.py', 'offload/upfix/__init__.py', 'offload/upfix/configs.py',
                *c.BASES.values(), 'offload/launch/configs/D-kv6-async-up1-control.env'):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(c.ROOT / rel, root / rel)
    for name in ('topk-fix', 'upf-0091+topk-fix'):
        arm = c.ARMS[name]
        got, base = source(c.CONFIG, arm, root), base_of(arm)
        assert c.pairs(got['OVERLAYS']) == c.compose(c.pairs(base['OVERLAYS']), arm)
        extra = shlex.split(got['EXP_DOCKER_ENV'])[len(shlex.split(base['EXP_DOCKER_ENV'])):]
        assert dict(s.split('=', 1) for s in extra) == c.TOPK_ENV
        assert (c.TOPK_SO, c.TOPK_SO_CONTAINER) in c.pairs(got['OVERLAYS'])
        assert ('offload/upfix/topk/overlay/' + c.TOPK_REL, c.PY + c.TOPK_REL) in c.pairs(got['OVERLAYS'])
    assert topk.REL == c.TOPK_REL
