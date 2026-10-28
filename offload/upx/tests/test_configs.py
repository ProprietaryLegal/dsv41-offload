"""CPU-only: source offload/upx/UPX.env exactly as start_exp.sh does (dsv41.env, PORT=8001, then the config); no host."""
from __future__ import annotations

import functools
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from offload.upx import build_0085b as b
from offload.upx import configs as c

FIELDS = ('OVERLAYS', 'EXP_DOCKER_ENV', 'EXTRA_DOCKER_ARGS', 'LABEL', 'SPEC_OVERRIDE', 'KV_BYTES_OVERRIDE', 'MAX_LEN_OVERRIDE')


PRODUCTION_COMMIT = '596ba1c'


@functools.cache
def production_env() -> str:
    """Production's dsv41.env at its promoted commit, as ~/dsv41-deploy/dsv41.env holds it (git object, read-only)."""
    text = subprocess.run(['git', 'show', f'{PRODUCTION_COMMIT}:dsv41.env'], cwd=c.ROOT, capture_output=True, text=True,
                          check=True).stdout
    path = Path(tempfile.mkdtemp(prefix='upx-prod-env-')) / 'dsv41.env'
    path.write_text(text)
    return str(path)


def source(config: str, env: dict[str, str], root: Path = c.ROOT, prelude: str = '') -> dict[str, str]:
    script = (f'set -eu; set -a; . ./dsv41.env; PORT=8001; {prelude}. "$1"; set +a; printf "%s\\0" '
              + ' '.join(f'"${{{f}-}}"' for f in FIELDS))
    clean = {k: v for k, v in os.environ.items() if not k.startswith(('UPX_', 'UPF_', 'UP2_', 'UP1_', 'K1_'))}
    clean['UPX_PROD_ENV'] = production_env()
    result = subprocess.run(['bash', '-c', script, 'upx-test', config], cwd=root, env={**clean, **env},
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(result.stderr.strip())
    return dict(zip(FIELDS, result.stdout.split('\0')))


def base_of(arm: dict[str, str]) -> dict[str, str]:
    return source(c.BASES[arm['UPX_BASE']], {k: v for k, v in arm.items() if k.startswith('UP2_')})


@pytest.mark.parametrize('name', list(c.ARMS))
def test_sourced_arm_equals_the_python_composition(name: str) -> None:
    arm = c.ARMS[name]
    got, base = source(c.CONFIG, arm), base_of(arm)
    assert c.pairs(got['OVERLAYS']) == c.compose(c.pairs(base['OVERLAYS']), arm)
    assert got['EXP_DOCKER_ENV'].split() == base['EXP_DOCKER_ENV'].split() + c.switch(arm)
    for field in FIELDS:
        if field not in ('OVERLAYS', 'EXP_DOCKER_ENV', 'LABEL'):
            assert got[field] == base[field], (name, field)
    if arm['UPX_0085B'] == '0':
        assert got == base, 'an arm with 0085b off must be its base config exactly (no mount, no switch)'
    else:
        assert got['LABEL'] == base['LABEL'] + '-0085b-' + arm['UPX_0085B']


def test_mounts_are_the_manifest_files_once_each() -> None:
    import json
    manifest = json.loads(b.MANIFEST.read_text())
    for name in ('release-tokens', 'release-file', 'release-file+0075'):
        mounts = c.pairs(source(c.CONFIG, c.ARMS[name])['OVERLAYS'])
        targets = [t for _, t in mounts]
        assert len(targets) == len(set(targets)), 'a container path is mounted twice'
        for row in manifest['files']:
            assert (row['overlay'], row['container_path']) in mounts, (name, row['overlay'])


def test_threshold_and_switch_values() -> None:
    got = source(c.CONFIG, {**c.ARMS['release-tokens'], 'UPX_RELEASE_TOKENS': '65536'})
    assert got['EXP_DOCKER_ENV'].split()[-1] == 'VLLM_DSV41_RELEASE_TOKENS=65536'
    got = source(c.CONFIG, c.ARMS['release-tokens'])
    assert got['EXP_DOCKER_ENV'].split()[-1] == f'VLLM_DSV41_RELEASE_TOKENS={c.DEFAULT_THRESHOLD}'
    got = source(c.CONFIG, c.ARMS['release-file'])
    assert got['EXP_DOCKER_ENV'].split()[-1] == f'VLLM_DSV41_RELEASE_FILE={c.RELEASE_FILE_CONTAINER}'
    assert 'VLLM_DSV41_RELEASE_TOKENS' not in got['EXP_DOCKER_ENV']


@pytest.mark.parametrize('field,value', [('UPX_BASE', 'production'), ('UPX_0085B', '1'), ('UPX_RELEASE_TOKENS', '-5'),
                                         ('UPX_RELEASE_TOKENS', '12k')])
def test_invalid_switches_are_refused(field: str, value: str) -> None:
    with pytest.raises(ValueError, match=f'invalid {field}'):
        source(c.CONFIG, {**c.ARMS['release-tokens'], field: value})


def test_snippet_refuses_a_base_that_already_mounts_or_sets_0085b() -> None:
    target = c.PY + b.WORKER
    with pytest.raises(ValueError, match='already mounts vllm/v1/worker/gpu_worker.py'):
        source(c.SNIPPET, {'UPX_0085B': 'file'}, prelude=f'OVERLAYS="x={target}"; EXP_DOCKER_ENV=; LABEL=t; ')
    with pytest.raises(ValueError, match='already sets VLLM_DSV41_RELEASE_TOKENS=1'):
        source(c.SNIPPET, {'UPX_0085B': 'file'}, prelude='OVERLAYS=; EXP_DOCKER_ENV="A=1 VLLM_DSV41_RELEASE_TOKENS=1"; LABEL=t; ')


def test_snippet_composes_onto_any_base_and_off_changes_nothing() -> None:
    prelude = 'OVERLAYS="a=/x/a.py b=/x/b.py"; EXP_DOCKER_ENV="K=1"; LABEL=PC1-chunk16384; '
    off = source(c.SNIPPET, {}, prelude=prelude)
    assert (off['OVERLAYS'], off['EXP_DOCKER_ENV'], off['LABEL']) == ('a=/x/a.py b=/x/b.py', 'K=1', 'PC1-chunk16384')
    on = source(c.SNIPPET, {'UPX_0085B': 'file'}, prelude=prelude)
    assert c.pairs(on['OVERLAYS'])[:2] == [('a', '/x/a.py'), ('b', '/x/b.py')]
    assert c.pairs(on['OVERLAYS'])[2:] == c.added({'UPX_0085B': 'file'})
    assert on['EXP_DOCKER_ENV'] == f'K=1 VLLM_DSV41_RELEASE_FILE={c.RELEASE_FILE_CONTAINER}'
    assert on['LABEL'] == 'PC1-chunk16384-0085b-file'


def test_compose_refuses_shadowing_directory_mounts() -> None:
    with pytest.raises(ValueError, match='shadows'):
        c.compose([('d', c.PY + 'vllm/v1/engine')], c.ARMS['release-file'])
    with pytest.raises(ValueError, match='already mounts'):
        c.compose([('f', c.PY + b.CORE)], c.ARMS['release-file'])


def test_release_file_command_is_atomic_and_rejects_negative() -> None:
    cmd = c.release_file_command(131072)
    assert cmd == ("mkdir -p ~/.cache/dsv41-serve && printf '%d\\n' 131072 > ~/.cache/dsv41-serve/.dsv41-release-tokens.new"
                   " && mv -f ~/.cache/dsv41-serve/.dsv41-release-tokens.new ~/.cache/dsv41-serve/dsv41-release-tokens")
    with pytest.raises(ValueError):
        c.release_file_command(-1)


def test_release_file_command_writes_what_the_worker_reads(tmp_path: Path) -> None:
    home = tmp_path / 'home'
    home.mkdir()
    subprocess.run(['bash', '-c', c.release_file_command(0) + ' && ' + c.release_file_command(131072)],
                   env={'HOME': str(home), 'PATH': os.environ['PATH']}, check=True)
    written = home / '.cache/dsv41-serve/dsv41-release-tokens'
    assert int(written.read_text().strip()) == 131072
    assert not (home / '.cache/dsv41-serve/.dsv41-release-tokens.new').exists()


def test_prod_base_is_production_verbatim() -> None:
    got = source(c.BASES['prod'], {})
    script = ('set -eu; . "$1"; printf "%s\\0" "$OFFLOAD_DOCKER_ENV $OFFLOAD_DSPARK_ENV" "$OFFLOAD_DOCKER_ARGS"')
    prod = subprocess.run(['bash', '-c', script, 'p', production_env()], capture_output=True, text=True, check=True)
    env, args = prod.stdout.split('\0')[:2]
    assert got['EXP_DOCKER_ENV'] == env and got['EXTRA_DOCKER_ARGS'] == args
    assert (got['OVERLAYS'], got['KV_BYTES_OVERRIDE'], got['MAX_LEN_OVERRIDE']) == ('', '6442450944', '327680')
    on = source(c.CONFIG, c.ARMS['release-file-prod'])
    assert on['EXTRA_DOCKER_ARGS'] == args and c.pairs(on['OVERLAYS']) == c.added(c.ARMS['release-file-prod'])


def test_prod_base_refuses_inherited_overrides_and_a_changed_ledger(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='inherited OVERLAYS'):
        source(c.CONFIG, {**c.ARMS['C-prod'], 'OVERLAYS': 'a=/b'})
    changed = tmp_path / 'dsv41.env'
    changed.write_text(Path(production_env()).read_text().replace('PROD_MAX_LEN=327680', 'PROD_MAX_LEN=524288'))
    with pytest.raises(ValueError, match='ledger changed'):
        source(c.CONFIG, {**c.ARMS['C-prod'], 'UPX_PROD_ENV': str(changed)})


def test_snippet_refuses_a_docker_arg_mount_of_a_0085b_file() -> None:
    target = c.PY + b.CORE
    with pytest.raises(ValueError, match='already mounts vllm/v1/engine/core.py'):
        source(c.SNIPPET, {'UPX_0085B': 'file'},
               prelude=f'OVERLAYS=; EXTRA_DOCKER_ARGS="-v /x/core.py:{target}:ro"; EXP_DOCKER_ENV=; LABEL=t; ')
